"""Segmentation loss, mIoU accumulation, and probe training state.

Port of canvit_specialize's loss.py / metrics.py / state.py / eval_utils.py
(unchanged math; the P2 gate compares numbers against specialize runs)."""

import logging
from dataclasses import dataclass
from functools import lru_cache
from typing import Any

import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.optim import AdamW

from canvit.core.metrics import mIoUAccumulator  # noqa: F401  (re-exported for this repo's consumers)
from canvit.core.policy import per_image_ce
from canvit.core.probes import SegmentationProbe

from .data import IGNORE_LABEL

log = logging.getLogger(__name__)


def ce_loss(logits: Tensor, masks: Tensor) -> Tensor:
    """Cross-entropy for semantic segmentation (masks nearest-resized to logits)."""
    if masks.shape[1:] != logits.shape[2:]:
        masks = F.interpolate(masks.unsqueeze(1).float(), logits.shape[2:], mode="nearest").squeeze(1).long()
    return F.cross_entropy(logits, masks, ignore_index=IGNORE_LABEL)


def upsample_preds(preds: Tensor, H: int, W: int) -> Tensor:
    """Nearest-upsample an integer LABEL map. For rendering only — for metrics use
    ``preds_from_logits``, which upsamples the logits instead (see its docstring)."""
    if preds.shape[1:] == (H, W):
        return preds
    return F.interpolate(preds.unsqueeze(1).float(), (H, W), mode="nearest").squeeze(1).long()


def preds_from_logits(logits: Tensor, H: int, W: int) -> Tensor:
    """Full-resolution predictions the PAPER way: bilinear-upsample the logits to the
    mask resolution, THEN argmax.

    The order matters and this repo had it backwards until 2026-07-29. Taking argmax at
    the probe grid first (64x64) locks each label into an 8x8 pixel block, so boundaries
    can only fall on the coarse grid; upsampling logits first lets them land at full
    resolution. The old order is strictly coarser and cost a MEASURED 0.19 mIoU at t4 on
    ADE20K val (2 seeds, +0.15/+0.17/+0.19/+0.19 at t1..t4 — reproducible to 0.01).

    This is the protocol every published ADE20K number is measured under. It was
    verified against CanViT-eval's independent implementation before that repo was
    archived: the two agreed on all ten timesteps to +0.00e+00 on exp34's probe
    (``canvit_claude/canvit/unification/20-eval-merge.md`` §5, and the archive marker in CanViT-eval).
    """
    if logits.shape[-2:] != (H, W):
        logits = F.interpolate(logits.float(), size=(H, W), mode="bilinear", align_corners=False)
    return logits.argmax(1)


@lru_cache(maxsize=8)
def _warn_score_res(res: int, full: int) -> None:
    """Once per (res, full) pair — this sits inside the per-glimpse reward path."""
    log.warning(
        "reward_score_res=%d does not divide the mask resolution %d, so the reward is "
        "scored at FULL %d instead (more accurate, ~2x slower). The reference value 128 "
        "assumes scene_size=512; set --cfg.reward-score-res to a divisor of %d to silence.",
        res, full, full, full)


def reward_ce(logits: Tensor, masks: Tensor, *, score_res: int | None) -> Tensor:
    """Per-image probe CE — the policy reward's raw material — scored at ``score_res``.

    ONE implementation for both policy entry points (`ade20k/rl_train.py::ce_from_logits`
    and `tasks/ade20k/task.py::BoundAde20kTask.per_image_loss`), because the reward must
    not depend on which trainer computes it. The RL repo's rule: bilinear-upsample the
    logits to ``score_res`` and STRIDE-subsample the masks down to it. ``score_res=None``
    means the masks' own resolution (full res).

    128 is the reference value, validated there against full 512 at Spearman 0.999 for
    candidate ranking at ~2x lower cost. The harness previously scored at the probe's
    native 64 instead — cheaper still, but never validated, and it is the last known
    config difference from the reference (doc 15 §A gap #5).
    """
    full = masks.shape[-1]
    res = score_res or full
    if full % res != 0:
        # Subsampling is a stride, so it needs divisibility. Fall back to FULL resolution
        # rather than assert: full res is strictly more accurate (it is what score_res
        # approximates for speed), so a wrong-but-plausible reward is impossible here —
        # only a slower one. 128 divides the reference's 512; other scene sizes land here.
        _warn_score_res(res, full)
        res = full
    m = masks if res == full else masks[:, :: full // res, :: full // res].contiguous()
    up = F.interpolate(logits.float(), size=(res, res), mode="bilinear", align_corners=False)
    return per_image_ce(up, m, ignore_label=IGNORE_LABEL).float()


def eval_probe_on_batch(probe: nn.Module, features: Tensor, masks: Tensor,
                        iou: mIoUAccumulator) -> Tensor:
    """Forward probe, upsample logits to mask resolution, argmax, update IoU.

    Returns the full-resolution predictions. The per-row IoU output needs exactly these and
    they were being discarded, so returning them is what keeps that an extra scatter-add
    rather than an extra forward. Existing callers ignore the value.
    """
    logits = probe(features.float())
    preds = preds_from_logits(logits, masks.shape[1], masks.shape[2])
    iou.update(preds, masks)
    return preds


@dataclass
class ProbeState:
    """Training state for one probe head."""

    name: str
    head: SegmentationProbe
    optimizer: AdamW
    scheduler: Any  # WarmupOneCycleLR or other LR scheduler
    n_timesteps: int = 0
    _best_mious: list[float] | None = None
    _loss_sum: Tensor | None = None
    _grad_norm_sum: Tensor | None = None
    _count: int = 0

    def init_best_mious(self, n_timesteps: int) -> None:
        self.n_timesteps = n_timesteps
        self._best_mious = [0.0] * n_timesteps

    @property
    def best_mious(self) -> list[float]:
        assert self._best_mious is not None, "call init_best_mious first"
        return self._best_mious

    @property
    def best_last_miou(self) -> float:
        return self.best_mious[-1]

    def update_best(self, mious: list[float]) -> bool:
        """Update per-timestep bests. Returns True if last timestep improved."""
        assert len(mious) == self.n_timesteps
        old_last = self.best_last_miou
        for t, v in enumerate(mious):
            if v > self.best_mious[t]:
                self.best_mious[t] = v
        return self.best_last_miou > old_last

    def accumulate(self, loss: Tensor, grad_norm: Tensor) -> None:
        """Accumulate loss/grad_norm. NO GPU sync."""
        if self._loss_sum is None:
            self._loss_sum = loss.detach().clone()
            self._grad_norm_sum = grad_norm.detach().clone()
        else:
            self._loss_sum += loss.detach()
            assert self._grad_norm_sum is not None
            self._grad_norm_sum += grad_norm.detach()
        self._count += 1

    def get_and_reset(self) -> tuple[float, float]:
        """Get averaged stats and reset. SYNCS here."""
        assert self._loss_sum is not None and self._grad_norm_sum is not None
        avg_loss = (self._loss_sum / self._count).item()
        avg_grad = (self._grad_norm_sum / self._count).item()
        self._loss_sum = self._grad_norm_sum = None
        self._count = 0
        return avg_loss, avg_grad
