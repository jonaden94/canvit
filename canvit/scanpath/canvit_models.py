"""CanViT search checkpoints as scanpath models (design §3).

Two readouts of one teacher-forced episode, after the glimpses at the human's fixations 0..t-1:

- ``policy``:  the learned scorer's logits -- the distribution the deployed policy picks
               fixation t from (its argmax at deploy).
- ``heatmap``: the task head's target heatmap -- "where is the target", the map ``greedy``
               fixates the argmax of. Needs no policy (a ``--preset probe`` checkpoint suffices).

**Same code path as the deploy rollout** (``harness.rollout.eval_viewpoints.
deploy_rollout_viewpoints``): the deploy selector (argmax, ``prime_on_policy=1``) is called at
every glimpse exactly as there, so the scorer, its rolling encoder reference (``enc.prev``), the
cue embedding and the view scale are the deployed ones. Only the CENTRE of each viewpoint
t >= 1 is replaced by the human's fixation t. t0 is the FULL anchor (the centre at the training
view scale), standing in for the human's start fixation, which COCO-Search18 records near the
centre. ``test_canvit_models.py`` pins this: forcing the policy's own argmax reproduces the
deploy rollout's scores exactly.

Forced centres are the human's continuous positions (clamped onto the image), not snapped to the
policy's 32x32 candidate cells: the backbone accepts any centre, and the human saw that one.

Built with the TRAINING constructors (``SearchRunTask.build_model`` / ``build_policy``) from the
same ``--cfg.search.*`` / ``--cfg.rl.*`` flags as the training run, then loaded STRICTLY: the
checkpoint does not record the policy's architecture, so a mismatched flag must fail loudly
rather than score a half-initialised scorer.
"""

from __future__ import annotations

import logging
from contextlib import nullcontext
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import torch
from torch import Tensor

from canvit.core import Viewpoint

from .data import clamp_to_content

log = logging.getLogger(__name__)


@dataclass
class SearchEpisode:
    """A loaded search model (+ policy) and its teacher-forced episode."""

    task: Any  # SearchRunTask
    model: Any  # CanViTForVisualSearch
    joint: Any | None  # JointPolicy, None for a probe-only checkpoint
    device: torch.device
    amp: bool = True

    @classmethod
    def load(cls, *, search_cfg, rl, ckpt: Path, device: torch.device, need_policy: bool) -> SearchEpisode:
        from canvit.search.task import SearchRunTask
        task = SearchRunTask(search_cfg, rl=rl)
        model, _ = task.build_model(device)
        payload = torch.load(ckpt, map_location=device, weights_only=False)
        _check_view_scale(payload, search_cfg)
        model.load_state_dict(payload["model_state"], strict=True)
        model.eval()
        joint = None
        sidecar = Path(ckpt).with_suffix(".policy.pt")
        if need_policy:
            if not sidecar.exists():
                raise FileNotFoundError(f"{sidecar} not found: the policy is saved beside the checkpoint as "
                                        "<ckpt>.policy.pt; a probe-only run has none (use --cfg.model canvit_heatmap)")
            joint = task.build_policy(model, device=device, canvas_grid=task.canvas_grid(model),
                                      generator=torch.Generator(device=device))
            joint.load_state_dict(torch.load(sidecar, map_location=device, weights_only=False))  # strict
            joint.scorer.eval()
        log.info("loaded %s (step %s)%s", ckpt, payload.get("step"), " + policy" if joint else "")
        return cls(task=task, model=model, joint=joint, device=device, amp=search_cfg.amp)

    def _amp(self):
        if self.device.type == "cuda" and self.amp:  # the training loop's rule, as search's evaluate
            dt = torch.bfloat16 if torch.cuda.get_device_capability(self.device) >= (8, 0) else torch.float16
            return torch.autocast("cuda", dtype=dt)
        return nullcontext()

    @torch.no_grad()
    def run(self, batch: dict, n: int) -> dict[str, Tensor]:
        """Glimpses at the anchor and the human fixations 1..n-1; returns ``policy`` (if loaded) and
        ``heatmap`` logits, each [B, n, G*G], row t-1 read after the glimpse at fixation t-1."""
        from canvit.harness.rollout.engine import _to_vp
        from canvit.harness.rollout.eval_viewpoints import deploy_selector
        from canvit.harness.rollout.viewpoint import ViewpointType
        from canvit.search.task import _canvas_hidden

        model, images, cue = self.model, batch["image"], batch["cue"]
        B, cg = images.shape[0], self.task.canvas_grid(model)
        if (cue < 0).any():
            raise ValueError("the search model needs a cue on every trial; this dataset has free-viewing trials")
        human = batch["fix"].nan_to_num(0.0)  # rows shorter than t: any finite centre, ignored downstream
        sel = deploy_selector(self.joint) if self.joint is not None else self.task._random_selector()
        ctx = sel.start_rollout(t0_type=ViewpointType.FULL, batch_size=B, device=images.device)
        if self.joint is not None:
            embed = self.task._policy_embed(model)
            ctx.cond = lambda: embed(cue)
        out: dict[str, list[Tensor]] = {"heatmap": []} | ({"policy": []} if self.joint is not None else {})
        state = None
        # ONE select per glimpse, as in the deploy rollout: the encoder keeps a rolling reference,
        # so a second call per glimpse would shift every later feature. The select at step t reads
        # the state after glimpse t-1, i.e. its scores ARE the prediction for fixation t; one last
        # select after glimpse n-1 scores fixation n.
        for t in range(n + 1):
            if t == n and self.joint is None:
                break
            named = sel.select(vp_type=ViewpointType.FULL if t == 0 else ViewpointType.RANDOM, ctx=ctx, t=t,
                               batch_size=B, device=images.device, state=state)
            if t > 0 and self.joint is not None:
                out["policy"].append(sel.last_aux["scores"].float().reshape(B, -1))
            if t == n:
                break
            vp = _to_vp(named)
            if t > 0:
                xy = clamp_to_content(human[:, t], batch["content"])
                vp = Viewpoint(centers=xy.flip(-1).to(vp.centers.dtype), scales=vp.scales)  # (y, x)
            if state is None:
                state = model.init_state(batch_size=B, canvas_grid_size=cg)
            with self._amp():
                state = model.canvit(image=images, state=state, viewpoint=vp).state
            heat, _ = model.head(_canvas_hidden(model, state, cg).float(), cue)
            out["heatmap"].append(heat.float().reshape(B, -1))
        return {k: torch.stack(v, 1) for k, v in out.items()}


def _check_view_scale(payload: dict, search_cfg) -> None:
    """Refuse a view scale that differs from the one the checkpoint records (design §6)."""
    rec = (payload.get("metadata") or {}).get("pretrain_view_scale")
    fs = search_cfg.foveated_scale
    if isinstance(rec, dict) and rec.get("fixed_scale") is not None and fs.mode == "fixed" \
            and abs(float(rec["fixed_scale"]) - fs.fixed_scale) > 1e-9:
        raise ValueError(f"--cfg.search.foveated-scale.fixed-scale {fs.fixed_scale} differs from the checkpoint's "
                         f"pretraining view scale {rec}: every glimpse would be out of distribution")


@dataclass
class CanViTSearchModel:
    """``readout='policy'`` (the learned scorer) or ``'heatmap'`` (the task head) as a scanpath model."""

    episode: SearchEpisode
    readout: Literal["policy", "heatmap"]

    @property
    def name(self) -> str:
        return f"canvit_{self.readout}"

    def forced_logits(self, batch: dict, n: int, grid: int) -> Tensor:
        logits = self.episode.run(batch, n)[self.readout]
        g = int(round(logits.shape[-1] ** 0.5))
        if g != grid:
            raise ValueError(f"the {self.readout} predicts on a {g}x{g} grid but --cfg.grid is {grid}: "
                             f"compare all models at the model's grid (--cfg.grid {g})")
        return logits


def precompute(model: CanViTSearchModel, loader, *, grid: int, scene_size: int):
    """Run the teacher-forced episode over every scanpath once (batched) and hand the predictions to
    pysaliency as a :class:`~canvit.scanpath.models.GridScanpathModel`, keyed by pysaliency's
    ``(scanpath_index, history length)``: every fixation t >= 1 of every scanpath gets one."""
    from .models import GridScanpathModel
    table: dict[tuple[int, int], Any] = {}
    for batch in loader:
        batch = {k: v.to(model.episode.device) for k, v in batch.items()}
        n = int(batch["n_fix"].max()) - 1
        if n < 1:
            continue
        logits = model.forced_logits(batch, n, grid).cpu().numpy()  # [B, n, G*G], row t-1 -> fixation t
        for b, (sp, L) in enumerate(zip(batch["scanpath"].tolist(), batch["n_fix"].tolist())):
            for t in range(1, L):
                table[(sp, t)] = logits[b, t - 1]
    log.info("%s: %d predictions", model.name, len(table))
    return GridScanpathModel(table, scene_size=scene_size, name=model.name)


__all__ = ["CanViTSearchModel", "SearchEpisode", "precompute"]
