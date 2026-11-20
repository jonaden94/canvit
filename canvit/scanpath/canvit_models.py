"""CanViT checkpoints of ANY task (search, distill/reconstruction, ade20k, in1k) as scanpath models
(design §3, §12).

Readouts of one teacher-forced episode, after the glimpses at the human's fixations 0..t-1:

- ``policy``:  the learned scorer's logits -- the distribution the deployed policy picks fixation t
               from (its argmax at deploy). Every task with a trained policy.
- ``heatmap``: search only: the task head's target heatmap -- "where is the target", the map
               ``greedy`` fixates the argmax of. Needs no policy (a ``--preset probe`` checkpoint suffices).

**Same code path as the deploy rollout** (``harness.rollout.eval_viewpoints.
deploy_rollout_viewpoints``): the deploy selector (argmax, ``prime_on_policy=1``) is called at
every glimpse exactly as there, so the scorer, its rolling encoder reference (``enc.prev``), the
cue embedding (search) and the view scale are the deployed ones. Only the CENTRE of each viewpoint
t >= 1 is replaced (by the human's fixation, or by a draw from the policy in :meth:`PolicyEpisode.sample`).
t0 is the FULL anchor (the centre at the training view scale), standing in for the human's start
fixation. ``test_canvit_models.py`` pins this: forcing the policy's own argmax reproduces the
deploy rollout's scores exactly.

One glimpse is ``backbone(image=<whole image>, state, viewpoint)`` for every task -- the step each
task's own evaluation takes (``search/task.py``, ``ade20k/task.py``, ``in1k/eval.py``,
``distill/validate.py``); the backbone is ``model.canvit`` for the task wrappers and the model
itself for distill. Foveated models only (they foveate the whole image themselves); a uniform-patcher
policy also picks a zoom, which has no human counterpart (owner, 2026-11-18), so it is refused.

Forced centres are the human's continuous positions (clamped onto the image), not snapped to the
policy's candidate cells: the backbone accepts any centre, and the human saw that one.

Built with the TRAINING constructors (each task's ``build_model`` / ``build_policy``, from the run's own
task and policy flags, ``sources.py``), then loaded STRICTLY: the checkpoint does not record the
policy's architecture, so a mismatched flag must fail loudly rather than score a half-initialised scorer.
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
class PolicyEpisode:
    """A loaded CanViT (+ policy) of any task and its teacher-forced / sampled episode."""

    task: Any  # the task's RunTask (sources.py builds it as the training CLI does)
    model: Any
    joint: Any | None  # JointPolicy; None only for a search probe (heatmap readout)
    device: torch.device
    amp: bool = True

    @property
    def is_search(self) -> bool:
        return getattr(self.task, "name", None) == "search"

    @classmethod
    def load(cls, *, task, ckpt: Path, device: torch.device, need_policy: bool) -> PolicyEpisode:
        payload = torch.load(ckpt, map_location=device, weights_only=False)
        # distill rebuilds its architecture FROM the checkpoint (as harness.evaluate does); the others ignore it
        model, _ = task.build_model(device, prior_model_config=payload.get("model_config"))
        if not task.is_foveated(model):
            raise ValueError(f"{ckpt}: not a foveated model. Scanpath evaluation needs fixation policies; a "
                             "uniform-patcher policy also chooses a zoom, which humans do not (design §3).")
        _check_view_scale(payload, getattr(task.cfg, "foveated_scale", None))
        model.load_state_dict(payload["model_state"], strict=True)
        model.eval()
        joint = None
        sidecar = Path(ckpt).with_suffix(".policy.pt")
        if need_policy:
            if not sidecar.exists():
                raise FileNotFoundError(f"{sidecar} not found: the policy is saved beside the checkpoint as "
                                        "<ckpt>.policy.pt; a probe-only run has none (search: --cfg.model "
                                        "canvit_heatmap)")
            joint = task.build_policy(model, device=device, canvas_grid=task.canvas_grid(model),
                                      generator=torch.Generator(device=device))
            joint.load_state_dict(torch.load(sidecar, map_location=device, weights_only=False))  # strict
            joint.scorer.eval()
        log.info("loaded %s %s (step %s)%s", getattr(task, "name", "?"), ckpt, payload.get("step"),
                 " + policy" if joint else "")
        return cls(task=task, model=model, joint=joint, device=device, amp=getattr(task.cfg, "amp", True))

    def _amp(self):
        if self.device.type == "cuda" and self.amp:  # the training loop's rule, as search's evaluate
            dt = torch.bfloat16 if torch.cuda.get_device_capability(self.device) >= (8, 0) else torch.float16
            return torch.autocast("cuda", dtype=dt)
        return nullcontext()

    @torch.no_grad()
    def _episode(self, batch: dict, n: int, next_xy) -> tuple[dict[str, Tensor], Tensor]:
        """The deploy rollout over glimpses 0..n-1, except that glimpse t >= 1 lands at
        ``next_xy(t, scores)`` ([B, 2] scene (x, y)), where ``scores`` [B, G*G] are the policy's scores for
        fixation t (None without a policy). Returns the ``policy`` (if loaded) and ``heatmap`` logits, each
        [B, n, G*G] (row t-1 read after glimpse t-1, i.e. the prediction for fixation t), and the centres
        of glimpses 0..n-1 [B, n, 2] (x, y)."""
        from canvit.harness.rollout.engine import _to_vp
        from canvit.harness.rollout.eval_viewpoints import deploy_selector
        from canvit.harness.rollout.viewpoint import ViewpointType
        from canvit.search.task import _canvas_hidden

        model, images, cue = self.model, batch["image"], batch["cue"]
        B, cg = images.shape[0], self.task.canvas_grid(model)
        backbone = getattr(model, "canvit", model)  # task wrappers hold it; distill's model IS the backbone
        if self.is_search and (cue < 0).any():
            raise ValueError("a search model needs a cue on every trial; this dataset has free-viewing trials")
        if self.joint is None and not self.is_search:
            raise ValueError("no policy loaded: only a search model has a policy-free readout (heatmap)")
        sel = deploy_selector(self.joint) if self.joint is not None else self.task._random_selector()
        ctx = sel.start_rollout(t0_type=ViewpointType.FULL, batch_size=B, device=images.device)
        if self.joint is not None and self.is_search:
            embed = self.task._policy_embed(model)
            ctx.cond = lambda: embed(cue)
        out: dict[str, list[Tensor]] = ({"heatmap": []} if self.is_search else {}) | (
            {"policy": []} if self.joint is not None else {})
        centres: list[Tensor] = []
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
            scores = sel.last_aux["scores"].float().reshape(B, -1) if t > 0 and self.joint is not None else None
            if scores is not None:
                out["policy"].append(scores)
            if t == n:
                break
            vp = _to_vp(named)
            if t > 0:
                xy = next_xy(t, scores)
                vp = Viewpoint(centers=xy.flip(-1).to(vp.centers.dtype), scales=vp.scales)  # (y, x)
            centres.append(vp.centers.flip(-1).float())
            if state is None:
                state = model.init_state(batch_size=B, canvas_grid_size=cg)
            with self._amp():
                state = backbone(image=images, state=state, viewpoint=vp).state
            if self.is_search:
                heat, _ = model.head(_canvas_hidden(model, state, cg).float(), cue)
                out["heatmap"].append(heat.float().reshape(B, -1))
        return {k: torch.stack(v, 1) for k, v in out.items()}, torch.stack(centres, 1)

    def run(self, batch: dict, n: int) -> dict[str, Tensor]:
        """Teacher forcing: glimpses at the anchor and the HUMAN fixations 1..n-1 (clamped onto the image);
        returns the logits of :meth:`_episode`."""
        human = batch["fix"].nan_to_num(0.0)  # rows shorter than t: any finite centre, ignored downstream
        return self._episode(batch, n, lambda t, scores: clamp_to_content(human[:, t], batch["content"]))[0]

    def sample(self, batch: dict, n: int, *, temperature: float, generator: torch.Generator | None = None) -> Tensor:
        """The model's OWN scanpath of n fixations (the anchor + n-1 chosen) [B, n, 2] scene (x, y): fixation t
        is the centre of a cell drawn from softmax(scores / temperature); temperature 0 = the argmax, i.e. the
        deploy rollout. Needs the policy."""
        if self.joint is None:
            raise ValueError("sampling scanpaths needs the policy (<ckpt>.policy.pt)")

        def draw(t, scores):
            g = int(round(scores.shape[1] ** 0.5))
            if temperature == 0:
                idx = scores.argmax(1)
            else:
                idx = torch.multinomial(torch.softmax(scores / temperature, 1), 1, generator=generator).squeeze(1)
            c = (torch.arange(g, device=scores.device, dtype=torch.float32) + 0.5) * (2.0 / g) - 1.0
            return torch.stack([c[idx % g], c[idx // g]], -1)  # cell (row, col) = (idx // g, idx % g)

        return self._episode(batch, n, draw)[1]


def _check_view_scale(payload: dict, fs) -> None:
    """Refuse a fixed view scale that differs from the one the checkpoint records (design §6)."""
    rec = (payload.get("metadata") or {}).get("pretrain_view_scale")
    if fs is None:
        return
    if isinstance(rec, dict) and rec.get("fixed_scale") is not None and fs.mode == "fixed" \
            and abs(float(rec["fixed_scale"]) - fs.fixed_scale) > 1e-9:
        raise ValueError(f"the task's foveated_scale.fixed_scale {fs.fixed_scale} differs from the checkpoint's "
                         f"pretraining view scale {rec}: every glimpse would be out of distribution")


@dataclass
class CanViTModel:
    """``readout='policy'`` (the learned scorer, any task) or ``'heatmap'`` (search's task head) as a scanpath
    model."""

    episode: PolicyEpisode
    readout: Literal["policy", "heatmap"]

    def __post_init__(self):
        if self.readout == "heatmap" and not self.episode.is_search:
            raise ValueError(f"readout 'heatmap' is the search task head; a {self.episode.task.name} model has "
                             "only the 'policy' readout")

    @property
    def name(self) -> str:
        return f"canvit_{getattr(self.episode.task, 'name', '?')}_{self.readout}"

    def forced_logits(self, batch: dict, n: int, grid: int) -> Tensor:
        logits = self.episode.run(batch, n)[self.readout]
        g = int(round(logits.shape[-1] ** 0.5))
        if g != grid:
            raise ValueError(f"the {self.readout} predicts on a {g}x{g} grid but --cfg.grid is {grid}: "
                             f"compare all models at the model's grid (--cfg.grid {g})")
        return logits


def precompute(model: CanViTModel, loader, *, grid: int, scene_size: int):
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


__all__ = ["CanViTModel", "PolicyEpisode", "precompute"]
