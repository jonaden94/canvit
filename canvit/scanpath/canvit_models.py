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

**The checkpoint is enough** (canvit main b029fd3+, Jonathan's self-contained checkpoints): the task and
policy configs come from the run config saved in it (``metadata["run_config"]``), rebuilt by
``harness.evaluate.parse`` exactly as ``harness.evaluate <task> --opts.ckpt`` does; the architecture from its
``model_config``; the policy from the same file (``policy_state_of``). Everything is loaded STRICTLY. A
checkpoint without a run config is refused: such a run predates the fovi port's checkpoints, so neither its
flags nor its foveation geometry are known (a run config implies canvit.core.foveation = fovi 27b1bde).

Free viewing with a SEARCH model (no cue): the readout is the uniform mixture over the cues of the per-cue
conditional densities, equal weights, nothing fitted (owner decision, 2026-11-23); named ``..._cuemix``.
"""

from __future__ import annotations

import logging
from contextlib import nullcontext
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import numpy as np
import torch
from torch import Tensor

from canvit.core import Viewpoint

from .data import clamp_to_content

log = logging.getLogger(__name__)


@dataclass
class PolicyEpisode:
    """A loaded CanViT (+ policy) of any task and its teacher-forced / sampled episode."""

    task: Any  # the task's RunTask, built from the checkpoint's run config (load)
    model: Any
    joint: Any | None  # JointPolicy; None only for a search probe (heatmap readout)
    device: torch.device
    amp: bool = True
    provenance: dict | None = None  # where the task came from (load); for the record

    @property
    def is_search(self) -> bool:
        return getattr(self.task, "name", None) == "search"

    @classmethod
    def load(cls, ckpt: Path, *, device: torch.device, need_policy: bool,
             overrides: tuple[str, ...] = ()) -> PolicyEpisode:
        """The task, model and policy of a self-contained checkpoint, as ``harness.evaluate`` builds them.
        ``overrides``: extra ``harness.evaluate`` flags that win over the run config (e.g. ``--cfg.retina <name>``)."""
        from canvit.harness.evaluate import adopt_checkpoint_provenance, parse
        from canvit.harness.infra.checkpoint import policy_state_of
        payload = torch.load(ckpt, map_location=device, weights_only=False)
        run_config = (payload.get("metadata") or {}).get("run_config")
        if run_config is None:
            raise ValueError(f"{ckpt} carries no run config (metadata['run_config'], written by canvit main since "
                             "2026-11-22). Scanpath evaluation reads the task, the policy and the foveation geometry "
                             "from it; an older checkpoint has none of them on record, so it is refused.")
        cmd, source = parse([run_config["argv"][0], "--opts.ckpt", str(ckpt), *overrides])
        task, _ = cmd.build()
        adopted = adopt_checkpoint_provenance(task.cfg, payload, source=ckpt)
        model, _ = task.build_model(device, prior_model_config=payload.get("model_config"))
        if not task.is_foveated(model):
            raise ValueError(f"{ckpt}: not a foveated model. Scanpath evaluation needs fixation policies; a "
                             "uniform-patcher policy also chooses a zoom, which humans do not (design §3).")
        _check_view_scale(payload, getattr(task.cfg, "foveated_scale", None))
        model.load_state_dict(payload["model_state"], strict=True)
        model.eval()
        joint = None
        if need_policy:
            state = policy_state_of(payload, path=Path(ckpt), device=device)
            if state is None:
                raise ValueError(f"{ckpt} holds no learned policy (its run trained none); a search probe has the "
                                 "task heatmap only (--cfg.model canvit_heatmap)")
            joint = task.build_policy(model, device=device, canvas_grid=task.canvas_grid(model),
                                      generator=torch.Generator(device=device))
            joint.load_state_dict(state)  # strict
            joint.scorer.eval()
        provenance = {"ckpt": str(ckpt), "step": payload.get("step"), "run_config_argv": run_config["argv"],
                      "run_config_drift": source["drift"] if source else None, "adopted_from_checkpoint": adopted,
                      "overrides": list(overrides), "retina": getattr(task.cfg, "retina", None),
                      "foveation": "canvit.core.foveation (fovi 27b1bde port; implied by the run config)"}
        log.info("loaded %s %s (step %s)%s", getattr(task, "name", "?"), ckpt, payload.get("step"),
                 " + policy" if joint else "")
        return cls(task=task, model=model, joint=joint, device=device, amp=getattr(task.cfg, "amp", True),
                   provenance=provenance)

    def _amp(self):
        if self.device.type == "cuda" and self.amp:  # the training loop's rule, as search's evaluate
            dt = torch.bfloat16 if torch.cuda.get_device_capability(self.device) >= (8, 0) else torch.float16
            return torch.autocast("cuda", dtype=dt)
        return nullcontext()

    @torch.no_grad()
    def _episode(self, batch: dict, n: int, next_xy, *, all_cues: int = 0) -> tuple[dict[str, Tensor], Tensor]:
        """The deploy rollout over glimpses 0..n-1, except that glimpse t >= 1 lands at
        ``next_xy(t, scores)`` ([B, 2] scene (x, y)), where ``scores`` [B, G*G] are the policy's scores for
        fixation t (None without a policy). Returns the ``policy`` (if loaded) and ``heatmap`` logits, each
        [B, n, G*G] (row t-1 read after glimpse t-1, i.e. the prediction for fixation t), and the centres
        of glimpses 0..n-1 [B, n, 2] (x, y).

        ``all_cues = K`` (search): the readouts for EVERY cue 0..K-1, [B, n, K, G*G], and ``next_xy`` gets the
        [B, K, G*G] scores of all cues. The cue enters the policy only as the scorer's ``cond`` (the encoder's
        features, and the backbone state, do not depend on it) and the heatmap only through the head, so the
        backbone runs once and the scorer and head run on the same features for each cue. The selector runs
        with cue 0, but its choice is never used: ``next_xy`` places every glimpse t >= 1."""
        from canvit.harness.rollout.engine import _to_vp
        from canvit.harness.rollout.eval_viewpoints import deploy_selector
        from canvit.harness.rollout.viewpoint import ViewpointType
        from canvit.search.task import _canvas_hidden

        model, images, cue = self.model, batch["image"], batch["cue"]
        B, cg = images.shape[0], self.task.canvas_grid(model)
        backbone = getattr(model, "canvit", model)  # task wrappers hold it; distill's model IS the backbone
        if all_cues:
            assert self.is_search, "all_cues is for search models"
            cue = torch.zeros_like(cue)
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
            if scores is not None and all_cues:
                feats = sel.last_aux["feats"].float().repeat_interleave(all_cues, 0)  # (b, k) rows, k fastest
                k_cue = torch.arange(all_cues, device=images.device).repeat(B)
                scores = sel.net(feats, cond=embed(k_cue)).float().reshape(B, all_cues, -1)
                out["policy"].append(scores)
            elif scores is not None:
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
            if self.is_search and all_cues:
                hidden = _canvas_hidden(model, state, cg).float().repeat_interleave(all_cues, 0)
                heat, _ = model.head(hidden, torch.arange(all_cues, device=images.device).repeat(B))
                out["heatmap"].append(heat.float().reshape(B, all_cues, -1))
            elif self.is_search:
                heat, _ = model.head(_canvas_hidden(model, state, cg).float(), cue)
                out["heatmap"].append(heat.float().reshape(B, -1))
        return {k: torch.stack(v, 1) for k, v in out.items()}, torch.stack(centres, 1)

    def run(self, batch: dict, n: int, *, all_cues: int = 0) -> dict[str, Tensor]:
        """Teacher forcing: glimpses at the anchor and the HUMAN fixations 1..n-1 (clamped onto the image);
        returns the logits of :meth:`_episode` (``all_cues``: for every cue, [B, n, K, G*G])."""
        human = batch["fix"].nan_to_num(0.0)  # rows shorter than t: any finite centre, ignored downstream
        return self._episode(batch, n, lambda t, scores: clamp_to_content(human[:, t], batch["content"]),
                             all_cues=all_cues)[0]

    def sample(self, batch: dict, n: int, *, temperature: float, generator: torch.Generator | None = None,
               cue_mixture: bool = False) -> Tensor:
        """The model's OWN scanpath of n fixations (the anchor + n-1 chosen) [B, n, 2] scene (x, y), drawn from
        the SAME density the likelihood scores: fixation t is a cell drawn with p(cell) = softmax(scores /
        temperature + log pixels of the image in the cell) (``models.grid_log_density``), placed at the cell
        centre clamped onto the image. ``cue_mixture`` (a search model on free viewing): p(cell) is the
        equal-weight mixture of the per-cue p(cell) at every step (``CanViTModel.mixed_logits``), from one
        backbone pass. Temperature 0 = the argmax of the scores (cue mixture: of the mixed p), i.e. the deploy
        rollout for one cue. Needs the policy and ``batch["hw"]``."""
        import math

        if self.joint is None:
            raise ValueError("sampling scanpaths needs the learned policy")
        K = 0
        if cue_mixture:
            from canvit.search.data import COCO_SEARCH18_CLASSES
            K = len(COCO_SEARCH18_CLASSES)
        S = batch["image"].shape[-1]

        def draw(t, scores):  # scores [B, G*G], or [B, K, G*G] for the cue mixture
            g = int(round(scores.shape[-1] ** 0.5))
            if temperature == 0 and not K:
                idx = scores.argmax(-1)
            else:
                count = _cell_counts(batch, S, g)
                logcount = torch.from_numpy(count).to(scores.device).float().log()
                if K:
                    logcount = logcount[:, None]
                logp = torch.log_softmax(scores.float() / (temperature or 1.0) + logcount, -1)
                if K:
                    logp = torch.logsumexp(logp, 1) - math.log(K)
                idx = logp.argmax(-1) if temperature == 0 else torch.multinomial(
                    logp.exp(), 1, generator=generator).squeeze(1)
            c = (torch.arange(g, device=scores.device, dtype=torch.float32) + 0.5) * (2.0 / g) - 1.0
            xy = torch.stack([c[idx % g], c[idx // g]], -1)  # cell (row, col) = (idx // g, idx % g)
            return xy if temperature == 0 and not K else clamp_to_content(xy, batch["content"])

        return self._episode(batch, n, draw, all_cues=K)[1]


def _frame(batch: dict, b: int) -> tuple[int, int] | None:
    """Sample b's display frame (data.ScanpathDataset ``frame``; (0, 0) = none)."""
    fr = tuple(int(v) for v in batch["frame"][b].tolist()) if "frame" in batch else (0, 0)
    return None if fr == (0, 0) else fr


def _cell_counts(batch: dict, scene: int, grid: int) -> np.ndarray:
    """[B, G*G] pixels of each sample's stimulus in each cell, in the sample's display frame."""
    from .models import _pixel_cells
    return np.stack([_pixel_cells(int(h), int(w), scene, grid, _frame(batch, b))[1]
                     for b, (h, w) in enumerate(batch["hw"].tolist())])


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
    model. ``cue_mixture``: a search model on free-viewing data -- every cue is run and the per-cue conditional
    densities are mixed with equal weights (:meth:`mixed_logits`)."""

    episode: PolicyEpisode
    readout: Literal["policy", "heatmap"]
    cue_mixture: bool = False

    def __post_init__(self):
        if self.readout == "heatmap" and not self.episode.is_search:
            raise ValueError(f"readout 'heatmap' is the search task head; a {self.episode.task.name} model has "
                             "only the 'policy' readout")
        if self.cue_mixture and not self.episode.is_search:
            raise ValueError("cue_mixture is for search models (a cue per class); this model takes no cue")

    @property
    def name(self) -> str:
        return f"canvit_{getattr(self.episode.task, 'name', '?')}_{self.readout}" + ("_cuemix" if self.cue_mixture
                                                                                     else "")

    def forced_logits(self, batch: dict, n: int, grid: int | None = None) -> Tensor:
        """[B, n, G*G] logits on the model's own G x G grid; ``grid`` (if given) must be G."""
        if self.cue_mixture:
            logits = self.mixed_logits(batch, n)
        else:
            logits = self.episode.run(batch, n)[self.readout]
        g = int(round(logits.shape[-1] ** 0.5))
        if grid is not None and g != grid:
            raise ValueError(f"the {self.readout} predicts on a {g}x{g} grid but --cfg.grid is {grid}: "
                             f"compare all models at the model's grid (--cfg.grid {g})")
        return logits

    def mixed_logits(self, batch: dict, n: int) -> Tensor:
        """[B, n, G*G] log of (1/K) sum_k p_k: p_k the per-pixel density (``models.grid_log_density``) of the
        episode run with cue k (k = every COCO-Search18 class), taken per cell. Already normalised over the
        image's pixels, so ``grid_log_density`` returns it unchanged; cells off the image are set to 0 (they
        hold no pixel). One backbone pass; the scorer / head run once per cue (:meth:`PolicyEpisode._episode`)."""
        import math

        from canvit.search.data import COCO_SEARCH18_CLASSES

        if not (batch["cue"] < 0).all():
            raise ValueError("cue_mixture is for free-viewing trials (cue -1); this batch has cues")
        K, S = len(COCO_SEARCH18_CLASSES), batch["image"].shape[-1]
        lg = self.episode.run(batch, n, all_cues=K)[self.readout].double()  # [B, n, K, G*G], one backbone pass
        g = int(round(lg.shape[-1] ** 0.5))
        count = _cell_counts(batch, S, g)
        logcount = torch.from_numpy(count).to(lg.device).double().log()[:, None, None]  # [B, 1, 1, G*G]
        on = logcount.isfinite()
        ld = torch.where(on, torch.log_softmax(lg + logcount, -1) - logcount, -math.inf)  # per-cue pixel log density
        mix = torch.logsumexp(ld, dim=2)  # [B, n, G*G]
        on = on[:, :, 0]
        return torch.where(on, mix - math.log(K), 0.0).float()


def precompute(model: CanViTModel, loader, *, grid: int | None = None, scene_size: int, frame: tuple[int, int] | None = None):
    """Run the teacher-forced episode over every scanpath once (batched) and hand the predictions to
    pysaliency as a :class:`~canvit.scanpath.models.GridScanpathModel`, keyed by pysaliency's
    ``(scanpath_index, history length)``: every fixation t >= 1 of every scanpath gets one. ``grid``: None = the
    model's own grid, else a check (:meth:`CanViTModel.forced_logits`)."""
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
    return GridScanpathModel(table, scene_size=scene_size, name=model.name, frame=frame)


def sample_scanpaths(ep: PolicyEpisode, loader, *, temperature: float, seed: int, max_len: int | None = None,
                     cue_mixture: bool = False) -> dict[str, np.ndarray]:
    """The model's OWN scanpaths (:meth:`PolicyEpisode.sample`), one per human scanpath of ``loader``, each as long
    as that human's (the model has no stop; at most ``max_len``), fixation 0 the centre anchor. ``cue_mixture``
    (a search model on free viewing): every fixation from the per-step cue mixture, the scored model. Returns
    ``xs``, ``ys`` [N, L] in STIMULUS pixels (NaN-padded), ``scanpath`` (the human scanpath's index), ``n_fix`` and
    ``cue`` [N] (-1: the mixture)."""
    from .data import letterbox_geometry
    gen = torch.Generator(device=ep.device).manual_seed(seed)
    xs, ys, sp, lens, cues = [], [], [], [], []
    for batch in loader:
        batch = {k: v.to(ep.device) for k, v in batch.items()}
        L = batch["n_fix"].clamp(max=max_len) if max_len else batch["n_fix"]
        xy = ep.sample(batch, int(L.max()), temperature=temperature, generator=gen,
                       cue_mixture=cue_mixture).cpu().numpy()  # scene (x, y)
        S = batch["image"].shape[-1]
        for b, (h, w) in enumerate(batch["hw"].tolist()):
            ch, cw, top, left = letterbox_geometry(h, w, S, _frame(batch, b))  # the inverse of data.pixel_to_scene
            px = ((xy[b, :, 0] + 1) / 2 * S - left) * (w / cw)
            py = ((xy[b, :, 1] + 1) / 2 * S - top) * (h / ch)
            n = int(L[b])
            px[n:], py[n:] = np.nan, np.nan
            xs.append(px), ys.append(py)
        sp += batch["scanpath"].tolist()
        lens += L.tolist()
        cues += batch["cue"].tolist()
    width = max(len(x) for x in xs)
    pad = lambda a: np.stack([np.pad(x, (0, width - len(x)), constant_values=np.nan) for x in a])  # noqa: E731
    return {"xs": pad(xs), "ys": pad(ys), "scanpath": np.array(sp), "n_fix": np.array(lens), "cue": np.array(cues),
            "temperature": np.array(temperature), "seed": np.array(seed)}


__all__ = ["CanViTModel", "PolicyEpisode", "precompute", "sample_scanpaths"]
