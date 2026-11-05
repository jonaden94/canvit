"""One validation-viewpoint interface for all three tasks.

Before this module each task picked its validation trajectory a different way, not by
design but by inheritance — every task's ``evaluate()`` was lifted from a different
ancestor and kept that ancestor's habit:

  * distill  <- the old pretrain loop's ``validate()``  -> quadtree C2F (uniform) /
                the deterministic centre+3x3 fixation trajectory (foveated)
  * ade20k   <- the specialize probe, which TRAINED on random viewpoints, so it
                validated on random too -> IID random, with no knob at all
  * in1k     <- canvit_eval's deploy convention -> C2F, and it already had the knob

Nothing about the tasks requires this. The *metrics* genuinely differ (mIoU vs top-1/5
vs cosine-to-teacher) and stay task-owned; the viewpoint sequence is a rollout concern
and belongs here. This module is that seam: one option set, one place where the
patcher-awareness rule lives, and ``"policy"`` — deploy the learned scorer by argmax —
available to every task instead of only to the standalone RL trainer.

**Defaults are deliberately NOT unified.** Each task keeps exactly the trajectory it
used before (see ``HISTORICAL_DEFAULTS``): flipping ade20k to C2F would silently break
comparability with every specialize probe number and every exp24 run, and flipping
distill would break the exp22/23/26 val curves. The knob is shared; the default is
per-task and documented. ``"auto"`` means "whatever this task has always done", so
every existing config is a no-op through this module.
"""

from __future__ import annotations

import logging
from dataclasses import replace
from functools import lru_cache
from typing import TYPE_CHECKING, Any, Literal

import torch

from canvit.core.viewpoint import Viewpoint

log = logging.getLogger(__name__)

if TYPE_CHECKING:  # the task configs import EvalPolicy from here, so stay dependency-light
    from canvit.harness.config import FoveatedScaleConfig

EvalPolicy = Literal["auto", "coarse_to_fine", "fine_to_coarse", "random", "full",
                     "fixation_grid", "entropy_coarse_to_fine", "policy"]

OPEN_LOOP: tuple[str, ...] = ("coarse_to_fine", "fine_to_coarse", "random", "full",
                              "fixation_grid")
"""Policies whose whole trajectory is known before the rollout starts.

Each name is a PRESET: one joint (center, scale) trajectory that has a published number
somewhere. They are not composable axes — the safe-box law draws
``centers = rand * (1 - scale)`` and the quadtree takes both off one tree, so center and
scale are jointly generated (eval-merge doc §5, Stage 3). What IS composable on top:
``t0``, and ``override_scale``."""

T0Anchor = Literal["full_anchor", "trajectory"]
"""Whether the episode opens on the full-scene anchor. ``None`` (the default everywhere)
means "whatever this preset always did", so it is a no-op unless set.

Only ``random`` and ``fine_to_coarse`` have a choice here: the quadtree's level 0 IS the
full scene, and ``fixation_grid`` / ``full`` open on their own first element. Setting it
for those would be a flag that does nothing, so they raise instead — see
``_resolve_t0``. This is the knob that separates canvit's ``random`` (anchored)
from canvit_eval's (not), a 0.0205 difference at t0 (F3)."""

_T0_APPLIES: tuple[str, ...] = ("random", "fine_to_coarse")

CLOSED_LOOP: tuple[str, ...] = ("policy", "entropy_coarse_to_fine")
"""Policies that need the live canvas state to pick the next glimpse, so their
trajectory cannot be precomputed. Both are driven by a ``(t, state) -> Viewpoint``
chooser through the task's rollout."""

PAPER_TABLE4_C64: dict[str, list[float]] = {
    # arXiv:2603.22570 Table 4, canvas 64^2, ADE20K val, squish-512, t = 0..4.
    # Mirrored from CanViT-PyTorch-RL/docs/paper-tables.md, which records them as
    # "verbatim targets for harness verification". Use as the validation target for
    # any baseline implemented here, and as dashed reference lines on Figure-4B plots.
    "entropy_coarse_to_fine": [39.6, 42.2, 43.3, 44.1, 44.7],   # EG-C2F (deterministic)
    "coarse_to_fine":         [39.6, 41.3, 42.5, 43.6, 44.7],   # C2F (paper: mean of n=10)
}
"""Paper row -> our policy name, for the rows we can actually reproduce.

VERIFIED 2026-07-30 on full ADE20K val (squish-512, c64, eval batch 32):

    EG-C2F  ours 39.58 42.22 43.31 44.05 44.67   vs paper  max|delta| = 0.05
    C2F     ours 39.58 41.23 42.54 43.54 44.71   vs paper  max|delta| = 0.07

EG-C2F is DETERMINISTIC in the paper, so that agreement is a real validation of the port,
not a coincidence of averaging. C2F is stochastic there (mean of n=10), so 0.07 is within
its CI.

**``random`` is NOT the paper's F-IID — do not label it as such.** It measures
39.58/41.37/42.18/42.84/43.42, i.e. +0.17..+0.42 ABOVE the F-IID row, growing with t.
``open_loop_viewpoints`` does pass ``start_with_full_scene=True`` so t0 matches, but the
glimpses then follow the safe-box AREA law (p(s) ~ 1-s over [min_scale, max_scale]) rather
than F-IID's fixed fovea-sized scale — a different, measurably stronger random policy.
The paper's F-IID and R-IID rows are both unreachable from this module today."""

HISTORICAL_DEFAULTS: dict[str, tuple[str, str]] = {
    # task        (uniform patcher,   foveated/square patcher)
    "distill":    ("coarse_to_fine",  "fixation_grid"),
    "ade20k":     ("random",          "random"),
    "in1k":       ("coarse_to_fine",  "coarse_to_fine"),
}
"""What ``"auto"`` resolves to, per task and patcher — i.e. what each task did before
this module existed. Written down in ONE table so the divergence is visible instead of
scattered across three ``evaluate()`` methods. Notes on the non-obvious entries:

* **distill/foveated -> fixation_grid.** The quadtree's varying scales are out of
  distribution for a fixed-scale foveated model (``fix_size = scale * H``), so the old
  loop used a deterministic centre + shuffled-3x3 trajectory at the TRAINING scale.
* **ade20k -> random.** Inherited from the specialize probe, which trained on random
  viewpoints. Consistent for a probe; wrong the moment a policy is in the loop, which
  is why ``"policy"`` exists.
* **in1k/foveated -> coarse_to_fine.** This is the known OOD footgun that distill
  avoids, retained on purpose: exp25's foveated in1k arrays were measured under it and
  changing the default would make them non-comparable. Pass ``--cfg.eval-policy
  fixation_grid`` (or ``full``) for a scale-pinned foveated deploy.
"""


def resolve(policy: str, *, task: str, is_foveated: bool) -> str:
    """Map ``"auto"`` onto the task's historical trajectory; pass anything else through."""
    assert task in HISTORICAL_DEFAULTS, f"unknown task {task!r}"
    if policy == "auto":
        uniform_default, foveated_default = HISTORICAL_DEFAULTS[task]
        return foveated_default if is_foveated else uniform_default
    assert policy in OPEN_LOOP or policy in CLOSED_LOOP, f"unknown eval policy {policy!r}"
    return policy


def make_random_viewpoints(
    batch_size: int, device: torch.device, n: int, *,
    min_scale: float, max_scale: float, start_with_full_scene: bool,
    is_foveated: bool = False,
    foveated_scale: FoveatedScaleConfig | None = None,
) -> list[Viewpoint]:
    """The probe rollout's viewpoint distribution, PATCHER-AWARE (eval-merge doc §2).

    Uniform patcher: specialize's law (core ``random_viewpoints`` — the same
    L²-safe-box law as pretrain's ``Viewpoint.random``).

    Foveated/square: delegated to :class:`RandomSelector`, the canonical random
    policy extracted from the pretraining loop in P1, so the probe sees exactly
    the scale/center law the backbone was trained under. This is not cosmetic:
    the foveated patcher derives its fixation window from the viewpoint scale
    (``fix_size = scale * H``), so feeding it the uniform safe-box law (scales
    ≤ 1) after it was pretrained at, say, ``fixed_scale=2.0`` puts every glimpse
    out of distribution — measured as mIoU *decreasing* monotonically with more
    glimpses (job 15025338; see p2-notes "foveated scale mismatch").

    Imports are function-local for the reason stated at the top of this module: the task
    configs import ``EvalPolicy`` from here, so a module-level ``harness.config`` import
    would close a cycle.
    """
    from canvit.core.policies import random_viewpoints
    from canvit.harness.config import FoveatedScaleConfig
    from canvit.harness.rollout.selector import RandomSelector
    from canvit.harness.rollout.viewpoint import ViewpointType

    if not is_foveated:
        return random_viewpoints(
            batch_size, device, n,
            min_scale=min_scale, max_scale=max_scale,
            start_with_full_scene=start_with_full_scene,
        )
    sel = RandomSelector(
        is_foveated=True,
        foveated_scale=foveated_scale or FoveatedScaleConfig(),
        min_viewpoint_scale=min_scale,
    )
    t0 = ViewpointType.FULL if start_with_full_scene else ViewpointType.RANDOM
    ctx = sel.start_rollout(t0_type=t0, batch_size=batch_size, device=device)
    types = [t0] + [ViewpointType.RANDOM] * (n - 1)
    return [
        sel.select(
            vp_type=vt, ctx=ctx, t=t, batch_size=batch_size, device=device, state=None  # type: ignore[arg-type]
        )
        for t, vt in enumerate(types)
    ]


@lru_cache(maxsize=8)
def _warn_off_scale(policy: str, trained: float, got: str) -> None:
    """Once per (policy, scale) per process — this sits inside a per-batch call."""
    log.warning(
        "eval_policy=%r puts a FIXED-SCALE foveated/square model OUT OF DISTRIBUTION: it "
        "was pretrained at view scale %s but this trajectory uses %s. The foveation window "
        "is fix_size = scale * H, so every glimpse is off-scale; the symptom is a metric "
        "that FALLS as glimpses accumulate. Measured cost on exp33/exp34: -0.114 top1 "
        "(in1k) and -0.128 mIoU at t9 (ade20k). Pass --cfg.eval-override-scale %s to keep "
        "this policy's CENTERS at the model's own scale, or use eval_policy='fixation_grid'.",
        policy, trained, got, trained)


def _check_scale_in_distribution(
    vps: list[Viewpoint], *, policy: str, is_foveated: bool, foveated_scale: Any,
) -> None:
    """Warn when the generated trajectory's scales do not match what a fixed-scale
    foveated model trained at.

    Checks the GENERATED SCALES rather than the policy name on purpose: a name-based table
    goes stale the moment a policy is added or an ``override_scale`` is passed, and this
    has to stay right for the combinations Stage 3 opens up (eval-merge doc §5). Silent for
    uniform models (their OOD axis is the glimpse crop in pixels, not the view scale) and
    for sampled-scale modes, which are scale-robust by construction.
    """
    if not is_foveated or foveated_scale is None:
        return
    if getattr(foveated_scale, "mode", None) != "fixed":
        return
    trained = getattr(foveated_scale, "fixed_scale", None)
    if trained is None or not vps:
        return
    scales = torch.cat([v.scales.reshape(-1) for v in vps])
    if torch.allclose(scales, torch.full_like(scales, float(trained))):
        return
    lo, hi = scales.min().item(), scales.max().item()
    got = f"{lo:g}" if lo == hi else f"scales in [{lo:g}, {hi:g}]"
    _warn_off_scale(policy, float(trained), got)


def _resolve_t0(policy: str, t0: T0Anchor | None) -> bool:
    """``start_with_full_scene`` for ``policy``. Unset => the preset's own historical value.

    An explicit value on a preset that cannot honour it is a HARD ERROR, not a warning: the
    request would otherwise be silently dropped and the caller would read the resulting
    number as having come from the config they typed (eval-merge doc §5, Stage 3).
    """
    if t0 is None:
        return True   # every preset here has historically opened on the full scene
    if policy not in _T0_APPLIES:
        raise ValueError(
            f"t0={t0!r} does not apply to eval_policy={policy!r}: its first glimpse IS its "
            f"own trajectory's first element (the quadtree's level 0 is the full scene; "
            f"fixation_grid opens on the centre fixation; full is the full scene). Passing "
            f"it would be silently ignored. It applies to {list(_T0_APPLIES)}.")
    return t0 == "full_anchor"


def open_loop_viewpoints(
    policy: str,
    *,
    batch_size: int,
    device: torch.device,
    n: int,
    is_foveated: bool,
    foveated_scale: FoveatedScaleConfig,
    min_scale: float = 0.05,
    max_scale: float = 1.0,
    foveated_eval_scale: float = 1.0,
    override_scale: float | None = None,
    t0: T0Anchor | None = None,
) -> list[Viewpoint]:
    """The precomputed trajectory for every policy except ``"policy"``.

    Delegates to the existing, tested generators rather than reimplementing them, so
    each option is bit-identical to the task that used to own it.

    ``override_scale`` replaces every generated scale while keeping the generated
    CENTERS, the same knob ``canvit_eval``'s ``EpisodeConfig.override_scale`` provides.
    It exists for one case: deploying a FIXED-SCALE FOVEATED backbone under a policy
    whose scales it never trained on. C2F is a quadtree over {1.0, 0.5, 0.25}, and a
    foveated model sets ``fix_size = scale * H``, so without pinning every glimpse is out
    of distribution and mIoU decays as glimpses accumulate. Pinning turns "C2F" into
    "C2F's fixation sequence at the model's own scale", which is the comparable
    measurement.

    Defaults to ``None`` = exact no-op. Deliberately NOT automatic for foveated models:
    ``HISTORICAL_DEFAULTS["in1k"]`` sends foveated runs to C2F unpinned on purpose (see
    that table's notes), and silently pinning would make exp25/exp29 non-comparable.
    """
    from canvit.core.policies import fine_to_coarse_viewpoints, repeated_full_scene
    from canvit.harness.rollout.viewpoint import make_eval_viewpoints, make_eval_viewpoints_foveated

    def _pin(vps: list[Viewpoint]) -> list[Viewpoint]:
        # `replace` rather than `Viewpoint(...)`: the generators return two different
        # Viewpoint classes (the harness one carries a debugging `name`, canvit.core's
        # does not), so naming fields here would break for one of them.
        if override_scale is None:
            return vps
        return [replace(v, scales=torch.full_like(v.scales, override_scale)) for v in vps]

    def _out(vps: list[Viewpoint]) -> list[Viewpoint]:
        vps = _pin(vps)
        _check_scale_in_distribution(vps, policy=policy, is_foveated=is_foveated,
                                     foveated_scale=foveated_scale)
        return vps

    anchor = _resolve_t0(policy, t0)

    if policy == "coarse_to_fine":
        return _out(make_eval_viewpoints(batch_size, device, n_viewpoints=n))
    if policy == "fine_to_coarse":
        # The quadtree walked the other way: finest level first, full scene last. Core has
        # had the generator all along; this repo simply never exposed it.
        vps = fine_to_coarse_viewpoints(batch_size, device, n)
        if anchor:
            vps = [Viewpoint(centers=torch.zeros(batch_size, 2, device=device),
                             scales=torch.ones(batch_size, device=device))] + vps[:-1]
        return _out(vps)
    if policy == "fixation_grid":
        return _out(make_eval_viewpoints_foveated(batch_size, device, n_viewpoints=n,
                                                 scale=foveated_eval_scale))
    if policy == "full":
        return _out(repeated_full_scene(batch_size, device, n))
    if policy == "random":
        from canvit.harness.config import FoveatedScaleConfig
        # Only the foveated branch reads it, and there a WRONG scale is not a soft
        # mismatch — `fix_size = scale * H` puts every glimpse out of distribution and
        # mIoU falls as glimpses accumulate (job 15025338). So silently defaulting it for
        # a foveated model would hide the exact bug this codebase already paid for once.
        assert not (is_foveated and foveated_scale is None), (
            "eval_policy='random' on a foveated/square model needs the pretraining "
            "foveated_scale; defaulting it would put every glimpse out of distribution.")
        return _out(make_random_viewpoints(
            batch_size, device, n, min_scale=min_scale, max_scale=max_scale,
            start_with_full_scene=anchor, is_foveated=is_foveated,
            foveated_scale=foveated_scale or FoveatedScaleConfig(),
        ))
    if policy in CLOSED_LOOP:
        raise ValueError(
            f"eval_policy={policy!r} is closed-loop: the next viewpoint depends on the "
            "canvas state, so it cannot be precomputed. Drive it through the task's "
            "closed-loop rollout instead."
        )
    raise ValueError(f"unknown eval policy: {policy!r}")


def deploy_selector(joint: Any) -> Any:
    """The learned policy in DEPLOY configuration: pure argmax, no exploration.

    ``prime_on_policy=1.0`` is the deployed rule (and consumes no RNG, so a deploy eval
    cannot perturb the training stream). The scorer is switched to eval mode by
    :func:`deploy_rollout_viewpoints`, which owns the train/eval restore.
    """
    from dataclasses import replace

    assert joint is not None and getattr(joint, "scorer", None) is not None, (
        "eval_policy='policy' needs a trained scorer, but this run has no policy. Either "
        "train one (--preset policy_only / joint) or pick an open-loop eval policy."
    )
    return replace(joint.policy_selector, mode="argmax", prime_on_policy=1.0)


def deploy_rollout_viewpoints(
    *,
    joint: Any,
    advance: Any,
    t0_type: Any,
    batch_size: int,
    device: torch.device,
    n: int,
    cond: Any = None,
) -> list[Viewpoint]:
    """Run the closed-loop deploy rollout, returning the viewpoints it actually took.

    ``cond`` (visual search): a zero-argument callable returning the target embedding for
    a FiLM-conditioned scorer; None for every unconditioned policy.

    ``advance(viewpoint, state, t) -> RecurrentState`` is the task's own single-glimpse
    step (called with ``state=None`` at t0, so it owns its own state init); this function
    owns only the selection. The scorer runs in eval mode under ``no_grad`` — deploy
    semantics, and the reason a policy eval cannot leak gradient or BatchNorm statistics
    into training. Train mode is restored on the way out.
    """
    from canvit.harness.rollout.engine import _to_vp
    from canvit.harness.rollout.viewpoint import ViewpointType

    # t0 must be the FULL anchor: the scorer needs a canvas to read, and at t=0 there is
    # no state yet. FULL delegates to the RandomSelector fallback, which needs none. This
    # matches all three tasks' validation (C2F, random and fixation_grid all start full)
    # and the RL reference, whose episodes open on the full scene.
    assert t0_type == ViewpointType.FULL, (
        f"eval_policy='policy' requires a FULL t0 anchor, got {t0_type}")

    sel = deploy_selector(joint)
    scorer = joint.scorer
    was_training = scorer.training
    scorer.eval()
    try:
        with torch.no_grad():
            ctx = sel.start_rollout(t0_type=t0_type, batch_size=batch_size, device=device)
            if cond is not None:
                ctx.cond = cond
            state, taken = None, []
            for t in range(n):
                vp_type = t0_type if t == 0 else ViewpointType.RANDOM
                named = sel.select(vp_type=vp_type, ctx=ctx, t=t, batch_size=batch_size,
                                   device=device, state=state)
                vp = _to_vp(named)
                taken.append(vp)
                state = advance(vp, state, t)
    finally:
        if was_training:
            scorer.train()
    return taken


__all__ = [
    "HISTORICAL_DEFAULTS",
    "CLOSED_LOOP",
    "EntropyGuidedC2F",
    "OPEN_LOOP",
    "PAPER_TABLE4_C64",
    "closed_loop_rollout",
    "entropy_c2f_chooser",
    "EvalPolicy",
    "deploy_rollout_viewpoints",
    "deploy_selector",
    "open_loop_viewpoints",
    "resolve",
]


# --- EG-C2F: entropy-guided coarse-to-fine (the paper's strongest heuristic) ------


def _tile_masks(crops: list[tuple[float, float, float]], canvas_grid: int,
                device: Any) -> Any:
    """[n_crops, G, G] bool: which canvas cells fall inside each crop.

    Ported verbatim from ``canvit_eval/policies.py::_build_tile_masks`` — the
    implementation the published EG-C2F numbers were produced with.
    """
    import torch

    G = canvas_grid
    assert G > 0 and (G & (G - 1)) == 0, f"canvas_grid must be a power of 2, got {G}"
    coords = torch.linspace(-1 + 1 / G, 1 - 1 / G, G, device=device)
    crops_t = torch.tensor(crops, device=device)
    cy, cx, s = crops_t[:, 0], crops_t[:, 1], crops_t[:, 2]
    row_in = (coords.unsqueeze(0) - cy.unsqueeze(1)).abs() <= s.unsqueeze(1)
    col_in = (coords.unsqueeze(0) - cx.unsqueeze(1)).abs() <= s.unsqueeze(1)
    return row_in.unsqueeze(2) & col_in.unsqueeze(1)


class EntropyGuidedC2F:
    """Coarse-to-fine quadtree levels, visited in order of DECREASING per-tile probe entropy.

    A port of ``canvit_eval/policies.py::EntropyGuidedC2F``, which is what produced the
    published EG-C2F row (paper Table 4 / `PAPER_TABLE4_C64`). Closed-loop: the pick needs
    the live canvas, because the entropy map is read off the probe at each step.

    Schedule: 3 quadtree levels — 1 full scene, then 4 half-quadrants, then 16 quarter-tiles
    (21 timesteps total). At the harness default ``n_timesteps=5`` that is the full scene
    plus the 4 half-quadrants ranked by entropy, which is exactly the paper's t0..t4 row.
    """

    def __init__(self, *, seg: Any, batch_size: int, device: Any, canvas_grid: int,
                 n_levels: int = 3) -> None:
        from canvit.core.policies import level_viewpoints

        self.seg, self.B, self.device, self.canvas_grid = seg, batch_size, device, canvas_grid
        self.levels = [level_viewpoints(lvl) for lvl in range(n_levels)]
        self.level_starts, t = [], 0
        for lvl in self.levels:
            self.level_starts.append(t)
            t += len(lvl)
        self.masks: list[Any] = [None] + [
            _tile_masks(self.levels[lvl], canvas_grid, device) for lvl in range(1, n_levels)]
        self.visited: list[Any] = [None] * n_levels

    def _entropy(self, state: Any) -> Any:
        """Per-cell probe entropy [B, G, G]. fp32 via ``head_logits`` (which disables
        autocast), so the ranking does not depend on the caller's amp context."""
        from canvit.core.policy import entropy_from_logits, head_logits

        return entropy_from_logits(
            head_logits(self.seg, state.canvas, canvas_grid=self.canvas_grid)).float()

    def __call__(self, t: int, state: Any) -> Viewpoint:
        import torch

        level_idx = sum(1 for s in self.level_starts[1:] if t >= s)
        pos_in_level = t - self.level_starts[level_idx]
        crops = self.levels[level_idx]

        if level_idx == 0:                      # the full-scene anchor; no choice to make
            cy, cx, s = crops[0]
            return Viewpoint(
                centers=torch.tensor([[cy, cx]], device=self.device).expand(self.B, -1).contiguous(),
                scales=torch.full((self.B,), s, device=self.device))

        if pos_in_level == 0:                   # entering a level: nothing visited yet
            self.visited[level_idx] = torch.zeros(
                self.B, len(crops), dtype=torch.bool, device=self.device)

        ent, masks, visited = self._entropy(state), self.masks[level_idx], self.visited[level_idx]
        n_cells = masks.sum(dim=(1, 2)).clamp(min=1).float()
        # mean entropy per candidate tile, then take the highest UNVISITED one
        scores = (ent.unsqueeze(1) * masks.unsqueeze(0).float()).sum(dim=(2, 3)) / n_cells
        scores = scores.masked_fill(visited, float("-inf"))
        chosen = scores.argmax(dim=1)
        visited.scatter_(1, chosen.unsqueeze(1), True)

        selected = torch.tensor(crops, device=self.device)[chosen]
        return Viewpoint(centers=selected[:, :2].contiguous(), scales=selected[:, 2].contiguous())


def closed_loop_rollout(*, chooser: Any, advance: Any, n: int) -> list[Viewpoint]:
    """Drive a closed-loop eval policy: ``chooser(t, state) -> Viewpoint``, then
    ``advance(vp, state, t) -> state`` (called with ``state=None`` at t0, so the task owns
    its own state init). Shared by the learned policy and EG-C2F so both take exactly the
    same path through the task's rollout, and no metric code has to branch on which."""
    import torch

    with torch.no_grad():
        state, taken = None, []
        for t in range(n):
            vp = chooser(t, state)
            taken.append(vp)
            state = advance(vp, state, t)
    return taken


def entropy_c2f_chooser(*, seg: Any, batch_size: int, device: Any, canvas_grid: int) -> Any:
    """A fresh :class:`EntropyGuidedC2F` — fresh because it carries per-rollout `visited`
    state, so reusing one across batches would silently exclude already-picked tiles."""
    return EntropyGuidedC2F(seg=seg, batch_size=batch_size, device=device,
                            canvas_grid=canvas_grid)
