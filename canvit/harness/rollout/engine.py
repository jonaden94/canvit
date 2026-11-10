"""Task-agnostic, TrainSpec-driven rollout engine (design doc 07 §3/§6).

This is the shared core of the unified harness: one recurrent glimpse rollout that
every task (distill / ade20k / in1k) drives, with the gradient regime and grad
routing taken from a :class:`~canvit.harness.spec.TrainSpec`. It subsumes
the historical ``train/step.py::training_step`` (distill's rollout) as the special
case ``bptt='chunked'`` + backbone-carrying-grad + ``DistillTask`` — the P1 parity
digest (``9a0100a1a3de3acd``) is the byte-exact regression guard for that case.

Seams (design §3):
  * ``RolloutTask`` — forward one glimpse (``forward_glimpse``), score the readout
    (``step_loss``), and expose the per-image loss for the policy reward
    (``per_image_loss``). The readout type is opaque to the engine.
  * ``Selector`` (from ``train/selector.py``) — where the next glimpse goes.
  * ``JointPolicy`` (optional) — the in-graph policy loss, unchanged from P4b.

Gradient regime (``spec.bptt.mode``):
  * ``none``    — backbone forward under ``no_grad``; only the head/policy carry
                  graph (probe / frozen-model policy). One backward at rollout end.
  * ``full``    — backbone carries grad; ONE backward over the whole rollout
                  (= ``chunked`` with ``chunk_size == n_glimpses``).
  * ``chunked`` — backbone carries grad; TBPTT: backward + detach every
                  ``chunk_size`` glimpses. Length is a fixed ``horizon`` or a
                  stochastic ``continue_prob`` extension (distill).

Grad routing to the backbone is enforced by the caller (via ``requires_grad`` +
the selector's ``feats_detached``); the engine only decides the ``no_grad`` context
of the glimpse forward and the backward cadence.
"""

from __future__ import annotations

import random as _random
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, NamedTuple, Protocol

import torch
import torch.distributed as dist
from torch import Tensor

from canvit.core import RecurrentState, Viewpoint
from canvit.harness.rollout.selector import RandomSelector, Selector
from canvit.harness.rollout.viewpoint import Viewpoint as NamedViewpoint
from canvit.harness.rollout.viewpoint import ViewpointType
from canvit.harness.spec import BpttSpec

if TYPE_CHECKING:  # heavy import kept off the runtime path
    from canvit.harness.policy.joint import JointPolicy


class GlimpseOut(NamedTuple):
    """What a task returns from ``forward_glimpse``: the opaque ``readout`` that
    ``step_loss``/``per_image_loss`` consume, plus the recurrent state and VPE to
    thread to the next glimpse.

    ``local_patches`` ([B,n_patch,local_dim], optional): the backbone's RAW local
    token output for this glimpse, BEFORE/AS written into the canvas (as opposed to
    the canvas state, which is AFTER writing and accumulated over every glimpse so
    far). Threaded to the NEXT glimpse's selector call for the "local_feat" policy
    feature group; None for tasks that do not populate it (zero behavior change).

    ``write_snapshots`` (optional): the glimpse's canvas right after each of its
    writes (``CanViTOutput.write_snapshots``), threaded the same way for the
    "depth_feat" policy feature group."""

    readout: Any
    state: RecurrentState
    vpe: Tensor | None
    local_patches: Tensor | None = None
    write_snapshots: list[Tensor] | None = None


class GlimpseLoss(Protocol):
    """Minimal contract the engine needs from ``step_loss``'s return value."""

    combined: Tensor  # scalar, with grad — the per-glimpse task loss


class TaskLoss(NamedTuple):
    """Convenience concrete :class:`GlimpseLoss` for tasks whose step loss is a single
    scalar (ade20k/in1k). Distill returns its richer ``LossOutput`` (also has
    ``.combined``), so the engine only ever reads ``.combined``."""

    combined: Tensor


class RolloutTask(Protocol):
    """The rollout-facing task seam (design §3.1). The full harness Task protocol
    adds data/eval/checkpoint methods; the engine only needs these three."""

    def forward_glimpse(
        self, *, model: Any, images: Tensor, state: RecurrentState,
        viewpoint: Viewpoint, backbone_no_grad: bool,
    ) -> GlimpseOut: ...

    def step_loss(self, readout: Any) -> Any:  # -> GlimpseLoss (has .combined)
        ...

    def per_image_loss(self, readout: Any) -> Tensor:  # [B], for the policy reward
        ...


@dataclass
class BranchResult:
    """Per-branch outcome the caller uses for logging/metrics (task-specific)."""

    t0_type: ViewpointType
    is_policy: bool
    mean_loss: Tensor          # detached: mean per-glimpse combined loss
    n_steps: int
    final_readout: Any         # last-chunk readout (for task metrics)
    metrics: dict[str, Tensor] = field(default_factory=dict)
    """Optional detached task metrics for this branch (empty unless the task provides
    the hooks): per-glimpse scalars from ``glimpse_metrics`` averaged over the branch,
    merged with one-shot ``final_metrics`` on the last readout. Distill uses these for
    its historical ``full/…`` / ``random/…`` wandb series."""


@dataclass
class RolloutViz:
    """Per-glimpse visualization data collected from branch 0 when ``collect_viz=True``.

    The engine stays task-neutral: it records the viewpoints and calls the task's
    optional ``viz_frame`` hook, whose return value is opaque here (distill returns
    teacher/predicted-scene sample data for the PCA figure; ade20k returns predicted
    masks for the segmentation overlay). Rendering + saving is the task's job.
    """

    viewpoints: list[NamedViewpoint] = field(default_factory=list)
    frames: list[Any] = field(default_factory=list)
    initial: Any = None   # optional task.viz_init(state_init) capture (pre-glimpse panels)


@dataclass
class RolloutResult:
    total_loss: Tensor                          # detached mean over branches
    branches: list[BranchResult] = field(default_factory=list)
    n_glimpses: int = 0
    policy_metrics: dict | None = None          # detached per-step means (joint mode)
    viz: RolloutViz | None = None               # only when collect_viz=True


def _to_vp(vp: NamedViewpoint) -> Viewpoint:
    return Viewpoint(centers=vp.centers, scales=vp.scales)


def sample_n_glimpses(bptt: BpttSpec, *, rng: _random.Random | Any = _random) -> int:
    """Trajectory length. Fixed ``horizon``, or stochastic TBPTT extension
    (``n=chunk_size``; ``while rand()<continue_prob: n+=chunk_size``) — the latter
    reproduces distill's historical draw byte-for-byte (same ``random.random()``
    call sequence)."""
    if bptt.continue_prob is not None:
        n = bptt.chunk_size
        while rng.random() < bptt.continue_prob:
            n += bptt.chunk_size
        return n
    assert bptt.horizon is not None
    return bptt.horizon


def run_rollout(
    *,
    model: Any,
    images: Tensor,
    task: RolloutTask,
    selector: Selector,
    bptt: BpttSpec,
    branches: list[ViewpointType],
    canvas_grid_size: int,
    amp_ctx: Any,
    task_weight: float = 1.0,
    collect_viz: bool = False,
    viz_task: Any = None,
    joint: "JointPolicy | None" = None,
    rng: Any = _random,
) -> RolloutResult:
    """Run every branch's rollout and return the total loss (backward already
    called inside, per the TBPTT cadence — no retain_graph). Mirrors the historical
    ``training_step`` for the distill config so the parity digest is unchanged.

    ``branches`` lists the t0 viewpoint type per branch (design D-D: distill passes
    ``[FULL]*n_full + [RANDOM]*n_random``; ade20k/in1k pass a single element).

    ``task_weight`` (design §4: ``loss = task_weight*task_loss + policy_weight*policy_loss``)
    scales the per-glimpse task loss in the graph; the policy loss is already scaled by
    ``joint.rl_weight`` (== ``policy_weight``). The default 1.0 is byte-exact to the pre-
    weight path (the ``==1.0`` guard makes the parity config hit the identical expression).
    """
    n_branches = len(branches)
    assert n_branches >= 1
    assert bptt.chunk_size >= 1
    device = images.device
    B = images.shape[0]

    core_model = getattr(model, "module", model)
    state_init = core_model.init_state(batch_size=B, canvas_grid_size=canvas_grid_size)

    n_glimpses = sample_n_glimpses(bptt, rng=rng)
    # DDP: each rank samples independently; broadcast so all ranks call backward()
    # the same number of times (else NCCL allreduce deadlocks).
    if dist.is_available() and dist.is_initialized():
        n_t = torch.tensor(n_glimpses, device=device)
        dist.broadcast(n_t, src=0)
        n_glimpses = int(n_t.item())

    # Backward cadence: 'none'/'full' => one backward at the end; 'chunked' => every
    # chunk_size glimpses. 'none' additionally runs the backbone forward under no_grad.
    backbone_no_grad = bptt.mode == "none"
    eff_chunk = n_glimpses if bptt.mode in ("none", "full") else bptt.chunk_size

    pol_acc = {"loss": torch.zeros((), device=device), "reward": torch.zeros((), device=device), "n": 0}
    results: list[BranchResult] = []
    # Viz is collected from branch 0 only (sample 0 inside the task's hook), matching
    # the historical training_step. Off by default => the parity path never touches it.
    # Hooks live on the RUN-level task (which also owns render_viz), not the per-batch
    # bound task the engine otherwise talks to — hence the explicit ``viz_task``.
    _viz_owner = viz_task if viz_task is not None else task
    viz_hook = getattr(_viz_owner, "viz_frame", None) if collect_viz else None
    viz = RolloutViz() if viz_hook is not None else None

    def _viz_capture(branch_idx: int, gout: GlimpseOut, vp_named: NamedViewpoint,
                     vp: Viewpoint, L: Any) -> None:
        if viz is None or branch_idx != 0:
            return
        viz.viewpoints.append(vp_named)
        viz.frames.append(viz_hook(model=model, images=images, gout=gout, viewpoint=vp, loss=L))

    def _tw_loss(L: Any) -> Tensor:
        """The per-glimpse task loss scaled by ``task_weight`` (design §4). The
        ``== 1.0`` guard makes the default/parity path hit the identical expression as
        before (no float multiply), so the digest ``9a0100a1a3de3acd`` is preserved."""
        lf = L.combined.float()
        return lf if task_weight == 1.0 else task_weight * lf

    def run_branch(t0_type: ViewpointType, branch_idx: int = 0) -> BranchResult:
        # Joint mode: pick this branch's selector. Policy branches are FULL-anchored
        # and carry the in-graph policy loss; non-policy branches run pure-random.
        if joint is not None:
            sel, is_policy = joint.branch_selector(t0_type)
            if is_policy:
                t0_type = ViewpointType.FULL
        else:
            sel, is_policy = selector, False
        prev_pi_loss: Tensor | None = None
        # VPG (terminal reward): the per-step policy aux, buffered until rollout end.
        # Empty for the historical inline-credit objectives (QReg/PG).
        defer = is_policy and joint is not None and joint.defers_credit
        traj: list[dict[str, Tensor]] = []

        # Optional per-glimpse task metrics (distill's scene/cls sub-losses), summed
        # here and averaged over the branch below. Absent hook => untouched fast path.
        gm_hook = getattr(task, "glimpse_metrics", None)
        acc: dict[str, Tensor] = {}

        def _acc(L: Any) -> None:
            if gm_hook is None:
                return
            for k, v in gm_hook(L).items():
                acc[k] = v if k not in acc else acc[k] + v

        ctx = sel.start_rollout(t0_type=t0_type, batch_size=B, device=device)

        # Pre-glimpse capture (initial scene/canvas panels), branch 0 only.
        if viz is not None and branch_idx == 0:
            viz_init = getattr(_viz_owner, "viz_init", None)
            if viz_init is not None:
                viz.initial = viz_init(model=model, images=images, state=state_init)

        # t0 forward
        with amp_ctx:
            vp0_named = sel.select(vp_type=t0_type, ctx=ctx, t=0, batch_size=B, device=device, state=state_init)
            vp0 = _to_vp(vp0_named)
            gout = task.forward_glimpse(
                model=model, images=images, state=state_init, viewpoint=vp0, backbone_no_grad=backbone_no_grad,
            )
            L = task.step_loss(gout.readout)
        _viz_capture(branch_idx, gout, vp0_named, vp0, L)
        _acc(L)
        if is_policy and not defer:  # seed the reward denominator with the FULL-anchor loss (no policy loss at t0)
            prev_pi_loss = task.per_image_loss(gout.readout).detach()

        chunk_loss = _tw_loss(L)
        total_detached = L.combined.detach().float()
        n_steps = 1
        state = gout.state
        # The glimpse JUST taken's raw local tokens (None at t0: no glimpse has
        # happened yet), threaded to the NEXT select() call for "local_feat".
        # Detached in lockstep with `state` at every TBPTT chunk boundary below.
        last_local = gout.local_patches
        # Same idea, for "depth_feat": the glimpse's canvas after each of its writes.
        last_write_snapshots = gout.write_snapshots
        final_readout = gout.readout

        # chunk_size==1: t0 is already a complete chunk.
        if eff_chunk == 1:
            (chunk_loss / n_glimpses / n_branches).backward()
            if n_glimpses > 1:
                state = RecurrentState(canvas=gout.state.canvas.detach(),
                                       recurrent_cls=gout.state.recurrent_cls.detach())
                last_local = last_local.detach() if last_local is not None else None
                last_write_snapshots = (
                    [s.detach() for s in last_write_snapshots] if last_write_snapshots is not None else None
                )
                chunk_loss = torch.zeros((), device=device)

        for t in range(1, n_glimpses):
            vp_named = sel.select(
                vp_type=ViewpointType.RANDOM, ctx=ctx, t=t, batch_size=B, device=device,
                state=state, local_patches=last_local, write_snapshots=last_write_snapshots,
            )
            vp = _to_vp(vp_named)
            with amp_ctx:
                gout = task.forward_glimpse(
                    model=model, images=images, state=state, viewpoint=vp, backbone_no_grad=backbone_no_grad,
                )
                L = task.step_loss(gout.readout)

            chunk_loss = chunk_loss + _tw_loss(L)
            _viz_capture(branch_idx, gout, vp_named, vp, L)
            _acc(L)

            if is_policy:
                assert joint is not None
                aux = sel.last_aux  # type: ignore[attr-defined]
                assert aux is not None, "policy selector produced no aux for a RANDOM glimpse"
            if is_policy and defer:
                # VPG: buffer this action's logits + V(s_t); the reward is only known at
                # the LAST glimpse, so the loss is built below (still inside the loop, so
                # it joins the FINAL chunk's backward — see the `t == n_glimpses - 1` block).
                assert joint is not None and aux is not None
                traj.append({"scores": aux["scores"], "flat_idx": aux["flat_idx"],
                             "value": joint.state_value(aux["feats"])})
            elif is_policy:
                assert joint is not None and prev_pi_loss is not None and aux is not None
                cur_pi = task.per_image_loss(gout.readout).detach()
                reward = (prev_pi_loss - cur_pi) / prev_pi_loss.clamp_min(1e-4)
                prev_pi_loss = cur_pi
                ploss = joint.glimpse_loss(depth=t, scores=aux["scores"], flat_idx=aux["flat_idx"], reward=reward)
                # chunk_loss is divided by n_glimpses below, but there are only
                # n_glimpses-1 POLICY glimpses (t0 is the anchor and carries no policy
                # loss). Undo that division and re-normalize by the policy count, so the
                # policy term is the MEAN over depths -- which is what the reference does:
                # `rl_train.rollout_and_loss` cats all depths into one [horizon*B, A]
                # tensor and takes a single `F.mse_loss`, i.e. one mean over horizon*B.
                #
                # Without this the harness's scorer gradient was exactly
                # (n_glimpses-1)/n_glimpses = 0.8x the reference's at horizon 4 -- a 20%
                # smaller effective policy LR at the same nominal `policy_lr`, so the
                # scorer was systematically under-trained at a fixed step budget. VPG
                # already compensated for the same division (see the `* n_glimpses` in the
                # deferred-credit branch below); the inline QReg/PG path did not.
                #
                # Only the GRAPH term is rescaled; `pol_acc["loss"]` keeps the raw
                # per-depth loss so the logged `policy_loss` series stays on the same scale
                # as earlier harness runs and as rl_train's logged `train_loss`.
                chunk_loss = chunk_loss + ploss * (n_glimpses / (n_glimpses - 1))
                pol_acc["loss"] = pol_acc["loss"] + ploss.detach()
                pol_acc["reward"] = pol_acc["reward"] + reward.mean().detach()
                pol_acc["n"] += 1

            if defer and traj and t == n_glimpses - 1:
                # Terminal reward: R = -(per-image task loss of the LAST glimpse), the
                # analogue of autoreg's R = -CE(final logits). Built HERE, before the
                # chunk backward, rather than after the loop: with
                # policy_grad_to_backbone=True the scorer's graph hangs off backbone
                # activations that the task backward frees, so a separate later backward
                # would raise. Folding it into chunk_loss keeps ONE backward per chunk.
                assert joint is not None
                reward = -task.per_image_loss(gout.readout).detach()
                # chunk_loss is divided by n_glimpses below, but vpg_loss already sums over
                # T and means over B (autoreg does NOT divide its RL loss by T). Pre-multiply
                # so the division cancels and `policy_weight` means the same thing at any
                # horizon.
                ploss = joint.trajectory_loss(
                    scores=torch.stack([a["scores"] for a in traj], dim=1),
                    flat_idx=torch.stack([a["flat_idx"] for a in traj], dim=1),
                    values=torch.stack([a["value"] for a in traj], dim=1),
                    reward=reward,
                ) * n_glimpses
                chunk_loss = chunk_loss + ploss
                pol_acc["loss"] = pol_acc["loss"] + ploss.detach() / n_glimpses
                pol_acc["reward"] = pol_acc["reward"] + reward.mean()
                pol_acc["n"] += 1

            total_detached = total_detached + L.combined.detach().float()
            final_readout = gout.readout
            n_steps += 1

            is_chunk_end = ((t + 1) % eff_chunk == 0)
            is_last = (t == n_glimpses - 1)
            if is_chunk_end:
                (chunk_loss / n_glimpses / n_branches).backward()  # no retain_graph
                if not is_last:
                    state = RecurrentState(canvas=gout.state.canvas.detach(),
                                           recurrent_cls=gout.state.recurrent_cls.detach())
                    last_local = gout.local_patches.detach() if gout.local_patches is not None else None
                    last_write_snapshots = (
                        [s.detach() for s in gout.write_snapshots] if gout.write_snapshots is not None else None
                    )
                    chunk_loss = torch.zeros((), device=device)
                else:
                    state = gout.state
                    last_local = gout.local_patches
                    last_write_snapshots = gout.write_snapshots
            else:
                state = gout.state
                last_local = gout.local_patches
                last_write_snapshots = gout.write_snapshots

        # A trailing partial chunk (n_glimpses not a multiple of eff_chunk) still
        # needs its backward. For distill (chunk_size divides its stochastic length
        # by construction) this never fires, so the parity path is unaffected.
        if n_glimpses % eff_chunk != 0:
            (chunk_loss / n_glimpses / n_branches).backward()

        metrics = {k: v / n_steps for k, v in acc.items()}
        fm_hook = getattr(task, "final_metrics", None)
        if fm_hook is not None:  # one-shot metrics on the last readout (distill's cos-sims)
            metrics.update(fm_hook(final_readout))

        return BranchResult(
            t0_type=t0_type, is_policy=is_policy,
            mean_loss=total_detached / n_steps, n_steps=n_steps, final_readout=final_readout,
            metrics=metrics,
        )

    for branch_idx, t0 in enumerate(branches):
        results.append(run_branch(t0, branch_idx))

    total_loss = torch.stack([r.mean_loss for r in results]).mean()

    policy_metrics = None
    if joint is not None and pol_acc["n"] > 0:
        n = pol_acc["n"]
        policy_metrics = {
            "policy_loss": pol_acc["loss"] / n,
            "reward_frac": pol_acc["reward"] / n,
            "prime_on_policy": joint.policy_selector.prime_on_policy,
        }
        # The objective's own detached diagnostics (VPG: entropy / value / adv_std / the
        # two loss terms). Empty for QReg/PG, which do not populate `last_step`.
        # NB `reward_frac` is a genuine FRACTION only for the inline objectives; under VPG
        # the reward is the raw terminal `-per_image_loss`, so the series is on a different
        # scale (~-0.7 for the ade20k probe, not ~+0.03).
        policy_metrics.update(joint.last_step)

    return RolloutResult(
        total_loss=total_loss, branches=results, n_glimpses=n_glimpses,
        policy_metrics=policy_metrics, viz=viz,
    )


__all__ = [
    "BranchResult",
    "GlimpseOut",
    "RolloutResult",
    "RolloutViz",
    "RolloutTask",
    "TaskLoss",
    "run_rollout",
    "sample_n_glimpses",
]
