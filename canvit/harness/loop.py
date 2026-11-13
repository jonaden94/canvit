"""The task-neutral step-based training driver (design §1, §Loop).

ONE loop for all three tasks. It owns the task-agnostic mechanics — per-step
rollout → grad-clip → optimizer/scheduler step, plus logging / checkpoint / eval
cadence — and delegates everything task-specific to the ``Task`` seam (data via the
caller's iterator, per-batch targets via ``task.bind``, metrics via ``on_eval``).
There is only a step loop (owner decision: step-based only). All three tasks —
including in1k, whose config is now step-based (``max_steps``) like the others — feed
this same driver.

This is deliberately thin: the heavy, task-specific machinery (distill's teacher /
normalizer / webdataset-resume, ade20k's mIoU eval, in1k's top-k) lives in each
``Task``, never here (design §1 "keep the outer loop thin"). DDP grad-sync for
in-rollout modules (backbone/scorer) is the manual-AllReduce path (design §9); it is
wired here for the scorer and completed in the DDP stage.
"""

from __future__ import annotations

import logging
import os
import signal
import subprocess
import time
from collections.abc import Callable, Iterator
from contextlib import nullcontext
from pathlib import Path
from typing import Any

import torch

from canvit.harness.infra import ddp
from canvit.harness.infra.checkpoint import save_checkpoint
from canvit.harness.optim.ema import EMATracker
from canvit.harness.rollout import run_rollout
from canvit.harness.rollout.viewpoint import ViewpointType
from canvit.harness.spec import TrainSpec

log = logging.getLogger(__name__)

# --- operational helpers (task-neutral; ported from train/loop.py) ---------

# SIGUSR1 (SLURM preemption / manual) → checkpoint after the current step. The
# handler only flips a flag; the loop does the save at a safe point. Installed by
# run() (see run.py); the inner loop merely polls the flag, so it stays signal-agnostic
# and unit-testable without a real signal.
_checkpoint_requested = False


def _handle_sigusr1(signum: int, frame: object) -> None:
    global _checkpoint_requested
    _checkpoint_requested = True
    log.info("SIGUSR1 received — will checkpoint after the current step")


def install_sigusr1_handler() -> None:
    """Install the SIGUSR1 → checkpoint handler. No-op off the main thread / where
    unsupported (e.g. inside a worker or on platforms without SIGUSR1)."""
    try:
        signal.signal(signal.SIGUSR1, _handle_sigusr1)
    except (ValueError, OSError, AttributeError):
        log.debug("SIGUSR1 handler not installed (not main thread / unsupported)")


def request_checkpoint() -> None:
    """Programmatically request a checkpoint at the next step boundary (test hook /
    non-signal trigger)."""
    global _checkpoint_requested
    _checkpoint_requested = True


def cancel_slurm_array() -> None:
    """Cancel the remaining tasks of the current SLURM array (crash-loop / done-early
    prevention). No-op outside a SLURM array job."""
    job_id = os.environ.get("SLURM_ARRAY_JOB_ID")
    if not job_id:
        return
    log.info("Cancelling remaining SLURM array tasks for job %s", job_id)
    try:
        subprocess.run(["scancel", job_id], check=False)
    except Exception as e:  # scancel missing / not permitted — never fatal
        log.warning("scancel failed: %s", e)


def grad_norms_by_module(
    module: Any, *, depth: int = 1, deep_prefixes: tuple[str, ...] = (),
) -> dict[str, float]:
    """Gradient L2 norms grouped by module-path prefix (ported from train/loop.py,
    task-agnostic). Each param contributes to the group named by its first ``depth``
    dot-separated components; params under a ``deep_prefixes`` entry are grouped one
    level deeper (longest match wins) — e.g. ``deep_prefixes=("patcher",)`` splits
    ``patcher`` into ``patcher.kpe`` / ``patcher.conditioner`` / … ."""
    groups: dict[str, list[torch.Tensor]] = {}
    for name, param in module.named_parameters():
        if param.grad is None:
            continue
        match = max(
            (p for p in deep_prefixes if name.startswith(p + ".")),
            key=lambda p: p.count("."), default=None,
        )
        ndepth = (len(match.split(".")) + 1) if match is not None else depth
        groups.setdefault(".".join(name.split(".")[:ndepth]), []).append(param.grad)
    return {
        prefix: torch.cat([g.flatten() for g in grads]).norm().item()
        for prefix, grads in groups.items()
    }


def branch_metrics(branches: list[Any]) -> dict[str, torch.Tensor]:
    """Per-branch-group metric series, keyed ``{t0_type}/{name}`` (train/loop.py's
    ``full/…`` and ``random/…``). Branches of the same t0 type are averaged, exactly like
    the historical ``aggregate(list[BranchMetrics])``. Beyond ``loss`` the names come
    from whatever the task's ``glimpse_metrics``/``final_metrics`` hooks produced, so
    this stays task-neutral (ade20k/in1k supply no hooks => just ``…/loss``)."""
    groups: dict[str, list[Any]] = {}
    for br in branches:
        groups.setdefault(br.t0_type.name.lower(), []).append(br)
    out: dict[str, torch.Tensor] = {}
    for name, brs in groups.items():
        out[f"{name}/loss"] = torch.stack([b.mean_loss for b in brs]).mean()
        for key in brs[0].metrics:
            out[f"{name}/{key}"] = torch.stack([b.metrics[key] for b in brs]).mean()
    return out


def apply_requires_grad(*, model: Any, head: Any, joint: Any, spec: TrainSpec) -> None:
    """Set ``requires_grad`` from the spec — the harness, not the task, decides what
    trains (design §3.1). ``model`` owns the backbone (+ head as a submodule); ``head``
    is the same head module (or None); ``joint`` owns the scorer."""
    core = getattr(model, "module", model)
    # Freeze/unfreeze the CanViT trunk. For task wrappers the trunk is core.canvit;
    # for the distill core model it is core itself (its heads are submodules).
    trunk = getattr(core, "canvit", core)
    trunk.requires_grad_(spec.train_backbone)
    if not spec.train_backbone:
        trunk.eval()
    if head is not None:
        head.requires_grad_(spec.train_head)
        # requires_grad_(False) does NOT freeze BatchNorm running statistics — the probe
        # head carries one (`head.bn`), and it kept updating from every training forward.
        # On a policy run that is corrosive: the reward IS the probe's CE reduction, so
        # the scorer chases a probe drifting under its feet, and the drift is data-order
        # (i.e. seed) dependent. Caught by exp27 arm B, where the POLICY-INDEPENDENT t0
        # mIoU came out 38.50 vs 38.75 on two seeds against a reference 39.6.
        # `rl_train` never had this: it does seg.eval() on the whole model.
        if not spec.train_head:
            head.eval()
    if joint is not None:
        joint.scorer.requires_grad_(spec.train_policy)


def run_training_loop(
    *,
    task: Any,                       # Task: batch_images(batch, device), bind(batch, device, model=, head=)
    model: Any,
    head: Any,
    optimizer: Any,
    scheduler: Any,
    selector: Any,
    spec: TrainSpec,
    branches: list[ViewpointType],
    canvas_grid: int,
    device: torch.device,
    train_batches: Iterator,
    n_steps: int,
    start_step: int = 0,
    joint: Any | None = None,
    amp_ctx: Any | None = None,
    grad_clip: float = 1.0,
    is_dist: bool = False,
    rank: int = 0,
    task_name: str = "task",
    model_config: dict | None = None,
    metadata: dict | None = None,
    log_every: int = 20,
    ckpt_dir: Path | None = None,
    ckpt_every: int = 0,             # 0 => only the end-of-run checkpoint
    eval_every: int = 0,             # 0 => no periodic eval
    ema_alpha: float = 0.0,          # >0 => also log an EMA-smoothed total_loss
    log_grad_norms: bool = False,    # log per-module grad L2 norms on log steps
    grad_norm_deep_prefixes: tuple[str, ...] = (),
    log_timing: bool = False,        # cumulative data% vs gpu% (bottleneck analysis)
    viz_every: int = 0,              # 0 => never; else collect+render viz every N steps
    run_dir: Path | None = None,     # where task.render_viz writes figures (locally)
    signal_checkpoint: bool = True,  # honor SIGUSR1 -> checkpoint (handler in run())
    best_metric: str | None = None,  # eval key to MAXIMIZE -> writes ckpt_dir/best.pt
    on_log: Callable[[int, dict], None] | None = None,
    on_eval: Callable[[int], dict] | None = None,
    on_best: Callable[[int, str, float], None] | None = None,
) -> dict:
    """Run ``n_steps`` of training and return the last step's metrics. ``train_batches``
    yields task-native batches; ``task.bind`` turns each into the per-glimpse
    ``RolloutTask`` the engine runs."""
    global _checkpoint_requested
    amp_ctx = amp_ctx or nullcontext()
    # Model and policy gradients are clipped SEPARATELY, each to grad_clip (train/loop.py
    # 874-878): one joint norm over the union would couple their magnitudes, so a large
    # scorer gradient would shrink the model's update (and vice versa).
    _scorer_ids = {id(p) for p in joint.scorer.parameters()} if joint is not None else set()
    trainable = [p for g in optimizer.param_groups for p in g["params"] if id(p) not in _scorer_ids]
    core = getattr(model, "module", model)
    ema = EMATracker(ema_alpha) if ema_alpha > 0 else None
    last: dict = {}

    def _write_ckpt(path: Path, step: int, *, update_latest: bool = True) -> None:
        save_checkpoint(
            path, model=model, optimizer=optimizer, scheduler=scheduler,
            step=step, task_name=task_name, spec=spec, model_config=model_config,
            metadata=metadata, joint=joint, update_latest=update_latest,
        )

    def _save(step: int) -> None:
        # Rank 0 writes; the others wait, so no rank races ahead of a half-written file.
        if is_dist and rank != 0:
            ddp.barrier()
            return
        if ckpt_dir is not None:
            _write_ckpt(ckpt_dir / f"step-{step}.pt", step)
        if is_dist:
            ddp.barrier()

    # Best-checkpoint selection (ade20k/train.py 177-182, in1k/train.py 210-214): the
    # task names an eval metric to MAXIMIZE and the improving checkpoint is kept as
    # `best.pt`, because what you publish is the best probe, not the last one. Per-job,
    # which is exactly the standalone semantics (neither standalone resumes, and both
    # run their whole schedule in one job). Tasks with no `best_metric` are unaffected.
    #
    # NO barrier here, unlike _save: validation is rank-0-only (on_eval returns {} on the
    # others), so only rank 0 ever gets a metric to compare. A barrier would deadlock.
    best_so_far: float | None = None

    def _track_best(step: int, metrics: dict) -> None:
        nonlocal best_so_far
        if not best_metric or best_metric not in metrics:
            return
        value = metrics[best_metric]
        if best_so_far is None or value > best_so_far:
            best_so_far = value
            if ckpt_dir is not None:
                # update_latest=False is load-bearing: `latest.pt` must keep pointing at
                # the NEWEST checkpoint. Repointing it at best.pt would make a resumed
                # array task silently restart from the best step, not the last one.
                _write_ckpt(ckpt_dir / "best.pt", step, update_latest=False)
                log.info("step %d  new best %s=%.4f -> best.pt", step, best_metric, value)
        if on_best is not None:
            on_best(step, best_metric, best_so_far)

    t_data_total = t_gpu_total = 0.0

    for step in range(start_step, start_step + n_steps):
        # Validate BEFORE the update, at step boundaries INCLUDING step 0 (train/loop.py
        # 689). Not cosmetic: it makes `step` the number of updates the evaluated weights
        # have had, so a resumed SLURM array task's val curve continues where the previous
        # one ended instead of being shifted by an eval interval, and step 0 is the only
        # record of the seeded model before this job touched it.
        #
        # train/loop.py additionally skips its `end_step`, but that is NOT translated
        # here: its range is `start..end_step` INCLUSIVE with training gated on
        # `step < end_step`, so its last iteration is a validate-only tail that would
        # re-validate exactly what the next job's step 0 validates. This loop has no such
        # tail (it trains every iterated step and checkpoints after), so the duplicate it
        # guards against cannot occur, and excluding the last step here would instead drop
        # a real validation point.
        if on_eval is not None and eval_every and step % eval_every == 0:
            last["eval"] = on_eval(step)
            _track_best(step, last["eval"])

        t0 = time.perf_counter()
        batch = next(train_batches)
        images = task.batch_images(batch, device)
        bound = task.bind(batch, device, model=model, head=head)
        t_data_total += time.perf_counter() - t0

        t1 = time.perf_counter()
        # Viz rides the SAME forward as training (no recomputation), like train/loop.py.
        do_viz = bool(viz_every) and run_dir is not None and step % viz_every == 0 \
            and hasattr(task, "render_viz")
        optimizer.zero_grad()
        result = run_rollout(
            model=model, images=images, task=bound, selector=selector, bptt=spec.bptt,
            branches=branches, canvas_grid_size=canvas_grid, amp_ctx=amp_ctx,
            task_weight=spec.task_weight, collect_viz=do_viz, viz_task=(task if do_viz else None),
            joint=joint,
        )
        # DDP: average gradients across ranks BEFORE clipping (design §9 — nothing is
        # DDP-wrapped, so every trainable module is synced by hand here). Clipping first
        # and averaging after is a different operation and would drift from 1-GPU.
        if is_dist:
            ddp.allreduce_grads(trainable)
        grad_norm = None
        if trainable:
            grad_norm = torch.nn.utils.clip_grad_norm_(trainable, grad_clip)
        if joint is not None:
            if is_dist:  # the scorer carries its own reward standardizers -> its own hook
                joint.allreduce_grads()
            torch.nn.utils.clip_grad_norm_(joint.scorer.parameters(), grad_clip)

        is_log = on_log is not None and step % log_every == 0
        # Read grad norms AFTER clip and BEFORE optimizer.step() (grads persist until
        # the next zero_grad, but reading pre-step keeps it unambiguous); log steps only.
        gnorms = (grad_norms_by_module(core, deep_prefixes=grad_norm_deep_prefixes)
                  if (is_log and log_grad_norms) else {})

        optimizer.step()
        scheduler.step()
        if joint is not None:
            joint.after_optimizer_step()  # QTD target-net Polyak update; no-op otherwise
        t_gpu_total += time.perf_counter() - t1

        # Metrics that are EMA-smoothed before logging (train/loop.py logs ONLY EMAs
        # under these names): the totals plus each branch group's per-glimpse series.
        smoothed: dict[str, torch.Tensor] = {
            "total_loss": result.total_loss,
            "n_glimpses": torch.tensor(float(result.n_glimpses)),
        }
        smoothed.update(branch_metrics(result.branches))

        last = {"step": step, "total_loss_raw": float(result.total_loss),
                "n_glimpses": result.n_glimpses}
        for k, v in smoothed.items():
            val = ema.update(k, v).item() if ema is not None else float(v)
            # Under DDP each rank EMAs its own batches; mean them so the logged series
            # describes the global training population (train/loop.py 913).
            last[k] = ddp.all_reduce_mean(val) if is_dist else val
        # Instantaneous (never smoothed), matching train/loop.py's metrics dict.
        if grad_norm is not None:
            last["grad_norm"] = float(grad_norm)
        if spec.bptt.continue_prob is not None:
            last["continue_prob"] = spec.bptt.continue_prob
        if log_timing and (t_data_total + t_gpu_total) > 0:
            tot = t_data_total + t_gpu_total
            last["data_pct"] = 100.0 * t_data_total / tot
            last["gpu_pct"] = 100.0 * t_gpu_total / tot
        if result.policy_metrics is not None:
            # Pass every key through rather than the three fixed ones, so an objective can
            # publish its own diagnostics (VPG: policy_entropy / value_mean / adv_std / the
            # two loss terms) without editing the loop. reward_frac / policy_loss /
            # prime_on_policy keep their names, so the existing wandb series are unchanged.
            for k, v in result.policy_metrics.items():
                last[k] = float(v)
        for k, v in gnorms.items():
            last[f"grad_norm/{k}"] = v

        if result.viz is not None and run_dir is not None:
            try:  # viz is a diagnostic — never let a plotting error kill training
                task.render_viz(result.viz, batch=batch, run_dir=run_dir, step=step)
            except Exception:
                log.exception("viz rendering failed at step %d (training continues)", step)

        if is_log:
            on_log(step, last)
        if ckpt_every and step > start_step and step % ckpt_every == 0:
            _save(step)

        # SIGUSR1 (SLURM preemption / manual): checkpoint at this safe boundary.
        if signal_checkpoint and _checkpoint_requested:
            log.info("checkpoint-on-signal at step %d", step)
            _save(step)
            _checkpoint_requested = False

    _save(start_step + n_steps)
    return last


__all__ = [
    "apply_requires_grad", "branch_metrics", "cancel_slurm_array", "grad_norms_by_module",
    "install_sigusr1_handler", "request_checkpoint", "run_training_loop",
]
