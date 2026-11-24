"""The single run orchestration for the unified harness (design §2 ``run.py``, D-B).

``run(task, spec, settings)`` is the one path all three peer tasks take:

    build_model → build_policy (joint) → apply_requires_grad → build_optimizer
    → build_loaders → build_selector → run_training_loop (+ eval/log/ckpt hooks)

Everything task-specific lives behind the ``RunTask`` seam (a run-level ``Task``:
``tasks/{distill,ade20k,in1k}/task.py``); everything task-neutral is here or in the
sibling harness modules (rollout / optim / checkpoint / loop / policy). This subsumed the
four former entry points (the ``train`` distill loop, ``ade20k.train``, ``in1k.train`` and
``ade20k.rl_train``) into one ``TrainSpec``-driven call, and as of the 2026-07-31
consolidation those four are DELETED — this is the only training entry point in the repo.
Historical launchers still reach them via commit pinning; see the README.

Config is **composed** (D-B), not one mega-dataclass: the task holds its own per-task
config; :class:`RunSettings` carries the harness-level, cross-cutting knobs; ``TrainSpec``
carries what-trains-under-which-loss. ``run`` writes checkpoints LOCALLY only (D-G); HF
publishing stays the manual ``python -m canvit.checkpoint.to_hf`` step.
"""

from __future__ import annotations

# DDP-safety: a per-rank/per-job MPLCONFIGDIR + the Agg backend, set BEFORE matplotlib is
# imported (some of this module's imports pull it transitively, and the distill viz path
# uses it). matplotlib's default ~/.cache/matplotlib on shared NFS races on the font cache
# across DDP ranks / concurrent jobs and hangs; this mirrors the old train/__main__.py.
# run.py IS the harness entry point (`python -m canvit.harness.run`), so guarding
# at module import is the earliest hook. os.environ.setdefault => an explicit MPLCONFIGDIR
# is respected; the /tmp dir is per-rank/job so ranks never share a cache.
import os

os.environ.setdefault(
    "MPLCONFIGDIR",
    f"/tmp/mpl_config_rank{os.environ.get('SLURM_PROCID', '0')}_job{os.environ.get('SLURM_JOB_ID', 'nojob')}",
)
os.makedirs(os.environ["MPLCONFIGDIR"], exist_ok=True)
import matplotlib

matplotlib.use("Agg")

# Gradient correctness under torch.compile (compile defaults ON for distill). backward()
# runs OUTSIDE autocast; torch.compile's default "same_as_forward" then silently corrupts
# gradients. train/loop.py sets this at its module top — mirror it here (unconditional; a
# no-op when compile is off) so compiled harness runs match the old loop's gradients.
import torch._functorch.config

torch._functorch.config.backward_pass_autocast = "off"  # type: ignore[attr-defined]
# The task/policy gradient split (loop: grad_split/*, log steps with policy_grad_to_backbone) runs
# autograd.grad(..., retain_graph=True) before the chunk's backward. A compiled backward with
# donated buffers refuses retain_graph (exp40 T3-full, job 17483087, 2026-11-23: "compiled with
# non-empty donated buffers which requires ... retain_graph=False"). Donation is only a memory
# optimisation; turning it off changes no numbers.
torch._functorch.config.donated_buffer = False  # type: ignore[attr-defined]

import logging
import time
from contextlib import nullcontext
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, Protocol, runtime_checkable

import torch

from canvit.harness.infra import ddp
from canvit.harness.infra.checkpoint import find_latest, load_checkpoint, restore_into
from canvit.harness.loop import (
    apply_requires_grad,
    cancel_slurm_array,
    install_sigusr1_handler,
    run_training_loop,
)
from canvit.harness.optim import build_optimizer_and_scheduler
from canvit.harness.rollout.viewpoint import ViewpointType
from canvit.harness.spec import TaskCaps, TrainSpec, check_spec

log = logging.getLogger(__name__)


@dataclass
class RunSettings:
    """Harness-level, task-neutral run knobs (design D-B composed config). Task-specific
    settings (data paths, model repo, augmentation, batch size, horizon) stay in the
    task's own config; ``TrainSpec`` owns what-trains-under-which-loss."""

    n_steps: int = 100
    start_step: int = 0
    grad_clip: float = 1.0
    amp: bool = True
    amp_dtype: Literal["bfloat16", "float16", "float32"] = "bfloat16"
    device: str = "cuda"
    seed: int = 0
    # DDP (single-GPU default; the loop AllReduces the scorer when world_size>1)
    world_size: int = 1
    rank: int = 0
    # cadence
    log_every: int = 20
    ckpt_every: int = 0          # 0 => only the end-of-run checkpoint
    eval_every: int = 0          # 0 => no periodic eval
    ckpt_dir: Path | None = None
    # resume + operational (task-neutral; parity with train/loop.py)
    resume: bool = True          # resume from find_latest(ckpt_dir) if a checkpoint exists
    signal_checkpoint: bool = True   # SIGUSR1 -> checkpoint after the current step
    use_failed_marker: bool = False  # SLURM crash-loop guard: FAILED file + scancel array
    ema_alpha: float = 0.1       # EMA-smoothed total_loss in the logs (0 => off)
    log_grad_norms: bool = True  # per-module grad L2 norms on log steps
    grad_norm_deep_prefixes: tuple[str, ...] = ()
    seed_ckpt: Path | None = None
    """SEED mode: load WEIGHTS ONLY from this local checkpoint and start a fresh run at
    step 0 (fresh optimizer/scheduler) — how production runs start from a pretrained
    model. Distinct from ``resume`` (full state, continues the step count), which takes
    priority: resume > seed_ckpt > fresh. Task-level HF seeding (distill's
    ``cfg.hf_seed_ckpt``, ade20k/in1k's ``cfg.model_repo``) happens in ``build_model``."""
    seed_optimizer: bool = False
    """With ``seed_ckpt``: also take over the seed's AdamW state (exp_avg, exp_avg_sq, step) for
    the parameters of the seed's single optimizer group, matched by name, into this run's
    ``backbone`` group (see :func:`load_seed_optimizer_state`). Other groups (policy) start
    fresh; lr, weight decay, betas and the schedule stay this run's."""
    seed_exp_avg_sq_scale: float = 1.0
    """Multiply the taken-over exp_avg_sq by this: the squared-gradient average grows when the
    batch shrinks (measured ~3x from exp38's 4x64 to 1x64, 2026-11-24)."""
    compile: bool = False        # torch.compile the model after build (perf)
    run_dir: Path | None = None
    """Root for this run's artifacts (`{run_dir}/checkpoints`, `{run_dir}/visualization`),
    the `logs_dir/run_group/run_name` convention. When set and ``ckpt_dir`` is not,
    checkpoints go to ``run_dir/checkpoints``."""
    log_timing: bool = True      # log cumulative data% vs gpu% (bottleneck analysis)
    viz_every: int = 0
    """Render the task's training-batch visualization every N steps (0 = off). Figures
    are written LOCALLY under ``run_dir/visualization/`` — never uploaded (D-G). Only
    tasks that implement ``render_viz`` (currently distill's multistep PCA) produce any."""
    # experiment tracker (off by default so run() is import-safe / offline)
    tracker: Literal["wandb", "none"] = "none"
    wandb_project: str | None = None
    wandb_entity: str | None = None
    wandb_dir: Path | None = None
    run_name: str | None = None
    run_config: dict | None = None
    """The run's full configuration (``cli.run_config_of``: the command line + every resolved
    field), set by the CLI. Saved in every checkpoint (``metadata["run_config"]``) and as
    ``run_dir/run_config.json``, so ``harness.evaluate --opts.ckpt`` can rebuild the run alone."""


@runtime_checkable
class RunTask(Protocol):
    """What ``run`` needs from a task (the run-level seam; the per-glimpse ``RolloutTask``
    seam in ``rollout.py`` is what ``bind`` returns). Concrete impls: the ``*RunTask``
    classes in ``tasks/<name>/task.py``."""

    name: str

    def caps(self) -> TaskCaps: ...
    def default_spec(self) -> TrainSpec: ...
    # prior_model_config: the resume checkpoint's model_config, which must win over
    # this run's config defaults (None when not resuming).
    def build_model(self, device: torch.device,
                    prior_model_config: dict | None = None) -> tuple[Any, Any]: ...  # (model, head|None)
    def canvas_grid(self, model: Any) -> int: ...
    def is_foveated(self, model: Any) -> bool: ...
    def branches(self) -> list[ViewpointType]: ...
    def build_loaders(self, *, world_size: int, rank: int) -> tuple[Any, Any]: ...  # (train_iter, val)
    def build_selector(self, *, device: torch.device, canvas_grid: int, is_foveated: bool) -> Any: ...
    def build_policy(self, model: Any, *, device: torch.device, canvas_grid: int,
                     generator: torch.Generator) -> Any: ...
    def trainable_param_groups(self, *, model: Any, head: Any, joint: Any,
                               spec: TrainSpec) -> dict[str, list]: ...
    def resume_start_step(self, payload: dict, scheduler: Any) -> int: ...  # default: scheduler.last_epoch
    def resume_state(self) -> dict: ...  # extra state this task needs to resume its data schedule
    def batch_images(self, batch: Any, device: torch.device) -> torch.Tensor: ...
    def bind(self, batch: Any, device: torch.device, *, model: Any, head: Any) -> Any: ...
    def evaluate(self, *, model: Any, head: Any, val_loader: Any, device: torch.device,
                 step: int, tracker: Any = None, run_dir: Path | None = None,
                 joint: Any = None) -> dict: ...
    # `joint` is the run's JointPolicy (None when no policy is trained) — needed so a task
    # can validate under `eval_policy="policy"`, i.e. deploy the scorer it just trained
    # instead of the open-loop trajectory it inherited. See harness/eval_viewpoints.py.
    def model_config(self, model: Any) -> dict: ...
    def checkpoint_metadata(self, model: Any) -> dict: ...


def _as_dict(cfg: Any) -> dict:
    """A task config as a plain dict (dataclass or not); {} when there is none."""
    from dataclasses import asdict, is_dataclass
    if cfg is None:
        return {}
    return asdict(cfg) if is_dataclass(cfg) else dict(getattr(cfg, "__dict__", {}))


def _flatten(d: dict, prefix: str = "") -> dict[str, Any]:
    """Nested dict -> dotted keys, non-scalars stringified (train/loop.py flatten_dict)."""
    flat: dict[str, Any] = {}
    for k, v in d.items():
        key = f"{prefix}{k}" if prefix else k
        if isinstance(v, dict):
            flat.update(_flatten(v, f"{key}."))
        else:
            flat[key] = v if isinstance(v, (int, float, bool, str, type(None))) else str(v)
    return flat


def _infinite(loader: Any):
    """Yield batches forever. Map-style loaders (ade20k) are re-iterated at exhaustion;
    webdataset/streaming loaders (distill/in1k) are effectively infinite already but this
    is harmless."""
    if hasattr(loader, "next"):        # distill WebDatasetTrainLoader
        while True:
            yield loader.next()
    while True:
        for batch in loader:
            yield batch


def load_seed_optimizer_state(optimizer: Any, names: list[str], params: list[Any], seed_payload: dict,
                              *, exp_avg_sq_scale: float = 1.0) -> int:
    """Copy a seed checkpoint's AdamW state into ``optimizer`` for ``params`` (named ``names``).

    The seed optimizer must have ONE param group; its params are identified by name through the
    seed's ``model_state`` order (a state_dict lists parameters in ``named_parameters`` order).
    The seed's parameters must be exactly ``names`` minus the ones new in this run (absent from
    the seed's model_state); every shape must match. Returns the number of tensors with state."""
    o = seed_payload.get("optimizer_state")
    if not o:
        raise ValueError("seed_optimizer: the seed checkpoint has no optimizer_state")
    if len(o["param_groups"]) != 1:
        raise ValueError(f"seed_optimizer: seed has {len(o['param_groups'])} param groups, expected 1")
    want = set(names)
    seed_names = [k for k in seed_payload["model_state"] if k in want]
    seed_ids = o["param_groups"][0]["params"]
    if len(seed_names) != len(seed_ids):
        raise ValueError(f"seed_optimizer: seed group has {len(seed_ids)} params, its model_state "
                         f"{len(seed_names)} of this run's backbone params")
    by_name = {n: o["state"][i] for n, i in zip(seed_names, seed_ids) if i in o["state"]}
    n_loaded = 0
    for name, p in zip(names, params, strict=True):
        st = by_name.get(name)
        if st is None:
            continue
        if st["exp_avg"].shape != p.shape:
            raise ValueError(f"seed_optimizer: {name} shape {tuple(st['exp_avg'].shape)} != {tuple(p.shape)}")
        optimizer.state[p] = {
            "step": st["step"].detach().to("cpu", torch.float32).clone(),  # as AdamW keeps it (not fused/capturable)
            "exp_avg": st["exp_avg"].to(device=p.device, dtype=p.dtype).clone(),
            "exp_avg_sq": (st["exp_avg_sq"].to(device=p.device, dtype=p.dtype) * exp_avg_sq_scale).clone(),
        }
        n_loaded += 1
    return n_loaded


def run(*, task: RunTask, spec: TrainSpec, settings: RunSettings) -> dict:
    """Train ``task`` under ``spec`` for ``settings.n_steps`` and return the last
    step's metrics. Task-neutral; every task-specific decision is delegated to ``task``.
    """
    # Topology first: under srun the environment knows the real rank/world size, and
    # RunSettings' single-GPU defaults would make every rank think it trains alone.
    dinfo = ddp.setup(device=settings.device, rank=settings.rank, world_size=settings.world_size)
    rank, world_size, device, is_dist = dinfo.rank, dinfo.world_size, dinfo.device, dinfo.is_dist
    use_cuda = device.type == "cuda"
    torch.manual_seed(settings.seed + rank)

    caps = task.caps()
    report = check_spec(spec, caps, is_dist=is_dist)
    if not report.ok:
        raise ValueError("invalid TrainSpec:\n  - " + "\n  - ".join(report.errors))
    for w in report.warnings:
        log.warning("TrainSpec warning: %s", w)

    log.info("=" * 60)
    log.info("Unified harness run: task=%s device=%s n_steps=%d", task.name, device, settings.n_steps)
    log.info("spec: train(bb=%s head=%s policy=%s) task_w=%.3g policy_w=%.3g task->bb=%s pol->bb=%s bptt=%s",
             spec.train_backbone, spec.train_head, spec.train_policy, spec.task_weight,
             spec.policy_weight, spec.task_grad_to_backbone, spec.policy_grad_to_backbone, spec.bptt)
    log.info("=" * 60)

    if settings.signal_checkpoint:
        install_sigusr1_handler()

    # Run-dir convention (logs_dir/run_group/run_name): checkpoints + visualizations
    # live under it. An explicit ckpt_dir still wins.
    run_dir = settings.run_dir
    ckpt_dir = settings.ckpt_dir or ((run_dir / "checkpoints") if run_dir else None)
    if ckpt_dir is None:
        # train/loop.py asserted `run_group is not None` here, so this case could not
        # happen; the harness derives everything and would instead train happily and throw
        # the weights away. Warn rather than raise: the parity/smoke harnesses run without
        # a run dir on purpose.
        log.warning("no ckpt_dir and no run_dir — this run will NOT write checkpoints "
                    "(pass --cfg.run-group/--cfg.run-name, or --opts.ckpt-dir)")

    # SLURM crash-loop guard (opt-in): a FAILED file left by a prior crash stops the
    # array from re-crashing forever (see the failed-token-blocks-resume incident).
    failed_marker = ((run_dir or ckpt_dir) / "FAILED") if (run_dir or ckpt_dir) else None
    if settings.use_failed_marker and failed_marker is not None and failed_marker.exists():
        log.error("FAILED marker exists (%s) — a previous job crashed; delete it to retry.", failed_marker)
        if rank == 0:
            cancel_slurm_array()
        raise RuntimeError(f"Refusing to start: {failed_marker} exists")

    # --- resume checkpoint is resolved BEFORE the model is built, because its
    # model_config must WIN over CLI defaults: building from the defaults and then
    # strict-loading saved weights of a different arch fails on missing/unexpected keys
    # (train/loop.py 254-261). ---------------------------------------------
    latest = find_latest(ckpt_dir) if (settings.resume and ckpt_dir is not None) else None
    prior_ckpt: dict | None = load_checkpoint(latest, device) if latest is not None else None

    # --- model + (optional) joint policy -----------------------------------
    model, head = task.build_model(
        device, prior_model_config=(prior_ckpt or {}).get("model_config") or None)
    canvas_grid = task.canvas_grid(model)
    is_foveated = task.is_foveated(model)

    if settings.compile:
        # Refuse rather than pretend: this compiles the WRAPPER's forward, which only
        # distill's task actually calls. A probe task steps `model.canvit(...)` /
        # `model.head(...)` directly, so this would return happily and change nothing —
        # you would pay the compile warmup and measure "no speedup" without a clue why.
        if not caps.supports_compile:
            raise ValueError(
                f"compile=True but task '{task.name}' does not support wrapper-level "
                "torch.compile: it bypasses the wrapper's forward (it steps .canvit / .head "
                "directly), so compiling here is a silent no-op. Compiling this task means "
                "compiling .canvit explicitly — a code change, not a flag.")
        log.info("Compiling model (torch.compile)")
        compiled = getattr(model, "compile", None)
        if callable(compiled):
            compiled()          # nn.Module.compile: patches this module's __call__
        else:
            model = torch.compile(model)  # type: ignore[assignment]

    joint = None
    if spec.train_policy or spec.policy_loss_active:
        gen = torch.Generator(device=device).manual_seed(settings.seed + rank)
        joint = task.build_policy(model, device=device, canvas_grid=canvas_grid, generator=gen)
        # ONE switch for "does the policy gradient reach the backbone": the spec's
        # policy_grad_to_backbone. The selector's feats_detached (from rl.feats_detached,
        # default True) used to be a second, independent switch, so
        # `--spec.policy-grad-to-backbone True` alone still detached the policy features and no
        # policy gradient reached the backbone, silently (found 2026-11-22; no launcher had
        # set either). The spec wins; a policy-token model cannot detach at all.
        sel = joint.policy_selector
        core_cfg = getattr(getattr(model, "module", model), "cfg", None)
        if getattr(core_cfg, "n_policy_tokens", 0):
            isolated = getattr(core_cfg, "policy_token_mode", None) == "isolated"
            if isolated and spec.policy_grad_to_backbone:
                raise ValueError(
                    "isolated policy tokens read the reconstruction path only detached, so no policy "
                    "gradient reaches it: use --spec.policy-grad-to-backbone False")
            if not isolated and not spec.policy_grad_to_backbone:
                raise ValueError(
                    "policy tokens are processed by the backbone, so the policy gradient reaches "
                    "it by construction: use --spec.policy-grad-to-backbone True")
            if isolated and spec.bptt.mode == "none":
                # bptt 'none' (a frozen model: --preset policy_only) runs every model forward
                # under no_grad, and the isolated policy stream lives INSIDE that forward: only
                # the readout would train, the stream and the token init never. 'full' is the
                # same cadence (one backward per rollout) with grad on; the frozen trunk still
                # records no graph (no parameter or input of it requires grad). Found 2026-11-24
                # before the first exp41 run.
                from dataclasses import replace as _replace
                spec = _replace(spec, bptt=_replace(spec.bptt, mode="full"))
                log.info("isolated policy tokens on a frozen model: bptt 'none' -> 'full' so the "
                         "policy stream trains (horizon %s)", spec.bptt.horizon)
        elif hasattr(sel, "feats_detached") and sel.feats_detached != spec.feats_detached:
            log.warning("policy selector feats_detached=%s overridden to %s to match "
                        "spec.policy_grad_to_backbone=%s", sel.feats_detached, spec.feats_detached,
                        spec.policy_grad_to_backbone)
            sel.feats_detached = spec.feats_detached
        # A terminal-reward objective cannot honor every spec cell; fail before training.
        from canvit.harness.policy import check_credit_regime
        check_credit_regime(joint=joint, spec=spec)

    # The harness — not the task — decides what trains (design §3.1).
    apply_requires_grad(model=model, head=head, joint=joint, spec=spec)

    # --- optimizer + scheduler (per trainable group, design D-E) -----------
    param_groups = task.trainable_param_groups(model=model, head=head, joint=joint, spec=spec)
    optimizer, scheduler = build_optimizer_and_scheduler(spec, param_groups)

    # --- RESUME / SEED / FRESH (task-neutral), BEFORE loaders so distill's
    # model-owned normalizer arrives already-initialized from the checkpoint --------
    # Priority mirrors train/loop.py: resume (full state, continues the step count)
    # > seed_ckpt (WEIGHTS ONLY, fresh opt/sched at step 0) > fresh.
    start_step = settings.start_step
    if prior_ckpt is not None:
        log.info("RESUME mode: continuing from %s", latest)
        restore_into(prior_ckpt, model=model, optimizer=optimizer, scheduler=scheduler,
                     joint=joint, path=latest, device=device)
        start_step = task.resume_start_step(prior_ckpt, scheduler)
        log.info("RESUME: start_step=%d", start_step)
    elif settings.seed_ckpt is not None:
        log.info("SEED mode: loading weights from %s (fresh opt/sched, step 0)", settings.seed_ckpt)
        seed_payload = load_checkpoint(settings.seed_ckpt, device)
        # weights only: no optimizer/scheduler => the run starts a fresh schedule.
        # A model with policy tokens seeded from one without: only those params start fresh.
        core = getattr(model, "module", model)
        new_params = (("policy_token_init", "policy_readout.", "policy_stream.")
                      if getattr(getattr(core, "cfg", None), "n_policy_tokens", 0) else ())
        restore_into(seed_payload, model=model, joint=joint, path=settings.seed_ckpt, device=device,
                     new_param_prefixes=new_params)
        stream = getattr(core, "policy_stream", None)
        if stream is not None and not any(k.startswith("policy_stream.") for k in seed_payload["model_state"]):
            # An isolated stream seeded from a model without one starts as a copy of the
            # (now pretrained) trunk: at step 0 it computes what read-only tokens would.
            stream.copy_from_trunk(core)
            log.info("SEED: isolated policy stream initialised from the seeded trunk")
        if settings.seed_optimizer:
            named = {id(p): n for n, p in core.named_parameters()}
            n = load_seed_optimizer_state(
                optimizer, [named[id(p)] for p in param_groups["backbone"]],
                list(param_groups["backbone"]), seed_payload, exp_avg_sq_scale=settings.seed_exp_avg_sq_scale)
            log.info("SEED: AdamW state of %d backbone tensors taken over (exp_avg_sq x %g)",
                     n, settings.seed_exp_avg_sq_scale)
        start_step = 0
    else:
        log.info("FRESH mode: no checkpoint, starting from scratch")

    # --- data + selector ---------------------------------------------------
    train_loader, val_loader = task.build_loaders(world_size=world_size, rank=rank)
    # Every rank must start from IDENTICAL weights and buffers, or averaged gradients are
    # applied to diverging models. Done after build_loaders because distill's standardizer
    # stats are model buffers filled there from one shard (train/loop.py 583-585), and
    # after the resume/seed restore so the loaded state is what gets broadcast.
    if is_dist:
        ddp.broadcast_parameters(model, joint.scorer if joint is not None else None,
                                 *((joint.critic, joint.target_net) if joint is not None else ()))
        log.info("DDP: broadcast parameters + buffers from rank 0")
    train_batches = _infinite(train_loader)
    selector = task.build_selector(device=device, canvas_grid=canvas_grid, is_foveated=is_foveated)
    branches = task.branches()
    if joint is not None and joint.second_branch is not None:
        if joint.keep_random_branch:
            raise ValueError("rl.second_branch replaces rl.keep_random_branch; set only one")
        branches = joint.branches(branches)
        log.info("branches (rl.second_branch=%s): %s", joint.second_branch, [b.name for b in branches])

    amp_ctx: Any = nullcontext()
    if settings.amp and use_cuda:
        # bfloat16 needs sm_80+ (Ampere); fall back to float16 on older GPUs rather than
        # silently running an emulated/broken autocast (train/loop.py 455-460).
        amp_dtype = settings.amp_dtype
        if amp_dtype == "bfloat16" and torch.cuda.get_device_capability(device) < (8, 0):
            log.warning("bfloat16 needs sm_80+; this GPU is sm_%d%d — using float16",
                        *torch.cuda.get_device_capability(device))
            amp_dtype = "float16"
        amp_ctx = torch.autocast("cuda", dtype=getattr(torch, amp_dtype))

    # --- tracker (optional) ------------------------------------------------
    prior_meta = (prior_ckpt or {}).get("metadata", {}) if prior_ckpt else {}
    tracker = None
    if settings.tracker == "wandb":
        from canvit.harness.infra.tracker import make_tracker
        # Resume the SAME wandb run across SLURM array tasks (train/loop.py 249-253):
        # a 245-job array is ONE experiment, and without the id round-trip each task
        # would open its own run and the curves would come out in 245 pieces.
        tracker = make_tracker(
            tracker="wandb", is_main=(rank == 0), is_seeding=False,
            run_name=settings.run_name or f"{task.name}-unified",
            wandb_project=settings.wandb_project, wandb_entity=settings.wandb_entity,
            wandb_dir=settings.wandb_dir, prev_comet_id=None,
            prev_wandb_id=prior_meta.get("wandb_run_id"),
        )
        # Hyperparameters (train/loop.py 279-282, 423): the task's flattened config,
        # the resolved TrainSpec, the SLURM job id and the trainable/total param split.
        n_total = sum(p.numel() for p in model.parameters())
        n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        params: dict[str, Any] = {
            **_flatten(_as_dict(getattr(task, "cfg", None))),
            "train_spec": str(spec), "task": task.name, "run_name": settings.run_name or task.name,
            "trainable_params": n_trainable, "total_params": n_total,
        }
        if slurm_job_id := os.environ.get("SLURM_JOB_ID"):
            params["slurm_job_id"] = slurm_job_id
        tracker.log_parameters(params)

    def on_log(step: int, m: dict) -> None:
        extra = ""
        if "reward_frac" in m:
            extra = f"  reward_frac={m['reward_frac']:+.4f}  policy_loss={m['policy_loss']:.4f}"
        log.info("step %d  loss=%.5f  n_glimpses=%.1f  lr=%.2e%s",
                 step, m["total_loss"], m["n_glimpses"], scheduler.get_last_lr()[0], extra)
        if tracker is not None:
            # train/loop.py's namespacing: everything under `train/` except the
            # per-module gradient norms, which have their own `grad_norm/` namespace.
            payload = {(k if k.startswith("grad_norm/") else f"train/{k}"): v
                       for k, v in m.items() if k != "step"}
            payload["train/lr"] = scheduler.get_last_lr()[0]
            tracker.log_metrics(payload, step=step)

    def on_best(step: int, name: str, value: float) -> None:
        """The loop decides + writes ``best.pt``; this only mirrors the running best to
        the tracker (ade20k/train.py logs a ``best_val_miou_t*`` series alongside)."""
        if tracker is not None:
            tracker.log_metrics({f"eval/best_{name}": value}, step=step)

    def on_eval(step: int) -> dict:
        # tracker/run_dir go through so a task's validation can log its OWN rich series
        # (distill's per-timestep val curves + probe accuracy) and write its figures to
        # the run dir — train/loop.py passed `exp` and `run_dir` straight into validate().
        # Validation is rank-0 only over the fixed subset (identical samples regardless of
        # world size — the same choice train/loop.py 692-696 makes); the others wait at the
        # barrier. Safe because evaluate() runs under no_grad with no collectives.
        # Exception: a task with `distributed_eval = True` (distill) is called on EVERY rank;
        # it splits the fixed subset by rank and all-reduces inside, and only rank 0 logs.
        distributed = is_dist and getattr(task, "distributed_eval", False)
        if is_dist and rank != 0 and not distributed:
            ddp.barrier()
            return {}
        t_val = time.perf_counter()
        extra = {"rank": rank, "world_size": world_size} if distributed else {}
        metrics = task.evaluate(model=model, head=head, val_loader=val_loader, device=device,
                                step=step, tracker=tracker, run_dir=run_dir, joint=joint, **extra)
        if is_dist and rank != 0:  # distributed eval: rank 0 holds the same metrics and logs them
            ddp.barrier()
            return {}
        val_seconds = time.perf_counter() - t_val
        log.info("step %d  eval (%.1fs): %s", step, val_seconds, metrics)
        if tracker is not None:
            # Per-task namespace: distill declares "val", the prefix its series have carried
            # since the old loop and which `validate` logged itself until it started
            # returning them instead (F8). Everyone else takes "eval".
            ns = getattr(task, "metrics_prefix", "eval")
            payload = {f"{ns}/{k}": v for k, v in metrics.items()}
            # How long validation costs, under the key ade20k/train.py logged it as. For a
            # probe this is the dominant non-training cost (63 val batches every 500
            # steps), so losing the series made it invisible.
            payload["timing/val_seconds"] = val_seconds
            tracker.log_metrics(payload, step=step)
        if is_dist:
            ddp.barrier()
        return metrics

    # Checkpoint metadata: ACCUMULATE config + provenance history across resumes
    # (train/loop.py semantics — `to_hf` reads training_config_history to recover
    # `pretrain_view_scale`, the foveated footgun; a single snapshot would lose the
    # earlier legs of a multi-job run).
    from datetime import UTC, datetime

    from canvit.checkpoint import current_provenance

    task_meta = task.checkpoint_metadata(model)
    cfg_history = dict(prior_meta.get("training_config_history") or {})
    prov_history = dict(prior_meta.get("provenance_history") or {})
    _now = datetime.now(UTC).isoformat()
    cfg_history[_now] = {**task_meta, "train_spec": str(spec)}
    prov_history[_now] = current_provenance()
    metadata = {
        **task_meta,
        # Carried forward so the NEXT array task resumes this same wandb run rather
        # than starting a fresh one (see the tracker block above).
        "wandb_run_id": (tracker.get_wandb_id() if tracker is not None
                         else prior_meta.get("wandb_run_id")),
        # What the NEXT job needs to resume this task's DATA schedule (distill's
        # WebDataset job_index + the invariants it is only valid under; {} for the
        # map-style tasks). Built after build_loaders, which is where it becomes known.
        "resume_state": task.resume_state(),
        "training_config_history": cfg_history,
        "provenance_history": prov_history,
        # This job's full configuration (a programmatic run without one keeps the prior's).
        "run_config": settings.run_config or prior_meta.get("run_config"),
    }
    if settings.run_config is not None and run_dir is not None and rank == 0:
        import json
        run_dir.mkdir(parents=True, exist_ok=True)
        (run_dir / "run_config.json").write_text(json.dumps(settings.run_config, indent=1) + "\n")

    try:
        last = run_training_loop(
            task=task, model=model, head=head, optimizer=optimizer, scheduler=scheduler,
            selector=selector, spec=spec, branches=branches, canvas_grid=canvas_grid, device=device,
            train_batches=train_batches, n_steps=settings.n_steps, start_step=start_step,
            joint=joint, amp_ctx=amp_ctx, grad_clip=settings.grad_clip, is_dist=is_dist, rank=rank,
            task_name=task.name, model_config=task.model_config(model),
            metadata=metadata, log_every=settings.log_every,
            ckpt_dir=ckpt_dir, ckpt_every=settings.ckpt_every, eval_every=settings.eval_every,
            ema_alpha=settings.ema_alpha, log_grad_norms=settings.log_grad_norms,
            log_timing=settings.log_timing,
            viz_every=settings.viz_every, run_dir=run_dir,
            grad_norm_deep_prefixes=settings.grad_norm_deep_prefixes,
            signal_checkpoint=settings.signal_checkpoint,
            best_metric=getattr(task, "best_metric", None),
            on_log=on_log, on_eval=(on_eval if settings.eval_every else None),
            on_best=on_best,
        )
    except Exception:
        # SLURM crash-loop guard: leave a FAILED marker + stop the array so the next
        # task doesn't re-crash on the same fault (opt-in; matches train/loop.py).
        if settings.use_failed_marker and failed_marker is not None and rank == 0:
            failed_marker.parent.mkdir(parents=True, exist_ok=True)
            failed_marker.write_text("crashed\n")
            cancel_slurm_array()
        raise
    if tracker is not None:
        tracker.end()
    log.info("run complete: %s", last)
    return last


def main(argv: list[str] | None = None) -> None:
    """``python -m canvit.harness.run {distill,ade20k,in1k} [flags]``.

    The CLI itself lives in :mod:`canvit.harness.cli` — tyro over each task's
    OWN config dataclass, so every field (incl. the nested ``--cfg.model.*`` patcher and
    ``--cfg.foveated-scale.*`` trees) is reachable. Imported lazily so ``run.py`` stays
    import-light and offline-safe."""
    from canvit.harness.cli import main as cli_main

    cli_main(argv)


__all__ = ["RunSettings", "RunTask", "run", "main"]


if __name__ == "__main__":
    main()
