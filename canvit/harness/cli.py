"""Full-fidelity CLI for the unified harness (``python -m canvit.harness.run``).

Built on ``tyro`` over each task's OWN config dataclass — the same idiom the three
standalone entry points already use (``train/__main__.py``, ``ade20k/__main__.py``,
``in1k/__main__.py``). Every config field is therefore reachable from the command
line, including the nested trees a hand-rolled argparse could not express:

    # foveated pretraining (exp22-style)
    python -m canvit.harness.run distill \
        --cfg.model.patcher-name foveated --cfg.foveated-scale.mode per_rollout \
        --cfg.run-group foveated --cfg.webdataset-dir /path/to/wds

    # ADE20K linear probe, and joint probe+policy
    python -m canvit.harness.run ade20k --preset probe
    python -m canvit.harness.run ade20k --preset joint --rl.use-rl True

**The task config is the source of truth.** :class:`~canvit.harness.run.RunSettings`
is DERIVED from it — ``cfg.steps_per_job`` sets ``n_steps``, ``cfg.val_every`` sets
``eval_every``, and ``compile`` / ``amp`` / ``grad_clip`` / ``tracker`` / ``wandb_*`` /
``seed_ckpt`` carry over — so a config that reproduced a run under the old entry point
reproduces it here, with no second place to set the same thing. The handful of knobs
with no task-config counterpart live in :class:`HarnessOpts` (``--opts.*``).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Annotated, Any, Literal

import tyro

from canvit.ade20k.config import Ade20kConfig
from canvit.distill.config import Config
from canvit.harness.config import JointPolicyConfig
from canvit.harness.run import RunSettings, run
from canvit.harness.spec import BpttSpec, GroupOptim, ScheduleSpec, TrainSpec, fixed_horizon_bptt
from canvit.in1k.config import In1kConfig
from canvit.search.config import SearchConfig, search_policy_config

log = logging.getLogger(__name__)

PresetName = Literal["default", "probe", "finetune", "policy_only", "joint"]


@dataclass
class HarnessOpts:
    """Harness-level knobs with no task-config counterpart (everything else is read
    off the task config, which stays the single source of truth)."""

    n_steps: int | None = None
    """Steps this job runs. None => the task's natural job length (distill:
    ``cfg.steps_per_job``, ade20k/in1k: ``cfg.max_steps``). Set it explicitly for
    short smoke runs."""
    eval_every: int | None = None
    """Validate every N steps. None => the task's own cadence (``cfg.val_every``)."""
    seed: int | None = None
    """RNG seed (``torch.manual_seed(seed + rank)``). None => the task config's own seed
    (distill/in1k ``cfg.seed``; ade20k has no seed field so None => 0). Set it to compare
    seed-to-seed variability — e.g. against the UNSEEDED standalone ade20k probe."""
    start_step: int = 0
    ckpt_every: int = 0
    """Periodic checkpoints every N steps (0 => only the end-of-run checkpoint)."""
    viz_every: int = 0
    """Render the task's training-batch visualization every N steps. 0 => the task
    config's own cadence (distill: ``val_every * viz_every_n_vals``; ade20k:
    ``cfg.viz_every``)."""
    ckpt_dir: Path | None = None
    """Explicit checkpoint dir. Overrides the ``logs_dir/run_group/run_name`` convention."""
    run_dir: Path | None = None
    """OVERRIDE for this run's artifact root (``visualization/``, and ``checkpoints/``
    when ``--opts.ckpt-dir`` is unset). Normally leave it unset: all three tasks derive
    ``cfg.logs_dir/cfg.run_group/cfg.run_name`` themselves."""
    ema_alpha: float | None = None
    """EMA smoothing of the logged loss series (0 => log raw). None => the task's own
    default (distill ``cfg.ema_alpha``; ade20k/in1k the harness default 0.1)."""
    resume: bool | None = None
    """Resume from ``find_latest(ckpt_dir)`` if a checkpoint exists. None => the task
    default: True for distill (array jobs must continue across tasks), False for the
    ade20k/in1k probes (single-job launchers mirroring the no-resume standalone, so
    re-running into a populated dir starts fresh instead of silently continuing the
    old run). Pass ``--opts.resume True`` to opt a probe into array-style resume."""
    signal_checkpoint: bool = True
    use_failed_marker: bool = False
    """SLURM crash-loop guard: write a FAILED marker and scancel the array on crash."""
    amp_dtype: Literal["bfloat16", "float16", "float32"] = "bfloat16"
    log_grad_norms: bool = True
    log_timing: bool = True


def _resolve_run_dir(logs_dir: Path, run_group: str, run_name: str | None,
                     *, prefix: str = "") -> tuple[Path, str]:
    """The ``logs_dir/run_group/run_name`` convention (train/loop.py 157-161), with an
    auto-generated timestamp name when ``run_name`` is unset. A group is MANDATORY (see
    :func:`_identity`), so a run dir is always returned and run artifacts can never land
    outside ``logs_dir``.
    ALL THREE tasks resolve their identity here — ade20k used to hardcode the tracker
    name ``"ade20k"`` and in1k defaulted to the constant ``"in1k-clf"``, so unnamed runs
    of either collided in the wandb UI. ``prefix`` labels the auto-generated name
    (distill passes none, keeping the old loop's bare timestamp)."""
    ts = datetime.now().strftime("%Y-%m-%d_%H-%M")
    name = run_name or (f"{prefix}_{ts}" if prefix else ts)
    return logs_dir / run_group / name, name


def _tracker(kind: str) -> str:
    if kind == "comet":
        raise NotImplementedError(
            "the harness supports tracker='wandb' or 'none' (comet was not ported); "
            "pass --cfg.tracker wandb")
    return kind


def _common(opts: HarnessOpts) -> dict[str, Any]:
    """HarnessOpts fields that map straight onto RunSettings."""
    out = {
        "start_step": opts.start_step, "ckpt_every": opts.ckpt_every,
        "viz_every": opts.viz_every,
        "signal_checkpoint": opts.signal_checkpoint,
        "use_failed_marker": opts.use_failed_marker, "amp_dtype": opts.amp_dtype,
        "log_grad_norms": opts.log_grad_norms, "log_timing": opts.log_timing,
    }
    if opts.ema_alpha is not None:   # else: the task's own default (see each build())
        out["ema_alpha"] = opts.ema_alpha
    return out


def _identity(cfg: Any, opts: HarnessOpts, *, prefix: str) -> dict[str, Any]:
    """Run identity + artifact roots + tracker, resolved THE SAME WAY for all three tasks
    from the config trio (``run_group`` / ``run_name`` / ``logs_dir``).

    ``run_group`` is REQUIRED — a run without one is refused. It used to be optional, and
    the fallback was silent and bad: no group meant no run dir, and ade20k/in1k then wrote
    to a flat ``probe_ckpt_dir`` / ``clf_ckpt_dir`` (``$CHECKPOINTS_DIR`` or
    ``checkpoints``, RELATIVE TO THE LAUNCH CWD), so an interactive run from the repo root
    put 3 GB of checkpoints in ``canvit/checkpoints/`` while every launcher-submitted run
    went to ``logs/``, and unrelated runs pooled in one directory overwriting each other's
    ``best.pt``. distill had no such dir and wrote nothing at all.

    ``ckpt_dir`` precedence: ``--opts.ckpt-dir`` > ``run_dir/checkpoints`` (derived in
    ``run()``). There is no third option."""
    # The tracker resolves FIRST so tracker="comet" keeps raising NotImplementedError
    # rather than being masked by the run-group error.
    tracker = _tracker(cfg.tracker)
    if cfg.run_group is None:
        raise ValueError(
            "--cfg.run-group is required: it names the experiment this run belongs to and "
            "fixes where every artifact goes (logs_dir/<run_group>/<run_name>/, holding "
            "checkpoints/ and visualization/). Pass --cfg.run-group <experiment>, "
            "optionally with --cfg.run-name <arm> — else the name is auto-generated from "
            "the task and a timestamp.")
    run_dir, run_name = _resolve_run_dir(cfg.logs_dir, cfg.run_group, cfg.run_name, prefix=prefix)
    return {
        "run_name": run_name, "run_dir": opts.run_dir or run_dir,
        "ckpt_dir": opts.ckpt_dir,
        "tracker": tracker, "wandb_project": cfg.wandb_project,
        "wandb_entity": cfg.wandb_entity, "wandb_dir": cfg.wandb_dir,
    }


@dataclass
class SpecOverrides:
    """Per-field overrides applied ON TOP of ``--preset`` (design §3.3).

    The preset says WHAT TRAINS as a coherent whole — the trainable modules, the loss
    weights, the grad routing, the BPTT regime and each group's tuned LR schedule. These
    override individual fields of that whole, so every combination `TrainSpec` can express
    is reachable from the CLI rather than only the five named ones.

    ``None`` means "leave the preset's value", so an all-default ``SpecOverrides`` is a
    no-op — `resolve_spec` returns exactly what it returned before this existed.

    Deliberately NOT exposed here: ``optim`` and ``bptt``. ``optim`` is a dict of nested
    dataclasses whose tuned per-group LR schedule is the thing presets exist to carry
    (flattening it onto the CLI would generate `--spec.optim.backbone.schedule.warmup-lr-ratio`
    and reintroduce the silent-misconfiguration class that filling it centrally closed).
    ``bptt`` is DERIVED: it follows ``train_backbone``, and is recomputed below when an
    override changes it.

    `check_spec` validates the result whatever its origin, so an incoherent combination is
    rejected before the model is built, and a vacuous-but-runnable one is warned about.
    """

    train_backbone: bool | None = None
    train_head: bool | None = None
    train_policy: bool | None = None
    task_weight: float | None = None
    policy_weight: float | None = None
    task_grad_to_backbone: bool | None = None
    policy_grad_to_backbone: bool | None = None

    def as_dict(self) -> dict[str, Any]:
        """The fields actually set, ready for ``dataclasses.replace``."""
        from dataclasses import fields as _fields

        return {f.name: v for f in _fields(self) if (v := getattr(self, f.name)) is not None}


@dataclass
class DistillCmd:
    """Pretraining: passive -> active dense latent distillation from DINOv3."""

    cfg: Config = field(default_factory=Config)
    preset: PresetName = "default"
    spec: SpecOverrides = field(default_factory=SpecOverrides)
    opts: HarnessOpts = field(default_factory=HarnessOpts)

    def build(self) -> tuple[Any, RunSettings]:
        from canvit.distill.task import DistillRunTask

        settings = RunSettings(
            # The shard-schedule window IS the job length for distill: a job trains
            # exactly steps_per_job steps and the next array task resumes at the next
            # shard block. Decoupling them corrupts the WebDataset resume.
            n_steps=self.opts.n_steps if self.opts.n_steps is not None else self.cfg.steps_per_job,
            eval_every=self.opts.eval_every if self.opts.eval_every is not None else self.cfg.val_every,
            log_every=self.cfg.log_every, grad_clip=self.cfg.grad_clip, amp=self.cfg.amp,
            seed=self.opts.seed if self.opts.seed is not None else self.cfg.seed,
            device=str(self.cfg.device), compile=self.cfg.compile,
            seed_ckpt=self.cfg.seed_ckpt,
            resume=self.opts.resume if self.opts.resume is not None else True,
            **_identity(self.cfg, self.opts, prefix=""),
            # Break `patcher` down into kpe / embed_head / conditioner (and the
            # conditioner one level deeper) so foveated runs get the old loop's
            # `grad_norm/patcher.kpe` + `grad_norm/patcher.conditioner.mlp` series
            # instead of a single aggregate (train/loop.py 881-883).
            grad_norm_deep_prefixes=(("patcher", "patcher.conditioner")
                                     if self.cfg.log_patcher_grad_detail else ()),
            **{"ema_alpha": self.cfg.ema_alpha, **_common(self.opts),
               # `pca_train`: the old loop rendered the TRAINING-batch PCA whenever it
               # validated AND on every viz_every_n_vals-th validation (train/loop.py
               # 680-681) — i.e. exactly `step % (val_every * viz_every_n_vals) == 0`.
               # Left at 0 the harness wrote `pca_val` only (exp23).
               "viz_every": self.opts.viz_every
               or self.cfg.val_every * max(1, self.cfg.viz_every_n_vals)},
        )
        return DistillRunTask(self.cfg), settings

    def lr_wd(self) -> tuple[float, float]:
        return self.cfg.peak_lr, self.cfg.weight_decay


@dataclass
class Ade20kCmd:
    """ADE20K semantic segmentation: frozen probe, finetune, or joint probe+policy."""

    cfg: Ade20kConfig = field(default_factory=Ade20kConfig)
    preset: PresetName = "default"
    spec: SpecOverrides = field(default_factory=SpecOverrides)
    rl: JointPolicyConfig = field(default_factory=JointPolicyConfig)
    """Viewpoint-policy config; only consulted for policy/joint presets."""
    opts: HarnessOpts = field(default_factory=HarnessOpts)

    def build(self) -> tuple[Any, RunSettings]:
        from canvit.ade20k.task import Ade20kRunTask

        settings = RunSettings(
            n_steps=self.opts.n_steps if self.opts.n_steps is not None else self.cfg.max_steps,
            eval_every=self.opts.eval_every if self.opts.eval_every is not None else self.cfg.val_every,
            log_every=self.cfg.log_every, grad_clip=self.cfg.grad_clip, amp=self.cfg.amp,
            seed=self.opts.seed if self.opts.seed is not None else self.cfg.seed,
            device=self.cfg.device,
            resume=self.opts.resume if self.opts.resume is not None else False,
            **_identity(self.cfg, self.opts, prefix="ade20k"),
            **{**_common(self.opts),
               # specialize's segmentation overlay cadence (cfg.viz_every, default 500).
               # Figures go to run_dir/visualization/, which every run now has.
               "viz_every": self.opts.viz_every or self.cfg.viz_every},
        )
        return Ade20kRunTask(self.cfg, rl=self.rl), settings

    def lr_wd(self) -> tuple[float, float]:
        return self.cfg.peak_lr, self.cfg.weight_decay


@dataclass
class In1kCmd:
    """ImageNet-1k classification: frozen linear probe, finetune, or joint clf+policy."""

    cfg: In1kConfig = field(default_factory=In1kConfig)
    preset: PresetName = "default"
    spec: SpecOverrides = field(default_factory=SpecOverrides)
    rl: JointPolicyConfig = field(default_factory=JointPolicyConfig)
    opts: HarnessOpts = field(default_factory=HarnessOpts)

    def build(self) -> tuple[Any, RunSettings]:
        from canvit.in1k.task import In1kRunTask

        # Per-job step budget = the shard-schedule window (cfg.steps_per_job; None => one
        # job of max_steps). The LR-cosine horizon is always the FULL run (cfg.max_steps),
        # so a multi-job array anneals across jobs. --opts.n-steps overrides the budget for
        # smoke runs only (it decouples from the shard window — don't resume such a run).
        eff_steps_per_job = self.cfg.steps_per_job if self.cfg.steps_per_job is not None else self.cfg.max_steps
        n_steps = self.opts.n_steps if self.opts.n_steps is not None else eff_steps_per_job
        settings = RunSettings(
            n_steps=n_steps,
            eval_every=self.opts.eval_every if self.opts.eval_every is not None else self.cfg.val_every,
            log_every=self.cfg.log_every, grad_clip=self.cfg.grad_clip, amp=self.cfg.amp,
            seed=self.opts.seed if self.opts.seed is not None else self.cfg.seed,
            device=self.cfg.device,
            resume=self.opts.resume if self.opts.resume is not None else False,
            **_identity(self.cfg, self.opts, prefix="in1k"),
            **_common(self.opts),
        )
        return In1kRunTask(self.cfg, rl=self.rl, total_steps=self.cfg.max_steps), settings

    def lr_wd(self) -> tuple[float, float]:
        return self.cfg.peak_lr, self.cfg.weight_decay


def _policy_warmup_steps(task: Any, pol: Any) -> int:
    """The scorer's LR ramp, in steps, on ANY task.

    ``policy_warmup_steps`` wins when set. Otherwise ``policy_warmup_frac`` is resolved
    against the task's run length — which only ade20k/in1k have at config time
    (``max_steps``). Distill is SLURM-array-shaped (``steps_per_job``) with no total, so
    the fraction cannot be resolved there and this WARNS instead of returning 0 quietly:
    a silently-absent ramp is exactly the deviation from the RL recipe that doc 15 §A
    gap #2 was about, and it would be invisible in a run's logs."""
    if pol.policy_warmup_steps > 0:
        return pol.policy_warmup_steps
    if pol.policy_warmup_frac <= 0:
        return 0
    total = getattr(task.cfg, "max_steps", None) or getattr(task.cfg, "cosine_total_steps", None) or 0
    if total <= 0:
        log.warning(
            "policy_warmup_frac=%.3f cannot be resolved on task %r: it has no config-time "
            "run length (no max_steps / cosine_total_steps). The scorer will train with NO "
            "LR ramp. Set --rl.policy-warmup-steps to get one.",
            pol.policy_warmup_frac, getattr(task, "name", "?"))
        return 0
    return int(pol.policy_warmup_frac * total)


def resolve_spec(task: Any, preset: str, lr: float, wd: float,
                 overrides: SpecOverrides | None = None) -> TrainSpec:
    """Pick the spec from ``preset`` (``default`` = the task's own ``default_spec``), apply
    any ``overrides`` on top, and give every trainable module an optimizer group (the
    presets ship an empty ``optim``, filled here from the task config's peak_lr /
    weight_decay).

    With no overrides this returns exactly what it returned before ``SpecOverrides``
    existed — including the ``default`` fast path — because the pinning digests and the
    phase-2 numeric gate are pinned to those specs."""
    from dataclasses import replace

    over = overrides.as_dict() if overrides is not None else {}
    horizon = getattr(task.cfg, "n_timesteps", 10)
    if preset == "default":
        spec = task.default_spec()
        if not over:
            return spec  # untouched fast path: byte-identical to the pre-override behaviour
        return _apply_overrides(task, spec, over, lr=lr, wd=wd, horizon=horizon)
    # Same rule the tasks' own default_spec uses, so `--preset finetune` and
    # `--cfg.mode finetune` agree on the regime instead of quietly differing.
    chunk = getattr(task.cfg, "bptt_chunk_size", 0)
    bptt_none = fixed_horizon_bptt(frozen=True, horizon=horizon)
    bptt_full = fixed_horizon_bptt(frozen=False, horizon=horizon, chunk_size=chunk)
    if preset == "probe":
        spec = TrainSpec.probe(bptt=bptt_none)
    elif preset == "finetune":
        spec = TrainSpec.finetune(bptt=bptt_full)
    elif preset == "policy_only":
        spec = TrainSpec.policy_only(bptt=bptt_none)
    elif preset == "joint":
        spec = TrainSpec.joint()
    else:
        raise ValueError(f"unknown preset {preset!r}")
    # Headless tasks (distill: heads live inside the forward, caps.has_head=False) can't
    # train a separate head — drop train_head so the head-bearing presets still apply
    # (distill 'finetune' => task-only backbone; distill 'joint' => backbone + policy).
    if not task.caps().has_head and spec.train_head:
        spec = replace(spec, train_head=False)
    return _apply_overrides(task, spec, over, lr=lr, wd=wd, horizon=horizon)


def _apply_overrides(task: Any, spec: TrainSpec, over: dict, *, lr: float, wd: float,
                     horizon: int) -> TrainSpec:
    """Apply ``over`` to ``spec``, re-derive ``bptt`` if needed, then fill optim groups.

    Order matters: overrides must land BEFORE the optim-group fill, or a module the
    override newly makes trainable arrives with no group and `check_spec` rejects the run
    with "optim[backbone] missing" instead of inheriting the task's tuned lr/wd/schedule."""
    from dataclasses import replace

    was_train_backbone = spec.train_backbone
    if over:
        spec = replace(spec, **over)
        # BPTT is derived from train_backbone, so a preset's value goes stale the moment an
        # override flips it: `--preset probe --spec.train-backbone True` would otherwise
        # train the backbone with bptt='none', i.e. no cross-timestep graph at all — a much
        # weaker regime than `--preset finetune`, silently. Recompute only when the value
        # actually CHANGED: distill's default bptt is stochastic ('chunked' with
        # continue_prob), which fixed_horizon_bptt cannot express, so re-deriving it on a
        # no-op override would quietly replace distill's training regime.
        if spec.train_backbone != was_train_backbone:
            spec = replace(spec, bptt=fixed_horizon_bptt(
                frozen=not spec.train_backbone, horizon=horizon,
                chunk_size=getattr(task.cfg, "bptt_chunk_size", 0)))
    # ade20k/in1k carry the policy config on the task (passed in); distill keeps it
    # inside its own config as `cfg.rl`.
    pol = getattr(task, "rl", None) or getattr(task.cfg, "rl", None) or JointPolicyConfig()
    # Inherit the task's OWN optimizer groups / LR schedule. Without this a non-default
    # preset silently fell back to a bare GroupOptim, i.e. ScheduleSpec()'s default
    # `warmup_constant, warmup_steps=0` — so `ade20k --preset finetune` threw away the
    # warmup_onecycle recipe and `in1k --preset finetune` its warmup_cosine, running a
    # flat LR with no warmup and no anneal. The preset is meant to say WHAT TRAINS, not
    # to reset how it is scheduled.
    base_optim = task.default_spec().optim
    task_sched = next((o.schedule for g, o in sorted(base_optim.items()) if g != "policy"), None)
    optim = dict(spec.optim)
    for m in spec.trainable_modules():
        if m in optim:
            continue
        if m == "policy":
            # The scorer carries its OWN optimizer recipe on every task — betas and the
            # ramp-then-hold schedule come from JointPolicyConfig, not from the task's
            # probe/finetune settings. Falling through to a bare GroupOptim gave the
            # scorer torch's betas and NO warmup, both deviations from the RL recipe.
            optim[m] = GroupOptim(
                lr=pol.policy_lr, weight_decay=pol.policy_weight_decay, betas=pol.policy_betas,
                schedule=ScheduleSpec(kind="warmup_constant",
                                      warmup_steps=_policy_warmup_steps(task, pol)),
            )
        elif (base := base_optim.get(m)) is not None:
            optim[m] = base  # the task already tuned this group (lr, wd and schedule)
        else:
            # A group the task's default spec never defines (e.g. `backbone` for a
            # probe-default task): task lr/wd, but keep the task's schedule shape.
            optim[m] = GroupOptim(lr=lr, weight_decay=wd,
                                  schedule=task_sched if task_sched is not None else ScheduleSpec())
    return replace(spec, optim=optim)


@dataclass
class SearchCmd:
    """Class-cued visual search (COCO-Search18) on a frozen foveated CanViT: a
    target-conditioned task head and policy, trained jointly by default."""

    cfg: SearchConfig = field(default_factory=SearchConfig)
    preset: PresetName = "default"
    spec: SpecOverrides = field(default_factory=SpecOverrides)
    rl: JointPolicyConfig = field(default_factory=search_policy_config)
    """The search policy defaults (`search_policy_config`), NOT the generic JointPolicyConfig ones."""
    opts: HarnessOpts = field(default_factory=HarnessOpts)

    def build(self) -> tuple[Any, RunSettings]:
        from canvit.search.task import SearchRunTask

        settings = RunSettings(
            n_steps=self.opts.n_steps if self.opts.n_steps is not None else self.cfg.max_steps,
            eval_every=self.opts.eval_every if self.opts.eval_every is not None else self.cfg.val_every,
            log_every=self.cfg.log_every, grad_clip=self.cfg.grad_clip, amp=self.cfg.amp,
            seed=self.opts.seed if self.opts.seed is not None else self.cfg.seed,
            device=self.cfg.device,
            resume=self.opts.resume if self.opts.resume is not None else False,
            **_identity(self.cfg, self.opts, prefix="search"),
            **{**_common(self.opts), "viz_every": self.opts.viz_every or self.cfg.viz_every},
        )
        return SearchRunTask(self.cfg, rl=self.rl), settings

    def lr_wd(self) -> tuple[float, float]:
        return self.cfg.peak_lr, self.cfg.weight_decay


Command = (
    Annotated[DistillCmd, tyro.conf.subcommand(name="distill")]
    | Annotated[Ade20kCmd, tyro.conf.subcommand(name="ade20k")]
    | Annotated[In1kCmd, tyro.conf.subcommand(name="in1k")]
    | Annotated[SearchCmd, tyro.conf.subcommand(name="search")]
)


def main(argv: list[str] | None = None) -> dict:
    import torch

    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(name)s %(message)s")
    # TF32 matmuls for the fp32 paths — train/__main__.py sets this before building
    # anything, and it changes both speed and numerics.
    torch.set_float32_matmul_precision("high")

    cmd: Any = tyro.cli(Command, args=argv)
    task, settings = cmd.build()
    spec = resolve_spec(task, cmd.preset, *cmd.lr_wd(), getattr(cmd, "spec", None))
    log.info("task=%s preset=%s n_steps=%d eval_every=%d run_dir=%s",
             task.name, cmd.preset, settings.n_steps, settings.eval_every, settings.run_dir)
    return run(task=task, spec=spec, settings=settings)


__all__ = ["Ade20kCmd", "DistillCmd", "HarnessOpts", "In1kCmd", "SearchCmd", "main", "resolve_spec"]
