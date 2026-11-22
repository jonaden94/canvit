"""Orthogonal training spec for the unified harness (design doc 07 §3.3/§4/§8).

``TrainSpec`` is the single, task-agnostic description of *what trains under which
loss* for one run. The same spec space applies identically to all three tasks
(distill / ade20k / in1k); a task only advertises its capabilities (``TaskCaps``:
does it have a head? can it drive a policy?) so the validator can reject
incoherent specs. Pure config + validation — no torch, no model — so it is
trivially CPU-unit-testable.

The orthogonal knobs (design §3.3) generate the full config cross product:

    train_backbone / train_head / train_policy   which params get gradients
    task_weight    / policy_weight               which losses are active
    task_grad_to_backbone                        task loss reaches trunk?  (probe vs finetune)
    policy_grad_to_backbone                       policy loss reaches trunk? (= not feats_detached)
    bptt                                          rollout gradient regime (none/full/chunked)

``feats_detached`` in the policy selector is exactly ``not policy_grad_to_backbone``.
``mode ∈ {frozen, finetune}`` (old IN1k) is ``train_backbone`` + ``task_grad_to_backbone``.
What P4b called ``use_rl`` is ``train_policy`` + ``policy_weight > 0`` (the removed
``JointPolicyConfig.use_rl`` field was never read; the spec decides).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

Module = Literal["backbone", "head", "policy"]
MODULES: tuple[Module, ...] = ("backbone", "head", "policy")


@dataclass(frozen=True)
class ScheduleSpec:
    """LR schedule for one optimizer param group (design D-E).

    ``warmup_constant``: linear warmup then hold (distill default).
    ``warmup_cosine``:   linear warmup then cosine decay to 0 (needs ``total_steps``).
    ``warmup_onecycle``: warmup + one-cycle anneal (needs ``total_steps``; the ADE20K
                         probe recipe).
    """

    kind: Literal["warmup_constant", "warmup_cosine", "warmup_onecycle"] = "warmup_constant"
    warmup_steps: int = 0
    total_steps: int | None = None
    start_lr: float | None = None
    warmup_lr_ratio: float | None = None  # onecycle: start_lr = peak_lr * ratio

    def errors(self, *, group: str) -> list[str]:
        errs: list[str] = []
        if self.warmup_steps < 0:
            errs.append(f"optim[{group}].schedule.warmup_steps must be >= 0")
        if self.kind in ("warmup_cosine", "warmup_onecycle") and self.total_steps is None:
            errs.append(f"optim[{group}].schedule.kind={self.kind!r} requires total_steps")
        if self.total_steps is not None and self.total_steps <= 0:
            errs.append(f"optim[{group}].schedule.total_steps must be > 0")
        # Reject warmup >= total for the decaying schedules: it silently pins the LR in
        # warmup for the whole run (LR never anneals → ~no learning). The standalone
        # warmup_cosine_scheduler asserts the same; without this the harness would run
        # the degenerate schedule quietly (found via the in1k gate, doc 11 §4).
        if (self.kind in ("warmup_cosine", "warmup_onecycle") and self.total_steps is not None
                and self.warmup_steps >= self.total_steps):
            errs.append(f"optim[{group}].schedule.warmup_steps ({self.warmup_steps}) must be "
                        f"< total_steps ({self.total_steps}) for kind={self.kind!r}")
        return errs


@dataclass(frozen=True)
class GroupOptim:
    """Optimizer settings for one trainable module group (design §7 D-E)."""

    lr: float
    weight_decay: float = 0.0
    betas: tuple[float, float] = (0.9, 0.999)
    """AdamW betas for THIS group. Per-group because the recipes genuinely differ: the
    RL viewpoint-scorer wants (0.9, 0.95) while the task recipes use torch's default.
    Before this existed the harness silently gave every group (0.9, 0.999), which is a
    real deviation from the CanViT-PyTorch-RL recipe (see doc 15 §A)."""
    schedule: ScheduleSpec = field(default_factory=ScheduleSpec)

    def errors(self, *, group: str) -> list[str]:
        errs = [] if self.lr > 0 else [f"optim[{group}].lr must be > 0"]
        if self.weight_decay < 0:
            errs.append(f"optim[{group}].weight_decay must be >= 0")
        if not all(0.0 <= b < 1.0 for b in self.betas):
            errs.append(f"optim[{group}].betas must each be in [0, 1), got {self.betas}")
        return errs + self.schedule.errors(group=group)


@dataclass(frozen=True)
class BpttSpec:
    """Rollout gradient regime over the glimpse sequence (design §6).

    One mechanism, three configs:

    * ``none``    — backbone forward under ``no_grad``; only head/policy carry graph
                    (probe / frozen-model-policy). ``horizon`` fixed.
    * ``full``    — one graph over the whole rollout; single backward at the end
                    (= ``chunked`` with ``chunk_size == horizon``). ``horizon`` fixed.
    * ``chunked`` — backward + detach at each chunk boundary. Length is either a
                    fixed ``horizon`` or a stochastic ``continue_prob`` extension
                    (distill's TBPTT: ``n = chunk_size``; ``while rand()<p: n += chunk_size``).

    Exactly one of ``horizon`` / ``continue_prob`` defines the length. ``continue_prob``
    is only meaningful with ``mode='chunked'``.

    **RULE — a FROZEN backbone ALWAYS takes ``mode='none'``.** Do not pair
    ``train_backbone=False`` with ``'full'`` or ``'chunked'``: ``bptt`` moves the BACKBONE
    and nothing else. The head reads the canvas state at t and never feeds back into it,
    so no head parameter influences a later timestep and there is no cross-timestep path
    to propagate. Measured 2026-07-28 (``tests/test_bptt_chunking.py``): head gradients
    are BIT-IDENTICAL between ``'none'`` and ``'full'`` whether the backbone is frozen or
    trainable. Keeping a graph over a frozen backbone changes no number and holds
    activations for the entire rollout. ``check_spec`` warns if you do it; the config path
    (``fixed_horizon_bptt``, used by every task's ``default_spec`` AND by ``resolve_spec``)
    enforces it for you, so this can only be reached by hand-building a ``TrainSpec``.

    Chunk length is otherwise free: ``horizon`` need NOT be divisible by ``chunk_size``.
    ``run_rollout`` flushes the trailing partial chunk and every chunk normalises by
    ``n_glimpses``, so 7 glimpses at chunk 3 runs ``[0,1,2][3,4,5][6]`` and accumulates the
    same total gradient as one graph. Prime horizons are fine.
    """

    mode: Literal["none", "full", "chunked"] = "chunked"
    chunk_size: int = 1
    horizon: int | None = None
    continue_prob: float | None = None

    def errors(self) -> list[str]:
        errs: list[str] = []
        if self.chunk_size < 1:
            errs.append("bptt.chunk_size must be >= 1")
        has_horizon = self.horizon is not None
        has_cprob = self.continue_prob is not None
        if has_horizon == has_cprob:
            errs.append("bptt: set exactly one of horizon (fixed length) or continue_prob (stochastic)")
        if has_horizon and self.horizon <= 0:  # type: ignore[operator]
            errs.append("bptt.horizon must be > 0")
        if has_cprob and not (0.0 <= self.continue_prob <= 1.0):  # type: ignore[operator]
            errs.append("bptt.continue_prob must be in [0, 1]")
        if has_cprob and self.mode != "chunked":
            errs.append(f"bptt.continue_prob (stochastic length) requires mode='chunked', not {self.mode!r}")
        return errs

    @property
    def stochastic(self) -> bool:
        return self.continue_prob is not None


def fixed_horizon_bptt(*, frozen: bool, horizon: int, chunk_size: int = 0) -> BpttSpec:
    """The BPTT regime for a fixed-length task (ade20k / in1k), from cfg.

    Shared so the two downstream tasks cannot drift apart — their `default_spec`s are
    deliberately structured the same way.

    * ``frozen`` (probe mode) -> ``none``, ALWAYS, ignoring ``chunk_size``. With the
      backbone under ``no_grad`` there is nothing for a graph to accumulate into: the
      head reads the canvas state at t and never feeds back into it, so no head
      parameter influences a later timestep. Measured 2026-07-28: head gradients are
      BIT-IDENTICAL between ``none`` and ``full`` whether the backbone is frozen or
      trainable — ``bptt`` moves the backbone only. Chunking a frozen backbone would
      cost memory and change nothing.
    * ``chunk_size <= 0`` (default) -> ``full``: one graph over the whole rollout.
    * ``chunk_size >= horizon`` -> ``full`` too; chunking at or beyond the horizon is
      the same computation, so collapse it rather than pretend otherwise.
    * otherwise -> ``chunked``: backward + detach every ``chunk_size`` glimpses.
      ``horizon`` need NOT be divisible by ``chunk_size`` — ``run_rollout`` flushes the
      trailing partial chunk, and every chunk normalises by ``n_glimpses``, so a prime
      horizon just ends with a short chunk (7 @ 3 -> [0,1,2][3,4,5][6]).
    """
    assert horizon > 0, f"horizon must be > 0, got {horizon}"
    if frozen:
        return BpttSpec(mode="none", horizon=horizon)
    if chunk_size <= 0 or chunk_size >= horizon:
        return BpttSpec(mode="full", horizon=horizon)
    return BpttSpec(mode="chunked", chunk_size=chunk_size, horizon=horizon)


@dataclass(frozen=True)
class TaskCaps:
    """What a task supports — supplied by the task so the validator can reject
    specs the task cannot honor."""

    has_head: bool
    supports_policy: bool
    supports_ddp: bool = True
    """False => this task refuses to run with world_size > 1 (checked by
    :func:`check_spec`, i.e. before the model is built). ade20k sets it: its loader is a
    plain map-style ``DataLoader(shuffle=True)`` with no ``DistributedSampler``, so ranks
    would draw OVERLAPPING samples instead of disjoint shards — no error, just a run whose
    effective batch is not what the config says."""
    supports_compile: bool = False
    """Whether ``RunSettings.compile`` actually compiles anything. ``run()`` compiles the
    model the harness holds, i.e. the wrapper's ``forward``; only distill's task calls that
    (``model(image=…)``). ade20k/in1k step ``model.canvit(...)`` / ``model.head(...)``
    directly, so wrapper-level compilation would be a SILENT no-op — compiling those means
    compiling ``.canvit`` explicitly. Default False so a new task has to opt in knowingly."""


@dataclass
class SpecReport:
    """Result of validating a ``TrainSpec`` against a task's capabilities."""

    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return not self.errors


@dataclass(frozen=True)
class TrainSpec:
    """What trains under which loss for one run (design §3.3). Task-agnostic;
    validate against a task's :class:`TaskCaps` with :func:`check_spec` (or
    :meth:`validate`)."""

    train_backbone: bool = False
    train_head: bool = False
    train_policy: bool = False
    task_weight: float = 1.0
    policy_weight: float = 0.0
    task_grad_to_backbone: bool = True
    policy_grad_to_backbone: bool = False
    bptt: BpttSpec = field(default_factory=lambda: BpttSpec(mode="chunked", chunk_size=1, horizon=1))
    optim: dict[str, GroupOptim] = field(default_factory=dict)

    # --- derived / adapters ------------------------------------------------
    @property
    def feats_detached(self) -> bool:
        """Selector flag: detach the canvas before the scorer iff the policy loss
        must NOT reach the backbone."""
        return not self.policy_grad_to_backbone

    @property
    def task_loss_active(self) -> bool:
        return self.task_weight > 0.0

    @property
    def policy_loss_active(self) -> bool:
        return self.policy_weight > 0.0

    def trainable_modules(self) -> tuple[Module, ...]:
        got = []
        if self.train_backbone:
            got.append("backbone")
        if self.train_head:
            got.append("head")
        if self.train_policy:
            got.append("policy")
        return tuple(got)  # type: ignore[return-value]

    def validate(self, caps: TaskCaps, *, is_dist: bool = False) -> None:
        """Raise ``ValueError`` listing all hard errors; warnings are returned by
        :func:`check_spec` and are the caller's to log."""
        report = check_spec(self, caps, is_dist=is_dist)
        if not report.ok:
            raise ValueError("invalid TrainSpec:\n  - " + "\n  - ".join(report.errors))

    # --- presets (document the mapping + used by per-task defaults) ---------
    @classmethod
    def probe(cls, *, bptt: BpttSpec | None = None, optim: dict[str, GroupOptim] | None = None) -> "TrainSpec":
        """Task-only, frozen backbone: train the head only (the ADE20K/IN1k probe)."""
        return cls(
            train_backbone=False, train_head=True, task_grad_to_backbone=False,
            bptt=bptt or BpttSpec(mode="none", horizon=10), optim=optim or {},
        )

    @classmethod
    def finetune(cls, *, bptt: BpttSpec | None = None, optim: dict[str, GroupOptim] | None = None) -> "TrainSpec":
        """Task-only, full model: train backbone + head end to end."""
        return cls(
            train_backbone=True, train_head=True, task_grad_to_backbone=True,
            bptt=bptt or BpttSpec(mode="full", horizon=10), optim=optim or {},
        )

    @classmethod
    def policy_only(cls, *, freeze_model: bool = True, bptt: BpttSpec | None = None,
                    optim: dict[str, GroupOptim] | None = None) -> "TrainSpec":
        """Train only the policy. ``freeze_model=True`` (default) freezes backbone +
        head (the RL flagship: frozen model, train the scorer)."""
        return cls(
            train_backbone=not freeze_model, train_head=False, train_policy=True,
            task_weight=0.0, policy_weight=1.0,
            task_grad_to_backbone=False, policy_grad_to_backbone=not freeze_model,
            bptt=bptt or BpttSpec(mode="none", horizon=10), optim=optim or {},
        )

    @classmethod
    def joint(cls, *, train_backbone: bool = True, task_grad_to_backbone: bool = True,
              policy_grad_to_backbone: bool = False, bptt: BpttSpec | None = None,
              optim: dict[str, GroupOptim] | None = None) -> "TrainSpec":
        """Joint task + policy. Defaults reproduce P4b (full task, policy-net-only)."""
        return cls(
            train_backbone=train_backbone, train_head=True, train_policy=True,
            task_weight=1.0, policy_weight=1.0,
            task_grad_to_backbone=task_grad_to_backbone,
            policy_grad_to_backbone=policy_grad_to_backbone,
            bptt=bptt or BpttSpec(mode="chunked", chunk_size=2, continue_prob=0.5),
            optim=optim or {},
        )


def check_spec(spec: TrainSpec, caps: TaskCaps, *, is_dist: bool = False) -> SpecReport:
    """Validate a :class:`TrainSpec` against a task's capabilities (design §8).

    Every combination is *allowed*; this rejects only incoherent specs (routing a
    loss into a frozen module, a loss with no target, a policy the task can't drive,
    an unsupported DDP cell) and warns on vacuous-but-runnable ones (the owner's
    rule: give all options, trust the user, warn on the degenerate). Task-specific
    warnings (e.g. frozen-backbone distill) are added by the task, not here.
    """
    e: list[str] = []
    w: list[str] = []

    # --- hard errors (reject) ---
    if not (spec.train_backbone or spec.train_head or spec.train_policy):
        e.append("nothing is trainable (train_backbone/head/policy all False)")
    if spec.task_weight < 0 or spec.policy_weight < 0:
        e.append("task_weight and policy_weight must be >= 0")
    if not spec.task_loss_active and not spec.policy_loss_active:
        e.append("no loss is active (task_weight == 0 and policy_weight == 0)")

    # policy trainability <-> policy loss must agree
    if spec.policy_loss_active and not spec.train_policy:
        e.append("policy_weight > 0 but train_policy is False (policy loss trains nothing)")
    if spec.train_policy and not spec.policy_loss_active:
        e.append("train_policy is True but policy_weight == 0 (policy would never learn)")

    # capability checks
    if spec.train_head and not caps.has_head:
        e.append("train_head is True but this task has no head")
    if spec.train_policy and not caps.supports_policy:
        e.append("train_policy is True but this task does not support a policy")

    # loss->backbone routing into a non-trainable backbone
    if spec.task_grad_to_backbone and not spec.train_backbone:
        e.append("task_grad_to_backbone is True but train_backbone is False (routing task loss into a frozen backbone)")
    if spec.policy_grad_to_backbone and not spec.train_backbone:
        e.append("policy_grad_to_backbone is True but train_backbone is False (routing policy loss into a frozen backbone)")
    if spec.policy_grad_to_backbone and not spec.train_policy:
        e.append("policy_grad_to_backbone is True but train_policy is False (no policy loss to route)")

    # DDP support matrix (design §9): the coupled cell is unsupported under DDP, and a
    # task whose loader cannot shard by rank refuses multi-GPU outright.
    if is_dist and not caps.supports_ddp:
        e.append(
            "this task does not support DDP (world_size > 1): its data loader ignores "
            "world_size/rank, so every rank would draw overlapping samples from the whole "
            "dataset instead of a disjoint shard. Run it on ONE GPU (NGPU=1, "
            "--ntasks-per-node=1)."
        )
    # policy_grad_to_backbone=True IS supported under DDP (2026-11-22): nothing is wrapped in
    # DistributedDataParallel; the loop AllReduces the .grad of every trainable model param by
    # hand after the rollout (loop.py, ddp.allreduce_grads), whichever loss put it there, and
    # the scorer through joint.allreduce_grads. The guard that refused it here was written with
    # the harness (88957e6) and never tested; a 2-GPU run (canvit_claude/canvit/
    # active_pretraining/ddp_policy_val_probe.py) ended with bit-identical params on both ranks.

    # bptt / optim internal validity
    e.extend(spec.bptt.errors())
    for group in spec.trainable_modules():
        go = spec.optim.get(group)
        if go is None:
            e.append(f"optim[{group}] missing: a trainable module needs an optimizer group")
        else:
            e.extend(go.errors(group=group))
    for group in spec.optim:
        if group not in MODULES:
            e.append(f"optim has unknown group {group!r} (expected a subset of {MODULES})")
        elif group not in spec.trainable_modules():
            w.append(f"optim[{group}] given but {group} is not trainable (ignored)")

    # --- warnings (run anyway) ---
    if spec.task_loss_active and not (spec.train_backbone or spec.train_head):
        w.append("task_weight > 0 but neither backbone nor head is trainable — the task loss "
                 "trains nothing (it only serves as the policy reward signal)")
    if spec.train_head and not spec.task_loss_active:
        w.append("train_head is True but task_weight == 0 — the head receives no gradient")
    if spec.train_backbone and not (spec.task_grad_to_backbone or spec.policy_grad_to_backbone):
        w.append("train_backbone is True but no loss is routed to the backbone — it won't learn")
    if not spec.train_backbone and spec.bptt.mode != "none" and not spec.train_policy:
        # THE RULE: frozen backbone => bptt MUST be 'none'. Warned rather than errored
        # only because it is wasteful, not wrong. See BpttSpec / fixed_horizon_bptt.
        w.append(
            f"bptt.mode={spec.bptt.mode!r} with a FROZEN backbone (train_backbone=False) — "
            "use mode='none'. bptt only ever moves the backbone: the head reads the canvas "
            "state at t and never feeds back into it, so no head parameter influences a "
            "later timestep. Measured 2026-07-28: head gradients are BIT-IDENTICAL between "
            "'none' and 'full'. Keeping a graph here buys nothing and costs activation "
            "memory for the whole rollout. (Not warned for train_policy runs — the "
            "policy's cross-timestep path has not been measured.)"
        )
    if spec.train_backbone and spec.bptt.mode == "none":
        # The MIRROR of the frozen-backbone warning above, and the one an override can
        # reach: bptt is derived from train_backbone by the presets, so flipping
        # train_backbone on via `--spec.train-backbone True` over a probe preset leaves
        # mode='none' behind. bptt moves the backbone and nothing else, so mode='none'
        # here means the backbone gets NO cross-timestep credit — a far weaker regime
        # than `--preset finetune`, and previously silent because the warning below only
        # fired the other way round. `_apply_overrides` re-derives bptt when the override
        # changes train_backbone, so reaching this warning means bptt was pinned
        # deliberately (or a task's own default_spec is inconsistent).
        w.append(
            "train_backbone is True but bptt.mode='none' — the backbone will get no "
            "cross-timestep gradient, only within-timestep credit. bptt moves the "
            "backbone only, so 'none' with a TRAINABLE backbone is a much weaker "
            "regime than 'full'/'chunked'. Use fixed_horizon_bptt(frozen=False, ...) "
            "unless you specifically want single-step credit."
        )
    if (spec.train_backbone and not spec.task_grad_to_backbone
            and spec.policy_grad_to_backbone and not spec.task_loss_active):
        w.append("backbone is trained solely by the policy loss (no task signal) — unusual but valid")

    return SpecReport(errors=e, warnings=w)
