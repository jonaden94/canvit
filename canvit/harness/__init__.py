"""Task-neutral training harness: the orchestrator AND everything shared between tasks.

Design: ``canvit_claude/canvit/unification/07-unified-harness-design.md``. The three tasks
(``distill`` / ``ade20k`` / ``in1k``) are equal peers that plug in via the task seam;
nothing in this package knows about DINOv3, segmentation, or classification.

Flat at this level are the entry points and the shared vocabulary — the five files to read
first, and the ones nearly everything imports:

- ``run.py``    — the process entry point (``python -m canvit.harness.run``)
- ``cli.py``    — tyro CLI, and ``resolve_spec``: ``--preset`` × task → ``TrainSpec``
- ``loop.py``   — the training loop: step cadence, validation, checkpointing, logging
- ``spec.py``   — ``TrainSpec`` / ``BpttSpec`` / ``GroupOptim`` + ``check_spec`` validation
- ``config.py`` — the two config knobs every task shares (``FoveatedScaleConfig``,
  ``JointPolicyConfig``); each task's own config lives in its own folder

Grouped into subpackages is the machinery:

- ``rollout/`` — how a batch becomes a sequence of glimpses: the engine, viewpoint
  geometry, selectors, eval-time viewpoint schedules
- ``policy/``  — the learned viewpoint-selection policy: builder, ``JointPolicy``, and the
  RL objectives (QReg / PG / VPG)
- ``optim/``   — optimizer construction, LR schedules, EMA
- ``infra/``   — plumbing with no training logic: checkpoint I/O, tracker, DDP, SLURM
  rendezvous, the resumable shard schedule, small utils
- ``viz/``     — the task-agnostic PCA / figure-I/O / metric leaves

``rollout``, ``policy`` and ``optim`` re-export their main module's API, so
``from canvit.harness.rollout import run_rollout`` (and the ``optim`` / ``policy``
equivalents) work exactly as they did when each was a single flat module.

The rule: **shared lives here, task-specific lives in that task's folder.** Until
2026-07-31 the shared primitives sat in a folder called ``train/`` alongside distill's
own code, because distill was once the whole repo — see
``canvit_claude/canvit/unification/18-package-restructure.md``.
"""

from canvit.harness.spec import (
    BpttSpec,
    GroupOptim,
    ScheduleSpec,
    SpecReport,
    TaskCaps,
    TrainSpec,
    check_spec,
)

__all__ = [
    "BpttSpec",
    "GroupOptim",
    "ScheduleSpec",
    "SpecReport",
    "TaskCaps",
    "TrainSpec",
    "check_spec",
]
