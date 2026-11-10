"""IN1k classification task — engine-facing core (design §3.1, §0 table).

Readout = per-glimpse recurrent CLS token [B,D]; the ``LN + Linear`` head is applied
OUTSIDE the backbone forward (frozen backbone → head still trains). Glimpse routing
matches ADE20K (uniform pre-crop / foveated full-image), so the ade20k helpers are
reused. The run-level ``Task`` wrapper (build_model via ``from_pretrained_with_new_head``,
webdataset loaders with ``with_epoch``, top-1/5 eval) lands with the neutral loop.
"""

from __future__ import annotations

from contextlib import nullcontext
from typing import Any

import torch
import torch.nn.functional as F
from torch import Tensor

from canvit.core import CanViTForImageClassification, RecurrentState, sample_at_viewpoint
from canvit.core.policy.features import INTRINSIC_GROUPS
from canvit.harness.rollout import GlimpseOut, TaskLoss
from canvit.harness.rollout.episode import consumes_full_image, derive_glimpse_px
from canvit.harness.rollout.viewpoint import ViewpointType

# in1k's classifier reads the CLS token, not a spatial probe, so the probe-entropy
# groups don't apply — the scorer uses the probe-free INTRINSIC groups (like distill).
POLICY_FEATURE_GROUPS: tuple[str, ...] = INTRINSIC_GROUPS


class BoundIn1kTask:
    """Per-batch IN1k :class:`RolloutTask`. Binds this batch's class labels; holds the
    classifier wrapper (``.canvit`` + ``.norm`` + ``.head``) and the glimpse routing."""

    def __init__(
        self, *, clf: CanViTForImageClassification, targets: Tensor, canvas_grid: int,
        glimpse_px: int | None = None, label_smoothing: float = 0.0,
    ):
        self.clf = clf
        self.targets = targets  # [B] class idx
        self.canvas_grid = canvas_grid
        self.label_smoothing = label_smoothing
        self.full_image = consumes_full_image(clf)
        self.glimpse_px = None if self.full_image else derive_glimpse_px(clf, glimpse_px)

    def forward_glimpse(
        self, *, model: Any, images: Tensor, state: RecurrentState,
        viewpoint: Any, backbone_no_grad: bool,
    ) -> GlimpseOut:
        clf = getattr(model, "module", model)  # unwrap DDP; in1k steps .canvit directly
        model_input = images if self.full_image else sample_at_viewpoint(
            spatial=images, viewpoint=viewpoint, glimpse_size_px=self.glimpse_px,
        )
        ctx = torch.no_grad() if backbone_no_grad else nullcontext()
        with ctx:
            out = clf.canvit(image=model_input, state=state, viewpoint=viewpoint)
        cls = out.state.recurrent_cls[:, 0].float()  # [B, D]
        return GlimpseOut(readout=cls, state=out.state, vpe=out.vpe)

    def _logits(self, readout: Tensor) -> Tensor:
        return self.clf.head(self.clf.norm(readout))  # [B, C]

    def step_loss(self, readout: Any) -> TaskLoss:
        logits = self._logits(readout)
        return TaskLoss(combined=F.cross_entropy(logits, self.targets, label_smoothing=self.label_smoothing))

    def per_image_loss(self, readout: Any) -> Tensor:
        return F.cross_entropy(self._logits(readout), self.targets, reduction="none")  # [B]


class In1kRunTask:
    """Run-level IN1k :class:`~canvit.harness.run.RunTask`. The trainable
    "head" is LN(``clf.norm``) + Linear(``clf.head``): ``from_pretrained_with_new_head``
    leaves ``clf.norm`` at requires_grad=True and the harness freezes only the trunk, so
    norm stays trainable in both frozen and finetune; the optimizer's "head" group is
    norm+head. Config composed (design D-B): task holds its ``In1kConfig``; joint policy
    config passed in (``rl``)."""

    name = "in1k"
    best_metric = "top1"
    """Eval key the harness maximizes for `best.pt` — matches `in1k/train.py`'s
    `best_top1` selection. NB the standalone also writes a `best-hf/` export alongside;
    here that stays a separate step via `python -m canvit.checkpoint.to_hf`."""

    def __init__(self, cfg, *, rl=None, total_steps=None):
        self.cfg = cfg
        self.rl = rl
        # LR-schedule (cosine) horizon = total training steps across ALL array jobs
        # (cfg.max_steps); passed down by the runner. None => warmup-only (no decay).
        self.total_steps = total_steps
        # Resumable shard schedule (mirrors DistillRunTask): the job this task starts at,
        # the saved schedule-state to validate against, and the loader/world_size once built.
        self._start_job_index = 0
        self._resume_saved: dict | None = None
        self._world_size: int | None = None
        self._train_loader = None

    @property
    def _steps_per_job(self) -> int:
        """Per-job step budget = the shard-schedule window. None => single job of max_steps."""
        return self.cfg.steps_per_job if self.cfg.steps_per_job is not None else self.cfg.max_steps

    def caps(self):
        from canvit.harness.spec import TaskCaps
        # DDP yes (the webdataset schedule shards by rank); wrapper-level compile no — like
        # ade20k this task steps .canvit/.head directly, so it would be a silent no-op.
        return TaskCaps(has_head=True, supports_policy=True, supports_compile=False)

    def default_spec(self):
        """cfg.mode drives the default: 'frozen' => probe (backbone frozen, bptt none);
        'finetune' => train backbone + head end to end (full-graph).

        The LR schedule reproduces ``in1k/train.py``'s AdamW + ``warmup_cosine_scheduler``:
        warmup from ``peak_lr * warmup_lr_ratio`` over ``cfg.warmup_steps`` then cosine to 0
        over ``total_steps`` (the run length; = ``cfg.max_steps`` for a full run) — the same
        step-based recipe the standalone now uses.
        """
        from canvit.harness.spec import GroupOptim, ScheduleSpec, TrainSpec, fixed_horizon_bptt
        T = self.cfg.n_timesteps
        if self.total_steps is None:
            sched = ScheduleSpec(kind="warmup_constant", warmup_steps=0)
        else:
            warmup = max(1, self.cfg.warmup_steps)
            sched = ScheduleSpec(kind="warmup_cosine", warmup_steps=warmup,
                                 total_steps=self.total_steps,
                                 start_lr=self.cfg.peak_lr * self.cfg.warmup_lr_ratio)
        head_go = GroupOptim(lr=self.cfg.peak_lr, weight_decay=self.cfg.weight_decay,
                             schedule=sched)
        finetune = self.cfg.mode == "finetune"
        bptt = fixed_horizon_bptt(frozen=not finetune, horizon=T,
                                  chunk_size=self.cfg.bptt_chunk_size)
        if finetune:
            return TrainSpec.finetune(bptt=bptt, optim={"backbone": head_go, "head": head_go})
        return TrainSpec.probe(bptt=bptt, optim={"head": head_go})

    def build_model(self, device, prior_model_config=None):
        # prior_model_config is unused: the backbone arch comes from the HF repo the head
        # was built on, so a resume rebuilds the same model from cfg.model_repo already.
        # mode-dependent head (fresh probe vs fused DINOv3 probe for finetune).
        from canvit.in1k.model import build_classifier
        clf = build_classifier(self.cfg, device)
        return clf, clf.head

    def canvas_grid(self, model):
        if self.cfg.canvas_grid is not None:
            return self.cfg.canvas_grid
        return self.cfg.scene_size // model.canvit.backbone.patch_size_px

    def is_foveated(self, model):
        return consumes_full_image(model)

    def branches(self):
        return [ViewpointType.FULL if self.cfg.train_start_full else ViewpointType.RANDOM]

    def build_val_loader(self):
        """The val loader alone (see the ade20k twin). ``None`` when ``cfg.val_dir`` is
        absent, matching what ``build_loaders`` hands the harness."""
        from canvit.in1k.data import make_val_loader
        return make_val_loader(self.cfg, world_size=1, rank=0) if self.cfg.val_dir.is_dir() else None

    def build_loaders(self, *, world_size, rank):
        from canvit.in1k.data import make_train_loader, make_val_loader
        loader, _ = make_train_loader(
            self.cfg, world_size=world_size, rank=rank,
            job_index=self._start_job_index, steps_per_job=self._steps_per_job,
        )
        self._train_loader, self._world_size = loader, world_size
        self._check_schedule_invariants(loader, world_size=world_size)  # fail before any work
        val = make_val_loader(self.cfg, world_size=world_size, rank=rank) if self.cfg.val_dir.is_dir() else None
        return loader, val

    def build_selector(self, *, device, canvas_grid, is_foveated):
        from canvit.harness.rollout.selector import RandomSelector
        return RandomSelector(is_foveated=is_foveated, foveated_scale=self.cfg.foveated_scale,
                              min_viewpoint_scale=self.cfg.min_vp_scale)

    def build_policy(self, model, *, device, canvas_grid, generator):
        from canvit.harness.config import JointPolicyConfig
        from canvit.harness.policy import build_policy
        rl = self.rl or JointPolicyConfig(feature_groups=POLICY_FEATURE_GROUPS)
        return build_policy(
            canvit=model.canvit, rl=rl, feature_groups=POLICY_FEATURE_GROUPS, device=device,
            canvas_grid=canvas_grid, min_viewpoint_scale=self.cfg.min_vp_scale,
            foveated_scale=self.cfg.foveated_scale, generator=generator, encode_model=model,
        )

    def policy_feature_groups(self):
        return POLICY_FEATURE_GROUPS

    def trainable_param_groups(self, *, model, head, joint, spec):
        groups: dict[str, list] = {}
        if spec.train_backbone:
            groups["backbone"] = list(model.canvit.parameters())
        if spec.train_head:  # in1k head = LN(norm) + Linear(head)
            groups["head"] = list(model.norm.parameters()) + list(model.head.parameters())
        if spec.train_policy:
            assert joint is not None
            params = list(joint.scorer.parameters())
            if joint.critic is not None:  # Q-Prop: critic rides the same param group/LR
                params += list(joint.critic.parameters())
            groups["policy"] = params
        return groups

    def resume_start_step(self, payload, scheduler):
        """Where this job starts. The SHARD SCHEDULE (not the scheduler) is authoritative:
        each job runs exactly steps_per_job shard-aligned steps, so the next job starts at
        ``(saved job_index + 1) * steps_per_job`` and reads the next shard slice. A
        checkpoint without a job_index, or a scheduler that disagrees with the derived step
        (a mid-job save, or a job that ran a different step count), is a hard error —
        resuming at any other offset silently re-processes or skips shards. Mirrors
        DistillRunTask; only reached on resume (fresh runs keep job_index=0)."""
        saved = (payload.get("metadata") or {}).get("resume_state") or {}
        if saved.get("job_index") is None:
            raise RuntimeError(
                "in1k shard-schedule resume requires a checkpoint carrying `job_index` "
                "(metadata.resume_state) — this one has none, so the next shard slice is "
                "unknown. Seed from it instead (RunSettings.seed_ckpt) to start fresh.")
        self._resume_saved = saved
        self._start_job_index = int(saved["job_index"]) + 1
        start_step = self._start_job_index * self._steps_per_job
        if scheduler.last_epoch != start_step:
            raise RuntimeError(
                f"in1k shard-schedule resume: scheduler.last_epoch={scheduler.last_epoch} "
                f"!= start_step={start_step} (= (saved job_index {saved['job_index']} + 1) "
                f"* steps_per_job {self._steps_per_job}). Checkpoint not written at an "
                "end-of-job boundary — resume from one, or seed to start a fresh schedule.")
        return start_step

    def _check_schedule_invariants(self, train_loader, *, world_size) -> None:
        """Refuse to resume when the shard-schedule inputs changed — the slice offset is
        only meaningful under the (world_size, batch, steps_per_job, samples_per_shard) it
        was computed with; changing any silently re-processes or skips shards. Mirrors
        DistillRunTask. No-op on a fresh run (``_resume_saved is None``)."""
        if self._resume_saved is None:
            return
        current = {
            "ddp_world_size": world_size,
            "batch_size": self.cfg.batch_size,
            "steps_per_job": self._steps_per_job,
            "samples_per_shard": train_loader.samples_per_shard,
        }
        saved = {k: self._resume_saved.get(k) for k in current}
        if saved != current:
            raise RuntimeError(
                "in1k shard-schedule resume config mismatch — refusing to resume (the "
                f"schedule offset would be wrong). Saved: {saved}. Current: {current}. To "
                "train with different values, seed from this checkpoint instead.")

    def resume_state(self):
        """Shard-schedule state stored in the checkpoint so the NEXT array job resumes
        shard-aligned (consumed by resume_start_step / _check_schedule_invariants). Empty
        before the loaders exist."""
        if self._train_loader is None:
            return {}
        return {
            "job_index": self._start_job_index,
            "ddp_world_size": self._world_size,
            "batch_size": self.cfg.batch_size,
            "steps_per_job": self._steps_per_job,
            "samples_per_shard": self._train_loader.samples_per_shard,
        }

    def batch_images(self, batch, device):
        return batch[0].to(device, non_blocking=True)

    def bind(self, batch, device, *, model, head):
        _, labels = batch
        return BoundIn1kTask(
            clf=model, targets=torch.as_tensor(labels, dtype=torch.long, device=device),
            canvas_grid=self.canvas_grid(model), glimpse_px=self.cfg.glimpse_px,
            label_smoothing=self.cfg.label_smoothing,
        )

    @torch.no_grad()
    def evaluate(self, *, model, head, val_loader, device, step, tracker=None, run_dir=None,
                 joint=None):
        # tracker/run_dir unused: this task returns its scalars for the caller to log
        # and renders no validation figures (owner: distill viz only).
        """Top-1/5 over the eval policy (reuses in1k/eval.py::evaluate)."""
        if val_loader is None:
            return {}
        from canvit.harness.rollout.eval_viewpoints import resolve
        from canvit.in1k.eval import evaluate as _eval
        amp = torch.autocast("cuda", dtype=torch.bfloat16) if device.type == "cuda" else nullcontext()
        is_fov = consumes_full_image(model)
        eval_policy = resolve(self.cfg.eval_policy, task="in1k", is_foveated=is_fov)
        accs = _eval(model, self.cfg, val_loader, device=device, canvas_grid=self.canvas_grid(model),
                     amp_ctx=amp, is_foveated=is_fov, eval_policy=eval_policy, joint=joint)
        return {"top1": accs[1], "top5": accs[5]}

    def model_config(self, model):
        from dataclasses import asdict

        from canvit.in1k.config import NUM_CLASSES

        # Record the architecture, not just the `model_repo` pointer — a FINETUNE changes
        # the backbone, so this checkpoint is the only place its weights exist and the
        # pointer describes the model it started from. These are exactly the constructor
        # arguments CanViTForImageClassification.from_checkpoint needs, read off the model
        # so they cannot disagree with the weights.
        core = getattr(model, "module", model)
        return {"task": "in1k", "n_classes": NUM_CLASSES, "canvas_grid": self.canvas_grid(model),
                "model_repo": self.cfg.model_repo, "mode": self.cfg.mode,
                "backbone_name": core.backbone_name, "canvit": asdict(core.canvit.cfg),
                "glimpse_grid_size": core.glimpse_grid_size}

    def checkpoint_metadata(self, model):
        from canvit.checkpoint import downstream_pretrain_view_scale
        return {"task": "in1k", "mode": self.cfg.mode, "scene_size": self.cfg.scene_size,
                "n_timesteps": self.cfg.n_timesteps,
                "model_repo": str(self.cfg.model_repo),
                "pretrain_view_scale": downstream_pretrain_view_scale(
                    patcher_name=getattr(model.canvit.cfg, "patcher_name", None),
                    foveated_scale=self.cfg.foveated_scale)}


__all__ = ["POLICY_FEATURE_GROUPS", "BoundIn1kTask", "In1kRunTask"]
