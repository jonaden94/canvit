"""Distill task — the DINOv3 feature-regression peer (design §3.1, §0 table).

Engine-facing core only (the per-batch :class:`RolloutTask` the rollout engine
consumes). Distill's readout is special: its recon/CLS heads live INSIDE the
pretraining forward, so ``forward_glimpse`` IS the model forward and ``step_loss``/
``per_image_loss`` delegate straight to the historical ``DistillTask`` (train/task.py).
This is the exact adapter proven byte-for-byte against the parity digest
``9a0100a1a3de3acd`` in ``harness/tests/test_rollout_parity.py``.

The run-level ``Task`` wrapper (build_model/build_loaders/evaluate) is added with the
neutral loop (design §11); its data pipeline is distill's existing webdataset +
normalizer machinery, unchanged.
"""

from __future__ import annotations

import logging
from contextlib import nullcontext
from typing import Any

import torch
from torch import Tensor

from canvit.core import RecurrentState
from canvit.core.policy.features import INTRINSIC_GROUPS
from canvit.distill.loss import DistillTask
from canvit.harness.rollout import GlimpseOut
from canvit.harness.rollout.viewpoint import ViewpointType

log = logging.getLogger(__name__)

# distill scorer uses the probe-free INTRINSIC feature groups (no task head to read).
POLICY_FEATURE_GROUPS: tuple[str, ...] = INTRINSIC_GROUPS


class BoundDistillTask:
    """Per-batch distill :class:`RolloutTask`. ``distill`` binds this batch's
    (standardized) teacher targets + the active loss terms."""

    def __init__(self, distill: DistillTask, metric_refs: dict[str, Any] | None = None):
        self.distill = distill
        # LOGGING ONLY (`final_metrics`): the destandardizers + this batch's RAW teacher
        # targets, so the raw-space cosines match train/loop.py's (which compares against
        # the true raw targets, not destandardize(standardize(x))). The loss path never
        # touches these, so they stay optional for the parity / A-B harnesses that
        # construct this class directly.
        self.metric_refs = metric_refs

    def forward_glimpse(
        self, *, model: Any, images: Tensor, state: RecurrentState,
        viewpoint: Any, backbone_no_grad: bool,
    ) -> GlimpseOut:
        # The pretraining wrapper handles glimpse cropping internally (glimpse_size_px
        # baked in) and computes scene/cls preds inside its forward; call it directly
        # (through the DDP wrapper when present, so head grads are AllReduced).
        ctx = torch.no_grad() if backbone_no_grad else nullcontext()
        with ctx:
            out = model(image=images, state=state, viewpoint=viewpoint)
        return GlimpseOut(readout=out, state=out.state, vpe=out.vpe)

    def step_loss(self, readout: Any) -> Any:
        return self.distill.step_loss(readout)

    def per_image_loss(self, readout: Any) -> Tensor:
        return self.distill.per_image_loss(readout)

    # --- logging-only hooks (train/loop.py's BranchMetrics; no effect on the loss) ---
    def glimpse_metrics(self, loss: Any) -> dict[str, Tensor]:
        """The two sub-losses, summed per glimpse and averaged over the branch by the
        engine — exactly `BranchMetrics.scene_patches_loss` / `.scene_cls_loss`."""
        return {"scene_patches_loss": loss.scene_patches_loss.detach(),
                "scene_cls_loss": loss.scene_cls_loss.detach()}

    def final_metrics(self, readout: Any) -> dict[str, Tensor]:
        """Cosine similarity of the LAST glimpse's predictions against the teacher, in
        both normalized and raw (destandardized) space — `BranchMetrics.*_cos_*`.
        Empty without ``metric_refs`` (raw space is undefined then)."""
        if not self.metric_refs:
            return {}
        import torch.nn.functional as F
        r = self.metric_refs
        scene_pred, cls_pred = readout.scene_pred, readout.cls_pred
        with torch.no_grad():
            scene_raw = r["scene_denorm"](scene_pred)
            cls_raw = r["cls_denorm"](cls_pred.unsqueeze(1)).squeeze(1)
            return {
                "scene_cos_raw": F.cosine_similarity(scene_raw, r["raw_scene_target"], dim=-1).mean(),
                "scene_cos_norm": F.cosine_similarity(scene_pred, self.distill.scene_target, dim=-1).mean(),
                "cls_cos_raw": F.cosine_similarity(cls_raw, r["raw_cls_target"], dim=-1).mean(),
                "cls_cos_norm": F.cosine_similarity(cls_pred, self.distill.cls_target, dim=-1).mean(),
            }


class DistillRunTask:
    """Run-level distill :class:`~canvit.harness.run.RunTask`. Distill's heads
    ride INSIDE the pretraining forward, so ``head=None`` (``has_head=False``) and the
    backbone group carries them. Reuses distill's existing machinery unchanged: model
    (``create_model``), webdataset loaders + model-owned normalizers (``create_loaders``
    / ``init_normalizer_stats_from_tar``), and the ``validate()`` eval. The normalizer is
    model-state, so it is initialized once (in ``build_loaders``, which by then has both
    the model — stashed in ``build_model`` — and the first shard)."""

    name = "distill"

    def __init__(self, cfg):
        self.cfg = cfg
        self._model = None
        self._device = None
        self._glimpse_size_px = None
        self.scene_norm = None
        self.cls_norm = None
        self._teacher = None
        self._teacher_compiled = False
        self._probe = None
        # WebDataset shard schedule (see resume_start_step): index of the job this
        # process runs, and the saved schedule inputs it must still agree with.
        self._start_job_index = 0
        self._resume_saved: dict | None = None
        self._train_loader = None
        self._world_size = 1

    def caps(self):
        from canvit.harness.spec import TaskCaps
        # heads live in the forward; supports_compile=True because this task DOES call the
        # wrapper's forward (`model(image=…)`), which is what run() compiles.
        return TaskCaps(has_head=False, supports_policy=True, supports_compile=True)

    def default_spec(self):
        """The historical distill regime: train backbone (+ in-forward heads), stochastic
        chunked TBPTT. Byte-exact to train/step.py under the parity digest."""
        from canvit.harness.spec import BpttSpec, GroupOptim, ScheduleSpec, TrainSpec
        sched = ScheduleSpec(
            kind="warmup_cosine" if self.cfg.cosine_total_steps else "warmup_constant",
            warmup_steps=self.cfg.warmup_steps, total_steps=self.cfg.cosine_total_steps,
            start_lr=self.cfg.start_lr,
        )
        return TrainSpec(
            train_backbone=True, train_head=False, task_grad_to_backbone=True,
            bptt=BpttSpec(mode="chunked", chunk_size=self.cfg.chunk_size,
                          continue_prob=self.cfg.continue_prob),
            optim={"backbone": GroupOptim(lr=self.cfg.peak_lr, weight_decay=self.cfg.weight_decay,
                                          schedule=sched)},
        )

    def build_model(self, device, prior_model_config=None):
        from canvit.distill.model import create_model, load_student_backbone
        assert not (self.cfg.seed_ckpt and self.cfg.hf_seed_ckpt), (
            "seed_ckpt and hf_seed_ckpt are mutually exclusive"
        )
        # HF SEED: the checkpoint's model config MUST win over CLI defaults, or the
        # arch won't match the weights (missing/unexpected keys on load).
        hf_seed_state = None
        if prior_model_config and prior_model_config.get("canvit"):
            # RESUME beats every seed mode (train/loop.py 219): rebuild the arch the
            # checkpoint was written with, so its weights load. Without this a CLI
            # default (e.g. canvas_update_mode) would silently override the saved arch
            # and the strict load would fail on missing/unexpected keys.
            import dacite

            from canvit import CanViTForPretrainingConfig
            log = __import__("logging").getLogger(__name__)
            ckpt_cfg = dacite.from_dict(CanViTForPretrainingConfig, prior_model_config["canvit"])
            if ckpt_cfg != self.cfg.model:
                log.warning("Overriding cfg.model from the checkpoint (was %s, now %s)",
                            self.cfg.model.canvas_update_mode, ckpt_cfg.canvas_update_mode)
            self.cfg.model = ckpt_cfg
        elif self.cfg.hf_seed_ckpt:
            from canvit.core.model.pretraining.hub import CanViTForPretrainingHFHub
            log = __import__("logging").getLogger(__name__)
            log.info("HF SEED mode: loading %s", self.cfg.hf_seed_ckpt)
            hf_model = CanViTForPretrainingHFHub.from_pretrained(self.cfg.hf_seed_ckpt)
            hf_seed_state = dict(hf_model.state_dict())
            self.cfg.model = hf_model.cfg
            del hf_model
        teacher = None
        if self.cfg.init_backbone_from_teacher:
            teacher = self._load_teacher(device)
        backbone = load_student_backbone(self.cfg, teacher=teacher)
        # `cfg.model.teacher_dim` is a documented PLACEHOLDER (train/config.py:113 —
        # "overridden by create_model based on actual teacher"): train/loop.py:294 passes
        # the REAL width as `create_model(backbone, teacher.embed_dim, cfg)`, and
        # create_model:63 assigns it back onto the config. Passing cfg.model.teacher_dim
        # here made that a self-assignment no-op, hardwiring the harness to the 768
        # default — correct for dinov3-vitb16 only. A vitl16 teacher is 1024 (5 exp21
        # launchers use it, all still on the old loop), which would have silently built
        # 768-wide distill heads on the first harness run.
        # A resume / HF-seed config is authoritative — its heads already have a width —
        # so only derive from the teacher on a fresh run.
        from_ckpt = bool(prior_model_config and prior_model_config.get("canvit")) or bool(
            self.cfg.hf_seed_ckpt)
        teacher_dim = (self.cfg.model.teacher_dim if from_ckpt
                       else self._teacher_embed_dim(device))
        bundle = create_model(backbone, teacher_dim, self.cfg)
        self._model, self._device = bundle.model, device
        self._glimpse_size_px = bundle.glimpse_size_px
        self.cls_norm, self.scene_norm = bundle.model.standardizers(self.cfg.canvas_patch_grid_size)
        if hf_seed_state is not None:
            from canvit.checkpoint import load_state_dict_flexible
            load_state_dict_flexible(bundle.model, hf_seed_state)
        return bundle.model, None

    def _teacher_embed_dim(self, device) -> int:
        """The teacher's REAL feature width. Read from the HF config when possible so a
        with-features run does not pay a full teacher load just to size the distill heads
        (the teacher is still loaded lazily later, for validation)."""
        if self._teacher is not None:
            return self._teacher.embed_dim
        try:
            from transformers import AutoConfig
            dim = getattr(AutoConfig.from_pretrained(self.cfg.teacher_repo_id),
                          "hidden_size", None)
            if isinstance(dim, int) and dim > 0:
                log.info("Teacher width from HF config (%s): %d", self.cfg.teacher_repo_id, dim)
                return dim
        except Exception as e:  # offline/unknown arch — fall back to the real thing
            log.warning("Could not read teacher width from the HF config (%s); "
                        "loading the teacher to size the distill heads", e)
        return self._load_teacher(device).embed_dim

    def _load_teacher(self, device):
        """The teacher, NEVER compiled. See `_teacher_for_forward` for why that matters."""
        if self._teacher is None:
            from canvit.distill.model import load_teacher
            self._teacher = load_teacher(self.cfg)
        return self._teacher

    def _teacher_for_forward(self, device):
        """The teacher for FORWARD passes, compiled under cfg.compile.

        Compilation is deliberately NOT done in `_load_teacher`, and the split is load-
        bearing: `compile_teacher` does `teacher.model = torch.compile(teacher.model)`,
        and the wrapper renames every parameter with an `_orig_mod.` prefix. Backbone
        teacher-init looks weights up BY NAME (`load_dinov3_weights_into_backbone` checks
        `model.layer.{n-1}.norm1.weight`), so initialising from a compiled teacher raises
        "Teacher has fewer than 12 transformer layers" — a hard startup crash for the
        production config (`init_backbone_from_teacher=True` + `compile=True`).
        train/loop.py gets this right by ordering: load_teacher(284) ->
        load_student_backbone(291) -> compile_teacher(303). Anything that reads the
        teacher STRUCTURALLY (weights, `.config`, `.embed_dim`) must use `_load_teacher`.

        train/loop.py:303 compiles the teacher alongside the model under cfg.compile
        ("Compiling teacher and model"); the harness used to compile only the student, so
        every harness run drove an EAGER teacher for validation targets and for raw-shard
        on-the-fly training targets. Measured impact on the targets is negligible
        (1-cos 1.2e-07, claude_dev/unification/teacher_compile_delta.py) — this is for speed and
        for not leaving a gratuitous asymmetry behind.
        """
        teacher = self._load_teacher(device)
        if self.cfg.compile and not self._teacher_compiled:
            from canvit.distill.model import compile_teacher
            log.info("Compiling teacher (torch.compile), matching train/loop.py")
            compile_teacher(teacher)
            self._teacher_compiled = True
        return teacher

    def canvas_grid(self, model):
        return self.cfg.canvas_patch_grid_size

    metrics_prefix = "val"
    """Tracker namespace for what ``evaluate`` returns. distill's per-timestep series have
    been ``val/scene_cos_*`` since the old loop, and ``validate`` used to log them itself;
    now that it RETURNS them (F8) the caller must keep the same namespace or every exp22-exp32
    dashboard breaks. ade20k/in1k have no attribute here and take the harness default,
    ``eval/``, which is what they have always used."""

    def is_foveated(self, model):
        return getattr(model.cfg, "patcher_name", "uniform") in ("foveated", "square")

    def branches(self):
        return ([ViewpointType.FULL] * self.cfg.n_full_start_branches
                + [ViewpointType.RANDOM] * self.cfg.n_random_start_branches)

    def build_val_loader(self):
        """The val loader alone (see the ade20k twin). Note ``build_loaders`` additionally
        seeds the normalizer from a training shard when the checkpoint carried no stats;
        standalone evaluation gets them from the checkpoint and asserts so."""
        from canvit.distill.data import create_imagefolder_val_loader
        return create_imagefolder_val_loader(self.cfg)

    def build_loaders(self, *, world_size, rank):
        from canvit.distill.data import create_loaders
        from canvit.distill.data.webdataset import (
            WebDatasetTrainLoader,
            init_normalizer_stats_from_tar,
            init_normalizer_stats_from_tar_raw,
        )
        loaders = create_loaders(self.cfg, job_index=self._start_job_index,
                                 world_size=world_size, rank=rank)
        train, val = loaders.train, loaders.val
        assert isinstance(train, WebDatasetTrainLoader), (
            "DistillRunTask currently supports the webdataset path only (set cfg.webdataset_dir)"
        )
        self._train_loader, self._world_size = train, world_size
        self._check_schedule_invariants(train, world_size=world_size)  # fail before any work
        # `reset_normalizer` forces a re-init even when the checkpoint carried stats.
        if self.cfg.reset_normalizer or not self.scene_norm.initialized:
            if train.has_features:
                init_normalizer_stats_from_tar(
                    train.normalizer_shard_paths(self.cfg.normalizer_shards),
                    self.scene_norm, self.cls_norm, self._device,
                    # `0` is the documented sentinel for "use the whole shard" (config.py);
                    # an `or 512` here silently capped it at 512 and gave the harness
                    # DIFFERENT target statistics than train/loop.py (exp23: 512 vs 4096
                    # samples off the same shard). Pass it through, like the raw branch.
                    self.cfg.normalizer_max_samples,
                )
            else:
                # Raw (jpg+json) shards carry no teacher features — seed the stats from
                # teacher forwards on the fly, like train/loop.py's raw branch.
                sz = self._scene_size_px()
                init_normalizer_stats_from_tar_raw(
                    train.normalizer_shard_paths(self.cfg.normalizer_shards),
                    self.scene_norm, self.cls_norm,
                    image_size=sz, compute_features=lambda imgs: self._teacher_targets(imgs, sz),
                    device=self._device, max_samples=self.cfg.normalizer_max_samples,
                )
        return train, val

    def _scene_size_px(self):
        """Scene resolution for the RAW-shard teacher forwards.

        train/loop.py:307-308 sizes this from the TEACHER's patch size
        (`teacher.model.config.patch_size`), not the student's: the scene must tokenize
        into exactly G x G *teacher* patches, since those are the distillation targets.
        The harness used `model.backbone.patch_size_px` (the STUDENT's) — identical while
        both are /16, as in every config to date, but silently wrong for a mixed pair.
        The teacher is already loaded on this path (it computes the targets), so reading
        it is free."""
        from canvit.distill.data import scene_size_px
        teacher_patch = getattr(self._load_teacher(self._device).model.config,
                                "patch_size", None)
        if teacher_patch is None:  # unknown teacher arch — student's is the best guess
            log.warning("Teacher exposes no patch_size; falling back to the student's")
            teacher_patch = self._model.backbone.patch_size_px
        return scene_size_px(self.cfg.canvas_patch_grid_size, teacher_patch)

    def _teacher_targets(self, images, sz):
        """Frozen-teacher features for RAW shards (the on-the-fly path). Resizes to the
        scene resolution first, exactly like train/loop.py's ``compute_raw_targets``."""
        from canvit.core.backbone.vit import NormFeatures
        teacher = self._teacher_for_forward(self._device)
        amp = (torch.autocast("cuda", dtype=torch.bfloat16)
               if self._device.type == "cuda" else nullcontext())
        with torch.no_grad(), amp:
            if images.shape[-1] != sz:
                images = torch.nn.functional.interpolate(images, size=(sz, sz), mode="bilinear",
                                                         align_corners=False)
            feats = teacher.forward_norm_features(images)
            return NormFeatures(patches=feats.patches.float(), cls=feats.cls.float())

    def build_selector(self, *, device, canvas_grid, is_foveated):
        from canvit.harness.rollout.selector import RandomSelector
        return RandomSelector(is_foveated=is_foveated, foveated_scale=self.cfg.foveated_scale,
                              min_viewpoint_scale=self.cfg.min_viewpoint_scale)

    def build_policy(self, model, *, device, canvas_grid, generator):
        from canvit.harness.policy import build_policy
        # distill: canvit == the pretraining model (its .cfg has canvas_dim/patcher_name);
        # encode_model defaults to SimpleNamespace(canvit=model) (probe-free INTRINSIC groups).
        return build_policy(
            canvit=model, rl=self.cfg.rl, feature_groups=POLICY_FEATURE_GROUPS, device=device,
            canvas_grid=canvas_grid, min_viewpoint_scale=self.cfg.min_viewpoint_scale,
            foveated_scale=self.cfg.foveated_scale, generator=generator, encode_model=None,
        )

    def policy_feature_groups(self):
        return POLICY_FEATURE_GROUPS

    def trainable_param_groups(self, *, model, head, joint, spec):
        groups: dict[str, list] = {}
        if spec.train_backbone:  # distill's in-forward heads ride the backbone group
            groups["backbone"] = list(model.parameters())
        if spec.train_policy:
            assert joint is not None
            params = list(joint.scorer.parameters())
            if joint.critic is not None:  # Q-Prop: critic rides the same param group/LR
                params += list(joint.critic.parameters())
            groups["policy"] = params
        return groups

    # --- visualization (ported from train/loop.py; saved LOCALLY, never uploaded) ---
    def viz_frame(self, *, model, images, gout, viewpoint, loss):
        """Per-glimpse viz sample for branch 0 / sample 0 — the engine's ``collect_viz``
        hook. Reuses the existing, tested ``extract_sample0_viz`` so the figure content
        is identical to the historical loop."""
        from canvit.distill.viz.sample import extract_sample0_viz
        core = getattr(model, "module", model)
        return extract_sample0_viz(
            gout.readout, images, viewpoint, loss.scene_pred, core, self._glimpse_size_px,
        )

    def viz_init(self, *, model, images, state):
        """Pre-glimpse panels (initial scene prediction + canvas) for sample 0, plus the
        denormalized input image — the engine calls this once per viz step, before t0."""
        from canvit.distill.viz.image import imagenet_denormalize_to_numpy
        core = getattr(model, "module", model)
        with torch.no_grad():
            init_scene = core.predict_teacher_scene(state.canvas)
            init_spatial = core.get_spatial(state.canvas[0:1])[0]
        return {
            "image": imagenet_denormalize_to_numpy(images[0]),
            "initial_scene": init_scene[0].detach().cpu().float().numpy(),
            "initial_canvas_spatial": init_spatial.detach().cpu().float().numpy(),
        }

    def render_viz(self, viz, *, batch, run_dir, step):
        """Render + save the multistep PCA figure to
        ``{run_dir}/visualization/pca_train/step-{step}.png`` (LOCAL disk — the current
        pretrain convention; the older wandb-backed upload path is deliberately NOT used)."""
        from canvit.distill.viz import plot_multistep_pca, save_figure

        if not viz.frames or not viz.initial:
            return
        samples, init = viz.frames, viz.initial
        # Teacher target for sample 0 (standardized), the figure's reference panel.
        _, raw_patches, _, _ = batch
        teacher = self.scene_norm(raw_patches[:1].to(self._device, dtype=torch.float32))

        img = init["image"]
        H, W = img.shape[:2]
        fov = [getattr(s, "foveated", None) for s in samples]
        sq = [getattr(s, "square", None) for s in samples]
        fig = plot_multistep_pca(
            full_img=img,
            teacher=teacher[0].detach().cpu().float().numpy(),
            scenes=[s.predicted_scene for s in samples],
            glimpses=[s.glimpse for s in samples],
            boxes=[vp.to_pixel_box(0, H, W) for vp in viz.viewpoints],
            names=[vp.name for vp in viz.viewpoints],
            scene_grid_size=self.cfg.canvas_patch_grid_size,
            glimpse_grid_size=self.cfg.glimpse_grid_size,
            initial_scene=init["initial_scene"],
            hidden_spatials=[s.canvas_spatial for s in samples],
            initial_hidden_spatial=init["initial_canvas_spatial"],
            foveated_samples=fov if any(f is not None for f in fov) else None,
            square_samples=sq if any(f is not None for f in sq) else None,
        )
        save_figure(fig, run_dir, "pca_train", step)

    # --- WebDataset multi-job (SLURM-array) resume, ported from train/loop.py -------
    def resume_start_step(self, payload, scheduler):
        """Where this job starts training.

        Sharded/feature path: the scheduler's step count, like the other two tasks.

        WebDataset path (the production one): the step is derived from the SHARD
        SCHEDULE, not the scheduler. Each job consumes exactly ``steps_per_job`` clean,
        shard-aligned steps, so the next job starts at ``(saved job_index + 1) *
        steps_per_job`` and reads the shard slice after the saved one. Resuming at any
        other offset silently re-processes or skips training shards, so both failure
        modes are hard errors instead: a checkpoint without a ``job_index``, and a
        checkpoint whose scheduler disagrees with the derived step (a mid-job save —
        SIGUSR1 preemption — or a job that ran a step count other than ``steps_per_job``).
        """
        if self.cfg.webdataset_dir is None:
            return scheduler.last_epoch
        saved = (payload.get("metadata") or {}).get("resume_state") or {}
        if saved.get("job_index") is None:
            raise RuntimeError(
                "WebDataset resume requires a checkpoint carrying `job_index` "
                "(metadata.resume_state) — this one has none, so the next shard slice "
                "is unknown. Seed from it instead (RunSettings.seed_ckpt) to start a "
                "fresh shard schedule from these weights."
            )
        self._resume_saved = saved
        self._start_job_index = int(saved["job_index"]) + 1
        start_step = self._start_job_index * self.cfg.steps_per_job
        if scheduler.last_epoch != start_step:
            raise RuntimeError(
                f"WebDataset resume: scheduler.last_epoch={scheduler.last_epoch} != "
                f"start_step={start_step} (= (saved job_index {saved['job_index']} + 1) "
                f"* steps_per_job {self.cfg.steps_per_job}). The checkpoint was not "
                "written at an end-of-job boundary — a mid-job / signal-triggered save, "
                "or the job ran a different number of steps than steps_per_job. Resume "
                "from an end-of-job checkpoint, or seed to start a fresh schedule."
            )
        return start_step

    def _check_schedule_invariants(self, train_loader, *, world_size) -> None:
        """Refuse to resume when the shard-schedule inputs changed. The slice offset
        (``job_index * shards_per_gpu * world_size``) is only meaningful under the
        (world_size, batch, steps_per_job, samples_per_shard) it was computed with;
        changing any of them silently re-processes or skips shards."""
        if self._resume_saved is None:
            return
        current = {
            "ddp_world_size": world_size,
            "batch_size_per_gpu": self.cfg.batch_size_per_gpu,
            "steps_per_job": self.cfg.steps_per_job,
            "samples_per_shard": train_loader.samples_per_shard,
        }
        saved = {k: self._resume_saved.get(k) for k in current}
        if saved != current:
            raise RuntimeError(
                "WebDataset resume config mismatch — refusing to resume because the "
                "shard-schedule offset would be wrong (silent shard re-processing or "
                f"skipping). Saved: {saved}. Current: {current}. To start a new training "
                "schedule with different values, seed from this checkpoint instead."
            )

    def resume_state(self):
        """Shard-schedule state stored in the checkpoint so the NEXT job resumes
        shard-aligned (consumed by ``resume_start_step`` / ``_check_schedule_invariants``).
        Empty off the WebDataset path or before the loaders exist."""
        if self.cfg.webdataset_dir is None or self._train_loader is None:
            return {}
        return {
            "job_index": self._start_job_index,
            "ddp_world_size": self._world_size,
            "batch_size_per_gpu": self.cfg.batch_size_per_gpu,
            "steps_per_job": self.cfg.steps_per_job,
            "samples_per_shard": self._train_loader.samples_per_shard,
        }

    def batch_images(self, batch, device):
        return batch[0].to(device, non_blocking=self.cfg.non_blocking_transfer)

    def bind(self, batch, device, *, model, head):
        images, raw_patches, raw_cls, _ = batch
        nb = self.cfg.non_blocking_transfer  # train/loop.py:631 honours this; we must too
        if raw_patches is None:
            # RAW (no-feature) shards: the loader supplies images only, so the frozen
            # teacher produces this batch's targets here (train/loop.py load_train_batch).
            feats = self._teacher_targets(images.to(device, non_blocking=nb), self._scene_size_px())
            raw_patches, raw_cls = feats.patches, feats.cls
        else:
            # non_blocking is NOT cosmetic here. Without it, `.to(device, dtype=float32)`
            # from a pinned fp16 tensor casts on the HOST first: measured 191 ms of
            # CPU-thread stall per step for the [B,1024,768] targets (vs 0.2 ms with it),
            # and the unpinned intermediate cannot overlap either. That stall is hidden
            # only to the extent the GPU still has queued work, which is why it cost ~10%
            # end-to-end and inflated step-time variance 5x (the hiding depends on the
            # stochastic rollout length). train/loop.py:655-656 always had this.
            raw_patches = raw_patches.to(device, dtype=torch.float32, non_blocking=nb)
            raw_cls = raw_cls.to(device, dtype=torch.float32, non_blocking=nb)
        return BoundDistillTask(
            DistillTask(
                scene_target=self.scene_norm(raw_patches),
                cls_target=self.cls_norm(raw_cls.unsqueeze(1)).squeeze(1),
                enable_scene_patches_loss=self.cfg.enable_scene_patches_loss,
                enable_scene_cls_loss=self.cfg.enable_scene_cls_loss,
            ),
            metric_refs={  # logging only — the raw-space cosine series
                "scene_denorm": self.scene_norm.destandardize,
                "cls_denorm": self.cls_norm.destandardize,
                "raw_scene_target": raw_patches, "raw_cls_target": raw_cls,
            },
        )

    @torch.no_grad()
    def evaluate(self, *, model, head, val_loader, device, step, tracker=None, run_dir=None,
                 joint=None):
        """The full distill validation phase (train/loop.py 689-734): ``validate()`` over
        the fixed val subset, logging its per-timestep cos/recon series through the REAL
        tracker, plus the IN1k linear-probe readout, the curve plots and the PCA figure on
        their historical cadences. Needs the teacher (cached, offline) for val targets.
        Best-effort: a readout, not parity-gated — returns {} if it can't run."""
        import tempfile
        from pathlib import Path

        from canvit.core.backbone.vit import NormFeatures
        from canvit.distill.probe import load_probe
        from canvit.distill.validate import validate
        from canvit.harness.infra.tracker import make_tracker
        try:
            teacher = self._teacher_for_forward(device)  # validation targets: compiled
            amp = torch.autocast("cuda", dtype=torch.bfloat16) if device.type == "cuda" else nullcontext()

            def compute_raw_targets(images, sz):
                with amp:
                    if images.shape[-1] != sz:
                        images = torch.nn.functional.interpolate(images, size=(sz, sz),
                                                                 mode="bilinear", align_corners=False)
                    feats = teacher.forward_norm_features(images)
                    return NormFeatures(patches=feats.patches.float(), cls=feats.cls.float())

            exp = tracker or make_tracker(
                tracker="none", is_main=True, is_seeding=False, run_name="distill-eval",
                wandb_project=None, wandb_entity=None, wandb_dir=None,
                prev_comet_id=None, prev_wandb_id=None)
            # Viz/curve cadence counts VALIDATIONS, as in train/loop.py — which assumes
            # the harness's eval_every is the config's val_every (the launcher sets both).
            val_count = step // max(1, self.cfg.val_every)
            fs = self.cfg.foveated_scale
            if self._probe is None:  # IN1k linear probe on the teacher's features
                self._probe = load_probe(self.cfg.teacher_name, device)
            with amp:
                series = validate(
                    exp=exp, step=step, model=model, compute_raw_targets=compute_raw_targets,
                    scene_normalizer=self.scene_norm, cls_normalizer=self.cls_norm,
                    val_batches=val_loader.batches(), device=device,
                    canvas_grid_size=self.cfg.canvas_patch_grid_size,
                    # Teacher-derived, like train/loop.py:308-309 (`scene_size`): the val
                    # scene must tokenize into G x G TEACHER patches, since the teacher
                    # produces the val targets. Was the student's patch size — identical
                    # while both are /16, wrong for a mixed pair.
                    scene_size_px=self._scene_size_px(),
                    glimpse_size_px=self._glimpse_size_px,
                    run_dir=run_dir or Path(tempfile.mkdtemp(prefix="distill_eval_")),
                    n_eval_viewpoints=self.cfg.n_eval_viewpoints,
                    min_viewpoint_scale=self.cfg.min_viewpoint_scale,
                    # foveated/square validate at the TRAINING scale for mode='fixed';
                    # sampled modes keep the scale-1 FULL anchor.
                    foveated_eval_scale=(fs.fixed_scale if fs.mode == "fixed" else 1.0),
                    # The shared validation-trajectory knob (harness/eval_viewpoints.py).
                    # "auto" (the default) reproduces this task's historical choice
                    # exactly: C2F for uniform, fixation_grid for foveated/square.
                    eval_policy=self.cfg.eval_policy, foveated_scale=fs, joint=joint,
                    override_scale=self.cfg.eval_override_scale,
                    prefix="val", probe=self._probe,
                    log_curves=(val_count % max(1, self.cfg.curve_every_n_vals) == 0),
                    log_pca=(val_count % max(1, self.cfg.viz_every_n_vals) == 0),
                    teacher=teacher, log_spatial_stats=self.cfg.log_spatial_stats,
                    teacher_name=self.cfg.teacher_name,
                )
            # `val_metric` is kept as an alias of scene_cos_raw_t{last} -- it is the key
            # exp22-exp36 logged and the one the loop's log line reads. The rest of the
            # series now comes back too, so a standalone evaluation sees more than one
            # scalar (eval-merge doc §5, F8).
            return {"val_metric": float(series["scene_cos_raw"]),
                    **{k: float(v) for k, v in series.items()}}
        except Exception as e:  # eval is a readout; never let it kill training
            import logging
            logging.getLogger(__name__).warning("distill evaluate() skipped: %s", e, exc_info=True)
            return {}

    def model_config(self, model):
        from dataclasses import asdict
        return {"task": "distill", "teacher_dim": self.cfg.model.teacher_dim,
                "canvas_grid": self.cfg.canvas_patch_grid_size, "backbone_name": self.cfg.backbone_name,
                # The FULL arch, so a resume can rebuild exactly this model before loading
                # the weights (see build_model's prior_model_config branch).
                "canvit": asdict(self.cfg.model)}

    def checkpoint_metadata(self, model):
        # pretrain_view_scale is the P6 FOOTGUN: the foveated/square view scale is NOT
        # in the HF config.json, so every downstream consumer must be told explicitly.
        # `to_hf` reads it from here / training_config_history. Only meaningful for the
        # foveated+square patchers at a fixed scale; None otherwise (uniform / sampled).
        from dataclasses import asdict

        fs = self.cfg.foveated_scale
        view_scale = fs.fixed_scale if (self.is_foveated(model) and fs.mode == "fixed") else None
        core = getattr(model, "module", model)
        return {
            "task": "distill",
            "scene_resolution": self.cfg.scene_resolution,
            "dataset": self.cfg.dataset,
            "patcher_name": getattr(self.cfg.model, "patcher_name", "uniform"),
            "foveated_scale_mode": fs.mode,
            "pretrain_view_scale": view_scale,
            "backbone_name": self.cfg.backbone_name,
            "glimpse_grid_size": self.cfg.glimpse_grid_size,
            # Needed to rebuild the model from an HF export (checkpoint/to_hf.py writes
            # both into config.json). `patch_stride` lives OUTSIDE model_config and is
            # the only record of an overlapping-patch model (exp21) — without it the
            # patch-embed conv is rebuilt non-overlapping and the weights mismatch.
            "canvas_patch_grid_sizes": list(core.canvas_patch_grid_sizes),
            "patch_stride": self.cfg.patch_stride,
            # Full view-scale config (not just the scalar above): `to_hf` needs mode +
            # distribution + range to describe a SAMPLED-scale model, where a single
            # `pretrain_view_scale` float is meaningless.
            "foveated_scale": asdict(fs),
            "teacher_repo_id": self.cfg.teacher_repo_id,
            "teacher_name": self.cfg.teacher_name,
        }


__all__ = ["POLICY_FEATURE_GROUPS", "BoundDistillTask", "DistillRunTask"]
