"""Configuration for CanViT pretraining."""

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

import torch

from canvit import CanViTForPretrainingConfig
from canvit.harness.config import FoveatedScaleConfig, JointPolicyConfig
from canvit.harness.infra.utils import get_sensible_device
from canvit.harness.rollout.eval_viewpoints import EvalPolicy

# Default HF repo for the teacher model
TEACHER_REPO_ID = "facebook/dinov3-vitb16-pretrain-lvd1689m"
# Short name used for shard paths and probe lookup (matches precomputed feature directories)
TEACHER_NAME = "dinov3_vitb16"


@dataclass
class Config:
    # Teacher
    teacher_repo_id: str = TEACHER_REPO_ID
    teacher_name: str = TEACHER_NAME
    # Student
    backbone_name: str = "vitb16"
    # Model config (PretrainingConfig via alias)
    # teacher_dim placeholder - overridden by create_model based on actual teacher
    model: CanViTForPretrainingConfig = field(
        default_factory=lambda: CanViTForPretrainingConfig(teacher_dim=768)
    )
    # Glimpse/canvas sizes (runtime, not in model config)
    glimpse_grid_size: int = 8  # tokens per glimpse side
    patch_stride: int | None = None  # uniform patcher: patch-embed conv stride;
    # None = patch_size (non-overlapping, default). Set < patch_size for overlapping
    # patches; glimpse_size_px is then (grid-1)*stride + patch_size.
    canvas_patch_grid_size: int = 32  # canvas spatial grid side length in tokens
    # Training
    batch_size_per_gpu: int = 64
    warmup_steps: int = 100_000
    start_lr: float | None = 1e-7  # None = peak_lr / warmup_steps
    peak_lr: float = 4e-4
    cosine_total_steps: int | None = None  # None = constant after warmup; set to enable cosine decay
    weight_decay: float = 1e-4
    min_viewpoint_scale: float = 0.05  # Minimum scale for random viewpoints
    n_full_start_branches: int = 1  # branches starting with FULL viewpoint at t0
    n_random_start_branches: int = 1  # branches starting with RANDOM viewpoint at t0
    foveated_scale: FoveatedScaleConfig = field(default_factory=FoveatedScaleConfig)
    """Per-glimpse view-scale sampling for foveated/square patchers (RANDOM
    glimpses). Default: fixed scale 1.0 = current full-image foveation."""
    rl: JointPolicyConfig = field(default_factory=JointPolicyConfig)
    """Joint task+policy training (P4b). OFF by default => byte-identical to
    pre-P4b pretraining. See :class:`JointPolicyConfig`."""
    chunk_size: int = 2  # BPTT chunk size (glimpses per chunk, gradient flows within)
    continue_prob: float = 0.5  # prob of adding another chunk to trajectory
    enable_scene_patches_loss: bool = True  # Scene (canvas) patch reconstruction loss
    enable_scene_cls_loss: bool = True  # Scene (global) CLS reconstruction loss
    ema_alpha: float = 0.1  # EMA smoothing for metrics
    grad_clip: float = 1.0
    steps_per_job: int = 4_992  # Steps this job does before exiting (for SLURM arrays)
    # Data
    val_dir: Path = Path("/mnt/vast-nhr/projects/nib00021/jonathan/datasets/imagenet1k-val")
    """IN1k val ImageFolder (raw images) for distill validation — glimpse rollouts scored
    by the frozen IN1k probes need actual pixels, so this is NOT the feature webdataset.
    Every launcher passes `--cfg.val-dir "$VAL_DIR"` from `.envrc.grete`, so this default
    only applies to ad-hoc local runs; `harness_train.sbatch` fails at launch if VAL_DIR
    is unset. Was a Nibi path (`/datasets/ILSVRC/...`) that does not exist on Grete."""
    train_index_dir: Path | None = None
    """Fallback source for ``val_index_dir`` when that is unset — despite the name this
    feeds VALIDATION's parquet index, not training (training reads WebDataset shards)."""
    val_image_dir: Path | None = None
    """Evaluate reconstruction on an arbitrary, LABEL-FREE image directory instead of the
    ImageNet-1k val ImageFolder (``val_dir``). Recursive and filename-sorted, so a flat
    folder (ADE20K's ``images/validation``) and a nested one both work.

    This is what ``canvit_eval``'s ``reconstruction`` task pointed at, and it is the only
    thing that task had which distill validation did not — the cosine-to-teacher series it
    computed is what ``validate`` already returns (eval-merge doc §5, Stage 4). ``val_dir``
    cannot serve this: ``IndexedImageFolder`` needs class subdirectories, and ADE20K's val
    images are flat.

    Labels are absent, so the IN1k linear-probe readout is skipped for these images (it is
    already gated on ``labels_are_in1k``); the cosine series are unaffected."""
    val_index_dir: Path | None = None  # parquet index cache for the val ImageFolder
    webdataset_dir: Path | None = None
    """THE training data source: pre-shuffled WebDataset tar shards under
    ``{webdataset_dir}/train-shuffled``. Required in practice — ``create_loaders``
    asserts it is set. Typed optional only so ``Config()`` stays constructible in tests.

    Shards may carry precomputed teacher features (cls.npy + ptch.npy) or be RAW
    (jpg+json only), in which case the frozen teacher computes targets on the fly — the
    exp21/exp22 path. Both go through ``WebDatasetTrainLoader``."""
    seed: int = 0  # for reproducibility (shard schedule permutations)
    # Run identification and checkpointing
    run_group: str | None = None
    """Run category (e.g. 'foveated', 'crop'). Combined with run_name to form
    `logs_dir / run_group / run_name /` as the root of all per-run artifacts."""
    run_name: str | None = None
    """Run name. Auto-generated from SLURM_ARRAY_JOB_ID or timestamp if None."""
    logs_dir: Path = Path("logs")
    """Root for run artifacts. Per-run files go under
    `logs_dir / run_group / run_name / {checkpoints,log}/`."""
    seed_ckpt: Path | None = None
    """Seed model weights from external checkpoint (.pt in CheckpointData format).
    Starts fresh (new experiment, step=0). Only used if no checkpoint exists in run_dir."""
    resume_from: Path | None = None
    """Continue another run's checkpoint with its full state (RunSettings.resume_from), e.g.
    an LR decay of that run: step count, AdamW and shard schedule continue; the LR is this
    config's. Only used if no checkpoint exists in run_dir."""
    hf_seed_ckpt: str | None = None
    """Seed model weights from HF Hub repo (e.g. '<org>/canvitb16-add-vpe-...'). Downloads
    config.json + model.safetensors, overrides cfg.model with the checkpoint's config.
    Mutually exclusive with seed_ckpt."""
    init_backbone_from_teacher: bool = False
    """Initialize the student ViT backbone from the (already-loaded) DINOv3 teacher's
    weights instead of random init. The two share the same ViT-B/16 design (12 layers,
    12 heads, head_dim 64, RoPE theta 100, LayerScale, 4x MLP), so the full transformer
    trunk transfers 1:1 (teacher q/k/v fused into the student's qkv, K-bias zero-filled);
    patch_embed transfers only when the student patch size matches the teacher's (16 ->
    vitb16 yes, vitb8 no -> left random). Backbone only; the rest of CanViT (canvas, VPE,
    heads) stays random. Applied on a fresh start; resume/seed_ckpt take precedence."""
    reset_normalizer: bool = False
    """Re-warmup normalizer stats when loading any checkpoint."""
    normalizer_max_samples: int = 0
    """Max samples per shard for normalizer stats. 0 = use all samples in the shard."""
    normalizer_shards: int = 4
    """How many shards to pool for the normalizer stats (see `normalizer_shard_paths`).

    The standardizers are POSITION-AWARE (`set_stats` reduces over dim 0 only), so each
    of the grid_size^2 x embed_dim means/vars is estimated from `normalizer_shards *
    samples_per_shard` samples — NOT from that many times the token count. At one shard
    (4096 samples) the sampling error is ~1.3% of a std on the mean and ~1.4% on the std;
    it falls as 1/sqrt(n_shards), so the default 4 puts it near 0.65%. Measured 2026-07-28
    on the in21k with-features set: shard-to-shard variation is 1.05x pure split-half
    sampling noise, i.e. shards are effectively i.i.d., so pooling more shards behaves
    exactly like drawing more samples. Cost is ~40 s per shard, paid ONCE per run (array
    task 0; later tasks load the frozen buffers from the checkpoint and skip init).

    The shards are the first n of the SORTED shard list, so every run pools the same
    shards regardless of `seed`, `job_index` or `rank` — unlike the historical
    `first_shard_path()`, which took the head of the seed-dependent schedule slice and so
    made the target statistics a function of the seed.

    Changing this value changes the target normalization and therefore the absolute loss
    scale: `*_cos_norm` and the losses are NOT comparable across different values. Metrics
    that are: `*_cos_raw` (destandardized back into true teacher-feature space) and
    `val/in1k_tts_top1_t*` (downstream probe accuracy, normalizer-independent).

    Runs before 2026-07-28 all used 1 shard, chosen by seed."""
    # Training
    num_workers: int = 4
    scene_resolution: int = 512
    dataset: str = "in21k"
    # Logging
    log_every: int = 20
    val_every: int = 1000
    n_eval_viewpoints: int = 10  # Number of viewpoints in validation
    eval_override_scale: float | None = None
    """Pin every eval glimpse to this scale while keeping the policy's CENTERS. Mirrors
    ``canvit_eval``'s ``EpisodeConfig.override_scale``, and ``Ade20kConfig``'s field of the
    same name — one knob on all three tasks, each keeping its own default (F4). ``None``
    (default) = off, an exact no-op.

    For a FIXED-SCALE FOVEATED backbone under a scale-varying policy: ``fix_size = scale * H``,
    so a glimpse at a scale the model never trained on is out of distribution and the metric
    decays as glimpses accumulate. Set this to the pretraining view scale to measure "that
    policy's centres at the model's own scale" instead. Leave unset for uniform backbones,
    whose out-of-distribution axis is the glimpse crop in pixels, not the view scale."""
    eval_policy: EvalPolicy = "auto"
    """Validation trajectory — the SHARED option set (harness/eval_viewpoints.py), the
    same one ade20k and in1k take. ``"auto"`` = this task's historical, PATCHER-DEPENDENT
    choice: quadtree ``coarse_to_fine`` for uniform, ``fixation_grid`` (deterministic
    centre + shuffled 3x3 at the training scale) for foveated/square, because the
    quadtree's varying scales are out of distribution for a fixed-scale foveated model.
    ``"policy"`` deploys a trained scorer by argmax. Left at ``"auto"`` for
    comparability: the exp22/23/26 val curves were all measured under it."""
    n_val_samples: int = 256
    """Number of validation samples evaluated per validation, independent of
    ``batch_size_per_gpu`` and of world size. A fixed, seeded random subset of the
    val set (the SAME images every validation — no cycling), clamped to the val set
    size. Evaluated on rank 0 only, in chunks of ``batch_size_per_gpu`` whose
    per-timestep metrics are aggregated (chunk size does not affect the result)."""
    val_seed: int = 0
    """Seed selecting the fixed validation subset. Same seed -> same images across
    runs and across world sizes."""
    viz_every_n_vals: int = 5  # Log viz every N validation runs
    curve_every_n_vals: int = 5  # Log curves every N validation runs
    log_spatial_stats: bool = True
    log_patcher_grad_detail: bool = True
    """Break the patcher's per-validation grad-norm logs into sub-components:
    ``patcher.kpe``, ``patcher.embed_head``, ``patcher.conditioner.*`` (FiLM MLP /
    learned per-patch code / etc.). When False, the patcher is reported as a
    single ``patcher`` group like every other top-level module. All other modules
    (backbone, scene_cls_head, …) are unaffected either way."""
    # Experiment tracker
    tracker: Literal["comet", "wandb", "none"] = "wandb"
    """Backend for parameter/metric/figure logging."""
    wandb_project: str | None = field(default_factory=lambda: os.environ.get("WANDB_PROJECT"))
    """W&B project name. Required when tracker='wandb'. Defaults to $WANDB_PROJECT
    (.envrc.grete sets it) — the same default Ade20kConfig/In1kConfig already had, which
    distill alone was ignoring. Every launcher passes it explicitly, so this only changes
    what a hand-run job does: land in the default project instead of asserting."""
    wandb_entity: str | None = field(default_factory=lambda: os.environ.get("WANDB_ENTITY") or None)
    """W&B entity (team or user). Falls back to $WANDB_ENTITY, then your default account."""
    wandb_dir: Path | None = field(
        default_factory=lambda: Path(d) if (d := os.environ.get("WANDB_DIR")) else None)
    """Directory wandb writes its run files into. Defaults to $WANDB_DIR (.envrc.grete
    sets it), then to wandb's own default (./wandb). Same rule as
    Ade20kConfig/In1kConfig; this used to be one user's absolute path, which no other
    project member can write to."""
    # Compilation and precision
    compile: bool = True
    combo_kernels: bool = False  # torch._inductor.config.combo_kernels (experimental)
    amp: bool = True
    non_blocking_transfer: bool = True  # Ablation: async CPU→GPU transfers
    # Optuna
    n_trials: int = 1
    # Runtime
    device: torch.device = field(default_factory=get_sensible_device)
