"""ImageNet-1k classification config (unification P5). Fresh CUDA task (D2),
mirroring ade20k/config.py, and step-based like it (max_steps / warmup_steps /
val_every) — the train stream is an infinite resampled WebDataset, so epochs were
only ever a derived batch count."""

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

from canvit.core import resolve_canvit_repo

from ..ade20k.config import (
    ResizeMode,
    _default_wandb_dir,
    _default_wandb_entity,
    _default_wandb_project,
)
from ..harness.config import FoveatedScaleConfig
from ..harness.rollout.eval_viewpoints import EvalPolicy

NUM_CLASSES = 1000


def _default_in1k_train_dir() -> Path:
    if root := os.environ.get("IN1K_TRAIN_DIR"):
        return Path(root)
    # WebDataset shards (jpg + json label), pre-resized to 512²; see p5-notes.md.
    return Path("/mnt/vast-nhr/projects/nib00021/jonathan/datasets/webdataset-imagenet-1k-no-features/train-shuffled")


def _default_in1k_val_dir() -> Path:
    if root := os.environ.get("IN1K_VAL_DIR"):
        return Path(root)
    # Synset-folder ImageFolder (n01440764/, …): ImageFolder's alphabetical class_idx
    # is what makes the labels match the webdataset's int labels, so the directory layout
    # is load-bearing, not incidental.
    return Path("/mnt/vast-nhr/projects/nib00021/jonathan/datasets/imagenet1k-val")


@dataclass
class In1kConfig:
    """CanViT ImageNet-1k classification: frozen-backbone linear probe (default)
    or full finetune, over a glimpse rollout."""

    model_repo: str = resolve_canvit_repo("canvitb16-add-vpe-pretrain-g128px-s512px-in21k-dv3b16-2026-02-02")
    retina: str | None = None
    """A MULTI-RETINA backbone is used through this one retina, by name as the backbone's patcher
    lists it (e.g. ``cmf0.75_res65_cps6``), for training and eval. Required for such a backbone,
    refused for a single-retina one (``core.patcher.foveated.fix_backbone_retina``)."""
    train_dir: Path = field(default_factory=_default_in1k_train_dir)
    val_dir: Path = field(default_factory=_default_in1k_val_dir)
    scene_size: int = 512

    mode: Literal["frozen", "finetune"] = "frozen"
    """``frozen`` (default, the P5 acceptance target): freeze the CanViT backbone,
    train only the LN+Linear head — the direct analogue of canvit_eval's frozen
    linear-clf-probe baseline. ``finetune``: train the whole classifier end to end
    (the ``...-finetune-...-in1k`` flagship)."""

    probe_repo: str | None = None
    """FINETUNE only: the DINOv3 in1k linear probe fused into the classifier head,
    reproducing the TPU flagship (``gcp_in1k_clf_ft/shared.py::load_classifier``). A
    finetune from a RANDOM head starts at chance and — at the tiny finetune LR — trains
    far too slowly. ``None`` => derive from the checkpoint's backbone via
    ``dinov3_in1k_probes.repos.probe_repo``. Ignored in ``frozen`` mode (fresh head)."""

    # Rollout (how the CLS token that feeds the head is produced)
    n_timesteps: int = 10
    bptt_chunk_size: int = 0
    """Truncated-BPTT chunk size for ``mode="finetune"``. 0 (default) = one graph over the
    whole rollout; n>0 = backward + detach every n glimpses, capping activation memory at
    n steps instead of ``n_timesteps``.

    ``n_timesteps`` need NOT be divisible by this: the rollout flushes the trailing
    partial chunk and every chunk normalises by ``n_timesteps``, so e.g. 7 glimpses at
    chunk 3 runs [0,1,2][3,4,5][6] with the same total gradient. ``n >= n_timesteps``
    collapses to one graph.

    IGNORED in ``mode="frozen"``: the backbone runs under ``no_grad`` there, and ``bptt``
    only ever moves the backbone — head gradients are bit-identical between ``none`` and
    ``full`` (measured 2026-07-28). Chunking a frozen backbone costs memory and changes
    nothing. See ``harness.spec.fixed_horizon_bptt``."""
    glimpse_px: int | None = None
    """Uniform-patcher glimpse crop px. None = derive from the model's
    glimpse_grid_size × patch size/stride (``harness/rollout/episode.py``). Ignored for
    foveated/square models (they consume the full image)."""
    canvas_grid: int | None = None
    """None = scene_size // patch_size."""

    # TRAINING viewpoint policy: IID random by default (matches the ade20k probe)
    min_vp_scale: float = 0.05
    max_vp_scale: float = 1.0
    train_start_full: bool = False
    foveated_scale: FoveatedScaleConfig = field(default_factory=FoveatedScaleConfig)
    """Foveated/square only: the view-scale law for the rollout, which MUST match
    how the backbone was pretrained (``fix_size = scale * H``; off-scale glimpses
    are out of distribution). Ignored for uniform models. See ade20k/config.py."""

    # EVAL viewpoint policy — the SHARED option set (harness/eval_viewpoints.py), the
    # same one distill and ade20k take. "auto" = this task's historical trajectory,
    # coarse-to-fine (the canvit_eval deploy default), for uniform AND foveated alike.
    # See HISTORICAL_DEFAULTS on why foveated keeps C2F despite the scale mismatch.
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

    # Training (step-based, like Ade20kConfig). Train uses the SAME resumable, shard-
    # aligned schedule as distill pretraining (canvit.harness.infra.schedule): a
    # seeded global shard permutation, each SLURM-array job consuming a contiguous block.
    max_steps: int = 200_000
    """Total optimizer steps = the LR-schedule (cosine) horizon across ALL array jobs.
    The old epoch-based default was 10 epochs ~= 200k steps @ batch 64 over 1,281,167 imgs."""
    steps_per_job: int | None = None
    """Steps ONE SLURM-array job runs before exiting (the shard-schedule window; the next
    job resumes at the next shard block). None => single job of `max_steps`. MUST be
    shard-aligned: steps_per_job * batch_size a multiple of images_per_shard (enforced by
    compute_shards_per_gpu), else clean cross-job resume is impossible."""
    batch_size: int = 64
    eval_batch_size: int = 64
    num_workers: int = 8
    peak_lr: float = 3e-4
    weight_decay: float = 1e-3
    warmup_steps: int = 10_000
    """LR warmup length in steps (was 0.5 epoch ~= 10k steps at batch_size=64)."""
    warmup_lr_ratio: float = 1e-6
    grad_clip: float = float("inf")
    label_smoothing: float = 0.0
    shuffle_buffer: int = 2000
    """Seeded within-stream shuffle buffer over the job's shard slice (0 = off). The
    shards are already globally pre-shuffled at creation, so this only adds cross-shard
    mixing / per-epoch reorder; seeded by `seed + job_index` so a re-run reproduces order
    (resume-safe: it permutes WITHIN the job's shards, never across the job boundary)."""

    # Data augmentation (train): RandomResizedCrop + flip (canonical IN1k probe recipe)
    augment: bool = True
    """Train-split augmentation. ``False`` makes the TRAIN split use the val
    preprocessing — the protocol policy training is defined under (doc 15 §A). Same name
    and meaning as ``Ade20kConfig.augment``; the implementations differ because the two
    tasks use different transform stacks, so this is a shared INTERFACE, not shared code.

    Default ``True``: every in1k probe/finetune number, including the exp25 arrays and the
    standalone gate (job 15046042), was measured with augmentation on."""
    aug_min_scale: float = 0.35
    aug_flip_prob: float = 0.5
    """Both ignored when ``augment=False``."""
    resize_mode: ResizeMode = "center_crop"
    """Val resize. ``center_crop`` (default) is ``canvit.core.preprocess`` — resize
    the short side, then centre-crop — and preserves geometry; ``squish`` keeps the full frame but distorts
    aspect ratio. Both work for every patcher — see Ade20kConfig.resize_mode for the
    trade-off (aspect-preserving reads better against human viewing for foveated models,
    squish for comparability with numbers measured under squish)."""

    # Debug / smoke: cap batches per eval (None = full). Train length is `max_steps`.
    limit_val_batches: int | None = None

    # Logging / checkpoints
    log_every: int = 50
    val_every: int = 20_000
    """Validate every N steps (was eval_every_epochs=1, i.e. ~20k steps at batch_size=64)."""
    device: str = "cuda"
    amp: bool = True
    seed: int = 0

    # Run identity — see Ade20kConfig for the shared contract. `run_name` used to default
    # to the constant "in1k-clf" and was read by the harness ONLY (the standalone always
    # auto-named), so every unnamed harness run showed up under one wandb name while the
    # standalone silently ignored the field.
    run_group: str | None = None
    run_name: str | None = None
    """None => auto: the descriptive `in1k_{mode}_{model}_{T}t_s{scene}_{ts}` name in the
    standalone, `in1k_{timestamp}` in the harness."""
    logs_dir: Path = Path("logs")
    tracker: Literal["comet", "wandb", "none"] = "wandb"
    wandb_project: str | None = field(default_factory=_default_wandb_project)
    wandb_entity: str | None = field(default_factory=_default_wandb_entity)
    wandb_dir: Path | None = field(default_factory=_default_wandb_dir)
