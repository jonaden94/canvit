"""Visual-search task config (class-cued search, COCO-Search18 first).

Design: ``claude_dev/search/00-design.md``. Haoze's defaults (autoregressive_detection,
branch haoze-wip) unless canvit has a natural equivalent; every formulation choice the
owner named is an option here or in the policy config (``search_policy_config``).
"""

from __future__ import annotations

import math
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

from canvit.harness.config import FoveatedScaleConfig, JointPolicyConfig


def _default_data_root() -> Path:
    if root := os.environ.get("COCO_SEARCH18_ROOT"):
        return Path(root)
    return Path("/user/henrich1/u25995/jonathan/datasets/coco_search18")


def _default_wandb_project() -> str | None:
    return os.environ.get("WANDB_PROJECT")


def _default_wandb_entity() -> str | None:
    return os.environ.get("WANDB_ENTITY") or None


def _default_wandb_dir() -> Path | None:
    if d := os.environ.get("WANDB_DIR"):
        return Path(d)
    return None


def _fixed_scale_2() -> FoveatedScaleConfig:
    """exp22/exp32/exp34/exp36 all pretrain the foveated backbone at view scale 2.0."""
    return FoveatedScaleConfig(mode="fixed", fixed_scale=2.0)


@dataclass
class SearchAugConfig:
    """Train-time augmentation (torchvision v2, boxes transformed with the image), applied
    to the original image before letterboxing. Defaults per the design doc §2.3."""

    hflip_p: float = 0.5
    crop: bool = True
    """Random scaled crop with the image's aspect kept (output = the letterboxed content size)."""
    crop_scale: tuple[float, float] = (0.6, 1.0)
    color_jitter: bool = False
    """Haoze: ColorJitter(0.4, 0.4, 0.4, 0.1) p 0.8, then greyscale p 0.1."""
    rotate: bool = False
    """Haoze: rotation +-60 deg, p 0.5 (the box becomes the axis-aligned box of the rotated box)."""
    affine: bool = False
    """Haoze: affine rot +-45 deg, translate +-10 %, scale 0.8-1.2, shear +-15 deg, p 0.5."""
    autocontrast: bool = False
    """Haoze: autocontrast p 0.5."""
    min_box_fraction: float = 1.0
    """A sample is kept only if at least this fraction of the transformed target box lies
    inside the frame; otherwise the geometric chain is redrawn."""
    max_tries: int = 10
    """Redraws before falling back to the un-augmented sample."""


@dataclass
class SearchConfig:
    # --- data ------------------------------------------------------------------
    data_root: Path = field(default_factory=_default_data_root)
    """COCO-Search18 root: ``images/<category>/<name>.jpg`` + the fixation JSONs."""
    split: Literal[1, 2] = 1
    """The dataset's ``split1`` (the paper's) or ``split2`` train/valid assignment."""
    scene_size: int = 512
    """Square scene side in px; the 1680x1050 images are letterboxed into it."""
    aug: SearchAugConfig = field(default_factory=SearchAugConfig)

    # --- model -----------------------------------------------------------------
    model_repo: str | None = None
    """The pretrained FOVEATED CanViT: an HF repo id, a local HF dir or a training ``.pt``.
    Required (there is no published foveated search backbone to default to)."""
    canvas_grid: int | None = None
    """None = scene_size // patch_size (32 for ViT-B/16 at 512 px)."""
    foveated_scale: FoveatedScaleConfig = field(default_factory=_fixed_scale_2)
    """Must match the backbone's pretraining view scale, or every glimpse is out of distribution."""
    head: Literal["unet", "linear"] = "unet"
    head_width: int = 64
    head_block_layers: int = 2
    cond_dim: int = 64
    """Width of the target-class embedding(s) (Haoze: 64)."""
    embedding: Literal["separate", "shared"] = "separate"
    """``separate``: the task head and the policy each learn their own class embedding.
    ``shared``: one embedding (in the head) read by both, each with its own FiLM projections."""
    detach_policy_embedding: bool = False
    """``shared`` only: the policy reads a detached copy, so only the task loss trains it."""
    dropout: float = 0.0
    """Dropout2d before the head's output conv. Off by default."""

    # --- loss --------------------------------------------------------------------
    loss: Literal["kl_gaussian", "nll_point"] = "kl_gaussian"
    """``kl_gaussian`` (Haoze): KL(target || softmax over the canvas grid), target = a
    normalised Gaussian per box centre. ``nll_point``: cross-entropy on the cell holding the
    box centre (the sigma -> 0 limit; Tom's comparison)."""
    target_sigma: float = 2.5 / 64
    """Gaussian sigma as a fraction of the scene side (Haoze: 2.5 cells on a 64x64 heatmap)."""
    supervise: Literal["last", "every"] = "last"
    """``last`` (Haoze): only the last glimpse's loss trains the head. ``every``: all glimpses."""

    # --- rollout / eval ----------------------------------------------------------------
    n_timesteps: int = 4
    """Glimpses per episode: the centred t0 fixation plus n-1 chosen ones (Haoze: 4)."""
    eval_policies: tuple[str, ...] = ("random", "greedy", "policy")
    """Fixation policies run at every validation. ``random``: seeded random fixations.
    ``greedy``: the argmax of the current task heatmap, never revisiting a cell.
    ``policy``: the learned scorer's argmax (skipped when no policy is trained)."""
    hit_box_frac: float = 0.1
    """Haoze's hit criterion: a square box of this fraction of the scene side (64 px at 640)
    around the heatmap argmax must intersect a target box."""
    eval_seed: int = 1234
    limit_val_batches: int | None = None

    # --- optimisation -----------------------------------------------------------------
    batch_size: int = 16
    eval_batch_size: int = 32
    num_workers: int = 4
    peak_lr: float = 1e-4
    """Task head (Haoze: AdamW lr 1e-4)."""
    weight_decay: float = 0.05
    """Task head (Haoze: 0.05)."""
    warmup_steps: int = 500
    max_steps: int = 20000
    grad_clip: float = 1.0

    # --- run identity / logging (same trio as every task) ------------------------------
    run_group: str | None = None
    run_name: str | None = None
    logs_dir: Path = Path("logs")
    log_every: int = 20
    val_every: int = 500
    viz_every: int = 0
    device: str = "cuda"
    amp: bool = True
    seed: int = 0
    tracker: Literal["comet", "wandb", "none"] = "wandb"
    wandb_project: str | None = field(default_factory=_default_wandb_project)
    wandb_entity: str | None = field(default_factory=_default_wandb_entity)
    wandb_dir: Path | None = field(default_factory=_default_wandb_dir)


# FoAViS's entropy target was 3.5 nats on a 64x64 policy grid. The same spatial spread on
# the 32x32 grid is 3.5 - ln(4096/1024) = 3.5 - ln 4 ~ 2.11 nats (e^H ~ 8.3 effective cells,
# ~0.8 % of the image, as in FoAViS).
SEARCH_ENTROPY_TARGET = 3.5 - math.log(4.0)


def search_policy_config() -> JointPolicyConfig:
    """The search task's policy defaults (design doc §5): PG on the plain canvas, a 32x32
    fixation grid, Haoze's terminal reward with FoAViS's per-step batch standardisation and
    no learned baseline, FoAViS's entropy target and dual step adapted to 32x32, and a
    scorer as small as the task head (64 wide, 2 blocks per stage)."""
    return JointPolicyConfig(
        objective="pg",
        select_bn_eval=False,  # the search scorer has no BatchNorm; keep it explicit
        feature_groups=("ln_feat",),
        centers_per_axis=32,
        width=64,
        block_layers=2,
        reward="terminal",
        reward_center="batch_mean",
        reward_scale="batch_std",
        entropy_target=SEARCH_ENTROPY_TARGET,
        alpha_lr=0.01,
        entropy_alpha_init=1e-3,
        entropy_alpha_min=1e-6,
        entropy_alpha_max=0.1,
        entropy_update="per_rollout",
    )
