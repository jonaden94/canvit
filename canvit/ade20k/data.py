"""ADE20K dataset + loaders + optimizer/scheduler/amp helpers.

Faithful port of canvit_specialize's datasets/ade20k.py + training/ade20k/common.py
(the P2 gate is reproducing specialize's probe numbers, so augmentation and
optimization are kept identical). This is the ONE train-time ADE20K pipeline of
the unified repo (master plan §3 — the specialize/RL duplicates retire with
their repos); validation-protocol comparability is anchored by the squish resize in
``canvit.core.data.ade20k.make_val_transforms``, which every ADE20K number in this
project — published, specialize-era and current — was measured under.
"""

import torch
from dinov3.eval.segmentation.schedulers import WarmupOneCycleLR
from dinov3.eval.segmentation.transforms import make_segmentation_train_transforms
from PIL import Image
from torch import Tensor
from torch.optim import AdamW
from torch.optim.lr_scheduler import LRScheduler
from torch.utils.data import DataLoader

# ADE20K dataset + val transforms + label constants are the shared val-protocol
# primitives; they live in core (canvit.core) so CanViT-eval uses the exact
# same definitions without depending on this training package. Re-exported here
# so this repo's own consumers keep importing them from `.data`.
from canvit.core.data.ade20k import (  # noqa: F401  (re-exported)
    IGNORE_LABEL,
    NUM_CLASSES,
    ADE20kDataset,
    make_val_transforms,
)

from .config import Ade20kConfig


def make_ade20k_val_loader(cfg: Ade20kConfig) -> DataLoader:
    """The ADE20K val loader on its own.

    Lifted out of :func:`make_ade20k_loaders` so standalone evaluation and training-time
    validation draw the val set from ONE place — building the train split as well just to
    reach it would scan a directory eval never touches.
    """
    if not cfg.ade20k_root.exists():
        raise FileNotFoundError(
            f"ADE20K root not found: {cfg.ade20k_root}. Set ADE20K_ROOT or pass --ade20k-root."
        )
    val_img_tf, val_mask_tf = make_val_transforms(cfg.scene_size, cfg.resize_mode)
    val_ds = ADE20kDataset(root=cfg.ade20k_root, split="validation",
                           img_transform=val_img_tf, mask_transform=val_mask_tf)
    return DataLoader(val_ds, cfg.eval_batch_size, num_workers=cfg.num_workers, pin_memory=True)


def make_ade20k_loaders(cfg: Ade20kConfig) -> tuple[DataLoader, DataLoader]:
    """Build ADE20K train/val data loaders (dinov3 train augmentation, squish val)."""
    if not cfg.ade20k_root.exists():
        raise FileNotFoundError(
            f"ADE20K root not found: {cfg.ade20k_root}. Set ADE20K_ROOT or pass --ade20k-root."
        )

    val_img_tf, val_mask_tf = make_val_transforms(cfg.scene_size, cfg.resize_mode)

    if cfg.augment:
        _train_aug = make_segmentation_train_transforms(
            img_size=cfg.scene_size,
            random_img_size_ratio_range=list(cfg.aug_scale_range),
            # Upstream annotation is Tuple[int] but implementation expects (H, W).
            crop_size=(cfg.scene_size, cfg.scene_size),  # pyright: ignore[reportArgumentType]
            flip_prob=cfg.aug_flip_prob,
            reduce_zero_label=True,
        )

        def train_transform(img: Image.Image, mask: Image.Image) -> tuple[Tensor, Tensor]:
            img_t, mask_t = _train_aug(img, mask)
            # dinov3's pipeline yields uint8 masks; the val transform (and the RL protocol's
            # unaugmented train split) int64. The probe loss casts for itself, but the policy
            # reward (`per_image_ce`) feeds the mask straight to cross_entropy, which rejects
            # uint8 -- every augmented policy run died at step 0 (2026-11-16). Same values.
            return img_t, mask_t.squeeze(0).long()

        train_ds = ADE20kDataset(root=cfg.ade20k_root, split="training",
                                 joint_transform=train_transform)
    else:
        # The RL protocol: the TRAIN split goes through the val transform, exactly as
        # ade20k/rl_train.py does. Not the same as identity-valued aug knobs — see
        # Ade20kConfig.augment for why (RandomCrop + PhotoMetricDistortion have no knob).
        train_ds = ADE20kDataset(root=cfg.ade20k_root, split="training",
                                 img_transform=val_img_tf, mask_transform=val_mask_tf)
    train_loader = DataLoader(
        train_ds, cfg.batch_size, shuffle=True, num_workers=cfg.num_workers, pin_memory=True, drop_last=True
    )
    return train_loader, make_ade20k_val_loader(cfg)


def make_optimizer_and_scheduler(
    params, *, lr: float, weight_decay: float, max_steps: int,
    warmup_steps: int, warmup_lr_ratio: float,
) -> tuple[AdamW, LRScheduler]:
    """AdamW + WarmupOneCycleLR (identical to specialize's probe recipe)."""
    optimizer = AdamW(params, lr=lr, weight_decay=weight_decay)
    scheduler = WarmupOneCycleLR(
        optimizer,
        max_lr=lr,
        total_steps=max_steps,
        warmup_iters=warmup_steps,
        warmup_ratio=warmup_lr_ratio,
        pct_start=0,
        anneal_strategy="cos",
        final_div_factor=float("inf"),
        use_beta1=False,
        update_momentum=False,
    )
    return optimizer, scheduler


def make_amp_ctx(amp: bool, device: torch.device) -> torch.autocast:
    amp_dtype = torch.bfloat16 if amp else torch.float32
    return torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=amp)
