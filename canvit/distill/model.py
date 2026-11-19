"""Model creation and loading utilities."""

import logging
from typing import NamedTuple

import torch

from canvit import CanViTForPretraining
from canvit.core import create_backbone
from canvit.core.backbone.vit import ViTBackbone
from canvit.core.teacher import DINOv3Teacher
from canvit.core.teacher import load_teacher as _load_teacher

from .config import Config

log = logging.getLogger(__name__)


class ModelBundle(NamedTuple):
    """Model with derived runtime parameters."""

    model: CanViTForPretraining
    glimpse_size_px: int


def load_teacher(cfg: Config) -> DINOv3Teacher:
    """Load frozen DINOv3 teacher from HuggingFace Hub."""
    return _load_teacher(cfg.teacher_repo_id, cfg.device)


def load_student_backbone(cfg: Config, teacher: DINOv3Teacher | None = None) -> ViTBackbone:
    """Load student backbone.

    Random init by default (pretrained weights otherwise arrive via the checkpoint /
    seed system). With ``cfg.init_backbone_from_teacher`` the backbone is instead
    seeded from the already-loaded DINOv3 ``teacher`` (backbone only; see
    ``load_dinov3_weights_into_backbone``). A later resume / seed_ckpt overwrites this.
    """
    backbone = create_backbone(cfg.backbone_name, patch_stride=cfg.patch_stride,
                               recurrence=cfg.model.recurrence)
    if cfg.init_backbone_from_teacher:
        if teacher is None:
            raise ValueError("init_backbone_from_teacher=True but no teacher was provided.")
        from canvit.core.backbone.dinov3_init import load_dinov3_weights_into_backbone

        load_dinov3_weights_into_backbone(backbone, teacher.model)
        log.info(f"Student backbone: {cfg.backbone_name} (init from DINOv3 teacher, patch_stride={cfg.patch_stride})")
    else:
        log.info(f"Student backbone: {cfg.backbone_name} (random init, patch_stride={cfg.patch_stride})")
    return backbone.to(cfg.device)


def create_model(
    student_backbone: ViTBackbone,
    teacher_dim: int,
    cfg: Config,
) -> ModelBundle:
    """Create CanViTForPretraining wrapping student backbone.

    glimpse_size_px is the side length of the pixel crop the uniform patcher
    takes internally from the full image; the foveated patcher ignores it and
    operates on the full image at the viewpoint's fixation point.
    """
    cfg.model.teacher_dim = teacher_dim

    # Uniform crop size is purely a function of training config + backbone.
    # The foveated patcher doesn't use it; pass it through anyway for symmetry.
    # With overlapping patches (patch_stride < patch_size) the glimpse window is
    # (grid-1)*stride + patch; patch_stride_px defaults to patch_size, so this
    # reduces to grid*patch (the non-overlapping case) bit-for-bit.
    glimpse_size_px = (
        (cfg.glimpse_grid_size - 1) * student_backbone.patch_stride_px
        + student_backbone.patch_size_px
    )

    model = CanViTForPretraining(
        backbone=student_backbone,
        cfg=cfg.model,
        glimpse_size_px=glimpse_size_px,
        backbone_name=cfg.backbone_name,
        canvas_patch_grid_sizes=[cfg.canvas_patch_grid_size],
    ).to(cfg.device)

    if cfg.model.patcher_name == "foveated":
        log.info(
            f"Model created (foveated): canvas={cfg.canvas_patch_grid_size}x{cfg.canvas_patch_grid_size}, "
            f"pattern_reference_size={cfg.model.foveated_patcher.pattern_reference_size}px, "
            f"n_patches={model.patcher.n_patches}, "
            f"student_dim={student_backbone.embed_dim} -> teacher_dim={teacher_dim}"
        )
    else:
        log.info(
            f"Model created: canvas={cfg.canvas_patch_grid_size}x{cfg.canvas_patch_grid_size}, "
            f"glimpse={cfg.glimpse_grid_size}x{cfg.glimpse_grid_size} ({glimpse_size_px}px), "
            f"student_dim={student_backbone.embed_dim} -> teacher_dim={teacher_dim}"
        )
    return ModelBundle(model, glimpse_size_px)


def compile_teacher(teacher: DINOv3Teacher) -> None:
    """Compile teacher HF model blocks in-place."""
    teacher.model = torch.compile(teacher.model)  # type: ignore[assignment]


def compile_model(model: CanViTForPretraining) -> None:
    """Compile CanViTForPretraining in-place."""
    model.compile()
