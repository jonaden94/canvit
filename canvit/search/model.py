"""CanViT + the target-conditioned search head, in one ``nn.Module``.

Mirrors ``CanViTForSemanticSegmentation``: a bare CanViT (``.canvit``, pretraining-only
modules dropped) and a head (``.head``) applied outside the backbone forward, so a frozen
backbone still trains the head.
"""

from __future__ import annotations

import logging
from typing import cast, get_args

from torch import nn

from canvit.core.backbone import BackboneName, create_backbone
from canvit.core.model.base.config import rebuild_canvit_config, serialize_canvit_config
from canvit.core.model.base.impl import CanViT
from canvit.core.model.pretraining.impl import CanViTForPretraining
from canvit.core.model_source import load_pretraining

from .head import SearchHead

log = logging.getLogger(__name__)

PRETRAINING_ONLY_PREFIXES = ("scene_cls_head.", "scene_patches_head.", "cls_standardizers.", "scene_standardizers.")


class CanViTForVisualSearch(nn.Module):
    def __init__(self, *, backbone_name: BackboneName, model_config: dict, n_classes: int,
                 glimpse_grid_size: int | None = None, **head_kw):
        super().__init__()
        cfg = rebuild_canvit_config(model_config)
        self.canvit = CanViT(backbone=create_backbone(backbone_name), cfg=cfg)
        self.backbone_name = backbone_name
        self.glimpse_grid_size = glimpse_grid_size
        self.canvit.glimpse_grid_size = glimpse_grid_size
        self.head_kw = dict(head_kw)
        self.head = SearchHead(canvas_dim=self.canvit.canvas_dim, n_classes=n_classes, **head_kw)

    def init_state(self, *, batch_size: int, canvas_grid_size: int):
        return self.canvit.init_state(batch_size=batch_size, canvas_grid_size=canvas_grid_size)

    @classmethod
    def from_pretrained_backbone(cls, pretrained_repo: str, *, n_classes: int, **head_kw) -> "CanViTForVisualSearch":
        """A pretrained CanViT (HF id, local HF dir or training ``.pt``) + a FRESH head."""
        log.info("Loading pretrained CanViT from %s (fresh %d-class search head)", pretrained_repo, n_classes)
        pretrained: CanViTForPretraining = load_pretraining(pretrained_repo)
        assert pretrained.backbone_name in get_args(BackboneName), f"unknown backbone {pretrained.backbone_name!r}"
        model = cls(backbone_name=cast(BackboneName, pretrained.backbone_name),
                    model_config=serialize_canvit_config(pretrained.cfg), n_classes=n_classes,
                    glimpse_grid_size=pretrained.glimpse_grid_size, **head_kw)
        base_sd = {k: v for k, v in pretrained.state_dict().items()
                   if not any(k.startswith(p) for p in PRETRAINING_ONLY_PREFIXES)}
        missing, unexpected = model.canvit.load_state_dict(base_sd, strict=False)
        assert not missing, f"Missing CanViT keys: {missing}"
        assert not unexpected, f"Unexpected CanViT keys: {unexpected}"
        return model
