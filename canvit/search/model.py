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
    """``prompt_kw`` (prompt-cued search, ``prompt.py``): ``adaln``, ``tokens``, ``condition_heads`` and
    ``scene_grid`` (keep the pretraining scene head + that grid's standardizer, for the distill loss and the
    cosine baseline; None = drop them). Without it (``n_classes`` given) the class-cued model of COCO-Search18."""

    def __init__(self, *, backbone_name: BackboneName, model_config: dict, n_classes: int | None,
                 glimpse_grid_size: int | None = None, prompt_kw: dict | None = None, **head_kw):
        super().__init__()
        cfg = rebuild_canvit_config(model_config)
        self.canvit = CanViT(backbone=create_backbone(backbone_name), cfg=cfg)
        self.backbone_name = backbone_name
        self.glimpse_grid_size = glimpse_grid_size
        self.canvit.glimpse_grid_size = glimpse_grid_size
        self.head_kw = dict(head_kw)
        self.prompt_kw = dict(prompt_kw) if prompt_kw is not None else None
        self.head = SearchHead(canvas_dim=self.canvit.canvas_dim, n_classes=n_classes, **head_kw)
        if self.prompt_kw is not None:
            from .prompt import PromptConditioning
            bb = self.canvit.backbone
            self.prompt = PromptConditioning(dim=bb.embed_dim, num_heads=bb.num_heads, n_blocks=bb.n_blocks,
                                             adaln=self.prompt_kw["adaln"], tokens=self.prompt_kw["tokens"])
            self.prompt.attach(bb)
            cond_dim = self.head_kw.get("cond_dim", 64)
            self.head_cond = nn.Linear(bb.embed_dim, cond_dim) if self.prompt_kw["condition_heads"] else None
            self.scene: nn.ModuleDict | None = None
            if self.prompt_kw.get("scene_grid") is not None:
                from canvit.core.standardizers import PatchStandardizer
                teacher_dim = self.prompt_kw["teacher_dim"]
                self.scene = nn.ModuleDict({
                    "norm": nn.LayerNorm(self.canvit.canvas_dim),
                    "proj": nn.Linear(self.canvit.canvas_dim, teacher_dim),
                    "std": PatchStandardizer(grid_size=self.prompt_kw["scene_grid"], embed_dim=teacher_dim)})

    def init_state(self, *, batch_size: int, canvas_grid_size: int):
        return self.canvit.init_state(batch_size=batch_size, canvas_grid_size=canvas_grid_size)

    def search_head(self, x, cue):
        """The task head on canvas tokens ``x`` [B, G, G, D] for this batch's cue: a class index (class-cued) or a
        ``prompt.Prompt`` (its 1x1 vector FiLM-conditions the head, or a zero vector with condition_heads off)."""
        if self.prompt_kw is None:
            return self.head(x, cue)
        if self.head_cond is not None:
            cond = self.head_cond(cue.vec)
        else:
            cond = cue.vec.new_zeros(cue.vec.shape[0], self.head_kw.get("cond_dim", 64))
        return self.head(x, None, cond=cond)

    def predict_scene(self, x):
        """Standardized DINOv3 patch features predicted from canvas tokens ``x`` [B, G, G, D] -> [B, G*G, T]."""
        assert self.scene is not None
        return self.scene["proj"](self.scene["norm"](x.flatten(1, 2)))

    @classmethod
    def from_model_config(cls, mc: dict) -> CanViTForVisualSearch:
        """The architecture a ``search`` checkpoint's ``model_config`` records (backbone + head),
        with UNTRAINED weights: the caller loads the checkpoint's ``model_state`` into it."""
        return cls(backbone_name=cast(BackboneName, mc["backbone_name"]), model_config=mc["canvit"],
                   n_classes=mc["n_classes"], glimpse_grid_size=mc.get("glimpse_grid_size"),
                   prompt_kw=mc.get("prompt_kw"), **mc["head_kw"])

    @classmethod
    def from_pretrained_backbone(cls, pretrained_repo: str, *, n_classes: int | None, prompt_kw: dict | None = None,
                                 **head_kw) -> "CanViTForVisualSearch":
        """A pretrained CanViT (HF id, local HF dir or training ``.pt``) + a FRESH head (and fresh prompt
        conditioning; with ``prompt_kw['scene_grid']`` the pretraining scene head and that grid's standardizer)."""
        log.info("Loading pretrained CanViT from %s (fresh search head, cue: %s)", pretrained_repo,
                 "prompt" if prompt_kw is not None else f"{n_classes} classes")
        pretrained: CanViTForPretraining = load_pretraining(pretrained_repo)
        assert pretrained.backbone_name in get_args(BackboneName), f"unknown backbone {pretrained.backbone_name!r}"
        if prompt_kw is not None and prompt_kw.get("scene_grid") is not None:
            prompt_kw = dict(prompt_kw, teacher_dim=pretrained.cfg.teacher_dim)
        model = cls(backbone_name=cast(BackboneName, pretrained.backbone_name),
                    model_config=serialize_canvit_config(pretrained.cfg), n_classes=n_classes,
                    glimpse_grid_size=pretrained.glimpse_grid_size, prompt_kw=prompt_kw, **head_kw)
        base_sd = {k: v for k, v in pretrained.state_dict().items()
                   if not any(k.startswith(p) for p in PRETRAINING_ONLY_PREFIXES)}
        missing, unexpected = model.canvit.load_state_dict(base_sd, strict=False)
        assert not missing, f"Missing CanViT keys: {missing}"
        assert not unexpected, f"Unexpected CanViT keys: {unexpected}"
        if getattr(model, "scene", None) is not None:
            g = str(model.prompt_kw["scene_grid"])  # type: ignore[index]
            if g not in pretrained.scene_standardizers:
                raise ValueError(f"{pretrained_repo} has scene standardizers for grids "
                                 f"{list(pretrained.scene_standardizers.keys())}, not {g}")
            model.scene["norm"].load_state_dict(pretrained.scene_patches_head["norm"].state_dict())
            model.scene["proj"].load_state_dict(pretrained.scene_patches_head["proj"].state_dict())
            model.scene["std"].load_state_dict(pretrained.scene_standardizers[g].state_dict())
        return model
