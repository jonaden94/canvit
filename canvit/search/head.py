"""The target-conditioned task head: canvas + target class -> heatmap over the canvas grid.

Built from the policy scorer's own blocks (``core.policy.net``): the same ConvNeXt-V2
U-Net and the same FiLM, so the task head and the policy share one implementation and
one size knob, with separate weights. Design doc §3.3.
"""

from __future__ import annotations

from typing import Literal

from torch import Tensor, nn

from canvit.core.policy.net import FiLM, N_UNET_ENC, _stage, build_unet, build_unet_film, unet_forward


class SearchHead(nn.Module):
    """``[B, G, G, D]`` canvas tokens + ``[B]`` target class -> ``[B, G, G]`` logits.

    ``unet``: LayerNorm -> 1x1 conv D->width -> FiLM -> U-Net (FiLM after every encoder and
    decoder stage) -> one more ConvNeXt stage -> 1x1 conv to one logit per cell.
    ``linear``: LayerNorm -> 1x1 conv D->width -> FiLM -> 1x1 conv (the sanity baseline).

    ``embed`` is the task's target embedding; with ``embedding='shared'`` the policy reads it
    too (``policy_embedding``)."""

    def __init__(self, *, canvas_dim: int, n_classes: int, cond_dim: int = 64,
                 kind: Literal["unet", "linear"] = "unet", width: int = 64, block_layers: int = 2,
                 dropout: float = 0.0):
        super().__init__()
        self.kind = kind
        self.embed = nn.Embedding(n_classes, cond_dim)
        self.ln = nn.LayerNorm(canvas_dim)
        self.proj = nn.Conv2d(canvas_dim, width, 1)
        self.film_in = FiLM(cond_dim, width)
        if kind == "unet":
            self.enc, self.dec = build_unet(width, block_layers)
            self.film_enc, self.film_dec = build_unet_film(cond_dim, width)
            self.post = _stage(width, width, block_layers)
        elif kind != "linear":
            raise ValueError(f"unknown search head {kind!r}")
        self.dropout = nn.Dropout2d(dropout) if dropout > 0 else nn.Identity()
        self.out = nn.Conv2d(width, 1, 1)

    def forward(self, x: Tensor, target: Tensor) -> Tensor:
        b, g, g2, _ = x.shape
        assert g == g2, f"square canvas expected, got {g}x{g2}"
        if self.kind == "unet":
            assert g % (2 ** (N_UNET_ENC - 1)) == 0, (
                f"the U-Net pools {N_UNET_ENC - 1} times; canvas grid {g} must be a multiple of "
                f"{2 ** (N_UNET_ENC - 1)}")
        cond = self.embed(target)
        h = self.proj(self.ln(x).permute(0, 3, 1, 2).contiguous())
        h = self.film_in(h, cond)
        if self.kind == "unet":
            h = unet_forward(h, self.enc, self.dec, film=(self.film_enc, self.film_dec), cond=cond)
            h = self.post(h)
        return self.out(self.dropout(h))[:, 0]

    def policy_embedding(self, target: Tensor, *, detach: bool) -> Tensor:
        """The SHARED target embedding as the policy reads it."""
        e = self.embed(target)
        return e.detach() if detach else e


__all__ = ["SearchHead"]
