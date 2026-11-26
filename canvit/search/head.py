"""The target-conditioned task head: canvas + target class -> heatmap over the canvas grid.

Built from the policy scorer's own blocks (``core.policy.net``): the same ConvNeXt-V2
U-Net and the same FiLM, so the task head and the policy share one implementation and
one size knob, with separate weights. Design doc §3.3.
"""

from __future__ import annotations

from typing import Literal

import torch
from torch import Tensor, nn

from canvit.core.policy.net import FiLM, N_UNET_ENC, _stage, build_unet, build_unet_film, unet_forward


class SearchHead(nn.Module):
    """``[B, G, G, D]`` canvas tokens + ``[B]`` target class -> ``([B, G, G] heatmap logits,
    [B] presence logit or None)``.

    ``unet``: LayerNorm -> 1x1 conv D->width -> FiLM -> U-Net (FiLM after every encoder and
    decoder stage) -> one more ConvNeXt stage -> 1x1 conv to one logit per cell.
    ``linear``: LayerNorm -> 1x1 conv D->width -> FiLM -> 1x1 conv (the sanity baseline).

    ``presence=True`` adds Haoze's present/absent readout: the last target-conditioned
    feature map (the one the output conv reads) mean-pooled, then a Linear to ONE logit,
    log p(present) - log p(absent) (his two-way softmax, written as a sigmoid). No LSTM: the
    canvas already carries the evidence across glimpses.

    ``embed`` is the task's target embedding; with ``embedding='shared'`` the policy reads it
    too (``policy_embedding``)."""

    def __init__(self, *, canvas_dim: int, n_classes: int | None, cond_dim: int = 64,
                 kind: Literal["unet", "linear"] = "unet", width: int = 64, block_layers: int = 2,
                 dropout: float = 0.0, presence: bool = False):
        super().__init__()
        self.kind = kind
        # None: prompt-cued search, the caller hands the FiLM condition to forward (``cond``).
        self.embed = nn.Embedding(n_classes, cond_dim) if n_classes is not None else None
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
        self.presence = nn.Linear(width, 1) if presence else None

    def forward(self, x: Tensor, target: Tensor | None, cond: Tensor | None = None) -> tuple[Tensor, Tensor | None]:
        b, g, g2, _ = x.shape
        assert g == g2, f"square canvas expected, got {g}x{g2}"
        if self.kind == "unet":
            assert g % (2 ** (N_UNET_ENC - 1)) == 0, (
                f"the U-Net pools {N_UNET_ENC - 1} times; canvas grid {g} must be a multiple of "
                f"{2 ** (N_UNET_ENC - 1)}")
        if cond is None:
            assert self.embed is not None and target is not None
            cond = self.embed(target)
        h = self.proj(self.ln(x).permute(0, 3, 1, 2).contiguous())
        h = self.film_in(h, cond)
        if self.kind == "unet":
            h = unet_forward(h, self.enc, self.dec, film=(self.film_enc, self.film_dec), cond=cond)
            h = self.post(h)
        pres = self.presence(h.mean(dim=(2, 3)))[:, 0] if self.presence is not None else None
        return self.out(self.dropout(h))[:, 0], pres

    def policy_embedding(self, target: Tensor, *, detach: bool) -> Tensor:
        """The SHARED target embedding as the policy reads it."""
        e = self.embed(target)
        return e.detach() if detach else e


def sincos_2d(g: int, width: int, device=None) -> Tensor:
    """[g*g, width] fixed 2-D sine-cosine position code of a g x g grid (row-major), half of the channels for y,
    half for x (DETR / MAE style)."""
    assert width % 4 == 0, f"width {width} must be a multiple of 4"
    q = width // 4
    freq = 1.0 / (10000 ** (torch.arange(q, device=device, dtype=torch.float32) / q))
    pos = torch.arange(g, device=device, dtype=torch.float32)[:, None] * freq  # [g, q]
    code = torch.cat([pos.sin(), pos.cos()], dim=-1)  # [g, 2q]
    y = code[:, None, :].expand(g, g, 2 * q)
    x = code[None, :, :].expand(g, g, 2 * q)
    return torch.cat([y, x], dim=-1).reshape(g * g, width)


class _ReadoutLayer(nn.Module):
    """One pre-norm decoder layer: self-attention among the queries (masked tokens excluded as keys), cross-attention
    queries -> canvas (keys = memory + position code, values = memory), MLP. One-way: the canvas is not updated."""

    def __init__(self, width: int, n_heads: int):
        super().__init__()
        self.ln_sa, self.ln_ca, self.ln_mlp = nn.LayerNorm(width), nn.LayerNorm(width), nn.LayerNorm(width)
        self.sa = nn.MultiheadAttention(width, n_heads, batch_first=True)
        self.ca = nn.MultiheadAttention(width, n_heads, batch_first=True)
        self.mlp = nn.Sequential(nn.Linear(width, 4 * width), nn.GELU(), nn.Linear(4 * width, width))

    def forward(self, q: Tensor, ignore: Tensor | None, mem: Tensor, mem_k: Tensor) -> Tensor:
        if q.shape[1] > 1:
            h = self.ln_sa(q)
            q = q + self.sa(h, h, h, key_padding_mask=ignore, need_weights=False)[0]
        q = q + self.ca(self.ln_ca(q), mem_k, mem, need_weights=False)[0]
        return q + self.mlp(self.ln_mlp(q))


class PromptQueryHead(nn.Module):
    """Cross-attention readout (owner variant a, 2026-11-26; a small DETR-like decoder, one-way): the prompt is the
    query, the canvas tokens are the memory. Replaces the FiLM U-Net head; same outputs.

    Queries: ``Linear(prompt.vec)`` (the output query) plus ``Linear(prompt.tokens)`` when the prompt has tokens
    (``prompt.token_mask`` False = an empty cell, never attended to). Memory: ``Linear(LN(canvas))`` [B, G*G, w];
    the cross-attention keys add a fixed 2-D sine-cosine position code. After ``n_layers`` layers:
    heatmap logit of cell i = ``<W_h LN(q0), W_k mem_i> / sqrt(w)`` -> [B, G, G]; presence = ``Linear(LN(q0))``."""

    def __init__(self, *, canvas_dim: int, prompt_dim: int, width: int = 256, n_layers: int = 2, n_heads: int = 8,
                 presence: bool = False):
        super().__init__()
        self.width = width
        self.q_vec = nn.Linear(prompt_dim, width)
        self.q_tok = nn.Linear(prompt_dim, width)
        self.mem_ln = nn.LayerNorm(canvas_dim)
        self.mem = nn.Linear(canvas_dim, width)
        self.layers = nn.ModuleList([_ReadoutLayer(width, n_heads) for _ in range(n_layers)])
        self.out_ln = nn.LayerNorm(width)
        self.heat_q = nn.Linear(width, width)
        self.heat_k = nn.Linear(width, width)
        self.presence = nn.Linear(width, 1) if presence else None

    def forward(self, x: Tensor, prompt) -> tuple[Tensor, Tensor | None]:
        b, g, g2, _ = x.shape
        assert g == g2, f"square canvas expected, got {g}x{g2}"
        mem = self.mem(self.mem_ln(x.flatten(1, 2)))  # [B, G*G, w]
        mem_k = mem + sincos_2d(g, self.width, device=x.device).to(mem.dtype)
        q = self.q_vec(prompt.vec)[:, None]
        ignore = None
        if prompt.tokens is not None:
            q = torch.cat([q, self.q_tok(prompt.tokens).to(q.dtype)], dim=1)
            ignore = torch.cat([torch.zeros(b, 1, dtype=torch.bool, device=x.device), ~prompt.token_mask], dim=1)
        for layer in self.layers:
            q = layer(q, ignore, mem, mem_k)
        q0 = self.out_ln(q[:, 0])
        heat = torch.einsum("bw,bnw->bn", self.heat_q(q0), self.heat_k(mem)) / self.width ** 0.5
        pres = self.presence(q0)[:, 0] if self.presence is not None else None
        return heat.view(b, g, g), pres


__all__ = ["PromptQueryHead", "SearchHead", "sincos_2d"]
