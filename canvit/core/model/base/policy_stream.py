"""Isolated policy stream (``policy_token_mode="isolated"``): the policy tokens run beside the
reconstruction path with their OWN weights and read it only through detached tensors, so the
policy loss reaches no reconstruction parameter (owner idea 2026-11-23, a "dorsal stream").

Per backbone block l, the tokens attend to [the block's local k, v (detached, RoPE as computed
by the block) | their own k, v] with their own q/k/v projections, then their own MLP. After a
canvas read they read the detached canvas (own read weights); after a canvas write they read the
detached local tokens the write used (own weights), i.e. they receive the write like the canvas.
Nothing in the reconstruction path reads the tokens. The readout scores the detached canvas.

``copy_from_trunk`` initialises the block and read weights from the trunk's (same shapes), so a
stream seeded from a pretrained model starts as the read-only tokens' computation; the write
receivers start at zero output (their ``out_proj`` is zero-initialised) and their k/v side is
copied from the canvas write.
"""

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from canvit.core.attention.base import CanvasAttention
from canvit.core.attention.read import CanvasReadAttention
from canvit.core.attention.write import CanvasWriteAttention
from canvit.core.backbone.vit import ViTBlock
from canvit.core.rope import RoPE


class PolicyBlock(ViTBlock):
    """A backbone-shaped block for the policy tokens (own weights, same parameter names as
    ``ViTBlock`` so the trunk's block can be copied in)."""

    def forward_isolated(self, r: Tensor, k_task: Tensor, v_task: Tensor) -> Tensor:
        """r [B, K, D] attends to [detached task k, v | its own k, v]; positionless (no RoPE)."""
        B, K, D = r.shape
        q_r, k_r, v_r = self.attn._qkv(self.norm1(r), None)
        a = F.scaled_dot_product_attention(
            q_r, torch.cat([k_task.detach(), k_r.to(k_task.dtype)], dim=2),
            torch.cat([v_task.detach(), v_r.to(v_task.dtype)], dim=2), scale=self.attn.scale)
        r = r + self.ls1(self.attn.proj(a.transpose(1, 2).reshape(B, K, D)))
        return r + self.ls2(self.mlp(self.norm2(r)))


class PolicyWriteReceiver(CanvasAttention):
    """The policy tokens' share of a canvas write: they query the local tokens (as the canvas
    does) and add the result to themselves. q: local_dim -> canvas_dim; k, v: local_dim ->
    canvas_dim; out: canvas_dim -> local_dim, zero-initialised (no effect at init)."""

    _mod_side = None

    def __init__(self, *, local_dim: int, canvas_dim: int, num_heads: int) -> None:
        super().__init__(q_in_dim=local_dim, kv_in_dim=local_dim, canvas_dim=canvas_dim,
                         out_dim=local_dim, num_heads=num_heads)
        self.q_proj = nn.Linear(local_dim, canvas_dim)
        self.k_proj = nn.Linear(local_dim, canvas_dim)
        self.v_proj = nn.Linear(local_dim, canvas_dim)
        self.out_proj = nn.Linear(canvas_dim, local_dim)
        nn.init.zeros_(self.out_proj.weight)
        nn.init.zeros_(self.out_proj.bias)


class PolicyStream(nn.Module):
    def __init__(self, *, n_blocks: int, local_dim: int, num_heads: int, ffn_ratio: float,
                 layerscale_init: float, canvas_dim: int, canvas_num_heads: int,
                 n_reads: int, n_writes: int) -> None:
        super().__init__()
        self.blocks = nn.ModuleList([PolicyBlock(local_dim, num_heads, ffn_ratio, layerscale_init)
                                     for _ in range(n_blocks)])
        self.reads = nn.ModuleList([CanvasReadAttention(local_dim=local_dim, canvas_dim=canvas_dim,
                                                        num_heads=canvas_num_heads) for _ in range(n_reads)])
        self.writes = nn.ModuleList([PolicyWriteReceiver(local_dim=local_dim, canvas_dim=canvas_dim,
                                                         num_heads=canvas_num_heads) for _ in range(n_writes)])

    def read(self, i: int, r: Tensor, canvas: Tensor, *, query_rope: RoPE, kv_rope: RoPE) -> Tensor:
        return r + self.reads[i](query=r, kv=canvas.detach(), query_rope=query_rope, kv_rope=kv_rope)

    def write(self, i: int, r: Tensor, local: Tensor, *, query_rope: RoPE, kv_rope: RoPE) -> Tensor:
        return r + self.writes[i](query=r, kv=local.detach(), query_rope=query_rope, kv_rope=kv_rope)

    def copy_from_trunk(self, model: nn.Module) -> None:
        """Copy the trunk's block and canvas-read weights into the stream (strict, same shapes),
        and the canvas writes' k/v side into the write receivers (their q starts fresh, their
        output at zero). Call after the trunk's weights are final (e.g. after a seed load)."""
        for dst, src in zip(self.blocks, model.backbone.blocks, strict=True):  # type: ignore[attr-defined]
            dst.load_state_dict(src.state_dict())
        for dst, src in zip(self.reads, model.canvas_read, strict=True):  # type: ignore[attr-defined]
            dst.load_state_dict(src.state_dict())
        for dst, src in zip(self.writes, model.canvas_write, strict=True):  # type: ignore[attr-defined]
            if isinstance(src, CanvasWriteAttention):  # asymmetric write: local -> canvas k/v
                for name in ("kv_norm", "k_proj", "v_proj"):
                    getattr(dst, name).load_state_dict(getattr(src, name).state_dict())
