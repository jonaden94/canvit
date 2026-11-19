"""Self-contained ViT backbone for CanViT."""

import logging
import math
from typing import NamedTuple

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from canvit.core.backbone.recurrence import (
    DepthRouter,
    ExpertMLP,
    RecurrenceConfig,
    aux_losses,
    expert_weight_cos_max,
)
from canvit.core.rope import RoPE, rope_apply_with_prefix

log = logging.getLogger(__name__)


class NormFeatures(NamedTuple):
    patches: Tensor  # [B, H*W, D]
    cls: Tensor  # [B, D]


# ---------------------------------------------------------------------------
# Building blocks
# ---------------------------------------------------------------------------


class PatchEmbed(nn.Module):
    def __init__(self, patch_size: int, embed_dim: int, stride: int | None = None) -> None:
        super().__init__()
        # stride defaults to patch_size (standard non-overlapping patchify). A
        # smaller stride yields overlapping patches: a grid of g = (in-patch)/stride+1
        # patches per side, each still a patch_size x patch_size conv (weights are
        # kernel-shaped, so they're identical/loadable regardless of stride).
        self.stride = stride if stride is not None else patch_size
        self.proj = nn.Conv2d(3, embed_dim, kernel_size=patch_size, stride=self.stride)

    def forward(self, x: Tensor) -> tuple[Tensor, int, int]:
        x = self.proj(x)  # [B, D, H, W]
        H, W = x.shape[2], x.shape[3]
        return x.flatten(2).transpose(1, 2), H, W  # [B, H*W, D]


class LayerScale(nn.Module):
    def __init__(self, dim: int, init_values: float) -> None:
        super().__init__()
        self.gamma = nn.Parameter(torch.full((dim,), init_values))

    def forward(self, x: Tensor) -> Tensor:
        return x * self.gamma


class MLP(nn.Module):
    def __init__(self, dim: int, hidden_dim: int) -> None:
        super().__init__()
        self.fc1 = nn.Linear(dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, dim)

    def forward(self, x: Tensor) -> Tensor:
        return self.fc2(F.gelu(self.fc1(x)))


class SelfAttention(nn.Module):
    def __init__(self, dim: int, num_heads: int) -> None:
        super().__init__()
        assert dim % num_heads == 0
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        self.qkv = nn.Linear(dim, dim * 3)
        mask = torch.ones(dim * 3)
        mask[dim : 2 * dim] = 0
        self.register_buffer("_bias_mask", mask, persistent=False)
        self.proj = nn.Linear(dim, dim)

    _bias_mask: Tensor

    def forward(self, x: Tensor, rope: RoPE) -> Tensor:
        B, N, D = x.shape
        assert self.qkv.bias is not None
        qkv = F.linear(x, self.qkv.weight, self.qkv.bias * self._bias_mask).reshape(B, N, 3, self.num_heads, self.head_dim)
        q, k, v = [qkv[:, :, i].transpose(1, 2) for i in range(3)]  # [B, H, N, D_h]
        q = rope_apply_with_prefix(x=q, rope=rope)
        k = rope_apply_with_prefix(x=k, rope=rope)
        out = F.scaled_dot_product_attention(q, k, v, scale=self.scale)
        return self.proj(out.transpose(1, 2).reshape(B, N, D))


class ViTBlock(nn.Module):
    """Pre-norm transformer block.

    Standard (``modulated=False``): LayerNorm-affine + LayerScale residual --
    CanViT's current block, unchanged.

    Modulated (``modulated=True``): no LayerNorm affine and no LayerScale;
    instead ``forward`` takes raw per-token modulation ``mod`` (``[n_tokens,
    6*D]``) and applies adaLN-zero -- ``gamma = 1 + raw``, ``beta = raw``, and a
    residual gate ``alpha = layerscale_init + raw`` -- to the attn and mlp
    branches. With ``mod=None`` (or all-zero raw) this reproduces the standard
    block's init behavior exactly (``x + layerscale_init * sublayer(LN(x))``).

    ``n_experts > 1`` (reViT, see ``canvit.core.backbone.recurrence``): the MLP is an
    ``ExpertMLP`` bank and ``forward`` must get this step's depth ``gate`` (``[E]``).
    With ``n_experts == 1`` (default) the MLP is the plain ``MLP`` and ``gate`` is unused.
    """

    def __init__(
        self,
        dim: int,
        num_heads: int,
        ffn_ratio: float,
        layerscale_init: float,
        *,
        modulated: bool = False,
        n_experts: int = 1,
    ) -> None:
        super().__init__()
        self.modulated = modulated
        self._alpha0 = layerscale_init  # residual-gate identity offset
        self.norm1 = nn.LayerNorm(dim, elementwise_affine=not modulated)
        self.attn = SelfAttention(dim, num_heads)
        self.norm2 = nn.LayerNorm(dim, elementwise_affine=not modulated)
        self.mlp: MLP | ExpertMLP = (
            MLP(dim, int(dim * ffn_ratio)) if n_experts == 1
            else ExpertMLP(dim, int(dim * ffn_ratio), n_experts)
        )
        if not modulated:
            self.ls1 = LayerScale(dim, layerscale_init)
            self.ls2 = LayerScale(dim, layerscale_init)

    def _ffn(self, h: Tensor, gate: Tensor | None) -> Tensor:
        # gate is None exactly when the MLP is a plain MLP -> the original call, unchanged.
        return self.mlp(h) if gate is None else self.mlp(h, gate)

    def forward(
        self, x: Tensor, rope: RoPE, mod: Tensor | None = None, gate: Tensor | None = None,
    ) -> Tensor:
        if not self.modulated:
            x = x + self.ls1(self.attn(self.norm1(x), rope))
            return x + self.ls2(self._ffn(self.norm2(x), gate))
        if mod is None:
            # Identity-at-init regime (raw == 0): gamma=1, beta=0, alpha=alpha0.
            x = x + self._alpha0 * self.attn(self.norm1(x), rope)
            return x + self._alpha0 * self._ffn(self.norm2(x), gate)
        # mod: [n_tokens, 6*D] raw -> per-token (gamma, beta, alpha) per branch.
        g1, b1, a1, g2, b2, a2 = mod.chunk(6, dim=-1)  # each [n_tokens, D]
        h = self.norm1(x)
        x = x + (self._alpha0 + a1) * self.attn((1.0 + g1) * h + b1, rope)
        h = self.norm2(x)
        return x + (self._alpha0 + a2) * self._ffn((1.0 + g2) * h + b2, gate)


# ---------------------------------------------------------------------------
# Backbone
# ---------------------------------------------------------------------------


class ViTBackbone(nn.Module):
    """ViT backbone: patch embedding + transformer blocks.

    ``n_blocks`` is the COMPUTATIONAL depth: how many block evaluations a forward runs
    (``run_block(t, ...)`` for ``t in range(n_blocks)``). Without recurrence (default)
    ``blocks`` has ``n_blocks`` independent entries and step ``t`` runs ``blocks[t]`` —
    the original backbone, unchanged. With ``recurrence.enabled`` (bViT / reViT, see
    ``canvit.core.backbone.recurrence``) ``blocks`` holds only ``n_templates`` shared
    blocks and step ``t`` runs ``blocks[t % n_templates]``, optionally with a reViT
    depth-router gate over an expert FFN bank and a bViT-TE step embedding.
    """

    embed_dim: int
    num_heads: int
    n_blocks: int
    patch_size_px: int
    ffn_ratio: float
    rope_base: float
    layerscale_init: float
    modulated: bool
    recurrence: RecurrenceConfig
    n_templates: int

    def __init__(
        self,
        *,
        embed_dim: int,
        num_heads: int,
        n_blocks: int,
        patch_size: int,
        ffn_ratio: float,
        rope_base: float,
        layerscale_init: float,
        modulated: bool = False,
        patch_stride: int | None = None,
        recurrence: RecurrenceConfig | None = None,
    ) -> None:
        super().__init__()
        assert embed_dim % num_heads == 0
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        # Recurrence (bViT / reViT). Off -> every recurrence attribute below is inert and
        # the module tree / state_dict / forward are exactly the original backbone's.
        self.recurrence = recurrence if recurrence is not None else RecurrenceConfig()
        rec = self.recurrence
        if rec.enabled:
            n_blocks = rec.depth if rec.depth is not None else n_blocks
            assert n_blocks % rec.n_templates == 0, (
                f"recurrence: depth {n_blocks} must be a multiple of n_templates {rec.n_templates}"
            )
        self.n_blocks = n_blocks
        self.n_templates = rec.n_templates if rec.enabled else n_blocks
        n_experts = rec.n_experts if rec.enabled else 1
        self.patch_size_px = patch_size
        # Patch-embed conv stride; defaults to patch_size (non-overlapping). A
        # smaller value makes the uniform patcher's patches overlap. Only the
        # uniform path uses patch_embed, so this is inert for foveated/square.
        self.patch_stride_px = patch_stride if patch_stride is not None else patch_size
        self.ffn_ratio = ffn_ratio
        self.rope_base = rope_base
        self.layerscale_init = layerscale_init
        self.modulated = modulated

        self.patch_embed = PatchEmbed(patch_size, embed_dim, stride=self.patch_stride_px)
        self.blocks = nn.ModuleList([
            ViTBlock(embed_dim, num_heads, ffn_ratio, layerscale_init,
                     modulated=modulated, n_experts=n_experts)
            for _ in range(self.n_templates)
        ])
        # reViT depth router (only with an expert bank) and bViT-TE step embedding. Both
        # attributes are None when unused, so they add no parameters / state_dict keys.
        self.router: DepthRouter | None = None
        if n_experts > 1:
            self.router = DepthRouter(n_experts, rec.router_hidden)
            # depth coordinate s_t = t / (L-1), reViT Eq. 1
            s = torch.arange(n_blocks, dtype=torch.float32) / max(n_blocks - 1, 1)
            self.register_buffer("_depth_coord", s[:, None], persistent=False)
        self.step_embed: nn.Parameter | None = None
        if rec.enabled and rec.step_embed:
            self.step_embed = nn.Parameter(torch.zeros(n_blocks, embed_dim))
        self._init_weights()
        if rec.enabled:
            log.info(
                "Recurrent backbone: depth=%d, n_templates=%d, n_experts=%d, step_embed=%s",
                n_blocks, self.n_templates, n_experts, rec.step_embed,
            )

    _depth_coord: Tensor

    def _init_weights(self) -> None:
        """Match DINOv3 init: trunc_normal_(std=0.02) for Linear weights, zeros for biases."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.trunc_normal_(module.weight, std=0.02)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, ExpertMLP):
                module.reset_parameters()
            elif isinstance(module, LayerScale):
                nn.init.constant_(module.gamma, self.layerscale_init)
            elif isinstance(module, PatchEmbed):
                proj = module.proj
                k = 1.0 / (proj.in_channels * proj.kernel_size[0] * proj.kernel_size[1])
                nn.init.uniform_(proj.weight, -math.sqrt(k), math.sqrt(k))
                if proj.bias is not None:
                    nn.init.uniform_(proj.bias, -math.sqrt(k), math.sqrt(k))
        if self.router is not None:
            # The loop above trunc_normal'd the router's Linears too; restore its own init
            # (default-init fc1, zero fc2 -> uniform gates). See DepthRouter.reset_parameters.
            self.router.reset_parameters()

    @property
    def head_dim(self) -> int:
        return self.embed_dim // self.num_heads

    # ------------------------------------------------------------------ recurrence

    def router_logits(self) -> Tensor | None:
        """reViT router logits ``[n_blocks, E]`` (``None`` without an expert bank)."""
        if self.router is None:
            return None
        return self.router(self._depth_coord)

    def depth_gates(self) -> Tensor | None:
        """reViT per-step expert gates ``[n_blocks, E]`` (fp32, rows sum to 1), or ``None``.
        Image-independent: compute once per forward and pass row ``t`` to ``run_block``."""
        z = self.router_logits()
        return None if z is None else torch.softmax(z.float(), dim=-1)

    def run_block(
        self, t: int, x: Tensor, rope: RoPE, mod: Tensor | None = None,
        gates: Tensor | None = None,
    ) -> Tensor:
        """Computational step ``t`` (``0 <= t < n_blocks``). Without recurrence this is
        exactly ``self.blocks[t](x, rope, mod)``."""
        if self.step_embed is not None:
            x = x + self.step_embed[t].to(x.dtype)
        block = self.blocks[t % self.n_templates]
        return block(x, rope, mod, None if gates is None else gates[t])

    def expert_banks(self) -> list[ExpertMLP]:
        return [b.mlp for b in self.blocks if isinstance(b.mlp, ExpertMLP)]  # type: ignore[union-attr]

    def recurrence_aux_loss(self) -> tuple[Tensor, dict[str, Tensor]] | None:
        """reViT auxiliary loss ``lambda_bal*bal + lambda_z*z + lambda_div*div`` and its
        raw terms, or ``None`` when there is no expert bank (bViT / standard backbone).
        Depends on parameters only, so add it once per optimizer step."""
        z = self.router_logits()
        if z is None:
            return None
        terms = aux_losses(z, self.expert_banks())
        rec = self.recurrence
        total = rec.lambda_bal * terms["bal"] + rec.lambda_z * terms["z"] + rec.lambda_div * terms["div"]
        return total, terms

    @torch.no_grad()
    def recurrence_stats(self) -> dict[str, float]:
        """Logging diagnostics for reViT: ``gate_max`` (max gate over all steps; 1/E =
        still uniform) and ``w_cos_max`` (-> 1.0 = experts collapsed). ``{}`` otherwise."""
        g = self.depth_gates()
        if g is None:
            return {}
        return {"gate_max": g.max().item(), "w_cos_max": expert_weight_cos_max(self.expert_banks())}
