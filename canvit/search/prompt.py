"""Prompt conditioning for search on the curated release: text and visual prompts, no class index.

Owner decisions of 2026-11-26 (``canvit_claude/canvit/search/04-prompt-conditioning-decisions.md``):

- **Prompt vectors** (decision 3): text = the Talk2DINO vector of the trial's text (``text_encoder.py``), visual =
  the release's pre-extracted DINOv3 cell features of another instance (``vp_feat`` [14, 768]: 1x1, then 2x2 and
  3x3 row by row). Every 768-d vector is L2-normalised, then one trainable Linear per prompt type maps it to the
  backbone width (``PromptEncoder``).
- **AdaLN in the backbone** (decision 4): per block, a zero-initialised Linear of the prompt's 1x1 vector gives
  ``(gamma1, beta1, gamma2, beta2)`` applied to the block's two LayerNorm outputs, ``h * (1 + gamma) + beta``.
  Zero at init = the pretrained model unchanged, bit for bit (``1 + 0`` and ``+ 0`` are exact).
- **Prompt tokens** (decision 4): none / 1x1 / 2x2 / 3x3 cell tokens (2x2 = exactly the 4 cells of the 2x2 grid,
  without the 1x1; empty cells, coverage 0, are masked); a text prompt gives one token. INTERPRETATION (to be
  confirmed by the owner): the local stream reads the tokens through a Flamingo-style gated cross-attention before
  every block, ``x + tanh(alpha) * XAttn(LN(x), tokens)`` with ``alpha`` = 0 at init, rather than by appending
  them to the self-attention sequence; only the gate makes "the pretrained model starts unchanged" exact.
  ``token_mode='prefix'`` (owner 2026-11-26): LLaMA-Adapter instead (``PromptPrefix``), the block's own
  self-attention also attends to the tokens through a separate, zero-gated softmax.
- Both attach to ``canvit.backbone.blocks`` through forward hooks (``PromptConditioning.attach``): ``canvit.core`` is
  not edited, the backbone's checkpoint keys stay, and the prompt's parameters live under ``prompt.*`` of the search
  model. The batch's prompt is set as a context (``with conditioning.use(prompt): ...``) around the backbone calls;
  without a context the hooks do nothing.
- **Visual share** (decision 2): ``assign_prompt_types`` decides per trial of an epoch whether it gets its visual
  prompt, and which text-only trials are dropped, so that the epoch's share of visual prompts is ``s``.
"""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Literal

import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor, nn

PROMPT_DIM = 768
"""Width of the prompt features: Talk2DINO's output and DINOv3 ViT-B's patch features."""
N_CELLS = 14
TOKEN_CELLS: dict[str, slice] = {"1x1": slice(0, 1), "2x2": slice(1, 5), "3x3": slice(5, 14)}
"""Which of the 14 visual-prompt cells each token option uses (the 1x1 cell is only ever its own option)."""
TokenOption = Literal["none", "1x1", "2x2", "3x3"]
PROMPT_TYPES = ("text", "visual")


# --- one batch's prompts -----------------------------------------------------------------------------------

@dataclass
class Prompt:
    """One batch's prompts after ``PromptEncoder``, on the model's device.

    ``vec`` [B, D]: the projected 1x1 vector (text vector, or the visual 1x1 cell): AdaLN, head and policy read it.
    ``tokens`` [B, K, D] / ``token_mask`` [B, K] (True = a real token): the prompt tokens; None with tokens='none'.
    ``raw`` [B, 768]: the unprojected L2-normalised 1x1 vector (the zero-shot cosine baseline reads it).
    ``visual`` [B] bool: which rows are visual prompts."""

    vec: Tensor
    tokens: Tensor | None
    token_mask: Tensor | None
    raw: Tensor
    visual: Tensor

    def __getitem__(self, idx: Tensor) -> Prompt:
        return self.rows(idx)

    def rows(self, idx: Tensor) -> Prompt:
        return Prompt(vec=self.vec[idx], tokens=None if self.tokens is None else self.tokens[idx],
                      token_mask=None if self.token_mask is None else self.token_mask[idx], raw=self.raw[idx],
                      visual=self.visual[idx])


@dataclass
class PromptInputs:
    """One batch's prompt INPUTS (on the device): ``visual`` [B] bool, text vectors [B, 768] or None, visual
    cells [B, 14, 768] / coverage [B, 14] or None. ``encode`` runs the (trainable) encoder afresh, so every glimpse
    builds its own graph and chunked BPTT never back-propagates through a graph a second time."""

    visual: Tensor
    text: Tensor | None = None
    vp_feat: Tensor | None = None
    vp_cov: Tensor | None = None

    def encode(self, encoder: PromptEncoder) -> Prompt:
        return encoder(visual=self.visual, text=self.text, vp_feat=self.vp_feat, vp_cov=self.vp_cov)

    def __getitem__(self, idx: Tensor) -> PromptInputs:
        def sub(t):
            return None if t is None else t[idx]
        return PromptInputs(visual=self.visual[idx], text=sub(self.text), vp_feat=sub(self.vp_feat),
                            vp_cov=sub(self.vp_cov))


class PromptEncoder(nn.Module):
    """Text vectors [B, 768] and / or visual cells [B, 14, 768] -> :class:`Prompt` (decision 3: L2-normalise every
    768-d vector, then one trainable Linear per prompt type, shared by the type's cells)."""

    def __init__(self, *, dim: int, tokens: TokenOption, in_dim: int = PROMPT_DIM):
        super().__init__()
        self.tokens = tokens
        self.text = nn.Linear(in_dim, dim)
        self.visual = nn.Linear(in_dim, dim)

    def forward(self, *, visual: Tensor, text: Tensor | None = None, vp_feat: Tensor | None = None,
                vp_cov: Tensor | None = None) -> Prompt:
        """``visual`` [B] bool says which rows use their visual prompt (``vp_feat`` / ``vp_cov`` needed then); the
        others use their text vector (``text`` needed then). Rows of the unused input are ignored."""
        b = visual.shape[0]
        dev = self.text.weight.device
        raw = torch.zeros(b, PROMPT_DIM, device=dev)
        vec = torch.zeros(b, self.text.out_features, device=dev)
        is_t, is_v = ~visual, visual
        if bool(is_t.any()):
            assert text is not None, "text rows need their text vectors"
            t = F.normalize(text[is_t].float(), dim=-1)
            raw[is_t], vec[is_t] = t, self.text(t).float()  # float: autocast may give half
        cells = None
        if bool(is_v.any()):
            assert vp_feat is not None and vp_cov is not None, "visual rows need vp_feat and vp_cov"
            cells = F.normalize(vp_feat[is_v].float(), dim=-1)  # [n_v, 14, 768]; an empty cell (zeros) stays 0
            raw[is_v] = cells[:, 0]
            vec[is_v] = self.visual(cells[:, 0]).float()
        if self.tokens == "none":
            return Prompt(vec=vec, tokens=None, token_mask=None, raw=raw, visual=visual)
        sl = TOKEN_CELLS[self.tokens]
        k = sl.stop - sl.start
        tok = torch.zeros(b, k, vec.shape[1], device=dev)
        mask = torch.zeros(b, k, dtype=torch.bool, device=dev)
        tok[is_t, 0], mask[is_t, 0] = vec[is_t], True  # a text prompt is one token
        if cells is not None:
            assert vp_cov is not None
            cov = vp_cov[is_v].to(dev)[:, sl]
            tv = self.visual(cells[:, sl]).float()  # [n_v, k, D]
            mv = cov > 0
            empty = ~mv.any(dim=1)  # no covered cell at this grid (cannot happen for 1x1): fall back to the 1x1 cell
            if bool(empty.any()):
                tv[empty, 0], mv[empty, 0] = vec[is_v][empty], True
            tok[is_v], mask[is_v] = tv, mv
        return Prompt(vec=vec, tokens=tok, token_mask=mask, raw=raw, visual=visual)


# --- gated conditioning of the backbone ------------------------------------------------------------------------

class GatedPromptXAttn(nn.Module):
    """Flamingo's gated cross-attention: ``x + tanh(alpha) * Attn(q=LN(x), kv=tokens)``, alpha = 0 at init."""

    def __init__(self, *, dim: int, num_heads: int):
        super().__init__()
        self.num_heads = num_heads
        self.norm = nn.LayerNorm(dim)
        self.q, self.k, self.v, self.out = (nn.Linear(dim, dim) for _ in range(4))
        self.alpha = nn.Parameter(torch.zeros(1))

    def forward(self, x: Tensor, tokens: Tensor, mask: Tensor) -> Tensor:
        b, n, d = x.shape
        h = self.num_heads

        def heads(t: Tensor) -> Tensor:
            return t.reshape(b, t.shape[1], h, d // h).transpose(1, 2)

        q = heads(self.q(self.norm(x)))
        k, v = heads(self.k(tokens).to(q.dtype)), heads(self.v(tokens).to(q.dtype))
        att = F.scaled_dot_product_attention(q, k, v, attn_mask=mask[:, None, None, :])
        return x + torch.tanh(self.alpha).to(x.dtype) * self.out(att.transpose(1, 2).reshape(b, n, d)).to(x.dtype)


class PromptPrefix(nn.Module):
    """LLaMA-Adapter-style prompt prefix for ONE backbone block (Zhang et al. 2023, zero-init attention).

    The block's self-attention queries also attend to the prompt tokens, in a SEPARATE softmax, scaled per head by
    a gate initialised at 0 and added to the attention output before its output projection:
    ``proj(Attn(q, k, v) + g * Attn(q, k_p, v_p))``. The tokens are not sequence tokens (nothing updates them).
    ``k_p, v_p`` = the block's OWN (live) k / v projections of ``LN(A tokens)``, with ``A`` a per-block Linear
    (identity at init) and a fresh LayerNorm: LLaMA-Adapter's per-layer prompt through the layer's projections,
    here starting from the shared prompt tokens. No RoPE on the prompt keys (they have no position)."""

    def __init__(self, *, dim: int, num_heads: int):
        super().__init__()
        self.adapt = nn.Linear(dim, dim)
        nn.init.eye_(self.adapt.weight)
        nn.init.zeros_(self.adapt.bias)
        self.norm = nn.LayerNorm(dim)
        self.gate = nn.Parameter(torch.zeros(num_heads))

    def forward(self, attn: nn.Module, x: Tensor, rope, tokens: Tensor, mask: Tensor) -> Tensor:
        """The term to add to ``attn``'s output for its input ``x`` (= ``proj`` of the gated prefix attention,
        without proj's bias). Every row has at least one unmasked token (``PromptEncoder``)."""
        from canvit.core.rope import rope_apply_with_prefix
        b, n, d = x.shape
        h, dh = attn.num_heads, attn.head_dim  # type: ignore[attr-defined]
        w, bias = attn.qkv.weight, attn.qkv.bias  # type: ignore[attr-defined]
        q = F.linear(x, w[:d], bias[:d]).reshape(b, n, h, dh).transpose(1, 2)
        q = rope_apply_with_prefix(x=q, rope=rope)  # the queries exactly as the block's own attention uses them
        t = self.norm(self.adapt(tokens)).to(x.dtype)
        m = t.shape[1]
        k = F.linear(t, w[d:2 * d]).reshape(b, m, h, dh).transpose(1, 2)  # k has no bias in this backbone
        v = F.linear(t, w[2 * d:], bias[2 * d:]).reshape(b, m, h, dh).transpose(1, 2)
        o = F.scaled_dot_product_attention(q, k.to(q.dtype), v.to(q.dtype), attn_mask=mask[:, None, None, :],
                                           scale=attn.scale)  # type: ignore[attr-defined]
        o = o * self.gate.to(o.dtype)[None, :, None, None]
        return F.linear(o.transpose(1, 2).reshape(b, n, d), attn.proj.weight)  # type: ignore[attr-defined]


class PromptConditioning(nn.Module):
    """AdaLN (``adaln``) and gated prompt tokens (``tokens``), per backbone block, attached by forward hooks.
    ``token_mode``: 'xattn' = Flamingo gated cross-attention before the block (``GatedPromptXAttn``), 'prefix' =
    LLaMA-Adapter prefix inside the block's self-attention (``PromptPrefix``)."""

    def __init__(self, *, dim: int, num_heads: int, n_blocks: int, adaln: bool, tokens: TokenOption,
                 token_mode: Literal["xattn", "prefix"] = "xattn"):
        super().__init__()
        if token_mode not in ("xattn", "prefix"):
            raise ValueError(f"token_mode {token_mode!r}: 'xattn' or 'prefix'")
        self.encoder = PromptEncoder(dim=dim, tokens=tokens)
        self.adaln = nn.ModuleList([nn.Linear(dim, 4 * dim) for _ in range(n_blocks)]) if adaln else None
        if self.adaln is not None:
            for lin in self.adaln:  # zero at init: gamma = beta = 0
                nn.init.zeros_(lin.weight)
                nn.init.zeros_(lin.bias)
        self.xattn = (nn.ModuleList([GatedPromptXAttn(dim=dim, num_heads=num_heads) for _ in range(n_blocks)])
                      if tokens != "none" and token_mode == "xattn" else None)
        self.prefix = (nn.ModuleList([PromptPrefix(dim=dim, num_heads=num_heads) for _ in range(n_blocks)])
                       if tokens != "none" and token_mode == "prefix" else None)
        self._ctx: Prompt | None = None
        self._handles: list = []

    @contextmanager
    def use(self, prompt: Prompt | None) -> Iterator[None]:
        """The backbone is conditioned on ``prompt`` inside the block (None: unconditioned)."""
        prev = self._ctx
        self._ctx = prompt
        try:
            yield
        finally:
            self._ctx = prev

    def attach(self, backbone: nn.Module) -> None:
        """Register the hooks on ``backbone.blocks`` (idempotent)."""
        if self._handles:
            return
        for i, blk in enumerate(backbone.blocks):  # type: ignore[attr-defined]
            if self.adaln is not None:
                self._handles.append(blk.norm1.register_forward_hook(self._adaln_hook(i, 0)))
                self._handles.append(blk.norm2.register_forward_hook(self._adaln_hook(i, 1)))
            if self.xattn is not None:
                self._handles.append(blk.register_forward_pre_hook(self._xattn_hook(i)))
            if self.prefix is not None:
                self._handles.append(blk.attn.register_forward_hook(self._prefix_hook(i)))

    def _adaln_hook(self, i: int, which: int):
        def hook(_module, _inputs, out: Tensor):
            if self._ctx is None:
                return out
            assert self.adaln is not None
            g, b = self.adaln[i](self._ctx.vec)[:, None, :].chunk(4, dim=-1)[2 * which: 2 * which + 2]
            return out * (1.0 + g.to(out.dtype)) + b.to(out.dtype)
        return hook

    def _xattn_hook(self, i: int):
        def hook(_module, args):
            ctx = self._ctx
            if ctx is None or ctx.tokens is None:
                return None
            assert self.xattn is not None and ctx.token_mask is not None
            return (self.xattn[i](args[0], ctx.tokens, ctx.token_mask), *args[1:])
        return hook


    def _prefix_hook(self, i: int):
        def hook(attn, args, out: Tensor):
            ctx = self._ctx
            if ctx is None or ctx.tokens is None:
                return None
            assert self.prefix is not None and ctx.token_mask is not None
            x, rope = args[0], args[1]  # SelfAttention.forward(x, rope)
            return out + self.prefix[i](attn, x, rope, ctx.tokens, ctx.token_mask).to(out.dtype)
        return hook


# --- visual share (decision 2) ---------------------------------------------------------------------------------

@dataclass
class ShareResult:
    keep: np.ndarray      # [n] bool: the trial stays in the epoch
    visual: np.ndarray    # [n] bool: the trial uses its visual prompt (only kept trials with a prompt)
    f: float              # fraction of the epoch's trials with a visual prompt
    p_keep_text: float    # probability that a text-only trial is kept


def assign_prompt_types(has_vp: np.ndarray, share: float, *, seed: int, epoch: int) -> ShareResult:
    """Per trial of an epoch (in epoch order): visual or text prompt, and whether it is dropped (decision 2, rule
    of the get_datasets session 2026-11-26). ``f`` = fraction with a visual prompt.

    - ``share <= f``: nothing dropped; a trial with a prompt gets it with probability share / f, all others text.
    - ``share > f``: every trial with a prompt gets it; a text-only trial is kept with probability
      ``f (1 - share) / (share (1 - f))``, so the expected share is ``share`` (0 at share 1: all text-only trials go).

    One uniform draw per trial from ``default_rng([seed, epoch, 2])``: a pure function of (seed, epoch, position),
    independent of batching, ranks and resume."""
    if not 0.0 <= share <= 1.0:
        raise ValueError(f"visual share {share} outside [0, 1]")
    has_vp = np.asarray(has_vp, dtype=bool)
    n = len(has_vp)
    u = np.random.default_rng([seed, epoch, 2]).random(n)
    f = float(has_vp.mean()) if n else 0.0
    if share > 0 and f == 0:
        raise ValueError("a visual share > 0 needs trials with visual prompts (a release with visual prompts)")
    if share <= f:
        visual = has_vp & (u < (share / f if f > 0 else 0.0))
        return ShareResult(keep=np.ones(n, bool), visual=visual, f=f, p_keep_text=1.0)
    p = f * (1 - share) / (share * (1 - f)) if f < 1 else 1.0
    keep = has_vp | (u < p)
    return ShareResult(keep=keep, visual=has_vp.copy(), f=f, p_keep_text=p)


__all__ = ["N_CELLS", "PROMPT_DIM", "PROMPT_TYPES", "TOKEN_CELLS", "GatedPromptXAttn", "Prompt",
           "PromptConditioning", "PromptEncoder", "PromptInputs", "ShareResult", "TokenOption", "assign_prompt_types"]
