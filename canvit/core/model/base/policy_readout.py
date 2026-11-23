"""Readout of the policy tokens: K tokens x H heads -> one distribution over the canvas cells.

Design note: canvit_claude/canvit/active_pretraining/02-policy-tokens-design.md (owner-approved
2026-11-22). Per token k and head h, a score for every canvas cell a:

    s[k,h,a] = q[k,h] . LN(canvas_a)[h] / sqrt(head_dim) + bias[k,h,a]

with q = W_q LN(token) projected up to the canvas width (the canvas side is not projected, as
in the canvas read attention), split into H heads. Each (k,h) gives a map softmax_a(s[k,h]);
the K*H maps are averaged AS PROBABILITIES with weights softmax over the K*H scores
W_w LN(token) (each token gives H of them). W_w and the bias start at zero, so at init every
map counts equally and there is no location preference. Computed in fp32 (bf16 ties the top
scores; FoAViS J.4). Returns log-probabilities, so ``log_softmax`` of the output is the output
itself and the existing PG loss / entropy code can consume it unchanged.
"""

import math

import torch
import torch.nn.functional as F
from torch import Tensor, nn


class PolicyTokenReadout(nn.Module):
    def __init__(self, *, local_dim: int, canvas_dim: int, n_tokens: int, n_heads: int, grid: int):
        super().__init__()
        assert canvas_dim % n_heads == 0
        self.n_tokens, self.n_heads, self.grid = n_tokens, n_heads, grid
        self.head_dim = canvas_dim // n_heads
        self.token_norm = nn.LayerNorm(local_dim)
        self.canvas_norm = nn.LayerNorm(canvas_dim)
        self.q_proj = nn.Linear(local_dim, canvas_dim)
        self.weight_proj = nn.Linear(local_dim, n_heads)
        nn.init.zeros_(self.weight_proj.weight)
        nn.init.zeros_(self.weight_proj.bias)
        self.bias = nn.Parameter(torch.zeros(n_tokens, n_heads, grid * grid))

    def forward(self, tokens: Tensor, canvas_spatial: Tensor) -> Tensor:
        """tokens [B, K, local_dim], canvas_spatial [B, grid^2, canvas_dim] -> log p [B, grid^2]."""
        B, K, _ = tokens.shape
        N = canvas_spatial.shape[1]
        assert K == self.n_tokens and N == self.grid ** 2, (K, N, self.n_tokens, self.grid)
        with torch.autocast(device_type=tokens.device.type, enabled=False):
            t = self.token_norm(tokens.float())
            q = self.q_proj(t).view(B, K, self.n_heads, self.head_dim)
            k = self.canvas_norm(canvas_spatial.float()).view(B, N, self.n_heads, self.head_dim)
            scores = torch.einsum("bkhd,bnhd->bkhn", q, k) / math.sqrt(self.head_dim) + self.bias
            log_maps = F.log_softmax(scores, dim=-1)                       # [B, K, H, N]
            log_w = F.log_softmax(self.weight_proj(t).reshape(B, K * self.n_heads), dim=-1)
            return torch.logsumexp(log_maps + log_w.view(B, K, self.n_heads, 1), dim=(1, 2))
