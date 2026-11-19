"""Single-block recurrence for the CanViT backbone: bViT and reViT, behind one switch.

Two papers, one config (:class:`RecurrenceConfig`). ``enabled=False`` (the default) builds
the standard backbone of independent blocks, bit-identical to before this module existed.

* **bViT** — Byra et al., *Investigating Single-Block Recurrence in Vision Transformers for
  Image Recognition*, arXiv:2605.10661. ONE transformer block applied ``depth`` times.
  ``n_experts=1``.
* **reViT** — Bulat et al., *One Block, Multiple Depths: Recurrent Vision Transformers with
  Depth-Programmed Experts*, arXiv:2610.12448. The same shared block, but its FFN is a bank
  of ``E`` experts whose WEIGHTS are merged per step (Eq. 1-3)::

      s_t = t / (L - 1),   g_t = softmax(psi(s_t)),   W_t = sum_e g_t,e * W_e

  and only the merged FFN touches the tokens. The gate depends on depth alone (the same
  for every image and token), so this is not token routing. Trained with three auxiliary
  losses (Eq. 6, 8, 9) that depend on parameters only: :func:`aux_losses`.

Ported from the standalone ImageNet implementation in ``recurrent-vit`` (``rvit/model.py``),
which reproduces bViT's ViT-S / bViT-S ImageNet numbers.

How it maps onto CanViT (see ``ViTBackbone``):

* ``backbone.n_blocks`` stays the COMPUTATIONAL depth (number of block evaluations), so the
  canvas read/write schedule (``compute_rw_positions``), the per-block token modulation and
  the forward loop all keep their meaning.
* ``backbone.blocks`` holds only ``n_templates`` parameterised blocks; step ``t`` runs
  ``blocks[t % n_templates]`` (bViT Table 3 "(2B,6S)" = 2 blocks looped 6 times).
* Canvas read/write cross-attention is NOT shared: every read/write position keeps its own
  weights, exactly as in the non-recurrent model. Only the ViT trunk recurs.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import Tensor, nn


@dataclass
class RecurrenceConfig:
    """bViT / reViT recurrence of the CanViT backbone. Off by default."""

    enabled: bool = False
    """Master switch. False (default) -> standard backbone (``n_blocks`` independent blocks),
    every other field ignored, bit-identical to a model built before this option existed."""
    depth: int | None = None
    """Computational depth L = number of block evaluations per glimpse. ``None`` -> the
    registry depth of the backbone (12 for vits16 / vitb16, 24 for vitl16)."""
    n_templates: int = 1
    """Parameter depth: number of independently parameterised blocks; step ``t`` runs
    template ``t % n_templates``. 1 = full sharing (bViT / reViT). Must divide ``depth``."""
    n_experts: int = 1
    """1 = bViT (plain FFN). >1 = reViT: an expert bank of this size in every template,
    merged per step by a depth router. The reViT paper's default is 4."""
    router_hidden: int = 16
    """Hidden width of the reViT depth router psi (paper App. A.1: 16)."""
    step_embed: bool = False
    """bViT-TE (bViT App. B, Eq. 8): add a learned, zero-initialised per-step vector to every
    token before each step, so the shared block can tell the steps apart."""
    lambda_bal: float = 1e-2
    """reViT balance loss weight (Eq. 6). Training-only; ignored when ``n_experts == 1``."""
    lambda_z: float = 1e-3
    """reViT router z-loss weight (Eq. 8)."""
    lambda_div: float = 1e-3
    """reViT expert-diversity loss weight (Eq. 9)."""

    def __post_init__(self) -> None:
        assert self.n_templates >= 1, f"n_templates must be >= 1, got {self.n_templates}"
        assert self.n_experts >= 1, f"n_experts must be >= 1, got {self.n_experts}"
        assert self.depth is None or self.depth >= 1, f"depth must be >= 1, got {self.depth}"
        assert self.router_hidden >= 1


class ExpertMLP(nn.Module):
    """GELU MLP stored as a bank of ``n_experts`` experts (reViT Eq. 2-3).

    Given a gate ``g`` (``[E]``, sums to 1) the expert weights AND biases are merged first;
    only the merged MLP is applied to the tokens, so the cost per token equals one MLP.
    Same math as ``canvit.core.backbone.vit.MLP`` (``fc2(gelu(fc1(x)))``).
    """

    def __init__(self, dim: int, hidden_dim: int, n_experts: int) -> None:
        super().__init__()
        self.n_experts = n_experts
        self.w1 = nn.Parameter(torch.empty(n_experts, hidden_dim, dim))
        self.b1 = nn.Parameter(torch.empty(n_experts, hidden_dim))
        self.w2 = nn.Parameter(torch.empty(n_experts, dim, hidden_dim))
        self.b2 = nn.Parameter(torch.empty(n_experts, dim))
        self.reset_parameters()

    def reset_parameters(self) -> None:
        """trunc_normal(0.02) per expert, zero biases — CanViT's / DINOv3's Linear init."""
        nn.init.trunc_normal_(self.w1, std=0.02)
        nn.init.trunc_normal_(self.w2, std=0.02)
        nn.init.zeros_(self.b1)
        nn.init.zeros_(self.b2)

    def forward(self, x: Tensor, gate: Tensor | None) -> Tensor:
        assert gate is not None, "ExpertMLP needs a depth gate (reViT); got None"
        g = gate.to(self.w1.dtype)
        w1 = torch.einsum("e,eij->ij", g, self.w1)
        b1 = g @ self.b1
        w2 = torch.einsum("e,eij->ij", g, self.w2)
        b2 = g @ self.b2
        return F.linear(F.gelu(F.linear(x, w1, b1)), w2, b2)

    def flat_experts(self) -> Tensor:
        """``[E, P]``: each expert's full parameter vector q_e (reViT Eq. 9)."""
        return torch.cat([self.w1.flatten(1), self.b1, self.w2.flatten(1), self.b2], dim=1)


class DepthRouter(nn.Module):
    """psi: R -> R^E. 2-layer MLP (SiLU), zero-initialised output => uniform gates at init."""

    def __init__(self, n_experts: int, hidden: int) -> None:
        super().__init__()
        self.fc1 = nn.Linear(1, hidden)
        self.fc2 = nn.Linear(hidden, n_experts)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        # fc1 keeps PyTorch's default init ON PURPOSE (as in rvit): the backbone-wide
        # trunc_normal(0.02) + zero-bias init would make silu(fc1(s)) ~ 0 for every s, i.e.
        # the router would see almost no depth signal and learn very slowly.
        self.fc1.reset_parameters()
        nn.init.zeros_(self.fc2.weight)
        nn.init.zeros_(self.fc2.bias)

    def forward(self, s: Tensor) -> Tensor:
        """``s``: ``[L, 1]`` depth coordinates -> logits ``[L, E]``."""
        return self.fc2(F.silu(self.fc1(s)))


def aux_losses(logits: Tensor, expert_banks: list[ExpertMLP]) -> dict[str, Tensor]:
    """reViT auxiliary losses, in fp32 (Eq. 6, 8, 9). Depend on parameters only, not data.

    * ``bal``: 1 - H(g_t)/log E, averaged over steps (Eq. 6) — keeps gates from going
      one-hot too early.
    * ``z``: (logsumexp z_t)^2 averaged over steps (Eq. 8) — keeps router logits small.
    * ``div``: mean squared pairwise cosine between experts' flattened parameters (Eq. 9),
      averaged over templates — keeps experts from collapsing into copies.
    """
    E = expert_banks[0].n_experts
    with torch.autocast(device_type=logits.device.type, enabled=False):
        z = logits.float()
        g = torch.softmax(z, dim=-1)
        ent = -(g * torch.log(g.clamp_min(1e-12))).sum(-1)
        bal = (1.0 - ent / math.log(E)).mean()
        zl = torch.logsumexp(z, dim=-1).pow(2).mean()
        iu = torch.triu_indices(E, E, offset=1, device=z.device)
        div = z.new_zeros(())
        for bank in expert_banks:
            q = F.normalize(bank.flat_experts().float(), dim=1)
            div = div + (q @ q.t())[iu[0], iu[1]].pow(2).mean()
        div = div / len(expert_banks)
    return {"bal": bal, "z": zl, "div": div}


@torch.no_grad()
def expert_weight_cos_max(expert_banks: list[ExpertMLP]) -> float:
    """Max pairwise cosine between experts' WEIGHT matrices (w1, w2; biases excluded),
    averaged over templates. Diagnostic: -> 1.0 means the experts collapsed into copies
    (then the router gets no gradient). ``div`` includes the biases and can look healthy
    while the weights are identical — seen in rvit's reViT-S deit3 run (2026-11-18)."""
    vals = []
    for bank in expert_banks:
        q = F.normalize(torch.cat([bank.w1.flatten(1), bank.w2.flatten(1)], 1).float(), dim=1)
        c = q @ q.t()
        vals.append(c[~torch.eye(len(c), dtype=torch.bool, device=c.device)].max())
    return torch.stack(vals).mean().item()


__all__ = [
    "DepthRouter", "ExpertMLP", "RecurrenceConfig", "aux_losses", "expert_weight_cos_max",
]
