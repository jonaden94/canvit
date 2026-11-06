"""Search targets, loss and the hit metric, on the canvas grid.

All coordinates are scene-normalised [-1, 1]; boxes are xyxy ``[B, N, 4]`` with NaN rows
as padding (``data.collate_search``). Cell (i, j) of a G x G grid has its centre at
``((j + 0.5) / G * 2 - 1, (i + 0.5) / G * 2 - 1)`` -- the same cell centres as the
policy's fixation candidates (``core.policy.net.fixation_candidates``).
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import Tensor


def cell_centres(grid: int, device: torch.device) -> Tensor:
    """[G] cell-centre coordinates along one axis in [-1, 1]."""
    return (torch.arange(grid, device=device, dtype=torch.float32) + 0.5) * (2.0 / grid) - 1.0


def box_centres(boxes: Tensor) -> tuple[Tensor, Tensor]:
    """``boxes`` [B, N, 4] xyxy -> (centres [B, N, 2] as (x, y), valid [B, N])."""
    valid = ~torch.isnan(boxes).any(-1)
    b = torch.nan_to_num(boxes)
    return torch.stack([(b[..., 0] + b[..., 2]) / 2, (b[..., 1] + b[..., 3]) / 2], dim=-1), valid


def target_map(boxes: Tensor, *, grid: int, kind: str, sigma: float) -> Tensor:
    """[B, G*G] target distribution over the canvas cells.

    ``kl_gaussian`` (Haoze, ``compute_search_loss.build_target_map``): one Gaussian per box
    centre, each normalised over the grid, summed and renormalised. ``sigma`` is a fraction
    of the scene side, so it is ``2 * sigma`` in [-1, 1] units. ``nll_point``: all mass on
    the cell holding each box centre (uniform over the boxes' cells) -- the sigma -> 0 limit.

    A sample without a box (target absent) gets an all-zero row, so its KL is exactly 0;
    the task masks those rows out of the heatmap loss and the policy reward anyway.
    """
    centres, valid = box_centres(boxes)  # [B, N, 2], [B, N]
    c = cell_centres(grid, boxes.device)
    if kind == "kl_gaussian":
        # Normalised in LOG space (a softmax over the grid): the same distribution as
        # exp(-d^2 / 2s^2) / sum, but a small sigma cannot underflow every cell to 0 (which
        # gave an all-zero target and a silent zero loss) -- it tends to the nearest cell.
        s = 2.0 * sigma
        dx2 = (c[None, None, :] - centres[..., 0:1]) ** 2  # [B, N, G]
        dy2 = (c[None, None, :] - centres[..., 1:2]) ** 2
        logg = -(dy2[..., :, None] + dx2[..., None, :]) / (2 * s * s)  # [B, N, G(y), G(x)]
        g = torch.softmax(logg.flatten(2), dim=-1).view_as(logg)
    elif kind == "nll_point":
        ix = ((centres[..., 0] + 1) / 2 * grid).floor().clamp(0, grid - 1).long()
        iy = ((centres[..., 1] + 1) / 2 * grid).floor().clamp(0, grid - 1).long()
        g = torch.zeros(*centres.shape[:2], grid, grid, device=boxes.device)
        g.view(*centres.shape[:2], -1).scatter_(-1, (iy * grid + ix)[..., None], 1.0)
    else:
        raise ValueError(f"unknown search loss {kind!r}")
    g = (g * valid[..., None, None]).sum(1)  # [B, G, G]
    return (g / g.flatten(1).sum(-1).clamp_min(1e-30)[:, None, None]).flatten(1)


def per_image_loss(logits: Tensor, target: Tensor) -> Tensor:
    """KL(target || softmax(logits)) per image [B]. ``logits`` [B, G, G] or [B, G*G],
    ``target`` [B, G*G]. For a one-hot target this is exactly the NLL of that cell."""
    logp = F.log_softmax(logits.float().flatten(1), dim=-1)
    return F.kl_div(logp, target, reduction="none").sum(-1)


def presence_loss(logit: Tensor, present: Tensor, *, kind: str, gamma: float) -> Tensor:
    """Per-sample loss [B] of the present/absent logit (log p(present) - log p(absent)).

    ``ce``: binary cross-entropy, identical to Haoze's 2-way cross-entropy. ``focal``: his
    ``focal_loss`` with alpha left out (it is a uniform weight; see ``presence_weight``):
    ``(1 - p_t) ** gamma * CE``, with p_t the probability of the true class."""
    ce = F.binary_cross_entropy_with_logits(logit.float(), present.float(), reduction="none")
    if kind == "ce":
        return ce
    if kind == "focal":
        return (1.0 - torch.exp(-ce)) ** gamma * ce
    raise ValueError(f"unknown presence loss {kind!r}")


def argmax_centres(logits: Tensor) -> Tensor:
    """[B, G, G] logits -> the argmax cell's centre [B, 2] as (x, y) in [-1, 1]."""
    b, g, _ = logits.shape
    idx = logits.flatten(1).argmax(-1)
    c = cell_centres(g, logits.device)
    return torch.stack([c[idx % g], c[idx // g]], dim=-1)


def percent_label(box_frac: float) -> str:
    """The metric-name suffix of the lenient criterion: 0.1 -> ``10_percent``."""
    return f"{box_frac * 100:g}_percent"


def inside_box(points: Tensor, boxes: Tensor) -> Tensor:
    """[B] bool: does each point (x, y) lie inside any of its target boxes (edges count)?
    The strict criterion (``found`` / ``hit``). Rows whose boxes are all NaN (target
    absent) are never inside."""
    x, y = points[:, 0, None], points[:, 1, None]
    valid = ~torch.isnan(boxes).any(-1)
    b = torch.nan_to_num(boxes)
    inside = (x >= b[..., 0]) & (x <= b[..., 2]) & (y >= b[..., 1]) & (y <= b[..., 3])
    return (inside & valid).any(-1)


def square_overlaps_box(points: Tensor, boxes: Tensor, *, box_frac: float) -> Tensor:
    """[B] bool: does a square of side ``box_frac`` x scene centred on each point (x, y)
    overlap any of its target boxes with positive area? The lenient criterion
    (``found_X_percent`` / ``hit_X_percent``): Haoze's ``check_box_intersection`` with
    threshold 0, i.e. GVSM's fixation box (64 px at 640 = 0.1). Rows whose boxes are all
    NaN (target absent) never overlap."""
    assert box_frac > 0, "a zero-size square never overlaps with positive area; use inside_box"
    half = box_frac  # side box_frac * 2 in [-1, 1] units -> half-side box_frac
    px, py = points[:, 0, None], points[:, 1, None]
    valid = ~torch.isnan(boxes).any(-1)
    b = torch.nan_to_num(boxes)
    ix = (torch.minimum(px + half, b[..., 2]) - torch.maximum(px - half, b[..., 0])).clamp_min(0)
    iy = (torch.minimum(py + half, b[..., 3]) - torch.maximum(py - half, b[..., 1])).clamp_min(0)
    return ((ix * iy > 0) & valid).any(-1)
