"""Square (axis-aligned, uniform-resolution) sampling patterns.

Three ways to build a frozen, axis-aligned ``m x m`` (or ``patch_size x
patch_size``) per-patch sampling pattern in the visual field, all returning the
same :class:`SquarePattern`:

  - :func:`build_fovi_square`             — squarified approximation of a fovi
    ``KNNPartitioningPatchEmbedding`` partition (one square per fovi patch,
    matched in count / extent / resolution).
  - :func:`build_fovi_square_regularized` — same, but outer rings are snapped to
    an integer image-pixel lattice (less periphery oversampling; foveal pattern
    preserved).
  - :func:`build_strided_square`          — standalone strided-square foveation
    (fovea + decreasing-resolution peripheral rings), no fovi dependency.

All patterns are defined **once at fixation ``(0.5, 0.5)``** and expressed in
fixation-window-normalized visual-field cartesian ``(x, y)`` coordinates
(``[-1, 1]`` = the foveation window). They can be re-placed at any fixation /
fixation size downstream via :func:`canvit.core.foveation.coords.transform_sampling_grid`.

The prototype/visualization code these were copied from lives in the fovi repo under
``notebooks/fovi_square_patches/`` (``square_patches.py``,
``square_patches_regularized.py``, ``square_patches_non_fovi.py``) and is left
untouched — this module is the standalone, dependency-light source of truth for
downstream (e.g. CanViT) use.

Conventions:
  - Visual-field coords are cartesian ``(x, y)`` in ``[-1, 1]`` (math, y-up).
  - ``positions_xy`` is ``[P, K, 2]`` (per patch, ``K`` samples, ordered by an
    ``indexing="ij"`` meshgrid: dim of the ``m x m`` grid 0 = x left->right,
    1 = y bottom->top).
  - ``pad_mask`` is ``[P, K]`` bool, ``True`` = the sample is out of field (reads
    zero / learned padding downstream). For the fovi builders this marks the
    outermost-ring padding strips (matching fovi's KNN padding); for the strided
    builder it marks samples outside the foveation window.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn.functional as F

from canvit.core.foveation.coords import transform_sampling_grid, xy_to_colrow


# =====================================================================
# Return type
# =====================================================================
@dataclass
class SquarePattern:
    """A frozen square sampling pattern in fixation-normalized visual field.

    Attributes:
        positions_xy: ``[P, K, 2]`` visual-field cartesian (x, y), frozen at
            fixation ``(0.5, 0.5)``. ``K = side * side``.
        pad_mask: ``[P, K]`` bool, ``True`` = out-of-field sample (padding).
        centers_xy: ``[P, 2]`` per-patch center (x, y) in the visual field.
        side: side length of the per-patch square grid (``m`` for the fovi
            builders, ``patch_size`` for the strided builder); ``K = side**2``.
        ring_idx: ``[P]`` concentric-ring index per patch (0 = innermost), or
            ``None`` when not applicable.
    """

    positions_xy: torch.Tensor
    pad_mask: torch.Tensor
    centers_xy: torch.Tensor
    side: int
    ring_idx: torch.Tensor | None = None

    @property
    def n_patches(self) -> int:
        return int(self.positions_xy.shape[0])

    @property
    def k(self) -> int:
        return int(self.positions_xy.shape[1])

    def subset(self, keep_mask: torch.Tensor) -> "SquarePattern":
        """Return a new pattern keeping only the patches selected by ``keep_mask``.

        ``keep_mask`` is a boolean tensor of length ``n_patches`` (or a long
        index tensor). ``side`` / ``K`` are per-patch constants and unchanged;
        only the patch (``P``) axis of ``positions_xy`` / ``pad_mask`` /
        ``centers_xy`` / ``ring_idx`` is subset.
        """
        idx = keep_mask
        return SquarePattern(
            positions_xy=self.positions_xy[idx].contiguous(),
            pad_mask=self.pad_mask[idx].contiguous(),
            centers_xy=self.centers_xy[idx].contiguous(),
            side=self.side,
            ring_idx=None if self.ring_idx is None else self.ring_idx[idx].contiguous(),
        )


# =====================================================================
# fovi-derived helpers (from notebooks/.../square_patches.py)
# =====================================================================
def _all_member_coords(pe) -> torch.Tensor:
    """``[N + N_pad, 2]`` visual-field cartesian (x, y): real then padded coords."""
    return torch.cat(
        [pe.in_coords.cartesian, pe.in_coords.cartesian_pad_coords], dim=0
    ).to(torch.float32)


def _patch_member_radii(pe) -> torch.Tensor:
    """Per-patch eccentricities of its ``k`` members (padded coords included). ``[P, k]``."""
    coords = _all_member_coords(pe)            # [Ntot, 2]
    knn = pe.knn_indices.long()                # [k, P]
    member_xy = coords[knn]                    # [k, P, 2]
    radii = member_xy.norm(dim=-1)             # [k, P]
    return radii.transpose(0, 1).contiguous()  # [P, k]


def ring_index_per_patch(pe) -> torch.Tensor:
    """Concentric-ring index per patch (0 = innermost), from center eccentricity."""
    radii = pe.out_coords.polar[:, 0].contiguous()
    unique = torch.unique(radii)
    return torch.searchsorted(unique, radii)


def _patch_edge_lengths(pe) -> torch.Tensor:
    """Per-patch axis-aligned square edge length via the radial heuristic. ``[P]``."""
    r_c = pe.out_coords.polar[:, 0].to(torch.float32)   # [P]
    radii = _patch_member_radii(pe)                      # [P, k]
    r_max = radii.max(dim=1).values
    r_min = radii.min(dim=1).values
    half = 0.5 * ((r_c - r_max).abs() + (r_c - r_min).abs())
    return 2.0 * half                                    # [P]


def compute_ring_edge_lengths(pe, rel_tol: float = 0.05, strict: bool = False) -> torch.Tensor:
    """One edge length per ring (the per-ring mean), broadcast to every patch in it.

    Patches in a ring are (near-)rotations of one another and the radial heuristic
    is rotation-invariant, so the per-patch edge should be ~constant within a ring.
    Prints a loud warning (or raises if ``strict``) when the within-ring
    ``(max-min)/mean`` exceeds ``rel_tol``. Returns ``[P]``.
    """
    edge = _patch_edge_lengths(pe)
    ring_idx = ring_index_per_patch(pe)
    radii_unique = torch.unique(pe.out_coords.polar[:, 0])
    edge_per_patch = torch.empty_like(edge)
    for ri in range(len(radii_unique)):
        mask = ring_idx == ri
        e = edge[mask]
        mean = float(e.mean())
        rel_spread = float(e.max() - e.min()) / max(mean, 1e-12)
        if rel_spread > rel_tol:
            details = (
                f"RING {ri}: REL_SPREAD={rel_spread:.4g} > REL_TOL={rel_tol} "
                f"(R={float(radii_unique[ri]):.4f}, N={int(mask.sum())}, "
                f"MIN={float(e.min()):.5f}, MAX={float(e.max()):.5f}, MEAN={mean:.5f})"
            )
            if strict:
                raise AssertionError(
                    f"ring {ri}: edge length not constant within ring — {details}."
                )
            bar = "!" * 80
            print(bar)
            print("!!! WARNING: EDGE LENGTH NOT CONSTANT WITHIN RING !!!")
            print(f"!!! {details}")
            print("!!! PROCEEDING WITH RING MEAN.")
            print(bar)
        edge_per_patch[mask] = mean
    return edge_per_patch


def global_m(pe) -> int:
    """Side length ``m`` of the square sample grid: ``round(sqrt(k))``."""
    k = int(pe._k)
    return int(round(k ** 0.5))


def build_square_positions(pe, edge_per_patch: torch.Tensor, m: int) -> torch.Tensor:
    """Axis-aligned ``m x m`` sample positions per patch, in the visual field. ``[P, m*m, 2]``."""
    centers = pe.out_coords.cartesian.to(torch.float32)              # [P, 2]
    offs = torch.linspace(-0.5, 0.5, m, dtype=torch.float32)         # [m]
    gx, gy = torch.meshgrid(offs, offs, indexing="ij")
    base = torch.stack([gx.reshape(-1), gy.reshape(-1)], dim=-1)     # [m*m, 2] in [-0.5, 0.5]^2
    return centers[:, None, :] + edge_per_patch[:, None, None] * base[None, :, :]


# =====================================================================
# Outermost-ring directional padding (from square_patches_regularized.py)
# =====================================================================
def _sector(cx: float, cy: float) -> int:
    """Map patch center (vf x, y) to one of 8 compass sectors (0=right, CCW)."""
    angle_deg = math.atan2(cy, cx) * 180.0 / math.pi  # (-180, 180]
    return int(math.floor((angle_deg + 22.5) / 45.0)) % 8


def _directional_mask(m: int, sector: int, n: int) -> torch.Tensor:
    """Boolean mask [m*m] where True = pad, for a patch in the given sector.

    ``n`` outermost strips are masked on each active side. Grid indexed as
    ``positions_xy.reshape(m, m, 2)``: dim 0 = x (left->right), dim 1 = y
    (bottom->top). Cardinal sectors mask one side; diagonal sectors mask two.
    """
    i = torch.arange(m).unsqueeze(1).expand(m, m)  # x-index, [m, m]
    j = torch.arange(m).unsqueeze(0).expand(m, m)  # y-index, [m, m]
    right = i >= m - n
    top = j >= m - n
    left = i < n
    bottom = j < n
    sides = {
        0: right,
        1: right | top,
        2: top,
        3: top | left,
        4: left,
        5: left | bottom,
        6: bottom,
        7: bottom | right,
    }
    return sides[sector].reshape(-1)  # [m*m], True = masked


def _outermost_pad_mask(
    positions_xy: torch.Tensor, ring_idx: torch.Tensor, m: int, pe,
) -> torch.Tensor:
    """Directional pad mask ``[P, m*m]`` for outermost-ring square patches.

    ``n_pad_rings`` is the number of distinct virtual padding-ring radii that
    actually appear as KNN neighbors of the outermost patches. fovi's
    ``cartesian_pad_coords`` contains more virtual rings than are used; only the
    nearest ones fill KNN slots, so we count exactly those. Each outermost patch
    then gets its ``n_pad_rings`` strips on the side facing away from the fovea
    masked; inner-ring patches get an all-False row.
    """
    vf = positions_xy
    P = vf.shape[0]
    max_ring = int(ring_idx.max().item())
    outer_ids = (ring_idx == max_ring).nonzero(as_tuple=True)[0]

    # Count distinct pad-coord radii that actually appear as KNN neighbors of the
    # outermost patches.
    n_real = pe.in_coords.cartesian.shape[0]
    pad_coords = pe.in_coords.cartesian_pad_coords
    pad_coords = pad_coords if torch.is_tensor(pad_coords) else torch.as_tensor(pad_coords)
    knn_idx = pe.knn_indices                          # [k, n_out]
    knn_pad = pe.knn_indices_pad_mask                 # [k, n_out], True=pad
    outer_knn_idx = knn_idx[:, outer_ids]
    outer_knn_pad = knn_pad[:, outer_ids]
    pad_slot_ids = outer_knn_idx[outer_knn_pad] - n_real   # indices into pad_coords
    if pad_slot_ids.numel() == 0:
        n_pad_rings = 0
    else:
        used_norms = pad_coords[pad_slot_ids].norm(dim=-1).round(decimals=3)
        n_pad_rings = int(torch.unique(used_norms).shape[0])

    full_mask = torch.zeros(P, m * m, dtype=torch.bool)
    if n_pad_rings == 0:
        return full_mask
    for p in outer_ids:
        grid = vf[p].reshape(m, m, 2)
        cx, cy = float(grid[:, :, 0].mean()), float(grid[:, :, 1].mean())
        sec = _sector(cx, cy)
        full_mask[p] = _directional_mask(m, sec, n_pad_rings)
    return full_mask


# =====================================================================
# image-pixel <-> vf helpers for regularization (square_patches_regularized.py)
# =====================================================================
def _relative_to_gridsample(positions_xy: torch.Tensor, fix_loc, fix_size, image_size: int) -> torch.Tensor:
    """Map visual-field (x, y) -> grid_sample grid (image-normalized), via fovi."""
    M = positions_xy.reshape(-1, 2).shape[0]
    grid = xy_to_colrow(positions_xy.reshape(-1, 2), do_norm=False, format="-11")
    fix_loc_t = torch.as_tensor(fix_loc, dtype=torch.float32).reshape(1, 2)
    fix_size_t = torch.as_tensor([fix_size, fix_size], dtype=torch.float32).reshape(1, 2)
    abs_grid = transform_sampling_grid(
        grid.reshape(1, 1, M, 2), fix_loc_t, fix_size_t, (image_size, image_size)
    )
    return abs_grid  # [1, 1, M, 2]


def _to_absolute_xy(positions_xy: torch.Tensor, fix_loc, fix_size, image_size: int) -> torch.Tensor:
    """Visual-field (x, y) -> absolute image coords (y math-up)."""
    abs_grid = _relative_to_gridsample(positions_xy, fix_loc, fix_size, image_size)
    ab = abs_grid.reshape(-1, 2).clone()
    ab[:, 1] = -ab[:, 1]
    return ab


def _abs_norm_to_pixel(abs_xy: torch.Tensor, image_size: int) -> torch.Tensor:
    """Image-normalized [-1, 1] (y math-up) -> continuous image-pixel coords."""
    return (abs_xy + 1) * image_size / 2.0 - 0.5


def _pixel_to_abs_norm(px_xy: torch.Tensor, image_size: int) -> torch.Tensor:
    """Inverse of :func:`_abs_norm_to_pixel`."""
    return (px_xy + 0.5) * 2.0 / image_size - 1.0


def _ring_decisions(pe, fix_size: int, rel_tol: float, m_override: int | None,
                    strict_nest_when_possible: bool, max_edge_change: float):
    """Per-ring: natural vs adapted (edge, spacing), and the keep-or-snap call."""
    m = int(m_override) if m_override is not None else global_m(pe)
    radii = torch.unique(pe.out_coords.polar[:, 0])
    ring_idx = ring_index_per_patch(pe)

    edge_per_patch_vf = compute_ring_edge_lengths(pe, rel_tol=rel_tol)
    ppu = fix_size / 2.0   # 1 vf-unit = fix_size/2 pixels (for fix_size==image_size)

    decisions = []
    last_outer_s = None
    for ri in range(len(radii)):
        mask = ring_idx == ri
        edge_vf = float(edge_per_patch_vf[mask][0])
        edge_px = edge_vf * ppu
        s_nat_px = edge_px / (m - 1)
        nested = False
        if s_nat_px < 1.0:
            kept = True
            s_adp_px = s_nat_px
            edge_adp_px = edge_px
            edge_adp_vf = edge_vf
            edge_chg_pct = 0.0
        else:
            kept = False
            s_round = float(max(1, round(s_nat_px)))
            s_adp_px = s_round
            if strict_nest_when_possible and last_outer_s is not None and last_outer_s >= 1:
                k_ideal = s_nat_px / last_outer_s
                k_lo = max(2, int(k_ideal))
                k_hi = k_lo + 1
                best = None
                for k in (k_lo, k_hi):
                    cand = float(k * last_outer_s)
                    cand_edge_px = cand * (m - 1)
                    rel_chg = abs(cand_edge_px - edge_px) / edge_px
                    if rel_chg <= max_edge_change:
                        d = abs(cand - s_nat_px)
                        if best is None or d < best[0]:
                            best = (d, cand)
                if best is not None:
                    s_adp_px = best[1]
                    nested = True
            edge_adp_px = s_adp_px * (m - 1)
            edge_adp_vf = edge_adp_px / ppu
            edge_chg_pct = (edge_adp_px - edge_px) / edge_px * 100.0
            last_outer_s = s_adp_px
        decisions.append(dict(
            ring=ri, n_patches=int(mask.sum()), r_center=float(radii[ri]),
            kept=kept, nested=nested,
            s_nat_px=s_nat_px, s_adp_px=s_adp_px,
            edge_nat_vf=edge_vf, edge_adp_vf=edge_adp_vf,
            edge_nat_px=edge_px, edge_adp_px=edge_adp_px,
            edge_change_pct=edge_chg_pct,
        ))
    return decisions, m, ppu


# =====================================================================
# strided-square geometry (from square_patches_non_fovi.py FoveatedPatches)
# =====================================================================
def _strided_grid_sizes(grid_size_fovea, patch_size, edge_length_multipliers):
    """Periphery grid sizes (patches per axis) for each layer."""
    prev_edge_length_px = patch_size * grid_size_fovea
    grid_sizes = []
    prev_mult = 1
    for layer_idx, mult in enumerate(edge_length_multipliers, start=1):
        patch_edge_px = mult * patch_size
        if prev_edge_length_px % patch_edge_px != 0:
            k = prev_edge_length_px // patch_size
            valid = [d for d in range(1, k + 1) if k % d == 0 and d > prev_mult]
            raise ValueError(
                f"Invalid edge_length_multiplier={mult} at peripheral layer {layer_idx}: "
                f"prev_edge_length_px={prev_edge_length_px} not divisible by "
                f"patch_edge_px={patch_edge_px}. Valid multipliers (> previous="
                f"{prev_mult}) would be {valid}."
            )
        grid_size = prev_edge_length_px / patch_edge_px + 2
        grid_sizes.append(int(grid_size))
        prev_edge_length_px = int(grid_size) * patch_edge_px
        prev_mult = mult
    return grid_sizes


def _strided_layer_indices(patch_size, edge_length_multiplier, grid_size, keep_inner=False,
                           samples_per_axis=None):
    """Relative (dy, dx) sampling positions for one patch layer (ring around center). [P, K, 2].

    Each patch covers a ``cell_size = edge_length_multiplier * patch_size`` px cell whose
    placement is fixed by ``patch_size`` / ``edge_length_multiplier`` / ``grid_size`` (the
    patch geometry — position and extent — is therefore independent of how densely the cell
    is sampled). ``samples_per_axis`` (``m``; default ``patch_size``) points per axis are
    placed evenly and centered within the cell: offset ``(j + 0.5) * cell_size / m - 0.5``
    for ``j in [0, m)``, whose mean is the cell-box center ``(cell_size - 1) / 2`` for any
    ``m``. So the patch center is invariant to ``m``; only the sample count/spacing changes.
    Positions are floats (sub-pixel for ``m != patch_size``); ``grid_sample`` interpolates.
    With ``m == patch_size`` this reproduces the native strided sampling exactly for the
    fovea and odd strides, and centers even strides (whose integer-pixel layout sat half a
    pixel off-center)."""
    assert edge_length_multiplier >= 1 and grid_size >= 2
    m = patch_size if samples_per_axis is None else int(samples_per_axis)
    assert m >= 1, f"samples_per_axis must be >= 1, got {m}"
    hole_size = grid_size - 2
    cell_size = edge_length_multiplier * patch_size
    hole_start = (grid_size - hole_size) // 2

    idx = torch.arange(grid_size, dtype=torch.int64)
    off_num = (2 * idx - 2 * hole_start - hole_size)
    starts_1d = (off_num * cell_size) // 2
    y0_all = starts_1d[:, None].expand(grid_size, grid_size)
    x0_all = starts_1d[None, :].expand(grid_size, grid_size)

    mask = torch.ones((grid_size, grid_size), dtype=torch.bool)
    if not keep_inner and hole_size > 0:
        mask[hole_start: hole_start + hole_size, hole_start: hole_start + hole_size] = False
    y0 = y0_all[mask]
    x0 = x0_all[mask]

    # m evenly-spaced, cell-centered sample offsets within the cell box.
    a = (torch.arange(m, dtype=torch.float32) + 0.5) * cell_size / m - 0.5   # [m]
    yy = a[:, None].expand(m, m)
    xx = a[None, :].expand(m, m)
    local = torch.stack([yy.reshape(-1), xx.reshape(-1)], dim=-1)            # [m*m, 2]
    top_left = torch.stack([y0, x0], dim=-1)[:, None, :].to(torch.float32)
    indices = top_left + local[None, :, :]                  # [P, m*m, 2] (dy, dx), float
    return indices


# =====================================================================
# Builders
# =====================================================================
def build_fovi_square(pe, m_override: int | None = None, rel_tol: float = 0.05) -> SquarePattern:
    """Squarified approximation of a fovi ``KNNPartitioningPatchEmbedding``.

    One axis-aligned square per fovi patch, matched in count (``m = round(sqrt
    k)`` or ``m_override``), extent and resolution (per-ring edge length). The
    pad mask marks the outermost-ring padding strips (fovi's KNN padding).
    """
    m = int(m_override) if m_override is not None else global_m(pe)
    edge_per_patch = compute_ring_edge_lengths(pe, rel_tol=rel_tol)
    ring_idx = ring_index_per_patch(pe)
    positions_xy = build_square_positions(pe, edge_per_patch, m)        # [P, m*m, 2]
    centers_xy = pe.out_coords.cartesian.to(torch.float32).clone()      # [P, 2]
    pad_mask = _outermost_pad_mask(positions_xy, ring_idx, m, pe)       # [P, m*m]
    return SquarePattern(
        positions_xy=positions_xy.contiguous(),
        pad_mask=pad_mask.contiguous(),
        centers_xy=centers_xy.contiguous(),
        side=m,
        ring_idx=ring_idx.contiguous(),
    )


def build_fovi_square_regularized(
    pe, fix_size: int, max_edge_change: float = 0.25, rel_tol: float = 0.05,
    m_override: int | None = None, strict_nest_when_possible: bool = False,
) -> SquarePattern:
    """Squarified fovi pattern with outer rings snapped to an integer image-pixel lattice.

    Inner rings (natural sample spacing < 1 px) keep fovi's center and edge
    unchanged; outer rings snap their spacing/edge/centers to a per-ring lattice
    on the ``fix_size``-pixel grid so peripheral samples land on integer pixels
    and within-ring samples coincide exactly. The pattern is defined at fixation
    ``(0.5, 0.5)`` with ``image_size == fix_size`` (the reference foveation
    window); the returned ``positions_xy`` are fix_loc-independent vf coords.
    """
    image_size = int(fix_size)
    fix_loc = (0.5, 0.5)
    decisions, m, _ = _ring_decisions(
        pe, fix_size, rel_tol, m_override, strict_nest_when_possible, max_edge_change
    )
    ring_idx = ring_index_per_patch(pe)
    P = pe.out_coords.shape[0]

    centers_vf = pe.out_coords.cartesian.to(torch.float32)
    centers_abs = _to_absolute_xy(centers_vf, fix_loc, fix_size, image_size)
    centers_px = _abs_norm_to_pixel(centers_abs, image_size)

    centers_px_adp = centers_px.clone()
    for r in decisions:
        mask = (ring_idx == r["ring"])
        if r["kept"]:
            continue
        s = r["s_adp_px"]
        offset = (m - 1) / 2.0 * s
        c = centers_px[mask]
        snapped = torch.round((c - offset) / s) * s + offset
        centers_px_adp[mask] = snapped

    offs = torch.linspace(-0.5, 0.5, m, dtype=torch.float32)
    base = torch.stack(
        torch.meshgrid(offs, offs, indexing="ij"), dim=-1
    ).reshape(-1, 2)                                            # [m*m, 2] in [-0.5, 0.5]^2

    positions_px = torch.empty(P, m * m, 2, dtype=torch.float32)
    for r in decisions:
        mask = (ring_idx == r["ring"])
        edge_px = r["edge_adp_px"]
        c = centers_px_adp[mask]
        positions_px[mask] = c[:, None, :] + edge_px * base[None, :, :]

    flat_abs = _pixel_to_abs_norm(positions_px.reshape(-1, 2), image_size)
    # For fix_loc=(0.5,0.5): vf = abs_norm * (image_size / fix_size) == abs_norm here.
    positions_xy = (flat_abs * (image_size / fix_size)).reshape(P, m * m, 2)
    centers_xy = _pixel_to_abs_norm(centers_px_adp, image_size) * (image_size / fix_size)

    pad_mask = _outermost_pad_mask(positions_xy, ring_idx, m, pe)

    over_edge = [r for r in decisions if abs(r["edge_change_pct"]) > max_edge_change * 100]
    if over_edge:
        bar = "!" * 80
        print(bar)
        print("!!! WARNING: ADAPTED EDGE EXCEEDS max_edge_change THRESHOLD !!!")
        for r in over_edge:
            print(
                f"!!! RING {r['ring']}: EDGE {r['edge_nat_px']:.2f} -> {r['edge_adp_px']:.2f} PX "
                f"({r['edge_change_pct']:+.1f}% > {max_edge_change*100:.0f}%)"
            )
        print(bar)

    return SquarePattern(
        positions_xy=positions_xy.contiguous(),
        pad_mask=pad_mask.contiguous(),
        centers_xy=centers_xy.contiguous(),
        side=m,
        ring_idx=ring_idx.contiguous(),
    )


def build_strided_square(
    grid_size_fovea: int, patch_size: int, edge_length_multipliers: list[int],
    fixation_size: int, add_to_patch_size: int = 0,
) -> SquarePattern:
    """Standalone strided-square foveation (no fovi dependency).

    The fovea is ``grid_size_fovea**2`` full-resolution patches; each peripheral
    layer uses ``stride = multiplier`` so its patches cover
    ``(multiplier*patch_size)**2`` pixels (resolution decreases outward).

    ``add_to_patch_size`` decouples sampling density from patch geometry: each patch
    is sampled with ``m = patch_size + add_to_patch_size`` points per axis
    (``K = m**2``), placed evenly and centered within its (unchanged) cell. The patch
    layout, count, centers and extents depend only on ``grid_size_fovea`` /
    ``patch_size`` / ``edge_length_multipliers`` and are independent of
    ``add_to_patch_size``. ``0`` (default) = native resolution; ``>0`` oversamples
    (sub-pixel); ``<0`` undersamples. Sub-pixel positions are bilinearly interpolated
    at sample time.

    The relative pixel layout is mapped into the visual field via
    ``vf = pixel_offset / (fixation_size / 2)``: with the deployment fixation size
    equal to ``fixation_size`` this reproduces native-pixel sampling. Samples
    landing outside the foveation window (``|vf| > 1``) are marked as padding.
    """
    assert isinstance(patch_size, int) and isinstance(grid_size_fovea, int)
    assert isinstance(add_to_patch_size, int)
    assert len(edge_length_multipliers) >= 1, "need at least one peripheral layer"
    assert all(
        edge_length_multipliers[i] < edge_length_multipliers[i + 1]
        for i in range(len(edge_length_multipliers) - 1)
    ), f"edge_length_multipliers must be strictly increasing, got {edge_length_multipliers}"

    m = patch_size + add_to_patch_size                  # samples per axis
    if m < 2:
        raise ValueError(
            f"patch_size + add_to_patch_size must be >= 2, got "
            f"{patch_size} + {add_to_patch_size} = {m}."
        )

    grid_sizes = _strided_grid_sizes(grid_size_fovea, patch_size, edge_length_multipliers)

    all_indices = []
    all_layer_ids = []
    # fovea
    idx = _strided_layer_indices(patch_size, 1, grid_size_fovea, keep_inner=True, samples_per_axis=m)
    all_indices.append(idx)
    all_layer_ids.append(torch.zeros(idx.shape[0], dtype=torch.int64))
    # periphery
    for i, (mult, gs) in enumerate(zip(edge_length_multipliers, grid_sizes)):
        idx = _strided_layer_indices(patch_size, mult, gs, samples_per_axis=m)
        all_indices.append(idx)
        all_layer_ids.append(torch.ones(idx.shape[0], dtype=torch.int64) * (i + 1))

    indices = torch.cat(all_indices, dim=0)             # [P, K, 2] (dy, dx), float
    ring_idx = torch.cat(all_layer_ids, dim=0)          # [P]

    if add_to_patch_size == 0:
        # Geometry sanity check for the native pattern: every sample lands on a
        # distinct (dy, dx). Skipped for resampled densities, where sub-pixel
        # samples may legitimately coincide and grid_sample handles overlaps.
        flat = indices.reshape(-1, 2)
        assert torch.unique(flat, dim=0).shape[0] == flat.shape[0], "Duplicate sampling indices!"

    # relative (dy, dx) pixel offsets -> vf (x, y): x = dx, y = -dy, scaled by fixation_size/2.
    s = fixation_size / 2.0
    dy = indices[..., 0].to(torch.float32)
    dx = indices[..., 1].to(torch.float32)
    positions_xy = torch.stack([dx / s, -dy / s], dim=-1)   # [P, K, 2]
    K = m * m

    pad_mask = (positions_xy.abs() > 1.0).any(dim=-1)        # [P, K]
    centers_xy = positions_xy.mean(dim=1)                    # [P, 2]

    return SquarePattern(
        positions_xy=positions_xy.contiguous(),
        pad_mask=pad_mask.contiguous(),
        centers_xy=centers_xy.contiguous(),
        side=int(m),
        ring_idx=ring_idx.contiguous(),
    )


# =====================================================================
# Per-ring pixel coverage / redundancy (notebook: fovi_ring_stats /
# ring_pixel_stats "new_pixel" column) and ring pruning
# =====================================================================
# These reproduce the per-ring "new pixels" accounting from
# ``notebooks/fovi_square_patches/fovi_plus_square_patches.ipynb``: rings are
# processed outermost -> innermost and, for each ring, we count the distinct
# pixel cells its samples land in that no outer ring has already covered. The
# pixel grid is a ``reference_size x reference_size`` lattice at centered
# fixation (``fix_loc = (0.5, 0.5)``), the same convention as
# ``canvit_pytorch.patcher.embed.count_unique_pixels`` and the notebook's
# ``_vf_unique_pixels`` (no clamp, out-of-window samples keep their own virtual
# cells). The count is a property of the sampling pattern at the reference
# scale, independent of where the sensor is fixated, so it can be computed once.


def _pixel_keys(xy: torch.Tensor, reference_size: int) -> set:
    """Distinct ``reference_size``-grid pixel cells (as ``(int, int)`` tuples).

    Mirrors ``count_unique_pixels`` / the notebook: ``floor((xy + 1)/2 * r)``,
    centered fixation, no clamp and no out-of-window exclusion.
    """
    r = int(reference_size)
    px = torch.floor((xy.reshape(-1, 2).to(torch.float64) + 1.0) * 0.5 * r).long()
    return set(map(tuple, px.tolist()))


def _ring_new_pixels(members_outer_first: list[torch.Tensor], reference_size: int) -> list[int]:
    """Per-ring new-pixel counts given each ring's sample ``(x, y)``, outer-first."""
    accumulated: set = set()
    new_counts: list[int] = []
    for member_xy in members_outer_first:
        keys = _pixel_keys(member_xy, reference_size)
        new = keys - accumulated
        accumulated |= keys
        new_counts.append(len(new))
    return new_counts


def fovi_ring_new_pixels(pe, reference_size: int):
    """Per-ring new-pixel counts for a fovi ``KNNPartitioning`` embedding.

    Faithful to the notebook's ``fovi_ring_stats`` pixel column: rings are the
    distinct output-patch eccentricities (``out_coords.polar`` radius), processed
    outermost -> inner; a ring's "members" are the **real** (non-padded) sensor
    indices its patches pool (``knn_indices`` with ``knn_indices_pad_mask``
    excluded), mapped to ``in_coords.cartesian`` pixels at ``reference_size``.

    Returns ``(radii_outer_first, n_patches_per_ring, new_pixels_per_ring)`` as
    Python lists (one entry per ring).
    """
    radii = pe.out_coords.polar[:, 0]
    sensor_xy = pe.in_coords.cartesian.to(torch.float32)
    knn = pe.knn_indices.long()                 # [k, P]
    pad = pe.knn_indices_pad_mask.bool()         # [k, P], True = padding slot
    radii_outer_first = torch.unique(radii).flip(0)  # largest (outermost) first

    members, n_patches, radii_list = [], [], []
    for r in radii_outer_first:
        patch_ids = (radii == r).nonzero(as_tuple=True)[0]
        sensors: set = set()
        for j in patch_ids.tolist():
            real = knn[~pad[:, j], j]
            sensors.update(int(s) for s in real.tolist())
        idx = torch.tensor(sorted(sensors), dtype=torch.long, device=sensor_xy.device)
        members.append(sensor_xy[idx] if idx.numel() else sensor_xy[:0])
        n_patches.append(int(patch_ids.numel()))
        radii_list.append(float(r))
    return radii_list, n_patches, _ring_new_pixels(members, reference_size)


def fovi_ring_keep_mask(pe, reference_size: int, min_new_pixels: int) -> torch.Tensor:
    """Boolean ``[P]`` keep-mask: drop patches whose ring adds ``< min_new_pixels``.

    ``min_new_pixels <= 0`` keeps every patch (returns all-True). The decision is
    per-ring, so a patch is kept iff its eccentricity ring contributes at least
    ``min_new_pixels`` new pixels at the ``reference_size`` scale.
    """
    radii = pe.out_coords.polar[:, 0]
    device = radii.device
    P = int(radii.shape[0])
    if min_new_pixels <= 0:
        return torch.ones(P, dtype=torch.bool, device=device)
    radii_outer_first, _, new_counts = fovi_ring_new_pixels(pe, reference_size)
    drop_radii = {radii_outer_first[i] for i, n in enumerate(new_counts) if n < min_new_pixels}
    keep = torch.tensor(
        [float(r) not in drop_radii for r in radii.tolist()],
        dtype=torch.bool, device=device,
    )
    return keep


def square_ring_new_pixels(pattern: SquarePattern, reference_size: int):
    """Per-ring new-pixel counts for a :class:`SquarePattern`.

    Faithful to the notebook's ``ring_pixel_stats`` pixel column: rings are the
    distinct ``ring_idx`` values processed outermost -> inner (by descending mean
    center eccentricity); a ring's members are **all** samples of its patches
    (``positions_xy``; out-of-field/padded samples are kept as their own virtual
    cells, exactly as in the notebook). Padding only ever lives on the outermost
    ring, so this never affects which inner ring is pruned.

    Returns ``(ring_ids_outer_first, n_patches_per_ring, new_pixels_per_ring)``.
    """
    if pattern.ring_idx is None:
        # Single implicit ring: all samples, one entry.
        new = _ring_new_pixels([pattern.positions_xy.reshape(-1, 2)], reference_size)
        return [0], [pattern.n_patches], new
    ring_idx = pattern.ring_idx
    center_r = pattern.centers_xy.norm(dim=-1)            # [P]
    unique_rings = torch.unique(ring_idx)
    # order rings outermost -> inner by mean center eccentricity (descending)
    mean_r = {int(rid): float(center_r[ring_idx == rid].mean()) for rid in unique_rings.tolist()}
    rings_outer_first = sorted(mean_r, key=lambda rid: -mean_r[rid])

    members, n_patches = [], []
    for rid in rings_outer_first:
        patch_ids = (ring_idx == rid).nonzero(as_tuple=True)[0]
        members.append(pattern.positions_xy[patch_ids].reshape(-1, 2))
        n_patches.append(int(patch_ids.numel()))
    return rings_outer_first, n_patches, _ring_new_pixels(members, reference_size)


def square_ring_keep_mask(pattern: SquarePattern, reference_size: int, min_new_pixels: int) -> torch.Tensor:
    """Boolean ``[P]`` keep-mask for a :class:`SquarePattern` (see
    :func:`fovi_ring_keep_mask`)."""
    P = pattern.n_patches
    device = pattern.positions_xy.device
    if min_new_pixels <= 0 or pattern.ring_idx is None:
        return torch.ones(P, dtype=torch.bool, device=device)
    rings_outer_first, _, new_counts = square_ring_new_pixels(pattern, reference_size)
    drop_rings = {rings_outer_first[i] for i, n in enumerate(new_counts) if n < min_new_pixels}
    keep = torch.tensor(
        [int(rid) not in drop_rings for rid in pattern.ring_idx.tolist()],
        dtype=torch.bool, device=device,
    )
    return keep
