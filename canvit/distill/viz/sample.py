"""Sample extraction for visualization (shared by train and validation)."""

from dataclasses import dataclass

import numpy as np
import torch
from torch import Tensor

from canvit import CanViTForPretraining
from canvit.core import CanViTOutput
from canvit.core.viewpoint import Viewpoint, sample_at_viewpoint

from .image import imagenet_denormalize_to_numpy


@dataclass
class FoveatedVizData:
    """Foveated-patcher viz tensors extracted for a single sample.

    All numpy, on CPU. Two coord systems are exposed so the renderer doesn't
    have to know about fovi's internals:

    - Visual-field frame ``[-1, 1]^2`` (row, col): retinal samples and patch
      centers relative to the fixation. Used by the "patches over RT
      (relative)" overlay column.
    - Image-pixel frame (x, y) in pixels: same retinal samples and patch
      centers projected into the full-image pixel frame, using the per-step
      viewpoint's fixation point. Used by all "absolute" foveated columns.
      Some pixel coords land outside the image when fixations are near edges
      — that's expected (out-of-image samples).
    """

    sample_cart_rowcol: np.ndarray         # [N_samples, 2] in [-1, 1]^2 (relative)
    sample_xy_pixel: np.ndarray            # [N_samples, 2] (x, y) image pixels
    sample_colors: np.ndarray              # [N_samples, 3] in [0, 1]
    sample_sizes: np.ndarray               # [N_samples] scatter sizes
    patch_cart_rowcol: np.ndarray          # [N_patches, 2] in [-1, 1]^2 (relative)
    patch_xy_pixel: np.ndarray             # [N_patches, 2] (x, y) image pixels
    knn_indices: np.ndarray                # [k, N_patches] int sample indices
    knn_pad_mask: np.ndarray               # [k, N_patches] bool, True = padded slot
    cart_pad_rowcol: np.ndarray | None     # [N_pad, 2] padded sample slots (for overlay context)
    out_polar_r: np.ndarray                # [N_patches] radii for ring coloring


@dataclass
class SquareVizData:
    """Square-patcher viz tensors extracted for a single sample (numpy, CPU).

    Square patches (fovi-derived or strided) expose per-pixel sample positions
    + a ``pad_mask``; deliberately-masked slots are dropped here, so the
    ``*_vis`` arrays already contain only the shown (non-masked) samples. Out-of-
    field non-masked samples are kept (they fall outside the image in the pixel
    frame, like the foveated path). Two coord systems, mirroring
    :class:`FoveatedVizData`:
      - VF frame ``[-1, 1]^2`` (row, col): for the "patches (rel)" overlay.
      - image-pixel (x, y): for the absolute scatter / reconstruction columns.
    """

    sample_xy_pixel: np.ndarray            # [N_vis, 2] (x, y) image pixels (non-masked)
    sample_cart_rowcol: np.ndarray         # [N_vis, 2] in [-1, 1]^2 (non-masked)
    sample_colors: np.ndarray              # [N_vis, 3] in [0, 1] (non-masked)
    sample_sizes: np.ndarray               # [N_vis] scatter sizes
    patch_boxes_rowcol: np.ndarray         # [N_patches, 4] (rowmin,colmin,rowmax,colmax) full square in VF
    patch_ring_idx: np.ndarray             # [N_patches] ring index (for outline color)
    patch_xy_pixel: np.ndarray             # [N_patches, 2] (x, y) patch centers in image pixels


@dataclass
class VizSampleData:
    """Viz data extracted for a single sample.

    Shape annotations:
        G = canvas grid size (e.g., 32)
        g = glimpse grid size (e.g., 8)
        D = teacher feature dim (e.g., 768)
        C = canvas hidden dim
    """

    glimpse: np.ndarray  # [H, W, 3] denormalized RGB (uniform: cropped window; foveated: full image)
    predicted_scene: np.ndarray  # [G², D] teacher-space prediction
    canvas_spatial: np.ndarray  # [G², C] raw hidden state
    local_patches: np.ndarray | None  # [g², D] local stream patches (None if not available)
    foveated: FoveatedVizData | None  # set only when model.patcher is foveated
    square: "SquareVizData | None" = None  # set only when model.patcher is square


def _as_numpy(x, dtype=None):
    """Tensor or numpy -> numpy array. fovi mixes both types on the patcher."""
    if isinstance(x, torch.Tensor):
        arr = x.detach().cpu().numpy()
    else:
        arr = np.asarray(x)
    if dtype is not None:
        arr = arr.astype(dtype, copy=False)
    return arr


def _rowcol_to_image_xy_pixel(
    rowcol_vf: np.ndarray,
    fix_center_rowcol: np.ndarray,
    image_size: int,
    fix_size_norm: float = 1.0,
) -> np.ndarray:
    """Map visual-field rowcol in [-1, 1]^2 to image-pixel (x, y).

    Mirrors the model's ``FoveatedPatcher.forward`` scene-position mapping
    (``scene_rowcol = fix_center + (fixation_size / image_size) * vf_rowcol``).
    ``fix_size_norm`` is that ``fixation_size / image_size`` ratio: 1.0 for
    full-image foveation (window == image), >1 when zoomed out (e.g. 722/512),
    <1 when zoomed in. Then convert image rowcol to pixel space:
        col_px = (col_image + 1) / 2 * (W - 1)
        row_px = (row_image + 1) / 2 * (H - 1)
    and return as (x=col_px, y=row_px).
    """
    image_rowcol = fix_center_rowcol[None, :] + fix_size_norm * rowcol_vf  # [N, 2]
    row_px = (image_rowcol[:, 0] + 1.0) * 0.5 * (image_size - 1)
    col_px = (image_rowcol[:, 1] + 1.0) * 0.5 * (image_size - 1)
    return np.stack([col_px, row_px], axis=1)  # (x, y)


def _extract_foveated_sample0(
    *,
    model: CanViTForPretraining,
    image: Tensor,
    viewpoint: Viewpoint,
) -> FoveatedVizData | None:
    """If ``model.patcher`` is foveated, capture the sample-level retinal
    output and the patcher's KNN partitioning state for the full image at the
    given viewpoint's fixation point.

    Cheap (~3 ms): re-runs only the retina on ``image[0:1]`` to obtain RGB at
    each sample point.
    """
    patcher = getattr(model, "patcher", None)
    if patcher is None or getattr(patcher, "kpe", None) is None:
        return None
    # Duck-type: foveated patcher has `retina` and `kpe` (KNNPartitioningPatchEmbedding).
    if not hasattr(patcher, "retina") or not hasattr(patcher, "kpe"):
        return None
    # Sample 0's retina (the only one unless the patcher has several, `cfg.retinas`).
    retina, kpe = patcher.retina, patcher.kpe
    if getattr(patcher, "n_retinas", 1) > 1:
        retina, kpe = patcher.geometry(patcher.active_retina(0))

    img0 = image[0:1]  # [1, 3, H, W]
    H = int(img0.shape[-1])
    # The deploy window matches the model forward: fix_size = scale * H (sample 0).
    # scale=1 -> full image; <1 zooms in; >1 zooms out. Using the viewpoint's
    # actual scale keeps the viz overlay faithful under sampled-scale training.
    scale0 = float(viewpoint.scales[0].item())
    fix_size = scale0 * float(H)
    fix_size_norm = scale0
    fix_loc = (viewpoint.centers[0:1].to(torch.float32) + 1.0) * 0.5  # [1, 2] in [0, 1]
    with torch.no_grad():
        sensor = retina(img0, fix_loc=fix_loc, fixation_size=fix_size)  # [1, 3, N_samples]

    # Per-sample RGB in [0, 1]. RetinalTransform output mirrors the ImageNet-
    # normalized image it consumed; denormalize so colors are display-ready.
    # Sensor shape [B, 3, N] -> reshape to [B, N, 3] for denormalize helper.
    s_chw = sensor[0].detach().cpu()  # [3, N_samples]
    s_nc = s_chw.transpose(0, 1).contiguous()  # [N_samples, 3]
    # Re-use imagenet_denormalize on a (1, 3, N, 1) faux-image then squeeze.
    s_faux = s_nc.transpose(0, 1).unsqueeze(-1)  # [3, N_samples, 1]
    sample_colors = imagenet_denormalize_to_numpy(s_faux)[..., 0, :]  # actually [N_samples, 3]
    sample_colors = np.clip(np.asarray(sample_colors, dtype=np.float32), 0.0, 1.0)

    sample_cart_rowcol = _as_numpy(retina.sampler.coords.cartesian_rowcol, dtype=np.float32)
    raw_sizes = getattr(retina, "scatter_sizes", None)
    if raw_sizes is not None:
        sample_sizes = _as_numpy(raw_sizes, dtype=np.float32) * 4.0 + 1.5
    else:
        sample_sizes = np.full(sample_cart_rowcol.shape[0], 3.0, dtype=np.float32)

    patch_cart_rowcol = _as_numpy(kpe.out_coords.cartesian_rowcol, dtype=np.float32)
    knn_indices = _as_numpy(kpe.knn_indices)
    knn_pad_mask = _as_numpy(kpe.knn_indices_pad_mask).astype(bool)
    out_polar_r = _as_numpy(kpe.out_coords.polar[:, 0], dtype=np.float32)

    cart_pad_rowcol: np.ndarray | None = None
    pad_rowcol_attr = getattr(kpe.in_coords, "cartesian_pad_rowcol", None)
    if pad_rowcol_attr is not None:
        cart_pad_rowcol = _as_numpy(pad_rowcol_attr, dtype=np.float32)
    else:
        # fovi only exposes pad coords in the xy frame (`cartesian_pad_coords`).
        # Real samples here are in the (row, col) = (-y, x) frame (the
        # `cartesian_rowcol` convention, xy_to_rowcol format='-11'); apply the
        # same map so pad markers/hulls align with the sample scatter.
        pad_xy = getattr(kpe.in_coords, "cartesian_pad_coords", None)
        if pad_xy is not None:
            pad_xy = _as_numpy(pad_xy, dtype=np.float32)
            cart_pad_rowcol = np.stack([-pad_xy[:, 1], pad_xy[:, 0]], axis=1)

    # Map every (row, col) in visual-field frame to image-pixel (x, y) using
    # the viewpoint's fixation and the configured fixation window:
    # ``scene_rowcol = fix_center + (fix_size / image_size) * vf_rowcol``.
    fix_center_rowcol = viewpoint.centers[0].detach().cpu().to(torch.float32).numpy()
    sample_xy_pixel = _rowcol_to_image_xy_pixel(sample_cart_rowcol, fix_center_rowcol, H, fix_size_norm)
    patch_xy_pixel = _rowcol_to_image_xy_pixel(patch_cart_rowcol, fix_center_rowcol, H, fix_size_norm)

    return FoveatedVizData(
        sample_cart_rowcol=sample_cart_rowcol,
        sample_xy_pixel=sample_xy_pixel,
        sample_colors=sample_colors,
        sample_sizes=sample_sizes,
        patch_cart_rowcol=patch_cart_rowcol,
        patch_xy_pixel=patch_xy_pixel,
        knn_indices=knn_indices,
        knn_pad_mask=knn_pad_mask,
        cart_pad_rowcol=cart_pad_rowcol,
        out_polar_r=out_polar_r,
    )


def _extract_square_sample0(
    *,
    model: CanViTForPretraining,
    image: Tensor,
    viewpoint: Viewpoint,
) -> "SquareVizData | None":
    """If ``model.patcher`` is a square patcher, capture its per-pixel sample
    positions / pad mask / ring index and sample per-pixel RGB via the patcher's
    own grid_sample (matching ``SquarePatcher.forward``). Deliberately-masked
    (``pad_mask``) slots are dropped; out-of-field non-masked slots are kept.
    """
    import torch.nn.functional as F

    from canvit.core.foveation.coords import transform_sampling_grid

    patcher = getattr(model, "patcher", None)
    # Duck-type the square patcher (no retina/kpe; exposes the pattern directly).
    if patcher is None or not hasattr(patcher, "sample_positions_xy") or not hasattr(patcher, "pad_mask"):
        return None

    img0 = image[0:1]  # [1, 3, H, W]
    H = int(img0.shape[-1])
    # Deploy window matches the forward: fix_size = scale * H (sample 0).
    scale0 = float(viewpoint.scales[0].item())
    fix_size = scale0 * float(H)
    fix_size_norm = scale0
    fix_loc = (viewpoint.centers[0:1].to(torch.float32) + 1.0) * 0.5  # [1, 2] in [0, 1]

    pos_xy = patcher.sample_positions_xy().detach().cpu()             # [P, K, 2] VF (x, y)
    pad = _as_numpy(patcher.pad_mask).astype(bool)                   # [P, K]
    ring = _as_numpy(patcher.ring_idx, dtype=np.int64)              # [P]
    P, K = pos_xy.shape[0], pos_xy.shape[1]

    # Per-pixel RGB: replicate SquarePatcher.forward's grid_sample (bilinear),
    # then imagenet-denormalize for display.
    with torch.no_grad():
        fix_size_t = torch.tensor([[fix_size, fix_size]], dtype=torch.float32)  # scale*H px
        grid = transform_sampling_grid(patcher._sample_colrow, fix_loc, fix_size_t, (H, H))
        samp = F.grid_sample(img0.to(torch.float32), grid, mode="bilinear",
                             padding_mode="zeros", align_corners=False)  # [1, 3, 1, P*K]
    s_faux = samp[0, :, 0, :].detach().cpu().unsqueeze(-1)            # [3, P*K, 1]
    colors = imagenet_denormalize_to_numpy(s_faux)[..., 0, :]        # [P*K, 3]
    colors = np.clip(np.asarray(colors, dtype=np.float32), 0.0, 1.0).reshape(P, K, 3)

    # VF rowcol (row=-y, col=x), matching the cartesian_rowcol convention.
    xy = pos_xy.numpy().astype(np.float32)
    rowcol = np.stack([-xy[..., 1], xy[..., 0]], axis=-1)            # [P, K, 2]

    # Full per-patch square extent in VF rowcol (over ALL slots, incl. masked).
    patch_boxes_rowcol = np.stack([
        rowcol[..., 0].min(axis=1), rowcol[..., 1].min(axis=1),
        rowcol[..., 0].max(axis=1), rowcol[..., 1].max(axis=1),
    ], axis=1).astype(np.float32)                                    # [P, 4]

    keep = ~pad                                                      # [P, K] non-masked
    sample_cart_rowcol = rowcol[keep]                                # [N_vis, 2]
    sample_colors = colors[keep]                                     # [N_vis, 3]
    sample_sizes = np.full(sample_cart_rowcol.shape[0], 4.0, dtype=np.float32)

    fix_center_rowcol = viewpoint.centers[0].detach().cpu().to(torch.float32).numpy()
    sample_xy_pixel = _rowcol_to_image_xy_pixel(sample_cart_rowcol, fix_center_rowcol, H, fix_size_norm)
    patch_centers_rowcol = _as_numpy(patcher._patch_rowcol, dtype=np.float32)  # [P, 2]
    patch_xy_pixel = _rowcol_to_image_xy_pixel(patch_centers_rowcol, fix_center_rowcol, H, fix_size_norm)

    return SquareVizData(
        sample_xy_pixel=sample_xy_pixel,
        sample_cart_rowcol=sample_cart_rowcol,
        sample_colors=sample_colors,
        sample_sizes=sample_sizes,
        patch_boxes_rowcol=patch_boxes_rowcol,
        patch_ring_idx=ring,
        patch_xy_pixel=patch_xy_pixel,
    )


def extract_sample0_viz(
    out: CanViTOutput,
    image: Tensor,
    viewpoint: Viewpoint,
    predicted_scene: Tensor,
    model: CanViTForPretraining,
    glimpse_size_px: int,
) -> VizSampleData:
    """Extract viz data for sample 0, move to CPU as numpy.

    Uniform mode: computes the cropped glimpse from ``image`` at ``viewpoint``
    for the "Glimpse" column. Foveated mode: stores the full image (the
    "Glimpse" column is replaced with foveated-specific panels in ``plot.py``).
    """
    patcher_name = getattr(model.cfg, "patcher_name", "uniform")
    is_foveated = patcher_name == "foveated"
    is_square = patcher_name == "square"

    if is_foveated or is_square:
        # No crop — the sample-based columns show the full image directly.
        glimpse_np = imagenet_denormalize_to_numpy(image[0].detach().cpu())
    else:
        glimpse_t = sample_at_viewpoint(
            spatial=image, viewpoint=viewpoint, glimpse_size_px=glimpse_size_px,
        )
        glimpse_np = imagenet_denormalize_to_numpy(glimpse_t[0].detach().cpu())

    scene_cpu = predicted_scene[0].detach().cpu().float()
    scene_np = scene_cpu.numpy()

    canvas_single = out.state.canvas[0:1].detach()
    spatial = model.get_spatial(canvas_single)[0]
    spatial_np = spatial.cpu().float().numpy()

    # Extract local stream patches if available
    local_np: np.ndarray | None = None
    if out.local_patches is not None:
        local_np = out.local_patches[0].detach().cpu().float().numpy()

    foveated = _extract_foveated_sample0(model=model, image=image, viewpoint=viewpoint) if is_foveated else None
    square = _extract_square_sample0(model=model, image=image, viewpoint=viewpoint) if is_square else None

    return VizSampleData(
        glimpse=glimpse_np,
        predicted_scene=scene_np,
        canvas_spatial=spatial_np,
        local_patches=local_np,
        foveated=foveated,
        square=square,
    )
