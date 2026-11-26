"""Matplotlib plotting utilities for visualization."""

from typing import NamedTuple

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.figure import Figure
from matplotlib.patches import Rectangle
from numpy.typing import NDArray
from sklearn.decomposition import PCA

from ...harness.rollout.viewpoint import PixelBox
from ...harness.viz.metrics import cosine_dissimilarity
from ...harness.viz.pca import fit_pca, pca_rgb
from ..probe import TopKPrediction
from .foveated_plot import (
    plot_patch_voronoi_absolute,
    plot_patches_overlay_relative,
    plot_samples_reconstruction_absolute,
    plot_samples_scatter_absolute,
    plot_square_patches_overlay_relative,
)
from .sample import FoveatedVizData, SquareVizData

# Type alias for RGBA color tuple
RGBA = tuple[float, float, float, float]


def timestep_colors(n: int) -> list[RGBA]:
    """Get n distinct colors from viridis colormap."""
    cmap = plt.get_cmap("viridis")
    return [cmap(i / max(1, n - 1)) for i in range(n)]


def plot_trajectory(
    *,
    img: NDArray[np.floating],
    boxes: list[PixelBox],
    names: list[str],
) -> Figure:
    """Plot [H, W, 3] image (values in [0, 1]) with viewpoint boxes overlaid."""
    assert len(boxes) == len(names)
    colors = timestep_colors(len(boxes))

    fig, ax = plt.subplots(figsize=(8, 8))
    ax.imshow(img)

    if len(boxes) > 1:
        xs = [b.center_x for b in boxes]
        ys = [b.center_y for b in boxes]
        ax.plot(xs, ys, "-", color="white", linewidth=1.5, alpha=0.7, zorder=1)

    for i, (box, name) in enumerate(zip(boxes, names, strict=True)):
        rect = Rectangle(
            (box.left, box.top),
            box.width,
            box.height,
            linewidth=2,
            edgecolor=colors[i],
            facecolor="none",
            label=f"t={i} ({name}, s={box.width / img.shape[1]:.2f})",
        )
        ax.add_patch(rect)
        ax.plot(box.center_x, box.center_y, "o", color=colors[i], markersize=6, zorder=2)

    ax.set_title("Viewpoint Trajectory")
    if boxes:
        ax.legend(loc="upper right", fontsize=8)
    ax.axis("off")
    plt.tight_layout()
    return fig


def policy_map_grid(m: NDArray[np.floating]) -> NDArray[np.floating]:
    """A policy's flat action distribution as a ``c x c`` grid over the full field (row = y).
    Foveated action space: ``c x c`` fixation cells. Uniform: ``n_scale x c x c`` (centre AND
    scale), summed over scales -- an approximate picture, since each scale's centres span a
    different safe box."""
    n_scale = next(s for s in range(1, 17) if m.size % s == 0
                   and round((m.size // s) ** 0.5) ** 2 == m.size // s)
    c = round((m.size // n_scale) ** 0.5)
    return m.reshape(n_scale, c, c).sum(0)


def plot_pca_grid(
    *,
    pca: PCA,
    reference: NDArray[np.floating],
    samples: list[NDArray[np.floating]],
    grid_size: int,
    titles: list[str],
) -> Figure:
    """Plot PCA viz comparing [G*G, D] reference (teacher) to samples (model outputs); PCA must be pre-fit on reference."""
    assert len(samples) == len(titles)
    n_cols = 1 + len(samples)

    reference_rgb = pca_rgb(pca, reference, grid_size, grid_size)

    fig, axes = plt.subplots(1, n_cols, figsize=(4 * n_cols, 4))

    axes[0].imshow(reference_rgb)
    axes[0].set_title("Teacher")
    axes[0].axis("off")

    for i, (sample, title) in enumerate(zip(samples, titles, strict=True)):
        sample_rgb = pca_rgb(pca, sample, grid_size, grid_size)
        mse = float(((sample - reference) ** 2).mean())
        axes[i + 1].imshow(sample_rgb)
        axes[i + 1].set_title(f"{title}\nMSE={mse:.4f}")
        axes[i + 1].axis("off")

    plt.tight_layout()
    return fig


class TimestepPredictions(NamedTuple):
    """Top-k predictions at a single timestep for a single sample."""
    predictions: list[TopKPrediction]
    gt_idx: int
    gt_name: str


def plot_multistep_pca(
    *,
    full_img: NDArray[np.floating],
    teacher: NDArray[np.floating],
    scenes: list[NDArray[np.floating]],
    glimpses: list[NDArray[np.floating]],
    boxes: list[PixelBox],
    names: list[str],
    scene_grid_size: int,
    glimpse_grid_size: int,
    initial_scene: NDArray[np.floating],
    locals_avp: list[NDArray[np.floating]] | None = None,
    locals_teacher: list[NDArray[np.floating]] | None = None,
    hidden_spatials: list[NDArray[np.floating]] | None = None,
    initial_hidden_spatial: NDArray[np.floating] | None = None,
    locals_teacher_cropped: list[NDArray[np.floating]] | None = None,
    show_locals: bool = False,
    timestep_predictions: list[TimestepPredictions] | None = None,
    foveated_samples: list[FoveatedVizData | None] | None = None,
    square_samples: list[SquareVizData | None] | None = None,
    policy_maps: list[NDArray[np.floating]] | None = None,
    fig_title: str | None = None,
) -> Figure:
    """Full multi-row visualization with all diagnostic columns.

    Trajectory column: the fixations so far (points joined by lines); the uniform patcher
    also draws each glimpse's box (for foveated / square glimpses a box has no meaning). With
    ``policy_maps`` (``policy_maps[k]`` = the action distribution that chose the glimpse at
    t = k + 1) the row's map is drawn under the fixations, with its entropy and maximum.
    ``fig_title``: a figure title (e.g. the retina of a multi-retina training sample).

    Row 0 = "init": learned spatial_hidden_init projected through scene_proj, BEFORE any glimpses
    Row 1+ = "t=0, t=1, ...": scene state AFTER processing each glimpse

    Uniform-patcher columns:  [Preds] | Trajectory | Glimpse | Teacher | Scene | [Hidden] | [Local Stream] | ... | Δ Scene | Error
    Foveated-patcher columns: [Preds] | Trajectory | Samples | Sample-Recon | Patches | Teacher | Scene | [Hidden] | [Patch-Recon] | ... | Δ Scene | Error
    """
    n_views = len(scenes)
    assert len(glimpses) == n_views
    assert len(boxes) == n_views
    assert len(names) == n_views

    # Detect foveated / square mode by presence of per-timestep viz data. Both
    # use the same "sample-based" column layout (Samples | Sample-Recon |
    # Patches), differing only in how each cell is rendered.
    is_foveated = (
        foveated_samples is not None
        and len(foveated_samples) == n_views
        and all(s is not None for s in foveated_samples)
    )
    is_square = (
        square_samples is not None
        and len(square_samples) == n_views
        and all(s is not None for s in square_samples)
    )
    is_sample = is_foveated or is_square

    if show_locals:
        assert locals_avp is not None and len(locals_avp) == n_views
        # locals_teacher is optional - if None, only show locals_avp

    show_hidden = hidden_spatials is not None
    if show_hidden:
        assert len(hidden_spatials) == n_views
        assert initial_hidden_spatial is not None

    show_teacher_locals = show_locals and locals_teacher is not None
    if show_teacher_locals:
        assert locals_teacher is not None and len(locals_teacher) == n_views

    show_cropped = show_locals and locals_teacher_cropped is not None
    if show_cropped:
        assert locals_teacher_cropped is not None
        assert len(locals_teacher_cropped) == n_views

    show_preds = timestep_predictions is not None
    if show_preds:
        assert len(timestep_predictions) == n_views

    S, G = scene_grid_size, glimpse_grid_size
    n_rows = n_views + 1
    colors = timestep_colors(n_views)

    # Fit PCA on teacher (used for scene comparisons)
    pca_teacher = fit_pca(teacher)
    teacher_rgb = pca_rgb(pca_teacher, teacher, S, S)
    initial_rgb = pca_rgb(pca_teacher, initial_scene, S, S, normalize=True)

    # Scene with own PCA (initial)
    pca_init_scene = fit_pca(initial_scene)
    initial_scene_own_rgb = pca_rgb(pca_init_scene, initial_scene, S, S)

    # Hidden PCA for init (own basis)
    if show_hidden:
        assert initial_hidden_spatial is not None
        pca_init_hidden = fit_pca(initial_hidden_spatial)
        initial_hidden_rgb = pca_rgb(pca_init_hidden, initial_hidden_spatial, S, S)
    else:
        initial_hidden_rgb = None

    # Precompute error maps (cosine dissimilarity, per-row scaling)
    initial_error = cosine_dissimilarity(initial_scene, teacher).reshape(S, S)
    error_maps = [initial_error] + [cosine_dissimilarity(s, teacher).reshape(S, S) for s in scenes]

    # Compute delta maps for scene predictions (cosine dissimilarity from prev)
    delta_scene_maps: list[NDArray[np.floating]] = [np.zeros((S, S))]
    prev_scene = initial_scene
    for scene in scenes:
        delta_scene_maps.append(cosine_dissimilarity(scene, prev_scene).reshape(S, S))
        prev_scene = scene

    # Compute delta maps for hidden states (if available)
    delta_hidden_maps: list[NDArray[np.floating]] | None = None
    if show_hidden:
        assert hidden_spatials is not None and initial_hidden_spatial is not None
        delta_hidden_maps = [np.zeros((S, S))]
        prev_hidden = initial_hidden_spatial
        for h in hidden_spatials:
            delta_hidden_maps.append(cosine_dissimilarity(h, prev_hidden).reshape(S, S))
            prev_hidden = h

    # Column indices. Foveated mode replaces the single "Glimpse" col with
    # three foveated cols, and the uniform "Local Stream" col with a single
    # patch-Voronoi col (rendered differently below).
    c = 0
    C_PREDS = c if show_preds else None
    if show_preds:
        c += 1
    C_TRAJ = c
    c += 1
    if is_sample:
        C_GLIMPSE = None
        C_FOV_SCATTER, C_FOV_RECON, C_FOV_PATCHES = c, c + 1, c + 2
        c += 3
    else:
        C_GLIMPSE = c
        C_FOV_SCATTER = C_FOV_RECON = C_FOV_PATCHES = None
        c += 1
    C_TEACHER, C_SCENE, C_SCENE_OWN = c, c + 1, c + 2
    c += 3
    C_HIDDEN = c if show_hidden else None
    if show_hidden:
        c += 1
    # Local stream col: same slot serves both uniform (g x g grid) and
    # foveated (patch-Voronoi). Either is gated on `show_locals` (== has_locals
    # in the caller), since `locals_avp` carries the data for both modes.
    C_LOCAL_AVP = c if show_locals else None
    if show_locals:
        c += 1
    C_LOCAL_TEACHER = c if show_teacher_locals else None
    if show_teacher_locals:
        c += 1
    C_CROPPED_TEACHER = c if show_cropped else None
    if show_cropped:
        c += 1
    C_DELTA_HIDDEN = c if show_hidden else None
    if show_hidden:
        c += 1
    C_DELTA_SCENE, C_ERROR = c, c + 1
    n_cols = c + 2

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(4 * n_cols, 4 * n_rows))

    # Row 0: Initial state (before any glimpses)
    row = 0
    if show_preds:
        assert C_PREDS is not None
        axes[row, C_PREDS].axis("off")
        axes[row, C_PREDS].set_title("Predictions")
    axes[row, C_TRAJ].imshow(full_img)
    axes[row, C_TRAJ].set_title("init")
    axes[row, C_TRAJ].axis("off")

    if is_sample:
        for col, title in zip(
            (C_FOV_SCATTER, C_FOV_RECON, C_FOV_PATCHES),
            ("(no glimpse)", "", ""),
            strict=True,
        ):
            assert col is not None
            axes[row, col].axis("off")
            if title:
                axes[row, col].set_title(title)
    else:
        assert C_GLIMPSE is not None
        axes[row, C_GLIMPSE].axis("off")
        axes[row, C_GLIMPSE].set_title("(no glimpse)")

    axes[row, C_TEACHER].imshow(teacher_rgb)
    axes[row, C_TEACHER].set_title("Teacher")
    axes[row, C_TEACHER].axis("off")

    axes[row, C_SCENE].imshow(initial_rgb)
    axes[row, C_SCENE].set_title("Scene (init)")
    axes[row, C_SCENE].axis("off")

    axes[row, C_SCENE_OWN].imshow(initial_scene_own_rgb)
    axes[row, C_SCENE_OWN].set_title("Scene own (init)")
    axes[row, C_SCENE_OWN].axis("off")

    if show_hidden:
        assert C_HIDDEN is not None and initial_hidden_rgb is not None
        axes[row, C_HIDDEN].imshow(initial_hidden_rgb)
        axes[row, C_HIDDEN].set_title("Hidden (init)")
        axes[row, C_HIDDEN].axis("off")

    if show_locals:
        assert C_LOCAL_AVP is not None
        axes[row, C_LOCAL_AVP].axis("off")
    if show_teacher_locals:
        assert C_LOCAL_TEACHER is not None
        axes[row, C_LOCAL_TEACHER].axis("off")

    if show_cropped:
        assert C_CROPPED_TEACHER is not None
        axes[row, C_CROPPED_TEACHER].axis("off")

    if show_hidden:
        assert C_DELTA_HIDDEN is not None
        axes[row, C_DELTA_HIDDEN].axis("off")
        axes[row, C_DELTA_HIDDEN].set_title("Δ Hidden")

    axes[row, C_DELTA_SCENE].axis("off")
    axes[row, C_DELTA_SCENE].set_title("Δ Scene")

    im_err = axes[row, C_ERROR].imshow(error_maps[0], cmap="hot")
    cos_dis = float(error_maps[0].mean())
    axes[row, C_ERROR].set_title(f"CosDis ({cos_dis:.4f})")
    axes[row, C_ERROR].axis("off")
    fig.colorbar(im_err, ax=axes[row, C_ERROR], fraction=0.046, pad=0.04)

    # Rows 1+: After each glimpse (t=0, t=1, ...)
    for t in range(n_views):
        row = t + 1
        scene_rgb = pca_rgb(pca_teacher, scenes[t], S, S, normalize=True)

        pca_scene_t = fit_pca(scenes[t])
        scene_own_rgb = pca_rgb(pca_scene_t, scenes[t], S, S)

        if show_hidden:
            assert hidden_spatials is not None
            pca_hidden_t = fit_pca(hidden_spatials[t])
            hidden_rgb = pca_rgb(pca_hidden_t, hidden_spatials[t], S, S)
        else:
            hidden_rgb = None

        local_avp_rgb = None
        local_teacher_rgb = None
        pca_cropped: PCA | None = None
        if show_locals and not is_sample:
            # Uniform-grid local stream: reshape [g*g, D] into a g x g RGB grid.
            # Sample-based paths (foveated/square) compute their PCA in the
            # renderer (different shape -> patch-Voronoi, not a grid).
            assert locals_avp is not None
            pca_local_avp = fit_pca(locals_avp[t])
            local_avp_rgb = pca_rgb(pca_local_avp, locals_avp[t], G, G)
        if show_teacher_locals:
            assert locals_teacher is not None
            pca_local_teacher = fit_pca(locals_teacher[t])
            local_teacher_rgb = pca_rgb(pca_local_teacher, locals_teacher[t], G, G)
            if show_cropped:
                assert locals_teacher_cropped is not None
                pca_cropped = fit_pca(locals_teacher_cropped[t])

        if show_preds:
            assert C_PREDS is not None and timestep_predictions is not None
            preds_t = timestep_predictions[t]
            ax = axes[row, C_PREDS]
            k = len(preds_t.predictions)
            y_pos = np.arange(k)
            probs = [p.probability for p in preds_t.predictions]
            names_short = [p.class_name[:20] for p in preds_t.predictions]
            bar_colors = ["green" if p.class_idx == preds_t.gt_idx else "steelblue" for p in preds_t.predictions]
            ax.barh(y_pos, probs, color=bar_colors)
            ax.set_yticks(y_pos)
            ax.set_yticklabels(names_short, fontsize=8)
            ax.set_xlim(0, 1)
            ax.invert_yaxis()
            ax.set_xlabel("prob", fontsize=8)
            gt_in_topk = any(p.class_idx == preds_t.gt_idx for p in preds_t.predictions)
            if not gt_in_topk:
                ax.set_title(f"GT: {preds_t.gt_name[:15]}", fontsize=8, color="green")

        ax = axes[row, C_TRAJ]
        ax.imshow(full_img)
        img_h, img_w = full_img.shape[:2]
        traj_title = f"t={t}"
        if policy_maps is not None and t >= 1:
            m = policy_maps[t - 1]
            # Cells span the normalized field [-1, 1], which the boxes map to pixel
            # centres [0, W-1] (viewpoint_to_pixel_box).
            ax.imshow(policy_map_grid(m), extent=(0, img_w - 1, img_h - 1, 0), cmap="magma",
                      alpha=0.6, interpolation="nearest")
            ent = float(-(m * np.log(np.clip(m, 1e-12, None))).sum())
            traj_title += f"  H={ent:.2f}  max={float(m.max()):.3f}"
        for i in range(1, t + 1):
            ax.plot(
                [boxes[i - 1].center_x, boxes[i].center_x],
                [boxes[i - 1].center_y, boxes[i].center_y],
                "-", color=colors[i], linewidth=1.5, alpha=0.7,
            )
        for i in range(t + 1):
            box = boxes[i]
            current = i == t
            if not is_sample:
                ax.add_patch(Rectangle(
                    (box.left, box.top), box.width, box.height,
                    linewidth=2, edgecolor=colors[i], facecolor="none", alpha=1.0 if current else 0.4,
                ))
            ax.plot(box.center_x, box.center_y, "o", color=colors[i],
                    markersize=9 if current else 5, markeredgecolor="white" if current else "none",
                    alpha=1.0 if current else 0.6)
        # Axes = the image (a box larger than the image used to shrink it).
        ax.set_xlim(-0.5, img_w - 0.5)
        ax.set_ylim(img_h - 0.5, -0.5)
        ax.set_title(traj_title)
        ax.axis("off")

        if is_foveated:
            assert foveated_samples is not None
            fov = foveated_samples[t]
            assert fov is not None
            # Pixel coords are pre-computed in FoveatedVizData using the
            # per-step viewpoint fixation (full-image foveation).
            sample_px = fov.sample_xy_pixel
            assert C_FOV_SCATTER is not None
            assert C_FOV_RECON is not None
            assert C_FOV_PATCHES is not None
            plot_samples_scatter_absolute(
                axes[row, C_FOV_SCATTER], full_img,
                sample_px, fov.sample_colors, sizes=fov.sample_sizes,
                title=f"Samples ({names[t]})" if t == 0 else f"Samples t={t}",
            )
            img_h, img_w = full_img.shape[:2]
            plot_samples_reconstruction_absolute(
                axes[row, C_FOV_RECON],
                sample_px, fov.sample_colors, img_h, img_w,
                title="Sample recon" if t == 0 else "",
            )
            plot_patches_overlay_relative(
                axes[row, C_FOV_PATCHES],
                fov.sample_cart_rowcol, fov.sample_colors, fov.sample_sizes,
                fov.knn_indices, fov.knn_pad_mask, fov.out_polar_r,
                cart_pad_xy=fov.cart_pad_rowcol,
                title="Patches (rel)" if t == 0 else "",
                show_padding=True,
            )
        elif is_square:
            assert square_samples is not None
            sq = square_samples[t]
            assert sq is not None
            assert C_FOV_SCATTER is not None and C_FOV_RECON is not None and C_FOV_PATCHES is not None
            sample_px = sq.sample_xy_pixel
            plot_samples_scatter_absolute(
                axes[row, C_FOV_SCATTER], full_img,
                sample_px, sq.sample_colors, sizes=sq.sample_sizes,
                title=f"Samples ({names[t]})" if t == 0 else f"Samples t={t}",
            )
            img_h, img_w = full_img.shape[:2]
            plot_samples_reconstruction_absolute(
                axes[row, C_FOV_RECON],
                sample_px, sq.sample_colors, img_h, img_w,
                title="Sample recon" if t == 0 else "",
            )
            plot_square_patches_overlay_relative(
                axes[row, C_FOV_PATCHES],
                sq.sample_cart_rowcol, sq.sample_colors, sq.sample_sizes,
                sq.patch_boxes_rowcol, sq.patch_ring_idx,
                title="Patches (rel)" if t == 0 else "",
            )
        else:
            assert C_GLIMPSE is not None
            axes[row, C_GLIMPSE].imshow(glimpses[t])
            axes[row, C_GLIMPSE].set_title(f"Glimpse ({names[t]})")
            axes[row, C_GLIMPSE].axis("off")

        axes[row, C_TEACHER].imshow(teacher_rgb)
        axes[row, C_TEACHER].axis("off")

        axes[row, C_SCENE].imshow(scene_rgb)
        axes[row, C_SCENE].set_title(f"Scene t={t}")
        axes[row, C_SCENE].axis("off")

        axes[row, C_SCENE_OWN].imshow(scene_own_rgb)
        axes[row, C_SCENE_OWN].set_title("Scene own" if t == 0 else "")
        axes[row, C_SCENE_OWN].axis("off")

        if show_hidden:
            assert C_HIDDEN is not None and hidden_rgb is not None
            axes[row, C_HIDDEN].imshow(hidden_rgb)
            axes[row, C_HIDDEN].set_title("Hidden" if t == 0 else "")
            axes[row, C_HIDDEN].axis("off")

        if show_locals:
            assert C_LOCAL_AVP is not None
            if is_foveated:
                # Foveated: patch-Voronoi in absolute (image) coords with sample hull mask.
                assert foveated_samples is not None
                fov = foveated_samples[t]
                assert fov is not None
                # local_patches[t] is [N_patches, D]; PCA -> [N_patches, 3] RGB.
                patch_pca = fit_pca(locals_avp[t])  # type: ignore[index]
                patch_rgb = patch_pca.transform(locals_avp[t])[:, :3]  # type: ignore[index]
                # Normalize each channel to [0, 1] for consistent display.
                lo = patch_rgb.min(axis=0, keepdims=True)
                hi = patch_rgb.max(axis=0, keepdims=True)
                patch_rgb = (patch_rgb - lo) / np.maximum(hi - lo, 1e-8)
                img_h, img_w = full_img.shape[:2]
                plot_patch_voronoi_absolute(
                    axes[row, C_LOCAL_AVP],
                    fov.patch_xy_pixel, patch_rgb, fov.sample_xy_pixel, img_h, img_w,
                    title="Local Stream (fov)" if t == 0 else "",
                )
            elif is_square:
                # Square: patch-Voronoi from patch centers, masked by the
                # non-masked sample hull (matches the foveated path).
                assert square_samples is not None
                sq = square_samples[t]
                assert sq is not None
                patch_pca = fit_pca(locals_avp[t])  # type: ignore[index]
                patch_rgb = patch_pca.transform(locals_avp[t])[:, :3]  # type: ignore[index]
                lo = patch_rgb.min(axis=0, keepdims=True)
                hi = patch_rgb.max(axis=0, keepdims=True)
                patch_rgb = (patch_rgb - lo) / np.maximum(hi - lo, 1e-8)
                img_h, img_w = full_img.shape[:2]
                plot_patch_voronoi_absolute(
                    axes[row, C_LOCAL_AVP],
                    sq.patch_xy_pixel, patch_rgb, sq.sample_xy_pixel, img_h, img_w,
                    title="Local Stream (sq)" if t == 0 else "",
                )
            else:
                assert local_avp_rgb is not None
                axes[row, C_LOCAL_AVP].imshow(local_avp_rgb)
                if t == 0:
                    axes[row, C_LOCAL_AVP].set_title("Local Stream")
                axes[row, C_LOCAL_AVP].axis("off")

        if show_teacher_locals:
            assert C_LOCAL_TEACHER is not None and local_teacher_rgb is not None
            axes[row, C_LOCAL_TEACHER].imshow(local_teacher_rgb)
            axes[row, C_LOCAL_TEACHER].set_title("Local Teacher" if t == 0 else "")
            axes[row, C_LOCAL_TEACHER].axis("off")

        if show_cropped:
            assert C_CROPPED_TEACHER is not None and locals_teacher_cropped is not None and pca_cropped is not None
            cropped_rgb = pca_rgb(pca_cropped, locals_teacher_cropped[t], G, G)
            axes[row, C_CROPPED_TEACHER].imshow(cropped_rgb)
            if t == 0:
                axes[row, C_CROPPED_TEACHER].set_title("Cropped Teacher")
            axes[row, C_CROPPED_TEACHER].axis("off")

        if show_hidden:
            assert C_DELTA_HIDDEN is not None and delta_hidden_maps is not None
            im_dh = axes[row, C_DELTA_HIDDEN].imshow(delta_hidden_maps[row], cmap="viridis")
            axes[row, C_DELTA_HIDDEN].set_title(f"Δh t={t}" if t == 0 else f"t={t}")
            axes[row, C_DELTA_HIDDEN].axis("off")
            fig.colorbar(im_dh, ax=axes[row, C_DELTA_HIDDEN], fraction=0.046, pad=0.04)

        im_ds = axes[row, C_DELTA_SCENE].imshow(delta_scene_maps[row], cmap="viridis")
        axes[row, C_DELTA_SCENE].set_title(f"Δs t={t}" if t == 0 else f"t={t}")
        axes[row, C_DELTA_SCENE].axis("off")
        fig.colorbar(im_ds, ax=axes[row, C_DELTA_SCENE], fraction=0.046, pad=0.04)

        im_err = axes[row, C_ERROR].imshow(error_maps[row], cmap="hot")
        cos_dis = float(error_maps[row].mean())
        axes[row, C_ERROR].set_title(f"CosDis ({cos_dis:.4f})")
        axes[row, C_ERROR].axis("off")
        fig.colorbar(im_err, ax=axes[row, C_ERROR], fraction=0.046, pad=0.04)

    if fig_title:
        fig.suptitle(fig_title, fontsize=24)
    plt.tight_layout()
    return fig
