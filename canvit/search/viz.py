"""Visual-search figure: one block per sample (one column per glimpse), in two rows of
blocks: target-present samples on top (green tint), target-absent below (orange tint).

Rows of a sample's block, for glimpse t:

1. **fixations** -- the scene with the target box (green), the fixations 0..t (white
   path, current one large), and the task head's argmax as a cross: green = inside the
   box (``found``), yellow = only its ``overlap_box_frac`` square overlaps the box
   (``found_X_percent``), red = neither, orange = target absent (no box);
   with a presence head the title carries p(present) after glimpse t.
2. **policy** (only when a policy chose the fixations) -- the policy's distribution over
   the 32 x 32 fixation grid that CHOSE fixation t (computed from the canvas after glimpse
   t-1), overlaid on the scene, with its entropy. t0 is the fixed centre, so it is blank.
3. **task head** -- softmax of the task head's heatmap after glimpse t, overlaid on the
   scene, with that glimpse's KL to the target map.
4. **canvas** -- PCA (3 components -> RGB) of the canvas features after glimpse t, fitted
   once per sample over all glimpses so colours are comparable across t.

Coordinates follow ``loss.py``: [-1, 1], fixation centres are (y, x), boxes are xyxy.
"""

from __future__ import annotations

from dataclasses import dataclass

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
from matplotlib.figure import Figure  # noqa: E402
from matplotlib.patches import Rectangle  # noqa: E402
from torch import Tensor  # noqa: E402

from canvit.harness.viz.pca import fit_pca, pca_rgb  # noqa: E402

from .loss import (  # noqa: E402
    argmax_centres,
    inside_box,
    per_image_loss,
    square_overlaps_box,
    target_map,
)

_MEAN = np.array([0.485, 0.456, 0.406])
# the current fixation: the same teal dot in every row (the X is the task head's guess)
_FIX = dict(s=60, c="cyan", edgecolors="black", linewidths=0.8, zorder=4)
_STD = np.array([0.229, 0.224, 0.225])


@dataclass
class SearchEpisode:
    """One rollout of the first ``n`` samples of a batch, on CPU.

    ``policy[t]`` is the distribution that chose fixation t, so ``policy[0]`` is unused
    (the centred t0). None when the fixations were not chosen by a policy."""

    images: Tensor             # [n, 3, S, S] ImageNet-normalised
    boxes: Tensor              # [n, N, 4] xyxy in [-1, 1], NaN rows = padding
    targets: list[str]         # [n] class names
    centers: Tensor            # [T, n, 2] fixation centres (y, x)
    logits: Tensor             # [T, n, G, G] task-head logits after each glimpse
    canvas: Tensor             # [T, n, G, G, D] canvas spatial features after each glimpse
    policy: Tensor | None      # [T, n, Gp, Gp] policy logits, or None
    present: Tensor            # [n] bool: target present
    presence: Tensor | None    # [T, n] presence logits (log-odds of present), or None
    overlap_box_frac: float
    loss_kind: str
    sigma: float


def _scene(img: Tensor) -> np.ndarray:
    x = img.float().permute(1, 2, 0).numpy() * _STD + _MEAN
    return np.clip(x, 0.0, 1.0)


def _px(v: Tensor | float, size: int) -> float:
    return (float(v) + 1.0) / 2.0 * size


def _draw_boxes(ax, boxes: Tensor, size: int) -> None:
    for b in boxes:
        if torch.isnan(b).any():
            continue
        x0, y0, x1, y1 = (_px(v, size) for v in b)
        ax.add_patch(Rectangle((x0, y0), x1 - x0, y1 - y0, fill=False, ec="lime", lw=1.2))


def _overlay(ax, scene: np.ndarray, prob: np.ndarray, size: int) -> None:
    ax.imshow(scene * 0.55)
    ax.imshow(prob / max(prob.max(), 1e-12), cmap="inferno", alpha=0.65, vmin=0.0, vmax=1.0,
              extent=(0, size, size, 0), interpolation="nearest")


def _draw_path(ax, cx: list[float], cy: list[float], t: int) -> None:
    """The scanpath up to fixation t: white line, past fixations as small white dots, the
    current one as the teal dot."""
    ax.plot(cx[:t + 1], cy[:t + 1], "-", color="white", lw=2.2, alpha=0.9)
    ax.scatter(cx[:t], cy[:t], s=14, c="white", edgecolors="black", linewidths=0.5, zorder=3)
    ax.scatter(cx[t:t + 1], cy[t:t + 1], **_FIX)


def _entropy(logits: Tensor) -> float:
    logp = torch.log_softmax(logits.flatten().float(), dim=0)
    return float(-(logp.exp() * logp).sum())


def make_search_figure(ep: SearchEpisode, *, title: str = "") -> Figure:
    T, n = ep.centers.shape[:2]
    G = ep.logits.shape[-1]
    size = ep.images.shape[-1]
    row_names = ["fixations"] + (["policy"] if ep.policy is not None else []) + ["task head", "canvas"]
    R = len(row_names)
    # one block (R rows x T glimpses) per sample, in two rows of blocks; the caller orders
    # target-present samples first, so 3 + 3 gives present on top, absent below. Each
    # block is tinted by its kind and titled with its cue.
    nrow = min(n, 2)
    ncol = -(-n // nrow)
    fig = plt.figure(figsize=(2.4 * T * ncol, 2.5 * R * nrow + 0.4), layout="constrained")
    blocks = np.atleast_1d(fig.subfigures(nrow, ncol, wspace=0.03, hspace=0.02)).flatten()

    tmap = target_map(ep.boxes, grid=G, kind=ep.loss_kind, sigma=ep.sigma)  # [n, G*G]
    for i in range(n):
        scene = _scene(ep.images[i])
        feats = ep.canvas[:, i].reshape(T, G * G, -1).float().numpy()
        pca = fit_pca(feats.reshape(T * G * G, -1), n_components=3)
        cy = [_px(ep.centers[t, i, 0], size) for t in range(T)]
        cx = [_px(ep.centers[t, i, 1], size) for t in range(T)]
        tp = bool(ep.present[i])
        sf = blocks[i]
        sf.set_facecolor("#e8f5e9" if tp else "#fff3e0")
        sf.suptitle(f"cue: {ep.targets[i]}  -  target {'present' if tp else 'ABSENT'}",
                    fontsize=11, fontweight="bold")
        axes = sf.subplots(R, T, squeeze=False)
        for j, name in enumerate(row_names):
            axes[j, 0].set_ylabel(name, fontsize=9)
        for ax in axes.flat:  # pin the view to the scene: markers near the border must not rescale it
            ax.set_xlim(0, size)
            ax.set_ylim(size, 0)
            ax.set_xticks([])
            ax.set_yticks([])
        for t in range(T):
            lg = ep.logits[t, i:i + 1]
            kl = float(per_image_loss(lg, tmap[i:i + 1]))
            guess = argmax_centres(lg)  # [1, 2] (x, y)
            found = bool(inside_box(guess, ep.boxes[i:i + 1]))
            near = bool(square_overlaps_box(guess, ep.boxes[i:i + 1], box_frac=ep.overlap_box_frac))
            pp = (f"  p(pres)={float(torch.sigmoid(ep.presence[t, i])):.2f}"
                  if ep.presence is not None else "")
            r = 0
            # 1. fixations
            ax = axes[r, t]
            ax.imshow(scene)
            _draw_boxes(ax, ep.boxes[i], size)
            _draw_path(ax, cx, cy, t)
            # the task head's guess = its argmax cell centre, a point: a cross, coloured by the
            # strict and the lenient localisation criteria
            if not tp:
                colour, verdict = "orange", "absent"
            elif found:
                colour, verdict = "lime", "found"
            elif near:
                colour, verdict = "yellow", f"found {ep.overlap_box_frac * 100:g}%"
            else:
                colour, verdict = "red", "miss"
            gx, gy = guess[0]
            ax.scatter([_px(gx, size)], [_px(gy, size)], s=90, marker="X", linewidths=0.8,
                       c=colour, edgecolors="black", zorder=5)
            ax.set_title(f"t{t}  {verdict}{pp}", fontsize=8)
            r += 1
            # 2. policy that chose fixation t
            if ep.policy is not None:
                ax = axes[r, t]
                if t == 0:
                    ax.imshow(scene * 0.55)
                    ax.set_title("centre (fixed)", fontsize=8)
                else:
                    pl = ep.policy[t, i]
                    prob = torch.softmax(pl.flatten().float(), 0).view_as(pl).numpy()
                    _overlay(ax, scene, prob, size)
                    ax.set_title(f"policy  H={_entropy(pl):.2f}", fontsize=8)
                _draw_boxes(ax, ep.boxes[i], size)
                _draw_path(ax, cx, cy, t)
                r += 1
            # 3. task-head heatmap after glimpse t
            ax = axes[r, t]
            prob = torch.softmax(lg.flatten().float(), 0).view(G, G).numpy()
            _overlay(ax, scene, prob, size)
            _draw_boxes(ax, ep.boxes[i], size)
            _draw_path(ax, cx, cy, t)
            ax.set_title(f"task head  KL={kl:.2f}" if tp else "task head (no target)", fontsize=8)
            r += 1
            # 4. canvas PCA after glimpse t
            ax = axes[r, t]
            ax.imshow(pca_rgb(pca, feats[t], G, G), extent=(0, size, size, 0), interpolation="nearest")
            _draw_boxes(ax, ep.boxes[i], size)
            ax.set_title("canvas PCA", fontsize=8)
    if title:
        fig.suptitle(title, fontsize=12)
    return fig


__all__ = ["SearchEpisode", "make_search_figure"]
