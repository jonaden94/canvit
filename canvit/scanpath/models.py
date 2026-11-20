"""Grid predictions as pysaliency models, and the fitted centre bias (design §3, §4.1).

pysaliency provides the models and the metrics; this file only adapts what it cannot know:

- :class:`GridScanpathModel` -- a ``pysaliency.ScanpathModel`` whose conditional log density comes
  from logits a CanViT episode precomputed on the G x G scene grid (``canvit_models.py``), keyed by
  pysaliency's ``(scanpath_index, history length)``. Each stimulus pixel takes the cell its centre
  falls in (letterbox of ``data.py``); a cell's mass is spread evenly over its pixels; cells with no
  stimulus pixel (letterbox padding) are dropped and the rest renormalised.
- :class:`DensitySaliencyMap` -- the density as a ``ScanpathSaliencyMapModel``, so pysaliency's
  NSS / AUC score the density itself (as in Kümmerer et al.), not the log density.
- :func:`fit_centre_bias` -- bandwidth and eps of pysaliency's ``CrossvalidatedBaselineModel``
  (leave-one-image-out KDE + uniform, Kümmerer's centre-bias baseline), chosen by log-likelihood
  on the TRAINING split; the model scored against is then built on the evaluated split.
"""

from __future__ import annotations

import logging
from functools import lru_cache

import numpy as np
import pysaliency
import torch
from pysaliency.baseline_utils import CrossvalidatedBaselineModel
from pysaliency.datasets import as_stimulus
from scipy.special import log_softmax

from .data import pixel_to_scene

log = logging.getLogger(__name__)


@lru_cache(maxsize=16)
def _pixel_cells(h: int, w: int, scene: int, grid: int) -> tuple[np.ndarray, np.ndarray]:
    """[h, w] flat cell index of every pixel centre, and [G*G] pixel count per cell."""
    yy, xx = np.meshgrid(np.arange(h) + 0.5, np.arange(w) + 0.5, indexing="ij")
    xy = pixel_to_scene(torch.from_numpy(np.stack([xx, yy], -1)), hw=(h, w), scene=scene).numpy()
    ij = np.clip(np.floor((xy + 1) / 2 * grid), 0, grid - 1).astype(np.int64)
    cell = ij[..., 1] * grid + ij[..., 0]
    return cell, np.bincount(cell.ravel(), minlength=grid * grid)


def grid_log_density(logits: np.ndarray, *, hw: tuple[int, int], scene: int) -> np.ndarray:
    """[G*G] scene-grid logits -> [h, w] log density over the stimulus pixels (sums to 1)."""
    grid = int(round(len(logits) ** 0.5))
    cell, count = _pixel_cells(hw[0], hw[1], scene, grid)
    on = count > 0
    logp = np.full(len(logits), -np.inf)
    logp[on] = log_softmax(logits[on].astype(np.float64))
    with np.errstate(divide="ignore"):
        return (logp - np.log(count))[cell]


class GridScanpathModel(pysaliency.ScanpathModel):
    """Precomputed grid logits as a pysaliency scanpath model. ``logits[(scanpath_index, t)]`` [G*G]
    is the prediction for fixation t after the human's fixations 0..t-1 (history length t)."""

    def __init__(self, logits: dict[tuple[int, int], np.ndarray], *, scene_size: int, name: str):
        super().__init__()
        self.logits, self.scene, self.name = logits, scene_size, name

    def conditional_log_density(self, stimulus, x_hist, y_hist, t_hist, attributes=None, out=None):
        key = (int(attributes["scanpath_index"]), len(x_hist))
        if key not in self.logits:
            raise KeyError(f"{self.name}: no prediction for scanpath {key[0]} after {key[1]} fixations "
                           "(was the episode run long enough?)")
        return grid_log_density(self.logits[key], hw=as_stimulus(stimulus).size, scene=self.scene)


class DensitySaliencyMap(pysaliency.ScanpathSaliencyMapModel):
    """A scanpath model's conditional density as a saliency map (for pysaliency's NSS and AUC)."""

    def __init__(self, model: pysaliency.ScanpathModel):
        super().__init__()
        self.model = model

    def conditional_saliency_map(self, stimulus, x_hist, y_hist, t_hist, attributes=None, out=None):
        return np.exp(self.model.conditional_log_density(stimulus, x_hist, y_hist, t_hist, attributes=attributes))


def fit_centre_bias(stimuli, fixations, *, n_images: int = 200, seed: int = 0,
                    bandwidths=tuple(10.0 ** np.arange(-2.2, -0.79, 0.2)),
                    epss=tuple(10.0 ** np.arange(-14, -1.9, 3.0))) -> dict[str, float]:
    """Grid search of ``CrossvalidatedBaselineModel(bandwidth, eps)``: mean log-likelihood (bits/fix)
    of the fixations on ``n_images`` random training stimuli, each scored by the KDE of all OTHER
    training stimuli. Returns the best ``{"bandwidth", "eps", "ll"}``; warns at a grid edge."""
    rng = np.random.default_rng(seed)
    pick = rng.choice(len(stimuli), size=min(n_images, len(stimuli)), replace=False)
    sub = fixations[np.isin(fixations.n, pick)]
    best = {"ll": -np.inf}
    for bw in bandwidths:
        for eps in epss:
            # memory_cache_size: pysaliency otherwise keeps every 14 MB map of every grid point (OOM at 20 GB)
            m = CrossvalidatedBaselineModel(stimuli, fixations, bandwidth=bw, eps=eps, memory_cache_size=4)
            ll = float(m.information_gain(stimuli, sub, baseline_model=pysaliency.UniformModel()))
            log.info("centre bias bandwidth=%.4f eps=%.0e: %.4f bits/fix over uniform", bw, eps, ll)
            if ll > best["ll"]:
                best = {"bandwidth": float(bw), "eps": float(eps), "ll": ll}
    for k, grid in (("bandwidth", bandwidths), ("eps", epss)):
        if best[k] in (grid[0], grid[-1]):
            log.warning("centre bias: %s=%g is at the edge of the searched grid", k, best[k])
    return best


__all__ = ["DensitySaliencyMap", "GridScanpathModel", "fit_centre_bias", "grid_log_density"]
