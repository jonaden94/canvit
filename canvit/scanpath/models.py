"""Grid predictions as pysaliency models, and the fitted centre bias (design §3, §4.1).

pysaliency provides the models and the metrics; this file only adapts what it cannot know:

- :class:`GridScanpathModel` -- a ``pysaliency.ScanpathModel`` whose conditional log density comes
  from logits a CanViT episode precomputed on the G x G scene grid (``canvit_models.py``), keyed by
  pysaliency's ``(scanpath_index, history length)``. Each stimulus pixel takes the cell its centre
  falls in (letterbox of ``data.py``). The cells are equal in SCENE area, so a cell's logit is a density
  per area: every pixel of a cell gets exp(logit), renormalised over the stimulus pixels -- a cell the
  letterbox edge cuts keeps its per-pixel density and loses mass with its lost pixels; cells with no
  stimulus pixel (letterbox padding) get none. (Before 2026-11-23 a cut cell kept the whole cell's mass,
  concentrated on its few pixels.)
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


def _log_count(count: np.ndarray) -> np.ndarray:
    with np.errstate(divide="ignore"):
        return np.log(count)  # -inf for cells without a stimulus pixel


def grid_log_density(logits: np.ndarray, *, hw: tuple[int, int], scene: int) -> np.ndarray:
    """[G*G] scene-grid logits -> [h, w] log density over the stimulus pixels (sums to 1): every pixel gets
    its cell's logit, normalised over the pixels (= cell probabilities ``softmax(logits + log pixels)``,
    spread evenly over each cell's pixels)."""
    grid = int(round(len(logits) ** 0.5))
    cell, count = _pixel_cells(hw[0], hw[1], scene, grid)
    logp = log_softmax(logits.astype(np.float64) + _log_count(count))
    with np.errstate(invalid="ignore"):  # -inf - -inf in cells no pixel reads
        return (logp - _log_count(count))[cell]


class GridScanpathModel(pysaliency.ScanpathModel):
    """Precomputed grid logits as a pysaliency scanpath model. ``logits[(scanpath_index, t)]`` [G*G]
    is the prediction for fixation t after the human's fixations 0..t-1 (history length t).

    ``temperature`` divides the logits: one global value, or per history length t (missing t: 1). It is a
    DIAGNOSTIC only (owner, 2026-11-20: one global entropy alpha in training; tau shows how far the policy's
    sharpness is from the human-optimal one) and never feeds back into training."""

    def __init__(self, logits: dict[tuple[int, int], np.ndarray], *, scene_size: int, name: str,
                 temperature: float | dict[int, float] = 1.0):
        super().__init__()
        self.logits, self.scene, self.name, self.temperature = logits, scene_size, name, temperature

    def tau(self, t: int) -> float:
        return self.temperature if isinstance(self.temperature, float | int) else self.temperature.get(t, 1.0)

    def with_temperature(self, temperature: float | dict[int, float]) -> GridScanpathModel:
        return GridScanpathModel(self.logits, scene_size=self.scene, name=self.name, temperature=temperature)

    def conditional_log_density(self, stimulus, x_hist, y_hist, t_hist, attributes=None, out=None):
        key = (int(attributes["scanpath_index"]), len(x_hist))
        if key not in self.logits:
            raise KeyError(f"{self.name}: no prediction for scanpath {key[0]} after {key[1]} fixations "
                           "(was the episode run long enough?)")
        return grid_log_density(self.logits[key] / self.tau(key[1]), hw=as_stimulus(stimulus).size,
                                scene=self.scene)

    def save(self, path) -> None:
        """The predictions as ``.npz`` (keys [N, 2] = (scanpath_index, t), float16 logits), e.g. for figures."""
        keys = np.array(list(self.logits), dtype=np.int64).reshape(-1, 2)
        np.savez_compressed(path, keys=keys, logits=np.stack([self.logits[tuple(k)] for k in keys]).astype(np.float16),
                            scene_size=self.scene, name=self.name)

    @classmethod
    def load(cls, path) -> GridScanpathModel:
        z = np.load(path)
        return cls({(int(a), int(b)): v.astype(np.float32) for (a, b), v in zip(z["keys"], z["logits"])},
                   scene_size=int(z["scene_size"]), name=str(z["name"]))


def _scored_cells(model: GridScanpathModel, stimuli, fixations):
    """For every fixation with a history: its logits [N, G*G], the cell it landed in [N], log(pixels of the
    image in each cell) [N, G*G] (-inf: none), and its history length [N]. (pysaliency's LL = log p(cell) -
    log(pixels in cell); the second term does not depend on the temperature, so fitting on cells is exact.)"""
    fx = without_start_fixations(fixations)
    logits, cells, logcount = [], [], []
    for i in range(len(fx)):
        h, w = stimuli.sizes[int(fx.n[i])]
        lg = model.logits[(int(fx.scanpath_index[i]), int(fx.scanpath_history_length[i]))]
        cell, count = _pixel_cells(int(h), int(w), model.scene, int(round(len(lg) ** 0.5)))
        logits.append(lg)
        cells.append(cell[fx.y_int[i], fx.x_int[i]])
        logcount.append(_log_count(count))
    return (np.stack(logits).astype(np.float64), np.array(cells), np.stack(logcount),
            np.asarray(fx.scanpath_history_length))


def _cell_log_probs(logits: np.ndarray, logcount: np.ndarray, tau) -> np.ndarray:
    """log p(cell) at temperature ``tau`` (scalar or [N, 1]): the temperature scales the logits, not the area."""
    return log_softmax(logits / tau + logcount, axis=1)


def fit_temperature(model: GridScanpathModel, stimuli, fixations, *, per_index: bool = False,
                    min_fixations: int = 50) -> float | dict[int, float]:
    """The temperature maximising the mean log-likelihood of the human fixations (fit on the TRAINING split).
    ``per_index``: one per history length t with >= ``min_fixations`` fixations (a secondary diagnostic)."""
    from scipy.optimize import minimize_scalar
    logits, cells, logcount, t = _scored_cells(model, stimuli, fixations)
    rows = np.arange(len(cells))

    def fit(mask) -> float:
        def nll(log_tau):
            logp = _cell_log_probs(logits[mask], logcount[mask], float(np.exp(log_tau)))
            return -logp[rows[: mask.sum()], cells[mask]].mean()
        log_tau = minimize_scalar(nll, bounds=(-4.0, 4.0), method="bounded").x
        if abs(abs(log_tau) - 4.0) < 1e-3:  # e.g. an uninformative map: flattening it is all the fit can do
            log.warning("temperature fit at the edge of [e^-4, e^4]: tau=%.3g", np.exp(log_tau))
        return float(np.exp(log_tau))

    if not per_index:
        return fit(np.ones(len(cells), dtype=bool))
    return {int(i): fit(t == i) for i in np.unique(t) if (t == i).sum() >= min_fixations}


def policy_entropy(model: GridScanpathModel, stimuli, fixations) -> tuple[np.ndarray, np.ndarray]:
    """Entropy (nats) of the predicted next-fixation distribution over the image's cells (``p(cell)`` as in
    :func:`grid_log_density`), at the model's temperature, per scored fixation; and the history length t."""
    logits, _, logcount, t = _scored_cells(model, stimuli, fixations)
    taus = np.array([model.tau(int(i)) for i in t])[:, None]
    logp = _cell_log_probs(logits, logcount, taus)
    return -(np.exp(logp) * np.where(np.isfinite(logp), logp, 0.0)).sum(1), t


class DensitySaliencyMap(pysaliency.ScanpathSaliencyMapModel):
    """A scanpath model's conditional density as a saliency map (for pysaliency's NSS and AUC)."""

    def __init__(self, model: pysaliency.ScanpathModel):
        super().__init__()
        self.model = model

    def conditional_saliency_map(self, stimulus, x_hist, y_hist, t_hist, attributes=None, out=None):
        d = np.exp(self.model.conditional_log_density(stimulus, x_hist, y_hist, t_hist, attributes=attributes))
        # A flat density is not exactly flat after exp(log): pysaliency's NSS then divides rounding noise by a
        # ~1e-22 std and returns +-1 (seen for UniformModel, 2026-11-20). Make it exactly constant -> NSS 0.
        if d.max() - d.min() <= 1e-9 * d.max():
            d = np.full_like(d, d.mean())
        return d


def without_start_fixations(fixations):
    """The fixations with a history: the start fixation of every scanpath is forced to the display centre,
    so in a centre bias it is a spike at one pixel, not viewing behaviour. Kept out of the centre bias's
    KDE and of its fit (measured 2026-11-20: it cost ~0.15 bits/fix and made the fit pick the narrowest
    bandwidth on the grid)."""
    return fixations[fixations.scanpath_history_length > 0]


def fit_centre_bias(stimuli, fixations, *, n_images: int = 200, seed: int = 0,
                    bandwidths=(0.01, 0.016, 0.025, 0.04, 0.063),
                    epss=(1e-7, 1e-4)) -> dict[str, float]:
    """Grid search of ``CrossvalidatedBaselineModel(bandwidth, eps)``: mean log-likelihood (bits/fix)
    of the fixations on ``n_images`` random training stimuli, each scored by the KDE of all OTHER
    training stimuli; start fixations excluded throughout (:func:`without_start_fixations`). Returns the
    best ``{"bandwidth", "eps", "ll"}``; warns at a grid edge. Grouping the leave-out by photo instead of
    stimulus (a photo can be a stimulus under several cues) changed the held-out score by <= 0.01 bits."""
    fixations = without_start_fixations(fixations)
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


__all__ = ["DensitySaliencyMap", "GridScanpathModel", "fit_centre_bias", "fit_temperature", "grid_log_density",
           "policy_entropy", "without_start_fixations"]
