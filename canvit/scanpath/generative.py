"""Generative scanpath metrics for COCO-Search18 target-present search (design §4.2).

They score a model's OWN scanpaths against the human ones, so they need no likelihood. Every
scanpath here, model or human, is [B, L, 2] (x, y) in scene coordinates [-1, 1], NaN-padded,
with ``n_fix`` [B] valid fixations, and INCLUDES fixation 0, the start (centre) fixation, as
the reference scanpaths do (HAT prepends the centre to its model's scanpaths;
``scanpath_decode(center_initial=True)``). Target boxes are [B, N, 4] xyxy in [-1, 1],
NaN-padded rows.

Reference followed, read 2026-11-18 (the IRL and HAT copies of these functions agree):
  https://github.com/cvlab-stonybrook/HAT/blob/main/common/metrics.py
  https://github.com/cvlab-stonybrook/HAT/blob/main/common/utils.py  (get_num_step2target, get_CDF,
      compute_search_cdf, cutFixOnTarget)
  https://github.com/cvlab-stonybrook/HAT/blob/main/hat/evaluation.py (how they are combined)
  https://github.com/cvlab-stonybrook/Scanpath_Prediction/blob/master/irl_dcb/metrics.py (Yang 2020)

**Target-fixation probability (TFP)**, Yang et al. CVPR 2020. A fixation hits the target when it
lies STRICTLY inside a target box (``X > x0 and X < x0 + w``, likewise y). ``hit`` = index of the
first hitting fixation, counting the start fixation as index 0 (the reference's ``num_step`` is
``hit + 1``). ``TFP(k)`` = fraction of trials with ``hit <= k``, i.e. target found within the
start fixation plus k more; the curve has entries k = 0..K, as the reference's
``compute_search_cdf(.., max_step=K)`` (K + 1 entries; HAT reports ``TFP_top{k}`` = entry k for
k >= 1). The reference averages the curve PER TASK (target category) and then over tasks;
:func:`tfp_curve` does the same. ``tfp_auc`` = the reference's ``compute_cdf_auc``,
``v[0] + v[-1] + sum(v[1:-1])``, which is the plain sum of the curve (not a trapezoid).
``probability_mismatch`` = ``sum_k |TFP_model(k) - TFP_human(k)|`` over all K + 1 entries, no
normalisation (HAT ``prob_mismatch``).

**Sequence Score (SS)**, Yang 2020 after Borji & Itti ICCV 2013 / Jiang et al. TNNLS 2016. All
human fixations on one (image, task) are clustered by MeanShift; each scanpath becomes the string
of its fixations' cluster ids; two strings score by Needleman-Wunsch with 0/1 match similarity and
gap 0, divided by the longer length (so it is LCS length / max length, in [0, 1]); a model
scanpath scores the mean over the humans. The cluster-building code was never released (HAT
README); its recipe is the authors' description in
https://github.com/cvlab-stonybrook/Scanpath_Prediction/issues/24 (ouyangzhibo, 2022-06-26 and
-07-28): bandwidth = ``sklearn.cluster.estimate_bandwidth`` (default quantile 0.3) of the image's
human fixations times the factor in (0.2, 0.5, 0.8, 1.0, 1.2, 1.5, 1.8) that maximises the
improved interaction rate ``(N_between - N_within) / N_clusters`` of consecutive fixations.

Coordinates: the reference fixations are pixels of the 512x320 resized image; ours are the same
image letterboxed isotropically into [-1, 1] (1680x1050 -> 512x320 content, one scale factor).
The bandwidth is estimated from the data, and ``estimate_bandwidth`` and ``MeanShift`` (flat
kernel, distance-based seeding, ``stop_thresh = 1e-3 * bandwidth``) are equivariant under an
isotropic scale + shift, so the clusters are the same as in pixels: no bandwidth conversion is
needed (checked against sklearn 1.8.0 source 2026-11-18).

Checked 2026-11-18 on split1 valid humans (3258 trials, 326 (image, task) pairs): the TFP curve
(K = 6) from scene coordinates equals the reference code run on the raw pixel JSON,
[0.007, 0.406, 0.727, 0.839, 0.875, 0.883, 0.887], AUC 4.624 (the published human 5.200 is on
the test split); human SS (leave-self-out) 0.491 vs the authors' corrected 0.476 on test.

Deviations from the reference, both by necessity:
- the clusters are fitted on target-present human fixations only (the reference also pooled the
  target-absent fixations of the image; this copy of the dataset has no TA data, design §2.1);
- truncation is the caller's: pass ``n_fix`` cut as wanted. HAT cuts TP model scanpaths at the
  first target hit (:func:`cut_at_target`) and truncates the model string to ``max_traj_length``
  but not the human ones; IRL truncates both to ``max_step``.

TODO: MultiMatch, SemSS (needs COCO panoptic labels), SS with durations (no durations here).
"""

from __future__ import annotations

import math

import numpy as np
import torch
from sklearn.cluster import MeanShift, estimate_bandwidth
from torch import Tensor

BANDWIDTH_FACTORS = (0.2, 0.5, 0.8, 1.0, 1.2, 1.5, 1.8)


def valid_mask(fix: Tensor, n_fix: Tensor) -> Tensor:
    """[B, L] True on the ``n_fix`` valid fixations; refuses lengths out of range or NaN inside them."""
    if fix.ndim != 3 or fix.shape[-1] != 2 or n_fix.shape != fix.shape[:1]:
        raise ValueError(f"want fix [B, L, 2] and n_fix [B], got {tuple(fix.shape)} and {tuple(n_fix.shape)}")
    if ((n_fix < 1) | (n_fix > fix.shape[1])).any():
        raise ValueError(f"n_fix must be in [1, {fix.shape[1]}], got {n_fix.tolist()}")
    valid = torch.arange(fix.shape[1], device=fix.device) < n_fix[:, None]
    if not torch.isfinite(fix[valid]).all():
        raise ValueError("non-finite coordinates inside the first n_fix fixations")
    return valid


# --- target-fixation probability ---------------------------------------------------------------

def target_hit_index(fix: Tensor, n_fix: Tensor, boxes: Tensor) -> Tensor:
    """[B] float: index of the first valid fixation strictly inside any target box (0 = the start
    fixation), ``inf`` if none is."""
    valid = valid_mask(fix, n_fix)
    if boxes.ndim != 3 or boxes.shape[0] != fix.shape[0] or boxes.shape[-1] != 4:
        raise ValueError(f"want boxes [B, N, 4], got {tuple(boxes.shape)}")
    if not torch.isfinite(boxes).all(-1).any(-1).all():
        raise ValueError("a trial without a target box has no TFP (target-absent trials are not scored)")
    x, y = fix[..., 0, None], fix[..., 1, None]  # [B, L, 1]
    b = boxes[:, None]  # [B, 1, N, 4]; NaN rows compare False
    inside = ((x > b[..., 0]) & (x < b[..., 2]) & (y > b[..., 1]) & (y < b[..., 3])).any(-1) & valid
    first = inside.int().argmax(1).float()  # argmax returns the first maximum
    return torch.where(inside.any(1), first, torch.full_like(first, math.inf))


def cut_at_target(hit: Tensor, n_fix: Tensor) -> Tensor:
    """``n_fix`` cut so each scanpath ends at its first target hit (HAT ``cutFixOnTarget``)."""
    return torch.where(torch.isfinite(hit), torch.minimum(n_fix, hit.long() + 1), n_fix)


def tfp_curve(hit: Tensor, task: Tensor, *, max_fix: int) -> Tensor:
    """[max_fix + 1]: entry k = fraction of trials with the target fixated by fixation k (start = 0),
    averaged within each task, then over tasks (reference ``compute_search_cdf``). Pass ``task``
    all-equal to pool the trials instead."""
    k = torch.arange(max_fix + 1, device=hit.device)
    found = (hit[:, None] <= k).float()  # [B, K + 1]
    return torch.stack([found[task == t].mean(0) for t in task.unique()]).mean(0)


def tfp_auc(curve: Tensor) -> Tensor:
    """Area under a TFP curve as the reference computes it: the sum of its entries."""
    return curve.sum()


def probability_mismatch(model_curve: Tensor, human_curve: Tensor) -> Tensor:
    """``sum_k |TFP_model(k) - TFP_human(k)|`` over all entries, unnormalised."""
    if model_curve.shape != human_curve.shape:
        raise ValueError(f"TFP curves differ in length: {tuple(model_curve.shape)} vs {tuple(human_curve.shape)}")
    return (model_curve - human_curve).abs().sum()


# --- sequence score ----------------------------------------------------------------------------

def _points(fix: Tensor, n_fix: Tensor) -> list[np.ndarray]:
    """One [n_fix, 2] float64 array per scanpath, padding dropped."""
    valid_mask(fix, n_fix)
    f = fix.detach().double().cpu().numpy()
    return [f[i, :n] for i, n in enumerate(n_fix.tolist())]


def _strings(ms: MeanShift, paths: list[np.ndarray]) -> list[list[int]]:
    return [ms.predict(p).tolist() for p in paths]


def improved_interaction_rate(ms: MeanShift, strings: list[list[int]]) -> float:
    """``(N_between - N_within) / N_clusters`` over consecutive fixation pairs (Jiang et al. 2016)."""
    same = sum(a == b for s in strings for a, b in zip(s, s[1:]))
    total = sum(len(s) - 1 for s in strings)
    return ((total - same) - same) / len(ms.cluster_centers_)


def fit_fixation_clusters(human_fix: Tensor, human_n_fix: Tensor) -> MeanShift:
    """MeanShift on all human fixations of ONE (image, task), bandwidth chosen as in the reference."""
    paths = _points(human_fix, human_n_fix)
    pts = np.concatenate(paths)
    base = estimate_bandwidth(pts)
    if base <= 0:
        raise ValueError(f"estimate_bandwidth is 0 on {len(pts)} fixations: at least 30% of them coincide")
    best, best_rate = None, -math.inf
    for f in BANDWIDTH_FACTORS:  # first maximum wins, as np.argmax in the reference
        ms = MeanShift(bandwidth=base * f).fit(pts)
        rate = improved_interaction_rate(ms, _strings(ms, paths))
        if rate > best_rate:
            best, best_rate = ms, rate
    assert best is not None
    return best


def nw_score(a: list[int], b: list[int]) -> float:
    """Needleman-Wunsch, 0/1 similarity, gap 0, divided by the longer length (reference ``nw_matching``)."""
    F = [[0.0] * (len(b) + 1) for _ in range(len(a) + 1)]
    for i in range(1, len(a) + 1):
        for j in range(1, len(b) + 1):
            F[i][j] = max(F[i - 1][j - 1] + float(a[i - 1] == b[j - 1]), F[i - 1][j], F[i][j - 1])
    return F[len(a)][len(b)] / max(len(a), len(b))


def sequence_score(model_fix: Tensor, model_n_fix: Tensor, human_fix: Tensor, human_n_fix: Tensor) -> Tensor:
    """[B_model] SS of each model scanpath against the humans of ONE (image, task): the mean
    ``nw_score`` over the human scanpaths, on clusters fitted to the human fixations."""
    ms = fit_fixation_clusters(human_fix, human_n_fix)
    gt = _strings(ms, _points(human_fix, human_n_fix))
    pred = _strings(ms, _points(model_fix, model_n_fix))
    return torch.tensor([np.mean([nw_score(p, g) for g in gt]) for p in pred], dtype=torch.float64)


def human_sequence_score(human_fix: Tensor, human_n_fix: Tensor) -> Tensor:
    """[H] each human scanpath against the OTHER humans of the (image, task) (the authors'
    correction in issue #24: comparing a scanpath with itself inflated the published 0.490)."""
    ms = fit_fixation_clusters(human_fix, human_n_fix)
    s = _strings(ms, _points(human_fix, human_n_fix))
    if len(s) < 2:
        raise ValueError("human consistency needs at least 2 human scanpaths")
    return torch.tensor([np.mean([nw_score(s[i], s[j]) for j in range(len(s)) if j != i])
                         for i in range(len(s))], dtype=torch.float64)


__all__ = ["BANDWIDTH_FACTORS", "cut_at_target", "fit_fixation_clusters", "human_sequence_score",
           "improved_interaction_rate", "nw_score", "probability_mismatch", "sequence_score",
           "target_hit_index", "tfp_auc", "tfp_curve", "valid_mask"]
