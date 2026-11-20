"""Teacher-forced likelihood evaluation on human scanpaths, scored by pysaliency (design §4.1).

    python -m canvit.harness.evaluate scanpath --cfg.model centre_bias --out results/cb.json

Conventions (pysaliency's and Kümmerer et al.'s, claude_dev/papers/scanpath_likelihood_eval.md):
every fixation with a history is scored -- fixation 0, the start fixation, is history only -- with
the full history; scores in bits/fixation. ``ll`` = information gain over pysaliency's
``UniformModel``, ``ig`` = over the centre bias (``CrossvalidatedBaselineModel``: leave-one-image-out
KDE on the evaluated split, bandwidth/eps fitted on the training split); ``nss`` / ``auc`` of the
conditional density. Each is averaged over images (``*_image``: per-image means, then over images --
the papers' headline) and over fixations (``*_fix``: pysaliency's default), overall and per
fixation index t (``*_t<t>``, fixation-averaged).
"""

from __future__ import annotations

import logging
import os
from dataclasses import asdict, dataclass, field, replace
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal

import numpy as np
import pysaliency
import torch
from pysaliency.baseline_utils import CrossvalidatedBaselineModel
from pysaliency.utils import average_values

from canvit.harness.config import JointPolicyConfig
from canvit.search.config import SearchConfig, search_policy_config

from .data import coco_search18, make_scanpath_loader
from .models import DensitySaliencyMap, fit_centre_bias

log = logging.getLogger(__name__)

RELEASE_FILES = ("COCOSearch18-images-TP.zip", "COCOSearch18-images-TA.zip", "COCOSearch18-fixations-TP.zip",
                 "coco_search18_fixations_TA_trainval.json")


def _env_path(name: str) -> Path:
    return Path(os.environ.get(name, ""))


@dataclass
class ScanpathConfig:
    model: Literal["uniform", "centre_bias", "canvit_policy", "canvit_heatmap"] = "centre_bias"
    """``canvit_policy``: a search run's learned scorer; ``canvit_heatmap``: its task head's heatmap
    (``canvit_models.py``). Both need ``ckpt``."""
    data_location: Path = field(default_factory=lambda: _env_path("SCANPATH_DATA_DIR"))
    """pysaliency's dataset location (``$SCANPATH_DATA_DIR``); COCO-Search18 is cached under it."""
    release_dirs: tuple[Path, ...] = field(default_factory=lambda: tuple(
        Path(p) for p in os.environ.get("COCO_SEARCH18_RELEASE_DIRS", "").split(":") if p))
    """Where the four COCO-Search18 release files are (``$COCO_SEARCH18_RELEASE_DIRS``, ':'-separated);
    only read when pysaliency's cache does not exist yet."""
    condition: Literal["both", "tp", "ta"] = "both"
    """Which scanpaths to score: target present, absent, or both (pysaliency's ``target_present``)."""
    part: Literal["validation", "train"] = "validation"
    grid: int = 32
    """Cells per axis the CanViT models predict on (the search policy's 32x32)."""
    scene_size: int = 512
    centre_bias_bandwidth: float | None = None
    """None = fit (with ``centre_bias_eps``) on the training split; ~minutes."""
    centre_bias_eps: float | None = None
    batch_size: int = 16
    num_workers: int = 4
    limit_scanpaths: int | None = None
    """Score only the first N scanpaths (smoke tests)."""
    device: str = "cuda"
    ckpt: Path | None = None
    """A search run's harness ``.pt`` (its policy is read from the ``.policy.pt`` beside it)."""
    search: SearchConfig = field(default_factory=SearchConfig)
    """The search run's task flags (``--cfg.search.model-repo``, head size, ...): the model is
    rebuilt with the training constructor and loaded strictly, so they must match the run's."""
    rl: JointPolicyConfig = field(default_factory=search_policy_config)
    """The search run's policy flags (``--cfg.rl.width``, ...); defaults = the search defaults."""


def load_coco_search18(cfg: ScanpathConfig):
    """pysaliency's COCO-Search18 split1 ``(stimuli_train, fixations_train, stimuli_val, fixations_val)``, with
    off-stimulus fixations clipped onto the stimulus in BOTH splits (the centre bias is fitted on train, and
    pysaliency's KDE refuses negative positions)."""
    files = {name: d / name for d in cfg.release_dirs for name in RELEASE_FILES if (d / name).exists()}
    st, ft, sv, fv = coco_search18(cfg.data_location, files)
    # Do not keep decoded images: the centre bias looks stimuli up by pixel hash (`stimulus_ids.index`), which
    # decodes EVERY stimulus once; with pysaliency's default cache that kept all 4326 training images (~22 GB,
    # the OOM of 2026-11-20). The hashes themselves stay cached.
    st.cached = sv.cached = False
    ft, n_train = clip_to_stimuli(st, ft)
    fv, n_val = clip_to_stimuli(sv, fv)
    log.info("clipped off-stimulus fixations onto the stimulus: %d train, %d validation", n_train, n_val)
    return st, ft, sv, fv


def select(fixations, *, condition: str, limit: int | None):
    """Scanpaths of one condition (and the first ``limit``), as pysaliency ``ScanpathFixations``."""
    keep = np.ones(len(fixations.scanpaths), dtype=bool)
    if condition != "both":
        present = np.asarray(fixations.scanpaths.scanpath_attributes["target_present"]).astype(bool)
        keep &= present == (condition == "tp")
    if limit is not None:
        keep &= np.cumsum(keep) <= limit
    return fixations.filter_scanpaths(np.flatnonzero(keep))


def clip_to_stimuli(stimuli, fixations):
    """-> (new ``ScanpathFixations`` with every fixation clipped onto its stimulus, number clipped).

    pysaliency does not clip, and an off-stimulus fixation indexes outside the map; the convention
    (pysaliency's ``clip_out_of_stimulus_fixations``, which fails on ``ScanpathFixations`` in 0.2.23)
    is to clip onto the edge. Clipped on a COPY of the scanpaths and rebuilt: ``ScanpathFixations``
    copies the coordinates at construction, so editing its scanpaths afterwards would not reach the
    scored fixations or their histories."""
    sp = fixations.scanpaths.copy()
    sizes = np.array([stimuli.sizes[int(n)] for n in sp.n], dtype=np.float64)  # (h, w) per scanpath
    clipped = 0
    for arr, lim in ((sp.xs, sizes[:, 1]), (sp.ys, sizes[:, 0])):
        data = arr._data
        valid = ~np.isnan(data)
        new = np.clip(data, 0, (lim - 1e-6)[:, None])
        clipped += int(np.sum(valid & (new != data)))
        data[valid] = new[valid]
    return pysaliency.ScanpathFixations(sp), clipped


def scores(model, stimuli, fixations, centre_bias) -> dict[str, float]:
    """pysaliency's per-fixation scores of ``model`` on the fixations with a history, averaged."""
    fx = fixations[fixations.scanpath_history_length > 0]
    # Stimulus by stimulus, so a model with a small map cache (the centre bias) computes each map once; no
    # average depends on the order (per-image means group by fx.n).
    fx = fx[np.argsort(fx.n, kind="stable")]
    uniform = pysaliency.UniformModel()
    ll_model = model.log_likelihoods(stimuli, fx) / np.log(2)
    per_fix = {
        "ll": ll_model - uniform.log_likelihoods(stimuli, fx) / np.log(2),
        "ig": ll_model - centre_bias.log_likelihoods(stimuli, fx) / np.log(2),
        "nss": DensitySaliencyMap(model).NSSs(stimuli, fx),
        "auc": DensitySaliencyMap(model).AUCs(stimuli, fx),
    }
    out: dict[str, float] = {"n_fixations": float(len(fx)), "n_scanpaths": float(len(np.unique(fx.scanpath_index)))}
    t = fx.scanpath_history_length
    for k, v in per_fix.items():
        out[f"{k}_image"] = float(average_values(v, fx, average="image"))
        out[f"{k}_fix"] = float(average_values(v, fx, average="fixation"))
        for i in range(1, int(t.max()) + 1):
            if (t == i).sum() >= 20:  # an index with fewer fixations is noise
                out[f"{k}_t{i}"] = float(v[t == i].mean())
    return out


def build_model(cfg: ScanpathConfig, stimuli, fixations, centre_bias, device: torch.device):
    if cfg.model == "uniform":
        return pysaliency.UniformModel()
    if cfg.model == "centre_bias":
        return centre_bias
    from .canvit_models import CanViTSearchModel, SearchEpisode, precompute
    if cfg.ckpt is None:
        raise ValueError(f"--cfg.model {cfg.model} needs --cfg.ckpt (a search run's .pt)")
    if cfg.search.scene_size != cfg.scene_size:
        raise ValueError(f"--cfg.search.scene-size {cfg.search.scene_size} != --cfg.scene-size {cfg.scene_size}: "
                         "the human fixations must be letterboxed into the scene the model was trained on")
    readout = "policy" if cfg.model == "canvit_policy" else "heatmap"
    ep = SearchEpisode.load(search_cfg=cfg.search, rl=cfg.rl, ckpt=cfg.ckpt, device=device,
                            need_policy=readout == "policy")
    loader = make_scanpath_loader(stimuli, fixations, scene_size=cfg.scene_size, batch_size=cfg.batch_size,
                                  num_workers=cfg.num_workers)
    return precompute(CanViTSearchModel(ep, readout), loader, grid=cfg.grid, scene_size=cfg.scene_size)


def run(cfg: ScanpathConfig) -> dict[str, Any]:
    device = torch.device(cfg.device if torch.cuda.is_available() else "cpu")
    st, ft, sv, fv = load_coco_search18(cfg)
    stimuli, fixations = (st, ft) if cfg.part == "train" else (sv, fv)
    fixations = select(fixations, condition=cfg.condition, limit=cfg.limit_scanpaths)
    if cfg.centre_bias_bandwidth is None or cfg.centre_bias_eps is None:
        fit = fit_centre_bias(st, ft)
    else:
        fit = {"bandwidth": cfg.centre_bias_bandwidth, "eps": cfg.centre_bias_eps}
    centre_bias = CrossvalidatedBaselineModel(stimuli, fixations, bandwidth=fit["bandwidth"], eps=fit["eps"],
                                              memory_cache_size=4)  # see scores(): fixations sorted by stimulus
    model = build_model(cfg, stimuli, fixations, centre_bias, device)
    metrics = scores(model, stimuli, fixations, centre_bias)
    return {
        "task": "scanpath",
        "model": cfg.model,
        "protocol": {k: (str(v) if isinstance(v, Path) else v) for k, v in asdict(replace(cfg)).items()}
        | {"centre_bias": fit, "off_stimulus_fixations": "clipped at load",
           "pysaliency": getattr(pysaliency, "__version__", "?")},
        "metrics": metrics,
        "evaluated_at": datetime.now(UTC).isoformat(),
    }


__all__ = ["ScanpathConfig", "clip_to_stimuli", "run", "scores", "select"]
