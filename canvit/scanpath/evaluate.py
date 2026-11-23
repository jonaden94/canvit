"""Teacher-forced likelihood evaluation on human scanpaths, scored by pysaliency (design §4.1).

    python -m canvit.harness.evaluate scanpath --cfg.dataset mit1003 --cfg.model centre_bias --out results/cb.json
    python -m canvit.harness.evaluate scanpath --cfg.dataset coco_search18 --cfg.condition tp \
        --cfg.model canvit_policy --cfg.ckpt <best.pt>        # the checkpoint alone defines the model

Datasets (``data.DATASETS``): the validation split where one exists, else the whole set; the centre bias is
fitted on the training split where one exists, else on the evaluated set (``data.fit_part``).

Conventions (pysaliency's and Kümmerer et al.'s, notes: canvit_claude/papers/, outside the repo):
every fixation with a history is scored -- fixation 0, the start fixation, is history only -- with
the full history; scores in bits/fixation. ``ll`` = information gain over pysaliency's
``UniformModel``, ``ig`` = over the centre bias (``CrossvalidatedBaselineModel``: leave-one-image-out
KDE on the evaluated split, bandwidth/eps fitted on ``fit_part``); ``nss`` / ``auc`` of the
conditional density. Each is averaged over images (``*_image``: per-image means, then over images --
the papers' headline) and over fixations (``*_fix``: pysaliency's default), overall and per
fixation index t (``*_t<t>``, fixation-averaged).
"""

from __future__ import annotations

import json
import logging
import os
from dataclasses import asdict, dataclass, field, replace
from datetime import UTC, datetime
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Literal

import numpy as np
import pysaliency
import torch
from pysaliency.baseline_utils import CrossvalidatedBaselineModel
from pysaliency.utils import average_values

from .data import DATASETS, _task_names, eval_part, fit_part, load_dataset, make_scanpath_loader
from .models import (
    DensitySaliencyMap,
    GridScanpathModel,
    fit_centre_bias,
    fit_temperature,
    policy_entropy,
    without_start_fixations,
)

log = logging.getLogger(__name__)

def _env_path(name: str) -> Path:
    return Path(os.environ.get(name, ""))


@dataclass
class ScanpathConfig:
    model: Literal["uniform", "centre_bias", "canvit_policy", "canvit_heatmap"] = "centre_bias"
    """``canvit_policy``: a run's learned policy (any task); ``canvit_heatmap``: a search run's task head
    (``canvit_models.py``). Both need ``ckpt``. A search model on free-viewing data is read as the uniform
    mixture over its cues (``..._cuemix``, owner decision 2026-11-23)."""
    dataset: Literal["coco_search18", "coco_freeview", "daemons", "mit1003", "cat2000", "figrim"] = "coco_search18"
    part: Literal["train", "validation", "all"] | None = None
    """None = ``data.eval_part(dataset)``: validation where the dataset has a split, else all."""
    data_location: Path = field(default_factory=lambda: _env_path("SCANPATH_DATA_DIR"))
    """pysaliency's dataset location (``$SCANPATH_DATA_DIR``); every dataset is cached under it."""
    release_dirs: tuple[Path, ...] = field(default_factory=lambda: tuple(
        Path(p) for p in os.environ.get("COCO_SEARCH18_RELEASE_DIRS", "").split(":") if p))
    """Where the four COCO-Search18 release files are (``$COCO_SEARCH18_RELEASE_DIRS``, ':'-separated);
    only read when pysaliency's cache does not exist yet."""
    condition: Literal["both", "tp", "ta"] = "both"
    """COCO-Search18: which scanpaths to score: target present, absent, or both (``target_present``)."""
    grid: int = 32
    """Cells per axis the CanViT models predict on (the search policy's 32x32)."""
    scene_size: int = 512
    centre_bias_fit: Path | None = None
    """A JSON ``{bandwidth, eps, ll}`` of the centre bias: read if it exists, else fitted on ``fit_part`` and
    written there (so every model of a dataset shares one fit). None: fitted, not written (~minutes)."""
    batch_size: int = 16
    num_workers: int = 4
    limit_scanpaths: int | None = None
    """Score only the first N scanpaths (smoke tests)."""
    device: str = "cuda"
    ckpt: Path | None = None
    """A self-contained harness checkpoint (run config, architecture and policy inside; canvit main b029fd3+)."""
    save_predictions: Path | None = None
    """CanViT models: write the predicted maps here (``.npz``, ``GridScanpathModel.save``), e.g. for figures."""
    fit_temperature_scanpaths: int = 0
    """CanViT models, DIAGNOSTIC: > 0 runs the model on this many scanpaths of ``fit_part`` (same condition),
    fits one global softmax temperature and one per fixation index by log-likelihood there, and reports the
    tempered likelihoods on the evaluated split under ``diagnostics``. Never fed back into training (owner,
    2026-11-20). On a dataset without a split the fit is in-sample (one parameter)."""


def split_name(cfg: ScanpathConfig) -> str:
    """The evaluated split in result names: ``<condition>_<part>`` for one COCO-Search18 condition, else the part."""
    part = cfg.part or eval_part(cfg.dataset)
    return f"{cfg.condition}_{part}" if cfg.condition != "both" else part


def load(cfg: ScanpathConfig, part: str):
    return load_dataset(cfg.dataset, part, location=cfg.data_location, release_dirs=cfg.release_dirs)


def select(fixations, *, condition: str, limit: int | None):
    """Scanpaths of one condition (and the first ``limit``), as pysaliency ``ScanpathFixations``."""
    keep = np.ones(len(fixations.scanpaths), dtype=bool)
    if condition != "both":
        present = np.asarray(fixations.scanpaths.scanpath_attributes["target_present"]).astype(bool)
        keep &= present == (condition == "tp")
    if limit is not None:
        keep &= np.cumsum(keep) <= limit
    return fixations.filter_scanpaths(np.flatnonzero(keep))


def scored_fixations(fixations):
    """The fixations that are scored -- every fixation with a history -- sorted by stimulus (stable), so a model
    with a small map cache (the centre bias) computes each map once; no average depends on the order."""
    fx = fixations[fixations.scanpath_history_length > 0]
    return fx[np.argsort(fx.n, kind="stable")]


def per_fixation_scores(model, stimuli, fx, centre_bias) -> dict[str, np.ndarray]:
    """pysaliency's per-fixation scores of ``model`` on ``fx`` (:func:`scored_fixations`, or a shard of it):
    ``ll`` / ``ig`` in bits over the uniform model / the centre bias, ``nss`` / ``auc`` of the conditional
    density, and ``n`` (stimulus), ``t`` (history length), ``scanpath`` (index) for :func:`aggregate`. Shards
    concatenate (key by key)."""
    ll_model = model.log_likelihoods(stimuli, fx) / np.log(2)
    return {
        "ll": ll_model - pysaliency.UniformModel().log_likelihoods(stimuli, fx) / np.log(2),
        "ig": ll_model - centre_bias.log_likelihoods(stimuli, fx) / np.log(2),
        "nss": DensitySaliencyMap(model).NSSs(stimuli, fx),
        "auc": DensitySaliencyMap(model).AUCs(stimuli, fx),
        "n": np.asarray(fx.n), "t": np.asarray(fx.scanpath_history_length), "scanpath": np.asarray(fx.scanpath_index),
    }


def aggregate(per_fix: dict[str, np.ndarray]) -> dict[str, float]:
    """The averages of :func:`per_fixation_scores`: over images (``*_image``: per-image means, then over images
    -- the papers' headline) and over fixations (``*_fix``: pysaliency's default), overall and per fixation
    index t (``*_t<t>``, fixation-averaged, for indices with >= 20 fixations)."""
    n, t = per_fix["n"], per_fix["t"]
    by_image = SimpleNamespace(n=n)  # all pysaliency's average_values reads
    out: dict[str, float] = {"n_fixations": float(len(n)), "n_scanpaths": float(len(np.unique(per_fix["scanpath"])))}
    for k, v in per_fix.items():
        if k in ("n", "t", "scanpath"):
            continue
        out[f"{k}_image"] = float(average_values(v, by_image, average="image"))
        out[f"{k}_fix"] = float(average_values(v, by_image, average="fixation"))
        for i in range(1, int(t.max()) + 1):
            if (t == i).sum() >= 20:  # an index with fewer fixations is noise
                out[f"{k}_t{i}"] = float(v[t == i].mean())
    return out


def scores(model, stimuli, fixations, centre_bias) -> dict[str, float]:
    """pysaliency's per-fixation scores of ``model`` on the fixations with a history, averaged."""
    fx = scored_fixations(fixations)
    return aggregate(per_fixation_scores(model, stimuli, fx, centre_bias))


def centre_bias_model(stimuli, fixations, fit: dict[str, float]) -> CrossvalidatedBaselineModel:
    """The centre bias the scores are relative to: pysaliency's leave-one-image-out KDE (+ uniform, ``fit``) on the
    EVALUATED fixations (after ``select``), start fixations excluded (``models.without_start_fixations``)."""
    return CrossvalidatedBaselineModel(stimuli, without_start_fixations(fixations), bandwidth=fit["bandwidth"],
                                       eps=fit["eps"], memory_cache_size=4)  # fixations sorted by stimulus


def build_model(cfg: ScanpathConfig, stimuli, fixations, centre_bias, device: torch.device):
    """-> (the model scored on ``fixations``, and for CanViT models a ``predict(stimuli, fixations)`` that runs
    the same model on other scanpaths -- the temperature diagnostic needs the training split -- else None,
    and for CanViT models the checkpoint's provenance, else None)."""
    if cfg.model == "uniform":
        return pysaliency.UniformModel(), None, None
    if cfg.model == "centre_bias":
        return centre_bias, None, None
    from .canvit_models import CanViTModel, PolicyEpisode, precompute
    if cfg.ckpt is None:
        raise ValueError(f"--cfg.model {cfg.model} needs --cfg.ckpt (a run's .pt)")
    readout = "policy" if cfg.model == "canvit_policy" else "heatmap"
    ep = PolicyEpisode.load(cfg.ckpt, device=device, need_policy=readout == "policy")
    c = ep.task.cfg
    task_scene = getattr(c, "scene_size", None) or getattr(c, "scene_resolution", None)
    if task_scene is not None and task_scene != cfg.scene_size:
        raise ValueError(f"the {ep.task.name} run's scene is {task_scene} px but --cfg.scene-size is "
                         f"{cfg.scene_size}: the human fixations must be letterboxed into the scene the model was "
                         "trained on")
    model = CanViTModel(ep, readout, cue_mixture=ep.is_search and _task_names(stimuli) is None)

    def predict(stimuli, fixations):
        loader = make_scanpath_loader(stimuli, fixations, scene_size=cfg.scene_size, batch_size=cfg.batch_size,
                                      num_workers=cfg.num_workers)
        return precompute(model, loader, grid=cfg.grid, scene_size=cfg.scene_size)

    return predict(stimuli, fixations), predict, ep.provenance


def centre_bias_fit(cfg: ScanpathConfig) -> dict[str, float]:
    """The centre bias's ``{bandwidth, eps, ll}``: from ``cfg.centre_bias_fit`` if it exists, else fitted on the
    dataset's ``fit_part`` (both conditions) and written there."""
    path = cfg.centre_bias_fit
    if path is not None and path.exists():
        return json.loads(path.read_text())
    st, ft = load(cfg, fit_part(cfg.dataset))
    fit = fit_centre_bias(st, ft) | {"dataset": cfg.dataset, "part": fit_part(cfg.dataset)}
    if path is not None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(fit, indent=1))
    return fit


def grid_diagnostics(model, stimuli, fixations, centre_bias, *, train=None) -> dict[str, Any]:
    """For a grid model: the policy's entropy per fixation index (nats), and with ``train = (stimuli,
    fixations, model on them)`` the fitted temperatures and the tempered likelihoods (DIAGNOSTIC)."""
    H, t = policy_entropy(model, stimuli, fixations)
    out: dict[str, Any] = {"entropy_fix": float(H.mean())} | {
        f"entropy_t{i}": float(H[t == i].mean()) for i in np.unique(t) if (t == i).sum() >= 20}
    if train is None:
        return out
    tau = fit_temperature(train[2], train[0], train[1])
    tau_t = fit_temperature(train[2], train[0], train[1], per_index=True)
    fx = without_start_fixations(fixations)
    uniform_ll = pysaliency.UniformModel().log_likelihoods(stimuli, fx)
    cb_ll = centre_bias.log_likelihoods(stimuli, fx)
    for name, temp in (("global", tau), ("per_index", tau_t)):
        ll = model.with_temperature(temp).log_likelihoods(stimuli, fx)

        def bits(ref):
            return float(average_values((ll - ref) / np.log(2), fx, average="image"))

        out[f"tempered_{name}"] = {"temperature": temp, "ll_image": bits(uniform_ll), "ig_image": bits(cb_ll)}
    H_tau, _ = policy_entropy(model.with_temperature(tau), stimuli, fixations)
    out["tempered_global"]["entropy_fix"] = float(H_tau.mean())
    return out


def run(cfg: ScanpathConfig) -> dict[str, Any]:
    device = torch.device(cfg.device if torch.cuda.is_available() else "cpu")
    part = cfg.part or eval_part(cfg.dataset)
    if cfg.condition != "both" and cfg.dataset != "coco_search18":
        raise ValueError(f"--cfg.condition {cfg.condition}: only COCO-Search18 has target-present/absent trials")
    if part not in DATASETS[cfg.dataset]:
        raise ValueError(f"{cfg.dataset} has the parts {DATASETS[cfg.dataset]}, not {part!r}")
    stimuli, fixations = load(cfg, part)
    fixations = select(fixations, condition=cfg.condition, limit=cfg.limit_scanpaths)
    fit = centre_bias_fit(cfg)
    centre_bias = centre_bias_model(stimuli, fixations, fit)
    model, predict, provenance = build_model(cfg, stimuli, fixations, centre_bias, device)
    metrics = scores(model, stimuli, fixations, centre_bias)
    diagnostics: dict[str, Any] = {}
    if isinstance(model, GridScanpathModel):
        if cfg.save_predictions is not None:
            model.save(cfg.save_predictions)
        train = None
        if cfg.fit_temperature_scanpaths > 0:
            assert predict is not None
            st, ft = (stimuli, fixations) if fit_part(cfg.dataset) == part else load(cfg, fit_part(cfg.dataset))
            ft_sub = select(ft, condition=cfg.condition, limit=cfg.fit_temperature_scanpaths)
            train = (st, ft_sub, predict(st, ft_sub))
        diagnostics = grid_diagnostics(model, stimuli, fixations, centre_bias, train=train)
        diagnostics["temperature_fit_part"] = fit_part(cfg.dataset) if train is not None else None
    return {
        "task": "scanpath",
        "model": getattr(model, "name", cfg.model) if isinstance(model, GridScanpathModel) else cfg.model,
        "dataset": cfg.dataset,
        "split": split_name(cfg),
        "protocol": {k: (str(v) if isinstance(v, Path) else v) for k, v in asdict(replace(cfg)).items()}
        | {"part": part, "centre_bias": fit, "off_stimulus_fixations": "clipped at load",
           "scored_fixations": "every fixation with a history (the start fixation is history only)",
           "checkpoint": provenance, "pysaliency": getattr(pysaliency, "__version__", "?")},
        "metrics": metrics,
        "diagnostics": diagnostics,
        "evaluated_at": datetime.now(UTC).isoformat(),
    }


__all__ = ["ScanpathConfig", "aggregate", "centre_bias_fit", "centre_bias_model", "grid_diagnostics", "per_fixation_scores",
           "run", "scored_fixations", "scores", "select", "split_name"]
