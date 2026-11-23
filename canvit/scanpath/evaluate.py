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
    cell_scores,
    fit_centre_bias,
    fit_temperature,
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
    density_metrics_fixations: int | None = 20000
    """NSS and AUC (~55 ms per fixation at 1080x1920 in pysaliency, LL ~2 ms) on a seeded random subset of at most
    this many scored fixations; LL and IG always on all (owner decision, 2026-11-23). None = all."""
    density_metrics_seed: int = 0
    processes: int = 1
    """CPU processes for pysaliency's per-fixation scoring (:func:`parallel_per_fixation_scores`)."""
    batch_size: int = 16
    num_workers: int = 4
    limit_scanpaths: int | None = None
    """Score only the first N scanpaths (smoke tests)."""
    device: str = "cuda"
    ckpt: Path | None = None
    """A self-contained harness checkpoint (run config, architecture and policy inside; canvit main b029fd3+)."""
    save_predictions: Path | None = None
    """CanViT models: write the predicted maps here (``.npz``, ``GridScanpathModel.save``), e.g. for figures."""
    save_samples: Path | None = None
    """CanViT policy models: also sample the model's OWN scanpaths, one per evaluated human scanpath and as long
    as it (``canvit_models.sample_scanpaths``), and write them here (``.npz``, stimulus pixels); a cue mixture
    writes ``<stem>_cuemix.npz``."""
    sample_temperature: float = 1.0
    """Softmax temperature of the sampled scanpaths (1 = the policy's own distribution; 0 = its argmax)."""
    sample_seed: int = 0
    samples_scanpaths: int | None = 2000
    """Sample for a seeded random subset of at most this many evaluated scanpaths (None: all)."""
    train_scanpaths: int = 0
    """CanViT models: > 0 also runs the model on a seeded random subset of this many TRAINING scanpaths (``fit_part``,
    same condition; :func:`subset_scanpaths`), for what is FITTED on train: the temperature DIAGNOSTIC (one global
    softmax temperature + one per fixation index, by log-likelihood; tempered likelihoods on the evaluated split
    under ``diagnostics``; never fed back into training, owner 2026-11-20) and, with ``save_train_predictions``,
    later fits (biases, design §12). A dataset without a split reuses the evaluated predictions (in-sample)."""
    train_seed: int = 0
    save_train_predictions: Path | None = None
    """Write the training-subset predictions here (``.npz``; keys index the SUBSET's scanpaths)."""


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


def subset_scanpaths(fixations, n: int, *, seed: int):
    """A seeded random subset of ``n`` scanpaths (all if fewer), in their original order, and their indices."""
    m = len(fixations.scanpaths)
    keep = np.arange(m) if m <= n else np.sort(np.random.default_rng(seed).choice(m, size=n, replace=False))
    return fixations.filter_scanpaths(keep), keep


def scored_fixations(fixations):
    """The fixations that are scored -- every fixation with a history -- sorted by stimulus (stable), so a model
    with a small map cache (the centre bias) computes each map once; no average depends on the order."""
    fx = fixations[fixations.scanpath_history_length > 0]
    return fx[np.argsort(fx.n, kind="stable")]


def density_metrics_mask(fx, max_n: int | None, seed: int) -> np.ndarray:
    """[len(fx)] which scored fixations get NSS / AUC: all, or a seeded random subset of ``max_n``. Draw it on the
    WHOLE :func:`scored_fixations` and slice it with the shards, so every model scores the same subset."""
    mask = np.ones(len(fx), dtype=bool)
    if max_n is not None and len(fx) > max_n:
        mask[:] = False
        mask[np.random.default_rng(seed).choice(len(fx), size=max_n, replace=False)] = True
    return mask


def per_fixation_scores(model, stimuli, fx, centre_bias, *, density_mask: np.ndarray | None = None
                        ) -> dict[str, np.ndarray]:
    """pysaliency's per-fixation scores of ``model`` on ``fx`` (:func:`scored_fixations`, or a shard of it):
    ``ll`` / ``ig`` in bits over the uniform model / the centre bias, ``nss`` / ``auc`` of the conditional
    density (only where ``density_mask``, NaN elsewhere; :func:`density_metrics_mask`), and ``n`` (stimulus),
    ``t`` (history length), ``scanpath`` (index) for :func:`aggregate`. Shards concatenate (key by key)."""
    ll_model = model.log_likelihoods(stimuli, fx) / np.log(2)
    mask = np.ones(len(fx), dtype=bool) if density_mask is None else density_mask
    nss, auc = np.full(len(fx), np.nan), np.full(len(fx), np.nan)
    if mask.any():
        nss[mask] = DensitySaliencyMap(model).NSSs(stimuli, fx[mask])
        auc[mask] = DensitySaliencyMap(model).AUCs(stimuli, fx[mask])
    return {
        "ll": ll_model - pysaliency.UniformModel().log_likelihoods(stimuli, fx) / np.log(2),
        "ig": ll_model - centre_bias.log_likelihoods(stimuli, fx) / np.log(2),
        "nss": nss, "auc": auc,
        "n": np.asarray(fx.n), "t": np.asarray(fx.scanpath_history_length), "scanpath": np.asarray(fx.scanpath_index),
    }


_POOL_ARGS: tuple = ()


def _score_shard(sl: slice) -> dict[str, np.ndarray]:
    model, stimuli, fx, centre_bias, mask = _POOL_ARGS
    return per_fixation_scores(model, stimuli, fx[sl], centre_bias, density_mask=mask[sl])


def parallel_per_fixation_scores(model, stimuli, fx, centre_bias, *, processes: int,
                                 density_mask: np.ndarray | None = None) -> dict[str, np.ndarray]:
    """:func:`per_fixation_scores` in ``processes`` forked workers, on contiguous shards cut at stimulus
    boundaries (``fx`` sorted by stimulus), concatenated in order: the same arrays as one call."""
    global _POOL_ARGS
    mask = np.ones(len(fx), dtype=bool) if density_mask is None else density_mask
    if processes <= 1:
        return per_fixation_scores(model, stimuli, fx, centre_bias, density_mask=mask)
    import multiprocessing as mp
    n = np.asarray(fx.n)
    starts = np.flatnonzero(np.r_[True, n[1:] != n[:-1]])  # first fixation of each stimulus
    idx = np.searchsorted(starts, np.linspace(0, len(fx), 8 * processes + 1)[1:-1])  # ~8 shards per worker
    cuts = np.unique(np.r_[0, starts[idx[idx < len(starts)]], len(fx)]).tolist()
    shards = [slice(a, b) for a, b in zip(cuts[:-1], cuts[1:])]
    _POOL_ARGS = (model, stimuli, fx, centre_bias, mask)
    try:
        with mp.get_context("fork").Pool(processes) as pool:  # the workers inherit the models (no pickling)
            parts = pool.map(_score_shard, shards, chunksize=1)
    finally:
        _POOL_ARGS = ()
    return {k: np.concatenate([p[k] for p in parts]) for k in parts[0]}


def aggregate(per_fix: dict[str, np.ndarray]) -> dict[str, float]:
    """The averages of :func:`per_fixation_scores`: over images (``*_image``: per-image means, then over images
    -- the papers' headline) and over fixations (``*_fix``: pysaliency's default), overall and per fixation
    index t (``*_t<t>``, fixation-averaged, for indices with >= 20 fixations). NaN entries (NSS / AUC outside
    the subset) are left out of every average."""
    out: dict[str, float] = {"n_fixations": float(len(per_fix["n"])),
                             "n_scanpaths": float(len(np.unique(per_fix["scanpath"]))),
                             "n_fixations_nss_auc": float(np.isfinite(per_fix["nss"]).sum())}
    for k, v in per_fix.items():
        if k in ("n", "t", "scanpath"):
            continue
        ok = np.isfinite(v)
        v, t = v[ok], per_fix["t"][ok]
        by_image = SimpleNamespace(n=per_fix["n"][ok])  # all pysaliency's average_values reads
        out[f"{k}_image"] = float(average_values(v, by_image, average="image"))
        out[f"{k}_fix"] = float(average_values(v, by_image, average="fixation"))
        for i in range(1, int(t.max()) + 1):
            if (t == i).sum() >= 20:  # an index with fewer fixations is noise
                out[f"{k}_t{i}"] = float(v[t == i].mean())
    return out


def scores(model, stimuli, fixations, centre_bias, *, density_metrics_fixations: int | None = None,
           density_metrics_seed: int = 0) -> dict[str, float]:
    """pysaliency's per-fixation scores of ``model`` on the fixations with a history, averaged."""
    fx = scored_fixations(fixations)
    mask = density_metrics_mask(fx, density_metrics_fixations, density_metrics_seed)
    return aggregate(per_fixation_scores(model, stimuli, fx, centre_bias, density_mask=mask))


def centre_bias_model(stimuli, fixations, fit: dict[str, float]) -> CrossvalidatedBaselineModel:
    """The centre bias the scores are relative to: pysaliency's leave-one-image-out KDE (+ uniform, ``fit``) on the
    EVALUATED fixations (after ``select``), start fixations excluded (``models.without_start_fixations``)."""
    return CrossvalidatedBaselineModel(stimuli, without_start_fixations(fixations), bandwidth=fit["bandwidth"],
                                       eps=fit["eps"], memory_cache_size=4)  # fixations sorted by stimulus


def build_model(cfg: ScanpathConfig, stimuli, fixations, centre_bias, device: torch.device):
    """-> (the model scored on ``fixations``, and for CanViT models a ``predict(stimuli, fixations)`` that runs
    the same model on other scanpaths -- the temperature diagnostic needs the training split -- and the
    ``CanViTModel``; for the baselines None, None)."""
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

    return predict(stimuli, fixations), predict, model


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


def grid_diagnostics(model, stimuli, fx, per_fix, *, train=None) -> dict[str, Any]:
    """For a grid model on the scored fixations ``fx`` (:func:`scored_fixations`; ``per_fix`` its
    :func:`per_fixation_scores`): the policy's entropy per fixation index (nats), and with ``train = (stimuli,
    fixations, model on them)`` the fitted temperatures and the tempered likelihoods (DIAGNOSTIC). A temperature
    changes pysaliency's LL (and IG) by exactly the change in log p(cell) (``models.cell_scores``)."""
    lp, H, t = cell_scores(model, stimuli, fx)
    out: dict[str, Any] = {"entropy_fix": float(H.mean())} | {
        f"entropy_t{i}": float(H[t == i].mean()) for i in np.unique(t) if (t == i).sum() >= 20}
    if train is None:
        return out
    tau = fit_temperature(train[2], train[0], train[1])
    tau_t = fit_temperature(train[2], train[0], train[1], per_index=True)
    by_image = SimpleNamespace(n=per_fix["n"])
    for name, temp in (("global", tau), ("per_index", tau_t)):
        lp_tau, H_tau, _ = cell_scores(model.with_temperature(temp), stimuli, fx)
        delta = (lp_tau - lp) / np.log(2)
        out[f"tempered_{name}"] = {
            "temperature": temp, "entropy_fix": float(H_tau.mean()),
            "ll_image": float(average_values(per_fix["ll"] + delta, by_image, average="image")),
            "ig_image": float(average_values(per_fix["ig"] + delta, by_image, average="image"))}
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
    model, predict, canvit = build_model(cfg, stimuli, fixations, centre_bias, device)
    fx = scored_fixations(fixations)
    per_fix = parallel_per_fixation_scores(model, stimuli, fx, centre_bias, processes=cfg.processes,
                                           density_mask=density_metrics_mask(fx, cfg.density_metrics_fixations,
                                                                             cfg.density_metrics_seed))
    metrics = aggregate(per_fix)
    diagnostics: dict[str, Any] = {}
    if isinstance(model, GridScanpathModel):
        if cfg.save_predictions is not None:
            model.save(cfg.save_predictions)
        train = None
        if cfg.train_scanpaths > 0:
            assert predict is not None
            if fit_part(cfg.dataset) == part:  # no split: the evaluated predictions themselves (in-sample)
                train = (stimuli, fixations, model)
            else:
                st, ft = load(cfg, fit_part(cfg.dataset))
                ft, _ = subset_scanpaths(select(ft, condition=cfg.condition, limit=None), cfg.train_scanpaths,
                                         seed=cfg.train_seed)
                train = (st, ft, predict(st, ft))
                if cfg.save_train_predictions is not None:
                    train[2].save(cfg.save_train_predictions)
        diagnostics = grid_diagnostics(model, stimuli, fx, per_fix, train=train)
        diagnostics["temperature_fit_part"] = None if train is None else (
            f"{fit_part(cfg.dataset)} (in-sample)" if train[2] is model else fit_part(cfg.dataset))
    if cfg.save_samples is not None:
        if canvit is None or canvit.readout != "policy":
            raise ValueError("--cfg.save-samples needs a CanViT policy model (--cfg.model canvit_policy)")
        from .canvit_models import sample_scanpaths
        sub, keep = subset_scanpaths(fixations, cfg.samples_scanpaths or len(fixations.scanpaths),
                                     seed=cfg.sample_seed)
        loader = make_scanpath_loader(stimuli, sub, scene_size=cfg.scene_size, batch_size=cfg.batch_size,
                                      num_workers=cfg.num_workers)
        smp = sample_scanpaths(canvit.episode, loader, temperature=cfg.sample_temperature, seed=cfg.sample_seed,
                               cue_mixture=canvit.cue_mixture)
        smp["scanpath"] = keep[smp["scanpath"]]  # index into the evaluated scanpaths, as the predictions' keys
        smp["n"] = np.asarray(fixations.scanpaths.n)[smp["scanpath"]]  # the stimulus index of each sample
        smp["model"] = np.array(canvit.name)
        samples_path = cfg.save_samples
        if canvit.cue_mixture:  # the file name says which model the samples come from
            samples_path = samples_path.with_name(f"{samples_path.stem}_cuemix{samples_path.suffix}")
        samples_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(samples_path, **smp)
    return {
        "task": "scanpath",
        "model": getattr(model, "name", cfg.model) if isinstance(model, GridScanpathModel) else cfg.model,
        "dataset": cfg.dataset,
        "split": split_name(cfg),
        "protocol": {k: (str(v) if isinstance(v, Path) else v) for k, v in asdict(replace(cfg)).items()}
        | {"part": part, "centre_bias": fit, "off_stimulus_fixations": "clipped at load",
           "scored_fixations": "every fixation with a history (the start fixation is history only)"
           + ("; DAEMONS forced fixations after the first are scored (pysaliency's default; owner, 2026-11-23)"
              if cfg.dataset == "daemons" else ""),
           "checkpoint": None if canvit is None else canvit.episode.provenance,
           "pysaliency": getattr(pysaliency, "__version__", "?")},
        "metrics": metrics,
        "diagnostics": diagnostics,
        "evaluated_at": datetime.now(UTC).isoformat(),
    }


__all__ = ["ScanpathConfig", "aggregate", "centre_bias_fit", "centre_bias_model", "density_metrics_mask",
           "grid_diagnostics", "parallel_per_fixation_scores", "per_fixation_scores", "run", "scored_fixations",
           "scores", "select", "split_name", "subset_scanpaths"]
