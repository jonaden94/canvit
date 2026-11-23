"""Teacher-forced likelihood evaluation on human scanpaths, scored by pysaliency (design §4.1).

    python -m canvit.harness.evaluate scanpath --cfg.model centre_bias --out results/cb.json

Conventions (pysaliency's and Kümmerer et al.'s, notes: canvit_claude/papers/, outside the repo):
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

from .data import coco_search18, make_scanpath_loader
from .models import (
    DensitySaliencyMap,
    GridScanpathModel,
    fit_centre_bias,
    fit_temperature,
    policy_entropy,
    without_start_fixations,
)
from .sources import PolicySource, SearchSource

log = logging.getLogger(__name__)

RELEASE_FILES = ("COCOSearch18-images-TP.zip", "COCOSearch18-images-TA.zip", "COCOSearch18-fixations-TP.zip",
                 "coco_search18_fixations_TA_trainval.json")


def _env_path(name: str) -> Path:
    return Path(os.environ.get(name, ""))


@dataclass
class ScanpathConfig:
    model: Literal["uniform", "centre_bias", "canvit_policy", "canvit_heatmap"] = "centre_bias"
    """``canvit_policy``: a run's learned policy (any task, ``source``); ``canvit_heatmap``: a search run's
    task head (``canvit_models.py``). Both need ``ckpt``."""
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
    """A run's harness ``.pt`` (its policy is read from the ``.policy.pt`` beside it)."""
    source_geometry: Literal["fovi-27b1bde", "fovi-c399d3b"] | None = None
    """REQUIRED for the CanViT models: the foveation geometry the run was trained with. Our episode builds
    fovi 27b1bde's (= canvit.core.foveation since the port, 2d4c032); c399d3b builds different patches (KNN k
    rule, planar geometry; docs/q_policy_foveated.md) that canvit cannot reproduce, so it is refused. A checkpoint
    does not record it: read the run's launcher (``FOVI_COMMIT``) or its SLURM log ("Pinned fovi -> ..."); a run
    pinned to a canvit commit after the port has 27b1bde. No default, so the choice is always on the record."""
    source: PolicySource = field(default_factory=SearchSource)
    save_predictions: Path | None = None
    """CanViT models: write the predicted maps here (``.npz``, ``GridScanpathModel.save``), e.g. for figures."""
    fit_temperature_scanpaths: int = 0
    """CanViT models, DIAGNOSTIC: > 0 runs the model on this many TRAINING scanpaths (same condition), fits one
    global softmax temperature and one per fixation index by log-likelihood there, and reports the tempered
    likelihoods on the evaluated split under ``diagnostics``. Never fed back into training (owner, 2026-11-20)."""
    """The task the run trained, with its own flags (``cfg.source:<task> --cfg.source.cfg.* --cfg.source.rl.*``,
    ``sources.py``): the model and policy are rebuilt with the training constructors and loaded strictly."""


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


def check_source_geometry(geometry: str | None) -> None:
    """Refuse a CanViT run whose foveation geometry is undeclared or not the one our episode builds (design §13)."""
    if geometry is None:
        raise ValueError("--cfg.source-geometry is required for a CanViT model: fovi-27b1bde (the geometry of "
                         "canvit.core.foveation, every run pinned to a canvit commit after the fovi port 2d4c032, "
                         "and FOVI_COMMIT=27b1bde runs such as exp39 search) or fovi-c399d3b. The checkpoint does "
                         "not record it: see the run's launcher FOVI_COMMIT or its SLURM log ('Pinned fovi -> ...').")
    if geometry != "fovi-27b1bde":
        raise ValueError(f"--cfg.source-geometry {geometry}: this run was trained on patches canvit no longer builds "
                         "(fovi c399d3b: other KNN k rule and field geometry; the port kept only 27b1bde). Scoring it "
                         "here would feed it different input than in training. Use a run trained on fovi 27b1bde or "
                         "later (owner decision 2026-11-20: the paper uses new-geometry runs only).")


def build_model(cfg: ScanpathConfig, stimuli, fixations, centre_bias, device: torch.device):
    """-> (the model scored on ``fixations``, and for CanViT models a ``predict(stimuli, fixations)`` that runs
    the same model on other scanpaths -- the temperature diagnostic needs the training split -- else None)."""
    if cfg.model == "uniform":
        return pysaliency.UniformModel(), None
    if cfg.model == "centre_bias":
        return centre_bias, None
    from .canvit_models import CanViTModel, PolicyEpisode, precompute
    if cfg.ckpt is None:
        raise ValueError(f"--cfg.model {cfg.model} needs --cfg.ckpt (a run's .pt)")
    check_source_geometry(cfg.source_geometry)
    task = cfg.source.task()
    c = task.cfg
    task_scene = getattr(c, "scene_size", None) or getattr(c, "scene_resolution", None)
    if task_scene is not None and task_scene != cfg.scene_size:
        raise ValueError(f"the {task.name} run's scene is {task_scene} px but --cfg.scene-size is {cfg.scene_size}: "
                         "the human fixations must be letterboxed into the scene the model was trained on")
    readout = "policy" if cfg.model == "canvit_policy" else "heatmap"
    ep = PolicyEpisode.load(task=task, ckpt=cfg.ckpt, device=device, need_policy=readout == "policy")

    def predict(stimuli, fixations):
        loader = make_scanpath_loader(stimuli, fixations, scene_size=cfg.scene_size, batch_size=cfg.batch_size,
                                      num_workers=cfg.num_workers)
        return precompute(CanViTModel(ep, readout), loader, grid=cfg.grid, scene_size=cfg.scene_size)

    return predict(stimuli, fixations), predict


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
    st, ft, sv, fv = load_coco_search18(cfg)
    stimuli, fixations = (st, ft) if cfg.part == "train" else (sv, fv)
    fixations = select(fixations, condition=cfg.condition, limit=cfg.limit_scanpaths)
    if cfg.centre_bias_bandwidth is None or cfg.centre_bias_eps is None:
        fit = fit_centre_bias(st, ft)
    else:
        fit = {"bandwidth": cfg.centre_bias_bandwidth, "eps": cfg.centre_bias_eps}
    centre_bias = CrossvalidatedBaselineModel(stimuli, without_start_fixations(fixations), bandwidth=fit["bandwidth"],
                                              eps=fit["eps"],
                                              memory_cache_size=4)  # see scores(): fixations sorted by stimulus
    model, predict = build_model(cfg, stimuli, fixations, centre_bias, device)
    metrics = scores(model, stimuli, fixations, centre_bias)
    diagnostics: dict[str, Any] = {}
    if isinstance(model, GridScanpathModel):
        if cfg.save_predictions is not None:
            model.save(cfg.save_predictions)
        train = None
        if cfg.fit_temperature_scanpaths > 0:
            assert predict is not None
            ft_sub = select(ft, condition=cfg.condition, limit=cfg.fit_temperature_scanpaths)
            train = (st, ft_sub, predict(st, ft_sub))
        diagnostics = grid_diagnostics(model, stimuli, fixations, centre_bias, train=train)
    return {
        "task": "scanpath",
        "model": cfg.model,
        "protocol": {k: (str(v) if isinstance(v, Path) else v) for k, v in asdict(replace(cfg)).items()}
        | {"centre_bias": fit, "off_stimulus_fixations": "clipped at load",
           "pysaliency": getattr(pysaliency, "__version__", "?")},
        "metrics": metrics,
        "diagnostics": diagnostics,
        "evaluated_at": datetime.now(UTC).isoformat(),
    }


__all__ = ["ScanpathConfig", "check_source_geometry", "clip_to_stimuli", "grid_diagnostics", "run", "scores",
           "select"]
