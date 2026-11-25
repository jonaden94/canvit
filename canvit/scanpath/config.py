"""The configuration of the human-scanpath evaluation (``python -m canvit.harness.evaluate scanpath``).

Kept free of pysaliency (an optional dependency, ``pip install canvit[scanpath]``), so that ``harness.evaluate``
can offer the subcommand without importing it; :mod:`canvit.scanpath.evaluate` does the work.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal


def _env_path(name: str) -> Path:
    return Path(os.environ.get(name, ""))


@dataclass
class ScanpathConfig:
    model: Literal["uniform", "centre_bias", "canvit_policy", "canvit_heatmap"] = "centre_bias"
    """``canvit_policy``: a run's learned policy (any task); ``canvit_heatmap``: a search run's task head
    (``canvit_models.py``). Both need ``ckpt``. A search model on free-viewing data is read as the uniform
    mixture over its cues (``..._cuemix``, owner decision 2026-11-23)."""
    dataset: Literal["coco_search18", "coco_freeview", "daemons", "mit1003", "cat2000", "figrim"] = "coco_search18"
    part: Literal["train", "validation", "test", "all"] | None = None
    """None = ``data.eval_part(dataset)``: validation where the dataset has a split, else all. ``test``: COCO-Search18's
    official test split (``data.coco_search18_test``), a second evaluation next to validation."""
    data_location: Path = field(default_factory=lambda: _env_path("SCANPATH_DATA_DIR"))
    """pysaliency's dataset location: every dataset is downloaded (first use) and cached under it. Required; the
    default is ``$SCANPATH_DATA_DIR`` if set."""
    release_dirs: tuple[Path, ...] = field(default_factory=lambda: tuple(
        Path(p) for p in os.environ.get("COCO_SEARCH18_RELEASE_DIRS", "").split(":") if p))
    """Local copies of the COCO-Search18 release files (default ``$COCO_SEARCH18_RELEASE_DIRS``, ':'-separated), read
    only when pysaliency's cache does not exist yet. Empty: pysaliency downloads split1 itself. The official TEST
    split always needs them (its images come from the release image zips; docs/scanpath.md)."""
    condition: Literal["both", "tp", "ta"] = "both"
    """COCO-Search18: which scanpaths to score: target present, absent, or both (``target_present``)."""
    cut_at_target: bool = False
    """COCO-Search18 TP: score each scanpath only up to its first fixation on the target
    (:func:`cut_at_first_target_fixation`), the format of the redistributed test split (HAT/Gazeformer), so that
    validation and test are comparable. The centre bias is then the KDE of the cut fixations too."""
    grid: int | None = None
    """Cells per axis of the CanViT model's predictions: None = the model's own grid (the search policy's 32x32;
    another task's policy may use another, e.g. ``rl.centers_per_axis``). A value is a check: a model on another
    grid is refused (``CanViTModel.forced_logits``), never resampled."""
    scene_size: int = 512
    display_frame: str | None = None
    """How a stimulus is placed in the scene: ``"scene"`` fills the scene as far as its aspect allows; ``"<W>x<H>"``
    letterboxes it first into the box of a W x H display centred in the scene (``data.letterbox_geometry``).
    None (the STANDARD, owner decision 2026-11-23) = the display shape of the checkpoint's own training data
    (the run task's ``display_frame`` attribute; refused for a task without one): a model only knows the geometry it
    was trained in.
    The search runs saw COCO-Search18's 1680x1050 displays, so scene rows outside that box were always padding
    for them. Set it only for a deliberate comparison (e.g. ``scene`` on FIGRIM/MIT1003). Recorded in the
    protocol with its source."""
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
    retina: str | None = None
    """A multi-retina backbone's retina to evaluate through, overriding the run config's ``--cfg.retina`` (None: the
    trained one). The task head and policy were trained through the run's retina only: any other is out of
    distribution for them."""
    shard: tuple[int, int] | None = None
    """``i N``: GPU stage only -- run the CanViT model on the i-th of N contiguous blocks of the evaluated scanpaths
    and write its predictions to ``save_predictions`` (no scoring). Shard 0 also writes the training-subset
    predictions and the samples. Then score once with ``--cfg.predictions <all shards>``."""
    predictions: tuple[Path, ...] = ()
    """Score these saved predictions (``GridScanpathModel.save``, e.g. every shard of one run) instead of running
    the model: CPU only. With ``train_scanpaths``, the training-subset predictions come from
    ``save_train_predictions``."""
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
