"""WebDataset-based training loader.

Used for training only; validation always reads the raw ImageNet-1k val
ImageFolder (see `create_loaders` / `IndexedImageFolder`), independent of the
training data source.

Each rank receives a deterministically computed list of shard paths (via
`schedule.compute_schedule_slice`). DataLoader workers then split those shards
via `wds.split_by_worker` — each worker streams one or more shards sequentially.
The actual `num_workers` is derived from `cfg.num_workers`, capped at
`shards_per_gpu`, and rounded down to a divisor of `shards_per_gpu` so every
worker gets the same number of shards.

Loader interface is the one `loop.py` drives (`next()` returning a Batch), so it can
call `train_loader.next()` without modification.
"""

from __future__ import annotations

import io
import json
import logging
import tarfile
from collections.abc import Callable, Iterator, Sequence
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import torch
import webdataset as wds
from PIL import Image
from torch import Tensor
from torch.utils.data import DataLoader

from canvit.core import CLSStandardizer, PatchStandardizer
from canvit.core.preprocess import preprocess

from ...harness.infra.schedule import compute_schedule_slice, compute_shards_per_gpu

if TYPE_CHECKING:
    from ..config import Config

log = logging.getLogger(__name__)


def _decode_jpg(data: bytes, image_size: int) -> Tensor:
    img = Image.open(io.BytesIO(data)).convert("RGB")
    tensor = preprocess(image_size)(img)
    assert isinstance(tensor, Tensor)
    return tensor


def _decode_label(data: bytes) -> int:
    return int(json.loads(data.decode("utf-8"))["label"])


def _decode_npy_fp16(data: bytes) -> Tensor:
    arr = np.load(io.BytesIO(data))
    return torch.from_numpy(arr.copy())


def _read_info(dir_: Path) -> dict:
    info_path = dir_ / "info.json"
    assert info_path.exists(), f"info.json not found at {info_path}"
    with open(info_path) as f:
        return json.load(f)


def _build_pipeline(
    shards: list[str],
    *,
    image_size: int,
    batch_size: int,
    use_worker_split: bool,
    has_features: bool = True,
) -> wds.WebDataset:
    """Build a WebDataset pipeline.

    With ``has_features=True`` (precomputed-feature shards) it yields
    ``(image, label, cls, ptch)`` batches. With ``has_features=False`` (raw
    shards carrying only ``jpg``/``json``) it yields ``(image, label)`` batches —
    teacher features are then computed on the fly in the training loop.

    wds.WebDataset already applies split_by_worker internally (via its default
    workersplitter parameter) at the shard-URL level — one shard per worker.
    We must NOT add split_by_worker again via .compose(), as that would apply
    it a second time at the decoded-sample level, keeping only every Nth sample
    per worker and reducing throughput by num_workers.

    nodesplitter=None: we pre-slice shards per rank before constructing the
    dataset, so no node-level splitting inside WebDataset is needed (the default
    single_node_only would raise ValueError under DDP).
    """
    # shardshuffle=False — the schedule already provides global shuffle.
    # empty_check=False — single-shard val/init pipelines should not warn.
    workersplitter = wds.split_by_worker if use_worker_split else None
    ds = wds.WebDataset(
        shards,
        shardshuffle=False,
        empty_check=False,
        nodesplitter=None,
        workersplitter=workersplitter,
    )
    if has_features:
        return (
            ds.to_tuple("jpg", "json", "cls.npy", "ptch.npy")
            .map_tuple(
                lambda d: _decode_jpg(d, image_size),
                _decode_label,
                _decode_npy_fp16,
                _decode_npy_fp16,
            )
            .batched(batch_size, partial=False)
        )
    return (
        ds.to_tuple("jpg", "json")
        .map_tuple(
            lambda d: _decode_jpg(d, image_size),
            _decode_label,
        )
        .batched(batch_size, partial=False)
    )


class WebDatasetTrainLoader:
    """Streams training samples from one rank's slice of the shard schedule.

    Yields `(images, raw_patches, raw_cls, labels)` per call to `next()`,
    matching the `next()` contract `loop.py` expects.
    """

    def __init__(
        self,
        *,
        train_dir: Path,
        seed: int,
        job_index: int,
        batch_size_per_gpu: int,
        steps_per_job: int,
        image_size: int,
        world_size: int,
        rank: int,
        num_workers: int,
    ) -> None:
        info = _read_info(train_dir)
        # Precomputed-feature shards carry cls.npy/ptch.npy. "Raw" shards have
        # only jpg+json; in that case teacher features are computed on the fly
        # in the training loop (load_train_batch -> compute_raw_targets).
        self.has_features: bool = "cls.npy" in info["keys"]
        log.info(
            "WebDatasetTrainLoader: %s (info.json keys = %s)",
            "PRECOMPUTED features" if self.has_features
            else "RAW images (teacher features computed on the fly)",
            info["keys"],
        )
        self.samples_per_shard: int = int(info["images_per_shard"])
        assert self.samples_per_shard % batch_size_per_gpu == 0, (
            f"samples_per_shard ({self.samples_per_shard}) must be divisible by "
            f"batch_size_per_gpu ({batch_size_per_gpu}) so each worker yields a "
            f"clean number of batches."
        )

        self.shards_per_gpu = compute_shards_per_gpu(
            steps_per_job, batch_size_per_gpu, self.samples_per_shard
        )
        self.shard_files: list[Path] = compute_schedule_slice(
            seed=seed,
            train_dir=train_dir,
            job_index=job_index,
            shards_per_gpu=self.shards_per_gpu,
            world_size=world_size,
            rank=rank,
        )
        self.train_dir = train_dir
        self.batch_size = batch_size_per_gpu
        self.image_size = image_size

        # Resolve num_workers: cap at shards_per_gpu (extra workers would get
        # zero shards), then round down to a divisor of shards_per_gpu so every
        # worker streams the same number of shards (keeps the per-worker batch
        # count uniform under .batched(partial=False)).
        requested = max(1, num_workers)
        capped = min(requested, self.shards_per_gpu)
        nw = capped
        while self.shards_per_gpu % nw != 0:
            nw -= 1
        self.num_workers = nw
        shards_per_worker = self.shards_per_gpu // self.num_workers
        if requested != self.num_workers:
            log.info(
                f"WebDatasetTrainLoader: requested num_workers={requested}, "
                f"using {self.num_workers} "
                f"({'capped to shards_per_gpu' if requested > self.shards_per_gpu else 'rounded down for divisibility'}); "
                f"each worker streams {shards_per_worker} shard(s)"
            )
        else:
            log.info(
                f"WebDatasetTrainLoader: num_workers={self.num_workers}; "
                f"each worker streams {shards_per_worker} shard(s)"
            )

        total_samples = len(self.shard_files) * self.samples_per_shard
        assert total_samples == steps_per_job * batch_size_per_gpu, (
            f"Sample count mismatch: {len(self.shard_files)} shards × {self.samples_per_shard} "
            f"= {total_samples} samples, but steps_per_job × batch_size = "
            f"{steps_per_job} × {batch_size_per_gpu} = {steps_per_job * batch_size_per_gpu}"
        )

        log.info(
            f"WebDatasetTrainLoader: rank={rank}/{world_size}, job_index={job_index}, "
            f"shards_per_gpu={self.shards_per_gpu}, num_workers={self.num_workers}, "
            f"samples_per_shard={self.samples_per_shard}, batch_size={batch_size_per_gpu}, "
            f"steps_per_job={steps_per_job}, total_samples={total_samples} ✓"
        )

        self._iter: Iterator | None = None
        self._loader: DataLoader | None = None

    def first_shard_path(self) -> Path:
        return self.shard_files[0]

    def normalizer_shard_paths(self, n_shards: int) -> list[Path]:
        """The first `n_shards` shards of the sorted training set, for normalizer stats.

        Deliberately NOT `self.shard_files[:n]`: that list comes from
        `compute_schedule_slice`, so it depends on `seed`, `job_index` and `rank` — two
        runs of the same config with different seeds seeded their standardizers from
        different shards. Pinning to the sorted head makes the stats a property of the
        DATASET, identical across every run and rank.

        Excludes the last shard, which `compute_schedule_slice` treats as partial.
        """
        assert n_shards >= 1, f"normalizer_shards must be >= 1, got {n_shards}"
        train_shards = sorted(self.train_dir.glob("shard-*.tar"))[:-1]
        assert len(train_shards) >= n_shards, (
            f"cfg.normalizer_shards={n_shards} but {self.train_dir} has only "
            f"{len(train_shards)} full shards (the last is excluded as partial). "
            f"Lower normalizer_shards, or point at a larger dataset."
        )
        return train_shards[:n_shards]

    def _ensure_iter(self) -> None:
        if self._iter is not None:
            return
        ds = _build_pipeline(
            [str(p) for p in self.shard_files],
            image_size=self.image_size,
            batch_size=self.batch_size,
            use_worker_split=True,
            has_features=self.has_features,
        )
        self._loader = DataLoader(
            ds,
            batch_size=None,
            num_workers=self.num_workers,
            pin_memory=True,
            persistent_workers=False,
            prefetch_factor=2 if self.num_workers > 0 else None,
        )
        self._iter = iter(self._loader)

    def next(self) -> tuple[Tensor, Tensor | None, Tensor | None, Tensor]:
        """Returns (images, raw_patches, raw_cls, labels).

        For raw (no-feature) shards, ``raw_patches`` and ``raw_cls`` are None —
        the training loop computes teacher features on the fly from ``images``.
        """
        self._ensure_iter()
        assert self._iter is not None
        raw_cls: Tensor | None
        raw_patches: Tensor | None
        if self.has_features:
            images, labels, raw_cls, raw_patches = next(self._iter)
        else:
            images, labels = next(self._iter)
            raw_cls = raw_patches = None
        # webdataset's .batched() returns labels as a list[int] — convert.
        if not isinstance(labels, Tensor):
            labels = torch.as_tensor(labels, dtype=torch.long)
        return images, raw_patches, raw_cls, labels


class _MomentAcc:
    """Streaming per-(token, channel) mean/var, so pooling shards costs O(1) memory.

    `set_stats` reduces over dim 0, so the obvious "concatenate every shard and call it
    once" needs n_shards * samples_per_shard * n_tokens * embed_dim floats resident —
    4 shards at 4096x1024x768 is ~51 GB. Accumulating sum/sum-of-squares instead keeps
    two [n_tokens, embed_dim] buffers regardless of how many shards are pooled.

    float64 because the sum of ~10^4 squared feature values loses meaningful precision
    in float32, and this runs once per training run.
    """

    def __init__(self, device: torch.device) -> None:
        self.n = 0
        self._sum: Tensor | None = None
        self._sumsq: Tensor | None = None
        self._device = device

    def add(self, batch: Tensor) -> None:
        """batch: [B, tokens, D]."""
        x = batch.to(self._device, torch.float64)
        if self._sum is None:
            self._sum = torch.zeros(x.shape[1:], dtype=torch.float64, device=self._device)
            self._sumsq = torch.zeros_like(self._sum)
        self._sum += x.sum(dim=0)
        self._sumsq += (x * x).sum(dim=0)
        self.n += x.shape[0]

    def moments(self) -> tuple[Tensor, Tensor]:
        assert self.n > 0 and self._sum is not None and self._sumsq is not None
        mean = self._sum / self.n
        # Matches set_stats' unbiased=False.
        var = (self._sumsq / self.n - mean * mean).clamp_min(0)
        return mean, var


def _freeze_stats(norm, acc: _MomentAcc) -> None:
    """Freeze `norm`'s buffers to the accumulated moments, via the public `set_stats`.

    `set_stats(data)` computes `data.mean(dim=0)` and `data.var(dim=0, unbiased=False)`,
    so we hand it a two-row surrogate carrying exactly the moments we streamed: for
    ``x = [m + s, m - s]``  ->  ``mean = m`` and ``var = ((+s)^2 + (-s)^2)/2 = s^2``.
    The frozen buffers are therefore the same values the full tensor would have produced,
    without ever materialising it. (Using the public API keeps this working if the
    standardizer ever gains validation or extra state in `set_stats`.)
    """
    mean, var = acc.moments()
    std = var.sqrt()
    norm.set_stats(torch.stack([mean + std, mean - std]).float())


def init_normalizer_stats_from_tar(
    shard_paths: Sequence[Path],
    scene_norm: PatchStandardizer,
    cls_norm: CLSStandardizer,
    device: torch.device,
    max_samples: int,
) -> None:
    """Initialise standardizer stats from one or more WebDataset tar shards.

    Streams `cls.npy` and `ptch.npy` entries directly from each tar via the stdlib
    `tarfile` module and accumulates moments across ALL of `shard_paths`, taking up to
    `max_samples` samples PER SHARD (0 = the whole shard). Pass
    `loader.normalizer_shard_paths(cfg.normalizer_shards)` to get a seed-independent
    selection.
    """
    log.info("Computing normalizer stats from %d tar(s): %s",
             len(shard_paths), ", ".join(p.name for p in shard_paths))
    assert len(shard_paths) > 0, "shard_paths must not be empty"

    scene_acc, cls_acc = _MomentAcc(device), _MomentAcc(device)
    chunk = 256  # amortise the host->device copy without holding a whole shard

    for shard_path in shard_paths:
        cls_buf: list[np.ndarray] = []
        ptch_buf: list[np.ndarray] = []
        keys_seen: dict[str, dict[str, np.ndarray]] = {}
        n_shard = 0

        def flush() -> None:
            if not cls_buf:
                return
            cls_acc.add(torch.from_numpy(np.stack(cls_buf)).unsqueeze(1))  # [B, 1, D]
            scene_acc.add(torch.from_numpy(np.stack(ptch_buf)))            # [B, T, D]
            cls_buf.clear()
            ptch_buf.clear()

        with tarfile.open(shard_path, "r") as tf:
            for member in tf:
                if not member.isfile():
                    continue
                name = member.name
                # entries are <key>.cls.npy and <key>.ptch.npy
                if name.endswith(".cls.npy"):
                    key, kind = name[: -len(".cls.npy")], "cls"
                elif name.endswith(".ptch.npy"):
                    key, kind = name[: -len(".ptch.npy")], "ptch"
                else:
                    continue
                f = tf.extractfile(member)
                assert f is not None
                arr = np.load(io.BytesIO(f.read()))
                keys_seen.setdefault(key, {})[kind] = arr

                entry = keys_seen[key]
                if "cls" in entry and "ptch" in entry:
                    cls_buf.append(entry["cls"])
                    ptch_buf.append(entry["ptch"])
                    del keys_seen[key]
                    n_shard += 1
                    if len(cls_buf) >= chunk:
                        flush()
                    if max_samples > 0 and n_shard >= max_samples:
                        break
        flush()
        assert n_shard > 0, f"No cls/ptch pairs found in {shard_path}"
        log.info(f"  Collected {n_shard} samples from {shard_path.name}")

    _freeze_stats(scene_norm, scene_acc)
    _freeze_stats(cls_norm, cls_acc)
    log.info(f"  Scene/CLS stats from {scene_acc.n} samples across {len(shard_paths)} shard(s)")
    torch.cuda.empty_cache()


def init_normalizer_stats_from_tar_raw(
    shard_paths: Sequence[Path],
    scene_norm: PatchStandardizer,
    cls_norm: CLSStandardizer,
    *,
    image_size: int,
    compute_features: Callable[[Tensor], object],
    device: torch.device,
    max_samples: int,
    sub_batch: int = 64,
) -> None:
    """Initialise standardizer stats from one or more RAW (no-feature) WebDataset shards.

    Streams ``jpg`` images from each tar, decodes them to ``image_size``, and computes
    teacher features on the fly via ``compute_features`` (which returns an object exposing
    ``.patches`` [B, T, D] and ``.cls`` [B, D]). Teacher forwards run in sub-batches of
    ``sub_batch`` and are folded straight into the moment accumulators, so neither the
    images nor the features are held for the whole run.

    ``max_samples`` is PER SHARD, and unlike the precomputed path its 0-sentinel is capped
    (teacher forwards for a full shard are expensive), so pooling n shards costs n teacher
    passes over ``cap`` images each.

    Only reached for a fresh (step-0) run on raw shards — resumed runs load standardizer
    stats from the checkpoint and skip init entirely.
    """
    assert len(shard_paths) > 0, "shard_paths must not be empty"
    # max_samples<=0 means "use the whole shard"; computing teacher features for
    # a full 4096-image shard is many forwards — cap to keep init quick/bounded.
    cap = max_samples if max_samples > 0 else 2048
    log.info(
        "Computing normalizer stats from %d RAW tar(s) (teacher on the fly): %s, "
        "up to %d samples each",
        len(shard_paths), ", ".join(p.name for p in shard_paths), cap,
    )

    scene_acc, cls_acc = _MomentAcc(device), _MomentAcc(device)

    for shard_path in shard_paths:
        imgs: list[Tensor] = []
        with tarfile.open(shard_path, "r") as tf:
            for member in tf:
                if not member.isfile() or not member.name.endswith(".jpg"):
                    continue
                f = tf.extractfile(member)
                assert f is not None
                imgs.append(_decode_jpg(f.read(), image_size))
                if len(imgs) >= cap:
                    break

        n = len(imgs)
        assert n > 0, f"No jpg images found in {shard_path}"
        for i in range(0, n, sub_batch):
            batch = torch.stack(imgs[i : i + sub_batch]).to(device, non_blocking=True)
            feats = compute_features(batch)
            cls_acc.add(feats.cls.detach().float().unsqueeze(1))    # type: ignore[attr-defined]
            scene_acc.add(feats.patches.detach().float())           # type: ignore[attr-defined]
        log.info(f"  Collected {n} samples from {shard_path.name}")

    _freeze_stats(scene_norm, scene_acc)
    _freeze_stats(cls_norm, cls_acc)
    log.info(f"  Scene/CLS stats from {scene_acc.n} samples across {len(shard_paths)} "
             f"shard(s) (computed live)")
    torch.cuda.empty_cache()
