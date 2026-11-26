"""A curated search release -> the search sample contract (``data.py``).

A release is a frozen folder ``$DATA_ROOT/search_release/<cut>-<version>/`` built by get_datasets; its
``README.md`` is the spec (sections cited below as "spec N"). canvit reads only that folder and the pack shards
it names; it imports nothing from get_datasets (contract: ``canvit_claude/shared/search_dataset_contract.md``).

Every sample is a dict::

    image:   float [3, S, S]  ImageNet-normalised, letterboxed into the square scene (as COCO-Search18)
    boxes:   float [N, 4]     the target box, scene-normalised [-1, 1] xyxy (N = 1 present, 0 absent)
    present: bool  []
    text:    str              the trial's prompt text (the cue; no class index, owner 2026-11-25)
    band:    long  []         val only: class-size band, an index into ``BANDS`` (spec 5)

and with ``cfg.visual_prompts`` (a release with visual prompts, spec 7) also::

    vp_feat:  float [14, 768] the visual prompt's DINOv3 cell features (1x1, 2x2, 3x3 grids on its mask's box)
    vp_cov:   float [14]      mask coverage of each cell (``vp_boxes.coverage``)
    vp_valid: bool  []        the trial has a visual prompt (``vp_idx >= 0``); False -> vp_feat, vp_cov all zero

Three parts (plus ``VisualPrompts``, the per-worker reader of ``vp_features.npy``):

- ``ReleasePictures``: ``images.parquet`` as plain numpy arrays (cheap to hand to workers) and the picture
  reader of spec 3 (one ``os.pread`` from the pack shard, PIL ``draft``, the orientation fix).
- ``ReleaseEpochs`` + ``EpochStream``: the training trials. The vendored reference sampler
  (``release_sampler.py``) builds an epoch's trial list in the MAIN process (~9 GB, ~16 s per epoch); each
  rank takes ``trials[rank::world]`` cut to ``len // world``, and the epochs follow each other as one endless
  stream. Workers receive single ``Trial`` tuples through the DataLoader's index queue, never the sampler.
  A position in the stream is (seed, epoch, offset), and an epoch's length is known without building it
  (``ReleaseEpochs.size``), so a resume at step s starts exactly at stream position s x batch_size.
- ``make_release_loaders``: the DataLoaders. Workers are SPAWNED, not forked: a forked worker would share the
  parent's 9 GB of pandas objects, and Python's reference counts and GC write to those pages, copying them
  into every worker.
"""

from __future__ import annotations

import io
import json
import logging
import math
import os
from collections.abc import Callable, Iterator
from pathlib import Path
from typing import NamedTuple

import numpy as np
import pandas as pd
import torch
from PIL import Image, ImageOps
from torch.utils.data import DataLoader, Sampler
from torchvision import tv_tensors

from .config import SearchAugConfig, SearchConfig
from .data import SearchDataset, collate_search

log = logging.getLogger(__name__)

Image.MAX_IMAGE_PIXELS = None  # release pictures reach 219 MP (spec 3)

FIXES = ("none", "exif", "rotate_90")
BANDS = ("lt10", "10-99", "100-299", "300-999", "1k-10k", "ge10k")
"""Class-size bands of spec 5: n = training examples of the class in ``examples.parquet``."""
_BAND_EDGES = (10, 100, 300, 1000, 10000)


class Trial(NamedTuple):
    """One search trial, as small as possible: it is pickled into a worker for every sample."""
    image_idx: int
    box: tuple[float, float, float, float] | None  # box_x, box_y, box_w, box_h (fractions after the fix); None = absent
    text: str
    band: int = -1  # val only (index into BANDS)
    vp_idx: int = -1  # visual prompt: row of vp_boxes / vp_features; -1 = none (or visual prompts off)


def release_meta(release_dir: Path) -> dict:
    return json.loads((Path(release_dir) / "meta.json").read_text())


# --- pictures (spec 3) ------------------------------------------------------------------------------------

class ReleasePictures:
    """``images.parquet`` (row = ``image_idx``) as numpy arrays, plus the reader. Opens one fd per shard per
    process, lazily; the fds are not pickled."""

    def __init__(self, release_dir: Path):
        meta = release_meta(release_dir)
        t = pd.read_parquet(Path(release_dir) / "images.parquet",
                            columns=["image_idx", "shard", "offset", "size", "fix", "width", "height"])
        if not (t["image_idx"].to_numpy() == np.arange(len(t))).all():
            raise ValueError("images.parquet: image_idx is not the row number")
        self.shard_names, codes = np.unique(t["shard"].to_numpy(dtype=object), return_inverse=True)
        self.shard_names = [str(s) for s in self.shard_names]
        self.shard = codes.astype(np.int16)
        self.offset = t["offset"].to_numpy(np.int64)
        self.size = t["size"].to_numpy(np.int64)
        fix = t["fix"].to_numpy(dtype=object)
        unknown = set(fix) - set(FIXES)
        if unknown:
            raise ValueError(f"images.parquet: unknown orientation fix {unknown}")
        self.fix = np.select([fix == f for f in FIXES], list(range(len(FIXES)))).astype(np.int8)
        self.width = t["width"].to_numpy(np.int32)
        self.height = t["height"].to_numpy(np.int32)
        self.shard_path = meta["shard_path"]
        self.pack_dir = meta["pack_dir"]
        self._fds: dict[int, int] = {}

    def __len__(self) -> int:
        return len(self.offset)

    def __getstate__(self):
        return {k: v for k, v in self.__dict__.items() if k != "_fds"}

    def __setstate__(self, state):
        self.__dict__.update(state)
        self._fds = {}

    def _fd(self, code: int) -> int:
        fd = self._fds.get(code)
        if fd is None:
            path = self.shard_path.format(pack_dir=self.pack_dir, shard=self.shard_names[code])
            fd = self._fds[code] = os.open(path, os.O_RDONLY)
        return fd

    def read(self, i: int, *, long_side: int | None = None) -> Image.Image:
        """Picture ``i`` in RGB, oriented as its boxes are. ``long_side``: let a JPEG decode at 1/2, 1/4 or 1/8
        scale as long as its longer side stays >= ``long_side`` (PIL ``draft``); None = full size."""
        n = int(self.size[i])
        data = os.pread(self._fd(int(self.shard[i])), n, int(self.offset[i]))
        if len(data) != n:
            raise OSError(f"image_idx {i}: read {len(data)} of {n} bytes")
        im = Image.open(io.BytesIO(data))
        if long_side is not None:
            w0, h0 = im.size
            s = long_side / max(w0, h0)
            if s < 1:
                im.draft("RGB", (math.ceil(w0 * s), math.ceil(h0 * s)))
        fix = FIXES[self.fix[i]]
        if fix == "exif":
            im = ImageOps.exif_transpose(im)
        elif fix == "rotate_90":
            im = im.transpose(Image.Transpose.ROTATE_90)  # 90 degrees counter-clockwise
        im = im.convert("RGB")
        # The fixed size is (width, height); a draft scales both sides by the same power of 2, rounding up.
        W, H = int(self.width[i]), int(self.height[i])
        w, h = im.size
        if (w, h) != (W, H) and (long_side is None or abs(w * H - h * W) > W + H):
            raise ValueError(f"image_idx {i}: decoded {w}x{h}, images.parquet says {W}x{H} after fix {fix!r}")
        return im


class VisualPrompts:
    """The visual prompts of a release (spec 7): ``vp_features.npy`` opened by mmap in each process on first use
    (never loaded whole; the memmap is not pickled into workers) and the cell coverages as one float32 array."""

    def __init__(self, release_dir: Path):
        release_dir = Path(release_dir)
        self.path = release_dir / "vp_features.npy"
        if not self.path.exists():
            raise FileNotFoundError(f"{release_dir} has no visual prompts (vp_features.npy); spec section 7")
        b = pd.read_parquet(release_dir / "vp_boxes.parquet", columns=["vp_idx", "coverage"])
        if not (b["vp_idx"].to_numpy() == np.arange(len(b))).all():
            raise ValueError("vp_boxes.parquet: vp_idx is not the row number")
        self.coverage = np.stack(b["coverage"].to_numpy()).astype(np.float32)  # [n, 14]
        self._feat: np.ndarray | None = None

    def __getstate__(self):
        return {k: v for k, v in self.__dict__.items() if k != "_feat"}

    def __setstate__(self, state):
        self.__dict__.update(state)
        self._feat = None

    @property
    def features(self) -> np.ndarray:
        if self._feat is None:
            self._feat = np.load(self.path, mmap_mode="r")
            if self._feat.shape[0] != len(self.coverage):
                raise ValueError(f"{self.path}: {self._feat.shape[0]} rows, vp_boxes has {len(self.coverage)}")
        return self._feat

    def sample(self, vp_idx: int) -> dict:
        """The sample's ``vp_*`` keys for visual prompt ``vp_idx`` (-1: zeros, ``vp_valid`` False)."""
        if vp_idx < 0:
            _, n_cells, dim = self.features.shape
            return {"vp_feat": torch.zeros(n_cells, dim), "vp_cov": torch.zeros(n_cells),
                    "vp_valid": torch.tensor(False)}
        return {"vp_feat": torch.from_numpy(np.asarray(self.features[vp_idx], dtype=np.float32)),
                "vp_cov": torch.from_numpy(self.coverage[vp_idx].copy()), "vp_valid": torch.tensor(True)}


def coco_test_prompts(release_dir: Path, pairs: list[tuple[str, str]]) -> list[int]:
    """A fixed visual prompt per COCO-Search18 trial (image name, category), for the zero-shot TEST only (training
    never reads ``vp_test``): among the release's ``vp_test`` boxes of the category, the one with the smallest
    ``blake2b(f"{image}|{category}|{box_id}", digest_size=8)`` (big-endian). -1 if the category has none."""
    import hashlib

    release_dir = Path(release_dir)
    t = pd.read_parquet(release_dir / "vp_test.parquet", columns=["category", "vp_idx"])
    box_id = pd.read_parquet(release_dir / "vp_boxes.parquet", columns=["box_id"])["box_id"].to_numpy(dtype=object)
    by_cat = {c: g.to_numpy() for c, g in t.groupby("category")["vp_idx"]}

    def key(image: str, cat: str, i: int) -> int:
        h = hashlib.blake2b(f"{image}|{cat}|{box_id[i]}".encode(), digest_size=8)
        return int.from_bytes(h.digest(), "big")

    return [int(min(by_cat[c], key=lambda i: key(name, c, i))) if c in by_cat else -1 for name, c in pairs]


class ReleaseDataset(SearchDataset):
    """``Trial`` -> the sample contract. The scene transform (augmentation, retention retries, letterbox) is
    ``SearchDataset``'s, the same as COCO-Search18's. ``aug=None`` = validation."""

    def __init__(self, pictures: ReleasePictures, *, scene_size: int, aug: SearchAugConfig | None,
                 visual_prompts: VisualPrompts | None = None):
        super().__init__([], scene_size=scene_size, aug=aug, visual_prompts=visual_prompts)
        self.pictures = pictures
        # Decode no smaller than the scene needs: a crop keeping area fraction a keeps sqrt(a) of each side,
        # which is then resized to the letterboxed content (longer side = scene_size).
        self.long_side = (math.ceil(scene_size / math.sqrt(aug.crop_scale[0])) if aug is not None and aug.crop
                          else scene_size)

    def __getitem__(self, trial: Trial) -> dict:  # type: ignore[override]
        im = self.pictures.read(trial.image_idx, long_side=self.long_side)
        w, h = im.size
        img = tv_tensors.Image(torch.from_numpy(np.array(im)).permute(2, 0, 1).contiguous())
        if trial.box is None:
            xyxy = torch.zeros(0, 4)
        else:
            bx, by, bw, bh = trial.box
            xyxy = torch.tensor([[bx * w, by * h, (bx + bw) * w, (by + bh) * h]], dtype=torch.float32)
        boxes = tv_tensors.BoundingBoxes(xyxy, format="XYXY", canvas_size=(h, w), clamping_mode=None)
        img, boxes = self._transform(img, boxes)
        image, b = self._scene(img, boxes)
        out = {"image": image, "boxes": b, "present": torch.tensor(trial.box is not None), "text": trial.text}
        if trial.band >= 0:
            out["band"] = torch.tensor(trial.band, dtype=torch.long)
        if self.visual_prompts is not None:
            out.update(self.visual_prompts.sample(trial.vp_idx))
        return out


# --- training trials (spec 4) -------------------------------------------------------------------------------

class ReleaseEpochs:
    """The reference sampler (vendored, ``release_sampler.py``) at the release's defaults, in this process."""

    def __init__(self, release_dir: Path, *, seed: int):
        from .release_sampler import SearchSampler
        self.release_dir = Path(release_dir)
        self.seed = seed
        self.defaults = release_meta(release_dir)["sampler_defaults"]
        self.sampler = SearchSampler(str(release_dir), **self.defaults)

    def size(self, e: int) -> int:
        """Trials in epoch ``e``, without building it: the epoch's first random draw (stochastic rounding of the
        expected draws) fixes the present count, and the absent count follows from it. Exact as long as no
        prompt runs out of negatives; ``trials`` checks it on every epoch it builds."""
        s = self.sampler
        rng = np.random.default_rng([self.seed, e])
        w = s.w
        n_present = int((np.floor(w).astype(int) + (rng.random(len(w)) < w - np.floor(w))).sum())
        return n_present + int(round(n_present * s.absent / (1 - s.absent)))

    def frame(self, e: int) -> pd.DataFrame:
        df = self.sampler.epoch(e, seed=self.seed)
        if len(df) != self.size(e):
            raise RuntimeError(f"epoch {e} has {len(df)} trials, size() predicted {self.size(e)}: a prompt ran out "
                               "of negatives, so resume positions computed from size() would be wrong.")
        return df

    def trials(self, e: int, *, rank: int, world_size: int, visual_prompts: bool = False) -> list[Trial]:
        """This rank's share (``rank_share``) of epoch ``e``; ``vp_idx`` only with ``visual_prompts``."""
        part = rank_share(self.frame(e), rank=rank, world_size=world_size)
        idx = part["image_idx"].to_numpy(np.int64)
        present = part["present"].to_numpy(bool)
        box = part[["box_x", "box_y", "box_w", "box_h"]].to_numpy(np.float64)
        text = part["text"].to_numpy(dtype=object)
        vp = part["vp_idx"].to_numpy(np.int64) if visual_prompts else np.full(len(part), -1)
        return [Trial(int(i), tuple(float(v) for v in b) if p else None, str(t), vp_idx=int(v))  # type: ignore[arg-type]
                for i, p, b, t, v in zip(idx, present, box, text, vp)]


def rank_share(df: pd.DataFrame, *, rank: int, world_size: int) -> pd.DataFrame:
    """Rows ``rank::world_size`` of an epoch, cut to ``len // world_size``: disjoint across ranks, the same
    count on every rank (the last ``len % world_size`` rows of the epoch are left out)."""
    return df.iloc[rank::world_size].iloc[:len(df) // world_size]


class EpochStream(Sampler):
    """One rank's endless stream of training trials: epoch 0, 1, 2, ... back to back, starting at stream
    position ``start`` (= the resume step x batch_size). ``length(e)`` is this rank's trial count of epoch e
    (cheap), ``trials(e)`` builds them (expensive, once per epoch)."""

    def __init__(self, *, length: Callable[[int], int], trials: Callable[[int], list[Trial]], start: int = 0):
        self.length, self.trials, self.start = length, trials, start

    def locate(self, pos: int) -> tuple[int, int]:
        """(epoch, offset in it) of stream position ``pos``."""
        e = 0
        while pos >= (n := self.length(e)):
            pos -= n
            e += 1
        return e, pos

    def __iter__(self) -> Iterator[Trial]:
        e, off = self.locate(self.start)
        while True:
            ts = self.trials(e)
            if len(ts) != self.length(e):
                raise RuntimeError(f"epoch {e}: {len(ts)} trials, length() says {self.length(e)}")
            log.info("search release: epoch %d (%d trials on this rank), from offset %d", e, len(ts), off)
            yield from ts[off:]
            e, off = e + 1, 0


# --- validation (spec 5) ----------------------------------------------------------------------------------

def class_band(n_examples: np.ndarray) -> np.ndarray:
    return np.searchsorted(np.asarray(_BAND_EDGES), n_examples, side="right")


def val_trials(release_dir: Path, *, visual_prompts: bool = False) -> list[Trial]:
    """All of ``val.parquet``, present and absent trials interleaved (so the first val batch, the one the
    figure draws, holds both; metrics do not depend on the order). With ``visual_prompts``, each trial's fixed
    ``vp_idx`` from ``val.parquet``."""
    release_dir = Path(release_dir)
    v = pd.read_parquet(release_dir / "val.parquet")
    n_cls = pd.read_parquet(release_dir / "examples.parquet", columns=["class"])["class"].value_counts()
    band = class_band(v["class"].map(n_cls).fillna(0).to_numpy(np.int64))
    box = v[["box_x", "box_y", "box_w", "box_h"]].to_numpy(np.float64)
    vp = v["vp_idx"].to_numpy(np.int64) if visual_prompts else np.full(len(v), -1)
    out = [Trial(int(i), tuple(float(x) for x in b) if p else None, str(t), int(k), int(q))  # type: ignore[arg-type]
           for i, p, b, t, k, q in zip(v["image_idx"].to_numpy(), v["present"].to_numpy(bool), box,
                                       v["text"].to_numpy(dtype=object), band, vp)]
    tp = [t for t in out if t.box is not None]
    ta = [t for t in out if t.box is None]
    n = min(len(tp), len(ta))
    return [t for pair in zip(tp, ta) for t in pair] + tp[n:] + ta[n:]


# --- loaders ----------------------------------------------------------------------------------------------

def _loader_kw(cfg: SearchConfig) -> dict:
    kw: dict = dict(num_workers=cfg.num_workers, pin_memory=True, collate_fn=collate_search)
    if cfg.num_workers > 0:
        kw.update(multiprocessing_context="spawn", persistent_workers=True)
    return kw


def _visual_prompts(cfg: SearchConfig) -> VisualPrompts | None:
    return VisualPrompts(cfg.release_dir) if cfg.visual_prompts else None


def make_release_val_loader(cfg: SearchConfig, pictures: ReleasePictures | None = None,
                            vp: VisualPrompts | None = None) -> DataLoader:
    pictures = pictures if pictures is not None else ReleasePictures(cfg.release_dir)
    vp = vp if vp is not None else _visual_prompts(cfg)
    ds = ReleaseDataset(pictures, scene_size=cfg.scene_size, aug=None, visual_prompts=vp)
    trials = val_trials(cfg.release_dir, visual_prompts=cfg.visual_prompts)
    return DataLoader(ds, sampler=trials, batch_size=cfg.eval_batch_size, **_loader_kw(cfg))  # type: ignore[arg-type]


def make_release_loaders(cfg: SearchConfig, *, world_size: int = 1, rank: int = 0,
                         start_step: int = 0) -> tuple[DataLoader, DataLoader]:
    """(train, val). Train is endless and starts at step ``start_step`` of this (seed, world_size,
    batch_size) stream."""
    meta = release_meta(cfg.release_dir)
    log.info("search release %s (%s), seed %d, rank %d/%d, start step %d", meta["name"], cfg.release_dir,
             cfg.seed, rank, world_size, start_step)
    pictures = ReleasePictures(cfg.release_dir)
    vp = _visual_prompts(cfg)
    epochs = ReleaseEpochs(cfg.release_dir, seed=cfg.seed)
    stream = EpochStream(length=lambda e: epochs.size(e) // world_size,
                         trials=lambda e: epochs.trials(e, rank=rank, world_size=world_size,
                                                        visual_prompts=cfg.visual_prompts),
                         start=start_step * cfg.batch_size)
    ds = ReleaseDataset(pictures, scene_size=cfg.scene_size, aug=cfg.aug, visual_prompts=vp)
    train = DataLoader(ds, sampler=stream, batch_size=cfg.batch_size, drop_last=True,  # type: ignore[arg-type]
                       **_loader_kw(cfg))
    return train, make_release_val_loader(cfg, pictures, vp)


__all__ = ["BANDS", "EpochStream", "ReleaseDataset", "ReleaseEpochs", "ReleasePictures", "Trial", "VisualPrompts",
           "class_band", "coco_test_prompts", "make_release_loaders", "rank_share", "make_release_val_loader",
           "release_meta", "val_trials"]
