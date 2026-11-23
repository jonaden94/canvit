"""Human scanpaths from pysaliency -> tensors for a CanViT episode (design §2).

pysaliency is the authority for the data: its loaders build the ``Stimuli`` and
``ScanpathFixations`` (pixel coordinates, x/y origin top-left, pixel i spanning [i, i+1), fixation 0
= the start fixation), and its metrics score them (``evaluate.py``). This file only adds what a
CanViT episode needs on top: the stimulus letterboxed into the square scene and the scanpath in
scene coordinates [-1, 1] -- with the same letterbox as ``search.data.SearchDataset``, so a
fixation and the pixels under it cannot disagree.

Every dataset comes from pysaliency's own loader (:func:`load_dataset`, ``DATASETS``), reading the cache
under ``$SCANPATH_DATA_DIR`` (built 2026-11-20, journal). COCO-Search18 comes from
``pysaliency.external_datasets.get_COCO_Search18`` with
``merge_tasks=False, unique_images=True`` (one stimulus per (image, target category), the cue in
the stimulus attribute ``task``; the default ``merge_tasks=True, unique_images=True`` raises in
pysaliency 0.2.23). Its download step is pointed at local copies of the four release files
(:func:`local_mirror`), each checked against pysaliency's own md5 -- the loader is otherwise
pysaliency's, unchanged. Checked 2026-11-18: u13879's files match all four md5s (pysaliency's TA
hash carries a stray trailing '.', which ``download_and_check`` would only have warned about).

A sample (one human scanpath):

    image:    float [3, S, S]  ImageNet-normalised, letterboxed (zero padding)
    cue:      long  []         index into COCO_SEARCH18_CLASSES; -1 = free viewing
    fix:      float [L, 2]     the fixations (x, y) in scene coordinates, NaN-padded by the collate
    n_fix:    long  []         number of fixations (fixation 0 included)
    boxes:    float [1, 4]     target box, scene xyxy (NaN: no target / target absent)
    content:  float [4]        the stimulus inside the scene, xyxy [-1, 1]
    hw:       long  [2]        the stimulus size in pixels (h, w)
    scanpath: long  []         index into ``fixations.scanpaths`` (pysaliency's ``scanpath_index``)
"""

from __future__ import annotations

import hashlib
import logging
import math
import os
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

import numpy as np
import torch
from torch import Tensor
from torch.utils.data import DataLoader, Dataset, Subset
from torchvision.transforms import v2
from torchvision.transforms.v2 import functional as TF

from canvit.core.preprocess import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from canvit.search.data import COCO_SEARCH18_CLASSES, letterbox_hw

log = logging.getLogger(__name__)


# --- loading (pysaliency) ------------------------------------------------------------------------

@contextmanager
def local_mirror(module, files: dict[str, Path]) -> Iterator[None]:
    """Within the block, ``module.download_and_check(url, target, md5)`` symlinks the local copy of
    ``basename(url)`` to ``target`` after checking its md5 (refused on mismatch, unlike pysaliency's
    warning), instead of downloading."""
    original = module.download_and_check

    def from_mirror(url, target, md5_hash, verify_ssl=True):
        name = os.path.basename(url)
        if name not in files:
            raise FileNotFoundError(f"{name} ({url}) has no local copy; pass it in `files`")
        h = hashlib.md5()
        with open(files[name], "rb") as f:
            for chunk in iter(lambda: f.read(1 << 24), b""):
                h.update(chunk)
        if h.hexdigest() != md5_hash.rstrip("."):
            raise ValueError(f"{files[name]}: md5 {h.hexdigest()} != pysaliency's {md5_hash}")
        os.symlink(Path(files[name]).resolve(), target)

    module.download_and_check = from_mirror
    try:
        yield
    finally:
        module.download_and_check = original


def _modify_image_numpy2(source_filename, target_filename, rst):
    """pysaliency 0.2.23's ``coco_search18._modify_image`` with its one numpy-2 failure fixed:
    ``image_data[pos] += -1`` on uint8 raises OverflowError under numpy >= 2 (NEP 50). Same random
    draws, same +-1 (never wraps: -1 only on a value > 0), so the stimuli -- and pysaliency's
    hash-based stimulus ids -- are those pysaliency builds under numpy 1. Reported upstream: TODO."""
    from PIL import Image
    image = Image.open(source_filename)
    image_data = np.array(image)
    width, height = image.size
    x_pos, y_pos = rst.randint(0, width), rst.randint(0, height)
    if image_data.ndim == 3:
        channel = rst.randint(0, image_data.shape[-1])
        image_pos = (y_pos, x_pos, channel)
    else:
        image_pos = (y_pos, x_pos)
    offset = -1 if image_data[image_pos] > 0 else 1
    image_data[image_pos] = int(image_data[image_pos]) + offset
    Image.fromarray(image_data).save(target_filename)


def coco_search18(location: Path, files: dict[str, Path]):
    """``(stimuli_train, fixations_train, stimuli_val, fixations_val)`` of COCO-Search18 split1, target
    present and absent, from pysaliency's loader. The first call builds pysaliency's cache under
    ``location/COCO-Search18_no-task-merge_unique-images/`` from ``files`` (the release's
    ``COCOSearch18-images-{TP,TA}.zip``, ``COCOSearch18-fixations-TP.zip``,
    ``coco_search18_fixations_TA_trainval.json``); later calls read the cache. Set ``TMPDIR`` to a
    big filesystem for the first call: pysaliency unpacks there (~20 GB; a node's /tmp is too small)."""
    from pysaliency.external_datasets import coco_search18 as cs18
    original = cs18._modify_image
    cs18._modify_image = _modify_image_numpy2
    try:
        with local_mirror(cs18, files):
            return cs18.get_COCO_Search18(location=str(location), merge_tasks=False, unique_images=True)
    finally:
        cs18._modify_image = original


RELEASE_FILES = ("COCOSearch18-images-TP.zip", "COCOSearch18-images-TA.zip", "COCOSearch18-fixations-TP.zip",
                 "coco_search18_fixations_TA_trainval.json")

DATASETS: dict[str, tuple[str, ...]] = {
    "coco_search18": ("train", "validation"),  # split1; target present AND absent (evaluate.select by condition)
    "coco_freeview": ("train", "validation"),
    "daemons": ("train", "validation"),
    "mit1003": ("all",),  # with the initial central fixation (as DeepGaze III), so it is history
    "cat2000": ("all",),  # train v1.1 (the test set has no public fixations)
    "figrim": ("all",),
}
"""The parts each dataset has: a train/validation split where the release has one, else the whole set."""


def eval_part(dataset: str) -> str:
    """The part scored: the validation split where one exists, else the whole set."""
    return "validation" if "validation" in DATASETS[dataset] else "all"


def fit_part(dataset: str) -> str:
    """The part the centre bias (and the temperature diagnostic) is fitted on: train where it exists, else the
    evaluated set itself (the centre bias is leave-one-image-out anyway)."""
    return "train" if "train" in DATASETS[dataset] else "all"


def load_dataset(dataset: str, part: str, *, location: Path, release_dirs: tuple[Path, ...] = ()):
    """``(stimuli, fixations)`` of one part of a dataset, from pysaliency's loader (cached under ``location``),
    off-stimulus fixations clipped onto the stimulus (:func:`clip_to_stimuli`). ``release_dirs``: only for
    COCO-Search18's first build (the four release files)."""
    from pysaliency import external_datasets as ed
    if part not in DATASETS[dataset]:
        raise ValueError(f"{dataset} has the parts {DATASETS[dataset]}, not {part!r}")
    loc = str(location)
    if dataset == "coco_search18":
        files = {name: d / name for d in release_dirs for name in RELEASE_FILES if (d / name).exists()}
        st, ft, sv, fv = coco_search18(location, files)
        stimuli, fixations = (st, ft) if part == "train" else (sv, fv)
    elif dataset == "coco_freeview":
        stimuli, fixations = getattr(ed, f"get_COCO_Freeview_{part}")(location=loc)
    elif dataset == "daemons":
        stimuli, fixations = getattr(ed, f"get_DAEMONS_{part}")(location=loc)
    elif dataset == "mit1003":
        stimuli, fixations = ed.get_mit1003_with_initial_fixation(location=loc)
    elif dataset == "cat2000":
        stimuli, fixations = ed.get_cat2000_train(location=loc, version="1.1")
    else:
        stimuli, fixations = ed.get_FIGRIM(location=loc)
    # Do not keep decoded images: the centre bias looks stimuli up by pixel hash (`stimulus_ids.index`), which
    # decodes EVERY stimulus once; with pysaliency's default cache that kept all 4326 COCO-Search18 training
    # images (~22 GB, the OOM of 2026-11-20). The hashes themselves stay cached.
    stimuli.cached = False
    fixations, n = clip_to_stimuli(stimuli, fixations)
    log.info("%s %s: %d stimuli, %d scanpaths; clipped %d off-stimulus fixations onto the stimulus", dataset, part,
             len(stimuli), len(fixations.scanpaths), n)
    return stimuli, fixations


def clip_to_stimuli(stimuli, fixations):
    """-> (new ``ScanpathFixations`` with every fixation clipped onto its stimulus, number clipped).

    pysaliency does not clip, and an off-stimulus fixation indexes outside the map; the convention
    (pysaliency's ``clip_out_of_stimulus_fixations``, which fails on ``ScanpathFixations`` in 0.2.23)
    is to clip onto the edge. Clipped on a COPY of the scanpaths and rebuilt: ``ScanpathFixations``
    copies the coordinates at construction, so editing its scanpaths afterwards would not reach the
    scored fixations or their histories."""
    import pysaliency
    sp = fixations.scanpaths.copy()
    sizes = np.array([stimuli.sizes[int(n)] for n in sp.n], dtype=np.float64).reshape(-1, 2)  # (h, w) per scanpath
    clipped = 0
    for arr, lim in ((sp.xs, sizes[:, 1]), (sp.ys, sizes[:, 0])):
        data = arr._data
        valid = ~np.isnan(data)
        new = np.clip(data, 0, (lim - 1e-6)[:, None])
        clipped += int(np.sum(valid & (new != data)))
        data[valid] = new[valid]
    return pysaliency.ScanpathFixations(sp), clipped


# --- scene geometry ------------------------------------------------------------------------------

def letterbox_geometry(h: int, w: int, scene: int) -> tuple[int, int, int, int]:
    """(content_h, content_w, top, left) of an (h, w) image letterboxed into the scene -- the numbers
    ``SearchDataset.__getitem__`` uses."""
    ch, cw = letterbox_hw(h, w, scene)
    return ch, cw, (scene - ch) // 2, (scene - cw) // 2


def pixel_to_scene(xy: Tensor, *, hw: tuple[int, int], scene: int) -> Tensor:
    """``xy`` [..., 2] stimulus pixels (x, y), continuous -> scene [-1, 1]. Not clamped."""
    h, w = hw
    ch, cw, top, left = letterbox_geometry(h, w, scene)
    x = (xy[..., 0] * (cw / w) + left) / scene * 2 - 1
    y = (xy[..., 1] * (ch / h) + top) / scene * 2 - 1
    return torch.stack([x, y], dim=-1)


def clamp_to_content(fix: Tensor, content: Tensor) -> Tensor:
    """``fix`` [B, 2] (x, y) clamped strictly inside ``content`` [B, 4]: a fixation off the stimulus
    (the eye tracker records a few) is glimpsed at the nearest edge."""
    eps = 1e-6
    x = torch.minimum(torch.maximum(fix[:, 0], content[:, 0] + eps), content[:, 2] - eps)
    y = torch.minimum(torch.maximum(fix[:, 1], content[:, 1] + eps), content[:, 3] - eps)
    return torch.stack([x, y], dim=-1)


# --- tensors for the episode ---------------------------------------------------------------------

def _task_names(stimuli) -> list[str] | None:
    """The cue per stimulus as a class name, or None for free viewing (no ``task`` attribute)."""
    if "task" not in getattr(stimuli, "attributes", {}):
        return None
    from pysaliency.external_datasets.coco_search18 import TASKS
    return [TASKS[int(t)] for t in stimuli.attributes["task"]]


class ScanpathDataset(Dataset):
    """pysaliency (stimuli, scanpath fixations) -> one sample per scanpath. Deterministic."""

    def __init__(self, stimuli, fixations, *, scene_size: int):
        self.stimuli = stimuli
        self.scanpaths = fixations.scanpaths
        self.scene = scene_size
        names = _task_names(stimuli)
        self.cues = None if names is None else [COCO_SEARCH18_CLASSES.index(n) for n in names]
        self.bbox = self.scanpaths.scanpath_attributes.get("target_bbox")  # xywh pixels, NaN if absent

    def __len__(self) -> int:
        return len(self.scanpaths)

    def __getitem__(self, i: int) -> dict:
        n = int(self.scanpaths.n[i])
        img = np.asarray(self.stimuli.stimuli[n])
        if img.ndim == 2:
            img = np.repeat(img[..., None], 3, axis=-1)
        img = torch.from_numpy(np.ascontiguousarray(img[..., :3])).permute(2, 0, 1)
        h, w = img.shape[-2:]
        ch, cw, top, left = letterbox_geometry(h, w, self.scene)
        img = v2.Resize((ch, cw), antialias=True)(img)
        img = TF.pad(img, [left, top, self.scene - cw - left, self.scene - ch - top], fill=0)
        image = TF.normalize(TF.to_dtype(img, torch.float32, scale=True),
                             mean=list(IMAGENET_DEFAULT_MEAN), std=list(IMAGENET_DEFAULT_STD))
        hw = (int(h), int(w))
        L = int(self.scanpaths.length[i])
        xy = torch.tensor(np.stack([self.scanpaths.xs[i][:L], self.scanpaths.ys[i][:L]], -1), dtype=torch.float32)
        boxes = torch.full((1, 4), math.nan)
        if self.bbox is not None and np.isfinite(self.bbox[i]).all():
            x, y, bw, bh = (float(v) for v in self.bbox[i])
            boxes = pixel_to_scene(torch.tensor([[x, y], [x + bw, y + bh]]), hw=hw, scene=self.scene).reshape(1, 4)
        content = pixel_to_scene(torch.tensor([[0.0, 0.0], [float(w), float(h)]]), hw=hw, scene=self.scene)
        return {"image": image.as_subclass(Tensor),
                "cue": torch.tensor(-1 if self.cues is None else self.cues[n], dtype=torch.long),
                "fix": pixel_to_scene(xy, hw=hw, scene=self.scene), "n_fix": torch.tensor(L, dtype=torch.long),
                "boxes": boxes, "content": content.reshape(4), "hw": torch.tensor(hw, dtype=torch.long),
                "scanpath": torch.tensor(i, dtype=torch.long)}


def collate_scanpath(batch: list[dict]) -> dict:
    """Stack; pad ``fix`` to the batch's longest scanpath with NaN."""
    L = max(s["fix"].shape[0] for s in batch)
    fix = torch.full((len(batch), L, 2), math.nan)
    for i, s in enumerate(batch):
        fix[i, : s["fix"].shape[0]] = s["fix"]
    keys = ("image", "cue", "n_fix", "boxes", "content", "hw", "scanpath")
    out = {k: torch.stack([s[k] for s in batch]) for k in keys}
    return out | {"fix": fix}


def make_scanpath_loader(stimuli, fixations, *, scene_size: int, batch_size: int, num_workers: int) -> DataLoader:
    """The scanpaths with at least one fixation after the start (the others have nothing to score), in order;
    samples keep their index into ``fixations.scanpaths``."""
    ds = ScanpathDataset(stimuli, fixations, scene_size=scene_size)
    keep = np.flatnonzero(np.asarray(fixations.scanpaths.length) >= 2).tolist()
    return DataLoader(Subset(ds, keep), batch_size=batch_size, shuffle=False, num_workers=num_workers,
                      collate_fn=collate_scanpath, pin_memory=True)


__all__ = ["DATASETS", "ScanpathDataset", "clamp_to_content", "clip_to_stimuli", "coco_search18", "collate_scanpath",
           "eval_part", "fit_part", "letterbox_geometry", "load_dataset", "local_mirror", "make_scanpath_loader",
           "pixel_to_scene"]
