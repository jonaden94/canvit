"""COCO-Search18 adapter -> the dataset-independent search sample contract.

Every sample is a dict:

    image:   float [3, S, S]  ImageNet-normalised, letterboxed into the square scene
    target:  long  []         class index into ``classes`` (the cue)
    boxes:   float [N, 4]     target boxes, scene-normalised [-1, 1] xyxy
    present: bool  []         target present (N >= 1) or absent (N == 0)

A new search dataset needs only an adapter that produces this; nothing downstream (head,
loss, policy, eval) reads anything dataset-specific.

COCO-Search18 facts this relies on (checked 2026-11-05, see claude_dev/search/00-design.md):
the TP fixation JSONs hold one record per (subject, trial), 10 subjects per
(image, target) pair, with the same target bbox ``[x, y, w, h]`` (1680x1050 px) for every
subject -- so the records are deduplicated on ``(name, task)``. The target-absent (TA)
trials are ``coco_search18_fixations_TA_trainval.json`` (no bbox; a ``split`` field with the
same 2163 train / 326 valid pairs as TP split1) and their images are a separate download
(checked 2026-11-06: all 2489 train/valid TA pairs present, 1680x1050 like TP).

ON-DISK LAYOUT (only this one is supported): the RESTRUCTURED release of 2026-11-06 (see
``README.md`` in the dataset root):

    images/<name>.jpg      all 5317 padded 1680x1050 images, TP and TA, one flat folder
    fixations/coco_search18_fixations_{TP_{train,valid}_split{1,2},TA_trainval}.json

The release's per-task folders (``images/<task>/``, ``coco_search18_images_TA/<task>/``)
held byte-identical copies under the same names, so a name identifies the image; whether
it is target-present depends on the (image, task) pair, which the fixation files give.
"""

from __future__ import annotations

import json
import logging
import math
from dataclasses import dataclass
from pathlib import Path

import torch
from torch import Tensor
from torch.utils.data import DataLoader, Dataset
from torchvision import tv_tensors
from torchvision.io import ImageReadMode, decode_image, read_file
from torchvision.transforms import v2
from torchvision.transforms.v2 import functional as TF

from canvit.core.preprocess import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD

from .config import SearchAugConfig, SearchConfig

log = logging.getLogger(__name__)


@dataclass(frozen=True)
class SearchRecord:
    path: Path
    target: int
    bbox_xywh: tuple[float, float, float, float] | None  # image pixels; None = target absent


COCO_SEARCH18_CLASSES = (
    "bottle", "bowl", "car", "chair", "clock", "cup", "fork", "keyboard", "knife", "laptop",
    "microwave", "mouse", "oven", "potted plant", "sink", "stop sign", "toilet", "tv",
)
"""The 18 target categories, sorted: a class's index is its position. Fixed by the dataset;
the same order the release's per-task folders sorted to, so indices are unchanged."""


def coco_search18_classes(root: Path) -> list[str]:
    """The 18 target categories (``COCO_SEARCH18_CLASSES``), after checking that ``root`` has
    the restructured layout -- the release's original layout is refused by name."""
    if not (root / "fixations").is_dir():
        hint = (" It looks like the ORIGINAL release layout (per-task image folders, JSONs in "
                "the root), which is no longer supported." if (root / "images" / "bottle").is_dir() else "")
        raise FileNotFoundError(f"{root}/fixations/ not found: expected the restructured COCO-Search18 "
                                f"layout (flat images/, fixations/; see the dataset's README.md).{hint}")
    return list(COCO_SEARCH18_CLASSES)


def _image_paths(root: Path, pairs, kind: str) -> list[Path]:
    """``images/<name>`` for each (name, task); refuses missing files up front."""
    paths = [root / "images" / name for name, _ in pairs]
    missing = [p for p in paths if not p.exists()]
    if missing:
        raise FileNotFoundError(f"{len(missing)} {kind} images missing from {root}/images/, e.g. {missing[0]}")
    return paths


def load_coco_search18(root: Path, *, split: int, part: str, classes: list[str]) -> list[SearchRecord]:
    """The unique (image, target) pairs of ``coco_search18_fixations_TP_{part}_split{split}.json``.
    Refuses a pair whose bbox differs between subjects (it would make the target ambiguous)."""
    path = root / "fixations" / f"coco_search18_fixations_TP_{part}_split{split}.json"
    trials = json.loads(path.read_text())
    cls_idx = {c: i for i, c in enumerate(classes)}
    by_pair: dict[tuple[str, str], tuple[float, ...]] = {}
    for tr in trials:
        key = (tr["name"], tr["task"])
        bbox = tuple(float(v) for v in tr["bbox"])
        prev = by_pair.setdefault(key, bbox)
        if prev != bbox:
            raise ValueError(f"{path.name}: {key} has two different target boxes {prev} vs {bbox}")
    pairs = sorted(by_pair.items())
    paths = _image_paths(root, [k for k, _ in pairs], "target-present")
    records = [SearchRecord(path=path, target=cls_idx[task], bbox_xywh=bbox)  # type: ignore[arg-type]
               for path, ((_, task), bbox) in zip(paths, pairs)]
    log.info("COCO-Search18 split%d %s: %d trials -> %d (image, target) pairs",
             split, part, len(trials), len(records))
    return records


def load_coco_search18_ta(root: Path, *, split: int, part: str, classes: list[str]) -> list[SearchRecord]:
    """The unique target-absent (image, cue) pairs of ``part`` ('train' / 'valid').

    The TA fixation file has no split1/split2 variants: its ``split`` field matches TP
    split1 (2163 / 326 pairs), so split2 is refused rather than silently mixed."""
    if split != 1:
        raise ValueError("the target-absent trials come with split1's train/valid assignment only; "
                         "use --cfg.split 1 or --cfg.condition tp")
    path = root / "fixations" / "coco_search18_fixations_TA_trainval.json"
    trials = [tr for tr in json.loads(path.read_text()) if tr["split"] == part]
    cls_idx = {c: i for i, c in enumerate(classes)}
    pairs = sorted({(tr["name"], tr["task"]) for tr in trials})
    paths = _image_paths(root, pairs, "target-absent")
    records = [SearchRecord(path=path, target=cls_idx[task], bbox_xywh=None)
               for path, (_, task) in zip(paths, pairs)]
    log.info("COCO-Search18 TA %s: %d trials -> %d (image, target) pairs", part, len(trials), len(records))
    return records


def load_records(cfg: SearchConfig, part: str, classes: list[str]) -> list[SearchRecord]:
    """TP pairs, plus the TA pairs when ``cfg.condition == 'both'``."""
    if cfg.condition == "both" and cfg.split != 1:  # refuse before reading anything
        raise ValueError("the target-absent trials come with split1's train/valid assignment only; "
                         "use --cfg.split 1 or --cfg.condition tp")
    recs = load_coco_search18(cfg.data_root, split=cfg.split, part=part, classes=classes)
    if cfg.condition == "both":
        ta = load_coco_search18_ta(cfg.data_root, split=cfg.split, part=part, classes=classes)
        # Interleave TP and TA, so the (unshuffled) first val batch -- the one the val figure
        # draws -- holds both kinds of trial. Metrics do not depend on the order.
        mixed = [r for pair in zip(recs, ta) for r in pair]
        n = min(len(recs), len(ta))
        recs = mixed + recs[n:] + ta[n:]
    return recs


def _photometric(aug: SearchAugConfig) -> list:
    ops: list = []
    if aug.color_jitter:
        ops += [v2.RandomApply([v2.ColorJitter(0.4, 0.4, 0.4, 0.1)], p=0.8), v2.RandomGrayscale(p=0.1)]
    if aug.autocontrast:
        ops += [v2.RandomAutocontrast(p=0.5)]
    return ops


def _geometric(aug: SearchAugConfig, content_hw: tuple[int, int]) -> list:
    """Everything that moves pixels, in Haoze's order (flip, rotate, affine), after the
    crop, which also resizes to the letterboxed content size."""
    h, w = content_hw
    ops: list = []
    if aug.crop:
        ops.append(v2.RandomResizedCrop(content_hw, scale=aug.crop_scale, ratio=(w / h, w / h), antialias=True))
    else:
        ops.append(v2.Resize(content_hw, antialias=True))
    if aug.hflip_p > 0:
        ops.append(v2.RandomHorizontalFlip(p=aug.hflip_p))
    if aug.rotate:
        ops.append(v2.RandomApply([v2.RandomRotation(60, fill=0)], p=0.5))
    if aug.affine:
        ops.append(v2.RandomApply([v2.RandomAffine(degrees=45, translate=(0.1, 0.1), scale=(0.8, 1.2),
                                                   shear=15, fill=0)], p=0.5))
    return ops


def box_inside_fraction(boxes_xyxy: Tensor, hw: tuple[int, int]) -> Tensor:
    """Per box, the fraction of its area that lies inside the ``hw`` frame."""
    h, w = hw
    x0, y0, x1, y1 = boxes_xyxy.unbind(-1)
    area = (x1 - x0).clamp_min(0) * (y1 - y0).clamp_min(0)
    ix = (x1.clamp(0, w) - x0.clamp(0, w)).clamp_min(0)
    iy = (y1.clamp(0, h) - y0.clamp(0, h)).clamp_min(0)
    return torch.where(area > 0, ix * iy / area.clamp_min(1e-6), torch.zeros_like(area))


def letterbox_hw(h: int, w: int, scene: int) -> tuple[int, int]:
    """The content size after fitting (h, w) into a scene x scene square, aspect kept."""
    s = scene / max(h, w)
    return max(1, round(h * s)), max(1, round(w * s))


class SearchDataset(Dataset):
    """Records -> the sample contract. ``aug=None`` = validation (deterministic)."""

    def __init__(self, records: list[SearchRecord], *, scene_size: int, aug: SearchAugConfig | None):
        self.records = records
        self.scene = scene_size
        self.aug = aug

    def __len__(self) -> int:
        return len(self.records)

    def _load(self, rec: SearchRecord) -> tuple[tv_tensors.Image, tv_tensors.BoundingBoxes]:
        img = tv_tensors.Image(decode_image(read_file(str(rec.path)), mode=ImageReadMode.RGB))
        h, w = img.shape[-2:]
        xywh = [rec.bbox_xywh] if rec.bbox_xywh is not None else []
        boxes = tv_tensors.BoundingBoxes(
            torch.tensor(xywh, dtype=torch.float32).reshape(-1, 4), format="XYWH",
            canvas_size=(h, w), clamping_mode=None)  # unclamped, so retention can be measured
        return img, TF.convert_bounding_box_format(boxes, new_format=tv_tensors.BoundingBoxFormat.XYXY)

    def _transform(self, img, boxes):
        h, w = img.shape[-2:]
        content = letterbox_hw(h, w, self.scene)
        if self.aug is None:
            return v2.Resize(content, antialias=True)(img, boxes)
        geo = v2.Compose(_geometric(self.aug, content))
        # No box (target absent) => the first draw is kept: any crop is valid.
        for _ in range(max(1, self.aug.max_tries)):
            gi, gb = geo(img, boxes)
            if bool((box_inside_fraction(gb.as_subclass(Tensor), content) >= self.aug.min_box_fraction).all()):
                break
        else:  # no draw kept the target: use the plain (un-augmented) geometry
            gi, gb = v2.Resize(content, antialias=True)(img, boxes)
        photo = _photometric(self.aug)
        if photo:
            gi = v2.Compose(photo)(gi)
        return gi, gb

    def __getitem__(self, i: int) -> dict:
        rec = self.records[i]
        img, boxes = self._load(rec)
        img, boxes = self._transform(img, boxes)
        ch, cw = img.shape[-2:]
        top, left = (self.scene - ch) // 2, (self.scene - cw) // 2
        # Letterbox: zero (black) padding, as the dataset's own 1680x1050 padding and Haoze's.
        img = TF.pad(img, [left, top, self.scene - cw - left, self.scene - ch - top], fill=0)
        b = boxes.as_subclass(Tensor).clone()
        b[:, [0, 2]] = (b[:, [0, 2]].clamp(0, cw) + left) / self.scene * 2 - 1
        b[:, [1, 3]] = (b[:, [1, 3]].clamp(0, ch) + top) / self.scene * 2 - 1
        image = TF.normalize(TF.to_dtype(img, torch.float32, scale=True),
                             mean=list(IMAGENET_DEFAULT_MEAN), std=list(IMAGENET_DEFAULT_STD))
        return {"image": image.as_subclass(Tensor), "target": torch.tensor(rec.target, dtype=torch.long),
                "boxes": b, "present": torch.tensor(rec.bbox_xywh is not None)}


def collate_search(batch: list[dict]) -> dict:
    """Stack images/targets/present; pad ``boxes`` to the batch max with NaN rows (at least
    one row, so an all-absent batch still has a ``[B, 1, 4]`` tensor)."""
    n = max(1, max(s["boxes"].shape[0] for s in batch))
    boxes = torch.full((len(batch), n, 4), math.nan)
    for i, s in enumerate(batch):
        boxes[i, : s["boxes"].shape[0]] = s["boxes"]
    return {"image": torch.stack([s["image"] for s in batch]),
            "target": torch.stack([s["target"] for s in batch]), "boxes": boxes,
            "present": torch.stack([s["present"] for s in batch])}


def make_search_loaders(cfg: SearchConfig) -> tuple[DataLoader, DataLoader, list[str]]:
    classes = coco_search18_classes(cfg.data_root)
    train = SearchDataset(load_records(cfg, "train", classes), scene_size=cfg.scene_size, aug=cfg.aug)
    val = SearchDataset(load_records(cfg, "valid", classes), scene_size=cfg.scene_size, aug=None)
    kw = dict(num_workers=cfg.num_workers, pin_memory=True, collate_fn=collate_search,
              persistent_workers=cfg.num_workers > 0)
    return (DataLoader(train, batch_size=cfg.batch_size, shuffle=True, drop_last=True, **kw),
            DataLoader(val, batch_size=cfg.eval_batch_size, shuffle=False, **kw),
            classes)


def make_search_val_loader(cfg: SearchConfig) -> tuple[DataLoader, list[str]]:
    classes = coco_search18_classes(cfg.data_root)
    val = SearchDataset(load_records(cfg, "valid", classes), scene_size=cfg.scene_size, aug=None)
    return (DataLoader(val, batch_size=cfg.eval_batch_size, shuffle=False, num_workers=cfg.num_workers,
                       pin_memory=True, collate_fn=collate_search), classes)
