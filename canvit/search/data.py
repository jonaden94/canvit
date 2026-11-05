"""COCO-Search18 adapter -> the dataset-independent search sample contract.

Every sample is a dict:

    image:   float [3, S, S]  ImageNet-normalised, letterboxed into the square scene
    target:  long  []         class index into ``classes``
    boxes:   float [N, 4]     target boxes, scene-normalised [-1, 1] xyxy (N >= 1)

A new search dataset needs only an adapter that produces this; nothing downstream (head,
loss, policy, eval) reads anything dataset-specific.

COCO-Search18 facts this relies on (checked 2026-11-05, see claude_dev/search/00-design.md):
the TP fixation JSONs hold one record per (subject, trial), 10 subjects per
(image, target) pair, with the same target bbox ``[x, y, w, h]`` (1680x1050 px) for every
subject -- so the records are deduplicated on ``(name, task)``. TA images are not on disk.
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
    bbox_xywh: tuple[float, float, float, float]  # in the image's own pixels


def coco_search18_classes(root: Path) -> list[str]:
    """The 18 target categories, sorted (= the ``images/`` subfolders)."""
    classes = sorted(p.name for p in (root / "images").iterdir() if p.is_dir())
    assert len(classes) == 18, f"expected 18 COCO-Search18 categories under {root}/images, got {classes}"
    return classes


def load_coco_search18(root: Path, *, split: int, part: str, classes: list[str]) -> list[SearchRecord]:
    """The unique (image, target) pairs of ``coco_search18_fixations_TP_{part}_split{split}.json``.
    Refuses a pair whose bbox differs between subjects (it would make the target ambiguous)."""
    path = root / f"coco_search18_fixations_TP_{part}_split{split}.json"
    trials = json.loads(path.read_text())
    cls_idx = {c: i for i, c in enumerate(classes)}
    by_pair: dict[tuple[str, str], tuple[float, ...]] = {}
    for tr in trials:
        key = (tr["name"], tr["task"])
        bbox = tuple(float(v) for v in tr["bbox"])
        prev = by_pair.setdefault(key, bbox)
        if prev != bbox:
            raise ValueError(f"{path.name}: {key} has two different target boxes {prev} vs {bbox}")
    records = [SearchRecord(path=root / "images" / task / name, target=cls_idx[task],
                            bbox_xywh=bbox)  # type: ignore[arg-type]
               for (name, task), bbox in sorted(by_pair.items())]
    log.info("COCO-Search18 split%d %s: %d trials -> %d (image, target) pairs",
             split, part, len(trials), len(records))
    return records


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
        boxes = tv_tensors.BoundingBoxes(
            torch.tensor([rec.bbox_xywh], dtype=torch.float32), format="XYWH",
            canvas_size=(h, w), clamping_mode=None)  # unclamped, so retention can be measured
        return img, TF.convert_bounding_box_format(boxes, new_format=tv_tensors.BoundingBoxFormat.XYXY)

    def _transform(self, img, boxes):
        h, w = img.shape[-2:]
        content = letterbox_hw(h, w, self.scene)
        if self.aug is None:
            return v2.Resize(content, antialias=True)(img, boxes)
        geo = v2.Compose(_geometric(self.aug, content))
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
                "boxes": b}


def collate_search(batch: list[dict]) -> dict:
    """Stack images/targets; pad ``boxes`` to the batch max with NaN rows."""
    n = max(s["boxes"].shape[0] for s in batch)
    boxes = torch.full((len(batch), n, 4), math.nan)
    for i, s in enumerate(batch):
        boxes[i, : s["boxes"].shape[0]] = s["boxes"]
    return {"image": torch.stack([s["image"] for s in batch]),
            "target": torch.stack([s["target"] for s in batch]), "boxes": boxes}


def make_search_loaders(cfg: SearchConfig) -> tuple[DataLoader, DataLoader, list[str]]:
    classes = coco_search18_classes(cfg.data_root)
    train = SearchDataset(load_coco_search18(cfg.data_root, split=cfg.split, part="train", classes=classes),
                          scene_size=cfg.scene_size, aug=cfg.aug)
    val = SearchDataset(load_coco_search18(cfg.data_root, split=cfg.split, part="valid", classes=classes),
                        scene_size=cfg.scene_size, aug=None)
    kw = dict(num_workers=cfg.num_workers, pin_memory=True, collate_fn=collate_search,
              persistent_workers=cfg.num_workers > 0)
    return (DataLoader(train, batch_size=cfg.batch_size, shuffle=True, drop_last=True, **kw),
            DataLoader(val, batch_size=cfg.eval_batch_size, shuffle=False, **kw),
            classes)


def make_search_val_loader(cfg: SearchConfig) -> tuple[DataLoader, list[str]]:
    classes = coco_search18_classes(cfg.data_root)
    val = SearchDataset(load_coco_search18(cfg.data_root, split=cfg.split, part="valid", classes=classes),
                        scene_size=cfg.scene_size, aug=None)
    return (DataLoader(val, batch_size=cfg.eval_batch_size, shuffle=False, num_workers=cfg.num_workers,
                       pin_memory=True, collate_fn=collate_search), classes)
