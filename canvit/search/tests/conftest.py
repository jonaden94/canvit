"""A synthetic COCO-Search18 in the dataset's real on-disk format -- the restructured layout
of 2026-11-06: all JPEGs (1680x1050) in one flat ``images/``, and in ``fixations/`` the
``coco_search18_fixations_TP_{train,valid}_split1.json`` with one record per subject plus the
target-absent trials as ``coco_search18_fixations_TA_trainval.json`` (no bbox, a ``split``
field) -- so the tests run the real adapter rather than hand-built tensors."""

import json

import pytest
import torch
from torchvision.io import write_jpeg

CATEGORIES = ["bottle", "bowl", "car", "chair", "clock", "cup", "fork", "keyboard", "knife",
              "laptop", "microwave", "mouse", "oven", "potted plant", "sink", "stop sign", "toilet", "tv"]
H, W = 1050, 1680


def _ta_record(name, task, subject, split):
    # The real TA JSON's field set (checked 2026-11-06): no bbox, condition "absent".
    return {"name": name, "subject": subject, "task": task, "condition": "absent", "X": [840.0],
            "Y": [525.0], "T": [200], "length": 1, "fixOnTarget": False, "correct": 1, "split": split}


def _record(name, task, bbox, subject, split):
    # Field set of the real JSON (see the dataset readme): only name/task/bbox are read.
    return {"name": name, "subject": subject, "task": task, "condition": "present", "bbox": bbox,
            "X": [840.0], "Y": [525.0], "T": [200], "length": 1, "correct": 1, "RT": 900, "split": split}


@pytest.fixture(scope="session")
def fake_cs18(tmp_path_factory):
    root = tmp_path_factory.mktemp("coco_search18")
    g = torch.Generator().manual_seed(0)
    (root / "images").mkdir()
    (root / "fixations").mkdir()
    pairs = {"train": [], "valid": []}
    for split, n in (("train", 6), ("valid", 4)):
        for i in range(n):
            task = CATEGORIES[i % len(CATEGORIES)]
            name = f"{split}_{i:06d}.jpg"
            img = torch.randint(0, 255, (3, H, W), generator=g, dtype=torch.uint8)
            write_jpeg(img, str(root / "images" / name), quality=50)
            bbox = [200.0 + 100 * i, 150.0 + 50 * i, 160.0, 120.0]
            pairs[split].append((name, task, bbox))
    for split, part in (("train", "train"), ("valid", "valid")):
        recs = [_record(n, t, b, s, split) for (n, t, b) in pairs[split] for s in range(1, 4)]  # 3 subjects
        (root / "fixations" / f"coco_search18_fixations_TP_{part}_split1.json").write_text(json.dumps(recs))
    ta = []
    for split, n in (("train", 6), ("valid", 4)):
        for i in range(n):
            task = CATEGORIES[(i + 3) % len(CATEGORIES)]
            name = f"ta_{split}_{i:06d}.jpg"
            img = torch.randint(0, 255, (3, H, W), generator=g, dtype=torch.uint8)
            write_jpeg(img, str(root / "images" / name), quality=50)
            ta += [_ta_record(name, task, s, split) for s in range(1, 4)]
    (root / "fixations" / "coco_search18_fixations_TA_trainval.json").write_text(json.dumps(ta))
    return root, pairs
