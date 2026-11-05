"""A synthetic COCO-Search18 in the dataset's real on-disk format (``images/<category>/``
JPEGs at 1680x1050, ``coco_search18_fixations_TP_{train,valid}_split1.json`` with one record
per subject), so the tests run the real adapter rather than hand-built tensors."""

import json

import pytest
import torch
from torchvision.io import write_jpeg

CATEGORIES = ["bottle", "bowl", "car", "chair", "clock", "cup", "fork", "keyboard", "knife",
              "laptop", "microwave", "mouse", "oven", "potted plant", "sink", "stop sign", "toilet", "tv"]
H, W = 1050, 1680


def _record(name, task, bbox, subject, split):
    # Field set of the real JSON (see the dataset readme): only name/task/bbox are read.
    return {"name": name, "subject": subject, "task": task, "condition": "present", "bbox": bbox,
            "X": [840.0], "Y": [525.0], "T": [200], "length": 1, "correct": 1, "RT": 900, "split": split}


@pytest.fixture(scope="session")
def fake_cs18(tmp_path_factory):
    root = tmp_path_factory.mktemp("coco_search18")
    g = torch.Generator().manual_seed(0)
    for c in CATEGORIES:
        (root / "images" / c).mkdir(parents=True)
    pairs = {"train": [], "valid": []}
    for split, n in (("train", 6), ("valid", 4)):
        for i in range(n):
            task = CATEGORIES[i % len(CATEGORIES)]
            name = f"{split}_{i:06d}.jpg"
            img = torch.randint(0, 255, (3, H, W), generator=g, dtype=torch.uint8)
            write_jpeg(img, str(root / "images" / task / name), quality=50)
            bbox = [200.0 + 100 * i, 150.0 + 50 * i, 160.0, 120.0]
            pairs[split].append((name, task, bbox))
    for split, part in (("train", "train"), ("valid", "valid")):
        recs = [_record(n, t, b, s, split) for (n, t, b) in pairs[split] for s in range(1, 4)]  # 3 subjects
        (root / f"coco_search18_fixations_TP_{part}_split1.json").write_text(json.dumps(recs))
    return root, pairs
