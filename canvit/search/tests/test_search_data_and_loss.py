"""The COCO-Search18 adapter (dedup, letterbox, augmentation + target retention) and the
search targets / loss / hit metric."""

import math

import pytest
import torch

from canvit.search.config import SearchAugConfig
from canvit.search.data import (
    SearchDataset,
    box_inside_fraction,
    coco_search18_classes,
    collate_search,
    letterbox_hw,
    load_coco_search18,
)
from canvit.search.loss import argmax_centres, hit, per_image_loss, target_map

S = 512


def _records(root, part="train"):
    classes = coco_search18_classes(root)
    return load_coco_search18(root, split=1, part=part, classes=classes), classes


# --------------------------------------------------------------------------- #
# adapter
# --------------------------------------------------------------------------- #
def test_records_are_deduplicated_per_image_and_target(fake_cs18):
    root, pairs = fake_cs18
    recs, classes = _records(root)
    assert len(recs) == len(pairs["train"])  # 3 subjects per pair -> one record
    assert classes == sorted(classes) and len(classes) == 18


def test_conflicting_boxes_are_refused(fake_cs18, tmp_path):
    import json
    import shutil
    root, _ = fake_cs18
    bad = tmp_path / "cs18"
    shutil.copytree(root, bad)
    path = bad / "coco_search18_fixations_TP_train_split1.json"
    recs = json.loads(path.read_text())
    recs[1]["bbox"] = [1.0, 2.0, 3.0, 4.0]  # subject 2 of pair 0 disagrees
    path.write_text(json.dumps(recs))
    with pytest.raises(ValueError, match="two different target boxes"):
        _records(bad)


def test_letterbox_maps_the_box_exactly(fake_cs18):
    root, pairs = fake_cs18
    recs, _ = _records(root, "valid")
    ds = SearchDataset(recs, scene_size=S, aug=None)
    ch, cw = letterbox_hw(1050, 1680, S)
    assert (ch, cw) == (320, 512)
    for i, rec in enumerate(recs):
        s = ds[i]
        assert s["image"].shape == (3, S, S)
        x, y, w, h = rec.bbox_xywh
        top = (S - ch) / 2
        sc = S / 1680
        want = torch.tensor([x * sc, y * sc + top, (x + w) * sc, (y + h) * sc + top]) / S * 2 - 1
        assert torch.allclose(s["boxes"][0], want, atol=2e-3), (s["boxes"][0], want)
        # the letterbox bands are black (= -mean/std after normalisation)
        assert torch.allclose(s["image"][:, 0, :], s["image"][:, 0, :1].expand(-1, S))


@pytest.mark.parametrize("aug", [
    SearchAugConfig(),
    SearchAugConfig(rotate=True, affine=True, color_jitter=True, autocontrast=True),
])
def test_augmentation_always_keeps_the_target(fake_cs18, aug):
    root, _ = fake_cs18
    recs, _ = _records(root)
    ds = SearchDataset(recs, scene_size=S, aug=aug)
    torch.manual_seed(0)
    ch, cw = letterbox_hw(1050, 1680, S)
    top = (S - ch) / 2
    for _ in range(3):
        for i in range(len(ds)):
            b = ds[i]["boxes"][0]
            assert torch.all(b[2:] > b[:2]), "the target box must not collapse"
            # inside the letterboxed content band
            assert b[1] >= (top / S * 2 - 1) - 1e-4 and b[3] <= ((top + ch) / S * 2 - 1) + 1e-4
            assert b[0] >= -1 - 1e-4 and b[2] <= 1 + 1e-4


def test_box_inside_fraction():
    boxes = torch.tensor([[0., 0., 10., 10.], [-5., 0., 5., 10.], [20., 20., 30., 30.]])
    assert torch.allclose(box_inside_fraction(boxes, (15, 15)), torch.tensor([1.0, 0.5, 0.0]))


def test_collate_pads_boxes_with_nan():
    a = {"image": torch.zeros(3, 4, 4), "target": torch.tensor(1), "boxes": torch.zeros(1, 4)}
    b = {"image": torch.zeros(3, 4, 4), "target": torch.tensor(2), "boxes": torch.ones(2, 4)}
    out = collate_search([a, b])
    assert out["boxes"].shape == (2, 2, 4) and torch.isnan(out["boxes"][0, 1]).all()


# --------------------------------------------------------------------------- #
# targets / loss / hit
# --------------------------------------------------------------------------- #
G = 32


def _box(cx, cy, half=0.05):
    return torch.tensor([[[cx - half, cy - half, cx + half, cy + half]]])


def test_target_maps_are_distributions():
    boxes = torch.cat([_box(0.1, -0.3), _box(-0.6, 0.7)])
    for kind in ("kl_gaussian", "nll_point"):
        t = target_map(boxes, grid=G, kind=kind, sigma=2.5 / 64)
        assert t.shape == (2, G * G)
        assert torch.allclose(t.sum(-1), torch.ones(2))


def test_gaussian_peaks_at_the_box_centre_cell():
    t = target_map(_box(0.1, -0.3), grid=G, kind="kl_gaussian", sigma=2.5 / 64)
    idx = t.argmax(-1).item()
    iy, ix = divmod(idx, G)
    assert ix == math.floor((0.1 + 1) / 2 * G) and iy == math.floor((-0.3 + 1) / 2 * G)


def test_nll_point_is_the_small_sigma_limit_of_kl_gaussian():
    boxes = _box(0.13, -0.27)
    logits = torch.randn(1, G, G)
    nll = per_image_loss(logits, target_map(boxes, grid=G, kind="nll_point", sigma=0.0))
    kl = per_image_loss(logits, target_map(boxes, grid=G, kind="kl_gaussian", sigma=1e-4))
    assert torch.allclose(nll, kl, atol=1e-4)
    # and NLL is exactly -log p(cell)
    ix, iy = math.floor((0.13 + 1) / 2 * G), math.floor((-0.27 + 1) / 2 * G)
    assert torch.allclose(nll, -torch.log_softmax(logits.flatten(1), -1)[:, iy * G + ix])


def test_multiple_boxes_share_the_mass():
    boxes = torch.cat([_box(0.5, 0.5), _box(-0.5, -0.5)], dim=1)  # one sample, two boxes
    t = target_map(boxes, grid=G, kind="nll_point", sigma=0.0)
    assert (t > 0).sum() == 2 and torch.allclose(t[t > 0], torch.tensor([0.5, 0.5]))


def test_hit_criterion():
    boxes = _box(0.5, 0.5, half=0.02)
    logits = torch.full((1, G, G), -10.0)
    iy, ix = math.floor((0.5 + 1) / 2 * G), math.floor((0.5 + 1) / 2 * G)
    logits[0, iy, ix] = 10.0
    assert hit(logits, boxes, box_frac=0.1).item()
    far = torch.full((1, G, G), -10.0)
    far[0, 0, 0] = 10.0
    assert not hit(far, boxes, box_frac=0.1).item()
    c = argmax_centres(logits)[0]
    assert abs(c[0] - 0.5) < 2 / G and abs(c[1] - 0.5) < 2 / G
