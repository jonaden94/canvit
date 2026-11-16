"""The train split's masks have the val split's dtype with and without augmentation.

dinov3's augmentation yields uint8 masks; the policy reward (`per_image_ce`) passes the mask
straight to `F.cross_entropy`, which rejects uint8, so every augmented policy run died at
step 0 (2026-11-16) until `make_ade20k_loaders` cast it."""

import numpy as np
import torch
from PIL import Image

from canvit.ade20k.config import Ade20kConfig
from canvit.ade20k.data import make_ade20k_loaders
from canvit.ade20k.metrics import reward_ce


def _fake_root(tmp_path):
    rng = np.random.default_rng(0)
    for split in ("training", "validation"):
        (tmp_path / "images" / split).mkdir(parents=True)
        (tmp_path / "annotations" / split).mkdir(parents=True)
        for i in range(2):
            Image.fromarray(rng.integers(0, 255, (300, 400, 3), dtype=np.uint8)).save(
                tmp_path / "images" / split / f"x_{i}.jpg")
            Image.fromarray(rng.integers(0, 151, (300, 400), dtype=np.uint8)).save(
                tmp_path / "annotations" / split / f"x_{i}.png")
    return tmp_path


def test_train_masks_are_int64_with_and_without_augmentation(tmp_path) -> None:
    root = _fake_root(tmp_path)
    for augment in (True, False):
        cfg = Ade20kConfig(ade20k_root=root, augment=augment, resize_mode="squish",
                           num_workers=0, batch_size=2)
        train, val = make_ade20k_loaders(cfg)
        _, m_train = train.dataset[0]
        _, m_val = val.dataset[0]
        assert m_train.dtype == m_val.dtype == torch.int64, (augment, m_train.dtype, m_val.dtype)
        # and the policy reward accepts it (the call that crashed)
        logits = torch.randn(1, 150, 32, 32)
        assert torch.isfinite(reward_ce(logits, m_train[None], score_res=128)).all()
