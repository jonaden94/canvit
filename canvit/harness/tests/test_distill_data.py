"""CPU tests for distill's two shard flavours and the normalizer-init rules.

WebDataset shards come in two kinds and the task must handle both (train/loop.py
546-575, 639-643):
  - **with features** (`cls.npy`/`ptch.npy`): teacher targets are precomputed.
  - **raw** (jpg+json only): the frozen teacher produces targets ON THE FLY, both to
    seed the standardizers and for every training batch. This is the exp21 path.
Plus `cfg.reset_normalizer`, which must re-init even when the checkpoint carried stats.

The loader and the teacher are stubbed (a real one needs tar shards + a GPU); the
dispatch, the reset rule and the on-the-fly bind are the real code. The raw path is
exercised against real shards by `canvit_claude/canvit/unification/harness_run_raw_shards.py`.
"""

from types import SimpleNamespace

import torch

from canvit.distill.config import Config
from canvit.distill.data.webdataset import WebDatasetTrainLoader
from canvit.distill.task import DistillRunTask

_G, _D, _BS, _PATCH = 8, 16, 2, 16


class _IdentityNorm:
    """Standardizer stub: bind() both standardizes the targets and hands the engine the
    destandardizer for the raw-space cosine metrics."""

    def __call__(self, x):
        return x

    def destandardize(self, x):
        return x


def _task(tmp_path, *, initialized, reset=False, has_features=True, normalizer_shards=4):
    cfg = Config(webdataset_dir="/nonexistent", batch_size_per_gpu=_BS, steps_per_job=64,
                 canvas_patch_grid_size=_G, reset_normalizer=reset,
                 normalizer_shards=normalizer_shards)
    t = DistillRunTask(cfg)
    t.scene_norm = SimpleNamespace(initialized=initialized)
    t.cls_norm = SimpleNamespace()
    t._device = torch.device("cpu")
    t._model = SimpleNamespace(backbone=SimpleNamespace(patch_size_px=_PATCH))
    loader = object.__new__(WebDatasetTrainLoader)
    loader.samples_per_shard, loader.has_features = 512, has_features
    # `normalizer_shard_paths` globs train_dir rather than reading shard_files (the
    # schedule slice), so the stub needs a real directory of shard-shaped names. 8 of
    # them: the default is 4 and the last is excluded as partial, so this leaves room
    # to raise the default without the fixture silently becoming the binding constraint.
    for i in range(8):
        (tmp_path / f"shard-{i:06d}.tar").touch()
    loader.train_dir = tmp_path
    loader.shard_files = [tmp_path / "shard-000000.tar"]
    return t, loader


def _stub_init(monkeypatch, task, loader):
    """Patch create_loaders + both normalizer initialisers; return the call records."""
    seen: dict = {"tar": [], "raw": []}
    monkeypatch.setattr("canvit.distill.data.create_loaders",
                        lambda cfg, **kw: SimpleNamespace(train=loader, val=None))
    monkeypatch.setattr("canvit.distill.data.webdataset.init_normalizer_stats_from_tar",
                        lambda *a, **k: seen["tar"].append((a, k)))
    monkeypatch.setattr("canvit.distill.data.webdataset.init_normalizer_stats_from_tar_raw",
                        lambda *a, **k: seen["raw"].append((a, k)))
    monkeypatch.setattr(task, "_teacher_targets", lambda imgs, sz: SimpleNamespace(
        patches=torch.zeros(imgs.shape[0], _G * _G, _D), cls=torch.zeros(imgs.shape[0], _D)))
    return seen


# --- which initialiser, and when -------------------------------------------
def test_feature_shards_use_the_precomputed_initializer(monkeypatch, tmp_path):
    t, loader = _task(tmp_path, initialized=False, has_features=True)
    seen = _stub_init(monkeypatch, t, loader)
    t.build_loaders(world_size=1, rank=0)
    assert len(seen["tar"]) == 1 and not seen["raw"]


def test_raw_shards_use_the_on_the_fly_initializer(monkeypatch, tmp_path):
    """Raw shards have no cls.npy/ptch.npy — calling the precomputed initialiser on them
    would read keys that aren't there."""
    t, loader = _task(tmp_path, initialized=False, has_features=False)
    seen = _stub_init(monkeypatch, t, loader)
    t.build_loaders(world_size=1, rank=0)
    assert len(seen["raw"]) == 1 and not seen["tar"]
    assert seen["raw"][0][1]["image_size"] == _G * _PATCH  # decoded at the scene resolution


def test_normalizer_shards_reaches_the_initializer(monkeypatch, tmp_path):
    """cfg.normalizer_shards must be threaded through, and the selection must be the
    sorted head of train_dir — NOT loader.shard_files, which is the seed-dependent
    schedule slice and would make the stats a function of the seed."""
    t, loader = _task(tmp_path, initialized=False, normalizer_shards=3)
    seen = _stub_init(monkeypatch, t, loader)
    t.build_loaders(world_size=1, rank=0)
    paths = seen["tar"][0][0][0]
    assert [p.name for p in paths] == [
        "shard-000000.tar", "shard-000001.tar", "shard-000002.tar"
    ]


def test_initialized_normalizer_is_not_reinitialized(monkeypatch, tmp_path):
    t, loader = _task(tmp_path, initialized=True)
    seen = _stub_init(monkeypatch, t, loader)
    t.build_loaders(world_size=1, rank=0)
    assert not seen["tar"] and not seen["raw"]


def test_reset_normalizer_forces_reinit(monkeypatch, tmp_path):
    """cfg.reset_normalizer must re-init even though the checkpoint carried stats —
    otherwise the flag silently does nothing on resume."""
    t, loader = _task(tmp_path, initialized=True, reset=True)
    seen = _stub_init(monkeypatch, t, loader)
    t.build_loaders(world_size=1, rank=0)
    assert len(seen["tar"]) == 1


# --- per-batch targets ------------------------------------------------------
def test_bind_computes_teacher_targets_when_the_batch_has_none(monkeypatch, tmp_path):
    """Raw shards yield (images, None, None, labels); bind must produce the targets
    rather than call .to() on a None."""
    t, loader = _task(tmp_path, initialized=True, has_features=False)
    _stub_init(monkeypatch, t, loader)
    t.scene_norm = t.cls_norm = _IdentityNorm()
    images = torch.randn(_BS, 3, _G * _PATCH, _G * _PATCH)
    bound = t.bind((images, None, None, None), torch.device("cpu"), model=None, head=None)
    assert bound.distill.scene_target.shape == (_BS, _G * _G, _D)
    assert bound.distill.cls_target.shape == (_BS, _D)


def test_checkpoint_model_config_wins_over_cli_defaults():
    """On RESUME the checkpoint's arch must override this run's config, or the strict
    weight load fails on missing/unexpected keys (train/loop.py 254-261). The full arch
    rides in model_config["canvit"], so the round-trip has to be lossless."""
    from unittest.mock import patch

    cfg = Config(webdataset_dir="/nonexistent", canvas_patch_grid_size=_G)
    cfg.model.canvas_update_mode = "additive"
    saved = DistillRunTask(cfg).model_config(None)          # what run() writes
    assert saved["canvit"]["canvas_update_mode"] == "additive"

    # a NEW run whose CLI default disagrees with the checkpoint
    cfg2 = Config(webdataset_dir="/nonexistent", canvas_patch_grid_size=_G)
    cfg2.model.canvas_update_mode = "convex"
    t2 = DistillRunTask(cfg2)
    # stop before any real model construction — we only pin the config resolution
    with patch("canvit.distill.model.load_student_backbone", side_effect=RuntimeError("stop")):
        try:
            t2.build_model(torch.device("cpu"), prior_model_config=saved)
        except RuntimeError as e:
            assert "stop" in str(e)
    assert t2.cfg.model.canvas_update_mode == "additive", "checkpoint arch must win"

    # no prior config (FRESH/SEED) => the run's own config stands
    cfg3 = Config(webdataset_dir="/nonexistent", canvas_patch_grid_size=_G)
    cfg3.model.canvas_update_mode = "convex"
    t3 = DistillRunTask(cfg3)
    with patch("canvit.distill.model.load_student_backbone", side_effect=RuntimeError("stop")):
        try:
            t3.build_model(torch.device("cpu"), prior_model_config=None)
        except RuntimeError:
            pass
    assert t3.cfg.model.canvas_update_mode == "convex"


def test_bind_uses_precomputed_targets_when_present(monkeypatch, tmp_path):
    t, loader = _task(tmp_path, initialized=True, has_features=True)
    _stub_init(monkeypatch, t, loader)
    t.scene_norm = t.cls_norm = _IdentityNorm()
    patches, cls = torch.randn(_BS, _G * _G, _D), torch.randn(_BS, _D)
    bound = t.bind((torch.randn(_BS, 3, 8, 8), patches, cls, None), torch.device("cpu"),
                   model=None, head=None)
    assert torch.allclose(bound.distill.scene_target, patches)  # NOT recomputed


# --- reconstruction on a label-free image directory --------------------------
# The one capability canvit_eval's `reconstruction` task had that distill validation did
# not: an arbitrary image folder as the val source. The cosine series it computed is what
# `validate` already returns -- and canvit_eval's `scene_cos_raw` was WRONG, comparing a
# normalized-space prediction against raw teacher features (measured 0.648 vs 0.927 at t9
# on exp32-fovi). So this is the capability ported, not the task (eval-merge doc §5, Stage 4).

def _png(path, colour=(10, 20, 30)):
    from PIL import Image
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (8, 8), colour).save(path)


def test_flat_image_dir_recurses_and_sorts_by_filename(tmp_path):
    """rglob so a flat folder (ADE20K val) and a nested tree both work; sorted by FILENAME
    so the order does not depend on the directory layout."""
    from canvit.distill.data import FlatImageDir

    _png(tmp_path / "b.png")
    _png(tmp_path / "nested" / "a.png")
    _png(tmp_path / "c.jpg")
    (tmp_path / "notes.bak").write_text("not an image")
    ds = FlatImageDir(tmp_path, transform=lambda im: torch.zeros(3, 8, 8))
    assert [p.name for p in ds.paths] == ["a.png", "b.png", "c.jpg"]


def test_flat_image_dir_labels_disable_the_in1k_readout(tmp_path):
    """-1 is the signal. It used to pass `labels_are_in1k`, which only checked the UPPER
    bound, so the probe readout would have reported an accuracy against garbage labels."""
    from canvit.distill.data import FlatImageDir
    from canvit.distill.probe import labels_are_in1k

    _png(tmp_path / "a.png")
    ds = FlatImageDir(tmp_path, transform=lambda im: torch.zeros(3, 8, 8))
    _, label = ds[0]
    assert label == -1
    assert not labels_are_in1k(torch.tensor([label]))


def test_labels_are_in1k_bounds():
    from canvit.distill.probe import labels_are_in1k

    assert labels_are_in1k(torch.tensor([0, 999]))
    assert not labels_are_in1k(torch.tensor([1000]))   # IN21k, the original purpose
    assert not labels_are_in1k(torch.tensor([-1]))     # no labels at all


def test_val_image_dir_selects_the_flat_loader(tmp_path):
    from canvit.distill.data import create_imagefolder_val_loader

    for i in range(3):
        _png(tmp_path / f"img{i}.png")
    cfg = Config(val_image_dir=tmp_path, webdataset_dir=tmp_path, tracker="none",
                 num_workers=0, batch_size_per_gpu=2)
    loader = create_imagefolder_val_loader(cfg)
    assert loader.n_samples == 3, "no class subdirectories needed, unlike val_dir"
    images, labels = next(iter(loader.batches()))
    assert images.shape[0] == 2 and labels.tolist() == [-1, -1]
