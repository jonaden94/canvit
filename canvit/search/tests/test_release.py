"""The curated search release loader (``canvit/search/release.py``).

Synthetic tests (always run): the picture reader on a fake pack shard (pread offsets, the orientation fixes,
draft), the box -> scene mapping, the rank split, and the trial stream's resume through a real (spawned)
DataLoader. Release tests (``slow``, skipped where the release is not mounted): spec section 6 checks 1 and 2,
i.e. both reference epochs row for row and every reference picture's size and 16x16 thumbnail.
"""

import io
import json
import math
import os
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch
from PIL import Image
from torch.utils.data import DataLoader

from canvit.search.config import SearchAugConfig
from canvit.search.data import collate_search
from canvit.search.release import (
    EpochStream,
    ReleaseDataset,
    ReleasePictures,
    Trial,
    class_band,
    rank_share,
)

RELEASE = Path("/user/henrich1/u25995/jonathan/datasets/visual_search/search_release/basic-v1")
# Releases whose reference epochs the slow parity test checks: basic-v1, basic-v2 (visual prompts), basic-v3-text and
# basic-v3 (+ the V3Det negatives, source_match on) once they exist, and any extra release dirs in
# $CANVIT_EXTRA_RELEASES (os.pathsep-separated; e.g. a partial release in progress).
# Nothing is ever imported from a release folder (it must never be written to, not even a __pycache__).
PARITY_RELEASES = [RELEASE, *(RELEASE.parent / n for n in ("basic-v2", "basic-v3-text", "basic-v3")),
                   *(Path(p) for p in os.environ.get("CANVIT_EXTRA_RELEASES", "").split(os.pathsep) if p)]
W, H = 320, 200  # stored picture size (before the fix)
BOX = (0.25, 0.5, 0.125, 0.25)  # x, y, w, h as fractions of the FIXED picture: a white rectangle


def _jpeg(img: Image.Image, exif_orientation: int | None = None) -> bytes:
    buf = io.BytesIO()
    kw = {}
    if exif_orientation is not None:
        ex = Image.Exif()
        ex[0x0112] = exif_orientation
        kw["exif"] = ex.tobytes()
    img.save(buf, format="JPEG", quality=95, **kw)
    return buf.getvalue()


def _white_box(w: int, h: int) -> Image.Image:
    a = np.zeros((h, w, 3), np.uint8)
    x, y, bw, bh = BOX
    a[round(y * h):round((y + bh) * h), round(x * w):round((x + bw) * w)] = 255
    return Image.fromarray(a)


@pytest.fixture(scope="module")
def fake_release(tmp_path_factory):
    """A release folder with ``meta.json`` + ``images.parquet`` and one pack shard holding three pictures whose
    FIXED form has the white box at BOX: stored upright (fix none), stored rotated with an EXIF orientation tag
    (fix exif), and stored rotated clockwise without a tag (fix rotate_90 turns it back)."""
    root = tmp_path_factory.mktemp("release")
    (root / "pack" / "shards").mkdir(parents=True)
    upright = _white_box(W, H)
    pics = [
        ("none", _jpeg(upright), W, H),
        # EXIF 6 = "rotate 90 CW to display": store the upright picture rotated 90 CCW (PIL ROTATE_90)
        ("exif", _jpeg(upright.transpose(Image.Transpose.ROTATE_90), exif_orientation=6), W, H),
        # rotate_90 = turn 90 CCW to fix: store it rotated 90 CW
        ("rotate_90", _jpeg(upright.transpose(Image.Transpose.ROTATE_270)), W, H),
    ]
    blob, rows = b"\0" * 512, []  # a leading header, so offsets are not 0
    for i, (fix, data, w, h) in enumerate(pics):
        rows.append(dict(image_idx=i, shard="pool-00-00000", offset=len(blob), size=len(data), fix=fix,
                         width=w, height=h))
        blob += data + b"\0" * 100
    (root / "pack" / "shards" / "pool-00-00000.tar").write_bytes(blob)
    df = pd.DataFrame(rows).astype({"image_idx": "int32", "width": "int32", "height": "int32"})
    df.to_parquet(root / "images.parquet")
    (root / "meta.json").write_text(json.dumps({"name": "fake-v1", "pack_dir": str(root / "pack"),
                                                "shard_path": "{pack_dir}/shards/{shard}.tar"}))
    return root


def test_reader_applies_every_fix(fake_release):
    pics = ReleasePictures(fake_release)
    ref = np.asarray(pics.read(0), dtype=np.float32)
    assert ref.shape == (H, W, 3)
    for i in (1, 2):
        im = np.asarray(pics.read(i), dtype=np.float32)
        assert im.shape == (H, W, 3), f"fix of picture {i}"
        assert np.abs(im - ref).mean() < 3, f"picture {i} is not oriented like the upright one"


def test_reader_draft_keeps_the_long_side(fake_release):
    pics = ReleasePictures(fake_release)
    im = pics.read(0, long_side=150)  # 320 -> 1/2 scale = 160 >= 150, 1/4 = 80 would be too small
    assert im.size == (160, 100)


def test_pictures_pickle_without_fds(fake_release):
    import pickle
    pics = ReleasePictures(fake_release)
    pics.read(0)
    assert pics._fds
    clone = pickle.loads(pickle.dumps(pics))
    assert clone._fds == {} and clone.read(0).size == (W, H)


def test_scene_box_covers_the_white_rectangle(fake_release):
    """No augmentation: the scene-normalised box lands exactly on the white pixels after the letterbox."""
    S = 64
    ds = ReleaseDataset(ReleasePictures(fake_release), scene_size=S, aug=None)
    for i in range(3):
        s = ds[Trial(i, BOX, "thing")]
        assert s["present"] and s["text"] == "thing" and s["boxes"].shape == (1, 4) and "target" not in s
        x0, y0, x1, y1 = ((s["boxes"][0] + 1) / 2 * S).tolist()
        bright = s["image"].mean(0) > 1.0  # white in ImageNet normalisation (~2.2), black ~ -1.9
        ys, xs = bright.nonzero(as_tuple=True)
        assert abs(xs.min().item() - x0) <= 1 and abs(xs.max().item() + 1 - x1) <= 1, i
        assert abs(ys.min().item() - y0) <= 1 and abs(ys.max().item() + 1 - y1) <= 1, i
    a = ds[Trial(0, None, "thing")]
    assert not a["present"] and a["boxes"].shape == (0, 4)


def test_augmented_scene_keeps_the_target(fake_release):
    ds = ReleaseDataset(ReleasePictures(fake_release), scene_size=64, aug=SearchAugConfig())
    assert ds.long_side == math.ceil(64 / math.sqrt(0.9))  # the release default crop 0.9-1.0
    torch.manual_seed(0)
    for _ in range(10):
        b = ds[Trial(0, BOX, "thing")]["boxes"]
        assert b.shape == (1, 4) and bool((b >= -1).all() and (b <= 1).all())


def test_collate_passes_text_and_band(fake_release):
    ds = ReleaseDataset(ReleasePictures(fake_release), scene_size=32, aug=None)
    batch = collate_search([ds[Trial(0, BOX, "a", 2)], ds[Trial(1, None, "b", 5)]])
    assert batch["text"] == ["a", "b"] and batch["band"].tolist() == [2, 5] and "target" not in batch
    assert batch["present"].tolist() == [True, False] and batch["boxes"].shape == (2, 1, 4)


def test_class_bands():
    assert class_band(np.array([1, 9, 10, 99, 100, 299, 300, 999, 1000, 9999, 10000])).tolist() == \
        [0, 0, 1, 1, 2, 2, 3, 3, 4, 4, 5]


def test_rank_shares_are_disjoint_and_equal():
    df = pd.DataFrame({"k": np.arange(103)})
    for world in (1, 2, 3, 4):
        parts = [rank_share(df, rank=r, world_size=world)["k"].tolist() for r in range(world)]
        assert all(len(p) == 103 // world for p in parts)
        flat = sum(parts, [])
        assert len(set(flat)) == len(flat) == world * (103 // world)


def _fake_epochs(world: int, rank: int):
    """Epoch e has 50 + 7e trials, labelled (e, i); this rank's share as Trials."""
    def full(e):
        return pd.DataFrame({"k": [f"{e}:{i}" for i in range(50 + 7 * e)]})

    def trials(e):
        return [Trial(0, None, k) for k in rank_share(full(e), rank=rank, world_size=world)["k"]]
    return (lambda e: len(full(e)) // world), trials


def test_stream_resume_continues_exactly():
    """A stream started at position p equals the uninterrupted stream from p, across epoch boundaries."""
    import itertools
    length, trials = _fake_epochs(world=1, rank=0)
    whole = [t.text for t in itertools.islice(EpochStream(length=length, trials=trials), 400)]
    assert whole[49:51] == ["0:49", "1:0"]  # epoch 0 has 50 trials
    for p in (0, 1, 49, 50, 51, 107, 300):
        tail = [t.text for t in itertools.islice(EpochStream(length=length, trials=trials, start=p), 400 - p)]
        assert tail == whole[p:], p


def test_stream_ranks_are_disjoint_per_epoch():
    import itertools
    world = 3
    seen = []
    for r in range(world):
        length, trials = _fake_epochs(world=world, rank=r)
        seen.append([t.text for t in itertools.islice(EpochStream(length=length, trials=trials), 200)])
    assert all(len(s) == 200 for s in seen)
    flat = sum(seen, [])
    assert len(set(flat)) == len(flat), "two ranks drew the same trial"


@pytest.mark.parametrize("workers", [0, 2])
def test_loader_resume_gives_the_same_batches(fake_release, workers):
    """Through a real DataLoader (spawned workers when > 0): the batches of a run resumed at step s are the
    uninterrupted run's batches s, s+1, ... (same trials, same order)."""
    B = 4
    length, trials = _fake_epochs(world=1, rank=0)
    pics = ReleasePictures(fake_release)

    def texts(start_step, n):
        ds = ReleaseDataset(pics, scene_size=32, aug=None)
        stream = EpochStream(length=length, trials=lambda e: [t._replace(image_idx=i % 3) for i, t in
                                                                enumerate(trials(e))], start=start_step * B)
        kw = dict(num_workers=workers, multiprocessing_context="spawn") if workers else {}
        dl = DataLoader(ds, sampler=stream, batch_size=B, drop_last=True, collate_fn=collate_search, **kw)
        out = []
        for b in dl:
            out.append(b["text"])
            if len(out) == n:
                break
        return out

    whole = texts(0, 30)
    assert texts(12, 18) == whole[12:]  # step 12 starts at trial 48; epoch 0 ends at 50


# --- the real release (spec section 6) --------------------------------------------------------------------

needs_release = pytest.mark.skipif(not (RELEASE / "meta.json").exists(), reason="search release not mounted")


@pytest.mark.slow
@pytest.mark.parametrize("release", PARITY_RELEASES, ids=lambda p: p.name)
def test_epochs_equal_the_reference(release):
    """Check 1: with the defaults and seed 0, epochs 0 and 1 equal reference/epoch_s0_e{0,1}.parquet (same rows,
    same order, every column; ``vp_idx`` too on a release with visual prompts). ~9 GB of RAM, ~1 min."""
    if not (release / "meta.json").exists():
        pytest.skip(f"{release} not on disk")
    from canvit.search.release import ReleaseEpochs
    ep = ReleaseEpochs(release, seed=0)
    for e in (0, 1):
        ref = pd.read_parquet(release / "reference" / f"epoch_s0_e{e}.parquet")
        got = ep.frame(e)
        assert list(got.columns) == list(ref.columns)
        assert len(got) == len(ref) == ep.size(e)
        for c in ref.columns:
            a, b = got[c].to_numpy(), ref[c].to_numpy()
            if c.startswith("box_"):
                assert np.array_equal(a.astype(np.float32), b, equal_nan=True), (e, c)
            else:
                assert (a.astype(object) == b.astype(object)).all(), (e, c)


@pytest.mark.slow
@pytest.mark.parametrize("release", PARITY_RELEASES, ids=lambda p: p.name)
def test_val_trials_are_val_parquet_as_is(release):
    """The val trials are val.parquet's rows as they are (no sampler; e.g. basic-v3's source-matched absent trials):
    every present trial, then interleaved with the absent ones in file order, with their picture, box, text and
    (on a visual-prompt release) vp_idx."""
    if not (release / "meta.json").exists():
        pytest.skip(f"{release} not on disk")
    from canvit.search.release import val_trials
    vp = (release / "vp_features.npy").exists()
    v = pd.read_parquet(release / "val.parquet")
    got = val_trials(release, visual_prompts=vp)
    tp, ta = np.flatnonzero(v["present"].to_numpy()), np.flatnonzero(~v["present"].to_numpy())
    n = min(len(tp), len(ta))
    order = [*(i for pair in zip(tp[:n], ta[:n]) for i in pair), *tp[n:], *ta[n:]]
    assert len(got) == len(v) == len(order)
    w = v.iloc[order]
    assert [t.image_idx for t in got] == w["image_idx"].tolist()
    assert [t.text for t in got] == w["text"].tolist()
    assert [t.box is not None for t in got] == w["present"].tolist()
    box = w[["box_x", "box_y", "box_w", "box_h"]].to_numpy(np.float64)
    assert all(t.box == tuple(float(x) for x in b) for t, b, p in zip(got, box, w["present"]) if p)
    assert [t.vp_idx for t in got] == (w["vp_idx"].tolist() if vp else [-1] * len(w))


@pytest.mark.slow
@needs_release
def test_pictures_equal_the_reference():
    """Check 2: every reference picture, read without draft and fixed, has the size (width, height) and a 16x16
    BOX thumbnail within a mean absolute difference of 2 of ``thumb16``."""
    ref = pd.read_parquet(RELEASE / "reference" / "images.parquet")
    pics = ReleasePictures(RELEASE)
    mads = []
    for r in ref.itertuples():
        im = pics.read(int(r.image_idx))
        assert im.size == (r.width, r.height), r.image_idx
        th = np.asarray(im.resize((16, 16), Image.Resampling.BOX), np.float32)
        want = np.frombuffer(r.thumb16, np.uint8).reshape(16, 16, 3).astype(np.float32)
        mads.append(np.abs(th - want).mean())
    assert max(mads) <= 2.0, f"worst thumbnail MAD {max(mads):.2f}"


def test_band_metrics_follow_the_overall_rules(fake_cs18):
    """evaluate() on val batches that carry a ``band``: with every trial in one band, that band's last-glimpse
    metrics equal the overall ones; the other bands are empty."""
    from canvit.search.release import BANDS
    from canvit.search.tests.test_search_task import _cfg, _rl, _TinyTask

    root, _ = fake_cs18
    task = _TinyTask(_cfg(root), rl=_rl())
    task.cfg.limit_val_batches = None  # every val batch, so the band counts cover the whole loader
    model, head = task.build_model(torch.device("cpu"))
    plain = task.build_val_loader()
    banded = [dict(b, band=torch.full_like(b["target"], 2)) for b in plain]
    m = task.evaluate(model=model, head=head, val_loader=banded, device=torch.device("cpu"), step=0)
    L = task.cfg.n_timesteps - 1
    for p in ("random", "greedy"):
        for k in ("found", "correct_found", "presence_acc"):
            assert m[f"{k}_t{L}_{p}_band{BANDS[2]}"] == m[f"{k}_t{L}_{p}"], (p, k)
    assert m[f"n_trials_band{BANDS[2]}"] == sum(len(b["present"]) for b in banded)
    assert m[f"n_trials_band{BANDS[0]}"] == 0
    assert not any("_band" in k for k in task.evaluate(model=model, head=head, val_loader=plain,
                                                         device=torch.device("cpu"), step=0))


# --- visual prompts (release spec section 7) ---------------------------------------------------------------

N_VP = 5


@pytest.fixture(scope="module")
def fake_vp_release(fake_release):
    """fake_release plus the visual-prompt tables the loader reads: vp_features.npy (float16 [5, 14, 768]),
    vp_boxes.parquet (vp_idx, box_id, coverage), vp_test.parquet (category, vp_idx)."""
    g = np.random.default_rng(0)
    np.save(fake_release / "vp_features.npy", g.normal(size=(N_VP, 14, 768)).astype(np.float16))
    pd.DataFrame({"vp_idx": np.arange(N_VP), "box_id": [f"src|ann:{i}" for i in range(N_VP)],
                  "coverage": [g.random(14).astype(np.float32) for _ in range(N_VP)]}
                 ).to_parquet(fake_release / "vp_boxes.parquet")
    pd.DataFrame({"category": ["bottle", "bottle", "bottle", "bowl"], "vp_idx": [0, 2, 4, 1]}
                 ).to_parquet(fake_release / "vp_test.parquet")
    return fake_release


def test_visual_prompt_reads_equal_the_array(fake_vp_release):
    """mmap reads (in a fresh, unpickled copy, as in a worker) equal np.load rows cast to float32; -1 gives
    zeros and vp_valid False."""
    import pickle

    from canvit.search.release import VisualPrompts
    full = np.load(fake_vp_release / "vp_features.npy")
    cov = np.stack(pd.read_parquet(fake_vp_release / "vp_boxes.parquet")["coverage"].to_numpy())
    vp = pickle.loads(pickle.dumps(VisualPrompts(fake_vp_release)))
    assert vp._feat is None
    for i in range(N_VP):
        s = vp.sample(i)
        assert s["vp_valid"] and s["vp_feat"].dtype == torch.float32
        assert np.array_equal(s["vp_feat"].numpy(), full[i].astype(np.float32))
        assert np.array_equal(s["vp_cov"].numpy(), cov[i])
    assert isinstance(vp._feat, np.memmap)
    z = vp.sample(-1)
    assert not z["vp_valid"] and not z["vp_feat"].any() and not z["vp_cov"].any() and z["vp_feat"].shape == (14, 768)
    with pytest.raises(FileNotFoundError):
        VisualPrompts(fake_vp_release / "nowhere")


def test_release_samples_carry_the_visual_prompt_only_when_asked(fake_vp_release):
    from canvit.search.release import VisualPrompts
    pics = ReleasePictures(fake_vp_release)
    plain = ReleaseDataset(pics, scene_size=32, aug=None)
    withvp = ReleaseDataset(pics, scene_size=32, aug=None, visual_prompts=VisualPrompts(fake_vp_release))
    a, b = plain[Trial(0, BOX, "x", vp_idx=3)], withvp[Trial(0, BOX, "x", vp_idx=3)]
    assert not any(k.startswith("vp_") for k in a)
    assert all(torch.equal(a[k], b[k]) for k in ("image", "boxes", "present"))
    batch = collate_search([b, withvp[Trial(1, None, "y")]])
    assert batch["vp_feat"].shape == (2, 14, 768) and batch["vp_valid"].tolist() == [True, False]


def test_coco_test_prompts_are_the_smallest_hash(fake_vp_release):
    import hashlib

    from canvit.search.release import coco_test_prompts

    def h(name, cat, box_id):
        return int.from_bytes(hashlib.blake2b(f"{name}|{cat}|{box_id}".encode(), digest_size=8).digest(), "big")
    pairs = [("a.jpg", "bottle"), ("b.jpg", "bottle"), ("a.jpg", "bowl"), ("a.jpg", "tv")]
    got = coco_test_prompts(fake_vp_release, pairs)
    want = [min((0, 2, 4), key=lambda i: h("a.jpg", "bottle", f"src|ann:{i}")),
            min((0, 2, 4), key=lambda i: h("b.jpg", "bottle", f"src|ann:{i}")), 1, -1]
    assert got == want and got == coco_test_prompts(fake_vp_release, pairs)


def test_coco_visual_prompts_for_eval_only(fake_cs18, fake_vp_release):
    from canvit.search.config import SearchConfig
    from canvit.search.data import make_search_loaders, make_search_val_loader

    root, _ = fake_cs18
    cfg = SearchConfig(data_root=root, release_dir=fake_vp_release, visual_prompts=True, num_workers=0,
                       eval_batch_size=4)
    with pytest.raises(ValueError, match="TEST only"):
        make_search_loaders(cfg)
    loader, classes = make_search_val_loader(cfg)
    b = next(iter(loader))
    assert b["vp_feat"].shape == (4, 14, 768)
    # each trial's prompt is a vp_test box of its own category, or none
    by_cat = {"bottle": {0, 2, 4}, "bowl": {1}}
    for rec in loader.dataset.records:
        assert rec.vp_idx in by_cat.get(rec.text, {-1})
