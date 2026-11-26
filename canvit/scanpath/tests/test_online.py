"""Human alignment during training (canvit.scanpath.online) on the tiny random search model and fake human data
(CPU): the cache, the grid-level scores against the full pipeline's density, and the training hook (off by default;
on, it changes nothing the run computes)."""

import logging

import numpy as np
import pysaliency
import pytest
import torch

from canvit.harness.run import RunSettings, run
from canvit.scanpath.data import letterbox_geometry, parse_display_frame
from canvit.scanpath.models import grid_log_density
from canvit.scanpath.online import SETS, HumanEval, build
from canvit.scanpath.tests.test_canvit_models import _episode, _in1k_episode
from canvit.search.tests.test_search_task import _cfg, _rl, _TinyTask

FIT = {"bandwidth": 0.05, "eps": 1e-3}


def _fv(stimuli, fixations):
    """The same images and scanpaths as free viewing: no ``task`` attribute (no cue)."""
    return pysaliency.FileStimuli(list(stimuli.filenames)), fixations


@pytest.fixture(scope="module")
def cache(fake_search, tmp_path_factory):
    _, _, sv, fv = fake_search
    frame = parse_display_frame(_TinyTask.display_frame)
    data = build((sv, fv), _fv(sv, fv), fit_cs18=FIT, fit_fv=FIT, scene=224, frame=frame, n_images=1, seed=0)
    path = tmp_path_factory.mktemp("online") / "human.pt"
    torch.save(data, path)
    return path


def _eval(cache, task, **kw):
    return HumanEval.load(cache, task=task, device=torch.device("cpu"), batch_size=3, **kw)


def test_the_cache_holds_the_same_images_for_every_set(cache):
    d = torch.load(cache, weights_only=False)
    for s in SETS:
        S = d["sets"][s]
        assert len(S["image"]) == len(S["cue"]) == len(S["fix"]) == len(S["length"]) > 0
        used = torch.unique(S["image"])
        for m in ("cb", "human"):
            assert torch.allclose(S[m][used].sum(-1), torch.ones(len(used)))
            assert (S[m][used] * (d["frac"][used] == 0)).sum() == 0  # no mass off the image
    assert set(d["sets"]["fv"]["image"].tolist()) == set(d["sets"]["tp"]["image"].tolist()) | set(
        d["sets"]["ta"]["image"].tolist())
    assert (d["sets"]["fv"]["cue"] == -1).all() and (d["sets"]["tp"]["cue"] >= 0).all()
    assert all(d["images"][i].any() for i in range(len(d["images"])))


def test_grid_ll_equals_the_full_pipelines_density(fake_cs18, cache):
    """ll_tp is pysaliency's LL of the policy's grid density (models.grid_log_density at the fixation's pixel),
    image-averaged; IG = LL minus the centre bias's grid LL."""
    ep = _episode(fake_cs18[0])
    he = _eval(cache, ep.task, max_fixations=3)
    got = he(task=ep.task, model=ep.model, joint=ep.joint)
    d, S = he.data, he.data["sets"]["tp"]
    rows = torch.arange(len(S["image"]))
    b = he._batch("tp", rows)
    logits = ep.run(b, 3)["policy"]
    vals, imgs = [], []
    for r in range(len(rows)):
        i = int(S["image"][r])
        h, w = (int(v) for v in d["hw"][i])
        frame = tuple(d["frame"]) if d["frame"] is not None else None
        for t in range(1, min(4, int(S["length"][r]))):
            ld = grid_log_density(logits[r, t - 1].numpy(), hw=(h, w), scene=d["scene"], frame=frame)
            x, y = S["fix"][r, t].tolist()  # scene -> pixel (inverse letterbox)
            ch, cw, top, left = letterbox_geometry(h, w, d["scene"], frame)
            px = int(((x + 1) / 2 * d["scene"] - left) * w / cw)
            py = int(((y + 1) / 2 * d["scene"] - top) * h / ch)
            vals.append((ld[min(py, h - 1), min(px, w - 1)] + np.log(h * w)) / np.log(2))
            imgs.append(i)
    want = np.mean([np.mean([v for v, j in zip(vals, imgs) if j == i]) for i in sorted(set(imgs))])
    assert got["ll_tp"] == pytest.approx(want, abs=0.05)  # cell edges: a fixation's pixel vs its exact position
    assert np.isfinite([got[k] for k in got]).all()
    assert got["entropy_tp"] > 0 and got["sacc_len_tp"] > 0 and got["sacc_len_human_tp"] > 0


def test_the_evaluation_leaves_rng_and_modes_untouched(fake_cs18, cache):
    ep = _episode(fake_cs18[0])
    ep.model.train()
    ep.joint.scorer.train()
    sels = [s for s in (ep.joint.policy_selector, getattr(ep.joint.policy_selector, "fallback", None),
                        ep.joint.random_selector) if isinstance(getattr(s, "generator", None), torch.Generator)]
    assert sels  # the tiny policy has generators, so the check below is not empty
    before = [torch.get_rng_state()] + [s.generator.get_state() for s in sels]
    _eval(cache, ep.task)(task=ep.task, model=ep.model, joint=ep.joint)
    after = [torch.get_rng_state()] + [s.generator.get_state() for s in sels]
    assert all(torch.equal(a, b) for a, b in zip(before, after))
    assert ep.model.training and ep.joint.scorer.training


def test_a_non_search_policy_is_scored_without_cues(cache):
    ep = _in1k_episode()  # 8 x 8 grid, no cue; its display frame differs, so the frame check is bypassed
    he = HumanEval(data=torch.load(cache, weights_only=False), device=torch.device("cpu"), batch_size=4)
    got = he(task=ep.task, model=ep.model, joint=ep.joint)
    assert set(got) == {f"{m}_{s}" for s in SETS for m in ("ig", "ll", "entropy", "entropy_human", "sacc_len",
                                                           "sacc_len_human")}
    assert np.isfinite(list(got.values())).all()
    assert got["entropy_tp"] <= np.log(64) + 1e-6


def test_a_cache_for_another_scene_or_frame_is_refused(fake_cs18, cache, tmp_path):
    ep = _episode(fake_cs18[0])
    d = torch.load(cache, weights_only=False)
    for change in ({"scene": 512}, {"frame": None}):
        p = tmp_path / "other.pt"
        torch.save(d | change, p)
        with pytest.raises(ValueError):
            HumanEval.load(p, task=ep.task, device=torch.device("cpu"))


def _run(task, **kw):
    settings = RunSettings(n_steps=2, eval_every=2, log_every=1, grad_clip=1.0, amp=False, seed=0, device="cpu",
                           tracker="none", **kw)
    return run(task=task, spec=task.default_spec(), settings=settings)


def test_training_with_the_human_eval_is_the_same_run(fake_cs18, cache, caplog):
    """On, the hook logs val/human/* at the eval step and changes nothing else the run computes."""
    root, _ = fake_cs18
    off = _run(_TinyTask(_cfg(root), rl=_rl()))
    with caplog.at_level(logging.INFO, logger="canvit.harness.run"):
        on = _run(_TinyTask(_cfg(root), rl=_rl()), human_eval_every=2, human_eval_data=cache)
    timing = {"data_pct", "gpu_pct"}  # wall-clock shares; everything the run computes must match exactly
    assert {k: v for k, v in off.items() if k not in timing} == {k: v for k, v in on.items() if k not in timing}
    assert "val/human/ig_tp" in caplog.text


def test_human_eval_every_must_be_a_multiple_of_eval_every(fake_cs18, cache):
    root, _ = fake_cs18
    with pytest.raises(ValueError, match="multiple"):
        _run(_TinyTask(_cfg(root), rl=_rl()), human_eval_every=3, human_eval_data=cache)
