"""Scanpath evaluation on pysaliency: geometry, the grid -> pixel density adapter, data selection and
clipping, the scores, the release-file mirror (design §7)."""

import hashlib
import math
from pathlib import Path

import numpy as np
import pysaliency
import pytest
import torch
from pysaliency.baseline_utils import CrossvalidatedBaselineModel

from canvit.scanpath.data import ScanpathDataset, clip_to_stimuli, collate_scanpath, local_mirror, pixel_to_scene
from canvit.scanpath.evaluate import scores, select
from canvit.scanpath.models import GridScanpathModel, _pixel_cells, fit_centre_bias, grid_log_density

from .conftest import H, W, scanpaths

CS18 = (1050, 1680)


# --- geometry ------------------------------------------------------------------------------------

def test_display_centre_maps_to_scene_origin():
    xy = pixel_to_scene(torch.tensor([[840.0, 525.0]]), hw=CS18, scene=512)
    assert torch.allclose(xy, torch.zeros(1, 2), atol=1e-6)


def test_display_corners_map_to_the_letterboxed_content():
    xy = pixel_to_scene(torch.tensor([[0.0, 0.0], [1680.0, 1050.0]]), hw=CS18, scene=512)
    top = 96 / 512 * 2 - 1  # 1680x1050 at 512: content 512x320, 96 px padding above and below
    assert torch.allclose(xy, torch.tensor([[-1.0, top], [1.0, -top]]), atol=1e-6)


def test_image_and_box_match_the_search_dataset(tmp_path):
    """The same file and box through SearchDataset (val path) and ScanpathDataset: one geometry."""
    from torchvision.io import write_jpeg

    from canvit.search.data import SearchDataset, SearchRecord
    p = tmp_path / "a.jpg"
    write_jpeg(torch.randint(0, 255, (3, *CS18), dtype=torch.uint8, generator=torch.Generator().manual_seed(1)), str(p))
    bbox = (300.0, 200.0, 160.0, 120.0)
    fix = scanpaths(1, per_stimulus=1, h=CS18[0], w=CS18[1])
    fix.scanpaths.scanpath_attributes["target_bbox"][0] = bbox
    got = ScanpathDataset(pysaliency.FileStimuli([str(p)], attributes={"task": np.array([4])}), fix, scene_size=512)[0]
    ref = SearchDataset([SearchRecord(path=p, target=4, bbox_xywh=bbox)], scene_size=512, aug=None)[0]
    assert torch.equal(got["image"], ref["image"]) and torch.allclose(got["boxes"], ref["boxes"], atol=1e-5)
    assert got["cue"].item() == 4


def test_fixation_lands_on_the_pixel_it_looked_at():
    img = np.zeros((*CS18, 3), dtype=np.uint8)
    img[290:330, 1180:1220] = 255  # centre (1200, 310)
    fix = scanpaths(1, per_stimulus=1, h=CS18[0], w=CS18[1])
    fix.scanpaths.xs._data[0, 1], fix.scanpaths.ys._data[0, 1] = 1200.0, 310.0
    s = ScanpathDataset(pysaliency.Stimuli([img]), fix, scene_size=512)[0]
    x, y = ((s["fix"][1] + 1) / 2 * 512).long().tolist()
    assert s["image"][:, y, x].min() > 1.5 and s["cue"].item() == -1  # normalised white; free viewing


def test_collate_pads_with_nan(fake_search):
    _, _, sv, fv = fake_search
    ds = ScanpathDataset(sv, fv, scene_size=64)
    b = collate_scanpath([ds[i] for i in range(5)])
    for i in range(5):
        n = int(b["n_fix"][i])
        assert torch.isfinite(b["fix"][i, :n]).all() and torch.isnan(b["fix"][i, n:]).all()
    present = np.asarray(fv.scanpaths.scanpath_attributes["target_present"][:5]).astype(bool)
    assert torch.isfinite(b["boxes"][present]).all() and torch.isnan(b["boxes"][~present]).all()


# --- grid -> pixel density -----------------------------------------------------------------------

def test_grid_density_sums_to_one_and_flat_logits_give_pysaliencys_uniform():
    ld = grid_log_density(np.random.default_rng(0).standard_normal(1024), hw=(H, W), scene=64)
    assert ld.shape == (H, W) and math.isclose(np.exp(ld).sum(), 1.0, rel_tol=1e-9)
    uniform = grid_log_density(np.zeros(1024), hw=(H, W), scene=64)
    assert np.allclose(uniform, pysaliency.UniformModel().log_density(np.zeros((H, W, 3))))


def test_a_cell_cut_by_the_letterbox_keeps_its_per_pixel_density():
    """Cells are equal in scene area, so a logit is a density per area: a cell the image edge cuts must not
    concentrate the whole cell's mass on its few pixels (the bug before 2026-11-23)."""
    h, w = 100, 168  # letterboxed into 64 px: content rows 13..51, so the top and bottom cell rows are cut
    cell, count = _pixel_cells(h, w, 64, 32)
    cut = 6 * 32 + 16  # cell row 6 = scene rows 12-13, of which only row 13 shows the image
    assert 0 < count[cut] < 0.75 * np.median(count[count > 0])
    logits = np.zeros(1024)
    logits[cut] = 1.0  # the cut cell is e times as dense as every other cell
    d = np.exp(grid_log_density(logits, hw=(h, w), scene=64))
    inside, other = d[cell == cut], d[cell != cut]
    assert np.allclose(inside, inside[0]) and np.allclose(other, other[0])
    assert math.isclose(inside[0] / other[0], math.e, rel_tol=1e-9) and math.isclose(d.sum(), 1.0, rel_tol=1e-9)


def test_grid_model_scores_the_cell_of_the_fixation(fake_search):
    _, _, sv, fv = fake_search
    fx = fv[fv.scanpath_history_length > 0]
    cell, count = _pixel_cells(H, W, 64, 32)
    table = {}
    for i in range(len(fx)):  # all mass on the cell of the fixation it predicts
        logits = np.full(1024, -1e9)
        logits[cell[fx.y_int[i], fx.x_int[i]]] = 0.0
        table[(int(fx.scanpath_index[i]), int(fx.scanpath_history_length[i]))] = logits
    m = GridScanpathModel(table, scene_size=64, name="delta")
    expected = -np.log([count[cell[fx.y_int[i], fx.x_int[i]]] for i in range(len(fx))])
    assert np.allclose(m.log_likelihoods(sv, fx), expected)
    with pytest.raises(KeyError, match="no prediction"):
        GridScanpathModel({}, scene_size=64, name="empty").log_likelihoods(sv, fx[:1])


# --- selection, clipping, scores -----------------------------------------------------------------

def test_select_by_condition_and_limit(fake_search):
    fv = fake_search[3]
    tp = np.asarray(fv.scanpaths.scanpath_attributes["target_present"]).astype(bool)
    assert len(select(fv, condition="tp", limit=None).scanpaths) == tp.sum()
    assert len(select(fv, condition="ta", limit=None).scanpaths) == (~tp).sum()
    assert len(select(fv, condition="both", limit=5).scanpaths) == 5


def test_clipping_reaches_fixations_and_histories(fake_search):
    sv = fake_search[2]
    fix = scanpaths(1, per_stimulus=1)
    fix.scanpaths.xs._data[0, 1] = W + 30.0  # off the stimulus
    fix = pysaliency.ScanpathFixations(fix.scanpaths)
    clipped, n = clip_to_stimuli(sv, fix)
    assert n == 1 and clipped.x.max() < W and np.nanmax(clipped.x_hist._data) < W


def test_scores_uniform_and_centre_bias(fake_search):
    st, ft, sv, fv = fake_search
    cb = CrossvalidatedBaselineModel(sv, fv, bandwidth=0.1, eps=1e-3)
    u = scores(pysaliency.UniformModel(), sv, fv, cb)
    c = scores(cb, sv, fv, cb)
    assert u["n_fixations"] == (fv.scanpath_history_length > 0).sum()
    assert abs(u["ll_image"]) < 1e-9 and abs(u["ll_fix"]) < 1e-9 and math.isclose(u["auc_fix"], 0.5)
    assert abs(c["ig_fix"]) < 1e-9 and c["ll_fix"] > 0  # central fixations: the centre bias wins


def test_fit_centre_bias_picks_from_the_grid(fake_search):
    st, ft, _, _ = fake_search
    fit = fit_centre_bias(st, ft, n_images=4, bandwidths=(0.05, 0.2), epss=(1e-3,))
    assert fit["bandwidth"] in (0.05, 0.2) and np.isfinite(fit["ll"])


# --- release-file mirror -------------------------------------------------------------------------

def test_local_mirror_checks_md5(tmp_path):
    import types
    f = tmp_path / "x.json"
    f.write_text("{}")
    md5 = hashlib.md5(b"{}").hexdigest()
    mod = types.SimpleNamespace(download_and_check=lambda *a, **k: None)
    with local_mirror(mod, {"x.json": f}):
        mod.download_and_check("http://h/x.json", str(tmp_path / "t1"), md5 + ".")  # pysaliency's stray '.'
        with pytest.raises(ValueError, match="md5"):
            mod.download_and_check("http://h/x.json", str(tmp_path / "t2"), "0" * 32)
    assert (tmp_path / "t1").resolve() == f.resolve() and mod.download_and_check("u", "t", "m") is None


def test_cli_parses_the_scanpath_subcommand():
    import tyro

    from canvit.harness.evaluate import Command, ScanpathEval
    from canvit.scanpath.evaluate import split_name
    cmd = tyro.cli(Command, args=["scanpath", "--cfg.model", "uniform", "--cfg.condition", "tp"])  # pyright: ignore
    assert isinstance(cmd, ScanpathEval) and cmd.cfg.model == "uniform" and cmd.cfg.condition == "tp"
    assert split_name(cmd.cfg) == "tp_validation"
    cmd = tyro.cli(Command, args=["scanpath", "--cfg.dataset", "mit1003"])  # pyright: ignore
    assert cmd.cfg.dataset == "mit1003" and split_name(cmd.cfg) == "all"


def test_dataset_parts():
    from canvit.scanpath.data import DATASETS, eval_part, fit_part, load_dataset
    assert [eval_part(d) for d in DATASETS] == ["validation"] * 3 + ["all"] * 3
    assert [fit_part(d) for d in DATASETS] == ["train"] * 3 + ["all"] * 3
    with pytest.raises(ValueError, match="parts"):
        load_dataset("mit1003", "validation", location=Path("/nonexistent"))


def test_flat_density_has_nss_zero(fake_search):
    from canvit.scanpath.models import DensitySaliencyMap
    _, _, sv, fv = fake_search
    fx = fv[fv.scanpath_history_length > 0][:20]
    assert np.allclose(DensitySaliencyMap(pysaliency.UniformModel()).NSSs(sv, fx), 0.0)


# --- temperature, entropy, saved predictions (diagnostics for the CanViT results) --------------------

def _random_grid_model(fix, *, seed=0, sharp_at_fixation=0.0):
    """Random logits for every scored fixation, optionally raised at the cell the fixation landed in."""
    rng = np.random.default_rng(seed)
    cell, _ = _pixel_cells(H, W, 64, 32)
    fx = fix[fix.scanpath_history_length > 0]
    table = {}
    for i in range(len(fx)):
        lg = rng.standard_normal(1024)
        lg[cell[fx.y_int[i], fx.x_int[i]]] += sharp_at_fixation
        table[(int(fx.scanpath_index[i]), int(fx.scanpath_history_length[i]))] = lg
    return GridScanpathModel(table, scene_size=64, name="random")


def test_temperature_one_is_the_model_and_the_fit_beats_its_neighbours(fake_search):
    from canvit.scanpath.models import fit_temperature
    _, _, sv, fv = fake_search
    fx = fv[fv.scanpath_history_length > 0]
    m = _random_grid_model(fv, sharp_at_fixation=3.0)
    assert np.allclose(m.with_temperature(1.0).log_likelihoods(sv, fx), m.log_likelihoods(sv, fx))
    tau = fit_temperature(m, sv, fv)

    def ll(t):
        return m.with_temperature(t).log_likelihoods(sv, fx).mean()
    assert ll(tau) >= ll(tau * 1.3) and ll(tau) >= ll(tau / 1.3)
    per_t = fit_temperature(m, sv, fv, per_index=True, min_fixations=1)
    assert set(per_t) == set(np.unique(fx.scanpath_history_length).tolist())


def test_flat_logits_have_the_entropy_of_the_image_cells(fake_search):
    from canvit.scanpath.models import policy_entropy
    _, _, sv, fv = fake_search
    m = _random_grid_model(fv)
    m = GridScanpathModel({k: np.zeros(1024) for k in m.logits}, scene_size=64, name="flat")
    H_, _ = policy_entropy(m, sv, fv)
    _, count = _pixel_cells(H, W, 64, 32)
    p = count[count > 0] / count.sum()  # flat per pixel: each cell weighted by its pixels on the image
    assert np.allclose(H_, -(p * np.log(p)).sum())


def test_saved_predictions_load_back(fake_search, tmp_path):
    _, _, sv, fv = fake_search
    fx = fv[fv.scanpath_history_length > 0]
    m = _random_grid_model(fv)
    m.save(tmp_path / "p.npz")
    back = GridScanpathModel.load(tmp_path / "p.npz")
    assert set(back.logits) == set(m.logits) and back.scene == 64
    assert np.allclose(back.log_likelihoods(sv, fx), m.log_likelihoods(sv, fx), atol=5e-3)  # float16 storage


def test_grid_diagnostics_report_entropy_and_tempered_likelihoods(fake_search):
    from canvit.scanpath.evaluate import grid_diagnostics
    st, ft, sv, fv = fake_search
    cb = CrossvalidatedBaselineModel(sv, fv, bandwidth=0.1, eps=1e-3)
    d = grid_diagnostics(_random_grid_model(fv), sv, fv, cb,
                         train=(st, ft, _random_grid_model(ft, seed=1, sharp_at_fixation=3.0)))
    # per-index entries need >= 20 fixations (entropy) / >= 50 (tau); this tiny split has fewer, so none appear
    assert np.isfinite(d["entropy_fix"]) and not any(k.startswith("entropy_t") for k in d)
    assert d["tempered_per_index"]["temperature"] == {}
    for k in ("tempered_global", "tempered_per_index"):
        assert {"temperature", "ll_image", "ig_image"} <= set(d[k])
    assert d["tempered_global"]["temperature"] > 0 and np.isfinite(d["tempered_global"]["entropy_fix"])


def test_sharded_per_fixation_scores_aggregate_to_the_same_numbers(fake_search):
    from canvit.scanpath.evaluate import aggregate, per_fixation_scores, scored_fixations
    _, _, sv, fv = fake_search
    cb = CrossvalidatedBaselineModel(sv, fv, bandwidth=0.1, eps=1e-3)
    fx = scored_fixations(fv)
    shards = [per_fixation_scores(cb, sv, fx[i:i + 7], cb) for i in range(0, len(fx), 7)]
    merged = {k: np.concatenate([sh[k] for sh in shards]) for k in shards[0]}
    whole = scores(cb, sv, fv, cb)
    assert aggregate(merged) == pytest.approx(whole, rel=1e-12, abs=1e-12)
    assert whole["n_scanpaths"] == len(np.unique(fv[fv.scanpath_history_length > 0].scanpath_index))
