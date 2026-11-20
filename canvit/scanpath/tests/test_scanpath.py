"""Scanpath evaluation on pysaliency: geometry, the grid -> pixel density adapter, data selection and
clipping, the scores, the release-file mirror (design §7)."""

import hashlib
import math

import numpy as np
import pysaliency
import pytest
import torch
from pysaliency.baseline_utils import CrossvalidatedBaselineModel

from canvit.scanpath.data import ScanpathDataset, collate_scanpath, local_mirror, pixel_to_scene
from canvit.scanpath.evaluate import clip_to_stimuli, scores, select
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

def test_grid_density_sums_to_one_and_area_weighted_logits_give_pysaliencys_uniform():
    cell, count = _pixel_cells(H, W, 64, 32)
    ld = grid_log_density(np.random.default_rng(0).standard_normal(1024), hw=(H, W), scene=64)
    assert ld.shape == (H, W) and math.isclose(np.exp(ld).sum(), 1.0, rel_tol=1e-9)
    with np.errstate(divide="ignore"):
        uniform = grid_log_density(np.log(count), hw=(H, W), scene=64)
    assert np.allclose(uniform, pysaliency.UniformModel().log_density(np.zeros((H, W, 3))))


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
    cmd = tyro.cli(Command, args=["scanpath", "--cfg.model", "uniform", "--cfg.condition", "tp"])  # pyright: ignore
    assert isinstance(cmd, ScanpathEval) and cmd.cfg.model == "uniform" and cmd.cfg.condition == "tp"


def test_cli_takes_the_policys_task_with_its_training_flags():
    import tyro

    from canvit.harness.evaluate import Command
    from canvit.scanpath.sources import In1kSource, SearchSource
    cmd = tyro.cli(Command, args=["scanpath", "--cfg.model", "canvit_policy", "cfg.source:in1k",  # pyright: ignore
                                  "--cfg.source.cfg.canvas-grid", "8", "--cfg.source.rl.width", "32"])
    assert isinstance(cmd.cfg.source, In1kSource)
    assert cmd.cfg.source.cfg.canvas_grid == 8 and cmd.cfg.source.rl.width == 32
    default = tyro.cli(Command, args=["scanpath"])  # pyright: ignore
    assert isinstance(default.cfg.source, SearchSource) and default.cfg.source.rl.centers_per_axis == 32


def test_flat_density_has_nss_zero(fake_search):
    from canvit.scanpath.models import DensitySaliencyMap
    _, _, sv, fv = fake_search
    fx = fv[fv.scanpath_history_length > 0][:20]
    assert np.allclose(DensitySaliencyMap(pysaliency.UniformModel()).NSSs(sv, fx), 0.0)
