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
from pysaliency.utils.variable_length_array import VariableLengthArray

from canvit.scanpath.data import ScanpathDataset, clip_to_stimuli, collate_scanpath, local_mirror, pixel_to_scene
from canvit.scanpath.evaluate import cut_at_first_target_fixation, scores, select
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


def test_cut_keeps_fixations_up_to_the_first_one_in_the_target_box():
    box = [0.1 * W, 0.2 * H, 0.2 * W, 0.2 * H]  # x 16.8..50.4, y 21..42
    inn, out = (30.0, 30.0), (W / 2, H / 2)  # the start fixation (index 0) in the box does not count
    paths = [[inn, out, inn, out, inn], [out, out, out], [out, inn]]
    fix = pysaliency.ScanpathFixations(pysaliency.Scanpaths(
        xs=VariableLengthArray([[p[0] for p in s] for s in paths]),
        ys=VariableLengthArray([[p[1] for p in s] for s in paths]), n=np.zeros(3, int),
        ts=VariableLengthArray([list(range(len(s))) for s in paths]),
        scanpath_attributes={"target_present": np.ones(3, int), "target_bbox": np.array([box] * 3)}))
    cut = cut_at_first_target_fixation(fix)
    assert list(zip(cut.scanpath_index, cut.scanpath_history_length)) == [(0, 0), (0, 1), (0, 2), (1, 0), (1, 1),
                                                                         (1, 2), (2, 0), (2, 1)]


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
    assert [eval_part(d) for d in DATASETS] == ["validation"] * 3 + ["all"] * 3  # test: opt-in (--cfg.part test)
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
    from canvit.scanpath.evaluate import grid_diagnostics, per_fixation_scores, scored_fixations
    st, ft, sv, fv = fake_search
    cb = CrossvalidatedBaselineModel(sv, fv, bandwidth=0.1, eps=1e-3)
    m = _random_grid_model(fv)
    fx = scored_fixations(fv)
    per_fix = per_fixation_scores(m, sv, fx, cb)
    d = grid_diagnostics(m, sv, fx, per_fix, train=(st, ft, _random_grid_model(ft, seed=1, sharp_at_fixation=3.0)))
    # per-index entries need >= 20 fixations (entropy) / >= 50 (tau); this tiny split has fewer, so none appear
    assert np.isfinite(d["entropy_fix"]) and not any(k.startswith("entropy_t") for k in d)
    assert d["tempered_per_index"]["temperature"] == {}
    g = d["tempered_global"]
    assert {"temperature", "ll_image", "ig_image", "entropy_fix"} <= set(g) and g["temperature"] > 0
    # the exact identity agrees with pysaliency run on the tempered model
    tempered = per_fixation_scores(m.with_temperature(g["temperature"]), sv, fx, cb)
    from pysaliency.utils import average_values
    for k in ("ll", "ig"):
        assert math.isclose(g[f"{k}_image"], average_values(tempered[k], fx, average="image"), rel_tol=1e-9)


def test_parallel_scoring_equals_one_process(fake_search):
    from canvit.scanpath.evaluate import parallel_per_fixation_scores, per_fixation_scores, scored_fixations
    _, _, sv, fv = fake_search
    cb = CrossvalidatedBaselineModel(sv, fv, bandwidth=0.1, eps=1e-3)
    m = _random_grid_model(fv)
    fx = scored_fixations(fv)
    one = per_fixation_scores(m, sv, fx, cb)
    par = parallel_per_fixation_scores(m, sv, fx, cb, processes=2)
    assert one.keys() == par.keys() and all(np.array_equal(one[k], par[k]) for k in one)


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


def test_nss_auc_on_a_seeded_subset_ll_on_all(fake_search):
    from canvit.scanpath.evaluate import aggregate, density_metrics_mask, per_fixation_scores, scored_fixations
    _, _, sv, fv = fake_search
    cb = CrossvalidatedBaselineModel(sv, fv, bandwidth=0.1, eps=1e-3)
    fx = scored_fixations(fv)
    mask = density_metrics_mask(fx, 10, seed=3)
    assert mask.sum() == 10 and np.array_equal(mask, density_metrics_mask(fx, 10, seed=3))
    assert density_metrics_mask(fx, None, 0).all()
    sub = scores(cb, sv, fv, cb, density_metrics_fixations=10, density_metrics_seed=3)
    full = scores(cb, sv, fv, cb)
    assert sub["ll_fix"] == full["ll_fix"] and sub["n_fixations_nss_auc"] == 10
    pf = per_fixation_scores(cb, sv, fx, cb)
    assert math.isclose(sub["nss_fix"], pf["nss"][mask].mean()) and math.isclose(sub["auc_fix"], pf["auc"][mask].mean())
    # shards with the sliced mask give the same numbers
    shards = [per_fixation_scores(cb, sv, fx[i:i + 7], cb, density_mask=mask[i:i + 7]) for i in range(0, len(fx), 7)]
    merged = {k: np.concatenate([sh[k] for sh in shards]) for k in shards[0]}
    assert aggregate(merged) == pytest.approx(sub, rel=1e-12, abs=1e-12)


# --- display frame (2026-11-23): stimuli letterboxed into the training display's box ----------------

FRAME = (1050, 1680)  # (H, W) of COCO-Search18's display, the search runs' training geometry


def test_display_frame_is_the_identity_for_stimuli_as_wide_as_the_frame():
    """CS18 / FreeView (1680x1050), CAT2000 / DAEMONS (1920x1080) and anything wider are unchanged by the frame."""
    from canvit.scanpath.data import letterbox_geometry, parse_display_frame
    assert parse_display_frame("scene") is None and parse_display_frame("1680x1050") == FRAME
    for hw in ((1050, 1680), (1080, 1920), (500, 1000), (321, 514)):
        assert letterbox_geometry(*hw, 512, FRAME) == letterbox_geometry(*hw, 512)
        _, count = _pixel_cells(*hw, 64, 32, FRAME)
        assert np.array_equal(count, _pixel_cells(*hw, 64, 32)[1])


def test_narrower_stimuli_stay_inside_the_frames_box():
    from canvit.scanpath.data import letterbox_geometry
    box_h = 320  # 1050/1680 of a 512 scene: content rows 96..415, as COCO-Search18 in training
    for h, w in ((1000, 1000), (768, 1024), (1024, 768)):
        ch, cw, top, left = letterbox_geometry(h, w, 512, FRAME)
        assert ch == box_h and top == 96 and abs(cw / ch - w / h) < 0.01 and left == (512 - cw) // 2
        assert letterbox_geometry(h, w, 512)[0] > box_h  # the plain letterbox would leave the box


def test_framed_sample_and_density(tmp_path):
    from torchvision.io import write_jpeg
    p = tmp_path / "square.jpg"
    write_jpeg(torch.randint(0, 255, (3, 80, 80), dtype=torch.uint8), str(p))
    stim = pysaliency.FileStimuli([str(p)])
    fix = scanpaths(1, per_stimulus=1, h=80, w=80)
    s = ScanpathDataset(stim, fix, scene_size=64, frame=FRAME)[0]
    assert torch.allclose(s["fix"][0], torch.zeros(2), atol=1e-6)  # the image centre is still the scene centre
    assert torch.allclose(s["content"][[1, 3]], torch.tensor([-0.625, 0.625]), atol=1e-6)  # rows 12..51 of 64
    assert s["frame"].tolist() == list(FRAME)
    from canvit.core.preprocess import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
    black = -torch.tensor(IMAGENET_DEFAULT_MEAN) / torch.tensor(IMAGENET_DEFAULT_STD)  # a zero pixel, normalised
    assert torch.allclose(s["image"][:, :12], black[:, None, None].expand(3, 12, 64))  # padding above the box
    ld = grid_log_density(np.random.default_rng(0).standard_normal(1024), hw=(80, 80), scene=64, frame=FRAME)
    assert math.isclose(np.exp(ld).sum(), 1.0, rel_tol=1e-9)
    m = GridScanpathModel({(0, 1): np.zeros(1024, np.float32)}, scene_size=64, name="x", frame=FRAME)
    m.save(tmp_path / "m.npz")
    assert GridScanpathModel.load(tmp_path / "m.npz").frame == FRAME
    m2 = GridScanpathModel({(0, 1): np.zeros(1024, np.float32)}, scene_size=64, name="x")
    m2.save(tmp_path / "m2.npz")
    with pytest.raises(ValueError, match="predictions of"):
        GridScanpathModel.load(tmp_path / "m.npz", tmp_path / "m2.npz")


def test_the_display_frame_comes_from_the_checkpoints_training_data(fake_search, tmp_path):
    from canvit.scanpath.evaluate import frame_is_identity, training_display_frame
    assert training_display_frame("search")[0] == "1680x1050"
    assert {training_display_frame(t)[0] for t in ("distill", "in1k", "ade20k")} == {"scene"}
    with pytest.raises(ValueError, match="not known"):
        training_display_frame("some_new_task")
    sv = fake_search[2]  # 105 x 168 = COCO-Search18's 16:10: the frame changes nothing
    assert frame_is_identity(sv, FRAME, 512)
    from torchvision.io import write_jpeg
    p = tmp_path / "sq.jpg"
    write_jpeg(torch.zeros(3, 50, 50, dtype=torch.uint8), str(p))
    assert not frame_is_identity(pysaliency.FileStimuli([str(p)]), FRAME, 512)


# --- COCO-Search18's official test split (data.coco_search18_test) ---------------------------------------------------

def test_cs18_test_split_rescales_tp_and_keeps_one_stimulus_per_image_and_task(tmp_path):
    import json
    import zipfile

    from PIL import Image

    from canvit.scanpath.data import CS18_TEST_JSON, coco_search18_test
    rel = tmp_path / "release"
    rel.mkdir()
    img = Image.fromarray(np.full((1050, 1680, 3), 128, np.uint8))
    for zname, root, entries in (("COCOSearch18-images-TP.zip", "images", [("bottle", "a.jpg"), ("cup", "a.jpg")]),
                                 ("COCOSearch18-images-TA.zip", "coco_search18_images_TA", [("bottle", "b.jpg")])):
        with zipfile.ZipFile(rel / zname, "w") as z:
            for task, name in entries:
                p = tmp_path / name
                img.save(p)
                z.write(p, f"{root}/{task}/{name}")
    tp = [{"name": "a.jpg", "subject": 1, "task": t, "condition": "present", "bbox": [0, 0, 10, 10], "X": [256, 512],
           "Y": [160, 0], "T": [100, 200, 300], "length": 2, "correct": 1, "split": "test"} for t in ("bottle", "cup")]
    ta = [{"name": "b.jpg", "subject": 2, "task": "bottle", "condition": "absent", "X": [840, 100], "Y": [525, 50],
           "T": [100, 200], "length": 2, "correct": 1, "split": "test"}]
    for cond, data in (("TP", tp), ("TA", ta)):
        path = tmp_path / CS18_TEST_JSON.format(cond=cond)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(data))
    stimuli, fix = coco_search18_test(tmp_path, (rel,))
    assert len(stimuli) == 3 and len(set(stimuli.stimulus_ids)) == 3  # same image, two tasks: two distinct stimuli
    assert [tuple(s) for s in stimuli.sizes] == [(1050, 1680)] * 3
    present = np.asarray(fix.scanpaths.scanpath_attributes["target_present"]).astype(bool)
    assert present.tolist() == [True, True, False]
    np.testing.assert_allclose(fix.scanpaths.xs[0], [840, 1680])  # TP: 512x320 -> 1680x1050
    np.testing.assert_allclose(fix.scanpaths.ys[0], [525, 0])
    np.testing.assert_allclose(fix.scanpaths.fixation_attributes["durations"][0], [0.1, 0.2])  # T cut to the X/Y
    np.testing.assert_allclose(fix.scanpaths.xs[2], [840, 100])  # TA: already display pixels
    assert "split" not in fix.scanpaths.scanpath_attributes
    again, _ = coco_search18_test(tmp_path, (tmp_path / "no-zips",))  # cached: the zips are not read again
    assert again.stimulus_ids[:] == stimuli.stimulus_ids[:]


@pytest.mark.skipif(not (Path(__import__("os").environ.get("SCANPATH_DATA_DIR", "/nonexistent"))
                         / "COCO-Search18_test_no-task-merge_unique-images").exists(),
                    reason="the real COCO-Search18 test cache is not built here")
def test_cs18_test_keys_are_disjoint_from_split1_and_fixations_land_in_the_image():
    import os

    from canvit.scanpath.data import coco_search18_test, load_dataset
    loc = Path(os.environ["SCANPATH_DATA_DIR"])
    rel = tuple(Path(p) for p in os.environ.get("COCO_SEARCH18_RELEASE_DIRS", "").split(":") if p)
    keys = {}
    for part in ("train", "validation", "test"):
        st, fx = load_dataset("coco_search18", part, location=loc, release_dirs=rel)
        keys[part] = {Path(f).parent.name + "/" + Path(f).stem for f in st.filenames}
        if part == "test":
            raw = coco_search18_test(loc, rel)[1]  # before clip_to_stimuli: the rescaled JSON coordinates themselves
            h, w = (np.array([st.sizes[int(n)] for n in raw.n])).T
            assert len(st) == 1224 and len(fx.scanpaths) == 6118 + 6120
            off = ~((raw.x >= 0) & (raw.x < w) & (raw.y >= 0) & (raw.y < h))
            assert off.sum() <= 10  # 6 of 53553 (tracker glitches above the image; val has 6 too), clipped like val's
            assert np.all((fx.x >= 0) & (fx.x < w) & (fx.y >= 0) & (fx.y < h))
            for cond in ("tp", "ta"):
                assert len(set(select(fx, condition=cond, limit=None).scanpaths.n)) == 612
    assert not keys["test"] & keys["train"] and not keys["test"] & keys["validation"]
