"""The teacher-forced CanViT search episode against the deploy rollout (design §7), on the search
tests' tiny random foveated backbone (CPU)."""

import numpy as np
import pytest
import torch

from canvit.harness.rollout.eval_viewpoints import deploy_rollout_viewpoints
from canvit.harness.rollout.viewpoint import ViewpointType
from canvit.scanpath.canvit_models import CanViTModel, PolicyEpisode
from canvit.search.tests.test_search_task import _cfg, _rl, _TinyTask

B, S, CG = 2, 224, 32


def _episode(root, *, policy=True):
    task = _TinyTask(_cfg(root, condition="tp"), rl=_rl())
    model, _ = task.build_model(torch.device("cpu"))
    model.eval()
    joint = None
    if policy:
        joint = task.build_policy(model, device=torch.device("cpu"), canvas_grid=CG,
                                  generator=torch.Generator().manual_seed(0))
        torch.manual_seed(1)
        for p in joint.scorer.parameters():  # non-trivial scores, so argmaxes differ between cells
            p.data.normal_(0, 0.5)
        joint.scorer.eval()
    return PolicyEpisode(task=task, model=model, joint=joint, device=torch.device("cpu"), amp=False)


def _batch(fix):
    g = torch.Generator().manual_seed(2)
    return {"image": torch.randn(B, 3, S, S, generator=g), "cue": torch.tensor([0, 5]), "fix": fix,
            "content": torch.tensor([[-1.0, -1.0, 1.0, 1.0]]).repeat(B, 1),  # no letterbox: no clamping
            "hw": torch.tensor([[S, S]] * B)}


def _deploy(ep, images, cue, n):
    """The deploy rollout with the search task's canvas update; returns the centres it took
    (x, y) and every scorer output, in call order."""
    scores = []
    hook = ep.joint.scorer.register_forward_hook(lambda m, i, o: scores.append(o.detach().reshape(B, -1)))

    def advance(vp, state, t):
        if state is None:
            state = ep.model.init_state(batch_size=B, canvas_grid_size=CG)
        return ep.model.canvit(image=images, state=state, viewpoint=vp).state

    embed = ep.task._policy_embed(ep.model)
    try:
        taken = deploy_rollout_viewpoints(joint=ep.joint, advance=advance, t0_type=ViewpointType.FULL,
                                          batch_size=B, device=images.device, n=n, cond=lambda: embed(cue))
    finally:
        hook.remove()
    return torch.stack([vp.centers.flip(-1) for vp in taken], 1), scores


def test_forcing_the_policys_own_choices_reproduces_the_deploy_rollout(fake_cs18):
    ep = _episode(fake_cs18[0])
    batch = _batch(torch.zeros(B, 3, 2))
    centres, deploy_scores = _deploy(ep, batch["image"], batch["cue"], n=3)  # scorer calls at t = 1, 2
    batch["fix"] = centres
    forced = ep.run(batch, n=2)["policy"]  # glimpses at fixations 0, 1; predicts fixations 1, 2
    assert len(deploy_scores) == 2
    for t in range(2):
        assert torch.equal(forced[:, t], deploy_scores[t])
    # and the deployed choice is the argmax of what we score (cell centres of the 32x32 grid)
    cell = ((centres[:, 1:] + 1) / 2 * CG).floor().long()
    assert torch.equal(forced.argmax(-1), cell[..., 1] * CG + cell[..., 0])


def test_sampling_at_temperature_zero_is_the_deploy_rollout(fake_cs18):
    ep = _episode(fake_cs18[0])
    batch = _batch(torch.zeros(B, 3, 2))
    centres, _ = _deploy(ep, batch["image"], batch["cue"], n=3)
    assert torch.allclose(ep.sample(batch, 3, temperature=0.0), centres, atol=1e-6)


def test_sampling_is_seeded_and_spreads_with_temperature(fake_cs18):
    ep = _episode(fake_cs18[0])
    batch = _batch(torch.zeros(B, 4, 2))
    a = ep.sample(batch, 4, temperature=5.0, generator=torch.Generator().manual_seed(0))
    b = ep.sample(batch, 4, temperature=5.0, generator=torch.Generator().manual_seed(0))
    c = ep.sample(batch, 4, temperature=5.0, generator=torch.Generator().manual_seed(1))
    assert torch.equal(a, b) and not torch.equal(a, c)
    assert torch.equal(a[:, 0], torch.zeros(B, 2))  # fixation 0 is the centre anchor


def test_human_fixations_change_the_later_predictions(fake_cs18):
    ep = _episode(fake_cs18[0])
    a = torch.tensor([[[0.0, 0.0], [0.6, 0.6], [0.0, 0.0]]]).repeat(B, 1, 1)
    b = torch.tensor([[[0.0, 0.0], [-0.6, -0.6], [0.0, 0.0]]]).repeat(B, 1, 1)
    pa, pb = ep.run(_batch(a), n=2), ep.run(_batch(b), n=2)
    for k in ("policy", "heatmap"):
        assert torch.equal(pa[k][:, 0], pb[k][:, 0])      # after the anchor only: identical
        assert not torch.equal(pa[k][:, 1], pb[k][:, 1])  # after a different fixation 1: not


def test_precompute_covers_every_scored_fixation_and_feeds_pysaliency(fake_cs18, fake_search):
    from canvit.scanpath.canvit_models import precompute
    from canvit.scanpath.data import make_scanpath_loader
    _, _, sv, fv = fake_search
    ep = _episode(fake_cs18[0])
    loader = make_scanpath_loader(sv, fv, scene_size=S, batch_size=4, num_workers=0)
    model = precompute(CanViTModel(ep, "policy"), loader, grid=CG, scene_size=S)
    fx = fv[fv.scanpath_history_length > 0]
    keys = set(zip(fx.scanpath_index.tolist(), fx.scanpath_history_length.tolist()))
    assert set(model.logits) == keys
    ll = model.log_likelihoods(sv, fx)
    assert np.isfinite(ll).all() and ll.shape == (len(fx),)


def test_heatmap_readout_needs_no_policy(fake_cs18):
    ep = _episode(fake_cs18[0], policy=False)
    out = CanViTModel(ep, "heatmap").forced_logits(_batch(torch.zeros(B, 3, 2)), 3, CG)
    assert out.shape == (B, 3, CG * CG) and torch.isfinite(out).all()
    with pytest.raises(ValueError, match="grid"):
        CanViTModel(ep, "heatmap").forced_logits(_batch(torch.zeros(B, 3, 2)), 3, 16)


# --- a non-search task: the same episode for any task's policy (design §12) --------------------------

def _in1k_episode():
    """A tiny FOVEATED in1k classifier with its policy, built by the task's own constructors."""
    from dataclasses import replace

    from canvit.core import CanViTForImageClassification
    from canvit.in1k.config import In1kConfig
    from canvit.in1k.task import In1kRunTask
    from canvit.search.tests.test_search_task import _FOVEATED
    torch.manual_seed(0)
    clf = CanViTForImageClassification(backbone_name="vits16", model_config=_FOVEATED, n_classes=10,
                                       glimpse_grid_size=8).eval()
    task = In1kRunTask(replace(In1kConfig(), canvas_grid=8), total_steps=1)
    assert task.is_foveated(clf)
    joint = task.build_policy(clf, device=torch.device("cpu"), canvas_grid=8,
                              generator=torch.Generator().manual_seed(0))
    torch.manual_seed(1)
    for p in joint.scorer.parameters():
        p.data.normal_(0, 0.5)
    joint.scorer.eval()
    return PolicyEpisode(task=task, model=clf, joint=joint, device=torch.device("cpu"), amp=False)


def test_non_search_policy_reproduces_its_deploy_rollout_without_a_cue():
    ep = _in1k_episode()
    assert not ep.is_search
    batch = _batch(torch.zeros(B, 3, 2)) | {"cue": torch.full((B,), -1)}  # free viewing: no cue
    scores = []
    hook = ep.joint.scorer.register_forward_hook(lambda m, i, o: scores.append(o.detach().reshape(B, -1)))

    def advance(vp, state, t):  # in1k's own step (in1k/eval.py): the backbone on the whole image
        if state is None:
            state = ep.model.init_state(batch_size=B, canvas_grid_size=8)
        return ep.model.canvit(image=batch["image"], state=state, viewpoint=vp).state

    try:
        taken = deploy_rollout_viewpoints(joint=ep.joint, advance=advance, t0_type=ViewpointType.FULL,
                                          batch_size=B, device=torch.device("cpu"), n=3)
    finally:
        hook.remove()
    batch["fix"] = torch.stack([vp.centers.flip(-1) for vp in taken], 1)
    out = ep.run(batch, n=2)
    assert set(out) == {"policy"}  # no task heatmap outside search
    # the generic policy has a BatchNorm and select_bn_eval=True: the scorer runs twice per selection (identical
    # in deploy's eval mode), so the hook saw 2 outputs per step
    per_select = len(scores) // 2
    assert ep.joint.policy_selector.select_bn_eval and per_select == 2
    for t in range(2):
        assert torch.equal(out["policy"][:, t], scores[t * per_select])
    assert torch.allclose(ep.sample(batch, 3, temperature=0.0), batch["fix"], atol=1e-6)
    with pytest.raises(ValueError, match="only the 'policy' readout"):
        CanViTModel(ep, "heatmap")


# --- self-contained checkpoints (main b029fd3+): the checkpoint alone builds the episode ---------------

from canvit.search.tests.test_search_eval import trained  # noqa: E402,F401 -- trains the tiny search run once


def test_load_builds_the_runs_task_and_policy_from_the_checkpoint_alone(trained):  # noqa: F811
    _, run_dir, _, argv = trained
    ckpt = run_dir / "checkpoints" / "best.pt"
    payload = torch.load(ckpt, map_location="cpu", weights_only=False)
    ep = PolicyEpisode.load(ckpt, device=torch.device("cpu"), need_policy=True)
    assert ep.is_search and ep.task.cfg.head_width == 16 and ep.task.rl.width == 16 and ep.amp is False
    assert ep.provenance["run_config_argv"] == argv and ep.provenance["run_config_drift"] == []

    def same(a, b):  # the joint's state is nested (scorer, encoder, ...)
        if isinstance(a, dict):
            return a.keys() == b.keys() and all(same(a[k], b[k]) for k in a)
        return torch.equal(a, b) if isinstance(a, torch.Tensor) else a == b
    assert same(ep.joint.state_dict(), payload["policy_state"])
    # the loaded episode is still the deploy rollout (forcing its own choices reproduces its scores)
    batch = _batch(torch.zeros(B, 3, 2))
    centres, deploy_scores = _deploy(ep, batch["image"], batch["cue"], n=3)
    batch["fix"] = centres
    forced = ep.run(batch, n=2)["policy"]
    for t in range(2):
        assert torch.equal(forced[:, t], deploy_scores[t])


def test_a_checkpoint_without_run_config_is_refused(tmp_path):
    ckpt = tmp_path / "old.pt"
    torch.save({"model_state": {}, "metadata": {}}, ckpt)
    with pytest.raises(ValueError, match="no run config"):
        PolicyEpisode.load(ckpt, device=torch.device("cpu"), need_policy=True)


# --- free viewing with a search model: the uniform mixture over the cues ------------------------------

def test_cue_mixture_is_the_mean_of_the_per_cue_pixel_densities(fake_cs18):
    from canvit.scanpath.models import _pixel_cells, grid_log_density
    from canvit.search.data import COCO_SEARCH18_CLASSES
    ep = _episode(fake_cs18[0])
    hw = (150, 224)  # a letterboxed stimulus, so some cells are cut and some off the image
    batch = _batch(torch.zeros(B, 3, 2)) | {"cue": torch.full((B,), -1), "hw": torch.tensor([hw] * B)}
    m = CanViTModel(ep, "policy", cue_mixture=True)
    assert m.name == "canvit_search_policy_cuemix"
    mixed = m.forced_logits(batch, 2, CG)  # [B, 2, G*G]
    cell, _ = _pixel_cells(*hw, S, CG)
    for b in range(B):
        for t in range(2):
            dens = [np.exp(grid_log_density(ep.run(batch | {"cue": torch.full((B,), k)}, 2)["policy"][b, t].numpy(),
                                            hw=hw, scene=S)) for k in range(len(COCO_SEARCH18_CLASSES))]
            expected = np.log(np.mean(dens, 0))
            got = mixed[b, t].numpy()
            assert np.allclose(got[cell], expected, atol=1e-5)  # the mixture, per pixel
            assert np.allclose(grid_log_density(got, hw=hw, scene=S), expected, atol=1e-5)  # pysaliency reads it as is
    with pytest.raises(ValueError, match="free-viewing"):
        m.forced_logits(_batch(torch.zeros(B, 3, 2)) | {"hw": torch.tensor([hw] * B)}, 2, CG)


def test_mixture_samples_come_from_the_mixed_density(fake_cs18):
    """Sampling with cue_mixture uses the scored per-step mixture: at temperature 0 the first chosen cell is the
    argmax of the mixed p(cell) after the anchor, and seeded draws land on the image."""
    from canvit.scanpath.models import _pixel_cells
    ep = _episode(fake_cs18[0])
    hw = (150, 224)  # letterboxed: content y in [-0.67, 0.67]
    batch = _batch(torch.zeros(B, 2, 2)) | {"cue": torch.full((B,), -1), "hw": torch.tensor([hw] * B),
                                            "content": torch.tensor([[-1.0, -0.67, 1.0, 0.67]]).repeat(B, 1)}
    mixed = CanViTModel(ep, "policy", cue_mixture=True).forced_logits(batch, 1, CG)[:, 0]  # log pixel density
    _, count = _pixel_cells(*hw, S, CG)
    p = mixed.double().exp() * torch.from_numpy(count)  # p(cell)
    assert torch.allclose(p.sum(-1), torch.ones(B, dtype=torch.float64), atol=1e-6)
    xy = ep.sample(batch, 2, temperature=0.0, cue_mixture=True)
    ij = ((xy[:, 1] + 1) / 2 * CG).floor().long()
    assert torch.equal(ij[:, 1] * CG + ij[:, 0], p.argmax(-1))
    xy = ep.sample(batch, 3, temperature=1.0, generator=torch.Generator().manual_seed(0), cue_mixture=True)
    assert (xy[..., 1].abs() <= 0.67).all()  # clamped onto the image


def test_sharded_predictions_equal_the_unsharded_run(fake_cs18, fake_search, tmp_path):
    """--cfg.shard i N writes one block's predictions; --cfg.predictions merges the files: the same table."""
    from canvit.scanpath.evaluate import ScanpathConfig, predict, shard_indices
    from canvit.scanpath.models import GridScanpathModel
    _, _, sv, fv = fake_search
    cfg = ScanpathConfig(scene_size=S, grid=CG, batch_size=4, num_workers=0)
    m = CanViTModel(_episode(fake_cs18[0]), "policy")
    whole = predict(cfg, m, sv, fv)
    paths = []
    for i in range(3):
        paths.append(tmp_path / f"shard{i}.npz")
        predict(cfg, m, sv, fv, indices=shard_indices(fv, (i, 3))).save(paths[-1], meta={"shard": i})
    merged = GridScanpathModel.load(*paths)
    assert merged.logits.keys() == whole.logits.keys() and merged.meta == {"shard": 0}
    for k, v in whole.logits.items():
        assert np.allclose(merged.logits[k], v, atol=1e-5)  # batch composition differs between the runs
    with pytest.raises(ValueError, match="overlap"):
        GridScanpathModel.load(paths[0], paths[0])
