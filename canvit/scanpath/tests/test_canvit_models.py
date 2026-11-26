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


def _episode(root, *, policy=True, **rl):
    task = _TinyTask(_cfg(root, condition="tp"), rl=_rl(**rl))
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

    try:  # the cue exactly as the search task's own deploy rollout builds it (evaluate)
        taken = deploy_rollout_viewpoints(joint=ep.joint, advance=advance, t0_type=ViewpointType.FULL,
                                          batch_size=B, device=images.device, n=n,
                                          cond=lambda: ep.task._policy_cond(ep.model, cue))
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


BETA_CONDITIONED = {"saccade_cost": "log", "saccade_cost_weight": 1.0, "saccade_cost_beta_mode": "conditioned",
                    "saccade_cost_beta_eval": 0.3}


def test_a_beta_conditioned_policy_reproduces_its_deploy_rollout_at_its_beta(fake_cs18):
    """A beta-conditioned search policy (saccade cost, stage 2): the episode builds the cue through the task's own
    conditioning hook, so forcing its own choices at the run's beta_eval reproduces its deploy rollout, a given beta
    changes the prediction, and a beta for a policy that takes none is refused."""
    ep = _episode(fake_cs18[0], **BETA_CONDITIONED)
    assert ep.resolved_saccade_beta() == 0.3  # default: the run's beta_eval
    batch = _batch(torch.zeros(B, 3, 2))
    centres, deploy_scores = _deploy(ep, batch["image"], batch["cue"], n=3)  # the task's deploy, at beta_eval
    batch["fix"] = centres
    forced = ep.run(batch, n=2)["policy"]
    for t in range(2):
        assert torch.equal(forced[:, t], deploy_scores[t])
    ep.saccade_beta = 0.3
    assert torch.equal(ep.run(batch, n=2)["policy"], forced)  # explicit = default, bit for bit
    ep.saccade_beta = 0.9
    assert not torch.allclose(ep.run(batch, n=2)["policy"], forced)  # beta reaches the scorer
    plain = _episode(fake_cs18[0])
    assert plain.resolved_saccade_beta() is None
    plain.saccade_beta = 0.3
    with pytest.raises(ValueError, match="takes no beta"):
        plain.run(batch, n=2)


def test_skipping_the_heatmap_leaves_the_policy_maps_and_centres_unchanged(fake_cs18):
    ep = _episode(fake_cs18[0])
    batch = _batch(torch.zeros(B, 3, 2))
    human = lambda t, scores: batch["fix"][:, t]  # noqa: E731
    for k in (0, 3):  # one cue, and several cues (the cue mixture's path)
        full, c_full = ep._episode(batch, 2, human, all_cues=k)
        lean, c_lean = ep._episode(batch, 2, human, all_cues=k, heatmap=False)
        assert set(full) == {"policy", "heatmap"} and set(lean) == {"policy"}
        assert torch.equal(lean["policy"], full["policy"]) and torch.equal(c_lean, c_full)
    with pytest.raises(ValueError, match="nothing to read out"):
        _episode(fake_cs18[0], policy=False)._episode(batch, 2, human, heatmap=False)


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


@pytest.mark.slow  # > 10 s on CPU; deselect with -m 'not slow'
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

def _in1k_episode(n_classes: int = 10):
    """A tiny FOVEATED in1k classifier with its policy, built by the task's own constructors."""
    from dataclasses import replace

    from canvit.core import CanViTForImageClassification
    from canvit.in1k.config import In1kConfig
    from canvit.in1k.task import In1kRunTask
    from canvit.search.tests.test_search_task import _FOVEATED
    torch.manual_seed(0)
    clf = CanViTForImageClassification(backbone_name="vits16", model_config=_FOVEATED, n_classes=n_classes,
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


@pytest.mark.slow  # > 10 s on CPU; deselect with -m 'not slow'
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
    # an override reaches the harness and wins over the run config (a single-retina backbone refuses a retina)
    with pytest.raises(ValueError, match="needs a multi-retina backbone"):
        PolicyEpisode.load(ckpt, device=torch.device("cpu"), need_policy=True, overrides=("--cfg.retina", "x"))


def test_a_checkpoint_without_run_config_is_refused(tmp_path):
    ckpt = tmp_path / "old.pt"
    torch.save({"model_state": {}, "metadata": {}}, ckpt)
    with pytest.raises(ValueError, match="no run config"):
        PolicyEpisode.load(ckpt, device=torch.device("cpu"), need_policy=True)


@pytest.mark.slow  # > 10 s on CPU; deselect with -m 'not slow'
def test_a_non_search_checkpoint_loads_and_scores_human_scanpaths(fake_search, tmp_path):
    """A checkpoint as the harness writes it for a non-search task (in1k: no cue, square training frame, the
    policy in ``policy_state``) builds the same episode, and its predictions score in pysaliency."""
    from canvit.harness.cli import command_from_run_config, run_config_of
    from canvit.harness.infra.checkpoint import save_checkpoint
    from canvit.scanpath.canvit_models import precompute
    from canvit.scanpath.data import make_scanpath_loader, parse_display_frame
    from canvit.scanpath.evaluate import training_display_frame
    ep = _in1k_episode(n_classes=1000)  # in1k rebuilds its head with the config's 1000 classes
    argv = ["in1k", "--cfg.canvas-grid", "8", "--cfg.device", "cpu", "--cfg.no-amp"]
    cmd, _ = command_from_run_config({"argv": argv})
    opt = torch.optim.SGD([torch.nn.Parameter(torch.zeros(1))], lr=0.1)
    ckpt = save_checkpoint(tmp_path / "best.pt", model=ep.model, optimizer=opt,
                           scheduler=torch.optim.lr_scheduler.LambdaLR(opt, lambda s: 1.0), step=1, task_name="in1k",
                           spec=ep.task.default_spec(), model_config=ep.task.model_config(ep.model), joint=ep.joint,
                           metadata={"run_config": run_config_of(cmd, argv), "pretrain_view_scale": {
                               "mode": "fixed", "fixed_scale": ep.task.cfg.foveated_scale.fixed_scale}},
                           update_latest=False)
    loaded = PolicyEpisode.load(ckpt, device=torch.device("cpu"), need_policy=True)
    assert loaded.task.name == "in1k" and not loaded.is_search and loaded.amp is False
    batch = _batch(torch.zeros(B, 3, 2)) | {"cue": torch.full((B,), -1)}
    assert torch.equal(loaded.run(batch, n=2)["policy"], ep.run(batch, n=2)["policy"])
    assert training_display_frame(loaded.task)[0] == "scene"
    # human search trials (with cues) through a model that takes none: the cue is ignored, no cue mixture
    from canvit.scanpath.evaluate import select
    _, _, sv, fv = fake_search
    fv = select(fv, condition="both", limit=8)  # target present and absent; keeps the test fast
    scene = loaded.task.cfg.scene_size
    loader = make_scanpath_loader(sv, fv, scene_size=scene, batch_size=4, num_workers=0,
                                  frame=parse_display_frame("scene"))
    model = precompute(CanViTModel(loaded, "policy"), loader, scene_size=scene)  # the policy's own grid (16, not 32)
    assert {v.size for v in model.logits.values()} == {16 * 16}
    with pytest.raises(ValueError, match="predicts on a 16x16 grid but --cfg.grid is 32"):
        precompute(CanViTModel(loaded, "policy"), loader, grid=32, scene_size=scene)  # a given grid is a check
    fx = fv[fv.scanpath_history_length > 0]
    assert set(model.logits) == set(zip(fx.scanpath_index.tolist(), fx.scanpath_history_length.tolist()))
    assert np.isfinite(model.log_likelihoods(sv, fx)).all()


@pytest.mark.slow  # > 10 s on CPU (builds a ViT-S backbone); deselect with -m 'not slow'
@pytest.mark.parametrize("name", ["distill", "ade20k"])
def test_distill_and_ade20k_checkpoints_load_to_the_same_policy(name, tmp_path):
    """A foveated distill / ade20k checkpoint as the harness writes it (model built by the task's own checkpoint path,
    so no teacher or pretrained download) loads to the same policy. distill's model IS the backbone; distill takes
    its backbone from the run config (argv), not from model_config. ~150 MB checkpoint: keep TMPDIR off a small /tmp."""
    from dataclasses import asdict

    from canvit.harness.cli import command_from_run_config, run_config_of
    from canvit.harness.infra.checkpoint import save_checkpoint
    from canvit.scanpath.evaluate import training_display_frame
    from canvit.search.tests.test_search_task import _FOVEATED
    argv = [name] + (["--cfg.backbone-name", "vits16"] if name == "distill" else []) + [
        "--cfg.device", "cpu", "--cfg.run-group", "g", "--cfg.logs-dir", str(tmp_path / "logs")]
    cmd, _ = command_from_run_config({"argv": argv})
    task, _ = cmd.build()
    torch.manual_seed(0)
    if name == "ade20k":
        from canvit.core.model.segmentation import CanViTForSemanticSegmentation
        model = CanViTForSemanticSegmentation(backbone_name="vits16", model_config=_FOVEATED, num_classes=150,
                                              glimpse_grid_size=8)
    else:
        import dacite

        from canvit import CanViTForPretrainingConfig
        mc = dacite.from_dict(CanViTForPretrainingConfig, asdict(task.cfg.model) | _FOVEATED | {"teacher_dim": 64},
                              config=dacite.Config(strict=False))
        model, _ = task.build_model(torch.device("cpu"), prior_model_config={"canvit": asdict(mc)})
    model.eval()
    assert task.is_foveated(model) and training_display_frame(task)[0] == "scene"
    joint = task.build_policy(model, device=torch.device("cpu"), canvas_grid=task.canvas_grid(model),
                              generator=torch.Generator().manual_seed(0))
    torch.manual_seed(1)
    for p in joint.scorer.parameters():
        p.data.normal_(0, 0.5)
    joint.scorer.eval()
    ep = PolicyEpisode(task=task, model=model, joint=joint, device=torch.device("cpu"), amp=False)
    opt = torch.optim.SGD([torch.nn.Parameter(torch.zeros(1))], lr=0.1)
    ckpt = save_checkpoint(tmp_path / "best.pt", model=model, optimizer=opt,
                           scheduler=torch.optim.lr_scheduler.LambdaLR(opt, lambda s: 1.0), step=1, task_name=name,
                           spec=task.default_spec(), model_config=task.model_config(model), joint=joint,
                           metadata={"run_config": run_config_of(cmd, argv), "pretrain_view_scale": {
                               "mode": "fixed", "fixed_scale": task.cfg.foveated_scale.fixed_scale}},
                           update_latest=False)
    loaded = PolicyEpisode.load(ckpt, device=torch.device("cpu"), need_policy=True)
    assert loaded.task.name == name and not loaded.is_search
    batch = _batch(torch.zeros(B, 3, 2)) | {"cue": torch.full((B,), -1)}
    assert torch.equal(loaded.run(batch, n=2)["policy"], ep.run(batch, n=2)["policy"])


# --- free viewing with a search model: the uniform mixture over the cues ------------------------------

@pytest.mark.slow  # > 10 s on CPU; deselect with -m 'not slow'
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
    per_cue = [ep.run(batch | {"cue": torch.full((B,), k)}, 2)["policy"] for k in range(len(COCO_SEARCH18_CLASSES))]
    for b in range(B):
        for t in range(2):
            dens = [np.exp(grid_log_density(pc[b, t].numpy(), hw=hw, scene=S)) for pc in per_cue]
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


@pytest.mark.slow  # > 10 s on CPU; deselect with -m 'not slow'
def test_sharded_predictions_equal_the_unsharded_run(fake_cs18, fake_search, tmp_path):
    """--cfg.shard i N writes one block's predictions; --cfg.predictions merges the files: the same table."""
    from canvit.scanpath.evaluate import ScanpathConfig, predict, select, shard_indices
    from canvit.scanpath.models import GridScanpathModel
    _, _, sv, fv = fake_search
    fv = select(fv, condition="both", limit=12)  # enough for three uneven shards; keeps the test fast
    cfg = ScanpathConfig(scene_size=S, grid=CG, batch_size=4, num_workers=0, display_frame="1680x1050")
    m = CanViTModel(_episode(fake_cs18[0]), "policy")
    whole = predict(cfg, m, sv, fv)
    paths = []
    for i in range(3):
        paths.append(tmp_path / f"shard{i}.npz")
        predict(cfg, m, sv, fv, indices=shard_indices(fv, (i, 3))).save(paths[-1], meta={"shard": i})
    merged = GridScanpathModel.load(*paths)
    assert merged.logits.keys() == whole.logits.keys() and merged.meta == {"shard": 0}
    assert merged.frame == whole.frame == (1050, 1680)
    for k, v in whole.logits.items():
        assert np.allclose(merged.logits[k], v, atol=1e-5)  # batch composition differs between the runs
    with pytest.raises(ValueError, match="overlap"):
        GridScanpathModel.load(paths[0], paths[0])
