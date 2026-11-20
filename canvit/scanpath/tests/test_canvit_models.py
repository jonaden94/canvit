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
            "content": torch.tensor([[-1.0, -1.0, 1.0, 1.0]]).repeat(B, 1)}  # no letterbox: no clamping


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
