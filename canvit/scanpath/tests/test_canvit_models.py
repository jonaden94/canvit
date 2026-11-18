"""The teacher-forced CanViT search episode against the deploy rollout (design §7), on the search
tests' tiny random foveated backbone (CPU)."""

import numpy as np
import pytest
import torch

from canvit.harness.rollout.eval_viewpoints import deploy_rollout_viewpoints
from canvit.harness.rollout.viewpoint import ViewpointType
from canvit.scanpath.canvit_models import CanViTSearchModel, SearchEpisode
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
    return SearchEpisode(task=task, model=model, joint=joint, device=torch.device("cpu"), amp=False)


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
    model = precompute(CanViTSearchModel(ep, "policy"), loader, grid=CG, scene_size=S)
    fx = fv[fv.scanpath_history_length > 0]
    keys = set(zip(fx.scanpath_index.tolist(), fx.scanpath_history_length.tolist()))
    assert set(model.logits) == keys
    ll = model.log_likelihoods(sv, fx)
    assert np.isfinite(ll).all() and ll.shape == (len(fx),)


def test_heatmap_readout_needs_no_policy(fake_cs18):
    ep = _episode(fake_cs18[0], policy=False)
    out = CanViTSearchModel(ep, "heatmap").forced_logits(_batch(torch.zeros(B, 3, 2)), 3, CG)
    assert out.shape == (B, 3, CG * CG) and torch.isfinite(out).all()
    with pytest.raises(ValueError, match="grid"):
        CanViTSearchModel(ep, "heatmap").forced_logits(_batch(torch.zeros(B, 3, 2)), 3, 16)
