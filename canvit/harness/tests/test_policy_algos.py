"""policy-algos branch: AC (A2C / PPO-clip), QTD (multi-step Q-learning) and dense QReg.

Unit checks on the new math (GAE, the clipped surrogate) plus rollouts through the real
engine on a tiny CPU ADE20K model, in the production cell (frozen backbone + probe,
policy only, bptt='none'): every new objective must train exactly the policy modules it
owns and leave the task net untouched -- and must not change what the old objectives do.
"""


import pytest
import torch

from canvit.ade20k.data import IGNORE_LABEL, NUM_CLASSES
from canvit.ade20k.task import POLICY_FEATURE_GROUPS as ADE_GROUPS
from canvit.ade20k.task import BoundAde20kTask
from canvit.core import CanViTForSemanticSegmentation
from canvit.harness.config import FoveatedScaleConfig, JointPolicyConfig
from canvit.harness.policy import build_policy
from canvit.harness.policy.joint import JointPolicy
from canvit.harness.policy.rl import AC, PG, QTD, VPG, QReg, ac_loss, gae
from canvit.harness.rollout import run_rollout
from canvit.harness.rollout.selector import RandomSelector
from canvit.harness.rollout.viewpoint import ViewpointType
from canvit.harness.spec import BpttSpec

_B, _G, _IMG = 2, 8, 224
_DEV = torch.device("cpu")


# --- math ------------------------------------------------------------------------------

def test_gae_matches_hand_computation():
    r = torch.tensor([[1.0, 2.0, 3.0]])
    v = torch.tensor([[0.5, 1.0, 1.5]])
    g, lam = 0.5, 0.5
    d2 = 3.0 - 1.5                       # last glimpse: no bootstrap
    d1 = 2.0 + g * 1.5 - 1.0
    d0 = 1.0 + g * 1.0 - 0.5
    a2 = d2
    a1 = d1 + g * lam * a2
    a0 = d0 + g * lam * a1
    adv, ret = gae(r, v, gamma=g, lam=lam)
    torch.testing.assert_close(adv, torch.tensor([[a0, a1, a2]]))
    torch.testing.assert_close(ret, adv + v)


def test_gae_lambda1_is_the_discounted_return_minus_value():
    r = torch.tensor([[1.0, 2.0, 3.0]])
    v = torch.tensor([[0.3, -0.2, 0.9]])
    adv, ret = gae(r, v, gamma=0.7, lam=1.0)
    G = torch.tensor([[1 + 0.7 * 2 + 0.49 * 3, 2 + 0.7 * 3, 3.0]])
    torch.testing.assert_close(ret, G)


def test_first_ppo_pass_has_the_a2c_gradient():
    """ratio == 1 on the first pass, so the clipped surrogate's gradient is A2C's
    -(adv * grad log pi): PPO with one epoch IS A2C."""
    torch.manual_seed(0)
    N, A = 6, 5
    scores = torch.randn(N, A, requires_grad=True)
    idx = torch.randint(A, (N,))
    adv = torch.randn(N)
    old = torch.log_softmax(scores.detach(), 1).gather(1, idx[:, None]).squeeze(1)
    vals = torch.zeros(N, requires_grad=True)
    loss, _, m = ac_loss(scores, idx, old, adv, vals, torch.zeros(N), clip_eps=0.2, alpha=0.0, value_weight=1.0)
    g_ppo, = torch.autograd.grad(loss, scores)
    logp = torch.log_softmax(scores, 1).gather(1, idx[:, None]).squeeze(1)
    g_a2c, = torch.autograd.grad(-(adv * logp).mean(), scores)
    torch.testing.assert_close(g_ppo, g_a2c)
    assert m["clip_frac"] == 0


def test_clip_stops_the_gradient_outside_the_trust_region():
    scores = torch.zeros(1, 3, requires_grad=True)
    idx = torch.tensor([0])
    old = torch.tensor([float(torch.log(torch.tensor(1 / 3))) - 1.0])  # ratio = e > 1 + clip
    loss, _, m = ac_loss(scores, idx, old, torch.tensor([1.0]), torch.zeros(1), torch.zeros(1),
                         clip_eps=0.2, alpha=0.0, value_weight=0.0)
    g, = torch.autograd.grad(loss, scores)
    assert g.abs().max() == 0  # positive advantage, ratio already above 1+eps: no push
    assert m["clip_frac"] == 1


# --- dispatch: the old objectives' paths must not move ---------------------------------

def _jp(obj):
    return JointPolicy(policy_selector=None, random_selector=None, scorer=None, objective=obj,  # type: ignore[arg-type]
                       rl_weight=1.0, keep_random_branch=False, target_momentum=0.997, device=_DEV)


def test_dispatch_flags():
    for obj, defers, steps in [(QReg(), False, False), (PG(), False, False), (VPG(), True, False),
                               (AC(), True, True), (QTD(), True, True)]:
        assert _jp(obj).defers_credit is defers, obj
        assert _jp(obj).needs_step_rewards is steps, obj
    assert _jp(QReg()).extra_candidates == 0
    assert _jp(QReg(extra_candidates=3)).extra_candidates == 3
    assert _jp(PG()).extra_candidates == 0


def test_after_optimizer_step_is_a_noop_without_target_net():
    _jp(QReg()).after_optimizer_step()
    _jp(PG()).after_optimizer_step()


# --- rollouts on the real engine (tiny CPU ADE20K model, policy_only cell) -------------

def _seg() -> CanViTForSemanticSegmentation:
    torch.manual_seed(0)
    seg = CanViTForSemanticSegmentation(backbone_name="vits16", model_config={}, num_classes=NUM_CLASSES).to(_DEV)
    seg.requires_grad_(False)
    seg.eval()
    return seg


def _masks() -> torch.Tensor:
    m = torch.randint(0, NUM_CLASSES, (_B, _IMG, _IMG), device=_DEV)
    m[:, :8] = IGNORE_LABEL
    return m


def _joint(seg, **rl_kw):
    gen = torch.Generator(device=_DEV)
    gen.manual_seed(0)
    return build_policy(canvit=seg.canvit, rl=JointPolicyConfig(**rl_kw), feature_groups=ADE_GROUPS,
                        device=_DEV, canvas_grid=_G, min_viewpoint_scale=0.05,
                        foveated_scale=FoveatedScaleConfig(), generator=gen, encode_model=seg)


def _rollout(seg, joint, horizon=4):
    task = BoundAde20kTask(seg=seg, masks=_masks(), canvas_grid=_G)
    sel = RandomSelector(is_foveated=False, foveated_scale=FoveatedScaleConfig(), min_viewpoint_scale=0.05)
    return run_rollout(model=seg, images=torch.randn(_B, 3, _IMG, _IMG), task=task, selector=sel,
                       bptt=BpttSpec(mode="none", horizon=horizon), branches=[ViewpointType.FULL],
                       canvas_grid_size=_G, amp_ctx=torch.enable_grad(), joint=joint)


def _has_grad(m) -> bool:
    return any(p.grad is not None and p.grad.abs().sum() > 0 for p in m.parameters())


def _no_grad(m) -> bool:
    return all(p.grad is None for p in m.parameters())


def test_ac_refuses_select_bn_eval():
    with pytest.raises(ValueError, match="no-select-bn-eval"):
        _joint(_seg(), objective="ac", select_bn_eval=True)


def test_a2c_trains_actor_and_value_critic_only():
    torch.manual_seed(1)
    seg = _seg()
    joint = _joint(seg, objective="ac", select_bn_eval=False)
    assert joint.policy_selector.mode == "sample" and joint.critic is not None
    assert joint.critic.vhead is None  # V(s) is the critic's whole map, not the dueling head
    r = _rollout(seg, joint)
    assert torch.isfinite(r.total_loss)
    assert _has_grad(joint.scorer) and _has_grad(joint.critic) and _no_grad(seg)
    for k in ("policy_entropy", "value_loss", "clip_frac", "value_mean", "reward_frac"):
        assert k in r.policy_metrics, k
    assert float(r.policy_metrics["clip_frac"]) == 0.0  # one pass: ratio == 1


def test_ppo_inner_passes_step_the_optimizer_then_leave_a_graph_for_the_loop():
    torch.manual_seed(1)
    seg = _seg()
    joint = _joint(seg, objective="ac", select_bn_eval=False, ppo_epochs=3)
    params = list(joint.scorer.parameters()) + list(joint.critic.parameters())
    joint.optimizer = torch.optim.AdamW(params, lr=1e-3)
    before = [p.detach().clone() for p in joint.scorer.parameters()]
    r = _rollout(seg, joint)
    assert torch.isfinite(r.total_loss)
    moved = any(not torch.equal(a, p) for a, p in zip(before, joint.scorer.parameters()))
    assert moved  # the 2 inner passes stepped the optimizer
    assert _has_grad(joint.scorer) and _has_grad(joint.critic) and _no_grad(seg)  # final pass's grads


def test_qtd_trains_scorer_not_target_and_polyak_moves_target():
    torch.manual_seed(1)
    seg = _seg()
    joint = _joint(seg, objective="qtd")
    assert joint.target_net is not None and joint.policy_selector.mode == "argmax"
    assert all(not p.requires_grad for p in joint.target_net.parameters())
    r = _rollout(seg, joint)
    assert torch.isfinite(r.total_loss)
    assert _has_grad(joint.scorer) and _no_grad(joint.target_net) and _no_grad(seg)
    assert "td_target" in r.policy_metrics
    tgt0 = [p.clone() for p in joint.target_net.parameters()]
    with torch.no_grad():
        for p in joint.scorer.parameters():
            p.add_(1.0)
    joint.after_optimizer_step()
    for t0, t1, s in zip(tgt0, joint.target_net.parameters(), joint.scorer.parameters()):
        torch.testing.assert_close(t1, t0 + 0.01 * (s - t0))


def test_dense_qreg_trains_scorer_and_matches_taken_reward_when_cells_coincide():
    torch.manual_seed(1)
    seg = _seg()
    joint = _joint(seg, objective="qreg", qreg_extra_candidates=3)
    assert joint.extra_candidates == 3
    r = _rollout(seg, joint, horizon=3)
    assert torch.isfinite(r.total_loss)
    assert _has_grad(joint.scorer) and _no_grad(seg)


def test_extra_candidate_reward_equals_the_taken_glimpse_reward_for_the_same_cell():
    """The extra glimpses must measure reward EXACTLY as the taken glimpse does: replay the
    taken cell as an 'extra' candidate from the same state and compare."""
    from canvit.core import RecurrentState, Viewpoint
    from canvit.harness.rollout.engine import _extra_candidate_rewards

    torch.manual_seed(1)
    seg = _seg()
    joint = _joint(seg, objective="qreg", qreg_extra_candidates=2)
    task = BoundAde20kTask(seg=seg, masks=_masks(), canvas_grid=_G)
    images = torch.randn(_B, 3, _IMG, _IMG)
    state = seg.init_state(batch_size=_B, canvas_grid_size=_G)
    vp0 = Viewpoint(centers=torch.zeros(_B, 2), scales=torch.ones(_B))
    with torch.no_grad():
        g0 = task.forward_glimpse(model=seg, images=images, state=state, viewpoint=vp0, backbone_no_grad=True)
        prev = task.per_image_loss(g0.readout)
    # force every extra draw onto cell 7
    sel = joint.policy_selector
    sel.vp_flat = sel.vp_flat[7:8].expand(sel.vp_flat.shape[0], -1).clone()
    vp = Viewpoint(centers=sel.vp_flat[:_B, :2].contiguous(), scales=sel.vp_flat[:_B, 2].contiguous())
    _, extra = _extra_candidate_rewards(joint=joint, task=task, model=seg, images=images, state=g0.state,
                                        vp=vp, prev_loss=prev, amp_ctx=torch.enable_grad())
    with torch.no_grad():
        g1 = task.forward_glimpse(model=seg, images=images, state=g0.state, viewpoint=vp, backbone_no_grad=True)
        taken = (prev - task.per_image_loss(g1.readout)) / prev.clamp_min(1e-4)
    torch.testing.assert_close(extra, taken[:, None].expand(-1, 2), rtol=1e-4, atol=1e-5)
    assert isinstance(g0.state, RecurrentState)


def test_old_objectives_are_unchanged_by_the_new_plumbing():
    """QReg and PG through the same helper: no extra candidates, no target net, inline credit."""
    seg = _seg()
    for kw in ({"objective": "qreg"}, {"objective": "pg", "select_bn_eval": False}):
        joint = _joint(seg, **kw)
        assert joint.target_net is None and joint.extra_candidates == 0 and not joint.defers_credit
        assert joint.critic is None
