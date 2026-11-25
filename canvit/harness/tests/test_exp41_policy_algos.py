"""exp41 (2026-11-24): policy learning on a FROZEN model, with four learning rules on two policy
architectures (a U-Net on the canvas, "P2", and isolated policy tokens, "T3-iso"):
  * PG (unchanged);
  * QReg, now also for policy tokens: their readout gives unbounded per-cell VALUES
    (``PolicyTokenReadout.values``), since its log-probabilities cannot be regressed on a reward;
  * dense QReg (K extra random cells glimpsed per policy state, ported from policy-algos 8f14161);
  * SAC-Discrete (ported from policy-algos 1a0bb3b) with twin canvas U-Net critics for both
    actors.
Also: on a frozen model (bptt 'none' runs the model forward under no_grad) the isolated policy
stream gets no gradient; run() switches such runs to 'full' (tested here at the engine level).
"""

import math

import pytest
import torch
import torch.nn.functional as F

from canvit.distill.loss import DistillTask
from canvit.distill.task import POLICY_FEATURE_GROUPS, BoundDistillTask
from canvit.harness.config import FoveatedScaleConfig, JointPolicyConfig
from canvit.harness.policy import build_policy
from canvit.harness.policy.joint import JointPolicy, TwinCritic
from canvit.harness.policy.rl import PG, SAC, VPG, QReg, sac_loss
from canvit.harness.rollout import run_rollout
from canvit.harness.rollout.selector import RandomSelector, TokenPolicySelector
from canvit.harness.rollout.viewpoint import ViewpointType
from canvit.harness.spec import BpttSpec
from canvit.harness.tests.test_policy_tokens import _B, _D, _DEV, _G, _IMG, _K, _model


def _task() -> BoundDistillTask:
    return BoundDistillTask(DistillTask(
        scene_target=torch.randn(_B, _G * _G, _D), cls_target=torch.randn(_B, _D),
        enable_scene_patches_loss=True, enable_scene_cls_loss=True))


def _build(model, **rl_kw) -> JointPolicy:
    gen = torch.Generator(device=_DEV).manual_seed(0)
    kw = dict(feature_groups=("ln_feat",), centers_per_axis=_G, select_bn_eval=False) | rl_kw
    return build_policy(canvit=model, rl=JointPolicyConfig(**kw), feature_groups=POLICY_FEATURE_GROUPS,
                        device=_DEV, canvas_grid=_G, min_viewpoint_scale=0.05,
                        foveated_scale=FoveatedScaleConfig(), generator=gen, scorer_input_norm="none")


def _frozen(tokens: bool):
    m = _model(n_policy_tokens=_K, policy_token_mode="isolated") if tokens else _model()
    m.requires_grad_(False)
    return m


def _roll(model, joint, mode="full", horizon=4):
    joint.set_trainable(True)
    return run_rollout(
        model=model, images=torch.randn(_B, 3, _IMG, _IMG), task=_task(),
        selector=RandomSelector(is_foveated=False, foveated_scale=FoveatedScaleConfig(), min_viewpoint_scale=0.05),
        bptt=BpttSpec(mode=mode, horizon=horizon), branches=[ViewpointType.FULL],
        canvas_grid_size=_G, amp_ctx=torch.enable_grad(), joint=joint)


def _has_grad(params) -> bool:
    return any(p.grad is not None and bool(p.grad.abs().sum() > 0) for p in params)


def _no_grad(params) -> bool:
    return all(p.grad is None or bool(p.grad.abs().sum() == 0) for p in params)


# --- the token readout's values -----------------------------------------------------------------

def test_token_values_are_the_weighted_raw_scores_and_log_probs_are_unchanged():
    torch.manual_seed(0)
    m = _model(n_policy_tokens=_K, policy_token_mode="isolated")
    ro = m.policy_readout
    with torch.no_grad():
        ro.bias.normal_()
        ro.weight_proj.weight.normal_()
    tok, can = torch.randn(_B, _K, ro.token_norm.normalized_shape[0]), torch.randn(_B, _G * _G, ro.canvas_norm.normalized_shape[0])
    s, log_w = ro._scores(tok, can)
    logp = ro(tok, can)
    ref = torch.logsumexp(F.log_softmax(s, -1) + log_w, dim=(1, 2))  # the pre-refactor expression
    torch.testing.assert_close(logp, ref)
    torch.testing.assert_close(logp.exp().sum(1), torch.ones(_B))
    v = ro.values(tok, can)
    torch.testing.assert_close(v, (log_w.exp() * s).sum(dim=(1, 2)))
    assert bool((v > 0).any()) and bool((v < 0).any())  # unbounded, unlike log p <= 0


def test_token_qreg_selects_by_values_and_regresses_them():
    m = _frozen(tokens=True)
    joint = _build(m, objective="qreg")
    sel = joint.policy_selector
    assert isinstance(sel, TokenPolicySelector) and sel.values and sel.mode == "argmax"
    r = _roll(m, joint)
    assert torch.isfinite(r.policy_metrics["policy_loss"])
    assert _has_grad(m.policy_readout.parameters()) and _has_grad(m.policy_stream.parameters())
    trunk = [p for n, p in m.named_parameters() if not n.startswith(("policy_",))]
    assert _no_grad(trunk)


def test_vpg_still_refused_for_policy_tokens():
    with pytest.raises(ValueError, match="not vpg"):
        _build(_model(n_policy_tokens=_K, policy_token_mode="isolated"), objective="vpg")


# --- frozen model: the isolated stream needs grad on ------------------------------------------

def test_frozen_isolated_stream_gets_gradient_only_without_no_grad():
    m = _frozen(tokens=True)
    joint = _build(m, objective="pg", entropy_target=4.0)
    _roll(m, joint, mode="none")
    assert _has_grad(m.policy_readout.parameters())
    assert _no_grad(m.policy_stream.parameters())  # the bug run() avoids: forward under no_grad
    for p in joint.policy_parameters():
        p.grad = None
    _roll(m, joint, mode="full")
    assert _has_grad(m.policy_stream.parameters()) and _has_grad(m.policy_readout.parameters())
    trunk = [p for n, p in m.named_parameters() if not n.startswith(("policy_",))]
    assert _no_grad(trunk)


# --- dense QReg --------------------------------------------------------------------------------

@pytest.mark.parametrize("tokens", [False, True])
def test_dense_qreg_rolls_out_and_trains_the_policy_only(tokens):
    m = _frozen(tokens)
    joint = _build(m, objective="qreg", qreg_extra_candidates=3)
    assert joint.extra_candidates == 3
    r = _roll(m, joint)
    assert torch.isfinite(r.policy_metrics["policy_loss"])
    assert _has_grad(joint.policy_parameters())
    trunk = [p for n, p in m.named_parameters() if not n.startswith(("policy_",))]
    assert _no_grad(trunk)


@pytest.mark.parametrize("tokens", [False, True])
def test_extra_candidate_reward_equals_the_taken_glimpse_reward_for_the_same_cell(tokens):
    """The extra glimpses must measure reward EXACTLY as the taken glimpse does: force every
    extra draw onto one cell and compare with taking that cell."""
    from canvit.core import Viewpoint
    from canvit.harness.rollout.engine import _extra_candidate_rewards

    m = _frozen(tokens)
    joint = _build(m, objective="qreg", qreg_extra_candidates=2)
    task, images = _task(), torch.randn(_B, 3, _IMG, _IMG)
    state = m.init_state(batch_size=_B, canvas_grid_size=_G)
    vp0 = Viewpoint(centers=torch.zeros(_B, 2), scales=torch.ones(_B))
    with torch.no_grad():
        g0 = task.forward_glimpse(model=m, images=images, state=state, viewpoint=vp0, backbone_no_grad=True)
        prev = task.per_image_loss(g0.readout)
    sel = joint.policy_selector
    sel.vp_flat = sel.vp_flat[7:8].expand(sel.vp_flat.shape[0], -1).clone()
    vp = Viewpoint(centers=sel.vp_flat[:_B, :2].contiguous(), scales=torch.full((_B,), 0.5))
    _, extra = _extra_candidate_rewards(joint=joint, task=task, model=m, images=images, state=g0.state,
                                        vp=vp, prev_loss=prev, amp_ctx=torch.enable_grad())
    with torch.no_grad():
        g1 = task.forward_glimpse(model=m, images=images, state=g0.state, viewpoint=vp, backbone_no_grad=True)
        taken = (prev - task.per_image_loss(g1.readout)) / prev.clamp_min(1e-4)
    torch.testing.assert_close(extra, taken[:, None].expand(-1, 2), rtol=1e-4, atol=1e-5)


def test_dispatch_flags():
    def jp(obj):
        return JointPolicy(policy_selector=None, random_selector=None, scorer=None, objective=obj,  # type: ignore[arg-type]
                           rl_weight=1.0, keep_random_branch=False, target_momentum=0.997, device=_DEV)
    for obj, defers, steps in [(QReg(), False, False), (PG(), False, False), (VPG(), True, False),
                               (SAC(), True, True)]:
        assert jp(obj).defers_credit is defers, obj
        assert jp(obj).needs_step_rewards is steps, obj
    assert jp(QReg(extra_candidates=3)).extra_candidates == 3 and jp(PG()).extra_candidates == 0
    jp(PG()).after_optimizer_step()  # no target net: a no-op


# --- SAC-Discrete ------------------------------------------------------------------------------

def test_sac_loss_matches_hand_computation():
    """Soft Bellman target with the exact expectation (eq. 10), the twin min, and the
    expected actor objective (eq. 12) on a 1-episode, 2-glimpse, 2-cell example."""
    scores = torch.tensor([[[0.0, 1.0], [2.0, 0.0]]], requires_grad=True)
    q1 = torch.tensor([[[1.0, 3.0], [0.5, 2.0]]], requires_grad=True)
    q2 = torch.tensor([[[2.0, 2.0], [1.0, 1.0]]], requires_grad=True)
    t1, t2 = torch.tensor([[[4.0, 1.0]]]), torch.tensor([[[3.0, 2.0]]])
    idx, r = torch.tensor([[1, 0]]), torch.tensor([[0.5, 0.25]])
    alpha, gamma = 0.3, 0.9
    loss, _, m = sac_loss(scores, q1, q2, t1, t2, idx, r, alpha=alpha, gamma=gamma)
    p1 = torch.softmax(torch.tensor([2.0, 0.0]), 0)
    v1 = (p1 * (torch.tensor([3.0, 1.0]) - alpha * p1.log())).sum()
    y = torch.tensor([0.5 + gamma * v1, 0.25])
    critic = 0.5 * (((torch.tensor([3.0, 0.5]) - y) ** 2).mean() + ((torch.tensor([2.0, 1.0]) - y) ** 2).mean())
    lp = torch.log_softmax(scores.detach(), 2)
    qmin = torch.min(q1, q2).detach()
    actor = (lp.exp() * (alpha * lp - qmin)).sum(2).mean()
    torch.testing.assert_close(loss, actor + critic)
    torch.testing.assert_close(m["td_target"], y.mean())
    g_s, g_q1 = torch.autograd.grad(loss, (scores, q1))
    s2 = scores.detach().clone().requires_grad_(True)
    lp2 = torch.log_softmax(s2, 2)
    g_ref, = torch.autograd.grad((lp2.exp() * (alpha * lp2 - qmin)).sum(2).mean(), s2)
    torch.testing.assert_close(g_s, g_ref)
    assert g_q1[0, 0, 0] == 0 and g_q1[0, 1, 1] == 0  # only the taken cells are regressed


@pytest.mark.parametrize("tokens", [False, True])
def test_sac_trains_actor_and_critics_not_target_or_model(tokens):
    m = _frozen(tokens)
    joint = _build(m, objective="sac", sac_target_entropy=4.0)
    assert isinstance(joint.objective, SAC) and joint.policy_selector.mode == "sample"
    assert isinstance(joint.critic, TwinCritic) and joint.critic.q1.vhead is None
    assert joint.objective.target_entropy == 4.0
    crit = list(joint.critic.parameters())
    assert all(any(c is p for p in joint.policy_parameters()) for c in crit)
    assert all(not p.requires_grad for p in joint.target_net.parameters())
    a0 = float(joint.log_alpha)
    r = _roll(m, joint)
    assert torch.isfinite(r.total_loss)
    assert _has_grad(joint.scorer.parameters()) and _has_grad(joint.critic.q1.parameters())
    assert _has_grad(joint.critic.q2.parameters())
    assert all(p.grad is None for p in joint.target_net.parameters())
    trunk = [p for n, p in m.named_parameters() if not n.startswith(("policy_",))]
    assert _no_grad(trunk)
    for k in ("policy_entropy", "critic_loss", "actor_loss", "alpha", "td_target", "reward_frac"):
        assert k in r.policy_metrics, k
    assert float(joint.log_alpha) != a0  # the temperature took its dual step


def test_sac_default_target_entropy_is_the_papers_over_the_actors_cells():
    joint = _build(_frozen(tokens=True), objective="sac")
    assert math.isclose(joint.objective.target_entropy, 0.98 * math.log(_G * _G))


def test_sac_token_critics_need_the_tokens_grid():
    with pytest.raises(ValueError, match="centers-per-axis"):
        _build(_frozen(tokens=True), objective="sac", centers_per_axis=_G // 2)


def test_sac_polyak_and_sidecar_roundtrip():
    m = _frozen(tokens=False)
    a = _build(m, objective="sac")
    tgt0 = [p.clone() for p in a.target_net.parameters()]
    with torch.no_grad():
        for p in a.critic.parameters():
            p.add_(1.0)
    a.after_optimizer_step()
    for t0, t1, c in zip(tgt0, a.target_net.parameters(), a.critic.parameters()):
        torch.testing.assert_close(t1, t0 + 0.005 * (c - t0))
    a.log_alpha.fill_(-1.25)
    b = _build(m, objective="sac")
    b.load_state_dict(a.state_dict())
    for x, y in zip(list(a.critic.parameters()) + list(a.target_net.parameters()),
                    list(b.critic.parameters()) + list(b.target_net.parameters())):
        assert torch.equal(x, y)
    assert float(b.log_alpha) == -1.25


def test_distill_policy_group_holds_actor_and_critics():
    from types import SimpleNamespace

    from canvit.distill.task import DistillRunTask
    m = _frozen(tokens=True)
    joint = _build(m, objective="sac")
    spec = SimpleNamespace(train_backbone=False, train_policy=True)
    groups = DistillRunTask.trainable_param_groups(None, model=m, head=None, joint=joint, spec=spec)
    assert {id(p) for p in groups["policy"]} == {id(p) for p in joint.policy_parameters()}
    assert len(groups["policy"]) == len(list(joint.scorer.parameters())) + len(list(joint.critic.parameters()))


# --- distill's own trajectory lengths (chunked TBPTT, stochastic length) --------------------------

def test_policy_only_on_distill_keeps_the_tasks_sampled_lengths_and_chunking():
    from canvit.distill.config import Config as DistillConfig
    from canvit.distill.task import DistillRunTask
    from canvit.harness.cli import resolve_spec
    task = DistillRunTask(DistillConfig())
    spec = resolve_spec(task, "policy_only", lr=4e-4, wd=1e-4)
    tb = task.default_spec().bptt
    assert spec.bptt == tb and spec.bptt.mode == "chunked" and spec.bptt.stochastic
    assert not spec.train_backbone and spec.task_weight == 0.0 and spec.train_policy


@pytest.mark.parametrize("tokens", [False, True])
@pytest.mark.parametrize("rl_kw", [{"objective": "pg", "entropy_target": 4.0}, {"objective": "qreg"},
                                   {"objective": "qreg", "qreg_extra_candidates": 2},
                                   {"objective": "sac", "sac_target_entropy": 4.0}])
def test_every_objective_under_distills_chunked_tbptt(tokens, rl_kw):
    import random
    m = _frozen(tokens)
    joint = _build(m, **rl_kw)
    joint.set_trainable(True)
    r = run_rollout(
        model=m, images=torch.randn(_B, 3, _IMG, _IMG), task=_task(),
        selector=RandomSelector(is_foveated=False, foveated_scale=FoveatedScaleConfig(), min_viewpoint_scale=0.05),
        bptt=BpttSpec(mode="chunked", chunk_size=2, continue_prob=0.9), branches=[ViewpointType.FULL],
        canvas_grid_size=_G, amp_ctx=torch.enable_grad(), joint=joint, rng=random.Random(3))
    assert r.n_glimpses >= 4  # several chunks
    assert torch.isfinite(r.policy_metrics["policy_loss"])
    assert _has_grad(joint.scorer.parameters())
    if tokens:  # the stream learns from the decisions inside a chunk
        assert _has_grad(m.policy_stream.parameters())
    if joint.critic is not None:
        assert _has_grad(joint.critic.parameters())
        assert all(p.grad is None for p in joint.target_net.parameters())
    trunk = [p for n, p in m.named_parameters() if not n.startswith(("policy_",))]
    assert _no_grad(trunk)


def test_inline_sac_critic_terms_equal_the_episode_critic_loss():
    """The engine builds the critic term of step t one glimpse late (with s_{t+1}); their mean
    over the steps must equal rl.sac_critic_loss on the whole episode (gamma > 0)."""
    from canvit.harness.policy.rl import sac_critic_loss
    torch.manual_seed(0)
    joint = _build(_frozen(tokens=False), objective="sac", sac_gamma=0.9)
    with torch.no_grad():  # targets != critics
        for p in joint.target_net.parameters():
            p.add_(0.01 * torch.randn_like(p))
    T, A = 3, joint.policy_selector.vp_flat.shape[0]
    m = _frozen(tokens=False)
    st = m.init_state(batch_size=_B, canvas_grid_size=_G)
    feats = []
    for _ in range(T):
        st.canvas.normal_()
        feats.append(joint.critic_feats(st))
    idx = torch.randint(A, (_B, T))
    r = torch.randn(_B, T)
    logp = torch.log_softmax(torch.randn(_B, T, A), dim=2)
    terms = [joint.sac_critic_term(feats=feats[t], flat_idx=idx[:, t], r=r[:, t],
                                   next_feats=feats[t + 1] if t + 1 < T else None,
                                   next_logp=logp[:, t + 1] if t + 1 < T else None) for t in range(T)]
    q1, q2 = joint.critic(torch.cat(feats, 0))
    t1, t2 = joint.target_net(torch.cat(feats[1:], 0))
    bta = lambda q, n: q.reshape(n, _B, A).transpose(0, 1)  # noqa: E731
    ref, _ = sac_critic_loss(logp, bta(q1, T), bta(q2, T), bta(t1, T - 1), bta(t2, T - 1), idx, r,
                             alpha=joint.log_alpha.exp(), gamma=0.9)
    torch.testing.assert_close(torch.stack(terms).mean(), ref, rtol=1e-5, atol=1e-6)
