"""The reward / centring / scaling / entropy options added for visual search (2026-11-05),
and the scorer's ``input_norm`` and FiLM conditioning.

Every option's ``default`` must reproduce the historical behaviour; that half is pinned by
``test_task_digests.py`` (``policy_*`` digests). This file checks the new behaviour itself
against hand-computed values, and that incoherent combinations are refused.
"""

from contextlib import nullcontext

import pytest
import torch

from canvit.harness.config import FoveatedScaleConfig, JointPolicyConfig
from canvit.harness.policy import build_policy
from canvit.harness.policy.rl import (
    PG,
    VPG,
    QReg,
    RewardSpec,
    RunningNorm,
    entropy_floor_step,
    resolve_reward,
    standardize_reward,
)


# --------------------------------------------------------------------------- #
# resolve_reward: defaults and refusals
# --------------------------------------------------------------------------- #
def test_defaults_resolve_to_the_historical_reward():
    hist = RewardSpec("fractional", "running_mean", "running_std")
    assert resolve_reward(objective=QReg()) == hist
    assert resolve_reward(objective=PG()) == hist
    assert resolve_reward(objective=PG(z_subtract_only=True)) == RewardSpec("fractional", "running_mean", "none")
    assert resolve_reward(objective=VPG()).kind == "terminal"


def test_foavis_and_search_combinations_resolve():
    assert resolve_reward(objective=PG(), reward="difference", center="batch_mean", scale="batch_std") == \
        RewardSpec("difference", "batch_mean", "batch_std")
    assert resolve_reward(objective=PG(), reward="terminal", center="batch_mean", scale="batch_std") == \
        RewardSpec("terminal", "batch_mean", "batch_std")
    assert resolve_reward(objective=PG(), center="none", scale="ema_abs").scale == "ema_abs"


@pytest.mark.parametrize("objective, kw, match", [
    (QReg(), {"reward": "terminal"}, "IMMEDIATE"),
    (VPG(), {"reward": "difference"}, "TERMINAL"),
    (VPG(), {"center": "batch_mean"}, "owns its baseline"),
    (PG(), {"center": "batch_mean", "scale": "running_std"}, "running mean"),
])
def test_incoherent_combinations_are_refused(objective, kw, match):
    with pytest.raises(ValueError, match=match):
        resolve_reward(objective=objective, **kw)


# --------------------------------------------------------------------------- #
# standardize_reward
# --------------------------------------------------------------------------- #
def _r(*v):
    return torch.tensor(v, dtype=torch.float32)


def test_running_norm_path_is_exactly_runningnorm():
    a, b = RunningNorm(momentum=0.9, device=torch.device("cpu")), RunningNorm(momentum=0.9, device=torch.device("cpu"))
    spec = RewardSpec("fractional", "running_mean", "running_std")
    for r in (_r(0.1, 0.3, -0.2), _r(0.5, 0.0, 0.2)):
        got, _ = standardize_reward(r, spec=spec, running=a, ema_abs=None)
        assert torch.equal(got, b.normalize(r))


def test_batch_mean_batch_std_is_foavis_per_step():
    r = _r(1.0, 2.0, 4.0, 7.0)
    got, _ = standardize_reward(r, spec=RewardSpec("difference", "batch_mean", "batch_std"),
                                running=RunningNorm(momentum=0.9, device=torch.device("cpu")), ema_abs=None)
    want = (r - r.mean()) / r.std(unbiased=False)
    assert torch.allclose(got, want)


def test_batch_std_leaves_an_all_equal_batch_unscaled():
    r = _r(0.3, 0.3, 0.3)
    for center in ("batch_mean", "none"):
        got, _ = standardize_reward(r, spec=RewardSpec("terminal", center, "batch_std"),
                                    running=RunningNorm(momentum=0.9, device=torch.device("cpu")), ema_abs=None)
        assert torch.isfinite(got).all()
        assert torch.allclose(got, r - r.mean() if center == "batch_mean" else r)


def test_ema_abs_is_haozes_normalizer():
    spec = RewardSpec("terminal", "none", "ema_abs")
    rn = RunningNorm(momentum=0.9, device=torch.device("cpu"))
    r1, r2 = _r(-1.0, -3.0), _r(-2.0, -100.0)
    got1, ema = standardize_reward(r1, spec=spec, running=rn, ema_abs=None)
    assert torch.allclose(got1, r1 / 2.0)  # first batch: scale = its own mean |r|
    got2, ema2 = standardize_reward(r2, spec=spec, running=rn, ema_abs=ema)
    want_scale = 0.99 * 2.0 + 0.01 * 51.0
    assert torch.allclose(ema2, torch.tensor(want_scale))
    assert torch.allclose(got2, (r2 / want_scale).clamp(-10, 10))
    assert got2.min() == -10.0  # the -100 is clipped


# --------------------------------------------------------------------------- #
# entropy dual step: the new upper clip
# --------------------------------------------------------------------------- #
def test_entropy_floor_step_respects_alpha_max():
    la = torch.tensor(0.0)
    entropy_floor_step(log_alpha=la, entropy=torch.tensor(0.0), target=100.0, alpha_lr=1.0,
                       alpha_min=1e-6, alpha_max=0.1)
    assert torch.isclose(la.exp(), torch.tensor(0.1))


# --------------------------------------------------------------------------- #
# build_policy refusals + the scorer options, on a tiny foveated model
# --------------------------------------------------------------------------- #
_G = 8


def _fov_seg():
    from canvit.harness.tests.test_task_digests import _fov_seg as make
    return make()


def _build(seg, rl, **kw):
    gen = torch.Generator()
    gen.manual_seed(0)
    return build_policy(
        canvit=seg.canvit, rl=rl, feature_groups=("ln_feat",), device=torch.device("cpu"),
        canvas_grid=_G, min_viewpoint_scale=0.1,
        foveated_scale=FoveatedScaleConfig(mode="fixed", fixed_scale=2.0), generator=gen,
        encode_model=seg, **kw)


@pytest.fixture(scope="module")
def seg():
    s = _fov_seg()
    s.requires_grad_(False)
    s.eval()
    return s


def test_pg_with_batchnorm_and_select_bn_eval_is_refused(seg):
    with pytest.raises(ValueError, match="off-policy"):
        _build(seg, JointPolicyConfig(objective="pg", select_bn_eval=True))
    # without the BatchNorm the two forwards coincide, so it is allowed
    _build(seg, JointPolicyConfig(objective="pg", select_bn_eval=True), scorer_input_norm="none")


def test_vpg_with_film_is_refused(seg):
    with pytest.raises(ValueError, match="UNCONDITIONED"):
        _build(seg, JointPolicyConfig(objective="vpg", select_bn_eval=False), film_cond_dim=16)


@pytest.mark.parametrize("rl", [
    JointPolicyConfig(objective="qreg", entropy_alpha_max=0.1),
    JointPolicyConfig(objective="vpg", select_bn_eval=False, entropy_update="per_rollout"),
    JointPolicyConfig(objective="pg", select_bn_eval=False, entropy_target=None, entropy_alpha_min=1e-6),
])
def test_entropy_knobs_without_a_dual_are_refused(seg, rl):
    with pytest.raises(ValueError, match="entropy dual ascent"):
        _build(seg, rl)


def test_alpha_init_sets_log_alpha(seg):
    jp = _build(seg, JointPolicyConfig(objective="pg", select_bn_eval=False, entropy_alpha_init=1e-3))
    assert torch.isclose(jp.log_alpha.exp(), torch.tensor(1e-3))


def test_input_norm_none_has_no_train_eval_difference(seg):
    jp = _build(seg, JointPolicyConfig(objective="pg", select_bn_eval=False, width=16, block_layers=1,
                                       centers_per_axis=32), scorer_input_norm="none")
    net = jp.scorer
    assert not any(isinstance(m, torch.nn.modules.batchnorm._BatchNorm) for m in net.modules())
    x = torch.randn(3, seg.canvit.canvas_dim, 32, 32)
    net.train()
    a = net(x)
    net.eval()
    assert torch.equal(a, net(x))


def test_film_scorer_needs_cond_and_ignores_it_at_init(seg):
    jp = _build(seg, JointPolicyConfig(objective="pg", select_bn_eval=False, width=16, block_layers=1,
                                       centers_per_axis=32), scorer_input_norm="none", film_cond_dim=8)
    net = jp.scorer.eval()
    x = torch.randn(2, seg.canvit.canvas_dim, 32, 32)
    with pytest.raises(ValueError, match="needs the target embedding"):
        net(x)
    # zero-initialised FiLM generators: at init the scores do not depend on the target
    assert torch.equal(net(x, cond=torch.randn(2, 8)), net(x, cond=torch.randn(2, 8)))
    # ... and the conditioning path is trainable: a non-zero generator makes them differ
    torch.nn.init.normal_(net.film_in.gen.net[-1].weight)
    assert not torch.equal(net(x, cond=torch.randn(2, 8)), net(x, cond=torch.randn(2, 8)))


# --------------------------------------------------------------------------- #
# The engine: terminal PG (deferred), difference PG (inline), per_rollout entropy,
# the step_weight hook and the policy_cond hook.
# --------------------------------------------------------------------------- #
class _Task:
    """BoundAde20kTask plus the two optional search hooks."""

    def __init__(self, seg, *, weights=None, cond=None):
        from canvit.harness.tests.test_task_digests import _seg_masks
        from canvit.ade20k.task import BoundAde20kTask
        self.inner = BoundAde20kTask(seg=seg, masks=_seg_masks(), canvas_grid=_G)
        self.weights = weights
        self.cond_calls = 0
        if cond is not None:
            def policy_cond():
                self.cond_calls += 1
                return cond
            self.policy_cond = policy_cond

    def forward_glimpse(self, **kw):
        return self.inner.forward_glimpse(**kw)

    def step_loss(self, readout):
        return self.inner.step_loss(readout)

    def per_image_loss(self, readout):
        return self.inner.per_image_loss(readout)

    def step_weight(self, t, n):
        return 1.0 if self.weights is None else self.weights[t]


def _rollout(seg, jp, task, horizon=3):
    from canvit.harness.rollout import run_rollout
    from canvit.harness.rollout.viewpoint import ViewpointType
    from canvit.harness.spec import BpttSpec
    from canvit.harness.tests.test_task_digests import _B, _IMG, _selector
    torch.manual_seed(0)
    return run_rollout(
        model=seg, images=torch.randn(_B, 3, _IMG, _IMG), task=task, selector=_selector(foveated=True),
        bptt=BpttSpec(mode="none", horizon=horizon), branches=[ViewpointType.FULL],
        canvas_grid_size=_G, amp_ctx=nullcontext(), joint=jp, task_weight=0.0)


@pytest.mark.parametrize("reward", ["terminal", "difference"])
def test_pg_rewards_run_and_train_the_scorer(seg, reward):
    rl = JointPolicyConfig(objective="pg", select_bn_eval=False, reward=reward, reward_center="batch_mean",
                           reward_scale="batch_std", width=16, block_layers=1, centers_per_axis=32,
                           entropy_update="per_rollout", entropy_alpha_init=1e-3)
    jp = _build(seg, rl, scorer_input_norm="none")
    assert jp.defers_credit is (reward == "terminal")
    la0 = jp.log_alpha.clone()
    res = _rollout(seg, jp, _Task(seg))
    assert res.policy_metrics is not None and torch.isfinite(res.policy_metrics["policy_loss"])
    grads = [p.grad for p in jp.scorer.parameters() if p.grad is not None]
    assert grads and any(g.abs().sum() > 0 for g in grads), "the policy loss must reach the scorer"
    assert not torch.equal(jp.log_alpha, la0), "per_rollout: one dual step per rollout"
    assert jp._rollout_entropies == []


def test_per_rollout_takes_exactly_one_dual_step(seg):
    """Inline path, 3 glimpses = 2 policy glimpses. per_rollout must move log(alpha) once,
    by alpha_lr * (target - mean entropy); per_glimpse would move it twice."""
    rl = JointPolicyConfig(objective="pg", select_bn_eval=False, reward="difference", width=16,
                           block_layers=1, centers_per_axis=32, entropy_update="per_rollout",
                           entropy_target=2.0, alpha_lr=0.01, entropy_alpha_init=1e-3,
                           entropy_alpha_min=1e-6, entropy_alpha_max=0.1)
    jp = _build(seg, rl, scorer_input_norm="none")
    seen = []
    orig = jp._entropy_step

    def spy(obj, entropy):
        seen.append(float(entropy))
        orig(obj, entropy)
    jp._entropy_step = spy
    _rollout(seg, jp, _Task(seg))
    assert len(seen) == 1


def test_step_weight_last_only_trains_the_head_on_the_last_glimpse_only(seg):
    """With weights [0, 0, 3] (the engine divides by n_glimpses=3), the head gradient
    equals that of the last glimpse's loss alone. The backbone runs under no_grad
    (bptt='none'), so each glimpse's readout is a plain tensor the reference can reuse."""
    from canvit.harness.rollout import run_rollout
    from canvit.harness.rollout.viewpoint import ViewpointType
    from canvit.harness.spec import BpttSpec
    from canvit.harness.tests.test_task_digests import _B, _IMG, _selector

    seg.head.requires_grad_(True)
    try:
        task = _Task(seg, weights=[0.0, 0.0, 3.0])
        readouts = []
        orig = task.step_loss

        def keep(readout):
            readouts.append(readout)
            return orig(readout)
        task.step_loss = keep
        seg.head.zero_grad()
        torch.manual_seed(0)
        run_rollout(model=seg, images=torch.randn(_B, 3, _IMG, _IMG), task=task,
                    selector=_selector(foveated=True), bptt=BpttSpec(mode="none", horizon=3),
                    branches=[ViewpointType.FULL], canvas_grid_size=_G, amp_ctx=nullcontext())
        got = [p.grad.clone() for p in seg.head.parameters()]
        assert len(readouts) == 3 and not readouts[-1].requires_grad

        seg.head.zero_grad()
        task.inner.step_loss(readouts[-1]).combined.backward()
        want = [p.grad.clone() for p in seg.head.parameters()]
        for a, b in zip(got, want):
            assert torch.allclose(a, b, atol=1e-7, rtol=1e-5)
        assert any(a.abs().sum() > 0 for a in got)
    finally:
        seg.head.requires_grad_(False)
        seg.head.zero_grad()


def test_policy_cond_reaches_a_film_scorer(seg):
    rl = JointPolicyConfig(objective="pg", select_bn_eval=False, reward="terminal", width=16,
                           block_layers=1, centers_per_axis=32)
    jp = _build(seg, rl, scorer_input_norm="none", film_cond_dim=8)
    task = _Task(seg, cond=torch.randn(2, 8, requires_grad=True))
    _rollout(seg, jp, task)
    assert task.cond_calls == 2, "one call per policy glimpse (horizon 3 => 2)"
    with pytest.raises(ValueError, match="needs the target embedding"):
        _rollout(seg, jp, _Task(seg))  # a conditioned scorer without the hook fails loudly
