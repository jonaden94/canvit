"""The optional per-saccade cost on the terminal-reward PG target (``JointPolicyConfig.saccade_cost*``).

Off (the default) and weight 0 must reproduce the historical trajectory loss bit for bit; a
positive weight must lower the target of the actions with the longest saccades. CPU only.
"""

from dataclasses import replace

import pytest
import torch

import canvit.harness.policy.joint as joint_mod
from canvit.harness.policy.joint import JointPolicy, SaccadeCost
from canvit.harness.policy.rl import PG, RewardSpec

B, T, A = 8, 3, 16
SEARCH_REWARD = RewardSpec("terminal", "batch_mean", "batch_std")


def _jp(cost: SaccadeCost = SaccadeCost()) -> JointPolicy:
    # Only the trajectory-loss fields are used; the selectors/scorer are not touched here.
    return JointPolicy(policy_selector=None, random_selector=None, scorer=None,  # type: ignore[arg-type]
                       objective=PG(entropy_bonus=1e-3, entropy_target=None), rl_weight=1.0,
                       keep_random_branch=False, target_momentum=0.997, device=torch.device("cpu"),
                       reward=SEARCH_REWARD, saccade_cost=cost)


def _batch(seed: int = 0):
    g = torch.Generator().manual_seed(seed)
    scores = torch.randn(B, T, A, generator=g)
    flat_idx = torch.randint(A, (B, T), generator=g)
    reward = -torch.rand(B, generator=g)
    saccade_len = 2 * torch.rand(B, T, generator=g)  # [-1, 1] units: 0 .. scene width
    mask = torch.arange(B) % 2 == 0  # as search: target-absent trials carry no reward
    return scores, flat_idx, reward, saccade_len, mask


def _loss_and_grad(jp: JointPolicy, scores: torch.Tensor, **kw):
    s = scores.clone().requires_grad_(True)
    loss = jp.trajectory_loss(scores=s, values=None, **kw)
    loss.backward()
    return loss.detach(), s.grad


@pytest.mark.parametrize("with_mask", [False, True])
@pytest.mark.parametrize("form", ["log", "linear"])
def test_no_cost_or_zero_weight_is_the_historical_loss_bit_for_bit(with_mask, form):
    scores, flat_idx, reward, saccade_len, mask = _batch()
    kw = {"flat_idx": flat_idx, "reward": reward} | ({"mask": mask} if with_mask else {})
    ref_loss, ref_grad = _loss_and_grad(_jp(), scores, **kw)
    for combine in ("standardized", "raw"):
        loss, grad = _loss_and_grad(_jp(SaccadeCost(form, 0.0, combine)), scores, saccade_len=saccade_len, **kw)
        assert torch.equal(loss, ref_loss) and torch.equal(grad, ref_grad)


def _targets(monkeypatch, jp: JointPolicy, **kw) -> torch.Tensor:
    seen = []
    real = joint_mod.pg_loss

    def spy(scores, flat_idx, target, **k):
        seen.append(target.clone())
        return real(scores, flat_idx, target, **k)

    monkeypatch.setattr(joint_mod, "pg_loss", spy)
    jp.trajectory_loss(values=None, **kw)
    return torch.stack(seen, dim=1)  # [B, T]


@pytest.mark.parametrize("form", ["log", "linear"])
def test_a_positive_weight_lowers_the_target_of_long_saccades(monkeypatch, form):
    scores, flat_idx, reward, saccade_len, _ = _batch()
    kw = {"scores": scores, "flat_idx": flat_idx, "reward": reward, "saccade_len": saccade_len}
    base = _targets(monkeypatch, _jp(SaccadeCost(form, 0.0)), **kw)
    beta = 0.3
    got = _targets(monkeypatch, _jp(SaccadeCost(form, beta)), **kw)
    shift = got - base  # = -beta * z_batch(c_t): per step, longer saccade -> lower target
    for t in range(T):
        order = saccade_len[:, t].argsort()
        assert (shift[order, t].diff() <= 1e-6).all()
        assert torch.allclose(shift[:, t].mean(), torch.tensor(0.0), atol=1e-6)  # centred: no net push
        assert torch.allclose(shift[:, t].std(unbiased=False), torch.tensor(beta), atol=1e-5)


def test_raw_combine_is_the_cost_subtracted_from_the_reward(monkeypatch):
    scores, flat_idx, reward, saccade_len, _ = _batch()
    lam, sc = 0.05, SaccadeCost("log", 0.05, "raw", r=1 / 32)
    got = _targets(monkeypatch, _jp(sc), scores=scores, flat_idx=flat_idx, reward=reward, saccade_len=saccade_len)
    want = _targets(monkeypatch, _jp(), scores=scores, flat_idx=flat_idx,
                    reward=reward - lam * torch.log1p(saccade_len * 32).sum(dim=1))
    assert torch.allclose(got, want)


def test_cost_forms():
    d = torch.tensor([0.0, 1 / 32, 2.0])
    assert torch.allclose(SaccadeCost("log", r=1 / 32).cost(d), torch.log1p(d * 32))
    assert torch.allclose(SaccadeCost("log", r=1 / 32).cost(d)[1], torch.tensor(2.0).log())  # d = r -> log 2
    assert torch.allclose(SaccadeCost("linear").cost(d), torch.tensor([0.0, 1 / 64, 1.0]))


# --------------------------------------------------------------------------- #
# Through run_rollout + build_policy, on the tiny foveated model of test_reward_options
# --------------------------------------------------------------------------- #
from canvit.harness.config import JointPolicyConfig  # noqa: E402
from canvit.harness.tests.test_reward_options import _build, _MaskedTask, _rollout, seg  # noqa: E402,F401


def _rl(**kw) -> JointPolicyConfig:
    return JointPolicyConfig(objective="pg", select_bn_eval=False, reward="terminal", reward_center="batch_mean",
                             reward_scale="batch_std", width=16, block_layers=1, centers_per_axis=32, **kw)


def _rollout_grads(seg, rl):
    from canvit.harness.tests.test_task_digests import _B
    torch.manual_seed(1)
    jp = _build(seg, rl, scorer_input_norm="none")
    out = _rollout(seg, jp, _MaskedTask(seg, mask=None, bump=torch.ones(_B)))
    return torch.cat([p.grad.flatten() for p in jp.scorer.parameters() if p.grad is not None]), out


def test_rollout_with_zero_weight_trains_exactly_as_without_and_logs_saccades(seg):
    base, out0 = _rollout_grads(seg, _rl())
    assert "saccade_len" not in out0.policy_metrics
    same, out = _rollout_grads(seg, _rl(saccade_cost="log"))
    assert torch.equal(base, same)
    d = float(out.policy_metrics["saccade_len"])
    assert 0.0 < d <= 2 * 2 ** 0.5  # centre units: at most the scene diagonal
    moved, _ = _rollout_grads(seg, _rl(saccade_cost="log", saccade_cost_weight=0.3))
    assert not torch.equal(base, moved)  # the cost reaches the scorer gradient


@pytest.mark.parametrize("kw, match", [
    ({"reward": "difference"}, "reward='terminal' only"),
    ({"objective": "qreg", "reward": "fractional", "reward_center": "default", "reward_scale": "default",
      "select_bn_eval": True}, "objective='pg'"),
])
def test_the_cost_is_refused_off_the_terminal_pg_path(seg, kw, match):
    with pytest.raises(ValueError, match=match):
        _build(seg, replace(_rl(saccade_cost="log", saccade_cost_weight=0.1), **kw), scorer_input_norm="none")


def test_a_weight_without_a_form_is_refused(seg):
    with pytest.raises(ValueError, match="saccade_cost='none'"):
        _build(seg, _rl(saccade_cost_weight=0.1), scorer_input_norm="none")


# --------------------------------------------------------------------------- #
# Stage 2: a beta-conditioned policy (per-trial beta). Written by Claude (AI, Claude Opus 5.5), 2026-11-25.
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("with_mask", [False, True])
@pytest.mark.parametrize("form", ["log", "linear"])
def test_a_constant_per_trial_beta_is_the_fixed_beta_bit_for_bit(with_mask, form):
    scores, flat_idx, reward, saccade_len, mask = _batch()
    kw = {"flat_idx": flat_idx, "reward": reward, "saccade_len": saccade_len} | ({"mask": mask} if with_mask else {})
    ref_loss, ref_grad = _loss_and_grad(_jp(SaccadeCost(form, 0.3)), scores, **kw)
    cond = _jp(SaccadeCost(form, 1.0, conditioned=True))  # weight = beta_max: unused when beta is given
    loss, grad = _loss_and_grad(cond, scores, saccade_beta=torch.full((B,), 0.3), **kw)
    assert torch.equal(loss, ref_loss) and torch.equal(grad, ref_grad)


def test_each_trial_is_shifted_by_its_own_beta(monkeypatch):
    scores, flat_idx, reward, saccade_len, _ = _batch()
    kw = {"scores": scores, "flat_idx": flat_idx, "reward": reward, "saccade_len": saccade_len}
    base = _targets(monkeypatch, _jp(SaccadeCost("log", 0.0)), **kw)
    unit = _targets(monkeypatch, _jp(SaccadeCost("log", 1.0)), **kw) - base  # -z_batch(c_t)
    beta = torch.linspace(0.0, 1.0, B)
    got = _targets(monkeypatch, _jp(SaccadeCost("log", 1.0, conditioned=True)), saccade_beta=beta, **kw) - base
    assert torch.allclose(got, beta[:, None] * unit, atol=1e-6)
    assert torch.equal(got[0], torch.zeros(T))  # beta = 0: the cost-free target


def test_a_conditioned_cost_needs_the_per_trial_beta():
    scores, flat_idx, reward, saccade_len, _ = _batch()
    with pytest.raises(AssertionError, match="per-trial beta"):
        _jp(SaccadeCost("log", 1.0, conditioned=True)).trajectory_loss(
            scores=scores, flat_idx=flat_idx, values=None, reward=reward, saccade_len=saccade_len)


def test_beta_features_and_sampling():
    f = SaccadeCost.features(torch.tensor([0.0, 0.1, 1.0]))
    assert f.shape == (3, joint_mod.BETA_FEATURES)
    assert torch.equal(f[0], torch.zeros(2))
    assert torch.allclose(f[1], torch.tensor([0.1, 0.6931472]))  # log1p(1) = log 2
    torch.manual_seed(0)
    b = SaccadeCost("log", 0.7, conditioned=True).sample_beta(1000, torch.device("cpu"))
    assert b.shape == (1000,) and 0.0 <= float(b.min()) and float(b.max()) <= 0.7 and float(b.std()) > 0.1


def test_rollout_with_a_constant_task_beta_trains_as_the_fixed_beta(seg):
    """Through run_rollout: the engine hands the task's ``saccade_beta`` to the loss; a constant
    beta reproduces the stage-1 scorer gradient exactly (same scorer, only the cost path differs)."""
    from canvit.harness.tests.test_task_digests import _B
    fixed, _ = _rollout_grads(seg, _rl(saccade_cost="log", saccade_cost_weight=0.3))
    torch.manual_seed(1)
    jp = _build(seg, _rl(saccade_cost="log", saccade_cost_weight=0.3), scorer_input_norm="none")
    jp.saccade_cost = replace(jp.saccade_cost, weight=1.0, conditioned=True)  # the unconditioned scorer is kept
    task = _MaskedTask(seg, mask=None, bump=torch.ones(_B))
    task.saccade_beta = torch.full((_B,), 0.3)
    _rollout(seg, jp, task)
    cond = torch.cat([p.grad.flatten() for p in jp.scorer.parameters() if p.grad is not None])
    assert torch.equal(fixed, cond)


@pytest.mark.parametrize("kw, match", [
    ({"saccade_cost_beta_mode": "conditioned", "saccade_cost": "log", "saccade_cost_weight": 1.0}, "FiLM"),
    ({"saccade_cost_beta_mode": "conditioned", "saccade_cost": "log"}, "weight > 0"),
    ({"saccade_cost_beta_mode": "conditioned", "saccade_cost": "log", "saccade_cost_weight": 1.0,
      "saccade_cost_combine": "raw"}, "standardized"),
    ({"saccade_cost": "log", "saccade_cost_weight": 0.3, "saccade_cost_beta_eval": 0.3}, "beta_mode='fixed'"),
])
def test_incoherent_stage2_settings_are_refused(seg, kw, match):
    with pytest.raises(ValueError, match=match):
        _build(seg, _rl(**kw), scorer_input_norm="none")
