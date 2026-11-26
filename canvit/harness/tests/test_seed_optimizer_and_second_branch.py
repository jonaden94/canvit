"""Two options for continuing a pretrained model with a policy (exp40 round 3, 2026-11-24):

  * ``RunSettings.seed_optimizer``: take over the seed's AdamW state for the task group, by
    parameter name, optionally scaling exp_avg_sq; continuing must equal continuing the seed's
    own optimizer, and parameters new in this run (or in other groups) start without state;
  * ``JointPolicyConfig.second_branch``: one policy branch plus a random-glimpse second branch
    (none / random start / center start) that never carries the policy.
"""

import pytest
import torch
from torch import nn

from canvit.distill.loss import DistillTask
from canvit.distill.task import BoundDistillTask
from canvit.harness.config import FoveatedScaleConfig
from canvit.harness.rollout import run_rollout
from canvit.harness.rollout.selector import RandomSelector
from canvit.harness.rollout.viewpoint import ViewpointType
from canvit.harness.run import load_seed_optimizer_state
from canvit.harness.spec import BpttSpec
from canvit.harness.tests.test_policy_tokens import _B, _D, _DEV, _G, _IMG, _K, _joint, _model


class _Seed(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.a = nn.Linear(3, 4)
        self.unused = nn.Parameter(torch.zeros(5))  # never gets a gradient -> no AdamW state
        self.b = nn.Linear(4, 2)
        self.register_buffer("buf", torch.ones(2))


class _Run(nn.Module):
    """The seed's parameters plus a new one in the middle (name matching, not position)."""

    def __init__(self) -> None:
        super().__init__()
        self.a = nn.Linear(3, 4)
        self.new = nn.Parameter(torch.zeros(7))
        self.unused = nn.Parameter(torch.zeros(5))
        self.b = nn.Linear(4, 2)
        self.register_buffer("buf", torch.ones(2))


def _grads(m: nn.Module, seed: int) -> None:
    g = torch.Generator().manual_seed(seed)
    for n, p in m.named_parameters():
        p.grad = None if n in ("unused", "new") else torch.randn(p.shape, generator=g)


def _seed_run():
    torch.manual_seed(0)
    seed = _Seed()
    opt = torch.optim.AdamW(seed.parameters(), lr=1e-2, weight_decay=1e-4)
    for s in range(3):
        _grads(seed, s)
        opt.step()
    return seed, opt, {"model_state": seed.state_dict(), "optimizer_state": opt.state_dict()}


def _run_opt(run: _Run, policy: nn.Parameter):
    names = [n for n, _ in run.named_parameters()]
    params = list(run.parameters())
    opt = torch.optim.AdamW([{"params": params, "lr": 1e-2, "weight_decay": 1e-4},
                             {"params": [policy], "lr": 1e-3}])
    return opt, names, params


def test_taken_over_state_continues_exactly_like_the_seed_optimizer():
    seed, seed_opt, payload = _seed_run()
    run = _Run()
    run.load_state_dict(seed.state_dict(), strict=False)
    policy = nn.Parameter(torch.zeros(3))
    opt, names, params = _run_opt(run, policy)
    n = load_seed_optimizer_state(opt, names, params, payload)
    assert n == 4  # a.weight, a.bias, b.weight, b.bias (unused has no state, new is new)
    assert run.new not in opt.state and run.unused not in opt.state and policy not in opt.state
    _grads(seed, 9)
    _grads(run, 9)
    seed_opt.step()
    opt.step()
    for name in ("a.weight", "a.bias", "b.weight", "b.bias"):
        assert torch.equal(dict(seed.named_parameters())[name], dict(run.named_parameters())[name]), name


def test_exp_avg_sq_scale_and_untouched_exp_avg():
    _, seed_opt, payload = _seed_run()
    run = _Run()
    opt, names, params = _run_opt(run, nn.Parameter(torch.zeros(3)))
    load_seed_optimizer_state(opt, names, params, payload, exp_avg_sq_scale=3.0)
    src = payload["optimizer_state"]["state"]
    by_name = dict(run.named_parameters())
    seed_names = [k for k in payload["model_state"] if k != "buf"]
    for sid, name in enumerate(seed_names):
        if sid not in src:
            continue
        st = opt.state[by_name[name]]
        assert torch.equal(st["exp_avg"], src[sid]["exp_avg"])
        assert torch.equal(st["exp_avg_sq"], src[sid]["exp_avg_sq"] * 3.0)
        assert torch.equal(st["step"], src[sid]["step"])


def test_refuses_a_seed_that_does_not_match():
    _, _, payload = _seed_run()
    run = _Run()
    opt, names, params = _run_opt(run, nn.Parameter(torch.zeros(3)))
    with pytest.raises(ValueError, match="no optimizer_state"):
        load_seed_optimizer_state(opt, names, params, {"model_state": payload["model_state"]})
    two = {**payload, "optimizer_state": {**payload["optimizer_state"],
                                          "param_groups": payload["optimizer_state"]["param_groups"] * 2}}
    with pytest.raises(ValueError, match="param groups"):
        load_seed_optimizer_state(opt, names, params, two)
    # a seed parameter that is not in this run's group (e.g. frozen here) -> count mismatch
    with pytest.raises(ValueError, match="seed group has"):
        keep = [i for i, n in enumerate(names) if n != "a.weight"]
        load_seed_optimizer_state(opt, [names[i] for i in keep], [params[i] for i in keep], payload)
    o = payload["optimizer_state"]
    wrong = {**o, "state": {k: ({**v, "exp_avg": torch.zeros(9, 9)} if k == 1 else v) for k, v in o["state"].items()}}
    with pytest.raises(ValueError, match="shape"):
        load_seed_optimizer_state(opt, names, params, {**payload, "optimizer_state": wrong})


# --- second branch ----------------------------------------------------------------------------

@pytest.mark.parametrize("second,expect", [
    ("none", [(ViewpointType.FULL, True)]),
    ("random_start", [(ViewpointType.FULL, True), (ViewpointType.RANDOM, False)]),
    ("center_start", [(ViewpointType.FULL, True), (ViewpointType.FULL, False)]),
])
def test_second_branch_runs_one_policy_branch_plus_a_random_one(second, expect):
    m = _model(n_policy_tokens=_K, policy_token_mode="isolated")
    joint = _joint(m, second_branch=second)
    branches = joint.branches([ViewpointType.FULL, ViewpointType.RANDOM])
    assert branches == [t for t, _ in expect]
    for i, t0 in enumerate(branches):
        sel, is_pol = joint.branch_selector(t0, i)
        assert is_pol == (i == 0)
        assert sel is (joint.policy_selector if i == 0 else joint.random_selector)
    task = BoundDistillTask(DistillTask(
        scene_target=torch.randn(_B, _G * _G, _D), cls_target=torch.randn(_B, _D),
        enable_scene_patches_loss=True, enable_scene_cls_loss=True))
    r = run_rollout(
        model=m, images=torch.randn(_B, 3, _IMG, _IMG), task=task,
        selector=RandomSelector(is_foveated=False, foveated_scale=FoveatedScaleConfig(), min_viewpoint_scale=0.05),
        bptt=BpttSpec(mode="chunked", chunk_size=2, horizon=4), branches=branches,
        canvas_grid_size=_G, amp_ctx=torch.enable_grad(), joint=joint)
    assert [(b.t0_type, b.is_policy) for b in r.branches] == expect
    assert r.policy_metrics is not None and torch.isfinite(r.policy_metrics["policy_loss"])


def test_second_branch_unset_keeps_the_task_branches_and_old_rule():
    m = _model(n_policy_tokens=_K, policy_token_mode="isolated")
    for keep in (False, True):
        joint = _joint(m, keep_random_branch=keep)
        tb = [ViewpointType.FULL, ViewpointType.RANDOM]
        assert joint.branches(tb) is tb
        assert joint.branch_selector(ViewpointType.FULL, 0)[1] is True
        assert joint.branch_selector(ViewpointType.RANDOM, 1)[1] is (not keep)
        assert joint.branch_selector(ViewpointType.FULL, 1)[1] is True  # FULL is always policy
