"""The post-hoc cost shapes are the training's: equal to ``SaccadeCost.cost`` (harness, branch saccade-cost)
on a grid of d, bit for bit, whenever both are importable. Written by Claude (AI, Claude Opus 5.5), 2026-11-25."""

import pytest
import torch

from canvit.scanpath.posthoc.cost_forms import FORMS, saccade_cost

D = torch.linspace(0.0, 2 * 2 ** 0.5, 1001)  # centre units: 0 .. the scene diagonal


def test_shapes():
    r = 1 / 32
    assert torch.allclose(saccade_cost("log", D, r), torch.log(1 + D / r))
    assert torch.allclose(saccade_cost("log", torch.tensor(r), r), torch.tensor(2.0).log())  # d = r -> log 2
    assert torch.equal(saccade_cost("linear", D, r), D / 2)
    with pytest.raises(AssertionError):
        saccade_cost("quadratic", D, r)


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
@pytest.mark.parametrize("grid", [8, 32])
@pytest.mark.parametrize("form", FORMS)
def test_equal_to_the_training_cost(form, grid, dtype):
    joint = pytest.importorskip("canvit.harness.policy.joint")
    if not hasattr(joint, "SaccadeCost"):
        pytest.skip("training-side saccade cost not on this branch (saccade-cost)")
    d = D.to(dtype)
    assert torch.equal(joint.SaccadeCost(form, r=1.0 / grid).cost(d), saccade_cost(form, d, 1.0 / grid))
