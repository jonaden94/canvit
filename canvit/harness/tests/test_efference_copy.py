"""The efference copy (``JointPolicyConfig.efference_copy``): the policy's extra input channel that says
where it looks NOW, and the search validation's saccade length (2026-11-27, exp44).

Off by default and then byte-identical to before (checked separately by the exp43-recipe parity probe:
canvit_claude/canvit/active_pretraining/exp44_saccade_cost.md)."""

from __future__ import annotations

from dataclasses import replace

import pytest
import torch

from canvit.core.policy import fixation_candidates
from canvit.core.policy.features import efference_copy_map
from canvit.harness.config import FoveatedScaleConfig
from canvit.harness.rollout.eval_viewpoints import deploy_selector
from canvit.harness.rollout.selector import PolicySelector, RandomSelector
from canvit.harness.rollout.viewpoint import ViewpointType

_G = 32


def test_map_peaks_at_the_fixation_cell():
    c = torch.tensor([[0.0, 0.0], [-1 + 1 / _G, 1 - 1 / _G]])     # scene centre; top-right cell centre
    m = efference_copy_map(c, _G, sigma=2 / _G)
    assert m.shape == (2, 1, _G, _G)
    assert m[1, 0].flatten().argmax().item() == 0 * _G + (_G - 1)   # row 0 (cy = -1 + 1/G), last column
    assert torch.isclose(m[1, 0, 0, _G - 1], torch.tensor(1.0))
    centre = m[0, 0, _G // 2 - 1: _G // 2 + 1, _G // 2 - 1: _G // 2 + 1]
    assert torch.allclose(centre, centre[0, 0].expand(2, 2))         # the 4 central cells are equally near


class _Net(torch.nn.Module):
    """Scores: a fixed one-hot per call (cell `pick`), so the selector's argmax is known; records its input."""

    def __init__(self, picks):
        super().__init__()
        self.picks, self.seen = list(picks), []

    def forward(self, feats, cond=None):
        self.seen.append(feats.detach().clone())
        s = torch.zeros(feats.shape[0], _G * _G)
        s[:, self.picks.pop(0)] = 1.0
        return s.view(-1, 1, _G, _G)


def _selector(picks, **kw):
    return PolicySelector(net=_Net(picks), encoder=lambda state: torch.zeros(2, 3, _G, _G),
                          vp_flat=fixation_candidates(_G).reshape(-1, 3),
                          fallback=RandomSelector(is_foveated=False, foveated_scale=FoveatedScaleConfig(),
                                                  min_viewpoint_scale=0.05), **kw)


def _rollout(sel, n):
    ctx = sel.start_rollout(t0_type=ViewpointType.FULL, batch_size=2, device=torch.device("cpu"))
    vps = [sel.select(vp_type=ViewpointType.FULL, ctx=ctx, t=0, batch_size=2, device=torch.device("cpu"),
                      state=None)]
    for t in range(1, n):
        vps.append(sel.select(vp_type=ViewpointType.RANDOM,  # every non-FULL glimpse is the policy's
                               ctx=ctx, t=t, batch_size=2,
                              device=torch.device("cpu"), state=None))
    return vps


def test_the_channel_follows_the_current_fixation():
    first, second = 5 * _G + 7, 20 * _G + 3
    sel = _selector([first, second, 0], efference_copy=True)
    _rollout(sel, 4)                                                  # t0 FULL + 3 policy glimpses
    seen = sel.net.seen
    assert all(f.shape[1] == 4 for f in seen)                        # 3 canvas channels + the efference copy
    assert seen[0][:, :3].abs().sum() == 0                            # the canvas groups are untouched
    peaks = [f[0, 3].flatten().argmax().item() for f in seen]
    full_peak = efference_copy_map(torch.zeros(1, 2), _G, 2 / _G)[0, 0].flatten().argmax().item()
    assert peaks == [full_peak, first, second]                        # t1 from the centre, then the last pick
    # a new rollout starts from the scene centre again; a deploy copy starts empty
    sel.net.picks = [1, 2]
    _rollout(sel, 2)
    assert sel.net.seen[-1][0, 3].flatten().argmax().item() == full_peak
    assert replace(sel)._last_centres is None


def test_off_is_the_old_input():
    sel = _selector([3, 4])
    _rollout(sel, 3)
    assert all(f.shape[1] == 3 for f in sel.net.seen) and sel._last_centres is None


def test_deploy_selector_keeps_the_flag():
    class J:
        scorer = object()
        policy_selector = _selector([0], efference_copy=True)
    d = deploy_selector(J())
    assert d.efference_copy and d._last_centres is None and d.mode == "argmax"


def test_refused_for_sac():
    from types import SimpleNamespace

    from canvit.harness.policy import build_policy
    from canvit.harness.tests.test_saccade_cost import _rl
    canvit = SimpleNamespace(cfg=SimpleNamespace(canvas_dim=8, n_policy_tokens=0))
    with pytest.raises(ValueError, match="efference_copy"):
        build_policy(canvit=canvit, rl=replace(_rl(), objective="sac", reward="default", reward_center="default",
                                               reward_scale="default", efference_copy=True),
                     feature_groups=("ln_feat",), device=torch.device("cpu"), canvas_grid=8,
                     min_viewpoint_scale=0.05, foveated_scale=FoveatedScaleConfig(),
                     generator=torch.Generator().manual_seed(0), encode_model=None, scorer_input_norm="none")
