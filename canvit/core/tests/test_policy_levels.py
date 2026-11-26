"""ViewpointScorer ``levels`` (U-Net depth; ported from policy-algos 2ba0249 for exp42): the default
must stay the historical net, fewer levels stop at a coarser bottleneck with fewer parameters."""

import torch

from canvit.core.policy.features import POLICY_GRID, feature_channels
from canvit.core.policy.net import ViewpointScorer

_CD, _GROUPS = 16, ("ln_feat",)


def _scorer(**kw) -> ViewpointScorer:
    torch.manual_seed(0)
    return ViewpointScorer(canvas_dim=_CD, width=16, n_scale=1, scales=(1.0,), centers_per_axis=16,
                           block_layers=1, groups=_GROUPS, action_space="fixation", input_norm="none", **kw)


def _feats(b: int = 3) -> torch.Tensor:
    torch.manual_seed(1)
    return torch.randn(b, feature_channels(_CD, _GROUPS), POLICY_GRID, POLICY_GRID)


def test_default_is_the_historical_net():
    a, b = _scorer(), _scorer(levels=6)
    assert len(a.enc) == 6 and len(a.dec) == 5
    sa, sb = a.state_dict(), b.state_dict()
    assert sa.keys() == sb.keys() and all(torch.equal(sa[k], sb[k]) for k in sa)
    x = _feats()
    assert torch.equal(a.eval()(x), b.eval()(x))


def test_levels_bottleneck_and_params():
    x = _feats()
    for levels in (1, 3, 6):
        net = _scorer(levels=levels).eval()
        assert len(net.enc) == levels and len(net.dec) == levels - 1
        seen = []
        h = net.enc[-1].register_forward_hook(lambda m, i, o: seen.append(o.shape[-1]))
        assert net(x).shape[0] == 3
        h.remove()
        assert seen == [POLICY_GRID >> (levels - 1)]
    n = [sum(p.numel() for p in _scorer(levels=k).parameters()) for k in (3, 6)]
    assert n[0] < n[1]
