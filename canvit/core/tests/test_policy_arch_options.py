"""ViewpointScorer architecture options added 2026-11-13 for the exp36_1 architecture ablation:
`levels` (U-Net depth) and `input_norm` (the Frontend's input BatchNorm). Both default to the
historical net, which must stay exactly as it was."""

import torch
from torch import nn

from canvit.core.policy.features import POLICY_GRID, feature_channels
from canvit.core.policy.net import ViewpointScorer

_CD, _GROUPS = 16, ("ln_feat",)


def _scorer(**kw) -> ViewpointScorer:
    torch.manual_seed(0)
    return ViewpointScorer(
        canvas_dim=_CD, width=16, n_scale=1, scales=(1.0,), centers_per_axis=16, block_layers=1,
        groups=_GROUPS, action_space="fixation", **kw,
    )


def _feats(b: int = 3) -> torch.Tensor:
    torch.manual_seed(1)
    return torch.randn(b, feature_channels(_CD, _GROUPS), POLICY_GRID, POLICY_GRID)


def test_defaults_are_the_historical_net() -> None:
    a, b = _scorer(), _scorer(levels=6, input_norm="bn")
    assert len(a.enc) == 6 and len(a.dec) == 5
    assert isinstance(a.frontend.bn, nn.BatchNorm2d)
    sa, sb = a.state_dict(), b.state_dict()
    assert sa.keys() == sb.keys() and all(torch.equal(sa[k], sb[k]) for k in sa)


def test_levels_shapes_and_bottleneck() -> None:
    x = _feats()
    for levels in range(1, 7):
        net = _scorer(levels=levels).eval()
        assert len(net.enc) == levels and len(net.dec) == levels - 1
        seen = []
        hook = net.enc[-1].register_forward_hook(lambda m, i, o: seen.append(o.shape[-1]))
        assert net(x).shape == (3, 1, 16, 16)
        hook.remove()
        assert seen == [POLICY_GRID >> (levels - 1)]  # bottleneck resolution


def test_fewer_levels_fewer_params() -> None:
    n = [sum(p.numel() for p in _scorer(levels=k).parameters()) for k in (3, 4, 6)]
    assert n[0] < n[1] < n[2]


def test_input_norm_none_has_no_batchnorm_and_no_train_eval_gap() -> None:
    net = _scorer(input_norm="none")
    assert not any(isinstance(m, nn.modules.batchnorm._BatchNorm) for m in net.modules())
    x = _feats()
    with torch.no_grad():
        train_out = net.train()(x)
        eval_out = net.eval()(x)
    assert torch.equal(train_out, eval_out)
    # the historical net does differ between the two modes (that is the gap removed)
    ref = _scorer()
    with torch.no_grad():
        assert not torch.allclose(ref.train()(x), ref.eval()(x))


def test_options_roundtrip_through_config(tmp_path) -> None:
    net = _scorer(levels=3, input_norm="none")
    net.save_pretrained(tmp_path)
    back = ViewpointScorer.from_pretrained(tmp_path)
    assert len(back.enc) == 3 and isinstance(back.frontend.bn, nn.Identity)
    x = _feats()
    with torch.no_grad():
        assert torch.equal(net.eval()(x), back.eval()(x))
