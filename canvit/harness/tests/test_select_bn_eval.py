"""BN mode (b) in the harness selector: available, off by default, digest-safe.

`PolicySelector` reused ONE train-mode scorer forward for both the glimpse choice and the
policy loss. The scorer carries a BatchNorm, so selection ran on batch statistics where the
RL reference uses running statistics — 45.7% of chosen glimpses differ, and mode (a) measured
0.19 mIoU t4 worse at matched CE (exp27 arm A vs arm C).
"""
import torch
import torch.nn as nn

from canvit.harness.rollout.selector import PolicySelector, RandomSelector
from canvit.harness.rollout.viewpoint import ViewpointType

_B, _A = 4, 8


class _Net(nn.Module):
    """Scorer stub whose BN makes train- and eval-mode scores differ, as the real one does."""

    def __init__(self):
        super().__init__()
        self.bn = nn.BatchNorm1d(_A)
        self.bn.running_mean.fill_(5.0)   # far from any batch mean -> modes disagree
        self.bn.running_var.fill_(1.0)

    def forward(self, x):
        return self.bn(x)


def _sel(**kw):
    return PolicySelector(
        net=_Net(), encoder=lambda s, **_kw: s, vp_flat=torch.rand(_A, 3),
        fallback=RandomSelector(is_foveated=False, foveated_scale=None, min_viewpoint_scale=0.05),
        generator=torch.Generator().manual_seed(0), **kw)


def _pick(sel, feats):
    sel.select(vp_type=ViewpointType.RANDOM, ctx=None, t=1, batch_size=_B,
               device=torch.device("cpu"), state=feats)
    return sel.last_aux["flat_idx"].clone()


def test_selector_primitive_stays_off_by_default():
    """`PolicySelector` is the low-level seam and keeps mode (a) as ITS default, so any
    caller that does not opt in is byte-identical to the pre-mode-(b) behaviour (the
    `run_rollout` parity digest is measured that way). The USER-FACING default lives in
    `JointPolicyConfig.select_bn_eval`, which is True."""
    from canvit.harness.config import JointPolicyConfig
    assert PolicySelector.select_bn_eval is False
    assert JointPolicyConfig().select_bn_eval is True, "band-reproducing config is the default"


def test_mode_b_changes_the_chosen_glimpse():
    torch.manual_seed(0)
    feats = torch.randn(_B, _A)
    a, b = _sel(select_bn_eval=False), _sel(select_bn_eval=True)
    b.net.load_state_dict(a.net.state_dict())
    assert not torch.equal(_pick(a, feats), _pick(b, feats)), (
        "the stub's BN was built so the modes must disagree; if they agree the "
        "eval-mode forward is not being used")


def test_mode_a_is_bit_identical_to_reusing_the_train_forward():
    """The digest guarantee: with the flag off, selection is still argmax of the SAME
    train-mode scores that feed the loss."""
    torch.manual_seed(0)
    feats = torch.randn(_B, _A)
    sel = _sel(select_bn_eval=False)
    idx = _pick(sel, feats)
    assert torch.equal(idx, sel.last_aux["scores"].detach().argmax(dim=1))


def test_the_loss_still_sees_train_mode_scores_under_mode_b():
    """Mode (b) must change only SELECTION — `scores` (the loss input) stays the
    grad-carrying train-mode forward, or the objective silently changes too."""
    torch.manual_seed(0)
    feats = torch.randn(_B, _A, requires_grad=True)
    sel = _sel(select_bn_eval=True)
    _pick(sel, feats)
    scores = sel.last_aux["scores"]
    assert scores.requires_grad, "loss input must still carry graph"
    assert sel.net.training, "train mode must be restored after selection"


def test_mode_b_adds_no_rng_draws():
    """RNG parity: the extra forward is no_grad/eval and must not consume the generator,
    so eps-greedy draws stay in the same order."""
    torch.manual_seed(0)
    feats = torch.randn(_B, _A)
    states = []
    for flag in (False, True):
        s = _sel(select_bn_eval=flag, prime_on_policy=0.5)
        _pick(s, feats)
        states.append(s.generator.get_state())
    assert torch.equal(states[0], states[1])
