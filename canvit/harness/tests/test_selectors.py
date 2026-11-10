"""P4a tests: PolicySelector / MixtureSelector through the P1 seam (fake net/encoder
— the real ones are covered in core's policy tests and ade20k/test_rl_train.py)."""

import torch

from canvit.harness.config import FoveatedScaleConfig
from canvit.harness.rollout.selector import MixtureSelector, PolicySelector, RandomSelector
from canvit.harness.rollout.viewpoint import ViewpointType

_B, _A = 4, 8
_CPU = torch.device("cpu")


class _FakeEncoder:
    def __init__(self):
        self.resets = 0

    def reset(self):
        self.resets += 1

    def __call__(self, state, **_kw):
        return torch.zeros(_B, 3, 32, 32)


def _make(mode: str, scores: torch.Tensor) -> PolicySelector:
    vp_flat = torch.stack(
        [torch.linspace(-0.5, 0.5, _A), torch.linspace(0.5, -0.5, _A), torch.full((_A,), 0.5)], dim=-1
    )
    rnd = RandomSelector(is_foveated=False, foveated_scale=FoveatedScaleConfig(), min_viewpoint_scale=0.1)
    return PolicySelector(
        net=lambda f: scores, encoder=_FakeEncoder(), vp_flat=vp_flat, fallback=rnd, mode=mode
    )


def test_policy_selector_argmax_and_aux() -> None:
    scores = torch.zeros(_B, _A)
    scores[:, 3] = 5.0  # candidate 3 wins everywhere
    sel = _make("argmax", scores)
    ctx = sel.start_rollout(t0_type=ViewpointType.FULL, batch_size=_B, device=_CPU)
    assert sel.encoder.resets == 1  # type: ignore[attr-defined]
    vp = sel.select(vp_type=ViewpointType.RANDOM, ctx=ctx, t=1, batch_size=_B, device=_CPU, state=None)
    assert vp.name == "policy"
    assert torch.allclose(vp.centers, sel.vp_flat[3, :2].expand(_B, 2))
    assert sel.last_aux is not None and (sel.last_aux["flat_idx"] == 3).all()
    # FULL delegates to the random fallback (t0 anchor)
    full = sel.select(vp_type=ViewpointType.FULL, ctx=ctx, t=0, batch_size=_B, device=_CPU, state=None)
    assert full.name == "full" and (full.scales == 1.0).all()


def test_policy_selector_sample_mode() -> None:
    scores = torch.full((_B, _A), -100.0)
    scores[:, 5] = 100.0  # ~all probability mass on candidate 5
    sel = _make("sample", scores)
    ctx = sel.start_rollout(t0_type=ViewpointType.RANDOM, batch_size=_B, device=_CPU)
    sel.select(vp_type=ViewpointType.RANDOM, ctx=ctx, t=0, batch_size=_B, device=_CPU, state=None)
    assert sel.last_aux is not None and (sel.last_aux["flat_idx"] == 5).all()


def test_mixture_extremes_and_blend() -> None:
    scores = torch.zeros(_B, _A)
    scores[:, 0] = 5.0
    pol = _make("argmax", scores)
    rnd = pol.fallback
    mix = MixtureSelector(random_sel=rnd, policy_sel=pol, p_policy=0.0)
    ctx = mix.start_rollout(t0_type=ViewpointType.RANDOM, batch_size=_B, device=_CPU)

    torch.manual_seed(0)
    vp = mix.select(vp_type=ViewpointType.RANDOM, ctx=ctx, t=0, batch_size=_B, device=_CPU, state=None)
    assert vp.name == "random" and not mix.last_mask.any()  # p=0 -> today's behavior

    mix.p_policy = 1.0
    vp = mix.select(vp_type=ViewpointType.RANDOM, ctx=ctx, t=1, batch_size=_B, device=_CPU, state=None)
    assert vp.name == "policy" and mix.last_mask.all()

    mix.p_policy = 0.5
    torch.manual_seed(1)
    vp = mix.select(vp_type=ViewpointType.RANDOM, ctx=ctx, t=2, batch_size=_B, device=_CPU, state=None)
    assert vp.name == "mixture"
    # policy rows carry candidate 0's center; masked rows must match exactly
    assert torch.allclose(vp.centers[mix.last_mask], pol.vp_flat[0, :2].expand(int(mix.last_mask.sum()), 2))


# --------------------------------------------------------------------------- #
# Foveated view scale: the policy chooses WHERE, `foveated_scale` chooses HOW WIDE.
#
# `fixation_candidates` has no scale dimension, so its table's scale column is a
# hardcoded 1.0. Taking that literally pinned every policy glimpse to full-field
# foveation, so on a model pretrained at another scale the policy's glimpses were out
# of distribution while t0's and the random ones were not. These tests pin the fix AND
# pin the two things it must not disturb: the uniform path (where the policy really
# does own the scale) and the random path's RNG stream.
# --------------------------------------------------------------------------- #

def _fixation_policy(fsc: FoveatedScaleConfig, *, n: int = _A) -> PolicySelector:
    """PolicySelector over a FIXATION table — scale column hardcoded 1.0, as
    `fixation_candidates` produces it."""
    vp_flat = torch.stack(
        [torch.linspace(-0.5, 0.5, n), torch.linspace(0.5, -0.5, n), torch.ones(n)], dim=-1
    )
    rnd = RandomSelector(is_foveated=True, foveated_scale=fsc, min_viewpoint_scale=0.1)
    scores = torch.zeros(_B, n)
    scores[:, 2] = 5.0
    return PolicySelector(
        net=lambda f: scores, encoder=_FakeEncoder(), vp_flat=vp_flat, fallback=rnd, mode="argmax"
    )


def _policy_scales(sel: PolicySelector, *, t0=ViewpointType.FULL, steps: int = 2):
    ctx = sel.start_rollout(t0_type=t0, batch_size=_B, device=_CPU)
    anchor = sel.select(vp_type=t0, ctx=ctx, t=0, batch_size=_B, device=_CPU, state=None)
    out = [sel.select(vp_type=ViewpointType.RANDOM, ctx=ctx, t=t, batch_size=_B,
                      device=_CPU, state=None).scales for t in range(1, steps + 1)]
    return anchor.scales, out


def test_foveated_policy_glimpses_use_the_configured_fixed_scale() -> None:
    """The bug: at fixed_scale != 1.0 the anchor used the configured scale and the policy
    used 1.0. They must agree — a view-scale mismatch is out of distribution for the
    backbone (see Ade20kConfig.foveated_scale)."""
    for fs in (0.5, 1.0, 2.0):
        sel = _fixation_policy(FoveatedScaleConfig(mode="fixed", fixed_scale=fs))
        anchor, steps = _policy_scales(sel)
        assert torch.allclose(anchor, torch.full((_B,), fs)), (fs, anchor)
        for s in steps:
            assert torch.allclose(s, torch.full((_B,), fs)), (fs, s)


def test_fixed_scale_one_is_bit_identical_to_the_old_hardcoded_value() -> None:
    """fixed_scale=1.0 is the default and the only configuration that ever worked, so the
    fix must reproduce it exactly — otherwise every existing foveated policy run changes."""
    sel = _fixation_policy(FoveatedScaleConfig(mode="fixed", fixed_scale=1.0))
    _, steps = _policy_scales(sel, steps=3)
    for s in steps:
        assert torch.equal(s, torch.ones(_B))


def test_foveated_policy_scale_consumes_no_rng_in_fixed_and_per_rollout() -> None:
    """`fixed` returns a constant and `per_rollout` reads the frozen ctx scale, so neither
    draws — the fix cannot shift the RNG stream of any run using them."""
    for fsc in (FoveatedScaleConfig(mode="fixed", fixed_scale=2.0),
                FoveatedScaleConfig(mode="per_rollout", min_scale=0.4, max_scale=0.9)):
        sel = _fixation_policy(fsc)
        ctx = sel.start_rollout(t0_type=ViewpointType.FULL, batch_size=_B, device=_CPU)
        torch.manual_seed(1234)
        before = torch.rand(1).item()
        torch.manual_seed(1234)
        sel.select(vp_type=ViewpointType.RANDOM, ctx=ctx, t=1, batch_size=_B,
                   device=_CPU, state=None)
        assert torch.rand(1).item() == before, fsc.mode


def test_per_glimpse_gives_the_policy_a_fresh_scale_each_glimpse() -> None:
    """The one mode whose stream does change — and the one that was plainly broken, since
    the model sees a fresh scale every glimpse but the policy always asked for 1.0."""
    torch.manual_seed(0)
    sel = _fixation_policy(FoveatedScaleConfig(mode="per_glimpse", min_scale=0.4, max_scale=0.9))
    _, steps = _policy_scales(sel, steps=3)
    for s in steps:
        assert ((s >= 0.4) & (s <= 0.9)).all(), s
    assert not torch.allclose(steps[0], steps[1])  # genuinely redrawn per glimpse


def test_uniform_policy_still_owns_its_own_scale() -> None:
    """On the uniform patcher the candidate table's scale column IS the policy's choice
    (safe-box centre+scale pairs). The foveated fix must not touch it."""
    scores = torch.zeros(_B, _A)
    scores[:, 3] = 5.0
    sel = _make("argmax", scores)  # is_foveated=False
    ctx = sel.start_rollout(t0_type=ViewpointType.FULL, batch_size=_B, device=_CPU)
    vp = sel.select(vp_type=ViewpointType.RANDOM, ctx=ctx, t=1, batch_size=_B,
                    device=_CPU, state=None)
    assert torch.allclose(vp.scales, torch.full((_B,), 0.5))  # straight from vp_flat


def test_view_scales_extraction_left_the_random_path_unchanged() -> None:
    """`view_scales` was split out of `_foveated_random_vp`; the random viewpoint it
    produces must be identical for the same seed, or the pinned distributions move."""
    for mode in ("fixed", "per_rollout", "per_glimpse"):
        fsc = FoveatedScaleConfig(mode=mode, fixed_scale=1.5, min_scale=0.4, max_scale=0.9)
        rnd = RandomSelector(is_foveated=True, foveated_scale=fsc, min_viewpoint_scale=0.1)
        got = []
        for _ in range(2):
            torch.manual_seed(7)
            ctx = rnd.start_rollout(t0_type=ViewpointType.RANDOM, batch_size=_B, device=_CPU)
            vp = rnd.select(vp_type=ViewpointType.RANDOM, ctx=ctx, t=1, batch_size=_B,
                            device=_CPU, state=None)
            got.append((vp.centers.clone(), vp.scales.clone()))
        assert torch.equal(got[0][0], got[1][0]) and torch.equal(got[0][1], got[1][1]), mode
