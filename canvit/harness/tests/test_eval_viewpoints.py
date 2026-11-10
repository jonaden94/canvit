"""The shared validation-viewpoint knob.

The load-bearing claim of this module is NOT that the options work — it is that
unifying them changed nothing. Every task keeps the trajectory it had, so every
existing config, every specialize probe number and every exp22/23/24/26 val curve
stays comparable. That is what most of these tests pin, by running the OLD generator
and the new dispatch under the same seed and demanding identical tensors.

The rest cover the genuinely new capability: ``"policy"``, i.e. deploying a trained
scorer by argmax instead of replaying a fixed trajectory.
"""

from __future__ import annotations

import pytest
import torch

from canvit.harness.config import FoveatedScaleConfig
from canvit.harness.rollout.eval_viewpoints import (
    HISTORICAL_DEFAULTS,
    OPEN_LOOP,
    deploy_rollout_viewpoints,
    deploy_selector,
    open_loop_viewpoints,
    resolve,
)

_B, _N, _DEV = 3, 5, torch.device("cpu")
_FS = FoveatedScaleConfig()


# --- the defaults did not move ----------------------------------------------
@pytest.mark.parametrize("task,is_fov,expected", [
    ("distill", False, "coarse_to_fine"),
    ("distill", True, "fixation_grid"),   # quadtree scales are OOD for fixed-scale foveated
    ("ade20k", False, "random"),          # inherited from the specialize probe
    ("ade20k", True, "random"),
    ("in1k", False, "coarse_to_fine"),    # canvit_eval's deploy convention
    ("in1k", True, "coarse_to_fine"),     # the known footgun, kept for exp25 comparability
])
def test_auto_resolves_to_each_tasks_historical_trajectory(task, is_fov, expected):
    assert resolve("auto", task=task, is_foveated=is_fov) == expected


def test_every_task_config_still_defaults_to_auto():
    """A task whose config drifted off "auto" would silently change its own history."""
    from canvit.ade20k.config import Ade20kConfig
    from canvit.distill.config import Config
    from canvit.in1k.config import In1kConfig

    assert Config().eval_policy == "auto"
    assert Ade20kConfig().eval_policy == "auto"
    assert In1kConfig().eval_policy == "auto"


def test_distill_uniform_is_bit_identical_to_the_old_generator():
    from canvit.harness.rollout.viewpoint import make_eval_viewpoints

    torch.manual_seed(0)
    old = make_eval_viewpoints(_B, _DEV, n_viewpoints=_N)
    torch.manual_seed(0)
    new = open_loop_viewpoints("coarse_to_fine", batch_size=_B, device=_DEV, n=_N,
                               is_foveated=False, foveated_scale=_FS)
    _assert_same_trajectory(old, new)


def test_distill_foveated_is_bit_identical_to_the_old_generator():
    from canvit.harness.rollout.viewpoint import make_eval_viewpoints_foveated

    old = make_eval_viewpoints_foveated(_B, _DEV, n_viewpoints=_N, scale=2.0)
    new = open_loop_viewpoints("fixation_grid", batch_size=_B, device=_DEV, n=_N,
                               is_foveated=True, foveated_scale=_FS, foveated_eval_scale=2.0)
    _assert_same_trajectory(old, new)
    # the scale really is carried through — this is the whole point of fixation_grid
    assert torch.allclose(new[0].scales, torch.full((_B,), 2.0))


def test_ade20k_random_is_bit_identical_to_the_old_generator():
    from canvit.harness.rollout.eval_viewpoints import make_random_viewpoints

    torch.manual_seed(0)
    old = make_random_viewpoints(_B, _DEV, _N, min_scale=0.05, max_scale=1.0,
                                 start_with_full_scene=True, is_foveated=False,
                                 foveated_scale=_FS)
    torch.manual_seed(0)
    new = open_loop_viewpoints("random", batch_size=_B, device=_DEV, n=_N,
                               is_foveated=False, foveated_scale=_FS)
    _assert_same_trajectory(old, new)


def _assert_same_trajectory(old, new):
    assert len(old) == len(new)
    for i, (a, b) in enumerate(zip(old, new)):
        torch.testing.assert_close(a.centers, b.centers, rtol=0, atol=0, msg=f"centers t{i}")
        torch.testing.assert_close(a.scales, b.scales, rtol=0, atol=0, msg=f"scales t{i}")


# --- the option set is the same for every task -------------------------------
@pytest.mark.parametrize("policy", OPEN_LOOP)
def test_every_open_loop_option_is_usable_by_every_task(policy):
    """The point of the unification: no option belongs to one task any more."""
    vps = open_loop_viewpoints(policy, batch_size=_B, device=_DEV, n=_N,
                               is_foveated=False, foveated_scale=_FS)
    assert len(vps) == _N
    for vp in vps:
        assert vp.centers.shape == (_B, 2) and vp.scales.shape == (_B,)


def test_policy_cannot_be_precomputed():
    with pytest.raises(ValueError, match="closed-loop"):
        open_loop_viewpoints("policy", batch_size=_B, device=_DEV, n=_N,
                             is_foveated=False, foveated_scale=_FS)


def test_unknown_policy_is_rejected_at_resolve():
    with pytest.raises(AssertionError, match="unknown eval policy"):
        resolve("greedy_oracle", task="ade20k", is_foveated=False)


def test_every_task_in_the_table_covers_both_patchers():
    for task, entry in HISTORICAL_DEFAULTS.items():
        assert len(entry) == 2, task
        for value in entry:
            assert value in OPEN_LOOP, f"{task}: {value} is not an open-loop default"


# --- deploying a trained scorer ----------------------------------------------
def test_deploy_without_a_policy_says_so():
    """The failure mode this replaces is silent: validating a policy run on random
    glimpses. An explicit error is the whole improvement."""
    with pytest.raises(AssertionError, match="needs a trained scorer"):
        deploy_selector(None)


def test_deploy_forces_pure_argmax():
    """Deployment must not explore, whatever the trainer's current epsilon is: a
    validation that took random glimpses would not measure the deployed policy, and
    drawing epsilon would consume RNG that training also uses."""
    joint = _tiny_joint(prime_on_policy=0.25)
    sel = deploy_selector(joint)
    assert sel.mode == "argmax" and sel.prime_on_policy == 1.0
    # ... and the trainer's own selector is untouched (deploy_selector copies).
    assert joint.policy_selector.prime_on_policy == 0.25


def test_deploy_requires_a_full_t0_anchor():
    from canvit.harness.rollout.viewpoint import ViewpointType

    with pytest.raises(AssertionError, match="FULL t0 anchor"):
        deploy_rollout_viewpoints(joint=_tiny_joint(), advance=lambda vp, st, t: st,
                                  t0_type=ViewpointType.RANDOM, batch_size=_B,
                                  device=_DEV, n=_N)


def test_deploy_rollout_is_closed_loop_and_restores_train_mode():
    """Each glimpse must be chosen from the state the previous one produced (that is
    what 'closed loop' means), and a validation must not leave the scorer in eval mode
    — the next training step would then update no BatchNorm statistics."""
    from canvit.harness.rollout.viewpoint import ViewpointType

    joint = _tiny_joint()
    joint.scorer.train()
    seen_states, advanced = [], []

    def advance(vp, state, t):
        seen_states.append(state)
        advanced.append(vp)
        return _FakeState(t)

    vps = deploy_rollout_viewpoints(joint=joint, advance=advance, t0_type=ViewpointType.FULL,
                                    batch_size=_B, device=_DEV, n=4)
    assert len(vps) == 4 and len(advanced) == 4
    assert seen_states[0] is None, "t0 must hand the task a None state to initialise"
    # every later selection saw the state the previous glimpse returned
    assert [s.t for s in seen_states[1:]] == [0, 1, 2]
    assert joint.scorer.training, "scorer left in eval mode would freeze its BatchNorm"


class _FakeState:
    def __init__(self, t):
        self.t = t


def _tiny_joint(prime_on_policy: float = 1.0):
    """A JointPolicy small enough to run on CPU: 2 candidates, a scorer that reads a
    stub encoder. Only the selection path is exercised here."""
    from dataclasses import dataclass

    import torch.nn as nn

    from canvit.harness.config import FoveatedScaleConfig
    from canvit.harness.rollout.selector import PolicySelector, RandomSelector

    class _Scorer(nn.Module):
        def __init__(self):
            super().__init__()
            self.w = nn.Parameter(torch.tensor([[1.0, -1.0]]))

        def forward(self, feats):
            return self.w.expand(feats.shape[0], -1)

    class _Encoder:
        def reset(self):
            pass

        def __call__(self, state, **_kw):
            return torch.zeros(_B, 1)

    scorer = _Scorer()
    sel = PolicySelector(
        net=scorer, encoder=_Encoder(),
        vp_flat=torch.tensor([[0.0, 0.0, 0.5], [0.5, 0.5, 0.5]]),
        fallback=RandomSelector(is_foveated=False, foveated_scale=FoveatedScaleConfig(),
                                min_viewpoint_scale=0.05),
        prime_on_policy=prime_on_policy,
    )

    @dataclass
    class _Joint:
        policy_selector: PolicySelector
        scorer: nn.Module

    return _Joint(policy_selector=sel, scorer=scorer)


# --- selection metric follows the eval policy --------------------------------
@pytest.mark.parametrize("eval_policy,expected", [
    ("auto", "miou_final"),
    ("random", "miou_final"),
    ("policy", "neg_ce_mean"),
])
def test_ade20k_best_metric_follows_the_eval_policy(eval_policy, expected):
    """Selecting a POLICY run on mIoU would put our checkpoints on a different axis
    from the qband reference band, which is defined on mean t1..tH CE."""
    from dataclasses import replace

    from canvit.ade20k.config import Ade20kConfig
    from canvit.ade20k.task import Ade20kRunTask

    task = Ade20kRunTask(replace(Ade20kConfig(), eval_policy=eval_policy))
    assert task.best_metric == expected


# --- the off-scale warning at the point of use -------------------------------
# HISTORICAL_DEFAULTS sends a foveated in1k run to unpinned coarse_to_fine ON PURPOSE, to
# keep exp25/exp29/exp33 comparable, and its docstring's "(or `full`)" advice is off-scale
# too. Measured cost of either: -0.114 top1 on in1k, -0.128 mIoU at t9 on ade20k. A doc
# paragraph did not stop it happening, so the warning has to be at the point of use.
# Checks the GENERATED SCALES, not the policy name, so it stays right for the combinations
# Stage 3 opens up.

_FIXED2 = FoveatedScaleConfig(mode="fixed", fixed_scale=2.0)


def _emit(policy: str, *, fs=_FIXED2, is_fov=True, override=None):
    torch.manual_seed(0)
    from canvit.harness.rollout.eval_viewpoints import _warn_off_scale
    _warn_off_scale.cache_clear()   # the warning is once-per-process
    return open_loop_viewpoints(
        policy, batch_size=_B, device=_DEV, n=_N, is_foveated=is_fov, foveated_scale=fs,
        foveated_eval_scale=fs.fixed_scale, override_scale=override,
    )


@pytest.mark.parametrize("policy", ["coarse_to_fine", "full"])
def test_off_scale_policies_warn_on_a_fixed_scale_foveated_model(policy, caplog):
    with caplog.at_level("WARNING"):
        _emit(policy)
    assert "OUT OF DISTRIBUTION" in caplog.text
    assert "eval-override-scale 2" in caplog.text  # names the remedy, with the value


@pytest.mark.parametrize("policy", ["fixation_grid", "random"])
def test_in_distribution_policies_stay_quiet(policy, caplog):
    """fixation_grid deploys at foveated_eval_scale and random goes through RandomSelector,
    so both already sit at the training scale. A warning here would be noise."""
    with caplog.at_level("WARNING"):
        _emit(policy)
    assert "OUT OF DISTRIBUTION" not in caplog.text


def test_an_explicit_pin_silences_it(caplog):
    """Pinning to the training scale is the documented remedy, so taking it must not warn —
    and the pin must actually reach every glimpse, which is what makes it in-distribution."""
    with caplog.at_level("WARNING"):
        vps = _emit("coarse_to_fine", override=2.0)
    scales = torch.cat([v.scales.reshape(-1) for v in vps])
    assert torch.allclose(scales, torch.full_like(scales, 2.0))
    assert "OUT OF DISTRIBUTION" not in caplog.text


def test_uniform_models_never_warn(caplog):
    """A uniform model's OOD axis is the glimpse crop in pixels, not the view scale."""
    with caplog.at_level("WARNING"):
        _emit("coarse_to_fine", fs=_FS, is_fov=False)
    assert "OUT OF DISTRIBUTION" not in caplog.text


def test_sampled_scale_modes_never_warn(caplog):
    """per_rollout / per_glimpse models saw a range of scales in training, so a varying
    trajectory is in distribution for them by construction."""
    with caplog.at_level("WARNING"):
        _emit("coarse_to_fine", fs=FoveatedScaleConfig(mode="per_glimpse"))
    assert "OUT OF DISTRIBUTION" not in caplog.text


# --- the newly exposed preset and the t0 modifier -----------------------------
# Stage 3 exposes `fine_to_coarse` (core has had the generator all along; this repo never
# surfaced it) and adds `t0`, the knob that separates canvit's `random` from
# canvit_eval's — a 0.0205 difference at t0 measured in Stage 0 (F3). Both randoms are
# legitimate; what is not legitimate is one flag meaning two things.

def test_fine_to_coarse_walks_the_quadtree_the_other_way():
    """Coarsest-last: the last glimpse of an unanchored f2c is the full scene, which is the
    first glimpse of c2f. If these ever come out the same, the reversal was lost."""
    torch.manual_seed(0)
    f2c = open_loop_viewpoints("fine_to_coarse", batch_size=_B, device=_DEV, n=5,
                               is_foveated=False, foveated_scale=_FS, t0="trajectory")
    assert f2c[-1].scales[0].item() == pytest.approx(1.0)   # ends full
    assert f2c[0].scales[0].item() < 1.0                    # starts fine
    torch.manual_seed(0)
    c2f = open_loop_viewpoints("coarse_to_fine", batch_size=_B, device=_DEV, n=5,
                               is_foveated=False, foveated_scale=_FS)
    assert not torch.equal(f2c[0].scales, c2f[0].scales)


def test_t0_full_anchor_prepends_the_full_scene():
    for policy in ("random", "fine_to_coarse"):
        torch.manual_seed(0)
        anchored = open_loop_viewpoints(policy, batch_size=_B, device=_DEV, n=5,
                                        is_foveated=False, foveated_scale=_FS,
                                        t0="full_anchor")
        torch.manual_seed(0)
        bare = open_loop_viewpoints(policy, batch_size=_B, device=_DEV, n=5,
                                    is_foveated=False, foveated_scale=_FS, t0="trajectory")
        assert anchored[0].scales[0].item() == pytest.approx(1.0), policy
        assert len(anchored) == len(bare) == 5, policy
        assert not torch.equal(anchored[1].centers, bare[1].centers), policy


def test_unset_t0_is_an_exact_no_op_for_every_preset():
    """The whole point: adding the knob must not move a single existing number. Compares the
    default (unset) against the value each preset historically used."""
    for policy in OPEN_LOOP:
        torch.manual_seed(0)
        default = open_loop_viewpoints(policy, batch_size=_B, device=_DEV, n=5,
                                       is_foveated=False, foveated_scale=_FS)
        torch.manual_seed(0)
        explicit = open_loop_viewpoints(
            policy, batch_size=_B, device=_DEV, n=5, is_foveated=False, foveated_scale=_FS,
            **({"t0": "full_anchor"} if policy in ("random", "fine_to_coarse") else {}))
        assert len(default) == len(explicit), policy
        for a, b in zip(default, explicit):
            assert torch.equal(a.centers, b.centers) and torch.equal(a.scales, b.scales), policy


@pytest.mark.parametrize("policy", ["coarse_to_fine", "full", "fixation_grid"])
def test_t0_on_a_preset_that_cannot_honour_it_is_a_hard_error(policy):
    """Silently dropping the flag would hand back a number the caller reads as having come
    from the config they typed. Errors name the flag and the presets it applies to."""
    with pytest.raises(ValueError, match="does not apply"):
        open_loop_viewpoints(policy, batch_size=_B, device=_DEV, n=5, is_foveated=False,
                             foveated_scale=_FS, t0="trajectory")


def test_override_scale_is_now_a_field_on_all_three_tasks():
    """F4: it was ade20k-only, and in1k is precisely the task whose historical default is
    the off-scale one. Same knob, each task keeping its own default (None = off)."""
    from canvit.ade20k.config import Ade20kConfig
    from canvit.distill.config import Config as DistillConfig
    from canvit.in1k.config import In1kConfig

    for cfg in (Ade20kConfig, DistillConfig, In1kConfig):
        assert "eval_override_scale" in cfg.__dataclass_fields__, cfg.__name__
        assert cfg.__dataclass_fields__["eval_override_scale"].default is None, cfg.__name__
