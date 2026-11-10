"""Task-agnostic joint-policy builder for the unified harness (design §7 D-D crown jewel).

Generalizes ``train/joint.py::build_joint_policy`` (which hardwired the distill case:
the core model IS the canvit, INTRINSIC feature groups, no probe) to ANY task by
splitting two roles that distill happened to conflate:

  * ``canvit``       — the recurrent backbone; its ``.cfg`` gives canvas_dim + patcher_name.
  * ``encode_model`` — what the ``StateEncoder`` featurizes: an object exposing ``.canvit``
                       (and, for probe-entropy feature groups, ``.head``). For distill this
                       is ``SimpleNamespace(canvit=canvit)``; for ade20k/in1k it is the task
                       wrapper (``seg`` / ``clf``) so probe-aware features can reach the head.
  * ``feature_groups`` — the task's scorer features (``Task.policy_feature_groups()``):
                       INTRINSIC for distill/in1k, the full set (with ent/ent_delta) for the
                       spatial segmentation probe in ade20k.

Reuses the existing :class:`JointPolicy` container, objectives, selectors and scorer —
this only rewires the encoder so joint task+policy works for all three tasks (distill
already worked; this unlocks ade20k/in1k). The old ``build_joint_policy`` stays until
the big-bang cutover.
"""

from __future__ import annotations

import logging
import math
from types import SimpleNamespace
from typing import Any

import torch

from canvit.core.policy import (
    StateEncoder,
    ViewpointScorer,
    candidate_viewpoints,
    fixation_candidates,
)
from canvit.harness.config import FoveatedScaleConfig, JointPolicyConfig
from canvit.harness.policy.joint import JointPolicy
from canvit.harness.policy.rl import PG, VPG, Objective, QReg
from canvit.harness.rollout.selector import PolicySelector, RandomSelector

log = logging.getLogger(__name__)


def build_policy(
    *,
    canvit: Any,
    rl: JointPolicyConfig,
    feature_groups: tuple[str, ...],
    device: torch.device,
    canvas_grid: int,
    min_viewpoint_scale: float,
    foveated_scale: FoveatedScaleConfig,
    generator: torch.Generator,
    encode_model: Any | None = None,
) -> JointPolicy:
    """Assemble a :class:`JointPolicy` for any task. ``encode_model`` defaults to
    ``SimpleNamespace(canvit=canvit)`` (the distill case); pass the task wrapper
    (``seg``/``clf``) so probe-aware feature groups can read its head. ``feature_groups``
    comes from the task (INTRINSIC vs the probe-entropy set)."""
    if encode_model is None:
        encode_model = SimpleNamespace(canvit=canvit)

    # `feature_groups` is the TASK's default; `rl.feature_groups` overrides it when the user
    # set it. Applied here rather than in each task because nothing else reads
    # `Task.policy_feature_groups()` at runtime, so this is the single funnel — and because
    # before this the field was silently discarded by all three tasks (see its docstring).
    if rl.feature_groups is not None:
        if tuple(rl.feature_groups) != tuple(feature_groups):
            log.info("policy feature groups overridden: task default %s -> %s",
                     tuple(feature_groups), tuple(rl.feature_groups))
        feature_groups = tuple(rl.feature_groups)

    is_foveated = getattr(canvit.cfg, "patcher_name", "uniform") in ("foveated", "square")
    if is_foveated:
        cand = fixation_candidates(rl.centers_per_axis)
        n_scale, scales, action_space = 1, (1.0,), "fixation"
    else:
        cand = candidate_viewpoints(rl.scales, rl.centers_per_axis)
        n_scale, scales, action_space = cand.shape[0], rl.scales, "safebox"
    vp_flat = cand.reshape(-1, 3).to(device)

    if rl.objective == "qreg":
        obj: Objective = QReg(prime_on_policy=rl.prime_on_policy, dueling=rl.dueling)
    elif rl.objective == "vpg":
        obj = VPG(
            entropy_bonus=rl.vpg_entropy_bonus, reinforce_weight=rl.vpg_reinforce_weight,
            baseline_weight=rl.vpg_baseline_weight, normalize_advantage=rl.vpg_normalize_advantage,
            value_bias_init=rl.vpg_value_bias_init,
        )
    else:
        obj = PG(entropy_bonus=rl.entropy_bonus, entropy_target=rl.entropy_target, alpha_lr=rl.alpha_lr,
                 qprop=rl.qprop)

    # VPG is defined for the FOVEATED action space (autoreg_tryout only ever ran a foveated
    # model). It runs on uniform and is internally consistent there, but it is not the same
    # problem: the action space gains a SCALE dimension the original never had, and t0
    # becomes a full-scene view instead of a centred fixation. Warn rather than refuse —
    # the mode stays the user's choice — but say it at the point of use, because a
    # config difference that moves the metric must not live only in a docstring.
    if isinstance(obj, VPG) and not is_foveated:
        log.warning(
            "objective='vpg' on a UNIFORM-patcher model (%d actions = %d scales x %d^2 "
            "centres). The autoreg_tryout recipe this ports was defined for FOVEATED "
            "models, where the action space is a pure %d^2 fixation heatmap with no scale "
            "dimension and t0 is a centred fixation. The run is internally consistent but "
            "is NOT the ported recipe; use a foveated/square model_repo for that.",
            vp_flat.shape[0], n_scale, rl.centers_per_axis, rl.centers_per_axis)

    # REINFORCE differentiates log pi(a|s) under the distribution `a` was SAMPLED from.
    # select_bn_eval samples from a separate EVAL-mode BN forward while the loss reads the
    # TRAIN-mode scores, so with it on VPG's score-function term is off-policy (no
    # importance weight) — a biased gradient. Measured on real rollout states: the two BN
    # modes rank actions very differently (argmax agrees 8-14%). It defaults True because
    # that reproduces the qband checkpoints for QReg/PG; autoreg_tryout cannot hit this at
    # all (its policy heads carry no BatchNorm, so sampling and scoring are one tensor).
    # REFUSE rather than warn (owner's call, 2026-07-31): `select_bn_eval` defaults True,
    # so the biased combination was what a plain `--rl.objective vpg` GOT, and a log line
    # is not enough to stop someone acting on the run's numbers. Opting in is one explicit
    # flag; there is no legitimate reason to want the biased gradient.
    if isinstance(obj, VPG) and rl.select_bn_eval:
        raise ValueError(
            "objective='vpg' requires --rl.no-select-bn-eval. With select_bn_eval=True the "
            "glimpse is SAMPLED from an eval-mode BN forward but log pi is differentiated "
            "from the TRAIN-mode scores, so the REINFORCE gradient is off-policy and biased "
            "(the two BN modes' argmax agrees only 8-14% on real rollout states). The True "
            "default exists to reproduce the qband checkpoints for QReg/PG, which do not "
            "sample; autoreg_tryout has no BN in its policy heads and cannot hit this.")

    # VPG needs the value head unconditionally — that IS its baseline — so it is not the
    # user's `dueling` knob to turn off. QReg honors the knob; plain PG never gets one
    # (softmax is shift-invariant, so a V(s) it cannot train would be dead weight).
    dueling = isinstance(obj, VPG) or (isinstance(obj, QReg) and obj.dueling)
    scorer = ViewpointScorer(
        canvas_dim=canvit.cfg.canvas_dim, width=rl.width, n_scale=n_scale, scales=scales,
        centers_per_axis=rl.centers_per_axis, block_layers=rl.block_layers, groups=feature_groups,
        dueling=dueling, action_space=action_space, readout=rl.policy_readout,
        local_dim=canvit.local_dim,
    ).to(device)
    if isinstance(obj, VPG) and obj.value_bias_init is not None:
        assert scorer.vhead is not None
        torch.nn.init.constant_(scorer.vhead[-1].bias, obj.value_bias_init)
    scorer.train()

    # Q-Prop critic: a SECOND, fully independent ViewpointScorer, built identically to
    # the actor `scorer` above (same action space / grid / width / blocks), used only as
    # an off-policy Q-value estimator feeding `pg_loss`'s `crit_all` -- never for action
    # selection (the actor's own PolicySelector is untouched). None for every other
    # objective/config, which is what keeps this change a no-op for qreg/vpg/plain-pg.
    critic: ViewpointScorer | None = None
    if isinstance(obj, PG) and obj.qprop:
        critic = ViewpointScorer(
            canvas_dim=canvit.cfg.canvas_dim, width=rl.width, n_scale=n_scale, scales=scales,
            centers_per_axis=rl.centers_per_axis, block_layers=rl.block_layers, groups=feature_groups,
            dueling=rl.dueling, action_space=action_space, readout=rl.policy_readout,
            local_dim=canvit.local_dim,
        ).to(device)
        critic.train()

    encoder = StateEncoder(encode_model, canvas_grid=canvas_grid, feature_groups=feature_groups)

    random_sel = RandomSelector(
        is_foveated=is_foveated, foveated_scale=foveated_scale, min_viewpoint_scale=min_viewpoint_scale
    )
    policy_sel = PolicySelector(
        net=scorer, encoder=encoder, vp_flat=vp_flat, fallback=random_sel,
        mode="argmax" if isinstance(obj, QReg) else "sample",  # PG and VPG are both on-policy
        prime_on_policy=rl.prime_on_policy if isinstance(obj, QReg) else 1.0,
        feats_detached=rl.feats_detached, select_bn_eval=rl.select_bn_eval, generator=generator,
    )

    jp = JointPolicy(
        policy_selector=policy_sel, random_selector=random_sel, scorer=scorer, objective=obj,
        rl_weight=rl.rl_weight, keep_random_branch=rl.keep_random_branch,
        target_momentum=rl.target_momentum, device=device,
        prime_target=rl.prime_on_policy, prime_warmup=rl.policy_warmup_steps,
        critic=critic,
    )
    if isinstance(obj, PG) and obj.entropy_target is not None:
        jp.log_alpha = torch.tensor(math.log(obj.entropy_bonus), device=device)
    return jp


def check_credit_regime(*, joint: Any, spec: Any) -> None:
    """Reject the one spec cell a TERMINAL-reward objective cannot honor.

    VPG applies its loss once at rollout end, so every step's scorer graph must survive
    to that point. With ``policy_grad_to_backbone=True`` that graph runs through backbone
    activations, which a CHUNKED task backward frees at each chunk boundary — the
    trajectory backward would then raise mid-run. ``mode='full'``/``'none'`` (one backward
    per rollout, which is what autoreg itself does) is fine, as is any chunked run whose
    policy gradient is detached from the backbone.

    Lives here rather than in ``spec.check_spec`` because the objective comes from
    ``JointPolicyConfig``, not ``TrainSpec``; called by ``run()`` right after the policy is
    built, so this fails before a single step of training."""
    if joint is None or not joint.defers_credit:
        return
    if spec.bptt.mode == "chunked" and spec.policy_grad_to_backbone:
        raise ValueError(
            f"objective={type(joint.objective).__name__} uses a TERMINAL reward, which is "
            "incompatible with bptt.mode='chunked' + policy_grad_to_backbone=True: the "
            "per-chunk task backward frees the backbone activations its trajectory loss "
            "still needs. Use bptt.mode='full' (one backward per rollout), or set "
            "policy_grad_to_backbone=False to keep the policy graph head-local."
        )


__all__ = ["build_policy", "check_credit_regime"]
