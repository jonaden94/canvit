"""Smoke tests: all three peer tasks drive the unified rollout engine, and the
TrainSpec grad regime routes gradients to exactly the right modules (design §5).

This is the concrete payoff of the harness — one engine, three tasks, the same
freeze/finetune knobs — validated on tiny CPU models. Joint task+policy per task
(the crown-jewel new capability) needs the per-task policy builder and is smoke-
tested once that lands (loop phase); here we cover the task-only cells.
"""

import torch

from canvit import CanViTForPretraining, CanViTForPretrainingConfig
from canvit.ade20k.data import IGNORE_LABEL, NUM_CLASSES
from canvit.ade20k.task import POLICY_FEATURE_GROUPS as ADE_GROUPS
from canvit.ade20k.task import BoundAde20kTask
from canvit.core import (
    CanViTForImageClassification,
    CanViTForSemanticSegmentation,
)
from canvit.distill.loss import DistillTask
from canvit.distill.task import POLICY_FEATURE_GROUPS as DISTILL_GROUPS
from canvit.distill.task import BoundDistillTask
from canvit.harness.config import FoveatedScaleConfig, JointPolicyConfig
from canvit.harness.policy import build_policy
from canvit.harness.rollout import run_rollout
from canvit.harness.rollout.selector import RandomSelector
from canvit.harness.rollout.viewpoint import ViewpointType
from canvit.harness.spec import BpttSpec
from canvit.in1k.task import POLICY_FEATURE_GROUPS as IN1K_GROUPS
from canvit.in1k.task import BoundIn1kTask

_B, _G, _IMG, _D, _C = 2, 8, 224, 384, 10
_DEV = torch.device("cpu")


def _selector() -> RandomSelector:
    return RandomSelector(is_foveated=False, foveated_scale=FoveatedScaleConfig(), min_viewpoint_scale=0.05)


def _zero_grads(*modules):
    for m in modules:
        for p in m.parameters():
            p.grad = None


def _has_grad(module) -> bool:
    return any(p.grad is not None and p.grad.abs().sum() > 0 for p in module.parameters())


def _no_grad(module) -> bool:
    return all(p.grad is None for p in module.parameters())


# --------------------------------------------------------------------------- #
# Distill (finetune only — its heads are inside the forward, no probe cell).
# --------------------------------------------------------------------------- #
def _distill_model() -> CanViTForPretraining:
    torch.manual_seed(0)
    from canvit.core import create_backbone
    return CanViTForPretraining(
        backbone=create_backbone("vits16"), cfg=CanViTForPretrainingConfig(teacher_dim=_D),
        glimpse_size_px=128, backbone_name="vits16", canvas_patch_grid_sizes=[_G],
    ).to(_DEV)


def test_distill_finetune_trains_backbone():
    torch.manual_seed(1)
    model = _distill_model()
    task = BoundDistillTask(DistillTask(
        scene_target=torch.randn(_B, _G * _G, _D), cls_target=torch.randn(_B, _D),
        enable_scene_patches_loss=True, enable_scene_cls_loss=True,
    ))
    _zero_grads(model)
    r = run_rollout(
        model=model, images=torch.randn(_B, 3, _IMG, _IMG), task=task, selector=_selector(),
        bptt=BpttSpec(mode="full", horizon=2), branches=[ViewpointType.FULL, ViewpointType.RANDOM],
        canvas_grid_size=_G, amp_ctx=torch.enable_grad(),
    )
    assert torch.isfinite(r.total_loss)
    assert _has_grad(model.backbone)


# --------------------------------------------------------------------------- #
# ADE20K (probe: frozen backbone / trains head ; finetune: trains backbone).
# --------------------------------------------------------------------------- #
def _seg() -> CanViTForSemanticSegmentation:
    torch.manual_seed(0)
    return CanViTForSemanticSegmentation(backbone_name="vits16", model_config={}, num_classes=NUM_CLASSES).to(_DEV)


def _seg_masks() -> torch.Tensor:
    m = torch.randint(0, NUM_CLASSES, (_B, _IMG, _IMG), device=_DEV)
    m[:, :8] = IGNORE_LABEL
    return m


def test_ade20k_probe_freezes_backbone_trains_head():
    torch.manual_seed(1)
    seg = _seg()
    seg.canvit.requires_grad_(False)
    seg.canvit.eval()
    task = BoundAde20kTask(seg=seg, masks=_seg_masks(), canvas_grid=_G)
    _zero_grads(seg)
    r = run_rollout(
        model=seg, images=torch.randn(_B, 3, _IMG, _IMG), task=task, selector=_selector(),
        bptt=BpttSpec(mode="none", horizon=2), branches=[ViewpointType.FULL],
        canvas_grid_size=_G, amp_ctx=torch.enable_grad(),
    )
    assert torch.isfinite(r.total_loss)
    assert _has_grad(seg.head)
    assert _no_grad(seg.canvit)


def test_ade20k_finetune_trains_backbone():
    torch.manual_seed(1)
    seg = _seg()
    task = BoundAde20kTask(seg=seg, masks=_seg_masks(), canvas_grid=_G)
    _zero_grads(seg)
    r = run_rollout(
        model=seg, images=torch.randn(_B, 3, _IMG, _IMG), task=task, selector=_selector(),
        bptt=BpttSpec(mode="full", horizon=2), branches=[ViewpointType.FULL],
        canvas_grid_size=_G, amp_ctx=torch.enable_grad(),
    )
    assert torch.isfinite(r.total_loss)
    assert _has_grad(seg.canvit)  # NEW capability: ADE20K full-model finetune


# --------------------------------------------------------------------------- #
# IN1k (frozen probe / finetune).
# --------------------------------------------------------------------------- #
def _clf() -> CanViTForImageClassification:
    torch.manual_seed(0)
    return CanViTForImageClassification(
        backbone_name="vits16", model_config={}, n_classes=_C, glimpse_grid_size=_G,
    ).to(_DEV)


def test_in1k_frozen_trains_head_only():
    torch.manual_seed(1)
    clf = _clf()
    clf.canvit.requires_grad_(False)
    clf.canvit.eval()
    task = BoundIn1kTask(clf=clf, targets=torch.randint(0, _C, (_B,)), canvas_grid=_G)
    _zero_grads(clf)
    r = run_rollout(
        model=clf, images=torch.randn(_B, 3, _IMG, _IMG), task=task, selector=_selector(),
        bptt=BpttSpec(mode="none", horizon=2), branches=[ViewpointType.FULL],
        canvas_grid_size=_G, amp_ctx=torch.enable_grad(),
    )
    assert torch.isfinite(r.total_loss)
    assert _has_grad(clf.head)
    assert _no_grad(clf.canvit)


def test_in1k_finetune_trains_backbone():
    torch.manual_seed(1)
    clf = _clf()
    task = BoundIn1kTask(clf=clf, targets=torch.randint(0, _C, (_B,)), canvas_grid=_G)
    _zero_grads(clf)
    r = run_rollout(
        model=clf, images=torch.randn(_B, 3, _IMG, _IMG), task=task, selector=_selector(),
        bptt=BpttSpec(mode="full", horizon=2), branches=[ViewpointType.FULL],
        canvas_grid_size=_G, amp_ctx=torch.enable_grad(),
    )
    assert torch.isfinite(r.total_loss)
    assert _has_grad(clf.canvit)  # NEW capability: IN1k policy-free finetune via the unified engine


# --------------------------------------------------------------------------- #
# Joint task+policy through the unified engine (the P4b mechanism). All three tasks go
# through the probe-aware `build_policy`; distill passes encode_model=None, which is
# exactly what `distill/task.py::build_policy` does in production.
# --------------------------------------------------------------------------- #
def test_distill_joint_trains_task_and_scorer():
    torch.manual_seed(1)
    model = _distill_model()
    gen = torch.Generator(device=_DEV)
    gen.manual_seed(0)
    joint = build_policy(
        canvit=model, rl=JointPolicyConfig(objective="qreg"), feature_groups=DISTILL_GROUPS,
        device=_DEV, canvas_grid=_G, min_viewpoint_scale=0.05,
        foveated_scale=FoveatedScaleConfig(), generator=gen, encode_model=None,
    )
    task = BoundDistillTask(DistillTask(
        scene_target=torch.randn(_B, _G * _G, _D), cls_target=torch.randn(_B, _D),
        enable_scene_patches_loss=True, enable_scene_cls_loss=True,
    ))
    _zero_grads(model, joint.scorer)
    r = run_rollout(
        model=model, images=torch.randn(_B, 3, _IMG, _IMG), task=task, selector=_selector(),
        bptt=BpttSpec(mode="chunked", chunk_size=2, horizon=4), branches=[ViewpointType.FULL],
        canvas_grid_size=_G, amp_ctx=torch.enable_grad(), joint=joint,
    )
    assert torch.isfinite(r.total_loss)
    assert r.policy_metrics is not None and "reward_frac" in r.policy_metrics
    assert _has_grad(joint.scorer)   # policy loss trained the scorer
    assert _has_grad(model.backbone)  # task (distill) loss still trained the backbone


def _joint_for(*, canvit, encode_model, groups):
    gen = torch.Generator(device=_DEV)
    gen.manual_seed(0)
    return build_policy(
        canvit=canvit, rl=JointPolicyConfig(objective="qreg"), feature_groups=groups,
        device=_DEV, canvas_grid=_G, min_viewpoint_scale=0.05, foveated_scale=FoveatedScaleConfig(),
        generator=gen, encode_model=encode_model,
    )


def test_ade20k_joint_trains_probe_and_scorer():
    # CROWN JEWEL: ADE20K joint task+policy (frozen backbone, train probe + policy) —
    # a capability the old ade20k trainer never had. Probe-aware scorer (ent features).
    torch.manual_seed(1)
    seg = _seg()
    seg.canvit.requires_grad_(False)
    seg.canvit.eval()
    joint = _joint_for(canvit=seg.canvit, encode_model=seg, groups=ADE_GROUPS)
    task = BoundAde20kTask(seg=seg, masks=_seg_masks(), canvas_grid=_G)
    _zero_grads(seg, joint.scorer)
    r = run_rollout(
        model=seg, images=torch.randn(_B, 3, _IMG, _IMG), task=task, selector=_selector(),
        bptt=BpttSpec(mode="none", horizon=4), branches=[ViewpointType.FULL],
        canvas_grid_size=_G, amp_ctx=torch.enable_grad(), joint=joint,
    )
    assert torch.isfinite(r.total_loss)
    assert r.policy_metrics is not None
    assert _has_grad(joint.scorer)   # policy loss trained the scorer
    assert _has_grad(seg.head)       # task CE trained the probe head
    assert _no_grad(seg.canvit)      # backbone stayed frozen


def test_in1k_joint_trains_head_and_scorer():
    # IN1k joint task+policy (frozen backbone, train head + policy); intrinsic scorer feats.
    torch.manual_seed(1)
    clf = _clf()
    clf.canvit.requires_grad_(False)
    clf.canvit.eval()
    joint = _joint_for(canvit=clf.canvit, encode_model=clf, groups=IN1K_GROUPS)
    task = BoundIn1kTask(clf=clf, targets=torch.randint(0, _C, (_B,)), canvas_grid=_G)
    _zero_grads(clf, joint.scorer)
    r = run_rollout(
        model=clf, images=torch.randn(_B, 3, _IMG, _IMG), task=task, selector=_selector(),
        bptt=BpttSpec(mode="none", horizon=4), branches=[ViewpointType.FULL],
        canvas_grid_size=_G, amp_ctx=torch.enable_grad(), joint=joint,
    )
    assert torch.isfinite(r.total_loss)
    assert r.policy_metrics is not None
    assert _has_grad(joint.scorer)
    assert _has_grad(clf.head)
    assert _no_grad(clf.canvit)


# --------------------------------------------------------------------------- #
# VPG (objective="vpg"): terminal reward => DEFERRED credit. The engine buffers the
# trajectory and applies one policy loss at rollout end, so these tests check the thing
# unit tests cannot: that the deferred path still reaches every module the spec says it
# should, on the real tasks, across the bptt regimes.
# --------------------------------------------------------------------------- #
def _vpg_joint(*, canvit, encode_model, groups, **rl_kw):
    gen = torch.Generator(device=_DEV)
    gen.manual_seed(0)
    # select_bn_eval=False is REQUIRED for vpg (build_policy raises otherwise): sampling
    # from an eval-mode BN forward while differentiating log pi from the train-mode scores
    # is an off-policy, biased REINFORCE gradient. The knob's True default serves QReg/PG,
    # which do not sample. Overridable so a test can assert the refusal.
    rl_kw.setdefault("select_bn_eval", False)
    return build_policy(
        canvit=canvit, rl=JointPolicyConfig(objective="vpg", **rl_kw),
        feature_groups=groups, device=_DEV, canvas_grid=_G, min_viewpoint_scale=0.05,
        foveated_scale=FoveatedScaleConfig(), generator=gen, encode_model=encode_model,
    )


def test_vpg_scorer_gets_a_value_head_and_samples_on_policy():
    seg = _seg()
    joint = _vpg_joint(canvit=seg.canvit, encode_model=seg, groups=ADE_GROUPS)
    assert joint.defers_credit
    # VPG forces dueling regardless of the config knob — the vhead IS its baseline.
    assert joint.scorer.vhead is not None
    assert joint.policy_selector.mode == "sample"  # on-policy, like PG (not QReg's argmax)


def test_vpg_ade20k_policy_only_trains_scorer_and_value_head():
    """The flagship cell: frozen backbone + frozen probe, train the policy only
    (= autoreg's rl_grad_mode='heads_only' with the task net also frozen)."""
    torch.manual_seed(1)
    seg = _seg()
    seg.requires_grad_(False)
    seg.eval()
    joint = _vpg_joint(canvit=seg.canvit, encode_model=seg, groups=ADE_GROUPS)
    task = BoundAde20kTask(seg=seg, masks=_seg_masks(), canvas_grid=_G)
    _zero_grads(seg, joint.scorer)
    r = run_rollout(
        model=seg, images=torch.randn(_B, 3, _IMG, _IMG), task=task, selector=_selector(),
        bptt=BpttSpec(mode="none", horizon=4), branches=[ViewpointType.FULL],
        canvas_grid_size=_G, amp_ctx=torch.enable_grad(), joint=joint,
    )
    assert torch.isfinite(r.total_loss)
    assert _has_grad(joint.scorer)
    assert _has_grad(joint.scorer.vhead)  # the baseline MSE reached V(s)
    assert _no_grad(seg)                  # nothing leaked into the frozen task net
    # The objective's own diagnostics ride through policy_metrics (loop.py logs them all).
    assert r.policy_metrics is not None
    for k in ("policy_entropy", "value_mean", "adv_std", "loss_reinforce", "loss_baseline"):
        assert k in r.policy_metrics, k


def test_vpg_ade20k_joint_trains_probe_and_policy():
    """Joint task+policy: the CE trains the probe while the terminal reward trains the
    policy — autoreg's use_aux_ce_loss=True + RL, in one rollout."""
    torch.manual_seed(1)
    seg = _seg()
    seg.canvit.requires_grad_(False)
    seg.canvit.eval()
    joint = _vpg_joint(canvit=seg.canvit, encode_model=seg, groups=ADE_GROUPS)
    task = BoundAde20kTask(seg=seg, masks=_seg_masks(), canvas_grid=_G)
    _zero_grads(seg, joint.scorer)
    r = run_rollout(
        model=seg, images=torch.randn(_B, 3, _IMG, _IMG), task=task, selector=_selector(),
        bptt=BpttSpec(mode="none", horizon=4), branches=[ViewpointType.FULL],
        canvas_grid_size=_G, amp_ctx=torch.enable_grad(), joint=joint,
    )
    assert torch.isfinite(r.total_loss)
    assert _has_grad(joint.scorer)
    assert _has_grad(seg.head)


def test_vpg_policy_grad_reaches_the_backbone_when_coupled():
    """policy_grad_to_backbone=True (autoreg's rl_grad_mode='all'): the policy loss must
    reshape the trunk too. Requires bptt='full' — chunked+coupled is refused in run.py
    because the chunk backward frees activations the deferred loss still needs."""
    torch.manual_seed(1)
    seg = _seg()
    joint = _vpg_joint(canvit=seg.canvit, encode_model=seg, groups=ADE_GROUPS,
                       feats_detached=False)
    task = BoundAde20kTask(seg=seg, masks=_seg_masks(), canvas_grid=_G)
    _zero_grads(seg, joint.scorer)
    r = run_rollout(
        model=seg, images=torch.randn(_B, 3, _IMG, _IMG), task=task, selector=_selector(),
        bptt=BpttSpec(mode="full", horizon=3), branches=[ViewpointType.FULL],
        canvas_grid_size=_G, amp_ctx=torch.enable_grad(), joint=joint,
    )
    assert torch.isfinite(r.total_loss)
    assert _has_grad(joint.scorer)
    assert _has_grad(seg.canvit)


def test_vpg_survives_chunked_bptt_when_the_policy_graph_is_head_local():
    """The case the run.py guard ALLOWS: chunked task backward + feats_detached=True. Each
    step's scorer graph is head-local (feats came from a no_grad forward), so it survives
    the chunk detach and the trajectory loss still backwards at the end."""
    torch.manual_seed(1)
    seg = _seg()
    joint = _vpg_joint(canvit=seg.canvit, encode_model=seg, groups=ADE_GROUPS)
    task = BoundAde20kTask(seg=seg, masks=_seg_masks(), canvas_grid=_G)
    _zero_grads(seg, joint.scorer)
    r = run_rollout(
        model=seg, images=torch.randn(_B, 3, _IMG, _IMG), task=task, selector=_selector(),
        bptt=BpttSpec(mode="chunked", chunk_size=2, horizon=4), branches=[ViewpointType.FULL],
        canvas_grid_size=_G, amp_ctx=torch.enable_grad(), joint=joint,
    )
    assert torch.isfinite(r.total_loss)
    assert _has_grad(joint.scorer)


def test_vpg_in1k_policy_only_is_task_agnostic():
    """The port is written against `task.per_image_loss`, so the classification task —
    the literal analogue of autoreg's ImageNet setup — needs no VPG-specific code."""
    torch.manual_seed(1)
    clf = _clf()
    clf.requires_grad_(False)
    clf.eval()
    joint = _vpg_joint(canvit=clf.canvit, encode_model=clf, groups=IN1K_GROUPS)
    task = BoundIn1kTask(clf=clf, targets=torch.randint(0, _C, (_B,)), canvas_grid=_G)
    _zero_grads(clf, joint.scorer)
    r = run_rollout(
        model=clf, images=torch.randn(_B, 3, _IMG, _IMG), task=task, selector=_selector(),
        bptt=BpttSpec(mode="none", horizon=3), branches=[ViewpointType.FULL],
        canvas_grid_size=_G, amp_ctx=torch.enable_grad(), joint=joint,
    )
    assert torch.isfinite(r.total_loss)
    assert _has_grad(joint.scorer)
    assert _no_grad(clf)


def test_vpg_reward_is_the_negated_terminal_task_loss():
    """R = -per_image_loss(LAST glimpse), not the per-glimpse fractional reduction the
    inline objectives use. Pinned because it is the single biggest semantic difference and
    a silent regression to fractional reward would still look healthy in wandb."""
    torch.manual_seed(1)
    seg = _seg()
    seg.requires_grad_(False)
    seg.eval()
    joint = _vpg_joint(canvit=seg.canvit, encode_model=seg, groups=ADE_GROUPS)
    masks = _seg_masks()
    task = BoundAde20kTask(seg=seg, masks=masks, canvas_grid=_G)
    _zero_grads(seg, joint.scorer)
    r = run_rollout(
        model=seg, images=torch.randn(_B, 3, _IMG, _IMG), task=task, selector=_selector(),
        bptt=BpttSpec(mode="none", horizon=3), branches=[ViewpointType.FULL],
        canvas_grid_size=_G, amp_ctx=torch.enable_grad(), joint=joint,
    )
    assert r.policy_metrics is not None
    reward = float(r.policy_metrics["reward_frac"])
    # A fresh random probe on ADE20K sits near chance CE = ln(150) ~ 5.0, so R is a large
    # NEGATIVE number. The inline fractional reward would be a small value around 0.
    assert reward < -1.0, reward


# --------------------------------------------------------------------------- #
# VPG on the FOVEATED patcher — the configuration the port actually targets.
# autoreg_tryout only ever ran a foveated model, and foveated mode is what makes the
# action space faithful: a pure fixation heatmap, no scale dimension, centred t0.
# --------------------------------------------------------------------------- #
def _seg_foveated() -> CanViTForSemanticSegmentation:
    from canvit.core.patcher import FoveatedPatcherConfig

    torch.manual_seed(0)
    return CanViTForSemanticSegmentation(
        backbone_name="vits16", num_classes=NUM_CLASSES,
        model_config={"patcher_name": "foveated", "foveated_patcher": FoveatedPatcherConfig()},
    ).to(_DEV)


def _fov_selector() -> RandomSelector:
    return RandomSelector(is_foveated=True, foveated_scale=FoveatedScaleConfig(),
                          min_viewpoint_scale=0.05)


def test_vpg_foveated_action_space_is_a_pure_fixation_heatmap():
    """The faithfulness claim, pinned: cpa**2 fixation centres, NO scale dimension. On the
    uniform patcher the same config yields n_scale * cpa**2 centre-AND-scale pairs, which is
    a different action space from the one autoreg's heatmap_head defined."""
    seg = _seg_foveated()
    joint = _vpg_joint(canvit=seg.canvit, encode_model=seg, groups=ADE_GROUPS,
                       centers_per_axis=16)
    assert joint.scorer.action_space == "fixation"
    assert joint.policy_selector.vp_flat.shape == (16 * 16, 3)
    assert joint.policy_selector.vp_flat[:, 2].unique().tolist() == [1.0]  # no scale dim

    uniform = _vpg_joint(canvit=_seg().canvit, encode_model=_seg(), groups=ADE_GROUPS,
                         centers_per_axis=16)
    assert uniform.scorer.action_space == "safebox"
    assert uniform.policy_selector.vp_flat.shape[0] == 2 * 16 * 16  # 2 scales x centres


def test_vpg_foveated_centers_per_axis_reproduces_autoregs_grid():
    """autoreg's splatting_grid_size=14 => a 14x14 heatmap. The action grid is a knob, so
    the original geometry is reachable exactly."""
    seg = _seg_foveated()
    joint = _vpg_joint(canvit=seg.canvit, encode_model=seg, groups=ADE_GROUPS,
                       centers_per_axis=14)
    assert joint.policy_selector.vp_flat.shape == (14 * 14, 3)


def test_vpg_foveated_policy_only_trains_scorer_and_value_head():
    """The flagship cell in the TARGET configuration: foveated model, frozen backbone +
    frozen probe, train the policy only (= autoreg's rl_grad_mode='heads_only')."""
    torch.manual_seed(1)
    seg = _seg_foveated()
    seg.requires_grad_(False)
    seg.eval()
    joint = _vpg_joint(canvit=seg.canvit, encode_model=seg, groups=ADE_GROUPS)
    task = BoundAde20kTask(seg=seg, masks=_seg_masks(), canvas_grid=_G)
    _zero_grads(seg, joint.scorer)
    r = run_rollout(
        model=seg, images=torch.randn(_B, 3, _IMG, _IMG), task=task, selector=_fov_selector(),
        bptt=BpttSpec(mode="none", horizon=4), branches=[ViewpointType.FULL],
        canvas_grid_size=_G, amp_ctx=torch.enable_grad(), joint=joint,
    )
    assert torch.isfinite(r.total_loss)
    assert _has_grad(joint.scorer)
    assert _has_grad(joint.scorer.vhead)
    assert _no_grad(seg)
    assert r.policy_metrics is not None
    # A fresh scorer over 256 candidates starts ~uniform: H ~ ln(256) = 5.545 nats.
    import math
    assert abs(float(r.policy_metrics["policy_entropy"]) - math.log(256)) < 0.3


def test_vpg_foveated_joint_trains_probe_and_policy():
    torch.manual_seed(1)
    seg = _seg_foveated()
    seg.canvit.requires_grad_(False)
    seg.canvit.eval()
    joint = _vpg_joint(canvit=seg.canvit, encode_model=seg, groups=ADE_GROUPS)
    task = BoundAde20kTask(seg=seg, masks=_seg_masks(), canvas_grid=_G)
    _zero_grads(seg, joint.scorer)
    r = run_rollout(
        model=seg, images=torch.randn(_B, 3, _IMG, _IMG), task=task, selector=_fov_selector(),
        bptt=BpttSpec(mode="none", horizon=4), branches=[ViewpointType.FULL],
        canvas_grid_size=_G, amp_ctx=torch.enable_grad(), joint=joint,
    )
    assert torch.isfinite(r.total_loss)
    assert _has_grad(joint.scorer)
    assert _has_grad(seg.head)


def test_vpg_warns_on_uniform_but_not_on_foveated(caplog):
    """The recipe is defined for foveated; on uniform it still runs, so this is a warning,
    not a refusal — but it must be emitted where the policy is built, not buried in a
    docstring."""
    import logging

    seg_u, seg_f = _seg(), _seg_foveated()
    with caplog.at_level(logging.WARNING, logger="canvit.harness.policy"):
        _vpg_joint(canvit=seg_u.canvit, encode_model=seg_u, groups=ADE_GROUPS)
    assert any("UNIFORM-patcher" in r.getMessage() for r in caplog.records)

    caplog.clear()
    with caplog.at_level(logging.WARNING, logger="canvit.harness.policy"):
        _vpg_joint(canvit=seg_f.canvit, encode_model=seg_f, groups=ADE_GROUPS)
    assert not any("UNIFORM-patcher" in r.getMessage() for r in caplog.records)

    # ...and the inline objectives never get the warning, whatever the patcher.
    caplog.clear()
    with caplog.at_level(logging.WARNING, logger="canvit.harness.policy"):
        _joint_for(canvit=seg_u.canvit, encode_model=seg_u, groups=ADE_GROUPS)
    assert not any("UNIFORM-patcher" in r.getMessage() for r in caplog.records)


# --------------------------------------------------------------------------- #
# Configurable policy NETWORK (`policy_readout`) and policy INPUT
# (`feature_groups`). Both were previously hardwired to the QReg/PG recipe: the
# scorer was always the U-Net, and `rl.feature_groups` was silently discarded because
# every task passed its own module constant. Together, readout='local' +
# feature_groups=('ln_feat',) is the autoreg_tryout policy net: a per-token linear on
# the (LayerNormed) raw canvas.
# --------------------------------------------------------------------------- #

def _groups_of(joint) -> tuple[str, ...]:
    return tuple(joint.policy_selector.encoder.feature_groups)


def test_feature_groups_default_is_each_tasks_own_set_bit_for_bit():
    """THE no-op proof. `rl.feature_groups` now defaults to None = 'task default', so
    making the knob real must not move any existing run: every task must still build the
    exact set it built when the field was ignored."""
    from canvit.core.policy.features import FEATURE_GROUPS, INTRINSIC_GROUPS
    from canvit.in1k.task import POLICY_FEATURE_GROUPS as IN1K

    assert JointPolicyConfig().feature_groups is None
    seg, clf = _seg(), _clf()
    ade = _vpg_joint(canvit=seg.canvit, encode_model=seg, groups=ADE_GROUPS)
    in1k = _vpg_joint(canvit=clf.canvit, encode_model=clf, groups=IN1K)
    assert _groups_of(ade) == tuple(FEATURE_GROUPS)      # ade20k: full probe-entropy set
    assert _groups_of(in1k) == tuple(INTRINSIC_GROUPS)   # in1k/distill: probe-free
    assert ade.scorer.frontend.groups == tuple(FEATURE_GROUPS)


def test_feature_groups_override_reaches_scorer_and_encoder():
    """'just the raw canvas': ln_feat is the canvas spatial features with a per-token
    LayerNorm and nothing else — no cos_*/delta/entropy derivation."""
    seg = _seg()
    j = _vpg_joint(canvit=seg.canvit, encode_model=seg, groups=ADE_GROUPS,
                   feature_groups=("ln_feat",))
    assert _groups_of(j) == ("ln_feat",)          # the encoder builds only that group
    assert j.scorer.frontend.groups == ("ln_feat",)  # and the scorer is sized for it
    assert len(j.scorer.frontend.sizes) == 1
    assert not j.policy_selector.encoder.needs_entropy  # no probe forward at all now


def test_local_readout_is_reachable_from_config_and_drops_the_unet():
    seg = _seg()
    unet = _vpg_joint(canvit=seg.canvit, encode_model=seg, groups=ADE_GROUPS)
    local = _vpg_joint(canvit=seg.canvit, encode_model=seg, groups=ADE_GROUPS,
                       policy_readout="local")
    assert unet.scorer.readout == "unet" and unet.scorer.enc is not None
    assert local.scorer.readout == "local" and local.scorer.enc is None
    assert sum(p.numel() for p in local.scorer.parameters()) \
        < sum(p.numel() for p in unet.scorer.parameters())


def test_autoreg_style_policy_trains_end_to_end():
    """The combination that matters: foveated fixation grid + local readout + raw-canvas
    features + VPG. Must train the scorer and the value head like any other config."""
    torch.manual_seed(1)
    seg = _seg_foveated()
    seg.requires_grad_(False)
    seg.eval()
    joint = _vpg_joint(canvit=seg.canvit, encode_model=seg, groups=ADE_GROUPS,
                       policy_readout="local", feature_groups=("ln_feat",),
                       centers_per_axis=32, select_bn_eval=False)
    task = BoundAde20kTask(seg=seg, masks=_seg_masks(), canvas_grid=_G)
    _zero_grads(seg, joint.scorer)
    r = run_rollout(
        model=seg, images=torch.randn(_B, 3, _IMG, _IMG), task=task, selector=_fov_selector(),
        bptt=BpttSpec(mode="none", horizon=4), branches=[ViewpointType.FULL],
        canvas_grid_size=_G, amp_ctx=torch.enable_grad(), joint=joint,
    )
    assert torch.isfinite(r.total_loss)
    assert _has_grad(joint.scorer)
    assert _has_grad(joint.scorer.vhead)
    assert _no_grad(seg)
    assert joint.policy_selector.vp_flat.shape == (32 * 32, 3)  # heatmap ON the state grid


def test_policy_levels_and_input_norm_reach_the_scorer_and_train():
    """`policy_levels` / `policy_input_norm` (2026-11-13 architecture ablation) are reachable
    from the config, the defaults build the historical net, and a shallower, BatchNorm-free
    scorer trains end to end on the foveated fixation grid."""
    from torch import nn

    seg = _seg_foveated()
    seg.requires_grad_(False)
    seg.eval()
    default = _vpg_joint(canvit=seg.canvit, encode_model=seg, groups=ADE_GROUPS)
    assert len(default.scorer.enc) == 6 and isinstance(default.scorer.frontend.bn, nn.BatchNorm2d)
    torch.manual_seed(1)
    joint = _vpg_joint(canvit=seg.canvit, encode_model=seg, groups=ADE_GROUPS,
                       feature_groups=("ln_feat",), policy_levels=3, policy_input_norm="none")
    assert len(joint.scorer.enc) == 3 and len(joint.scorer.dec) == 2
    assert not any(isinstance(m, nn.modules.batchnorm._BatchNorm) for m in joint.scorer.modules())
    task = BoundAde20kTask(seg=seg, masks=_seg_masks(), canvas_grid=_G)
    _zero_grads(seg, joint.scorer)
    r = run_rollout(
        model=seg, images=torch.randn(_B, 3, _IMG, _IMG), task=task, selector=_fov_selector(),
        bptt=BpttSpec(mode="none", horizon=4), branches=[ViewpointType.FULL],
        canvas_grid_size=_G, amp_ctx=torch.enable_grad(), joint=joint,
    )
    assert torch.isfinite(r.total_loss)
    assert _has_grad(joint.scorer)
    assert _no_grad(seg)


def test_vpg_refuses_select_bn_eval_but_qreg_still_gets_it():
    """`select_bn_eval` defaults True (it reproduces the qband checkpoints for QReg/PG), but
    for VPG it means sampling from one distribution and differentiating log pi of another.

    This REFUSES as of 2026-07-31 (owner's call); it used to warn and continue, which meant
    a bare `--rl.objective vpg` trained on a biased off-policy gradient with only a log line
    to say so. The third assertion is the one that matters most: the refusal must NOT leak
    into QReg/PG, for which mode (b) IS the reference behaviour."""
    import pytest

    seg = _seg()
    with pytest.raises(ValueError, match="no-select-bn-eval"):
        _vpg_joint(canvit=seg.canvit, encode_model=seg, groups=ADE_GROUPS, select_bn_eval=True)

    # VPG with the unbiased mode builds fine.
    assert _vpg_joint(canvit=seg.canvit, encode_model=seg, groups=ADE_GROUPS,
                      select_bn_eval=False).defers_credit

    # QReg at the True DEFAULT must be untouched by the refusal.
    assert _joint_for(canvit=seg.canvit, encode_model=seg, groups=ADE_GROUPS) is not None
