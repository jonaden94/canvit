"""The task-vs-policy gradient split (``run_rollout(grad_split_params=...)``).

Two properties make it trustworthy: it must not change training (``.grad`` identical with
and without the split), and what it reports as the policy share must be the policy term's
gradient (with the task loss weighted 0, the policy share IS the whole ``.grad``).
"""

import torch

from canvit import CanViTForPretraining, CanViTForPretrainingConfig
from canvit.core import create_backbone
from canvit.distill.loss import DistillTask
from canvit.distill.task import POLICY_FEATURE_GROUPS, BoundDistillTask
from canvit.harness.config import FoveatedScaleConfig, JointPolicyConfig
from canvit.harness.loop import _grad_split_metrics
from canvit.harness.policy import build_policy
from canvit.harness.rollout import run_rollout
from canvit.harness.rollout.selector import RandomSelector
from canvit.harness.rollout.viewpoint import ViewpointType
from canvit.harness.spec import BpttSpec

_B, _G, _IMG, _D = 2, 8, 224, 384
_DEV = torch.device("cpu")


def _rollout(*, split: bool, task_weight: float = 1.0):
    torch.manual_seed(0)
    model = CanViTForPretraining(
        backbone=create_backbone("vits16"), cfg=CanViTForPretrainingConfig(teacher_dim=_D),
        glimpse_size_px=128, backbone_name="vits16", canvas_patch_grid_sizes=[_G],
    ).to(_DEV)
    gen = torch.Generator(device=_DEV)
    gen.manual_seed(0)
    joint = build_policy(  # PG whose gradient reaches the backbone (feats not detached)
        canvit=model, rl=JointPolicyConfig(objective="pg", select_bn_eval=False, feats_detached=False),
        feature_groups=POLICY_FEATURE_GROUPS, device=_DEV, canvas_grid=_G, min_viewpoint_scale=0.05,
        foveated_scale=FoveatedScaleConfig(), generator=gen, encode_model=None,
    )
    g = torch.Generator().manual_seed(1)
    task = BoundDistillTask(DistillTask(
        scene_target=torch.randn(_B, _G * _G, _D, generator=g), cls_target=torch.randn(_B, _D, generator=g),
        enable_scene_patches_loss=True, enable_scene_cls_loss=True,
    ))
    params = list(model.parameters())
    r = run_rollout(
        model=model, images=torch.randn(_B, 3, _IMG, _IMG, generator=g), task=task,
        selector=RandomSelector(is_foveated=False, foveated_scale=FoveatedScaleConfig(), min_viewpoint_scale=0.05),
        bptt=BpttSpec(mode="chunked", chunk_size=2, horizon=4), branches=[ViewpointType.FULL],
        canvas_grid_size=_G, amp_ctx=torch.enable_grad(), joint=joint, task_weight=task_weight,
        grad_split_params=params if split else None,
    )
    return params, r


def test_split_leaves_training_gradient_unchanged():
    p_off, r_off = _rollout(split=False)
    p_on, r_on = _rollout(split=True)
    assert r_off.policy_grads is None and r_on.policy_grads is not None
    for a, b in zip(p_off, p_on):
        assert (a.grad is None) == (b.grad is None)
        if a.grad is not None:
            assert torch.equal(a.grad, b.grad)


def test_policy_share_is_the_policy_terms_gradient():
    params, r = _rollout(split=True, task_weight=0.0)  # .grad = policy term only
    assert r.policy_grads is not None
    n_reached = 0
    for p, gp in zip(params, r.policy_grads):
        if gp is None:
            assert p.grad is None or not p.grad.any()
            continue
        n_reached += 1
        assert torch.allclose(gp, p.grad, rtol=1e-5, atol=1e-8)
    assert n_reached > 0, "the policy gradient reached no model parameter"
    m = _grad_split_metrics(params, r.policy_grads)
    assert m["grad_split/policy_norm"] > 0 and m["grad_split/task_norm"] < 1e-6 * m["grad_split/policy_norm"]
