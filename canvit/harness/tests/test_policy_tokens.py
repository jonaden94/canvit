"""Policy tokens (CanViTConfig.n_policy_tokens; design note
canvit_claude/canvit/active_pretraining/02-policy-tokens-design.md).

The load-bearing properties:
  * 0 tokens = today's model (no params, no state field);
  * READ-ONLY tokens leave the pretrained computation bit-identical (canvas, CLS, patches),
    so every read-only arm starts from exactly the pretrained model;
  * the readout is a proper distribution over the canvas cells, uniform-weighted at init;
  * joint PG training reaches the tokens, the readout AND the backbone;
  * seeding a token model from a token-free checkpoint is strict except for the new params.
"""

import math
from dataclasses import replace

import pytest
import torch

from canvit import CanViTForPretraining, CanViTForPretrainingConfig
from canvit.core import Viewpoint, create_backbone
from canvit.distill.loss import DistillTask
from canvit.distill.task import POLICY_FEATURE_GROUPS, BoundDistillTask
from canvit.harness.config import FoveatedScaleConfig, JointPolicyConfig
from canvit.harness.infra.checkpoint import restore_into
from canvit.harness.policy import build_policy
from canvit.harness.rollout import run_rollout
from canvit.harness.rollout.selector import RandomSelector
from canvit.harness.rollout.viewpoint import ViewpointType
from canvit.harness.spec import BpttSpec

_B, _G, _IMG, _D, _K = 2, 8, 224, 384, 3
_DEV = torch.device("cpu")


def _model(**cfg_kw) -> CanViTForPretraining:
    torch.manual_seed(0)
    cfg = CanViTForPretrainingConfig(teacher_dim=_D, policy_grid=_G, **cfg_kw)
    return CanViTForPretraining(backbone=create_backbone("vits16"), cfg=cfg, glimpse_size_px=128,
                                backbone_name="vits16", canvas_patch_grid_sizes=[_G]).to(_DEV)


def _vps():
    return [Viewpoint(centers=torch.zeros(_B, 2), scales=torch.ones(_B)),
            Viewpoint(centers=torch.full((_B, 2), 0.3), scales=torch.full((_B,), 0.5))]


def _roll(model, images):
    state = model.init_state(batch_size=_B, canvas_grid_size=_G)
    outs = []
    with torch.no_grad():
        for vp in _vps():
            out = model(image=images, state=state, viewpoint=vp)
            state = out.state
            outs.append(out)
    return outs


def test_zero_tokens_adds_nothing():
    m = _model()
    assert m.policy_token_init is None and m.policy_readout is None
    assert not any(k.startswith(("policy_token_init", "policy_readout")) for k in m.state_dict())
    assert m.init_state(batch_size=_B, canvas_grid_size=_G).policy_tokens is None


def test_readonly_tokens_leave_the_pretrained_computation_bit_identical():
    base, tok = _model(), _model(n_policy_tokens=_K, policy_token_mode="readonly")
    tok.load_state_dict(base.state_dict(), strict=False)  # same pretrained weights
    images = torch.randn(_B, 3, _IMG, _IMG, generator=torch.Generator().manual_seed(1))
    for ob, ot in zip(_roll(base, images), _roll(tok, images)):
        assert torch.equal(ob.state.canvas, ot.state.canvas)
        assert torch.equal(ob.state.recurrent_cls, ot.state.recurrent_cls)
        assert torch.equal(ob.local_patches, ot.local_patches)
        assert ot.state.policy_tokens is not None and ot.state.policy_tokens.shape == (_B, _K, base.local_dim)


def test_full_tokens_participate():
    base, tok = _model(), _model(n_policy_tokens=_K, policy_token_mode="full")
    tok.load_state_dict(base.state_dict(), strict=False)
    images = torch.randn(_B, 3, _IMG, _IMG, generator=torch.Generator().manual_seed(1))
    ob, ot = _roll(base, images)[-1], _roll(tok, images)[-1]
    assert ot.state.policy_tokens.shape == (_B, _K, base.local_dim)
    assert not torch.equal(ob.state.canvas, ot.state.canvas), "full tokens must change the computation"


@pytest.mark.parametrize("mode", ["readonly", "full"])
def test_readout_is_a_distribution_and_uniformly_weighted_at_init(mode):
    m = _model(n_policy_tokens=_K, policy_token_mode=mode, policy_heads=4)
    images = torch.randn(_B, 3, _IMG, _IMG)
    state = _roll(m, images)[-1].state
    logp = m.policy_log_probs(state)
    assert logp.shape == (_B, _G * _G) and logp.dtype == torch.float32
    assert torch.allclose(logp.exp().sum(-1), torch.ones(_B), atol=1e-5)
    # at init (zero weight layer, zero bias): the plain mean of the K*H per-head softmax maps
    ro = m.policy_readout
    t = ro.token_norm(state.policy_tokens)
    q = ro.q_proj(t).view(_B, _K, 4, -1)
    k = ro.canvas_norm(m.get_spatial(state.canvas)).view(_B, _G * _G, 4, -1)
    maps = torch.softmax(torch.einsum("bkhd,bnhd->bkhn", q, k) / math.sqrt(q.shape[-1]), -1)
    assert torch.allclose(logp.exp(), maps.mean(dim=(1, 2)), atol=1e-6)


def test_reinit_starts_every_glimpse_from_the_init():
    carry = _model(n_policy_tokens=_K)
    reinit = _model(n_policy_tokens=_K, policy_token_reinit=True)
    reinit.load_state_dict(carry.state_dict())
    images = torch.randn(_B, 3, _IMG, _IMG)
    oc, orr = _roll(carry, images), _roll(reinit, images)
    assert torch.equal(oc[0].state.policy_tokens, orr[0].state.policy_tokens)  # t0: both start from init
    assert not torch.equal(oc[1].state.policy_tokens, orr[1].state.policy_tokens)
    # reinit at t1 = processing the init again with the t1 inputs (state otherwise identical)
    st = replace(oc[0].state, policy_tokens=None)
    with torch.no_grad():
        o1 = carry(image=images, state=st, viewpoint=_vps()[1])  # None -> init
    assert torch.equal(o1.state.policy_tokens, orr[1].state.policy_tokens)


def _joint(model, **rl_kw):
    gen = torch.Generator(device=_DEV).manual_seed(0)
    return build_policy(canvit=model, rl=JointPolicyConfig(objective="pg", feats_detached=False, **rl_kw),
                        feature_groups=POLICY_FEATURE_GROUPS, device=_DEV, canvas_grid=_G,
                        min_viewpoint_scale=0.05, foveated_scale=FoveatedScaleConfig(), generator=gen)


@pytest.mark.parametrize("mode", ["readonly", "full"])
def test_joint_pg_trains_tokens_readout_and_backbone(mode):
    m = _model(n_policy_tokens=_K, policy_token_mode=mode)
    joint = _joint(m)
    assert joint.scorer is m.policy_readout
    task = BoundDistillTask(DistillTask(
        scene_target=torch.randn(_B, _G * _G, _D), cls_target=torch.randn(_B, _D),
        enable_scene_patches_loss=True, enable_scene_cls_loss=True))
    r = run_rollout(
        model=m, images=torch.randn(_B, 3, _IMG, _IMG), task=task,
        selector=RandomSelector(is_foveated=False, foveated_scale=FoveatedScaleConfig(), min_viewpoint_scale=0.05),
        bptt=BpttSpec(mode="chunked", chunk_size=2, horizon=4), branches=[ViewpointType.FULL],
        canvas_grid_size=_G, amp_ctx=torch.enable_grad(), joint=joint)
    assert r.policy_metrics is not None and torch.isfinite(r.policy_metrics["policy_loss"])
    # entropy (nats over the policy_grid^2 cells) and alpha are logged for the inline PG path
    ent = float(r.policy_metrics["policy_entropy"])
    assert 0 < ent <= math.log(_G * _G) + 1e-4 and torch.isfinite(r.policy_metrics["alpha"])
    for name, p in [("init", m.policy_token_init), ("q_proj", m.policy_readout.q_proj.weight),
                    ("backbone", m.backbone.blocks[0].attn.qkv.weight)]:
        assert p.grad is not None and p.grad.abs().sum() > 0, name


def test_builder_refuses_what_tokens_cannot_honour():
    m = _model(n_policy_tokens=_K)
    with pytest.raises(ValueError, match="PG only"):
        build_policy(canvit=m, rl=JointPolicyConfig(objective="qreg", feats_detached=False),
                     feature_groups=POLICY_FEATURE_GROUPS, device=_DEV, canvas_grid=_G,
                     min_viewpoint_scale=0.05, foveated_scale=FoveatedScaleConfig(),
                     generator=torch.Generator())


def test_seed_from_a_token_free_checkpoint_is_strict_except_new_params():
    base, tok = _model(), _model(n_policy_tokens=_K)
    payload = {"model_state": base.state_dict(), "step": 7}
    restore_into(payload, model=tok, new_param_prefixes=("policy_token_init", "policy_readout."))
    for k, v in base.state_dict().items():
        assert torch.equal(tok.state_dict()[k], v), k
    with pytest.raises(RuntimeError):  # without the allowance: the strict load refuses
        restore_into(payload, model=_model(n_policy_tokens=_K))
    bad = {"model_state": {k: v for k, v in base.state_dict().items() if "blocks.0." not in k}}
    with pytest.raises(RuntimeError, match="missing keys outside"):
        restore_into(bad, model=_model(n_policy_tokens=_K),
                     new_param_prefixes=("policy_token_init", "policy_readout."))


def test_deploy_takes_the_argmax_cell():
    from canvit.harness.rollout.eval_viewpoints import deploy_rollout_viewpoints

    m = _model(n_policy_tokens=_K)
    joint = _joint(m)
    images = torch.randn(_B, 3, _IMG, _IMG)
    states = []

    def advance(vp, state, t):
        if state is None:
            state = m.init_state(batch_size=_B, canvas_grid_size=_G)
        s = m(image=images, state=state, viewpoint=vp).state
        states.append(s)
        return s

    maps: list = []
    vps = deploy_rollout_viewpoints(joint=joint, advance=advance, t0_type=ViewpointType.FULL,
                                    batch_size=_B, device=_DEV, n=3, prob_maps=maps) \
        if "prob_maps" in deploy_rollout_viewpoints.__code__.co_varnames else \
        deploy_rollout_viewpoints(joint=joint, advance=advance, t0_type=ViewpointType.FULL,
                                  batch_size=_B, device=_DEV, n=3)
    table = joint.policy_selector.vp_flat
    for t in (1, 2):
        with torch.no_grad():
            idx = m.policy_log_probs(states[t - 1]).argmax(-1)
        assert torch.equal(vps[t].centers, table[idx, :2])
