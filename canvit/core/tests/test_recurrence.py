"""Tests for the bViT / reViT recurrent backbone option (``RecurrenceConfig``).

The property that matters most is the first one: with recurrence OFF (the default) the
model is the old model — same module tree, same state_dict keys, same numbers. The
digest tests in ``harness/tests`` guard that end to end; these check it directly.
"""

import dataclasses

import pytest
import torch

from canvit.core import CanViT, CanViTConfig, Viewpoint, create_backbone
from canvit.core.backbone import RecurrenceConfig
from canvit.core.backbone.dinov3_init import load_dinov3_weights_into_backbone
from canvit.core.backbone.recurrence import ExpertMLP
from canvit.core.backbone.vit import MLP
from canvit.core.model.base.config import rebuild_canvit_config, serialize_canvit_config

GLIMPSE = 64  # vits16 -> 4x4 patches
CANVAS_GRID = 8

BVIT = RecurrenceConfig(enabled=True)
REVIT = RecurrenceConfig(enabled=True, n_experts=4)


def _model(rec: RecurrenceConfig | None, *, seed: int = 0) -> CanViT:
    torch.manual_seed(seed)
    cfg = CanViTConfig() if rec is None else CanViTConfig(recurrence=rec)
    bb = create_backbone("vits16") if rec is None else create_backbone("vits16", recurrence=rec)
    return CanViT(backbone=bb, cfg=cfg, glimpse_size_px=GLIMPSE)


def _forward(model: CanViT):
    g = torch.Generator().manual_seed(1)
    img = torch.randn(2, 3, GLIMPSE, GLIMPSE, generator=g)
    vp = Viewpoint.full_scene(batch_size=2, device=img.device)
    state = model.init_state(batch_size=2, canvas_grid_size=CANVAS_GRID)
    return model(image=img, state=state, viewpoint=vp, canvas_grid_size=CANVAS_GRID)


# --------------------------------------------------------------------------- off == old


def test_off_is_the_old_model():
    old, off = _model(None).eval(), _model(RecurrenceConfig()).eval()
    assert list(old.state_dict()) == list(off.state_dict())
    for (k, a), (_, b) in zip(old.state_dict().items(), off.state_dict().items()):
        assert torch.equal(a, b), k
    with torch.inference_mode():
        o1, o2 = _forward(old), _forward(off)
    assert torch.equal(o1.state.canvas, o2.state.canvas)
    assert torch.equal(o1.local_patches, o2.local_patches)


def test_disabled_config_ignores_its_other_fields():
    """enabled=False must win: n_experts/n_templates/step_embed are inert."""
    rec = RecurrenceConfig(enabled=False, n_experts=4, n_templates=2, step_embed=True, depth=6)
    bb = create_backbone("vits16", recurrence=rec)
    assert bb.n_blocks == 12 and len(bb.blocks) == 12
    assert bb.router is None and bb.step_embed is None
    assert all(isinstance(b.mlp, MLP) for b in bb.blocks)


# --------------------------------------------------------------------------- bViT


def test_bvit_shares_one_block():
    m = _model(BVIT)
    bb = m.backbone
    assert bb.n_blocks == 12 and len(bb.blocks) == 1 and bb.n_templates == 1
    # One ViT-S block's parameters instead of twelve.
    old = _model(None).backbone
    n = sum(p.numel() for p in bb.blocks.parameters())
    assert n * 12 == sum(p.numel() for p in old.blocks.parameters())
    # The read/write schedule is over computational depth, unchanged.
    assert (m.read_after_blocks, m.write_after_blocks) == ((1, 5, 9), (3, 7, 11))
    out = _forward(m)
    assert torch.isfinite(out.state.canvas).all()


def test_partial_sharing_and_step_embed():
    rec = RecurrenceConfig(enabled=True, n_templates=3, step_embed=True)
    m = _model(rec)
    bb = m.backbone
    assert len(bb.blocks) == 3 and bb.step_embed is not None and bb.step_embed.shape == (12, 384)
    out = _forward(m)
    (out.state.canvas.square().mean() + out.local_patches.square().mean()).backward()
    assert bb.step_embed.grad is not None and bb.step_embed.grad.abs().sum() > 0
    with pytest.raises(AssertionError, match="multiple of n_templates"):
        create_backbone("vits16", recurrence=RecurrenceConfig(enabled=True, n_templates=5))


def test_depth_override_changes_compute_not_params():
    m = _model(RecurrenceConfig(enabled=True, depth=24))
    assert m.backbone.n_blocks == 24 and len(m.backbone.blocks) == 1
    assert len(m.read_after_blocks) == 6 and m.write_after_blocks[-1] == 23
    assert torch.isfinite(_forward(m).state.canvas).all()


# --------------------------------------------------------------------------- reViT


def test_expert_mlp_with_identical_experts_equals_mlp():
    torch.manual_seed(0)
    mlp = MLP(16, 32)
    bank = ExpertMLP(16, 32, 3)
    with torch.no_grad():
        bank.w1.copy_(mlp.fc1.weight.expand(3, -1, -1))
        bank.b1.copy_(mlp.fc1.bias.expand(3, -1))
        bank.w2.copy_(mlp.fc2.weight.expand(3, -1, -1))
        bank.b2.copy_(mlp.fc2.bias.expand(3, -1))
    x = torch.randn(2, 5, 16)
    gate = torch.softmax(torch.randn(3), 0)
    torch.testing.assert_close(bank(x, gate), mlp(x))


def test_expert_mlp_merges_weights_not_outputs():
    """One-hot gate -> exactly that expert; the merge is on weights (Eq. 2-3)."""
    torch.manual_seed(0)
    bank = ExpertMLP(8, 16, 4)
    x = torch.randn(3, 8)
    for e in range(4):
        one_hot = torch.nn.functional.one_hot(torch.tensor(e), 4).float()
        ref = torch.nn.functional.linear(
            torch.nn.functional.gelu(torch.nn.functional.linear(x, bank.w1[e], bank.b1[e])),
            bank.w2[e], bank.b2[e])
        torch.testing.assert_close(bank(x, one_hot), ref)


def test_revit_init_and_gradients():
    m = _model(REVIT)
    bb = m.backbone
    assert len(bb.blocks) == 1 and isinstance(bb.blocks[0].mlp, ExpertMLP)
    g = bb.depth_gates()
    assert g is not None and g.shape == (12, 4)
    torch.testing.assert_close(g, torch.full((12, 4), 0.25))  # zero-init router -> uniform
    # Router fc1 keeps PyTorch's default init (NOT the backbone's trunc_normal + zero bias).
    assert bb.router is not None and bb.router.fc1.bias.abs().sum() > 0

    out = _forward(m)
    aux = bb.recurrence_aux_loss()
    assert aux is not None
    total, terms = aux
    assert set(terms) == {"bal", "z", "div"} and all(torch.isfinite(v) for v in terms.values())
    assert terms["bal"].item() == pytest.approx(0.0, abs=1e-6)  # uniform gates: max entropy
    (out.state.canvas.square().mean() + out.local_patches.square().mean() + total).backward()
    assert bb.router.fc2.weight.grad is not None and bb.router.fc2.weight.grad.abs().sum() > 0
    assert bb.blocks[0].mlp.w1.grad is not None
    stats = bb.recurrence_stats()
    assert stats["gate_max"] == pytest.approx(0.25) and stats["w_cos_max"] < 0.1


def test_no_aux_loss_without_experts():
    assert _model(None).backbone.recurrence_aux_loss() is None
    assert _model(BVIT).backbone.recurrence_aux_loss() is None
    assert _model(BVIT).backbone.recurrence_stats() == {}


def test_revit_with_modulated_backbone():
    """Composes with the adaLN `*_modulate` backbone (modulation is per computational step)."""
    from canvit.core.modulation import ViTModulationConfig
    cfg = CanViTConfig(vit_modulation=ViTModulationConfig(enabled=True), recurrence=REVIT)
    m = CanViT(backbone=create_backbone("vits16_modulate", recurrence=REVIT), cfg=cfg,
               glimpse_size_px=GLIMPSE)
    assert m.token_modulation is not None and len(m.token_modulation(m.patcher.patch_positions()).block) == 12
    assert torch.isfinite(_forward(m).state.canvas).all()


# --------------------------------------------------------------------------- plumbing


def test_backbone_and_config_must_agree():
    with pytest.raises(AssertionError, match="recurrence"):
        CanViT(backbone=create_backbone("vits16"), cfg=CanViTConfig(recurrence=BVIT))
    with pytest.raises(AssertionError, match="recurrence"):
        CanViT(backbone=create_backbone("vits16", recurrence=BVIT), cfg=CanViTConfig())


@pytest.mark.parametrize("rec", [RecurrenceConfig(), BVIT, REVIT,
                                 RecurrenceConfig(enabled=True, n_templates=2, depth=8)])
def test_config_roundtrip(rec):
    cfg = CanViTConfig(recurrence=rec)
    assert rebuild_canvit_config(serialize_canvit_config(cfg)).recurrence == rec
    import dacite
    assert dacite.from_dict(CanViTConfig, dataclasses.asdict(cfg)).recurrence == rec


def test_old_config_without_field_loads_as_off():
    d = serialize_canvit_config(CanViTConfig())
    del d["recurrence"]
    assert rebuild_canvit_config(d).recurrence == RecurrenceConfig()


def test_state_dict_roundtrip_into_rebuilt_model():
    src = _model(REVIT, seed=0)
    cfg = rebuild_canvit_config(serialize_canvit_config(src.cfg))
    dst = CanViT(backbone=create_backbone("vits16", recurrence=cfg.recurrence), cfg=cfg,
                 glimpse_size_px=GLIMPSE)
    dst.load_state_dict(src.state_dict(), strict=True)
    with torch.inference_mode():
        assert torch.equal(_forward(src.eval()).state.canvas, _forward(dst.eval()).state.canvas)
    # ...and a recurrent checkpoint does NOT load into a standard model (strict).
    with pytest.raises(RuntimeError):
        _model(None).load_state_dict(src.state_dict(), strict=True)


def test_dinov3_init_refuses_recurrent_backbone():
    with pytest.raises(ValueError, match="recurrent"):
        load_dinov3_weights_into_backbone(create_backbone("vits16", recurrence=BVIT), torch.nn.Linear(1, 1))
