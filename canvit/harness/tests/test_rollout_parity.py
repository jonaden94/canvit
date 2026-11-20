"""Byte-exact parity: the generalized rollout engine reproduces distill's loss stream.

Replicates ``canvit_claude/canvit/unification/parity_probe.py`` (same tiny CPU model, same pinned
RNG, same 25 steps) but drives the loss through ``harness.rollout.run_rollout`` with
a distill adapter instead of ``train/step.py::training_step``. If the sha256[:16] of
the loss stream still equals the recorded ``9a0100a1a3de3acd``, the generalized
engine is a byte-for-byte superset of the historical distill rollout (design §6).

Run: ``.venv-cu126/bin/python -m pytest canvit/harness/tests/test_rollout_parity.py``
"""

import hashlib
import random
from contextlib import nullcontext

import torch

from canvit import CanViTForPretraining, CanViTForPretrainingConfig
from canvit.core import create_backbone
from canvit.distill.loss import DistillTask
from canvit.harness.config import FoveatedScaleConfig
from canvit.harness.rollout import GlimpseOut, run_rollout
from canvit.harness.rollout.selector import RandomSelector
from canvit.harness.rollout.viewpoint import ViewpointType
from canvit.harness.spec import BpttSpec

_B, _G, _D = 2, 8, 384
_N_STEPS = 25
_DEVICE = torch.device("cpu")
_EXPECTED_DIGEST = "9a0100a1a3de3acd"  # recorded distill loss-stream digest (parity guard)


class _DistillAdapter:
    """Wraps a per-step ``DistillTask`` as a ``RolloutTask``: the distill forward IS
    the pretraining model forward (heads computed inside), so ``step_loss`` /
    ``per_image_loss`` delegate straight through."""

    def __init__(self, task: DistillTask):
        self.task = task

    def forward_glimpse(self, *, model, images, state, viewpoint, backbone_no_grad):
        # Distill trains the backbone (backbone_no_grad is always False here); the
        # ctx is threaded through for uniformity with the other tasks.
        ctx = torch.no_grad() if backbone_no_grad else nullcontext()
        with ctx:
            out = model(image=images, state=state, viewpoint=viewpoint)
        return GlimpseOut(readout=out, state=out.state, vpe=out.vpe)

    def step_loss(self, readout):
        return self.task.step_loss(readout)

    def per_image_loss(self, readout):
        return self.task.per_image_loss(readout)


def _build_model() -> CanViTForPretraining:
    torch.manual_seed(1234)
    backbone = create_backbone("vits16").to(_DEVICE)
    cfg = CanViTForPretrainingConfig(teacher_dim=_D)
    return CanViTForPretraining(
        backbone=backbone, cfg=cfg, glimpse_size_px=128,
        backbone_name="vits16", canvas_patch_grid_sizes=[_G],
    ).to(_DEVICE)


def test_rollout_reproduces_distill_parity_digest():
    torch.use_deterministic_algorithms(True)
    model = _build_model()
    opt = torch.optim.AdamW(model.parameters(), lr=1e-4)

    selector = RandomSelector(is_foveated=False, foveated_scale=FoveatedScaleConfig(), min_viewpoint_scale=0.1)
    bptt = BpttSpec(mode="chunked", chunk_size=2, continue_prob=0.5)
    branches = [ViewpointType.FULL, ViewpointType.RANDOM]  # n_full=1, n_random=1 (full first)

    random.seed(4321)      # drives trajectory length (continue_prob draws)
    torch.manual_seed(5678)  # drives batches + viewpoint sampling + dropout
    losses: list[str] = []
    for _ in range(_N_STEPS):
        images = torch.randn(_B, 3, 224, 224, device=_DEVICE)
        scene_target = torch.randn(_B, _G * _G, _D, device=_DEVICE)
        cls_target = torch.randn(_B, _D, device=_DEVICE)
        # raw_* targets are consumed only by distill's cos-sim metrics (not the loss
        # stream); the parity digest is over total_loss, so we omit them here.
        _ = torch.randn(_B, _G * _G, _D, device=_DEVICE)  # keep RNG draw parity with the probe
        _ = torch.randn(_B, _D, device=_DEVICE)

        task = _DistillAdapter(DistillTask(
            scene_target=scene_target, cls_target=cls_target,
            enable_scene_patches_loss=True, enable_scene_cls_loss=True,
        ))

        opt.zero_grad()
        result = run_rollout(
            model=model, images=images, task=task, selector=selector,
            bptt=bptt, branches=branches, canvas_grid_size=_G, amp_ctx=nullcontext(),
        )
        opt.step()
        losses.append(result.total_loss.item().hex())

    digest = hashlib.sha256("".join(losses).encode()).hexdigest()[:16]
    assert digest == _EXPECTED_DIGEST, (
        f"rollout engine loss-stream digest {digest} != recorded distill {_EXPECTED_DIGEST}; "
        "the generalized engine is NOT byte-for-byte with the historical distill rollout."
    )
