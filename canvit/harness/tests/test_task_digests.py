"""Pinning digests for the ade20k / in1k rollouts — the refactor guard.

WHAT THESE ARE. ``test_rollout_parity.py``'s distill digest (``9a0100a1a3de3acd``)
originally certified agreement with the now-deleted standalone trainer. ade20k and in1k
have no old side left to agree with, so the digests here are **pinning** digests: they
assert only that today's numbers equal the numbers recorded when this file was written.
That is exactly the guarantee a pure file-move refactor needs, and nothing more — a
pre-existing bug is pinned in along with everything else. They are NOT a correctness
claim.

Recorded 2026-07-31 on the tree at ``8554c1f``, immediately before the package
restructure (``train/`` split into ``harness/`` + ``distill/``). If the restructure is a
pure move, every digest below is unchanged.

WHAT EACH DIGEST COVERS. Both halves of the training path:
  * the per-step ``total_loss`` stream — forward numerics, glimpse routing, BPTT
    chunking, loss reduction;
  * a fingerprint of every parameter after N optimizer steps — so a change to gradient
    flow (what is frozen, where ``detach`` lands, how chunks normalise) moves it too,
    even if step 0's loss is untouched.

FIDELITY. The spec of each config (bptt mode, branch composition, what is frozen) is the
production one from ``claude_dev/unification/capability_matrix.md``. ``horizon`` and step count
are cut down to keep this a CPU unit test; the digest's job is to detect change, not to
reproduce a training curve.

Run: ``.venv-cu126/bin/python -m pytest canvit/harness/tests/test_task_digests.py``
"""

import hashlib
from contextlib import nullcontext

import torch

from canvit.ade20k.data import IGNORE_LABEL, NUM_CLASSES
from canvit.ade20k.task import POLICY_FEATURE_GROUPS as ADE20K_GROUPS
from canvit.ade20k.task import BoundAde20kTask
from canvit.core import (
    CanViTForImageClassification,
    CanViTForPretraining,
    CanViTForPretrainingConfig,
    CanViTForSemanticSegmentation,
    create_backbone,
)
from canvit.core.model.base.config import coerce_nested_configs
from canvit.distill.loss import DistillTask
from canvit.distill.task import BoundDistillTask
from canvit.harness.config import FoveatedScaleConfig, JointPolicyConfig
from canvit.harness.policy import build_policy
from canvit.harness.rollout import run_rollout
from canvit.harness.rollout.selector import RandomSelector
from canvit.harness.rollout.viewpoint import ViewpointType
from canvit.harness.spec import BpttSpec
from canvit.in1k.task import BoundIn1kTask

_B, _G, _IMG, _N_CLS = 2, 8, 224, 10
_D = 384  # teacher width for the distill config (vits16)
_FIXED_SCALE = 2.0  # exp22/exp32/exp34/exp36 all pretrain and evaluate foveated at 2.0

# The exp32-fovi patcher geometry, value-for-value from
# slurm/runs/exp32_pretrain_lrdrop/exp32-fovi.sh -- the configuration that actually ran
# 245 array tasks, so a drift here is a drift in something we have results for.
_FOVEATED_MODEL_CONFIG = {
    "patcher_name": "foveated",
    "foveated_patcher": {
        "fov": 35, "resolution": 64, "cmf_a": 0.5, "cart_patch_size": 5,
        "arch_flag": "doubleres",
        "conditioning": {"mode": "film", "film": {"fourier": {"num_features": 256, "sigma": 4}}},
    },
}
_N_STEPS = 6
_HORIZON = 3
_DEVICE = torch.device("cpu")

# Recorded on 8554c1f. A pure file move must not change these.
_EXPECTED = {
    # Recorded 2026-07-31 on 8554c1f (uniform, policy-free).
    "ade20k_probe": "b9fd07bdac4f68bd",
    "ade20k_finetune": "28fb8bff5010a010",
    "in1k_probe": "00ed4e8f2279b20f",
    "in1k_finetune": "6f4accd7c2ad3dba",
    # Recorded 2026-09-09 on 4a7bf4f — the foveated and policy paths, which until then
    # had no numeric guard at all. Each verified to be sensitive to the thing it guards:
    # scaling the policy loss by 0.8 (the bug that once sat in the harness for weeks)
    # moves the policy digests, and fov 35 -> 34 moves the foveated ones.
    # The three foveated digests were re-recorded 2026-11-05 against fovi 27b1bde, which
    # ports upstream's minimum full-coverage KNN k and planar field geometry (k 46 -> 47
    # here). Under fovi c399d3b they were 15c1fd86283f05cf / 4e3314c856be78a1 /
    # ec460e57758e93d7.
    "distill_foveated": "6b5b0834f18355f7",
    "ade20k_probe_foveated": "c495ad71872f2f7d",
    "policy_qreg_uniform": "1b2ae61d6c9075ae",
    "policy_qreg_foveated": "93e815bf99e65370",
    # Recorded 2026-11-05 on aa78fad (fovi 27b1bde), before the search-v1 engine changes.
    # These three also hash the scorer's weights (`also_hash`); each was checked to move
    # under a small change of what it guards (PG entropy_bonus 0.01 -> 0.02 and rl_weight
    # 0.8, VPG baseline weight 0.5, QReg rl_weight 0.8). PG is the inline per-glimpse
    # credit path, VPG the deferred terminal-reward path; neither had a digest before.
    "policy_qreg_foveated_scorer": "964fda302528c40e",
    "policy_pg_foveated": "51fa46fe6bbc664c",
    "policy_vpg_foveated": "625ccbd492174ac8",
}


def _selector(*, foveated: bool = False) -> RandomSelector:
    """The no-arg call is the uniform selector the four original digests recorded under —
    keep it byte-identical or those digests move for the wrong reason. ``foveated=True``
    pins the view scale, which is what a fixed-scale foveated backbone requires: the
    window is ``fix_size = scale * H``, so an unpinned scale puts every glimpse out of
    distribution."""
    scale = (FoveatedScaleConfig(mode="fixed", fixed_scale=_FIXED_SCALE) if foveated
             else FoveatedScaleConfig())
    return RandomSelector(
        is_foveated=foveated, foveated_scale=scale, min_viewpoint_scale=0.1
    )


def _param_fingerprint(model: torch.nn.Module) -> str:
    """sha256 over every parameter's raw bytes, in sorted-name order. Sensitive to any
    numeric drift, including one that only shows up after several optimizer steps."""
    h = hashlib.sha256()
    for name, p in sorted(model.named_parameters(), key=lambda kv: kv[0]):
        h.update(name.encode())
        h.update(p.detach().contiguous().numpy().tobytes())
    return h.hexdigest()[:16]


def _run_digest(*, model, make_task, bptt: BpttSpec, trainable, seed: int,
                selector: RandomSelector | None = None, joint=None,
                task_weight: float = 1.0, also_hash: torch.nn.Module | None = None) -> str:
    """N seeded steps of ``run_rollout`` + ``opt.step()``; hash the loss stream and the
    resulting weights together.

    ``selector=None`` and ``joint=None`` reproduce the original uniform, policy-free path
    exactly, so the four digests recorded on 8554c1f are unaffected by these parameters.

    ``also_hash`` adds a module that is not part of ``model`` to the weight fingerprint.
    The policy digests need it for the scorer: ``model`` is the task wrapper, so without it
    the scorer's weights enter only through the viewpoints they choose, and a few small
    optimizer steps rarely change a sampled fixation. Measured 2026-11-05: without it, the
    PG digest did not move when ``entropy_bonus`` went 0.01 -> 0.02. ``None`` leaves the
    existing digests' hash input unchanged."""
    torch.use_deterministic_algorithms(True)
    opt = torch.optim.AdamW(trainable, lr=1e-4)
    selector = selector if selector is not None else _selector()

    torch.manual_seed(seed)  # batches + viewpoint sampling + dropout
    losses: list[str] = []
    for _ in range(_N_STEPS):
        images = torch.randn(_B, 3, _IMG, _IMG, device=_DEVICE)
        task = make_task()
        opt.zero_grad()
        result = run_rollout(
            model=model, images=images, task=task, selector=selector, bptt=bptt,
            branches=[ViewpointType.FULL],  # both tasks: train_start_full=True
            canvas_grid_size=_G, amp_ctx=nullcontext(), joint=joint,
            task_weight=task_weight,
        )
        opt.step()
        assert torch.isfinite(result.total_loss), "digest is meaningless on a NaN stream"
        losses.append(result.total_loss.item().hex())

    extra = "" if also_hash is None else _param_fingerprint(also_hash)
    return hashlib.sha256(
        ("".join(losses) + _param_fingerprint(model) + extra).encode()
    ).hexdigest()[:16]


# --------------------------------------------------------------------------- #
# ADE20K — probe (head only, bptt=none) and finetune (backbone+head, bptt=full).
# --------------------------------------------------------------------------- #
def _seg() -> CanViTForSemanticSegmentation:
    torch.manual_seed(0)
    return CanViTForSemanticSegmentation(
        backbone_name="vits16", model_config={}, num_classes=NUM_CLASSES
    ).to(_DEVICE)


def _seg_masks() -> torch.Tensor:
    m = torch.randint(0, NUM_CLASSES, (_B, _IMG, _IMG), device=_DEVICE)
    m[:, :8] = IGNORE_LABEL
    return m


def ade20k_probe_digest() -> str:
    seg = _seg()
    seg.canvit.requires_grad_(False)
    seg.canvit.eval()  # frozen backbone runs in eval mode (BN), as the probe recipe does
    return _run_digest(
        model=seg,
        make_task=lambda: BoundAde20kTask(seg=seg, masks=_seg_masks(), canvas_grid=_G),
        bptt=BpttSpec(mode="none", horizon=_HORIZON),
        trainable=list(seg.head.parameters()),
        seed=101,
    )


def ade20k_finetune_digest() -> str:
    seg = _seg()
    return _run_digest(
        model=seg,
        make_task=lambda: BoundAde20kTask(seg=seg, masks=_seg_masks(), canvas_grid=_G),
        bptt=BpttSpec(mode="full", horizon=_HORIZON),
        trainable=list(seg.parameters()),
        seed=102,
    )


# --------------------------------------------------------------------------- #
# IN1k — probe (head only, bptt=none) and finetune (backbone+head, bptt=full).
# --------------------------------------------------------------------------- #
def _clf() -> CanViTForImageClassification:
    torch.manual_seed(0)
    return CanViTForImageClassification(
        backbone_name="vits16", model_config={}, n_classes=_N_CLS, glimpse_grid_size=_G,
    ).to(_DEVICE)


def _targets() -> torch.Tensor:
    return torch.randint(0, _N_CLS, (_B,), device=_DEVICE)


def in1k_probe_digest() -> str:
    clf = _clf()
    clf.canvit.requires_grad_(False)
    clf.canvit.eval()
    return _run_digest(
        model=clf,
        make_task=lambda: BoundIn1kTask(clf=clf, targets=_targets(), canvas_grid=_G),
        bptt=BpttSpec(mode="none", horizon=_HORIZON),
        trainable=list(clf.head.parameters()),
        seed=201,
    )


def in1k_finetune_digest() -> str:
    clf = _clf()
    return _run_digest(
        model=clf,
        make_task=lambda: BoundIn1kTask(clf=clf, targets=_targets(), canvas_grid=_G),
        bptt=BpttSpec(mode="full", horizon=_HORIZON),
        trainable=list(clf.parameters()),
        seed=202,
    )


# --------------------------------------------------------------------------- #
# FOVEATED and POLICY coverage.
#
# Everything above pins the UNIFORM, policy-free paths. Nothing did the same for the two
# things the current science actually rests on: the foveated backbone (exp32/exp34/exp36)
# and the viewpoint policy (exp35/exp36). A regression in either used to be visible only
# in a multi-hour GPU run -- the harness policy gradient sat at exactly 0.8x the reference
# for weeks for want of a check like this.
#
# Configs are the ones that ran at scale, so a moved digest corresponds to results we hold.
# Horizon and step count are cut down to keep these CPU unit tests, as above.
# --------------------------------------------------------------------------- #
def _fov_seg() -> CanViTForSemanticSegmentation:
    torch.manual_seed(0)
    return CanViTForSemanticSegmentation(
        backbone_name="vits16", model_config=_FOVEATED_MODEL_CONFIG, num_classes=NUM_CLASSES
    ).to(_DEVICE)


def _distill_model() -> CanViTForPretraining:
    """exp32-fovi's model: the foveated patcher on the distill pretraining wrapper."""
    torch.manual_seed(0)
    return CanViTForPretraining(
        backbone=create_backbone("vits16"),
        # coerce_nested_configs: the wrapper classes accept nested dicts, but building a
        # config directly does not -- `foveated_patcher` would stay a dict and fail deep in
        # the patcher. This is the helper every loader uses for the same reason.
        cfg=CanViTForPretrainingConfig(
            teacher_dim=_D, **coerce_nested_configs(_FOVEATED_MODEL_CONFIG)),
        glimpse_size_px=128, backbone_name="vits16", canvas_patch_grid_sizes=[_G],
    ).to(_DEVICE)


def _joint(*, canvit, groups, encode_model, foveated: bool, rl: JointPolicyConfig | None = None):
    """A QReg scorer by default, the objective every policy campaign here has used.
    ``rl`` swaps in another objective (the PG / VPG digests)."""
    gen = torch.Generator(device=_DEVICE)
    gen.manual_seed(0)
    scale = (FoveatedScaleConfig(mode="fixed", fixed_scale=_FIXED_SCALE) if foveated
             else FoveatedScaleConfig())
    return build_policy(
        canvit=canvit, rl=rl if rl is not None else JointPolicyConfig(objective="qreg"),
        feature_groups=groups,
        device=_DEVICE, canvas_grid=_G, min_viewpoint_scale=0.1, foveated_scale=scale,
        generator=gen, encode_model=encode_model,
    )


def distill_foveated_digest() -> str:
    """exp32-fovi: foveated distill pretraining. The uniform equivalent is
    ``test_rollout_parity.py``'s ``9a0100a1a3de3acd``; the foveated path had nothing."""
    model = _distill_model()
    return _run_digest(
        model=model,
        make_task=lambda: BoundDistillTask(DistillTask(
            scene_target=torch.randn(_B, _G * _G, _D), cls_target=torch.randn(_B, _D),
            enable_scene_patches_loss=True, enable_scene_cls_loss=True,
        )),
        bptt=BpttSpec(mode="chunked", chunk_size=2, horizon=_HORIZON),
        trainable=list(model.parameters()),
        seed=301,
        selector=_selector(foveated=True),
    )


def ade20k_probe_foveated_digest() -> str:
    """exp34's ``ade20k-fovi-ti-1196k``: frozen foveated backbone, train the head."""
    seg = _fov_seg()
    seg.canvit.requires_grad_(False)
    seg.canvit.eval()
    return _run_digest(
        model=seg,
        make_task=lambda: BoundAde20kTask(seg=seg, masks=_seg_masks(), canvas_grid=_G),
        bptt=BpttSpec(mode="none", horizon=_HORIZON),
        trainable=list(seg.head.parameters()),
        seed=302,
        selector=_selector(foveated=True),
    )


def policy_qreg_uniform_digest() -> str:
    """exp35's recipe shape: ``policy_only`` on a uniform backbone -- backbone and head
    frozen, scorer trained, ``task_weight=0.0``, so this digest is the POLICY loss and the
    scorer's weights and nothing else."""
    seg = _seg()
    seg.canvit.requires_grad_(False)
    seg.canvit.eval()
    seg.head.requires_grad_(False)
    joint = _joint(canvit=seg.canvit, groups=ADE20K_GROUPS, encode_model=seg, foveated=False)
    return _run_digest(
        model=seg,
        make_task=lambda: BoundAde20kTask(seg=seg, masks=_seg_masks(), canvas_grid=_G),
        bptt=BpttSpec(mode="none", horizon=_HORIZON),
        trainable=list(joint.scorer.parameters()),
        seed=303,
        joint=joint,
        task_weight=0.0,
    )


def policy_qreg_foveated_digest() -> str:
    """exp36's recipe shape: the same, on the foveated backbone at scale 2.0."""
    seg = _fov_seg()
    seg.canvit.requires_grad_(False)
    seg.canvit.eval()
    seg.head.requires_grad_(False)
    joint = _joint(canvit=seg.canvit, groups=ADE20K_GROUPS, encode_model=seg, foveated=True)
    return _run_digest(
        model=seg,
        make_task=lambda: BoundAde20kTask(seg=seg, masks=_seg_masks(), canvas_grid=_G),
        bptt=BpttSpec(mode="none", horizon=_HORIZON),
        trainable=list(joint.scorer.parameters()),
        seed=304,
        joint=joint,
        selector=_selector(foveated=True),
        task_weight=0.0,
    )


def _policy_foveated_digest(rl: JointPolicyConfig, seed: int) -> str:
    """``policy_qreg_foveated``'s setup with another objective: frozen foveated backbone
    and head, scorer trained, ``task_weight=0.0``."""
    seg = _fov_seg()
    seg.canvit.requires_grad_(False)
    seg.canvit.eval()
    seg.head.requires_grad_(False)
    joint = _joint(canvit=seg.canvit, groups=ADE20K_GROUPS, encode_model=seg, foveated=True, rl=rl)
    return _run_digest(
        model=seg,
        make_task=lambda: BoundAde20kTask(seg=seg, masks=_seg_masks(), canvas_grid=_G),
        bptt=BpttSpec(mode="none", horizon=_HORIZON),
        trainable=list(joint.scorer.parameters()),
        seed=seed,
        joint=joint,
        selector=_selector(foveated=True),
        task_weight=0.0,
        also_hash=joint.scorer,
    )


def policy_qreg_foveated_scorer_digest() -> str:
    """``policy_qreg_foveated`` with the scorer's weights hashed too. That digest does NOT
    move when the policy loss is scaled by 0.8 (measured 2026-11-05: over its 6 steps the
    argmax fixations, and so the task losses, never change), so on its own it does not
    guard the foveated QReg policy loss. Kept alongside rather than re-recorded, so the
    original pin stays as it was."""
    return _policy_foveated_digest(JointPolicyConfig(objective="qreg"), seed=304)


def policy_pg_foveated_digest() -> str:
    """PG: on-policy sampling, the INLINE per-glimpse fractional reward, RunningNorm, and
    the entropy dual step. ``select_bn_eval=False``, the on-policy setting (with it on, PG
    samples from an eval-mode forward but differentiates the train-mode one)."""
    return _policy_foveated_digest(JointPolicyConfig(objective="pg", select_bn_eval=False), seed=305)


def policy_vpg_foveated_digest() -> str:
    """VPG: the DEFERRED terminal reward, the learned value head, per-step advantage
    normalisation. ``select_bn_eval=False`` is required for VPG (build_policy refuses True)."""
    return _policy_foveated_digest(JointPolicyConfig(objective="vpg", select_bn_eval=False), seed=306)


DIGESTS = {
    "ade20k_probe": ade20k_probe_digest,
    "ade20k_finetune": ade20k_finetune_digest,
    "in1k_probe": in1k_probe_digest,
    "in1k_finetune": in1k_finetune_digest,
    "distill_foveated": distill_foveated_digest,
    "ade20k_probe_foveated": ade20k_probe_foveated_digest,
    "policy_qreg_uniform": policy_qreg_uniform_digest,
    "policy_qreg_foveated": policy_qreg_foveated_digest,
    "policy_qreg_foveated_scorer": policy_qreg_foveated_scorer_digest,
    "policy_pg_foveated": policy_pg_foveated_digest,
    "policy_vpg_foveated": policy_vpg_foveated_digest,
}


def test_ade20k_probe_digest():
    assert ade20k_probe_digest() == _EXPECTED["ade20k_probe"]


def test_ade20k_finetune_digest():
    assert ade20k_finetune_digest() == _EXPECTED["ade20k_finetune"]


def test_in1k_probe_digest():
    assert in1k_probe_digest() == _EXPECTED["in1k_probe"]


def test_in1k_finetune_digest():
    assert in1k_finetune_digest() == _EXPECTED["in1k_finetune"]


# To run just the path you touched, select by NAME: `-k foveated` (36 tests across the
# suite), `-k policy` (62), `-k digest` (9). That is a filter for the inner loop, not a
# gate — it matches names, so it misses a test that exercises foveated code under another
# name. Run the whole suite before pinning a commit for a real job: every expensive bug in
# this repo's history was a cross-cutting surprise that "only touched" something else.
def test_distill_foveated_digest():
    assert distill_foveated_digest() == _EXPECTED["distill_foveated"]


def test_ade20k_probe_foveated_digest():
    assert ade20k_probe_foveated_digest() == _EXPECTED["ade20k_probe_foveated"]


def test_policy_qreg_uniform_digest():
    assert policy_qreg_uniform_digest() == _EXPECTED["policy_qreg_uniform"]


def test_policy_qreg_foveated_digest():
    assert policy_qreg_foveated_digest() == _EXPECTED["policy_qreg_foveated"]


def test_policy_qreg_foveated_scorer_digest():
    assert policy_qreg_foveated_scorer_digest() == _EXPECTED["policy_qreg_foveated_scorer"]


def test_policy_pg_foveated_digest():
    assert policy_pg_foveated_digest() == _EXPECTED["policy_pg_foveated"]


def test_policy_vpg_foveated_digest():
    assert policy_vpg_foveated_digest() == _EXPECTED["policy_vpg_foveated"]


if __name__ == "__main__":  # record / re-check the digests
    for name, fn in DIGESTS.items():
        print(f"{name:20s} {fn()}")
