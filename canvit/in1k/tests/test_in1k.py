"""CPU smoke tests for the IN1k classification task (P5): the glimpse rollout
produces per-timestep CLS tokens for both patcher routings; frozen mode trains
only LN+head (backbone untouched) while finetune reaches the backbone; metrics
count correctly."""

import os
from pathlib import Path

import pytest
import torch

from canvit.core import CanViTForImageClassification
from canvit.core.patcher import FoveatedPatcherConfig

from ...harness.rollout.episode import consumes_full_image
from ...harness.rollout.eval_viewpoints import make_random_viewpoints
from ..config import FoveatedScaleConfig
from ..metrics import TopKAccuracy, ce_loss, topk_correct
from ..rollout import eval_viewpoints, rollout_cls_tokens

_IN1K_SHARDS = Path("/mnt/vast-nhr/projects/nib00021/jonathan/datasets") \
    / "webdataset-imagenet-1k-no-features/train-shuffled"


def _readable_dir(p: Path) -> bool:
    """``is_dir()`` RAISES on a path that exists but is not readable (a user outside group
    HPC_nib00021), which failed the whole suite at collection instead of skipping this test."""
    try:
        return p.is_dir() and os.access(p, os.R_OK | os.X_OK)
    except OSError:
        return False

_B, _G, _T, _IMG, _C = 2, 8, 3, 224, 10
_DEVICE = torch.device("cpu")


def _tiny_clf(model_config: dict) -> CanViTForImageClassification:
    torch.manual_seed(0)
    return CanViTForImageClassification(
        backbone_name="vits16", model_config=model_config, n_classes=_C, glimpse_grid_size=_G
    ).to(_DEVICE)


def _logits_and_loss(clf, freeze: bool):
    torch.manual_seed(1)
    images = torch.randn(_B, 3, _IMG, _IMG, device=_DEVICE)
    targets = torch.randint(0, _C, (_B,), device=_DEVICE)
    is_fov = consumes_full_image(clf)
    vps = make_random_viewpoints(
        _B, _DEVICE, _T, min_scale=0.05, max_scale=1.0, start_with_full_scene=True,
        is_foveated=is_fov, foveated_scale=FoveatedScaleConfig(),
    )
    cls_tokens = rollout_cls_tokens(
        clf=clf, images=images, viewpoints=vps, canvas_grid=_G, glimpse_px=None, freeze_backbone=freeze
    )
    assert len(cls_tokens) == _T
    logits = [clf.head(clf.norm(c)) for c in cls_tokens]
    assert logits[0].shape == (_B, _C)
    loss = torch.stack([ce_loss(lg, targets) for lg in logits]).mean()
    assert torch.isfinite(loss)
    loss.backward()
    return targets


def test_uniform_frozen_trains_head_only() -> None:
    clf = _tiny_clf({})
    assert not consumes_full_image(clf)
    clf.canvit.requires_grad_(False)
    clf.canvit.eval()
    _logits_and_loss(clf, freeze=True)
    head_grads = [p.grad for p in clf.head.parameters() if p.grad is not None]
    assert head_grads and any(g.abs().sum() > 0 for g in head_grads)
    assert all(p.grad is None for p in clf.canvit.parameters())


def test_foveated_frozen_full_image_routing() -> None:
    clf = _tiny_clf({"patcher_name": "foveated", "foveated_patcher": FoveatedPatcherConfig()})
    assert consumes_full_image(clf)  # foveated => full-image routing
    clf.canvit.requires_grad_(False)
    clf.canvit.eval()
    _logits_and_loss(clf, freeze=True)
    assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in clf.head.parameters())


def test_finetune_reaches_backbone() -> None:
    clf = _tiny_clf({})
    _logits_and_loss(clf, freeze=False)
    assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in clf.canvit.parameters())


def test_eval_viewpoints_shapes() -> None:
    for policy in ("coarse_to_fine", "full", "random"):
        vps = eval_viewpoints(
            policy, _B, _DEVICE, _T, is_foveated=False, foveated_scale=FoveatedScaleConfig()
        )
        assert len(vps) == _T
        assert vps[0].centers.shape == (_B, 2)


@pytest.mark.skipif(not _readable_dir(_IN1K_SHARDS), reason="IN1k webdataset shards not mounted on this node")
def test_train_pipeline_decodes_real_shards() -> None:
    """End-to-end on the real IN1k shards: the webdataset pipeline decodes
    jpg+json into (image [B,3,S,S], labels in [0,999]) with the train aug."""
    from ..data import build_train_pipeline, make_train_transform

    size = 128  # small crop for a quick CPU decode; real runs use scene_size=512
    tfm = make_train_transform(size, min_scale=0.35, flip_prob=0.5)
    shard_files = sorted(_IN1K_SHARDS.glob("shard-*.tar"))[:2]
    ds = build_train_pipeline(shard_files, transform=tfm, batch_size=4,
                              num_workers=0, shuffle_buffer=0, shuffle_seed=0)
    images, labels = next(iter(ds))
    assert images.shape == (4, 3, size, size), images.shape
    labels_t = torch.as_tensor(labels)
    assert labels_t.shape == (4,) and (labels_t >= 0).all() and (labels_t < 1000).all()


def test_metrics_topk() -> None:
    logits = torch.tensor([[5.0, 0, 0, 0], [0, 0, 3.0, 0], [0, 1.0, 0, 0]])
    targets = torch.tensor([0, 2, 2])  # first two correct@1, third wrong@1 but... class2 not in top1
    c = topk_correct(logits, targets, ks=(1, 2))
    assert c[1] == 2  # rows 0,1 correct at top-1
    acc = TopKAccuracy(ks=(1,))
    acc.update(logits, targets)
    assert abs(acc.compute()[1] - 2 / 3) < 1e-6


def test_finetune_fuses_a_probe_head_and_frozen_does_not(monkeypatch):
    """``mode=finetune`` MUST build its head from a pretrained DINOv3 probe, never a
    random one.

    A random head starts at chance (loss = ln(1000) ~ 6.9) and, at the finetune LR of
    6.25e-6, never recovers — the run looks healthy and learns almost nothing. exp33 catches
    this by eye ("watch the first train/full/loss sit well below 6.9"); this pins the
    routing so a refactor cannot quietly swap the two branches. The constructors are
    stubbed, so no weights are fetched and no network is touched.
    """
    from canvit.core import CanViTForImageClassification as Clf
    from canvit.in1k import model as in1k_model

    called: list[str] = []
    monkeypatch.setattr(Clf, "from_pretrained_with_probe",
                        classmethod(lambda cls, **kw: called.append("probe") or _Stub()))
    monkeypatch.setattr(Clf, "from_pretrained_with_new_head",
                        classmethod(lambda cls, **kw: called.append("new_head") or _Stub()))
    monkeypatch.setattr(in1k_model, "_resolve_probe_repo", lambda cfg: "stub/probe")

    from ..config import In1kConfig
    in1k_model.build_classifier(In1kConfig(mode="finetune"), _DEVICE)
    assert called == ["probe"], "finetune must fuse a pretrained probe into the head"

    called.clear()
    in1k_model.build_classifier(In1kConfig(mode="frozen"), _DEVICE)
    assert called == ["new_head"], "frozen mode trains from zero, so a fresh head is right"


class _Stub(torch.nn.Module):
    """Stands in for the classifier; ``build_classifier`` only calls ``.to(device)``."""

    def to(self, *a, **k):  # noqa: D102
        return self
