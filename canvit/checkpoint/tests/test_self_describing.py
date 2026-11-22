"""Downstream checkpoints must record their ARCHITECTURE, not just a ``model_repo`` path.

A pointer is not a description. It can move, be unreadable by another member of the
project, or -- for a FINETUNE -- stop describing the model that was trained, because
finetuning changed the backbone and the checkpoint is then the only place those weights
exist. So ``ade20k``/``in1k`` ``model_config()`` records exactly the constructor arguments
``CanViTFor{SemanticSegmentation,ImageClassification}.from_checkpoint`` needs.

The full weight-level round trip lives in CanViT-PyTorch's ``tests/test_checkpoint_sources.py``
(it owns the constructors). What is pinned here is the trainer's half: that every required
key is recorded, and that values are read off the MODEL rather than the config, so they
cannot disagree with the weights.
"""

from types import SimpleNamespace

from canvit.ade20k.config import Ade20kConfig
from canvit.ade20k.task import Ade20kRunTask
from canvit.core.model.base.config import CanViTConfig
from canvit.in1k.config import In1kConfig
from canvit.in1k.task import In1kRunTask

# Deliberately NOT the defaults: a value that survives only because it matches the default
# proves nothing about whether it was actually read off the model.
_ARCH = CanViTConfig(canvas_num_heads=4, canvas_head_dim=32, enable_vpe=False)


def _fake_model(*, dropout=0.25, use_ln=False, grid=6, backbone="vits16"):
    """Just the attributes ``model_config`` reads. A real model would need the source HF
    repo, which would make this test need network and a cache."""
    return SimpleNamespace(
        backbone_name=backbone,
        canvit=SimpleNamespace(cfg=_ARCH),
        glimpse_grid_size=grid,
        head=SimpleNamespace(dropout_p=dropout, use_ln=use_ln),
    )


def test_ade20k_records_the_architecture():
    mc = Ade20kRunTask(Ade20kConfig(model_repo="somewhere", canvas_grid=32)).model_config(
        _fake_model())
    # exactly what CanViTForSemanticSegmentation.from_checkpoint indexes
    assert mc["backbone_name"] == "vits16"
    assert mc["canvit"]["canvas_num_heads"] == 4 and mc["canvit"]["enable_vpe"] is False
    assert mc["num_classes"] == 150
    assert mc["glimpse_grid_size"] == 6
    assert mc["dropout"] == 0.25          # read off the head, not Ade20kConfig.dropout (0.1)
    assert mc["use_ln"] is False          # ditto: the wrapper's default is True
    assert mc["model_repo"] == "somewhere"   # kept as provenance, no longer load-bearing


def test_in1k_records_the_architecture():
    mc = In1kRunTask(In1kConfig(model_repo="somewhere", canvas_grid=32)).model_config(
        _fake_model())
    assert mc["backbone_name"] == "vits16"
    assert mc["canvit"]["canvas_head_dim"] == 32
    assert mc["n_classes"] == 1000
    assert mc["glimpse_grid_size"] == 6
    assert mc["model_repo"] == "somewhere"


def test_ddp_wrapped_model_is_unwrapped():
    """in1k supports DDP, so ``model`` may be a DistributedDataParallel whose attributes
    live under ``.module``. Reading through the wrapper would raise AttributeError mid-run,
    at the first checkpoint save rather than at startup."""
    inner = _fake_model(backbone="vitb16")
    wrapped = SimpleNamespace(module=inner)
    mc = In1kRunTask(In1kConfig(model_repo="somewhere", canvas_grid=32)).model_config(wrapped)
    assert mc["backbone_name"] == "vitb16"
    assert mc["canvit"]["canvas_num_heads"] == 4


def _built_from_model_config(task, mc):
    """``build_model(prior_model_config=mc)``: the path a resume and ``harness.evaluate`` take.
    ``model_repo='somewhere'`` does not exist, so reading it would fail the test."""
    import torch
    model, head = task.build_model(torch.device("cpu"), prior_model_config=mc)
    assert head is model.head
    return model


def test_ade20k_builds_the_recorded_architecture_without_its_model_repo():
    from dataclasses import asdict
    task = Ade20kRunTask(Ade20kConfig(model_repo="somewhere", canvas_grid=32))
    mc = task.model_config(_fake_model())
    model = _built_from_model_config(task, {**mc, "canvit": asdict(_ARCH)})
    assert task.model_config(model) == mc                  # the same architecture, round trip


def test_in1k_builds_the_recorded_architecture_without_its_model_repo():
    from dataclasses import asdict
    task = In1kRunTask(In1kConfig(model_repo="somewhere", canvas_grid=32))
    mc = task.model_config(_fake_model())
    model = _built_from_model_config(task, {**mc, "canvit": asdict(_ARCH)})
    assert task.model_config(model) == mc
