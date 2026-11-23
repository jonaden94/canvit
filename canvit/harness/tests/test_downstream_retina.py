"""``--cfg.retina`` is a downstream-task option, not a search one: ade20k, in1k and search all use
a multi-retina backbone through one named retina (``core.patcher.foveated.fix_backbone_retina``),
require the choice for such a backbone and refuse it for a single-retina one. Tiny 2-retina
foveated ViT-S, architecture given as a checkpoint's ``model_config``."""

import pytest
import torch

from canvit.ade20k.config import Ade20kConfig
from canvit.ade20k.task import Ade20kRunTask
from canvit.in1k.config import In1kConfig
from canvit.in1k.task import In1kRunTask

_FOV = {"fov": 50, "resolution": 64, "cmf_a": 0.5, "cart_patch_size": 5, "arch_flag": "doubleres",
        "rf_interp": "bilinear",
        "conditioning": {"mode": "film", "film": {"fourier": {"num_features": 256, "sigma": 4}}}}
_MULTI = {"patcher_name": "foveated",
          "foveated_patcher": {**_FOV, "retinas": [[0.75, 32, 3, False], [2.2, 32, 3, False]]}}
_SINGLE = {"patcher_name": "foveated", "foveated_patcher": _FOV}


def _mc(canvit):
    return {"backbone_name": "vits16", "canvit": canvit, "glimpse_grid_size": None,
            "num_classes": 150, "n_classes": 1000}


_TASKS = [lambda **kw: Ade20kRunTask(Ade20kConfig(model_repo="nowhere", canvas_grid=32, **kw)),
          lambda **kw: In1kRunTask(In1kConfig(model_repo="nowhere", canvas_grid=32, **kw))]


@pytest.mark.parametrize("make", _TASKS, ids=["ade20k", "in1k"])
def test_a_multi_retina_backbone_is_used_through_the_named_retina(make):
    model, _ = make(retina="cmf2.2_res32_cps3").build_model(torch.device("cpu"), prior_model_config=_mc(_MULTI))
    patcher = model.canvit.patcher
    assert patcher.fixed_retina == 1 and patcher.active_retina(0) == 1


@pytest.mark.parametrize("make", _TASKS, ids=["ade20k", "in1k"])
def test_the_retina_is_required_for_multi_and_refused_for_single(make):
    with pytest.raises(ValueError, match="multi-retina backbone: choose one retina"):
        make().build_model(torch.device("cpu"), prior_model_config=_mc(_MULTI))
    with pytest.raises(ValueError, match="needs a multi-retina backbone"):
        make(retina="cmf0.75_res32_cps3").build_model(torch.device("cpu"), prior_model_config=_mc(_SINGLE))
