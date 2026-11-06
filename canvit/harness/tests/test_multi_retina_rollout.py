"""``run_rollout`` with a multi-retina patcher: each sample's retina is drawn once per
step, held for every glimpse of every branch, and cleared afterwards; a single-retina
patcher draws nothing (no RNG consumed, so the parity digest cannot move)."""

from types import SimpleNamespace

import torch
from torch import nn

from canvit.harness.config import FoveatedScaleConfig
from canvit.harness.rollout import GlimpseOut, run_rollout
from canvit.harness.rollout.engine import _patcher_of
from canvit.harness.rollout.selector import RandomSelector
from canvit.harness.rollout.viewpoint import ViewpointType
from canvit.harness.spec import BpttSpec

_B = 6


class _Patcher:
    def __init__(self, n_retinas: int):
        self.n_retinas = n_retinas
        self.active = None
        self.calls: list = []

    def set_retinas(self, retinas):
        self.calls.append(retinas)
        self.active = retinas


class _Model(nn.Module):
    def __init__(self, n_retinas: int):
        super().__init__()
        self.patcher = _Patcher(n_retinas)

    def init_state(self, batch_size, canvas_grid_size):
        return None


class _Task:
    """Records the patcher's retina choice at every glimpse; the loss is a toy one."""

    def __init__(self):
        self.w = nn.Parameter(torch.ones(()))
        self.seen: list = []

    def forward_glimpse(self, *, model, images, state, viewpoint, backbone_no_grad):
        self.seen.append(model.patcher.active)
        return GlimpseOut(readout=self.w * 1.0, state=state, vpe=None)

    def step_loss(self, readout):
        return SimpleNamespace(combined=readout ** 2)

    def per_image_loss(self, readout):
        return (readout ** 2).expand(_B)


def _rollout(model, task, n_branches=2):
    selector = RandomSelector(is_foveated=True, foveated_scale=FoveatedScaleConfig(fixed_scale=2.0),
                              min_viewpoint_scale=0.1)
    return run_rollout(model=model, images=torch.zeros(_B, 3, 8, 8), task=task, selector=selector,
                       bptt=BpttSpec(mode="full", horizon=3),
                       branches=[ViewpointType.FULL] * n_branches, canvas_grid_size=4,
                       amp_ctx=torch.autocast("cpu", enabled=False))


def test_retinas_drawn_once_held_for_every_glimpse_then_cleared():
    torch.manual_seed(0)
    model, task = _Model(n_retinas=9), _Task()
    _rollout(model, task)
    calls = model.patcher.calls
    assert len(calls) == 2 and calls[1] is None                  # one draw, then cleared
    ids = calls[0]
    assert ids.shape == (_B,) and ids.dtype == torch.long and 0 <= int(ids.min()) and int(ids.max()) < 9
    assert len(task.seen) == 2 * 3                               # 2 branches x 3 glimpses
    assert all(s is ids for s in task.seen)                      # same choice throughout
    torch.manual_seed(0)
    model2 = _Model(n_retinas=9)
    _rollout(model2, _Task())
    assert torch.equal(model2.patcher.calls[0], ids)             # torch's seeded RNG


def test_single_retina_draws_nothing():
    torch.manual_seed(0)
    state = torch.get_rng_state()
    model, task = _Model(n_retinas=1), _Task()
    _rollout(model, task)
    assert model.patcher.calls == [] and task.seen == [None] * 6
    # RandomSelector draws its viewpoints on the rollout's device RNG (CPU here), so compare
    # with a rollout that has no patcher at all: the same RNG state afterwards.
    after_single = torch.get_rng_state()
    torch.set_rng_state(state)
    plain = nn.Module()
    plain.init_state = lambda batch_size, canvas_grid_size: None
    _rollout(plain, _NoPatcherTask())
    assert torch.equal(torch.get_rng_state(), after_single)


class _NoPatcherTask(_Task):
    def forward_glimpse(self, *, model, images, state, viewpoint, backbone_no_grad):
        return GlimpseOut(readout=self.w * 1.0, state=state, vpe=None)


def test_patcher_found_on_model_and_wrapper():
    class Wrapper(nn.Module):
        def __init__(self, canvit):
            super().__init__()
            self.canvit = canvit

    m = _Model(n_retinas=3)
    assert _patcher_of(m) is m.patcher
    assert _patcher_of(Wrapper(m)) is m.patcher
    assert _patcher_of(nn.Linear(2, 2)) is None
