"""The two saccade-cost shapes, exactly as the RL training defines them (``SaccadeCost.cost`` in
``canvit/harness/policy/joint.py`` on branch ``saccade-cost``; the same formulas, tested equal there).

d: distance between consecutive glimpse centres in scene units [-1, 1] (scene width = 2).
r: half a policy-grid cell = 1 / grid (the training's ``1 / centers_per_axis``).

    log:     c(d) = log(1 + d / r)
    linear:  c(d) = d / 2            (the fraction of the scene width)

Post hoc fits only the weight (lambda / alpha) of one of these, plus tau. When ``saccade-cost`` is merged,
``posthoc`` imports the harness function and this module goes; importers to change then: ``saccade_cost.py``,
``priors.py`` (amplitude prior, helper_3), ``tests/test_priors.py``; ``tests/test_cost_forms.py`` goes too.

Written by Claude (AI, Claude Opus 5.5), 2026-11-25, at Valentin's request (two cost shapes, one definition)."""

import torch
from torch import Tensor

FORMS = ("log", "linear")


def saccade_cost(form: str, d: Tensor, r: float | Tensor) -> Tensor:
    if form == "log":
        return torch.log1p(d / r)
    assert form == "linear", f"no cost for form={form!r}"
    return d / 2.0
