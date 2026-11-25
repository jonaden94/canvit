"""saccade_cost: the fit recovers a known cost weight (the training's cost shapes); the validation shift is exact at
the record's own model. Written by Claude (AI, Claude Opus 5.5), 2026-11-24; two shapes only since 2026-11-25."""

import numpy as np
import pytest
import torch

from canvit.scanpath.posthoc import saccade_cost as sc


def _data(key: str, w: float, form: str = "log", n: int = 4000, cells: int = 64, seed: int = 0) -> dict:
    rng = np.random.default_rng(seed)
    d = rng.uniform(0.0, 2.0, (n, cells))  # scene units
    r = np.full(n, 1 / 32)
    logits = rng.normal(0.0, 1.0, (n, cells))
    c = np.log1p(d / r[:, None]) if form == "log" else d / 2
    z = logits / 3.0 - w * c  # tau = 3
    p = np.exp(z - z.max(1, keepdims=True))
    p /= p.sum(1, keepdims=True)
    cell = np.array([rng.choice(cells, p=q) for q in p])
    return sc._tensors(key, {"logits": logits, "logcount": np.zeros((n, cells)), "cell": cell,
                             "n": np.arange(n) // 4, "d": d, "r": r})


@pytest.mark.parametrize("form, w", [("log", 1.5), ("linear", 3.0)])
def test_fit_recovers_the_cost_weight(form, w):
    D = _data("tp", w=w, form=form)
    p, se, cov = sc.fit(form, [D])
    assert list(p) == ["log_tau_tp", "w"]
    assert abs(p["w"] - w) < 4 * se["w"] + 0.05 * w
    assert abs(np.exp(p["log_tau_tp"]) - 3.0) < 0.6
    assert se["w"] > 0 and cov.shape == (2, 2) and np.isclose(np.sqrt(cov[-1, -1]), se["w"])


def test_only_the_training_shapes():
    assert set(sc.FORMS) == {"none", "log", "linear"}


def test_centre_bias_base():
    D = _data("tp", w=1.0, form="linear")
    D["log_cb"] = torch.zeros_like(D["logits"])  # a flat centre bias: the base is uniform over the cells
    p, se, _ = sc.fit("linear", [D], base="cb")
    assert list(p) == ["w"] and p["w"] > 0  # a falling prior; no tau without the policy
    ig = sc.val_ig("none", {}, D, record_ig=0.0, base="cb")
    assert np.isclose(ig[0], 0.0)  # the centre bias over itself


def test_val_shift_is_zero_at_the_record_model():
    D = _data("tp", w=0.0, n=200)
    ig = sc.val_ig("none", {"log_tau_tp": 0.0}, D, record_ig=-1.234)  # tau = 1, no prior = the record itself
    assert np.isclose(ig[0], -1.234)
    ig2 = sc.val_ig("log", {"w": 0.0}, D, record_ig=-1.234, taus=[1.0, 2.0])
    assert np.isclose(ig2[0], -1.234) and not np.isclose(ig2[1], -1.234)
    assert torch.allclose(sc.log_p("none", {}, D, tau=1.0).exp().sum(1), torch.ones(200, dtype=torch.float64))
