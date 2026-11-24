"""saccade_cost: the fit recovers a known log prior; the validation shift is exact at the record's own model.
Written by Claude (AI, Claude Opus 5.5), 2026-11-24."""

import numpy as np
import torch

from canvit.scanpath import saccade_cost as sc


def _data(key: str, a: float, n: int = 4000, cells: int = 64, seed: int = 0) -> dict:
    rng = np.random.default_rng(seed)
    d = rng.uniform(0.0, 30.0, (n, cells))
    r = np.full(n, 0.8)
    logits = rng.normal(0.0, 1.0, (n, cells))
    z = logits / 3.0 + a * np.log(d + r[:, None])  # tau = 3
    p = np.exp(z - z.max(1, keepdims=True))
    p /= p.sum(1, keepdims=True)
    cell = np.array([rng.choice(cells, p=q) for q in p])
    return sc._tensors(key, {"logits": logits, "logcount": np.zeros((n, cells)), "cell": cell,
                             "n": np.arange(n) // 4, "d": d, "r": r})


def test_fit_recovers_log_prior():
    D = _data("tp", a=-1.5)
    p, se = sc.fit("log", [D])
    assert abs(p["a"] + 1.5) < 4 * se["a"] + 0.05
    assert abs(np.exp(p["log_tau_tp"]) - 3.0) < 0.6
    assert se["a"] > 0


def test_val_shift_is_zero_at_the_record_model():
    D = _data("tp", a=0.0, n=200)
    ig = sc.val_ig("none", {"log_tau_tp": 0.0}, D, record_ig=-1.234)  # tau = 1, no prior = the record itself
    assert np.isclose(ig[0], -1.234)
    ig2 = sc.val_ig("log", {"a": 0.0}, D, record_ig=-1.234, taus=[1.0, 2.0])
    assert np.isclose(ig2[0], -1.234) and not np.isclose(ig2[1], -1.234)
    assert torch.allclose(sc.log_p("none", {}, D, tau=1.0).exp().sum(1), torch.ones(200, dtype=torch.float64))
