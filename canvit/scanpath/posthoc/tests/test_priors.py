"""Human-bias priors (priors.py): reduction to the model, exactness of the IG shift, and recovery of known
parameters from synthetic fixations drawn from the prior model itself."""

import numpy as np
import pytest
import torch

from canvit.scanpath.posthoc.priors import (
    CellBatch,
    cell_centres,
    cell_log_mean,
    cell_log_probs,
    features,
    fit,
    image_average,
    log_probs,
)

G = 16


def synthetic(n: int, seed: int = 0, logits_scale: float = 0.0) -> CellBatch:
    """Fixations on a G x G grid with every cell on the image (logcount 0), random histories of 1-4 fixations."""
    rng = np.random.default_rng(seed)
    hist = [rng.uniform(-0.9, 0.9, size=(rng.integers(1, 5), 2)) for _ in range(n)]
    return CellBatch(logits=logits_scale * rng.normal(size=(n, G * G)), logcount=np.zeros((n, G * G)),
                     cell=np.zeros(n, dtype=int), t=np.array([len(h) for h in hist]), n=rng.integers(0, 20, n),
                     hist=hist, log_cb=np.log(np.full((n, G * G), 1.0 / (G * G))))


def draw(b: CellBatch, params: dict, seed: int = 1) -> CellBatch:
    """The same batch with each fixation's cell drawn from the model with ``params``."""
    lp = log_probs({k: torch.tensor(v, dtype=torch.float64) for k, v in params.items()}, features(b)).numpy()
    rng = np.random.default_rng(seed)
    cells = np.array([rng.choice(G * G, p=np.exp(r) / np.exp(r).sum()) for r in lp])
    return CellBatch(**{**b.__dict__, "cell": cells})


def test_no_prior_is_the_model_at_tau_one():
    b = synthetic(50, logits_scale=2.0)
    lp = log_probs({}, features(b)).numpy()
    expected = b.logits - np.log(np.exp(b.logits).sum(1, keepdims=True))
    np.testing.assert_allclose(lp, expected, atol=1e-6)


def test_cells_off_the_image_get_no_mass():
    b = synthetic(10, logits_scale=1.0)
    b.logcount[:, :G] = -np.inf  # the first row of cells is letterbox padding
    lp = log_probs({"amp_w": torch.tensor(1.0, dtype=torch.float64)}, features(b)).numpy()
    assert np.all(np.isneginf(lp[:, :G])) and np.allclose(np.exp(lp).sum(1), 1.0)


def test_cell_centres_are_row_major_and_symmetric():
    c = cell_centres(4)
    assert np.allclose(c[0], [-0.75, -0.75]) and np.allclose(c[1], [-0.25, -0.75]) and np.allclose(c[4], [-0.75, -0.25])


def test_fit_recovers_known_prior_parameters():
    truth = {"amp_w": 1.5, "ret_w": -2.0, "ret_log_s": float(np.log(0.15)), "dir_card": 0.8,
             "dir_lr": -0.5}
    b = draw(synthetic(4000), truth)
    est = fit(b, ("amplitude", "return", "direction"), with_model=False)
    for k, v in truth.items():
        assert est[k] == pytest.approx(v, abs=0.25 * max(1.0, abs(v))), (k, est[k], v)


def test_fit_recovers_the_temperature():
    b = synthetic(3000, logits_scale=3.0)
    b = draw(b, {"log_tau": float(np.log(2.5))})
    est = fit(b, (), with_model=True)
    assert np.exp(est["log_tau"]) == pytest.approx(2.5, rel=0.1)


def test_fitting_never_lowers_the_training_likelihood():
    b = draw(synthetic(1500, logits_scale=2.0), {"log_tau": 0.5, "amp_w": 1.5})
    base = cell_log_probs(b, {"log_tau": 0.0}).mean()
    for priors in ((), ("amplitude",), ("amplitude", "return", "direction", "centre")):
        assert cell_log_probs(b, fit(b, priors)).mean() >= base - 1e-9


def test_image_average_is_mean_of_per_image_means():
    v, n = np.array([1.0, 3.0, 10.0]), np.array([0, 0, 1])
    assert image_average(v, n) == pytest.approx((2.0 + 10.0) / 2)


def test_cell_log_mean_matches_the_per_cell_logsumexp():
    """The bincount version equals the per-cell masks it replaced (also with a large dynamic range)."""
    from scipy.special import logsumexp

    from canvit.scanpath.models import _pixel_cells
    rng = np.random.default_rng(0)
    for h, w, frame in ((48, 80, None), (60, 45, None), (30, 50, (1050, 1680))):
        cell, count = _pixel_cells(h, w, 64, 8, frame)
        ld = rng.normal(0.0, 30.0, (h, w))
        ref = np.zeros(len(count))
        for c in np.unique(cell):
            ref[c] = logsumexp(ld[cell == c]) - np.log(count[c])
        assert np.allclose(cell_log_mean(ld, cell, count), ref, rtol=0, atol=1e-10)


def test_the_amplitude_prior_is_the_training_log_cost():
    """log p moves by -amp_w * log(1 + d / r) per cell, r = 1 / G (cost_forms, the RL training's definition)."""
    from canvit.scanpath.posthoc.cost_forms import saccade_cost
    b = synthetic(20, logits_scale=1.0)
    f = features(b)
    w = torch.tensor(0.7, dtype=torch.float64)
    shift = (log_probs({"amp_w": w}, f) - log_probs({}, f)).numpy()
    cur = np.stack([h[-1] for h in b.hist])
    d = np.linalg.norm(cell_centres(G)[None] - cur[:, None], axis=-1)
    want = -0.7 * saccade_cost("log", torch.as_tensor(d), 1.0 / G).numpy()
    np.testing.assert_allclose(shift - shift.mean(1, keepdims=True), want - want.mean(1, keepdims=True), atol=1e-5)
