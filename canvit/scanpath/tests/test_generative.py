"""Generative scanpath metrics (design §4.2): hand-built cases with known answers."""

import math

import pytest
import torch

from canvit.scanpath.generative import (
    cut_at_target,
    fit_fixation_clusters,
    human_sequence_score,
    nw_score,
    probability_mismatch,
    sequence_score,
    target_hit_index,
    tfp_auc,
    tfp_curve,
)

NAN = math.nan
BOX = torch.tensor([[[0.4, 0.4, 0.6, 0.6]]])  # [1, 1, 4]


def paths(*ps, length=None):
    """Scanpaths as lists of (x, y) -> NaN-padded [B, L, 2] and n_fix [B]."""
    L = length or max(len(p) for p in ps)
    fix = torch.full((len(ps), L, 2), NAN)
    for i, p in enumerate(ps):
        fix[i, : len(p)] = torch.tensor(p)
    return fix, torch.tensor([len(p) for p in ps])


# --- TFP ---------------------------------------------------------------------------------------

def test_hit_at_fixation_two():
    fix, n = paths([(0, 0), (-0.5, -0.5), (0.5, 0.5), (0.9, 0.9)])
    hit = target_hit_index(fix, n, BOX)
    assert hit.tolist() == [2.0]
    curve = tfp_curve(hit, torch.zeros(1), max_fix=3)
    assert curve.tolist() == [0.0, 0.0, 1.0, 1.0]  # TFP(1) = 0, TFP(2) = 1
    assert tfp_auc(curve).item() == 2.0


def test_box_is_strict_and_any_box_counts():
    on_edge = [(0, 0), (0.4, 0.5)]  # x == x0: not inside, as the reference's X > x0
    fix, n = paths(on_edge, [(0, 0), (0.1, 0.1), (-0.5, -0.5)])
    boxes = torch.tensor([[[0.4, 0.4, 0.6, 0.6], [NAN] * 4],
                          [[0.4, 0.4, 0.6, 0.6], [-0.6, -0.6, -0.4, -0.4]]])  # NaN row = padding
    assert target_hit_index(fix, n, boxes).tolist() == [math.inf, 2.0]


def test_start_fixation_counts_as_index_zero():
    fix, n = paths([(0.5, 0.5), (0, 0)])
    hit = target_hit_index(fix, n, BOX)
    assert hit.tolist() == [0.0]
    assert tfp_curve(hit, torch.zeros(1), max_fix=2).tolist() == [1.0, 1.0, 1.0]


def test_padding_beyond_n_fix_is_ignored():
    fix, _ = paths([(0, 0), (-0.5, -0.5), (0.5, 0.5)], length=5)
    n = torch.tensor([2])  # the hit at index 2 lies in the padding
    assert target_hit_index(fix, n, BOX).tolist() == [math.inf]
    assert cut_at_target(torch.tensor([1.0, math.inf]), torch.tensor([4, 4])).tolist() == [2, 4]


def test_curve_averages_within_then_over_tasks():
    hit = torch.tensor([1.0, math.inf, math.inf, math.inf])
    task = torch.tensor([0, 1, 1, 1])
    assert tfp_curve(hit, task, max_fix=1).tolist() == [0.0, 0.5]  # pooled would be 0.25
    assert tfp_curve(hit, torch.zeros(4), max_fix=1).tolist() == [0.0, 0.25]


def test_probability_mismatch():
    hit = torch.tensor([1.0, 3.0, math.inf])
    task = torch.zeros(3)
    human = tfp_curve(hit, task, max_fix=4)
    assert probability_mismatch(tfp_curve(hit.clone(), task, max_fix=4), human).item() == 0.0
    model = tfp_curve(torch.tensor([1.0, 1.0, 1.0]), task, max_fix=4)  # [0, 1, 1, 1, 1]
    # human [0, 1/3, 1/3, 2/3, 2/3]: |diff| = 0 + 2/3 + 2/3 + 1/3 + 1/3
    assert probability_mismatch(model, human).item() == pytest.approx(2.0)
    with pytest.raises(ValueError, match="length"):
        probability_mismatch(model[:3], human)


def test_refuses_invalid_input():
    fix, n = paths([(0, 0), (NAN, 0.5)])
    with pytest.raises(ValueError, match="non-finite"):
        target_hit_index(fix, n, BOX)
    fix, n = paths([(0, 0)])
    with pytest.raises(ValueError, match="target box"):
        target_hit_index(fix, n, torch.full((1, 1, 4), NAN))
    with pytest.raises(ValueError, match="n_fix"):
        target_hit_index(fix, torch.tensor([2]), BOX)


# --- sequence score ----------------------------------------------------------------------------

CORNERS = [(-0.5, -0.5), (0.5, -0.5), (0.5, 0.5), (-0.5, 0.5)]
PATH = [(0.0, 0.0)] + CORNERS  # start fixation, then four well-separated places
REVERSED = [(0.0, 0.0)] + CORNERS[::-1]


def humans(jitters=((0, 0), (0.01, 0), (0, 0.01))):
    return paths(*[[(x + dx, y + dy) for x, y in PATH] for dx, dy in jitters])


def test_nw_score():
    assert nw_score([1, 2, 3], [1, 2, 3]) == 1.0
    assert nw_score([1, 2, 3, 4], [4, 3, 2, 1]) == 0.25
    assert nw_score([1, 2], [1, 3, 2, 4]) == 0.5  # two matches over the longer length 4
    assert nw_score([5], [6]) == 0.0


def test_one_cluster_per_place():
    ms = fit_fixation_clusters(*humans())
    assert len(ms.cluster_centers_) == 5


def test_identical_scores_one_and_reversed_lower():
    model, n = paths(PATH, REVERSED)
    ss = sequence_score(model, n, *humans())
    assert ss[0].item() == 1.0
    assert ss[1].item() == pytest.approx(2 / 5)  # common: the start and one corner
    assert human_sequence_score(*humans()).tolist() == [1.0, 1.0, 1.0]


def test_sequence_score_ignores_padding():
    model, n = paths(PATH, REVERSED)
    padded = torch.cat([model, torch.full((2, 3, 2), NAN)], dim=1)
    padded[0, 5] = torch.tensor([0.5, 0.5])  # a real-looking point beyond n_fix
    assert torch.equal(sequence_score(padded, n, *humans()), sequence_score(model, n, *humans()))


def test_sequence_score_is_scale_free():
    """The bandwidth is estimated from the data, so pixels vs scene coordinates do not matter."""
    rng = torch.Generator().manual_seed(0)
    h_fix, h_n = paths(*[(torch.rand(6, 2, generator=rng) * 2 - 1).tolist() for _ in range(5)])
    m_fix, m_n = paths(*[(torch.rand(4, 2, generator=rng) * 2 - 1).tolist() for _ in range(3)])
    to_px = lambda f: (f + 1) * 256  # noqa: E731  scene -> 512 px, isotropic as the letterbox
    assert torch.allclose(sequence_score(m_fix, m_n, h_fix, h_n),
                          sequence_score(to_px(m_fix), m_n, to_px(h_fix), h_n))
