"""Equivalence of the two sample average uniqueness implementations.

Sample average uniqueness (AFML Snippet 4.4, repeats counted) has two implementations
in the package, on purpose:

* :func:`finlab.weights.sample_average_uniqueness`, the dense reference. It takes a
  bar-by-label indicator matrix and the draws, and uses one matrix product.
* :func:`finlab.monte_carlo._sample_uniqueness`, the numba kernel used by the Monte
  Carlo path. It takes the number of bars and the span of each draw.

The tests feed both the same labels and draws and require equal results. The dense
matrix is built here with numpy, independently of either library function, and is
checked against :func:`finlab.weights.indicator_matrix` in every case.

Fixed seeds: ``range(200)`` for the randomized sweep and ``range(30)`` for the
edge-case sweeps. In the sweep, ``n_bars = 1 + seed % 40`` so every size from 1 to 40
occurs, and the number of labels is drawn from 1 to 30.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from finlab.monte_carlo import _sample_uniqueness
from finlab.weights import indicator_matrix, sample_average_uniqueness

MAX_BARS = 40
MAX_LABELS = 30
SWEEP_SEEDS = range(200)
EDGE_SEEDS = range(30)

# Relative tolerance. Both implementations add the same terms 1/c_t over the same spans,
# but in a different order: the dense one uses a matrix product and np.mean, the kernel
# uses sequential loops. Each span sum has at most MAX_BARS terms and the mean over draws
# has at most 2 * MAX_LABELS terms, all in (0, 1]. A worst-case bound on the rounding is
# about (40 + 60) * 2.2e-16 = 2.2e-14, so 1e-12 leaves a margin of about 50. A real
# difference in the formula would be far larger than that.
RTOL = 1e-12


def _random_spans(rng, n_bars, n_labels):
    """Inclusive bar spans ``[start, end]`` with the awkward cases mixed in.

    Each label is one of: an exact copy of an earlier label (identical span), a copy of
    an earlier start with a new end (duplicate start), a length-1 span, a span ending at
    the last bar, or a random span.
    """
    starts = np.empty(n_labels, dtype=np.int64)
    ends = np.empty(n_labels, dtype=np.int64)
    for k in range(n_labels):
        kind = int(rng.integers(0, 5))
        if kind == 0 and k > 0:
            j = int(rng.integers(0, k))
            starts[k], ends[k] = starts[j], ends[j]
        elif kind == 1 and k > 0:
            starts[k] = starts[int(rng.integers(0, k))]
            ends[k] = rng.integers(starts[k], n_bars)
        else:
            starts[k] = rng.integers(0, n_bars)
            if kind == 2:
                ends[k] = starts[k]
            elif kind == 3:
                ends[k] = n_bars - 1
            else:
                ends[k] = rng.integers(starts[k], n_bars)
    return starts, ends


def _random_draws(rng, n_labels):
    """Draw positions with repeats. The size is ``n_labels`` half the time, else 1 to 2n."""
    if rng.random() < 0.5:
        size = n_labels
    else:
        size = int(rng.integers(1, 2 * n_labels + 1))
    return rng.integers(0, n_labels, size=size).astype(np.int64)


def _indicator(n_bars, starts, ends):
    """Dense 0/1 matrix, bars as rows and labels as columns, built from the spans."""
    bars = np.arange(n_bars)[:, None]
    return ((bars >= starts) & (bars <= ends)).astype(np.float64)


def _assert_agree(starts, ends, n_bars, draws, where):
    """Feed both implementations the same labels and draws; return the shared value.

    ``draws=None`` means every label once, in order, for both implementations.
    """
    m = _indicator(n_bars, starts, ends)
    t1 = pd.Series(ends, index=pd.Index(starts))
    np.testing.assert_array_equal(
        indicator_matrix(np.arange(n_bars), t1).to_numpy(), m, err_msg=f"indicator: {where}"
    )
    pos = np.arange(len(starts), dtype=np.int64) if draws is None else draws
    dense = sample_average_uniqueness(m, draws)
    kernel = _sample_uniqueness(n_bars, starts[pos], ends[pos])
    np.testing.assert_allclose(kernel, dense, rtol=RTOL, atol=0, err_msg=where)
    return dense


def test_both_implementations_agree_on_random_samples():
    coverage = {
        "duplicate start": 0,
        "identical span": 0,
        "length-one span": 0,
        "span ends at last bar": 0,
        "repeated draw": 0,
        "single draw": 0,
    }
    checks = 0
    for seed in SWEEP_SEEDS:
        rng = np.random.default_rng(seed)
        n_bars = 1 + seed % MAX_BARS
        n_labels = int(rng.integers(1, MAX_LABELS + 1))
        starts, ends = _random_spans(rng, n_bars, n_labels)
        draws = _random_draws(rng, n_labels)
        where = (
            f"seed={seed} n_bars={n_bars} starts={starts.tolist()} "
            f"ends={ends.tolist()} draws={draws.tolist()}"
        )
        _assert_agree(starts, ends, n_bars, draws, where)
        _assert_agree(starts, ends, n_bars, None, where + " draws=None")
        checks += 2

        coverage["duplicate start"] += int(len(np.unique(starts)) < n_labels)
        coverage["identical span"] += int(
            len({(int(s), int(e)) for s, e in zip(starts, ends)}) < n_labels
        )
        coverage["length-one span"] += int(np.any(ends == starts))
        coverage["span ends at last bar"] += int(np.any(ends == n_bars - 1))
        coverage["repeated draw"] += int(len(np.unique(draws)) < len(draws))
        coverage["single draw"] += int(len(draws) == 1)

    assert checks == 2 * len(SWEEP_SEEDS)
    # The sweep must actually hit each awkward case, or the agreement means little.
    missing = [name for name, count in coverage.items() if count == 0]
    assert not missing, f"sweep never produced: {missing}; counts={coverage}"


@pytest.mark.parametrize(
    ("n_bars", "n_labels"), [(1, 1), (1, MAX_LABELS), (MAX_BARS, 1), (MAX_BARS, MAX_LABELS)]
)
def test_size_corners_agree(n_bars, n_labels):
    for seed in EDGE_SEEDS:
        rng = np.random.default_rng(seed)
        starts, ends = _random_spans(rng, n_bars, n_labels)
        draws = _random_draws(rng, n_labels)
        where = f"n_bars={n_bars} n_labels={n_labels} seed={seed} draws={draws.tolist()}"
        _assert_agree(starts, ends, n_bars, draws, where)
        _assert_agree(starts, ends, n_bars, None, where + " draws=None")


def test_single_draw_has_uniqueness_exactly_one():
    # One draw covers its bars alone, so c = 1 there and every term is exactly 1.
    for seed in EDGE_SEEDS:
        rng = np.random.default_rng(seed)
        n_bars = 1 + seed % MAX_BARS
        n_labels = int(rng.integers(1, MAX_LABELS + 1))
        starts, ends = _random_spans(rng, n_bars, n_labels)
        for d in range(n_labels):
            where = (
                f"seed={seed} n_bars={n_bars} starts={starts.tolist()} ends={ends.tolist()} d={d}"
            )
            value = _assert_agree(starts, ends, n_bars, np.array([d], dtype=np.int64), where)
            assert value == 1.0, where


def test_all_draws_identical_give_one_over_n():
    # n copies of one label: c = n on its span, so each term is 1/n and the mean is 1/n.
    for seed in EDGE_SEEDS:
        rng = np.random.default_rng(seed)
        n_bars = 1 + seed % MAX_BARS
        n_labels = int(rng.integers(1, MAX_LABELS + 1))
        starts, ends = _random_spans(rng, n_bars, n_labels)
        d = int(rng.integers(0, n_labels))
        for n in (1, 2, 5, 17):
            draws = np.full(n, d, dtype=np.int64)
            where = f"seed={seed} n_bars={n_bars} starts={starts.tolist()} ends={ends.tolist()} draws={draws.tolist()}"
            value = _assert_agree(starts, ends, n_bars, draws, where)
            np.testing.assert_allclose(value, 1.0 / n, rtol=RTOL, atol=0, err_msg=where)


def test_disjoint_labels_have_uniqueness_one():
    # Labels on separate blocks of bars, each drawn at most once: c = 1 on every span.
    for seed in EDGE_SEEDS:
        rng = np.random.default_rng(seed)
        n_labels = int(rng.integers(1, MAX_LABELS + 1))
        lengths = rng.integers(1, 5, size=n_labels).astype(np.int64)
        gaps = rng.integers(0, 3, size=n_labels).astype(np.int64)
        starts = np.cumsum(lengths + gaps) - lengths
        ends = starts + lengths - 1
        n_bars = int(ends[-1]) + 1 + int(rng.integers(0, 3))
        k = int(rng.integers(1, n_labels + 1))
        draws = rng.permutation(n_labels)[:k].astype(np.int64)
        where = f"seed={seed} n_bars={n_bars} starts={starts.tolist()} ends={ends.tolist()} draws={draws.tolist()}"
        value = _assert_agree(starts, ends, n_bars, draws, where)
        np.testing.assert_allclose(value, 1.0, rtol=RTOL, atol=0, err_msg=where)
        value_all = _assert_agree(starts, ends, n_bars, None, where + " draws=None")
        np.testing.assert_allclose(value_all, 1.0, rtol=RTOL, atol=0, err_msg=where)
