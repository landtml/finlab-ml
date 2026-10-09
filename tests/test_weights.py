"""Tests for finlab.weights (AFML Chapter 4).

Fast functions are checked against brute-force references written in this
file: concurrency and uniqueness by explicit double loops over bars and labels,
return attribution by per-label loops, and the sequential bootstrap by a
dense naive loop that consumes the same uniforms.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from finlab.weights import (
    average_uniqueness,
    indicator_matrix,
    num_co_events,
    sample_weight_by_return,
    sequential_bootstrap,
    time_decay,
)


def _random_labels(n_bars: int, n_labels: int, max_span: int, seed: int):
    """Random bar grid and labels whose end times are bar times."""
    rng = np.random.default_rng(seed)
    index = pd.bdate_range("2021-01-01", periods=n_bars)
    starts = np.sort(rng.choice(n_bars - max_span - 1, size=n_labels, replace=False))
    spans = rng.integers(0, max_span, size=n_labels)
    ends = np.minimum(starts + spans, n_bars - 1)
    t1 = pd.Series(index[ends], index=index[starts])
    return index, t1


# ---------------------------------------------------------------------------
# Brute-force references
# ---------------------------------------------------------------------------


def naive_concurrency(index, t1):
    """c_t = #{i : t0_i <= t <= t1_i}, by an explicit double loop over bars and labels."""
    out = np.zeros(len(index), dtype=np.int64)
    for k, b in enumerate(index):
        for t0, t_end in t1.items():
            if t0 <= b <= t_end:
                out[k] += 1
    return out


def naive_avg_uniqueness(index, t1, c):
    out = []
    for t0, t_end in t1.items():
        vals = [1.0 / c[k] for k, b in enumerate(index) if t0 <= b <= t_end]
        out.append(sum(vals) / len(vals) if vals else np.nan)
    return np.array(out)


def naive_return_weights(index, t1, close):
    logp = np.log(close.reindex(index).to_numpy(dtype=float))
    r = np.zeros(len(index))
    r[1:] = np.diff(logp)
    c = naive_concurrency(index, t1)
    raw = []
    for t0, t_end in t1.items():
        acc = 0.0
        for k, b in enumerate(index):
            if t0 <= b <= t_end:
                acc += r[k] / c[k]
        raw.append(abs(acc))
    raw = np.array(raw)
    return raw * len(raw) / raw.sum()


def naive_indicator(index, t1):
    m = np.zeros((len(index), len(t1)))
    for i, (t0, t_end) in enumerate(t1.items()):
        for k, b in enumerate(index):
            if t0 <= b <= t_end:
                m[k, i] = 1.0
    return m


def naive_sequential_draws(m, uniforms):
    """Snippet 4.5 with the same inverse-CDF rule as the kernel, as explicit loops."""
    n_bars, n_ev = m.shape
    colsum = (m != 0).sum(axis=0)
    c = np.zeros(n_bars)
    draws = []
    for u in uniforms:
        avg = np.zeros(n_ev)
        for j in range(n_ev):
            if colsum[j] > 0:
                s = 0.0
                for t in range(n_bars):
                    if m[t, j] != 0:
                        s += 1.0 / (1.0 + c[t])
                avg[j] = s / colsum[j]
        total = 0.0
        for j in range(n_ev):
            total += avg[j]
        target = u * total
        acc = 0.0
        pick = -1
        last = -1
        for j in range(n_ev):
            if avg[j] > 0.0:
                last = j
                acc += avg[j]
                if acc > target:
                    pick = j
                    break
        if pick < 0:
            pick = last
        draws.append(pick)
        for t in range(n_bars):
            if m[t, pick] != 0:
                c[t] += 1.0
    return np.array(draws, dtype=np.int64)


def sample_avg_uniqueness(m, draws):
    """Average uniqueness of a bootstrap sample, counting repeats (Snippet 4.4 style)."""
    c = (m[:, draws] != 0).sum(axis=1).astype(float)
    vals = []
    for d in draws:
        rows = np.nonzero(m[:, d])[0]
        vals.append(np.mean(1.0 / c[rows]))
    return float(np.mean(vals))


# ---------------------------------------------------------------------------
# Concurrency and uniqueness
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("seed", range(4))
def test_num_co_events_matches_naive_double_loop(seed):
    index, t1 = _random_labels(120, 25, max_span=15, seed=seed)
    got = num_co_events(index, t1)
    np.testing.assert_array_equal(got.to_numpy(), naive_concurrency(index, t1))


def test_num_co_events_open_labels_run_to_last_bar():
    index = pd.bdate_range("2021-01-01", periods=10)
    t1 = pd.Series([pd.NaT, index[3]], index=index[[0, 2]])
    got = num_co_events(index, t1).to_numpy()
    np.testing.assert_array_equal(got, [1, 1, 2, 2, 1, 1, 1, 1, 1, 1])


def test_average_uniqueness_is_one_without_overlap():
    index = pd.bdate_range("2021-01-01", periods=12)
    t1 = pd.Series(index[[2, 6, 10]], index=index[[0, 4, 8]])
    c = num_co_events(index, t1)
    avg = average_uniqueness(index, t1, c)
    np.testing.assert_allclose(avg.to_numpy(), 1.0, rtol=0, atol=1e-15)


@pytest.mark.parametrize("seed", range(3))
def test_average_uniqueness_matches_naive(seed):
    index, t1 = _random_labels(150, 30, max_span=20, seed=seed + 10)
    c = num_co_events(index, t1)
    got = average_uniqueness(index, t1, c)
    ref = naive_avg_uniqueness(index, t1, c.to_numpy())
    np.testing.assert_allclose(got.to_numpy(), ref, rtol=1e-12)
    assert np.all((got.to_numpy() > 0) & (got.to_numpy() <= 1.0 + 1e-12))


def test_average_uniqueness_is_reciprocal_harmonic_mean_of_concurrency():
    index, t1 = _random_labels(80, 15, max_span=12, seed=21)
    c = num_co_events(index, t1)
    avg = average_uniqueness(index, t1, c).to_numpy()
    for k, (t0, t_end) in enumerate(t1.items()):
        span_c = c[(index >= t0) & (index <= t_end)].to_numpy().astype(float)
        harmonic = len(span_c) / np.sum(1.0 / span_c)
        assert avg[k] == pytest.approx(1.0 / harmonic, rel=1e-12)


def test_indicator_matrix_matches_naive():
    index, t1 = _random_labels(60, 12, max_span=10, seed=5)
    got = indicator_matrix(index, t1)
    assert got.shape == (len(index), len(t1))
    np.testing.assert_array_equal(got.to_numpy(), naive_indicator(index, t1))


# ---------------------------------------------------------------------------
# Sequential bootstrap
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("seed", range(4))
def test_sequential_bootstrap_matches_naive_with_same_uniforms(seed):
    index, t1 = _random_labels(90, 18, max_span=14, seed=30 + seed)
    m = indicator_matrix(index, t1)
    n = 25
    got = sequential_bootstrap(m, n_samples=n, seed=seed)
    uniforms = np.random.default_rng(seed).random(n)
    ref = naive_sequential_draws(m.to_numpy(), uniforms)
    np.testing.assert_array_equal(got, ref)


def test_sequential_bootstrap_is_reproducible_and_in_range():
    index, t1 = _random_labels(50, 10, max_span=8, seed=40)
    m = indicator_matrix(index, t1)
    a = sequential_bootstrap(m, seed=123)
    b = sequential_bootstrap(m, seed=123)
    np.testing.assert_array_equal(a, b)
    assert a.shape == (len(t1),)
    assert a.min() >= 0 and a.max() < len(t1)


def test_sequential_bootstrap_has_higher_average_uniqueness_than_standard():
    index, t1 = _random_labels(200, 60, max_span=30, seed=50)
    m = indicator_matrix(index, t1).to_numpy()
    n = m.shape[1]
    seq_scores, std_scores = [], []
    for seed in range(40):
        seq = sequential_bootstrap(m, n_samples=n, seed=seed)
        seq_scores.append(sample_avg_uniqueness(m, seq))
        rng = np.random.default_rng(1000 + seed)
        std = rng.integers(0, n, size=n)
        std_scores.append(sample_avg_uniqueness(m, std))
    assert np.mean(seq_scores) > np.mean(std_scores)


# ---------------------------------------------------------------------------
# Return attribution and time decay
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("seed", range(3))
def test_sample_weight_by_return_matches_naive_and_sums_to_n(seed):
    index, t1 = _random_labels(120, 25, max_span=18, seed=60 + seed)
    rng = np.random.default_rng(seed)
    close = pd.Series(100.0 * np.exp(np.cumsum(rng.normal(0, 0.01, len(index)))), index=index)
    got = sample_weight_by_return(index, t1, close)
    np.testing.assert_allclose(got.to_numpy(), naive_return_weights(index, t1, close), rtol=1e-10)
    assert got.sum() == pytest.approx(len(t1), rel=1e-12)
    assert np.all(got.to_numpy() >= 0)


def test_time_decay_properties():
    w = pd.Series([0.5, 0.2, 1.0, 0.3, 0.8])
    one = time_decay(w, c_last=1.0)
    np.testing.assert_allclose(one.to_numpy(), 1.0)  # c = 1: no decay

    half = time_decay(w, c_last=0.5)
    assert half.iloc[-1] == pytest.approx(1.0)  # newest gets weight 1
    assert np.all(np.diff(half.to_numpy()) >= 0)  # weights grow with recency
    assert np.all((half.to_numpy() >= 0) & (half.to_numpy() <= 1.0))

    cum = w.cumsum().to_numpy()
    total = cum[-1]
    expected = 0.5 + (1 - 0.5) * cum / total  # a = c, b = (1 - c) / T
    np.testing.assert_allclose(half.to_numpy(), expected, rtol=1e-12)

    neg = time_decay(w, c_last=-0.5)
    assert neg.iloc[-1] == pytest.approx(1.0)
    assert neg.iloc[0] == 0.0  # oldest part is clipped to zero for c < 0


def test_time_decay_respects_original_order():
    w = pd.Series([0.3, 0.8, 0.5], index=[3, 1, 2])
    out = time_decay(w, c_last=0.4)
    assert list(out.index) == [3, 1, 2]
    ordered = time_decay(w.sort_index(), c_last=0.4)
    np.testing.assert_allclose(out.loc[[1, 2, 3]].to_numpy(), ordered.to_numpy())


def test_time_decay_rejects_bad_c():
    with pytest.raises(ValueError):
        time_decay(np.ones(3), c_last=1.5)
    with pytest.raises(ValueError):
        time_decay(np.ones(3), c_last=-1.0)
