"""Tests for finlab.fracdiff (AFML ch. 5).

The fast numba kernels are cross-checked against naive references written with
explicit Python loops and ``np.dot`` (the book's Snippets 5.2 and 5.3 style).
The weights are also checked against the closed-form binomial series
``omega_k = (-1)^k C(d, k)`` from AFML 5.4.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from scipy.special import binom

from finlab.fracdiff import (
    find_min_d,
    frac_diff,
    frac_diff_ffd,
    get_weights,
    get_weights_ffd,
)


# --------------------------------------------------------------------------- #
# Naive references
# --------------------------------------------------------------------------- #


def naive_ffd(x: np.ndarray, w_lag: np.ndarray) -> np.ndarray:
    """Snippet 5.3 in spirit: one ``np.dot`` per observation, weights oldest-first."""
    w_book = w_lag[::-1]  # the book stores weights oldest-first
    width = len(w_lag) - 1
    out = np.full(len(x), np.nan)
    for t in range(width, len(x)):
        out[t] = np.dot(w_book, x[t - width : t + 1])
    return out


def naive_expanding(x: np.ndarray, d: float, thres: float) -> np.ndarray:
    """Snippet 5.2 in spirit, with the weight-loss rule recomputed from scratch per t."""
    n = len(x)
    w_lag = np.empty(n)
    w_lag[0] = 1.0
    for k in range(1, n):
        w_lag[k] = -w_lag[k - 1] * (d - k + 1) / k
    total = float(np.sum(np.abs(w_lag)))
    first = n
    for t in range(n):
        lost = sum(abs(w_lag[j]) for j in range(t + 1, n)) / total
        if lost <= thres:
            first = t
            break
    out = np.full(n, np.nan)
    for t in range(first, n):
        out[t] = np.dot(w_lag[: t + 1][::-1], x[: t + 1])
    return out


def naive_closed_form_weights(d: float, size: int) -> np.ndarray:
    return np.array([(-1.0) ** k * binom(d, k) for k in range(size)])


def log_random_walk(seed: int, n: int = 3000) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return np.log(100.0) + np.cumsum(rng.normal(0.0, 0.01, n))


# --------------------------------------------------------------------------- #
# Weights
# --------------------------------------------------------------------------- #


def test_weights_d0_is_identity() -> None:
    w = get_weights(0.0, 6)
    np.testing.assert_array_equal(w, [1.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    np.testing.assert_array_equal(get_weights_ffd(0.0, 1e-5), [1.0])


def test_weights_d1_is_first_difference() -> None:
    w = get_weights(1.0, 6)
    np.testing.assert_array_equal(w, [1.0, -1.0, 0.0, 0.0, 0.0, 0.0])
    np.testing.assert_array_equal(get_weights_ffd(1.0, 1e-5), [1.0, -1.0])


@pytest.mark.parametrize("d", [0.2, 0.4, 0.7, 1.0, 1.3, 2.0])
def test_weights_match_binomial_series(d: float) -> None:
    np.testing.assert_allclose(
        get_weights(d, 40), naive_closed_form_weights(d, 40), rtol=1e-11, atol=1e-14
    )


@pytest.mark.parametrize("d", [0.2, 0.4, 0.7, 1.3])
def test_ffd_weight_partial_sums_have_closed_form(d: float) -> None:
    """Sum of omega_0..omega_L equals (-1)^L C(d-1, L): the partial sums of (1-B)^d at B=1.

    For d in (0, 1) they lie in (0, 1] and tend to 0, which is why the
    truncated window removes the level and leaves a zero-mean, differenced series.
    """
    w = get_weights_ffd(d, 1e-5)
    L = len(w) - 1
    partial = np.cumsum(w)
    expected = np.array([(-1.0) ** j * binom(d - 1, j) for j in range(L + 1)])
    np.testing.assert_allclose(partial, expected, rtol=1e-9, atol=1e-12)
    if 0 < d < 1:
        assert np.all(partial[1:] > 0) and np.all(partial[1:] <= 1.0)
        assert np.all(np.diff(partial[1:]) < 0)  # partial sums strictly decrease to 0


def test_ffd_window_respects_threshold() -> None:
    d, thres = 0.4, 1e-3
    w = get_weights_ffd(d, thres)
    full = get_weights(d, 5000)
    L = len(w)
    assert np.all(np.abs(w[1:]) >= thres)  # every kept weight clears the cut
    assert abs(full[L]) < thres  # the first dropped weight is below it
    np.testing.assert_allclose(w, full[:L], rtol=0, atol=0)


# --------------------------------------------------------------------------- #
# FFD kernel vs naive dot product
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("d", [0.1, 0.4, 0.8, 1.0, 1.5])
@pytest.mark.parametrize("seed", [0, 1])
def test_ffd_kernel_matches_naive_dot(d: float, seed: int) -> None:
    x = log_random_walk(seed, n=2500)
    thres = 1e-4
    fast = frac_diff_ffd(x, d, thres=thres)
    slow = naive_ffd(x, get_weights_ffd(d, thres))
    np.testing.assert_allclose(fast, slow, rtol=1e-10, atol=1e-12)


def test_ffd_handles_nans_by_forward_fill_and_masks_original_nans() -> None:
    rng = np.random.default_rng(3)
    x = log_random_walk(3, n=800)
    holes = rng.random(800) < 0.05
    holes[0] = False
    xn = x.copy()
    xn[holes] = np.nan
    filled = pd.Series(xn).ffill().to_numpy()
    fast = frac_diff_ffd(xn, 0.5, thres=1e-3)
    slow = naive_ffd(filled, get_weights_ffd(0.5, 1e-3))
    keep = ~holes
    np.testing.assert_allclose(fast[keep], slow[keep], rtol=1e-10, atol=1e-12)
    assert np.all(np.isnan(fast[holes]))


def test_ffd_keeps_pandas_container_and_length() -> None:
    x = log_random_walk(4, n=500)
    idx = pd.date_range("2020-01-01", periods=500, freq="D")
    s = pd.Series(x, index=idx, name="logp")
    out = frac_diff_ffd(s, 0.4, thres=1e-3)
    assert isinstance(out, pd.Series) and out.name == "logp"
    assert out.index.equals(idx)
    df = pd.DataFrame({"a": x, "b": x[::-1].copy()}, index=idx)
    out_df = frac_diff_ffd(df, 0.4, thres=1e-3)
    assert list(out_df.columns) == ["a", "b"]
    np.testing.assert_allclose(out_df["a"].to_numpy(), out.to_numpy(), equal_nan=True)


# --------------------------------------------------------------------------- #
# Expanding window
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("d", [0.3, 0.6, 1.0])
@pytest.mark.parametrize("seed", [5, 6])
def test_expanding_matches_naive(d: float, seed: int) -> None:
    x = log_random_walk(seed, n=400)
    thres = 0.01
    np.testing.assert_allclose(
        frac_diff(x, d, thres=thres), naive_expanding(x, d, thres), rtol=1e-10, atol=1e-12, equal_nan=True
    )


def test_frac_diff_d1_reproduces_first_differences() -> None:
    x = log_random_walk(7, n=300)
    out = frac_diff(x, 1.0)  # default thres=0.01
    assert np.isnan(out[0])
    np.testing.assert_allclose(out[1:], np.diff(x), rtol=0, atol=1e-14)


def test_frac_diff_d0_returns_the_series() -> None:
    x = log_random_walk(8, n=200)
    np.testing.assert_allclose(frac_diff(x, 0.0), x, rtol=0, atol=1e-14)


def test_thres_one_keeps_every_observation() -> None:
    x = log_random_walk(9, n=100)
    assert np.all(np.isfinite(frac_diff(x, 0.5, thres=1.0)))


# --------------------------------------------------------------------------- #
# Memory preservation
# --------------------------------------------------------------------------- #


def test_memory_correlation_decreases_with_d() -> None:
    """Correlation of frac_diff_ffd(x, d) with x falls as d rises (AFML 5.6, Figure 5.5).

    Checked on five seeded random walks. Each seed is monotone; the
    averages are strictly decreasing. This is a test on fixed data, not a
    proof for all series (see docs/proofs/fracdiff.md).
    """
    grid = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
    rows = []
    for seed in range(5):
        # The FFD window at d=0.2, thres=1e-4 is ~2000 weights, so n must exceed it.
        x = log_random_walk(seed, n=6000)
        corrs = []
        for d in grid:
            f = np.asarray(frac_diff_ffd(x, d, thres=1e-4))
            m = np.isfinite(f)
            corrs.append(np.corrcoef(x[m], f[m])[0, 1])
        rows.append(corrs)
    arr = np.array(rows)
    assert np.all(np.isfinite(arr))
    assert np.allclose(arr[:, 0], 1.0)  # d = 0 keeps all memory
    assert np.all(np.diff(arr, axis=1) < 0)  # per seed, strictly decreasing
    means = arr.mean(axis=0)
    assert np.all(np.diff(means) < 0)


# --------------------------------------------------------------------------- #
# find_min_d
# --------------------------------------------------------------------------- #


def lag1_adf_surrogate(x: np.ndarray) -> float:
    """Test-only stand-in for an ADF test: p-value falls as lag-1 autocorrelation falls."""
    rho = np.corrcoef(x[1:], x[:-1])[0, 1]
    return 0.01 if rho < 0.5 else 0.9


def test_find_min_d_returns_first_passing_grid_value() -> None:
    x = log_random_walk(10, n=2000)
    grid = np.linspace(0.0, 1.0, 11)
    res = find_min_d(x, adf_pvalue=0.05, adf_test=lag1_adf_surrogate, d_grid=grid, thres=1e-3)

    # Re-derive the answer independently with a plain loop over the grid.
    expected = None
    for d in grid:
        f = np.asarray(frac_diff_ffd(x, float(d), thres=1e-3))
        f = f[np.isfinite(f)]
        if lag1_adf_surrogate(f) <= 0.05:
            expected = float(d)
            break
    assert res.d == expected
    assert res.d is not None and res.pvalue is not None and res.pvalue <= 0.05
    assert res.corr is not None and -1.0 <= res.corr <= 1.0


def test_find_min_d_reports_none_when_nothing_passes() -> None:
    x = log_random_walk(11, n=500)
    res = find_min_d(x, adf_test=lambda z: 0.99, d_grid=[0.0, 0.5, 1.0])
    assert res.d is None and res.pvalue is None and res.corr is None


def test_find_min_d_needs_injected_adf_test() -> None:
    with pytest.raises(TypeError, match="adf_test"):
        find_min_d(log_random_walk(12, n=50))


def test_validation() -> None:
    with pytest.raises(ValueError):
        get_weights_ffd(0.5, thres=0.0)
    with pytest.raises(ValueError):
        frac_diff_ffd(np.arange(10.0), float("nan"))
    with pytest.raises(ValueError):
        get_weights(0.5, 0)
