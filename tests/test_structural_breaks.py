"""Tests for finlab.structural_breaks (AFML ch. 17).

Each fast routine is cross-checked against a brute-force reference written as
literal loops with ``np.linalg.lstsq`` / explicit sums, on seeded random inputs.
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest

from finlab.structural_breaks import (
    adf_stat,
    chu_stinchcombe_white,
    csw_critical_value,
    cusum_test,
    sadf,
)

# ---------------------------------------------------------------------------
# Naive references
# ---------------------------------------------------------------------------


def naive_recursive_residuals(y, x):
    """Literal BDE recursion: refit OLS on each prefix, then standardise."""
    t_total, p = x.shape
    w = np.full(t_total, np.nan)
    for t in range(p, t_total):
        xp, yp = x[:t], y[:t]
        gram_inv = np.linalg.inv(xp.T @ xp)
        beta = gram_inv @ (xp.T @ yp)
        f = 1.0 + x[t] @ gram_inv @ x[t]
        w[t] = (y[t] - x[t] @ beta) / math.sqrt(f)
    return w


def naive_cusum(y, x):
    p = x.shape[1]
    w = naive_recursive_residuals(y, x)
    resid = w[p:]
    sigma = math.sqrt(np.mean(resid**2))
    stat = np.full(len(y), np.nan)
    for t in range(p, len(y)):
        stat[t] = np.sum(w[p : t + 1]) / sigma
    return w, stat


def naive_adf(y, lag, const):
    """Design built from first principles, OLS via lstsq, OLS standard errors."""
    n_levels = len(y)
    rows = []
    targets = []
    for t in range(lag + 1, n_levels):
        dy = y[t] - y[t - 1]
        row = [y[t - 1]] + [y[t - l] - y[t - l - 1] for l in range(1, lag + 1)]
        if const in ("c", "ct", "ctt"):
            row.append(1.0)
        if const in ("ct", "ctt"):
            row.append(float(t - lag - 1))
        if const == "ctt":
            row.append(float(t - lag - 1) ** 2)
        rows.append(row)
        targets.append(dy)
    x = np.array(rows, dtype=float)
    yy = np.array(targets)
    n, p = x.shape
    beta, *_ = np.linalg.lstsq(x, yy, rcond=None)
    resid = yy - x @ beta
    sigma2 = resid @ resid / (n - p)
    cov = sigma2 * np.linalg.inv(x.T @ x)
    return beta[0] / math.sqrt(cov[0, 0])


def naive_sadf(y, min_length, lags, const):
    out = np.full(len(y), np.nan)
    for e in range(min_length - 1, len(y)):
        taus = [naive_adf(y[s : e + 1], lags, const) for s in range(0, e - min_length + 2)]
        out[e] = max(taus)
    return out


# ---------------------------------------------------------------------------
# CUSUM on recursive residuals (AFML 17.3.1)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_cusum_matches_naive_mean_model(seed):
    rng = np.random.default_rng(seed)
    y = rng.standard_normal(120).cumsum() * 0.1 + rng.standard_normal(120)
    res = cusum_test(y)
    w_ref, s_ref = naive_cusum(y, np.ones((120, 1)))
    np.testing.assert_allclose(res.residuals, w_ref, rtol=1e-9, atol=1e-12, equal_nan=True)
    np.testing.assert_allclose(res.statistic, s_ref, rtol=1e-9, atol=1e-9, equal_nan=True)
    assert res.initial_size == 1


@pytest.mark.parametrize("seed", [3, 4])
def test_cusum_matches_naive_with_features(seed):
    rng = np.random.default_rng(seed)
    x = np.column_stack([np.ones(150), rng.standard_normal(150)])
    y = x @ np.array([0.5, -1.0]) + rng.standard_normal(150)
    res = cusum_test(y, exog=x)
    w_ref, s_ref = naive_cusum(y, x)
    np.testing.assert_allclose(res.residuals, w_ref, rtol=1e-8, atol=1e-10, equal_nan=True)
    np.testing.assert_allclose(res.statistic, s_ref, rtol=1e-8, atol=1e-8, equal_nan=True)
    assert res.initial_size == 2


def test_cusum_grows_after_known_mean_shift():
    """Injected mean shift at t=100: |S_t| after the shift exceeds |S_t| before it."""
    rng = np.random.default_rng(42)
    y = rng.standard_normal(200)
    y[100:] += 3.0  # shift in units of the noise sd
    res = cusum_test(y)
    s = res.statistic
    before = np.nanmax(np.abs(s[1:100]))
    after = np.abs(s[-1])
    assert after > before
    assert after > 3.0 * before / 2.0
    # The same series without the shift stays small at the end.
    y0 = np.random.default_rng(42).standard_normal(200)
    assert abs(cusum_test(y0).statistic[-1]) < after / 2


def test_cusum_singular_initial_block_raises():
    x = np.zeros((10, 2))
    x[:, 0] = 1.0  # first two rows are collinear: X_2'X_2 is singular
    with pytest.raises(ValueError):
        cusum_test(np.arange(10.0), exog=x)


def test_cusum_pandas_and_result_shape():
    s = pd.Series(np.random.default_rng(5).standard_normal(50))
    res = cusum_test(s)
    assert res.statistic.shape == (50,)
    assert res.z.shape == (50,)
    assert np.isnan(res.statistic[0])
    # Pandas input must give the same values as the array path.
    ref = cusum_test(s.to_numpy())
    np.testing.assert_allclose(res.statistic, ref.statistic, rtol=1e-12, equal_nan=True)
    np.testing.assert_allclose(res.residuals, ref.residuals, rtol=1e-12, equal_nan=True)


# ---------------------------------------------------------------------------
# Chu-Stinchcombe-White (AFML 17.3.2)
# ---------------------------------------------------------------------------


def naive_csw_fixed(y, n0):
    out = np.full(len(y), np.nan)
    for t in range(n0 + 1, len(y)):
        sig2 = sum((y[i] - y[i - 1]) ** 2 for i in range(1, t + 1)) / t
        out[t] = (y[t] - y[n0]) / (math.sqrt(sig2) * math.sqrt(t - n0))
    return out


def naive_csw_sup(y):
    out = np.full(len(y), np.nan)
    arg = np.full(len(y), -1)
    for t in range(1, len(y)):
        sig2 = sum((y[i] - y[i - 1]) ** 2 for i in range(1, t + 1)) / t
        vals = [(y[t] - y[n]) / (math.sqrt(sig2) * math.sqrt(t - n)) for n in range(t)]
        out[t] = max(vals)
        arg[t] = int(np.argmax(vals))
    return out, arg


@pytest.mark.parametrize("seed", [0, 1])
def test_csw_fixed_reference_matches_naive(seed):
    rng = np.random.default_rng(seed)
    y = rng.standard_normal(80).cumsum()
    res = chu_stinchcombe_white(y, reference=7)
    np.testing.assert_allclose(res.statistic, naive_csw_fixed(y, 7), rtol=1e-10, equal_nan=True)
    assert res.reference is None


@pytest.mark.parametrize("seed", [2, 3])
def test_csw_sup_matches_naive(seed):
    rng = np.random.default_rng(seed)
    y = rng.standard_normal(60).cumsum()
    res = chu_stinchcombe_white(y)
    s_ref, arg_ref = naive_csw_sup(y)
    np.testing.assert_allclose(res.statistic, s_ref, rtol=1e-10, equal_nan=True)
    np.testing.assert_array_equal(res.reference[1:], arg_ref[1:])


def test_csw_critical_value_book_constant():
    # AFML 17.3.2: b_0.05 = 4.6, one-sided c = sqrt(b + log(t - n)).
    assert csw_critical_value(1, 4.6) == pytest.approx(math.sqrt(4.6))
    assert csw_critical_value(math.e, 4.6) == pytest.approx(math.sqrt(5.6))
    with pytest.raises(ValueError):
        csw_critical_value(0)


@pytest.mark.parametrize("reference", [None, 0, 1])
def test_csw_constant_leading_values_do_not_raise(reference):
    # sigma_t = 0 while the series is still flat, so the statistic is undefined there.
    y = np.array([0.0, 0.0, 0.0, 1.0, 2.0, 4.0])
    res = chu_stinchcombe_white(y, reference=reference)
    assert len(res.statistic) == len(y)
    assert len(res.critical) == len(y)
    assert len(res.reject) == len(y)


@pytest.mark.parametrize("reference", [None, 0])
def test_csw_zero_volatility_entries_are_nan(reference):
    y = np.array([0.0, 0.0, 0.0, 1.0, 2.0, 4.0])
    res = chu_stinchcombe_white(y, reference=reference)
    # sigma_1 = sigma_2 = 0: the statistic is NaN there and never rejects.
    assert np.isnan(res.statistic[1]) and np.isnan(res.statistic[2])
    assert not res.reject[1] and not res.reject[2]
    # From index 3 on sigma is positive, so the statistic is finite.
    assert np.all(np.isfinite(res.statistic[3:]))


def test_csw_zero_volatility_pandas_shape_and_index():
    idx = pd.date_range("2024-01-01", periods=5, freq="D")
    y = pd.Series([0.0, 0.0, 1.0, 2.0, 3.0], index=idx)
    res = chu_stinchcombe_white(y)
    assert isinstance(res.statistic, pd.Series)
    assert list(res.statistic.index) == list(idx)
    assert np.isnan(res.statistic.iloc[1])
    assert np.isfinite(res.statistic.iloc[2:]).all()


@pytest.mark.parametrize("seed", [4, 5])
def test_csw_normal_series_has_no_nan_after_start(seed):
    rng = np.random.default_rng(seed)
    y = rng.standard_normal(50).cumsum()
    res = chu_stinchcombe_white(y)
    assert np.all(np.isfinite(res.statistic[1:]))
    fixed = chu_stinchcombe_white(y, reference=3)
    assert np.all(np.isfinite(fixed.statistic[4:]))


def test_csw_detects_late_drift():
    rng = np.random.default_rng(7)
    y = rng.standard_normal(300).cumsum() * 0.01
    drift = np.zeros(300)
    drift[150:] = np.linspace(0.0, 1.0, 150)  # upward drift of 1.0 over the second half
    res = chu_stinchcombe_white(y + drift, reference=140)
    # S_{140,299} ~ 1.0 / (0.01 * sqrt(159)) ~ 8, far above c ~ 3.1.
    assert res.statistic[-1] > 5.0
    assert res.reject[-1]
    # Without drift the same reference point is not rejected at the end.
    assert not chu_stinchcombe_white(y, reference=140).reject[-1]


# ---------------------------------------------------------------------------
# ADF and SADF (AFML 17.4.2)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("const", ["nc", "c", "ct", "ctt"])
@pytest.mark.parametrize("lag", [0, 1, 3])
def test_adf_matches_naive(const, lag):
    rng = np.random.default_rng(10 + lag)
    y = rng.standard_normal(120).cumsum()
    assert adf_stat(y, lag, const) == pytest.approx(naive_adf(y, lag, const), rel=1e-9)


def test_adf_rejects_bad_constant():
    with pytest.raises(ValueError):
        adf_stat(np.arange(30.0), 1, "trend")


@pytest.mark.parametrize("const", ["c", "ct"])
def test_sadf_matches_naive(const):
    rng = np.random.default_rng(21)
    y = rng.standard_normal(45).cumsum()
    got = sadf(y, min_length=15, lags=1, constant=const)
    ref = naive_sadf(y, 15, 1, const)
    np.testing.assert_allclose(got, ref, rtol=1e-8, equal_nan=True)


def test_sadf_random_walk_vs_explosive():
    """Explosive exp-growth series has a much larger SADF than a random walk."""
    rng = np.random.default_rng(99)
    n = 200
    walk = rng.standard_normal(n).cumsum()
    explosive = np.exp(0.02 * np.arange(n)) * (1.0 + 0.01 * rng.standard_normal(n))
    s_walk = sadf(walk, min_length=30, lags=1)
    s_exp = sadf(explosive, min_length=30, lags=1)
    assert isinstance(s_exp, np.ndarray)  # numpy in, numpy out
    assert s_exp[-1] > 2.0
    assert np.nanmax(s_walk) < 2.0
    assert s_exp[-1] > np.nanmax(s_walk) + 1.0


def test_sadf_pandas_index_and_nan_prefix():
    idx = pd.date_range("2020-01-01", periods=60, freq="D")
    s = pd.Series(np.random.default_rng(3).standard_normal(60).cumsum(), index=idx)
    out = sadf(s, min_length=20, lags=1)
    assert isinstance(out, pd.Series)
    assert out.index.equals(idx)
    assert out.iloc[:19].isna().all()
    assert out.iloc[19:].notna().all()
    np.testing.assert_allclose(out.to_numpy(), sadf(s.to_numpy(), min_length=20, lags=1),
                               rtol=1e-12, equal_nan=True)


def test_sadf_rejects_short_min_length():
    with pytest.raises(ValueError):
        sadf(np.arange(50.0).cumsum(), min_length=3, lags=2)


def test_cusum_near_collinear_regressors_raise_not_finite():
    rng = np.random.default_rng(1)
    y = np.cumsum(rng.normal(size=300))
    x = rng.normal(size=300)
    with pytest.raises(ValueError, match="not finite"):
        cusum_test(y, np.column_stack([x, x * (1 + 1e-6)]))


def _magnitude_message(exc_info):
    text = str(exc_info.value)
    assert "magnitude" in text
    assert "collinear" not in text and "reproduce" not in text


def test_cusum_large_magnitude_series_names_magnitude_not_collinearity():
    with pytest.raises(ValueError) as exc_info:
        cusum_test(np.linspace(1, 2, 50) * 1e160)
    _magnitude_message(exc_info)


def test_cusum_underflowing_series_names_magnitude_not_exact_fit():
    rng = np.random.default_rng(7)
    with pytest.raises(ValueError) as exc_info:
        cusum_test(rng.normal(size=60) * 1e-200)
    _magnitude_message(exc_info)


def test_cusum_constant_series_keeps_no_variation_message():
    with pytest.raises(ValueError, match="no variation in series"):
        cusum_test(np.ones(50))


def test_cusum_exact_fit_regressor_raises_reproduce_message():
    rng = np.random.default_rng(2)
    y = np.cumsum(rng.normal(size=300))[:3]
    with pytest.raises(ValueError, match="reproduce"):
        cusum_test(y, y.reshape(-1, 1))


def test_adf_rejects_non_integer_lag():
    y = np.random.default_rng(3).standard_normal(120).cumsum()
    with pytest.raises(ValueError):
        adf_stat(y, 1.9)
    assert np.isfinite(adf_stat(y, 1))
    assert np.isfinite(adf_stat(y, np.int64(1)))


@pytest.mark.parametrize("bad", [float("nan"), float("inf"), float("-inf"), "x", "2"])
def test_adf_non_finite_or_non_numeric_lag_is_value_error(bad):
    y = np.random.default_rng(4).standard_normal(120).cumsum()
    with pytest.raises(ValueError, match="lag must be an integer"):
        adf_stat(y, bad)


@pytest.mark.parametrize("bad", [float("nan"), float("inf"), "x"])
def test_sadf_non_finite_or_non_numeric_lags_is_value_error(bad):
    y = np.random.default_rng(4).standard_normal(120).cumsum()
    with pytest.raises(ValueError, match="lags must be an integer"):
        sadf(y, min_length=30, lags=bad)


def test_adf_integer_valued_float_and_numpy_integer_lags_work():
    y = np.random.default_rng(6).standard_normal(120).cumsum()
    assert adf_stat(y, 2.0) == pytest.approx(adf_stat(y, 2))
    assert adf_stat(y, np.int64(2)) == pytest.approx(adf_stat(y, 2))


def test_sadf_rejects_non_integer_lags():
    y = np.random.default_rng(4).standard_normal(60).cumsum()
    with pytest.raises(ValueError):
        sadf(y, min_length=20, lags=1.5)
