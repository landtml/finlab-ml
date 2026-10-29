"""Tests for quantowl.structural_breaks (AFML ch. 17).

Each fast routine is cross-checked against a brute-force reference written as
literal loops with ``np.linalg.lstsq`` / explicit sums, on seeded random inputs.
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest

from quantowl.structural_breaks import (
    adf_stat,
    cadf,
    chow_dfc,
    chu_stinchcombe_white,
    csw_critical_value,
    cusum_test,
    qadf,
    sadf,
    sadf_null_quantiles,
    sdfc,
    smt,
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


def test_cusum_large_magnitude_series_is_scale_invariant():
    # The CUSUM statistic is a ratio to sigma, so it does not depend on scale and the large side is
    # rescaled first: a 1e160 series gives the same statistic as the unit-scale series.
    base = cusum_test(np.linspace(1, 2, 50)).statistic
    np.testing.assert_allclose(cusum_test(np.linspace(1, 2, 50) * 1e160).statistic, base, rtol=1e-9)


def test_cusum_underflowing_series_matches_unit_scale():
    # A 1e-200 series is rescaled before the statistic is computed, so the statistic is a ratio to
    # sigma and matches scale 1, as adf_stat does.
    rng = np.random.default_rng(7)
    y = rng.normal(size=60)
    np.testing.assert_allclose(cusum_test(y * 1e-200).statistic, cusum_test(y).statistic, rtol=1e-9)


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


# --------------------------------------------------------------------------- #
# CSW at extreme magnitudes
# --------------------------------------------------------------------------- #


def _csw_walk() -> np.ndarray:
    return np.cumsum(np.random.default_rng(1).normal(size=300))


@pytest.mark.parametrize("scale", [1e-170, 1e-200])
def test_csw_small_scale_matches_unit_scale(scale):
    # The chu statistic is scale-invariant, so a 1e-170 series is rescaled and gives the unit-scale
    # value, not a refusal.
    base = chu_stinchcombe_white(_csw_walk()).statistic
    np.testing.assert_allclose(chu_stinchcombe_white(_csw_walk() * scale).statistic, base, rtol=1e-9)


def test_csw_ordinary_scale_output_is_finite_and_unchanged():
    res = chu_stinchcombe_white(_csw_walk())
    assert np.all(np.isfinite(res.statistic[1:]))
    assert np.nanmax(np.abs(res.statistic)) == pytest.approx(3.2997035863686834, rel=1e-9)
    assert int(res.reject.sum()) == 8


def test_csw_constant_series_still_all_nan_without_error():
    res = chu_stinchcombe_white(np.full(50, 2.0))
    assert np.all(np.isnan(res.statistic))
    assert int(res.reject.sum()) == 0


@pytest.mark.parametrize("min_length", [30.5, float("nan")])
def test_sadf_rejects_non_integer_min_length(min_length):
    y = np.cumsum(np.random.default_rng(2).normal(size=100))
    with pytest.raises(ValueError, match="min_length"):
        sadf(y, min_length=min_length)


def test_sadf_accepts_integer_valued_min_length():
    y = np.cumsum(np.random.default_rng(2).normal(size=100))
    np.testing.assert_allclose(
        sadf(y, min_length=30.0)[-3:], sadf(y, min_length=30)[-3:], rtol=0, atol=0
    )


def test_csw_non_integer_reference_names_reference():
    y = np.cumsum(np.random.default_rng(1).normal(size=50))
    with pytest.raises(ValueError, match="reference must be an integer"):
        chu_stinchcombe_white(y, reference=1.5)


def test_sadf_nan_min_length_names_min_length():
    y = np.cumsum(np.random.default_rng(2).normal(size=100))
    with pytest.raises(ValueError, match="min_length"):
        sadf(y, min_length=float("nan"))


def test_csw_critical_value_rejects_b_that_makes_the_root_negative() -> None:
    # b = -10 made every critical value NaN, so the test never rejected.
    from quantowl.structural_breaks import csw_critical_value

    with pytest.raises(ValueError, match="b"):
        csw_critical_value(np.array([1, 5, 20]), b=-10.0)
    with pytest.raises(ValueError, match="b"):
        chu_stinchcombe_white(np.cumsum(np.ones(60)), b=-10.0)


def test_csw_critical_value_valid_b_unchanged() -> None:
    from quantowl.structural_breaks import csw_critical_value

    np.testing.assert_allclose(
        csw_critical_value(np.array([1, 2]), b=4.6), np.sqrt(4.6 + np.log([1.0, 2.0]))
    )


def test_adf_stat_constant_series_reports_no_variation() -> None:
    # A constant series must be reported as having no variation, not as too few observations, and
    # constant='nc' must not silently return 0.0.
    from quantowl.structural_breaks import adf_stat

    for constant in ("c", "nc"):
        with pytest.raises(ValueError, match="no variation"):
            adf_stat(np.ones(50), lag=0, constant=constant)


def _random_walk_levels(seed: int = 0, n: int = 200) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return np.cumsum(rng.normal(size=n)) + 10.0


def test_adf_stat_is_scale_invariant_at_extreme_scale() -> None:
    # The Gram matrix overflowed at 1e160, and the error blamed the series length.
    from quantowl.structural_breaks import adf_stat

    x = _random_walk_levels()
    base = adf_stat(x, lag=1)
    assert adf_stat(x * 1e160, lag=1) == pytest.approx(base, rel=1e-8)
    assert adf_stat(x * 1e-160, lag=1) == pytest.approx(base, rel=1e-8)
    assert adf_stat(x * 1e200, lag=1) == pytest.approx(base, rel=1e-8)


def test_sadf_is_scale_invariant_at_extreme_scale() -> None:
    # Every output was NaN at 1e200 and no error was raised.
    from quantowl.structural_breaks import sadf

    x = _random_walk_levels()
    base = np.asarray(sadf(x, min_length=20))
    for factor in (1e200, 1e-200):
        got = np.asarray(sadf(x * factor, min_length=20))
        np.testing.assert_array_equal(np.isnan(got), np.isnan(base))
        np.testing.assert_allclose(got, base, rtol=1e-8, equal_nan=True)



def test_sadf_rejects_constant_series_like_adf_stat() -> None:
    # sadf returned NaN silently where adf_stat raises.
    from quantowl.structural_breaks import sadf

    with pytest.raises(ValueError, match="constant"):
        sadf(np.ones(40), min_length=10)


def test_csw_critical_value_nan_lag_is_named_as_lag() -> None:
    # A NaN lag passed the lag >= 1 check (NaN compares false), then failed the b check, so the error
    # blamed b. A non-finite lag is now reported as a lag.
    from quantowl.structural_breaks import csw_critical_value

    with pytest.raises(ValueError, match="t - n must be finite"):
        csw_critical_value(np.nan)
    with pytest.raises(ValueError, match="b must satisfy"):
        csw_critical_value(np.array([1, 5]), b=np.nan)


def test_structural_breaks_reject_a_descending_index() -> None:
    # Values were read in array order, so a descending DatetimeIndex gave the statistics of the
    # reversed series. bars.py sorts by index; here a non-ascending index is refused.
    import pandas as pd

    from quantowl.fracdiff import frac_diff, frac_diff_ffd
    from quantowl.structural_breaks import adf_stat, chu_stinchcombe_white, sadf

    rw = np.cumsum(np.random.default_rng(5).standard_normal(60)) + 10
    s = pd.Series(rw, index=pd.date_range("2024-01-01", periods=60, freq="D"))
    r = s.iloc[::-1]
    with pytest.raises(ValueError, match="sorted"):
        chu_stinchcombe_white(r)
    with pytest.raises(ValueError, match="sorted"):
        adf_stat(r, 1)
    with pytest.raises(ValueError, match="sorted"):
        sadf(r, 8, 1)
    with pytest.raises(ValueError, match="sorted"):
        frac_diff_ffd(r, 0.4, 1e-3)
    with pytest.raises(ValueError, match="sorted"):
        frac_diff(r, 0.4)
    # An ascending Series, and a plain array, are unchanged.
    assert chu_stinchcombe_white(s).statistic.shape[0] == 60


def test_cusum_test_and_chu_are_scale_invariant_at_large_magnitude() -> None:
    # adf_stat and sadf rescale their input, but cusum_test and chu_stinchcombe_white refused a series
    # near 1e155 although both statistics are ratios that do not change with scale.
    from quantowl import structural_breaks as sb

    y = np.cumsum(np.random.default_rng(0).standard_normal(200))
    big = y * 1e155
    base_c = sb.cusum_test(y).statistic
    np.testing.assert_allclose(sb.cusum_test(big).statistic, base_c, rtol=1e-9, atol=1e-12)
    np.testing.assert_allclose(sb.chu_stinchcombe_white(big).statistic,
                               sb.chu_stinchcombe_white(y).statistic, rtol=1e-9, atol=1e-12)
    assert sb.cusum_test(big).sigma == pytest.approx(sb.cusum_test(y).sigma * 1e155, rel=1e-9)


def test_integer_arguments_refuse_bool() -> None:
    # bool is numbers.Integral, so True was read as 1 (lag and reference).
    from quantowl import structural_breaks as sb

    y = np.cumsum(np.random.default_rng(0).standard_normal(60))
    with pytest.raises(ValueError):
        sb.adf_stat(y, True)
    with pytest.raises(ValueError):
        sb.chu_stinchcombe_white(y, reference=True)


def test_cusum_test_refuses_residuals_that_overflow_in_original_units() -> None:
    # The statistic and sigma stayed finite, but residuals = w_s * y_scale overflowed to inf for a
    # series near 1.7e308. The residuals must be refused rather than returned as inf.
    import quantowl.structural_breaks as sb

    y = np.random.default_rng(0).choice([-1.0, 1.0], 60) * 1.7e308
    with pytest.raises(ValueError, match="not finite or underflow"):
        sb.cusum_test(y)


def test_cusum_test_large_series_keeps_finite_residuals() -> None:
    # A series of 1e155 still gives the unscaled statistics, with finite residuals in its units.
    import quantowl.structural_breaks as sb

    y = np.cumsum(np.random.default_rng(1).standard_normal(60))
    base = sb.cusum_test(y)
    big = sb.cusum_test(y * 1e155)
    assert np.all(np.isfinite(big.residuals[1:]))
    assert np.allclose(big.statistic[1:], base.statistic[1:], rtol=1e-9)


def test_csw_critical_value_refuses_infinite_b() -> None:
    # b = +inf gave infinite critical values, so chu_stinchcombe_white never rejected. The lag check
    # already refuses a non-finite value, and b now gets the same rule.
    import quantowl.structural_breaks as sb

    with pytest.raises(ValueError, match="b must be finite"):
        sb.csw_critical_value(np.array([5.0]), b=np.inf)
    with pytest.raises(ValueError, match="b must"):
        sb.csw_critical_value(np.array([5.0]), b=-np.inf)
    # The default b still gives finite critical values, and NaN b keeps its own message.
    assert np.isfinite(sb.csw_critical_value(np.array([5.0]))).all()
    with pytest.raises(ValueError, match="b must satisfy"):
        sb.csw_critical_value(np.array([5.0]), b=np.nan)


def test_cusum_and_chu_small_scale_match_unit_scale() -> None:
    # The rescale covered only the large side, so a series of magnitude 1e-160 gave chu 1.38622
    # against 1.38505 at scale 1 (8e-4 off) and cusum off by 1e-6. The statistics are scale-invariant,
    # so they must match scale 1 to rounding, as adf_stat already does.
    from quantowl import structural_breaks as sb

    y = np.cumsum(np.random.default_rng(0).standard_normal(120)) + 10
    base_chu = sb.chu_stinchcombe_white(y).statistic[2]
    base_cusum = sb.cusum_test(y).statistic[-1]
    assert sb.chu_stinchcombe_white(y * 1e-160).statistic[2] == pytest.approx(base_chu, rel=1e-10)
    assert sb.cusum_test(y * 1e-160).statistic[-1] == pytest.approx(base_cusum, rel=1e-10)
    assert sb.chu_stinchcombe_white(y * 1e-200).statistic[2] == pytest.approx(base_chu, rel=1e-10)
    assert sb.cusum_test(y * 1e-200).statistic[-1] == pytest.approx(base_cusum, rel=1e-10)


def test_large_side_scale_still_matches_unit_scale() -> None:
    # The large side is unchanged, and the small-side rescale does not touch a normal scale.
    from quantowl import structural_breaks as sb

    y = np.cumsum(np.random.default_rng(0).standard_normal(120)) + 10
    base = sb.cusum_test(y).statistic[-1]
    assert sb.cusum_test(y * 1e150).statistic[-1] == pytest.approx(base, rel=1e-10)
    assert sb.cusum_test(y).statistic[-1] == pytest.approx(base, rel=0, abs=0)


def test_cusum_test_refuses_descending_exog_index_and_mismatched_labels():
    # exog's index was never read, so a reversed exog frame gave the statistics of the numpy input,
    # and labels that differed from series were paired by position.
    idx = pd.date_range("2020-01-01", periods=40, freq="D")
    rng = np.random.default_rng(1)
    y = pd.Series(rng.normal(size=40), index=idx)
    x = pd.DataFrame({"f": rng.normal(size=40)}, index=idx)
    with pytest.raises(ValueError, match="sorted"):
        cusum_test(y, x.iloc[::-1])
    with pytest.raises(ValueError, match="same index"):
        cusum_test(y, x.set_axis(idx + pd.Timedelta(hours=12)))
    # Matching labelled exog keeps its value.
    assert np.allclose(
        cusum_test(y, x).statistic, cusum_test(y.to_numpy(), x.to_numpy()).statistic, equal_nan=True
    )


def test_cusum_test_pairs_the_same_instants_across_timezones():
    idx = pd.date_range("2020-01-01", periods=40, freq="h", tz="UTC")
    rng = np.random.default_rng(2)
    y = pd.Series(rng.normal(size=40), index=idx)
    x = pd.DataFrame({"f": rng.normal(size=40)}, index=idx)
    x_berlin = x.set_axis(idx.tz_convert("Europe/Berlin"))
    assert np.allclose(
        cusum_test(y, x_berlin).statistic, cusum_test(y, x).statistic, atol=1e-12, equal_nan=True
    )
    with pytest.raises(ValueError, match="same index"):
        cusum_test(y, x.set_axis(idx.tz_convert("Europe/Berlin") + pd.Timedelta(minutes=30)))


@pytest.mark.parametrize("constant", ["c", "ct"])
def test_adf_stat_is_invariant_to_a_large_level_shift(constant):
    rng = np.random.default_rng(3)
    base = np.cumsum(rng.standard_normal(2000) * 0.01)
    assert adf_stat(base + 1e6, lag=1, constant=constant) == pytest.approx(
        adf_stat(base, lag=1, constant=constant), abs=1e-8
    )


def test_sadf_is_invariant_to_a_large_level_shift():
    rng = np.random.default_rng(4)
    base = np.cumsum(rng.standard_normal(300) * 0.01)
    shifted = sadf(base + 1e6, min_length=30, lags=1)
    assert not np.isinf(shifted).any()
    assert np.allclose(shifted, sadf(base, min_length=30, lags=1), atol=1e-8, equal_nan=True)


# ---------------------------------------------------------------------------
# Chow-type Dickey-Fuller DFC and SDFC (AFML ch. 17)
# ---------------------------------------------------------------------------


def naive_dfc(y, brk):
    """Literal OLS of dy_t on y_{t-1} * 1{t >= brk}, no intercept, rows t = 1..T-1."""
    t_idx = np.arange(1, len(y))
    dy = y[t_idx] - y[t_idx - 1]
    x = (y[t_idx - 1] * (t_idx >= brk)).reshape(-1, 1)
    beta, *_ = np.linalg.lstsq(x, dy, rcond=None)
    resid = dy - x @ beta
    n, p = x.shape
    cov = (resid @ resid / (n - p)) * np.linalg.inv(x.T @ x)
    return beta[0] / math.sqrt(cov[0, 0])


@pytest.mark.parametrize(
    "n_obs, tau, brk",
    [(100, 0.07, 7), (120, 0.15, 18), (120, 0.4, 48), (100, 0.29, 29), (50, 0.5, 25)],
)
def test_chow_dfc_break_is_ceil_of_tau_times_t(n_obs, tau, brk):
    # 0.07 * 100 is slightly above 7 in floats, so the rounding decides between 7 and 8.
    y = np.random.default_rng(n_obs).standard_normal(n_obs).cumsum()
    assert chow_dfc(y, tau) == pytest.approx(naive_dfc(y, brk), rel=1e-9)
    assert not math.isclose(naive_dfc(y, brk + 1), naive_dfc(y, brk), rel_tol=1e-6)


def test_chow_dfc_matches_direct_ols_at_mid_break():
    y = np.random.default_rng(42).standard_normal(120).cumsum()
    assert chow_dfc(y, 0.5) == pytest.approx(naive_dfc(y, 60), rel=1e-9)


def test_chow_dfc_is_scale_invariant():
    y = np.random.default_rng(5).standard_normal(90).cumsum()
    base = chow_dfc(y, 0.4)
    assert chow_dfc(y * 1e200, 0.4) == pytest.approx(base, rel=1e-9)
    assert chow_dfc(y * 1e-200, 0.4) == pytest.approx(base, rel=1e-9)


def test_chow_dfc_pandas_and_numpy_agree():
    idx = pd.date_range("2021-01-01", periods=80, freq="D")
    s = pd.Series(np.random.default_rng(6).standard_normal(80).cumsum(), index=idx)
    assert chow_dfc(s, 0.5) == chow_dfc(s.to_numpy(), 0.5)


def test_sdfc_equals_max_of_direct_loop():
    y = np.random.default_rng(11).standard_normal(120).cumsum()
    res = sdfc(y, tau0=0.15)
    breaks = np.arange(18, 103)  # ceil(0.15 * 120) = 18 and ceil(0.85 * 120) = 102
    ref = np.array([naive_dfc(y, b) for b in breaks])
    np.testing.assert_array_equal(res.breaks, breaks)
    np.testing.assert_allclose(res.dfc, ref, rtol=1e-9)
    assert res.statistic == pytest.approx(ref.max(), rel=1e-9)
    assert res.break_index == breaks[int(np.argmax(ref))]
    assert isinstance(res.dfc, np.ndarray)


def test_sdfc_bounds_a_fine_tau_grid_and_is_attained_at_its_argmax():
    y = np.random.default_rng(11).standard_normal(120).cumsum()
    res = sdfc(y, tau0=0.15)
    grid = [chow_dfc(y, tau) for tau in np.linspace(0.15, 0.85, 701)]
    assert max(grid) <= res.statistic + 1e-12
    assert chow_dfc(y, res.break_index / len(y)) == pytest.approx(res.statistic, rel=1e-9)


def test_sdfc_is_larger_for_a_regime_switch_to_explosive_growth():
    rng = np.random.default_rng(7)
    n = 300
    shocks = rng.standard_normal(n)
    switch = np.empty(n)
    switch[0] = 0.0
    for t in range(1, n):
        rho = 1.0 if t < 150 else 1.01
        switch[t] = rho * switch[t - 1] + shocks[t]
    walk = np.cumsum(shocks)
    assert sdfc(switch).statistic > sdfc(walk).statistic + 2.0


def test_sdfc_undefined_breaks_are_nan_and_skipped():
    # y_{t-1} = 0 for every t >= 21, so DFC is NaN for each break b >= 21 and finite below it.
    y = np.r_[np.arange(1.0, 21.0), np.zeros(20)]
    res = sdfc(y, tau0=0.15)
    assert np.isnan(res.dfc[res.breaks >= 21]).all()
    assert np.isfinite(res.dfc[res.breaks <= 20]).all()
    assert res.break_index <= 20


def test_sdfc_refuses_when_every_admissible_break_is_undefined():
    y = np.r_[1.0, np.zeros(29)]
    with pytest.raises(ValueError, match="undefined"):
        sdfc(y, tau0=0.15)


def test_chow_dfc_undefined_when_no_post_break_level_is_nonzero():
    y = np.r_[np.arange(1.0, 21.0), np.zeros(20)]
    with pytest.raises(ValueError, match="undefined"):
        chow_dfc(y, 0.6)


@pytest.mark.parametrize("bad", [0.0, 0.5, 0.6, -0.1, float("nan"), True, "0.2"])
def test_sdfc_rejects_tau0_outside_open_interval(bad):
    with pytest.raises(ValueError, match="tau0"):
        sdfc(np.random.default_rng(1).standard_normal(60).cumsum(), tau0=bad)


@pytest.mark.parametrize("bad", [0.0, 1.0, 1.5, float("nan"), True])
def test_chow_dfc_rejects_tau_outside_open_interval(bad):
    with pytest.raises(ValueError, match="tau"):
        chow_dfc(np.random.default_rng(1).standard_normal(60).cumsum(), bad)


def test_chow_dfc_refuses_break_after_the_last_observation():
    y = np.random.default_rng(2).standard_normal(50).cumsum()
    with pytest.raises(ValueError, match="after the last observation"):
        chow_dfc(y, 0.999)  # ceil(49.95) = 50 > T - 1


def test_chow_dfc_and_sdfc_refuse_short_and_constant_series():
    with pytest.raises(ValueError, match="three observations"):
        chow_dfc([1.0, 2.0], 0.5)
    with pytest.raises(ValueError, match="constant"):
        chow_dfc(np.ones(30), 0.5)
    with pytest.raises(ValueError, match="constant"):
        sdfc(np.full(30, 2.0))


def test_sdfc_refuses_descending_index():
    idx = pd.date_range("2021-01-01", periods=40, freq="D")[::-1]
    s = pd.Series(np.random.default_rng(3).standard_normal(40).cumsum(), index=idx)
    with pytest.raises(ValueError, match="ascending"):
        sdfc(s)


# ---------------------------------------------------------------------------
# Quantile and conditional ADF (AFML 17.4.2)
# ---------------------------------------------------------------------------


def naive_adf_set(y, min_length, lag, const, e):
    """The ADF statistics whose maximum is SADF at end point e (windows of >= min_length levels)."""
    return [naive_adf(y[s : e + 1], lag, const) for s in range(0, e - min_length + 2)]


def naive_qadf(y, min_length, lag, const, q):
    out = np.full(len(y), np.nan)
    for e in range(min_length - 1, len(y)):
        out[e] = np.quantile(naive_adf_set(y, min_length, lag, const, e), q)
    return out


def naive_cadf(y, min_length, lag, const, q):
    out = np.full(len(y), np.nan)
    for e in range(min_length - 1, len(y)):
        taus = naive_adf_set(y, min_length, lag, const, e)
        threshold = np.quantile(taus, q)
        out[e] = np.mean([tau for tau in taus if tau >= threshold])
    return out


@pytest.mark.parametrize("const", ["c", "ct"])
@pytest.mark.parametrize("q", [0.5, 0.9])
def test_qadf_matches_naive_quantile(const, q):
    y = np.random.default_rng(21).standard_normal(45).cumsum()
    np.testing.assert_allclose(
        qadf(y, 15, 1, q=q, constant=const), naive_qadf(y, 15, 1, const, q), rtol=1e-8, equal_nan=True
    )


@pytest.mark.parametrize("const", ["c", "ct"])
@pytest.mark.parametrize("q", [0.5, 0.9])
def test_cadf_matches_naive_tail_mean(const, q):
    y = np.random.default_rng(22).standard_normal(45).cumsum()
    np.testing.assert_allclose(
        cadf(y, 15, 1, q=q, constant=const), naive_cadf(y, 15, 1, const, q), rtol=1e-8, equal_nan=True
    )


def test_qadf_and_cadf_at_q_one_equal_sadf_exactly():
    y = np.random.default_rng(21).standard_normal(45).cumsum()
    base = sadf(y, min_length=15, lags=1)
    np.testing.assert_array_equal(qadf(y, 15, 1, q=1.0), base)
    np.testing.assert_array_equal(cadf(y, 15, 1, q=1.0), base)


def test_quantile_rule_is_numpy_linear():
    from quantowl.structural_breaks import _quantile_sorted

    rng = np.random.default_rng(8)
    for n in [1, 2, 3, 7, 40]:
        vals = np.sort(rng.standard_normal(n))
        for q in [0.0, 0.05, 0.25, 0.5, 0.73, 0.95, 1.0]:
            assert _quantile_sorted(vals, q) == pytest.approx(np.quantile(vals, q), rel=1e-12, abs=1e-15)


def test_qadf_dispersion_matches_naive_band():
    y = np.random.default_rng(23).standard_normal(45).cumsum()
    level, dispersion = qadf(y, 15, 1, q=0.9, v=0.05)
    np.testing.assert_allclose(
        level, naive_qadf(y, 15, 1, "c", 0.9), rtol=1e-8, equal_nan=True
    )
    expected = naive_qadf(y, 15, 1, "c", 0.95) - naive_qadf(y, 15, 1, "c", 0.85)
    np.testing.assert_allclose(dispersion, expected, rtol=1e-8, equal_nan=True)
    assert np.all(dispersion[14:] >= -1e-12)


def test_qadf_is_nondecreasing_in_q_and_cadf_is_bracketed():
    y = np.random.default_rng(24).standard_normal(60).cumsum()
    grid = [0.1, 0.5, 0.9, 0.95, 1.0]
    values = np.vstack([qadf(y, 20, 1, q=q) for q in grid])
    finite = np.all(np.isfinite(values[:, 19:]), axis=0)
    assert np.all(np.diff(values[:, 19:][:, finite], axis=0) >= 0)
    upper = np.asarray(sadf(y, min_length=20, lags=1))
    for q in [0.5, 0.9]:
        c = cadf(y, 20, 1, q=q)
        lower = qadf(y, 20, 1, q=q)
        ok = np.isfinite(c)
        assert np.all(c[ok] >= lower[ok] - 1e-12)
        assert np.all(c[ok] <= upper[ok] + 1e-12)


def test_qadf_and_cadf_pandas_index_and_numpy_agree():
    idx = pd.date_range("2021-01-01", periods=60, freq="D")
    s = pd.Series(np.random.default_rng(25).standard_normal(60).cumsum(), index=idx)
    for fn in (qadf, cadf):
        out = fn(s, 20, 1, q=0.9)
        assert isinstance(out, pd.Series)
        assert out.index.equals(idx)
        np.testing.assert_allclose(out.to_numpy(), fn(s.to_numpy(), 20, 1, q=0.9), rtol=1e-12, equal_nan=True)
    level, dispersion = qadf(s, 20, 1, q=0.9, v=0.05)
    assert isinstance(level, pd.Series) and isinstance(dispersion, pd.Series)


@pytest.mark.parametrize("bad", [-0.1, 1.1, float("nan"), True, "0.9"])
def test_qadf_and_cadf_reject_q_outside_unit_interval(bad):
    y = np.random.default_rng(26).standard_normal(60).cumsum()
    with pytest.raises(ValueError, match="q must"):
        qadf(y, 20, 1, q=bad)
    with pytest.raises(ValueError, match="q must"):
        cadf(y, 20, 1, q=bad)


@pytest.mark.parametrize(
    "q, v, message",
    [(0.9, 0.0, "v must"), (0.9, -0.05, "v must"), (0.02, 0.05, "outside"), (0.98, 0.05, "outside")],
)
def test_qadf_rejects_bad_dispersion_band(q, v, message):
    y = np.random.default_rng(27).standard_normal(60).cumsum()
    with pytest.raises(ValueError, match=message):
        qadf(y, 20, 1, q=q, v=v)


def test_qadf_and_cadf_reject_short_window_and_bad_lag():
    y = np.random.default_rng(28).standard_normal(60).cumsum()
    with pytest.raises(ValueError, match="lag=2"):
        qadf(y, 5, 2)
    with pytest.raises(ValueError, match="lag must be an integer"):
        cadf(y, 20, 1.5)
    with pytest.raises(ValueError, match="lag must be >= 0"):
        qadf(y, 20, -1)
    with pytest.raises(ValueError, match="constant"):
        cadf(y, 20, 1, constant="trend")


def test_qadf_and_cadf_refuse_constant_series():
    with pytest.raises(ValueError, match="constant"):
        qadf(np.full(40, 1.0), 15, 1)
    with pytest.raises(ValueError, match="constant"):
        cadf(np.full(40, 1.0), 15, 1)


# ---------------------------------------------------------------------------
# Sub- and super-martingale tests (AFML 17.4.3)
# ---------------------------------------------------------------------------


def naive_smt_tstat(y, t0, t1, kind):
    """OLS on the window [t0, t1] (0-based) with raw 1-based positions and lstsq."""
    pos = np.arange(t0, t1 + 1) + 1.0
    yy = y[t0 : t1 + 1]
    if kind in ("poly2", "exp", "power"):
        yy = np.log(yy)
    if kind in ("poly1", "poly2"):
        x, k = np.column_stack([np.ones_like(pos), pos, pos**2]), 2
    elif kind == "exp":
        x, k = np.column_stack([np.ones_like(pos), pos]), 1
    else:
        x, k = np.column_stack([np.ones_like(pos), np.log(pos)]), 1
    beta = np.linalg.lstsq(x, yy, rcond=None)[0]
    resid = yy - x @ beta
    n, p = x.shape
    cov = (resid @ resid / (n - p)) * np.linalg.inv(x.T @ x)
    return beta[k] / math.sqrt(cov[k, k])


def naive_smt(y, min_length, kind, phi):
    out = np.full(len(y), np.nan)
    for e in range(min_length - 1, len(y)):
        out[e] = max(
            abs(naive_smt_tstat(y, s, e, kind)) / (e - s) ** phi for s in range(0, e - min_length + 2)
        )
    return out


def _positive_growth_series(n: int = 80, seed: int = 3) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return np.exp(0.01 * np.arange(n) + 0.05 * rng.standard_normal(n))


@pytest.mark.parametrize("kind", ["poly1", "poly2", "exp", "power"])
@pytest.mark.parametrize("phi", [0.0, 0.5])
def test_smt_matches_naive_loop_with_raw_positions(kind, phi):
    # The naive loop uses the raw 1-based positions, so this also checks the window centring.
    y = _positive_growth_series()
    np.testing.assert_allclose(
        smt(y, 12, kind, phi), naive_smt(y, 12, kind, phi), rtol=1e-7, equal_nan=True
    )


@pytest.mark.parametrize("kind", ["poly1", "poly2", "exp", "power"])
def test_smt_is_scale_invariant(kind):
    y = _positive_growth_series(n=60, seed=4)
    base = np.asarray(smt(y, 12, kind))
    for factor in (1e200, 1e-200):
        np.testing.assert_allclose(np.asarray(smt(y * factor, 12, kind)), base, rtol=1e-7, equal_nan=True)


@pytest.mark.parametrize("kind", ["poly1", "poly2", "exp", "power"])
def test_smt_with_phi_one_is_no_larger_than_phi_zero(kind):
    y = _positive_growth_series(n=60, seed=5)
    plain = np.asarray(smt(y, 12, kind, 0.0))
    damped = np.asarray(smt(y, 12, kind, 1.0))
    ok = np.isfinite(plain)
    assert np.all(damped[ok] <= plain[ok] + 1e-12)


def test_smt_exponential_growth_exceeds_random_walk_for_exp_kind():
    # Measured on seed 0 (see docs/proofs/structural_breaks.md, section 7). Pinned values.
    rng = np.random.default_rng(0)
    expo = np.exp(0.01 * np.arange(200)) * (1.0 + 0.01 * rng.standard_normal(200))
    walk = np.exp(np.cumsum(0.01 * np.random.default_rng(100).standard_normal(200)))
    exp_kind = np.asarray(smt(expo, 30, "exp"))
    walk_kind = np.asarray(smt(walk, 30, "exp"))
    assert float(exp_kind[-1]) == pytest.approx(845.1877, rel=1e-4)
    assert float(np.nanmax(walk_kind)) == pytest.approx(32.8009, rel=1e-4)
    assert np.nanmax(exp_kind) > 20.0 * np.nanmax(walk_kind)


def test_smt_poly2_does_not_flag_pure_exponential_growth():
    # A pure exponential is linear in logs, so the quadratic term of the log fit is near zero.
    rng = np.random.default_rng(0)
    expo = np.exp(0.01 * np.arange(200)) * (1.0 + 0.01 * rng.standard_normal(200))
    walk = np.exp(np.cumsum(0.01 * np.random.default_rng(100).standard_normal(200)))
    poly2_expo = np.asarray(smt(expo, 30, "poly2"))
    assert float(np.nanmax(poly2_expo)) == pytest.approx(4.0016, rel=1e-4)
    assert np.nanmax(poly2_expo) < np.nanmax(np.asarray(smt(walk, 30, "poly2")))


def test_smt_poly1_accepts_negative_values_and_log_kinds_refuse_them():
    y = np.random.default_rng(6).standard_normal(60)
    assert np.any(y <= 0.0)
    assert np.isfinite(np.asarray(smt(y, 12, "poly1"))[11:]).all()
    for kind in ("poly2", "exp", "power"):
        with pytest.raises(ValueError, match=f"kind={kind!r}.*strictly positive"):
            smt(y, 12, kind)


@pytest.mark.parametrize("kind", ["poly2", "exp", "power"])
def test_smt_log_kinds_refuse_zero_by_name(kind):
    y = _positive_growth_series(n=40)
    y[7] = 0.0
    with pytest.raises(ValueError, match="found 1 value"):
        smt(y, 12, kind)


def test_smt_pandas_index_and_nan_prefix():
    idx = pd.date_range("2021-01-01", periods=60, freq="D")
    s = pd.Series(_positive_growth_series(n=60, seed=8), index=idx)
    out = smt(s, 12, "exp", 0.5)
    assert isinstance(out, pd.Series)
    assert out.index.equals(idx)
    assert out.iloc[:11].isna().all()
    assert out.iloc[11:].notna().all()
    np.testing.assert_allclose(out.to_numpy(), smt(s.to_numpy(), 12, "exp", 0.5), rtol=1e-12, equal_nan=True)


@pytest.mark.parametrize("bad", ["linear", None, 1, "POLY1"])
def test_smt_refuses_unknown_kind(bad):
    with pytest.raises(ValueError, match="kind must be one of"):
        smt(_positive_growth_series(n=40), 12, bad)


@pytest.mark.parametrize("bad", [-0.1, 1.5, float("nan"), True, "0.5"])
def test_smt_refuses_phi_outside_unit_interval(bad):
    with pytest.raises(ValueError, match="phi"):
        smt(_positive_growth_series(n=40), 12, "exp", bad)


def test_smt_refuses_min_length_too_small_for_the_regressors():
    y = _positive_growth_series(n=40)
    with pytest.raises(ValueError, match="too small for kind='poly1'"):
        smt(y, 3, "poly1")
    with pytest.raises(ValueError, match="too small for kind='exp'"):
        smt(y, 2, "exp")
    assert np.isfinite(np.asarray(smt(y, 3, "exp"))[2:]).all()


@pytest.mark.parametrize("bad", [30.5, float("nan")])
def test_smt_refuses_non_integer_min_length(bad):
    with pytest.raises(ValueError, match="min_length must be an integer"):
        smt(_positive_growth_series(n=40), bad, "exp")


def test_smt_refuses_min_length_beyond_the_series():
    with pytest.raises(ValueError, match="exceeds the series length"):
        smt(_positive_growth_series(n=40), 41, "exp")


# ---------------------------------------------------------------------------
# Simulated critical values for SADF (random-walk null)
# ---------------------------------------------------------------------------


def _explosive_path(seed: int, n_calm: int = 100, n_boom: int = 100) -> np.ndarray:
    """AR(1) with phi = 0.5 for ``n_calm`` steps, then phi = 1.03 for ``n_boom`` steps."""
    rng = np.random.default_rng(seed)
    eps = rng.standard_normal(n_calm + n_boom)
    y = np.zeros(n_calm + n_boom)
    for i in range(1, y.size):
        phi = 0.5 if i < n_calm else 1.03
        y[i] = phi * y[i - 1] + eps[i]
    return y


def test_sadf_null_same_seed_gives_identical_output():
    first_frame, first_max = sadf_null_quantiles(60, min_length=12, n_sim=40, seed=5)
    second_frame, second_max = sadf_null_quantiles(60, min_length=12, n_sim=40, seed=5)
    pd.testing.assert_frame_equal(first_frame, second_frame)
    pd.testing.assert_series_equal(first_max, second_max)


def test_sadf_null_different_seeds_give_different_output():
    frame_a, max_a = sadf_null_quantiles(60, min_length=12, n_sim=40, seed=5)
    frame_b, max_b = sadf_null_quantiles(60, min_length=12, n_sim=40, seed=6)
    assert not frame_a.equals(frame_b)
    assert not max_a.equals(max_b)


def test_sadf_null_shape_and_index_follow_sadf_positions():
    n_obs, min_length = 60, 12
    frame, maximum = sadf_null_quantiles(n_obs, min_length=min_length, lags=1, constant="c", n_sim=20, seed=1)
    assert frame.shape == (n_obs, 3)
    assert frame.index.equals(pd.RangeIndex(n_obs))
    assert list(frame.columns) == [0.90, 0.95, 0.99]
    assert list(maximum.index) == [0.90, 0.95, 0.99]
    walk = np.cumsum(np.random.default_rng(2).standard_normal(n_obs))
    sadf_nan = np.isnan(np.asarray(sadf(walk, min_length=min_length, lags=1, constant="c")))
    np.testing.assert_array_equal(frame[0.95].isna().to_numpy(), sadf_nan)


def test_sadf_null_quantile_columns_are_non_decreasing():
    frame, maximum = sadf_null_quantiles(80, min_length=15, n_sim=200, seed=2)
    values = frame.to_numpy()
    rows = ~np.isnan(values[:, 0])
    assert rows.any()
    assert np.all(np.diff(values[rows], axis=1) >= 0)
    assert np.all(np.diff(maximum.to_numpy()) >= 0)


def test_sadf_null_matches_a_direct_simulation_loop():
    n_obs, min_length, n_sim, levels = 50, 12, 30, (0.5, 0.9)
    frame, maximum = sadf_null_quantiles(
        n_obs, min_length=min_length, lags=1, constant="c", quantiles=levels, n_sim=n_sim, seed=3
    )
    rng = np.random.default_rng(3)
    sims = np.vstack(
        [sadf(np.cumsum(rng.standard_normal(n_obs)), min_length=min_length, lags=1, constant="c")
         for _ in range(n_sim)]
    )
    expected = np.full((n_obs, len(levels)), np.nan)
    expected[min_length - 1 :] = np.quantile(sims[:, min_length - 1 :], levels, axis=0).T
    np.testing.assert_allclose(frame.to_numpy(), expected, rtol=1e-12, equal_nan=True)
    expected_max = np.quantile(sims[:, min_length - 1 :].max(axis=1), levels)
    np.testing.assert_allclose(maximum.to_numpy(), expected_max, rtol=1e-12)


def test_sadf_null_refuses_nc_by_name():
    with pytest.raises(ValueError, match="constant='nc'"):
        sadf_null_quantiles(60, min_length=12, constant="nc", n_sim=5)


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"n_sim": 0}, "n_sim must be >= 1, got 0"),
        ({"n_sim": -3}, "n_sim must be >= 1, got -3"),
        ({"n_sim": 2.5}, "n_sim must be an integer"),
        ({"quantiles": (0.9, 1.0)}, "quantiles must be in"),
        ({"quantiles": (0.0, 0.9)}, "quantiles must be in"),
        ({"quantiles": (0.95, 0.9)}, "strictly increasing"),
        ({"quantiles": ()}, "must not be empty"),
        ({"quantiles": 0.95}, "sequence of numbers"),
        ({"seed": -1}, "seed must be >= 0, got -1"),
        ({"n_obs": 1}, "n_obs must be >= 2, got 1"),
        ({"min_length": 40, "n_obs": 30}, "min_length=40 exceeds n_obs=30"),
        ({"min_length": 5}, "min_length=5 too small"),
        ({"lags": -1}, "lags must be >= 0"),
        ({"constant": "trend"}, "constant must be one of"),
    ],
)
def test_sadf_null_rejects_bad_arguments(kwargs, message):
    args = {"n_obs": 60, "min_length": 12, "n_sim": 5, "seed": 0}
    args.update(kwargs)
    with pytest.raises(ValueError, match=message):
        sadf_null_quantiles(**args)


def test_sadf_null_quantile_flags_an_explosive_path():
    _, maximum = sadf_null_quantiles(200, min_length=20, n_sim=200, seed=7)
    q95 = maximum[0.95]
    flagged = 0
    for k in range(40):
        s = np.asarray(sadf(_explosive_path(k), min_length=20, lags=1, constant="c"))
        flagged += bool(np.nanmax(s[19:]) > q95)
    assert flagged > 20, f"only {flagged} of 40 explosive paths exceeded q95={q95:.3f}"


def test_sadf_is_invariant_to_scale_and_level_for_intercept_specs():
    # The simulated null relies on this: scale and start level of a walk leave SADF unchanged.
    walk = np.cumsum(np.random.default_rng(4).standard_normal(80))
    for constant in ("c", "ct", "ctt"):
        base = np.asarray(sadf(walk, min_length=15, lags=1, constant=constant))
        for a, b in ((3.0, 0.0), (-0.5, 0.0), (1.0, 50.0), (1e-3, -7.0)):
            got = np.asarray(sadf(a * walk + b, min_length=15, lags=1, constant=constant))
            np.testing.assert_allclose(got, base, rtol=1e-6, equal_nan=True)


@pytest.mark.slow
def test_sadf_null_95_quantile_has_nominal_size_on_fresh_walks():
    n_obs, min_length, n_test = 100, 10, 2000
    _, maximum = sadf_null_quantiles(n_obs, min_length=min_length, lags=1, constant="c", n_sim=1000, seed=11)
    q95 = maximum[0.95]
    rng = np.random.default_rng(12)  # independent of the simulation seed
    exceed = 0
    for _ in range(n_test):
        s = np.asarray(sadf(np.cumsum(rng.standard_normal(n_obs)), min_length=min_length, lags=1, constant="c"))
        exceed += bool(np.nanmax(s[min_length - 1 :]) > q95)
    rate = exceed / n_test
    band = 3 * math.sqrt(0.05 * 0.95 / n_test)
    assert abs(rate - 0.05) <= band, f"rate {rate:.4f} outside 0.05 +/- {band:.4f} (q95={q95:.3f})"


def test_sadf_null_with_min_length_equal_to_n_obs_has_one_end_point():
    pointwise, maximum = sadf_null_quantiles(40, min_length=40, n_sim=30, seed=3)
    assert pointwise.dropna().index.tolist() == [39]
    np.testing.assert_allclose(maximum.to_numpy(), pointwise.loc[39].to_numpy(), rtol=0, atol=1e-12)
