"""Tests for finlab.microstructure (AFML ch. 19).

Each fast implementation is cross-checked against a brute-force reference on
seeded random inputs. Hand-made examples pin down sign and boundary behaviour.
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest

from finlab.microstructure import (
    amihud_illiquidity,
    becker_parkinson_volatility,
    bulk_volume_classification,
    corwin_schultz_spread,
    hasbrouck_lambda,
    hasbrouck_lambda_tstat,
    kyle_lambda,
    kyle_lambda_tstat,
    roll_measure,
    signed_root_dollar_volume,
    tick_rule,
    vpin,
)

SQRT2 = math.sqrt(2.0)
DEN = 3.0 - 2.0 * SQRT2
K2 = math.sqrt(8.0 / math.pi)


# ---------------------------------------------------------------------------
# Naive references (deliberately simple, loop-based)
# ---------------------------------------------------------------------------
def naive_tick_rule(prices: list[float]) -> list[int]:
    b = [1]
    for t in range(1, len(prices)):
        dp = prices[t] - prices[t - 1]
        if dp > 0:
            b.append(1)
        elif dp < 0:
            b.append(-1)
        else:
            b.append(b[-1])
    return b


def naive_roll(prices: list[float]) -> float:
    dp = [prices[t] - prices[t - 1] for t in range(1, len(prices))]
    pairs = [(dp[t], dp[t - 1]) for t in range(1, len(dp))]
    mx = sum(a for a, _ in pairs) / len(pairs)
    my = sum(b for _, b in pairs) / len(pairs)
    cov = sum((a - mx) * (b - my) for a, b in pairs) / len(pairs)
    return 2.0 * math.sqrt(max(0.0, -cov))


def naive_corwin_schultz(high: list[float], low: list[float], sl: int) -> list[float]:
    n = len(high)
    out = [math.nan] * n
    for t in range(sl, n):
        beta_terms = []
        for k in range(sl):
            a = t - k
            b = t - k - 1
            beta_terms.append(
                math.log(high[a] / low[a]) ** 2 + math.log(high[b] / low[b]) ** 2
            )
        beta = sum(beta_terms) / sl
        gamma = math.log(max(high[t - 1], high[t]) / min(low[t - 1], low[t])) ** 2
        alpha = (SQRT2 - 1.0) * math.sqrt(beta) / DEN - math.sqrt(gamma / DEN)
        if alpha < 0:
            alpha = 0.0
        out[t] = 2.0 * (math.exp(alpha) - 1.0) / (1.0 + math.exp(alpha))
    return out


def naive_becker_parkinson(high: list[float], low: list[float], sl: int) -> list[float]:
    n = len(high)
    out = [math.nan] * n
    for t in range(max(sl, 1), n):
        beta = 0.0
        for k in range(sl):
            a = t - k
            b = t - k - 1
            beta += math.log(high[a] / low[a]) ** 2 + math.log(high[b] / low[b]) ** 2
        beta /= sl
        gamma = math.log(max(high[t - 1], high[t]) / min(low[t - 1], low[t])) ** 2
        s = (2 ** -0.5 - 1.0) * math.sqrt(beta) / (K2 * DEN) + math.sqrt(
            gamma / (K2 * K2 * DEN)
        )
        out[t] = max(s, 0.0)
    return out


def naive_vpin(
    prices: list[float], volume: list[int], bucket: int, n_window: int
) -> list[float]:
    """Trade-by-trade reference: expand each unit of volume into one unit trade."""
    dp = [prices[t] - prices[t - 1] for t in range(1, len(prices))]
    # Causal scale: for change dp[j], the sample sd of dp[0..j-1]; 0.5 if fewer
    # than two past changes or zero spread. Matches the expanding default.
    buy_frac = [0.5]
    for j, x in enumerate(dp):
        past = dp[:j]
        if len(past) < 2:
            buy_frac.append(0.5)
            continue
        mean = sum(past) / len(past)
        sigma = math.sqrt(sum((y - mean) ** 2 for y in past) / (len(past) - 1))
        if sigma <= 0.0:
            buy_frac.append(0.5)
            continue
        buy_frac.append(0.5 * (1.0 + math.erf((x / sigma) / SQRT2)))
    units: list[tuple[float, float]] = []
    for v, bf in zip(volume, buy_frac):
        for _ in range(int(v)):
            units.append((bf, 1.0 - bf))
    total = len(units)
    n_full = total // bucket
    bs = []
    for k in range(n_full):
        chunk = units[k * bucket : (k + 1) * bucket]
        bs.append((sum(u[0] for u in chunk), sum(u[1] for u in chunk)))
    out = []
    for tau in range(n_full):
        if tau < n_window - 1:
            out.append(math.nan)
            continue
        s = sum(abs(bs[k][0] - bs[k][1]) for k in range(tau - n_window + 1, tau + 1))
        out.append(s / (n_window * bucket))
    return out


def naive_hasbrouck_lambda(bar_close: list[float], srdv: list[float]) -> float:
    """Loop over bars: ``sum(x y) / sum(x^2)`` with ``y`` the log-price change into each bar."""
    sxx = 0.0
    sxy = 0.0
    for tau in range(1, len(bar_close)):
        y = math.log(bar_close[tau]) - math.log(bar_close[tau - 1])
        sxx += srdv[tau] * srdv[tau]
        sxy += srdv[tau] * y
    return sxy / sxx


def naive_hasbrouck(bar_close: list[float], srdv: list[float]) -> tuple[float, float]:
    """Loop over bars straight from the OLS formulas; returns ``(lambda, t)``."""
    lam = naive_hasbrouck_lambda(bar_close, srdv)
    ssr = 0.0
    sxx = 0.0
    for tau in range(1, len(bar_close)):
        y = math.log(bar_close[tau]) - math.log(bar_close[tau - 1])
        e = y - lam * srdv[tau]
        ssr += e * e
        sxx += srdv[tau] * srdv[tau]
    se = math.sqrt(ssr / (len(bar_close) - 2) / sxx)
    return lam, lam / se


def naive_signed_root_dollar_volume(
    prices: list[float], volumes: list[float], signs: list[float], bar_ids: list[int]
) -> list[float]:
    """Add each trade's ``b_t sqrt(p_t V_t)`` to its bar, one trade at a time."""
    out = [0.0] * (max(bar_ids) + 1)
    for p, v, s, b in zip(prices, volumes, signs, bar_ids):
        out[b] += s * math.sqrt(p * v)
    return out


# ---------------------------------------------------------------------------
# Tick rule
# ---------------------------------------------------------------------------
def test_tick_rule_hand_path() -> None:
    # up, flat (carries up), down, flat (carries down), up
    prices = np.array([10.0, 10.5, 10.5, 10.2, 10.2, 10.3])
    assert tick_rule(prices).tolist() == [1, 1, 1, -1, -1, 1]


def test_tick_rule_first_element_and_leading_decline() -> None:
    assert tick_rule(np.array([10.0, 9.5, 9.5])).tolist() == [1, -1, -1]
    assert tick_rule(np.array([10.0])).tolist() == [1]


def test_tick_rule_series_roundtrip() -> None:
    p = pd.Series([1.0, 2.0, 1.5], index=pd.date_range("2024-01-01", periods=3))
    out = tick_rule(p)
    assert isinstance(out, pd.Series)
    assert out.index.equals(p.index)
    assert out.tolist() == [1, 1, -1]


@pytest.mark.parametrize("seed", range(5))
def test_tick_rule_matches_naive(seed: int) -> None:
    rng = np.random.default_rng(seed)
    # round to a tick grid so zero moves occur frequently
    p = np.round(100 + np.cumsum(rng.normal(size=400)), 1)
    assert tick_rule(p).tolist() == naive_tick_rule(p.tolist())


# ---------------------------------------------------------------------------
# Roll measure
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("seed", range(4))
def test_roll_matches_naive(seed: int) -> None:
    rng = np.random.default_rng(seed)
    p = 50 + np.cumsum(rng.normal(size=300)) + 0.2 * rng.choice([-1, 1], size=300)
    assert roll_measure(p) == pytest.approx(naive_roll(p.tolist()), rel=1e-12)


def test_roll_recovers_spread_from_bid_ask_bounce() -> None:
    # p_t = m_t + b_t * c with m a random walk: Roll should estimate 2c.
    # The bounce must dominate the walk's increments. With c = 0.05 and increment sd 0.5
    # the relative error is about 11% per seed, so a 5% check passes by luck. With
    # c = 0.5 and increment sd 0.05 it is about 0.2%, and the 5% check is meaningful.
    rng = np.random.default_rng(7)
    T = 200_000
    c = 0.5
    m = np.cumsum(rng.normal(scale=0.05, size=T))
    b = rng.choice([-1.0, 1.0], size=T)
    p = m + b * c
    assert roll_measure(p) == pytest.approx(2 * c, rel=0.05)


def test_roll_zero_when_serial_covariance_nonnegative() -> None:
    # A strictly monotone path has constant price changes: covariance is 0.
    assert roll_measure(np.arange(10, dtype=float)) == 0.0


# ---------------------------------------------------------------------------
# Corwin-Schultz and Becker-Parkinson
# ---------------------------------------------------------------------------
def _random_hl(rng: np.random.Generator, n: int) -> tuple[np.ndarray, np.ndarray]:
    mid = 100 * np.exp(np.cumsum(rng.normal(scale=0.01, size=n)))
    high = mid * (1 + np.abs(rng.normal(scale=0.01, size=n)))
    low = mid * (1 - np.abs(rng.normal(scale=0.01, size=n)))
    return high, low


@pytest.mark.parametrize("sl", [1, 2, 4])
@pytest.mark.parametrize("seed", range(3))
def test_corwin_schultz_matches_naive_and_is_nonnegative(seed: int, sl: int) -> None:
    rng = np.random.default_rng(seed)
    high, low = _random_hl(rng, 120)
    fast = corwin_schultz_spread(high, low, sl=sl)
    ref = np.array(naive_corwin_schultz(high.tolist(), low.tolist(), sl))
    assert np.all(np.isnan(fast[:sl]))
    np.testing.assert_allclose(fast[sl:], ref[sl:], rtol=1e-10, atol=1e-14)
    assert np.all(fast[sl:] >= 0.0)


def test_corwin_schultz_clips_negative_alpha_to_zero() -> None:
    # Bars alternate between two disjoint ranges (an overnight-style gap).
    # The two-bar range gamma is then far larger than the single-bar beta,
    # so alpha < 0 and the spread is clipped to 0.
    high = np.tile([101.0, 102.1], 5)
    low = np.tile([100.9, 102.0], 5)
    s = corwin_schultz_spread(high, low)
    assert np.all(s[1:] == 0.0)
    # A constant range gives alpha = sqrt(gamma) > 0, a positive spread.
    flat = corwin_schultz_spread(np.full(10, 100.5), np.full(10, 99.5))
    assert np.all(flat[1:] > 0.0)


def test_corwin_schultz_series_roundtrip() -> None:
    idx = pd.date_range("2024-01-01", periods=30, freq="D")
    rng = np.random.default_rng(3)
    h, lo = _random_hl(rng, 30)
    out = corwin_schultz_spread(pd.Series(h, index=idx), pd.Series(lo, index=idx))
    assert isinstance(out, pd.Series) and out.index.equals(idx)
    ref = np.array(naive_corwin_schultz(h.tolist(), lo.tolist(), 1))
    assert np.isnan(out.iloc[0])
    np.testing.assert_allclose(out.to_numpy()[1:], ref[1:], rtol=1e-10, atol=1e-14)


@pytest.mark.parametrize("seed", range(3))
def test_becker_parkinson_matches_naive(seed: int) -> None:
    rng = np.random.default_rng(seed)
    high, low = _random_hl(rng, 100)
    fast = becker_parkinson_volatility(high, low, sl=2)
    ref = np.array(naive_becker_parkinson(high.tolist(), low.tolist(), 2))
    np.testing.assert_allclose(fast[2:], ref[2:], rtol=1e-10, atol=1e-14)
    assert np.all(fast[2:] >= 0.0)


def test_becker_parkinson_recovers_volatility_on_gbm() -> None:
    # Book formula, checked by simulation (see docs/proofs/microstructure.md).
    # One GBM path is cut into consecutive bars, so each bar starts where the last one
    # ended, which the two-bar term of the estimator assumes.
    rng = np.random.default_rng(0)
    sigma = 0.02
    nb, steps = 20_000, 400
    inc = rng.normal(0.0, sigma / math.sqrt(steps), size=nb * steps)
    path = np.concatenate([[0.0], np.cumsum(inc)])
    bars = np.stack([path[k * steps : (k + 1) * steps + 1] for k in range(nb)])
    hi = np.exp(bars.max(axis=1))
    lo = np.exp(bars.min(axis=1))
    est = becker_parkinson_volatility(hi, lo, sl=1)
    # At 400 monitoring steps per bar the mean is about 5% low (0.01893 at seed 0,
    # measured). Finer monitoring shrinks the gap but does not remove it, so the 10%
    # tolerance covers the bias.
    assert np.nanmean(est) == pytest.approx(sigma, rel=0.10)


# ---------------------------------------------------------------------------
# Kyle lambda and Amihud
# ---------------------------------------------------------------------------
def test_kyle_lambda_recovers_known_slope_exactly() -> None:
    # dp_t = 2.5 * x_t exactly, so OLS must return 2.5 with zero residual.
    rng = np.random.default_rng(1)
    x = rng.choice([-1.0, 1.0], size=500) * rng.integers(1, 10, size=500)
    dp = 2.5 * x
    prices = np.concatenate([[0.0], np.cumsum(dp[1:])])
    signed = np.concatenate([[0.0], x[1:]])
    assert kyle_lambda(prices, signed) == pytest.approx(2.5, abs=1e-12)
    lam, t = kyle_lambda_tstat(prices, signed)
    assert lam == pytest.approx(2.5, abs=1e-12)
    assert math.isinf(t)


@pytest.mark.parametrize("seed", range(3))
def test_kyle_lambda_noisy_matches_lstsq(seed: int) -> None:
    rng = np.random.default_rng(seed)
    T = 400
    x = rng.choice([-1.0, 1.0], size=T) * rng.exponential(5.0, size=T)
    dp = 0.8 * x + rng.normal(scale=0.5, size=T)
    prices = np.concatenate([[0.0], np.cumsum(dp[1:])])
    signed = np.concatenate([[0.0], x[1:]])
    lam_ref = np.linalg.lstsq(signed[1:, None], np.diff(prices), rcond=None)[0][0]
    lam, t = kyle_lambda_tstat(prices, signed)
    assert lam == pytest.approx(lam_ref, rel=1e-10)
    assert abs(lam - 0.8) < 0.05
    assert abs(t) > 10  # the slope is well identified
    # Independent t-value: OLS through the origin, standard error with n - 1 degrees of freedom.
    xr, yr = signed[1:], np.diff(prices)
    resid = yr - lam_ref * xr
    se_ref = math.sqrt(float(resid @ resid) / (len(xr) - 1) / float(xr @ xr))
    assert t == pytest.approx(lam_ref / se_ref, rel=1e-10)


def test_kyle_lambda_rejects_zero_signed_volume() -> None:
    with pytest.raises(ValueError, match="identically zero"):
        kyle_lambda(np.arange(5.0), np.zeros(5))


def test_amihud_regression_and_ratio_match_naive() -> None:
    rng = np.random.default_rng(4)
    r = rng.normal(scale=0.01, size=50)
    dv = rng.uniform(1e5, 1e6, size=50)
    num = sum(abs(a) * b for a, b in zip(r, dv))
    den = sum(b * b for b in dv)
    assert amihud_illiquidity(r, dv) == pytest.approx(num / den, rel=1e-12)
    ratio = sum(abs(a) / b for a, b in zip(r, dv)) / len(r)
    assert amihud_illiquidity(r, dv, method="ratio") == pytest.approx(ratio, rel=1e-12)


def test_amihud_rejects_bad_method() -> None:
    with pytest.raises(ValueError, match="unknown method"):
        amihud_illiquidity(np.ones(3), np.ones(3), method="nope")


# ---------------------------------------------------------------------------
# Hasbrouck lambda and signed root dollar volume
# ---------------------------------------------------------------------------
def test_hasbrouck_lambda_recovers_known_slope_exactly() -> None:
    # log closes follow lambda * x exactly. exp and log round each close, so the
    # recovered slope agrees to about 1e-14, not bit for bit.
    rng = np.random.default_rng(1)
    x = rng.choice([-1.0, 1.0], size=300) * rng.uniform(1.0, 10.0, size=300)
    lam = 0.37
    logc = np.r_[0.0, np.cumsum(lam * x[1:])]
    srdv = x.copy()
    srdv[0] = 123.0  # element 0 is unused, so this value must not change the result
    assert hasbrouck_lambda(np.exp(logc), srdv) == pytest.approx(lam, abs=1e-10)
    got, t = hasbrouck_lambda_tstat(np.exp(logc), srdv)
    assert got == pytest.approx(lam, abs=1e-10)
    assert abs(t) > 1e6  # residuals are rounding noise, so the t-value is huge


@pytest.mark.parametrize("seed", range(3))
def test_hasbrouck_lambda_matches_numpy_lstsq(seed: int) -> None:
    rng = np.random.default_rng(seed)
    B = 300
    x = rng.choice([-1.0, 1.0], size=B) * rng.exponential(1.0, size=B)
    y = 0.8 * x[1:] + rng.normal(scale=0.5, size=B - 1)
    bar_close = np.exp(np.r_[0.0, np.cumsum(y)])
    lam_ref = np.linalg.lstsq(x[1:, None], np.diff(np.log(bar_close)), rcond=None)[0][0]
    lam, t = hasbrouck_lambda_tstat(bar_close, x)
    assert lam == pytest.approx(lam_ref, rel=1e-10)
    # The slope is within four standard errors of the truth, where se is the t-test's own se.
    se = lam / t
    assert abs(lam - 0.8) < 4 * abs(se)


@pytest.mark.parametrize("seed", range(3))
def test_hasbrouck_lambda_matches_naive_loop(seed: int) -> None:
    rng = np.random.default_rng(seed)
    B = 120
    srdv = rng.choice([-1.0, 1.0], size=B) * rng.uniform(0.5, 50.0, size=B)
    log_steps = np.r_[0.0, 0.01 * srdv[1:] + rng.normal(0, 0.02, B - 1)]
    bar_close = 100 * np.exp(np.cumsum(log_steps))
    lam, t = hasbrouck_lambda_tstat(bar_close, srdv)
    lam_ref, t_ref = naive_hasbrouck(bar_close.tolist(), srdv.tolist())
    assert lam == pytest.approx(lam_ref, rel=1e-10)
    assert t == pytest.approx(t_ref, rel=1e-9)
    assert hasbrouck_lambda(bar_close, srdv) == pytest.approx(lam_ref, rel=1e-10)


def test_hasbrouck_tstat_matches_hand_computation() -> None:
    # Regressor (1, 2, 3) against response (2, 3, 5), from bars 1 to 3.
    # sum(xy) = 23, sum(x^2) = 14, so lambda = 23/14. SSR = 38 - 23**2/14 = 3/14.
    # se^2 = SSR / ((3 - 1) * 14) = 3/392, so t^2 = (23/14)**2 * 392/3 = 1058/3.
    bar_close = np.exp([0.0, 2.0, 5.0, 10.0])
    srdv = np.array([0.0, 1.0, 2.0, 3.0])
    lam, t = hasbrouck_lambda_tstat(bar_close, srdv)
    assert lam == pytest.approx(23.0 / 14.0, rel=1e-12)
    assert t == pytest.approx(math.sqrt(1058.0 / 3.0), rel=1e-12)


def test_hasbrouck_scales_with_a_tiny_regressor() -> None:
    # Squares of a regressor near 1e-160 are subnormal, so sum(x^2) loses precision and the
    # standard error overflowed to inf, which gave t = 0 for a fit with t of about 18.8.
    bar_close = np.exp([0.0, 2.0, 5.0, 10.0])
    srdv = np.array([0.0, 1.0, 2.0, 3.0])
    lam0, t0 = hasbrouck_lambda_tstat(bar_close, srdv)
    lam, t = hasbrouck_lambda_tstat(bar_close, srdv * 1e-160)
    assert t == pytest.approx(t0, rel=1e-12)
    assert lam == pytest.approx(lam0 * 1e160, rel=1e-12)
    assert hasbrouck_lambda(bar_close, srdv * 1e-160) == pytest.approx(lam0 * 1e160, rel=1e-12)


def test_hasbrouck_lambda_rejects_zero_regressor() -> None:
    with pytest.raises(ValueError, match="identically zero"):
        hasbrouck_lambda(np.array([1.0, 2.0, 3.0, 4.0]), np.zeros(4))
    with pytest.raises(ValueError, match="identically zero"):
        hasbrouck_lambda_tstat(np.array([1.0, 2.0, 3.0, 4.0]), np.zeros(4))


@pytest.mark.parametrize(
    ("bar_close", "srdv", "match"),
    [
        ([100.0, 0.0, 101.0, 102.0], [0.0, 1.0, 2.0, 3.0], "strictly positive"),
        ([100.0, -1.0, 101.0, 102.0], [0.0, 1.0, 2.0, 3.0], "strictly positive"),
        ([100.0, 101.0, 102.0], [0.0, 1.0, 2.0, 3.0], "same length"),
        ([100.0, 101.0], [0.0, 1.0], "at least 3"),
        ([100.0, math.nan, 101.0, 102.0], [0.0, 1.0, 2.0, 3.0], "finite"),
        ([100.0, 101.0, 102.0, 103.0], [0.0, 1.0, math.inf, 3.0], "finite"),
    ],
)
def test_hasbrouck_refuses_bad_input(bar_close, srdv, match: str) -> None:
    with pytest.raises(ValueError, match=match):
        hasbrouck_lambda(bar_close, srdv)
    with pytest.raises(ValueError, match=match):
        hasbrouck_lambda_tstat(bar_close, srdv)


def test_hasbrouck_refuses_labels_that_differ() -> None:
    idx = pd.date_range("2024-01-01", periods=5, freq="D")
    close = pd.Series(np.linspace(100.0, 104.0, 5), index=idx)
    srdv = pd.Series(np.arange(5.0), index=idx + pd.Timedelta(hours=12))
    with pytest.raises(ValueError, match="same index"):
        hasbrouck_lambda(close, srdv)


def test_hasbrouck_lambda_large_regressor_stays_finite() -> None:
    # The squared regressor overflows near 1e154. The scaled fallback must still
    # return the slope, which is about 3e-200 here.
    rng = np.random.default_rng(9)
    x = 1e200 * rng.choice([-1.0, 1.0], size=40) * rng.uniform(0.5, 2.0, size=40)
    x[0] = 0.0
    lam_true = 3e-200
    bar_close = np.exp(np.r_[0.0, np.cumsum(lam_true * x[1:])])
    got = hasbrouck_lambda(bar_close, x)
    assert np.isfinite(got)
    assert got == pytest.approx(lam_true, rel=1e-9)
    lam, t = hasbrouck_lambda_tstat(bar_close, x)
    assert lam == pytest.approx(lam_true, rel=1e-9)
    assert not np.isnan(t)


def test_signed_root_dollar_volume_hand_example() -> None:
    # Bar 0: +sqrt(4 * 1) = 2, then -sqrt(1 * 1) = -1, so the bar sums to 1.
    # Bar 1 has no trades and gets 0. Bar 2: +sqrt(9 * 4) = 6, and the trade with
    # sign 0 adds nothing.
    out = signed_root_dollar_volume(
        [4.0, 1.0, 9.0, 16.0], [1.0, 1.0, 4.0, 1.0], [1, -1, 1, 0], [0, 0, 2, 2]
    )
    assert out.tolist() == [1.0, 0.0, 6.0]


@pytest.mark.parametrize("seed", range(3))
def test_signed_root_dollar_volume_matches_naive(seed: int) -> None:
    rng = np.random.default_rng(seed)
    n = 300
    prices = 50 + 10 * rng.random(n)
    volumes = rng.integers(1, 100, size=n).astype(float)
    signs = rng.choice([-1.0, 0.0, 1.0], size=n)
    bar_ids = np.sort(rng.integers(0, 40, size=n))  # some bars may be empty
    got = signed_root_dollar_volume(prices, volumes, signs, bar_ids)
    ref = naive_signed_root_dollar_volume(
        prices.tolist(), volumes.tolist(), signs.tolist(), bar_ids.tolist()
    )
    assert got.shape == (int(bar_ids.max()) + 1,)
    np.testing.assert_allclose(got, ref, rtol=1e-12, atol=0.0)


def test_signed_root_dollar_volume_large_prices_stay_finite() -> None:
    # p * V = 1e400 overflows float64, but sqrt(p) * sqrt(V) = 1e200 does not.
    got = signed_root_dollar_volume([1e200], [1e200], [1], [0])
    assert got.tolist() == [pytest.approx(1e200, rel=1e-12)]


def test_hasbrouck_trade_level_matches_bar_reference() -> None:
    # Trades are grouped into bars of six, and each bar closes at its last trade.
    rng = np.random.default_rng(5)
    bars, per = 50, 6
    n = bars * per
    bar_ids = np.repeat(np.arange(bars), per)
    prices = 100 * np.exp(np.cumsum(rng.normal(scale=0.01, size=n)))
    volumes = rng.integers(1, 40, size=n).astype(float)
    signs = tick_rule(prices)
    srdv = signed_root_dollar_volume(prices, volumes, signs, bar_ids)
    bar_close = prices[per - 1 :: per]
    ref_srdv = naive_signed_root_dollar_volume(
        prices.tolist(), volumes.tolist(), signs.tolist(), bar_ids.tolist()
    )
    np.testing.assert_allclose(srdv, ref_srdv, rtol=1e-12, atol=0.0)
    lam, t = hasbrouck_lambda_tstat(bar_close, srdv)
    lam_ref, t_ref = naive_hasbrouck(bar_close.tolist(), srdv.tolist())
    assert lam == pytest.approx(lam_ref, rel=1e-10)
    assert t == pytest.approx(t_ref, rel=1e-9)


@pytest.mark.parametrize(
    ("prices", "volumes", "signs", "bar_ids", "match"),
    [
        ([0.0, 1.0], [1.0, 1.0], [1, 1], [0, 0], "prices must be strictly positive"),
        ([1.0, 1.0], [0.0, 1.0], [1, 1], [0, 0], "volumes must be strictly positive"),
        ([1.0, 1.0], [1.0, 1.0], [1, 2], [0, 0], "signs must be -1, 0 or 1"),
        ([1.0, 1.0], [1.0, 1.0], [1, math.nan], [0, 0], "finite"),
        ([1.0, 1.0], [1.0, 1.0], [1, 1], [0, -1], "non-negative"),
        ([1.0, 1.0], [1.0, 1.0], [1, 1], [0.0, 1.0], "integers"),
        ([1.0, 1.0], [1.0, 1.0], [1, 1], [True, False], "integers"),
        ([1.0, 1.0], [1.0, 1.0], [1, 1], [0], "same length"),
        ([], [], [], [], "at least one trade"),
        ([1.0, 1.0], [1.0, 1.0], [1, 1], [0, 10_000_000], "limit"),
        ([[1.0, 2.0]], [[1.0, 2.0]], [[1, 1]], [0, 0], "one-dimensional"),
    ],
)
def test_signed_root_dollar_volume_refuses_bad_input(
    prices, volumes, signs, bar_ids, match: str
) -> None:
    with pytest.raises(ValueError, match=match):
        signed_root_dollar_volume(prices, volumes, signs, bar_ids)


def test_signed_root_dollar_volume_refuses_labels_that_differ() -> None:
    idx = pd.date_range("2024-01-01", periods=3, freq="D")
    prices = pd.Series([1.0, 2.0, 3.0], index=idx)
    bar_ids = pd.Series([0, 0, 1], index=idx + pd.Timedelta(hours=12))
    with pytest.raises(ValueError, match="same index"):
        signed_root_dollar_volume(prices, [1.0, 1.0, 1.0], [1, 1, 1], bar_ids)


# ---------------------------------------------------------------------------
# BVC and VPIN
# ---------------------------------------------------------------------------
def test_bvc_buy_fraction_is_half_at_zero_change_and_monotone() -> None:
    p = np.array([10.0, 10.0, 11.0, 9.0])
    bf = bulk_volume_classification(p, np.ones(4), sigma=1.0)
    assert bf[1] == pytest.approx(0.5)
    assert bf[2] > 0.5 > bf[3]
    assert np.all((bf >= 0) & (bf <= 1))
    # Exact values: buy_frac = Phi(dp / sigma) with dp = (0, 1, -2) and sigma = 1.
    assert bf[2] == pytest.approx(0.5 * (1.0 + math.erf(1.0 / SQRT2)), rel=1e-12)
    assert bf[3] == pytest.approx(0.5 * (1.0 + math.erf(-2.0 / SQRT2)), rel=1e-12)


@pytest.mark.parametrize("seed", range(3))
def test_vpin_matches_trade_level_reference(seed: int) -> None:
    rng = np.random.default_rng(seed)
    n = 150
    prices = 100 + np.cumsum(rng.normal(scale=0.3, size=n))
    volume = rng.integers(1, 9, size=n)  # integer volumes make the expansion exact
    bucket = 25
    n_window = 4
    fast = vpin(prices, volume.astype(float), bucket_volume=float(bucket), n_window=n_window)
    ref = np.array(naive_vpin(prices.tolist(), volume.tolist(), bucket, n_window))
    assert fast.shape == ref.shape
    np.testing.assert_allclose(fast, ref, rtol=1e-9, atol=1e-12, equal_nan=True)


@pytest.mark.parametrize("seed", range(3))
def test_vpin_bounded_in_unit_interval(seed: int) -> None:
    rng = np.random.default_rng(seed)
    prices = 100 + np.cumsum(rng.normal(size=500))
    volume = rng.exponential(10.0, size=500)
    out = vpin(prices, volume, bucket_volume=50.0, n_window=10)
    finite = out[~np.isnan(out)]
    assert finite.size > 0
    assert np.all(finite >= 0.0) and np.all(finite <= 1.0 + 1e-12)


def test_vpin_one_sided_flow_is_one() -> None:
    # Every price rises by far more than sigma, so every trade is a buy
    # (buy_frac -> 1) and VPIN -> 1 once bucket 0 has left the window.
    prices = np.arange(1, 201, dtype=float) * 1000.0
    volume = np.full(200, 10.0)
    out = vpin(prices, volume, bucket_volume=100.0, n_window=5, sigma=1.0)
    # The first observation has no prior price (buy_frac = 0.5), so windows
    # that include bucket 0 are not pure. Check windows after bucket 0 leaves.
    finite = out[5:]
    assert np.all(np.isfinite(finite))
    np.testing.assert_allclose(finite, 1.0, atol=1e-9)


def test_bvc_default_scale_is_causal() -> None:
    """Perturbing future prices must not change earlier buy fractions."""
    rng = np.random.default_rng(11)
    p = 100 + np.cumsum(rng.normal(scale=0.5, size=200))
    v = np.ones_like(p)
    base = bulk_volume_classification(p, v)
    q = p.copy()
    q[120:] += 50.0  # large jump in the future
    shifted = bulk_volume_classification(q, v)
    np.testing.assert_array_equal(base[:120], shifted[:120])


@pytest.mark.parametrize("bad_price, bad_volume", [(np.inf, 100.0), (-np.inf, 100.0), (np.nan, 100.0)])
def test_vpin_rejects_non_finite_prices(bad_price, bad_volume) -> None:
    prices = np.array([10.0, 10.1, bad_price, 10.3, 10.2, 10.4])
    volume = np.full(6, 100.0)
    with pytest.raises(ValueError, match="prices"):
        vpin(prices, volume, bucket_volume=200.0, n_window=2)


@pytest.mark.parametrize("bad_volume", [np.inf, -np.inf, np.nan])
def test_vpin_rejects_non_finite_volume(bad_volume) -> None:
    prices = np.array([10.0, 10.1, 10.2, 10.3, 10.2, 10.4])
    volume = np.array([100.0, 100.0, bad_volume, 100.0, 100.0, 100.0])
    with pytest.raises(ValueError, match="volume"):
        vpin(prices, volume, bucket_volume=200.0, n_window=2)


# ---------------------------------------------------------------------------
# Non-finite input is rejected, not silently absorbed
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("bad", [np.nan, np.inf, -np.inf])
def test_price_estimators_reject_non_finite_prices(bad) -> None:
    prices = np.array([10.0, 10.1, bad, 10.3, 10.2, 10.4])
    volume = np.full(6, 100.0)
    with pytest.raises(ValueError, match="prices"):
        tick_rule(prices)
    with pytest.raises(ValueError, match="prices"):
        roll_measure(prices)
    with pytest.raises(ValueError, match="prices"):
        bulk_volume_classification(prices, volume)
    with pytest.raises(ValueError, match="prices"):
        kyle_lambda(prices, volume)
    with pytest.raises(ValueError, match="prices"):
        kyle_lambda_tstat(prices, volume)


@pytest.mark.parametrize("bad", [np.nan, np.inf, -np.inf])
def test_volume_based_estimators_reject_non_finite_volume(bad) -> None:
    prices = np.array([10.0, 10.1, 10.2, 10.3, 10.2, 10.4])
    volume = np.array([100.0, 100.0, bad, 100.0, 100.0, 100.0])
    with pytest.raises(ValueError, match="signed_volume"):
        kyle_lambda(prices, volume)
    with pytest.raises(ValueError, match="signed_volume"):
        kyle_lambda_tstat(prices, volume)
    with pytest.raises(ValueError, match="volume"):
        bulk_volume_classification(prices, volume)


@pytest.mark.parametrize("method", ["regression", "ratio"])
@pytest.mark.parametrize("bad", [np.nan, np.inf])
def test_amihud_rejects_non_finite_inputs(method, bad) -> None:
    returns = np.array([0.01, -0.02, 0.03, 0.01])
    dollar = np.array([1e6, 2e6, 1.5e6, 1e6])
    bad_returns = returns.copy()
    bad_returns[1] = bad
    bad_dollar = dollar.copy()
    bad_dollar[2] = bad
    with pytest.raises(ValueError, match="returns"):
        amihud_illiquidity(bad_returns, dollar, method=method)
    with pytest.raises(ValueError, match="dollar_volume"):
        amihud_illiquidity(returns, bad_dollar, method=method)


def test_vpin_rejects_bucket_count_above_limit() -> None:
    prices = np.linspace(100.0, 101.0, 300)
    v = np.ones(300)
    with pytest.raises(ValueError, match="bucket"):
        vpin(prices, v, bucket_volume=1e-9, n_window=5)



def test_vpin_rejects_non_integer_n_window() -> None:
    rng = np.random.default_rng(7)
    prices = 100.0 + rng.standard_normal(300).cumsum()
    v = rng.uniform(1.0, 2.0, 300)
    with pytest.raises(ValueError):
        vpin(prices, v, bucket_volume=50.0, n_window=2.0)
    out = vpin(prices, v, bucket_volume=50.0, n_window=np.int64(2))
    assert out.shape[0] > 0


def test_vpin_accepts_two_million_buckets() -> None:
    # 2e9 shares in 1,000-share buckets is 2e6 buckets: valid high-volume input, under the cap.
    prices = np.linspace(100.0, 101.0, 200)
    v = np.full(200, 1e7)
    out = vpin(prices, v, bucket_volume=1000.0, n_window=5)
    assert out.shape[0] == 2_000_000
    finite = out[np.isfinite(out)]
    assert finite.size > 0
    assert np.all((finite >= 0.0) & (finite <= 1.0))


def test_vpin_rejects_bucket_count_above_ten_million() -> None:
    prices = np.linspace(100.0, 101.0, 200)
    v = np.full(200, 5.5e7)  # 1.1e10 shares / 1,000 = 1.1e7 buckets
    with pytest.raises(ValueError, match=r"1\.1e\+07 .*exceeds 10,000,000"):
        vpin(prices, v, bucket_volume=1000.0, n_window=5)


@pytest.mark.parametrize("bucket_volume", [float("nan"), float("inf"), 0.0, -1.0])
def test_vpin_rejects_nonfinite_or_nonpositive_bucket_volume(bucket_volume: float) -> None:
    rng = np.random.default_rng(0)
    p = 100 + np.cumsum(rng.normal(0, 0.1, 50))
    v = rng.uniform(1, 2, 50)
    with pytest.raises(ValueError, match="bucket_volume"):
        vpin(p, v, bucket_volume=bucket_volume, n_window=2)


@pytest.mark.parametrize("bad_sl", [1.5, float("nan"), float("inf")])
def test_hl_estimators_reject_non_integer_sl(bad_sl: float) -> None:
    # sl=1.5 must be refused: it would give numbers that match neither sl=1 nor sl=2.
    high = np.array([1.0, 1.2, 1.1, 1.5, 1.4, 1.9, 1.7])
    low = np.array([1.0, 1.0, 1.1, 1.2, 1.2, 1.3, 1.4])
    with pytest.raises(ValueError, match="sl must be an integer"):
        corwin_schultz_spread(high, low, sl=bad_sl)
    with pytest.raises(ValueError, match="sl must be an integer"):
        becker_parkinson_volatility(high, low, sl=bad_sl)


def test_hl_estimators_integer_valued_sl_unchanged() -> None:
    high = np.array([1.0, 1.2, 1.1, 1.5, 1.4, 1.9, 1.7])
    low = np.array([1.0, 1.0, 1.1, 1.2, 1.2, 1.3, 1.4])
    np.testing.assert_array_equal(
        corwin_schultz_spread(high, low, sl=2.0), corwin_schultz_spread(high, low, sl=2)
    )


def test_hl_estimators_reject_infinite_prices() -> None:
    # An inf bar must be refused: it turned the next spread into NaN. tick_rule already rejects inf.
    high = np.array([np.inf, 2.0, 2.0])
    low = np.array([1.0, 1.0, 1.0])
    with pytest.raises(ValueError, match="finite"):
        corwin_schultz_spread(high, low)
    with pytest.raises(ValueError, match="finite"):
        becker_parkinson_volatility(high, low)


def test_roll_measure_large_prices_stay_finite() -> None:
    # The product of price changes must not overflow for prices near 1e160.
    from finlab.microstructure import roll_measure

    s = roll_measure(np.array([1e160, 3e160] * 5))
    assert np.isfinite(s)
    assert s == pytest.approx(4e160, rel=1e-9)


def test_amihud_regression_large_dollar_volume_stays_finite() -> None:
    # dot(dv, dv) must not overflow for dollar volume near 1e160, which would give NaN.
    from finlab.microstructure import amihud_illiquidity

    dv = 1e160 * np.array([1.0, 2.0, 3.0])
    assert amihud_illiquidity(dv, dv) == pytest.approx(1.0, rel=1e-12)
    # A value that would silently overflow to 0.0: 5e157 / 5e320 = 1e-163.
    got = amihud_illiquidity(np.array([0.01, 0.02]), np.array([1e160, 2e160]))
    assert got == pytest.approx(1e-163, rel=1e-9)


def test_kyle_lambda_tstat_large_volume_is_finite() -> None:
    # sum(x*x) must not overflow for volume near 1e160, which would make lambda and t NaN.
    from finlab.microstructure import kyle_lambda, kyle_lambda_tstat

    v = 1e160 * np.array([1.0, 2.0, 3.0, 4.0])
    p = np.r_[0.0, np.cumsum(2 * v)]
    sv = np.r_[0.0, v]
    lam, t = kyle_lambda_tstat(p, sv)
    assert lam == pytest.approx(2.0, rel=1e-9)
    assert not np.isnan(t)
    assert kyle_lambda(p, sv) == pytest.approx(2.0, rel=1e-9)


def test_becker_parkinson_huge_sl_raises_value_error() -> None:
    # sl = 10**20 must raise ValueError, not reach the numba kernel and leak a raw OverflowError.
    from finlab.microstructure import becker_parkinson_volatility

    h = np.ones(50) * 1.01
    lo = np.ones(50)
    with pytest.raises(ValueError, match="sl"):
        becker_parkinson_volatility(h, lo, sl=10**20)


def test_amihud_regression_fallback_keeps_a_representable_tiny_value() -> None:
    # The rescaled denominator m * sum(dv^2) must not overflow once m * n is above
    # about 1.8e308, so the result was 0.0 where the true value 1.5e-310 is representable.
    from finlab.microstructure import amihud_illiquidity

    # abs=0 is needed: pytest.approx adds an absolute tolerance of 1e-12 by default, which would
    # accept the 0.0 this bug returned.
    assert amihud_illiquidity([0.01, 0.02], [1e308, 1e308], method="regression") == pytest.approx(
        1.5e-310, rel=1e-6, abs=0.0
    )
    assert amihud_illiquidity([0.01, 0.02], [1e300, 1e300], method="regression") == pytest.approx(
        1.5e-302, rel=1e-12, abs=0.0
    )


def _labelled_paths(n: int = 40):
    rng = np.random.default_rng(3)
    idx = pd.date_range("2024-01-01", periods=n, freq="D")
    p = 100 * np.exp(np.cumsum(rng.normal(scale=0.01, size=n)))
    v = rng.integers(1, 50, size=n).astype(float)
    return idx, p, v


def _two_input_calls(v_mean: float):
    return {
        "kyle_lambda": kyle_lambda,
        "bulk_volume_classification": bulk_volume_classification,
        "amihud_illiquidity": lambda r, dv: amihud_illiquidity(r, dv),
        "vpin": lambda p, v: vpin(p, v, 4.0 * v_mean, 2),
    }


@pytest.mark.parametrize("name", ["kyle_lambda", "bulk_volume_classification", "vpin"])
def test_descending_labelled_series_is_refused_not_read_reversed(name):
    # A descending index must be refused, not read as reversed time, since the statistic would
    # change with the labels. The structural functions refuse this order; so do these.
    idx, p, v = _labelled_paths()
    fn = _two_input_calls(float(np.mean(v)))[name]
    s_p = pd.Series(p, index=idx)
    s_v = pd.Series(v, index=idx)
    with pytest.raises(ValueError, match="sorted"):
        fn(s_p.iloc[::-1], s_v.iloc[::-1])


def test_descending_labelled_returns_are_refused_in_amihud():
    idx, p, v = _labelled_paths()
    r = pd.Series(np.r_[0.0, np.diff(np.log(p))], index=idx)
    dv = pd.Series(v * p, index=idx)
    with pytest.raises(ValueError, match="sorted"):
        amihud_illiquidity(r.iloc[::-1], dv.iloc[::-1])


@pytest.mark.parametrize("name", ["kyle_lambda", "bulk_volume_classification", "vpin"])
def test_labelled_series_with_different_labels_is_refused(name):
    # Two Series with different labels must be refused, not paired by position.
    idx, p, v = _labelled_paths()
    fn = _two_input_calls(float(np.mean(v)))[name]
    s_p = pd.Series(p, index=idx)
    shifted = pd.Series(v, index=idx + pd.Timedelta(hours=12))
    with pytest.raises(ValueError, match="same index"):
        fn(s_p, shifted)


def test_corwin_schultz_refuses_high_and_low_with_different_labels():
    idx, p, v = _labelled_paths()
    s_hi = pd.Series(p * 1.01, index=idx)
    s_lo = pd.Series(p * 0.99, index=idx[::-1])
    with pytest.raises(ValueError):
        corwin_schultz_spread(s_hi, s_lo)


def test_labelled_series_with_matching_index_is_unchanged():
    # Valid input keeps its value: labelled and matching gives the same result as numpy.
    idx, p, v = _labelled_paths()
    assert kyle_lambda(pd.Series(p, index=idx), pd.Series(v, index=idx)) == kyle_lambda(p, v)
    got = bulk_volume_classification(pd.Series(p, index=idx), pd.Series(v, index=idx))
    assert np.array_equal(np.asarray(got), bulk_volume_classification(p, v))


def test_kyle_lambda_pairs_the_same_instants_across_timezones() -> None:
    rng = np.random.default_rng(4)
    idx = pd.date_range("2021-03-01 09:00", periods=60, freq="h", tz="UTC")
    prices = pd.Series(100 + np.cumsum(rng.standard_normal(60)), index=idx)
    signed = pd.Series(rng.standard_normal(60), index=idx)
    berlin = pd.Series(signed.to_numpy(), index=idx.tz_convert("Europe/Berlin"))
    assert kyle_lambda(prices, berlin) == pytest.approx(kyle_lambda(prices, signed), abs=1e-12)


def test_kyle_lambda_refuses_different_instants_across_timezones() -> None:
    rng = np.random.default_rng(4)
    idx = pd.date_range("2021-03-01 09:00", periods=60, freq="h", tz="UTC")
    prices = pd.Series(100 + np.cumsum(rng.standard_normal(60)), index=idx)
    signed = pd.Series(rng.standard_normal(60), index=idx.tz_convert("Europe/Berlin") + pd.Timedelta(minutes=30))
    with pytest.raises(ValueError, match="same index"):
        kyle_lambda(prices, signed)
