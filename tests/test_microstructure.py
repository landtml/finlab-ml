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
    kyle_lambda,
    kyle_lambda_tstat,
    roll_measure,
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
    rng = np.random.default_rng(7)
    T = 200_000
    c = 0.05
    m = np.cumsum(rng.normal(scale=0.5, size=T))
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
    rng = np.random.default_rng(0)
    sigma = 0.02
    nb, steps = 20_000, 400
    inc = rng.normal(0.0, sigma / math.sqrt(steps), size=(nb, steps))
    path = np.concatenate([np.zeros((nb, 1)), np.cumsum(inc, axis=1)], axis=1)
    hi = np.exp(path.max(axis=1))
    lo = np.exp(path.min(axis=1))
    est = becker_parkinson_volatility(hi, lo, sl=1)
    # Discrete monitoring of 400 steps biases the range slightly low.
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


def test_kyle_lambda_rejects_zero_signed_volume() -> None:
    with pytest.raises(ValueError):
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
    with pytest.raises(ValueError):
        amihud_illiquidity(np.ones(3), np.ones(3), method="nope")


# ---------------------------------------------------------------------------
# BVC and VPIN
# ---------------------------------------------------------------------------
def test_bvc_buy_fraction_is_half_at_zero_change_and_monotone() -> None:
    p = np.array([10.0, 10.0, 11.0, 9.0])
    bf = bulk_volume_classification(p, np.ones(4), sigma=1.0)
    assert bf[1] == pytest.approx(0.5)
    assert bf[2] > 0.5 > bf[3]
    assert np.all((bf >= 0) & (bf <= 1))


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
