"""Tests for finlab.strategy_risk (AFML ch. 15: binomial Sharpe, implied precision and frequency, failure)."""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest
from scipy.stats import norm

import finlab.strategy_risk as sr
from finlab.strategy_risk import (
    binomial_sharpe,
    implied_frequency,
    implied_precision,
    mixture_returns,
    probability_of_failure,
)

SEED = 20261013
ROOT5 = math.sqrt(5.0)


def _theta_reference(p: float, n: float, pt: float, sl: float) -> float:
    """Per-bet mean and standard deviation from the two-point law, then scaled by sqrt(n)."""
    values = np.array([pt, sl])
    probs = np.array([p, 1.0 - p])
    mean = float(probs @ values)
    std = math.sqrt(float(probs @ (values - mean) ** 2))
    return mean / std * math.sqrt(n)


def _precision_by_bisection(sl: float, pt: float, n: float, target: float) -> float:
    """Slow reference: theta is strictly increasing on (0, 1), so bisection finds its root."""
    lo, hi = 1e-12, 1.0 - 1e-12
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        if _theta_reference(mid, n, pt, sl) < target:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


def _threshold_by_bisection(r_pos: float, r_neg: float, freq: float, target: float) -> float:
    """Slow reference for the failure threshold: bisection on binomial_sharpe itself."""
    lo, hi = 1e-12, 1.0 - 1e-12
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        if _theta_reference(mid, freq, r_pos, r_neg) < target:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


# ---------------------------------------------------------------------------
# binomial_sharpe (eq. 15.1 and 15.3)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("p", [0.3, 0.55, 0.9])
@pytest.mark.parametrize("n", [1.0, 52.0, 252.0])
def test_binomial_sharpe_symmetric_matches_eq_15_1(p: float, n: float) -> None:
    expected = (2 * p - 1) * math.sqrt(n) / (2 * math.sqrt(p * (1 - p)))
    assert binomial_sharpe(p, n) == pytest.approx(expected, rel=1e-12)


def test_binomial_sharpe_symmetric_value_at_052_bets() -> None:
    # Hand value: (2 * 0.55 - 1) * sqrt(52) / (2 * sqrt(0.55 * 0.45)) = 0.7247430753...
    assert binomial_sharpe(0.55, 52) == pytest.approx(0.72474307534, rel=1e-10)


def test_binomial_sharpe_asymmetric_hand_case() -> None:
    # pt = 0.02, sl = -0.01, p = 0.6: mean = 0.02 * 0.6 - 0.01 * 0.4 = 0.008;
    # std = 0.03 * sqrt(0.24); theta = mean / std * sqrt(252).
    expected = 0.008 / (0.03 * math.sqrt(0.24)) * math.sqrt(252)
    assert binomial_sharpe(0.6, 252, pt=0.02, sl=-0.01) == pytest.approx(expected, rel=1e-12)


@pytest.mark.parametrize(
    ("p", "n", "pt", "sl"),
    [(0.55, 52.0, 1.0, -1.0), (0.6, 252.0, 0.02, -0.01), (0.2, 10.0, 3.0, -0.5)],
)
def test_binomial_sharpe_equals_two_point_moments(p: float, n: float, pt: float, sl: float) -> None:
    assert binomial_sharpe(p, n, pt, sl) == pytest.approx(_theta_reference(p, n, pt, sl), rel=1e-12)


def test_binomial_sharpe_matches_simulated_bets() -> None:
    # Simulated check: 2,000,000 seeded bets, per-bet mean over sample std, scaled by sqrt(52).
    # Standard error of the sample mean is about 0.35 % here, so a 2 % tolerance is about 5 SE.
    rng = np.random.default_rng(SEED)
    wins = rng.random(2_000_000) < 0.6
    bets = np.where(wins, 1.0, -1.0)
    sample = bets.mean() / bets.std(ddof=1) * math.sqrt(52)
    assert sample == pytest.approx(binomial_sharpe(0.6, 52), rel=0.02)


def test_binomial_sharpe_scales_with_sqrt_of_frequency() -> None:
    assert binomial_sharpe(0.6, 4.0 * 52) == pytest.approx(
        2.0 * binomial_sharpe(0.6, 52), rel=1e-12
    )


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"p": 0.0}, "p must be strictly between"),
        ({"p": 1.0}, "p must be strictly between"),
        ({"p": 1.5}, "p must be strictly between"),
        ({"p": float("nan")}, "p must be finite"),
        ({"p": True}, "p must be a real number"),
        ({"freq": 0.0}, "freq must be positive"),
        ({"freq": -3.0}, "freq must be positive"),
        ({"freq": True}, "freq must be a real number"),
        ({"freq": float("inf")}, "freq must be finite"),
        ({"pt": 0.0}, "pt must be positive"),
        ({"pt": -1.0}, "pt must be positive"),
        ({"sl": 0.0}, "sl must be negative"),
        ({"sl": 0.5}, "sl must be negative"),
        ({"pt": float("nan")}, "pt must be finite"),
    ],
)
def test_binomial_sharpe_refuses_bad_inputs(kwargs: dict, message: str) -> None:
    args = {"p": 0.55, "freq": 52.0, "pt": 1.0, "sl": -1.0}
    args.update(kwargs)
    with pytest.raises(ValueError, match=message):
        binomial_sharpe(**args)


# ---------------------------------------------------------------------------
# implied_precision (Snippet 15.3, binHR)
# ---------------------------------------------------------------------------


def test_implied_precision_hand_case_closed_form() -> None:
    # sl = -1, pt = 1, freq = 4, target = 1: a = 20, b = -20, c = 4, disc = 80,
    # so p = (5 + sqrt(5)) / 10 and (2p - 1) = 1 / sqrt(5), sqrt(p(1-p)) = 1 / sqrt(5).
    p = implied_precision(-1.0, 1.0, 4.0, 1.0)
    assert p == pytest.approx((5.0 + ROOT5) / 10.0, rel=1e-12)
    assert binomial_sharpe(p, 4.0) == pytest.approx(1.0, rel=1e-12)


@pytest.mark.parametrize("sl", [-0.5, -1.0, -2.0])
@pytest.mark.parametrize("pt", [0.3, 1.0, 3.0])
@pytest.mark.parametrize("freq", [1.0, 4.0, 52.0, 260.0])
@pytest.mark.parametrize("p", [0.05, 0.3, 0.5, 0.6, 0.9])
def test_implied_precision_round_trip(sl: float, pt: float, freq: float, p: float) -> None:
    # The target comes from binomial_sharpe, so it is negative for p below the zero-mean hit rate.
    # Both signs go through the same root selection.
    target = binomial_sharpe(p, freq, pt, sl)
    assert implied_precision(sl, pt, freq, target) == pytest.approx(p, rel=1e-9)


def test_implied_precision_negative_target_returns_smaller_root() -> None:
    # Hand values: with sl = -1, pt = 1, freq = 52, the target -1 is solved by the p below 1/2.
    p = implied_precision(-1.0, 1.0, 52.0, -1.0)
    assert 0.0 < p < 0.5
    assert binomial_sharpe(p, 52.0) == pytest.approx(-1.0, rel=1e-12)


@pytest.mark.parametrize("sl", [-0.5, -1.0, -2.0])
@pytest.mark.parametrize("pt", [0.3, 1.0, 3.0])
@pytest.mark.parametrize("freq", [1.0, 52.0, 260.0])
@pytest.mark.parametrize("target", [-2.0, -0.5, 0.0, 0.5, 2.0, 6.0])
def test_implied_precision_matches_slow_bisection(
    sl: float, pt: float, freq: float, target: float
) -> None:
    # Slow reference: bisection on the two-point mean and standard deviation, with no quadratic.
    fast = implied_precision(sl, pt, freq, target)
    slow = _precision_by_bisection(sl, pt, freq, target)
    assert 0.0 < fast < 1.0
    assert fast == pytest.approx(slow, rel=1e-8)


def test_implied_precision_zero_target_is_zero_mean_hit_rate() -> None:
    # Zero Sharpe needs the mean bet return to be zero: p = -sl / (pt - sl) = 1/3 for pt = 2, sl = -1.
    assert implied_precision(-1.0, 2.0, 52.0, 0.0) == pytest.approx(1.0 / 3.0, rel=1e-12)


def test_implied_precision_discriminant_is_nonnegative_on_grid() -> None:
    # Checked by test: b^2 - 4ac equals target^2 d^2 (target^2 d^2 - 4 freq sl pt) >= 0 when sl < 0 < pt.
    for sl in (-0.5, -1.0, -2.0):
        for pt in (0.3, 1.0, 3.0):
            for freq in (1.0, 52.0, 260.0):
                for target in (-3.0, -0.1, 0.0, 0.1, 3.0):
                    d = pt - sl
                    a = (freq + target**2) * d * d
                    b = (2 * freq * sl - target**2 * d) * d
                    c = freq * sl * sl
                    disc = b * b - 4 * a * c
                    closed = target**2 * d * d * (target**2 * d * d - 4 * freq * sl * pt)
                    # Rounding in b^2 - 4ac scales with b^2, so the absolute tolerance does too.
                    scale = max(1.0, b * b)
                    assert disc >= -1e-12 * scale
                    assert disc == pytest.approx(closed, rel=1e-9, abs=1e-12 * scale)


@pytest.mark.parametrize(
    ("args", "message"),
    [
        ((0.0, 1.0, 52.0, 1.0), "sl must be negative"),
        ((0.5, 1.0, 52.0, 1.0), "sl must be negative"),
        ((-1.0, 0.0, 52.0, 1.0), "pt must be positive"),
        ((-1.0, 1.0, 0.0, 1.0), "freq must be positive"),
        ((-1.0, 1.0, 52.0, float("nan")), "target_sharpe must be finite"),
        ((-1.0, 1.0, 52.0, True), "target_sharpe must be a real number"),
    ],
)
def test_implied_precision_refuses_bad_inputs(args: tuple, message: str) -> None:
    with pytest.raises(ValueError, match=message):
        implied_precision(*args)


# ---------------------------------------------------------------------------
# implied_frequency (Snippet 15.4, binFreq)
# ---------------------------------------------------------------------------


def test_implied_frequency_hand_case_closed_form() -> None:
    # sl = -1, pt = 1, p = (5 + sqrt(5)) / 10, target 1: freq = 4 * 0.2 / 0.2 = 4.
    p = (5.0 + ROOT5) / 10.0
    assert implied_frequency(-1.0, 1.0, p, 1.0) == pytest.approx(4.0, rel=1e-12)


def test_implied_frequency_negative_target_with_negative_mean() -> None:
    # sl = -1, pt = 1, p = 0.3: mean = -0.4, width = 2, so freq = 4 * 0.21 / 0.16 = 5.25.
    assert implied_frequency(-1.0, 1.0, 0.3, -1.0) == pytest.approx(5.25, rel=1e-12)


@pytest.mark.parametrize("p", [0.1, 0.3, 0.55, 0.7, 0.95])
@pytest.mark.parametrize("n", [1.0, 52.0, 260.0])
def test_implied_frequency_round_trip(p: float, n: float) -> None:
    target = binomial_sharpe(p, n, 2.0, -1.0)
    assert implied_frequency(-1.0, 2.0, p, target) == pytest.approx(n, rel=1e-9)


def test_implied_frequency_spurious_root_is_nan() -> None:
    # p = 0.3 has a negative mean bet return. The squared root gives a frequency whose
    # Sharpe is -1, not +1, so no positive frequency gives +1.
    assert math.isnan(implied_frequency(-1.0, 1.0, 0.3, 1.0))


@pytest.mark.parametrize(
    ("p", "target"),
    [
        (0.7, 0.0),  # zero Sharpe needs no bets
        (0.5, 1.0),  # sl = -1, pt = 1: mean bet return is exactly 0 at p = 0.5
    ],
)
def test_implied_frequency_nan_when_no_positive_frequency_exists(p: float, target: float) -> None:
    assert math.isnan(implied_frequency(-1.0, 1.0, p, target))


@pytest.mark.parametrize(
    ("args", "message"),
    [
        ((0.0, 1.0, 0.6, 1.0), "sl must be negative"),
        ((-1.0, 0.0, 0.6, 1.0), "pt must be positive"),
        ((-1.0, 1.0, 0.0, 1.0), "p must be strictly between"),
        ((-1.0, 1.0, 1.0, 1.0), "p must be strictly between"),
        ((-1.0, 1.0, 0.6, float("inf")), "target_sharpe must be finite"),
    ],
)
def test_implied_frequency_refuses_bad_inputs(args: tuple, message: str) -> None:
    with pytest.raises(ValueError, match=message):
        implied_frequency(*args)


# ---------------------------------------------------------------------------
# probability_of_failure (Snippet 15.5, probFailure)
# ---------------------------------------------------------------------------


def _book_example_returns(seed: int = 0) -> np.ndarray:
    return mixture_returns(0.05, -0.1, 0.05, 0.1, 0.75, 2600, seed=seed)


def test_probability_of_failure_book_example_is_in_unit_interval() -> None:
    value = probability_of_failure(_book_example_returns(), 260, 2.0)
    assert 0.0 <= value <= 1.0


def test_probability_of_failure_book_example_seed0_value() -> None:
    # Measured value of the example with seed 0 (see docs/proofs/strategy_risk.md, section 4).
    assert probability_of_failure(_book_example_returns(0), 260, 2.0) == pytest.approx(
        0.4829238935, rel=1e-9
    )


def test_probability_of_failure_matches_bisection_reference() -> None:
    # Hand sample: 3 positive returns and 2 non-positive ones. p = 0.6, rPos = 0.03, rNeg = -0.02.
    returns = np.array([0.04, 0.02, 0.03, -0.01, -0.03])
    freq, target = 52.0, 0.5
    p, r_pos, r_neg = 0.6, 0.03, -0.02
    thresh = _threshold_by_bisection(r_pos, r_neg, freq, target)
    expected = norm.cdf(thresh, loc=p, scale=p * (1 - p))
    assert probability_of_failure(returns, freq, target) == pytest.approx(expected, rel=1e-8)


def test_probability_of_failure_accepts_pandas_series() -> None:
    returns = _book_example_returns(1)
    assert probability_of_failure(pd.Series(returns), 260, 2.0) == probability_of_failure(
        returns, 260, 2.0
    )


@pytest.mark.parametrize(
    ("returns", "message"),
    [
        (np.array([0.01, 0.02]), "at least one value <= 0"),
        (np.array([-0.01, -0.02]), "at least one positive value"),
        (np.array([0.01, 0.0, 0.0]), "every non-positive return is zero"),
        (np.array([0.01, np.nan, -0.01]), "returns must be finite"),
        (np.array([[0.01, 0.02], [-0.01, 0.0]]), "one-dimensional or one column"),
        (np.array([0.01, "x", -0.01], dtype=object), "returns must be real numbers"),
    ],
)
def test_probability_of_failure_refuses_bad_returns(returns: np.ndarray, message: str) -> None:
    with pytest.raises(ValueError, match=message):
        probability_of_failure(returns, 260, 2.0)


def test_probability_of_failure_refuses_bad_freq_and_target() -> None:
    returns = np.array([0.02, -0.01, 0.03, -0.02])
    with pytest.raises(ValueError, match="freq must be positive"):
        probability_of_failure(returns, 0.0, 2.0)
    with pytest.raises(ValueError, match="target_sharpe must be finite"):
        probability_of_failure(returns, 260, float("nan"))


# ---------------------------------------------------------------------------
# mixture_returns (Snippet 15.5, mixGaussians)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("prob1", "expected_count"),
    [(0.0, 0), (0.5, 1300), (0.7, 1820), (0.75, 1950), (1.0, 2600)],
)
def test_mixture_returns_component_counts(prob1: float, expected_count: int) -> None:
    # Zero sigma makes each draw equal to its component mean, so the count can be read back exactly.
    # 2600 * 0.7 is slightly below 1820 in floating point, so the count is rounded, not truncated.
    out = mixture_returns(1.0, -1.0, 0.0, 0.0, prob1, 2600, seed=SEED)
    assert out.shape == (2600,)
    assert int(np.sum(out == 1.0)) == expected_count


def test_mixture_returns_is_reproducible_and_shuffled() -> None:
    a = mixture_returns(0.05, -0.1, 0.05, 0.1, 0.75, 2600, seed=3)
    b = mixture_returns(0.05, -0.1, 0.05, 0.1, 0.75, 2600, seed=3)
    c = mixture_returns(0.05, -0.1, 0.05, 0.1, 0.75, 2600, seed=4)
    assert a.dtype == np.float64
    np.testing.assert_array_equal(a, b)
    assert not np.array_equal(a, c)


def test_mixture_returns_shuffles_components() -> None:
    # Before the shuffle the first 1950 values are all component 1 (mean 1.0), so no -1.0 appears there.
    out = mixture_returns(1.0, -1.0, 0.0, 0.0, 0.75, 2600, seed=SEED)
    assert np.any(out[:1950] == -1.0)


def test_mixture_returns_sample_mean_within_four_standard_errors() -> None:
    # Analytic mean 0.75 * 0.05 + 0.25 * (-0.1) = 0.0125. Variance 0.00859375 from the two components.
    n = 2_000_000
    out = mixture_returns(0.05, -0.1, 0.05, 0.1, 0.75, n, seed=SEED)
    sd = math.sqrt(0.00859375)
    assert float(out.mean()) == pytest.approx(0.0125, abs=4 * sd / math.sqrt(n))


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"sigma1": -0.05}, "sigma1 and sigma2 must be non-negative"),
        ({"sigma2": -0.1}, "sigma1 and sigma2 must be non-negative"),
        ({"prob1": 1.2}, "prob1 must be between 0 and 1"),
        ({"prob1": float("nan")}, "prob1 must be finite"),
        ({"mu1": float("inf")}, "mu1 must be finite"),
        ({"n_obs": 0}, "n_obs must be >= 1"),
        ({"n_obs": 2.5}, "n_obs must be an integer"),
        ({"n_obs": True}, "n_obs must be an integer"),
    ],
)
def test_mixture_returns_refuses_bad_inputs(kwargs: dict, message: str) -> None:
    args = {"mu1": 0.05, "mu2": -0.1, "sigma1": 0.05, "sigma2": 0.1, "prob1": 0.75, "n_obs": 2600}
    args.update(kwargs)
    with pytest.raises(ValueError, match=message):
        mixture_returns(**args)


# ---------------------------------------------------------------------------
# Module surface
# ---------------------------------------------------------------------------


def test_module_all_lists_public_functions() -> None:
    assert set(sr.__all__) == {
        "binomial_sharpe",
        "implied_precision",
        "implied_frequency",
        "probability_of_failure",
        "mixture_returns",
    }
    for name in sr.__all__:
        assert callable(getattr(sr, name))


# ---------------------------------------------------------------------------
# Extreme magnitudes: theta, the hit rate and the frequency depend on the payoffs only through
# their ratio, so a common scale must leave them unchanged and must not overflow or underflow.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("scale", [1e-310, 1e-300, 1e308])
def test_binomial_sharpe_is_invariant_to_common_payoff_scale(scale: float) -> None:
    expected = binomial_sharpe(0.6, 52.0, 1.0, -1.0)
    assert binomial_sharpe(0.6, 52.0, scale, -scale) == pytest.approx(expected, rel=1e-12)


@pytest.mark.parametrize("scale", [1e-310, 1e-300, 1e308])
def test_implied_precision_is_invariant_to_common_payoff_scale(scale: float) -> None:
    expected = implied_precision(-1.0, 1.0, 52.0, 1.0)
    assert implied_precision(-scale, scale, 52.0, 1.0) == pytest.approx(expected, rel=1e-12)


@pytest.mark.parametrize("scale", [1e-310, 1e-300, 1e308])
def test_implied_frequency_is_invariant_to_common_payoff_scale(scale: float) -> None:
    expected = implied_frequency(-1.0, 1.0, 0.6, 1.0)
    assert implied_frequency(-scale, scale, 0.6, 1.0) == pytest.approx(expected, rel=1e-12)


def test_probability_of_failure_is_invariant_to_tiny_return_scale() -> None:
    # Returns near 1e-300 make the threshold's payoff arithmetic underflow.
    tiny = np.array([1e-300, -1e-300, 2e-300, -2e-300])
    assert probability_of_failure(tiny, 52.0, 1.0) == pytest.approx(
        probability_of_failure([1.0, -1.0, 2.0, -2.0], 52.0, 1.0), rel=1e-9
    )
