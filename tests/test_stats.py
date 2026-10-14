"""Tests for finlab.stats (AFML ch. 14: Sharpe, PSR, expected max SR, DSR, minTRL)."""

from __future__ import annotations

import dataclasses
import math
import warnings
from collections.abc import Callable

import numpy as np
import pandas as pd
import pytest
from scipy.integrate import quad
from scipy.stats import norm

import finlab.stats as stats_module
from finlab.stats import (
    DeflatedSharpeResult,
    bet_timing,
    deflated_sharpe,
    deflated_sharpe_ratio,
    drawdown_time_under_water,
    expected_max_sharpe,
    hhi_concentration,
    holding_period,
    min_track_record_length,
    probabilistic_sharpe_ratio,
    return_concentration,
    sharpe_ratio,
)

Z95 = 1.6448536269514722  # norm.ppf(0.95), hard-coded for the hand computation


def test_sharpe_ratio_annualisation_matches_manual() -> None:
    rng = np.random.default_rng(0)
    x = rng.normal(0.001, 0.01, size=500)
    manual_per_period = x.mean() / x.std(ddof=1)
    assert sharpe_ratio(x, periods_per_year=252, annualize=False) == pytest.approx(
        manual_per_period, rel=1e-12
    )
    assert sharpe_ratio(x, periods_per_year=252) == pytest.approx(
        manual_per_period * math.sqrt(252), rel=1e-12
    )
    # Series input with NaNs gives the same answer as the cleaned array.
    s = pd.Series(np.concatenate([x, [np.nan]]))
    assert sharpe_ratio(s, periods_per_year=12) == pytest.approx(
        manual_per_period * math.sqrt(12), rel=1e-12
    )


@pytest.mark.parametrize("value", [0.01, -0.003, 1e-6])
def test_sharpe_ratio_constant_returns_give_nan_with_warning(value: float) -> None:
    # Float noise in np.std must not turn a constant series into a huge finite Sharpe.
    with pytest.warns(UserWarning, match="constant"):
        assert math.isnan(sharpe_ratio(np.full(10, value)))


def test_sharpe_ratio_identically_zero_returns_are_zero_without_warning() -> None:
    # An all-zero trial has Sharpe ratio 0 by definition; it is not undefined like a nonzero constant.
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        assert sharpe_ratio(np.zeros(10), annualize=False) == 0.0
    assert not [w for w in caught if issubclass(w.category, UserWarning)]


def test_psr_is_half_when_sr_equals_benchmark() -> None:
    for skew, kurt in [(0.0, 3.0), (-1.2, 6.0), (0.7, 4.5)]:
        for n in (10, 250, 5000):
            assert probabilistic_sharpe_ratio(0.3, 0.3, n, skew, kurt) == pytest.approx(
                0.5, abs=1e-15
            )


def test_psr_matches_closed_form() -> None:
    sr, b, n, g3, g4 = 0.12, 0.05, 120, -0.4, 5.0
    denom = math.sqrt(1 - g3 * sr + (g4 - 1) / 4 * sr * sr)
    expected = norm.cdf((sr - b) * math.sqrt(n - 1) / denom)
    assert probabilistic_sharpe_ratio(sr, b, n, g3, g4) == pytest.approx(
        expected, rel=1e-12
    )


def test_psr_fat_tails_lower_confidence() -> None:
    gauss = probabilistic_sharpe_ratio(0.2, 0.0, 100, 0.0, 3.0)
    fat = probabilistic_sharpe_ratio(0.2, 0.0, 100, 0.0, 9.0)
    assert fat < gauss


def test_expected_max_sharpe_properties() -> None:
    assert expected_max_sharpe(1, 0.5) == 0.0
    assert expected_max_sharpe(10, 0.0) == 0.0
    values = [expected_max_sharpe(n, 1.0) for n in (2, 5, 20, 100, 1000)]
    assert all(b > a for a, b in zip(values, values[1:]))
    # Scales with sqrt(V[SR]).
    assert expected_max_sharpe(50, 4.0) == pytest.approx(
        2 * expected_max_sharpe(50, 1.0), rel=1e-12
    )
    # Below the classical sqrt(2 log N) bound.
    assert expected_max_sharpe(1000, 1.0) < math.sqrt(2 * math.log(1000))


def test_deflated_sharpe_decreases_with_trials() -> None:
    sr_hat, var_sr, n_obs = 0.25, 0.04, 250
    dsr = [
        deflated_sharpe_ratio(sr_hat, n, var_sr, n_obs) for n in (1, 2, 5, 10, 100, 1000)
    ]
    assert all(b < a for a, b in zip(dsr, dsr[1:]))


# closed form, not a book value: DSR equals PSR evaluated at expected_max_sharpe (AFML 14.7.3).
def test_deflated_sharpe_equals_psr_against_expected_max() -> None:
    sr_hat, n, var_sr, t = 0.3, 40, 0.02, 300
    sr_star = expected_max_sharpe(n, var_sr)
    assert deflated_sharpe_ratio(sr_hat, n, var_sr, t, -0.5, 6.0) == pytest.approx(
        probabilistic_sharpe_ratio(sr_hat, sr_star, t, -0.5, 6.0), rel=1e-14
    )


def test_dsr_independent_reimplementation_of_the_formula() -> None:
    """Independent reimplementation of the AFML 14.7.3 formula with scipy primitives. It checks the implementation, not the accuracy of the expected-maximum approximation (see section 6 of the proof note)."""
    gamma = 0.5772156649015329  # Euler-Mascheroni constant, typed here rather than imported

    def dsr_reference(
        sr_hat: float, n: int, var_sr: float, n_obs: int, skew: float, kurt: float
    ) -> float:
        sr_star = math.sqrt(var_sr) * (
            (1 - gamma) * norm.ppf(1 - 1 / n) + gamma * norm.ppf(1 - 1 / (n * math.e))
        )
        denom = math.sqrt(1 - skew * sr_hat + (kurt - 1) / 4 * sr_hat**2)
        return float(norm.cdf((sr_hat - sr_star) * math.sqrt(n_obs - 1) / denom))

    # (N, var_sr, sr_hat, T, skew, kurt)
    cases = [
        (2, 0.01, 0.2, 250, -0.3, 4.5),
        (5, 0.04, 0.5, 120, 0.0, 3.0),
        (50, 0.0025, 0.15, 500, 0.2, 3.8),
    ]
    for n, var_sr, sr_hat, n_obs, skew, kurt in cases:
        expected = dsr_reference(sr_hat, n, var_sr, n_obs, skew, kurt)
        assert deflated_sharpe_ratio(sr_hat, n, var_sr, n_obs, skew, kurt) == pytest.approx(
            expected, rel=1e-12
        )
        assert deflated_sharpe(sr_hat, n, var_sr, n_obs, skew, kurt).dsr == pytest.approx(
            expected, rel=1e-12
        )


def test_min_track_record_length_hand_computation() -> None:
    # Gaussian returns: denominator term = 1 + (3-1)/4 * 0.5^2 = 1.125.
    # minTRL = 1 + 1.125 * (z_0.95 / 0.5)^2 = 1 + 1.125 * 10.8221... = 13.1749...
    hand = 1.0 + 1.125 * (Z95 / 0.5) ** 2
    assert min_track_record_length(0.5, 0.0, 0.0, 3.0, 0.95) == pytest.approx(
        hand, rel=1e-12
    )
    assert hand == pytest.approx(13.1749455434, abs=1e-9)


@pytest.mark.parametrize("bad_benchmark", [float("nan"), float("inf")])
def test_min_track_record_length_rejects_non_finite_benchmark(bad_benchmark: float) -> None:
    # A NaN benchmark fails every comparison, so the call must raise instead of returning NaN.
    with pytest.raises(ValueError, match="sr_benchmark"):
        min_track_record_length(0.5, bad_benchmark, 0.0, 3.0, 0.95)


@pytest.mark.parametrize("bad_sr_hat", [float("nan"), float("inf")])
def test_min_track_record_length_rejects_non_finite_sr_hat(bad_sr_hat: float) -> None:
    with pytest.raises(ValueError, match="sr_hat must be finite"):
        min_track_record_length(bad_sr_hat, 0.0, 0.0, 3.0, 0.95)


@pytest.mark.parametrize("bad_n_obs", [float("nan"), float("inf")])
def test_probabilistic_sharpe_ratio_rejects_non_finite_n_obs(bad_n_obs: float) -> None:
    # NaN must not pass the n_obs < 2 check, and inf must not return 1.0 silently.
    with pytest.raises(ValueError, match="n_obs"):
        probabilistic_sharpe_ratio(0.1, 0.0, bad_n_obs)


@pytest.mark.parametrize("bad_sr_hat", [float("nan"), float("inf")])
def test_probabilistic_sharpe_ratio_rejects_non_finite_sr_hat(bad_sr_hat: float) -> None:
    with pytest.raises(ValueError, match="sr_hat must be finite"):
        probabilistic_sharpe_ratio(bad_sr_hat, 0.0, 100)


def test_min_track_record_length_round_trip_gives_target_psr() -> None:
    n_min = min_track_record_length(0.2, 0.05, -0.3, 4.5, 0.9)
    n_int = math.ceil(n_min)
    assert probabilistic_sharpe_ratio(0.2, 0.05, n_int, -0.3, 4.5) >= 0.9
    assert probabilistic_sharpe_ratio(0.2, 0.05, n_int - 1, -0.3, 4.5) < 0.9


def test_min_track_record_length_rejects_bad_input() -> None:
    with pytest.raises(ValueError):
        min_track_record_length(0.1, 0.1)
    with pytest.raises(ValueError):
        min_track_record_length(0.1, 0.0, prob=1.0)


def test_psr_rejects_short_track_and_bad_moments() -> None:
    with pytest.raises(ValueError):
        probabilistic_sharpe_ratio(0.1, 0.0, 1)
    with pytest.raises(ValueError):
        # 1 - 10*1 + 0 < 0 for skew 10 at SR = 1
        probabilistic_sharpe_ratio(1.0, 0.0, 100, 10.0, 3.0)


# ---------------------------------------------------------------------------
# Result object, exports and input validation
# ---------------------------------------------------------------------------


def test_deflated_sharpe_result_matches_float_api() -> None:
    res = deflated_sharpe(0.3, 40, 0.02, 250, -0.5, 6.0)
    assert isinstance(res, DeflatedSharpeResult)
    assert res.dsr == deflated_sharpe_ratio(0.3, 40, 0.02, 250, -0.5, 6.0)
    assert res.sr_star == expected_max_sharpe(40, 0.02)
    assert (res.sr_hat, res.n_trials, res.n_obs, res.skew, res.kurt) == (0.3, 40, 250, -0.5, 6.0)
    assert res.dsr == probabilistic_sharpe_ratio(0.3, res.sr_star, 250, -0.5, 6.0)


def test_deflated_sharpe_result_is_frozen_and_exported() -> None:
    res = deflated_sharpe(0.3, 40, 0.02, 250)
    with pytest.raises(dataclasses.FrozenInstanceError):
        res.dsr = 0.0  # type: ignore[misc]
    assert {"deflated_sharpe", "DeflatedSharpeResult"} <= set(stats_module.__all__)


_VALID = dict(sr_hat=0.3, n_trials=40, var_sr=0.02, n_obs=250, skew=0.0, kurt=3.0)


@pytest.mark.parametrize(
    "override, match",
    [
        ({"n_trials": 0}, r"n_trials must be >= 1"),
        ({"n_trials": -3}, r"n_trials must be >= 1"),
        ({"n_trials": True}, r"n_trials must be an integer"),
        ({"n_trials": 2.5}, r"n_trials must be an integer"),
        ({"n_trials": float("nan")}, r"n_trials must be an integer"),
        ({"n_trials": "40"}, r"n_trials must be an integer"),
        ({"var_sr": -0.02}, r"var_sr must be non-negative"),
        ({"var_sr": float("nan")}, r"var_sr must be finite"),
        ({"var_sr": float("inf")}, r"var_sr must be finite"),
        ({"n_obs": 1}, r"n_obs must be >= 2"),
        ({"n_obs": False}, r"n_obs must be an integer"),
        ({"n_obs": 250.5}, r"n_obs must be an integer"),
        ({"sr_hat": float("nan")}, r"sr_hat must be finite"),
        ({"sr_hat": float("-inf")}, r"sr_hat must be finite"),
        ({"skew": float("nan")}, r"skew must be finite"),
        ({"kurt": float("inf")}, r"kurt must be finite"),
    ],
)
def test_deflated_sharpe_rejects_invalid_inputs(override: dict, match: str) -> None:
    args = {**_VALID, **override}
    with pytest.raises(ValueError, match=match):
        deflated_sharpe(**args)
    with pytest.raises(ValueError, match=match):
        deflated_sharpe_ratio(**args)


def test_expected_max_sharpe_rejects_invalid_inputs() -> None:
    with pytest.raises(ValueError, match="n_trials must be an integer"):
        expected_max_sharpe(2.5, 1.0)
    with pytest.raises(ValueError, match="n_trials must be >= 1"):
        expected_max_sharpe(0, 1.0)
    with pytest.raises(ValueError, match="var_sr must be finite"):
        expected_max_sharpe(10, float("nan"))
    with pytest.raises(ValueError, match="var_sr must be non-negative"):
        expected_max_sharpe(10, -1.0)


def test_deflated_sharpe_accepts_integral_counts_and_zero_variance() -> None:
    base = deflated_sharpe(0.3, 40, 0.02, 250).dsr
    assert deflated_sharpe(0.3, 40.0, 0.02, 250.0).dsr == base
    assert deflated_sharpe(0.3, np.int64(40), 0.02, np.int64(250)).dsr == base
    # var_sr = 0: all trials share one SR, so SR* = 0 and DSR = PSR(SR_hat, 0).
    zero = deflated_sharpe(0.3, 40, 0.0, 250)
    assert zero.sr_star == 0.0
    assert zero.dsr == pytest.approx(probabilistic_sharpe_ratio(0.3, 0.0, 250), rel=1e-14)


# ---------------------------------------------------------------------------
# Hand-computed cases
# ---------------------------------------------------------------------------


def test_psr_hand_case_gaussian_denominator_sqrt3() -> None:
    # Gaussian moments (skew 0, kurt 3): PSR denominator = sqrt(1 + (3-1)/4 * SR^2)
    #   = sqrt(1 + SR^2/2). With SR_hat = 2 and T = 4: sqrt(1 + 4/2) = sqrt(3), and
    #   sqrt(T - 1) = sqrt(3).
    # Benchmark 0: z = (2 - 0) * sqrt(3) / sqrt(3) = 2, so PSR = Phi(2) = 0.97725
    #   (standard table value 0.9772498680518208).
    # Benchmark 1: z = (2 - 1) * sqrt(3) / sqrt(3) = 1, so PSR = Phi(1) = 0.84134
    #   (standard table value 0.8413447460685429).
    assert probabilistic_sharpe_ratio(2.0, 0.0, 4, 0.0, 3.0) == pytest.approx(
        0.9772498680518208, abs=1e-12
    )
    assert probabilistic_sharpe_ratio(2.0, 1.0, 4, 0.0, 3.0) == pytest.approx(
        0.8413447460685429, abs=1e-12
    )


def test_min_track_record_length_hand_case_gives_four() -> None:
    # Gaussian moments, SR_hat = 2, benchmark 0: D^2 = 1 + 2 = 3 and SR - SR* = 2.
    # prob = Phi(2) gives z = Phi^{-1}(prob) = 2, so
    #   minTRL = 1 + D^2 * (z / (SR - SR*))^2 = 1 + 3 * (2 / 2)^2 = 4.
    # Consistency with the PSR hand case: PSR(T = 4) = Phi(2) = prob, and PSR(T = 3)
    #   has z = 2 * sqrt(2) / sqrt(3) = 1.633, so it is below prob.
    prob = 0.9772498680518208
    assert min_track_record_length(2.0, 0.0, 0.0, 3.0, prob) == pytest.approx(4.0, abs=1e-9)
    assert probabilistic_sharpe_ratio(2.0, 0.0, 4, 0.0, 3.0) == pytest.approx(prob, abs=1e-12)
    assert probabilistic_sharpe_ratio(2.0, 0.0, 3, 0.0, 3.0) < prob


def test_dsr_hand_computed_case_n_equals_one() -> None:
    # N = 1: expected_max_sharpe returns 0 by convention, so SR* = 0 and DSR = PSR against zero.
    # Inputs: SR_hat = 0.2, var_sr = 0.01 (not used when N = 1), T = 101, skew = 0, kurt = 3.
    # Denominator (Gaussian moments): sqrt(1 - 0 + (3 - 1)/4 * 0.2^2) = sqrt(1 + 0.2^2 / 2)
    #   = sqrt(1.02) = 1.0099505.
    # z = (0.2 - 0) * sqrt(101 - 1) / sqrt(1.02) = 0.2 * 10 / 1.0099505 = 1.9802951
    #   (the exact value is 2 / sqrt(1.02) = 1.98029509, rounded to 7 places).
    # DSR = Phi(z) = 0.976165 (six decimals, from scipy).
    dsr = deflated_sharpe_ratio(0.2, 1, 0.01, 101, 0.0, 3.0)
    assert deflated_sharpe(0.2, 1, 0.01, 101, 0.0, 3.0).sr_star == 0.0
    assert dsr == pytest.approx(norm.cdf(2.0 / math.sqrt(1.02)), rel=1e-9)
    assert dsr == pytest.approx(0.976165, abs=1e-6)
    assert dsr == pytest.approx(probabilistic_sharpe_ratio(0.2, 0.0, 101, 0.0, 3.0), rel=1e-14)


# ---------------------------------------------------------------------------
# Expected maximum: exact closed forms and an independent numerical reference
#
# The approximation error is measured against quadrature, not taken from the book.
# ---------------------------------------------------------------------------

_QUAD_BREAKS = [0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0]


def _quad(integrand: Callable[[float], float]) -> float:
    value, _ = quad(
        integrand, -40.0, 40.0, points=_QUAD_BREAKS, limit=500, epsabs=1e-13, epsrel=1e-13
    )
    return float(value)


def _exact_expected_max(n: int) -> float:
    """E[max of n IID N(0,1)] = integral of x * n * phi(x) * Phi(x)^(n-1) dx, by quadrature."""

    def integrand(x: float) -> float:
        return x * n * norm.pdf(x) * norm.cdf(x) ** (n - 1)

    return _quad(integrand)


def _total_mass(n: int) -> float:
    """Integral of the density n * phi * Phi^(n-1); must be 1 (checks the integrator)."""

    def integrand(x: float) -> float:
        return n * norm.pdf(x) * norm.cdf(x) ** (n - 1)

    return _quad(integrand)


def test_numerical_reference_integrates_a_density() -> None:
    for n in (1, 2, 5, 10, 50, 100, 1000):
        assert _total_mass(n) == pytest.approx(1.0, abs=1e-9)


def test_expected_max_n1_convention_agrees_with_exact_value() -> None:
    # expected_max_sharpe returns 0 for N = 1 by convention (the formula needs Phi^{-1}(0)).
    # The exact value for one standard normal is E[Z] = 0, so the convention is also exact.
    assert expected_max_sharpe(1, 0.5) == 0.0
    assert _exact_expected_max(1) == pytest.approx(0.0, abs=1e-12)


def test_expected_max_n2_closed_form_for_exact_maximum_approximation_error_recorded() -> None:
    # Closed form for the exact maximum: max(a, b) = (a + b)/2 + |a - b|/2 and, for
    # Z1, Z2 IID N(0,1), E|Z1 - Z2| = sqrt(2) * sqrt(2/pi) = 2/sqrt(pi). So
    # E[max of 2 N(0,1)] = 1/sqrt(pi) = 0.5641895835.
    exact = 1.0 / math.sqrt(math.pi)
    assert _exact_expected_max(2) == pytest.approx(exact, abs=1e-10)
    # Recorded values for the implementation (measured; the approximation is not exact):
    approx = expected_max_sharpe(2, 1.0)
    assert approx == pytest.approx(0.5197553443, abs=1e-9)
    assert approx - exact == pytest.approx(-0.0444342393, abs=1e-9)
    assert (approx - exact) / exact == pytest.approx(-0.078758, abs=1e-6)
    # Scaling by sqrt(V): the exact maximum is sqrt(V)/sqrt(pi), and the code scales the same way.
    assert expected_max_sharpe(2, 4.0) == pytest.approx(2 * approx, rel=1e-12)


def test_expected_max_matches_independent_numerical_reference() -> None:
    # Exact = quadrature; approx = expected_max_sharpe(N, 1.0). Measured values:
    #   N     exact         approx        approx - exact   relative
    #   2     0.5641895835  0.5197553443  -0.0444342393    -7.876 %
    #   5     1.1629644736  1.1925940010  +0.0296295274    +2.548 %
    #   10    1.5387527308  1.5745983013  +0.0358455705    +2.3295 %
    #   50    2.2490736294  2.2763030934  +0.0272294640    +1.211 %
    #   100   2.5075936364  2.5306028932  +0.0230092568    +0.918 %
    #   1000  3.2414357691  3.2551215137  +0.0136857445    +0.422 %
    # The bounds are the measured |relative error| plus a small margin.
    bounds = {2: 0.080, 5: 0.026, 10: 0.024, 50: 0.0125, 100: 0.0095, 1000: 0.0045}
    for n, bound in bounds.items():
        exact = _exact_expected_max(n)
        approx = expected_max_sharpe(n, 1.0)
        assert abs((approx - exact) / exact) < bound, n
    # Sign: the approximation is below the exact maximum at N = 2 and above it for N >= 5.
    assert expected_max_sharpe(2, 1.0) < _exact_expected_max(2)
    for n in (5, 10, 50, 100, 1000):
        assert expected_max_sharpe(n, 1.0) > _exact_expected_max(n)


@pytest.mark.parametrize("returns", [[], [0.003547]], ids=["empty", "single"])
def test_sharpe_ratio_warns_on_fewer_than_two_observations(returns: list[float]) -> None:
    with pytest.warns(UserWarning, match="sharpe_ratio.*fewer than two"):
        assert math.isnan(sharpe_ratio(returns, periods_per_year=252.0))


def test_psr_warns_on_infinite_benchmark_and_returns_its_limit() -> None:
    # An infinite sr_benchmark gives z = -inf, so PSR is 0 with a warning. An infinite sr_hat raises.
    with pytest.warns(UserWarning, match="probabilistic_sharpe_ratio.*non-finite"):
        assert probabilistic_sharpe_ratio(0.5, np.inf, 60, -0.2, 3.5) == 0.0


def test_sharpe_ratio_rejects_multi_column_ndarray() -> None:
    # A 2-D array was flattened, pooling every column into one series.
    a = np.array([[1.0, 100], [2, 200], [3, 300], [1.5, 150]])
    with pytest.raises(ValueError, match="one column"):
        sharpe_ratio(a, annualize=False)


def test_sharpe_ratio_single_column_and_row_vector_unchanged() -> None:
    r = np.array([1.0, 2.0, 3.0, 1.5])
    want = sharpe_ratio(r, annualize=False)
    assert sharpe_ratio(r.reshape(-1, 1), annualize=False) == pytest.approx(want)
    # A row vector is several columns, so it is refused like any other multi-column input.
    # Accepting it would pool the columns into one series.
    with pytest.raises(ValueError, match="one column"):
        sharpe_ratio(r.reshape(1, -1), annualize=False)


@pytest.mark.parametrize("kurt", [float("inf"), float("nan")])
def test_psr_rejects_non_finite_kurtosis(kurt: float) -> None:
    # Infinite kurtosis gave exactly 0.5 ("no skill") with no warning.
    from finlab.stats import probabilistic_sharpe_ratio

    with pytest.raises(ValueError, match="kurtosis must be finite"):
        probabilistic_sharpe_ratio(1.0, 0.0, 100, 0.0, kurt)


def test_psr_rejects_non_finite_skew() -> None:
    from finlab.stats import probabilistic_sharpe_ratio

    with pytest.raises(ValueError, match="skew must be finite"):
        probabilistic_sharpe_ratio(1.0, 0.0, 100, float("inf"), 3.0)


def test_psr_finite_skew_kurtosis_unchanged() -> None:
    from finlab.stats import probabilistic_sharpe_ratio

    assert probabilistic_sharpe_ratio(1.0, 0.0, 100, 0.0, 3.0) == pytest.approx(
        0.9999999999999998, rel=1e-12
    )


def test_sharpe_ratio_tiny_scale_returns_are_finite_and_scale_invariant() -> None:
    # An absolute 1e-12 floor treated returns of scale 1e-13 as constant.
    base = np.array([1.0, 2.0, 3.0, 1.5])
    tiny = base * 1e-13
    assert np.isfinite(sharpe_ratio(tiny, annualize=False))
    assert sharpe_ratio(tiny, annualize=False) == pytest.approx(
        sharpe_ratio(base, annualize=False), rel=1e-12
    )


def test_sharpe_ratio_still_flags_float_noise_as_constant_at_unit_scale() -> None:
    with pytest.warns(UserWarning, match="constant"):
        assert math.isnan(sharpe_ratio(np.array([1.0, 1.0 + 1e-13, 1.0, 1.0])))


def test_sharpe_ratio_rejects_single_row_multi_column_ndarray() -> None:
    # A 1x3 array was raveled into three observations of one series.
    with pytest.raises(ValueError, match="one column"):
        sharpe_ratio(np.array([[0.01, 0.02, 0.03]]))


def test_min_track_record_length_rejects_non_finite_skew_with_clear_message() -> None:
    # A NaN skew must be named as the cause, not reported as "skew/kurtosis incompatible".
    from finlab.stats import min_track_record_length

    with pytest.raises(ValueError, match="skew must be finite"):
        min_track_record_length(0.2, 0.1, skew=float("nan"))


def test_min_track_record_length_rejects_non_finite_kurtosis_with_clear_message() -> None:
    from finlab.stats import min_track_record_length

    with pytest.raises(ValueError, match="kurt must be finite"):
        min_track_record_length(0.2, 0.1, kurt=float("inf"))


def test_sharpe_ratio_rejects_multi_column_dataframe() -> None:
    # A multi-column DataFrame was reduced to its first column with no warning, while the same data
    # as an ndarray is refused. Both containers now agree.
    x = np.random.default_rng(0).normal(0.001, 0.01, 50)
    with pytest.raises(ValueError, match="one column"):
        sharpe_ratio(pd.DataFrame(np.column_stack([x, -x, x])), annualize=False)
    with pytest.raises(ValueError, match="one column"):
        sharpe_ratio(pd.DataFrame([[0.1, 0.2, 0.3]]), annualize=False)


def test_sharpe_ratio_one_column_dataframe_unchanged() -> None:
    x = np.random.default_rng(0).normal(0.001, 0.01, 50)
    assert sharpe_ratio(pd.DataFrame({"a": x}), annualize=False) == pytest.approx(
        sharpe_ratio(x, annualize=False)
    )


def test_sharpe_ratio_unused_periods_per_year_ignored_when_not_annualized() -> None:
    # periods_per_year was validated even when annualize=False, where it is not used, so a per-period
    # call with NaN or inf raised. The value is only checked when used.
    x = np.random.default_rng(0).normal(0.001, 0.01, 50)
    want = sharpe_ratio(x, annualize=False)
    assert sharpe_ratio(x, periods_per_year=float("nan"), annualize=False) == pytest.approx(want)
    assert sharpe_ratio(x, periods_per_year=float("inf"), annualize=False) == pytest.approx(want)
    with pytest.raises(ValueError, match="periods_per_year"):
        sharpe_ratio(x, periods_per_year=float("nan"), annualize=True)


def test_psr_rejects_nan_benchmark() -> None:
    # A NaN sr_benchmark returned NaN with a warning, while minTRL refuses it. An infinite benchmark
    # stays a documented value (0.0).
    from finlab.stats import probabilistic_sharpe_ratio

    with pytest.raises(ValueError, match="sr_benchmark must not be NaN"):
        probabilistic_sharpe_ratio(0.1, float("nan"), 100)


def test_probabilistic_sharpe_accepts_zero_dim_array_skew() -> None:
    # A 0-d array holds one real number, so it is accepted (0.8395...).
    import numpy as np

    from finlab.stats import min_track_record_length, probabilistic_sharpe_ratio

    assert probabilistic_sharpe_ratio(0.1, 0.0, 100, np.array(0.0), 3.0) == pytest.approx(
        0.8395254171844497, rel=1e-12
    )
    assert min_track_record_length(0.2, 0.1, np.array(0.0), np.array(3.0)) == pytest.approx(
        276.96543231773217, rel=1e-12
    )
    with pytest.raises(ValueError, match="skew must be a real number"):
        probabilistic_sharpe_ratio(0.1, 0.0, 100, np.array([0.0, 0.1]), 3.0)


def test_sharpe_ratio_refuses_strings_bools_and_none() -> None:
    # Strings became floats, bools became 1/0 and None became NaN.
    from finlab.stats import sharpe_ratio

    with pytest.raises(ValueError, match="returns must be real numbers"):
        sharpe_ratio(["0.01", "0.02", "-0.01"])
    with pytest.raises(ValueError, match="returns must be real numbers"):
        sharpe_ratio([True, False, True])
    with pytest.raises(ValueError, match="returns must be real numbers"):
        sharpe_ratio([None, 0.01, 0.02, -0.01])


def test_sharpe_ratio_refuses_none_entries_as_intended() -> None:
    # A None entry is refused by name in a list. A pandas Series converts None to NaN before the
    # function sees it, so only the list form is refused here.
    from finlab.stats import sharpe_ratio

    with pytest.raises(ValueError, match="real numbers"):
        sharpe_ratio([0.01, None, -0.02, 0.03])


def test_min_track_record_length_refuses_prob_at_or_below_half() -> None:
    # For sr_hat > sr_benchmark, PSR(T) is above 0.5 for every T, so prob <= 0.5 has no solution. The
    # function used to return a T that solves PSR = 1 - prob (0.95 at T = 273 for prob 0.05).
    from finlab.stats import min_track_record_length, probabilistic_sharpe_ratio

    with pytest.raises(ValueError, match="prob"):
        min_track_record_length(0.1, 0.0, prob=0.05)
    # The refusal applies to prob < 0.5 only (prob = 0.5 has the root T = 1, tested below), so 0.4 is refused.
    with pytest.raises(ValueError, match="prob"):
        min_track_record_length(0.1, 0.0, prob=0.4)
    # Prob above 0.5 still solves PSR = prob at the returned length.
    T = min_track_record_length(0.1, 0.0, prob=0.95)
    assert probabilistic_sharpe_ratio(0.1, 0.0, math.ceil(T), 0.0, 3.0) == pytest.approx(0.95, abs=0.01)


def test_min_track_record_length_prob_half_gives_the_root() -> None:
    # At prob = 0.5 the root is T = 1 (z = 0), which is returned. The refusal is for prob < 0.5
    # only, where PSR(T) stays above 0.5 for every T.
    from finlab.stats import min_track_record_length

    assert min_track_record_length(0.1, 0.0, prob=0.5) == pytest.approx(1.0)
    with pytest.raises(ValueError, match="prob"):
        min_track_record_length(0.1, 0.0, prob=0.4)


# ---------------------------------------------------------------------------
# Bets, holding period, concentration and drawdown (AFML Snippets 14.1 to 14.4)
# ---------------------------------------------------------------------------


def _daily(values: list[float] | np.ndarray) -> pd.Series:
    values = np.asarray(values, dtype=float)
    return pd.Series(values, index=pd.date_range("2024-01-01", periods=values.size, freq="D"))


def _bet_timing_reference(values: list[float]) -> list[int]:
    # The bet-end rule, one row at a time. A flat row ends a bet when the row before it was not
    # flat; the first row counts as not flat. A sign flip ends a bet. The last row is always included.
    n = len(values)
    ends: set[int] = set()
    for t in range(n):
        was_open = True if t == 0 else values[t - 1] != 0
        if values[t] == 0 and was_open:
            ends.add(t)
        if t > 0 and values[t] * values[t - 1] < 0:
            ends.add(t)
    ends.add(n - 1)
    return sorted(ends)


def _holding_period_reference(positions: list[float], days: list[float]) -> float:
    # Snippet 14.2 transcribed row by row in pure Python. The numba kernel is compared with it.
    closed: list[tuple[float, float]] = []
    t_entry = 0.0
    for i in range(1, len(positions)):
        p_diff = positions[i] - positions[i - 1]
        if p_diff * positions[i - 1] >= 0:
            if positions[i] != 0:
                t_entry = (t_entry * positions[i - 1] + days[i] * p_diff) / positions[i]
        elif positions[i] * positions[i - 1] < 0:
            closed.append((days[i] - t_entry, abs(positions[i - 1])))
            t_entry = days[i]
        else:
            closed.append((days[i] - t_entry, abs(p_diff)))
    total = sum(w for _, w in closed)
    return sum(dt * w for dt, w in closed) / total if total > 0 else math.nan


def _drawdown_reference(pnl: list[float]) -> tuple[list[int], list[float], list[float]]:
    # One scan of the path. A high-water mark (HWM) starts at a row where pnl rises above every earlier
    # value. Its drawdown is the lowest pnl before pnl rises above the HWM again. Daily rows, so a row
    # difference is a number of days.
    marks: list[tuple[int, float, float]] = []
    row, hwm, low = 0, pnl[0], pnl[0]
    for t in range(1, len(pnl)):
        if pnl[t] > hwm:
            marks.append((row, hwm, low))
            row, hwm, low = t, pnl[t], pnl[t]
        else:
            low = min(low, pnl[t])
    marks.append((row, hwm, low))
    kept = [(r, h, lo) for r, h, lo in marks if h > lo]
    rows = [r for r, _, _ in kept]
    dd = [1.0 - lo / h for _, h, lo in kept]
    tuw = [(rows[k + 1] - rows[k]) / 365.25 for k in range(len(rows) - 1)]
    return rows, dd, tuw


def test_bet_timing_flattening_flip_and_trailing_open_position() -> None:
    # Rows 0 to 6: long, long, flip to short, short, flat, long, long. The last position is still open.
    positions = _daily([1, 1, -1, -1, 0, 1, 1])
    bets = bet_timing(positions)
    assert isinstance(bets, pd.DatetimeIndex)
    assert list(bets) == [positions.index[2], positions.index[4], positions.index[6]]


def test_bet_timing_flat_first_row_is_a_bet_end() -> None:
    # The NaN that shift(1) leaves in front of row 0 compares unequal to zero, as in the book code.
    positions = _daily([0, 1, 1, 0])
    assert list(bet_timing(positions)) == [positions.index[0], positions.index[3]]


def test_bet_timing_reports_a_flat_last_row_even_when_it_ends_no_bet() -> None:
    # Row 1 is the flattening. Row 2 is flat but did not end a bet; it is included as the last timestamp.
    positions = _daily([1, 0, 0])
    assert list(bet_timing(positions)) == [positions.index[1], positions.index[2]]


def test_bet_timing_single_open_row_is_its_own_bet_end() -> None:
    positions = _daily([1])
    assert list(bet_timing(positions)) == [positions.index[0]]


def test_bet_timing_matches_row_by_row_rule_on_random_positions() -> None:
    rng = np.random.default_rng(14)
    for _ in range(20):
        raw = rng.integers(-2, 3, size=60).astype(float)
        positions = _daily(raw)
        rows = positions.index.get_indexer(bet_timing(positions)).tolist()
        assert rows == _bet_timing_reference(raw.tolist())


def test_bet_timing_refuses_bad_input() -> None:
    with pytest.raises(TypeError, match="positions must be a pandas Series"):
        bet_timing(np.array([1.0, 0.0]))
    with pytest.raises(TypeError, match="DatetimeIndex"):
        bet_timing(pd.Series([1.0, 0.0]))
    with pytest.raises(ValueError, match="at least one observation"):
        bet_timing(_daily([]))
    with pytest.raises(ValueError, match="must not contain NaN"):
        bet_timing(_daily([1.0, np.nan, 0.0]))
    # pandas 2.0 gives a string Series dtype object, which is refused by the element check
    # ("must be real numbers"); pandas 3 has a str dtype, refused by the dtype check ("must hold").
    with pytest.raises(ValueError, match="real numbers"):
        bet_timing(pd.Series(["1", "0"], index=pd.date_range("2024-01-01", periods=2)))
    with pytest.raises(ValueError, match="must be real numbers"):
        bet_timing(pd.Series(["1", "0"], index=pd.date_range("2024-01-01", periods=2), dtype=object))
    with pytest.raises(ValueError, match="must hold real numbers"):
        bet_timing(pd.Series([True, False], index=pd.date_range("2024-01-01", periods=2)))
    with pytest.raises(ValueError, match="strictly increasing"):
        bet_timing(pd.Series([1.0, 0.0], index=pd.DatetimeIndex(["2024-01-02", "2024-01-01"])))
    with pytest.raises(ValueError, match="strictly increasing"):
        bet_timing(pd.Series([1.0, 0.0], index=pd.DatetimeIndex(["2024-01-01", "2024-01-01"])))


def test_bet_timing_sees_flips_between_tiny_positions() -> None:
    # The product of two opposite positions of 1e-200 underflows to -0.0, which is not < 0.
    # A flip must be read from the signs, so the flips on rows 1 and 3 are both reported.
    got = bet_timing(_daily([1e-200, -1e-200, -1e-200, 1e-200]))
    assert list(got) == list(pd.date_range("2024-01-01", periods=4, freq="D")[[1, 3]])


@pytest.mark.parametrize("scale", [1e-200, 1e-300])
def test_bet_timing_is_unchanged_by_scaling_positions(scale: float) -> None:
    rng = np.random.default_rng(11)
    for _ in range(20):
        values = rng.integers(-2, 3, 25).astype(float)
        assert bet_timing(_daily(values * scale)).equals(bet_timing(_daily(values)))


@pytest.mark.parametrize(
    ("values", "expected"),
    [([1.0, 0.5, 0.0, 0.0], 1.5), ([1.0, -1.0, 0.0, 0.0], 1.0)],
)
@pytest.mark.parametrize("scale", [1.0, 1e-200, 1e-300])
def test_holding_period_closes_tiny_positions(values: list[float], expected: float, scale: float) -> None:
    # Holding period does not depend on the unit of size. The tiny products used to read as an
    # increase, so the closing row was dropped and the result was NaN.
    assert holding_period(_daily(np.array(values) * scale)) == pytest.approx(expected, rel=1e-12)


@pytest.mark.parametrize("scale", [1e-200, 1e-300])
def test_holding_period_is_unchanged_by_scaling_random_positions(scale: float) -> None:
    rng = np.random.default_rng(12)
    for _ in range(20):
        values = rng.integers(-2, 3, 25).astype(float)
        assert holding_period(_daily(values * scale)) == pytest.approx(
            holding_period(_daily(values)), rel=1e-12, nan_ok=True
        )


def test_holding_period_one_long_closed_after_three_days() -> None:
    # Opened on day 0 with size 1 and closed on day 3.
    assert holding_period(_daily([1, 1, 1, 0])) == pytest.approx(3.0, rel=1e-12)


def test_holding_period_scale_in_weights_the_entry_by_size() -> None:
    # Day 0: one unit. Day 1: a second unit, so the entry time is (0 * 1 + 1 * 1) / 2 = 0.5.
    # Day 2: both units close, each held 2 - 0.5 = 1.5 days on average.
    assert holding_period(_daily([1, 2, 0])) == pytest.approx(1.5, rel=1e-12)


def test_holding_period_partial_close_weights_each_closing_step_by_size() -> None:
    # Day 0: two units. Day 1: one closes after 1 day. Day 3: the other closes after 3 days. Mean 2.
    positions = pd.Series(
        [2.0, 1.0, 0.0], index=pd.DatetimeIndex(["2024-01-01", "2024-01-02", "2024-01-04"])
    )
    assert holding_period(positions) == pytest.approx(2.0, rel=1e-12)


def test_holding_period_flip_closes_the_old_side_and_restarts_the_entry() -> None:
    # Long from day 0 closes on the flip at day 2 (2 days). The short opened at day 2 closes on day 5 (3 days).
    positions = pd.Series(
        [1.0, -1.0, -1.0, 0.0],
        index=pd.DatetimeIndex(["2024-01-01", "2024-01-03", "2024-01-05", "2024-01-06"]),
    )
    assert holding_period(positions) == pytest.approx(2.5, rel=1e-12)


def test_holding_period_uses_fractional_days() -> None:
    positions = pd.Series([1.0, 0.0], index=pd.DatetimeIndex(["2024-01-01 00:00", "2024-01-02 12:00"]))
    assert holding_period(positions) == pytest.approx(1.5, rel=1e-12)


def test_holding_period_is_nan_when_nothing_is_closed() -> None:
    assert math.isnan(holding_period(_daily([1, 1, 1])))
    assert math.isnan(holding_period(_daily([0])))


def test_holding_period_refuses_bad_input() -> None:
    with pytest.raises(TypeError, match="positions must be a pandas Series"):
        holding_period([1.0, 0.0])
    with pytest.raises(ValueError, match="must not contain NaN"):
        holding_period(_daily([1.0, np.nan, 0.0]))
    with pytest.raises(ValueError, match="at least one observation"):
        holding_period(_daily([]))


def test_holding_period_numba_kernel_matches_python_transcription() -> None:
    rng = np.random.default_rng(2)
    for trial in range(30):
        n = int(rng.integers(2, 120))
        raw = rng.integers(-3, 4, size=n).astype(float) if trial % 2 else rng.normal(0.0, 1.0, n)
        index = pd.Timestamp("2024-01-01") + pd.to_timedelta(
            np.concatenate([[0.0], np.cumsum(rng.uniform(0.1, 3.0, n - 1))]), unit="D"
        )
        days = ((index - index[0]) / pd.Timedelta(days=1)).tolist()
        expected = _holding_period_reference(raw.tolist(), days)
        got = holding_period(pd.Series(raw, index=index))
        if math.isnan(expected):
            assert math.isnan(got)
        else:
            assert got == pytest.approx(expected, rel=1e-12)


def test_hhi_equal_returns_give_zero() -> None:
    assert hhi_concentration([0.01] * 5) == pytest.approx(0.0, abs=1e-12)


def test_hhi_single_nonzero_return_gives_one() -> None:
    assert hhi_concentration([0.0, 0.0, 0.0, 0.5]) == pytest.approx(1.0, rel=1e-12)


def test_hhi_two_or_fewer_returns_is_nan() -> None:
    assert math.isnan(hhi_concentration([0.1, 0.2]))
    assert math.isnan(hhi_concentration([0.1]))


def test_hhi_hand_computed_three_returns() -> None:
    # Weights (1/6, 1/3, 1/2) give sum w^2 = 7/18, so HHI = (7/18 - 1/3) / (2/3) = 1/12.
    assert hhi_concentration([1.0, 2.0, 3.0]) == pytest.approx(1 / 12, rel=1e-12)


def test_hhi_is_scale_invariant() -> None:
    r = np.array([0.02, -0.01, 0.0, 0.03, 0.01])
    assert hhi_concentration(10.0 * r) == pytest.approx(hhi_concentration(r), rel=1e-12)


def test_hhi_equal_huge_returns_give_zero() -> None:
    # The sum of three 1e308 overflows to inf. The HHI is scale-free, so it must not be refused.
    assert hhi_concentration([1e308, 1e308, 1e308]) == pytest.approx(0.0, abs=1e-12)


def test_hhi_refuses_a_scalar_by_name() -> None:
    # A scalar is not a return series; it must not come back as NaN.
    with pytest.raises(ValueError, match="returns"):
        hhi_concentration(0.02)


def test_hhi_single_signed_returns_stay_in_unit_interval() -> None:
    # With one sign the weights lie in [0, 1], so sum w^2 <= sum w = 1 and HHI <= 1.
    rng = np.random.default_rng(5)
    for _ in range(50):
        r = rng.uniform(0.001, 1.0, int(rng.integers(3, 40)))
        for sign in (1.0, -1.0):
            assert -1e-12 <= hhi_concentration(sign * r) <= 1.0 + 1e-12


def test_hhi_mixed_sign_can_exceed_one() -> None:
    # Weights (2, -1, 0) sum to one with sum w^2 = 5, so HHI = (5 - 1/3) / (2/3) = 7.
    assert hhi_concentration([2.0, -1.0, 0.0]) == pytest.approx(7.0, rel=1e-12)


def test_hhi_refuses_zero_sum_and_missing_values_by_name() -> None:
    with pytest.raises(ValueError, match="returns sum to zero"):
        hhi_concentration([1.0, -1.0, 2.0, -2.0])
    with pytest.raises(ValueError, match="returns must not contain NaN"):
        hhi_concentration([0.1, np.nan, 0.2, 0.3])
    with pytest.raises(ValueError, match="returns must be finite"):
        hhi_concentration([0.1, np.inf, 0.2])


def test_return_concentration_hand_computed() -> None:
    # Six returns: two in January, one in February, three in March.
    dates = pd.DatetimeIndex(
        ["2024-01-05", "2024-01-20", "2024-02-10", "2024-03-01", "2024-03-15", "2024-03-30"]
    )
    res = return_concentration(pd.Series([0.1, 0.2, -0.1, 0.3, -0.1, -0.2], index=dates))
    # Positive (0.1, 0.2, 0.3): 1/12. Negative (0.1, 0.1, 0.2), weights (1/4, 1/4, 1/2): 1/16.
    # Counts per month (2, 1, 3): 1/12.
    assert res.positive == pytest.approx(1 / 12, rel=1e-12)
    assert res.negative == pytest.approx(1 / 16, rel=1e-12)
    assert res.time == pytest.approx(1 / 12, rel=1e-12)


def test_return_concentration_periods_without_returns_count_as_zero() -> None:
    # Counts per month (Jan 2, Feb 0, Mar 1): weights (2/3, 0, 1/3), HHI = (5/9 - 1/3) / (2/3) = 1/3.
    dates = pd.DatetimeIndex(["2024-01-05", "2024-01-20", "2024-03-03"])
    assert return_concentration(pd.Series([0.1, -0.1, 0.2], index=dates)).time == pytest.approx(
        1 / 3, rel=1e-12
    )


def test_return_concentration_all_zero_side_is_nan() -> None:
    # No gains at all: the positive sub-series is all zeros, so its HHI is undefined and reported as NaN.
    res = return_concentration(_daily([0, 0, 0, -0.1, -0.2, -0.3]))
    assert math.isnan(res.positive)
    assert res.negative == pytest.approx(1 / 12, rel=1e-12)
    assert math.isnan(res.time)


def test_return_concentration_equal_huge_gains_give_zero_not_nan() -> None:
    # Three equal gains of 1e308 sum to inf. Equal gains have HHI 0, so NaN would be wrong.
    res = return_concentration(_daily([1e308, 1e308, 1e308]))
    assert res.positive == pytest.approx(0.0, abs=1e-12)


def test_return_concentration_empty_series_is_all_nan() -> None:
    res = return_concentration(pd.Series([], index=pd.DatetimeIndex([]), dtype=float))
    assert all(math.isnan(v) for v in (res.positive, res.negative, res.time))


def test_return_concentration_result_is_frozen() -> None:
    res = return_concentration(_daily([0.1, -0.1, 0.2, 0.3]))
    with pytest.raises(dataclasses.FrozenInstanceError):
        res.time = 0.0  # type: ignore[misc]


def test_return_concentration_refuses_bad_input() -> None:
    with pytest.raises(TypeError, match="returns must be a pandas Series"):
        return_concentration(np.array([0.1, 0.2]))
    with pytest.raises(TypeError, match="DatetimeIndex"):
        return_concentration(pd.Series([0.1, -0.1, 0.2]))
    with pytest.raises(ValueError, match="returns must not contain NaN"):
        return_concentration(_daily([0.1, np.nan, 0.2]))


def test_drawdown_hand_built_path_with_two_drawdowns() -> None:
    # pnl by day: 1, 2, 1, 3, 2, 3, 4, 2, 5. Reported HWMs: day 1 (2, low 1), day 3 (3, low 2), day 6 (4, low 2).
    # Day 0 (value 1) and day 8 (value 5) have no drawdown and are not reported.
    pnl = _daily([1, 2, 1, 3, 2, 3, 4, 2, 5])
    dd, tuw = drawdown_time_under_water(pnl)
    assert dd.tolist() == pytest.approx([0.5, 1 / 3, 0.5], rel=1e-12)
    assert list(dd.index) == [pnl.index[1], pnl.index[3], pnl.index[6]]
    assert tuw.tolist() == pytest.approx([2 / 365.25, 3 / 365.25], rel=1e-12)
    assert list(tuw.index) == [pnl.index[1], pnl.index[3]]


def test_drawdown_in_dollars_is_hwm_minus_low() -> None:
    dd, _ = drawdown_time_under_water(_daily([1, 2, 1, 3, 2, 3, 4, 2, 5]), dollars=True)
    assert dd.tolist() == pytest.approx([1.0, 1.0, 2.0], rel=1e-12)


def test_drawdown_open_last_drawdown_has_no_time_under_water_entry() -> None:
    # The path ends below the HWM of 3 set on day 3, so that drawdown is still open.
    pnl = _daily([1, 2, 1, 3, 2])
    dd, tuw = drawdown_time_under_water(pnl)
    assert dd.tolist() == pytest.approx([0.5, 1 / 3], rel=1e-12)
    assert list(tuw.index) == [pnl.index[1]]
    assert tuw.tolist() == pytest.approx([2 / 365.25], rel=1e-12)


def test_drawdown_time_under_water_runs_to_next_drawdown_high_not_next_high() -> None:
    # pnl by day: 0, 1, 0.5, 2, 3, 2, 4. The HWM of 1 (day 1) recovers on day 3, when pnl is 2.
    # The HWM of 2 (day 3) has no drawdown and is skipped, so the interval runs to day 4 (HWM 3): 3 days.
    pnl = _daily([0, 1, 0.5, 2, 3, 2, 4])
    dd, tuw = drawdown_time_under_water(pnl, dollars=True)
    assert dd.tolist() == pytest.approx([0.5, 1.0], rel=1e-12)
    assert tuw.tolist() == pytest.approx([3 / 365.25], rel=1e-12)
    assert list(tuw.index) == [pnl.index[1]]


def test_drawdown_fraction_needs_strictly_positive_pnl() -> None:
    with pytest.raises(ValueError, match="pnl must be strictly positive"):
        drawdown_time_under_water(_daily([1.0, 0.0, 2.0]))
    # Dollar drawdowns accept zero and negative pnl: the HWM 1 has low -1, so its drawdown is 2.
    dd, _ = drawdown_time_under_water(_daily([1.0, -1.0, 2.0]), dollars=True)
    assert dd.tolist() == pytest.approx([2.0], rel=1e-12)


def test_drawdown_single_observation_gives_empty_series() -> None:
    dd, tuw = drawdown_time_under_water(_daily([1.0]))
    assert len(dd) == 0 and len(tuw) == 0


def test_drawdown_refuses_bad_input() -> None:
    with pytest.raises(TypeError, match="pnl must be a pandas Series"):
        drawdown_time_under_water(np.array([1.0, 2.0]))
    with pytest.raises(TypeError, match="DatetimeIndex"):
        drawdown_time_under_water(pd.Series([1.0, 2.0]))
    with pytest.raises(ValueError, match="strictly increasing"):
        drawdown_time_under_water(
            pd.Series([1.0, 2.0], index=pd.DatetimeIndex(["2024-01-02", "2024-01-01"]))
        )
    with pytest.raises(ValueError, match="must not contain NaN"):
        drawdown_time_under_water(_daily([1.0, np.nan]))
    with pytest.raises(TypeError, match="dollars must be a bool"):
        drawdown_time_under_water(_daily([1.0, 2.0]), dollars=1)


def test_drawdown_matches_row_by_row_scan_on_random_equity_curves() -> None:
    rng = np.random.default_rng(4)
    for _ in range(10):
        n = int(rng.integers(2, 200))
        pnl = _daily(np.exp(np.cumsum(rng.normal(0.0, 0.03, n))))
        rows, dd_ref, tuw_ref = _drawdown_reference(pnl.tolist())
        dd, tuw = drawdown_time_under_water(pnl)
        assert list(dd.index) == list(pnl.index[np.asarray(rows, dtype=int)])
        assert dd.tolist() == pytest.approx(dd_ref, rel=1e-12)
        assert tuw.tolist() == pytest.approx(tuw_ref, rel=1e-12)


def test_stats_all_exports_resolve() -> None:
    for name in stats_module.__all__:
        assert hasattr(stats_module, name), name
