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
    deflated_sharpe,
    deflated_sharpe_ratio,
    expected_max_sharpe,
    min_track_record_length,
    probabilistic_sharpe_ratio,
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
