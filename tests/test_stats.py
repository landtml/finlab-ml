"""Tests for finlab.stats (AFML ch. 14: Sharpe, PSR, expected max SR, DSR, minTRL)."""

from __future__ import annotations

import dataclasses
import math
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


def test_deflated_sharpe_equals_psr_against_expected_max() -> None:
    sr_hat, n, var_sr, t = 0.3, 40, 0.02, 300
    sr_star = expected_max_sharpe(n, var_sr)
    assert deflated_sharpe_ratio(sr_hat, n, var_sr, t, -0.5, 6.0) == pytest.approx(
        probabilistic_sharpe_ratio(sr_hat, sr_star, t, -0.5, 6.0), rel=1e-14
    )


def test_min_track_record_length_hand_computation() -> None:
    # Gaussian returns: denominator term = 1 + (3-1)/4 * 0.5^2 = 1.125.
    # minTRL = 1 + 1.125 * (z_0.95 / 0.5)^2 = 1 + 1.125 * 10.8221... = 13.1749...
    hand = 1.0 + 1.125 * (Z95 / 0.5) ** 2
    assert min_track_record_length(0.5, 0.0, 0.0, 3.0, 0.95) == pytest.approx(
        hand, rel=1e-12
    )
    assert hand == pytest.approx(13.1749455434, abs=1e-9)


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


# ---------------------------------------------------------------------------
# Expected maximum: exact closed forms and an independent numerical reference
#
# Label for the numerical reference checks: independent numerical reference,
# approximation error measured, not a book value.
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
    # Label: independent numerical reference, approximation error measured, not a book value.
    # Measured (exact = quadrature; approx = expected_max_sharpe(N, 1.0)):
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
