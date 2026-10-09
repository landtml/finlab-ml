"""Tests for finlab.stats (AFML ch. 14: Sharpe, PSR, expected max SR, DSR, minTRL)."""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest
from scipy.stats import norm

from finlab.stats import (
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
    assert hand == pytest.approx(13.174943, abs=1e-5)


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
