"""Tests for finlab.bet_sizing (AFML ch. 10). Book worked examples are included."""
import math

import numpy as np
import pandas as pd
import pytest

from finlab.bet_sizing import (
    average_active_signals,
    calibrate_sigmoid_width,
    discretize_signal,
    limit_price,
    prob_bet_size,
    sigmoid_bet_size,
    target_position,
)


def test_book_sigmoid_calibration_and_targets():
    # AFML 10.6: calibrate to m*=0.95 at divergence 10, Q=100.
    w = calibrate_sigmoid_width(10.0, 0.95)
    assert sigmoid_bet_size(w, 10.0) == pytest.approx(0.95)
    assert target_position(w, 110, 100, 100) == 95
    assert target_position(w, 115, 100, 100) == 97


def test_book_limit_price():
    # AFML 10.6: order of size 97 from flat with f=115, p=100 is priced below 112.3657.
    w = calibrate_sigmoid_width(10.0, 0.95)
    assert limit_price(97, 0, 115.0, w, 100) == pytest.approx(112.3657, abs=1e-4)


def test_limit_price_is_between_market_and_forecast():
    w = 2.0
    p = limit_price(50, 0, 105.0, w, 100)
    assert 100.0 < p < 105.0  # between market price and forecast for a long order
    assert math.isnan(limit_price(10, 10, 105.0, w, 100))


@pytest.mark.parametrize(
    ("current", "target"),
    [(0, 100), (0, -100), (99, 100), (-99, -100)],
)
def test_limit_price_at_cap_is_nan(current, target):
    # A target at max_position gives m == 1, where the cap term has no finite limit.
    assert math.isnan(limit_price(target, current, 105.0, 2.0, 100))


@pytest.mark.parametrize(
    ("current", "target", "forecast", "w", "expected"),
    [
        (0, 97, 115.0, calibrate_sigmoid_width(10.0, 0.95), 112.36573855883363),
        (0, 99, 105.0, 2.0, 103.71886638294875),
        (0, -99, 105.0, 2.0, 103.71886638294875),
        (-30, 50, 105.0, 2.0, 104.38670427147159),
    ],
)
def test_limit_price_below_cap_unchanged(current, target, forecast, w, expected):
    assert limit_price(target, current, forecast, w, 100) == pytest.approx(expected, rel=1e-14)


def test_two_class_probability_mapping():
    m = prob_bet_size([0.5, 0.9, 0.1, 1.0, 0.0])
    assert m[0] == 0.0
    assert m[1] == pytest.approx(-m[2])
    assert m[1] > 0
    assert m[3] == pytest.approx(1.0) and m[4] == pytest.approx(-1.0)


def test_probability_outside_unit_interval_rejected():
    with pytest.raises(ValueError):
        prob_bet_size([1.2])


def test_multiclass_probability_sign_follows_side():
    m_long = prob_bet_size([0.8], side=[1], num_classes=3)
    m_short = prob_bet_size([0.8], side=[-1], num_classes=3)
    assert m_long[0] > 0 and m_short[0] == pytest.approx(-m_long[0])


def test_discretize_signal_rounds_to_step():
    out = discretize_signal([0.33, -0.61, 0.0], 0.2)
    np.testing.assert_allclose(out, [0.4, -0.6, 0.0])
    with pytest.raises(ValueError):
        discretize_signal([0.1], 0.0)


def _naive_average(signals, t1, grid):
    out = []
    for t in grid:
        active = [s for s, s0, s1 in zip(signals.values, signals.index, t1.values)
                  if s0 <= t <= s1]
        out.append(np.mean(active) if active else 0.0)
    return np.array(out)


def test_average_active_signals_matches_naive_loop():
    rng = np.random.default_rng(0)
    idx = pd.date_range("2020-01-01", periods=60, freq="D")
    sig = pd.Series(rng.uniform(-1, 1, 30), index=idx[:30])
    lengths = rng.integers(1, 12, 30)
    t1 = pd.Series([idx[min(i + L, 59)] for i, L in enumerate(lengths)], index=idx[:30])
    got = average_active_signals(sig, t1, index=idx).to_numpy()
    want = _naive_average(sig, t1, idx)
    np.testing.assert_allclose(got, want, atol=1e-12)


def test_calibrate_sigmoid_width_warns_on_infinite_x_and_still_returns_inf() -> None:
    with pytest.warns(UserWarning, match="calibrate_sigmoid_width.*non-finite"):
        assert calibrate_sigmoid_width(np.inf, 0.5) == math.inf


def test_discretize_signal_warns_on_infinite_element_and_still_returns_inf() -> None:
    signal = np.array([0.2, np.inf, -0.3])
    with pytest.warns(UserWarning, match="discretize_signal.*non-finite"):
        out = discretize_signal(signal, 0.1)
    assert np.isinf(out[1])
@pytest.mark.parametrize("bad", [[0.5, -0.5], np.zeros((2, 2))])
def test_average_active_signals_rejects_non_pandas_input(bad) -> None:
    idx = pd.date_range("2024-01-01", periods=2)
    signals = pd.Series([0.5, -0.5], index=idx)
    t1 = pd.Series([idx[1], idx[1]], index=idx)
    with pytest.raises(TypeError, match="signals"):
        average_active_signals(bad, t1)
    with pytest.raises(TypeError, match="t1"):
        average_active_signals(signals, bad)
