"""Tests for finlab.bet_sizing (AFML ch. 10). Book worked examples are included."""
import math
import warnings

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


def test_sigmoid_large_divergence_reaches_its_limit_without_overflow():
    # x * x overflows for |x| above about 1.3e154; the limit must still be reached.
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        assert float(sigmoid_bet_size(1.0, 1e200)) == pytest.approx(1.0)
        assert float(sigmoid_bet_size(1.0, -1e200)) == pytest.approx(-1.0)
    assert target_position(1.0, 1e200, 0.0, 10) == 10


def test_calibrate_sigmoid_width_rejects_negative_x():
    # The bet sign follows x, so a negative divergence cannot give a positive m_star.
    with pytest.raises(ValueError, match="x"):
        calibrate_sigmoid_width(-10.0, 0.95)
    assert calibrate_sigmoid_width(10.0, 0.95) == pytest.approx(10.8033, abs=1e-4)


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
        (0, -99, 105.0, 2.0, 106.28113361705125),  # mirror of (0, 99) about the forecast
    ],
)
def test_limit_price_below_cap_unchanged(current, target, forecast, w, expected):
    assert limit_price(target, current, forecast, w, 100) == pytest.approx(expected, rel=1e-14)


def test_limit_price_crossing_zero_averages_all_eighty_fills():
    # Order from -30 to 50 fills positions -29..50; the average is over those 80 signed positions.
    expected = sum(105.0 - (p / 100) * math.sqrt(2.0 / (1 - (p / 100) ** 2)) for p in range(-29, 51)) / 80
    assert limit_price(50, -30, 105.0, 2.0, 100) == pytest.approx(expected, rel=1e-14)


def test_limit_price_short_target_is_mirror_of_long():
    # A short order fills at prices below the forecast's mirror image: its limit sits above it.
    long_price = limit_price(3, 0, 10.0, 1.0, 10)
    short_price = limit_price(-3, 0, 10.0, 1.0, 10)
    assert short_price > 10.0
    assert short_price == pytest.approx(2 * 10.0 - long_price)


@pytest.mark.parametrize(
    ("reduce_args", "fill_args"),
    [((1, 3), (2, 0)), ((-1, -3), (-2, 0))],
)
def test_limit_price_reduction_prices_the_levels_crossed(reduce_args, fill_args):
    # Cutting 3 -> 1 fills the same levels (1 and 2) as building 0 -> 2, so the prices match.
    reduced = limit_price(*reduce_args, 10.0, 1.0, 10)
    assert math.isfinite(reduced)
    assert reduced == pytest.approx(limit_price(*fill_args, 10.0, 1.0, 10), rel=1e-14)


@pytest.mark.parametrize(
    ("target", "current", "expected"),
    [(1, -10, 10.6055), (-1, 2, 10.0), (-2, 3, 10.0)],
)
def test_limit_price_averages_fills_across_zero(target, current, expected):
    # Each unit fill is priced at the signed position after it, so a flip averages over
    # |target - current| fills, not over the absolute levels between them.
    assert limit_price(target, current, 10.0, 1.0, 10) == pytest.approx(expected, abs=1e-4)


@pytest.mark.parametrize(
    ("target", "current", "expected"),
    [
        (1, 0, 9.899496218474079),
        (5, 0, 9.673420114512794),
        (-5, 0, 10.326579885487206),
        (9, 0, 9.248758841730664),
        (3, 1, 9.740695201875747),
        (7, 2, 9.388306488100442),
        (-3, -1, 10.259304798124253),
        (-7, -2, 10.611693511899558),
        (2, 0, 9.847686036621074),
        (0, 3, 9.898457357747382),
        (0, -3, 10.101542642252618),
        (0, 5, 9.788890168350719),
        (4, 1, 9.681651541093169),
        (1, 4, 9.793628874075191),
        (-6, -2, 10.519567875169546),
        (-2, -6, 10.383098911477529),
        (9, 4, 8.858875746764475),
    ],
)
def test_limit_price_same_sign_and_flat_values_unchanged(target, current, expected):
    assert limit_price(target, current, 10.0, 1.0, 10) == pytest.approx(expected, rel=1e-14)


@pytest.mark.parametrize(("target", "current"), [(0, 0), (8, 8), (-8, -8)])
def test_limit_price_no_order_is_nan(target, current):
    assert math.isnan(limit_price(target, current, 10.0, 1.0, 10))


@pytest.mark.parametrize(("target", "current", "max_position"), [(101, 0, 100), (-101, 0, 100), (2, 5, 3)])
def test_limit_price_rejects_positions_beyond_max_position(target, current, max_position):
    with pytest.raises(ValueError, match="max_position"):
        limit_price(target, current, 105.0, 2.0, max_position)


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


def test_average_active_signals_nan_signal_only_poisons_its_active_span() -> None:
    idx = pd.date_range("2024-01-01", periods=5, freq="D")
    signals = pd.Series([1.0, np.nan, 1.0, -1.0, 1.0], index=idx)
    t1 = pd.Series(idx[[2, 3, 4, 4, 4]], index=idx)
    got = average_active_signals(signals, t1).to_numpy()
    assert got[0] == pytest.approx(1.0)
    assert np.isnan(got[1:4]).all()  # the NaN event is active at 1, 2 and 3
    assert got[4] == pytest.approx(1.0 / 3.0)  # events 2, 3 and 4 remain, with mean 1/3


def test_average_active_signals_inf_signal_only_affects_its_active_span() -> None:
    idx = pd.date_range("2024-01-01", periods=5, freq="D")
    signals = pd.Series([np.inf, 1.0, 1.0, 1.0, 1.0], index=idx)
    t1 = pd.Series(idx[[0, 2, 3, 4, 4]], index=idx)
    assert average_active_signals(signals, t1).tolist() == [np.inf, 1.0, 1.0, 1.0, 1.0]


def test_average_active_signals_negative_inf_signal_only_affects_its_active_span() -> None:
    idx = pd.date_range("2024-01-01", periods=5, freq="D")
    signals = pd.Series([-np.inf, 0.0, 1.0, 0.0, 0.0], index=idx)
    t1 = pd.Series(idx[[0, 2, 4, 3, 4]], index=idx)
    assert average_active_signals(signals, t1).tolist() == [-np.inf, 0.0, 0.5, 0.5, 0.5]


def test_average_active_signals_opposite_infs_active_together_is_nan() -> None:
    idx = pd.date_range("2024-01-01", periods=3, freq="D")
    signals = pd.Series([np.inf, -np.inf, 1.0], index=idx)
    t1 = pd.Series(idx[[2, 2, 2]], index=idx)
    got = average_active_signals(signals, t1).to_numpy()
    assert got[0] == np.inf  # only the +inf event is active at 0
    assert np.isnan(got[1:]).all()  # both infinities are active at 1 and 2


def test_average_active_signals_rejects_nat_end_times() -> None:
    idx = pd.date_range("2024-01-01", periods=5, freq="D")
    signals = pd.Series([1.0, 1.0, 1.0, -1.0, 1.0], index=idx)
    t1 = pd.Series([idx[2], idx[3], idx[4], idx[4], pd.NaT], index=idx)
    with pytest.raises(ValueError, match="NaT") as info:
        average_active_signals(signals, t1)
    assert str(idx[4].date()) in str(info.value)  # the offending row is named


def test_average_active_signals_rejects_t1_before_start() -> None:
    idx = pd.date_range("2020-01-01", periods=5, freq="D")
    sig = pd.Series([0.5, 1.0, -1.0, 0.25, 0.75], index=idx)
    t1 = pd.Series(idx[[1, 0, 3, 4, 4]], index=idx)  # event 1 starts day 1, ends day 0
    with pytest.raises(ValueError, match="before") as info:
        average_active_signals(sig, t1)
    assert str(idx[1].date()) in str(info.value)  # the offending row is named


def test_average_active_signals_valid_t1_unchanged() -> None:
    idx = pd.date_range("2020-01-01", periods=5, freq="D")
    sig = pd.Series([0.5, 1.0, -1.0, 0.25, 0.75], index=idx)
    t1 = pd.Series(idx[[1, 2, 3, 4, 4]], index=idx)
    got = average_active_signals(sig, t1).round(4).tolist()
    assert got == [0.5, 0.75, 0.0, -0.375, 0.5]


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


def test_multiclass_probability_below_uniform_rejected():
    # The predicted label's probability is a maximum, so it cannot be below 1/k.
    with pytest.raises(ValueError, match="1/num_classes"):
        prob_bet_size([0.2], side=[1], num_classes=3)
    with pytest.raises(ValueError, match="1/num_classes"):
        prob_bet_size([0.0], side=[1], num_classes=3)
    assert prob_bet_size([1.0 / 3], side=[1], num_classes=3)[0] == pytest.approx(0.0, abs=1e-12)


def test_prob_bet_size_applies_side_for_two_classes():
    # A short side must flip the sign of the two-class bet, not be silently dropped.
    long_bet = prob_bet_size([0.7])
    np.testing.assert_allclose(prob_bet_size([0.7], side=[1]), long_bet)
    np.testing.assert_allclose(prob_bet_size([0.7], side=[-1]), -long_bet)


@pytest.mark.parametrize("bad_side", [[2], [0], [np.nan]])
def test_prob_bet_size_rejects_side_outside_plus_minus_one(bad_side):
    with pytest.raises(ValueError, match="side"):
        prob_bet_size([0.9], side=bad_side)


def test_prob_bet_size_names_both_lengths_on_side_mismatch():
    with pytest.raises(ValueError, match="side has length 2 but prob has length 3"):
        prob_bet_size([0.7, 0.2, 0.5], side=[1, -1], num_classes=3)
    with pytest.raises(ValueError, match="side has length 2 but prob has length 3"):
        prob_bet_size([0.7, 0.2, 0.5], side=[1, -1])
