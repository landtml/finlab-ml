"""Tests for quantowl.bet_sizing (AFML ch. 10). Book worked examples are included."""
import math
import warnings

import numpy as np
import pandas as pd
import pytest

from quantowl.bet_sizing import (
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


@pytest.mark.parametrize("bad_w", [-1.0, np.inf, -np.inf, np.nan])
def test_limit_price_rejects_invalid_width(bad_w):
    with pytest.raises(ValueError, match="w must be"):
        limit_price(3, 0, 10.0, bad_w, 10)


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


@pytest.mark.parametrize("bad", [[1.5, -2.2], [0.2, 1.0001], [np.nan, 0.1], [np.inf, 0.1]])
def test_discretize_signal_rejects_values_outside_unit_interval(bad) -> None:
    with pytest.raises(ValueError, match="discretize_signal"):
        discretize_signal(bad, 0.5)


def test_discretize_signal_in_range_unchanged_at_edges() -> None:
    np.testing.assert_allclose(discretize_signal([1.0, -1.0, 0.0], 0.5), [1.0, -1.0, 0.0])
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


def test_prob_bet_size_aligns_side_by_label():
    # A side Series listed in another order must be matched to the probability by date.
    days = pd.to_datetime(["2021-01-04", "2021-01-05"])
    prob = pd.Series([0.9, 0.2], index=days)
    side = pd.Series([-1.0, 1.0], index=days[::-1])
    np.testing.assert_allclose(prob_bet_size(prob, side=side), prob_bet_size([0.9, 0.2], side=[1, -1]))


def test_prob_bet_size_refuses_side_with_a_label_missing_from_prob():
    days = pd.to_datetime(["2021-01-04", "2021-01-05", "2021-01-06"])
    prob = pd.Series([0.9, 0.2], index=days[:2])
    side = pd.Series([1.0, -1.0], index=[days[0], days[2]])
    with pytest.raises(ValueError, match="labels differ"):
        prob_bet_size(prob, side=side)


def test_average_active_signals_tz_aware_index_matches_naive() -> None:
    # .values dropped the timezone, so searchsorted raised TypeError.
    naive = pd.date_range("2024-01-01", periods=5, freq="D")
    aware = naive.tz_localize("Asia/Taipei")
    sig_vals = [0.5, 1.0, -1.0, 0.25, 0.75]
    t1_pos = [1, 2, 3, 4, 4]
    want = average_active_signals(
        pd.Series(sig_vals, index=naive), pd.Series(naive[t1_pos], index=naive)
    ).tolist()
    got = average_active_signals(
        pd.Series(sig_vals, index=aware), pd.Series(aware[t1_pos], index=aware)
    ).tolist()
    assert got == want


def test_discretize_signal_stays_in_unit_interval_when_step_does_not_divide_it() -> None:
    # A step that does not divide 1 must not push a bet past 1: round(1/0.6) * 0.6 = 1.2.
    got = discretize_signal([1.0, -1.0, 0.5], 0.6)
    assert np.all(np.abs(got) <= 1.0)
    assert got[0] == 1.0 and got[1] == -1.0


def test_discretize_signal_step_dividing_one_unchanged() -> None:
    got = discretize_signal([0.26, -0.74, 1.0], 0.25)
    assert got.tolist() == [0.25, -0.75, 1.0]


def test_calibrate_sigmoid_width_rejects_overflowing_width() -> None:
    # x * x overflowed to inf for finite x, so the round trip gave 0.
    with pytest.raises(ValueError, match="overflow"):
        calibrate_sigmoid_width(1e200, 0.5)


def test_calibrate_sigmoid_width_rejects_negative_target_for_positive_x() -> None:
    # m* = -0.95 gave the same width as +0.95 for x > 0.
    with pytest.raises(ValueError, match="m_star"):
        calibrate_sigmoid_width(10.0, -0.95)


@pytest.mark.parametrize("bad_w", [float("nan")])
def test_sigmoid_bet_size_rejects_nan_width(bad_w: float) -> None:
    # The check "if w <= 0" let NaN through.
    with pytest.raises(ValueError, match="w must be"):
        sigmoid_bet_size(bad_w, 1.0)


@pytest.mark.parametrize("bad_forecast", [float("nan"), float("inf")])
def test_limit_price_rejects_non_finite_forecast(bad_forecast: float) -> None:
    # The forecast was never checked, so NaN passed through as a NaN limit.
    with pytest.raises(ValueError, match="forecast"):
        limit_price(1, 0, bad_forecast, 1.0, 4)


@pytest.mark.parametrize("bad_target", [2.0, 2.5])
def test_limit_price_rejects_non_integer_target(bad_target: float) -> None:
    # A float target raised TypeError from range().
    with pytest.raises(ValueError, match="target must be an integer"):
        limit_price(bad_target, 0, 100.0, 1.0, 4)


@pytest.mark.parametrize("bad_cap", [2.5, 4.0])
def test_limit_price_rejects_non_integer_max_position(bad_cap: float) -> None:
    # max_position = 2.5 was accepted silently.
    with pytest.raises(ValueError, match="max_position must be an integer"):
        limit_price(1, 0, 100.0, 1.0, bad_cap)


def test_sigmoid_bet_size_infinite_divergence_is_at_limit() -> None:
    # inf / hypot(sqrt(w), inf) was inf / inf, which gave NaN.
    from quantowl.bet_sizing import sigmoid_bet_size, target_position

    assert float(sigmoid_bet_size(1.0, float("inf"))) == 1.0
    assert float(sigmoid_bet_size(1.0, float("-inf"))) == -1.0
    assert target_position(1.0, float("inf"), 0.5, 10) == 10


def test_calibrate_sigmoid_width_rejects_underflowing_width() -> None:
    # x * x underflowed to 0 at x = 1e-170, so the width was 0 and sigmoid_bet_size then raised.
    # At 1e-160 the width was subnormal and did not round-trip.
    from quantowl.bet_sizing import calibrate_sigmoid_width

    with pytest.raises(ValueError, match="underflows"):
        calibrate_sigmoid_width(1e-170, 0.95)
    # The (1e-160, 0.95) case used to be refused because the 1e-9 round trip ignored the precision
    # of a subnormal width. It now returns a width whose round trip is within that precision.
    w = calibrate_sigmoid_width(1e-160, 0.95)
    assert w > 0.0
    from quantowl.bet_sizing import sigmoid_bet_size

    assert float(sigmoid_bet_size(w, 1e-160)) == pytest.approx(0.95, abs=1e-3)


def test_calibrate_sigmoid_width_tiny_target_raises_value_error() -> None:
    # abs(m_star) ** -2 raised a raw OverflowError for m_star below ~1e-154, before the width guard
    # was reached. It is a ValueError like the other width overflows.
    from quantowl.bet_sizing import calibrate_sigmoid_width

    with pytest.raises(ValueError, match="overflows"):
        calibrate_sigmoid_width(1.0, 1e-200)


def test_calibrate_sigmoid_width_keeps_representable_width() -> None:
    # x * x overflowed before the factor (below 1) was applied, so a width of about 2e304 was
    # rejected as an overflow. Scaling before squaring keeps it.
    from quantowl.bet_sizing import calibrate_sigmoid_width, sigmoid_bet_size

    w = calibrate_sigmoid_width(1e155, 0.999999)
    assert w == pytest.approx(2.000003000004e304, rel=1e-9)
    assert float(sigmoid_bet_size(w, 1e155)) == pytest.approx(0.999999, abs=1e-12)


def test_calibrate_sigmoid_width_keeps_subnormal_width_that_round_trips() -> None:
    # The subnormal-width guard rejected every subnormal width, even one that calibrates back to
    # m_star within float tolerance. The round trip decides now, not the magnitude.
    from quantowl.bet_sizing import calibrate_sigmoid_width, sigmoid_bet_size

    w = calibrate_sigmoid_width(1e-155, 0.5)
    assert 0.0 < w < np.finfo(np.float64).tiny
    assert float(sigmoid_bet_size(w, 1e-155)) == pytest.approx(0.5, abs=1e-9)
    w2 = calibrate_sigmoid_width(3e-155, 0.99)
    assert float(sigmoid_bet_size(w2, 3e-155)) == pytest.approx(0.99, abs=1e-9)


def test_target_position_rejects_non_integer_cap() -> None:
    # limit_price refused a float cap, but target_position accepted 2.5 and returned 0 with no error.
    from quantowl.bet_sizing import target_position

    with pytest.raises(ValueError, match="max_position must be an integer"):
        target_position(1.0, 0.6, 0.4, 2.5)


def test_limit_price_rejects_bool_position() -> None:
    # bool is a numbers.Integral, so True was accepted as a target of 1.
    from quantowl.bet_sizing import limit_price

    with pytest.raises(ValueError, match="target must be an integer"):
        limit_price(True, 0, 0.5, 1.0, 10)
    with pytest.raises(ValueError, match="max_position must be an integer"):
        limit_price(1, 0, 0.5, 1.0, True)


def test_target_position_rejects_non_finite_width_and_price() -> None:
    # An infinite width gave 0 and an infinite market price gave a full short, while limit_price
    # rejects the same width. An infinite forecast stays valid (the sigmoid limit is +/-1).
    from quantowl.bet_sizing import target_position

    with pytest.raises(ValueError, match="w must be > 0 and finite"):
        target_position(float("inf"), 0.6, 0.4, 10)
    with pytest.raises(ValueError, match="market_price must be finite"):
        target_position(1.0, 0.6, float("inf"), 10)
    with pytest.raises(ValueError, match="forecast must not be NaN"):
        target_position(1.0, float("nan"), 0.4, 10)
    assert target_position(1.0, float("inf"), 0.5, 10) == 10


def test_average_active_signals_naive_grid_with_aware_signals() -> None:
    # Comparing Index objects made a naive grid against tz-aware signals raise TypeError inside the
    # sweep. The expected values are [1, .75, .75, .5, 0, 0] (UTC instants).
    from quantowl.bet_sizing import average_active_signals

    d = pd.date_range("2020-01-01", periods=2, freq="D", tz="UTC")
    sig = pd.Series([1.0, 0.5], index=d)
    t1 = pd.Series(pd.date_range("2020-01-03", periods=2, freq="D", tz="UTC"), index=d)
    grid = pd.date_range("2020-01-01", periods=6, freq="D")
    got = average_active_signals(sig, t1, index=grid)
    assert got.tolist() == pytest.approx([1.0, 0.75, 0.75, 0.5, 0.0, 0.0])


def test_average_active_signals_naive_t1_with_aware_signals() -> None:
    # The before-start check compared a naive t1 with aware signals and raised.
    from quantowl.bet_sizing import average_active_signals

    d = pd.date_range("2020-01-01", periods=2, freq="D", tz="UTC")
    sig = pd.Series([1.0, 0.5], index=d)
    t1 = pd.Series(pd.date_range("2020-01-03", periods=2, freq="D"), index=d)
    grid = pd.date_range("2020-01-01", periods=6, freq="D")
    assert average_active_signals(sig, t1, index=grid).tolist() == pytest.approx(
        [1.0, 0.75, 0.75, 0.5, 0.0, 0.0]
    )


def test_discretize_signal_rejects_step_whose_quotient_overflows() -> None:
    # For steps below about 1.7e-309, signal / step overflowed to inf, and the clip turned a 0.3 bet
    # into a full 1.0 with no error. Such a step is refused instead.
    from quantowl.bet_sizing import discretize_signal

    with pytest.raises(ValueError, match="step is too small"):
        discretize_signal([0.3], 1e-309)
    assert discretize_signal([0.3], 1e-6).tolist() == pytest.approx([0.3])


def test_prob_bet_size_negative_zero_is_probability_zero() -> None:
    # -0.0 is the same probability as 0.0, but 1 / sqrt(-0.0) flipped the sign, so the largest short
    # became the largest long. np.round(-0.0001, 3) is -0.0 in practice.
    from quantowl.bet_sizing import prob_bet_size

    assert prob_bet_size([-0.0]).tolist() == prob_bet_size([0.0]).tolist() == [-1.0]
    assert prob_bet_size(np.round([-0.0001], 3)).tolist() == [-1.0]


def test_prob_bet_size_rejects_non_integer_num_classes() -> None:
    # num_classes=2.5 took the "> 2" branch and was computed with 2.5, while the error said side was
    # required. It is now refused as a non-integer.
    from quantowl.bet_sizing import prob_bet_size

    with pytest.raises(ValueError, match="num_classes must be an integer"):
        prob_bet_size([0.9], side=[1], num_classes=2.5)


def test_prob_bet_size_accepts_integral_float_num_classes() -> None:
    # 2.0 is a whole number and is accepted. Only non-integral values such as 2.5 are refused.
    from quantowl.bet_sizing import prob_bet_size

    assert np.allclose(prob_bet_size([0.7], num_classes=2.0), prob_bet_size([0.7], num_classes=2))
    with pytest.raises(ValueError, match="num_classes must be an integer"):
        prob_bet_size([0.7], num_classes=2.5)


def test_prob_bet_size_scalar_prob_with_length_one_side_names_shapes() -> None:
    # The message said "side has length 1 but prob has length 1".
    from quantowl.bet_sizing import prob_bet_size

    with pytest.raises(ValueError, match="side shape \\(1,\\) does not match prob shape \\(\\)"):
        prob_bet_size(0.7, side=[1])


def test_average_active_signals_integer_index_is_averaged() -> None:
    # The tz check read .tz on a plain Index and raised AttributeError. The expected result is
    # [0.5, 0.4].
    from quantowl.bet_sizing import average_active_signals

    out = average_active_signals(pd.Series([0.5, 0.3], index=[1, 2]), pd.Series([2, 3], index=[1, 2]))
    assert out.tolist() == [0.5, 0.4]


def test_average_active_signals_mismatched_index_types_raise_value_error() -> None:
    # A non-datetime index with datetime end times must fail as ValueError.
    from quantowl.bet_sizing import average_active_signals

    t1 = pd.Series(pd.date_range("2020-01-02", periods=2), index=[1, 2])
    with pytest.raises(ValueError):
        average_active_signals(pd.Series([0.5, 0.3], index=[1, 2]), t1)


def test_average_active_signals_mixed_awareness_keeps_caller_timezone() -> None:
    # The UTC conversion leaked into the result index (naive, shifted by 8 h).
    from quantowl.bet_sizing import average_active_signals

    idx = pd.date_range("2020-01-01", periods=2, tz="Asia/Taipei")
    t1 = pd.Series(pd.date_range("2020-01-02", periods=2), index=idx)
    out = average_active_signals(pd.Series([0.5, 0.3], index=idx), t1)
    assert out.index.equals(idx)
    assert str(out.index.tz) == "Asia/Taipei"


def test_limit_price_rejects_zero_width_like_target_position() -> None:
    # target_position refused w = 0 while limit_price returned the forecast. Both now refuse, since
    # w must be > 0.
    from quantowl.bet_sizing import limit_price, target_position

    with pytest.raises(ValueError, match="w must be"):
        limit_price(2, 0, 0.6, 0.0, 10)
    with pytest.raises(ValueError, match="w must be"):
        target_position(0.0, 0.6, 0.5, 10)


def test_calibrate_sigmoid_width_exact_for_representable_input() -> None:
    # The sqrt((1-m)(1+m)) step gave 299.99999999999994 where the exact width is 300.0.
    from quantowl.bet_sizing import calibrate_sigmoid_width

    assert calibrate_sigmoid_width(10.0, 0.5) == 300.0
    assert calibrate_sigmoid_width(np.float32(10), 0.5) == 300.0


def test_calibrate_sigmoid_width_large_x_does_not_overflow_the_square() -> None:
    # x * x overflows near 1e154 but the width is representable, so the scaling must still hold.
    from quantowl.bet_sizing import calibrate_sigmoid_width

    w = calibrate_sigmoid_width(1e155, 0.99999)
    assert math.isfinite(w) and w > 0


def test_calibrate_sigmoid_width_subnormal_width_that_calibrates_is_returned() -> None:
    # (1e-160, 0.5) returns the subnormal width 3e-320; an earlier guard refused it because the 1e-9
    # round trip ignores the precision of a subnormal width.
    from quantowl.bet_sizing import calibrate_sigmoid_width, sigmoid_bet_size

    w = calibrate_sigmoid_width(1e-160, 0.5)
    assert w == pytest.approx(3e-320, rel=1e-2, abs=0.0)  # abs=0: subnormal values are below the 1e-12 default
    w2 = calibrate_sigmoid_width(1e-160, 0.999)
    assert w2 == pytest.approx(2e-323, rel=0.5, abs=0.0)
    assert float(sigmoid_bet_size(w2, 1e-160)) == pytest.approx(0.999, abs=1e-3)


def test_calibrate_sigmoid_width_pin_subnormal_widths_that_calibrate() -> None:
    # Subnormal widths whose rounding step moves the bet by at most 1e-3 are still returned, so the
    # refusal must not catch these.
    from quantowl.bet_sizing import calibrate_sigmoid_width, sigmoid_bet_size

    for x, m in ((1e-160, 0.5), (1e-160, 0.999), (1e-155, 0.5)):
        w = calibrate_sigmoid_width(x, m)
        assert 0.0 < w < np.finfo(np.float64).tiny
        assert float(sigmoid_bet_size(w, x)) == pytest.approx(m, abs=1e-3)


def test_calibrate_sigmoid_width_refuses_subnormal_width_too_coarse_to_calibrate() -> None:
    # At x = 1.15e-162 the only subnormal width is 5e-324, one rounding step. Its bet was 0.41 for a
    # target of 0.5 (0.09 off), yet the tolerance was m/2 because x * x underflowed, so the width was
    # accepted. The width must be refused instead.
    from quantowl.bet_sizing import calibrate_sigmoid_width, sigmoid_bet_size

    with pytest.raises(ValueError, match="underflows"):
        calibrate_sigmoid_width(1.15084165586488e-162, 0.6746939996559512)
    with pytest.raises(ValueError, match="underflows"):
        calibrate_sigmoid_width(1e-162, 0.5)
    # Every width that is accepted in the subnormal range round-trips within 1e-3.
    rng = np.random.default_rng(12)
    for x in 10.0 ** rng.uniform(-165, -150, 200):
        for m in rng.uniform(0.01, 0.99, 3):
            try:
                w = calibrate_sigmoid_width(float(x), float(m))
            except ValueError:
                continue
            if w < np.finfo(np.float64).tiny:
                assert abs(float(sigmoid_bet_size(w, float(x))) - m) <= 1e-3


def test_limit_price_refuses_zero_width_as_intended() -> None:
    # w = 0 is refused by limit_price, as target_position refuses it. The zero width is refused on
    # purpose rather than priced at the forecast.
    from quantowl.bet_sizing import limit_price, target_position

    with pytest.raises(ValueError, match="w must be > 0"):
        limit_price(1, 0, 12.0, 0.0, 5)
    with pytest.raises(ValueError, match="w must be > 0"):
        target_position(0.0, 12.0, 10.0, 5)


def test_position_arguments_refuse_whole_float_as_intended() -> None:
    # A position argument must be an integer type, so 5.0 is refused by both limit_price and
    # target_position. numpy integers are still accepted.
    from quantowl.bet_sizing import limit_price, target_position

    with pytest.raises(ValueError, match="max_position must be an integer"):
        limit_price(2, 0, 12.0, 1.0, 5.0)
    with pytest.raises(ValueError, match="max_position must be an integer"):
        target_position(0.5, 12.0, 10.0, 5.0)
    assert math.isfinite(limit_price(np.int64(2), np.int64(0), 12.0, 1.0, np.int64(5)))
    assert target_position(0.5, 12.0, 10.0, np.int64(5)) == 4


def test_calibrate_sigmoid_width_is_accurate_when_m_is_near_one() -> None:
    # 1 - m * m cancelled when m is near 1, so the width had a relative error of 5.5e-10 at (10,
    # 1 - 1e-8). Factoring 1 - m * m as (1 - m)(1 + m) keeps it near 1e-15.
    from decimal import Decimal, localcontext

    from quantowl.bet_sizing import calibrate_sigmoid_width, sigmoid_bet_size

    m = 1 - 1e-8
    w = calibrate_sigmoid_width(10.0, m)
    with localcontext() as ctx:
        ctx.prec = 60
        md = Decimal(m)
        exact = Decimal(100) * (1 - md * md) / (md * md)
        assert abs((Decimal(w) - exact) / exact) < Decimal("1e-13")
    assert float(sigmoid_bet_size(w, 10.0)) == pytest.approx(m, rel=1e-14)


def test_target_position_large_cap_is_not_snapped() -> None:
    # The snap tolerance scaled with |k|, so at 5e5 shares it moved a bet 0.00025 share up to the
    # next integer. int(m * cap) gives -500484 (truncation toward zero).
    from quantowl.bet_sizing import target_position

    assert target_position(4.084685686539295, -1.1683697144921645, 0.0, 10**6) == -500484
    # A bet clearly below the position still truncates; this is a documented limit, not a snap.
    assert target_position(0.03772579937783819, 47.77086633466709, 47.7, 29) == 9


def test_discretize_signal_step_floor_is_documented() -> None:
    # Overflow starts for a step below about |signal| * 5.6e-309, so a step of 1e-308 is valid and a
    # step of 1e-309 is refused by name. The docstring states this floor.
    from quantowl.bet_sizing import discretize_signal

    assert discretize_signal([0.3], 1e-308).tolist() == pytest.approx([0.3])
    assert discretize_signal([-0.3], 1e-308).tolist() == pytest.approx([-0.3])
    with pytest.raises(ValueError, match="step is too small"):
        discretize_signal([0.3], 1e-309)


def test_average_active_signals_accepts_the_same_instants_in_two_zones() -> None:
    idx = pd.date_range("2020-01-01", periods=10, freq="D", tz="UTC")
    sig = pd.Series(np.linspace(-1, 1, 10), index=idx)
    t1 = pd.Series(idx + pd.Timedelta(days=3), index=idx)
    t1_tpe = pd.Series(t1.to_numpy(), index=idx.tz_convert("Asia/Taipei")).dt.tz_convert("Asia/Taipei")
    want = average_active_signals(sig, t1)
    got = average_active_signals(sig, t1_tpe)
    np.testing.assert_allclose(got.to_numpy(), want.to_numpy())
    shifted = pd.Series(t1_tpe.to_numpy(), index=t1_tpe.index + pd.Timedelta(hours=1))
    with pytest.raises(ValueError, match="share an index"):
        average_active_signals(sig, shifted)
