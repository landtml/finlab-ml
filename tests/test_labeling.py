"""Tests for quantowl.labeling (AFML Chapter 3 and trend scanning).

Each fast function is checked against a brute-force reference written
independently in this file: the triple barrier against a pandas path loop in
the style of Snippet 3.2, daily volatility against a per-bar loop, and trend
scanning against scipy.stats.linregress.
"""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
import pytest
from scipy import stats

from quantowl.labeling import (
    add_vertical_barrier,
    drop_labels,
    get_bins,
    get_daily_vol,
    get_events,
    meta_labels,
    trend_scanning_labels,
)


def _random_walk(n: int, seed: int, vol: float = 0.01) -> pd.Series:
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range("2019-01-01", periods=n)
    return pd.Series(100.0 * np.exp(np.cumsum(rng.normal(0.0, vol, n))), index=idx)


# ---------------------------------------------------------------------------
# Brute-force references
# ---------------------------------------------------------------------------


def naive_events(close, t_events, pt_sl, target, min_ret, vert=None, side=None):
    """Snippet 3.2 logic: first touch by pandas path slices, one event at a time."""
    rows = []
    for t0 in t_events:
        trg = target.loc[t0]
        if not (trg > min_ret):
            continue
        sd = 1.0 if side is None else float(side.loc[t0])
        t_v = vert.loc[t0] if vert is not None else pd.NaT
        end = close.index[-1] if pd.isna(t_v) else t_v
        path = close.loc[t0:end]
        rel = (path / close.loc[t0] - 1.0) * sd
        t_pt = rel[rel > pt_sl[0] * trg].index.min() if pt_sl[0] > 0 else pd.NaT
        # With side omitted, the stop uses pt_sl[0] too (the documented rule).
        sl_mult = pt_sl[0] if side is None else pt_sl[1]
        t_sl = rel[rel < -sl_mult * trg].index.min() if sl_mult > 0 else pd.NaT
        if not pd.isna(t_sl) and (pd.isna(t_pt) or t_sl <= t_pt):
            t1, code = t_sl, "sl"
        elif not pd.isna(t_pt):
            t1, code = t_pt, "pt"
        elif not pd.isna(t_v):
            t1, code = t_v, "vertical"
        else:
            t1, code = pd.NaT, "none"
        rows.append((t0, t1, trg, sd, code))
    return pd.DataFrame(rows, columns=["t0", "t1", "trgt", "side", "barrier"]).set_index("t0")


def naive_daily_vol(close, span):
    """Book's Snippet 3.1 logic: last bar strictly before t - 1 day, per bar."""
    rets_idx, rets = [], []
    for i, t in enumerate(close.index):
        prev = [j for j in range(i) if close.index[j] < t - pd.Timedelta(days=1)]
        if prev:
            j = prev[-1]
            rets_idx.append(t)
            rets.append(close.iloc[i] / close.iloc[j] - 1.0)
    vol = pd.Series(rets, index=pd.DatetimeIndex(rets_idx)).ewm(span=span).std()
    return vol.reindex(close.index)


# ---------------------------------------------------------------------------
# Snippet 3.1 and 3.4
# ---------------------------------------------------------------------------


def test_daily_vol_matches_per_bar_reference():
    close = _random_walk(150, seed=1)
    got = get_daily_vol(close, span=20)
    ref = naive_daily_vol(close, span=20)
    pd.testing.assert_series_equal(got, ref, check_names=False, rtol=1e-10)


def test_daily_vol_first_day_is_nan():
    close = _random_walk(10, seed=2)
    vol = get_daily_vol(close, span=5)
    assert vol.iloc[0] != vol.iloc[0]  # NaN: no bar a day earlier


def test_vertical_barrier_is_first_bar_at_or_after_expiry():
    close = _random_walk(40, seed=3)
    t_events = close.index[[0, 5, -1]]
    v = add_vertical_barrier(t_events, close, num_days=3)
    assert v.iloc[0] == close.index[3]
    assert v.iloc[1] == close.index[8]
    assert pd.isna(v.iloc[2])  # last bar + 3 days is past the last bar


# ---------------------------------------------------------------------------
# Triple barrier
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_triple_barrier_matches_pandas_reference(seed):
    close = _random_walk(400, seed=seed, vol=0.02)
    rng = np.random.default_rng(100 + seed)
    t_events = close.index[np.sort(rng.choice(380, size=60, replace=False))]
    target = pd.Series(rng.uniform(0.005, 0.04, len(close)), index=close.index)
    vert = add_vertical_barrier(t_events, close, num_days=12)
    vert = vert.where(vert.notna(), pd.NaT)
    pt_sl = (1.5, 1.0)
    got = get_events(close, t_events, pt_sl, target, 0.0, vertical_barrier_times=vert)
    ref = naive_events(close, t_events, pt_sl, target, 0.0, vert=vert)
    assert list(got.index) == list(ref.index)
    pd.testing.assert_series_equal(got["t1"], ref["t1"], check_names=False)
    assert list(got["barrier"]) == list(ref["barrier"])
    np.testing.assert_allclose(got["trgt"].to_numpy(), ref["trgt"].to_numpy())


def test_triple_barrier_meta_mode_matches_pandas_reference():
    close = _random_walk(300, seed=7, vol=0.02)
    rng = np.random.default_rng(7)
    t_events = close.index[np.sort(rng.choice(280, size=50, replace=False))]
    target = pd.Series(0.02, index=close.index)
    side = pd.Series(rng.choice([-1.0, 1.0], size=len(t_events)), index=t_events)
    vert = add_vertical_barrier(t_events, close, num_days=15)
    pt_sl = (2.0, 0.5)
    got = get_events(close, t_events, pt_sl, target, 0.0, vertical_barrier_times=vert, side=side)
    ref = naive_events(close, t_events, pt_sl, target, 0.0, vert=vert, side=side)
    pd.testing.assert_series_equal(got["t1"], ref["t1"], check_names=False)
    assert list(got["barrier"]) == list(ref["barrier"])


def test_triple_barrier_hits_upper_on_rising_path():
    close = pd.Series(100.0 * 1.01 ** np.arange(60), index=pd.bdate_range("2020-01-01", periods=60))
    t_events = close.index[:-1]
    target = pd.Series(0.005, index=close.index)
    ev = get_events(close, t_events, (1.0, 1.0), target, 0.0)
    assert set(ev["barrier"]) == {"pt"}
    # 1% per bar exceeds the 0.5% upper barrier on the first bar after entry.
    expected = close.index[1:]
    np.testing.assert_array_equal(ev["t1"].to_numpy(), expected.to_numpy())


def test_triple_barrier_hits_lower_on_falling_path():
    close = pd.Series(100.0 * 0.99 ** np.arange(60), index=pd.bdate_range("2020-01-01", periods=60))
    t_events = close.index[:-1]
    target = pd.Series(0.005, index=close.index)
    ev = get_events(close, t_events, (1.0, 1.0), target, 0.0)
    assert set(ev["barrier"]) == {"sl"}
    np.testing.assert_array_equal(ev["t1"].to_numpy(), close.index[1:].to_numpy())


def test_triple_barrier_meta_side_flips_which_barrier_is_hit():
    close = pd.Series(100.0 * 1.01 ** np.arange(60), index=pd.bdate_range("2020-01-01", periods=60))
    t_events = close.index[:-1]
    target = pd.Series(0.005, index=close.index)
    long_ev = get_events(close, t_events, (1.0, 1.0), target, 0.0,
                         side=pd.Series(1.0, index=t_events))
    short_ev = get_events(close, t_events, (1.0, 1.0), target, 0.0,
                          side=pd.Series(-1.0, index=t_events))
    assert set(long_ev["barrier"]) == {"pt"}
    assert set(short_ev["barrier"]) == {"sl"}


def test_triple_barrier_vertical_barrier_applies_when_no_horizontal_touch():
    close = pd.Series(100.0, index=pd.bdate_range("2020-01-01", periods=30))
    t_events = close.index[:5]
    target = pd.Series(0.05, index=close.index)
    vert = pd.Series(close.index[[4, 5, 6, 7, 8]], index=t_events)
    ev = get_events(close, t_events, (1.0, 1.0), target, 0.0, vertical_barrier_times=vert)
    assert set(ev["barrier"]) == {"vertical"}
    np.testing.assert_array_equal(ev["t1"].to_numpy(), vert.to_numpy())


def test_min_ret_drops_small_targets():
    close = _random_walk(50, seed=4)
    t_events = close.index[:10]
    target = pd.Series(0.01, index=close.index)
    target.iloc[3] = 0.001
    ev = get_events(close, t_events, (1.0, 1.0), target, 0.005)
    assert close.index[3] not in ev.index
    assert len(ev) == 9


def test_get_events_rejects_unknown_event_time():
    close = _random_walk(20, seed=5)
    bad = pd.DatetimeIndex([pd.Timestamp("1990-01-01")])
    with pytest.raises(ValueError):
        get_events(close, bad, (1.0, 1.0), pd.Series(0.01, index=close.index), 0.0)


def test_get_bins_and_meta_labels_agree_with_definition():
    close = _random_walk(200, seed=9, vol=0.02)
    rng = np.random.default_rng(9)
    t_events = close.index[np.sort(rng.choice(180, size=40, replace=False))]
    target = pd.Series(0.02, index=close.index)
    ev = get_events(close, t_events, (1.0, 1.0), target, 0.0,
                    vertical_barrier_times=add_vertical_barrier(t_events, close, 8))
    bins = get_bins(ev, close)  # side-and-size labels: no side multiplication
    ret_direct = close.loc[bins.index.map(lambda t: ev.loc[t, "t1"])].to_numpy() / close.loc[bins.index].to_numpy() - 1.0
    np.testing.assert_allclose(bins["ret"].to_numpy(), ret_direct, rtol=1e-12)
    np.testing.assert_array_equal(bins["bin"].to_numpy(), np.sign(ret_direct).astype(int))

    primary = pd.Series(rng.choice([-1.0, 1.0], size=len(ev)), index=ev.index)
    meta = meta_labels(ev, close, primary)
    assert set(meta["bin"].unique()) <= {0, 1}
    pnl = primary.loc[meta.index].to_numpy() * bins.loc[meta.index, "ret"].to_numpy()
    np.testing.assert_allclose(meta["ret"].to_numpy(), pnl, rtol=1e-12)
    np.testing.assert_array_equal(meta["bin"].to_numpy(), (pnl > 0).astype(int))


def test_meta_labels_is_one_exactly_when_primary_side_was_right():
    close = pd.Series(100.0 * 1.01 ** np.arange(40), index=pd.bdate_range("2020-01-01", periods=40))
    t_events = close.index[:-2]
    target = pd.Series(0.005, index=close.index)
    ev = get_events(close, t_events, (1.0, 1.0), target, 0.0)
    primary = pd.Series(np.where(np.arange(len(ev)) % 2 == 0, 1.0, -1.0), index=ev.index)
    meta = meta_labels(ev, close, primary)
    # Rising path: a long call is right (bin 1), a short call is wrong (bin 0).
    np.testing.assert_array_equal(meta["bin"].to_numpy(), (primary.to_numpy() > 0).astype(int))


def test_meta_labels_rejects_abstention():
    close = _random_walk(30, seed=11)
    ev = get_events(close, close.index[:5], (1.0, 1.0), pd.Series(0.01, index=close.index), 0.0)
    with pytest.raises(ValueError):
        meta_labels(ev, close, np.array([1.0, 0.0, 1.0, -1.0, 1.0]))


def test_get_bins_meta_refuses_nan_side():
    close = _random_walk(30, seed=11)
    ev = get_events(close, close.index[:5], (1.0, 1.0), pd.Series(0.01, index=close.index), 0.0)
    ev["side"] = np.array([1.0, np.nan, 1.0, -1.0, np.nan])
    with pytest.raises(ValueError, match="side is NaN for 2 event"):
        get_bins(ev, close, meta=True)


def test_get_bins_without_meta_ignores_nan_side():
    close = _random_walk(30, seed=11)
    ev = get_events(close, close.index[:5], (1.0, 1.0), pd.Series(0.01, index=close.index), 0.0)
    ev["side"] = 1.0
    with pytest.warns(UserWarning):
        plain = get_bins(ev, close)
    ev["side"] = np.array([1.0, np.nan, 1.0, -1.0, np.nan])
    with pytest.warns(UserWarning):
        nan_side = get_bins(ev, close)
    pd.testing.assert_frame_equal(nan_side, plain)


# ---------------------------------------------------------------------------
# Snippet 3.8 -- drop_labels
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("seed", range(5))
def test_drop_labels_leaves_each_class_above_min_pct(seed):
    rng = np.random.default_rng(seed)
    labels = rng.choice([-1, 0, 1, 2], size=2000, p=[0.4, 0.3, 0.25, 0.05])
    ev = pd.DataFrame({"bin": labels}, index=pd.RangeIndex(2000))
    out = drop_labels(ev, min_pct=0.1)
    freq = out["bin"].value_counts(normalize=True)
    assert freq.shape[0] == 3
    assert freq.min() >= 0.1
    assert 2 not in set(out["bin"])  # the 5% class is always removed


def test_drop_labels_stops_at_two_classes_as_in_book():
    ev = pd.DataFrame({"bin": [0] * 50 + [1] * 45 + [2] * 5}, index=pd.RangeIndex(100))
    out = drop_labels(ev, min_pct=0.1)
    assert set(out["bin"]) == {0, 1}  # class 2 dropped, then the rule stops at two


def test_drop_labels_keeps_balanced_input():
    ev = pd.DataFrame({"bin": [-1, 0, 1] * 20}, index=pd.RangeIndex(60))
    out = drop_labels(ev, min_pct=0.1)
    assert len(out) == 60


# ---------------------------------------------------------------------------
# Trend scanning
# ---------------------------------------------------------------------------


def naive_trend(close, windows, threshold):
    y = np.log(close.to_numpy(dtype=float))
    n = y.shape[0]
    wins = sorted({int(w) for w in windows})
    t_out = np.full(n, np.nan)
    w_out = np.full(n, np.nan)
    b_out = np.full(n, np.nan)
    for i in range(n):
        best, best_t, best_w = -1.0, 0.0, 0
        for length in wins:
            if i + length > n:
                continue
            res = stats.linregress(np.arange(length), y[i:i + length])
            t = res.slope / res.stderr
            if abs(t) > best:
                best, best_t, best_w = abs(t), t, length
        if best >= 0:
            t_out[i] = best_t
            w_out[i] = best_w
            b_out[i] = np.sign(best_t) if best > threshold else 0.0
    return t_out, w_out, b_out


@pytest.mark.parametrize("seed", [0, 1])
def test_trend_scanning_matches_linregress_reference(seed):
    close = _random_walk(120, seed=seed, vol=0.02)
    windows = [5, 9, 16]
    got = trend_scanning_labels(close, windows, t_threshold=1.0)
    t_ref, w_ref, b_ref = naive_trend(close, windows, 1.0)
    np.testing.assert_allclose(got["t_value"].to_numpy(), t_ref, rtol=1e-7, atol=1e-10)
    np.testing.assert_array_equal(got["window"].to_numpy(), w_ref)
    np.testing.assert_array_equal(got["bin"].to_numpy(), b_ref)


def test_trend_scanning_tail_is_undefined_and_threshold_zeroes_weak_trends():
    close = _random_walk(60, seed=3)
    out = trend_scanning_labels(close, [10, 20], t_threshold=1e9)
    assert out["bin"].iloc[-9:].isna().all()  # no 10-bar window fits in the last 9 bars
    assert (out["bin"].dropna() == 0).all()  # threshold this high zeroes everything


def test_trend_scanning_labels_perfect_uptrend_as_plus_one():
    close = pd.Series(np.exp(0.01 * np.arange(50)) * 100, index=pd.bdate_range("2020-01-01", periods=50))
    out = trend_scanning_labels(close, [10], t_threshold=1.96)
    assert set(out["bin"].dropna()) == {1.0}
    # Floating-point residuals keep SSE tiny but nonzero, so t is huge and finite.
    assert out["t_value"].iloc[0] > 1e6


def test_get_bins_side_without_meta_warns_and_is_unchanged():
    import warnings

    close = _random_walk(200, seed=11, vol=0.02)
    rng = np.random.default_rng(11)
    t_events = close.index[np.sort(rng.choice(180, size=30, replace=False))]
    target = pd.Series(0.02, index=close.index)
    side = pd.Series(rng.choice([-1.0, 1.0], size=len(t_events)), index=t_events)
    ev = get_events(close, t_events, (1.0, 1.0), target, 0.0,
                    vertical_barrier_times=add_vertical_barrier(t_events, close, 8),
                    side=side)
    with pytest.warns(UserWarning, match="raw return"):
        warned = get_bins(ev, close)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        silent = get_bins(ev, close, meta=False)
    pd.testing.assert_frame_equal(warned, silent)
    # The bin is the sign of the raw return, not of the side-adjusted P&L.
    raw = close.loc[warned.index.map(lambda t: ev.loc[t, "t1"])].to_numpy() / close.loc[warned.index].to_numpy() - 1.0
    np.testing.assert_array_equal(warned["bin"].to_numpy(), np.sign(raw).astype(int))


def test_get_bins_meta_true_does_not_warn():
    import warnings

    close = _random_walk(200, seed=12, vol=0.02)
    rng = np.random.default_rng(12)
    t_events = close.index[np.sort(rng.choice(180, size=30, replace=False))]
    target = pd.Series(0.02, index=close.index)
    side = pd.Series(rng.choice([-1.0, 1.0], size=len(t_events)), index=t_events)
    ev = get_events(close, t_events, (1.0, 1.0), target, 0.0,
                    vertical_barrier_times=add_vertical_barrier(t_events, close, 8),
                    side=side)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        get_bins(ev, close, meta=True)


def test_get_bins_without_side_column_does_not_warn():
    import warnings

    close = _random_walk(200, seed=13, vol=0.02)
    rng = np.random.default_rng(13)
    t_events = close.index[np.sort(rng.choice(180, size=30, replace=False))]
    target = pd.Series(0.02, index=close.index)
    ev = get_events(close, t_events, (1.0, 1.0), target, 0.0,
                    vertical_barrier_times=add_vertical_barrier(t_events, close, 8))
    ev = ev.drop(columns=["side"])
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        get_bins(ev, close)


@pytest.mark.parametrize("bad", [[1.0, 2.0], np.zeros((2, 2))])
def test_pandas_only_labeling_helpers_reject_non_pandas_input(bad) -> None:
    idx = pd.date_range("2024-01-01", periods=4)
    close = pd.Series([10.0, 10.5, 10.2, 10.8], index=idx)
    events = pd.DataFrame(
        {"t1": idx[1:], "trgt": [0.01] * 3, "side": [1.0] * 3},
        index=idx[:3],
    )
    with pytest.raises(TypeError, match="pandas"):
        get_daily_vol(bad)
    with pytest.raises(TypeError, match="events"):
        get_bins(bad, close)
    with pytest.raises(TypeError, match="close"):
        get_bins(events, bad)
    with pytest.raises(TypeError, match="events"):
        meta_labels(bad, close, side=pd.Series([1.0] * 3, index=events.index))
    with pytest.raises(TypeError, match="close"):
        meta_labels(events, bad, side=pd.Series([1.0] * 3, index=events.index))


# ---------------------------------------------------------------------------
# Timezone mismatch between target and close
# ---------------------------------------------------------------------------


def test_get_events_rejects_naive_target_with_aware_close():
    close = _random_walk(60, seed=11).tz_localize("America/New_York")
    tv = add_vertical_barrier(close.index[:3], close, 2)
    target = pd.Series(0.02, index=pd.bdate_range("2019-01-01", periods=60))
    with pytest.raises(ValueError, match="timezone") as info:
        get_events(close, close.index[:3], (1, 1), target, 0.01, vertical_barrier_times=tv)
    assert "America/New_York" in str(info.value)
    assert "None" in str(info.value)


# ---------------------------------------------------------------------------
# Timezone of the index is kept on barrier and t1 values
# ---------------------------------------------------------------------------


def test_vertical_barrier_and_events_keep_index_timezone():
    close = _random_walk(60, seed=12).tz_localize("America/New_York")
    events = close.index[:20:5]
    vt = add_vertical_barrier(events, close, 10)
    assert vt.dtype == close.index.dtype
    target = pd.Series(0.02, index=close.index)
    ev = get_events(close, events, (1, 1), target, 0.01, vertical_barrier_times=vt)
    assert ev["t1"].dtype == close.index.dtype
    bins = get_bins(ev, close)
    assert len(bins) == len(events)


# ---------------------------------------------------------------------------
# Vertical barrier between bars
# ---------------------------------------------------------------------------


def test_vertical_barrier_between_bars_is_priced_at_the_last_bar_before_it():
    c = _random_walk(40, seed=13)
    t0 = c.index[0]
    between = pd.Series([c.index[10] + pd.Timedelta(hours=12)], index=[t0])
    on_grid = pd.Series([c.index[10]], index=[t0])
    trgt = pd.Series(10.0, index=c.index)
    with warnings.catch_warnings():
        warnings.simplefilter("error", UserWarning)
        ev = get_events(c, [t0], (10, 10), trgt, 0.01, vertical_barrier_times=between)
        ev_grid = get_events(c, [t0], (10, 10), trgt, 0.01, vertical_barrier_times=on_grid)
    assert ev.loc[t0, "t1"] == c.index[10] + pd.Timedelta(hours=12)
    expected = c.iloc[10] / c.iloc[0] - 1.0
    assert get_bins(ev, c).loc[t0, "ret"] == pytest.approx(expected)
    assert get_bins(ev_grid, c).loc[t0, "ret"] == pytest.approx(expected)


def test_get_bins_still_drops_a_t1_after_the_last_bar():
    c = _random_walk(40, seed=13)
    ev = pd.DataFrame({"t1": [c.index[-1] + pd.Timedelta(days=1)]}, index=[c.index[0]])
    assert get_bins(ev, c).empty


# ---------------------------------------------------------------------------
# Input order does not change daily volatility
# ---------------------------------------------------------------------------


def test_get_daily_vol_is_independent_of_input_order():
    s = _random_walk(30, seed=14)
    shuffled = s.iloc[np.random.default_rng(1).permutation(len(s))]
    expected = get_daily_vol(s, 5).reindex(shuffled.index)
    pd.testing.assert_series_equal(get_daily_vol(shuffled, 5), expected)


@pytest.mark.parametrize("bad", [np.nan, np.inf, -np.inf])
def test_get_daily_vol_rejects_non_finite_prices(bad):
    idx = pd.bdate_range("2024-01-01", periods=30)
    px = np.linspace(100.0, 110.0, 30)
    px[12] = bad
    with pytest.raises(ValueError, match="close"):
        get_daily_vol(pd.Series(px, index=idx))


def test_get_daily_vol_allows_leading_nan_before_first_price():
    # A column padded with NaN before listing: the NaN prefix is skipped, not rejected.
    rng = np.random.default_rng(0)
    idx = pd.bdate_range("2020-01-01", periods=300)
    c = pd.Series(100 * np.exp(np.cumsum(rng.normal(0, 0.01, 300))), index=idx)
    c.iloc[:20] = np.nan
    vol = get_daily_vol(c)
    assert np.all(np.isfinite(vol.iloc[-200:].to_numpy()))
    assert np.isfinite(vol.iloc[-1])


def test_get_daily_vol_interior_nan_after_listing_still_raises():
    rng = np.random.default_rng(0)
    idx = pd.bdate_range("2020-01-01", periods=300)
    c = pd.Series(100 * np.exp(np.cumsum(rng.normal(0, 0.01, 300))), index=idx)
    c.iloc[:20] = np.nan
    c.iloc[150] = np.nan
    with pytest.raises(ValueError, match="finite"):
        get_daily_vol(c)


def test_get_daily_vol_duplicate_timestamps_raise_clear_error():
    idx = pd.bdate_range("2024-01-01", periods=30)
    s = pd.Series(np.linspace(100.0, 110.0, 30), index=idx)
    dup = pd.concat([s, s.iloc[-3:]]).sort_index(kind="stable")
    with pytest.raises(ValueError, match="duplicate timestamps"):
        get_daily_vol(dup)


@pytest.mark.parametrize("bad_price", [0.0, -5.0])
def test_get_daily_vol_rejects_non_positive_prices(bad_price: float) -> None:
    # A zero price made the return inf, which ewm silently skipped, and a negative price gave finite
    # values. get_events and get_bins already reject these.
    idx = pd.date_range("2020-01-01", periods=10, freq="D")
    px = pd.Series(np.arange(1.0, 11.0), idx)
    px.iloc[3] = bad_price
    with pytest.raises(ValueError, match="strictly positive"):
        get_daily_vol(px, span=3)


@pytest.mark.parametrize("bad_min_ret", [float("nan"), float("inf")])
def test_get_events_rejects_non_finite_min_ret(bad_min_ret: float) -> None:
    # A NaN min_ret made "target > min_ret" False for every event, so all events were dropped
    # silently. Non-finite pt_sl values are already rejected.
    from quantowl.labeling import get_events

    idx = pd.date_range("2020-01-01", periods=10, freq="D")
    close = pd.Series(np.arange(100.0, 110.0), idx)
    with pytest.raises(ValueError, match="min_ret"):
        get_events(close, idx[:3], [1, 1], pd.Series(0.05, index=idx[:3]), bad_min_ret)


def test_trend_scanning_rejects_non_integral_window() -> None:
    # A window of 3.9 was silently truncated to 3.
    from quantowl.labeling import trend_scanning_labels

    cs = pd.Series(np.exp(np.linspace(0, 1, 10)), pd.date_range("2020-01-01", periods=10))
    with pytest.raises(ValueError, match="window"):
        trend_scanning_labels(cs, [3.9])


def test_get_events_side_none_uses_pt_sl_zero_for_the_stop() -> None:
    # With side omitted, the stop width came from pt_sl[1], not pt_sl[0].
    from quantowl.labeling import get_events

    idx = pd.date_range("2020-01-01", periods=6, freq="D")
    close = pd.Series([100.0, 98.0, 97.0, 96.0, 95.0, 94.0], index=idx)
    tgt = pd.Series(0.01, index=idx)
    out = get_events(close, idx[:1], (1.0, 3.0), tgt, 0.0)
    # Stop width is 1 * 0.01, so the -2% move on bar 1 stops the event out.
    assert out.loc[idx[0], "t1"] == idx[1]
    assert out.loc[idx[0], "barrier"] == "sl"




def test_get_events_rejects_non_positive_target_when_min_ret_is_negative() -> None:
    # A negative target passed the filter trgt > min_ret and gave a stop on bar 1.
    from quantowl.labeling import get_events

    idx = pd.date_range("2020-01-01", periods=6, freq="D")
    close = pd.Series([100.0, 98.0, 97.0, 96.0, 95.0, 94.0], index=idx)
    with pytest.raises(ValueError, match="positive"):
        get_events(close, idx[:1], (1.0, 1.0), pd.Series(-0.5, index=idx), -1.0)


def test_trend_scanning_rejects_string_window_sizes() -> None:
    # A string is iterable, so '345' was read as the windows 3, 4 and 5.
    s = pd.Series(np.linspace(100, 120, 50))
    with pytest.raises(ValueError, match="window"):
        trend_scanning_labels(s, "345")


def test_trend_scanning_rejects_non_finite_threshold() -> None:
    # A NaN t_threshold labelled every bar 0 with no error.
    s = pd.Series(np.r_[np.linspace(100, 120, 20), np.linspace(120, 100, 20)])
    with pytest.raises(ValueError, match="t_threshold"):
        trend_scanning_labels(s, [5, 10], t_threshold=float("nan"))


def test_get_daily_vol_rejects_nat_label() -> None:
    # A NaT timestamp received a finite volatility.
    s = pd.Series([1.0, 2.0, 3.0], index=pd.DatetimeIndex(["2020-01-01", pd.NaT, "2020-01-03"]))
    with pytest.raises(ValueError, match="NaT"):
        get_daily_vol(s)


@pytest.mark.parametrize("bad", [-2, float("nan"), float("inf"), 1e15])
def test_add_vertical_barrier_rejects_bad_num_days(bad: float) -> None:
    # A negative num_days gave expiries before the event, and NaN, inf and huge values leaked numpy,
    # pandas or OverflowError messages.
    idx = pd.date_range("2020-01-01", periods=6, freq="D")
    s = pd.Series(range(6), index=idx)
    with pytest.raises(ValueError, match="num_days"):
        add_vertical_barrier(idx[3:5], s, num_days=bad)


def test_get_events_side_none_rejects_one_zero_barrier() -> None:
    # With side omitted both barriers use pt_sl[0], so (0, 2) silently switched the stop off and
    # (2, 0) put a stop at 2. Those one-sided specs are refused by name.
    from quantowl.labeling import get_events

    idx = pd.date_range("2020-01-01", periods=6, freq="D")
    close = pd.Series([100.0, 98.0, 97.0, 96.0, 95.0, 94.0], index=idx)
    tgt = pd.Series(0.01, index=idx)
    with pytest.raises(ValueError, match="pt_sl"):
        get_events(close, idx[:1], (0, 2), tgt, 0.0)
    with pytest.raises(ValueError, match="pt_sl"):
        get_events(close, idx[:1], (2, 0), tgt, 0.0)
    # With side given, pt_sl[1] is the stop, so a zero stop is a real stop-free spec.
    side = pd.Series(1, index=idx[:1])
    out = get_events(close, idx[:1], (2, 0), tgt, 0.0, side=side)
    assert len(out) == 1


def test_trend_scanning_accepts_one_shot_window_iterable() -> None:
    # The integer-window loop consumed a generator, so the windows read after it were empty and the
    # call raised. A one-shot iterable gives the same result as a list.
    from quantowl.labeling import trend_scanning_labels

    c = pd.Series([100, 100, 95, 100, 101, 102, 103, 104.0], index=pd.date_range("2020-01-01", periods=8))
    want = trend_scanning_labels(c, [3, 4])
    got = trend_scanning_labels(c, (w for w in [3, 4]))
    pd.testing.assert_frame_equal(got, want)


def test_get_events_accepts_negative_infinite_min_ret() -> None:
    # Every non-finite min_ret used to be refused. Only NaN and +inf are refused now (both drop every
    # event, which is what the guard is for). -inf drops none.
    from quantowl.labeling import get_events

    idx = pd.date_range("2020-01-01", periods=8, freq="D")
    c = pd.Series([100, 100, 95, 100, 101, 102, 103, 104.0], index=idx)
    out = get_events(c, idx[:1], (1, 1), pd.Series(0.01, index=idx), float("-inf"))
    assert out["barrier"].tolist() == ["sl"]


def test_add_vertical_barrier_rejects_unsorted_close_index() -> None:
    # searchsorted ran on an unsorted grid and gave expiries at the end of the series. get_events
    # already refuses an unsorted close, so this function refuses it too.
    from quantowl.labeling import add_vertical_barrier

    idx = pd.date_range("2020-01-01", periods=10, freq="D")
    c = pd.Series(range(100, 110), index=idx, dtype=float)
    ev = idx[:2]
    with pytest.raises(ValueError, match="increasing"):
        add_vertical_barrier(ev, c.iloc[::-1], 2)
    assert add_vertical_barrier(ev, c, 2).tolist() == [idx[2], idx[3]]


def test_get_events_names_a_duplicate_target_index() -> None:
    # A target with a repeated label made reindex raise pandas' duplicate-label error. close and
    # t_events already get a named message, so target gets one too.
    from quantowl.labeling import get_events

    idx = pd.date_range("2020-01-01", periods=8, freq="D")
    c = pd.Series([100, 100, 95, 100, 101, 102, 103, 104.0], index=idx)
    dup = pd.Series([0.01] * 8, index=[idx[0]] * 8)
    with pytest.raises(ValueError, match="target index must be unique"):
        get_events(c, idx[:1], (1, 1), dup, 0.0)


def _ev_fixture():
    idx = pd.date_range("2020-01-01", periods=50, freq="D")
    rng = np.random.default_rng(0)
    close = pd.Series(100 * np.exp(np.cumsum(rng.normal(0, 0.01, 50))), index=idx)
    return idx, close


def test_get_events_warns_when_pt_sl_1_is_ignored_without_side() -> None:
    # With side omitted, pt_sl[0] sets both barriers (documented), so pt_sl = (1, 2) ran with a stop
    # of 1 and no sign of pt_sl[1]. The value is kept; the call now warns.
    from quantowl.labeling import get_events

    idx, close = _ev_fixture()
    with pytest.warns(UserWarning, match=r"pt_sl\[1\] is ignored"):
        out = get_events(close, idx[:20], [1, 2], pd.Series(0.02, index=idx), 0.0)
    assert out["barrier"].value_counts().to_dict() == {"pt": 12, "sl": 8}


def test_get_events_refuses_zero_width_targets_under_negative_min_ret() -> None:
    # A zero width is refused under a negative min_ret: a retained zero width would stop out on its
    # first bar. The docstring now says so.
    from quantowl.labeling import get_events

    idx, close = _ev_fixture()
    target = pd.Series(0.02, index=idx)
    target.iloc[3] = 0.0
    assert len(get_events(close, idx[:20], [1, 1], target, 0.0)) == 19
    with pytest.raises(ValueError, match="positive"):
        get_events(close, idx[:20], [1, 1], target, -0.5)
    with pytest.raises(ValueError, match="positive"):
        get_events(close, idx[:20], [1, 1], target, -np.inf)


def test_add_vertical_barrier_refuses_duplicate_grid_timestamps() -> None:
    # The docstring says "strictly increasing" for close, but a duplicated timestamp passed
    # is_monotonic_increasing, so expiries were computed on a grid get_events then refuses.
    from quantowl.labeling import add_vertical_barrier

    idx = pd.date_range("2020-01-01", periods=10)
    c = pd.Series(100.0 + np.arange(10.0), index=idx)
    dup = pd.concat([c.iloc[:5], c.iloc[4:9]])
    with pytest.raises(ValueError, match="strictly increasing"):
        add_vertical_barrier(c.index[:3], dup, 1)


def test_trend_scanning_refuses_negative_threshold() -> None:
    # The threshold bounds |t|, so a negative one labelled every bar +1 or -1 and none 0. It must be
    # non-negative.
    from quantowl.labeling import trend_scanning_labels

    idx = pd.date_range("2020-01-01", periods=50, freq="D")
    close = pd.Series(100 * np.exp(np.cumsum(np.random.default_rng(0).normal(0, 0.01, 50))), index=idx)
    with pytest.raises(ValueError, match="non-negative"):
        trend_scanning_labels(close, [3], -1.0)




def test_trend_scanning_refuses_string_window_sizes() -> None:
    # float('3') parsed the string, so ['3'] ran as window 3.
    from quantowl.labeling import trend_scanning_labels

    idx = pd.date_range("2020-01-01", periods=50, freq="D")
    close = pd.Series(100 * np.exp(np.cumsum(np.random.default_rng(0).normal(0, 0.01, 50))), index=idx)
    with pytest.raises(ValueError, match="integers"):
        trend_scanning_labels(close, ["3"])


def test_get_events_side_with_missing_rows_names_the_coverage() -> None:
    # A side covering 5 of 9 events was reported as a bad value ("must equal -1 or +1"), when the
    # index is what is missing. The message now names the coverage.
    import pandas as pd

    from quantowl import labeling as L

    idx = pd.date_range("2020-01-01", periods=60, freq="D")
    rng = np.random.default_rng(0)
    close = pd.Series(100 * np.exp(np.cumsum(rng.normal(0, 0.01, 60))), idx)
    t_events = idx[5:50:5]
    with pytest.raises(ValueError, match="covers 5 of 9 events"):
        L.get_events(close, t_events, [1, 1], pd.Series(0.02, idx), 0.0, side=pd.Series(1.0, t_events[:5]))


def test_get_events_refuses_one_zero_pt_sl_with_side_omitted_as_intended() -> None:
    # With side omitted, a one-zero pt_sl is refused, and a non-zero pt_sl[1] is ignored with a
    # warning. Both rules are intended.
    import pandas as pd

    from quantowl import labeling as L

    idx = pd.date_range("2020-01-01", periods=60, freq="D")
    rng = np.random.default_rng(0)
    close = pd.Series(100 * np.exp(np.cumsum(rng.normal(0, 0.01, 60))), idx)
    t_events = idx[5:50:5]
    vb = L.add_vertical_barrier(t_events, close, 3)
    for pt_sl in ([1, 0], [0, 1]):
        with pytest.raises(ValueError, match="side omitted"):
            L.get_events(close, t_events, pt_sl, pd.Series(0.02, idx), 0.0, vb)
    with pytest.warns(UserWarning):
        assert len(L.get_events(close, t_events, [2, 1], pd.Series(0.02, idx), 0.0, vb)) == 9


def test_get_events_refuses_non_datetime_close_index() -> None:
    # An integer close index was read as nanoseconds, so t1 came out as 1970 timestamps and get_bins
    # failed later with a raw TypeError. get_daily_vol refuses it.
    idx = pd.RangeIndex(30)
    close = pd.Series(100.0 + np.arange(30.0), index=idx)
    tgt = pd.Series(0.02, index=idx)
    with pytest.raises(ValueError, match="DatetimeIndex"):
        get_events(close, idx[:5], [1, 1], tgt, 0.0)


def test_get_events_refuses_target_that_labels_no_event() -> None:
    # A target stamped at noon against midnight events reindexed to all NaN, so every event was
    # dropped with no message.
    from quantowl.labeling import get_events

    idx = pd.date_range("2020-01-01", periods=30, freq="D")
    close = pd.Series(100.0 + np.arange(30.0), index=idx)
    tgt = pd.Series(0.02, index=idx + pd.Timedelta(hours=12))
    with pytest.raises(ValueError, match="target"):
        get_events(close, idx[:10], [1, 1], tgt, 0.0)


def test_get_events_refuses_vertical_barriers_that_label_no_event() -> None:
    # Vertical barriers with a default RangeIndex lost every expiry silently.
    from quantowl.labeling import add_vertical_barrier, get_events

    idx = pd.date_range("2020-01-01", periods=30, freq="D")
    close = pd.Series(100.0 + np.arange(30.0), index=idx)
    tgt = pd.Series(0.02, index=idx)
    vb = add_vertical_barrier(idx[:10], close, 3)
    with pytest.raises(ValueError, match="vertical_barrier_times"):
        get_events(close, idx[:10], [1, 1], tgt, 0.0, vertical_barrier_times=pd.Series(vb.to_numpy()))


def test_get_events_names_timezone_for_naive_vertical_index_with_aware_events() -> None:
    close = _random_walk(60, seed=13).tz_localize("Europe/Berlin")
    events = close.index[:3]
    target = pd.Series(0.02, index=events)
    naive_vertical = pd.Series(close.index[10:13].tz_localize(None), index=events.tz_localize(None))
    with pytest.raises(ValueError, match="timezone") as info:
        get_events(close, events, (1, 1), target, 0.01, vertical_barrier_times=naive_vertical)
    assert "vertical_barrier_times" in str(info.value)
    assert "Europe/Berlin" in str(info.value)


def test_get_events_names_the_timezone_when_t_events_and_close_differ() -> None:
    c = _random_walk(40, seed=13)
    aware = c.tz_localize("America/New_York")
    trgt = pd.Series(0.01, index=aware.index)
    with pytest.raises(ValueError, match="timezone"):
        get_events(aware, aware.index[:5].tz_localize(None), (1, 1), trgt, 0.0)
    with pytest.raises(ValueError, match="timezone"):
        get_events(c, aware.index[:5], (1, 1), pd.Series(0.01, index=c.index), 0.0)


def test_get_bins_warns_when_every_event_is_dropped() -> None:
    idx, close = _ev_fixture()
    events = pd.DataFrame({"t1": pd.NaT, "trgt": 0.02}, index=idx[:5])
    events["t1"] = pd.to_datetime(events["t1"])
    with pytest.warns(UserWarning, match="empty"):
        out = get_bins(events, close)
    assert out.empty
    good = pd.DataFrame({"t1": idx[5:10]}, index=idx[:5])
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        assert not get_bins(good, close).empty


@pytest.mark.parametrize("num_days", [0.5, 0.7, 1 / 7, 1, 2, 3])
def test_fractional_num_days_on_us_index_matches_ns_index(num_days: float) -> None:
    # Expiry is built in ns; searchsorted on a microsecond grid used to refuse the lossy cast.
    grid_ns = pd.date_range("2020-01-01 09:00", periods=60, freq="7h31min", unit="ns")
    events = grid_ns[:5]
    close_ns = pd.Series(np.arange(60.0), index=grid_ns)
    close_us = pd.Series(np.arange(60.0), index=grid_ns.as_unit("us"))
    ref = add_vertical_barrier(events, close_ns, num_days=num_days)
    got = add_vertical_barrier(events.as_unit("us"), close_us, num_days=num_days)
    np.testing.assert_array_equal(
        got.to_numpy().astype("datetime64[ns]"), ref.to_numpy().astype("datetime64[ns]")
    )
