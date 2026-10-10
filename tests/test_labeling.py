"""Tests for finlab.labeling (AFML Chapter 3 and trend scanning).

Each fast function is checked against a brute-force reference written
independently in this file: the triple barrier against a pandas path loop in
the style of Snippet 3.2, daily volatility against a per-bar loop, and trend
scanning against scipy.stats.linregress.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from scipy import stats

from finlab.labeling import (
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
        t_sl = rel[rel < -pt_sl[1] * trg].index.min() if pt_sl[1] > 0 else pd.NaT
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
