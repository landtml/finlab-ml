"""Tests for quantowl.bars (AFML ch. 2).

Every fast (numba) bar and CUSUM routine is cross-checked against a naive,
pure-Python reference on seeded random inputs. The naive imbalance and runs
references compute each EWMA as a closed form over the full history
(``E_H = (1-a)^H E_0 + sum_j a (1-a)^{H-1-j} x_j``), so they do not share code
with the recursive numba updates they check.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from quantowl.bars import (
    OHLCV_COLUMNS,
    _tick_rule,
    cusum_filter,
    dollar_bars,
    imbalance_bars,
    run_bars,
    tick_bars,
    time_bars,
    volume_bars,
)


# --------------------------------------------------------------------------- #
# Naive references (pure Python)
# --------------------------------------------------------------------------- #


def naive_threshold_ends(x: np.ndarray, thr: float) -> list[int]:
    ends, acc = [], 0.0
    for t, xv in enumerate(x):
        acc += xv
        if acc >= thr:
            ends.append(t)
            acc = 0.0
    return ends


def naive_ohlcv(p: np.ndarray, v: np.ndarray, ends: list[int]) -> np.ndarray:
    rows, start = [], 0
    for e in ends:
        sp = [float(q) for q in p[start : e + 1]]
        sv = [float(q) for q in v[start : e + 1]]
        dollar = sum(a * b for a, b in zip(sp, sv))
        rows.append([sp[0], max(sp), min(sp), sp[-1], sum(sv), dollar, e - start + 1])
        start = e + 1
    return np.array(rows, dtype=np.float64).reshape(-1, 7)


def naive_tick_rule(p: np.ndarray, b0: float) -> list[float]:
    b = [b0]
    for t in range(1, len(p)):
        dp = p[t] - p[t - 1]
        if dp == 0:
            b.append(b[-1])
        else:
            b.append(1.0 if dp > 0 else -1.0)
    return b


def closed_form_ewma(init: float, xs: list[float], alpha: float) -> float:
    """E after the EWMA recursion E <- alpha*x + (1-alpha)*E over ``xs``, from ``init``."""
    H = len(xs)
    e = (1.0 - alpha) ** H * init
    for j, x in enumerate(xs):
        e += alpha * (1.0 - alpha) ** (H - 1 - j) * x
    return e


def _alpha(span: float) -> float:
    return 2.0 / (span + 1.0)


def naive_imbalance(
    p: np.ndarray,
    mag: np.ndarray,
    init_T: int,
    span_bars: float,
    span_ticks: float,
    b0: float,
) -> list[int]:
    n = len(p)
    b = naive_tick_rule(p, b0)
    s = [b[t] * mag[t] for t in range(n)]
    m = min(init_T, n)
    e_s0 = sum(s[:m]) / m
    a_bars, a_ticks = _alpha(span_bars), _alpha(span_ticks)
    lengths: list[int] = []
    hist_s: list[float] = []  # signed values of all ticks in closed bars

    def thresholds() -> float:
        e_T = closed_form_ewma(float(init_T), [float(L) for L in lengths], a_bars)
        e_s = closed_form_ewma(e_s0, hist_s, a_ticks)
        return e_T * abs(e_s)

    ends: list[int] = []
    start, theta = 0, 0.0
    thr = thresholds()
    for t in range(n):
        theta += s[t]
        if abs(theta) >= thr:
            ends.append(t)
            lengths.append(t - start + 1)
            hist_s.extend(s[start : t + 1])
            start, theta = t + 1, 0.0
            thr = thresholds()
    return ends


def naive_runs(
    p: np.ndarray,
    mag: np.ndarray,
    init_T: int,
    span_bars: float,
    span_ticks: float,
    b0: float,
) -> list[int]:
    n = len(p)
    b = naive_tick_rule(p, b0)
    buy = [mag[t] if b[t] > 0 else 0.0 for t in range(n)]
    sell = [mag[t] if b[t] < 0 else 0.0 for t in range(n)]
    m = min(init_T, n)
    e_b0 = sum(buy[:m]) / m
    e_s0 = sum(sell[:m]) / m
    a_bars, a_ticks = _alpha(span_bars), _alpha(span_ticks)
    lengths: list[int] = []
    hist_b: list[float] = []
    hist_s: list[float] = []

    def thresholds() -> float:
        e_T = closed_form_ewma(float(init_T), [float(L) for L in lengths], a_bars)
        e_b = closed_form_ewma(e_b0, hist_b, a_ticks)
        e_s = closed_form_ewma(e_s0, hist_s, a_ticks)
        return e_T * max(e_b, e_s)

    ends: list[int] = []
    start = 0
    run_b = run_s = 0.0
    thr = thresholds()
    for t in range(n):
        run_b += buy[t]
        run_s += sell[t]
        if max(run_b, run_s) >= thr:
            ends.append(t)
            lengths.append(t - start + 1)
            hist_b.extend(buy[start : t + 1])
            hist_s.extend(sell[start : t + 1])
            start = t + 1
            run_b = run_s = 0.0
            thr = thresholds()
    return ends


def naive_cusum(x: np.ndarray, h: float) -> list[int]:
    """Snippet 2.4 transcribed with explicit loops (positions in ``x``)."""
    events = []
    s_pos, s_neg = 0.0, 0.0
    for i in range(1, len(x)):
        d = x[i] - x[i - 1]
        s_pos, s_neg = max(0.0, s_pos + d), min(0.0, s_neg + d)
        if s_neg < -h:
            s_neg = 0.0
            events.append(i)
        elif s_pos > h:
            s_pos = 0.0
            events.append(i)
    return events


# --------------------------------------------------------------------------- #
# Synthetic data
# --------------------------------------------------------------------------- #


def make_ticks(seed: int, n: int = 1500, drift: float = 0.0):
    """Seeded tick data with price ties (rounded to cents) and integer volumes."""
    rng = np.random.default_rng(seed)
    steps = rng.normal(drift, 0.05, n)
    p = np.round(100.0 + np.cumsum(steps), 2)
    p = np.where(p <= 0.5, 0.5, p)
    v = rng.integers(1, 200, n).astype(np.float64)
    return p, v


# --------------------------------------------------------------------------- #
# Standard bars
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_tick_bars_match_naive(seed: int) -> None:
    p, v = make_ticks(seed)
    for thr in (7, 50):
        bars = tick_bars(p, thr, volumes=v)
        ends = naive_threshold_ends(np.ones(len(p)), thr)
        np.testing.assert_array_equal(bars.index.to_numpy(), np.array(ends, dtype=np.int64))
        np.testing.assert_allclose(
            bars[list(OHLCV_COLUMNS)].to_numpy(),
            naive_ohlcv(p, v, ends),
            rtol=1e-12,
            atol=0,
        )


@pytest.mark.parametrize("seed", [0, 3])
def test_volume_and_dollar_bars_match_naive(seed: int) -> None:
    p, v = make_ticks(seed)
    vb = volume_bars(p, v, 5_000.0)
    ends_v = naive_threshold_ends(v, 5_000.0)
    np.testing.assert_array_equal(vb.index.to_numpy(), np.array(ends_v))
    np.testing.assert_allclose(
        vb[list(OHLCV_COLUMNS)].to_numpy(), naive_ohlcv(p, v, ends_v), rtol=1e-12
    )

    db = dollar_bars(p, v, 300_000.0)
    ends_d = naive_threshold_ends(p * v, 300_000.0)
    np.testing.assert_array_equal(db.index.to_numpy(), np.array(ends_d))
    np.testing.assert_allclose(
        db[list(OHLCV_COLUMNS)].to_numpy(), naive_ohlcv(p, v, ends_d), rtol=1e-12
    )


def test_standard_bars_hit_threshold_exactly() -> None:
    p, v = make_ticks(7)
    bars = tick_bars(p, 25, volumes=v)
    assert (bars["ticks"] == 25).all()
    assert len(bars) == len(p) // 25  # the partial trailing bar is dropped
    vb = volume_bars(p, v, 2_000.0)
    assert (vb["volume"] >= 2_000.0).all()
    # Removing the last tick of each bar would drop volume below threshold.
    assert (vb["volume"] - v[vb.index.to_numpy()] < 2_000.0).all()


def test_bar_invariants() -> None:
    p, v = make_ticks(11)
    bars = dollar_bars(p, v, 200_000.0)
    assert (bars["high"] >= bars[["open", "close"]].max(axis=1) - 1e-12).all()
    assert (bars["low"] <= bars[["open", "close"]].min(axis=1) + 1e-12).all()
    closed_ticks = int(bars["ticks"].sum())
    assert closed_ticks == int(bars.index.to_numpy()[-1]) + 1
    np.testing.assert_allclose(
        bars["volume"].sum(), v[:closed_ticks].sum(), rtol=1e-12
    )


def test_time_bars_right_closed_and_dropping_empty() -> None:
    idx = pd.to_datetime(
        ["2024-01-01 00:00:10", "2024-01-01 00:00:59", "2024-01-01 00:01:00", "2024-01-01 00:03:30"]
    )
    s = pd.Series([10.0, 11.0, 12.0, 13.0], index=idx)
    vol = pd.Series([1.0, 2.0, 3.0, 4.0], index=idx)
    out = time_bars(s, vol, freq="1min")
    # Intervals (00:00, 00:01] and (00:03, 00:04]; the two empty minutes are dropped.
    assert list(out.index) == [pd.Timestamp("2024-01-01 00:01:00"), pd.Timestamp("2024-01-01 00:04:00")]
    assert out.loc[pd.Timestamp("2024-01-01 00:01:00"), "open"] == 10.0
    assert out.loc[pd.Timestamp("2024-01-01 00:01:00"), "close"] == 12.0
    assert out.loc[pd.Timestamp("2024-01-01 00:01:00"), "ticks"] == 3
    assert out.loc[pd.Timestamp("2024-01-01 00:04:00"), "volume"] == 4.0


def test_time_bars_warns_that_trailing_interval_is_kept() -> None:
    idx = pd.to_datetime(["2020-01-02 10:00:10", "2020-01-02 10:00:50", "2020-01-02 10:01:20"])
    s = pd.Series([1.0, 2.0, 3.0], index=idx)
    with pytest.warns(UserWarning, match="not closed"):
        out = time_bars(s, freq="1min")
    # The trailing (10:01, 10:02] interval holds one tick and is kept, not dropped.
    assert list(out.index) == [pd.Timestamp("2020-01-02 10:01:00"), pd.Timestamp("2020-01-02 10:02:00")]
    assert out["ticks"].tolist() == [2, 1]


def test_numpy_input_gives_positional_index() -> None:
    p, v = make_ticks(4, n=200)
    bars = tick_bars(p, 10)
    assert isinstance(bars.index, pd.RangeIndex) or bars.index.dtype.kind == "i"
    assert list(bars.index[:2]) == [9, 19]
    assert bars["volume"].isna().all()  # no volumes given


def test_series_index_is_used_as_bar_timestamp() -> None:
    p, v = make_ticks(5, n=120)
    idx = pd.date_range("2024-03-01", periods=120, freq="s")
    bars = tick_bars(pd.Series(p, index=idx), 30)
    assert list(bars.index) == [idx[29], idx[59], idx[89], idx[119]]


def test_unsorted_series_gives_the_same_bars_as_sorted() -> None:
    p, v = make_ticks(13, n=600)
    idx = pd.date_range("2024-03-01", periods=600, freq="s")
    perm = np.random.default_rng(13).permutation(600)
    sorted_bars = tick_bars(pd.Series(p, index=idx), 50, volumes=pd.Series(v, index=idx))
    shuffled = tick_bars(
        pd.Series(p[perm], index=idx[perm]),
        50,
        volumes=pd.Series(v[perm], index=idx[perm]),
    )
    assert shuffled.index.is_monotonic_increasing
    pd.testing.assert_frame_equal(shuffled, sorted_bars)


def test_time_bars_pair_each_tick_with_its_own_timestamp() -> None:
    idx = pd.to_datetime(["2020-01-02 10:00:50", "2020-01-02 10:00:10", "2020-01-02 10:00:30"])
    s = pd.Series([4.0, 1.0, 2.0], index=idx)
    expected_index = pd.DatetimeIndex(["2020-01-02 10:01:00"])
    with pytest.warns(UserWarning, match="not closed"):
        bars = time_bars(s, None, "1min")
    assert list(bars.index) == list(expected_index)
    assert bars["open"].iloc[0] == 1.0
    assert bars["high"].iloc[0] == 4.0
    assert bars["low"].iloc[0] == 1.0
    assert bars["close"].iloc[0] == 4.0
    assert bars["ticks"].iloc[0] == 3


def test_time_bars_volume_follows_unsorted_ticks() -> None:
    idx = pd.to_datetime(["2020-01-02 10:00:50", "2020-01-02 10:00:10", "2020-01-02 10:00:30"])
    s = pd.Series([4.0, 1.0, 2.0], index=idx)
    vol_array = np.array([40.0, 10.0, 20.0])
    vol_series = pd.Series(vol_array, index=idx)
    with pytest.warns(UserWarning, match="not closed"):
        from_array = time_bars(s, vol_array, "1min")
        from_series = time_bars(s, vol_series, "1min")
    for bars in (from_array, from_series):
        assert bars["volume"].iloc[0] == 70.0
        assert bars["dollar_value"].iloc[0] == 1.0 * 10.0 + 2.0 * 20.0 + 4.0 * 40.0
        assert bars.index.is_monotonic_increasing


# --------------------------------------------------------------------------- #
# Tick rule
# --------------------------------------------------------------------------- #


def test_tick_rule_carries_sign_through_zero_changes() -> None:
    p = np.array([1.0, 1.0, 2.0, 2.0, 2.0, 1.0, 1.0])
    b = _tick_rule(p, 1.0)
    np.testing.assert_array_equal(b, [1.0, 1.0, 1.0, 1.0, 1.0, -1.0, -1.0])
    b_neg = _tick_rule(p, -1.0)
    assert b_neg[0] == -1.0 and b_neg[1] == -1.0 and b_neg[2] == 1.0


@pytest.mark.parametrize("seed", [0, 8])
def test_tick_rule_matches_naive(seed: int) -> None:
    p, _ = make_ticks(seed, n=400)
    np.testing.assert_array_equal(_tick_rule(p, 1.0), np.array(naive_tick_rule(p, 1.0)))


# --------------------------------------------------------------------------- #
# Imbalance and runs bars
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("kind", ["tick", "volume", "dollar"])
@pytest.mark.parametrize("seed", [0, 1])
def test_imbalance_bars_match_naive(kind: str, seed: int) -> None:
    p, v = make_ticks(seed, n=1200, drift=0.02)
    init_T, sb, st, b0 = 40, 10.0, 200.0, 1.0
    mag = {"tick": np.ones_like(p), "volume": v, "dollar": p * v}[kind]
    bars = imbalance_bars(
        p, v, kind=kind, init_T=init_T, span_bars=sb, span_ticks=st, b0=b0
    )
    ends = naive_imbalance(p, mag, init_T, sb, st, b0)
    np.testing.assert_array_equal(bars.index.to_numpy(), np.array(ends, dtype=np.int64))
    np.testing.assert_allclose(
        bars[list(OHLCV_COLUMNS)].to_numpy(), naive_ohlcv(p, v, ends), rtol=1e-10
    )


@pytest.mark.parametrize("kind", ["tick", "volume", "dollar"])
@pytest.mark.parametrize("seed", [2, 5])
def test_run_bars_match_naive(kind: str, seed: int) -> None:
    p, v = make_ticks(seed, n=1200, drift=-0.01)
    init_T, sb, st, b0 = 40, 10.0, 200.0, -1.0
    mag = {"tick": np.ones_like(p), "volume": v, "dollar": p * v}[kind]
    bars = run_bars(p, v, kind=kind, init_T=init_T, span_bars=sb, span_ticks=st, b0=b0)
    ends = naive_runs(p, mag, init_T, sb, st, b0)
    np.testing.assert_array_equal(bars.index.to_numpy(), np.array(ends, dtype=np.int64))


def test_imbalance_threshold_is_exact_on_a_monotone_tape() -> None:
    """Rising prices give b_t = 1 everywhere, so E[b] = 1 exactly and E[T] stays at init_T.

    The rule then closes a bar when theta_T = T reaches E[T] * 1, so every bar has
    exactly init_T ticks. Runs bars behave the same way: buys reach E[T] * max(1, 0).
    """
    n, init_T = 1000, 50
    p = 100.0 + np.arange(n) * 0.01
    imb = imbalance_bars(p, kind="tick", init_T=init_T, span_bars=20, span_ticks=100)
    assert (imb["ticks"] == init_T).all()
    runs = run_bars(p, kind="tick", init_T=init_T, span_bars=20, span_ticks=100)
    assert (runs["ticks"] == init_T).all()


def test_imbalance_and_runs_validate_inputs() -> None:
    p, _ = make_ticks(0, n=50)
    with pytest.raises(ValueError):
        imbalance_bars(p, kind="volume")  # needs volumes
    with pytest.raises(ValueError):
        imbalance_bars(p, kind="bogus")
    with pytest.raises(ValueError):
        run_bars(p, init_T=0)
    with pytest.raises(ValueError):
        imbalance_bars(p, span_bars=0.5)
    with pytest.raises(ValueError):
        tick_bars(np.array([1.0, -2.0, 3.0]), 2)


# --------------------------------------------------------------------------- #
# CUSUM filter
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("seed", [0, 4])
def test_cusum_on_levels_matches_snippet_2_4(seed: int) -> None:
    rng = np.random.default_rng(seed)
    x = np.cumsum(rng.normal(0, 0.01, 1500))
    h = 0.05
    idx = pd.date_range("2024-01-01", periods=len(x), freq="min")
    events = cusum_filter(pd.Series(x, index=idx), h)
    expected = naive_cusum(x, h)
    assert len(events) > 0
    np.testing.assert_array_equal(events, idx[expected])
    np.testing.assert_array_equal(cusum_filter(x, h), np.array(expected))


@pytest.mark.parametrize("seed", [6])
def test_cusum_on_returns_includes_first_observation(seed: int) -> None:
    rng = np.random.default_rng(seed)
    y = rng.normal(0, 0.02, 800)
    h = 0.03
    events = cusum_filter(y, h, is_returns=True)
    # Naive reference on returns: the first value is itself a candidate event.
    s_pos = s_neg = 0.0
    expected = []
    for i, d in enumerate(y):
        s_pos, s_neg = max(0.0, s_pos + d), min(0.0, s_neg + d)
        if s_neg < -h:
            s_neg = 0.0
            expected.append(i)
        elif s_pos > h:
            s_pos = 0.0
            expected.append(i)
    np.testing.assert_array_equal(events, np.array(expected, dtype=np.int64))


def test_cusum_rejects_non_finite_and_bad_threshold() -> None:
    with pytest.raises(ValueError):
        cusum_filter(np.array([0.0, np.nan, 1.0]), 0.1)
    with pytest.raises(ValueError):
        cusum_filter(np.array([0.0, 1.0]), 0.0)


def test_cusum_shuffled_series_gives_the_same_events_as_sorted() -> None:
    rng = np.random.default_rng(4)
    ts = pd.date_range("2021-01-04 09:00", periods=400, freq="7s")
    p = pd.Series(100 + np.cumsum(rng.normal(0, 0.1, 400)), index=ts)
    shuffled = p.iloc[rng.permutation(400)]
    expected = cusum_filter(p, 1.0)
    pd.testing.assert_index_equal(cusum_filter(shuffled, 1.0), expected)


def test_cusum_filter_rejects_multi_column_array_and_frame() -> None:
    # A (6, 2) input was flattened to a 12-long series and returned positions.
    arr = np.arange(12.0).reshape(6, 2)
    with pytest.raises(ValueError, match="one column"):
        cusum_filter(arr, 1.0)
    with pytest.raises(ValueError, match="one column"):
        cusum_filter(pd.DataFrame(arr, columns=["a", "b"]), 1.0)


def test_cusum_filter_single_column_input_unchanged() -> None:
    x = np.arange(12.0)
    want = cusum_filter(x, 1.0)
    np.testing.assert_array_equal(cusum_filter(x.reshape(-1, 1), 1.0), want)
    np.testing.assert_array_equal(cusum_filter(pd.DataFrame({"a": x}), 1.0), want)


@pytest.mark.parametrize("bad_init_T", [float("inf"), float("nan")])
def test_imbalance_bars_non_finite_init_T_is_value_error(bad_init_T: float) -> None:
    # int(inf) raised OverflowError, bypassing the usual ValueError.
    from quantowl.bars import imbalance_bars

    with pytest.raises(ValueError, match="init_T"):
        imbalance_bars(np.array([1.0, 2, 1, 2.0]), init_T=bad_init_T)


def test_tick_bars_rejects_fractional_threshold() -> None:
    # A threshold of 2.5 silently built bars of 3 ticks.
    from quantowl.bars import tick_bars

    with pytest.raises(ValueError, match="integer"):
        tick_bars(np.arange(1.0, 11.0), 2.5)



def test_dollar_bars_rejects_overflowing_dollar_product() -> None:
    # price * volume overflowed to inf, and inf showed up in dollar_value.
    from quantowl.bars import dollar_bars

    with pytest.raises(ValueError, match="overflow"):
        dollar_bars(np.array([1e300] * 3), np.array([1e10] * 3), 1e300)


def test_time_bars_rejects_zero_length_freq() -> None:
    # freq="0min" reached pandas' resample and raised ZeroDivisionError. A bad frequency string
    # already gives a ValueError, and a zero-length one gets the same.
    idx = pd.date_range("2024-01-01", periods=4, freq="20s")
    with pytest.raises(ValueError, match="frequency"):
        time_bars(pd.Series([1.0, 2.0, 3.0, 4.0], index=idx), None, "0min")


def test_cusum_filter_one_column_dataframe_returns_labels() -> None:
    # A one-column DataFrame returned integer positions, while a Series returns its index labels,
    # which is what the docstring promises for a labelled input.
    from quantowl.bars import cusum_filter

    rw = np.cumsum(np.random.default_rng(5).standard_normal(60)) + 10
    idx = pd.date_range("2024-01-01", periods=60, freq="D")
    got = cusum_filter(pd.DataFrame({"a": rw}, index=idx), 0.5)
    want = cusum_filter(pd.Series(rw, index=idx), 0.5)
    pd.testing.assert_index_equal(pd.Index(got), pd.Index(want))


def test_bar_sums_that_overflow_are_refused() -> None:
    # A per-tick price * volume overflow was refused, but the bar sums were not checked, so a bar
    # dollar value or volume of inf was returned with no error. Every bar kind and time_bars now
    # refuses it. Only inf is tested: NaN volume is the valid "no volume" case.
    from quantowl.bars import dollar_bars, time_bars, volume_bars

    with pytest.raises(ValueError, match="overflows"):
        dollar_bars(np.array([1e300, 1e300]), np.array([1e8, 1e8]), 1.7e308)
    with pytest.raises(ValueError, match="overflows"):
        volume_bars(np.array([1.0, 1.0]), np.array([1e308, 1e308]), 1.7e308)
    idx = pd.date_range("2024-01-01", periods=2, freq="min")
    with pytest.raises(ValueError, match="overflows"):
        time_bars(pd.Series([1e200, 1.0], index=idx), pd.Series([1e200, 1.0], index=idx), "1min")


def test_volume_bars_are_not_refused_for_dollar_only_overflow() -> None:
    # The bar-sum check tested the dollar column for volume bars, which never use it. Four 1-tick
    # bars with volume 1e200 (and dollar inf) are now returned.
    from quantowl.bars import volume_bars

    out = volume_bars(np.full(4, 1e200), np.full(4, 1e200), threshold=1e200)
    assert out["volume"].tolist() == [1e200] * 4
    assert out["ticks"].tolist() == [1, 1, 1, 1]


def test_cusum_filter_refuses_single_row_multi_column_input() -> None:
    # A 1 x 2 array was flattened and returned column positions as event positions.
    from quantowl.bars import cusum_filter

    with pytest.raises(ValueError, match="one-dimensional or one column"):
        cusum_filter(np.array([[1.0, 2.0]]), threshold=0.5)
    assert cusum_filter(np.array([[1.0], [2.0]]), threshold=0.5).tolist() == [1]


def test_tick_bars_refuses_bool_threshold() -> None:
    # threshold=True was read as one tick per bar.
    from quantowl.bars import tick_bars

    with pytest.raises(ValueError, match="positive integer"):
        tick_bars(np.arange(1.0, 9.0), threshold=True)


def test_imbalance_and_run_dollar_bars_refuse_overflowing_dollar_product() -> None:
    # price * volume overflowed for every tick, so imbalance_bars(kind="dollar") formed no bars and
    # returned an empty frame with no error. dollar_bars and run_bars refuse it.
    from quantowl.bars import imbalance_bars, run_bars

    rng = np.random.default_rng(4)
    p = 100 * np.exp(np.cumsum(rng.normal(0, 0.001, 400)))
    v = np.abs(np.random.default_rng(4).normal(500, 100, 400))
    with pytest.raises(ValueError, match="overflows"):
        imbalance_bars(p * 1e160, v * 1e160, "dollar", init_T=20)
    with pytest.raises(ValueError, match="overflows"):
        run_bars(p * 1e160, v * 1e160, "dollar", init_T=20)


def test_tick_bars_keep_the_dollar_overflow_refusal_as_intended() -> None:
    # Tick and time bars keep the dollar check, so their refusal stays; only the volume kind skips it.
    # volume_bars accepts the same ticks and returns them.
    from quantowl.bars import imbalance_bars, tick_bars, volume_bars

    assert volume_bars(np.full(400, 1e160), np.full(400, 1e160), 1e10).shape == (400, 7)
    with pytest.raises(ValueError, match="overflows"):
        tick_bars(np.full(400, 1e160), 50, np.full(400, 1e160))
    with pytest.raises(ValueError, match="overflows"):
        imbalance_bars(np.full(400, 1e160), np.full(400, 1e160), "tick", init_T=20)
    assert imbalance_bars(np.full(400, 1e160), np.full(400, 1e160), "volume", init_T=20).shape[1] == 7


def test_bars_and_cusum_refuse_bool_init_t_and_threshold() -> None:
    # init_T=True and cusum_filter(threshold=True) were read as 1. tick_bars already refuses a bool
    # threshold, so the siblings now agree.
    from quantowl.bars import cusum_filter, imbalance_bars, tick_bars

    p = np.linspace(100, 101, 40)
    with pytest.raises(ValueError, match="init_T"):
        imbalance_bars(p, np.ones(40), "volume", init_T=True)
    with pytest.raises(ValueError, match="threshold"):
        cusum_filter(p, threshold=True)
    with pytest.raises(ValueError, match="threshold"):
        tick_bars(p, True)


def test_time_bars_names_mixed_timezones_in_the_index() -> None:
    a = pd.Series([1.0, 2.0], index=pd.to_datetime(["2024-01-01 10:00:10", "2024-01-01 10:00:20"]).tz_localize("Europe/Berlin"))
    b = pd.Series([3.0, 4.0], index=pd.to_datetime(["2024-01-01 09:00:30", "2024-01-01 09:00:40"]).tz_localize("UTC"))
    mixed = pd.concat([a, b])
    with pytest.raises(ValueError, match="mixed timezones.*tz_convert"):
        time_bars(mixed, freq="1min")
    fixed = pd.concat([a.tz_convert("UTC"), b])
    assert len(time_bars(fixed, freq="1min")) >= 1


# --------------------------------------------------------------------------- #
# Naive wall-clock indexes through a DST fall-back hour
# --------------------------------------------------------------------------- #


def _naive_fall_back_feed() -> pd.DatetimeIndex:
    # Arrival order across the New York fall-back on 2024-11-03: 01:00 to 01:59 happens twice.
    return pd.DatetimeIndex(
        [
            "2024-11-03 00:50",
            "2024-11-03 01:00",
            "2024-11-03 01:30",
            "2024-11-03 01:00",
            "2024-11-03 01:30",
            "2024-11-03 02:00",
        ]
    )


def test_naive_repeated_label_non_adjacent_is_refused() -> None:
    idx = _naive_fall_back_feed()
    p = np.linspace(100.0, 101.0, len(idx))
    with pytest.raises(ValueError, match="2024-11-03 01:00.*timezone"):
        tick_bars(pd.Series(p, index=idx), 2)
    with pytest.raises(ValueError, match="2024-11-03 01:00.*timezone"):
        cusum_filter(pd.Series(p, index=idx), 0.1)


def test_naive_unsorted_with_adjacent_repeat_is_still_sorted() -> None:
    idx = pd.DatetimeIndex(
        ["2024-03-01 10:00", "2024-03-01 09:00", "2024-03-01 09:00", "2024-03-01 08:00"]
    )
    bars = tick_bars(pd.Series([3.0, 2.0, 2.5, 1.0], index=idx), 2)
    assert list(bars.index) == [pd.Timestamp("2024-03-01 09:00"), pd.Timestamp("2024-03-01 10:00")]
    assert list(bars["close"]) == [2.0, 3.0]


def test_aware_fall_back_index_is_sorted_by_instant() -> None:
    aware = _naive_fall_back_feed().tz_localize(
        "America/New_York", ambiguous=np.array([False, True, True, False, False, False])
    )
    p = np.linspace(100.0, 101.0, len(aware))
    bars = tick_bars(pd.Series(p, index=aware), 2)
    assert bars.index.is_monotonic_increasing
    assert len(bars) == 3
