"""Tests for finlab.bars (AFML ch. 2).

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

from finlab.bars import (
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
