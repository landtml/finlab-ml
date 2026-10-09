"""Financial data structures (AFML ch. 2): bars and event sampling.

Implements, from Lopez de Prado's *Advances in Financial Machine Learning*:

* Section 2.3.1 -- standard bars: time bars (:func:`time_bars`), tick bars
  (:func:`tick_bars`), volume bars (:func:`volume_bars`), dollar bars
  (:func:`dollar_bars`).
* Section 2.3.2.1-2.3.2.2 -- imbalance bars: tick, volume and dollar imbalance
  bars (:func:`imbalance_bars`).
* Section 2.3.2.3-2.3.2.4 -- runs bars: tick, volume and dollar runs bars
  (:func:`run_bars`).
* Section 2.5.2.1 and Snippet 2.4 -- the symmetric CUSUM filter
  (:func:`cusum_filter`).

Threshold conventions
---------------------
* **Fixed** thresholds: time bars (``freq``), standard bars (``threshold``) and
  the CUSUM filter (``threshold``, the filter size *h*) do not adapt.
* **Expected** (dynamic) thresholds: imbalance and runs bars close a bar when
  the accumulated statistic exceeds an *expectation* of the statistic,
  ``E0[T] * |E0[b v]|`` (imbalance) or ``E0[T] * max{...}`` (runs). Both
  factors are exponentially weighted averages (EWMA) estimated from prior
  bars and prior ticks. Their hyper-parameters (``init_T``, ``span_bars``,
  ``span_ticks``) are fixed; the threshold itself moves with the data.

Bar-close timestamps
--------------------
Every bar is indexed by the timestamp of its closing observation. For
threshold-based bars this is the tick that closes the bar; for time bars it is
the right edge of the interval (intervals are right-closed). The incomplete
trailing bar is dropped, because it has not closed yet.

Scope
-----
The kernels are numba-compiled loops over ticks. Each is O(n). Estimation
details the book leaves open (EWMA initialisation, the tick-rule seed ``b0``)
are documented on each function.

Not covered
-----------
The futures roll and ETF trick (Section 2.4), PCA risk weights (2.4.2),
sequential or entropy-based sampling (2.5.1), the volume-clock resampling of
order-book data, and the auction-outlier filtering the book discusses in 2.3.1.2.
"""

from __future__ import annotations

from typing import Literal

import numpy as np
import pandas as pd
from numpy.typing import ArrayLike, NDArray

from finlab._jit import jit

__all__ = [
    "OHLCV_COLUMNS",
    "time_bars",
    "tick_bars",
    "volume_bars",
    "dollar_bars",
    "imbalance_bars",
    "run_bars",
    "cusum_filter",
]

OHLCV_COLUMNS: tuple[str, ...] = (
    "open",
    "high",
    "low",
    "close",
    "volume",
    "dollar_value",
    "ticks",
)

Kind = Literal["tick", "volume", "dollar"]
_KINDS: tuple[str, ...] = ("tick", "volume", "dollar")


# --------------------------------------------------------------------------- #
# Numba kernels (all O(n), no Python loops over observations)
# --------------------------------------------------------------------------- #


@jit
def _cum_boundaries(x: NDArray[np.float64], threshold: float) -> NDArray[np.int64]:
    """Close a bar at every tick ``t`` where the running sum of ``x`` reaches ``threshold``.

    Standard bars (AFML 2.3.1): ``x`` is 1 for tick bars, the volume for volume
    bars, and price times volume for dollar bars. The threshold is fixed.
    """
    n = x.shape[0]
    ends = np.empty(n, dtype=np.int64)
    k = 0
    acc = 0.0
    for t in range(n):
        acc += x[t]
        if acc >= threshold:
            ends[k] = t
            k += 1
            acc = 0.0
    return ends[:k]


@jit
def _tick_rule(p: NDArray[np.float64], b0: float) -> NDArray[np.float64]:
    """Tick rule of AFML 2.3.2.1: ``b_t = b_{t-1}`` if ``dp_t = 0``, else ``sign(dp_t)``.

    ``b_0`` (the first element) is the seed ``b0``. The sequence is global, so the
    sign carries across bar boundaries, as the book requires.
    """
    n = p.shape[0]
    b = np.empty(n, dtype=np.float64)
    b[0] = b0
    for t in range(1, n):
        dp = p[t] - p[t - 1]
        if dp > 0.0:
            b[t] = 1.0
        elif dp < 0.0:
            b[t] = -1.0
        else:
            b[t] = b[t - 1]
    return b


@jit
def _ewma_update_bar_length(e_t: float, length: int, alpha: float) -> float:
    return alpha * float(length) + (1.0 - alpha) * e_t


@jit
def _imbalance_ends(
    s: NDArray[np.float64],
    init_T: int,
    alpha_bars: float,
    alpha_ticks: float,
) -> NDArray[np.int64]:
    """Bar boundaries for imbalance bars (AFML 2.3.2.1-2.3.2.2).

    ``s[t] = b_t * v_t`` (``v_t = 1`` for tick imbalance). A bar closes at the
    first tick where ``|theta| >= E[T] * |E[s]|``, with ``theta`` the running
    sum of ``s`` inside the bar.

    Expectations are frozen while a bar is open. At close, ``E[T]`` is updated
    with the bar length and ``E[s]`` with the bar's per-tick values, both as EWMA
    steps. The initial ``E[T] = init_T`` and ``E[s]`` is the mean of the first
    ``min(init_T, n)`` values (bootstrap; the only look-ahead in the estimator).
    """
    n = s.shape[0]
    m = min(init_T, n)
    e_T = float(init_T)
    e_s = 0.0
    for j in range(m):
        e_s += s[j]
    e_s = e_s / float(m)

    ends = np.empty(n, dtype=np.int64)
    k = 0
    start = 0
    theta = 0.0
    thr = e_T * abs(e_s)
    for t in range(n):
        theta += s[t]
        if abs(theta) >= thr:
            ends[k] = t
            k += 1
            e_T = _ewma_update_bar_length(e_T, t - start + 1, alpha_bars)
            for j in range(start, t + 1):
                e_s = alpha_ticks * s[j] + (1.0 - alpha_ticks) * e_s
            start = t + 1
            theta = 0.0
            thr = e_T * abs(e_s)
    return ends[:k]


@jit
def _runs_ends(
    buy: NDArray[np.float64],
    sell: NDArray[np.float64],
    init_T: int,
    alpha_bars: float,
    alpha_ticks: float,
) -> NDArray[np.int64]:
    """Bar boundaries for runs bars (AFML 2.3.2.3-2.3.2.4).

    ``buy[t] = 1{b_t = 1} v_t`` and ``sell[t] = 1{b_t = -1} v_t``. A bar closes
    at the first tick where ``max(sum buy, sum sell) >= E[T] * max(E[buy], E[sell])``.
    The expectation ``E[buy]`` estimates ``P[b=1] E[v | b=1]`` per tick (and
    ``E[sell]`` estimates ``(1 - P[b=1]) E[v | b=-1]``), each by EWMA over prior
    ticks. Tick runs bars use ``v = 1``.

    Initialisation follows the same rules as :func:`_imbalance_ends`.
    """
    n = buy.shape[0]
    m = min(init_T, n)
    e_T = float(init_T)
    e_b = 0.0
    e_s = 0.0
    for j in range(m):
        e_b += buy[j]
        e_s += sell[j]
    e_b = e_b / float(m)
    e_s = e_s / float(m)

    ends = np.empty(n, dtype=np.int64)
    k = 0
    start = 0
    run_b = 0.0
    run_s = 0.0
    thr = e_T * max(e_b, e_s)
    for t in range(n):
        run_b += buy[t]
        run_s += sell[t]
        if max(run_b, run_s) >= thr:
            ends[k] = t
            k += 1
            e_T = _ewma_update_bar_length(e_T, t - start + 1, alpha_bars)
            for j in range(start, t + 1):
                e_b = alpha_ticks * buy[j] + (1.0 - alpha_ticks) * e_b
                e_s = alpha_ticks * sell[j] + (1.0 - alpha_ticks) * e_s
            start = t + 1
            run_b = 0.0
            run_s = 0.0
            thr = e_T * max(e_b, e_s)
    return ends[:k]


@jit
def _ohlcv(
    p: NDArray[np.float64], v: NDArray[np.float64], ends: NDArray[np.int64]
) -> NDArray[np.float64]:
    """Aggregate ticks into bars ``(start, end]`` given the closing indices ``ends``.

    Columns: open, high, low, close, volume, dollar value, tick count.
    """
    m = ends.shape[0]
    out = np.empty((m, 7), dtype=np.float64)
    start = 0
    for k in range(m):
        e = ends[k]
        hi = -np.inf
        lo = np.inf
        vol = 0.0
        dol = 0.0
        for t in range(start, e + 1):
            if p[t] > hi:
                hi = p[t]
            if p[t] < lo:
                lo = p[t]
            vol += v[t]
            dol += p[t] * v[t]
        out[k, 0] = p[start]
        out[k, 1] = hi
        out[k, 2] = lo
        out[k, 3] = p[e]
        out[k, 4] = vol
        out[k, 5] = dol
        out[k, 6] = float(e - start + 1)
        start = e + 1
    return out


@jit
def _cusum_events(y: NDArray[np.float64], h: float) -> NDArray[np.int64]:
    """Symmetric CUSUM filter (AFML Snippet 2.4), returning event positions in ``y``.

    The reset-to-zero and the strict comparisons follow Snippet 2.4 exactly.
    """
    n = y.shape[0]
    ev = np.empty(n, dtype=np.int64)
    k = 0
    s_pos = 0.0
    s_neg = 0.0
    for t in range(n):
        s_pos = max(0.0, s_pos + y[t])
        s_neg = min(0.0, s_neg + y[t])
        if s_neg < -h:
            s_neg = 0.0
            ev[k] = t
            k += 1
        elif s_pos > h:
            s_pos = 0.0
            ev[k] = t
            k += 1
    return ev[:k]


# --------------------------------------------------------------------------- #
# Input handling
# --------------------------------------------------------------------------- #


def _as_arrays(
    prices: ArrayLike | pd.Series,
    volumes: ArrayLike | pd.Series | None,
    need_volume: bool,
) -> tuple[pd.Index, NDArray[np.float64], NDArray[np.float64]]:
    """Validate and convert tick inputs to contiguous float64 arrays.

    Returns the index used to label bars (the input index for a Series, a
    RangeIndex for an array), the prices and the volumes. Volumes are NaN when
    absent, which is allowed only for kinds that do not use them.
    """
    if isinstance(prices, pd.Series):
        index = pd.Index(prices.index)
        p = prices.to_numpy(dtype=np.float64)
    else:
        p = np.asarray(prices, dtype=np.float64)
        if p.ndim != 1:
            raise ValueError(f"prices must be 1-D, got shape {p.shape}.")
        index = pd.RangeIndex(p.shape[0])
    p = np.ascontiguousarray(p)
    if p.size == 0:
        raise ValueError("prices is empty.")
    if not np.all(np.isfinite(p)) or np.any(p <= 0.0):
        raise ValueError("prices must be finite and strictly positive.")

    if volumes is None:
        if need_volume:
            raise ValueError("this bar type needs volumes; pass volumes=...")
        v = np.full(p.shape[0], np.nan)
    else:
        if isinstance(volumes, pd.Series) and isinstance(prices, pd.Series):
            if not volumes.index.equals(prices.index):
                raise ValueError("prices and volumes Series must share the same index.")
        v = np.asarray(volumes, dtype=np.float64).reshape(-1)
        if v.shape[0] != p.shape[0]:
            raise ValueError(
                f"volumes length {v.shape[0]} != prices length {p.shape[0]}."
            )
        if not np.all(np.isfinite(v)) or np.any(v < 0.0):
            raise ValueError("volumes must be finite and non-negative.")
        v = np.ascontiguousarray(v)
    return index, p, v


def _check_threshold(threshold: float) -> float:
    if not np.isfinite(threshold) or threshold <= 0:
        raise ValueError(f"threshold must be a positive finite number, got {threshold!r}.")
    return float(threshold)


def _check_init_T(init_T: int) -> int:
    if int(init_T) != init_T or init_T < 1:
        raise ValueError(f"init_T must be a positive integer, got {init_T!r}.")
    return int(init_T)


def _alpha(span: float, name: str) -> float:
    if not np.isfinite(span) or span < 1:
        raise ValueError(f"{name} must be >= 1, got {span!r}.")
    return 2.0 / (float(span) + 1.0)


def _magnitude(p: NDArray[np.float64], v: NDArray[np.float64], kind: str) -> NDArray[np.float64]:
    """Per-tick size used by a bar type: 1 (tick), volume, or price times volume (dollar)."""
    if kind == "tick":
        return np.ones(p.shape[0], dtype=np.float64)
    if kind == "volume":
        return v
    return p * v


def _check_kind(kind: str) -> None:
    if kind not in _KINDS:
        raise ValueError(f"kind must be one of {_KINDS}, got {kind!r}.")


def _bars_frame(
    index: pd.Index,
    p: NDArray[np.float64],
    v: NDArray[np.float64],
    ends: NDArray[np.int64],
) -> pd.DataFrame:
    """Build the OHLCV frame indexed by the bar-close label."""
    ohlcv = _ohlcv(p, v, ends)
    out = pd.DataFrame(ohlcv, columns=list(OHLCV_COLUMNS))
    out["ticks"] = out["ticks"].astype(np.int64)
    out.index = index[ends]
    return out


# --------------------------------------------------------------------------- #
# Public API: standard bars (AFML 2.3.1)
# --------------------------------------------------------------------------- #


def time_bars(
    prices: pd.Series,
    volumes: pd.Series | None = None,
    freq: str = "1min",
) -> pd.DataFrame:
    """Time bars: aggregate ticks over fixed calendar intervals (AFML 2.3.1.1).

    Parameters
    ----------
    prices : pd.Series
        Tick prices, indexed by a ``DatetimeIndex``.
    volumes : pd.Series, optional
        Tick volumes on the same index. Without it, volume columns are NaN.
    freq : str, default "1min"
        Pandas offset alias for the bar length. **Fixed** threshold.

    Returns
    -------
    pd.DataFrame
        OHLCV columns (see :data:`OHLCV_COLUMNS`), indexed by the right edge of
        each interval. Empty intervals are dropped.

    Notes
    -----
    Intervals are right-closed: a tick stamped exactly at the edge belongs to the
    bar that ends there. Time bars oversample quiet periods, as AFML 2.3.1.1 notes.

    Scope
    -----
    Uses pandas ``resample`` (vectorised, not a numba loop).
    """
    if not isinstance(prices, pd.Series) or not isinstance(prices.index, pd.DatetimeIndex):
        raise TypeError("time_bars needs a pd.Series indexed by a DatetimeIndex.")
    _, p, v = _as_arrays(prices, volumes, need_volume=False)
    df = pd.DataFrame(
        {"price": p, "volume": v, "dollar": p * v},
        index=prices.index,
    )
    rs = df.resample(freq, closed="right", label="right")
    out = pd.DataFrame(
        {
            "open": rs["price"].first(),
            "high": rs["price"].max(),
            "low": rs["price"].min(),
            "close": rs["price"].last(),
            "volume": rs["volume"].sum(min_count=1),
            "dollar_value": rs["dollar"].sum(min_count=1),
            "ticks": rs["price"].count(),
        }
    )
    out = out[out["ticks"] > 0].copy()
    out["ticks"] = out["ticks"].astype(np.int64)
    return out


def tick_bars(
    prices: ArrayLike | pd.Series,
    threshold: int,
    volumes: ArrayLike | pd.Series | None = None,
) -> pd.DataFrame:
    """Tick bars: close a bar every ``threshold`` ticks (AFML 2.3.1.2).

    Parameters
    ----------
    prices : array-like or pd.Series
        Tick prices. A Series supplies the bar-close timestamps.
    threshold : int
        Number of ticks per bar. **Fixed** threshold.
    volumes : array-like or pd.Series, optional
        Tick volumes, used only for the ``volume`` and ``dollar_value`` columns.

    Returns
    -------
    pd.DataFrame
        OHLCV frame indexed by the closing tick's timestamp (or position).

    Scope
    -----
    Numba kernel :func:`_cum_boundaries`, O(n).
    """
    thr = _check_threshold(threshold)
    index, p, v = _as_arrays(prices, volumes, need_volume=False)
    ends = _cum_boundaries(np.ones(p.shape[0], dtype=np.float64), thr)
    return _bars_frame(index, p, v, ends)


def volume_bars(
    prices: ArrayLike | pd.Series,
    volumes: ArrayLike | pd.Series,
    threshold: float,
) -> pd.DataFrame:
    """Volume bars: close a bar every ``threshold`` units traded (AFML 2.3.1.3).

    Parameters
    ----------
    prices : array-like or pd.Series
        Tick prices.
    volumes : array-like or pd.Series
        Tick volumes (non-negative).
    threshold : float
        Volume per bar. **Fixed** threshold.

    Returns
    -------
    pd.DataFrame
        OHLCV frame indexed by the closing tick.

    Scope
    -----
    Numba kernel :func:`_cum_boundaries` applied to ``volumes``.
    """
    thr = _check_threshold(threshold)
    index, p, v = _as_arrays(prices, volumes, need_volume=True)
    ends = _cum_boundaries(v, thr)
    return _bars_frame(index, p, v, ends)


def dollar_bars(
    prices: ArrayLike | pd.Series,
    volumes: ArrayLike | pd.Series,
    threshold: float,
) -> pd.DataFrame:
    """Dollar bars: close a bar every ``threshold`` of ``price * volume`` (AFML 2.3.1.4).

    Parameters
    ----------
    prices : array-like or pd.Series
        Tick prices.
    volumes : array-like or pd.Series
        Tick volumes (non-negative).
    threshold : float
        Dollar value per bar, in the price currency. **Fixed** threshold.

    Returns
    -------
    pd.DataFrame
        OHLCV frame indexed by the closing tick.

    Scope
    -----
    Numba kernel :func:`_cum_boundaries` applied to ``price * volume``.
    """
    thr = _check_threshold(threshold)
    index, p, v = _as_arrays(prices, volumes, need_volume=True)
    ends = _cum_boundaries(p * v, thr)
    return _bars_frame(index, p, v, ends)


# --------------------------------------------------------------------------- #
# Public API: imbalance and runs bars (AFML 2.3.2)
# --------------------------------------------------------------------------- #


def imbalance_bars(
    prices: ArrayLike | pd.Series,
    volumes: ArrayLike | pd.Series | None = None,
    kind: Kind = "tick",
    *,
    init_T: int = 100,
    span_bars: float = 20,
    span_ticks: float = 1000,
    b0: float = 1.0,
) -> pd.DataFrame:
    """Imbalance bars (AFML 2.3.2.1-2.3.2.2).

    A bar closes at the first tick ``T`` where the signed imbalance
    ``theta_T = sum_{t<=T} b_t v_t`` satisfies
    ``|theta_T| >= E0[T] * |E0[b_t v_t]|``.

    Parameters
    ----------
    prices : array-like or pd.Series
        Tick prices. The tick rule ``b_t`` is built from consecutive differences.
    volumes : array-like or pd.Series, optional
        Tick volumes. Required for ``kind`` ``"volume"`` and ``"dollar"``.
    kind : {"tick", "volume", "dollar"}, default "tick"
        ``v_t`` is 1, the volume, or price times volume respectively.
    init_T : int, default 100
        Initial expected bar length ``E0[T]``, and the number of leading ticks
        used to bootstrap ``E0[b_t v_t]``. Fixed.
    span_bars : float, default 20
        EWMA span for ``E[T]``, in bars. Fixed.
    span_ticks : float, default 1000
        EWMA span for ``E[b_t v_t]``, in ticks. Fixed.
    b0 : float, default 1.0
        Seed of the tick rule (``b_0``). The book sets it to the terminal value of
        the previous bar. Within one call that is automatic; a fresh call uses ``b0``.

    Returns
    -------
    pd.DataFrame
        OHLCV frame indexed by the closing tick.

    Notes
    -----
    The expected imbalance is **dynamic** (EWMA estimates, frozen while a bar is
    open and updated at close). The bootstrap is the only look-ahead: the initial
    ``E[b_t v_t]`` uses the first ``init_T`` ticks. If the bootstrap mean is exactly
    zero, the threshold is zero and every tick closes a bar, as the rule implies.

    Scope
    -----
    Tick imbalance bars are the ``kind="tick"`` case. The book's ``TIB``, ``VIB``
    and ``DIB`` are the three kinds.
    """
    _check_kind(kind)
    _check_init_T(init_T)
    a_bars = _alpha(span_bars, "span_bars")
    a_ticks = _alpha(span_ticks, "span_ticks")
    index, p, v = _as_arrays(prices, volumes, need_volume=kind != "tick")
    b = _tick_rule(p, float(b0))
    s = np.ascontiguousarray(b * _magnitude(p, v, kind))
    ends = _imbalance_ends(s, int(init_T), a_bars, a_ticks)
    return _bars_frame(index, p, v, ends)


def run_bars(
    prices: ArrayLike | pd.Series,
    volumes: ArrayLike | pd.Series | None = None,
    kind: Kind = "tick",
    *,
    init_T: int = 100,
    span_bars: float = 20,
    span_ticks: float = 1000,
    b0: float = 1.0,
) -> pd.DataFrame:
    """Runs bars (AFML 2.3.2.3-2.3.2.4).

    A bar closes at the first tick where
    ``max{sum_{b=1} v_t, sum_{b=-1} v_t} >= E0[T] * max{E0[buy], E0[sell]}``,
    with ``v_t = 1`` for tick runs, the volume for volume runs, and the dollar
    value for dollar runs.

    Parameters
    ----------
    prices : array-like or pd.Series
        Tick prices.
    volumes : array-like or pd.Series, optional
        Tick volumes. Required for ``kind`` ``"volume"`` and ``"dollar"``.
    kind : {"tick", "volume", "dollar"}, default "tick"
        Which size measure ``v_t`` the runs are counted in.
    init_T : int, default 100
        Initial expected bar length ``E0[T]``, and the bootstrap window.
    span_bars : float, default 20
        EWMA span for ``E[T]``, in bars. Fixed.
    span_ticks : float, default 1000
        EWMA span for the per-tick buy and sell expectations, in ticks. Fixed.
    b0 : float, default 1.0
        Seed of the tick rule.

    Returns
    -------
    pd.DataFrame
        OHLCV frame indexed by the closing tick.

    Notes
    -----
    Sequence breaks are allowed: runs are counted per side without offsetting,
    as the book specifies. ``E0[buy]`` estimates ``P[b=1] E0[v | b=1]`` and
    ``E0[sell]`` estimates ``(1 - P[b=1]) E0[v | b=-1]``, each by EWMA over prior
    ticks. The threshold is **dynamic**; the spans are **fixed**.

    Scope
    -----
    Numba kernel :func:`_runs_ends`, O(n).
    """
    _check_kind(kind)
    _check_init_T(init_T)
    a_bars = _alpha(span_bars, "span_bars")
    a_ticks = _alpha(span_ticks, "span_ticks")
    index, p, v = _as_arrays(prices, volumes, need_volume=kind != "tick")
    b = _tick_rule(p, float(b0))
    mag = _magnitude(p, v, kind)
    buy = np.ascontiguousarray(np.where(b > 0.0, mag, 0.0))
    sell = np.ascontiguousarray(np.where(b < 0.0, mag, 0.0))
    ends = _runs_ends(buy, sell, int(init_T), a_bars, a_ticks)
    return _bars_frame(index, p, v, ends)


# --------------------------------------------------------------------------- #
# Public API: CUSUM event filter (AFML 2.5.2.1, Snippet 2.4)
# --------------------------------------------------------------------------- #


def cusum_filter(
    series: ArrayLike | pd.Series,
    threshold: float,
    *,
    is_returns: bool = False,
) -> pd.Index | NDArray[np.int64]:
    """Symmetric CUSUM filter: return the event times (AFML 2.5.2.1, Snippet 2.4).

    The filter tracks ``S+_t = max(0, S+_{t-1} + y_t)`` and
    ``S-_t = min(0, S-_{t-1} + y_t)`` and emits an event, then resets that
    side to zero, whenever ``S-_t < -h`` or ``S+_t > h``.

    Parameters
    ----------
    series : array-like or pd.Series
        Levels (e.g. log prices), with ``y_t = x_t - x_{t-1}``, unless
        ``is_returns=True``. Also accepts returns directly.
    threshold : float
        Filter size ``h``. **Fixed**.
    is_returns : bool, default False
        If True, ``series`` already holds the increments ``y_t``. The first
        observation is then eligible as an event.

    Returns
    -------
    pd.Index or np.ndarray
        For a Series: the index labels of the events. For an array: the integer
        positions of the events in the input.

    Notes
    -----
    Snippet 2.4 uses strict comparisons (``> h``, ``< -h``), while the text
    writes ``>= h``. This module follows the snippet. The two differ only on a
    measure-zero set for continuous data. Non-finite values are rejected rather
    than silently propagating through the sums.

    Scope
    -----
    Numba kernel :func:`_cusum_events`, O(n). The threshold is fixed, not
    volatility-scaled; scale ``h`` yourself (e.g. to a multiple of the
    return standard deviation, as in Chapter 5, exercise 6(a) of the book) if needed.
    """
    h = _check_threshold(threshold)
    if isinstance(series, pd.Series):
        index: pd.Index | None = pd.Index(series.index)
        x = series.to_numpy(dtype=np.float64)
    else:
        index = None
        x = np.asarray(series, dtype=np.float64).reshape(-1)
    if not np.all(np.isfinite(x)):
        raise ValueError("series must be finite.")
    if is_returns:
        y = np.ascontiguousarray(x)
        offset = 0
    else:
        if x.shape[0] < 2:
            return index[:0] if index is not None else np.empty(0, dtype=np.int64)
        y = np.ascontiguousarray(np.diff(x))
        offset = 1
    pos = _cusum_events(y, h) + offset
    if index is None:
        return pos.astype(np.int64)
    return index[pos]
