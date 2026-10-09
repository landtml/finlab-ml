"""Labeling for supervised learning (AFML Chapter 3).

Implements the labeling pipeline of AFML Chapter 3:

* Snippet 3.1 -- :func:`get_daily_vol`, an exponentially weighted volatility of
  one-day returns, used to set barrier widths.
* Snippet 3.4 -- :func:`add_vertical_barrier`, the expiry (vertical) barrier.
* Snippets 3.2-3.3 and 3.6 -- :func:`get_events`, the triple-barrier method.
  The book scans each event's path with a pandas loop; here the scan is a
  numba kernel over event positions.
* Snippets 3.5 and 3.7 -- :func:`get_bins`, the side-and-size label, and the
  meta-labeling variant when a side is supplied.
* Section 3.6 -- :func:`meta_labels`, the binary "was the primary side right?"
  label built from primary-model side predictions.
* Snippet 3.8 -- :func:`drop_labels`, the recursive removal of rare classes.

Trend-scanning labels (:func:`trend_scanning_labels`) are *not* part of AFML
Chapter 3. They follow the trend-scanning method attributed to Lopez de Prado
(2019) and Hudson & Thames, implemented from the literal definition in the
function docstring. The exact citation was not checked against a primary
source while writing this module.

Not covered: the multiprocessing engine ``mpPandasObj`` of Chapter 20 (the
numba kernels replace it for these functions), the exercises at the end of
the chapter, and the sampling of events (Chapter 2) that produces
``t_events``.
"""

from __future__ import annotations

from typing import Sequence

import numpy as np
import pandas as pd
from numba import prange

from ._jit import jit, pjit

__all__ = [
    "get_daily_vol",
    "add_vertical_barrier",
    "get_events",
    "get_bins",
    "meta_labels",
    "drop_labels",
    "trend_scanning_labels",
    "BARRIER_NAMES",
]

#: Names of the barrier codes returned in the ``barrier`` column of
#: :func:`get_events`. Code ``0`` means no barrier was touched (an open event).
BARRIER_NAMES: tuple[str, ...] = ("none", "pt", "sl", "vertical")


# ---------------------------------------------------------------------------
# Snippet 3.1 -- daily volatility
# ---------------------------------------------------------------------------


def get_daily_vol(close: pd.Series, span: int = 100) -> pd.Series:
    """Exponentially weighted standard deviation of daily returns (Snippet 3.1).

    Parameters
    ----------
    close : pd.Series
        Strictly positive prices indexed by a ``DatetimeIndex``.
    span : int, default 100
        ``span`` passed to :meth:`pandas.Series.ewm`.

    Returns
    -------
    pd.Series
        Volatility aligned to ``close.index``. Each bar ``t`` is compared with
        the last bar strictly before ``t - 1 day``. Bars with no such bar (the
        first day of the sample) are NaN.

    Notes
    -----
    Snippet 3.1 in the book applies ``ewm(span).std()`` to the return series
    built as above. This function does the same, with the lookup written as
    a vectorised ``searchsorted``.

    Scope: one-day look-back on a time index. Irregular sampling is handled
    by the calendar lookback, not by a fixed bar count.
    """
    if not isinstance(close.index, pd.DatetimeIndex):
        raise TypeError("close must be indexed by a DatetimeIndex.")
    px = close.to_numpy(dtype=float)
    idx = close.index
    prev = idx.searchsorted(idx - pd.Timedelta(days=1), side="left") - 1
    valid = prev >= 0
    cur = np.nonzero(valid)[0]
    rets = px[cur] / px[prev[valid]] - 1.0
    ret_s = pd.Series(rets, index=idx[cur])
    vol = ret_s.ewm(span=span).std()
    out = vol.reindex(idx)
    out.name = "daily_vol"
    return out


# ---------------------------------------------------------------------------
# Snippet 3.4 -- vertical barrier
# ---------------------------------------------------------------------------


def add_vertical_barrier(
    t_events: pd.Index | Sequence, close: pd.Series, num_days: float = 0
) -> pd.Series:
    """Expiry timestamp for each event, ``num_days`` calendar days later (Snippet 3.4).

    Parameters
    ----------
    t_events : pd.Index or sequence of timestamps
        Event start times.
    close : pd.Series
        Price series whose index supplies the bar grid.
    num_days : float, default 0
        Calendar days until expiry. ``0`` means the barrier is the first bar at
        or after the event time.

    Returns
    -------
    pd.Series
        Indexed by ``t_events``. Values are the first bar at or after
        ``event + num_days``. Events whose expiry falls after the last bar
        have NaT. Snippet 3.4 instead drops those events from the output.

    Scope: calendar-day offsets only.
    """
    t_events = pd.DatetimeIndex(t_events)
    grid = close.index
    pos = grid.searchsorted(t_events + pd.Timedelta(days=num_days), side="left")
    valid = pos < len(grid)
    vals = np.full(len(t_events), np.datetime64("NaT", "ns"))
    vals[valid] = grid.to_numpy()[pos[valid]]
    return pd.Series(pd.DatetimeIndex(vals), index=t_events)


# ---------------------------------------------------------------------------
# Snippets 3.2 / 3.3 / 3.6 -- triple-barrier first-touch scan
# ---------------------------------------------------------------------------


@pjit
def _scan_first_touch(
    close: np.ndarray,
    starts: np.ndarray,
    ends: np.ndarray,
    trgt: np.ndarray,
    side: np.ndarray,
    pt_mult: float,
    sl_mult: float,
) -> tuple[np.ndarray, np.ndarray]:
    """First horizontal-barrier touch per event, scanned over bar positions.

    For event ``i`` the path is ``close[starts[i] + 1 .. ends[i]]``. The side
    adjusted return ``side[i] * (close[j] / close[starts[i]] - 1)`` is compared
    with ``+pt_mult * trgt[i]`` (profit taking) and ``-sl_mult * trgt[i]``
    (stop loss). A multiplier of 0 disables that barrier. If both are touched
    on the same bar the stop loss is recorded.

    Returns
    -------
    hit_pos : int64 array
        Bar position of the first horizontal touch, or -1.
    code : int8 array
        ``1`` profit taking, ``2`` stop loss, ``0`` no horizontal touch.
    """
    n_ev = starts.shape[0]
    hit_pos = np.full(n_ev, -1, dtype=np.int64)
    code = np.zeros(n_ev, dtype=np.int8)
    for i in prange(n_ev):
        s = starts[i]
        e = ends[i]
        p0 = close[s]
        sd = side[i]
        up = pt_mult * trgt[i]
        lo = -sl_mult * trgt[i]
        for j in range(s + 1, e + 1):
            r = (close[j] / p0 - 1.0) * sd
            if sl_mult > 0.0 and r < lo:
                hit_pos[i] = j
                code[i] = 2
                break
            if pt_mult > 0.0 and r > up:
                hit_pos[i] = j
                code[i] = 1
                break
    return hit_pos, code


def _check_prices(close: pd.Series) -> np.ndarray:
    if not isinstance(close, pd.Series):
        raise TypeError("close must be a pandas Series.")
    px = close.to_numpy(dtype=float)
    if not np.all(np.isfinite(px)) or np.any(px <= 0):
        raise ValueError("close must contain only finite, strictly positive prices.")
    if not close.index.is_monotonic_increasing or close.index.has_duplicates:
        raise ValueError("close.index must be strictly increasing.")
    return np.ascontiguousarray(px)


def get_events(
    close: pd.Series,
    t_events: pd.Index | Sequence,
    pt_sl: Sequence[float],
    target: pd.Series,
    min_ret: float,
    vertical_barrier_times: pd.Series | None = None,
    side: pd.Series | None = None,
) -> pd.DataFrame:
    """Triple-barrier events: first barrier touch per event (Snippets 3.2, 3.3, 3.6).

    Parameters
    ----------
    close : pd.Series
        Strictly positive prices, strictly increasing index.
    t_events : pd.Index or sequence of timestamps
        Event start times. Each must be an index label of ``close``.
    pt_sl : two non-negative floats
        ``(pt, sl)``. ``pt`` multiplies ``trgt`` to give the profit-taking
        width and ``sl`` the stop-loss width. ``0`` disables that barrier.
    target : pd.Series
        Unit barrier width (absolute return), indexed by time. Its values at
        ``t_events`` are used. Snippet 3.3 calls this ``trgt``.
    min_ret : float
        Events with ``target <= min_ret`` (or NaN) are dropped.
    vertical_barrier_times : pd.Series, optional
        Expiry timestamp per event, indexed by ``t_events`` (see
        :func:`add_vertical_barrier`). NaT or a missing value means no vertical
        barrier for that event. Without it the path runs to the last bar.
    side : pd.Series, optional
        Side in ``{-1, +1}`` per event, indexed by ``t_events``. Supplying it
        turns on meta-labeling mode (Snippet 3.6): each side is applied to its
        own event. Without it every event is long and the two horizontal
        barriers are symmetric (``pt_sl[0]`` is used for both, as in the book
        for ``pt_sl[0] == pt_sl[1]``).

    Returns
    -------
    pd.DataFrame
        Indexed by the retained event start times, with columns:

        ``t1``
            Time of the first barrier touch (NaT if none was touched and there
            is no vertical barrier).
        ``trgt``
            Unit barrier width used for the event.
        ``side``
            Side used for the event (``+1`` when ``side`` was not given).
        ``barrier``
            One of ``"pt"``, ``"sl"``, ``"vertical"``, ``"none"``. Tie rule: a
            horizontal touch on the same timestamp as the vertical barrier
            takes precedence over it, and a stop loss takes precedence over a
            profit take on the same bar.

    Notes
    -----
    The path of event ``i`` is ``close[t0 .. min(t1_vertical, last bar)]``.
    The horizontal touch is the first bar where
    ``side * (close_j / close_t0 - 1)`` is strictly above ``pt * trgt`` or
    strictly below ``-sl * trgt``. The returned ``t1`` is the minimum over the
    touched barriers, which is the book's ``min(axis=1)`` over the three
    barrier times. The scan is one numba-parallel loop over events, not a
    Python loop with pandas slicing.

    Scope: long-only or per-event side on one price series. Vertical barriers
    may fall between bars. Their timestamp is reported as given.
    """
    px = _check_prices(close)
    grid = close.index
    ev_index = pd.Index(t_events)
    if ev_index.has_duplicates:
        raise ValueError("t_events must not contain duplicates.")
    if not ev_index.isin(grid).all():
        raise ValueError("every t_events label must be an index label of close.")
    pt_sl_arr = np.asarray(pt_sl, dtype=float).reshape(-1)
    if pt_sl_arr.shape[0] != 2 or np.any(pt_sl_arr < 0) or not np.all(np.isfinite(pt_sl_arr)):
        raise ValueError("pt_sl must be two finite, non-negative floats.")

    trgt_all = target.reindex(ev_index).to_numpy(dtype=float)
    keep = np.isfinite(trgt_all) & (trgt_all > min_ret)
    ev_index = ev_index[keep]
    trgt = trgt_all[keep]

    starts = grid.get_indexer(ev_index)
    if np.any(starts < 0):
        raise ValueError("every t_events label must be an index label of close.")
    starts = starts.astype(np.int64)

    if side is None:
        side_v = np.ones(len(ev_index), dtype=float)
    else:
        side_v = side.reindex(ev_index).to_numpy(dtype=float)
        if not np.all(np.isin(side_v, (-1.0, 1.0))):
            raise ValueError("side must equal -1 or +1 for every retained event.")

    n_bars = len(grid)
    ends = np.full(len(ev_index), n_bars - 1, dtype=np.int64)
    vert_time = pd.DatetimeIndex([pd.NaT] * len(ev_index))
    has_vert = np.zeros(len(ev_index), dtype=bool)
    if vertical_barrier_times is not None:
        vt = vertical_barrier_times.reindex(ev_index)
        vert_time = pd.DatetimeIndex(vt.to_numpy())
        has_vert = ~pd.isna(vert_time)
        if np.any(has_vert):
            ends_v = grid.searchsorted(vert_time[has_vert], side="right") - 1
            if np.any(ends_v < starts[has_vert]):
                raise ValueError("a vertical barrier falls before its event start.")
            ends[has_vert] = ends_v

    hit_pos, code = _scan_first_touch(
        px,
        starts,
        ends,
        np.ascontiguousarray(trgt),
        np.ascontiguousarray(side_v),
        float(pt_sl_arr[0]),
        float(pt_sl_arr[1]),
    )

    n = len(ev_index)
    t1 = np.full(n, np.datetime64("NaT", "ns"))
    grid_vals = grid.to_numpy()
    horiz = code > 0
    t1[horiz] = grid_vals[hit_pos[horiz]]
    vert_only = (~horiz) & has_vert
    code_out = code.astype(np.int64)
    code_out[vert_only] = 3
    t1[vert_only] = vert_time.to_numpy()[vert_only]

    out = pd.DataFrame(
        {
            "t1": pd.DatetimeIndex(t1),
            "trgt": trgt,
            "side": side_v,
            "barrier": np.asarray(BARRIER_NAMES, dtype=object)[code_out],
        },
        index=ev_index,
    )
    return out


# ---------------------------------------------------------------------------
# Snippets 3.5 / 3.7 -- side and size labels
# ---------------------------------------------------------------------------


def get_bins(events: pd.DataFrame, close: pd.Series, meta: bool = False) -> pd.DataFrame:
    """Return and sign label for each event (Snippets 3.5 and 3.7).

    Parameters
    ----------
    events : pd.DataFrame
        Output of :func:`get_events`, with column ``t1``.
    close : pd.Series
        Prices used to evaluate the start and end of each event.
    meta : bool, default False
        ``False``: side-and-size labels (Snippet 3.5). ``True``: meta-labels
        (Snippet 3.7), which multiply the return by ``events["side"]`` and
        return 0/1. Use ``True`` when ``side`` was supplied to
        :func:`get_events`, since that function always returns a ``side``
        column, and it is ``+1`` for every event when ``side`` was omitted.

    Returns
    -------
    pd.DataFrame
        Indexed by the events with a non-NaT ``t1`` and a finite return.
        Columns:

        ``ret``
            Return from the event start to ``t1``. When ``side`` is present it
            is multiplied by ``side``, so it is the P&L of the bet.
        ``bin``
            Without ``side``: ``sign(ret)``, in ``{-1, 0, 1}``. With ``side``
            (meta-labeling): ``1`` if the P&L is strictly positive, else
            ``0``.

    Notes
    -----
    Prices are taken at the start time and at ``t1`` with a backward fill
    (the first bar at or after the timestamp), as in Snippet 3.5. An event
    whose ``t1`` lies after the last bar has no end price and is dropped.

    Scope: one price series. Returns are simple, not log.
    """
    ev = events.dropna(subset=["t1"])
    t1_vals = pd.DatetimeIndex(ev["t1"])
    px_start = close.reindex(ev.index, method="bfill").to_numpy(dtype=float)
    px_end = close.reindex(t1_vals, method="bfill").to_numpy(dtype=float)
    ret = px_end / px_start - 1.0
    if meta:
        if "side" not in ev.columns:
            raise ValueError("meta=True requires a 'side' column in events.")
        ret = ret * ev["side"].to_numpy(dtype=float)
    finite = np.isfinite(ret)
    ret = ret[finite]
    index = ev.index[finite]
    if meta:
        bin_ = (ret > 0).astype(np.int64)
    else:
        bin_ = np.sign(ret).astype(np.int64)
    return pd.DataFrame({"ret": ret, "bin": bin_}, index=index)


def meta_labels(
    events: pd.DataFrame, close: pd.Series, side: pd.Series | np.ndarray
) -> pd.DataFrame:
    """Binary meta-label from primary-model sides (Section 3.6).

    Parameters
    ----------
    events : pd.DataFrame
        Output of :func:`get_events` (``t1`` required). Its ``side`` column, if
        any, is replaced.
    close : pd.Series
        Prices, as in :func:`get_bins`.
    side : pd.Series or array
        Primary model's side per event, values in ``{-1, +1}``. A Series is
        aligned by label. An array is aligned by position.

    Returns
    -------
    pd.DataFrame
        Same as :func:`get_bins` with ``meta=True``. ``bin`` is ``1`` when the primary side was
        right (its P&L is positive), else ``0``.

    Notes
    -----
    Zero sides ("no bet") are rejected. Drop them before calling this function
    if the primary model abstains.
    """
    ev = events.copy()
    if isinstance(side, pd.Series):
        ev["side"] = side.reindex(ev.index).to_numpy(dtype=float)
    else:
        arr = np.asarray(side, dtype=float).reshape(-1)
        if arr.shape[0] != len(ev):
            raise ValueError("side array must have one entry per event.")
        ev["side"] = arr
    if ev["side"].isna().any():
        raise ValueError("side is missing for some events.")
    if not np.all(np.isin(ev["side"].to_numpy(), (-1.0, 1.0))):
        raise ValueError("side must equal -1 or +1 (drop abstentions first).")
    return get_bins(ev, close, meta=True)


# ---------------------------------------------------------------------------
# Snippet 3.8 -- drop rare labels
# ---------------------------------------------------------------------------


def drop_labels(events: pd.DataFrame, min_pct: float = 0.05) -> pd.DataFrame:
    """Recursively drop the rarest label until every class is common enough (Snippet 3.8).

    Parameters
    ----------
    events : pd.DataFrame
        Must contain a ``bin`` column.
    min_pct : float, default 0.05
        Minimum class frequency (a fraction of the rows that remain).

    Returns
    -------
    pd.DataFrame
        Subset of ``events`` (same order) with the dropped classes removed.

    Notes
    -----
    At each step the class with the smallest frequency is removed while that
    frequency is below ``min_pct`` and at least three classes remain. The
    book's rule ("unless only two classes are left") is kept: with two classes
    the loop stops even if one is below ``min_pct``. The loop terminates
    because each step removes one class and there are finitely many classes.

    Scope: removes whole classes only. It never removes single rows.
    """
    if not 0.0 <= min_pct < 1.0:
        raise ValueError("min_pct must be in [0, 1).")
    out = events
    while True:
        freq = out["bin"].value_counts(normalize=True)
        if freq.shape[0] < 3 or freq.min() >= min_pct:
            break
        out = out[out["bin"] != freq.idxmin()]
    return out


# ---------------------------------------------------------------------------
# Trend-scanning labels (Lopez de Prado 2019 / Hudson & Thames; not AFML Ch. 3)
# ---------------------------------------------------------------------------


@jit
def _slope_t(y: np.ndarray, start: int, length: int) -> float:
    """t-statistic of the OLS slope of ``y[start:start+length]`` on ``0..length-1``."""
    xbar = 0.5 * (length - 1)
    ybar = 0.0
    for k in range(length):
        ybar += y[start + k]
    ybar /= length
    sxx = length * (length * length - 1.0) / 12.0
    sxy = 0.0
    for k in range(length):
        sxy += (k - xbar) * (y[start + k] - ybar)
    b = sxy / sxx
    a = ybar - b * xbar
    sse = 0.0
    for k in range(length):
        resid = y[start + k] - a - b * k
        sse += resid * resid
    dof = length - 2
    s2 = sse / dof
    if s2 <= 0.0:
        if b == 0.0:
            return 0.0
        return np.inf if b > 0.0 else -np.inf
    se = np.sqrt(s2 / sxx)
    return b / se


@pjit
def _trend_scan_kernel(
    logp: np.ndarray, windows: np.ndarray, threshold: float
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """For every start bar, the window with the largest ``|t|`` and its label."""
    n = logp.shape[0]
    t_best = np.full(n, np.nan)
    w_best = np.full(n, np.nan)
    lab = np.full(n, np.nan)
    for i in prange(n):
        best_abs = -1.0
        best_t = 0.0
        best_w = 0
        for w in range(windows.shape[0]):
            length = windows[w]
            if i + length > logp.shape[0]:
                continue
            t = _slope_t(logp, i, length)
            if abs(t) > best_abs:
                best_abs = abs(t)
                best_t = t
                best_w = length
        if best_abs >= 0.0:
            t_best[i] = best_t
            w_best[i] = best_w
            if best_abs > threshold:
                lab[i] = 1.0 if best_t > 0.0 else -1.0
            else:
                lab[i] = 0.0
    return t_best, w_best, lab


def trend_scanning_labels(
    close: pd.Series,
    window_sizes: Sequence[int],
    t_threshold: float = 1.96,
) -> pd.DataFrame:
    """Trend-scanning label from the best-fitting forward window (not AFML Ch. 3).

    Parameters
    ----------
    close : pd.Series
        Strictly positive prices. Log prices are regressed.
    window_sizes : sequence of int
        Forward window lengths in bars (each ``>= 3``).
    t_threshold : float, default 1.96
        A label is non-zero only when the chosen ``|t|`` exceeds this value.
        This is a design choice, not a value taken from the literature.

    Returns
    -------
    pd.DataFrame
        Indexed by ``close.index``. Columns:

        ``t_value``
            Signed t-statistic of the slope of ``log(close)`` against time over
            the chosen window, starting at the bar.
        ``window``
            Length of the chosen window. Float, NaN where no window fits.
        ``bin``
            ``sign(t_value)`` if ``|t_value| > t_threshold``, else ``0``. NaN
            where no window fits (the last ``min(window_sizes) - 1`` bars).

    Notes
    -----
    Method, for bar ``i``:

    1. For every window length ``L`` with ``i + L <= n``, regress
       ``y_k = log close[i + k]`` on ``x_k = k`` for ``k = 0..L-1`` by OLS.
    2. ``t = b / SE(b)``, where ``SE(b)^2 = s^2 / Sxx`` with
       ``s^2 = SSE / (L - 2)`` and ``Sxx = L (L^2 - 1) / 12``.
    3. Keep the ``L`` with the largest ``|t|`` (first one on a tie).
    4. Label ``sign(t)`` if ``|t| > t_threshold``, else ``0``.

    A perfect fit (``SSE = 0``) gives ``t = +-inf``. The per-bar loop is
    parallel (numba ``prange``) and the window loop is serial.

    Scope: forward-looking by construction. Each label uses bars ``i..i+L-1``.
    Do not use these labels as features.
    """
    if isinstance(close, pd.Series):
        px = _check_prices(close)
        index = close.index
    else:
        raise TypeError("close must be a pandas Series.")
    windows = np.asarray(sorted({int(w) for w in window_sizes}), dtype=np.int64)
    if windows.size == 0 or windows.min() < 3:
        raise ValueError("window_sizes must be non-empty integers >= 3.")
    if windows.max() > len(px):
        windows = windows[windows <= len(px)]
        if windows.size == 0:
            raise ValueError("every window is longer than the series.")
    t_best, w_best, lab = _trend_scan_kernel(
        np.log(px), windows, float(t_threshold)
    )
    return pd.DataFrame(
        {"t_value": t_best, "window": w_best, "bin": lab}, index=index
    )
