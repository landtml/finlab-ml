"""Bet sizing from predicted probabilities and forecasts (AFML ch. 10).

Functions here translate model output into a position size in ``[-1, 1]`` (or
into an integer target position) and handle the book's practical concerns:
averaging overlapping bets, discretization to limit overtrading, and dynamic
sizing as the market price moves toward the forecast.

Formulas (AFML section numbers in brackets):

* Probability to size, two outcomes [10.3]: ``z = (p - 1/2) / sqrt(p (1 - p))``,
  ``m = 2 Phi(z) - 1``.
* Probability to size, one-vs-rest [10.3]: with ``p~ = max_i p_i`` and ``k``
  outcomes, ``z = (p~ - 1/k) / sqrt(p~ (1 - p~))``, ``m = x (2 Phi(z) - 1)``
  where ``x`` is the predicted label.
* Sigmoid dynamic sizing [10.6]: ``m(w, x) = x / sqrt(w + x^2)`` with divergence
  ``x = f - p``; the width is calibrated from a target pair via
  ``w = x^2 (m*^-2 - 1)``.
* Target position [10.6]: ``q = int(m(w, f - p) * Q)`` for maximum position ``Q``.
* Breakeven limit price [10.6]: the price at which the order becomes unprofitable
  given the sigmoid inverse ``L(f, w, m) = f - m sqrt(w / (1 - m^2))``.
* Discretization [10.5]: ``m* = round(m / d) * d`` with step ``d``.

Scope
-----
Bet sizes are a function of the probability or forecast you provide; this module
does not estimate them. The concurrency-based averaging of :func:`average_active_signals`
uses the event lifespans ``t1`` and is a plain sweep over positions.

Not covered
-----------
The mixture-of-Gaussians budgeting approach [10.2] (it needs a fitted mixture
model) and the meta-labeling pipeline itself (see :mod:`finlab.labeling`).
"""

from __future__ import annotations

import math
import warnings

import numpy as np
import numpy.typing as npt
import pandas as pd
from scipy.special import ndtr

from ._checks import _require_type

__all__ = [
    "prob_bet_size",
    "sigmoid_bet_size",
    "calibrate_sigmoid_width",
    "target_position",
    "limit_price",
    "average_active_signals",
    "discretize_signal",
]

FloatArray = npt.NDArray[np.float64]


def prob_bet_size(
    prob: npt.ArrayLike,
    side: npt.ArrayLike | None = None,
    num_classes: int = 2,
) -> FloatArray:
    """Map predicted probabilities to bet sizes in ``[-1, 1]``.

    Parameters
    ----------
    prob : array-like
        Predicted probability of the outcome. For ``num_classes == 2`` this is the
        probability of label ``+1``. For ``num_classes > 2`` it is the probability
        of the predicted label (the maximum class probability).
    side : array-like, optional
        Predicted label in ``{-1, +1}``. Gives the sign of the bet. Required when
        ``num_classes > 2``; for two classes it is applied when passed and the bet
        is long when it is omitted.
    num_classes : int, default 2
        Number of possible labels ``k``.

    Returns
    -------
    numpy.ndarray
        Bet sizes ``m = 2 Phi(z) - 1`` (two classes) or ``side * (2 Phi(z) - 1)``.

    Notes
    -----
    Probabilities of exactly 0 or 1 give ``z = +/- inf`` and bet sizes of exactly
    ``+/- 1``; probabilities outside ``[0, 1]`` are rejected, and for
    ``num_classes > 2`` so are probabilities below ``1/num_classes``.
    """
    p = np.asarray(prob, dtype=np.float64)
    if np.any((p < 0) | (p > 1)) or np.any(np.isnan(p)):
        raise ValueError("probabilities must lie in [0, 1].")
    if num_classes < 2:
        raise ValueError(f"num_classes must be >= 2, got {num_classes!r}.")
    s = None if side is None else np.asarray(side, dtype=np.float64)
    if s is not None and s.ndim > 0 and s.shape != p.shape:
        # Checked here so the mismatch names both lengths, not a numpy broadcast error.
        n_prob = p.shape[0] if p.ndim else 1
        raise ValueError(f"side has length {s.shape[0]} but prob has length {n_prob}.")
    if s is not None and not np.all(np.isin(s, (-1.0, 1.0))):
        raise ValueError("side values must be -1 or +1.")
    with np.errstate(divide="ignore", invalid="ignore"):
        if num_classes == 2:
            z = (p - 0.5) / np.sqrt(p * (1.0 - p))
            m = 2.0 * ndtr(z) - 1.0
            if s is not None:
                m = s * m
        else:
            if s is None:
                raise ValueError("side is required when num_classes > 2.")
            if np.any(p < 1.0 / num_classes):
                # Below the uniform probability the bet would point against the predicted label.
                raise ValueError(f"probabilities must be at least 1/num_classes = {1.0 / num_classes!r}.")
            z =(p - 1.0 / num_classes) / np.sqrt(p * (1.0 - p))
            m = s * (2.0 * ndtr(z) - 1.0)
    # Degenerate endpoints: p in {0, 1} gives z = +/- inf; ndtr handles it, but the
    # 0/0 limits of the two-class case at p = 0.5 are exactly 0.
    return np.where(np.isnan(m) & (p == 0.5), 0.0, m)


def sigmoid_bet_size(w: float, x: npt.ArrayLike) -> FloatArray:
    """Sigmoid bet size ``m(w, x) = x / sqrt(w + x^2)`` [10.6].

    Parameters
    ----------
    w : float
        Width coefficient ``w > 0``. Larger values give smaller bets for the same
        divergence.
    x : array-like
        Divergence ``f - p`` between forecast and market price.

    Returns
    -------
    numpy.ndarray
        Values in ``(-1, 1)`` for finite ``x``.
    """
    if w <= 0:
        raise ValueError(f"w must be > 0, got {w!r}.")
    x = np.asarray(x, dtype=np.float64)
    # hypot avoids the overflow of x * x for very large |x|, where the bet is at its limit.
    return x / np.hypot(math.sqrt(w), x)


def calibrate_sigmoid_width(x: float, m_star: float) -> float:
    """Width ``w`` such that ``m(w, x) == m_star`` [10.6].

    Solves ``m = x / sqrt(w + x^2)`` for ``w``: ``w = x^2 (m*^-2 - 1)``.

    Parameters
    ----------
    x : float
        A nonzero divergence at which the bet size is specified.
    m_star : float
        Desired bet size in ``(0, 1)``.

    Examples
    --------
    >>> round(calibrate_sigmoid_width(10.0, 0.95), 4)
    10.8033
    """
    if x == 0:
        raise ValueError("x must be nonzero to calibrate the sigmoid width.")
    if not 0.0 < abs(m_star) < 1.0:
        raise ValueError(f"|m_star| must be in (0, 1), got {m_star!r}.")
    if not math.isfinite(x):
        # An infinite divergence yields an infinite width; the caller should know why.
        warnings.warn(
            "calibrate_sigmoid_width: non-finite x gives a non-finite width",
            UserWarning,
            stacklevel=2,
        )
    return float(x * x * (abs(m_star) ** -2 - 1.0))


def target_position(w: float, forecast: float, market_price: float, max_position: int) -> int:
    """Integer target position ``int(m(w, f - p) * Q)`` [10.6].

    Truncates toward zero, as the book does, so the position never exceeds
    ``max_position`` in absolute value.
    """
    if max_position < 1:
        raise ValueError(f"max_position must be >= 1, got {max_position!r}.")
    m = float(sigmoid_bet_size(w, forecast - market_price))
    return int(m * max_position)


def limit_price(
    target: int,
    current: int,
    forecast: float,
    w: float,
    max_position: int,
) -> float:
    """Breakeven limit price for moving from ``current`` to ``target`` [10.6].

    Implements the book's formula
    ``p_bar = (1/|target - current|) sum_p L(f, w, p / Q)`` where ``p`` runs over
    the signed position after each unit fill, from ``current + sgn(target - current)``
    to ``target``. ``L(f, w, m) = f - m sqrt(w / (1 - m^2))`` inverts the sigmoid in
    the market price. Returns ``nan`` when ``target == current`` (no order) and when
    ``abs(target) == max_position`` (the cap term has no finite limit). Raises
    ``ValueError`` when ``abs(target)`` or ``abs(current)`` exceeds ``max_position``.

    Notes
    -----
    Pricing an order between the current price and the forecast: the returned
    value lies between them, and approaches the forecast as the order is filled.
    """
    if max_position < 1:
        raise ValueError(f"max_position must be >= 1, got {max_position!r}.")
    if abs(target) > max_position:
        raise ValueError(f"target {target!r} exceeds max_position {max_position!r}.")
    if abs(current) > max_position:
        raise ValueError(f"current {current!r} exceeds max_position {max_position!r}.")
    if target == current:
        return float("nan")
    if abs(target) == max_position:
        return float("nan")  # the cap term m == 1 has no finite limit price
    step = 1 if target > current else -1
    # Each unit fill is priced at the signed position it reaches, so an order that crosses zero
    # averages over |target - current| fills on both sides of zero.
    n_fills = abs(target - current)
    total = 0.0
    position = current
    for _ in range(n_fills):
        position += step
        m = position / max_position
        total += forecast - m * math.sqrt(w / (1.0 - m * m))
    return total / n_fills


def average_active_signals(
    signals: pd.Series,
    t1: pd.Series,
    index: pd.Index | None = None,
) -> pd.Series:
    """Average the signals of all bets active at each timestamp [10.4].

    Parameters
    ----------
    signals : pd.Series
        Bet signal per event, indexed by event start time.
    t1 : pd.Series
        Event end time per event (aligned with ``signals``).
    index : pd.Index, optional
        Timestamps at which to report the average. Defaults to ``signals.index``.

    Returns
    -------
    pd.Series
        Mean of the signals of active events at each timestamp; ``0`` where no
        event is active. A timestamp where a NaN signal is active is NaN; a ``+inf``
        (``-inf``) signal makes it ``+inf`` (``-inf``), or NaN when both are active.

    Raises
    ------
    ValueError
        If ``t1`` contains NaT, or ``signals`` and ``t1`` have different indexes.

    Notes
    -----
    Implemented as a difference-array sweep over positions, so the cost is
    ``O(n log n)`` in the sort, not quadratic in the number of overlapping events.
    """
    _require_type(signals, "signals", pd.Series)
    _require_type(t1, "t1", pd.Series)
    if not signals.index.equals(t1.index):
        raise ValueError("signals and t1 must share an index.")
    nat_rows = t1.index[t1.isna().to_numpy()]
    if len(nat_rows) > 0:
        # searchsorted sends NaT past the grid, which would keep the event active forever.
        raise ValueError(f"t1 contains NaT at rows {list(nat_rows)}; every event needs an end time.")
    before_start = signals.index[t1.to_numpy() < signals.index.to_numpy()]
    if len(before_start) > 0:
        # An end before its start would make the event silently disappear from the average.
        raise ValueError(
            f"t1 is before the start at rows {list(before_start)}; every event must end at or after it."
        )
    if index is None:
        index = signals.index
    grid = pd.Index(index).sort_values()
    start = grid.searchsorted(signals.index.values, side="left")
    # Position of the last grid point not after t1.
    end = grid.searchsorted(t1.values, side="right") - 1
    valid = (end >= start) & (start < len(grid))
    n = len(grid)
    total = np.zeros(n + 1)
    count = np.zeros(n + 1)
    vals = signals.to_numpy(dtype=np.float64)[valid]
    s = start[valid]
    e = end[valid]
    # Non-finite signals are left out of the sums and tracked separately, so they do not spread past their span.
    is_nan = np.isnan(vals)
    is_pinf = np.isposinf(vals)
    is_ninf = np.isneginf(vals)
    special = is_nan | is_pinf | is_ninf
    clean = np.where(special, 0.0, vals)
    present = (~special).astype(np.float64)
    np.add.at(total, s, clean)
    np.add.at(total, e + 1, -clean)
    np.add.at(count, s, present)
    np.add.at(count, e + 1, -present)
    flags = {}
    for name, mask in (("nan", is_nan), ("pinf", is_pinf), ("ninf", is_ninf)):
        flag = np.zeros(n + 1)
        np.add.at(flag, s, mask.astype(np.float64))
        np.add.at(flag, e + 1, -mask.astype(np.float64))
        flags[name] = np.cumsum(flag)[:n] > 0
    total = np.cumsum(total)[:n]
    count = np.cumsum(count)[:n]
    with np.errstate(divide="ignore", invalid="ignore"):
        avg = np.where(count > 0, total / np.maximum(count, 1), 0.0)
    avg = np.where(flags["pinf"], np.inf, avg)
    avg = np.where(flags["ninf"], -np.inf, avg)
    avg = np.where(flags["pinf"] & flags["ninf"], np.nan, avg)
    avg = np.where(flags["nan"], np.nan, avg)
    return pd.Series(avg, index=grid, name="signal")


def discretize_signal(signal: npt.ArrayLike, step: float) -> FloatArray:
    """Round bet sizes to multiples of ``step`` to limit overtrading [10.5].

    ``m* = round(m / step) * step``. ``step`` must be in ``(0, 1]``.
    """
    if not 0.0 < step <= 1.0:
        raise ValueError(f"step must be in (0, 1], got {step!r}.")
    s = np.asarray(signal, dtype=np.float64)
    if np.isinf(s).any():
        # Infinite bet sizes survive rounding as infinite targets; flag them instead of passing them on.
        warnings.warn(
            "discretize_signal: non-finite (inf) signal element, output contains inf",
            UserWarning,
            stacklevel=2,
        )
    return np.round(s / step) * step
