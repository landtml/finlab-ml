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

import numpy as np
import numpy.typing as npt
import pandas as pd
from scipy.special import ndtr

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
        Predicted label in ``{-1, +1}`` (only used when ``num_classes > 2``).
        Gives the sign of the bet.
    num_classes : int, default 2
        Number of possible labels ``k``.

    Returns
    -------
    numpy.ndarray
        Bet sizes ``m = 2 Phi(z) - 1`` (two classes) or ``side * (2 Phi(z) - 1)``.

    Notes
    -----
    Probabilities of exactly 0 or 1 give ``z = +/- inf`` and bet sizes of exactly
    ``+/- 1``; probabilities outside ``[0, 1]`` are rejected.
    """
    p = np.asarray(prob, dtype=np.float64)
    if np.any((p < 0) | (p > 1)) or np.any(np.isnan(p)):
        raise ValueError("probabilities must lie in [0, 1].")
    if num_classes < 2:
        raise ValueError(f"num_classes must be >= 2, got {num_classes!r}.")
    with np.errstate(divide="ignore", invalid="ignore"):
        if num_classes == 2:
            z = (p - 0.5) / np.sqrt(p * (1.0 - p))
            m = 2.0 * ndtr(z) - 1.0
        else:
            if side is None:
                raise ValueError("side is required when num_classes > 2.")
            s = np.asarray(side, dtype=np.float64)
            z = (p - 1.0 / num_classes) / np.sqrt(p * (1.0 - p))
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
    return x / np.sqrt(w + x * x)


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
    ``p_bar = (1/|target - current|) sum_j L(f, w, j / Q)`` over
    ``j = |current + sgn(target - current)| .. |target|``, where
    ``L(f, w, m) = f - m sqrt(w / (1 - m^2))`` inverts the sigmoid in the market
    price. Returns ``nan`` when ``target == current`` (no order).

    Notes
    -----
    Pricing an order between the current price and the forecast: the returned
    value lies between them, and approaches the forecast as the order is filled.
    """
    if max_position < 1:
        raise ValueError(f"max_position must be >= 1, got {max_position!r}.")
    if target == current:
        return float("nan")
    step = 1 if target > current else -1
    lo = abs(current + step)
    hi = abs(target)
    if hi < lo:
        return float("nan")
    total = 0.0
    for j in range(lo, hi + 1):
        m = j / max_position
        total += forecast - m * math.sqrt(w / (1.0 - m * m))
    return total / (hi - lo + 1)


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
        event is active.

    Notes
    -----
    Implemented as a difference-array sweep over positions, so the cost is
    ``O(n log n)`` in the sort, not quadratic in the number of overlapping events.
    """
    if not signals.index.equals(t1.index):
        raise ValueError("signals and t1 must share an index.")
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
    np.add.at(total, s, vals)
    np.add.at(total, e + 1, -vals)
    np.add.at(count, s, 1.0)
    np.add.at(count, e + 1, -1.0)
    total = np.cumsum(total)[:n]
    count = np.cumsum(count)[:n]
    with np.errstate(divide="ignore", invalid="ignore"):
        avg = np.where(count > 0, total / np.maximum(count, 1), 0.0)
    return pd.Series(avg, index=grid, name="signal")


def discretize_signal(signal: npt.ArrayLike, step: float) -> FloatArray:
    """Round bet sizes to multiples of ``step`` to limit overtrading [10.5].

    ``m* = round(m / step) * step``. ``step`` must be in ``(0, 1]``.
    """
    if not 0.0 < step <= 1.0:
        raise ValueError(f"step must be in (0, 1], got {step!r}.")
    s = np.asarray(signal, dtype=np.float64)
    return np.round(s / step) * step
