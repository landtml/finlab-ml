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
import numbers
import warnings

import numpy as np
import numpy.typing as npt
import pandas as pd
from scipy.special import ndtr

from ._checks import _require_type, _same_index, as_datetime_index, tz_mismatch_message

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


def _as_integer(value: object) -> int | None:
    """Return ``value`` as an int when it is a whole number, else None. bool is never whole here."""
    if isinstance(value, (bool, np.bool_)):
        return None
    if isinstance(value, numbers.Integral):
        return int(value)
    if isinstance(value, numbers.Real) and math.isfinite(value) and float(value).is_integer():
        return int(math.floor(value))
    return None


def _align_side_to_prob(prob: pd.Series, side: pd.Series) -> pd.Series:
    """Return ``side`` reindexed to the labels of ``prob``, refusing labels that do not match."""
    if not (prob.index.is_unique and side.index.is_unique):
        raise ValueError("prob and side labels differ or repeat: both need unique labels to align by label.")
    missing_side = int(np.count_nonzero(~prob.index.isin(side.index)))
    missing_prob = int(np.count_nonzero(~side.index.isin(prob.index)))
    if missing_side or missing_prob:
        raise ValueError(
            f"prob and side labels differ: {missing_side} prob label(s) have no side, "
            f"{missing_prob} side label(s) have no prob."
        )
    return side.reindex(prob.index)


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
        is long when it is omitted. When ``prob`` and ``side`` are both pandas
        Series, ``side`` is matched to ``prob`` by label; labels that do not match
        on both sides raise ``ValueError``. Arrays are matched by position.
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
    if isinstance(prob, pd.Series) and isinstance(side, pd.Series) and not prob.index.equals(side.index):
        # Labels decide the match when both inputs carry them; arrays stay positional.
        side = _align_side_to_prob(prob, side)
    # Adding +0.0 turns a signed zero into +0.0: -0.0 would flip the sign of z at p = 0.
    p = np.asarray(prob, dtype=np.float64) + 0.0
    if np.any((p < 0) | (p > 1)) or np.any(np.isnan(p)):
        raise ValueError("probabilities must lie in [0, 1].")
    whole = _as_integer(num_classes)
    if whole is None:
        # A non-integral float such as 2.5 must not take the "> 2" branch, so it is refused.
        # A whole float such as 2.0 is accepted: only a non-integral value is refused.
        raise ValueError(f"num_classes must be an integer, got {num_classes!r}.")
    num_classes = whole
    if num_classes < 2:
        raise ValueError(f"num_classes must be >= 2, got {num_classes!r}.")
    s = None if side is None else np.asarray(side, dtype=np.float64)
    if s is not None and s.ndim > 0 and s.shape != p.shape:
        # Checked here so the mismatch names both lengths, not a numpy broadcast error.
        if p.ndim == 0:
            # A scalar prob has no length, so say the shapes instead of "length 1 vs 1".
            raise ValueError(f"side shape {s.shape} does not match prob shape ().")
        raise ValueError(f"side has length {s.shape[0]} but prob has length {p.shape[0]}.")
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
    if not w > 0:  # also rejects NaN, which "w <= 0" let through
        raise ValueError(f"w must be > 0, got {w!r}.")
    x = np.asarray(x, dtype=np.float64)
    # hypot avoids the overflow of x * x for very large |x|, where the bet is at its limit.
    # At x = +-inf the ratio is inf / inf, so take the limit +-1 directly.
    with np.errstate(invalid="ignore"):
        bet = x / np.hypot(math.sqrt(w), x)
    return np.where(np.isinf(x), np.sign(x), bet)[()]  # [()] keeps a scalar input scalar


def calibrate_sigmoid_width(x: float, m_star: float) -> float:
    """Width ``w`` such that ``m(w, x) == m_star`` [10.6].

    Solves ``m = x / sqrt(w + x^2)`` for ``w``: ``w = x^2 (m*^-2 - 1)``.

    Parameters
    ----------
    x : float
        A positive divergence at which the bet size is specified. A negative
        divergence is rejected, since the bet sign follows ``x``.
    m_star : float
        Desired bet size in ``(0, 1)``.

    Examples
    --------
    >>> round(calibrate_sigmoid_width(10.0, 0.95), 4)
    10.8033
    """
    if x == 0:
        raise ValueError("x must be nonzero to calibrate the sigmoid width.")
    if x < 0:
        raise ValueError(f"x must be positive to calibrate the sigmoid width, got {x!r}.")
    if m_star < 0:
        # For x > 0 the bet is always positive, so a negative target is unattainable, not a mirror.
        raise ValueError(f"m_star must be positive for x > 0, got {m_star!r}.")
    if not 0.0 < abs(m_star) < 1.0:
        raise ValueError(f"|m_star| must be in (0, 1), got {m_star!r}.")
    if not math.isfinite(x):
        # An infinite divergence yields an infinite width; the caller should know why.
        warnings.warn(
            "calibrate_sigmoid_width: non-finite x gives a non-finite width",
            UserWarning,
            stacklevel=2,
        )
    # w = x^2 (m^-2 - 1) = (x/m)^2 (1 - m^2). The scaling comes first and the square is a plain
    # product, so a representable width is not lost to an overflow of x * x, and a tiny m
    # gives an inf width, which the overflow check below reports. The ** operator is avoided
    # because Python raises OverflowError from it.
    m = float(abs(m_star))
    y = float(x) / m
    q = y * y
    if math.isfinite(q):
        # Square first, then scale by (1 - m^2): exact for representable inputs, e.g. w = 300 at
        # x = 10, m = 0.5. The root form below rounds that to 299.99999999999994.
        # 1 - m * m is written as (1 - m)(1 + m): the factored form keeps its relative error near 1e-15
        # when m is close to 1, where the subtraction would cancel.
        w = float(q * ((1.0 - m) * (1.0 + m)))
    else:
        # y * y overflows above about 1.3e154, so the factor is rooted first to keep the product finite.
        ys = y * math.sqrt((1.0 - m) * (1.0 + m))
        w = float(ys * ys)
    if math.isfinite(x) and w < np.finfo(np.float64).tiny:
        # A zero width breaks sigmoid_bet_size, and a subnormal one may not calibrate
        # back. A subnormal width that does calibrate back is kept, so the round
        # trip decides, not the magnitude.
        back = float(sigmoid_bet_size(w, x)) if w > 0.0 else float("nan")
        # A subnormal w is off by up to half the smallest subnormal step. Its effect on the bet is
        # |dm/dw| * err = (m / 2) / (x^2 + w) * err. The whole step is used as the error (twice the
        # rounding bound, so it is safe), and that bound is added to the 1e-9 floor. Main returned
        # 3e-320 for (1e-160, 0.5); the round trip decides, not the magnitude.
        step = float(np.nextafter(0.0, 1.0))
        rounding = (step / (float(x) * float(x) + w)) * (abs(m_star) / 2.0) if w > 0.0 else float("inf")
        # x * x underflows for |x| below about 1.5e-162, so the denominator above is w alone and
        # the bound reaches m / 2 for a width of one step. A subnormal width is only kept when its
        # rounding moves the bet by at most 1e-3, the precision the subnormal tests check.
        if not (rounding <= 1e-3 and abs(back - m_star) <= 1e-9 + rounding):
            raise ValueError(f"the width underflows for x = {x!r}; x is too small to calibrate.")
    if math.isfinite(x) and not math.isfinite(w):
        # x * x overflows for |x| above about 1.3e154; an inf width would calibrate to a bet of 0.
        raise ValueError(f"the width overflows for x = {x!r}; x is too large to calibrate.")
    return w


def _require_position_int(name: str, value: object) -> None:
    """Reject anything that is not an integer position; bool is refused too."""
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, numbers.Integral):
        raise ValueError(f"{name} must be an integer, got {value!r}.")


def target_position(w: float, forecast: float, market_price: float, max_position: int) -> int:
    """Integer target position ``int(m(w, f - p) * Q)`` [10.6].

    Truncates toward zero, as the book does, so the position never exceeds
    ``max_position`` in absolute value.
    """
    _require_position_int("max_position", max_position)
    if max_position < 1:
        raise ValueError(f"max_position must be >= 1, got {max_position!r}.")
    # Same checks as limit_price: an infinite width or price must not give 0 or a full short.
    if not (math.isfinite(w) and w > 0):
        raise ValueError(f"w must be > 0 and finite, got {w!r}.")
    if not math.isfinite(market_price):
        raise ValueError(f"market_price must be finite, got {market_price!r}.")
    if math.isnan(forecast):
        raise ValueError(f"forecast must not be NaN, got {forecast!r}.")
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
    ``ValueError`` when ``abs(target)`` or ``abs(current)`` exceeds ``max_position``,
    or when ``w`` is negative or not finite.

    Notes
    -----
    Pricing an order between the current price and the forecast: the returned
    value lies between them, and approaches the forecast as the order is filled.
    """
    # Positions are whole units; a float target or cap made range() or the cap arithmetic fail or
    # silently truncate, so both must be integers.
    for name, value in (("target", target), ("current", current), ("max_position", max_position)):
        _require_position_int(name, value)
    if max_position < 1:
        raise ValueError(f"max_position must be >= 1, got {max_position!r}.")
    if abs(target) > max_position:
        raise ValueError(f"target {target!r} exceeds max_position {max_position!r}.")
    if abs(current) > max_position:
        raise ValueError(f"current {current!r} exceeds max_position {max_position!r}.")
    if not (w > 0 and math.isfinite(w)):
        # w = 0 is refused as in target_position and sigmoid_bet_size; a negative width has no
        # square root, and an infinite one gives an infinite, non-price limit.
        raise ValueError(f"w must be > 0 and finite, got {w!r}.")
    if not math.isfinite(forecast):
        # A NaN forecast would pass through every fill as a NaN limit price.
        raise ValueError(f"forecast must be finite, got {forecast!r}.")
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


def _is_aware(index: pd.Index) -> bool:
    """True for a tz-aware datetime index. A plain Index has no timezone, so it is naive."""
    return isinstance(index, pd.DatetimeIndex) and index.tz is not None


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
    if not _same_index(signals.index, t1.index):
        raise ValueError(
            tz_mismatch_message(signals.index, t1.index, "signals", "t1")
            or "signals and t1 must share an index."
        )
    nat_rows = t1.index[t1.isna().to_numpy()]
    if len(nat_rows) > 0:
        # searchsorted sends NaT past the grid, which would keep the event active forever.
        raise ValueError(f"t1 contains NaT at rows {list(nat_rows)}; every event needs an end time.")
    sig_idx = as_datetime_index(pd.Index(signals.index))
    t1_idx = as_datetime_index(pd.Index(t1))
    if len({_is_aware(i) for i in (sig_idx, t1_idx)}) > 1:
        # Signals and t1 with mixed awareness: compare on the UTC wall clock.
        sig_idx, t1_idx = (
            i.tz_convert("UTC").tz_localize(None) if _is_aware(i) else i for i in (sig_idx, t1_idx)
        )
    try:
        before_start = signals.index[np.asarray(t1_idx < sig_idx, dtype=bool)]
    except TypeError as exc:
        # An integer index with datetime end times is not a comparison the sweep can make.
        raise ValueError("signals and t1 hold values that cannot be compared (mixed types).") from exc
    if len(before_start) > 0:
        # An end before its start would make the event silently disappear from the average.
        raise ValueError(
            f"t1 is before the start at rows {list(before_start)}; every event must end at or after it."
        )
    if index is None:
        index = signals.index
    grid = pd.Index(index).sort_values()
    # The output keeps the caller's index (and timezone); only the comparison copies are converted.
    out_index = grid
    grid = as_datetime_index(grid)
    if len({_is_aware(i) for i in (grid, sig_idx, t1_idx)}) > 1:
        # Naive and tz-aware values cannot be compared. The aware values are compared on their UTC
        # wall clock: the aware side is put on UTC and the timezone dropped for the sweep.
        grid, sig_idx, t1_idx = (
            i.tz_convert("UTC").tz_localize(None) if _is_aware(i) else i
            for i in (grid, sig_idx, t1_idx)
        )
    # Pass Index objects, not .values: .values drops the timezone of a tz-aware index.
    try:
        start = grid.searchsorted(sig_idx, side="left")
        # Position of the last grid point not after t1.
        end = grid.searchsorted(t1_idx, side="right") - 1
    except TypeError as exc:
        raise ValueError("signals and t1 hold values that cannot be compared (mixed types).") from exc
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
    return pd.Series(avg, index=out_index, name="signal")


def discretize_signal(signal: npt.ArrayLike, step: float) -> FloatArray:
    """Round bet sizes to multiples of ``step`` to limit overtrading [10.5].

    ``m* = round(m / step) * step``, clipped to ``[-1, 1]``. The clip matters only when
    ``step`` does not divide 1 (for example 0.6 rounds 1.0 up to 1.2), so the result
    is then the endpoint ``+/-1``, which is not a multiple of ``step``. ``step`` must be
    in ``(0, 1]``, and every signal must be finite and lie in ``[-1, 1]``; otherwise
    ``ValueError`` is raised. Floor: ``signal / step`` overflows float64 when ``step`` is
    below about ``|signal| * 5.6e-309`` (about 1.7e-309 for a signal of 0.3); such a step
    is refused with ``ValueError`` rather than returned as an endpoint.
    """
    if not 0.0 < step <= 1.0:
        raise ValueError(f"step must be in (0, 1], got {step!r}.")
    s = np.asarray(signal, dtype=np.float64)
    if not np.all(np.isfinite(s) & (np.abs(s) <= 1.0)):
        # Bet sizes live in [-1, 1]; values outside it would pass through as oversized targets.
        raise ValueError("discretize_signal: signal values must be finite and lie in [-1, 1].")
    with np.errstate(over="ignore"):
        q = s / step
    if not np.all(np.isfinite(q)):
        # A step this small makes s / step overflow, and the clip would turn the bet into +/-1.
        raise ValueError(f"step is too small: signal / step overflows for step = {step!r}.")
    return np.clip(np.round(q) * step, -1.0, 1.0)
