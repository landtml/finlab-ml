"""Structural break tests (AFML ch. 17).

Implements, following the book's notation:

* :func:`cusum_test` -- Brown-Durbin-Evans CUSUM test on recursive residuals
  (AFML 17.3.1).
* :func:`chu_stinchcombe_white` -- Chu-Stinchcombe-White CUSUM test on levels
  (AFML 17.3.2), either from a fixed reference point ``n`` or as the supremum
  over backward-shifting reference points.
* :func:`adf_stat` -- the Dickey-Fuller tau-statistic of the regression in
  AFML 17.4.2 on a single window.
* :func:`sadf` -- the Supremum Augmented Dickey-Fuller series (AFML 17.4.2,
  Snippet 17.1 inner loop wrapped in the outer end-point loop).
* :func:`qadf` and :func:`cadf` -- the quantile and conditional ADF series over the
  same ADF statistics that :func:`sadf` maximises (AFML 17.4.2).
* :func:`chow_dfc` and :func:`sdfc` -- the Chow-type Dickey-Fuller statistic at
  one break fraction, and its supremum over admissible breaks (AFML ch. 17).
* :func:`smt` -- the sub- and super-martingale test statistic, the supremum over
  start points of a curvature or growth coefficient (AFML 17.4.3).

Sign and input conventions
--------------------------
* Explosiveness tests (CSW, SADF, QADF, CADF and SDFC) are one-sided and look for
  *upward* departures: a large positive statistic means the series grew faster than a
  random walk. The ADF slope ``beta`` is tested with ``H0: beta <= 0`` against
  ``H1: beta > 0``, so SADF uses ``tau = beta_hat / se(beta_hat)`` unchanged.
* :func:`smt` takes the absolute value of its t-statistic, so it flags upward and
  downward curvature alike and is not one-sided.
* The book applies these tests to **log prices** (AFML 17.4.2.1). The functions
  here do not take logs for you; pass ``np.log(prices)`` if you want the book's
  setting.

Literal readings and deviations
-------------------------------
* Brown-Durbin-Evans: the book's ``f_t`` and ``sigma_omega`` formulas are
  garbled in the text. We use the standard recursive residual
  ``w_t = (y_t - x_t' b_{t-1}) / sqrt(1 + x_t' (X_{t-1}'X_{t-1})^{-1} x_t)``
  and ``sigma_w^2 = mean(w^2)`` (the book's ``E[w_t]`` is taken as 0, its value
  under H0). Starting from an initial block of ``k = p`` observations, the
  statistic ``S_t`` sums ``t - k`` recursive residuals, so its null variance is
  ``t - k`` (the book writes ``t - k - 1``; our count is explicit in
  :class:`CusumResult`).
* Chu-Stinchcombe-White: the critical value is read as
  ``c_alpha[n, t] = sqrt(b_alpha + log(t - n))``, with the book's ``b_0.05 = 4.6``.
  The book's printed radical sign is ambiguous, and this grouping is our reading.
* ADF constants follow the *code* of Snippet 17.2, not the prose: ``'nc'`` has no
  intercept, ``'c'`` adds an intercept, ``'ct'`` adds an intercept and a linear
  trend, ``'ctt'`` also adds a quadratic trend. The trend is the absolute
  position in the sample, as in the snippet. Standard OLS errors are used
  (``sigma^2 = SSR / (n - p)``).
* SADF window length: ``min_length`` counts *levels* in the smallest window,
  so the book's ``tau`` (minimum number of differences) corresponds to
  ``min_length - 1``.

Scope
-----
Not covered: the Brown-Durbin-Evans critical boundaries (not given in the book text, so no p-values are returned for
the CUSUM path); multi-bubble variants; and the parallel SADF implementation of ch. 20.
"""

from __future__ import annotations

import math
import numbers
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from ._checks import _same_index, tz_mismatch_message
from ._jit import jit

__all__ = [
    "CusumResult",
    "CSWResult",
    "SDFCResult",
    "cusum_test",
    "chu_stinchcombe_white",
    "csw_critical_value",
    "adf_stat",
    "sadf",
    "qadf",
    "cadf",
    "chow_dfc",
    "sdfc",
    "smt",
]

_CONSTANT_CODES = {"nc": 0, "c": 1, "ct": 2, "ctt": 3}
_CSW_B_05 = 4.6  # b_0.05 from AFML 17.3.2 (Monte Carlo, Chu-Stinchcombe-White)


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------


def _require_integer(value: Any, message: str) -> None:
    # Type and finiteness come first so that int() never sees NaN, inf or text. bool is an Integral,
    # so it is refused by name.
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, numbers.Real):
        raise ValueError(message)
    if not isinstance(value, numbers.Integral) and not math.isfinite(value):
        raise ValueError(message)
    if value != math.floor(value):
        raise ValueError(message)


def _require_real_in(
    value: Any,
    name: str,
    low: float,
    high: float,
    *,
    low_open: bool = False,
    high_open: bool = False,
) -> float:
    # bool is refused by name, as in _require_integer; NaN and inf fail the finiteness test.
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, numbers.Real) or not math.isfinite(value):
        raise ValueError(f"{name} must be a finite number, got {value!r}.")
    v = float(value)
    below = v <= low if low_open else v < low
    above = v >= high if high_open else v > high
    if below or above:
        left = "(" if low_open else "["
        right = ")" if high_open else "]"
        raise ValueError(f"{name} must be in {left}{low}, {high}{right}, got {value!r}.")
    return v


def _as_float_vector(series: Any) -> tuple[np.ndarray, pd.Index | None]:
    index = series.index if isinstance(series, (pd.Series, pd.DataFrame)) else None
    if index is not None and not index.is_monotonic_increasing:
        # The tests read values in array order, so a descending index gave the reversed series' statistics
        # under the original labels. Refused rather than sorted.
        raise ValueError("series index must be sorted in ascending order (oldest first).")
    values = np.asarray(series, dtype=np.float64)
    if values.ndim == 2 and values.shape[1] == 1:
        values = values[:, 0]
    if values.ndim != 1:
        raise ValueError(f"expected a 1-D series, got shape {values.shape}.")
    if not np.all(np.isfinite(values)):
        raise ValueError("series contains NaN or infinite values.")
    return np.ascontiguousarray(values), index


# Series whose largest magnitude lies outside [1/_SAFE_MAGNITUDE, _SAFE_MAGNITUDE] are rescaled
# before the ADF regression, whose Gram matrix squares the values.
_SAFE_MAGNITUDE = 1e100


def _rescale_for_adf(y: np.ndarray) -> np.ndarray:
    """Divide ``y`` by its largest magnitude when that is outside the safe range.

    The ADF tau-statistic does not change when the series is scaled, but the Gram matrix
    squares the values. Near 1e154 the squares overflow, and near 1e-154 they underflow, so
    the regression fails. Series inside the safe range are returned unchanged.
    """
    m = float(np.max(np.abs(y))) if y.size else 0.0
    if m > _SAFE_MAGNITUDE or 0.0 < m < 1.0 / _SAFE_MAGNITUDE:
        return y / m
    return y


def _large_scale(values: np.ndarray) -> float:
    """Largest magnitude of ``values`` when it is outside the safe range, else 1.0.

    Both sides are handled, as in ``_rescale_for_adf``: above 1e100 the squares overflow, and below
    1e-100 they underflow. The statistics are ratios to sigma, so the scale cancels.
    """
    m = float(np.max(np.abs(values))) if values.size else 0.0
    return m if (m > _SAFE_MAGNITUDE or 0.0 < m < 1.0 / _SAFE_MAGNITUDE) else 1.0


def _center_for_adf(y: np.ndarray, const: int) -> np.ndarray:
    # With an intercept in the model tau does not change under a level shift. At levels near 1e6 the
    # y[t-1] column makes the Gram matrix ill-conditioned, so centre it. 'nc' has no intercept.
    return y - y.mean() if const >= 1 and y.size else y


def _wrap(values: np.ndarray, index: pd.Index | None) -> Any:
    return values if index is None else pd.Series(values, index=index)


@jit
def _invert_small(a: np.ndarray) -> np.ndarray:
    """Gauss-Jordan inverse with partial pivoting. Returns NaNs if singular."""
    p = a.shape[0]
    m = np.zeros((p, 2 * p))
    for i in range(p):
        for j in range(p):
            m[i, j] = a[i, j]
        m[i, p + i] = 1.0
    for c in range(p):
        piv = c
        best = abs(m[c, c])
        for r in range(c + 1, p):
            if abs(m[r, c]) > best:
                best = abs(m[r, c])
                piv = r
        if best == 0.0:
            return np.full((p, p), np.nan)
        if piv != c:
            for j in range(2 * p):
                tmp = m[c, j]
                m[c, j] = m[piv, j]
                m[piv, j] = tmp
        d = m[c, c]
        for j in range(2 * p):
            m[c, j] /= d
        for r in range(p):
            if r != c:
                f = m[r, c]
                if f != 0.0:
                    for j in range(2 * p):
                        m[r, j] -= f * m[c, j]
    out = np.empty((p, p))
    for i in range(p):
        for j in range(p):
            out[i, j] = m[i, p + j]
    return out


# ---------------------------------------------------------------------------
# Brown-Durbin-Evans CUSUM on recursive residuals (AFML 17.3.1)
# ---------------------------------------------------------------------------


@jit
def _recursive_residuals(y: np.ndarray, x: np.ndarray) -> tuple[np.ndarray, bool]:
    """1-step-ahead recursive residuals of RLS, started from the first ``p`` rows."""
    n_obs, p = x.shape
    w = np.full(n_obs, np.nan)
    gram = np.zeros((p, p))
    xty = np.zeros(p)
    for t in range(p):
        for i in range(p):
            xty[i] += x[t, i] * y[t]
            for j in range(p):
                gram[i, j] += x[t, i] * x[t, j]
    inv = _invert_small(gram)
    if np.isnan(inv[0, 0]):
        return w, False
    beta = np.zeros(p)
    for i in range(p):
        for j in range(p):
            beta[i] += inv[i, j] * xty[j]
    px = np.zeros(p)
    for t in range(p, n_obs):
        for i in range(p):
            px[i] = 0.0
            for j in range(p):
                px[i] += inv[i, j] * x[t, j]
        f = 1.0
        pred = 0.0
        for i in range(p):
            f += x[t, i] * px[i]
            pred += x[t, i] * beta[i]
        err = y[t] - pred
        w[t] = err / math.sqrt(f)
        for i in range(p):
            beta[i] += px[i] * err / f
            for j in range(p):
                inv[i, j] -= px[i] * px[j] / f
    return w, True


@dataclass(frozen=True)
class CusumResult:
    """Output of :func:`cusum_test`.

    Attributes
    ----------
    statistic : np.ndarray
        ``S_t`` for every observation (0-based). Entries before ``initial_size``
        are NaN. ``S_t`` is the cumulative recursive residual sum divided by
        ``sigma``.
    z : np.ndarray
        ``S_t / sqrt(t - k + 1)``, which is N(0, 1) under H0 with known sigma
        (the number of residuals summed at position ``t`` is ``t - k + 1``
        0-based).
    residuals : np.ndarray
        Recursive residuals ``w_t`` (NaN for the first ``initial_size`` points).
    sigma : float
        ``sigma_w = sqrt(mean(w_t^2))`` over the residuals.
    initial_size : int
        ``k``, the number of observations used to start the recursion.
    """

    statistic: np.ndarray
    z: np.ndarray
    residuals: np.ndarray
    sigma: float
    initial_size: int


_MAGNITUDE_MESSAGE = (
    "recursive residuals are not finite or underflow; the series magnitude may be outside "
    "the usable range, so rescale the series"
)


def _extreme_magnitude(*arrays: np.ndarray) -> bool:
    # Squared residuals overflow above about 1e154 and underflow below about 1e-154,
    # so a largest absolute value far outside [1e-100, 1e100] points at scale.
    hi = float(np.max(np.abs(np.concatenate([a.ravel() for a in arrays]))))
    return hi > 1e100 or 0.0 < hi < 1e-100


def cusum_test(series: Any, exog: Any = None) -> CusumResult:
    """Brown-Durbin-Evans CUSUM test on recursive residuals (AFML 17.3.1).

    Parameters
    ----------
    series : array-like or pd.Series
        Dependent variable ``y_t``, ordered in time.
    exog : array-like, optional
        Features ``x_t`` as a ``(T, p)`` array (1-D is one feature). If omitted,
        the design is a constant, which tests a mean shift in ``series``.

    Returns
    -------
    CusumResult
        Per-point statistics and the recursive residuals. A series with a mean
        shift at ``tau`` has ``|S_t|`` drifting away from zero after ``tau``.

    Notes
    -----
    The recursion starts from the first ``p`` rows of ``exog``, which must make
    ``X_p'X_p`` invertible (otherwise ``ValueError``). ``S_t`` at 0-based
    position ``t`` sums the residuals ``w_p, ..., w_t``. The book's critical
    boundaries are not part of the text, so none are applied here.

    Scope: only the CUSUM path of AFML 17.3.1. See the module docstring for the
    literal-reading decisions.
    """
    y, index = _as_float_vector(series)
    n_obs = y.shape[0]
    if exog is None:
        x = np.ones((n_obs, 1))
    else:
        # exog's labels are read too. A descending index and labels that differ from the series
        # are refused, as the series index already is.
        exog_index = exog.index if isinstance(exog, (pd.Series, pd.DataFrame)) else None
        if exog_index is not None:
            if not exog_index.is_monotonic_increasing:
                raise ValueError("exog index must be sorted in ascending order (oldest first).")
            # Same-instant labels in another zone or unit are aligned (shared with microstructure).
            if index is not None and not _same_index(exog_index, index):
                raise ValueError(
                    tz_mismatch_message(exog_index, index, "exog", "series")
                    or "exog and series must share the same index."
                )
        x = np.asarray(exog, dtype=np.float64)
        if x.ndim == 1:
            x = x.reshape(-1, 1)
        if x.shape[0] != n_obs:
            raise ValueError(f"exog has {x.shape[0]} rows, series has {n_obs}.")
        if not np.all(np.isfinite(x)):
            raise ValueError("exog contains NaN or infinite values.")
    x = np.ascontiguousarray(x)
    p = x.shape[1]
    if n_obs - p < 2:
        raise ValueError("need at least two recursive residuals (T - p >= 2).")

    # Both statistics are ratios of residual sums to sigma, so scaling y (or a column of exog) by a
    # constant leaves them unchanged. The recursion runs on the scaled values so its squares stay finite
    # the residuals and sigma are reported back in the series' own units.
    y_scale = _large_scale(y)
    y_s = y / y_scale
    if exog is not None:
        col_scale = np.array([_large_scale(x[:, j]) for j in range(p)])
        x = x / col_scale
    w_s, ok = _recursive_residuals(y_s, x)
    if not ok:
        raise ValueError("initial design X_p'X_p is singular; use more leading rows.")
    resid_s = w_s[p:]
    sigma_s = float(math.sqrt(np.mean(resid_s * resid_s)))
    sigma = sigma_s * y_scale
    w = w_s * y_scale
    if not np.all(np.isfinite(w[p:])):
        # The residuals are reported in the series' units, and a residual past the float64 maximum
        # cannot be stored, so it is refused as main refused it instead of returning inf.
        raise ValueError(_MAGNITUDE_MESSAGE)
    resid = resid_s
    # The default constant regressor is ignored here; only user inputs can carry scale.
    scale_arrays = (y,) if exog is None else (y, x)
    if not math.isfinite(sigma_s):
        if _extreme_magnitude(*scale_arrays):
            raise ValueError(_MAGNITUDE_MESSAGE)
        raise ValueError(
            "recursive residuals are not finite; this is usually caused by near-collinear "
            "regressors. Remove one of the regressors."
        )
    if not sigma_s > 0.0:
        if np.all(y == y[0]):
            raise ValueError("recursive residuals are all zero; no variation in series.")
        if _extreme_magnitude(*scale_arrays):
            raise ValueError(_MAGNITUDE_MESSAGE)
        raise ValueError(
            "recursive residuals are all zero because the regressors reproduce the series "
            "exactly; the CUSUM test has nothing to measure."
        )

    stat = np.full(n_obs, np.nan)
    z = np.full(n_obs, np.nan)
    cum = np.cumsum(resid)
    counts = np.arange(1, resid.shape[0] + 1, dtype=np.float64)
    stat[p:] = cum / sigma_s
    z[p:] = cum / (sigma_s * np.sqrt(counts))
    return CusumResult(
        statistic=stat,
        z=z,
        residuals=w,
        sigma=sigma,
        initial_size=p,
    )


# ---------------------------------------------------------------------------
# Chu-Stinchcombe-White CUSUM on levels (AFML 17.3.2)
# ---------------------------------------------------------------------------


@jit
def _prefix_sigma(y: np.ndarray) -> np.ndarray:
    """sigma_t = sqrt(t^-1 sum_{i=1}^t (y_i - y_{i-1})^2) for 0-based t >= 1."""
    n_obs = y.shape[0]
    out = np.full(n_obs, np.nan)
    acc = 0.0
    for t in range(1, n_obs):
        d = y[t] - y[t - 1]
        acc += d * d
        out[t] = math.sqrt(acc / t)
    return out


@jit
def _csw_fixed(y: np.ndarray, sigma: np.ndarray, ref: int) -> np.ndarray:
    n_obs = y.shape[0]
    out = np.full(n_obs, np.nan)
    for t in range(ref + 1, n_obs):
        denom = sigma[t] * math.sqrt(t - ref)
        if denom == 0.0:
            # Flat prefix: sigma_t = 0 and the statistic is undefined, keep NaN.
            continue
        out[t] = (y[t] - y[ref]) / denom
    return out


@jit
def _csw_sup(y: np.ndarray, sigma: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    n_obs = y.shape[0]
    best = np.full(n_obs, np.nan)
    arg = np.full(n_obs, -1, dtype=np.int64)
    for t in range(1, n_obs):
        if sigma[t] == 0.0:
            # Flat prefix: the statistic is undefined, keep NaN with no argmax.
            continue
        bval = -np.inf
        barg = -1
        for n in range(t):
            s = (y[t] - y[n]) / (sigma[t] * math.sqrt(t - n))
            if s > bval:
                bval = s
                barg = n
        best[t] = bval
        arg[t] = barg
    return best, arg


def csw_critical_value(lag: int | np.ndarray, b: float = _CSW_B_05) -> np.ndarray:
    """One-sided critical value ``sqrt(b + log(t - n))`` (AFML 17.3.2).

    Parameters
    ----------
    lag : int or array of int
        ``t - n`` (must be >= 1).
    b : float, default 4.6
        Monte Carlo constant; the book gives ``b_0.05 = 4.6``.

    Returns
    -------
    np.ndarray
        Critical values with the same shape as ``lag``.
    """
    lag_arr = np.asarray(lag, dtype=np.float64)
    if not np.all(np.isfinite(lag_arr)) or np.any(lag_arr < 1):
        # NaN fails "< 1" to False, so it must be caught here, not left to the b check.
        raise ValueError("t - n must be finite and >= 1.")
    if math.isinf(b) and b > 0:
        # +inf gives infinite critical values, so the test would never reject. It is refused by name.
        raise ValueError("b must be finite; an infinite b gives infinite critical values.")
    arg = b + np.log(lag_arr)
    if not np.all(arg >= 0):
        # The square root of a negative value is NaN, and NaN critical values never reject.
        raise ValueError("b must satisfy b + log(t - n) >= 0 for every lag (b is too negative or NaN).")
    return np.sqrt(arg)


@dataclass(frozen=True)
class CSWResult:
    """Output of :func:`chu_stinchcombe_white`.

    Attributes
    ----------
    statistic : np.ndarray
        ``S_{n,t}`` for a fixed reference ``n``, or ``S_t = sup_n S_{n,t}``
        when ``reference`` is None. NaN where undefined.
    critical : np.ndarray
        ``c_alpha`` at each ``t`` (for the sup, at the maximising ``n``).
    reject : np.ndarray
        Boolean mask, ``statistic > critical`` (one-sided test for an upward
        break).
    reference : np.ndarray or None
        Maximising reference index ``n*_t`` for the sup form; None otherwise.
    """

    statistic: np.ndarray
    critical: np.ndarray
    reject: np.ndarray
    reference: np.ndarray | None


def chu_stinchcombe_white(
    series: Any,
    reference: int | None = None,
    b: float = _CSW_B_05,
) -> CSWResult:
    """Chu-Stinchcombe-White CUSUM test on levels (AFML 17.3.2).

    Parameters
    ----------
    series : array-like or pd.Series
        Levels ``y_t`` (log prices, per the book).
    reference : int, optional
        Fixed 0-based reference index ``n``. If None, the supremum over all
        ``n in [0, t-1]`` is used (the book's backward-shifting variant).
    b : float, default 4.6
        Critical-value constant (book: ``b_0.05 = 4.6``).

    Returns
    -------
    CSWResult
        Statistics, critical values, and the one-sided rejection mask.

    Notes
    -----
    ``S_{n,t} = (y_t - y_n) / (sigma_t sqrt(t - n))`` and
    ``sigma_t^2 = (t)^-1 sum_{i=1}^t (y_i - y_{i-1})^2`` in 0-based indices (the
    book's ``(t-1)^-1 sum_{i=2}^t`` in 1-based). Under H0 (no drift) ``S_{n,t}``
    is approximately N(0, 1). The sup form is O(T^2).
    """
    y, index = _as_float_vector(series)
    n_obs = y.shape[0]
    if n_obs < 2:
        raise ValueError("need at least two observations.")
    # The statistic is a ratio to sigma, so a large-magnitude series is scaled first.
    y = y / _large_scale(y)
    sigma = _prefix_sigma(y)
    # The full-sample sigma is zero only if every difference is zero (constant series,
    # reported as NaN below) or the squares underflowed; it is inf if they overflowed.
    if not np.all(y == y[0]) and (not math.isfinite(sigma[-1]) or sigma[-1] == 0.0):
        raise ValueError(
            "sigma is zero or not finite because the squared differences underflow or "
            "overflow; the series magnitude is likely outside the usable range, so rescale "
            "the series"
        )
    if reference is None:
        stat, ref = _csw_sup(y, sigma)
        ref_out = ref
        lag = np.arange(n_obs, dtype=np.float64) - ref.astype(np.float64)
        crit = np.full(n_obs, np.nan)
        ok = ref >= 0
        crit[ok] = csw_critical_value(lag[ok], b)
        stat = np.where(ok, stat, np.nan)
    else:
        _require_integer(reference, "reference must be an integer.")
        if not 0 <= reference < n_obs - 1:
            raise ValueError(f"reference must be in [0, {n_obs - 2}], got {reference}.")
        reference = int(reference)
        stat = _csw_fixed(y, sigma, reference)
        crit = np.full(n_obs, np.nan)
        t_idx = np.arange(reference + 1, n_obs)
        crit[reference + 1 :] = csw_critical_value(t_idx - reference, b)
        ref_out = None
    reject = np.zeros(n_obs, dtype=bool)
    valid = np.isfinite(stat) & np.isfinite(crit)
    reject[valid] = stat[valid] > crit[valid]
    if index is not None:
        stat_out: Any = pd.Series(stat, index=index)
        crit_out: Any = pd.Series(crit, index=index)
        reject_out: Any = pd.Series(reject, index=index)
        ref_final: Any = None if ref_out is None else pd.Series(ref_out, index=index)
    else:
        stat_out, crit_out, reject_out = stat, crit, reject
        ref_final = ref_out
    return CSWResult(
        statistic=stat_out,
        critical=crit_out,
        reject=reject_out,
        reference=ref_final,
    )


# ---------------------------------------------------------------------------
# ADF and SADF (AFML 17.4.2)
# ---------------------------------------------------------------------------


@jit
def _n_deterministic(const: int) -> int:
    if const == 0:
        return 0
    if const == 1:
        return 1
    if const == 2:
        return 2
    return 3


@jit
def _adf_tau(y: np.ndarray, start: int, end: int, lag: int, const: int) -> float:
    """tau = beta_hat / se(beta_hat) for the AFML 17.4.2 regression on y[start..end].

    Regression rows: targets ``dy_t`` for ``t in [start+lag+1, end]`` with regressors
    ``y_{t-1}``, ``dy_{t-1}, ..., dy_{t-lag}`` and the deterministic terms.
    """
    n_levels = end - start + 1
    n_rows = n_levels - 1 - lag
    n_det = _n_deterministic(const)
    p = 1 + lag + n_det
    if n_rows <= p:
        return np.nan
    x = np.empty((n_rows, p))
    yy = np.empty(n_rows)
    for r in range(n_rows):
        t = start + lag + 1 + r
        yy[r] = y[t] - y[t - 1]
        x[r, 0] = y[t - 1]
        for ll in range(1, lag + 1):
            x[r, ll] = y[t - ll] - y[t - ll - 1]
        k = 1 + lag
        if const >= 1:
            x[r, k] = 1.0
            k += 1
        if const >= 2:
            x[r, k] = float(t - lag - 1)
            k += 1
        if const >= 3:
            x[r, k] = float(t - lag - 1) * float(t - lag - 1)
            k += 1
    gram = np.zeros((p, p))
    xty = np.zeros(p)
    for r in range(n_rows):
        for i in range(p):
            xty[i] += x[r, i] * yy[r]
            for j in range(p):
                gram[i, j] += x[r, i] * x[r, j]
    inv = _invert_small(gram)
    if np.isnan(inv[0, 0]):
        return np.nan
    beta = np.zeros(p)
    for i in range(p):
        for j in range(p):
            beta[i] += inv[i, j] * xty[j]
    ssr = 0.0
    for r in range(n_rows):
        fitted = 0.0
        for i in range(p):
            fitted += x[r, i] * beta[i]
        e = yy[r] - fitted
        ssr += e * e
    sigma2 = ssr / (n_rows - p)
    var0 = sigma2 * inv[0, 0]
    if var0 <= 0.0:
        if beta[0] == 0.0:
            return 0.0
        return math.copysign(np.inf, beta[0])
    return beta[0] / math.sqrt(var0)


@jit
def _sadf_kernel(
    y: np.ndarray, min_length: int, lag: int, const: int
) -> tuple[np.ndarray, np.ndarray]:
    n_obs = y.shape[0]
    out = np.full(n_obs, np.nan)
    arg = np.full(n_obs, -1, dtype=np.int64)
    for e in range(min_length - 1, n_obs):
        best = -np.inf
        best_s = -1
        for s in range(0, e - min_length + 2):
            tau = _adf_tau(y, s, e, lag, const)
            if not np.isnan(tau) and tau > best:
                best = tau
                best_s = s
        if best_s >= 0:
            out[e] = best
            arg[e] = best_s
    return out, arg


@jit
def _quantile_sorted(vals: np.ndarray, q: float) -> float:
    """Quantile of sorted ``vals`` by linear interpolation, as ``np.quantile`` (method "linear").

    The blend takes the form numpy uses, measured from the nearer end. At ``q = 1`` the
    result is the largest value exactly.
    """
    n = vals.shape[0]
    h = (n - 1) * q
    lo = int(math.floor(h))
    frac = h - lo
    if frac == 0.0 or lo >= n - 1:
        return vals[lo]
    a = vals[lo]
    b = vals[lo + 1]
    if a == b:
        return a
    diff = b - a
    if frac >= 0.5:
        return b - diff * (1.0 - frac)
    return a + diff * frac


@jit
def _adf_set_kernel(
    y: np.ndarray, min_length: int, lag: int, const: int, q: float, q_lo: float, q_hi: float
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Quantiles and tail mean of the ADF set ``A_t`` at every end point.

    ``A_t`` holds the same statistics whose maximum is ``SADF_t`` (windows of at least
    ``min_length`` levels). Returns ``QADF`` at ``q``, at ``q_lo`` and at ``q_hi``, and
    ``CADF``, the mean of the elements of ``A_t`` that are at least ``QADF_{t,q}``.
    """
    n_obs = y.shape[0]
    out_q = np.full(n_obs, np.nan)
    out_lo = np.full(n_obs, np.nan)
    out_hi = np.full(n_obs, np.nan)
    out_c = np.full(n_obs, np.nan)
    for e in range(min_length - 1, n_obs):
        n_win = e - min_length + 2
        taus = np.empty(n_win)
        k = 0
        for s in range(0, n_win):
            tau = _adf_tau(y, s, e, lag, const)
            if not np.isnan(tau):
                taus[k] = tau
                k += 1
        if k == 0:
            continue
        vals = np.sort(taus[:k])
        qv = _quantile_sorted(vals, q)
        out_q[e] = qv
        out_lo[e] = _quantile_sorted(vals, q_lo)
        out_hi[e] = _quantile_sorted(vals, q_hi)
        total = 0.0
        cnt = 0
        for i in range(k):
            if vals[i] >= qv:
                total += vals[i]
                cnt += 1
        if cnt > 0:
            out_c[e] = total / cnt
    return out_q, out_lo, out_hi, out_c


def _constant_code(constant: str) -> int:
    if constant not in _CONSTANT_CODES:
        raise ValueError(f"constant must be one of {sorted(_CONSTANT_CODES)}, got {constant!r}.")
    return _CONSTANT_CODES[constant]


def _window_inputs(
    series: Any, min_length: Any, lag: Any, constant: str, lag_name: str
) -> tuple[np.ndarray, pd.Index | None, int, int, int]:
    """Validate the arguments shared by sadf, qadf and cadf.

    Returns the levels, the index, the integer lag, the constant code and the integer
    ``min_length``. ``lag_name`` is the argument name used in messages (sadf says ``lags``).
    """
    y, index = _as_float_vector(series)
    if y.size > 0 and np.all(y == y[0]):
        # A constant series is refused here, as adf_stat refuses it.
        raise ValueError("series has no variation (it is constant), so the ADF regression is singular.")
    _require_integer(lag, f"{lag_name} must be an integer.")
    _require_integer(min_length, "min_length must be an integer.")
    if lag < 0:
        raise ValueError(f"{lag_name} must be >= 0.")
    code = _constant_code(constant)
    n_det = _n_deterministic(code)
    min_required = 2 * lag + n_det + 3  # rows = len - 1 - lag must exceed p = 1 + lag + n_det
    if min_length < min_required:
        raise ValueError(
            f"min_length={min_length} too small for {lag_name}={lag}, constant={constant!r}; "
            f"need >= {min_required}."
        )
    if min_length > y.shape[0]:
        raise ValueError("min_length exceeds the series length.")
    return y, index, int(lag), code, int(min_length)


def adf_stat(series: Any, lag: int, constant: str = "c") -> float:
    """Augmented Dickey-Fuller tau-statistic on a single window (AFML 17.4.2).

    Fits ``dy_t = beta*y_{t-1} + sum_{l=1}^{lag} gamma_l dy_{t-l} + deterministic + eps_t``
    and returns ``beta_hat / se(beta_hat)``.

    Parameters
    ----------
    series : array-like or pd.Series
        Levels (log prices, per the book).
    lag : int
        Number of lagged differences ``L >= 0``.
    constant : {'nc', 'c', 'ct', 'ctt'}, default 'c'
        Deterministic terms: none, intercept, intercept + trend, intercept +
        trend + trend^2. See the module docstring for the code-based reading.

    Returns
    -------
    float
        The tau-statistic. Under H0 (unit root) it is not normal. Explosive
        behaviour gives large positive values.
    """
    y, _ = _as_float_vector(series)
    if y.size > 0 and np.all(y == y[0]):
        # A constant series makes the regressors singular. Without this, the failure reported
        # "too few observations", and constant='nc' silently returned 0.0.
        raise ValueError("series has no variation (it is constant), so the ADF regression is singular.")
    _require_integer(lag, "lag must be an integer.")
    if lag < 0:
        raise ValueError("lag must be >= 0.")
    code = _constant_code(constant)
    # The tau-statistic is scale-invariant, so rescale instead of failing at 1e160.
    tau = _adf_tau(_center_for_adf(_rescale_for_adf(y), code), 0, y.shape[0] - 1, int(lag), code)
    if np.isnan(tau):
        raise ValueError("too few observations for the requested lag and constant.")
    return float(tau)


def sadf(
    series: Any,
    min_length: int,
    lags: int = 1,
    constant: str = "c",
) -> Any:
    """Supremum Augmented Dickey-Fuller series (AFML 17.4.2).

    For each end point ``t >= min_length - 1`` (0-based), computes
    ``SADF_t = sup_{t0} ADF_{t0, t}`` over start points ``t0 in [0, t - min_length + 1]``,
    each ADF fitted on the window ``y[t0..t]``. The right edge is fixed at ``t``.

    Parameters
    ----------
    series : array-like or pd.Series
        Levels (log prices, per the book).
    min_length : int
        Smallest window (number of levels) used. Corresponds to the book's ``tau``
        plus one.
    lags : int, default 1
        ADF lag order ``L``.
    constant : {'nc', 'c', 'ct', 'ctt'}, default 'c'
        Deterministic terms. See :func:`adf_stat`.

    Returns
    -------
    pd.Series or np.ndarray
        ``SADF_t`` for every end point, NaN before ``min_length - 1``. A
        ``pd.Series`` is returned when the input is a Series.

    Notes
    -----
    Cost is O(T^2) ADF regressions, so O(T^3) work overall in the rows. The
    outer end-point loop and the inner start-point loop are both compiled with
    numba. ``lags`` and ``constant`` follow the arguments of Snippet 17.1.
    """
    y, index, lags, code, min_length = _window_inputs(series, min_length, lags, constant, "lags")
    # Rescale as adf_stat does, so windows of values near 1e200 do not silently give NaN.
    out, _ = _sadf_kernel(_center_for_adf(_rescale_for_adf(y), code), min_length, lags, code)
    return _wrap(out, index)


def qadf(
    series: Any,
    min_length: int,
    lag: int,
    q: float = 0.95,
    constant: str = "c",
    v: float | None = None,
) -> Any:
    """Quantile ADF series (AFML 17.4.2).

    For each end point ``t`` (0-based), let ``A_t`` be the ADF statistics over the same
    start points as :func:`sadf`, so that ``max(A_t) = SADF_t``. Then ``QADF_{t,q}`` is the
    ``q``-quantile of ``A_t``.

    Parameters
    ----------
    series : array-like or pd.Series
        Levels ``y_t`` in time order (log prices, per the book).
    min_length : int
        Smallest window (number of levels), as in :func:`sadf`.
    lag : int
        ADF lag order ``L``.
    q : float, default 0.95
        Quantile level in ``[0, 1]``.
    constant : {'nc', 'c', 'ct', 'ctt'}, default 'c'
        Deterministic terms. See :func:`adf_stat`.
    v : float, optional
        Half-width of the dispersion band, with ``v > 0``, ``q - v >= 0`` and
        ``q + v <= 1``. When given, the dispersion ``QADF_{t,q+v} - QADF_{t,q-v}`` is
        returned as well.

    Returns
    -------
    pd.Series or np.ndarray
        ``QADF_{t,q}`` for every end point, NaN before ``min_length - 1``. Without ``v`` this
        is the whole result. With ``v`` the result is the tuple ``(qadf, dispersion)``, each
        in the same style. A ``pd.Series`` is returned for a Series input.

    Notes
    -----
    Quantiles are linear interpolations between order statistics, the rule of
    ``np.quantile`` with its default ``method="linear"``. At ``q = 1`` the result equals
    :func:`sadf` exactly. The cost is that of :func:`sadf` plus a sort of ``A_t`` at each
    end point.
    """
    y, index, lag, code, min_length = _window_inputs(series, min_length, lag, constant, "lag")
    q = _require_real_in(q, "q", 0.0, 1.0)
    if v is None:
        q_lo = q_hi = q
    else:
        v = _require_real_in(v, "v", 0.0, 1.0, low_open=True)
        if q - v < 0.0 or q + v > 1.0:
            raise ValueError(f"v={v!r} puts q - v or q + v outside [0, 1] for q={q!r}.")
        q_lo, q_hi = q - v, q + v
    out_q, out_lo, out_hi, _ = _adf_set_kernel(
        _center_for_adf(_rescale_for_adf(y), code), min_length, lag, code, q, q_lo, q_hi
    )
    if v is None:
        return _wrap(out_q, index)
    return _wrap(out_q, index), _wrap(out_hi - out_lo, index)


def cadf(
    series: Any,
    min_length: int,
    lag: int,
    q: float = 0.95,
    constant: str = "c",
) -> Any:
    """Conditional ADF series (AFML 17.4.2).

    For each end point ``t`` (0-based), ``CADF_{t,q}`` is the mean of the elements of ``A_t``
    (the ADF statistics of :func:`sadf`) that are at least ``QADF_{t,q}``.

    Parameters
    ----------
    series : array-like or pd.Series
        Levels ``y_t`` in time order (log prices, per the book).
    min_length : int
        Smallest window (number of levels), as in :func:`sadf`.
    lag : int
        ADF lag order ``L``.
    q : float, default 0.95
        Quantile level in ``[0, 1]`` that sets the tail.
    constant : {'nc', 'c', 'ct', 'ctt'}, default 'c'
        Deterministic terms. See :func:`adf_stat`.

    Returns
    -------
    pd.Series or np.ndarray
        ``CADF_{t,q}`` for every end point, NaN before ``min_length - 1``. A ``pd.Series``
        is returned for a Series input.

    Notes
    -----
    The tail is ``{a in A_t : a >= QADF_{t,q}}``. The conditional dispersion, the standard
    deviation of that tail, is not returned. At ``q = 1`` the tail is the
    maximum alone, so the result equals :func:`sadf`, except when several elements tie
    exactly at the maximum. Then the mean may differ from the maximum by rounding.
    """
    y, index, lag, code, min_length = _window_inputs(series, min_length, lag, constant, "lag")
    q = _require_real_in(q, "q", 0.0, 1.0)
    _, _, _, out_c = _adf_set_kernel(
        _center_for_adf(_rescale_for_adf(y), code), min_length, lag, code, q, q, q
    )
    return _wrap(out_c, index)


# ---------------------------------------------------------------------------
# Sub- and super-martingale tests (AFML 17.4.3)
# ---------------------------------------------------------------------------

# Kind codes: 0 = poly1 (levels, quadratic in t), 1 = poly2 (logs, quadratic in t),
# 2 = exp (logs, linear in t), 3 = power (logs, linear in log t).
_SMT_KINDS = {"poly1": 0, "poly2": 1, "exp": 2, "power": 3}


@jit
def _smt_tstat(yv: np.ndarray, base: np.ndarray, s: int, e: int, kind: int) -> float:
    """t-statistic of the tested coefficient on the window ``s..e`` (0-based, inclusive).

    The tested coefficient is the quadratic term for kinds 0 and 1, and the slope for kinds 2
    and 3. The regressors are built from ``base`` centred on the window mean. Centring leaves
    the tested coefficient and its standard error unchanged, because only the intercept and
    the other coefficient move. It keeps the Gram matrix well conditioned far from the origin.
    """
    n_rows = e - s + 1
    quad = kind <= 1
    p = 3 if quad else 2
    if n_rows <= p:
        return np.nan
    mean = 0.0
    for r in range(s, e + 1):
        mean += base[r]
    mean /= n_rows
    x = np.empty((n_rows, p))
    yy = np.empty(n_rows)
    for r in range(n_rows):
        z = base[s + r] - mean
        x[r, 0] = 1.0
        x[r, 1] = z
        if quad:
            x[r, 2] = z * z
        yy[r] = yv[s + r]
    gram = np.zeros((p, p))
    xty = np.zeros(p)
    for r in range(n_rows):
        for i in range(p):
            xty[i] += x[r, i] * yy[r]
            for j in range(p):
                gram[i, j] += x[r, i] * x[r, j]
    inv = _invert_small(gram)
    if np.isnan(inv[0, 0]):
        return np.nan
    beta = np.zeros(p)
    for i in range(p):
        for j in range(p):
            beta[i] += inv[i, j] * xty[j]
    ssr = 0.0
    for r in range(n_rows):
        fitted = 0.0
        for i in range(p):
            fitted += x[r, i] * beta[i]
        resid = yy[r] - fitted
        ssr += resid * resid
    k = 2 if quad else 1
    var_b = ssr / (n_rows - p) * inv[k, k]
    if var_b <= 0.0:
        if beta[k] == 0.0:
            return 0.0
        return math.copysign(np.inf, beta[k])
    return beta[k] / math.sqrt(var_b)


@jit
def _smt_kernel(
    yv: np.ndarray, base: np.ndarray, min_length: int, kind: int, phi: float
) -> np.ndarray:
    n_obs = yv.shape[0]
    out = np.full(n_obs, np.nan)
    for e in range(min_length - 1, n_obs):
        best = -np.inf
        found = False
        for s in range(0, e - min_length + 2):
            tt = _smt_tstat(yv, base, s, e, kind)
            if np.isnan(tt):
                continue
            val = abs(tt) / (float(e - s) ** phi)
            if val > best:
                best = val
                found = True
        if found:
            out[e] = best
    return out


def smt(
    series: Any,
    min_length: int,
    kind: str = "poly1",
    phi: float = 0.0,
) -> Any:
    """Sub- and super-martingale test statistic ``SMT_t`` (AFML 17.4.3).

    For each end point ``t`` (0-based), and each start point ``t0`` with the window
    ``[t0, t]`` of at least ``min_length`` levels, an OLS fit on the window gives the
    coefficient ``b`` and its standard error. Then

        SMT_t = sup_{t0} |b_hat| / (se(b_hat) * (t - t0) ** phi).

    Parameters
    ----------
    series : array-like or pd.Series
        Levels ``y_t`` in time order. The log kinds need strictly positive values.
    min_length : int
        Smallest window (number of levels). It must leave at least one residual degree of
        freedom: at least 4 for ``'poly1'`` and ``'poly2'``, and at least 3 for ``'exp'`` and
        ``'power'``.
    kind : {'poly1', 'poly2', 'exp', 'power'}, default 'poly1'
        Regression on the window ``[t0, t]``:

        - ``'poly1'``: ``y_t = a + g t + b t^2 + e``, the statistic is on ``b``.
        - ``'poly2'``: ``log y_t = a + g t + b t^2 + e``, the statistic is on ``b``.
        - ``'exp'``: ``log y_t = log(a) + b t + e``, the statistic is on ``b``.
        - ``'power'``: ``log y_t = log(a) + b log(t) + e``, the statistic is on ``b``.
    phi : float, default 0.0
        Exponent in ``[0, 1]`` that down-weights long windows. ``phi = 0`` gives the plain
        supremum of the absolute t-statistic.

    Returns
    -------
    pd.Series or np.ndarray
        ``SMT_t`` for every end point, NaN before ``min_length - 1``. A ``pd.Series`` is
        returned for a Series input.

    Notes
    -----
    Time is the 1-based position in the full series, so ``t`` ranges over ``1..T``. The
    1-based convention keeps ``log(t)`` defined for ``'power'``. The window set is the one of
    :func:`sadf`, with start points ``0..t - min_length + 1``, so that ``min_length`` means the
    same thing in every function of this module. The standard error uses
    ``sigma^2 = SSR / (n - p)``. The statistic takes the absolute value, so both upward and
    downward curvature are flagged. The cost is O(T^2) fits.
    """
    y, index = _as_float_vector(series)
    if isinstance(kind, str) and kind in _SMT_KINDS:
        code = _SMT_KINDS[kind]
    else:
        raise ValueError(f"kind must be one of {sorted(_SMT_KINDS)}, got {kind!r}.")
    _require_integer(min_length, "min_length must be an integer.")
    phi = _require_real_in(phi, "phi", 0.0, 1.0)
    p = 3 if code <= 1 else 2
    if min_length < p + 1:
        raise ValueError(f"min_length={min_length} too small for kind={kind!r}; need >= {p + 1}.")
    if min_length > y.shape[0]:
        raise ValueError("min_length exceeds the series length.")
    if code == 0:
        # The statistic is a ratio, so the scale cancels. Rescale as the ADF code does, so the
        # squares in the Gram matrix stay finite.
        yv = _rescale_for_adf(y)
    else:
        n_bad = int(np.sum(y <= 0.0))
        if n_bad:
            raise ValueError(
                f"kind={kind!r} takes log(series), so every value must be strictly positive; "
                f"found {n_bad} value(s) <= 0."
            )
        yv = np.log(y)
    positions = np.arange(1, y.shape[0] + 1, dtype=np.float64)
    base = np.log(positions) if code == 3 else positions
    out = _smt_kernel(yv, base, int(min_length), code, phi)
    return _wrap(out, index)


# ---------------------------------------------------------------------------
# Chow-type Dickey-Fuller tests (AFML ch. 17)
# ---------------------------------------------------------------------------


def _break_index(tau: float, n_obs: int) -> int:
    """First post-break position ``ceil(tau * T)``: ``D_t = 1`` for ``t >= ceil(tau * T)``.

    The product is rounded to 9 decimals first. Without that, ``0.07 * 100`` exceeds 7 in
    floating point, and the break would move one step later.
    """
    return math.ceil(round(tau * n_obs, 9))


def _chow_inputs(series: Any) -> np.ndarray:
    y, _ = _as_float_vector(series)
    if y.size < 3:
        raise ValueError("need at least three observations for the Chow-type DF test.")
    if np.all(y == y[0]):
        raise ValueError("series has no variation (it is constant), so the DFC regression is degenerate.")
    # DFC is a ratio of residual sums to sigma, so it is scale-invariant. The rescaling keeps
    # the squares finite, as in adf_stat.
    return _rescale_for_adf(y)


@jit
def _dfc_at(y: np.ndarray, brk: int) -> float:
    """DFC at break index ``brk``: OLS of ``dy_t`` on ``y_{t-1} * 1{t >= brk}``, no intercept.

    Rows run over ``t = 1..T-1``. Pre-break rows have a zero regressor, so they add to the
    residual sum of squares only. The standard error uses ``sigma^2 = SSR / (T - 2)``, the
    same guard as ``_adf_tau``. Returns NaN when every post-break ``y_{t-1}`` is zero.
    """
    n_obs = y.shape[0]
    if n_obs - 2 < 1:
        return np.nan
    sxx = 0.0
    sxy = 0.0
    for t in range(max(brk, 1), n_obs):
        x = y[t - 1]
        sxx += x * x
        sxy += x * (y[t] - y[t - 1])
    if sxx == 0.0:
        return np.nan
    delta = sxy / sxx
    ssr = 0.0
    for t in range(1, n_obs):
        fitted = delta * y[t - 1] if t >= brk else 0.0
        e = (y[t] - y[t - 1]) - fitted
        ssr += e * e
    var0 = ssr / (n_obs - 2) / sxx
    if var0 <= 0.0:
        if delta == 0.0:
            return 0.0
        return math.copysign(np.inf, delta)
    return delta / math.sqrt(var0)


@jit
def _dfc_path(y: np.ndarray, b_lo: int, b_hi: int) -> np.ndarray:
    out = np.empty(b_hi - b_lo + 1)
    for k in range(out.shape[0]):
        out[k] = _dfc_at(y, b_lo + k)
    return out


def chow_dfc(series: Any, tau: float) -> float:
    """Chow-type Dickey-Fuller statistic ``DFC(tau)`` for a break at fraction ``tau``.

    Fits by OLS, with no intercept and no lags, ``dy_t = delta * y_{t-1} * D_t + eps_t``, where
    ``D_t = 0`` for ``t < ceil(tau * T)`` and ``D_t = 1`` from there on. Returns
    ``delta_hat / se(delta_hat)``.

    Parameters
    ----------
    series : array-like or pd.Series
        Levels ``y_t`` in time order (log prices, per the book). Refused if it has
        fewer than three values, a non-finite value, or no variation.
    tau : float
        Break fraction in the open interval (0, 1).

    Returns
    -------
    float
        ``DFC(tau)``. Large positive values indicate explosive behaviour after the
        break, the direction :func:`sdfc` takes its supremum over.

    Notes
    -----
    Convention: the break index is ``b = ceil(tau * T)`` (computed after rounding
    ``tau * T`` to 9 decimals), and the 0-based observation ``t`` is post-break when
    ``t >= b``. A ``b`` of 0 is the same model as 1, because there is no regressor at
    ``t = 0``. The regression runs on all ``T - 1`` rows, so the pre-break rows enter the
    residual variance.

    The statistic is invariant to scaling ``series``. Refused, by name, when no
    observation lies after the break, or when every post-break ``y_{t-1}`` is zero.
    """
    y = _chow_inputs(series)
    tau = _require_real_in(tau, "tau", 0.0, 1.0, low_open=True, high_open=True)
    n_obs = y.shape[0]
    brk = max(_break_index(tau, n_obs), 1)
    if brk > n_obs - 1:
        raise ValueError(f"tau={tau!r} places the break after the last observation (T={n_obs}).")
    value = _dfc_at(y, brk)
    if math.isnan(value):
        raise ValueError("DFC is undefined: every y_{t-1} after the break is zero.")
    return float(value)


@dataclass(frozen=True)
class SDFCResult:
    """Output of :func:`sdfc`.

    Attributes
    ----------
    statistic : float
        ``SDFC``, the maximum of ``DFC`` over the admissible break indices.
    break_index : int
        Argmax break index ``b*``, the 0-based first post-break position. The break
        fraction is ``b* / T``.
    breaks : np.ndarray
        The admissible break indices ``b_lo, ..., b_hi`` (inclusive).
    dfc : np.ndarray
        ``DFC`` at each entry of ``breaks``. NaN where every post-break ``y_{t-1}`` is zero.
    """

    statistic: float
    break_index: int
    breaks: np.ndarray
    dfc: np.ndarray


def sdfc(series: Any, tau0: float = 0.15) -> SDFCResult:
    """Supremum Chow-type Dickey-Fuller statistic (AFML ch. 17).

    ``SDFC = sup_{tau in [tau0, 1 - tau0]} DFC(tau)``. The supremum runs over the
    distinct break indices ``b = ceil(tau * T)`` in that range, so it is a finite maximum.

    Parameters
    ----------
    series : array-like or pd.Series
        Levels ``y_t`` in time order (log prices, per the book).
    tau0 : float, default 0.15
        Trimming fraction in the open interval (0, 0.5). Breaks are searched in
        ``[tau0, 1 - tau0]``.

    Returns
    -------
    SDFCResult
        The supremum, its argmax break index, and the full ``DFC`` path over the
        admissible breaks. The argmax is an index into the series, not a label. For a
        ``pd.Series`` input, use ``series.index[result.break_index]``.

    Notes
    -----
    The break index rounding is the one documented in :func:`chow_dfc`. The path is
    computed with a numba kernel, and the cost is O(T^2) in the rows. The lower and upper
    indices are clamped to ``[1, T - 1]``, so every break leaves an observation after it.
    """
    y = _chow_inputs(series)
    tau0 = _require_real_in(tau0, "tau0", 0.0, 0.5, low_open=True, high_open=True)
    n_obs = y.shape[0]
    b_lo = max(_break_index(tau0, n_obs), 1)
    b_hi = min(_break_index(1.0 - tau0, n_obs), n_obs - 1)
    if b_lo > b_hi:
        raise ValueError(f"tau0={tau0!r} leaves no admissible break for T={n_obs}.")
    dfc = _dfc_path(y, b_lo, b_hi)
    if np.all(np.isnan(dfc)):
        raise ValueError("DFC is undefined at every admissible break: every y_{t-1} after each break is zero.")
    k = int(np.nanargmax(dfc))
    return SDFCResult(
        statistic=float(dfc[k]),
        break_index=b_lo + k,
        breaks=np.arange(b_lo, b_hi + 1),
        dfc=dfc,
    )
