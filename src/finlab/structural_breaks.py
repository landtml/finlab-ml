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

Sign and input conventions
--------------------------
* Explosiveness tests (CSW and SADF) are one-sided and look for *upward*
  departures: a large positive statistic means the series grew faster than a
  random walk. The ADF slope ``beta`` is tested with ``H0: beta <= 0`` against
  ``H1: beta > 0``, so SADF uses ``tau = beta_hat / se(beta_hat)`` unchanged.
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
Not covered: the Chow-type ``DFC`` / ``SDFC`` tests (17.4.1); quantile and
conditional ADF (17.4.2.4-5); the Brown-Durbin-Evans critical boundaries (not
given in the book text, so no p-values are returned for the CUSUM path);
multi-bubble and sub/super-martingale variants; and the parallel SADF
implementation of ch. 20.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from ._jit import jit

__all__ = [
    "CusumResult",
    "CSWResult",
    "cusum_test",
    "chu_stinchcombe_white",
    "csw_critical_value",
    "adf_stat",
    "sadf",
]

_CONSTANT_CODES = {"nc": 0, "c": 1, "ct": 2, "ctt": 3}
_CSW_B_05 = 4.6  # b_0.05 from AFML 17.3.2 (Monte Carlo, Chu-Stinchcombe-White)


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------


def _as_float_vector(series: Any) -> tuple[np.ndarray, pd.Index | None]:
    index = series.index if isinstance(series, (pd.Series, pd.DataFrame)) else None
    values = np.asarray(series, dtype=np.float64)
    if values.ndim == 2 and values.shape[1] == 1:
        values = values[:, 0]
    if values.ndim != 1:
        raise ValueError(f"expected a 1-D series, got shape {values.shape}.")
    if not np.all(np.isfinite(values)):
        raise ValueError("series contains NaN or infinite values.")
    return np.ascontiguousarray(values), index


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

    w, ok = _recursive_residuals(y, x)
    if not ok:
        raise ValueError("initial design X_p'X_p is singular; use more leading rows.")
    resid = w[p:]
    sigma = float(math.sqrt(np.mean(resid * resid)))
    if sigma == 0.0:
        raise ValueError("recursive residuals are all zero; no variation in series.")

    stat = np.full(n_obs, np.nan)
    z = np.full(n_obs, np.nan)
    cum = np.cumsum(resid)
    counts = np.arange(1, resid.shape[0] + 1, dtype=np.float64)
    stat[p:] = cum / sigma
    z[p:] = cum / (sigma * np.sqrt(counts))
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
        out[t] = (y[t] - y[ref]) / (sigma[t] * math.sqrt(t - ref))
    return out


@jit
def _csw_sup(y: np.ndarray, sigma: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    n_obs = y.shape[0]
    best = np.full(n_obs, np.nan)
    arg = np.full(n_obs, -1, dtype=np.int64)
    for t in range(1, n_obs):
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
    if np.any(lag_arr < 1):
        raise ValueError("t - n must be >= 1.")
    return np.sqrt(b + np.log(lag_arr))


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
    sigma = _prefix_sigma(y)
    if reference is None:
        stat, ref = _csw_sup(y, sigma)
        ref_out = ref
        lag = np.arange(n_obs, dtype=np.float64) - ref.astype(np.float64)
        crit = np.full(n_obs, np.nan)
        ok = ref >= 0
        crit[ok] = csw_critical_value(lag[ok], b)
        stat = np.where(ok, stat, np.nan)
    else:
        if not 0 <= reference < n_obs - 1:
            raise ValueError(f"reference must be in [0, {n_obs - 2}], got {reference}.")
        stat = _csw_fixed(y, sigma, int(reference))
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


def _constant_code(constant: str) -> int:
    if constant not in _CONSTANT_CODES:
        raise ValueError(f"constant must be one of {sorted(_CONSTANT_CODES)}, got {constant!r}.")
    return _CONSTANT_CODES[constant]


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
    if lag < 0:
        raise ValueError("lag must be >= 0.")
    code = _constant_code(constant)
    tau = _adf_tau(y, 0, y.shape[0] - 1, int(lag), code)
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
    y, index = _as_float_vector(series)
    if lags < 0:
        raise ValueError("lags must be >= 0.")
    code = _constant_code(constant)
    n_det = _n_deterministic(code)
    min_required = 2 * lags + n_det + 3  # rows = len - 1 - lags must exceed p = 1 + lags + n_det
    if min_length < min_required:
        raise ValueError(
            f"min_length={min_length} too small for lags={lags}, constant={constant!r}; "
            f"need >= {min_required}."
        )
    if min_length > y.shape[0]:
        raise ValueError("min_length exceeds the series length.")
    out, _ = _sadf_kernel(y, int(min_length), int(lags), code)
    return _wrap(out, index)
