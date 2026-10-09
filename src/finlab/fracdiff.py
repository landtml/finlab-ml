"""Fractionally differentiated features (AFML ch. 5).

Implements, from Lopez de Prado's *Advances in Financial Machine Learning*:

* Section 5.4 -- the fractional difference operator ``(1 - B)^d`` as a binomial
  series, with weights ``omega_k`` (5.4.1) and the iterative recursion
  ``omega_k = -omega_{k-1} (d - k + 1) / k`` (5.4.2).
* Section 5.5.1, Snippets 5.1-5.2 -- the expanding-window method
  (:func:`get_weights`, :func:`frac_diff`), with the weight-loss tolerance
  ``thres`` (the book's ``tau``).
* Section 5.5.2, Snippet 5.3 -- the fixed-width window method (FFD):
  :func:`get_weights_ffd`, :func:`frac_diff_ffd`.
* Section 5.6, Snippet 5.4 -- the smallest ``d`` that passes a stationarity
  test: :func:`find_min_d`.

Conventions
-----------
Weights are returned in **lag order**: ``w[k]`` is ``omega_k``, the weight on
``X_{t-k}``. The book's snippets store them reversed (oldest first) because
they use ``np.dot`` with a slice. Results here align with the input: a
Series or DataFrame keeps its index, and warm-up rows are NaN. The book's
snippets drop those rows.

Threshold conventions
---------------------
``thres`` is a **fixed** tolerance. For FFD it cuts the window at the first
weight with ``|omega_k| < thres``. For the expanding window it sets the
skipped prefix so that the weight lost, ``lambda_t``, is at most ``thres``.
Neither threshold depends on the data.

Scope
-----
All weight and filter kernels are numba loops over observations: O(n * width)
for FFD and O(n^2) for the expanding window. Memory is O(n) plus the weights.

Not covered
-----------
Cointegration tests (Engle-Granger, Johansen) and the Jarque-Bera test from the
chapter exercises. The ADF test itself is not implemented: :func:`find_min_d`
takes an injected ``adf_test`` callable (see its docstring). The book's
statsmodels call (Snippet 5.4) is not a dependency here.
"""

from __future__ import annotations

from typing import Callable, NamedTuple

import numpy as np
import pandas as pd
from numba import prange
from numpy.typing import ArrayLike, NDArray

from finlab._jit import jit, pjit

__all__ = [
    "get_weights",
    "get_weights_ffd",
    "frac_diff",
    "frac_diff_ffd",
    "find_min_d",
    "MinDResult",
]

# Safety cap on the FFD window length. A finite threshold always terminates
# (|omega_k| decays like k^{-d-1}), so this is a guard against absurdly small thres.
_MAX_WIDTH = 10_000_000


# --------------------------------------------------------------------------- #
# Numba kernels
# --------------------------------------------------------------------------- #


@jit
def _ffd_width(d: float, thres: float, cap: int) -> int:
    """Number of FFD weights: the first ``k`` with ``|omega_k| < thres`` (exclusive).

    The recursion mirrors :func:`_fill_weights` exactly, so the two agree bit for bit.
    """
    w = 1.0
    k = 1
    while k < cap:
        w_new = -w * (d - k + 1) / k
        if abs(w_new) < thres:
            break
        w = w_new
        k += 1
    return k


@jit
def _fill_weights(d: float, n: int) -> NDArray[np.float64]:
    """Lag-ordered weights ``omega_0 .. omega_{n-1}`` by the recursion of AFML 5.4.2."""
    w = np.empty(n, dtype=np.float64)
    w[0] = 1.0
    for k in range(1, n):
        w[k] = -w[k - 1] * (d - k + 1) / k
    return w


_BLOCK = 2048  # observations per block; keeps out/x slices in cache


@pjit
def _fixed_width_apply(x: NDArray[np.float64], w: NDArray[np.float64]) -> NDArray[np.float64]:
    """FFD: ``out[t] = sum_k w[k] x[t-k]`` for ``t >= L-1``; NaN before that.

    Parallel over blocks of ``t`` (``prange``). Inside a block the loop order is
    ``k`` outer, ``t`` inner, so each ``out[t]`` accumulates ``k = 0, 1, ...`` in a
    fixed order. The result does not depend on the thread count, and it matches a
    per-observation dot product to rounding. The inner loop is elementwise, with no
    reassociation, so ``fastmath`` is not needed.
    """
    n = x.shape[0]
    L = w.shape[0]
    out = np.full(n, np.nan)
    n_blocks = max(0, (n - (L - 1) + _BLOCK - 1) // _BLOCK)
    for b in prange(n_blocks):
        t0 = (L - 1) + b * _BLOCK
        t1 = min(t0 + _BLOCK, n)
        for t in range(t0, t1):
            out[t] = 0.0
        for k in range(L):
            wk = w[k]
            for t in range(t0, t1):
                out[t] += wk * x[t - k]
    return out


@pjit
def _expanding_apply(
    x: NDArray[np.float64], w: NDArray[np.float64], first: int
) -> NDArray[np.float64]:
    """Expanding window: ``out[t] = sum_{k<=t} w[k] x[t-k]`` for ``t >= first``.

    Same blocked, parallel layout as :func:`_fixed_width_apply`. Lag ``k``
    contributes only to ``t >= k``.
    """
    n = x.shape[0]
    out = np.full(n, np.nan)
    n_blocks = max(0, (n - first + _BLOCK - 1) // _BLOCK)
    for b in prange(n_blocks):
        t0 = first + b * _BLOCK
        t1 = min(t0 + _BLOCK, n)
        for t in range(t0, t1):
            out[t] = 0.0
        for k in range(t1):
            wk = w[k]
            for t in range(max(t0, k), t1):
                out[t] += wk * x[t - k]
    return out


# --------------------------------------------------------------------------- #
# Input handling
# --------------------------------------------------------------------------- #


def _check_d(d: float) -> float:
    if not np.isfinite(d):
        raise ValueError(f"d must be finite, got {d!r}.")
    return float(d)


def _check_thres(thres: float, *, strict: bool) -> float:
    if not np.isfinite(thres) or thres < 0 or (strict and thres <= 0):
        raise ValueError(f"thres must be {'> 0' if strict else '>= 0'}, got {thres!r}.")
    return float(thres)


def _apply_per_column(
    series: pd.Series | pd.DataFrame | ArrayLike,
    fn: Callable[[NDArray[np.float64]], NDArray[np.float64]],
) -> pd.Series | pd.DataFrame | NDArray[np.float64]:
    """Run ``fn`` on a 1-D float array or on each column of a 2-D one, keeping the container.

    NaNs are forward-filled before ``fn`` (the book's ``fillna(method='ffill')``), and
    the output is set back to NaN wherever the original value was NaN.
    """
    if isinstance(series, pd.DataFrame):
        cols = {}
        for name in series.columns:
            cols[name] = _apply_one(series[name], fn)
        return pd.DataFrame(cols, index=series.index)
    if isinstance(series, pd.Series):
        return pd.Series(_apply_one(series, fn), index=series.index, name=series.name)
    arr = np.asarray(series, dtype=np.float64)
    if arr.ndim == 1:
        return _apply_one(arr, fn)
    if arr.ndim == 2:
        return np.column_stack([_apply_one(arr[:, j], fn) for j in range(arr.shape[1])])
    raise ValueError(f"series must be 1-D or 2-D, got {arr.ndim}-D.")


def _apply_one(values: ArrayLike | pd.Series, fn: Callable[[NDArray[np.float64]], NDArray[np.float64]]) -> NDArray[np.float64]:
    raw = np.asarray(values, dtype=np.float64).reshape(-1)
    filled = pd.Series(raw).ffill().to_numpy(dtype=np.float64)
    out = fn(np.ascontiguousarray(filled))
    out[np.isnan(raw)] = np.nan
    return out


# --------------------------------------------------------------------------- #
# Weights
# --------------------------------------------------------------------------- #


def get_weights(d: float, size: int) -> NDArray[np.float64]:
    """Expanding-window weights ``omega_0 .. omega_{size-1}`` (AFML 5.4, Snippet 5.1).

    Parameters
    ----------
    d : float
        Differencing order. Any real value is allowed, not only ``[0, 1]``.
    size : int
        Number of weights (``>= 1``), normally the series length.

    Returns
    -------
    np.ndarray
        Lag-ordered weights of length ``size``. ``w[0] = 1``.

    Notes
    -----
    Recursion ``omega_k = -omega_{k-1} (d - k + 1) / k`` (AFML 5.4.2). For
    ``d = 0`` the output is ``[1, 0, 0, ...]``, and for ``d = 1`` it is
    ``[1, -1, 0, ...]``.

    Scope
    -----
    Numba loop of length ``size``.
    """
    d = _check_d(d)
    if int(size) != size or size < 1:
        raise ValueError(f"size must be a positive integer, got {size!r}.")
    return _fill_weights(d, int(size))


def get_weights_ffd(d: float, thres: float = 1e-5) -> NDArray[np.float64]:
    """Fixed-width-window weights, lag-ordered (AFML 5.5.2, Snippet 5.3).

    Parameters
    ----------
    d : float
        Differencing order (any real value).
    thres : float, default 1e-5
        Cut-off. Weights are kept while ``|omega_k| >= thres``. **Fixed**. Must be > 0.

    Returns
    -------
    np.ndarray
        Lag-ordered weights ``omega_0 .. omega_{L}`` with ``L = width``. The window
        width is ``len(w) - 1``. ``w[0] = 1``.

    Notes
    -----
    Implements the first ``l*`` with ``|omega_{l*}| < thres`` as the exclusive end
    of the window, as in Snippet 5.3 (``if abs(w_) < thres: break``). The book
    text gives the recursion (5.4.2) but no source for this helper, so the
    helper is derived from it.

    Scope
    -----
    Two numba passes, O(width).
    """
    d = _check_d(d)
    thres = _check_thres(thres, strict=True)
    n = _ffd_width(d, thres, _MAX_WIDTH)
    if n >= _MAX_WIDTH:
        raise RuntimeError("FFD window hit the safety cap; increase thres.")
    return _fill_weights(d, n)


# --------------------------------------------------------------------------- #
# Fractional differentiation
# --------------------------------------------------------------------------- #


def frac_diff_ffd(
    series: pd.Series | pd.DataFrame | ArrayLike,
    d: float,
    thres: float = 1e-5,
) -> pd.Series | pd.DataFrame | NDArray[np.float64]:
    """Fixed-width-window fractional differentiation (AFML 5.5.2, Snippet 5.3).

    Parameters
    ----------
    series : pd.Series, pd.DataFrame, or array-like
        Values to differentiate (1-D, or 2-D with one column per series). Typically
        log prices. NaNs are forward-filled before the window is applied.
    d : float
        Differencing order. Any real value.
    thres : float, default 1e-5
        Weight cut-off, fixed. See :func:`get_weights_ffd`.

    Returns
    -------
    pd.Series, pd.DataFrame or np.ndarray
        Same container and length as the input. The first ``width`` rows are NaN, as
        are rows whose original value was NaN.

    Notes
    -----
    Every output uses the same weight vector. Hence no expanding-window drift.
    Stationarity and memory are as in AFML 5.5.2 and 5.6. This is a claim from
    the book; see ``docs/proofs/fracdiff.md``.

    Scope
    -----
    Numba kernel :func:`_fixed_width_apply`, O(n * width).
    """
    w = get_weights_ffd(d, thres)
    return _apply_per_column(series, lambda x: _fixed_width_apply(x, w))


def frac_diff(
    series: pd.Series | pd.DataFrame | ArrayLike,
    d: float,
    thres: float = 0.01,
) -> pd.Series | pd.DataFrame | NDArray[np.float64]:
    """Expanding-window fractional differentiation (AFML 5.5.1, Snippet 5.2).

    Parameters
    ----------
    series : pd.Series, pd.DataFrame, or array-like
        Values to differentiate. NaNs are forward-filled.
    d : float
        Differencing order (any real value).
    thres : float, default 0.01
        Maximum tolerated weight loss ``lambda_t``. **Fixed**. ``thres >= 1``
        keeps every observation. Points with ``lambda_t > thres`` are NaN.

    Returns
    -------
    pd.Series, pd.DataFrame or np.ndarray
        Same container and length as the input.

    Notes
    -----
    Weight loss at ``t`` is ``lambda_t = sum_{j>t} |omega_j| / sum_j |omega_j|``, with
    ``j`` over the full-length weights (AFML 5.5.1). Output starts at the first
    ``t`` with ``lambda_t <= thres``. Since the window grows, the result drifts
    (AFML Figure 5.3); FFD avoids this.

    Scope
    -----
    O(n^2) work (numba). Use :func:`frac_diff_ffd` for long series.
    """
    thres = _check_thres(thres, strict=False)
    d = _check_d(d)

    def run(x: NDArray[np.float64]) -> NDArray[np.float64]:
        n = x.shape[0]
        if n == 0:
            return np.empty(0, dtype=np.float64)
        w = _fill_weights(d, n)
        a = np.abs(w)
        total = a.sum()
        lam = (total - np.cumsum(a)) / total
        ok = np.flatnonzero(lam <= thres)
        first = int(ok[0]) if ok.size else n
        return _expanding_apply(x, w, first)

    return _apply_per_column(series, run)


# --------------------------------------------------------------------------- #
# Minimum-d search (AFML 5.6)
# --------------------------------------------------------------------------- #


class MinDResult(NamedTuple):
    """Result of :func:`find_min_d`.

    Attributes
    ----------
    d : float or None
        Smallest grid value that passes the test, or None if no grid value passes.
    pvalue : float or None
        Test p-value at ``d``.
    corr : float or None
        Pearson correlation between the original series and the fractionally
        differentiated one, over the valid rows. Measures the memory kept.
    """

    d: float | None
    pvalue: float | None
    corr: float | None


def find_min_d(
    series: pd.Series | ArrayLike,
    adf_pvalue: float = 0.05,
    *,
    adf_test: Callable[[NDArray[np.float64]], float] | None = None,
    d_grid: ArrayLike | None = None,
    thres: float = 1e-2,
) -> MinDResult:
    """Smallest ``d`` whose FFD series passes a stationarity test (AFML 5.6, Snippet 5.4).

    Parameters
    ----------
    series : pd.Series or array-like
        Values to test, typically log prices.
    adf_pvalue : float, default 0.05
        Significance level. A d passes when ``pvalue <= adf_pvalue``.
    adf_test : callable
        **Required.** ``adf_test(x) -> float`` takes the finite FFD values as a 1-D
        float array and returns the p-value of a unit-root test (ADF, with the
        null of a unit root). finlab does not bundle one: the book's ADF
        critical values (MacKinnon) are not in the text, so an ADF implementation
        must be injected, e.g. ``statsmodels.tsa.stattools.adfuller(x, maxlag=1,
        regression="c", autolag=None)[1]``. That is not a dependency here.
    d_grid : array-like, optional
        Candidate values, scanned in increasing order. Default ``linspace(0, 1, 11)``,
        as in Snippet 5.4.
    thres : float, default 1e-2
        FFD weight cut-off, passed to :func:`frac_diff_ffd`. Snippet 5.4 uses 0.01.

    Returns
    -------
    MinDResult
        ``(d, pvalue, corr)`` at the first passing grid value, or
        ``(None, None, None)`` if none passes.

    Notes
    -----
    The search is a linear scan of the grid, not a bisection. Passing is not
    assumed monotone in ``d``, which is the book's own procedure. The test is
    applied to the FFD values with the NaN warm-up removed. A grid value of
    ``d = 0`` is the original series.

    Scope
    -----
    Cost is one FFD pass plus one ``adf_test`` call per grid value.
    """
    if adf_test is None:
        raise TypeError(
            "find_min_d needs an adf_test callable (x -> p-value); finlab does not bundle an ADF test."
        )
    if not 0.0 < adf_pvalue < 1.0:
        raise ValueError(f"adf_pvalue must be in (0, 1), got {adf_pvalue!r}.")
    grid = np.linspace(0.0, 1.0, 11) if d_grid is None else np.sort(np.asarray(d_grid, dtype=np.float64).reshape(-1))
    raw = np.asarray(series, dtype=np.float64).reshape(-1)
    for d in grid:
        fd = np.asarray(frac_diff_ffd(raw, float(d), thres=thres), dtype=np.float64)
        valid = np.isfinite(fd)
        x = fd[valid]
        if x.size < 3:
            continue
        p = float(adf_test(x))
        if p <= adf_pvalue:
            corr = _corr(raw[valid], x)
            return MinDResult(float(d), p, corr)
    return MinDResult(None, None, None)


def _corr(a: NDArray[np.float64], b: NDArray[np.float64]) -> float:
    """Pearson correlation over pairs where both are finite; NaN if undefined."""
    m = np.isfinite(a) & np.isfinite(b)
    a = a[m]
    b = b[m]
    if a.size < 2 or np.std(a) == 0.0 or np.std(b) == 0.0:
        return float("nan")
    return float(np.corrcoef(a, b)[0, 1])
