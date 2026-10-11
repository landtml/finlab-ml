"""Hierarchical Risk Parity (HRP) portfolio weights.

Implements AFML chapter 16, "Machine Learning Asset Allocation":

* :func:`correlation_distance` -- the metric ``d = sqrt(0.5 * (1 - rho))``,
  section 16.4.1.
* Tree clustering with :func:`scipy.cluster.hierarchy.linkage`, section 16.4.1
  and Snippet 16.1. Only the single-linkage default is used in the book.
* :func:`quasi_diagonalize` -- the leaf ordering of Snippet 16.2, section 16.4.2.
* :func:`hrp_weights` -- the recursive bisection of Snippet 16.3, section 16.4.3,
  with inverse-variance allocation inside each cluster.

Scope: a single covariance (or correlation) matrix of N assets. Returns
long-only weights summing to one. The covariance matrix is never inverted,
so it may be singular; it must have a strictly positive diagonal.

Not covered: the Euclidean "distance of distances" variant of section 16.4.1
(the book's Example 16.2 clustering), constrained bisection (section 16.4.3's
closing remark), the out-of-sample Monte Carlo study of section 16.6, and
the Appendix 16.A.3 data generator.
"""

from __future__ import annotations

from typing import Optional, Union

import numpy as np
import pandas as pd
import scipy.cluster.hierarchy as sch
from scipy.spatial.distance import squareform

from ._checks import _require_real_values
from ._jit import jit

__all__ = ["correlation_distance", "quasi_diagonalize", "hrp_weights"]

_ALLOWED_METHODS = ("single", "complete", "average", "weighted")

MatrixLike = Union[np.ndarray, pd.DataFrame]


def correlation_distance(corr: MatrixLike) -> MatrixLike:
    """Map a correlation matrix to the metric ``d = sqrt(0.5 * (1 - rho))``.

    Parameters
    ----------
    corr : array-like or pandas.DataFrame, shape (N, N)
        Correlation matrix with entries in ``[-1, 1]``.

    Returns
    -------
    numpy.ndarray or pandas.DataFrame, shape (N, N)
        Distance matrix with entries in ``[0, 1]``, zero diagonal and
        ``d_ij = 0`` only for perfectly correlated pairs.

    Notes
    -----
    The distance is zero exactly when the correlation is 1, which includes any
    positively scaled or shifted copy of a series. So it is a metric on
    standardized series only. The book states the metric property
    in AFML section 16.4.1 (Appendix 16.A.1); that statement was not checked in
    this repository. The correlations are clipped to ``[-1, 1]`` first, so
    floating-point noise in an estimated matrix does not produce NaN.

    Scope: square input with finite entries in [-1, 1] (up to float noise).
    Symmetry is checked by :func:`hrp_weights`, not here.
    """
    raw = np.asarray(corr)
    _require_real_values("corr", raw)
    arr = np.asarray(raw, dtype=np.float64)
    if arr.ndim != 2 or arr.shape[0] != arr.shape[1]:
        raise ValueError(f"corr must be square, got shape {arr.shape}")
    # The distance is defined for a correlation matrix, whose diagonal is 1. A non-unit diagonal
    # gave nonzero self-distances in the public output (hrp_weights zeroed them, so it hid this).
    # The tolerance is 1e-6, as in hrp_weights, so a float32-stored corr is accepted.
    if np.all(np.isfinite(arr)) and not np.allclose(np.diag(arr), 1.0, rtol=0.0, atol=1e-6):
        raise ValueError("corr must have a unit diagonal")
    # +inf must not map to distance 0, -2 must not be clipped silently, and NaN must not pass through.
    # The clip now only absorbs float noise just outside [-1, 1].
    if not np.all(np.isfinite(arr)):
        raise ValueError("corr must contain only finite values")
    if np.any(np.abs(arr) > 1.0 + 1e-6):
        raise ValueError("corr entries must lie in [-1, 1]")
    rho = np.clip(arr, -1.0, 1.0)
    dist = np.sqrt(0.5 * (1.0 - rho))
    if isinstance(corr, pd.DataFrame):
        return pd.DataFrame(dist, index=corr.index, columns=corr.columns)
    return dist


def quasi_diagonalize(link: np.ndarray) -> list[int]:
    """Order the leaves of a linkage tree so that similar assets are adjacent.

    Parameters
    ----------
    link : numpy.ndarray, shape (N-1, 4)
        Linkage matrix as returned by :func:`scipy.cluster.hierarchy.linkage`.

    Returns
    -------
    list of int
        A permutation of ``0..N-1``. Visiting the root's left child before
        its right child, depth first, gives the order.

    Notes
    -----
    This is the procedure of Snippet 16.2 without its pandas index tricks.
    Replacing each merged cluster by its two constituents, left first,
    preserves the order of the clustering, so the result is the same
    sequence. Scope: the output is an ordering only. It reorders assets with no change of
    basis, and the covariance is re-indexed by it with no other change.
    """
    z = np.asarray(link)
    # Non-finite entries would reach int() below and fail there with OverflowError.
    if not np.all(np.isfinite(z)):
        raise ValueError("link must contain only finite values.")
    n = int(z.shape[0]) + 1
    if n == 1:
        return [0]
    children = z[:, :2]
    if not np.all(children == np.floor(children)):
        raise ValueError("link child ids must be integers.")
    ids = children.astype(np.int64)
    # Row i is node n + i, so it may use leaves 0..n-1 and the nodes of rows before it.
    limit = n + np.arange(n - 1)[:, None]
    bad_rows = np.flatnonzero(np.any((ids < 0) | (ids >= limit), axis=1))
    if bad_rows.size > 0:
        raise ValueError(
            f"link rows {bad_rows.tolist()} have a child id that is not a leaf or an earlier row."
        )
    flat = ids.ravel()
    if np.unique(flat).size != flat.size:
        # Each node merges into exactly one parent; a repeated child would drop leaves from the order.
        raise ValueError("link uses a child id more than once.")
    order: list[int] = []
    stack = [2 * n - 2]  # root cluster id in scipy's numbering
    while stack:
        node = stack.pop()
        if node < n:
            order.append(int(node))
        else:
            row = node - n
            stack.append(int(z[row, 1]))  # right child, popped after left
            stack.append(int(z[row, 0]))
    return order


@jit
def _cluster_variance(cov: np.ndarray, s: int, e: int) -> float:
    """Variance of the inverse-variance portfolio of the block ``[s, e)``."""
    k = e - s
    inv_sum = 0.0
    for i in range(s, e):
        inv_sum += 1.0 / cov[i, i]
    wt = np.empty(k, dtype=np.float64)
    for i in range(k):
        wt[i] = (1.0 / cov[s + i, s + i]) / inv_sum
    var = 0.0
    for i in range(k):
        for j in range(k):
            var += wt[i] * cov[s + i, s + j] * wt[j]
    return var


@jit
def _recursive_bisection(cov: np.ndarray) -> np.ndarray:
    """Recursive bisection on a quasi-diagonal covariance (Snippet 16.3).

    ``cov`` must already be in quasi-diagonal order. Returns weights in the
    same order.
    """
    n = cov.shape[0]
    w = np.ones(n, dtype=np.float64)
    stack_s = np.empty(n + 1, dtype=np.int64)
    stack_e = np.empty(n + 1, dtype=np.int64)
    stack_s[0] = 0
    stack_e[0] = n
    top = 1
    while top > 0:
        top -= 1
        s = stack_s[top]
        e = stack_e[top]
        if e - s < 2:
            continue
        m = s + (e - s) // 2  # |L1| = int(|L| / 2), order preserved
        v1 = _cluster_variance(cov, s, m)
        v2 = _cluster_variance(cov, m, e)
        tot = v1 + v2
        if tot > 0.0:
            # A tiny negative eigenvalue can make v1 negative and push alpha past 1.
            alpha = min(max(1.0 - v1 / tot, 0.0), 1.0)  # split factor in [0, 1]
        else:
            alpha = 0.5
        for i in range(s, m):
            w[i] *= alpha
        for i in range(m, e):
            w[i] *= 1.0 - alpha
        stack_s[top] = s
        stack_e[top] = m
        top += 1
        stack_s[top] = m
        stack_e[top] = e
        top += 1
    return w


def hrp_weights(
    cov: MatrixLike,
    corr: Optional[MatrixLike] = None,
    method: str = "single",
) -> Union[np.ndarray, pd.Series]:
    """Hierarchical Risk Parity weights (AFML section 16.4).

    Parameters
    ----------
    cov : array-like or pandas.DataFrame, shape (N, N)
        Covariance matrix. Only the diagonal and the off-diagonal entries
        used by the allocation are read. Must have a strictly positive
        diagonal. It need not be invertible.
    corr : array-like or pandas.DataFrame, shape (N, N), optional
        Correlation matrix used for clustering. If omitted, it is derived
        from ``cov``. If a DataFrame, it is aligned to ``cov``'s labels.
    method : {"single", "complete", "average", "weighted"}, default "single"
        Linkage passed to :func:`scipy.cluster.hierarchy.linkage`. The book
        uses ``"single"``. ``"ward"`` is excluded because it requires
        Euclidean observations, which a distance matrix lacks.

    Returns
    -------
    pandas.Series or numpy.ndarray, shape (N,)
        Weights in the original asset order, nonnegative and summing to 1.
        A Series indexed like ``cov`` if ``cov`` is a DataFrame.

    Notes
    -----
    Stages (AFML 16.4):

    1. Tree clustering on ``d = sqrt(0.5 (1 - rho))``.
    2. Quasi-diagonalization: reorder assets by the tree's leaf order.
    3. Recursive bisection: split each block into halves ``L1``, ``L2`` with
       ``|L1| = floor(|L|/2)``. Let ``V_j`` be the variance of the
       inverse-variance portfolio of ``L_j``, and ``alpha = 1 - V_1/(V_1+V_2)``.
       Multiply the weights in ``L1`` by ``alpha`` and those in ``L2`` by
       ``1 - alpha``.

    No matrix is inverted. The only divisions are by cluster variances and
    by the diagonal entries of ``cov``.

    Scope: one set of N assets, no constraints. For N = 1 the result is
    ``[1.0]``. Proved in ``docs/proofs/hrp.md``.
    """
    if method not in _ALLOWED_METHODS:
        raise ValueError(f"method must be one of {_ALLOWED_METHODS}, got {method!r}")
    if isinstance(cov, pd.DataFrame):
        labels = cov.index
    else:
        labels = None
    raw_cov = np.asarray(cov)
    _require_real_values("cov", raw_cov)
    C = np.asarray(raw_cov, dtype=np.float64)
    if C.ndim != 2 or C.shape[0] != C.shape[1]:
        raise ValueError(f"cov must be square, got shape {C.shape}")
    n = C.shape[0]
    if n == 0:
        raise ValueError("cov is empty")
    diag = np.diag(C)
    if not np.all(np.isfinite(C)):
        raise ValueError("cov contains non-finite values")
    if np.any(diag <= 0.0):
        raise ValueError("cov must have a strictly positive diagonal")
    # A non-PSD matrix gives negative weights, which break the long-only contract. The tolerance
    # admits float noise on the zero eigenvalues of a singular covariance, and the rounding of a
    # covariance stored in float32 (about 6e-8 relative); 1e-6 covers that and still refuses
    # a real negative eigenvalue or asymmetry. PSD is tested on the correlation-normalized
    # matrix, so the answer does not depend on the units of the assets.
    scale = float(np.max(np.abs(C)))
    if not np.allclose(C, C.T, rtol=1e-6, atol=1e-12 * scale):
        raise ValueError("cov must be symmetric")
    sd_c = np.sqrt(diag)
    Z = 0.5 * (C + C.T) / np.outer(sd_c, sd_c)
    lam_min = float(np.linalg.eigvalsh(0.5 * (Z + Z.T))[0])
    if lam_min < -1e-6:
        raise ValueError(f"cov must be positive semidefinite; its smallest eigenvalue is {lam_min!r}.")

    # The shape check runs for every N, so a single asset cannot hide a mismatched corr.
    if corr is None:
        sd = np.sqrt(diag)
        R = C / np.outer(sd, sd)
    else:
        if isinstance(corr, pd.DataFrame) and labels is not None:
            # A label missing from corr gave a KeyError from pandas; name the labels instead.
            missing = [lab for lab in labels if lab not in corr.index or lab not in corr.columns]
            if missing:
                raise ValueError(f"corr is missing labels of cov: {missing[:5]!r}.")
            R = np.asarray(corr.loc[labels, labels], dtype=np.float64)
        else:
            raw_corr = np.asarray(corr)
            _require_real_values("corr", raw_corr)
            R = np.asarray(raw_corr, dtype=np.float64)
        if R.shape != C.shape:
            raise ValueError("corr and cov must have the same shape")
        if not np.all(np.isfinite(R)):
            raise ValueError("corr must contain only finite values")
        # Symmetry is checked on the full matrix, not only the upper triangle, so an asymmetric corr is refused.
        # The tolerances are 1e-6, as for cov: a corr stored in float32 is valid.
        if not np.allclose(R, R.T, rtol=1e-6, atol=1e-6):
            raise ValueError("corr must be symmetric")
        # The range check also runs here, because correlation_distance is skipped when N == 1.
        if np.any(np.abs(R) > 1.0 + 1e-6):
            raise ValueError("corr entries must lie in [-1, 1]")
        # The unit diagonal is checked here for every N, as correlation_distance does for N > 1.
        if not np.allclose(np.diag(R), 1.0, rtol=0.0, atol=1e-6):
            raise ValueError("corr must have a unit diagonal")

    if n == 1:
        w = np.ones(1)
    else:
        D = np.asarray(correlation_distance(R), dtype=np.float64)
        np.fill_diagonal(D, 0.0)
        link = sch.linkage(squareform(D, checks=False), method=method)
        order = np.asarray(quasi_diagonalize(link), dtype=np.int64)
        Cs = np.ascontiguousarray(C[np.ix_(order, order)])
        ws = _recursive_bisection(Cs)
        w = np.empty(n, dtype=np.float64)
        w[order] = ws

    if labels is not None:
        return pd.Series(w, index=labels, name="hrp_weight")
    return w
