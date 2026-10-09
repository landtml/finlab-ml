"""Sample weights and sequential bootstrap for overlapping labels (AFML Chapter 4).

Implements:

* Snippet 4.1 -- :func:`num_co_events`, the concurrency ``c_t`` of labels.
  The book loops over events with pandas slices. Here a numba difference-array
  sweep computes all counts in one pass.
* Section 4.4, Snippet 4.2 -- :func:`average_uniqueness`, the mean of
  ``1 / c_t`` over each label's lifespan.
* :func:`sample_average_uniqueness`, the average uniqueness of a bootstrap sample,
  counting repeated draws (section 4.4 topic; snippet numbering not verified).
* Snippet 4.3 -- :func:`indicator_matrix`, the bar-by-label matrix ``1_{t,i}``.
* Snippet 4.5 -- :func:`sequential_bootstrap`, the uniqueness-driven draw of
  labels, with the inner loop in numba.
* Snippet 4.10 -- :func:`sample_weight_by_return`, weights from absolute
  return attribution, normalised to sum to the number of labels.
* Snippet 4.11 -- :func:`time_decay`, the piecewise-linear decay applied to
  cumulative uniqueness.

Not covered: the multiprocessing engine ``mpPandasObj`` (Chapter 20), the
Monte Carlo study of Snippets 4.7-4.9 (not reproduced here; benchmarks/bench_weights.py
measures speed only), class weights (end of Section 4.8), and the bagging classifier
of Chapter 6. A separate Monte Carlo experiment, designed in this repository, is in
:mod:`finlab.monte_carlo`; its claims are in docs/proofs/monte_carlo.md. The book's
study is not reproduced there either.

Conventions: ``index`` is the sorted grid of bars. ``t1`` is a Series whose
index holds label start times and whose values hold label end times (NaT
means the label runs to the last bar). Labels are identified by their
position in ``t1``.
"""

from __future__ import annotations

from typing import Sequence

import numpy as np
import pandas as pd

from ._jit import jit, pjit

__all__ = [
    "num_co_events",
    "average_uniqueness",
    "sample_average_uniqueness",
    "indicator_matrix",
    "sequential_bootstrap",
    "sample_weight_by_return",
    "time_decay",
]


# ---------------------------------------------------------------------------
# Span helpers
# ---------------------------------------------------------------------------


def _grid(index: pd.Index | Sequence) -> pd.Index:
    grid = pd.Index(index)
    if grid.has_duplicates or not grid.is_monotonic_increasing:
        raise ValueError("index must be strictly increasing bar times.")
    return grid


def _spans(grid: pd.Index, t1: pd.Series) -> tuple[np.ndarray, np.ndarray]:
    """Bar positions ``[start, end]`` (inclusive) of each label.

    ``start`` is the first bar at or after the label's start time. ``end`` is
    the last bar at or before ``t1``. NaT ``t1`` gives ``end = n - 1``. A label
    that starts after the last bar gets ``end < start`` and an empty span.
    """
    n = len(grid)
    starts = np.asarray(grid.searchsorted(pd.Index(t1.index), side="left"), dtype=np.int64)
    vals = t1.to_numpy()
    ends = np.full(len(vals), n - 1, dtype=np.int64)
    valid = ~pd.isna(vals)
    if np.any(valid):
        ends[valid] = (
            np.asarray(grid.searchsorted(pd.Index(vals[valid]), side="right"), dtype=np.int64) - 1
        )
    return starts, ends


@jit
def _concurrency_sweep(n_bars: int, starts: np.ndarray, ends: np.ndarray) -> np.ndarray:
    """``c[t] = #{i : starts[i] <= t <= ends[i]}`` via a difference array."""
    diff = np.zeros(n_bars + 1, dtype=np.int64)
    for k in range(starts.shape[0]):
        s = starts[k]
        e = ends[k]
        if e >= s and s >= 0:
            diff[s] += 1
            diff[e + 1] -= 1
    out = np.empty(n_bars, dtype=np.int64)
    run = 0
    for t in range(n_bars):
        run += diff[t]
        out[t] = run
    return out


@jit
def _span_sums(v: np.ndarray, starts: np.ndarray, ends: np.ndarray) -> np.ndarray:
    """Sum of ``v`` over each label's inclusive span (0 for empty spans)."""
    out = np.zeros(starts.shape[0], dtype=np.float64)
    for k in range(starts.shape[0]):
        acc = 0.0
        for t in range(starts[k], ends[k] + 1):
            acc += v[t]
        out[k] = acc
    return out


@jit
def _indicator(n_bars: int, starts: np.ndarray, ends: np.ndarray) -> np.ndarray:
    """Dense ``(n_bars, n_labels)`` 0/1 matrix ``1_{t,i}``."""
    m = np.zeros((n_bars, starts.shape[0]), dtype=np.float64)
    for k in range(starts.shape[0]):
        for t in range(starts[k], ends[k] + 1):
            m[t, k] = 1.0
    return m


# ---------------------------------------------------------------------------
# Snippet 4.1 -- concurrency
# ---------------------------------------------------------------------------


def num_co_events(index: pd.Index | Sequence, t1: pd.Series) -> pd.Series:
    """Number of labels whose lifespan covers each bar (Snippet 4.1).

    Parameters
    ----------
    index : pd.Index or array
        Strictly increasing bar times.
    t1 : pd.Series
        Label start times (index) and end times (values). NaT ends run to the
        last bar.

    Returns
    -------
    pd.Series
        ``c_t`` indexed by ``index``, integer-valued, zero where no label is
        active.

    Notes
    -----
    ``c_t = sum_i 1[t0_i <= t <= t1_i]``. A label is counted at the bar where
    it starts and at the bar where it ends. Computed in O(n + I) with a
    difference array: ``+1`` at each start, ``-1`` just after each end, then a
    running sum.

    Scope: bars are discrete. Label times are mapped to bar positions with
    ``searchsorted``, so label times need not be bar times.
    """
    grid = _grid(index)
    starts, ends = _spans(grid, t1)
    c = _concurrency_sweep(len(grid), starts, ends)
    return pd.Series(c, index=grid, name="c_t")


# ---------------------------------------------------------------------------
# Section 4.4 -- average uniqueness
# ---------------------------------------------------------------------------


def average_uniqueness(
    index: pd.Index | Sequence, t1: pd.Series, c_t: pd.Series | np.ndarray
) -> pd.Series:
    """Average uniqueness of each label over its lifespan (Section 4.4, Snippet 4.2).

    Parameters
    ----------
    index : pd.Index or array
        Bar grid, as in :func:`num_co_events`.
    t1 : pd.Series
        Label start times (index) and end times (values).
    c_t : pd.Series or array
        Concurrency per bar, aligned with ``index`` by position (usually the
        output of :func:`num_co_events`).

    Returns
    -------
    pd.Series
        ``ū_i = (sum_{t in span_i} 1/c_t) / |span_i|``, indexed by ``t1.index``.
        Labels with an empty span get NaN.

    Notes
    -----
    Uniqueness at bar ``t`` is ``u_{t,i} = 1_{t,i} / c_t``. The average is
    taken over the bars in the label's span. It equals the reciprocal of the
    harmonic mean of ``c_t`` over the span. It is 1 exactly when ``c_t = 1``
    on the whole span (no overlap), and it lies in ``(0, 1]`` otherwise.

    Scope: bars where ``c_t = 0`` contribute 0. Such bars cannot lie inside any
    label's span, so this only matters for malformed input.
    """
    grid = _grid(index)
    starts, ends = _spans(grid, t1)
    c = np.asarray(c_t, dtype=np.float64).reshape(-1)
    if c.shape[0] != len(grid):
        raise ValueError("c_t must have one entry per bar in index.")
    u = np.zeros_like(c)
    np.divide(1.0, c, out=u, where=c > 0)
    sums = _span_sums(u, starts, ends)
    lengths = (ends - starts + 1).astype(np.float64)
    with np.errstate(invalid="ignore", divide="ignore"):
        avg = np.where(lengths > 0, sums / np.maximum(lengths, 1.0), np.nan)
    return pd.Series(avg, index=t1.index, name="avg_u")


def sample_average_uniqueness(
    index_matrix: pd.DataFrame | np.ndarray,
    draws: Sequence[int] | np.ndarray | None = None,
) -> float:
    """Average uniqueness of a bootstrap sample, repeats included (section 4.4).

    Parameters
    ----------
    index_matrix : pd.DataFrame or 2-D array
        Indicator matrix with bars as rows and labels as columns, as returned by
        :func:`indicator_matrix`. Non-zero entries count as 1.
    draws : 1-D integer array, optional
        Column positions of the sampled labels, possibly with repeats. ``None``
        means every column once, in order.

    Returns
    -------
    float
        Mean over the drawn labels of their average uniqueness. Each repeat of a
        label counts separately.

    Raises
    ------
    ValueError
        If ``index_matrix`` is not two-dimensional, if ``draws`` is empty, not
        one-dimensional, not integer-valued, or has a position outside the
        columns, or if a drawn label covers no bar.

    Notes
    -----
    Concurrency is taken over the sample: ``c_t = sum_k 1_{t, d_k}``, where
    ``d_k`` runs over the draws, so a label drawn twice covers each of its bars
    twice. For a drawn label ``d``, its uniqueness is
    ``ū_d = (1 / |L_d|) sum_{t in L_d} 1 / c_t``, and the result is
    ``(1/n) sum_k ū_{d_k}`` over the ``n`` draws.

    With ``draws=None`` the sample is the full set of labels, so the result equals
    ``average_uniqueness(index, t1, num_co_events(index, t1)).mean()``. The sum over
    draws is computed with one dense matrix product, so the cost is
    O(n_bars * n_draws).
    """
    if isinstance(index_matrix, pd.DataFrame):
        m = index_matrix.to_numpy(dtype=np.float64)
    else:
        m = np.asarray(index_matrix, dtype=np.float64)
    if m.ndim != 2:
        raise ValueError("index_matrix must be two-dimensional.")
    n_bars, n_ev = m.shape
    if draws is None:
        pos = np.arange(n_ev, dtype=np.int64)
    else:
        pos = np.asarray(draws)
        if pos.ndim != 1:
            raise ValueError("draws must be one-dimensional.")
    if pos.size == 0:
        raise ValueError("draws must not be empty.")
    if draws is not None and not np.issubdtype(pos.dtype, np.integer):
        raise ValueError("draws must contain integer column positions.")
    if pos.min() < 0 or pos.max() >= n_ev:
        raise ValueError("draws contains a position outside the columns of index_matrix.")
    sub = (m[:, pos] != 0.0).astype(np.float64)  # one column per draw, repeats kept
    span_len = sub.sum(axis=0)
    if np.any(span_len == 0.0):
        raise ValueError("a drawn label covers no bar.")
    c = sub.sum(axis=1)
    inv = np.zeros(n_bars, dtype=np.float64)
    np.divide(1.0, c, out=inv, where=c > 0)
    u = (inv @ sub) / span_len
    return float(np.mean(u))


# ---------------------------------------------------------------------------
# Snippet 4.3 -- indicator matrix
# ---------------------------------------------------------------------------


def indicator_matrix(index: pd.Index | Sequence, t1: pd.Series) -> pd.DataFrame:
    """Binary bar-by-label matrix ``1_{t,i}`` (Snippet 4.3).

    Parameters
    ----------
    index : pd.Index or array
        Bar grid (rows).
    t1 : pd.Series
        Label start times (index) and end times (values) (columns).

    Returns
    -------
    pd.DataFrame
        Float matrix of shape ``(len(index), len(t1))``. Rows are bars, columns
        are labels, labelled by ``index`` and ``t1.index``.

    Notes
    -----
    The book's orientation is kept (bars as rows, labels as columns). The
    sequential bootstrap draws labels, that is, columns.
    """
    grid = _grid(index)
    starts, ends = _spans(grid, t1)
    m = _indicator(len(grid), starts, ends)
    return pd.DataFrame(m, index=grid, columns=t1.index)


# ---------------------------------------------------------------------------
# Snippet 4.5 -- sequential bootstrap
# ---------------------------------------------------------------------------


@jit
def _seq_bootstrap_kernel(
    n_bars: int,
    ptr: np.ndarray,
    rows: np.ndarray,
    colsum: np.ndarray,
    uniforms: np.ndarray,
) -> np.ndarray:
    """Sequential-bootstrap draws, using a CSC-style sparse indicator matrix.

    Label ``j`` covers bars ``rows[ptr[j]:ptr[j+1]]`` (ascending). For each draw
    the candidate weight is the average uniqueness of ``j`` given the labels
    already drawn (``c[t]`` counts of earlier draws):
    ``avg[j] = (sum_t 1 / (1 + c[t])) / colsum[j]``. The draw is the first ``j``
    whose cumulative weight exceeds ``uniforms[k] * total``.
    """
    n_ev = ptr.shape[0] - 1
    n_draw = uniforms.shape[0]
    c = np.zeros(n_bars, dtype=np.float64)
    avg = np.zeros(n_ev, dtype=np.float64)
    draws = np.empty(n_draw, dtype=np.int64)
    for k in range(n_draw):
        total = 0.0
        for j in range(n_ev):
            acc = 0.0
            if colsum[j] > 0.0:
                for q in range(ptr[j], ptr[j + 1]):
                    acc += 1.0 / (1.0 + c[rows[q]])
                acc = acc / colsum[j]
            avg[j] = acc
            total += acc
        target = uniforms[k] * total
        acc = 0.0
        pick = -1
        last_pos = -1
        for j in range(n_ev):
            if avg[j] > 0.0:
                last_pos = j
                acc += avg[j]
                if acc > target:
                    pick = j
                    break
        if pick < 0:
            pick = last_pos
        draws[k] = pick
        for q in range(ptr[pick], ptr[pick + 1]):
            c[rows[q]] += 1.0
    return draws


def sequential_bootstrap(
    index_matrix: pd.DataFrame | np.ndarray,
    n_samples: int | None = None,
    seed: int | None = None,
) -> np.ndarray:
    """Draw labels by sequential bootstrap (Snippet 4.5).

    Parameters
    ----------
    index_matrix : pd.DataFrame or 2-D array
        Indicator matrix with bars as rows and labels as columns, as returned by
        :func:`indicator_matrix`. Non-zero entries count as 1.
    n_samples : int, optional
        Number of draws. Defaults to the number of labels, as in the book.
    seed : int, optional
        Seed for :func:`numpy.random.default_rng`. ``None`` gives fresh entropy.

    Returns
    -------
    np.ndarray
        Integer array of label positions (columns of ``index_matrix``), length
        ``n_samples``. Repeats are allowed, but they become less likely as the
        sample accumulates.

    Notes
    -----
    Draw ``k`` picks label ``j`` with probability proportional to its average
    uniqueness given the labels already drawn,
    ``delta_j = ū_j^{(k)} / sum_l ū_l^{(k)}``, where
    ``ū_j^{(k)} = (1/T_j) sum_t 1_{t,j} / (1 + c_t^{(k-1)})``. Uniforms are
    drawn with numpy and mapped to labels by inverse CDF in numba, so the output
    is reproducible for a given seed and matches a naive loop that uses the
    same uniforms.

    Scope: cost is O(n_samples * (T + nnz)) with nnz the number of 1 entries.
    """
    if isinstance(index_matrix, pd.DataFrame):
        m = index_matrix.to_numpy(dtype=np.float64)
    else:
        m = np.asarray(index_matrix, dtype=np.float64)
    if m.ndim != 2:
        raise ValueError("index_matrix must be two-dimensional.")
    n_bars, n_ev = m.shape
    if n_ev == 0:
        raise ValueError("index_matrix has no labels.")
    binary = (m != 0.0)
    if not binary.any():
        raise ValueError("index_matrix has no active entries.")
    n_draw = n_ev if n_samples is None else int(n_samples)
    if n_draw < 1:
        raise ValueError("n_samples must be >= 1.")
    j_idx, t_idx = np.nonzero(binary.T)
    colsum = np.bincount(j_idx, minlength=n_ev).astype(np.float64)
    ptr = np.zeros(n_ev + 1, dtype=np.int64)
    ptr[1:] = np.cumsum(colsum.astype(np.int64))
    rows = t_idx.astype(np.int64)
    rng = np.random.default_rng(seed)
    uniforms = rng.random(n_draw)
    return _seq_bootstrap_kernel(n_bars, ptr, rows, colsum, uniforms)


# ---------------------------------------------------------------------------
# Snippet 4.10 -- return attribution
# ---------------------------------------------------------------------------


def sample_weight_by_return(
    index: pd.Index | Sequence, t1: pd.Series, close: pd.Series
) -> pd.Series:
    """Sample weights from absolute return attribution (Snippet 4.10).

    Parameters
    ----------
    index : pd.Index or array
        Bar grid. ``close`` is looked up on it.
    t1 : pd.Series
        Label start times (index) and end times (values).
    close : pd.Series
        Strictly positive prices indexed by (a superset of) ``index``.

    Returns
    -------
    pd.Series
        ``w_i = w̃_i * I / sum_j w̃_j`` indexed by ``t1.index``, with
        ``w̃_i = | sum_{t in span_i} r_t / c_t |``, ``r_t = log close_t - log close_{t-1}``
        (0 at the first bar) and ``c_t`` from :func:`num_co_events`. The weights
        sum to ``I``.

    Notes
    -----
    Log returns are additive, so the sum over a span is the label's log return
    after each bar's return is split among the ``c_t`` labels active at that
    bar. The start bar's return is included, as in the book.

    Scope: a label with zero attributed return gets weight 0. Raises if every
    attributed return is zero.
    """
    grid = _grid(index)
    px = close.reindex(grid).to_numpy(dtype=np.float64)
    if not np.all(np.isfinite(px)) or np.any(px <= 0):
        raise ValueError("close must cover every bar in index with positive prices.")
    r = np.zeros(len(grid), dtype=np.float64)
    r[1:] = np.diff(np.log(px))
    c = num_co_events(grid, t1).to_numpy(dtype=np.float64)
    v = np.zeros_like(r)
    np.divide(r, c, out=v, where=c > 0)
    starts, ends = _spans(grid, t1)
    w_tilde = np.abs(_span_sums(v, starts, ends))
    total = w_tilde.sum()
    if not total > 0.0:
        raise ValueError("all attributed returns are zero; weights are undefined.")
    w = w_tilde * (len(w_tilde) / total)
    return pd.Series(w, index=t1.index, name="w")


# ---------------------------------------------------------------------------
# Snippet 4.11 -- time decay
# ---------------------------------------------------------------------------


def time_decay(
    weights: pd.Series | np.ndarray, c_last: float = 1.0
) -> pd.Series | np.ndarray:
    """Piecewise-linear decay of weights by cumulative uniqueness (Snippet 4.11).

    Parameters
    ----------
    weights : pd.Series or array
        Non-negative weights (typically the uniqueness ``ū_i``). Series are
        ordered by index. Arrays are taken as already chronological.
    c_last : float in (-1, 1], default 1.0
        Weight given to the oldest observation. ``1`` means no decay.

    Returns
    -------
    pd.Series or np.ndarray
        Decay factors in the original order, same type as ``weights``.

    Notes
    -----
    Let ``C_i = sum_{j <= i} w_j`` over the chronological order and
    ``T = C_last``. The factor is ``d_i = max(0, a + b C_i)`` with
    ``a = 1 - bT`` (so the newest observation has ``d = 1``) and
    * for ``c_last in [0, 1]``: ``b = (1 - c_last) / T``, so ``a = c_last``;
    * for ``c_last in (-1, 0)``: ``b = 1 / ((1 + c_last) T)``, and the factor is
      clipped to 0 for the oldest part of the sample.
    Decay runs on cumulative uniqueness, not on calendar time, so redundant
    observations do not shrink the weights too fast.

    Scope: the oldest observation has ``C = w_1``, not 0, so its factor is
    ``c_last + (1 - c_last) w_1 / T``. It equals ``c_last`` only in the limit of
    small ``w_1 / T`` (see docs/proofs/weights.md).
    """
    if not (-1.0 < c_last <= 1.0):
        raise ValueError("c_last must lie in (-1, 1].")
    is_series = isinstance(weights, pd.Series)
    if is_series:
        ws = weights.sort_index()
        values = ws.to_numpy(dtype=np.float64)
    else:
        values = np.asarray(weights, dtype=np.float64).reshape(-1)
    if np.any(values < 0.0) or not np.all(np.isfinite(values)):
        raise ValueError("weights must be finite and non-negative.")
    cum = np.cumsum(values)
    total = cum[-1] if cum.size else 0.0
    if not total > 0.0:
        raise ValueError("weights must have a positive sum.")
    if c_last >= 0.0:
        slope = (1.0 - c_last) / total
    else:
        slope = 1.0 / ((c_last + 1.0) * total)
    const = 1.0 - slope * total
    dec = np.maximum(const + slope * cum, 0.0)
    if is_series:
        out = pd.Series(dec, index=ws.index, name=weights.name)
        return out.reindex(weights.index)
    return dec
