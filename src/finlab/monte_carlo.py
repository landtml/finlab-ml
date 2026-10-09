"""Seeded Monte Carlo trial runner.

``run_trials`` calls a user function once per trial, giving each trial its own
:class:`numpy.random.Generator`, and collects the returned scalars into a
DataFrame with one row per trial.

Random streams
--------------
Trial ``i`` draws from
``np.random.default_rng(np.random.SeedSequence(seed, spawn_key=(i,)))``.
The stream of trial ``i`` therefore depends only on ``(seed, i)``. The output
does not change with ``num_threads``, with how trials are chunked across
workers, or with ``n_iter`` (a longer run extends a shorter one).

Scope
-----
Trials run through :func:`finlab.parallel.mp_pandas_obj`, so ``num_threads > 1``
uses a process pool and requires ``func`` to be defined at module level. All
random draws are made by numpy. The bootstrap trial below also calls numba
kernels from :mod:`finlab.weights`, but those take the uniforms as input and
draw nothing themselves.

Bootstrap uniqueness experiment
-------------------------------
:func:`bootstrap_uniqueness_mc` runs the experiment of AFML section 4.5.4, the
Monte Carlo comparison of standard and sequential bootstrap uniqueness. It
follows the printed listings of Snippets 4.7 and 4.8 (label generation and the
two bootstraps), with one change: random numbers come from a seeded
:class:`numpy.random.Generator`, not numpy's legacy global generator. So a single
trial is not the book's trial, but the design is the same. The book's Snippet 4.9
runs 1E6 trials through its own job engine; :func:`bootstrap_uniqueness_mc` runs
``n_iter`` trials (10,000 by default) through :func:`run_trials`.

Not covered
-----------
Convergence diagnostics, variance reduction, formal tests of which bootstrap
gives higher uniqueness, and distributed execution across machines.
"""

from __future__ import annotations

import functools
from collections.abc import Mapping
from typing import Any, Callable

import numpy as np
import pandas as pd

from ._jit import jit
from .parallel import mp_pandas_obj
from .weights import _concurrency_sweep, _seq_bootstrap_kernel, _span_sums

__all__ = ["run_trials", "random_t1", "bootstrap_uniqueness_trial", "bootstrap_uniqueness_mc"]


def _is_int(value: Any) -> bool:
    return isinstance(value, (int, np.integer)) and not isinstance(value, bool)


def _check_positive_int(name: str, value: Any) -> None:
    if not _is_int(value) or value < 1:
        raise ValueError(f"{name} must be a positive integer, got {value!r}.")


def _resolve_seed(seed: int | None) -> int:
    if seed is None:
        # Fresh entropy, drawn once so that every trial in this run shares it.
        return int(np.random.SeedSequence().entropy)
    if not _is_int(seed) or seed < 0:
        raise ValueError(f"seed must be None or a non-negative integer, got {seed!r}.")
    return int(seed)


def _trial_rng(seed: int, i: int) -> np.random.Generator:
    return np.random.default_rng(np.random.SeedSequence(seed, spawn_key=(i,)))


def _trial_molecule(
    molecule: pd.Index,
    func: Callable[..., Mapping[str, float]],
    seed: int,
    kwargs: dict[str, Any],
    columns: list[str],
) -> pd.DataFrame:
    """Run the trials in ``molecule`` and return one row per trial, indexed by trial."""
    rows = []
    for i in molecule:
        result = func(_trial_rng(seed, int(i)), **kwargs)
        if set(result.keys()) != set(columns):
            raise ValueError(
                f"trial {int(i)} returned keys {sorted(result.keys())}, "
                f"expected {sorted(columns)} (the keys of trial 0)."
            )
        rows.append([result[c] for c in columns])
    return pd.DataFrame(rows, index=molecule, columns=columns)


def run_trials(
    func: Callable[..., Mapping[str, float]],
    n_iter: int,
    seed: int | None = 0,
    num_threads: int = 1,
    **kwargs: Any,
) -> pd.DataFrame:
    """Run ``func`` ``n_iter`` times with independent, reproducible random streams.

    Parameters
    ----------
    func : callable
        Called as ``func(rng, **kwargs)`` where ``rng`` is a
        :class:`numpy.random.Generator` dedicated to that trial. Returns a
        mapping from result name to a float. Every trial must return the same
        keys. Must be defined at module level when ``num_threads > 1``.
    n_iter : int
        Number of trials; must be a positive integer.
    seed : int or None, default 0
        Non-negative integer seed. ``None`` draws fresh entropy once for the
        whole run. The seed used is recorded in ``result.attrs["seed"]``, so a
        run made with ``seed=None`` can be reproduced by passing that value.
    num_threads : int, default 1
        Number of worker processes; must be a positive integer. ``1`` runs in
        the calling process. The results do not depend on this value.
    **kwargs
        Passed through to ``func``. Names that clash with the parameters of
        ``run_trials`` (``func``, ``n_iter``, ``seed``, ``num_threads``) cannot be
        used.

    Returns
    -------
    pd.DataFrame
        One row per trial, in trial order, with index ``pd.RangeIndex(n_iter)``
        and columns taken from the keys of trial 0. ``attrs["seed"]`` holds the
        integer seed used.

    Raises
    ------
    ValueError
        If ``n_iter`` or ``num_threads`` is not a positive integer, if ``seed``
        is neither ``None`` nor a non-negative integer, if ``func`` returns no
        keys, or if a trial returns a different set of keys from trial 0.

    Notes
    -----
    Trial 0 is evaluated once in the calling process to fix the column names,
    and then evaluated again as part of the run. Because each trial's stream is
    fixed by ``(seed, i)``, both evaluations give the same result. The extra cost
    is one trial out of ``n_iter``.
    """
    _check_positive_int("n_iter", n_iter)
    _check_positive_int("num_threads", num_threads)
    seed_value = _resolve_seed(seed)

    columns = list(func(_trial_rng(seed_value, 0), **kwargs).keys())
    if not columns:
        raise ValueError("func must return at least one result key.")

    worker = functools.partial(
        _trial_molecule, func=func, seed=seed_value, kwargs=kwargs, columns=columns
    )
    out = mp_pandas_obj(worker, pd.RangeIndex(n_iter), num_threads=num_threads)
    out.index = pd.RangeIndex(n_iter)
    out.attrs["seed"] = seed_value
    return out


# ---------------------------------------------------------------------------
# Bootstrap uniqueness experiment
# ---------------------------------------------------------------------------


def _check_label_args(n_obs: int, n_bars: int, max_h: int) -> None:
    if not (_is_int(n_obs) and _is_int(n_bars) and _is_int(max_h)):
        raise ValueError("n_obs, n_bars and max_h must be integers.")
    if n_obs < 1 or n_bars < 1:
        raise ValueError(f"n_obs and n_bars must be at least 1, got {n_obs}, {n_bars}.")
    if max_h < 2:
        raise ValueError(f"max_h must be at least 2 (lengths are 1..max_h-1), got {max_h}.")


def _random_labels(
    n_obs: int, n_bars: int, max_h: int, rng: np.random.Generator
) -> tuple[np.ndarray, np.ndarray]:
    """Snippet 4.7 in numpy: sorted distinct starts and the end of each, from ``rng``.

    Same draws as :func:`random_t1`. Used directly by the trial, which avoids building
    a pandas object on every trial.
    """
    starts = rng.integers(0, n_bars, size=n_obs).astype(np.int64)
    lengths = rng.integers(1, max_h, size=n_obs).astype(np.int64)
    # Keep the last draw for each start, as the book's overwrite loop does.
    uniq, first_in_reversed = np.unique(starts[::-1], return_index=True)
    last_draw = n_obs - 1 - first_in_reversed
    return uniq, uniq + lengths[last_draw]


def random_t1(
    n_obs: int,
    n_bars: int,
    max_h: int,
    seed: int | np.random.Generator | None = None,
) -> pd.Series:
    """Random label set, as in AFML Snippet 4.7 (``getRndT1``).

    Parameters
    ----------
    n_obs : int
        Number of draws, at least 1. Labels can collapse (see Notes), so the
        result can have fewer than ``n_obs`` labels.
    n_bars : int
        Number of bars at the start of the sample. Start bars are drawn from
        ``0..n_bars-1``.
    max_h : int
        Upper bound of the label length, at least 2. Lengths are drawn from
        ``1..max_h-1``, because the book draws ``randint(1, maxH)``, which excludes
        ``maxH``.
    seed : int, numpy.random.Generator or None, default None
        A Generator is used directly. Any other value is passed to
        :func:`numpy.random.default_rng`.

    Returns
    -------
    pd.Series
        Index: the distinct start bars, sorted ascending. Values: the end bar of
        each label, ``start + length``. Ends are not clipped, so they can exceed
        ``n_bars - 1``.

    Raises
    ------
    ValueError
        If an argument is not an integer, if ``n_obs < 1`` or ``n_bars < 1``, or if
        ``max_h < 2``.

    Notes
    -----
    Draw order: ``n_obs`` start bars, ``rng.integers(0, n_bars, size=n_obs)``, then
    ``n_obs`` lengths, ``rng.integers(1, max_h, size=n_obs)``.

    The book's loop assigns ``t1.loc[ix] = val`` for each draw, so a start drawn more
    than once keeps the length of its last draw. The result is built the same way.
    """
    _check_label_args(n_obs, n_bars, max_h)
    rng = seed if isinstance(seed, np.random.Generator) else np.random.default_rng(seed)
    uniq, ends = _random_labels(n_obs, n_bars, max_h, rng)
    return pd.Series(ends, index=pd.Index(uniq))


@jit
def _spans_to_csc(starts: np.ndarray, ends: np.ndarray) -> tuple:
    """CSC layout of the indicator matrix: label ``j`` covers ``rows[ptr[j]:ptr[j+1]]``."""
    n_ev = starts.shape[0]
    ptr = np.zeros(n_ev + 1, dtype=np.int64)
    for j in range(n_ev):
        length = ends[j] - starts[j] + 1
        if length < 0:
            length = 0
        ptr[j + 1] = ptr[j] + length
    rows = np.empty(ptr[n_ev], dtype=np.int64)
    colsum = np.empty(n_ev, dtype=np.float64)
    for j in range(n_ev):
        q = ptr[j]
        for t in range(starts[j], ends[j] + 1):
            rows[q] = t
            q += 1
        colsum[j] = float(ptr[j + 1] - ptr[j])
    return ptr, rows, colsum


@jit
def _sample_uniqueness(n_bars: int, starts: np.ndarray, ends: np.ndarray) -> float:
    """Average uniqueness of a sample given the spans of its draws, repeats kept.

    Same quantity as :func:`finlab.weights.sample_average_uniqueness`, computed
    from spans. Needs at least one draw.
    """
    n = starts.shape[0]
    c = _concurrency_sweep(n_bars, starts, ends)
    u = np.zeros(n_bars, dtype=np.float64)
    for t in range(n_bars):
        if c[t] > 0:
            u[t] = 1.0 / c[t]
    sums = _span_sums(u, starts, ends)
    acc = 0.0
    for k in range(n):
        acc += sums[k] / float(ends[k] - starts[k] + 1)
    return acc / n


@jit
def _trial_uniqueness(
    n_bars: int,
    starts: np.ndarray,
    ends: np.ndarray,
    std_draws: np.ndarray,
    uniforms: np.ndarray,
) -> tuple:
    """Average uniqueness of a standard and a sequential bootstrap sample.

    ``starts`` and ``ends`` give each label's span in bars. ``std_draws`` holds
    label positions (repeats allowed). The sequential sample is drawn by
    :func:`finlab.weights._seq_bootstrap_kernel` with one entry of ``uniforms``
    per draw. Returns ``(std_u, seq_u)``.
    """
    ptr, rows, colsum = _spans_to_csc(starts, ends)
    seq_draws = _seq_bootstrap_kernel(n_bars, ptr, rows, colsum, uniforms)
    std_u = _sample_uniqueness(n_bars, starts[std_draws], ends[std_draws])
    seq_u = _sample_uniqueness(n_bars, starts[seq_draws], ends[seq_draws])
    return std_u, seq_u


def bootstrap_uniqueness_trial(
    rng: np.random.Generator, n_obs: int, n_bars: int, max_h: int
) -> dict[str, float]:
    """One trial: average uniqueness of a standard and a sequential bootstrap.

    Parameters
    ----------
    rng : numpy.random.Generator
        Stream for this trial, as supplied by :func:`run_trials`.
    n_obs : int
        Number of draws passed to :func:`random_t1`.
    n_bars : int
        Number of start bars, passed to :func:`random_t1`.
    max_h : int
        Upper bound of the label length, passed to :func:`random_t1`.

    Returns
    -------
    dict[str, float]
        ``"std_u"``: average uniqueness of the standard bootstrap sample.
        ``"seq_u"``: average uniqueness of the sequential bootstrap sample.

    Notes
    -----
    Follows AFML Snippet 4.8. The bar grid runs from 0 to the largest label end,
    as in ``range(t1.max() + 1)`` in the book. Each sample has as many draws as
    there are labels (after :func:`random_t1` has merged repeated starts), as the
    book's ``np.random.choice(indM.columns, size=indM.shape[1])`` and
    ``seqBootstrap(indM)`` do.

    Draw order from ``rng``, fixed by this docstring:

    1. :func:`random_t1`: ``n_obs`` start bars, then ``n_obs`` lengths.
    2. ``std_draws = rng.integers(0, m, size=m)``: the standard bootstrap, with
       repeats. ``m`` is the number of labels.
    3. ``uniforms = rng.random(m)``: one uniform per sequential draw.

    The sequential sample uses ``uniforms`` through the same inverse-CDF rule as
    :func:`finlab.weights.sequential_bootstrap`. Both averages keep repeated
    draws. Both lie in ``(0, 1]``.
    """
    _check_label_args(n_obs, n_bars, max_h)
    starts, ends = _random_labels(n_obs, n_bars, max_h, rng)
    m = len(starts)
    grid = int(ends.max()) + 1
    std_draws = rng.integers(0, m, size=m)
    uniforms = rng.random(m)
    std_u, seq_u = _trial_uniqueness(grid, starts, ends, std_draws, uniforms)
    return {"std_u": float(std_u), "seq_u": float(seq_u)}


def bootstrap_uniqueness_mc(
    n_obs: int = 10,
    n_bars: int = 100,
    max_h: int = 5,
    n_iter: int = 10_000,
    seed: int | None = 0,
    num_threads: int = 1,
) -> pd.DataFrame:
    """Monte Carlo comparison of standard and sequential bootstrap uniqueness.

    Parameters
    ----------
    n_obs : int, default 10
        Number of draws in :func:`random_t1`. The book's Snippet 4.9 uses 10.
    n_bars : int, default 100
        Number of start bars. The book's Snippet 4.9 uses 100.
    max_h : int, default 5
        Upper bound of the label length. The book's Snippet 4.9 uses 5.
    n_iter : int, default 10_000
        Number of trials. The book's Snippet 4.9 uses 1E6. The default is smaller
        for speed; it is not the book's value.
    seed : int or None, default 0
        Seed for :func:`run_trials`.
    num_threads : int, default 1
        Number of worker processes. The result does not depend on it.

    Returns
    -------
    pd.DataFrame
        One row per trial, with columns ``["std_u", "seq_u"]``. Each value is the
        average uniqueness of one bootstrap sample (see
        :func:`bootstrap_uniqueness_trial`).

    Notes
    -----
    The three experiment defaults (``n_obs``, ``n_bars``, ``max_h``) are the values
    printed in Snippet 4.9. ``n_iter`` is a smaller run count than the book's 1E6.
    The book reports medians of about 0.6 for the standard bootstrap and 0.7 for the
    sequential bootstrap (text after Figure 4.2). Those medians are compared with the
    measured ones in ``docs/proofs/monte_carlo.md``, not asserted here.
    """
    return run_trials(
        bootstrap_uniqueness_trial,
        n_iter,
        seed=seed,
        num_threads=num_threads,
        n_obs=n_obs,
        n_bars=n_bars,
        max_h=max_h,
    )
