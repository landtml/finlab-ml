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
:func:`bootstrap_uniqueness_mc` compares the average uniqueness of a standard
bootstrap sample with that of a sequential bootstrap sample (Chapter 4 style).
The label design (:func:`random_t1`) is a design choice made here, because the
book text is not in this repository. The defaults are not verified against the
book.

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


def random_t1(
    n_obs: int,
    n_bars: int,
    max_h: int,
    seed: int | np.random.Generator | None = None,
) -> pd.Series:
    """Random label set on the bar grid ``0..n_bars-1``.

    This label design is a choice made for the bootstrap experiment. It is not
    taken from the book.

    Parameters
    ----------
    n_obs : int
        Number of labels, with ``1 <= n_obs <= n_bars``.
    n_bars : int
        Number of bars. Bars are the integers ``0..n_bars-1``.
    max_h : int
        Largest label length in bars, at least 1.
    seed : int, numpy.random.Generator or None, default None
        A Generator is used directly. Any other value is passed to
        :func:`numpy.random.default_rng`.

    Returns
    -------
    pd.Series
        Index: the distinct start bars, sorted ascending. Values: the end bar of
        each label, as integers. Label ``i`` runs from ``index[i]`` to
        ``values[i]``.

    Raises
    ------
    ValueError
        If an argument is not an integer, if ``n_obs`` is outside ``1..n_bars``,
        or if ``max_h < 1``.

    Notes
    -----
    Draw order: ``n_obs`` distinct start bars, ``rng.choice(n_bars, size=n_obs,
    replace=False)``, sorted. Then one length per label in sorted order,
    ``rng.integers(1, max_h + 1, size=n_obs)``. The end is
    ``min(start + length, n_bars - 1)``. So ``end - start`` lies in ``1..max_h``,
    except where an end is clipped at the last bar, where it can be 0.
    """
    if not (_is_int(n_obs) and _is_int(n_bars) and _is_int(max_h)):
        raise ValueError("n_obs, n_bars and max_h must be integers.")
    if not 1 <= n_obs <= n_bars:
        raise ValueError(f"need 1 <= n_obs <= n_bars, got n_obs={n_obs}, n_bars={n_bars}.")
    if max_h < 1:
        raise ValueError(f"max_h must be at least 1, got {max_h}.")
    rng = seed if isinstance(seed, np.random.Generator) else np.random.default_rng(seed)
    starts = np.sort(rng.choice(n_bars, size=n_obs, replace=False)).astype(np.int64)
    lengths = rng.integers(1, max_h + 1, size=n_obs).astype(np.int64)
    ends = np.minimum(starts + lengths, n_bars - 1)
    return pd.Series(ends, index=pd.Index(starts))


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
        Number of labels and of draws in each bootstrap sample.
    n_bars : int
        Number of bars.
    max_h : int
        Largest label length in bars.

    Returns
    -------
    dict[str, float]
        ``"std_u"``: average uniqueness of the standard bootstrap sample.
        ``"seq_u"``: average uniqueness of the sequential bootstrap sample.

    Notes
    -----
    Draw order from ``rng``, fixed by this docstring:

    1. ``random_t1(n_obs, n_bars, max_h, rng)``: start bars, then lengths.
    2. ``std_draws = rng.integers(0, n_obs, size=n_obs)``: the standard bootstrap,
       with repeats.
    3. ``uniforms = rng.random(n_obs)``: one uniform per sequential draw.

    The sequential sample uses ``uniforms`` through the same inverse-CDF rule as
    :func:`finlab.weights.sequential_bootstrap`. Both averages keep repeated
    draws. Both lie in ``(0, 1]``.
    """
    t1 = random_t1(n_obs, n_bars, max_h, rng)
    starts = t1.index.to_numpy(dtype=np.int64)
    ends = t1.to_numpy(dtype=np.int64)
    std_draws = rng.integers(0, n_obs, size=n_obs)
    uniforms = rng.random(n_obs)
    std_u, seq_u = _trial_uniqueness(n_bars, starts, ends, std_draws, uniforms)
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
        Number of labels and of draws, passed to :func:`random_t1`.
    n_bars : int, default 100
        Number of bars.
    max_h : int, default 5
        Largest label length in bars.
    n_iter : int, default 10_000
        Number of trials. This default is far below any book-scale run. The book
        text is not in this repository, so that claim is not checked here.
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
    The defaults (``n_obs``, ``n_bars``, ``max_h``, ``n_iter``) are design choices
    of this module. They are not verified against the book.
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
