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
random draws are made by numpy inside ``func``; numba is not involved.

Not covered
-----------
Convergence diagnostics, variance reduction, and distributed execution across
machines.
"""

from __future__ import annotations

import functools
from collections.abc import Mapping
from typing import Any, Callable

import numpy as np
import pandas as pd

from .parallel import mp_pandas_obj

__all__ = ["run_trials"]


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
