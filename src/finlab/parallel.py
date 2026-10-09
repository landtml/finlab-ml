"""Multiprocessing and partitioning helpers (AFML ch. 20).

``mp_pandas_obj`` applies a function to molecules (contiguous chunks) of a pandas
index, optionally in parallel with a process pool, and concatenates the results.
Two partition schemes from the book balance work across workers:

* :func:`lin_parts` -- equal-size chunks (use when the cost is linear in the
  number of atoms).
* :func:`nested_parts` -- chunks sized so that work growing quadratically (pairs
  of atoms) is balanced, as in the book's snippet 20.6.

Scope
-----
Uses :mod:`concurrent.futures` from the standard library, so there is no joblib
dependency. The function passed in must be picklable (defined at module level)
when ``num_threads > 1``.

Not covered
-----------
Distributed execution across machines, and the book's "vectorization over
loops" examples, which are a matter of how the user writes ``func``.
"""

from __future__ import annotations

import os
from concurrent.futures import ProcessPoolExecutor
from typing import Any, Callable

import numpy as np
import numpy.typing as npt
import pandas as pd

__all__ = ["lin_parts", "nested_parts", "mp_pandas_obj"]


def lin_parts(num_atoms: int, num_threads: int) -> npt.NDArray[np.int64]:
    """Boundary positions for ``num_threads`` equal-size chunks (snippet 20.5).

    Returns an array ``parts`` of length ``min(num_threads, num_atoms) + 1`` with
    ``parts[0] == 0`` and ``parts[-1] == num_atoms``; chunk ``k`` is
    ``[parts[k], parts[k + 1])``.
    """
    if num_atoms < 0 or num_threads < 1:
        raise ValueError("num_atoms must be >= 0 and num_threads >= 1.")
    n = min(num_threads, max(num_atoms, 1))
    parts = np.ceil(np.linspace(0, num_atoms, n + 1)).astype(np.int64)
    return parts


def nested_parts(
    num_atoms: int, num_threads: int, upper_triangle: bool = False
) -> npt.NDArray[np.int64]:
    """Chunk boundaries for work that grows quadratically with the atom index.

    Implements snippet 20.6: the first chunks are kept small, so that each chunk
    touches roughly the same number of pairs. With ``upper_triangle=True`` the
    chunks are balanced for the upper triangle of a pairwise matrix (cost of atom
    ``i`` proportional to ``num_atoms - i``).
    """
    if num_atoms < 0 or num_threads < 1:
        raise ValueError("num_atoms must be >= 0 and num_threads >= 1.")
    parts = [0]
    n_threads = min(num_threads, max(num_atoms, 1))
    if num_atoms == 0:
        return np.array([0], dtype=np.int64)
    if upper_triangle:
        # Balance pairs: chunk boundary at the point where the cumulative pair
        # count reaches the k-th fraction of the total.
        weights = np.arange(num_atoms, 0, -1, dtype=np.float64)
    else:
        weights = np.arange(1, num_atoms + 1, dtype=np.float64)
    cum = np.cumsum(weights)
    total = cum[-1]
    for k in range(1, n_threads):
        target = total * k / n_threads
        parts.append(int(np.searchsorted(cum, target, side="left")) + 1)
    parts.append(num_atoms)
    out = np.unique(np.clip(np.array(parts, dtype=np.int64), 0, num_atoms))
    return out


def _run_molecule(func: Callable[..., Any], molecule: pd.Index, kwargs: dict[str, Any]):
    return func(molecule=molecule, **kwargs)


def mp_pandas_obj(
    func: Callable[..., Any],
    index: pd.Index,
    num_threads: int = 1,
    linear: bool = True,
    **kwargs: Any,
) -> pd.Series | pd.DataFrame:
    """Apply ``func`` to molecules of ``index`` and concatenate the results.

    Parameters
    ----------
    func : callable
        Called as ``func(molecule=<pd.Index chunk>, **kwargs)``. It must return a
        pandas object indexed by (a subset of) its molecule. Must be picklable.
    index : pd.Index
        The atoms to split (e.g. event start times).
    num_threads : int, default 1
        Worker processes. ``1`` runs serially in-process (no pickling needed).
        ``0`` or negative means ``os.cpu_count()``.
    linear : bool, default True
        Use :func:`lin_parts`; otherwise :func:`nested_parts`.
    **kwargs
        Passed through to ``func``.

    Returns
    -------
    pd.Series or pd.DataFrame
        The concatenated results, in molecule order.
    """
    if num_threads <= 0:
        num_threads = os.cpu_count() or 1
    parts = lin_parts(len(index), num_threads) if linear else nested_parts(len(index), num_threads)
    molecules = [index[parts[k] : parts[k + 1]] for k in range(len(parts) - 1)]
    molecules = [m for m in molecules if len(m) > 0]
    if not molecules:
        return func(molecule=index[:0], **kwargs)
    if num_threads == 1 or len(molecules) == 1:
        pieces = [_run_molecule(func, m, kwargs) for m in molecules]
    else:
        with ProcessPoolExecutor(max_workers=num_threads) as pool:
            futures = [pool.submit(_run_molecule, func, m, kwargs) for m in molecules]
            pieces = [f.result() for f in futures]
    return pd.concat(pieces)
