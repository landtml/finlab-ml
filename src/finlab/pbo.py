"""Probability of backtest overfitting via combinatorially symmetric CV.

Implements AFML chapter 11 section 11.6 (Strategy selection) and the CSCV
procedure of Bailey, Borwein, Lopez de Prado and Zhu (2017), as described in
AFML 11.6 (pp. 156-157 of the book text):

* :func:`probability_of_backtest_overfitting` -- PBO and the logit distribution
  lambda_c over all C(S, S/2) train/test splits.

The T x N performance matrix is cut into S contiguous row blocks. Every choice
of S/2 blocks forms an in-sample (training) set; its complement is the
out-of-sample (testing) set. For each split the trial with the best in-sample
performance is selected, its out-of-sample relative rank w is computed, and the
logit lambda = log(w / (1 - w)) is recorded. PBO is the fraction of splits with
lambda <= 0, i.e. where the in-sample winner is at or below the out-of-sample
median.

Notes
-----
* Relative rank: ``w = rank / (N + 1)`` where ``rank`` is the number of trials
  whose out-of-sample performance is <= that of the selected trial (ties count
  in the selected trial's favour). Hence w in (0, 1) and lambda is finite.
* Default performance measure: per-column Sharpe ratio, mean / std(ddof=1), on
  the rows of each block set. A column with zero variance gets SR = 0 when its
  mean is zero and +/-inf otherwise.
* The default path is a numba kernel over all combinations. A user-supplied
  ``performance`` callable is evaluated in a Python loop, which is slower.
* The split count C(S, S/2) includes each train/test pair in both orders, as in
  the book's algorithm; S = 16 gives 12,870 splits (the book's text says 12,780,
  which is a typo).

Scope
-----
Single-metric PBO on a fixed trial matrix. No purging/embargo (CSCV splits
contiguous blocks as in the book), no stochastic subsampling of combinations.

Not covered: the performance-degradation and probability-of-loss statistics of
Bailey et al. (2017), the stochastic dominance test, and CPCV paths (see
``finlab.cv``).
"""

from __future__ import annotations

import itertools
import math
from dataclasses import dataclass
from typing import Callable, Optional, Union

import numpy as np
import pandas as pd

from finlab._jit import jit

__all__ = ["PBOResult", "probability_of_backtest_overfitting"]

PerformanceFn = Callable[[np.ndarray], np.ndarray]


@dataclass(frozen=True, eq=False)
class PBOResult:
    """Result of :func:`probability_of_backtest_overfitting`.

    Attributes
    ----------
    pbo : float
        Fraction of splits with logit <= 0, in [0, 1].
    logits : numpy.ndarray
        Logit lambda_c for each of the C(S, S/2) splits, in enumeration order
        of ``itertools.combinations(range(S), S // 2)``.
    n_combinations : int
        Number of splits, C(S, S/2).
    """

    pbo: float
    logits: np.ndarray
    n_combinations: int


def probability_of_backtest_overfitting(
    performance_matrix: Union[pd.DataFrame, np.ndarray],
    n_partitions: int = 16,
    performance: Optional[PerformanceFn] = None,
) -> PBOResult:
    """Estimate the probability of backtest overfitting by CSCV.

    Parameters
    ----------
    performance_matrix : DataFrame or ndarray, shape (T, N)
        Rows are synchronous time observations (e.g. periodic PnL), columns are
        the N trial strategies. Must be finite. A DataFrame's columns are only
        used for shape; values are read positionally.
    n_partitions : int, default 16
        Number S of contiguous row blocks. Must be even, >= 2, divide T, and
        satisfy T / S >= 2.
    performance : callable, optional
        ``performance(sub)`` takes a (rows, N) ndarray and returns a length-N
        array of trial scores (higher is better). Defaults to the per-column
        Sharpe ratio, which uses the numba kernel.

    Returns
    -------
    PBOResult
        ``pbo``, the logit array and the number of splits.

    Raises
    ------
    ValueError
        On a non-2-D or non-finite matrix, N < 2, S odd or < 2, T not divisible
        by S, or blocks shorter than 2 rows.

    Notes
    -----
    Logits are lambda_c = log(w_c / (1 - w_c)); PBO = mean(lambda_c <= 0).
    Values near 0.5 mean the in-sample winner is out-of-sample median in about
    half of the splits, which is what pure-noise trials produce (see
    ``docs/proofs/pbo.md``).
    """
    mat = _validate(performance_matrix, n_partitions)
    T, N = mat.shape
    S = n_partitions
    L = T // S
    k = S // 2
    combos = np.array(list(itertools.combinations(range(S), k)), dtype=np.int64)
    blocks = np.ascontiguousarray(mat.reshape(S, L, N))

    if performance is None:
        logits = _cscv_sharpe_logits(blocks, combos)
    else:
        logits = _cscv_callable_logits(blocks, combos, performance)

    pbo = float(np.mean(logits <= 0.0))
    return PBOResult(pbo=pbo, logits=logits, n_combinations=int(combos.shape[0]))


def _validate(
    performance_matrix: Union[pd.DataFrame, np.ndarray], n_partitions: int
) -> np.ndarray:
    """Coerce to a finite float64 (T, N) array and check the CSCV constraints."""
    if isinstance(performance_matrix, pd.DataFrame):
        arr = performance_matrix.to_numpy(dtype=np.float64)
    else:
        arr = np.asarray(performance_matrix, dtype=np.float64)
    if arr.ndim != 2:
        raise ValueError("performance_matrix must be 2-D (T x N)")
    if not np.all(np.isfinite(arr)):
        raise ValueError("performance_matrix must contain only finite values")
    T, N = arr.shape
    S = int(n_partitions)
    if N < 2:
        raise ValueError("need at least N >= 2 trial columns")
    if S < 2 or S % 2 != 0:
        raise ValueError("n_partitions must be an even integer >= 2")
    if T % S != 0:
        raise ValueError(f"T={T} must be divisible by n_partitions={S}")
    if T // S < 2:
        raise ValueError("each block must contain at least 2 rows")
    return arr


@jit
def _sharpe_of_mask(blocks: np.ndarray, mask: np.ndarray, col: int) -> float:
    """Sharpe ratio of column ``col`` over the blocks flagged in ``mask``."""
    S, L, _ = blocks.shape
    cnt = 0
    total = 0.0
    for s in range(S):
        if mask[s]:
            for t in range(L):
                total += blocks[s, t, col]
                cnt += 1
    mu = total / cnt
    ss = 0.0
    for s in range(S):
        if mask[s]:
            for t in range(L):
                d = blocks[s, t, col] - mu
                ss += d * d
    sd = math.sqrt(ss / (cnt - 1))
    if sd == 0.0:
        if mu == 0.0:
            return 0.0
        return math.inf if mu > 0.0 else -math.inf
    return mu / sd


@jit
def _cscv_sharpe_logits(blocks: np.ndarray, combos: np.ndarray) -> np.ndarray:
    """Logit lambda_c for every split, with Sharpe ratio as the performance."""
    S, _, N = blocks.shape
    C = combos.shape[0]
    logits = np.empty(C, dtype=np.float64)
    in_train = np.zeros(S, dtype=np.bool_)
    out_test = np.zeros(S, dtype=np.bool_)
    train_sr = np.empty(N, dtype=np.float64)
    test_sr = np.empty(N, dtype=np.float64)
    for c in range(C):
        for s in range(S):
            in_train[s] = False
        for i in range(combos.shape[1]):
            in_train[combos[c, i]] = True
        for s in range(S):
            out_test[s] = not in_train[s]
        for n in range(N):
            train_sr[n] = _sharpe_of_mask(blocks, in_train, n)
            test_sr[n] = _sharpe_of_mask(blocks, out_test, n)
        best = 0
        for n in range(1, N):
            if train_sr[n] > train_sr[best]:
                best = n
        rank = 0
        for n in range(N):
            if test_sr[n] <= test_sr[best]:
                rank += 1
        omega = rank / (N + 1.0)
        logits[c] = math.log(omega / (1.0 - omega))
    return logits


def _cscv_callable_logits(
    blocks: np.ndarray, combos: np.ndarray, performance: PerformanceFn
) -> np.ndarray:
    """Logit lambda_c for every split, using a user performance callable."""
    S, L, N = blocks.shape
    logits = np.empty(combos.shape[0], dtype=np.float64)
    for c, combo in enumerate(combos):
        in_train = set(int(s) for s in combo)
        train_idx = sorted(in_train)
        test_idx = [s for s in range(S) if s not in in_train]
        train = blocks[train_idx].reshape(-1, N)
        test = blocks[test_idx].reshape(-1, N)
        train_sr = np.asarray(performance(train), dtype=np.float64).ravel()
        test_sr = np.asarray(performance(test), dtype=np.float64).ravel()
        if train_sr.shape != (N,) or test_sr.shape != (N,):
            raise ValueError("performance must return one score per column")
        best = int(np.argmax(train_sr))
        rank = int(np.sum(test_sr <= test_sr[best]))
        omega = rank / (N + 1.0)
        logits[c] = math.log(omega / (1.0 - omega))
    return logits
