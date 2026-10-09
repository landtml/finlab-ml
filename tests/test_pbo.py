"""Tests for finlab.pbo (AFML ch. 11-12: CSCV and probability of backtest overfitting)."""

from __future__ import annotations

import itertools
import math

import numpy as np
import pandas as pd
import pytest

from finlab.pbo import PBOResult, probability_of_backtest_overfitting


# ---------------------------------------------------------------------------
# Brute-force reference: literal transcription of the book's CSCV steps 1-7.
# Deliberately independent of the numba kernel (no shared helpers).
# ---------------------------------------------------------------------------
def naive_cscv_logits(M: np.ndarray, S: int) -> np.ndarray:
    T, N = M.shape
    L = T // S
    blocks = [M[s * L : (s + 1) * L] for s in range(S)]
    logits = []
    for J in itertools.combinations(range(S), S // 2):
        Jbar = [s for s in range(S) if s not in J]
        train = np.vstack([blocks[s] for s in J])
        test = np.vstack([blocks[s] for s in Jbar])
        R = [train[:, n].mean() / train[:, n].std(ddof=1) for n in range(N)]
        n_star = int(np.argmax(R))
        Rbar = [test[:, n].mean() / test[:, n].std(ddof=1) for n in range(N)]
        rank = sum(1 for n in range(N) if Rbar[n] <= Rbar[n_star])
        omega = rank / (N + 1)
        logits.append(math.log(omega / (1 - omega)))
    return np.array(logits)


def naive_pbo(M: np.ndarray, S: int) -> float:
    lam = naive_cscv_logits(M, S)
    return sum(1 for x in lam if x <= 0) / len(lam)


@pytest.mark.parametrize(
    "seed, T, N, S",
    [(1, 48, 2, 4), (2, 64, 5, 8), (3, 96, 7, 6), (4, 120, 10, 4), (5, 80, 3, 8)],
)
def test_fast_cscv_matches_naive_loop(seed: int, T: int, N: int, S: int) -> None:
    rng = np.random.default_rng(seed)
    M = rng.normal(0.0005, 0.01, size=(T, N))
    fast = probability_of_backtest_overfitting(M, n_partitions=S)
    slow = naive_cscv_logits(M, S)
    assert fast.n_combinations == math.comb(S, S // 2) == len(slow)
    np.testing.assert_allclose(fast.logits, slow, rtol=1e-10, atol=1e-12)
    assert fast.pbo == naive_pbo(M, S)


def test_callable_path_agrees_with_default_sharpe() -> None:
    rng = np.random.default_rng(11)
    M = rng.normal(0.0, 1.0, size=(64, 6))

    def sharpe(sub: np.ndarray) -> np.ndarray:
        return sub.mean(axis=0) / sub.std(axis=0, ddof=1)

    fast = probability_of_backtest_overfitting(M, n_partitions=8)
    generic = probability_of_backtest_overfitting(M, n_partitions=8, performance=sharpe)
    np.testing.assert_allclose(generic.logits, fast.logits, rtol=1e-10, atol=1e-12)
    assert generic.pbo == fast.pbo


def test_dataframe_input_equals_ndarray() -> None:
    rng = np.random.default_rng(12)
    M = rng.normal(size=(32, 4))
    df = pd.DataFrame(M, columns=["a", "b", "c", "d"])
    a = probability_of_backtest_overfitting(df, n_partitions=4)
    b = probability_of_backtest_overfitting(M, n_partitions=4)
    assert isinstance(a, PBOResult)
    np.testing.assert_array_equal(a.logits, b.logits)
    assert a.pbo == b.pbo


def test_pure_noise_many_trials_gives_pbo_near_half() -> None:
    # Under pure noise the IS winner's OOS rank is uniform (exchangeability of
    # columns), so E[PBO] = 1/2, not 1. A single realisation is noisy because
    # the 70 splits share blocks (sd ~ 0.16 across seeds), so average 30 seeds.
    pbos = []
    for seed in range(30):
        rng = np.random.default_rng(2024 + seed)
        M = rng.normal(0.0, 0.01, size=(320, 60))
        res = probability_of_backtest_overfitting(M, n_partitions=8)
        assert res.n_combinations == 70
        pbos.append(res.pbo)
    assert 0.4 <= float(np.mean(pbos)) <= 0.6


def test_true_edge_gives_pbo_near_zero() -> None:
    rng = np.random.default_rng(7)
    M = rng.normal(0.0, 0.01, size=(320, 30))
    M[:, 0] += 0.02  # one trial with a genuine, persistent edge
    res = probability_of_backtest_overfitting(M, n_partitions=8)
    assert res.pbo < 0.05
    assert np.all(res.logits[res.logits > 0] > 0)


def test_logits_bounded_by_rank_range() -> None:
    rng = np.random.default_rng(3)
    N = 9
    M = rng.normal(size=(60, N))
    res = probability_of_backtest_overfitting(M, n_partitions=6)
    lo, hi = math.log(1 / N), math.log(N / 1)
    assert np.all(res.logits >= lo - 1e-12) and np.all(res.logits <= hi + 1e-12)
    assert res.pbo == pytest.approx(float(np.mean(res.logits <= 0)), abs=0)


@pytest.mark.parametrize(
    "shape, S, msg",
    [
        ((40, 1), 4, "N >= 2"),
        ((40, 3), 3, "even"),
        ((40, 3), 1, "even"),
        ((42, 3), 4, "divisible"),
        ((4, 3), 4, "at least 2 rows"),
    ],
)
def test_validation_errors(shape, S: int, msg: str) -> None:
    M = np.zeros(shape)
    M[0, 0] = 1.0  # avoid all-zero degenerate matrix
    with pytest.raises(ValueError, match=msg):
        probability_of_backtest_overfitting(M, n_partitions=S)


def test_non_finite_rejected() -> None:
    M = np.ones((16, 3))
    M[2, 1] = np.nan
    with pytest.raises(ValueError, match="finite"):
        probability_of_backtest_overfitting(M, n_partitions=4)
