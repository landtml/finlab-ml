"""Tests for finlab.pbo (AFML ch. 11-12: CSCV and probability of backtest overfitting)."""

from __future__ import annotations

import itertools
import math

import numpy as np
import pandas as pd
import pytest

from finlab.pbo import PBOResult, _sharpe_of_mask, probability_of_backtest_overfitting


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
    assert np.all(res.logits > 0)  # in every split the in-sample pick ranks above the OOS median


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


# ---------------------------------------------------------------------------
# n_partitions must be an integer. Without the up-front check, 4.5 would pass as
# int(4.5) == 4 and then fail in range(4.5) with an unclear TypeError, '4' would
# fail inside T // S, and True would be reported as "not even".
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("bad", [4.5, 8.9, float("nan"), float("inf"), True, "4", None])
def test_n_partitions_must_be_integer(bad) -> None:
    M = np.random.default_rng(31).normal(size=(32, 4))
    with pytest.raises(ValueError, match="n_partitions must be an integer"):
        probability_of_backtest_overfitting(M, n_partitions=bad)


@pytest.mark.parametrize("ok", [4.0, np.int64(4), np.float64(4.0)])
def test_integral_partition_count_equals_int(ok) -> None:
    M = np.random.default_rng(32).normal(size=(32, 4))
    ref = probability_of_backtest_overfitting(M, n_partitions=4)
    res = probability_of_backtest_overfitting(M, n_partitions=ok)
    np.testing.assert_array_equal(res.logits, ref.logits)
    assert res.pbo == ref.pbo
    assert res.n_combinations == ref.n_combinations == 6


# ---------------------------------------------------------------------------
# Hand-computed cases. Input: T = 8, N = 2, S = 4, so each block has 2 rows.
# Block s has two identical rows with value x_s. A split's training pool then
# holds {x_s, x_s, x_t, x_t} with mean (x_s + x_t) / 2, sum of squared deviations
# (x_s - x_t)^2, sample sd |x_s - x_t| / sqrt(3), and Sharpe
#     SR = (sqrt(3) / 2) * g,   g = (x_s + x_t) / |x_s - x_t|.
# The common factor sqrt(3) / 2 does not change any comparison, so the table of
# g values decides every rank. With N = 2: rank in {1, 2}, omega = rank / 3,
# logit = log(1/2) = -log 2 for rank 1 and log 2 for rank 2. Splits are listed
# in itertools.combinations order; the test set of split {s, t} is the complement.
# ---------------------------------------------------------------------------
def test_hand_computed_T8_N2_S4_logits_and_pbo() -> None:
    # Hand-computed, constructed input, not a book value.
    A = np.repeat([1.0, 2.0, 3.0, 4.0], 2)
    B = np.repeat([1.0, 4.0, 2.0, 3.0], 2)
    res = probability_of_backtest_overfitting(np.column_stack([A, B]), n_partitions=4)
    # train {0,1}: g_A = 3,     g_B = 5/3 -> A wins;  test {2,3}: g_A = 7, g_B = 5   -> A top, rank 2
    # train {0,2}: g_A = 2,     g_B = 3   -> B wins;  test {1,3}: g_A = 3, g_B = 7   -> B top, rank 2
    # train {0,3}: g_A = 5/3,   g_B = 2   -> B wins;  test {1,2}: g_A = 5, g_B = 3   -> B lower, rank 1
    # train {1,2}: g_A = 5,     g_B = 3   -> A wins;  test {0,3}: g_A = 5/3, g_B = 2 -> A lower, rank 1
    # train {1,3}: g_A = 3,     g_B = 7   -> B wins;  test {0,2}: g_A = 2, g_B = 3   -> B top, rank 2
    # train {2,3}: g_A = 7,     g_B = 5   -> A wins;  test {0,1}: g_A = 3, g_B = 5/3 -> A top, rank 2
    expected = np.array(
        [math.log(2), math.log(2), -math.log(2), -math.log(2), math.log(2), math.log(2)]
    )
    assert res.n_combinations == 6
    np.testing.assert_allclose(res.logits, expected, rtol=0, atol=1e-12)
    # Two splits have logit <= 0 (the rank-1 rows), so PBO = 2/6.
    assert res.pbo == pytest.approx(1 / 3, abs=1e-12)


def test_hand_computed_out_of_sample_tie_counts_for_selected_trial() -> None:
    # Hand-computed, constructed input, not a book value.
    # Blocks of A: (1, 2, 3, 4). Blocks of B: (5, 6, 3, 4). Blocks 2 and 3 are equal in both.
    # g_B = 11, 4, 9, 3, 5, 7 for the pairs {0,1}, {0,2}, {0,3}, {1,2}, {1,3}, {2,3}.
    # train {0,1}: g_A = 3, g_B = 11 -> B wins. Test {2,3}: A = B = 7 (exact tie).
    #              Rank = #{n : test_n <= test_B} = 2 because the tie counts for the
    #              selected trial, so logit = +log 2. A strict rule would give -log 2.
    # train {0,2}: g_A = 2, g_B = 4 -> B; test {1,3}: g_A = 3, g_B = 5 -> B top, rank 2.
    # train {0,3}: g_A = 5/3, g_B = 9 -> B; test {1,2}: g_A = 5, g_B = 3 -> rank 1.
    # train {1,2}: g_A = 5, g_B = 3 -> A; test {0,3}: g_A = 5/3, g_B = 9 -> rank 1.
    # train {1,3}: g_A = 3, g_B = 5 -> B; test {0,2}: g_A = 2, g_B = 4 -> B top, rank 2.
    # train {2,3}: A = B = 7 (exact in-sample tie) -> first index, A wins;
    #              test {0,1}: g_A = 3, g_B = 11 -> B top, A lower, rank 1 -> -log 2.
    A = np.repeat([1.0, 2.0, 3.0, 4.0], 2)
    B = np.repeat([5.0, 6.0, 3.0, 4.0], 2)
    res = probability_of_backtest_overfitting(np.column_stack([A, B]), n_partitions=4)
    expected = np.array(
        [math.log(2), math.log(2), -math.log(2), -math.log(2), math.log(2), -math.log(2)]
    )
    np.testing.assert_allclose(res.logits, expected, rtol=0, atol=1e-12)
    assert res.pbo == pytest.approx(1 / 2, abs=1e-12)


# ---------------------------------------------------------------------------
# Closed-form examples on constructed inputs (not book values).
# ---------------------------------------------------------------------------
def test_closed_form_identical_columns_give_pbo_zero_constructed_input() -> None:
    """Closed form, constructed input, not a book value.

    All N identical columns give bitwise-equal Sharpe ratios in every split.
    Train ties, so the in-sample argmax is column 0 (first index). Test ties, so
    rank = N (ties count for the selected trial), omega = N/(N+1), logit = log N > 0,
    PBO = 0.
    """
    rng = np.random.default_rng(22)
    x = rng.normal(0.0, 0.01, size=(32, 1))
    N = 5
    res = probability_of_backtest_overfitting(np.hstack([x] * N), n_partitions=4)
    np.testing.assert_allclose(res.logits, math.log(N), rtol=0, atol=1e-12)
    assert res.pbo == 0.0


def test_closed_form_dominant_column_gives_pbo_zero_constructed_input() -> None:
    """Closed form, constructed input, not a book value.

    Column 0 is noise plus 1.0, the others are noise with sd 0.01. Column 0's block
    means exceed every other column's in every block (asserted below), so its Sharpe
    ratio is the largest in every split, in and out of sample. Rank = N, PBO = 0.
    """
    rng = np.random.default_rng(21)
    M = rng.normal(0.0, 0.01, size=(32, 3))
    M[:, 0] += 1.0
    block_means = M.reshape(4, 8, 3).mean(axis=1)
    # Premise of the closed form: column 0 is strictly better in every block.
    assert np.all(block_means[:, [0]] > block_means[:, 1:])
    res = probability_of_backtest_overfitting(M, n_partitions=4)
    np.testing.assert_allclose(res.logits, math.log(3), rtol=0, atol=1e-12)
    assert res.pbo == 0.0


def test_closed_form_mirror_columns_give_pbo_one_constructed_input() -> None:
    """Closed form, constructed input, not a book value.

    A construction in which the in-sample winner is the out-of-sample worst in every
    split (item 3c). Shown for N = 2, S = 4 only.

    T = 8, N = 2, S = 4. Column A has block means mu = (3, -1, -1, -1), with rows
    mu_s +/- 0.5 (so every block has nonzero variance). Column B = -A. Pooled mean of
    a pair of blocks {s, t} is (mu_s + mu_t) / 2: {0,1}, {0,2}, {0,3} give +1, and
    {1,2}, {1,3}, {2,3} give -1. Each split's test pair is the complement, so its
    pooled mean has the opposite sign. SR_B = -SR_A exactly, since negation leaves the
    sample sd unchanged. So the in-sample winner has a negative out-of-sample score
    while the other column is positive: rank 1, omega = 1/3, logit = -log 2 in all
    6 splits, PBO = 1.
    """
    mu = np.array([3.0, -1.0, -1.0, -1.0])
    a_rows = np.stack([mu + 0.5, mu - 0.5], axis=1).reshape(-1)
    M = np.column_stack([a_rows, -a_rows])
    res = probability_of_backtest_overfitting(M, n_partitions=4)
    assert res.n_combinations == 6
    np.testing.assert_allclose(res.logits, -math.log(2), rtol=0, atol=1e-12)
    assert res.pbo == 1.0


# ---------------------------------------------------------------------------
# Zero-variance columns, ties and degenerate inputs.
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("value", [0.0, 0.01, -0.01])
def test_all_equal_returns_give_finite_pbo_and_logits(value: float) -> None:
    # Every column is constant, so every Sharpe is the same by the module rule
    # (0 for value 0, +/-inf otherwise). All scores tie, so rank = N and logit = log N.
    M = np.full((32, 3), value)
    res = probability_of_backtest_overfitting(M, n_partitions=4)
    assert 0.0 <= res.pbo <= 1.0
    assert np.all(np.isfinite(res.logits))
    np.testing.assert_allclose(res.logits, math.log(3), rtol=0, atol=1e-12)
    assert res.pbo == 0.0


def test_constant_column_among_noise_is_top_in_every_split() -> None:
    # Column 0 is constant at 0.1, so its Sharpe is +inf in every split (exact zero-
    # variance rule). Noise columns have finite Sharpe ratios. Column 0 is the in-sample
    # winner and the out-of-sample top, so rank = N and logit = log N in every split.
    rng = np.random.default_rng(23)
    M = rng.normal(0.0, 0.01, size=(32, 3))
    M[:, 0] = 0.1
    res = probability_of_backtest_overfitting(M, n_partitions=4)
    assert np.all(np.isfinite(res.logits))
    np.testing.assert_allclose(res.logits, math.log(3), rtol=0, atol=1e-12)
    assert res.pbo == 0.0


@pytest.mark.parametrize("value", [0.1, 0.3, 1 / 3, 0.07, 1e-3])
def test_zero_variance_sharpe_is_exactly_inf_not_rounding_dependent(value: float) -> None:
    # Closed form. The mean of n copies of value can differ from value by an ulp, so a
    # test on the sum of squares would return a huge finite Sharpe ratio (about 7e15 for
    # 0.1). Exact equality of the selected values gives +/-inf as documented.
    mask = np.ones(4, dtype=np.bool_)
    blocks = np.full((4, 3, 1), value)
    assert _sharpe_of_mask(blocks, mask, 0) == math.inf
    assert _sharpe_of_mask(-blocks, mask, 0) == -math.inf
    assert _sharpe_of_mask(np.zeros((4, 3, 1)), mask, 0) == 0.0


def test_zero_variance_rule_uses_only_selected_blocks() -> None:
    # Blocks 0 and 1 are constant at 0.1; blocks 2 and 3 hold 5.0.
    blocks = np.full((4, 2, 1), 0.1)
    blocks[2:] = 5.0
    assert _sharpe_of_mask(blocks, np.array([True, True, False, False]), 0) == math.inf
    # Selecting blocks 0 and 2 mixes values, so the Sharpe ratio is finite and matches numpy.
    got = _sharpe_of_mask(blocks, np.array([True, False, True, False]), 0)
    x = np.array([0.1, 0.1, 5.0, 5.0])
    assert got == pytest.approx(x.mean() / x.std(ddof=1), rel=1e-12)


def test_nonconstant_sharpe_matches_hand_value() -> None:
    # Closed form. Values 1, 2, 3, 4 (one row per block): mean 2.5, sum of squared
    # deviations 5, sample sd sqrt(5/3), Sharpe 2.5 / sqrt(5/3).
    blocks = np.arange(1.0, 5.0).reshape(4, 1, 1)
    got = _sharpe_of_mask(blocks, np.ones(4, dtype=np.bool_), 0)
    assert got == pytest.approx(2.5 / math.sqrt(5 / 3), rel=1e-12)


# ---------------------------------------------------------------------------
# Invalid input: each rule has a clear message.
# ---------------------------------------------------------------------------
def _noise(T: int, N: int) -> np.ndarray:
    return np.random.default_rng(41).normal(0.0, 0.01, size=(T, N))


@pytest.mark.parametrize(
    "M, S, msg",
    [
        (np.arange(1.0, 41.0), 4, "2-D"),  # 1-D input
        (np.zeros((0, 3)), 4, "at least 2 rows"),  # zero rows
        (np.zeros((16, 0)), 4, "N >= 2"),  # zero columns
        (_noise(40, 3), 5, "even"),  # S odd
        (_noise(40, 3), 0, "even"),  # S = 0
        (_noise(40, 3), -2, "even"),  # S < 0
        (_noise(2, 3), 2, "at least 2 rows"),  # blocks of 1 row
        (_noise(6, 3), 4, "divisible"),  # T not divisible by S
    ],
)
def test_invalid_input_messages(M, S: int, msg: str) -> None:
    with pytest.raises(ValueError, match=msg):
        probability_of_backtest_overfitting(M, n_partitions=S)


def test_one_trial_column_rejected() -> None:
    with pytest.raises(ValueError, match="N >= 2"):
        probability_of_backtest_overfitting(_noise(16, 1), n_partitions=4)


@pytest.mark.parametrize("bad", [np.inf, -np.inf])
def test_infinite_entries_rejected(bad: float) -> None:
    M = _noise(16, 3)
    M[2, 1] = bad
    with pytest.raises(ValueError, match="finite"):
        probability_of_backtest_overfitting(M, n_partitions=4)
