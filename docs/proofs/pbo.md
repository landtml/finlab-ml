# Proofs and derivations: `finlab.pbo` (AFML ch. 11 section 11.6, CSCV)

Status labels:

- **Proved here**: derived in this document.
- **Checked by test**: asserted in `tests/test_pbo.py`, or measured by simulation (stated as such).
- **Claimed from the book only**: stated in AFML or Bailey et al. (2017), not re-derived.

## 1. CSCV construction

Input: a T x N matrix M, S even, S | T. Blocks M_1..M_S of L = T/S rows.
For each k-subset J of {1..S} with k = S/2, the training set is the union of
blocks in J and the test set is the complement J-bar. There are
C(S, S/2) such splits. The procedure follows AFML 11.6 steps 1-7.

**Proved here (count).** The number of splits is C(S, S/2) by definition of
the binomial coefficient. For S = 16, C(16, 8) = 12,870
(`tests/test_doc_claims.py::test_pbo_split_count_s16`). The book's text gives
12,780 in the surrounding paragraph. That is a claim about the book's text and
is not checked here.

## 2. Relative rank and logit

For split c, let n* = argmax_n R_n (in-sample Sharpe, first index on ties) and
let rank_c = #{n : Rbar_n <= Rbar_{n*}} (out-of-sample), so rank_c in {1, ..., N}.
Define omega_c = rank_c / (N + 1) and lambda_c = log(omega_c / (1 - omega_c)).

**Proved here.**
1. omega_c in [1/(N+1), N/(N+1)] subset (0, 1), so lambda_c is finite and
   lambda_c in [log(1/N), log N].
2. lambda_c <= 0 iff omega_c <= 1/2 iff rank_c <= (N+1)/2. So lambda_c <= 0 means
   the in-sample winner ranks at or below the out-of-sample median.
3. lambda_c = 0 iff rank_c = (N+1)/2, which is the median in the book's sense (AFML 11.6 step 7).

**Checked by test.** The logit range bound in item 1 (`test_logits_bounded_by_rank_range`).

PBO = (1/C(S,S/2)) * #{c : lambda_c <= 0}. This is the book's
PBO = integral_{-inf}^{0} f(lambda) d lambda, estimated by the relative frequency
over splits.

## 3. Pure noise gives E[PBO] = 1/2 (not 1)

The book's remarks about PBO are about overfit selection. For pure noise the
selected trial is unrelated to its out-of-sample performance, so the expected
PBO is 1/2 when N is even. It is derived here and is not attributed to the
book.

**Proved here.** Assume the N columns are independent, identically distributed
across trials (pure noise, same distribution in each column), with continuous
distribution so ties have probability zero. Fix split c. The training blocks
determine n*. The test-set scores (Rbar_1, ..., Rbar_N) depend only on the test
blocks, which are disjoint from the training blocks. Because the blocks are
independent, the test vector is independent of the training data.
Conditional on the training data, the test scores are exchangeable across
columns, and n* is fixed. So rank_c is uniform on {1, ..., N}, and

    P(lambda_c <= 0) = P(rank_c <= (N+1)/2) = floor((N+1)/2) / N.

For even N this is exactly 1/2. For odd N it is (N+1)/(2N), slightly above 1/2.
Taking expectations over the data gives E[PBO] = P(lambda_c <= 0) by linearity.

**Realised values are noisy.** The C(S, S/2) splits share blocks, so their
logits are dependent, and the realised PBO for one matrix has a large spread.
**Checked by test** (`tests/test_doc_claims.py::test_pbo_pure_noise_simulation_seeds_0_to_59`):
T = 320, S = 8, normal returns with mean 0 and sd 0.01, seeds 0 to 59 (`numpy.random.default_rng(seed)`).
For N = 60: mean PBO = 0.5029, sd = 0.1566 (population sd, `ddof=0`), 5th and 95th
percentiles 0.255 and 0.745. For N = 10 the sd is 0.2355; for N = 2 the mean is 0.4686
and the sd is 0.2978. These are simulation results, not proofs.

**Checked by test.** `test_pure_noise_many_trials_gives_pbo_near_half` averages
PBO over 30 seeds and asserts the mean is in [0.4, 0.6].

## 4. One real edge gives PBO near 0

**Sketch only (not a proof).** Suppose column 1 has a persistent positive mean
mu > 0 in every block and the other columns are pure noise with variance
sigma^2 small relative to mu. With high probability the in-sample winner is
column 1 in every split, and its out-of-sample score is the top score, so
rank_c = N and lambda_c = log N > 0. Then PBO -> 0 as the signal-to-noise
ratio grows. The formal bound is not derived here.

**Checked by test.** `test_true_edge_gives_pbo_near_zero` (N = 30, S = 8,
edge 0.02 against noise sd 0.01): PBO < 0.05.

## 5. Implementation equivalence

The numba kernel `_cscv_sharpe_logits` and the naive reference in the test file
compute the same quantity: the same split enumeration (`itertools.combinations`
order), the same first-index argmax, and the same `<=` rank rule. They differ
only in floating-point summation order (numba uses a two-pass sequential sum;
numpy uses pairwise summation).

**Checked by test.** Logits agree to rtol 1e-10 and atol 1e-12, and PBO matches
exactly, on random matrices with (T, N, S) in {(48,2,4), (64,5,8), (96,7,6),
(120,10,4), (80,3,8)}. `test_fast_cscv_matches_naive_loop`. The callable path
(`performance=` with a Sharpe function) matches the default path too.

## 6. Limitations

- The rank uses ties in the selected trial's favour (`<=`). With continuous
  returns ties have probability zero.
- Sharpe is undefined for zero-variance columns; the kernel returns 0 or +/-inf by
  the rule in the module docstring, which is a convention and not from the book.
- The split count is exponential in S; S = 16 gives 12,870 splits, which is the
  practical upper limit used in the benchmark.

## Summary

| Statement | Status |
|---|---|
| Number of splits C(S, S/2); C(16, 8) = 12,870 (the book's 12,780 is a book claim, not checked) | Proved here |
| omega in (0,1), lambda finite, lambda <= 0 iff rank <= (N+1)/2 | Proved here |
| Pure noise gives P(lambda <= 0) = floor((N+1)/2)/N, so E[PBO] = 1/2 for even N | Proved here |
| Pure-noise PBO is near 1 | Not supported; contradicted by the proof above |
| Pure-noise PBO mean, sd and percentiles at N = 60, 10, 2 (seeds 0 to 59) | Checked by test (`test_pbo_pure_noise_simulation_seeds_0_to_59`) |
| One real edge gives PBO near 0 | Checked by test; bound sketched only |
| Fast kernel equals naive loop | Checked by test |
| CSCV procedure and PBO definition | Claimed from book (AFML 11.6; Bailey et al. 2017) |
