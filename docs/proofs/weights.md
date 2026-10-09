# Proofs and derivations: `finlab.weights`

Source: AFML Chapter 4 (Sample Weights), Snippets 4.1 to 4.11.

Each claim is tagged as one of:

* **Proved here**: a proof is given in this document.
* **Checked by test**: verified by `tests/test_weights.py` against a brute-force reference.
* **Claimed from the book only**: stated in AFML and used as given.

## Notation

Bars are `t = 1..T` with sorted times. Label `i` has start bar `s_i` and end
bar `e_i >= s_i` (its lifespan is `L_i = {s_i, ..., e_i}`). The indicator is
`1_{t,i} = 1[t in L_i]`, and the number of labels is `I`.

## 1. Concurrency (Snippet 4.1)

**Definition.** `c_t = sum_i 1_{t,i}`.

**Proposition 1.1 (difference-array sweep).** Let `D` be the array with
`D[s_i] += 1` and `D[e_i + 1] -= 1` for every label. Then `c_t = sum_{k <= t} D[k]`.

*Proof.* `sum_{k <= t} D[k] = #{i : s_i <= t} - #{i : e_i + 1 <= t}`, which equals
`#{i : s_i <= t} - #{i : e_i < t}`. Since `e_i < t` implies `s_i <= e_i < t`, the
second set is a subset of the first. Their difference is
`#{i : s_i <= t, e_i >= t} = c_t`. *(Proved here.)*

The sweep does O(T + I) work, with no Python loop over labels. `_concurrency_sweep`
implements it. Label start and end times are mapped to bar positions by
`searchsorted`, so label times need not be bar times.

* **Checked by test.** `test_num_co_events_matches_naive_double_loop` (4 seeds)
  compares with an explicit double loop over bars and labels, using timestamp
  comparisons. `test_num_co_events_open_labels_run_to_last_bar` checks NaT ends.

## 2. Average uniqueness (Section 4.4, Snippet 4.2)

**Definition.** `u_{t,i} = 1_{t,i} / c_t`, and
`ū_i = (sum_t u_{t,i}) / (sum_t 1_{t,i})`, the mean over the lifespan.

**Proposition 2.1 (bounds).** `0 < ū_i <= 1`. Also `ū_i = 1` if and only if
`c_t = 1` for every `t` in `L_i` (no other label overlaps label `i`).

*Proof.* On `L_i`, label `i` itself is active, so `c_t >= 1`. Hence each
`u_{t,i} = 1/c_t` lies in `(0, 1]`, and so does their mean. A mean of numbers in
`(0, 1]` equals 1 exactly when every number equals 1. *(Proved here.)*

**Proposition 2.2 (harmonic form).** `ū_i = 1 / H_i`, where `H_i` is the harmonic
mean of `c_t` over `L_i`.

*Proof.* `H_i = |L_i| / sum_{t in L_i} (1/c_t)`, so `1/H_i = (1/|L_i|) sum_{t in L_i} 1/c_t = ū_i`.
*(Proved here. Matches the book's remark in Section 4.4.)*

* **Checked by test.** `test_average_uniqueness_is_one_without_overlap` (disjoint
  labels give exactly 1), `test_average_uniqueness_matches_naive` (3 seeds), and
  `test_average_uniqueness_is_reciprocal_harmonic_mean_of_concurrency`.

## 2a. Snippet 4.4: sample average uniqueness

**Definition.** Let `d_1, ..., d_n` be a bootstrap sample of label positions, with
repeats. Concurrency is taken over the sample, `c_t = sum_k 1_{t, d_k}`, so a label
drawn twice covers each of its bars twice. For a drawn label `d`,
`ū_d = (1/|L_d|) sum_{t in L_d} 1/c_t`. The sample statistic is
`S = (1/n) sum_k ū_{d_k}`. The function `sample_average_uniqueness` computes `S`.
The definition follows the task specification. The book's text was not consulted.

**Proposition 2a.1 (range).** `0 < S <= 1`, provided every drawn label covers at
least one bar.

*Proof.* Each drawn label `d_k` is active on its own span, so `c_t >= 1` there. Each
`ū_{d_k}` is a mean of values `1/c_t` in `(0, 1]`, hence lies in `(0, 1]`. So does
their mean `S`. *(Proved here.)* A drawn label with an empty span would give a NaN
mean, and the function raises for that case instead.

**Proposition 2a.2 (single draw).** A sample of size 1 gives `S = 1`.

*Proof.* With `n = 1`, `c_t = 1` on the span of the one drawn label, so every term
`1/c_t` equals 1 and `S = 1`. *(Proved here.)*

**Proposition 2a.3 (identical spans).** If all `n` draws have the same span `L` with
`|L| = ell`, whether or not they are distinct, then `S = 1/n`.

*Proof.* On `L`, every draw covers every bar, so `c_t = n` for `t in L`. Each drawn
label has `ū = (1/ell) * ell * (1/n) = 1/n`. The mean over `n` equal values is
`1/n`. *(Proved here.)*

**Proposition 2a.4 (full sample).** With every label drawn once, `c_t` equals
`num_co_events` and `S` equals `average_uniqueness(...).mean()`.

*Proof.* Each label appears once, so `sum_k 1_{t, d_k}` is the sum of all indicator
columns, which is `c_t` from Snippet 4.1. The per-label means are the values of
Snippet 4.2 and the mean over labels is `S`. *(Proved here.)*

**Proposition 2a.5 (order invariance).** `S` depends only on the multiset of draws.

*Proof.* `c_t` is a sum over the draws, and `S` is an average over them. Both are
unchanged by a permutation of the draws. *(Proved here.)*

* **Checked by test.** `test_sample_average_uniqueness_matches_naive_with_repeats`
  (5 seeds, 40 draws from 20 labels, repeats present) compares with the naive
  reference `sample_avg_uniqueness` at `rtol = 1e-12`. The same test checks the
  range `(0, 1]`.
* **Checked by test.** `test_sample_average_uniqueness_full_sample_equals_average_uniqueness_mean`
  checks Proposition 2a.4 numerically. `test_sample_average_uniqueness_single_label_is_one`
  checks Proposition 2a.2 and the repeat case `[2, 2, 2]`, which gives `1/3`.
* **Checked by test.** `test_sample_average_uniqueness_identical_spans_give_one_over_n`
  checks Proposition 2a.3 for sample sizes 1, 3 and 8, with repeats.
* **Checked by test (hand-computed value).**
  `test_sample_average_uniqueness_hand_computed_repeat_case`. Labels cover bars
  `{0, 1}` and `{1, 2}`. The sample `[0, 0, 1]` has `c = [2, 3, 1]` and gives
  `S = 1/2`. The sample `[0, 1]` gives `3/4`.
* **Checked by test.** `test_sample_average_uniqueness_does_not_depend_on_draw_order`
  (Proposition 2a.5). `test_sample_average_uniqueness_dataframe_and_array_agree`
  checks that DataFrame and ndarray inputs match. `test_sample_average_uniqueness_rejects_bad_input`
  checks the `ValueError` cases: non-2-D input, out-of-range or negative positions,
  empty or non-1-D or non-integer draws, and a drawn label with no bar.
* **Not verified.** Whether this matches the listing of Snippet 4.4 in the book.
  The book's text was not consulted, so no book figure or wording is reproduced.

## 3. Indicator matrix (Snippet 4.3)

`indicator_matrix` returns the `T x I` matrix with entries `1_{t,i}`. It is
checked by `test_indicator_matrix_matches_naive`. The book's orientation (bars as
rows, labels as columns) is kept, and the bootstrap draws columns.

## 4. Sequential bootstrap (Section 4.5, Snippet 4.5)

Let `phi^{(k)}` be the `k` labels drawn so far (with repeats), and let `c^{(k)}_t`
be the count of draws in `phi^{(k)}` that cover bar `t`. The uniqueness of a
candidate `j` given the draws is `u^{(k)}_{t,j} = 1_{t,j} / (1 + c^{(k)}_t)`, and
its average over `L_j` is

    ū^{(k)}_j = (1 / |L_j|) sum_{t in L_j} 1 / (1 + c^{(k)}_t).

The `(k+1)`-th draw picks `j` with probability `delta^{(k)}_j = ū^{(k)}_j / sum_l ū^{(k)}_l`.

**Proposition 4.1 (valid distribution and inverse-CDF sampling).** When
`sum_l ū^{(k)}_l > 0`, `delta^{(k)}` is a probability vector. With `C_j = sum_{l<=j} ū_l`
and `u ~ U[0,1)`, the rule "pick the smallest `j` with `C_j > u * total`" yields
`P(pick = j) = ū_j / total`.

*Proof.* The weights are nonnegative and sum to `total`, so `delta` sums to 1. The
event `pick = j` is `C_{j-1} <= u total < C_j`. Its probability is
`(C_j - C_{j-1}) / total = ū_j / total`. *(Proved here.)* The kernel's
`last_pos` fallback is reached only when `u * total` equals or exceeds `C_J` by
rounding, and it picks the last label with positive weight. *(Proved here.)*

**Proposition 4.2 (monotone decrease of overlap weights).** For every `j`,
`ū^{(k+1)}_j <= ū^{(k)}_j`. Hence the uniqueness of any candidate never increases
as the sample grows.

*Proof.* Each `c^{(k)}_t` is nondecreasing in `k`, so each term `1/(1 + c_t)` is
nonincreasing. The average of nonincreasing terms over the same set is
nonincreasing. *(Proved here.)* This gives the "decreasingly likely overlap"
property of the book in its weakest form. It does not prove a rate.

**Kernel and reference.** The numba kernel `_seq_bootstrap_kernel` stores the
matrix in CSC form. It sums each `ū_j` in ascending `t` order and uses the same
inverse-CDF rule. A dense Python reference with the same uniforms produces
identical draws, since the floating-point operations are performed in the same
order.

* **Checked by test.** `test_sequential_bootstrap_matches_naive_with_same_uniforms`
  (4 seeds) checks exact equality of draws. `test_sequential_bootstrap_is_reproducible_and_in_range`
  checks seeding.
* **Checked by test.** `test_sequential_bootstrap_has_higher_average_uniqueness_than_standard`
  (40 seeds, 200 bars, 60 labels with spans up to 30): mean sample uniqueness
  of the sequential draws exceeds that of the standard bootstrap.
* **Measured, not a proof.** With 200 seeds on the same design the means were
  0.1963 (sequential) and 0.1913 (standard), with standard deviations of about
  0.013 and 0.015. The gain is real in this setting but small (about 2.6%
  relative). The book's own figures are not reproduced here.
* **Claimed from the book only.** That the sequential sample is "much closer to
  IID" than the standard bootstrap, and the Monte Carlo results of Snippets
  4.7-4.9 (not reproduced; `benchmarks/bench_weights.py` measures speed only).

## 5. Return attribution (Snippet 4.10)

Let `r_t = log p_t - log p_{t-1}` (0 at the first bar, as in the book's NaN skip).
Define

    w~_i = | sum_{t in L_i} r_t / c_t |,      w_i = w~_i * I / sum_j w~_j.

**Proposition 5.1 (normalisation).** `sum_i w_i = I`, provided `sum_j w~_j > 0`.

*Proof.* `sum_i w_i = (I / sum_j w~_j) sum_i w~_i = I`. *(Proved here.)*
The function raises when the sum is 0 (weights would be undefined).

**Remark (additivity).** Without the `1/c_t` split, `sum_{t in L_i} r_t` is the
log return of the label's lifespan. The split assigns each bar's return equally
among the labels active on that bar. *(Interpretation; not a theorem.)*

* **Checked by test.** `test_sample_weight_by_return_matches_naive_and_sums_to_n`
  (3 seeds) compares with a per-label loop, and checks the sum.

## 6. Time decay (Section 4.7, Snippet 4.11)

Let `C_i = sum_{j <= i} w_j` over the chronological order, `T = C_I`, and let
`d_i = max(0, a + b C_i)`. The boundary conditions are

1. `d` at the newest observation is 1: `a + b T = 1`, so `a = 1 - bT`.
2. The contingency on `c_last` (the weight of the oldest observation):
   * `c_last in [0, 1]`: `b = (1 - c_last) / T`, which gives `a = c_last`.
   * `c_last in (-1, 0)`: `b = 1 / ((1 + c_last) T)`, which gives
     `a = c_last / (1 + c_last)` (negative), so the factor is clipped to 0 for the
     oldest part of the sample.

**Proposition 6.1 (boundary conditions hold).** The formulas above give `d_I = 1`.

*Proof.* For `c >= 0`: `a + bT = c + (1 - c) = 1`. For `c in (-1, 0)`:
`a + bT = c/(1+c) + 1/(1+c) = 1`. *(Proved here.)*

**Proposition 6.2 (the oldest weight).** For `c_last in [0, 1]`, the oldest
observation has `C_1 = w_1`, so its factor is

    d_1 = c_last + (1 - c_last) w_1 / T.

It equals `c_last` only in the limit `w_1 / T -> 0`.

*Proof.* Direct substitution of `C_1 = w_1` into `d = a + b C`. *(Proved here.)*

This corrects the book's comment that the oldest observation "gets weight
`clfLastW`": the piecewise-linear function is evaluated at cumulative
uniqueness, and the first observation sits at `C_1 = w_1`, not at 0.

**Proposition 6.3 (monotone in time).** Since `w_j >= 0`, `C_i` is nondecreasing, so
`d_i` is nondecreasing before clipping. For `c_last >= 0`, `d` lies in `[c_last, 1]`.
*(Proved here.)*

* **Checked by test.** `test_time_decay_properties` (`c = 1` gives all ones, `c = 0.5`
  matches `c + (1 - c) C / T` exactly, monotone, newest = 1, `c < 0` zeroes the
  oldest). `test_time_decay_respects_original_order` (index order is restored).
  `test_time_decay_rejects_bad_c` (c outside `(-1, 1]` raises). The exact formula
  in Proposition 6.2 is checked by the `c = 0.5` case.

## Not covered

The multiprocessing engine `mpPandasObj` (Chapter 20), the Monte Carlo study
(Snippets 4.7-4.9), class weights (end of Section 4.8), the bagging classifier of
Chapter 6, and the exercises.
