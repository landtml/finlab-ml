# Proofs and derivations: `finlab.monte_carlo`

Scope: `run_trials`, `random_t1`, `bootstrap_uniqueness_trial` and `bootstrap_uniqueness_mc`
in `src/finlab/monte_carlo.py`. The kernels it uses from `finlab.weights` are described in
[weights.md](weights.md).

The experiment follows the printed listings of AFML Snippets 4.7 and 4.8 (section 4.5.4).
Those listings were read from the copy of the book provided in this repository's branch. The
book was not run, and no number below is a book figure unless it is marked as one.

Each claim is tagged as one of:

* **Proved here**: a proof is given in this document.
* **Checked by test**: asserted by `tests/test_monte_carlo.py` or `tests/test_weights.py`.
* **Checked against the printed book text**: a comparison made by reading the book, not by a test.
* **Measured (not a proof)**: a number from a run on the development machine, with its settings.
* **Claimed from the book only**: a statement about AFML that was not checked here.

## Notation

Trial `i` of a run with seed `s` uses the generator `G(s, i) = default_rng(SeedSequence(s, spawn_key=(i,)))`.
Labels have start bars `s_j` and end bars `e_j = s_j + l_j`, with `l_j >= 1`. Label `j` has span
`L_j = {s_j, ..., e_j}`. For a sample of draws `d_1, ..., d_n` (repeats allowed), `c_t` is the number
of draws whose span contains bar `t`, and `S = (1/n) sum_k (1/|L_{d_k}|) sum_{t in L_{d_k}} 1/c_t`
is the sample average uniqueness. This is the same quantity as
`finlab.weights.sample_average_uniqueness`.

## 1. Proved here

**Proposition 1.1 (worker-count invariance).** Let `F(i) = func(G(s, i), **kwargs)` be the
row of trial `i`. The output of `run_trials(func, n, seed=s, num_threads=k)` has row `i` equal
to `F(i)` for every `k >= 1`.

*Proof.* Each row is computed by `_trial_molecule` from a freshly made `G(s, i)` and the same
`func` and `kwargs`. `mp_pandas_obj` splits the index into molecules and concatenates the piece
results in molecule order (`pd.concat(pieces)` in `finlab.parallel`). The molecule split decides
only which process computes row `i`, not its value. So the frame is the same for every `k`.
The argument assumes `func` is a deterministic function of its arguments, and draws only from
the generator it is given. If `func` reads other state, the claim does not hold. *(Proved here,
given that `func` uses only its `rng` argument.)*

**Proposition 1.2 (prefix property).** For `m <= n`, the first `m` rows of a run of length `n`
equal the rows of a run of length `m` with the same seed.

*Proof.* Both runs compute `F(i)` for `i < m`, by Proposition 1.1. The column names come from
trial 0, which is `F(0)` in both runs. *(Proved here.)*

**Proposition 1.3 (bounds).** For every trial of `bootstrap_uniqueness_trial`, `std_u` and
`seq_u` lie in `(0, 1]`.

*Proof.* `random_t1` gives each label a start `s_j` in `0..n_bars-1` and an end `e_j = s_j + l_j`
with `l_j` in `1..max_h-1`. So `e_j > s_j` and every span has at least one bar. The bar grid of a
trial runs from 0 to `max_j e_j`, so every span lies inside it. Each drawn label is active on its
own span, so `c_t >= 1` there, and each term `1/c_t` lies in `(0, 1]`. Each per-draw average over
a span lies in `(0, 1]`, and so does their mean over the draws. The standard sample is a set of
draws of the labels. The sequential sample consists of labels with positive weight, by
Proposition 4.1 of [weights.md](weights.md), so its spans are also nonempty. *(Proved here.)*

**Proposition 1.4 (one observation).** If `n_obs = 1`, then `std_u = seq_u = 1`.

*Proof.* With one draw, `random_t1` returns one label, and each sample has one draw, that label.
So `c_t = 1` on its span and `S = 1`. For the sequential draw there is one candidate, and the
inverse-CDF rule `min{j : C_j > u * total}` with `C_0 = total > u * total` (for `u < 1`) selects it
every time. So `seq_u = 1` as well. *(Proved here.)*

**Proposition 1.5 (identical spans).** Let all `n` draws have the same span `L`, with
`|L| = ell`. Then `S = 1/n` for the standard and the sequential sample.

*Proof.* On `L`, every draw covers every bar, so `c_t = n`. Each draw has
`(1/ell) * ell * (1/n) = 1/n`, and their mean is `1/n`. *(Proved here.)*

*Scope note.* `random_t1` returns distinct start bars, so its labels never share a span. The
identity applies to inputs of `_trial_uniqueness` where the spans are equal, which is how the
test below exercises it.

## 2. Checked by test

* **Kernel against dense reference (rtol 1e-12).** `test_trial_uniqueness_kernel_matches_dense_reference`
  (6 seeds, 30 bars, `n_obs = 12`, `max_h = 6`) compares `_trial_uniqueness` with a dense
  indicator-matrix path for both `std_u` and `seq_u`. `test_trial_matches_dense_naive_reference`
  does the same for the whole trial, with the draws taken from the same generator.
  `test_trial_uniqueness_hand_computed_repeat_case` checks a hand value of `2/3` for a sample with
  repeats.
* **Serial matches parallel.** `test_num_threads_does_not_change_result` (`run_trials`, 2 and 3
  threads, 50 trials) and `test_bootstrap_uniqueness_mc_num_threads_does_not_change_result`.
* **Determinism for a fixed seed.** `test_same_seed_gives_identical_frame`,
  `test_bootstrap_uniqueness_mc_same_seed_gives_same_frame`,
  `test_trial_is_deterministic_for_fixed_rng_seed`, and `test_seed_none_records_integer_and_is_reproducible`.
  `test_different_seeds_give_different_results` checks that seeds matter.
* **Prefix property and per-trial stream.** `test_prefix_property` (Proposition 1.2) and
  `test_trial_matches_direct_call` (Proposition 1.1, for three trial indices).
* **Documented draw order.** `test_trial_follows_documented_draw_order` (3 seeds) rebuilds the
  trial from the documented steps: start bars, lengths, standard draws, then uniforms.
* **Snippet 4.7 loop.** `test_random_t1_matches_snippet_4_7_loop_with_same_draws` (10 seeds) feeds the
  same draws to a loop written as the book's `t1.loc[ix] = val` assignments, and checks that
  `random_t1` returns the same Series, including the last-draw-wins rule for repeated starts.
* **Label generator.** `test_random_t1_index_is_unique_sorted_and_within_grid`,
  `test_random_t1_lengths_lie_in_one_to_max_h_minus_one`,
  `test_random_t1_ends_are_not_clipped_at_the_last_bar`,
  `test_random_t1_can_merge_repeated_starts`, `test_random_t1_same_seed_gives_same_series`,
  `test_random_t1_generator_and_int_seed_agree`, and `test_random_t1_rejects_bad_arguments`.
* **Identities.** `test_single_draw_sample_has_uniqueness_one` (Proposition 1.4, 3 seeds) and
  `test_identical_spans_give_one_over_n` (Proposition 1.5, `n = 1, 3, 7`, with repeats).
* **Bounds.** `test_trial_values_lie_in_unit_interval` (200 trials, Proposition 1.3), and the
  range assertion in `test_sequential_bootstrap_uniqueness_exceeds_standard_by_four_se`.
* **Argument checks.** `test_invalid_n_iter_raises`, `test_invalid_num_threads_raises`,
  `test_invalid_seed_raises`, and `test_inconsistent_keys_raise`.
* **Statistical check (one seed).** `test_sequential_bootstrap_uniqueness_exceeds_standard_by_four_se`
  asserts that the paired mean gap at seed 0, `n_iter = 2000`, exceeds 4 standard errors. It is a
  regression check at one seed, not a proof, and the margin is in section 4.
* **Slow check.** `test_large_run_sequential_median_exceeds_standard_median` (marked `slow`, which
  `pyproject.toml` describes as long-running brute-force cross-checks) uses `n_iter = 20_000`,
  seed 7, and asserts that the sequential median exceeds the standard median.

## 3. Checked against the printed book text

These were checked by reading the printed listings and the text after Figure 4.2 (p. 68). No test compares
against the book's output.

* **Snippet 4.7.** The book's prose says lengths run from 0 to maxH, but the listing excludes 0, and this code follows the listing. The start is `randint(0, numBars)` and the length is `randint(1, maxH)`. In
  numpy `randint` excludes the upper bound, so lengths run from 1 to `maxH - 1`. `random_t1`
  uses the same bounds. A repeated start is overwritten by its later draw, so a label set can
  have fewer than `numObs` labels. `random_t1` does the same.
* **Snippet 4.8.** The sequential draw uses the inverse-CDF rule on the uniforms, not `np.random.choice(p=...)`; both give the same distribution. The bar grid is `range(t1.max() + 1)`, so it runs to the last label end, not to
  `numBars`. The standard sample is `np.random.choice(indM.columns, size=indM.shape[1])`, which has
  one draw per label. `bootstrap_uniqueness_trial` uses the number of labels as the sample size.
  `seqBootstrap(indM)` also defaults to one draw per label.
* **Snippet 4.9 defaults.** `numObs=10`, `numBars=100`, `maxH=5`, as used by `bootstrap_uniqueness_mc`.
  The book's `numIters` is 1E6. The default here is 10,000.
* **Snippet 4.4.** `getAvgUniqueness` takes the uniqueness of each column over the rows it covers,
  with the concurrency taken over the matrix passed in. Snippet 4.8 applies it to the bootstrapped
  matrix, so repeats are counted, as in `sample_average_uniqueness`.
* **Figure 4.2 text (p. 68).** The book states, in the text after the figure, that "the median of the average uniqueness for the
  standard method is 0.6, and the median of the average uniqueness for the sequential method is
  0.7." The experiment's measured medians match this to one decimal place (section 4).

Differences from the book's run, which the comparison has to allow for:

* The random numbers come from a seeded `numpy.random.Generator`, not numpy's global generator.
  The same integer seed does not give the book's random stream, so no single trial is the book's
  trial.
* The book's run uses 1E6 iterations and its own job engine (`mpEngine`, Chapter 20). This module
  runs `n_iter` trials through `finlab.parallel.mp_pandas_obj`.

## 4. Measured (not a proof)

These numbers come from runs on the development machine. They depend on the machine and on the
settings listed. They are not theorems.

**Statistical gap, default sizes.** `bootstrap_uniqueness_mc(n_iter=2000, seed=s)` with `n_obs=10`,
`n_bars=100`, `max_h=5`. `d = seq_u - std_u` is taken per trial. Its standard error is the sample
standard deviation of `d` over `sqrt(2000)`.

| seed | mean `d` | standard error | ratio | mean `std_u` | mean `seq_u` |
|---|---|---|---|---|---|
| 0 | 0.084357 | 0.002807 | 30.0 | 0.6069 | 0.6912 |
| 1 | 0.087352 | 0.002860 | 30.5 | 0.6094 | 0.6968 |
| 2 | 0.082604 | 0.002995 | 27.6 | 0.6119 | 0.6945 |
| 3 | 0.083943 | 0.002911 | 28.8 | 0.6119 | 0.6958 |
| 42 | 0.085824 | 0.002858 | 30.0 | 0.6073 | 0.6932 |

**Medians against the book's text (p. 68).** `bootstrap_uniqueness_mc(n_iter=20000, seed=0, num_threads=4)`
gives medians of `0.6` for `std_u` and `0.7` for `seq_u`, to four decimal places. The book's
text gives 0.6 and 0.7 for the same two statistics. Seed 7, the slow test's seed, gives the same
medians. This is a comparison at one decimal place, from a run of 20,000 trials and not 1E6. It
supports that the design matches the book's experiment closely enough to reproduce those
medians. It is not a test of the book's full distribution.

**Repeated starts.** With 10 draws on 100 start bars, the chance of at least one repeated start is
`1 - prod_{i=0}^{9} (1 - i/100)`, about `0.372`. For seeds `s = 0..1999`, `random_t1(10, 100, 5, seed=np.random.default_rng(s))` gives fewer than 10 labels in
`0.369` of label sets. This matches the calculation.

**Another experiment in this repository, not comparable.** [weights.md](weights.md) reports `0.1963` (sequential)
against `0.1913` (standard) over 200 seeds. That design has 200 bars, 60 labels and spans up to 30,
as described there. Its settings differ from the experiment above, so the two results must not be
compared as one number.

**Benchmark, one run.** `benchmarks/bench_monte_carlo.py`, default sizes (`n_obs=10`, `n_bars=100`,
`max_h=5`):

* Per trial: `146.7 us` for a dense NumPy path against `3.99 us` for the jitted `_trial_uniqueness`,
  a ratio of `36.8x`. The dense path computes the same quantities with matrix operations, so this
  measures the implementation, not a different method. The script also checks the two paths
  against each other and stops if they differ by more than `1e-9`. That check is in the script,
  not in the test suite.
* First call in a fresh process with an empty numba cache: `2496 ms`, which is compile time.
* `bootstrap_uniqueness_mc(n_iter=20000)`: `8509` trials/s with `num_threads=1`, and `25389`
  trials/s with the script's parallel run, `num_threads=min(4, os.cpu_count())`, a speedup of
  `2.98x`. Each parallel run includes creating its process pool.

## 5. Claimed from the book only

These statements are about AFML and were not checked here.

* **The sequential sample is closer to IID.** The book says the sequential bootstrap sample "will be
  much closer to IID than samples drawn from the standard bootstrap method", and that this "can be
  verified by measuring an increase in" average uniqueness. The experiment here measures average
  uniqueness, which is the quantity the book proposes for that check. It does not measure closeness
  to IID, so it does not test the IID claim.
* **The ANOVA result.** The text after Figure 4.2 says an ANOVA test on the difference of means gives "a
  vanishingly small probability". This module does not run that test.
* **The figure's source run.** The book does not state how many trials produced Figure 4.2. This
  document does not claim it was the 1E6 run, and makes no claim that the figure is reproduced.

## Limits

No convergence diagnostics are given. The paired gap is reported at fixed settings only. No formal
test of which bootstrap gives higher uniqueness is included. The median comparison in section 4 is
at one decimal place and at 20,000 trials. Results are machine-dependent where timings are concerned.
