# Proofs and derivations: `finlab.monte_carlo`

Scope: `run_trials`, `random_t1`, `bootstrap_uniqueness_trial` and `bootstrap_uniqueness_mc`
in `src/finlab/monte_carlo.py`. The kernels it uses from `finlab.weights` are described in
[weights.md](weights.md).

The book text is not in this repository, and it was not consulted for this work. Nothing
here reproduces a book figure. Where a claim is about the book, it is marked as such.

Each claim is tagged as one of:

* **Proved here**: a proof is given in this document.
* **Checked by test**: asserted by `tests/test_monte_carlo.py` or `tests/test_weights.py`.
* **Measured (not a proof)**: a number from a run on the development machine, with its settings.
* **Claimed from the book only**: a statement about AFML that was not checked here.

## Notation

Trial `i` of a run with seed `s` uses the generator `G(s, i) = default_rng(SeedSequence(s, spawn_key=(i,)))`.
A label has start bar `s_j` and end bar `e_j`, and its span is `L_j = {s_j, ..., e_j}`.
For a sample of draws `d_1, ..., d_n` (repeats allowed), `c_t` is the number of draws whose span
contains bar `t`, and `S = (1/n) sum_k (1/|L_{d_k}|) sum_{t in L_{d_k}} 1/c_t` is the sample
average uniqueness. This is the same quantity as `finlab.weights.sample_average_uniqueness`.

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

*Proof.* A label has `s_j` in `0..n_bars-1` and `e_j = min(s_j + l_j, n_bars - 1)` with `l_j >= 1`.
So `e_j >= s_j` and every span has at least one bar. Each drawn label is active on its own span,
so `c_t >= 1` there, and each term `1/c_t` lies in `(0, 1]`. Each per-draw average over a span
lies in `(0, 1]`, and so does their mean over the `n` draws. The standard sample is a
set of draws of the labels. The sequential sample consists of labels with positive weight, by
Proposition 4.1 of [weights.md](weights.md), so its spans are also nonempty. *(Proved here.)*

**Proposition 1.4 (one observation).** If `n_obs = 1`, then `std_u = seq_u = 1`.

*Proof.* With one label, every standard draw is that label, so `c_t = 1` on its span and
`S = 1`. For the sequential draw there is one candidate, and the inverse-CDF rule
`min{j : C_j > u * total}` with `C_0 = total > u * total` (for `u < 1`) selects it every time.
So `seq_u = 1` as well. *(Proved here.)*

**Proposition 1.5 (identical spans).** Let all `n` draws have the same span `L`, with
`|L| = ell`. Then `S = 1/n` for the standard and the sequential sample.

*Proof.* On `L`, every draw covers every bar, so `c_t = n`. Each draw has
`(1/ell) * ell * (1/n) = 1/n`, and their mean is `1/n`. *(Proved here.)*

*Scope note.* `random_t1` gives distinct start bars, so its labels never share a span. The
identity applies to inputs of `_trial_uniqueness` where the spans are equal, which is how the
test below exercises it.

## 2. Checked by test

* **Kernel against dense reference (rtol 1e-12).** `test_trial_uniqueness_matches_naive_reference`
  (6 seeds, `n_bars = 40`, `n_obs = 12`, `max_h = 6`) compares `_trial_uniqueness` with a dense
  indicator-matrix path for both `std_u` and `seq_u`. `test_trial_uniqueness_hand_computed_repeat_case`
  checks a hand value of `2/3` for a sample with repeats.
* **Serial matches parallel.** `test_num_threads_does_not_change_result` (`run_trials`, 2 and 3
  threads, 50 trials) and `test_bootstrap_uniqueness_mc_num_threads_does_not_change_result`.
* **Determinism for a fixed seed.** `test_same_seed_gives_identical_frame`,
  `test_bootstrap_uniqueness_mc_same_seed_gives_same_frame`,
  `test_trial_is_deterministic_for_fixed_rng_seed`, and `test_seed_none_records_integer_and_is_reproducible`.
  `test_different_seeds_give_different_results` checks that seeds matter.
* **Prefix property and per-trial stream.** `test_prefix_property` (Proposition 1.2) and
  `test_trial_matches_direct_call` (Proposition 1.1, for three trial indices).
* **Documented draw order.** `test_trial_follows_documented_draw_order` (3 seeds) rebuilds the
  trial from the documented steps: labels, then standard draws, then uniforms.
* **Identities.** `test_single_label_sample_has_uniqueness_one` (Proposition 1.4, 3 seeds) and
  `test_identical_spans_give_one_over_n` (Proposition 1.5, `n = 1, 3, 7`, with repeats).
* **Bounds.** `test_trial_values_lie_in_unit_interval` (200 trials, Proposition 1.3), and the
  range assertion in `test_sequential_bootstrap_uniqueness_exceeds_standard_by_four_se`.
* **Label generator.** `test_random_t1_index_is_unique_sorted_and_values_are_ordered` (4 seeds),
  `test_random_t1_span_lengths_respect_max_h_and_clip_at_last_bar`,
  `test_random_t1_same_seed_gives_same_series`, `test_random_t1_generator_and_int_seed_agree`,
  and `test_random_t1_rejects_bad_arguments`.
* **Argument checks.** `test_invalid_n_iter_raises`, `test_invalid_num_threads_raises`,
  `test_invalid_seed_raises`, and `test_inconsistent_keys_raise`.
* **Statistical check (one seed).** `test_sequential_bootstrap_uniqueness_exceeds_standard_by_four_se`
  asserts that the paired mean gap at seed 0, `n_iter = 2000`, exceeds 4 standard errors. It is a
  regression check at one seed, not a proof, and the margin is in the measured section below.
* **Slow check.** `test_large_run_sequential_median_exceeds_standard_median` (marked `slow`, which
  `pyproject.toml` describes as long-running brute-force cross-checks) uses `n_iter = 20_000`,
  seed 7, and asserts that the sequential median exceeds the standard median.

## 3. Measured (not a proof)

These numbers come from runs on the development machine. They depend on the machine and on the
settings listed. They are not theorems.

**Statistical gap, default sizes.** `bootstrap_uniqueness_mc(n_iter=2000, seed=0)` with
`n_obs=10`, `n_bars=100`, `max_h=5`. The paired gap `d = seq_u - std_u` has mean `0.084453`,
with standard error `0.002728` (sample standard deviation over `sqrt(2000)`). Mean `std_u` is about
`0.593` and mean `seq_u` about `0.678`. The same design at seeds 1, 2, 3 and 42 gives paired means
`0.080511`, `0.084183`, `0.085663` and `0.083638`. These are from one re-run on the development
machine while writing this note.

**Another experiment in this repository, not comparable.** [weights.md](weights.md) reports `0.1963` (sequential)
against `0.1913` (standard) over 200 seeds. That design has 200 bars, 60 labels and spans up to 30,
as described there. Its label design and sizes differ from the experiment above, so the two
results must not be compared as one number.

**Benchmark, one run.** `benchmarks/bench_monte_carlo.py`, default sizes (`n_obs=10`, `n_bars=100`,
`max_h=5`):

* Per trial: `154.2 us` for a dense NumPy path against `4.45 us` for the jitted `_trial_uniqueness`,
  a ratio of `34.6x`. The dense path computes the same quantities with matrix operations, so this
  measures the implementation, not a different method. The script also checks the two paths
  against each other and stops if they differ by more than `1e-9`. That check is in the script,
  not in the test suite.
* First call in a fresh process with an empty numba cache: `2402 ms`, which is compile time.
* `bootstrap_uniqueness_mc(n_iter=20000)`: `8727` trials/s with `num_threads=1`, and `29307`
  trials/s with the script's parallel run, `num_threads=min(4, os.cpu_count())`, a speedup of
  `3.36x`. Each parallel run includes creating its process pool.

## 4. Claimed from the book only

The book was not consulted. The statements below are not checked here and may be wrong.

* **The Monte Carlo study the other docs refer to as Snippets 4.7-4.9.** No claim is made that this
  module reproduces it. Its label design, sizes, defaults, draw order and the uniform-to-draw rule
  are choices of this repository and are not verified against the book.
* **The defaults and run sizes.** The default `n_iter = 10_000` is a short run for a quick check.
  How it compares with the book's run sizes is not known from this repository, so no comparison is
  made.
* **The reported medians.** Whether the book's reported medians look like the `std_u` and `seq_u`
  medians here is not checked.
* **The sequential sample is closer to IID.** This is the book's motivation for the sequential
  bootstrap, as the other proof note records. The experiment here measures average uniqueness,
  which is one quantity. It does not measure closeness to IID, so it does not test that claim.

## Limits

No convergence diagnostics are given. The paired gap is reported at fixed settings only. No formal
test of which bootstrap gives higher uniqueness is included. Results are machine-dependent where
timings are concerned.
