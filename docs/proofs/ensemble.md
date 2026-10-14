# Proof notes: finlab.ensemble (AFML ch. 6)

Module: `src/finlab/ensemble.py`. Tests: `tests/test_ensemble.py`.

## What is proved here

**Reproducibility.** Given `seed`, every bag's draws are a deterministic function
of the data and the seed: the bag seeds are drawn once from one generator, and
each draw uses its own `default_rng(seed_b)`. Checked by
`test_same_seed_gives_identical_bags`.

**Sequential bags are more unique (empirical, one setting).** Sequential bootstrap
(see `docs/proofs/weights.md`) is greedy in the average-uniqueness sense, so each
bag's draws have higher average uniqueness than i.i.d. draws. This is checked in
one setting by `test_sequential_bags_have_higher_average_uniqueness_than_iid`. It
is not a general theorem here: the book does not give a general bound, and the
margin depends on the horizon and the sample.

**Same bags with `oob_score=False` (proved here).** The out-of-bag pass runs after the
bag loop and draws no random numbers. The only `oob_score` check inside the loop raises
and does nothing else. So the draws and the predictions do not depend on `oob_score`.
Checked by `test_oob_score_leaves_bags_and_predictions_unchanged`.

## What is claimed from the book only

- That sequential bagging reduces the redundancy of bags with overlapping labels
  (AFML ch. 6), and that subsampling to the effective sample size is a sensible
  default. The module does not reproduce the book's bagging-accuracy calculation
  (Snippet 6.1), which is analytic rather than a Monte Carlo study.
- That labels overlapping in time leak between in-bag and out-of-bag samples, so the
  out-of-bag score is optimistic for overlapping labels and does not replace purged
  cross-validation (AFML ch. 6 and ch. 7). Not verified against the text.

## Out-of-bag score

**Definition (checked by test).** For row i, let B_i be the set of bags whose draws do
not contain i. `oob_decision_function_[i]` is the mean of those bags' `predict_proba`
over B_i, and is NaN when B_i is empty. `oob_score_` is the accuracy of the argmax over
the rows with B_i nonempty, weighted by `sample_weight` when weights are given.
`test_oob_decision_function_matches_slow_reference` compares the fast path with a loop
written from this definition, for sequential and i.i.d. bags.

**Worked case (checked by test).** `test_oob_hand_case_with_fixed_draws` fixes the draws
to [0, 0, 1], [1, 2, 2] and [0, 1, 2] for n = 3. Then B_0 = {1}, B_1 is empty and
B_2 = {0} (bag indices). The bags give class-1 probabilities 0.6 and 0.2, so the rows
are [0.4, 0.6] and [0.8, 0.2]. Row 0 is predicted 1 and row 2 is predicted 0; both true
labels are 0, so the unweighted accuracy is 1/2. With weights 1, 100 and 3 the NaN row
drops out, and the accuracy is (1 * 0 + 3 * 1) / (1 + 3) = 3/4.

**Fit-time overhead (measured).** `PYTHONPATH=src python3 benchmarks/bench_ensemble.py`,
three runs on 2026-10-14 on a shared 4-vCPU machine, with other jobs running (1-minute load
average 2.8 at 10:59). n = 1000 rows, 50 bags, nearest-centroid base, best of 3 per setting.

- Sequential bags: fit 1.55 to 1.62 s with `oob_score=False` and 1.51 to 1.56 s with
  `oob_score=True` (0.96x to 0.98x). The out-of-bag pass is within run-to-run noise here,
  because the sequential draws dominate the fit.
- i.i.d. bags: 0.0068 to 0.0098 s with `oob_score=False` and 0.0114 to 0.0170 s with
  `oob_score=True` (1.43x to 1.73x). The out-of-bag pass adds about 4 to 7 ms over 50 bags.

## Scope

Classification with hashable labels. Sample weights passed to `fit` are sliced per
bag but are not themselves re-normalised inside the bag. The out-of-bag score uses the
draws of one fit and does not purge overlapping labels.
