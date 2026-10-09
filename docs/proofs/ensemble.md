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

## What is claimed from the book only

- That sequential bagging reduces the redundancy of bags with overlapping labels
  (AFML ch. 6), and that subsampling to the effective sample size is a sensible
  default. The module does not reproduce the book's Monte Carlo study.

## Scope

Classification with hashable labels. Sample weights passed to `fit` are sliced per
bag but are not themselves re-normalised inside the bag. Out-of-bag estimation is
not implemented.
