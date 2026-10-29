# Proof notes: finlab.tuning (AFML ch. 9)

Module: `src/finlab/tuning.py`. Tests: `tests/test_tuning.py`.

## Proved here

**T1 (selection is an argmax over candidates).** `grid_search` and
`randomized_search` return the candidate `theta*` maximising the mean CV score
`S(theta) = (1/K) sum_k score_k(theta)`, with ties resolved to the earlier candidate.
*Proof.* The loop keeps the first candidate attaining the running maximum, updating
only on strict improvement (`mean > best`). ∎

**T2 (leakage-freeness is inherited, not re-proved).** The search is leakage-free
exactly when the splitter passed to it is: each `score_k` is computed on a test
block after purging and embargo from `finlab.cv`, whose no-leakage property is proved
in `docs/proofs/cpcv.md`. The tuner adds no training data beyond what the splitter
yields.

**T3 (the reported best score is optimistically biased).** Let `S_j` be the CV mean
for candidate `j`, and assume the `S_j` are random with the same expectation `mu`
but noise. Then `E[max_j S_j] >= max_j E[S_j] = mu`, with strict inequality whenever
the `S_j` are not almost surely equal.
*Proof.* `max_j S_j >= S_i` for every `i`, so `E[max] >= E[S_i]` for each `i`. Strictness
follows since `max` of non-degenerate random variables with equal means exceeds the mean
on a set of positive probability. ∎
Consequence: `best_score` overstates the performance of the chosen model on new data.
Nested CV is needed for an unbiased estimate and is not implemented here.

## Checked by test (`test_grid_search_recovers_best_threshold`, `test_grid_search_ties_prefer_first_candidate`, `test_randomized_search_is_reproducible`, `test_cv_score_matches_manual_loop`, `test_empty_grid_rejected`)

- Grid search recovers the best threshold on a dataset built with a known optimum.
- Ties prefer the first candidate.
- Randomized search with a fixed seed is reproducible (identical results table).
- `cv_score` equals a manual loop over the splits (mean and standard deviation).
- An empty grid is rejected.

## Pipeline weights (AFML Snippet 9.2)

Snippet 9.2 (`MyPipeline`) overrides `Pipeline.fit` so that `sample_weight` reaches the last
step as `<last step>__sample_weight`. `finlab.tuning` makes the same routing in
`fit_model`, which `cv_score` calls. The routing is not a book result; it is a rule of this
module.

**Checked by test** `test_pipeline_gets_sample_weight_on_its_final_step`: with weights, the
fit of a Pipeline model receives `thresh__sample_weight` equal to `w[train]` on every fold, and
`test_pipeline_without_sample_weight_is_fit_with_no_fit_params` shows that no weights means no
fit parameters.

**Checked by test** `test_grid_search_with_sklearn_pipeline_matches_manual_weighted_loop`: the
score from `grid_search` on a scikit-learn `Pipeline` equals, exactly, a manual loop that fits
the same Pipeline with `lr__sample_weight` on each fold.

**Checked by test** `test_grid_search_with_sklearn_pipeline_and_metadata_routing_matches_manual_loop`:
with `enable_metadata_routing=True`, the same score comes from passing `sample_weight` directly.

## Not covered

Nested CV, Bayesian search, and multiple-testing correction of the selected score
(see `finlab.stats.deflated_sharpe_ratio` for one such correction in the backtest
setting).
