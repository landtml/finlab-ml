# Proof notes: finlab.onc (ONC, López de Prado, Lewis & Boudt 2019)

Module: `src/finlab/onc.py`. Tests: `tests/test_onc.py`. Not part of AFML's text.
The implementation is the simplified version described in the module: k-means on the
correlation rows, with `k` chosen by mean silhouette, and no repair step.

## Proved here

**O1 (silhouette lies in [-1, 1]).** For each variable, `s_i = (b_i - a_i)/max(a_i, b_i)`
with `a_i, b_i >= 0`, so `|s_i| <= 1`.
*Proof.* `|b - a| <= max(a, b)` for non-negative `a, b`. Singletons are set to `0`. ∎

**O2 (the distance is a metric on correlations).** `d_ij = sqrt(1/2 (1 - rho_ij))`
is symmetric, zero on the diagonal, and satisfies the triangle inequality, because it is
half the Euclidean (L2) distance between standardised return vectors: for unit-variance
`x, y` with correlation `rho`, `E(x - y)^2 = 2 - 2 rho`, so `d = sqrt(E(x-y)^2) / 2`.
*Proof sketch.* `d` is a positive multiple of an L2 norm of `x - y`, and a norm
satisfies the triangle inequality. ∎

**O3 (`onc` picks the best silhouette over the candidates it evaluates).** The returned
`k` maximises the mean silhouette over `k in [2, max_k]`, with ties to the smaller `k`.
*Proof.* The loop updates only on strict improvement. ∎

## Checked by test

- On a block-structured correlation matrix with noise and shuffled rows, `onc` recovers
  the true partition for a fixed seed.
- `silhouette_scores` matches a naive double loop (exact to 1e-12).
- Singleton clusters give silhouette `0`.
- Deterministic for a fixed seed; invalid matrices are rejected.

## Claimed or not proved

- That k-means on correlation rows recovers true clusters in general. The tests use a
  designed block structure only.
- The repair step and quality-ratio ranking of the published ONC are not implemented,
  so results can differ from the paper on harder data.
- Lloyd's iteration does not increase inertia; this is the standard result and is not
  re-proved here.
