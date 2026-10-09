# finlab: Proofs and Checks for `finlab.hrp` (AFML ch. 16)

Component: [`src/finlab/hrp.py`](../../src/finlab/hrp.py).
Tests: [`tests/test_hrp.py`](../../tests/test_hrp.py).

Status tags as in [`microstructure.md`](microstructure.md):
**[proved here]**, **[checked by test]**, **[claimed from the book only]**.

---

## 1. Distance (AFML 16.4.1)

**Definition.** `d_ij = sqrt(1/2 (1 - rho_ij))`.

**Proposition 1.1 [proved here].** `d` is a metric on `N` return series, that is,
non-negative, zero exactly on identical series, symmetric, and it satisfies the
triangle inequality. (The book proves this in Appendix 16.A.1. The proof below is
an independent route.)

*Proof.* Standardize each series `X_i` to a vector `z_i` with zero mean and
`||z_i||^2 = T`. Then `rho_ij = z_i . z_j / T`, and

    ||z_i - z_j||^2 = ||z_i||^2 + ||z_j||^2 - 2 z_i . z_j = 2T (1 - rho_ij).

Hence `d_ij = ||z_i - z_j|| / (2 sqrt(T))`. This is a positive multiple of the
Euclidean distance between the embedded vectors, so the four metric axioms hold.
Zero distance means `z_i = z_j`, that is, `rho = 1`, which is coincidence up to
scale. ∎

**Note.** The correlation is clipped to `[-1, 1]` before the square root, so
that noise from the estimated matrix cannot produce NaN. Clipping changes
nothing for a valid correlation matrix.

**[checked by test]** Book's Example 16.1 (`rho = [[1, .7, .2], [.7, 1, -.2], [.2, -.2, 1]]`)
gives the book's distance matrix to 5e-5.

## 2. Tree clustering (AFML 16.4.1, Snippet 16.1)

The tree is `scipy.cluster.hierarchy.linkage(squareform(D), method)`.

**Implementation note [checked by test].** Snippet 16.1 passes the square matrix
`D` directly to `linkage`. SciPy treats a square matrix as `N` observation
vectors, not as a distance matrix, and emits a `ClusterWarning`. For a 3-asset
example it returns merge heights `0.566` and `0.975`, not the distances
`0.387` and `0.632`. The implementation therefore uses `squareform` to condense
`D` first. This is a deviation from the printed snippet, and it is required for
the book's stated intent.

**Deviation from the text [noted].** The book's Example 16.2 clusters on the
Euclidean distance between columns of `D`. We follow Snippet 16.1 and cluster on
`D` directly, with a choice of linkage (`single` by default).

## 3. Quasi-diagonalization (AFML 16.4.2, Snippet 16.2)

**Proposition 3.1 [proved here].** `quasi_diagonalize(link)` returns a permutation
of `0, ..., N-1`.

*Proof.* A linkage matrix over `N` leaves has `N-1` rows. Row `m` merges two
clusters into node `N + m`, so the root is node `2N - 2`. Each node is either a
leaf (`< N`) or has exactly two children, and every leaf is a descendant of the
root exactly once, because the merged clusters are disjoint by construction. The
stack traversal visits each node once and pushes both children of every internal
node. The output is the sequence of leaves in left-to-right depth-first order,
so it lists each index exactly once. ∎

**Proposition 3.2 [proved here].** The output equals the book's Snippet 16.2 order.
*Proof sketch.* Snippet 16.2 repeatedly replaces each cluster ID in the top row by
its two constituents, keeping their positions (`sort_index` on the even/odd
index layout). That is an in-order expansion of the tree, which is the depth-first
left-to-right traversal above. ∎

**[checked by test]** For random covariances (N up to 25), the output is a
permutation, and it equals a recursive list-based expansion of the same linkage.

## 4. Inverse-variance allocation is optimal for diagonal covariance (Appendix 16.A.2)

**Proposition 4.1 [proved here].** If `Sigma = diag(s_1^2, ..., s_N^2)` with
`s_i > 0`, then `argmin_{w : 1'w = 1} w' Sigma w` is `w_i ∝ 1/s_i^2`.

*Proof.* The objective is strictly convex, and the constraint is affine, so the
KKT conditions are necessary and sufficient. The Lagrangian gives
`2 s_i^2 w_i = mu` for every `i`, so `w_i = mu / (2 s_i^2)`. Imposing `1'w = 1`
gives `mu = 2 / sum_j (1/s_j^2)`, which yields `w_i = (1/s_i^2) / sum_j (1/s_j^2)`. ∎

**[checked by test]** `hrp_weights(diag(1, 4, 0.25, 9))` equals the inverse-variance weights to 1e-12.

## 5. Cluster variance and the bisection split (AFML 16.4.3, Snippet 16.3)

For a block `L` with covariance `V_L`, let `w~ = diag(V_L)^{-1} / tr(diag(V_L)^{-1})`
(inverse-variance weights) and `V~_L = w~' V_L w~`.

**Proposition 5.1 [proved here].** If the block is split into `L1`, `L2` with
variances `V1 = V~_{L1}` and `V2 = V~_{L2}`, and `alpha = 1 - V1/(V1 + V2) = V2/(V1 + V2)`,
then the total weight `alpha` goes to `L1` and `1 - alpha` to `L2`. Their shares
are inverse to variance: `alpha / (1 - alpha) = (1/V1) / (1/V2)`.

*Proof.* `alpha = V2/(V1+V2)` and `1 - alpha = V1/(V1+V2)`. Then
`alpha/(1-alpha) = V2/V1 = (1/V1)/(1/V2)`. ∎

**Proposition 5.2 (mass conservation) [proved here].** At every step, the weights
of a block sum to the weight that block received from its parent. The final
weights are nonnegative and sum to 1.

*Proof.* By induction on the depth. The root has weight 1. A block of weight `W`
is split into parts of weight `alpha W` and `(1 - alpha) W`, with `alpha in [0, 1]`
because `V1, V2 >= 0`. This is the value of `V1, V2` when the diagonal is
positive, which the implementation requires. The total over the leaves is the
root's weight, 1. ∎

**Edge case.** If `V1 + V2 = 0`, the implementation sets `alpha = 1/2`. Positive
diagonal entries make `V_L > 0`, so this branch is never reached from a valid
input.

**[checked by test]** Weights sum to 1 to 1e-12 and are nonnegative on 25 seeded
random covariances, including singular ones (`T < N` and an explicit rank-2 case).
Results match a list-based recursive reference for `single`, `complete` and
`average` linkages (rtol 1e-10).

## 6. Block-diagonal covariance (cluster-level inverse variance)

**Proposition 6.1 [proved here] (two blocks of two).** Let
`C = diag(C_A, C_B)` with `C_A, C_B` each `2 x 2`, and with the cross-block
correlation zero. Let the within-block correlation be `rho` with `|rho| < 1`, and
suppose `rho > 0`, so that under single linkage each block forms before the two
blocks merge. (Under the distance `d = sqrt(1/2 (1-rho))`, within-block distances
`sqrt(1/2 (1-rho)) < sqrt(1/2) = d_cross` require `rho > 0`.) Then the HRP weights
are

    w_i = (V_b^{-1} / (V_A^{-1} + V_B^{-1})) * (s_i^{-2} / sum_{j in b(i)} s_j^{-2}),

where `b(i)` is the block containing `i`, and `V_b = w~_b' C_b w~_b` is the
book's cluster variance.

*Proof.* Single linkage merges the two within-block pairs before any cross-block
pair, so the leaf order is `(A, B)` with each block contiguous, and the top split
is `L1 = A`, `L2 = B`. The bisection sizes are `|L1| = |L2| = 2`. By Proposition
5.1, the top split gives `A` the weight `V_B/(V_A+V_B) = (1/V_A)/(1/V_A+1/V_B)`,
which is the formula above. Within a block of two, the split is between two
singletons, whose variances are `s_i^2`. By Proposition 5.1 this gives the
inverse-variance shares `s_i^{-2}/(s_1^{-2}+s_2^{-2})`. ∎

**Corollary 6.2 [proved here].** For diagonal `C` (all blocks of size one), the
weights reduce to the global inverse-variance allocation, the case of Proposition 4.1.

*Proof.* With singleton blocks `V_b = s_b^2`, the top-level weight is
`(1/s_b^2)/sum_j (1/s_j^2)`, and this is the within-block share, so the product is
`s_i^{-2}/sum_j s_j^{-2}`. ∎

**[checked by test]** `test_hrp_block_diagonal_matches_cluster_level_inverse_variance`
checks the formula of Proposition 6.1 to 1e-10 for specific `s` and `rho = 0.5`.
`test_hrp_diagonal_covariance_is_inverse_variance` checks Corollary 6.2.

**Scope.** Proposition 6.1 is the two-block case. For larger blocks, the within-block
bisection is not inverse-variance (it recurses through the block), so the
block-level formula holds only for the top split, not for the within-block
weights.

## 7. Complexity

**Proposition 7.1 [proved here].** The bisection costs `sum over all internal
blocks of (size)^2` operations, which is at most `N^3 / 3` (a chain-shaped tree).
The sum over each level is at most `N^2`.

*Proof.* Each block of size `k` costs `O(k^2)` to compute its two cluster variances
(an inverse-variance vector and a quadratic form). The blocks of one tree level are
disjoint, so their costs sum to at most `N^2`. Each leaf has `O(log N)` ancestors
in a balanced tree, but a chain-shaped tree has `N - 1` levels, giving `N^3/3`. ∎

**[claimed from the book only]** The book states best-case `O(log N)` and
worst-case `O(N)` for the bisection (`T(n) = Theta(n)`). Those counts are of
bisection steps and omit the cost of each cluster variance. With the `O(k^2)`
variance cost, the implementation's worst case is `O(N^3)`. The linkage step is
`O(N^2)` in SciPy's single-linkage implementation, and the benchmark (`benchmarks/bench_hrp.py`)
is dominated by that step for large `N`.

## 8. Summary

| Result | Status |
|---|---|
| `d = sqrt(1/2(1-rho))` is a metric (Prop. 1.1) | proved here; book proves it in App. 16.A.1 |
| Example 16.1 distance matrix | checked by test |
| `squareform` is required for `linkage` (Snippet 16.1) | checked by test (deviation from printed snippet) |
| Quasi-diagonalization is a permutation (Prop. 3.1) | proved here; checked by test |
| Quasi-diagonal order equals Snippet 16.2 (Prop. 3.2) | proved here (sketch); checked by test |
| Inverse-variance optimal for diagonal covariance (Prop. 4.1) | proved here; checked by test |
| Bisection split is inverse-variance (Prop. 5.1) | proved here; checked by test |
| Weights nonnegative, sum to 1 (Prop. 5.2) | proved here; checked by test on 25 random covariances, incl. singular |
| Block-diagonal cluster-level allocation (Prop. 6.1, Cor. 6.2) | proved here; checked by test |
| Complexity `O(N^3)` worst case (Prop. 7.1) | proved here; the book's `T(n)=Theta(n)` is claimed from the book only |
| The book reports lower out-of-sample variance for HRP than CLA and IVP (Section 16.6) | claimed from the book only; not reproduced here |
