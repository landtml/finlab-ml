# finlab: Proofs and Checks for `finlab.hrp` (AFML ch. 16)

Component: [`src/finlab/hrp.py`](../../src/finlab/hrp.py).
Tests: [`tests/test_hrp.py`](../../tests/test_hrp.py).

Status tags as in [`microstructure.md`](microstructure.md):
**[proved here]**, **[checked by test]**, **[claimed from the book only]**.

---

## 1. Distance (AFML 16.4.1)

**Definition.** `d_ij = sqrt(1/2 (1 - rho_ij))`.

**Proposition 1.1 [proved here].** `d` is a pseudometric on `N` return series: it is
non-negative, symmetric, satisfies the triangle inequality, and `d_ij = 0` if and only if
`rho_ij = 1`. The condition `rho_ij = 1` holds exactly when series `j` is a positive affine
function of series `i`, that is, `X_j = a + b X_i` with `b > 0`. So a scaled or shifted copy
has distance zero without being identical to the original. Hence `d` is a metric on the
standardized series, that is, on series taken up to positive affine maps. (The book proves
the metric property in Appendix 16.A.1. The proof below is an independent route.)

*Proof.* Standardize each series `X_i` to a vector `z_i` with zero mean and
`||z_i||^2 = T`. Then `rho_ij = z_i . z_j / T`, and

    ||z_i - z_j||^2 = ||z_i||^2 + ||z_j||^2 - 2 z_i . z_j = 2T (1 - rho_ij).

Hence `d_ij = ||z_i - z_j|| / (2 sqrt(T))`. This is a positive multiple of the
Euclidean distance between the embedded vectors, so the four metric axioms hold.
Zero distance means `z_i = z_j`, that is, `rho = 1`. Since `z_i` and `z_j` are standardized
copies of `X_i` and `X_j`, `z_i = z_j` holds if and only if `X_j = a + b X_i` with `b > 0`
(the standardization removes the shift and the positive scale). Identical series are the
case `a = 0`, `b = 1`. ∎

**Note.** The correlation is clipped to `[-1, 1]` before the square root, so
that noise from the estimated matrix cannot produce NaN. Clipping changes
nothing for a valid correlation matrix.

**[checked by test]** `test_correlation_distance_book_example_16_1` (`rho = [[1, .7, .2], [.7, 1, -.2], [.2, -.2, 1]]`)
compares `correlation_distance` with the 4-decimal matrix written in the test, to 5e-5. That matrix is `sqrt(1/2 (1 - rho))` rounded to 4 decimals. Whether it equals the book's printed table is not checked here.

**[checked by test]** `test_hrp_scaled_copy_has_zero_correlation_distance` takes a seeded series `x` (250 draws) and its copies `2x + 1` and `0.5x - 3`, which are not identical to `x`. Their distance is `0` to 1e-7 (for `0.5x - 3`, `rho = 1 - 2e-16` in floating point, so `d` is about `1e-8`). The copy `-3x` has `rho = -1` and distance `1`.

## 2. Tree clustering (AFML 16.4.1, Snippet 16.1)

The tree is `scipy.cluster.hierarchy.linkage(squareform(D), method)`.

**Implementation note [checked by test].** Snippet 16.1 passes the square matrix
`D` directly to `linkage`. SciPy treats a square matrix as `N` observation
vectors, not as a distance matrix, and emits a `ClusterWarning`. For a 3-asset
example it returns merge heights `0.566` and `0.975`, not the distances
`0.387` and `0.632` (`tests/test_doc_claims.py::test_hrp_square_matrix_linkage_heights`). The implementation therefore uses `squareform` to condense
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

**[checked by test]** For random covariances (10 seeds, N from 2 to 24, `test_quasi_diagonalize_is_permutation_and_matches_naive`), the output is a
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

**Proposition 7.1 [proved here].** Let `H(N)` be the sum of `size^2` over the internal
blocks of the bisection of `N >= 2` leaves, where a block of size `k >= 2` splits into
`floor(k/2)` and `ceil(k/2)` (Snippet 16.3 cuts the quasi-diagonal list at its midpoint).
Then `H(N) <= N^2 ceil(log2 N)`. The variance work of the bisection, the sum over splits of
`|L1|^2 + |L2|^2`, is at most `H(N)`.

*Proof.* Each child of a block of size `k` has size at most `ceil(k/2)`, and
`ceil(ceil(x)/2) = ceil(x/2)`. By induction on depth `j`, every block at depth `j` has
size at most `ceil(N/2^j)`. Blocks at one depth are disjoint, so their sizes sum to at most
`N`, and their squares sum to at most `N ceil(N/2^j) <= N^2`. An internal block at depth `j`
needs `ceil(N/2^j) >= 2`, that is `2^j < N`, which holds for exactly `ceil(log2 N)` depths
`j = 0, ..., ceil(log2 N) - 1`. Summing over them gives the bound. For the second claim,
`|L1|^2 + |L2|^2 <= (|L1| + |L2|)^2` for each split. ∎

*Note.* The split tree depends only on `N`. The linkage tree fixes the leaf order, not the
splits, so a chain-shaped linkage tree does not make the bisection expensive. The bound
`N^3/3` fails for `N = 2` to `5`, for example `H(3) = 13 > 9`, and it is not used. The values
are `H(N) = 4, 13, 24, 42` for `N = 2, 3, 4, 5`.

**[checked by test]** `test_hrp_bisection_cost_bound_n2_to_30` checks the bound for
`N = 2` to `30`, the identity `W = H + N - N^2` for the variance work `W`, and the failure of
`N^3/3` at `N = 2` to `5`. It replicates the split schedule in Python and checks that the
replica gives the same weights as `_recursive_bisection`. Numerically, `H(N)/N^2` stays
below 2 for `N = 2` to `30` (observed only, not proved).

Each split computes two cluster variances, each costing `O(k^2)` for its block (an
inverse-variance vector and a quadratic form). By Proposition 7.1 the bisection's worst case is
therefore `O(N^2 log N)`.

**[claimed from the book only]** The book states best-case `O(log N)` and
worst-case `O(N)` for the bisection (`T(n) = Theta(n)`). Those counts are of
bisection steps and omit the cost of each cluster variance. Including that cost, the bound
above gives `O(N^2 log N)`. A chain-shaped split tree would give `O(N^3)`, but the midpoint
split never produces one. The linkage step is
`O(N^2)` in SciPy's single-linkage implementation, and the benchmark (`benchmarks/bench_hrp.py`)
is dominated by that step for large `N`.

## 8. Summary

| Result | Status |
|---|---|
| `d = sqrt(1/2(1-rho))` is a pseudometric on series, a metric on standardized series (Prop. 1.1) | proved here; checked by test for scaled copies; book cites App. 16.A.1 |
| Example 16.1 distance matrix | checked by test |
| `squareform` is required for `linkage` (Snippet 16.1) | checked by test (deviation from printed snippet) |
| Quasi-diagonalization is a permutation (Prop. 3.1) | proved here; checked by test |
| Quasi-diagonal order equals Snippet 16.2 (Prop. 3.2) | proved here (sketch); checked by test |
| Inverse-variance optimal for diagonal covariance (Prop. 4.1) | proved here; checked by test |
| Bisection split is inverse-variance (Prop. 5.1) | proved here; checked by test |
| Weights nonnegative, sum to 1 (Prop. 5.2) | proved here; checked by test on 25 random covariances, incl. singular |
| Block-diagonal cluster-level allocation (Prop. 6.1, Cor. 6.2) | proved here; checked by test |
| Bisection cost `H(N) <= N^2 ceil(log2 N)`, so `O(N^2 log N)` (Prop. 7.1) | proved here; checked by test for N = 2 to 30; the book's `T(n)=Theta(n)` is claimed from the book only |
| The book reports lower out-of-sample variance for HRP than CLA and IVP (Section 16.6) | claimed from the book only; not reproduced here |
