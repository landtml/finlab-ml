# Proofs and derivations: `finlab.importance` (AFML chapter 8)

Every claim below is tagged with one of three labels:

- **[Proved]**: derived here from definitions, with a proof.
- **[Checked]**: verified by a test in `tests/test_importance.py`. This is numerical
  evidence on seeded inputs, not a proof.
- **[Book]**: stated in AFML chapter 8. Not proved or tested here.

Notation: $n$ samples, $p$ features, $K$ CV folds. $\mathbf X\in\mathbb R^{n\times p}$.

---

## 1. MDI (snippet 8.2)

Let $\mathrm{imp}_{t,j}\ge 0$ be the impurity decrease attributed to feature $j$ by tree
$t$, with $t=1,\dots,T$. Let $S_j=\{t:\mathrm{imp}_{t,j}\neq 0,\ \mathrm{imp}_{t,j}\text{ not NaN}\}$
and $c_j=|S_j|$. The module computes

$$
\bar m_j=\frac1{c_j}\sum_{t\in S_j}\mathrm{imp}_{t,j},\qquad
s_j=\sqrt{\frac{1}{c_j-1}\sum_{t\in S_j}(\mathrm{imp}_{t,j}-\bar m_j)^2},\qquad
\mathrm{se}_j=\frac{s_j}{\sqrt{c_j}} .
$$

Normalization: $\tau=\sum_j \bar m_j$, and the output is $m_j=\bar m_j/\tau$,
$\mathrm{se}_j'=\mathrm{se}_j/\tau$.

**[Proved] (1a) Sum to one.** $\sum_j m_j=\tau/\tau=1$ whenever $\tau>0$. The code
raises if $\tau\le 0$, so this always holds for returned output. Sums run over features
with finite $\bar m_j$. Features with $c_j=0$ are NaN and are excluded by `nansum`.

**[Proved] (1b) Bounds.** If all $\mathrm{imp}_{t,j}\ge 0$, then $\bar m_j\ge 0$ and
$\tau\ge\bar m_j$, so $0\le m_j\le 1$. The impurity decrease of a split is non-negative
in exact arithmetic, so this bound applies to real forests. The module does not check
the sign; a caller that passes negative values loses the guarantee.

**[Proved] (1c) Scale equivariance.** Dividing by $\tau$ scales $\mathrm{se}_j$ by the
same factor, so the normalized table is the unnormalized table divided by $\tau$. The
ranking of features is unchanged by the normalization.

**[Proved] (1d) Two-pass variance.** The code computes $\bar m_j$ first, then
$\sum(v-\bar m_j)^2$. This is the textbook definition of the sample variance, with no
cancellation from $\sum v^2-(\sum v)^2/c$.

**[Book] (1e) Zeros are not draws.** Under $\texttt{max\_features}=1$, a feature
receives a non-zero importance only in trees where it was drawn, so a zero means "not
drawn". Averaging zeros would bias the mean toward zero. The book recommends replacing
zeros with NaN (note 1b). The module does this. The bias argument is the book's and is
not tested here.

**[Book] (1f) In-sample.** MDI is computed on the training data, so it is in-sample.
This is a property of the method, not of the code.

**[Book] (1g) Substitution.** With two identical features, MDI gives each about half of
the importance. Not tested.

---

## 2. MDA (snippet 8.3)

For fold $k$ with training set $\mathcal T_k$ and test set $\mathcal E_k$, let $\hat f_k$
be the model fit on $\mathcal T_k$, and let $\mathbf X^{(k,j,\pi)}$ be the test matrix with
column $j$ reordered by permutation $\pi$. The score $S$ is accuracy,

$$
S(\hat f, \mathbf X, \mathbf y, \mathbf w)=\frac{\sum_{i\in\mathcal E} w_i\,\mathbf 1[\hat f(\mathbf x_i)=y_i]}{\sum_{i\in\mathcal E} w_i}.
$$

Baseline $b_k=S(\hat f_k,\mathbf X_{\mathcal E_k})$. Drop for feature $j$ in fold $k$ is
$d_{k,j}=\frac1R\sum_{r=1}^R\big(b_k-S(\hat f_k,\mathbf X^{(k,j,\pi_{k,j,r})})\big)$,
where $R$ is `n_repeats`. Output mean $\bar d_j=\frac1K\sum_k d_{k,j}$ and
$\mathrm{se}_j=\mathrm{sd}_k(d_{k,j})/\sqrt K$.

**[Proved] (2a) Marginal preserved.** For any permutation $\pi$, column $j$ of
$\mathbf X^{(k,j,\pi)}$ has the same empirical distribution as column $j$ of
$\mathbf X_{\mathcal E_k}$. The other columns are unchanged. So the only change is the
pairing between feature $j$ and the rest of the row (including $y$). This is why
permutation isolates the contribution of feature $j$ to predictions, not its marginal
distribution.

**[Proved] (2b) Weighted accuracy is a proper average.** With $w_i\ge 0$ and
$\sum w_i>0$, $S\in[0,1]$. The drop $b_k-S_{\text{perm}}$ lies in $[-1,1]$.

**[Proved] (2c) Repeat averaging.** $d_{k,j}$ is a mean of $R$ values, so averaging
over repeats leaves the expectation unchanged and reduces the variance of the
permutation randomness by $R$ (for i.i.d. draws).

**[Checked]** On data where only feature 0 carries the label (test
`test_mda_informative_feature_has_largest_drop_and_noise_is_near_zero`), the drop for
feature 0 is about 0.48 across seeds 0–5, and the drops for the four noise features stay
within $\pm 0.003$.

**[Checked]** The MDA output equals a naive loop-based reference exactly, with the same
RNG draw order (test `test_mda_matches_naive_reference_exactly`). This checks
implementation, not statistics.

**[Book] (2d) MDA can conclude all features are unimportant.** Because the score is OOS,
a model with no predictive power gives drops near zero. Not tested as a theorem.

**[Book] (2e) Substitution effects.** With two identical features, permuting one leaves
the other intact, so each shows a drop near zero even when both matter. Orthogonalization
(section 4) only mitigates linear substitution. Not tested.

**[Book] (2f) Purged CV.** The CV must be purged and embargoed (chapter 7). The module
accepts any `cv.split(X)` and does not enforce purging itself. It is the caller's
responsibility.

---

## 3. SFI (snippet 8.4)

For each feature $j$, fit $\hat g_{k,j}$ on the single column $\mathbf x_{\cdot j}$ of the
training fold, and score it on the same column of the test fold. The output is
$\bar s_j=\frac1K\sum_k S_{k,j}$ and $\mathrm{se}_j=\mathrm{sd}_k(S_{k,j})/\sqrt K$. The
baseline is the all-feature model's OOS score per fold.

**[Proved] (3a) No substitution between features.** The score $S_{k,j}$ is a function of
column $j$ and $y$ only. Changing any other column leaves it unchanged, so two identical
features get the same SFI score. This is why SFI avoids the substitution dilution of MDI
and MDA, as the book states.

**[Proved] (3b) Joint effects are lost.** A feature whose information appears only in
combination with another gets a score of the single-feature model, which can equal chance.
This follows from 3a. Example: $y=x_1\cdot x_2$ with $x_1,x_2$ independent symmetric
signs gives $\Pr(y=1\mid x_1)=\tfrac12$ for every $x_1$, so $S_{\cdot 1}$ equals the
majority-class rate. The module reports this honestly.

**[Checked]** On the same data as the MDA test, feature 0 scores about 0.98 and the noise
features score about 0.47–0.54 across seeds 0–5. The test asserts the ranking and a
noise band of $\pm 0.1$ around chance.

---

## 4. Orthogonal features (snippet 8.5)

Given $\mathbf X$, let $\mu_j$ and $\sigma_j$ be the sample mean and sample standard
deviation (ddof = 1) of column $j$. Define $\mathbf Z$ by $Z_{ij}=(X_{ij}-\mu_j)/\sigma_j$.
Let $\mathbf Z'\mathbf Z\,\mathbf W=\mathbf W\boldsymbol\Lambda$ be the eigendecomposition
with $\boldsymbol\Lambda=\mathrm{diag}(\lambda_1\ge\dots\ge\lambda_p)$ and $\mathbf W$
orthogonal. Let $\mathbf P=\mathbf Z\mathbf W_k$, where $\mathbf W_k$ holds the first $k$
columns. The kept $k$ is the smallest with $\sum_{i\le k}\lambda_i/\sum_i\lambda_i\ge\tau$.

**[Proved] (4a) Diagonal Gram matrix.** $\mathbf P'\mathbf P=\mathbf W'\mathbf Z'\mathbf Z\mathbf W=\mathbf W'\mathbf W\boldsymbol\Lambda\mathbf W'\mathbf W=\boldsymbol\Lambda$, using $\mathbf W'\mathbf W=\mathbf I$. Off-diagonal entries are zero, so the columns of $\mathbf P$ are orthogonal.
For the truncated case, $\mathbf P_k'\mathbf P_k=\boldsymbol\Lambda_k$ by the same argument.

**[Proved] (4b) Uncorrelated components.** Since $\mathrm{cov}(\mathbf P)=\mathbf P'\mathbf P/(n-1)$ when columns are mean-zero (they are, because $\mathbf Z$ is centered), $\mathrm{cov}(\mathbf P)=\boldsymbol\Lambda/(n-1)$ is diagonal. This is what the test `test_orthogonal_features_are_uncorrelated_and_match_eigenvalues` checks.

**[Proved] (4c) Correlation matrix.** $\mathbf Z'\mathbf Z/(n-1)$ is the correlation matrix
$\mathbf R$, so $\boldsymbol\Lambda/(n-1)$ are the eigenvalues of $\mathbf R$. This is why
the code's output $\lambda_i/(n-1)$ matches a naive correlation computed by loops (test
`test_orthogonal_features_eigenvalues_agree_with_naive_correlation`).

**[Proved] (4d) Trace identity.** $\mathrm{tr}(\boldsymbol\Lambda)=\mathrm{tr}(\mathbf Z'\mathbf Z)=\sum_j\sum_i Z_{ij}^2=p(n-1)$, because each standardized column has sum of squares $n-1$. So the explained-variance ratio of component $i$ is $\lambda_i/(p(n-1))$.

**[Proved] (4e) Minimal $k$.** Let $c_i=\sum_{l\le i}\lambda_l/\sum_l\lambda_l$, which is nondecreasing because $\lambda_l\ge 0$. `np.searchsorted(c, τ, side="left")` returns the smallest index $i$ with $c_i\ge\tau$. The code keeps $k=i+1$ components, so $k$ is the smallest count reaching $\tau$. The result is capped at $p$, which covers floating-point cases where $c_p$ rounds to slightly less than 1 when $\tau=1$. Negative eigenvalues from rounding are clipped to 0 before summing.

**[Proved] (4f) Transform reproduces components.** For training data $\mathbf X$,
`OrthogonalFeatures.transform` computes $((\mathbf X-\boldsymbol\mu)/\boldsymbol\sigma)\mathbf W_k = \mathbf Z\mathbf W_k=\mathbf P$, which is the fitted $\mathbf P$. New data uses the training $\boldsymbol\mu,\boldsymbol\sigma$, so it is not refit. This is checked in `test_transform_reproduces_fitted_components_and_applies_to_new_data`.

**[Checked]** Off-diagonal covariance of $\mathbf P$ is below $10^{-9}$ relative to the diagonal, and $\mathbf W$ is orthonormal to $10^{-10}$, on seeded correlated data.

**[Book] (4g) Dimension reduction.** Dropping small-eigenvalue components speeds up
convergence. Not tested as a claim about convergence.

**[Book] (4h) PCA ranking vs MDI.** The book reports a Pearson correlation of 0.8491
between eigenvalues and MDI levels, and a weighted Kendall's tau of 0.8206 (figure 8.1).
This is from the book's own data and is not reproduced here. The weighted tau (snippet
8.6) is not implemented.

**[Book] (4i) Reconstruction.** Truncating to $k$ components gives the best rank-$k$
approximation of $\mathbf Z$ in Frobenius norm (Eckart–Young). This is a standard result
and is not proved in the module or in the book's chapter 8. It is not tested.

---

## Summary

| Claim | Status |
|---|---|
| MDI means sum to 1, bounded in [0,1] for non-negative input | Proved (1a, 1b) |
| MDI normalization preserves ranking | Proved (1c) |
| MDA permutation preserves column marginal | Proved (2a) |
| MDA on informative vs noise features | Checked |
| MDA exact match to naive loop reference | Checked |
| SFI has no cross-feature substitution | Proved (3a) |
| SFI ranks informative feature first | Checked |
| $P'P=\Lambda$, components uncorrelated | Proved (4a, 4b) |
| Minimal $k$ for variance threshold | Proved (4e) |
| Transform reproduces training components | Proved (4f) |
| Eigenvalues match naive correlation | Checked |
| MDI zeros should be NaN (max_features=1 argument) | Book (1e) |
| MDI in-sample, substitution dilution | Book (1f, 1g) |
| MDA can find all features unimportant; substitution | Book (2d, 2e) |
| PCA-MDI correlation 0.8491, weighted tau 0.8206 | Book (4h) |
| Dimension reduction speeds convergence | Book (4g) |
| Truncated PCA is best rank-k approximation | Not in book; not tested (4i) |
