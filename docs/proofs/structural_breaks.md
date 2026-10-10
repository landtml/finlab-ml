# finlab.structural_breaks: derivations and status (AFML ch. 17)

Component under test: [`src/finlab/structural_breaks.py`](../../src/finlab/structural_breaks.py).
Tests: [`tests/test_structural_breaks.py`](../../tests/test_structural_breaks.py).
Benchmark: [`benchmarks/bench_structural_breaks.py`](../../benchmarks/bench_structural_breaks.py).

Each result below carries one status tag:

- **[proved here]**: derived in this document.
- **[checked by test]** (the claim names its test function, for example `test_csw_sup_matches_naive`): the code is compared with a brute-force reference, or a
  property is asserted on seeded inputs. This is numerical evidence, not a proof.
- **[claimed from the book only]**: stated in AFML (or the cited paper) and used
  as given. We did not re-derive it here.

---

## 1. Brown-Durbin-Evans recursive residuals (AFML 17.3.1)

Model: `y_t = x_t' beta + eps_t`, `eps_t` i.i.d. `N(0, sigma^2)` under H0 (the
book's `beta_t = beta` for all `t`). Let `X_t` stack `x_1..x_t` and
`P_t = (X_t' X_t)^{-1}`, with `beta_hat_t = P_t X_t' y_{1:t}`. Start from an
initial block of `k = p` observations, where `X_p` is invertible.

### 1.1 Recursive least squares update **[proved here]**

For `t > k`, adding row `x_t` to `X_{t-1}` and applying Sherman-Morrison to
`(A + x x')^{-1} = A^{-1} - A^{-1}xx'A^{-1} / (1 + x'A^{-1}x)` with `A = X_{t-1}'X_{t-1}`:

    P_t = P_{t-1} - (P_{t-1} x_t x_t' P_{t-1}) / f_t,        f_t = 1 + x_t' P_{t-1} x_t.

Since `X_t'y_{1:t} = X_{t-1}'y_{1:t-1} + x_t y_t`,

    beta_hat_t = beta_hat_{t-1} + P_t x_t (y_t - x_t' beta_hat_{t-1}).

Using `P_t x_t = P_{t-1} x_t / f_t` gives
`beta_hat_t = beta_hat_{t-1} + (P_{t-1} x_t / f_t) (y_t - x_t' beta_hat_{t-1})`, which is the
update coded in `_recursive_residuals`.

### 1.2 Standardised one-step residual **[proved here]**

Define `w_t = (y_t - x_t' beta_hat_{t-1}) / sqrt(f_t)` for `t = k+1, ..., T`. This is
the book's `omega_t` with `f_t` written in the standard form (the book's
`f_t = sigma^2 [1 + x_t'(X_t'X_t)^{-1}x_t]` is garbled in the text; the standard
form uses `X_{t-1}`, the data before `t`).

Linear representation: `beta_hat_{t-1} - beta = P_{t-1} X_{t-1}' eps_{1:t-1}`, so

    y_t - x_t' beta_hat_{t-1} = eps_t - x_t' P_{t-1} X_{t-1}' eps_{1:t-1}.

Hence `E[w_t] = 0`. Since `eps_t` is independent of `eps_{1:t-1}`,
`Var(y_t - x_t'beta_hat_{t-1}) = sigma^2 (1 + x_t'P_{t-1}x_t) = sigma^2 f_t`, and dividing
by `sqrt(f_t)` gives `Var(w_t) = sigma^2`.

### 1.3 Independence of the recursive residuals **[claimed from the book only]**

The book (and Brown, Durbin and Evans 1975) state that under H0 the `w_t` are
independent `N(0, sigma^2)`. We use this to get the null distribution of `S_t`.
We did not re-derive the independence here. A check of the orthogonality of the
coefficient vectors would be needed for a full proof.

### 1.4 The CUSUM statistic and its null variance **[proved here, conditional on 1.3]**

With `k = p` and `S_t = sum_{j=k+1}^{t} w_j / sigma_w` (sum of `t - k` terms, 1-based `t`),

    Var_H0(S_t) = (t - k) * sigma^2 / sigma_w^2  ->  t - k  when sigma_w = sigma.

*Discrepancy with the book.* AFML writes `S_t ~ N[0, t-k-1]`. Our count of terms is
`t - k`, which gives `t - k` for the sum over `j = k+1..t`. The book's `t - k - 1`
corresponds to a convention where the initial block is of size `k+1`. The code
reports the count `t - p + 1` (0-based `t`), which is `t - p` in 1-based indexing,
and the `z` field divides by `sqrt` of that count.

### 1.5 Scale estimate **[checked by test: `test_cusum_matches_naive_mean_model`, `test_cusum_matches_naive_with_features`; the formula is a book choice]**

The book's `sigma_w^2 = (T-k)^{-1} sum (w_t - E[w_t])^2`. We take `E[w_t] = 0`
(its value under H0), so `sigma_w^2 = mean(w_t^2)`. The book's text does not say
whether `E[w_t]` should be the sample mean instead. The test checks the fast
recursion against the literal refit-OLS reference, so the recursion is verified;
the scale choice is a convention.

### 1.6 Critical values **[not implemented]**

The book gives no Brown-Durbin-Evans boundary constants, so the code returns `S_t`
and `z_t` without a rejection rule. The test of a known injected mean shift
checks only the qualitative property that `|S_t|` after the break exceeds `|S_t|`
before it.

**Test status [checked by test].** `test_cusum_matches_naive_*` compares `w_t` and
`S_t` against a literal refit-OLS loop on seeded random inputs (mean model and a
two-column design). `test_cusum_grows_after_known_mean_shift` injects a shift of
3 sd at `t = 100` and checks `|S_last| > 1.5 max_{t<100} |S_t|` (last index 199). Observed values (seed 42; pinned in `tests/test_doc_claims.py`):
`max_{t<100}|S_t| = 3.78`, `|S_last| = 117.6`.

---

## 2. Chu-Stinchcombe-White CUSUM on levels (AFML 17.3.2)

### 2.1 Statistic **[proved here, under the stated H0]**

`S_{n,t} = (y_t - y_n) / (sigma_t sqrt(t - n))`, with
`sigma_t^2 = t^{-1} sum_{i=1}^{t} (y_i - y_{i-1})^2` (0-based; the book's
`(t-1)^{-1} sum_{i=2}^{t}` in 1-based indexing).

Under H0 (`Delta y_i` i.i.d. with variance `sigma^2`):
`y_t - y_n = sum_{i=n+1}^t Delta y_i` has variance `(t-n) sigma^2`. Since
`sigma_t^2 -> sigma^2` almost surely (law of large numbers for `Delta y_i^2`,
which holds when `E[Delta y_i^4] < inf`), Slutsky's lemma gives
`S_{n,t} -> N(0, 1)` in distribution as `t - n -> inf`.

### 2.2 Critical value **[claimed from the book only, with a reading of the radical]**

`c_alpha[n, t] = sqrt(b_alpha + log(t - n))`, with `b_0.05 = 4.6`, taken as given; its source page has not been checked. (Monte Carlo,
Chu, Stinchcombe and White; Homm and Breitung 2012). The book's printed radical
is ambiguous. We read the radical as covering `b + log(t-n)`. The code exposes
`b` as a parameter, so the reading can be changed without touching the kernel.

The test `test_csw_critical_value_book_constant` checks the implemented formula at
two points. It does not verify the Monte Carlo constant.

### 2.3 Supremum over reference points **[proved here, as a definition]**

`S_t = sup_{n in [0, t-1]} S_{n,t}` (the book's backward-shifting variant). The
supremum over a finite set is attained, so `S_t` is well defined. The
implementation returns the maximising `n*_t`, and the rejection rule uses
`c_alpha[n*_t, t]`. That rule is our convention, since the book gives none for the
supremum.

**Test status [checked by test].** `test_csw_fixed_reference_matches_naive` and
`test_csw_sup_matches_naive` compare against literal loops (the sup test also
checks the argmax). `test_csw_detects_late_drift` shows a drift of 1.0 over the
second half of a 300-point series gives `S_{140,299} = 6.48` against `c = 3.11`,
and no rejection without drift. The value 6.48 (6.4823) and the `c` of 3.11 are pinned in `tests/test_doc_claims.py`.

---

## 3. Augmented Dickey-Fuller and SADF (AFML 17.4.2)

### 3.1 The regression **[proved here, as a specification]**

The book's regression (17.4.2) is
`dy_t = alpha + beta y_{t-1} + sum_{l=1}^{L} gamma_l dy_{t-l} + eps_t`.
Snippet 17.2 puts the lagged level first, so `beta` is column 0. The deterministic
block depends on the constant option: `'nc'` none, `'c'` intercept, `'ct'`
intercept plus trend, `'ctt'` intercept plus trend and trend squared. The trend is
the absolute sample position (as in the snippet, which does not reset it for
sub-windows).

### 3.2 Tau statistic **[proved here]**

OLS gives `beta_hat = (X'X)^{-1} X'y`. With `sigma_hat^2 = SSR / (n - p)` and
`Var(beta_hat) = sigma^2 (X'X)^{-1}`, the standard error of `beta_0` is
`sqrt(sigma_hat^2 [(X'X)^{-1}]_{00})`, and `tau = beta_hat_0 / se(beta_hat_0)`.
This is the textbook OLS result.

**Null distribution [claimed from the book only].** Under the unit-root null,
`tau` follows the Dickey-Fuller distribution, not Student's t. We did not verify
this.

### 3.3 SADF definition **[proved here, as a definition]**

For end point `t` (0-based) and minimum window `m` (number of levels), the book's
`SADF_t = sup_{t0 in [1, t-tau]} ADF_{t0,t}` becomes
`SADF_t = max_{s in [0, t - m + 1]} tau(y[s..t])`. The set of start points is
finite, so the maximum exists. The NaN prefix `t < m - 1` is by construction.

**Lemma (regression count) [proved here].** For a series of length `T` with
minimum window `m`, the number of ADF regressions is

    sum_{t = m-1}^{T-1} (t - m + 2) = (T - m + 1)(T - m + 2) / 2,

which is the book's count with `tau = m - 1` (1-based), and is `O(T^2)`.

**Admissibility of the window size [proved here].** The regression has
`n = (window length) - 1 - L` rows and `p = 1 + L + d` columns (`d` the number of
deterministic terms). The code requires `n > p`, that is `m >= 2L + d + 3`. The
guard in `sadf` implements this.

### 3.4 Exponential growth and explosiveness **[claimed from the book only]**

AFML 17.4.2.3 and the cited Phillips, Wu and Yu (2011) theory: for an
explosive AR(1) (`rho > 1`), the tau statistic diverges with the sample size, while
under a random walk the supremum stays `O_p(1)`. We did not prove this here.

**Test status [checked by test].** `test_adf_matches_naive` compares against a
design built from first principles and `np.linalg.lstsq`, for all four constant
options and `L in {0,1,3}`. `test_sadf_matches_naive` compares the full SADF
series for two constants. `test_sadf_random_walk_vs_explosive` on seeded inputs:
random walk `max SADF = 1.37`; exponential growth `exp(0.02 t) (1 + 0.01 eps)`
gives `SADF_{199} = 14.55` (pinned in `tests/test_doc_claims.py`, together with the random-walk value 1.37). These numbers show the expected separation on this
sample. They are not a size or power study.

### 3.5 Log prices **[claimed from the book only]**

AFML 17.4.2.1 argues for log prices (multiplicative regime changes keep the
regression coefficients stable). The functions accept whatever series is passed.
They do not take logs, and the docstrings say so.

---

## 4. Summary

| Result | Status |
|---|---|
| RLS update (Sherman-Morrison), `beta_t`, `P_t`, `f_t` | proved here; checked by test (`test_cusum_matches_naive_mean_model`, `test_cusum_matches_naive_with_features`) |
| Recursive residual mean 0 and variance `sigma^2` | proved here |
| Independence of recursive residuals | claimed from the book only (BDE 1975) |
| BDE `S_t` null variance `t - k` (book: `t - k - 1`) | proved here under 1.3; convention discrepancy noted |
| BDE critical boundaries | not in the book text; not implemented |
| CSW `S_{n,t} -> N(0,1)` | proved here (Slutsky) |
| CSW critical value `sqrt(b + log(t-n))`, `b = 4.6` | claimed from the book only; radical read as one term |
| CSW sup form | definition proved here; rejection rule is our convention |
| ADF tau as OLS t-statistic | proved here (textbook OLS) |
| ADF null distribution | claimed from the book only |
| SADF definition, regression count `(T-m+1)(T-m+2)/2` | proved here |
| Explosive tau diverges | claimed from the book only (PWY 2011) |
| Fast kernels equal literal references | checked by test (`test_cusum_matches_naive_mean_model`, `test_csw_sup_matches_naive`, `test_adf_matches_naive`, `test_sadf_matches_naive`) |
