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

---

## 5. Chow-type Dickey-Fuller DFC and SDFC (AFML ch. 17)

### 5.1 Model **[claimed from the book only (not verified against the text)]**

`dy_t = delta * y_{t-1} * D_t + eps_t` with no intercept and no lags, where `D_t = 0` for
`t < tau* T` and `D_t = 1` after. The section number is not verified: section 2 of this note
already cites 17.3.2 for Chu-Stinchcombe-White.

### 5.2 Break convention **[proved here, as a definition]**

The code sets the first post-break position to `b = ceil(tau T)`, with `T` the series length
and `t` the 0-based index. `D_t = 1` exactly when `t >= b`. This is the literal reading of
`D_t = 0` for `t < tau T`. Before the ceiling, `tau T` is rounded to 9 decimals, because the
float product `0.07 * 100` is slightly above 7, and a plain ceiling would move the break
from 7 to 8. `b = 0` is the same model as `b = 1`, since there is no regressor at `t = 0`.

### 5.3 DFC as an OLS t-statistic **[proved here, textbook OLS]**

Rows run over `t = 1..T-1`. Let `x_t = y_{t-1} 1{t >= b}` and `dy_t = y_t - y_{t-1}`. With one
regressor and no intercept, `(X'X)^{-1} = 1 / sum x_t^2`, so

    delta_hat = sum x_t dy_t / sum x_t^2,
    sigma_hat^2 = SSR / (T - 2),          SSR = sum (dy_t - delta_hat x_t)^2,
    se(delta_hat) = sqrt(sigma_hat^2 / sum x_t^2),
    DFC(tau) = delta_hat / se(delta_hat).

The pre-break rows have `x_t = 0`. They add to the residual sum of squares only, so the
residual variance uses all `T - 1` rows. This is our reading of the model as written.
The kernel `_dfc_at` computes these sums directly. It uses the same variance guard as
`_adf_tau`.

### 5.4 Scale invariance **[proved here]**

Replace `y` by `c y`. Then `x_t` and `dy_t` scale by `c`, so `delta_hat` is unchanged,
`SSR` scales by `c^2`, `sigma_hat` by `c`, and `sqrt(sum x_t^2)` by `c`. Hence
`se(delta_hat)` is unchanged and `DFC` is invariant. The code rescales with the ADF guard
`_rescale_for_adf`, so the sums stay finite for `|y|` outside `[1e-100, 1e100]`.

### 5.5 SDFC as a finite maximum **[proved here, as a definition]**

`DFC(tau)` depends on `tau` only through `b = ceil(tau T)`. So
`sup_{tau in [tau0, 1 - tau0]} DFC(tau)` is the maximum of `DFC(b)` over the integers
`b in [b_lo, b_hi]`, with `b_lo = max(ceil(tau0 T), 1)` and `b_hi = min(ceil((1 - tau0) T), T - 1)`.
The clamp keeps at least one post-break observation. The maximum exists because the set is
finite.

### 5.6 Null distribution and rejection rule **[not implemented]**

The null distribution of `DFC` and `SDFC`, and their critical values, are not derived here.
The functions return statistics with no p-values and no rejection rule.

### 5.7 Test status **[checked by test]**

- `test_chow_dfc_break_is_ceil_of_tau_times_t` checks five `(T, tau)` pairs against a literal
  `np.linalg.lstsq` regression. It includes the float case `0.07 * 100`. It also checks that
  the break one step later gives a different value.
- `test_chow_dfc_matches_direct_ols_at_mid_break` and `test_chow_dfc_is_scale_invariant` cover
  the closed form and section 5.4.
- `test_sdfc_equals_max_of_direct_loop` compares every admissible break with the literal loop,
  and checks the argmax index. `test_sdfc_bounds_a_fine_tau_grid_and_is_attained_at_its_argmax`
  checks that a 701-point `tau` grid never exceeds the supremum.
- `test_sdfc_is_larger_for_a_regime_switch_to_explosive_growth` compares a random walk with a
  series that grows at 1.01 per step after `t = 150`.
- `test_sdfc_undefined_breaks_are_nan_and_skipped`, `test_sdfc_refuses_when_every_admissible_break_is_undefined`
  and `test_chow_dfc_undefined_when_no_post_break_level_is_nonzero` cover the undefined case.
  The refusal tests cover `tau0`, `tau`, a break after the last observation, short, constant
  and descending-index inputs.

**Measured [a shared 4-vCPU machine, 2026-10-13]**: running the body of
`test_sdfc_is_larger_for_a_regime_switch_to_explosive_growth` (seed 7, `n = 300`, switch at
`t = 150`) gives `SDFC = 11.5959` with argmax `b = 96`. The same shocks with `rho = 1` give
`SDFC = 1.4516`. `DFC` at the true break `b = 150` is `11.51`. The argmax does not locate the
break date in this sample. The path is nearly flat over the post-break range, so the statistic
measures the explosive regime, not its start date. These are single seeded runs, not a size or
power study.

---

## 6. Quantile and conditional ADF (AFML 17.4.2)

### 6.1 Definitions **[claimed from the book only (not verified against the text)]**

For end point `t`, with `A_t` the set of ADF statistics defined in 6.2:

    QADF_{t,q} = q-quantile of A_t,
    dispersion = QADF_{t,q+v} - QADF_{t,q-v},
    CADF_{t,q} = mean of { a in A_t : a >= QADF_{t,q} },
    conditional dispersion = standard deviation of that same set.

The section number is not verified. The conditional dispersion is not returned by `cadf`.

### 6.2 The set A_t **[proved here, as a definition]**

`A_t = { ADF(y[s..t]) : s in [0, t - m + 1] }` for minimum window `m`, as in section 3.3.
`max A_t = SADF_t` by definition. Using start points `0..t-m` would drop the shortest
admissible window, so `max A_t` would not equal `SADF_t` in general. The code uses the
section 3.3 range.

### 6.3 Quantile rule **[proved here, as a definition]**

For sorted values `a_(1) <= ... <= a_(n)` and `h = (n-1) q`, with `lo = floor(h)` and
`gamma = h - lo`, the code returns `a_(lo+1) + gamma (a_(lo+2) - a_(lo+1))`. This is the rule
of `np.quantile` with `method="linear"`. The code evaluates the blend from the nearer end,
which is the form numpy's `_lerp` uses (read from the installed numpy 2.5.4 source). Two
consequences:

- `q = 1` gives `h = n - 1`, so the result is `a_(n)`, the maximum, exactly.
- `QADF_{t,q}` is nondecreasing in `q`, because the interpolation of sorted values is.
  So the dispersion `QADF_{t,q+v} - QADF_{t,q-v}` is nonnegative.

### 6.4 Bounds and identities **[proved here]**

Every element of the tail is at least `QADF_{t,q}`, and every element of `A_t` is at most
`max A_t = SADF_t`. Hence

    QADF_{t,q} <= CADF_{t,q} <= SADF_t.

At `q = 1`, `QADF_{t,1} = SADF_t` by 6.3. The tail is then the set of elements equal to the
maximum. With no ties it is the maximum alone, so `CADF_{t,1} = SADF_t` exactly. With exact
ties, the mean of equal floats can differ from the maximum by rounding.

### 6.5 Test status **[checked by test]**

- `test_qadf_matches_naive_quantile` and `test_cadf_matches_naive_tail_mean` compare both
  series with literal loops built on `np.quantile` and `naive_adf`, for two constants and two
  levels of `q`.
- `test_qadf_and_cadf_at_q_one_equal_sadf_exactly` checks equality, not closeness, at `q = 1`.
- `test_quantile_rule_is_numpy_linear` compares the interpolation with `np.quantile` for
  sizes 1 to 40 and several `q`.
- `test_qadf_dispersion_matches_naive_band` covers the band. `test_qadf_is_nondecreasing_in_q_and_cadf_is_bracketed`
  checks the monotonicity and the bounds in 6.4.
- Refusal tests cover `q`, the band `v`, the window length, the lag, the constant and
  constant input. `test_qadf_and_cadf_pandas_index_and_numpy_agree` covers index handling.

**Measured [a shared 4-vCPU machine, 2026-10-13, `PYTHONPATH=src python benchmarks/bench_structural_breaks.py`]**:
at `n = 300`, `min_length = 30`, `lag = 1`, `q = 0.95`, the literal loop takes 5.6927 s and
the numba kernel takes 0.111150 s (best of 3), a 51x speedup. The same run gives 5.7630 s and
0.111788 s for `sadf`. At `n = 1000` with `min_length = 50`, `qadf` takes 4.064 s and `sadf`
takes 4.002 s. These are single runs on a shared machine, so the timings carry some noise.

---

## 7. Sub- and super-martingale tests (AFML 17.4.3)

### 7.1 Definitions **[claimed from the book only (not verified against the text)]**

For end point `t` and start point `t0`, fit by OLS on the window `[t0, t]`:

- `poly1`: `y_t = a + g t + b t^2 + e`,
- `poly2`: `log y_t = a + g t + b t^2 + e`,
- `exp`: `log y_t = log(a) + b t + e`,
- `power`: `log y_t = log(a) + b log(t) + e`.

The statistic is `SMT_t = sup_{t0} |b_hat| / (se(b_hat) (t - t0)^phi)` with `phi` in `[0, 1]`.
The section number is not verified. The absolute value in the statistic is the form stated for
this module. Whether the book's test is signed is not verified.

### 7.2 Time index and window set **[proved here, as definitions]**

Time is the 1-based position in the full series, so `t` runs over `1..T`. The 1-based convention
keeps `log(t)` defined for `power`. The start points are `t0 in [0, t - m + 1]`, the set of
section 3.3, so that `m` (the minimum window in levels) means the same thing in every function
of the module. The code uses `t - t0` as the difference of positions, which equals the number
of steps in the window.

### 7.3 Centring the time regressor **[proved here]**

The code centres the regressor on the window mean, `z = t - c`, with `c` the mean of `t` (or of
`log t` for `power`) over the window. Centring does not change the tested coefficient or its
standard error:

- For the quadratic kinds, `t = z + c` gives `a + g t + b t^2 = (a + g c + b c^2) + (g + 2 b c) z + b z^2`.
  The coefficient of `z^2` is `b`. The design matrix changes by a unit upper-triangular `M`
  with `M^{-1}` having last row `(0, 0, 1)`, so `[(X'X)^{-1}]_{33}` is unchanged. The residuals,
  the SSR and `sigma_hat^2` are unchanged as well.
- For the linear kinds the same argument applies to the slope.

The power kind is centred in `log t`, not in `t`. The positions are the global 1-based ones
before centring, so the regression is the one in 7.1.

### 7.4 Scale invariance **[proved here]**

For `poly1`, replacing `y` by `c y` scales `b_hat` and `se(b_hat)` by `c`, so the ratio is
unchanged. The code rescales `y` as the ADF code does. For the log kinds, `log(c y) = log y + log c`
only shifts the intercept, so `b_hat` and its standard error are unchanged.

### 7.5 Effect of `phi` **[proved here]**

Each window has `t - t0 >= m - 1 >= 1` steps, so `(t - t0)^phi >= 1` for `phi >= 0`. Hence
`SMT_t(phi) <= SMT_t(0)` for every `phi` in `[0, 1]`.

### 7.6 Pure exponential growth and the choice of kind **[proved here]**

For `y_t = exp(c t)`, `log y_t = c t` is linear in `t`, so the quadratic coefficient of the
`poly2` fit is zero in exact arithmetic. The `poly2` test therefore has no signal from pure
exponential growth. The `exp` kind is the test designed for it. The measured values below
show the difference.

### 7.7 Test status **[checked by test]**

- `test_smt_matches_naive_loop_with_raw_positions` compares every kind, with `phi` 0 and 0.5,
  against a loop that uses `np.linalg.lstsq` on the raw 1-based positions. The match confirms
  the centring in 7.3. The largest difference in a run on 2026-10-13 was 3.2e-11.
- `test_smt_is_scale_invariant` checks 7.4 at `1e200` and `1e-200`.
- `test_smt_with_phi_one_is_no_larger_than_phi_zero` checks 7.5.
- `test_smt_exponential_growth_exceeds_random_walk_for_exp_kind` and
  `test_smt_poly2_does_not_flag_pure_exponential_growth` pin the seed-0 values below.
- Refusal tests cover the kind, `phi`, the window length for each kind, a non-integer or
  oversized `min_length`, and non-positive values under the log kinds, which name the kind and
  the count. `test_smt_pandas_index_and_nan_prefix` covers index handling.

### 7.8 Measured values **[measured, a shared 4-vCPU machine, 2026-10-13]**

Seeded series of length 200 with `min_length = 30` and `phi = 0`. The exponential series is
`exp(0.01 t) (1 + 0.01 eps)` with seed 0. The walk is `exp(cumsum(0.01 eps))` with seed 100.

| Kind | Growth series, end point | Growth series, max | Walk, max |
|---|---|---|---|
| `exp` | 845.1877 | 845.1877 | 32.8009 |
| `poly1` | 74.4752 | 75.5309 | 16.0536 |
| `poly2` | 1.7761 | 4.0016 | 15.8933 |
| `power` | 118.379 | 119.0495 | 30.1005 |

The walk column is the same walk (seed 100) for every kind, so each row compares the growth
series with the walk under one kind. Seeds 1 and 2 give the same ordering for `exp` and `poly1`
(growth maxima 877.6 and 843.9 for `exp`, against walk maxima 42.4 and 29.1). For `poly2` the
growth maximum is below the walk maximum for all three seeds: 2.10 to 4.00 for the growth series
against 15.9 to 18.4 for the walk. These are single seeded runs, not a size or power study. No
critical values are derived.

**Benchmark [measured, a shared 4-vCPU machine, 2026-10-13, `PYTHONPATH=src python benchmarks/bench_structural_breaks.py`]**:
at `n = 200`, `min_length = 30`, kind `poly1`, the literal loop takes 0.5590 s and the numba kernel
takes 0.016012 s (best of 3), a 35x speedup. At `n = 1000` with `min_length = 50`, `smt` takes
1.787 s. Single runs on a shared machine, so the timings carry some noise.

### 7.9 Not implemented **[not implemented]**

The null distribution and critical values of `SMT` are not derived here. The functions return
statistics with no rejection rule.
