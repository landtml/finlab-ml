# finlab.fracdiff: derivations and status of each claim

Component under test: [`src/finlab/fracdiff.py`](../../src/finlab/fracdiff.py) (AFML ch. 5).
Tests: [`tests/test_fracdiff.py`](../../tests/test_fracdiff.py). Benchmark: [`benchmarks/bench_fracdiff.py`](../../benchmarks/bench_fracdiff.py).

Tags used below:

- **[proved here]**: a derivation is given in this document.
- **[checked by test]** (the claim names its test function, for example `test_ffd_kernel_matches_naive_dot`): the implementation agrees with a naive reference or with a closed form on seeded inputs. This is evidence on fixed data, not a proof.
- **[claimed from the book only]**: from AFML, not proved or tested here.

Notation. `B` is the backshift operator, `B^k X_t = X_{t-k}`. Generalised binomial coefficients are `C(x, k) = x(x-1)...(x-k+1) / k!` for real `x` and integer `k >= 0`, with `C(x, 0) = 1`.

---

## 1. Binomial series and weights (AFML 5.4)

**Equation 5.1 (book).** For real `d`,

    (1 - B)^d = sum_{k>=0} omega_k B^k,     omega_k = (-1)^k C(d, k).

**Equation 5.2 (book, 5.4.2).** The weights satisfy `omega_0 = 1` and `omega_k = -omega_{k-1} (d - k + 1) / k`.

**Proposition 1.1 [proved here].** Equation 5.2 follows from Equation 5.1.

*Proof.* From the product form `C(d, k) = d(d-1)...(d-k+1) / k!`, for every real `d` and `k >= 1`,

    C(d, k) = C(d, k-1) * (d - k + 1) / k,

with no division by `C(d, k-1)`. Multiplying by `(-1)^k` gives `omega_k = (-1)^k C(d,k) = -omega_{k-1} (d - k + 1) / k`. With `omega_0 = C(d,0) = 1`, this is Equation 5.2. ∎

**Proposition 1.2 [proved here] (special cases).**

- `d = 0`: `C(0, k) = 0` for `k >= 1`, so `omega = (1, 0, 0, ...)`.
- `d = 1`: `C(1, k) = 0` for `k >= 2`, and `C(1,1) = 1`, so `omega = (1, -1, 0, ...)`.

*Proof.* Direct from the definition of `C(x,k)`, since the falling factorial contains the factor `(x - 0) = 0` when `x = 0`, and the factor `(x - 1) = 0` when `x = 1`, for `k` large enough. ∎

**[checked by test]** `get_weights(0, 6) == [1, 0, 0, 0, 0, 0]`, `get_weights(1, 6) == [1, -1, 0, 0, 0, 0]`, `get_weights_ffd(1) == [1, -1]`, and `get_weights(d, 40)` matches `(-1)^k binom(d, k)` from scipy for several `d` (`test_weights_*`).

---

## 2. Partial sums of the weights (FFD window sum property)

**Proposition 2.1 [proved here].** For integer `L >= 0`,

    S_L := sum_{k=0}^{L} omega_k = (-1)^L C(d-1, L).

*Proof.* Induction on `L`. For `L = 0`, `S_0 = omega_0 = 1 = C(d-1, 0)`. For `L >= 1`, `S_L - S_{L-1} = omega_L = (-1)^L C(d, L)`. Pascal's rule for generalised binomials, `C(x, L) = C(x-1, L) + C(x-1, L-1)` (valid for every real `x`), with `x = d` gives `C(d, L) = C(d-1, L) + C(d-1, L-1)`. Therefore `(-1)^L C(d-1, L) - (-1)^{L-1} C(d-1, L-1) = (-1)^L [C(d-1, L) + C(d-1, L-1)] = (-1)^L C(d, L) = omega_L`. The induction hypothesis then gives `S_L = (-1)^L C(d-1, L)`. ∎

**Proposition 2.2 [proved here] (behaviour for 0 < d < 1).** For `0 < d < 1` and `L >= 1`,

    S_L = (-1)^L C(d-1, L) = prod_{i=1}^{L} (1 - d/i) in (0, 1],

and `S_L` is strictly decreasing in `L` with limit `0`.

*Proof.* Write `x = d - 1 in (-1, 0)`. Then `C(x, L) = prod_{i=1}^{L} (x - i + 1) / L!`. Each factor `x - i + 1 = d - i` is negative because `i >= 1 > d`. So the sign of `C(x, L)` is `(-1)^L`, and `(-1)^L C(x, L) = prod_{i=1}^{L} (i - d) / i = prod_{i=1}^{L} (1 - d/i)`. Each factor lies in `(0, 1)`, so the product lies in `(0, 1]`, is strictly decreasing in `L`, and tends to `0` because `sum_i d/i` diverges. ∎

**Consequence (window tail mass) [proved here].** For `0 < d < 1`, `omega_k <= 0` for all `k >= 1`, and `sum_{k>=0} omega_k = 0` (the limit of `S_L`). Hence the omitted tail satisfies `sum_{k>L} |omega_k| = S_L = prod_{i=1}^{L}(1 - d/i)`, which is of order `L^{-d} / Gamma(1-d)`. For a bounded series `|X_t| <= M`, the truncation error of the FFD output is at most `M * S_L` in absolute value. This is a bound on the sum of the omitted weights, not on each weight, and the FFD cut-off `thres` acts on each weight separately.

**[checked by test]** The FFD partial sums equal `(-1)^L C(d-1, L)` to `1e-9` relative, for `d` in `{0.2, 0.4, 0.7, 1.3}` (`test_ffd_weight_partial_sums_have_closed_form`). For `0 < d < 1` they lie in `(0, 1]` and decrease, as Proposition 2.2 requires.

---

## 3. Monotonicity of the FFD weights (why the cut-off is well-defined)

**Proposition 3.1 [proved here].** For `0 < d < 1`, `|omega_k|` is strictly decreasing for `k >= 1`.

*Proof.* `|omega_k| / |omega_{k-1}| = |d - k + 1| / k = (k - 1 - d) / k` for `k >= 2`, and `k - 1 - d > 0` for `k >= 2` since `d < 1`. Also `(k - 1 - d)/k < 1`. For `k = 1`, `|omega_1| = d < 1 = |omega_0|`. ∎

Consequence: the rule "keep weights while `|omega_k| >= thres`" in `get_weights_ffd` (Snippet 5.3, `if abs(w_) < thres: break`) keeps the prefix up to the first weight below `thres`, and the window width is the length of that prefix. For `d` in `(0, 1)` this is the prefix of a monotone sequence. For other `d` the magnitudes need not be monotone, and the code still stops at the first crossing, as the snippet does. The stop is finite for any `thres > 0`, because `|omega_k|` is of order `k^{-d-1}`, which tends to 0 when `d` is not a nonnegative integer. For a nonnegative integer `d`, `omega_k = 0` after `k = d + 1`.

**[checked by test]** `test_ffd_window_respects_threshold`: every kept weight is at least `thres`, and the first dropped one is below it.

---

## 4. Expanding-window weight loss (AFML 5.5.1)

**Definition (book).** `lambda_t = sum_{j > t} |omega_j| / sum_{j >= 0} |omega_j|` is the fraction of weight lost when `X_t` is computed with only the lags available at `t`. The sums run over the full-length weight vector of the series (length `n`). The first output index is `t* = min{t : lambda_t <= thres}`.

**Proposition 4.1 [proved here].** `lambda_t` is non-increasing in `t`, equals `0` at `t = n-1`, and so `t*` is well-defined.

*Proof.* `lambda_t` is the tail sum of nonnegative terms divided by a constant, so it decreases as the tail shrinks. At `t = n-1` the tail is empty, so `lambda_{n-1} = 0 <= thres`. ∎

*Remark.* Snippet 5.2 computes the same quantity by `np.cumsum(abs(w))` over the reversed weights, then `skip = count(w_ > thres)`. Its index bookkeeping differs by one from the text. This module uses the text definition directly. `test_expanding_matches_naive` recomputes `lambda_t` from scratch for each `t`.

**[checked by test]** `frac_diff(x, 1.0)` equals the first differences of `x` for `t >= 1`, and `frac_diff(x, 0.0) = x`, with the default `thres = 0.01`. `thres >= 1` keeps every observation (`test_frac_diff_*`, `test_thres_one_*`). For `d = 1` the expanding window needs only one lag. Here `lambda_0 = 1/2` and `lambda_1 = 0`, so `t* = 1`.

---

## 5. Algebraic correctness of the kernels

**Proposition 5.1 [proved here].** For a window of `L` weights, the blocked kernels compute `out[t] = sum_{k=0}^{L-1} omega_k x_{t-k}` for `t >= L-1`, with the accumulation order `k = 0, 1, ...` for each `t`. The result is independent of the block size and of the thread count, because each `out[t]` is written by exactly one block, and the order within it is fixed.

*Proof.* Blocks partition the index range `[L-1, n)` into disjoint intervals, and each block loops over `k` in increasing order. Each `out[t]` is written only in the block that contains `t`, with the same operation sequence (`0 + omega_0 x_t`, then `+ omega_1 x_{t-1}`, and so on). The arithmetic is floating-point and deterministic for a fixed order. ∎

**[checked by test]** The FFD kernel matches a naive per-observation `np.dot` implementation to `1e-10` for several `d`, with and without NaN gaps (`test_ffd_kernel_matches_naive_dot`, `test_ffd_handles_nans_*`). The expanding kernel matches a naive loop (`test_expanding_matches_naive`).

**Scope of the NaN rule [proved here].** Values are forward-filled before the convolution. The output is set to NaN wherever the original input was NaN, and in the first `L-1` rows. This matches the book's `fillna(method='ffill')` plus the `isfinite` skip of Snippets 5.2 and 5.3, but the output keeps the input length instead of dropping rows.

---

## 6. Memory and stationarity (AFML 5.6)

**Claim (book).** `d*` is the minimum `d` with a stationary FFD series, and `0 < d* << 1` preserves memory. The correlation with the original series `corr(X_t, tilde X_t(d))` decreases in `d`, and at the ADF threshold for E-mini futures the correlation is about `0.995` (Figure 5.5).

**Status.** These are claimed from the book only. They rest on empirical ADF tests on futures and are not proved here.

**[checked by test]** On five seeded log-price random walks of length `6000`, with `thres = 1e-4` and `d` in `{0, 0.2, ..., 1}`, the correlation with the original series is `1` at `d = 0` up to rounding (the FFD output is the input itself), decreases within each seed, and its average decreases (`test_memory_correlation_decreases_with_d`). This is a check on fixed data. It does not prove monotonicity for every series.

**Minimum-d search [status].** `find_min_d` scans a grid and returns the first `d` where the injected test returns `p <= adf_pvalue`. The grid scan does not assume monotone passing. The search is correct for the injected test by construction, and the test `test_find_min_d_returns_first_passing_grid_value` checks it. Whether the injected test's p-value means what the book means by stationarity is the caller's responsibility. The book's ADF critical values (MacKinnon) are not in the text, so no ADF is bundled.

---

## 7. Summary of status

| Item | Status |
| --- | --- |
| Binomial recursion for omega_k (Eq. 5.2) | proved here |
| d = 0 and d = 1 weights | proved here; checked by test (`test_weights_d0_is_identity`, `test_weights_d1_is_first_difference`) |
| Partial sums `(-1)^L C(d-1, L)` (Prop. 2.1) | proved here (Pascal's rule); checked by test (`test_ffd_weight_partial_sums_have_closed_form`) |
| Partial sums in (0,1] and decreasing for 0<d<1 (Prop. 2.2) | proved here; checked by test (`test_ffd_weight_partial_sums_have_closed_form`) |
| Omitted tail mass for 0<d<1 | proved here |
| Monotone `abs(omega_k)` for 0<d<1 (Prop. 3.1) | proved here |
| Expanding-window `lambda_t` monotone, `t*` well-defined | proved here |
| Blocked kernels match naive dot products | proved here (ordering argument); checked by test (`test_ffd_kernel_matches_naive_dot`, `test_expanding_matches_naive`) |
| Memory correlation decreases in d | checked by test (`test_memory_correlation_decreases_with_d`) on five seeded random walks; claimed from the book in general |
| FFD yields stationarity at `d*` (AFML 5.6) | claimed from the book only |
| Correlation about 0.995 at the ADF threshold (Figure 5.5) | claimed from the book only (futures data) |
| `find_min_d` returns the first passing grid value | checked by test (`test_find_min_d_returns_first_passing_grid_value`), with an injected test |
