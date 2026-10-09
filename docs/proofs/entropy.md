# finlab.entropy: derivations and status (AFML ch. 18)

Component under test: [`src/finlab/entropy.py`](../../src/finlab/entropy.py).
Tests: [`tests/test_entropy.py`](../../tests/test_entropy.py).
Benchmark: [`benchmarks/bench_entropy.py`](../../benchmarks/bench_entropy.py).

Status tags: **[proved here]**, **[checked by test]** (numerical evidence on
seeded inputs), **[claimed from the book only]** (stated in AFML or its cited
papers and not re-derived here).

---

## 1. Shannon entropy (AFML 18.2)

`H[X] = -sum_x p[x] log2 p[x]`, with `0 <= H[X] <= log2|A|`, equality on the right
iff `p` is uniform. **[proved here]**: `H >= 0` since `-log2 p >= 0`. The upper
bound follows from Jensen's inequality applied to the concave `log2`:
`H = sum p log2(1/p) <= log2 sum p (1/p) = log2 |A|`, with equality iff `p`
is uniform on `A`.

## 2. Plug-in estimator (AFML 18.3, Snippet 18.1)

For a message `x_1^n` over an alphabet of `K` symbols and word length `w`, let
`p_hat_w(y)` be the frequency of word `y` among the `m = n - w + 1` overlapping
words, and

    H_hat_{n,w} = -(1/w) sum_y p_hat_w(y) log2 p_hat_w(y).                (18.3)

**Choice of window count [a deviation from the snippet].** Snippet 18.1 iterates
`i in range(w, n)` and so omits the final word `x_{n-w+1}^n`. We use all `m`
windows, normalised by `m`. This is the empirical distribution of the complete
set of length-`w` substrings.

**Bounds [proved here].** The distribution over words has at most `min(m, K^w)`
atoms, so by 18.2, `H(p_hat_w) <= log2 min(m, K^w) <= w log2 K`. Dividing by `w`:
`0 <= H_hat_{n,w} <= log2 K`.

**Special cases [proved here].**
- A constant message has one word with `p = 1`, so `H_hat = 0`.
- If `K^w <= m` and the words are equidistributed, `H_hat = log2 K`. The
  uniform 4-symbol test below meets this.

**Consistency [claimed from the book only].** For a stationary ergodic process,
`H_hat_{n,w} -> H_w / w` as `n -> inf` for fixed `w`, and the entropy rate
`lim_w H_w / w` is the Shannon rate (Gao et al. 2008, cited in 18.3). We did not
prove this. The estimator is biased for `n` not much larger than `K^w`.

**Overflow guard [proved here].** The word code `sum_j c_j K^{w-1-j}` lies in
`[0, K^w)`, so an int64 code is safe if `K^w < 2^62`. The wrapper rejects larger
values instead of wrapping.

## 3. Lempel-Ziv dictionary (AFML 18.4, Snippet 18.2)

The parse is deterministic: the initial phrase is `x_0`, and at position `i` the
next phrase is the shortest substring `x_i^j` that is not already in the
dictionary. A trailing repetition is not added. The pseudocode in Snippet 18.2
is transcribed literally, including the `i = j + 1` update after the inner loop.

**Lemma (distinct phrases) [proved here].** Every phrase added is absent from the
dictionary when it is added, so the dictionary entries are pairwise distinct. Each
phrase has length at least one and phrases do not overlap, so the phrase count
satisfies `c <= n`.

**Lempel-Ziv entropy estimate (`lempel_ziv_entropy`).** The book describes the LZ
rate qualitatively ("the number of items in a Lempel-Ziv dictionary relative to the
length of the message") and gives no formula. We use `c log2 c / n`, which is the
standard LZ78 normalisation from the literature (Ziv and Lempel 1978). **This
formula is not in AFML.** It is labelled external here, and it is biased for short
messages.

## 4. Kontoyiannis / Gao et al. (2008) estimator (AFML 18.4, Snippet 18.4)

### 4.1 Match length **[proved here]**

For centre `i` and window length `n`, Snippet 18.3 defines `L_i^n` as one plus the
length of the longest substring `x_i^{i+l}` that also appears as `x_j^{j+l}` for some
`j in [i-n, i-1]`. The fast kernel stops at the first length without a match.
That is valid: if `x_i^{i+l}` has a match at `j`, then so does the prefix of
length `l-1` (same `j`), so matchability is monotone in the length. The result is
the same as the snippet's loop over all `l`.

### 4.2 Estimator **[proved here as a definition; checked by test]**

Expanding window (`n = i`, centres `i = 1..floor(N/2)`):

    H_hat_N = [ (1/M) sum_i log2(i+1) / L_i^i ]    (the snippet's h).

Sliding window (`n = w`, centres `i = w..N-w`):

    H_hat_{N,w} = [ (1/M) sum_i log2(w+1) / L_i^w ]

The snippet's `log2(n+1)` is kept as written. The snippet's `log2 n` version is the
one in the book's display equation, which we did not implement. The snippet also
defines a redundancy `r = 1 - h / log2 N`. That normalisation is not an alphabet
normalisation, so we do not return it.

**Odd-length caveat [proved here].** For the expanding window the centres are
`i <= floor(N/2)`, so when `N` is odd the last symbol is never a centre. It can
still lie inside a match. This is why the book asks for an even length.

### 4.3 Book examples (AFML 18.4) **[checked by test, with one discrepancy]**

Two claims from the text, using the snippet's formula:

- `10000111` and `10000110` have the same estimate (the final bit is irrelevant
  because of the unmatchable `11`). **Checked by test, exact equality.**
- The book states the entropy rates of `11100001` and `01100001` (as given, forward
  order) as 0.96 and 0.84. The snippet gives `0.9682` and `0.8432`. **The second
  matches to rounding. The first is 0.968 by the snippet, which rounds to 0.97,
  not 0.96.** We tried the expanding-window variants `log2(i)`, `log2(i+1)` (the
  snippet) and `log2(2i)` in the numerator, and none reproduces both examples. The most likely explanation is a
  truncated or typo'd printed value. The test asserts the snippet's value to
  0.001 and checks that it is within 0.01 of the printed 0.96.

### 4.4 Consistency and the bias/variance window **[claimed from the book only]**

Kontoyiannis (1998) proves `H_hat -> H` a.s. under stationarity, ergodicity and the
Doeblin condition. Gao et al. (2008) remove the Doeblin condition by using the
`log2(n+1)` modification, and recommend `N ~ n + (log2 n)^2` (e.g. `N = 2^8`
gives `n ~ 198`, `k ~ 58`). Those statements are cited from the book, not derived
here.

**Test status [checked by test].** `test_match_length_matches_literal_snippet`
compares the early-exit kernel with the literal loop (no early exit) for many
`(i, n)`. `test_kontoyiannis_expanding_matches_naive` and
`test_kontoyiannis_sliding_matches_naive` compare the full estimator with the
literal snippet on odd and even lengths and several windows. Agreement is to
`rel = 1e-12`.

## 5. Encoding schemes (AFML 18.5)

### 5.1 Binary encoding **[proved here]**

`c_t = 1{r_t > 0}` on `{r_t != 0}`. Zero returns are dropped as the book
prescribes. The codes are a deterministic function of the sign.

### 5.2 Quantile encoding **[proved here, with a tie caveat]**

With edges `e_k = Q(k/q)` (the empirical `k/q` quantiles of the reference sample,
`k = 1..q-1`) and `c(r) = #{k : e_k <= r}` (`numpy.searchsorted`, side 'right'):

- `c` is non-decreasing in `r`, takes values in `{0..q-1}`, and `c(r) = k` iff
  `e_k <= r < e_{k+1}`. **[proved here]**
- The in-sample bin counts are `n/q` up to ties and interpolation (linear
  quantiles can fall between observations). **Checked by test:** for `n = 1000`,
  `q = 10`, every bin has 99 to 101 points. The exact count is not guaranteed by
  the construction.

### 5.3 Sigma encoding **[proved here]**

With `lo = min r`, `K = ceil((max r - lo) / sigma)` (at least 1), code
`c(r) = floor((r - lo)/sigma)` clipped to `K - 1`. Each code except the last
covers an interval of width `sigma`. The clip is needed only when `max r` is an
exact multiple of `sigma`, where the unclipped code would be `K`. The test
`test_encode_sigma_bins_have_fixed_width` checks the intervals and the clip.

## 6. Entropy of a Gaussian (AFML 18.6)

For `X ~ N(0, sigma^2)`, the differential entropy is `h = (1/2) log(2 pi e sigma^2)`.
**[standard result; not implemented]**. The book prints `H ~ 1.42` for the standard
normal. With the natural logarithm, `(1/2) ln(2 pi e) = 1.4189`, so the printed
value is in nats, not bits. In bits it is `2.048`. This is noted here only for
readers who compare the book's numbers with the code. The Gaussian benchmark and
the entropy-implied volatility are out of scope.

## 7. Summary

| Result | Status |
|---|---|
| Shannon bounds `0 <= H <= log2 K` | proved here |
| Plug-in estimator bounds, constant and uniform cases | proved here |
| Plug-in consistency for stationary ergodic processes | claimed from the book only |
| Plug-in uses all `n-w+1` windows (snippet drops the last) | deviation; documented |
| Word-code overflow guard | proved here |
| LZ parse: distinct phrases, `c <= n` | proved here |
| LZ entropy `c log2 c / n` | **not in AFML**; standard LZ78 normalisation, labelled external |
| Match length monotone, early exit is exact | proved here; checked by test |
| Kontoyiannis estimator as implemented | proved here as definition; checked by test vs literal snippet |
| Kontoyiannis consistency, bias/variance rule | claimed from the book only (Kontoyiannis 1998; Gao et al. 2008) |
| Book's `10000111` / `10000110` equality | checked by test (exact) |
| Book's `0.96` for `11100001` | **discrepancy**: snippet gives 0.968; test checks 0.968 and 0.96 to 0.01 |
| Book's `0.84` for `01100001` | checked by test (0.8432) |
| Binary, quantile, sigma encoders | proved here (sign, monotone, width); tie caveat noted |
| Gaussian entropy constant 1.42 | book value is in nats; not implemented |
