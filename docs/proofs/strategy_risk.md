# Proofs and derivations: `finlab.strategy_risk` (AFML ch. 15, Snippets 15.1 to 15.5)

Status labels used throughout:

- **Proved here**: derived in this document from stated assumptions.
- **Checked by test** (the claim names its test function): a numerical property asserted in `tests/test_strategy_risk.py`.
- **Claimed from the book only**: stated in AFML, with the snippet or equation number as cited here. Not verified against the text and not re-derived here.
- **Convention (not from the book)**: a choice made in this code, not a published value.
- **Measured**: a run of the code, with the date and machine class given.

Notation: a bet wins `pt > 0` with probability `p` and returns `sl < 0` otherwise. `D = pt - sl`
is the width of the two outcomes. `n` is the number of bets a year (`freq`). `theta` is the
annualised Sharpe ratio of the year's bets. `t` is the target Sharpe ratio (`target_sharpe`).

---

## 1. Binomial Sharpe ratio

**Claimed from the book only (AFML eq. 15.1 for symmetric payoffs and eq. 15.3 for asymmetric payoffs).**

    theta = ((pt - sl) p + sl) / ((pt - sl) sqrt(p (1 - p))) * sqrt(n).

**Proved here (two-point law).** Let X = pt with probability p and X = sl with probability 1 - p.
Then E X = p pt + (1 - p) sl = D p + sl. For the variance, pt - E X = D (1 - p) and sl - E X = -D p, so

    Var X = p (D (1 - p))^2 + (1 - p) (D p)^2 = D^2 p (1 - p).

**Proved here (annual Sharpe of n IID bets).** The total return over one year is the sum of n IID copies of X.
Its mean is n E X and its standard deviation is sqrt(n Var X), so the ratio is sqrt(n) E X / sqrt(Var X),
which is the formula above.

**Proved here (symmetric reduction).** With pt = 1 and sl = -1, D = 2, E X = 2p - 1 and sqrt(Var X) = 2 sqrt(p (1 - p)),
so theta = (2p - 1) sqrt(n) / (2 sqrt(p (1 - p))), which is eq. 15.1.

**Checked by test.**
- `test_binomial_sharpe_symmetric_matches_eq_15_1`: the symmetric closed form on a grid of p and n.
- `test_binomial_sharpe_symmetric_value_at_052_bets`: hand value at p = 0.55, n = 52, 0.72474307534.
- `test_binomial_sharpe_asymmetric_hand_case`: pt = 0.02, sl = -0.01, p = 0.6, n = 252, with mean 0.008 and std 0.03 sqrt(0.24).
- `test_binomial_sharpe_equals_two_point_moments`: the closed form against the mean and standard deviation of the two-point law computed from the probabilities.
- `test_binomial_sharpe_scales_with_sqrt_of_frequency`: theta(4n) = 2 theta(n).
- `test_binomial_sharpe_matches_simulated_bets`: 2,000,000 seeded bets (seed 20261013, p = 0.6, pt = 1, sl = -1, n = 52).
  The per-bet mean over the sample standard deviation, scaled by sqrt(52), is 1.474889 against 1.471960 from the closed form,
  a relative difference of 0.20 %. The tolerance is 2 %, about five standard errors of the sample mean.
  **Measured** 2026-10-13 on a shared 4-vCPU machine.

---

## 2. Implied precision

**Proved here (quadratic).** Set theta(p) = t and square. Since n > 0, this is equivalent to
n (D p + sl)^2 = t^2 D^2 p (1 - p) with the sign of D p + sl matching t. Expanding gives

    a p^2 + b p + c = 0,  a = (n + t^2) D^2,  b = (2 n sl - t^2 D) D,  c = n sl^2,

the coefficients of Snippet 15.3 (claimed from the book only). The squared equation holds exactly when
theta(p) = t or theta(p) = -t. The sign condition is handled in section 3 for frequency and by the root choice below.

**Proved here (discriminant).** Using D + sl = pt,

    b^2 - 4ac = t^2 D^2 (t^2 D^2 - 4 n sl pt) = t^2 D^2 (t^2 D^2 + 4 n |sl| pt).

Since sl < 0 < pt, this is nonnegative for every real t. A negative discriminant therefore cannot occur
for valid inputs, so the module has no refusal or nan branch for it.

**Proved here (theta is a bijection from (0, 1) onto the real line).** Write theta(p) = (sqrt(n) / D) g(p) with
g(p) = (D p + sl) / sqrt(p (1 - p)). Then

    g'(p) = (p (pt + sl) - sl) / (2 (p (1 - p))^{3/2}).

The numerator is linear in p. It equals -sl > 0 at p = 0 and pt > 0 at p = 1, so it is positive on [0, 1].
Hence g is strictly increasing on (0, 1). Its limits are -inf as p goes to 0 (because sl < 0) and +inf as p goes to 1
(because pt > 0). Therefore theta maps (0, 1) bijectively onto the real line. For every real t there is exactly one
p in (0, 1) with theta(p) = t. The derivative formula agrees with a central finite difference to 8e-10 relative
(measured 2026-10-13 on a shared 4-vCPU machine).

**Proved here (root choice).** The quadratic has exactly the solutions of theta(p) = t and theta(p) = -t in (0, 1).
Both exist by the bijection. Because theta is increasing, the larger root solves theta = +|t| and the smaller root
solves theta = -|t|. The module returns:

- for t >= 0, the larger root (-b + sqrt(disc)) / (2a), which is the Snippet 15.3 formula;
- for t < 0, the smaller root. Since (-b - sqrt(disc)) (-b + sqrt(disc)) = b^2 - disc = 4ac, the smaller root is (-b - sqrt(disc)) / (2a) = 2c / (-b + sqrt(disc)).

For t = 0 the two roots coincide at -sl / D = |sl| / D.

**Proved here (numerical form).** Both root formulas are written with the factored discriminant
sqrt(disc) = |t| D sqrt(t^2 D^2 + 4 n |sl| pt), and -b = t^2 D^2 + 2 n |sl| D. Every term is nonnegative, so no
cancellation occurs. The direct form b^2 - 4ac gives relative errors near 1e-8 at t = 0, where the discriminant is 0
(**Measured** 2026-10-13 on a shared 4-vCPU machine, against the bisection reference).

**Checked by test.**
- `test_implied_precision_hand_case_closed_form`: sl = -1, pt = 1, n = 4, t = 1 gives a = 20, b = -20, c = 4, disc = 80,
  and p = (5 + sqrt 5) / 10. Check: p (1 - p) = 1/5 and 2p - 1 = 1/sqrt 5, so theta = 1.
- `test_implied_precision_round_trip`: 180 cases of (sl, pt, n, p), with target = binomial_sharpe(p, n, pt, sl).
  Both signs of target occur because p is on both sides of the zero-mean hit rate. **Measured** 2026-10-13:
  maximum relative error in p 5.6e-16.
- `test_implied_precision_matches_slow_bisection`: 162 cases against bisection on the two-point moments, which uses no quadratic.
  The tolerance is 1e-8 relative. **Measured** 2026-10-13: maximum relative error 5.0e-16.
- `test_implied_precision_negative_target_returns_smaller_root`: sl = -1, pt = 1, n = 52, t = -1 gives p < 1/2.
- `test_implied_precision_zero_target_is_zero_mean_hit_rate`: t = 0 gives p = -sl / (pt - sl) = 1/3 for pt = 2, sl = -1.
- `test_implied_precision_discriminant_is_nonnegative_on_grid`: b^2 - 4ac equals the factored form on a grid, and is nonnegative.

---

## 3. Implied frequency

**Proved here.** Solving n (D p + sl)^2 = t^2 D^2 p (1 - p) for n gives

    n = t^2 D^2 p (1 - p) / (D p + sl)^2.

**Proved here (spurious root).** For D p + sl != 0 and t != 0, theta(p, n) = sqrt(n) (D p + sl) / (D sqrt(p (1 - p))).
Since sqrt(n) > 0, theta(p, n) has the sign of D p + sl. The n above gives theta(p, n)^2 = t^2 exactly, so
theta(p, n) = t holds if and only if the sign of D p + sl matches the sign of t. When the signs differ, the
function returns nan. The module checks theta(p, n) against t with a relative tolerance of 1e-9 and an absolute tolerance of 1e-12.

**Convention (not from the book).** nan is returned when no positive frequency gives the target: t = 0 (a zero Sharpe
needs n = 0), D p + sl = 0 (theta is 0 for every n), or the signs differ. The formula would divide by zero in the middle case.

**Checked by test.**
- `test_implied_frequency_hand_case_closed_form`: sl = -1, pt = 1, p = (5 + sqrt 5) / 10, t = 1 gives n = 4.
- `test_implied_frequency_negative_target_with_negative_mean`: sl = -1, pt = 1, p = 0.3, t = -1 gives n = 5.25. Check: 4 * 0.21 / 0.16.
- `test_implied_frequency_round_trip`: n is recovered from binomial_sharpe for p in {0.1, 0.3, 0.55, 0.7, 0.95} and n in {1, 52, 260}.
- `test_implied_frequency_spurious_root_is_nan`: p = 0.3, t = +1 gives nan, because the squared root gives theta = -1.
- `test_implied_frequency_nan_when_no_positive_frequency_exists`: t = 0 and the zero-mean case (p = 0.5 with sl = -1, pt = 1).

---

## 4. Probability of failure

**Claimed from the book only (AFML Snippet 15.5, probFailure).** With `p` the share of returns above zero, `rPos` the mean of
those returns, `rNeg` the mean of the others, and `thresP = implied_precision(rNeg, rPos, freq, t)`:

    risk = norm.cdf(thresP, loc = p, scale = p (1 - p)).

**Proved here (what the threshold means).** The threshold is the hit rate at which a bet with payoffs `rPos` and `rNeg`
and frequency `freq` has Sharpe ratio `t`. With those payoffs, the module's `binomial_sharpe(thresP, freq, rPos, rNeg) = t`.
The output is a normal CDF value at that threshold, so it lies in [0, 1].

**Scale argument (not changed here).** The scale `p (1 - p)` is the Bernoulli variance of one return's sign. The standard
error of the sample share `p` for `n` returns is `sqrt(p (1 - p) / n)`. The ratio of the book's scale to the standard error is

    p (1 - p) / sqrt(p (1 - p) / n) = sqrt(n p (1 - p)).

For the example in section 4.1 (2600 returns, seed 0) the ratio sqrt(n p (1 - p)) is 24.10
(**Measured** 2026-10-13 on a shared 4-vCPU machine). The two scales give different answers, shown
below. The module keeps the book's argument, because the scale is part of the formula it reproduces.

### 4.1 The book example

`mixture_returns(0.05, -0.1, 0.05, 0.1, 0.75, 2600, seed=0)`, then `probability_of_failure(returns, 260, 2.0)`.

**Measured** 2026-10-13 on a shared 4-vCPU machine, seed 0:

| quantity | value |
|---|---|
| returns above zero | 1724 of 2600 |
| `p` | 0.6630769231 |
| `rPos` | 0.0631952482 |
| `rNeg` | -0.0926459252 |
| `thresP` | 0.6535114569 |
| scale `p (1 - p)` (book) | 0.2234059172 |
| standard error `sqrt(p (1 - p) / n)` | 0.0092695920 |
| risk with the book's scale | 0.4829238935 |
| risk with the standard-error scale (not the module output) | 0.1510551058 |

Other seeds give 0.4579 (seed 1), 0.4835 (seed 2), 0.4491 (seed 3), 0.4691 (seed 4) and 0.4570 (seed 5),
all with the book's scale. The six seeds span 0.449 to 0.484.

**Checked by test.**
- `test_probability_of_failure_matches_bisection_reference`: a five-return sample with p = 0.6, rPos = 0.03, rNeg = -0.02.
  The threshold is found by bisection on `binomial_sharpe` itself, which checks the mapping of `rNeg` to `sl` and `rPos` to `pt`.
- `test_probability_of_failure_book_example_is_in_unit_interval`.
- `test_probability_of_failure_book_example_seed0_value`: pins 0.4829238935 (the **Measured** value above).
- `test_probability_of_failure_accepts_pandas_series`, and the refusals for all-positive, all-nonpositive, all-zero losses,
  NaN, and multi-column input.

---

## 5. Two-Gaussian mixture

**Convention (not from the book).** The draw is `round(n_obs * prob1)` values from N(mu1, sigma1), the rest from N(mu2, sigma2),
then a shuffle, all from one `numpy.random.default_rng(seed)` stream. The count is rounded because `2600 * 0.7` is
slightly below 1820 in floating point, and truncation would give 1819.

**Proved here (moments).** The mixture mean is prob1 mu1 + (1 - prob1) mu2. Its second moment is
prob1 (sigma1^2 + mu1^2) + (1 - prob1)(sigma2^2 + mu2^2). For the book example (mu1 = 0.05, mu2 = -0.1,
sigma1 = 0.05, sigma2 = 0.1, prob1 = 0.75), the mean is 0.0125 and the variance is 0.00875 - 0.0125^2 = 0.00859375.

**Checked by test.**
- `test_mixture_returns_component_counts`: with zero sigma each draw equals its mean, so the count of mu1 values
  is read back exactly: 0, 1300, 1820, 1950 and 2600 for prob1 = 0, 0.5, 0.7, 0.75 and 1.
- `test_mixture_returns_shuffles_components`: with mu1 = 1, mu2 = -1 and zero sigma, a value of -1 appears among the first 1950 entries.
- `test_mixture_returns_is_reproducible_and_shuffled`: the same seed gives the same array, and another seed does not.
- `test_mixture_returns_sample_mean_within_four_standard_errors`: 2,000,000 draws with seed 20261013.
  The sample mean is 0.0123841, a difference of -0.000116 from 0.0125. The tolerance is four standard errors, 0.000262.
  **Measured** 2026-10-13 on a shared 4-vCPU machine.
- `test_mixture_returns_refuses_bad_seed_by_name` and `test_mixture_returns_refuses_draws_that_overflow_float64`: a bool, negative or float seed, and draws beyond the float64 range, are refused by name.

---

## 6. Float64 range

**Proved here (scale invariance).** Theta depends on the payoffs only through their ratio. Multiplying pt and sl by
c > 0 multiplies the mean and the standard deviation of one bet by c, so theta does not change. The quadratic of
section 2 has coefficients homogeneous of degree two in (pt, sl), so its roots, and the hit rate, do not change
either. The implied frequency of section 3 is homogeneous of degree zero in (pt, sl), so it does not change. The
module divides both payoffs by max(pt, -sl) before the arithmetic, which places the outcome width in (1, 2].

**Checked by test.** `test_binomial_sharpe_is_invariant_to_common_payoff_scale`,
`test_implied_precision_is_invariant_to_common_payoff_scale` and `test_implied_frequency_is_invariant_to_common_payoff_scale`
(scales 1e-310, 1e-300 and 1e308); `test_probability_of_failure_is_invariant_to_tiny_return_scale`;
`test_probability_of_failure_accepts_returns_near_float_max`.

**Convention (not from the book): refusals.** A result that float64 cannot represent raises ValueError naming the
argument: a hit rate that is not a normal float strictly inside (0, 1); a frequency that overflows or lies below the
normal range; a target whose square overflows; an integer beyond the float64 range.

**Checked by test.** `test_implied_precision_names_target_when_its_square_overflows`,
`test_implied_frequency_names_target_when_the_frequency_overflows`,
`test_implied_precision_refuses_hit_rates_that_are_not_normal_floats_in_unit_interval`,
`test_implied_frequency_refuses_frequencies_below_the_normal_float_range` and
`test_scalars_beyond_float64_range_are_refused_by_name`.

**Measured (2026-10-14, a shared 4-vCPU machine).** The round trip
`binomial_sharpe(implied_precision(sl, pt, freq, t), freq, pt, sl)` at sl = -1, pt = 1, freq = 52 has relative error
2.2e-12 at t = 1e3, 6.2e-10 at t = 1e5, 1.7e-6 at t = 1e6 and 3.2e-2 at t = 1e8. Near p = 1 the spacing of float64
limits the hit rate, so the round trip holds to float64 precision only for moderate targets.

---

## Summary

| Statement | Status |
|---|---|
| Two-point mean and variance; annual Sharpe of n IID bets | Proved here |
| Symmetric reduction to eq. 15.1 | Proved here; eq. number claimed from the book only |
| Quadratic for implied precision; root choice; factored discriminant | Proved here |
| Discriminant is nonnegative for sl < 0 < pt, so no negative-discriminant case exists | Proved here |
| theta is a bijection from (0, 1) onto the real line | Proved here (derivative identity checked to 8e-10 by finite difference) |
| Implied precision against the slow reference | Checked by test (`test_implied_precision_matches_slow_bisection`) |
| Implied frequency, spurious root and nan cases | Proved here; checked by test |
| Probability of failure formula and scale `p (1 - p)` | Claimed from the book only; scale issue measured (section 4.1) |
| Mixture count, moments and shuffle | Proved here (moments); checked by test |
| Simulated binomial Sharpe check | Checked by test; measured 0.20 % relative difference |
| Rounding of the mixture count and the nan policy | Convention (not from the book) |
| Float64 range: scale invariance, refusals, measured round-trip error | Proved here (invariance); checked by test; measured (section 6) |
