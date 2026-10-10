# Proofs and derivations: `finlab.stats` (AFML ch. 14, section 14.7)

Status labels used throughout:

- **Proved here**: derived in this document from stated assumptions.
- **Checked by test** (the claim names its test function, for example `test_psr_matches_closed_form`): a numerical property asserted in `tests/test_stats.py`.
- **Claimed from the book only**: stated in AFML or the cited papers; not re-derived here.
- **Convention (not from the book)**: a choice made in this code, not a published value.
- **Numerical reference, approximation error measured, not a book value**: compared with an
  independent quadrature in `tests/test_stats.py` (section 6).

Notation: T observations, non-annualised Sharpe ratio SR = mu/sigma, skewness
gamma3, non-excess kurtosis gamma4 (gamma4 = 3 for Normal), Phi = standard
normal CDF, Phi^{-1} its inverse.

---

## 1. Asymptotic variance of the sample Sharpe ratio

**Claim (AFML 14.7.2, citing Bailey and Lopez de Prado 2012; the variance is
due to Mertens 2002).** For IID returns with finite fourth moment,

    Var[SR_hat] = (1 - gamma3 SR + (gamma4 - 1)/4 SR^2) / T   (asymptotically).

**Proved here (delta method).** Let x_t be IID with mean mu, variance sigma^2,
and standardised form x = mu + sigma z, E z = 0, E z^2 = 1, E z^3 = gamma3,
E z^4 = gamma4. Write m = sample mean and s2 = sample second moment of x.
The estimator is g(m, s2) = m (s2 - m^2)^{-1/2}, with population value
g(mu, E x^2) = SR.

Gradient at the population value (sigma^2 = E x^2 - mu^2):

    d g / d m  = sigma^{-1} (1 + SR^2)
    d g / d s2 = -SR / (2 sigma^2)

Per-observation covariances, using x^2 - E x^2 = 2 mu sigma z + sigma^2 (z^2 - 1):

    Var(x)     = sigma^2
    Cov(x, x^2) = 2 mu sigma^2 + sigma^3 gamma3
    Var(x^2)   = 4 mu^2 sigma^2 + 4 mu sigma^3 gamma3 + sigma^4 (gamma4 - 1)

Delta-method variance times T is the quadratic form a^2 Var(x) + 2ab Cov + b^2 Var(x^2),
with a = sigma^{-1}(1+SR^2) and b = -SR/(2 sigma^2). Substituting mu/sigma = SR:

    a^2 Var(x)        = (1 + SR^2)^2
    2ab Cov(x, x^2)   = -(1 + SR^2) SR (2 SR + gamma3)
    b^2 Var(x^2)      = SR^4 + SR^3 gamma3 + SR^2 (gamma4 - 1)/4

Summing and collecting terms (the SR^4 terms and the SR^2 cross terms cancel,
leaving 1 - gamma3 SR + SR^2 (gamma4 - 1)/4) gives the claim.
Check: gamma3 = 0, gamma4 = 3 gives (1 + SR^2/2)/T, the familiar Gaussian result.

**Assumptions:** IID returns, finite fourth moment, a consistent estimator of
the population moments. The result is asymptotic; finite-T accuracy is
claimed from the book only.

## 2. Probabilistic Sharpe ratio

**Claim (AFML eq. on p. 203).**

    PSR[SR*] = Phi[ (SR_hat - SR*) sqrt(T - 1) / sqrt(1 - gamma3 SR_hat + (gamma4 - 1)/4 SR_hat^2) ].

**Proved here (given section 1).** Under H0: SR = SR*, the standardised statistic
(SR_hat - SR*) / sqrt(V_hat / T), with V_hat the plug-in variance from section 1,
converges in distribution to N(0, 1) by Slutsky's theorem. Hence
P(SR > SR* | data) is approximated by Phi of the standardised statistic.
Replacing T by T - 1 is the small-sample convention of the book, which is
**claimed from the book only**.

**Checked by test (`test_psr_is_half_when_sr_equals_benchmark`, `test_psr_matches_closed_form`, `test_psr_fat_tails_lower_confidence`).**
- PSR = 0.5 exactly when SR_hat = SR* (any skew, kurtosis, T): the argument of
  Phi is 0. (`test_psr_is_half_when_sr_equals_benchmark`)
- The implementation equals the closed form for a non-trivial (SR, gamma3, gamma4, T). (`test_psr_matches_closed_form`)
- Fatter tails lower PSR. (`test_psr_fat_tails_lower_confidence`)

## 3. Expected maximum Sharpe ratio

**Claim (AFML 14.7.3 and the box "Euler-Mascheroni approximation").** For N
independent trials under H0 with cross-trial variance V,

    SR* = sqrt(V) [ (1 - gamma) Phi^{-1}(1 - 1/N) + gamma Phi^{-1}(1 - 1/(N e)) ],

where gamma = 0.5772... is the Euler-Mascheroni constant. The approximation is
from Bailey et al. (2014) and is **claimed from the book only**.

**Proved here (scaling).** If the trial Sharpe ratios are IID N(0, V) under H0,
then SR_n = sqrt(V) Z_n with Z_n standard normal, so
E[max_n SR_n] = sqrt(V) E[max_n Z_n]. The factor sqrt(V) in the formula is therefore exact,
and only the standard-normal term is approximate.

**Proved here (edge cases of the implementation).** For N = 1, the book's
formula is undefined (Phi^{-1}(0) = -inf); the implementation returns 0,
which is the expected maximum of a single draw with no selection effect.

**Checked by test (`test_expected_max_sharpe_properties`, `test_expected_max_matches_independent_numerical_reference`) and simulation (`test_stats_expected_max_monte_carlo_seed0`).**
- Monotone increasing in N and proportional to sqrt(V). (`test_expected_max_sharpe_properties`)
- Below the classical bound sqrt(2 log N) (cited from the literature, not verified here;
  checked at N = 1000 by `test_expected_max_sharpe_properties`).
- Monte Carlo of E[max of N standard normals] (20,000 draws) against the
  formula, sigma = 1: N = 2: 0.570 vs 0.520; N = 10: 1.542 vs 1.575;
  N = 100: 2.508 vs 2.531; N = 1000: 3.244 vs 3.255. The largest error is
  0.0504 (N = 2). The draws use `numpy.random.default_rng(0)`, with
  one `standard_normal((20000, N))` call per N in the order 2, 10, 100, 1000
  (`tests/test_doc_claims.py::test_stats_expected_max_monte_carlo_seed0`).
- **Numerical reference, approximation error measured, not a book value.** The exact E[max]
  by quadrature shows the formula is 7.9% below the exact maximum at N = 2, 2.5% above at N = 5,
  and 0.4% above at N = 1000 (table in section 6). The formula is not exact for small N.
  (`test_expected_max_matches_independent_numerical_reference`)

## 4. Deflated Sharpe ratio

**Definition (AFML 14.7.3).** DSR = PSR[SR*] with SR* from section 3.

**Proved here (monotonicity in N).** Phi^{-1}(1 - 1/N) and Phi^{-1}(1 - 1/(N e)) are both
increasing in N (their arguments increase with N). Since 1 - gamma > 0 and
gamma > 0, SR* is increasing in N for V > 0. PSR is decreasing in SR* because
Phi is increasing and its argument (SR_hat - SR*) sqrt(T-1)/D is decreasing in SR*
(D > 0). Hence DSR is strictly decreasing in N whenever V > 0.
**Checked by test:** `test_deflated_sharpe_decreases_with_trials`.

**Proved here (equality with PSR).** By construction,
`deflated_sharpe_ratio(...) == probabilistic_sharpe_ratio(..., expected_max_sharpe(...), ...)`.
**Checked by test:** `test_deflated_sharpe_equals_psr_against_expected_max`.

**Proved here (result object).** `deflated_sharpe(...)` returns the same `dsr` as
`deflated_sharpe_ratio(...)` (exact equality, since the latter returns `.dsr`), together with
`sr_star`. Counts must be integers (bool rejected) and the other inputs must be finite.
**Checked by test:** `test_deflated_sharpe_result_matches_float_api`,
`test_deflated_sharpe_rejects_invalid_inputs`.

**Convention (not from the book).** Values of DSR above 0.95 are a common significance
convention (not from the book). The book is not consulted for it here, so it is not labelled as
a book value.

## 5. Minimum track record length

**Derivation (proved here, from section 2).** Let p in (0, 1), D = sqrt(1 - gamma3 SR + (gamma4-1)/4 SR^2),
and suppose SR > SR*. Then PSR(T) = p holds iff (SR - SR*) sqrt(T - 1)/D = Phi^{-1}(p).
Solving for T:

    minTRL = 1 + D^2 ( Phi^{-1}(p) / (SR - SR*) )^2.

PSR is increasing in T when SR > SR*, so minTRL is the unique threshold for PSR >= p.
The implementation raises a ValueError for SR <= SR*, where no finite T
gives p > 1/2. The formula is from Bailey and Lopez de Prado (2012). It is cited from the
literature here, not from AFML.

**Hand computation (checked by test: `test_min_track_record_length_hand_computation`).** Gaussian returns (gamma3 = 0, gamma4 = 3),
SR = 0.5, SR* = 0, p = 0.95:
D^2 = 1 + (2/4)(0.25) = 1.125, Phi^{-1}(0.95) = 1.6448536,
minTRL = 1 + 1.125 (1.6448536/0.5)^2 = 13.1749455 (13.174946 to six decimals).
**Checked by test:** `test_min_track_record_length_hand_computation` (tolerance 1e-5),
and to 1e-9 in `tests/test_doc_claims.py::test_stats_min_track_record_length_exact`.
The round trip PSR(ceil(minTRL)) >= p > PSR(ceil(minTRL) - 1) is checked by
`test_min_track_record_length_round_trip_gives_target_psr`, at other parameters.

The default `prob = 0.95` of minTRL is a common convention, not a book value.
**Checked by test:** `test_min_track_record_length_hand_case_gives_four` (below).

## 6. Closed-form and numerical-reference checks

Label for this section: **numerical reference, approximation error measured, not a book value.**

**Proved here (N = 1).** For one standard normal, E[max] = E[Z] = 0. This equals the
convention in `expected_max_sharpe` (0 for N = 1), so the convention is also the exact value.
**Checked by test:** `test_expected_max_n1_convention_agrees_with_exact_value`.

**Proved here (N = 2, exact maximum).** max(a, b) = (a + b)/2 + |a - b|/2. For Z1, Z2 IID
N(0, 1), E|Z1 - Z2| = sqrt(2) * sqrt(2/pi) = 2/sqrt(pi). Hence E[max of 2] = 1/sqrt(pi) = 0.5641895835.
The implementation gives 0.5197553443, an error of -0.0444342393 (-7.876 %).
**Checked by test (closed form, approximation error recorded: `test_expected_max_n2_closed_form_for_exact_maximum_approximation_error_recorded`):**
`test_expected_max_n2_closed_form_for_exact_maximum_approximation_error_recorded`.

**Numerical reference.** The exact E[max of N IID N(0, 1)] is the integral of
x * N * phi(x) * Phi(x)^(N-1) dx, evaluated by `scipy.integrate.quad` on [-40, 40]. The density
N * phi * Phi^(N-1) integrates to 1 (`test_numerical_reference_integrates_a_density`), and the
N = 2 quadrature reproduces 1/sqrt(pi) to better than 1e-10. Measured (`expected_max_sharpe(N, 1.0)`
against the quadrature):

| N | exact (quadrature) | `expected_max_sharpe(N, 1)` | difference | relative |
|---|---|---|---|---|
| 2 | 0.5641895835 | 0.5197553443 | -0.0444342393 | -7.876 % |
| 5 | 1.1629644736 | 1.1925940010 | +0.0296295274 | +2.548 % |
| 10 | 1.5387527308 | 1.5745983013 | +0.0358455705 | +2.3295 % |
| 50 | 2.2490736294 | 2.2763030934 | +0.0272294640 | +1.211 % |
| 100 | 2.5075936364 | 2.5306028932 | +0.0230092568 | +0.918 % |
| 1000 | 3.2414357691 | 3.2551215137 | +0.0136857445 | +0.422 % |

The formula is below the exact maximum at N = 2 and above it for N >= 5. The relative error
falls as N grows, but it is not small for small N. The test bounds on |relative error| are
0.080, 0.026, 0.024, 0.0125, 0.0095 and 0.0045 for the six rows
(`test_expected_max_matches_independent_numerical_reference`). The Monte Carlo means in
section 3 (same seeded draws) differ from the exact values by 1.0, 0.7, 0.1 and 1.0 Monte Carlo
standard errors for N = 2, 10, 100 and 1000 (standard errors 0.0058, 0.0041, 0.0030, 0.0025),
so they do not contradict the quadrature.

**Checked by test (hand cases: `test_psr_hand_case_gaussian_denominator_sqrt3`, `test_min_track_record_length_hand_case_gives_four`).**
- Gaussian moments, denominator sqrt(3): with SR_hat = 2 and T = 4, the PSR denominator is
  sqrt(1 + 4/2) = sqrt(3) and sqrt(T - 1) = sqrt(3), so z = 2 at benchmark 0 and z = 1 at
  benchmark 1. PSR = Phi(2) = 0.97725 and Phi(1) = 0.84134 (standard table values, compared to
  1e-12). (`test_psr_hand_case_gaussian_denominator_sqrt3`)
- minTRL: SR_hat = 2, benchmark 0, Gaussian, prob = Phi(2): minTRL = 1 + 3 (2/2)^2 = 4, and
  PSR(T = 4) = prob. (`test_min_track_record_length_hand_case_gives_four`)

## Summary

| Statement | Status |
|---|---|
| Var[SR_hat] asymptotic formula | Proved here (delta method) |
| PSR form and T-1 convention | Proved here asymptotically; T-1 claimed from book |
| Expected-max approximation | Claimed from book; scaling by sqrt(V) proved here; accuracy checked by MC and by numerical reference (section 6) |
| Expected max at N = 1 and the N = 2 exact value 1/sqrt(pi) | Proved here; checked by test (`test_expected_max_n1_convention_agrees_with_exact_value`, `test_expected_max_n2_closed_form_for_exact_maximum_approximation_error_recorded`) |
| Approximation error against exact quadrature | Numerical reference, approximation error measured, not a book value (section 6) |
| DSR monotone in N | Proved here |
| minTRL formula | Proved here; formula from literature (not in AFML text) |
| 0.95 level (DSR threshold, minTRL default) | Convention (not from the book) |
| PSR = 0.5 at equality, hand values, monotonicity checks | Checked by test (`test_psr_is_half_when_sr_equals_benchmark`, `test_psr_hand_case_gaussian_denominator_sqrt3`, `test_expected_max_sharpe_properties`) |
