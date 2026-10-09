# Proofs and derivations: `finlab.stats` (AFML ch. 14, section 14.7)

Status labels used throughout:

- **Proved here**: derived in this document from stated assumptions.
- **Checked by test**: a numerical property asserted in `tests/test_stats.py`.
- **Claimed from the book only**: stated in AFML or the cited papers; not re-derived here.

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

Summing and collecting terms (the SR^2 and SR^4 terms cancel, leaving
1 - gamma3 SR + SR^2 (gamma4 - 1)/4) gives the claim.
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

**Checked by test.**
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

**Checked by test and simulation.**
- Monotone increasing in N and proportional to sqrt(V). (`test_expected_max_sharpe_properties`)
- Below the bound sqrt(2 log N) quoted by the book (checked at N = 1000).
- Monte Carlo of E[max of N standard normals] (20,000 draws) against the
  formula, sigma = 1: N = 2: 0.570 vs 0.520; N = 10: 1.542 vs 1.575;
  N = 100: 2.508 vs 2.531; N = 1000: 3.244 vs 3.255. The error is at most
  about 0.05 over this range.

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

## 5. Minimum track record length

**Derivation (proved here, from section 2).** Let p in (0, 1), D = sqrt(1 - gamma3 SR + (gamma4-1)/4 SR^2),
and suppose SR > SR*. Then PSR(T) = p holds iff (SR - SR*) sqrt(T - 1)/D = Phi^{-1}(p).
Solving for T:

    minTRL = 1 + D^2 ( Phi^{-1}(p) / (SR - SR*) )^2.

PSR is increasing in T when SR > SR*, so minTRL is the unique threshold for PSR >= p.
The implementation raises a ValueError for SR <= SR*, where no finite T
gives p > 1/2. The formula is from Bailey and Lopez de Prado (2012), which is
**not** in the AFML text supplied with this repo; cited from the literature.

**Hand computation (checked by test).** Gaussian returns (gamma3 = 0, gamma4 = 3),
SR = 0.5, SR* = 0, p = 0.95:
D^2 = 1 + (2/4)(0.25) = 1.125, Phi^{-1}(0.95) = 1.6448536,
minTRL = 1 + 1.125 (1.6448536/0.5)^2 = 13.174943.
**Checked by test:** `test_min_track_record_length_hand_computation`. The
round trip PSR(ceil(minTRL)) >= p > PSR(ceil(minTRL) - 1) is also checked.

## Summary

| Statement | Status |
|---|---|
| Var[SR_hat] asymptotic formula | Proved here (delta method) |
| PSR form and T-1 convention | Proved here asymptotically; T-1 claimed from book |
| Expected-max approximation | Claimed from book; scaling by sqrt(V) proved here; accuracy checked by MC |
| DSR monotone in N | Proved here |
| minTRL formula | Proved here; formula from literature (not in AFML text) |
| PSR = 0.5 at equality, hand values, monotonicity checks | Checked by test |
