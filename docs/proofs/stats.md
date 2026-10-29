# Proofs and derivations: `quantowl.stats` (AFML ch. 14: section 14.7 and snippets 14.1 to 14.4)

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
**Checked by test** (`test_stats_sharpe_variance_formula_by_simulation`, Gaussian case only):
with SR = 0.1 and T = 2000, 4000 simulated samples (seed 20261010) give Var[SR_hat] = 5.080e-4
against (1 + SR^2/2)/T = 5.025e-4, a relative difference of +1.09 % (tolerance 10 %). This is a
simulation check of the finite-T value, not a proof of it.

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

**Checks of the DSR formula (three tests).** `test_dsr_independent_reimplementation_of_the_formula`
(independent reimplementation, not a book value) rebuilds AFML 14.7.3 from scipy primitives alone,
without calling `expected_max_sharpe`, `deflated_sharpe_ratio` or `deflated_sharpe`, and matches both
public functions at rel 1e-12 for three parameter sets. `test_dsr_hand_computed_case_n_equals_one`
(hand-computed, closed form) takes N = 1, where SR* = 0 by convention, and checks DSR = Phi(2/sqrt(1.02))
= 0.976165 by hand. `test_deflated_sharpe_equals_psr_against_expected_max` (closed form, not a book
value) checks DSR against PSR at `expected_max_sharpe`.

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
**Checked by test:** `test_min_track_record_length_hand_computation` (pytest.approx: rel 1e-12
against the formula, abs 1e-9 against the stored constant 13.1749455434),
and to 1e-9 in `tests/test_doc_claims.py::test_stats_min_track_record_length_exact`.
The round trip PSR(ceil(minTRL)) >= p > PSR(ceil(minTRL) - 1) is checked by
`test_min_track_record_length_round_trip_gives_target_psr`, at other parameters.

The default `prob = 0.95` of minTRL is a common convention, not a book value.
**Checked by test:** `test_stats_default_prob_is_0_95` (the default call equals the explicit
`prob = 0.95` call, rel 1e-12).

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

## 7. Bet timing (Snippet 14.1)

**Source.** Snippet 14.1 as implemented here. Claimed from the book only (not verified against the
text).

**Proved here (set description).** Let Z = {t : p_t = 0}, and let O = {t : p_{t-1} != 0}, where
row 0 is put in O. Let F = {t >= 1 : p_t * p_{t-1} < 0}. The function returns the sorted set
(Z intersect O) union F, plus the last row when it is not already there. Each step is set algebra
on row labels, so the output is exactly this set.

**Edges of the rule.** Row 0 belongs to O because the NaN that `shift(1)` leaves in front of it
compares unequal to zero. A flat first row is therefore a bet end. The last row is always in the
result, so a flat last row that ends no bet is still reported. The last-row line is claimed from
the book only (not verified against the text).

**Checked by test.** `test_bet_timing_flattening_flip_and_trailing_open_position` (positions
1, 1, -1, -1, 0, 1, 1 give rows 2, 4, 6), `test_bet_timing_flat_first_row_is_a_bet_end` (0, 1, 1, 0
gives rows 0, 3), `test_bet_timing_reports_a_flat_last_row_even_when_it_ends_no_bet` (1, 0, 0 gives
rows 1, 2), `test_bet_timing_matches_row_by_row_rule_on_random_positions` (20 seeded integer paths
against the row-by-row statement above). Flips are read from signs, so
`test_bet_timing_sees_flips_between_tiny_positions` (sizes 1e-200) and
`test_bet_timing_is_unchanged_by_scaling_positions` (paths scaled by 1e-200 and 1e-300) hold.

## 8. Holding period (Snippet 14.2)

**Convention (not from the book).** Units are pooled: the open position has one entry time, the
size-weighted mean of the times its units were entered. The book's averaging is claimed from the
book only (not verified against the text).

**Proved here (invariant).** Let tau_t be the pooled entry time after row t, and let the closed
size and holding time be accumulated as the loop runs. Induction on t:

- Increase or unchanged, with p_t != 0. The code sets tau_t = (tau_{t-1} p_{t-1} + t dp) / p_t,
  where dp = p_t - p_{t-1}. Since p_{t-1} dp >= 0, p_{t-1} and dp have the same sign, and dividing
  numerator and denominator by the sign gives the mean of tau_{t-1} with weight |p_{t-1}| and t with
  weight |dp|. When p_{t-1} = 0 the formula gives tau_t = t, and no units were open.
- Same-side decrease, p_t p_{t-1} >= 0. The |dp| units that close at t are held t - tau_{t-1}. The
  pooled mean of the remaining units is unchanged, so tau is kept.
- Flip, p_t p_{t-1} < 0. The whole |p_{t-1}| closes at t, held t - tau_{t-1}. The new position
  enters at t, so tau_t = t.
- Closing to zero, p_t = 0. This is the same-side decrease with |dp| = |p_{t-1}|.

Hence the loop adds (t - tau) times the size closed at each closing row, and returns the sum of
those products over the sum of sizes closed. That is the size-weighted average holding period of
the closed size. NaN when nothing is closed, as in the book. The time unit is float days from the
first row.

**Checked by test.** `test_holding_period_one_long_closed_after_three_days` (3.0),
`test_holding_period_scale_in_weights_the_entry_by_size` (1.5),
`test_holding_period_partial_close_weights_each_closing_step_by_size` (2.0),
`test_holding_period_flip_closes_the_old_side_and_restarts_the_entry` (2.5),
`test_holding_period_uses_fractional_days` (1.5 for 36 hours),
`test_holding_period_numba_kernel_matches_python_transcription` (30 seeded cases, irregular
timestamps, relative tolerance 1e-12). Sizes are scaled by their largest magnitude before the
kernel, so `test_holding_period_closes_tiny_positions`, `test_holding_period_is_unchanged_by_scaling_random_positions`
and `test_holding_period_handles_sizes_near_1e306_over_a_long_history` (entry 3000 days in, size 1e306) hold.

**Measured.** `PYTHONPATH=src python3 benchmarks/bench_stats.py`, four runs on 2026-10-14, a shared
4-vCPU machine. For 100,000 rows the naive Python loop took 0.0187 to 0.0212 s and the numba kernel
0.0016 to 0.0018 s, a speedup of 11.2x to 11.7x. The two results agree to 6e-14. Timings vary with
the machine and are not tested.

## 9. Concentration (Snippet 14.3)

**Proved here (range, one sign).** Let the n values all have one sign, with sum S != 0, and let
w_i = r_i / S. Then w_i >= 0 and sum w_i = 1, so 0 <= w_i <= 1 and sum w_i^2 <= sum w_i = 1. By
Cauchy-Schwarz, sum w_i^2 >= (sum w_i)^2 / n = 1 / n. So h = sum w_i^2 lies in [1/n, 1], and
HHI = (h - 1/n) / (1 - 1/n) lies in [0, 1].

- HHI = 0 exactly when h = 1/n, which is equality in Cauchy-Schwarz: all w_i are equal, so all r_i
  are equal.
- HHI = 1 exactly when h = 1. Then sum w_i (1 - w_i) = 0 with each term non-negative, so each w_i
  is 0 or 1. With sum w_i = 1, exactly one return is non-zero.

**Proved here (mixed signs).** The range argument needs one sign. For r = (2, -1, 0), S = 1 and
w = (2, -1, 0), so h = 5 and HHI = (5 - 1/3) / (2/3) = 7.

**Checked by test.** `test_hhi_equal_returns_give_zero`, `test_hhi_single_nonzero_return_gives_one`,
`test_hhi_hand_computed_three_returns` (1/12 for returns 1, 2, 3),
`test_hhi_single_signed_returns_stay_in_unit_interval` (seeded, both signs),
`test_hhi_mixed_sign_can_exceed_one` (7), `test_hhi_is_scale_invariant`,
`test_hhi_two_or_fewer_returns_is_nan`, `test_hhi_refuses_zero_sum_and_missing_values_by_name`,
`test_return_concentration_hand_computed` (positive 1/12, negative 1/16, time 1/12),
`test_return_concentration_periods_without_returns_count_as_zero` (time 1/3, with an empty month),
`test_return_concentration_all_zero_side_is_nan`. The sums are taken after dividing by the
largest magnitude, so `test_hhi_equal_huge_returns_give_zero` and
`test_return_concentration_equal_huge_gains_give_zero_not_nan` (three gains of 1e308) hold, and
`test_hhi_refuses_a_scalar_by_name` covers a scalar input.

**Convention (not from the book).** `positive` takes returns >= 0, so zeros count as positive. A
component is NaN when its sub-series has two or fewer values, or sums to zero. A positive sub-series
sums to zero only when every value is zero. The time component uses calendar periods of `freq`,
month end by default (`pd.offsets.MonthEnd()`), and counts periods with no returns as zero.

**Measured.** `pd.Series([1.0], index=pd.DatetimeIndex(["2024-01-01"])).groupby(pd.Grouper(freq="M")).count()`
raises `ValueError: Invalid frequency: M` on pandas 3.0.6, checked 2026-10-14 on a shared 4-vCPU
machine.

**Claimed from the book only.** The HHI formula and the sign and time split of Snippet 14.3, as implemented here.

## 10. Drawdown and time under water (Snippet 14.4)

**Proved here (runs).** The running maximum M_t never decreases. Rows with equal M_t are therefore
contiguous, and grouping by the value of M_t gives the runs of consecutive rows in time order. The
first row of each run is the time that high-water mark (HWM) was set. The code takes each run's
minimum with `np.minimum.reduceat`, which is the group-by minimum of the book's rule.

**Proved here (what a reported drawdown is).** For an HWM H set at row s, the run extends until
the first later row whose pnl exceeds H. The drawdown is H - L (`dollars=True`) or 1 - L/H
(`dollars=False`), where L is the lowest pnl in that run. An HWM is reported only when L < H.

**Proved here (range).** With `dollars=False` and pnl > 0, every reported HWM has H > L > 0, so
1 - L/H lies in (0, 1).

**Checked by test.** `test_drawdown_hand_built_path_with_two_drawdowns` (fractions 1/2, 1/3, 1/2;
time under water 2/365.25 and 3/365.25 years), `test_drawdown_in_dollars_is_hwm_minus_low`,
`test_drawdown_open_last_drawdown_has_no_time_under_water_entry`,
`test_drawdown_fraction_needs_strictly_positive_pnl`,
`test_drawdown_time_under_water_runs_to_next_drawdown_high_not_next_high`,
`test_drawdown_matches_row_by_row_scan_on_random_equity_curves` (10 seeded equity curves against a
one-pass scan).

**Known property of the rule.** Time under water runs from a reported HWM to the next reported
HWM, not to the next HWM. A mark with no later drawdown in between is skipped, so the interval can
pass the recovery. Pnl 0, 1, 0.5, 2, 3, 2, 4 on consecutive days, with `dollars=True`, gives 3 days for the mark at day 1.
Pnl is back above 1 at day 3, so the recovery took 2 days. The rule is Snippet 14.4 as implemented here, claimed from the book only (not verified against the text). The 3-day value
is checked by test.

**Convention (not from the book).** One year is 365.25 days. The book divides a `DatetimeIndex`
difference by `np.timedelta64(1, 'Y')`. On pandas 3.0.6 that raises `ValueError: Unit Y is not
supported` (checked 2026-10-14 on a shared 4-vCPU machine). A plain numpy array divided by the
same value raises `TypeError` instead. So the conversion is replaced.

**Claimed from the book only.** The high-water-mark grouping and the dd and tuw definitions of
Snippet 14.4, as implemented here.

## Summary

| Statement | Status |
|---|---|
| Var[SR_hat] asymptotic formula | Proved here (delta method); Gaussian value at SR = 0.1, T = 2000 checked by simulation (`test_stats_sharpe_variance_formula_by_simulation`) |
| PSR form and T-1 convention | Proved here asymptotically; T-1 claimed from book |
| Expected-max approximation | Claimed from book; scaling by sqrt(V) proved here; accuracy checked by MC and by numerical reference (section 6) |
| Expected max at N = 1 and the N = 2 exact value 1/sqrt(pi) | Proved here; checked by test (`test_expected_max_n1_convention_agrees_with_exact_value`, `test_expected_max_n2_closed_form_for_exact_maximum_approximation_error_recorded`) |
| Approximation error against exact quadrature | Numerical reference, approximation error measured, not a book value (section 6) |
| DSR monotone in N | Proved here |
| minTRL formula | Proved here; formula from literature (not in AFML text) |
| 0.95 level (DSR threshold, minTRL default) | Convention (not from the book) |
| PSR = 0.5 at equality, hand values, monotonicity checks | Checked by test (`test_psr_is_half_when_sr_equals_benchmark`, `test_psr_hand_case_gaussian_denominator_sqrt3`, `test_expected_max_sharpe_properties`) |
| Bet-end set of Snippet 14.1 (flattenings, flips, last row) | Proved here as set algebra on row labels; edge cases checked by test (`test_bet_timing_*`); the rule itself claimed from the book only (not verified against the text) |
| Pooled entry time and holding period (Snippet 14.2) | Proved here (invariant under each row type); numba kernel equals the Python transcription (`test_holding_period_numba_kernel_matches_python_transcription`); speed measured (section 8) |
| HHI in [0, 1] for one-signed returns; 0 at equal returns, 1 at one non-zero | Proved here (Cauchy-Schwarz); checked by test (`test_hhi_single_signed_returns_stay_in_unit_interval`) |
| HHI above 1 with mixed signs (example 7) | Proved here (arithmetic); checked by test (`test_hhi_mixed_sign_can_exceed_one`) |
| Drawdown runs, dd in (0, 1) for positive pnl | Proved here (contiguous runs of the running maximum) |
| Time under water skips marks with no later drawdown | Checked by test (`test_drawdown_time_under_water_runs_to_next_drawdown_high_not_next_high`); the rule claimed from the book only |
| One year = 365.25 days | Convention (not from the book); the book's `'Y'` divisor fails on pandas 3.0.6 (section 10) |
