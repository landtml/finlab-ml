# Proofs and derivations: `quantowl.trading_rules` (AFML ch. 13, Snippets 13.1 and 13.2)

Status labels used below:

- **Proved here**: derived in this document from the stated assumptions.
- **Checked by test** (the claim names its test function): asserted in `tests/test_trading_rules.py`.
- **Measured**: a number from a run, with the command, the date and the machine class. Not a proof.
- **Claimed from the book only (not verified against the text)**: stated in AFML. Section and snippet numbers are as given in the lane spec.
- **Convention (not from the book)**: a choice made in this code.

Notation: P_t is the price, f the forecast, phi the AR(1) coefficient, sigma the innovation
scale, eps_t i.i.d. N(0, 1), and d_t = P_t - f.

---

## 1. Half-life of the OU model

**Claimed from the book only (not verified against the text).** AFML 13.5 gives the model
P_t = (1 - phi) f + phi P_{t-1} + sigma eps_t, and the half-life tau = -log 2 / log phi.

**Proved here.** Subtract f from both sides. Since (1 - phi) f - f = -phi f,

    d_t = phi d_{t-1} + sigma eps_t.

eps_t has mean zero and is independent of the past, so E[d_t | d_0] = phi^t d_0 for t >= 0.
Each step multiplies the expected deviation by phi.

Let tau be the time at which the expected deviation halves, phi^tau = 1/2. Taking logarithms
gives tau log(phi) = -log 2. For 0 < phi < 1,

    tau = -log 2 / log phi,    equivalently    phi = 2^(-1/tau).

**Proved here (properties on 0 < phi < 1).**

- tau is positive and finite.
- d tau / d phi = log 2 / (phi (log phi)^2) > 0, so tau increases with phi.
- tau tends to 0 as phi tends to 0 from above, and to infinity as phi tends to 1 from below.

**Checked by test.** `test_ou_phi_of_five_steps_half_life` (phi = 2^(-0.2) for tau = 5),
`test_ou_half_life_round_trip`, `test_deviation_halves_after_half_life`,
`test_half_life_increases_with_phi`.

## 2. Stationary variance and the innovation scale

**Proved here.** Suppose the process is stationary, so d_t has a fixed variance V. Since eps_t is
independent of d_{t-1}, V = phi^2 V + sigma^2, and therefore V = sigma^2 / (1 - phi^2). The
stationary mean of d is zero, because E[d_t] = phi^t d_0 tends to zero.

`fit_ou` returns an estimate of the innovation scale sigma, not of sqrt(V).

**Convention (not from the book).** `otr_grid` starts each path at `start` and does not draw the
start from the stationary distribution.

## 3. Least-squares fit (`fit_ou`)

**Proved here (estimator).** Let x_t = P_{t-1} - f and y_t = P_t - f. The sum of squares
S(phi) = sum (y_t - phi x_t)^2 is quadratic in phi with leading coefficient sum x_t^2. Setting
dS/dphi = 0 gives phi_hat = sum(x_t y_t) / sum(x_t^2). The sum of x_t^2 must be positive, which
is why the refusal for an all-equal lagged path exists.

**Proved here (consistency, given stationarity).** Assume the ergodic theorem for a stationary
AR(1) series with finite variance. The ergodic theorem is assumed here, not proved. Then
(1/n) sum x_t y_t tends to E[d_{t-1} d_t] = phi Var(d), and (1/n) sum x_t^2 tends to Var(d). The
first limit uses E[d_{t-1} d_t] = phi E[d_{t-1}^2], which holds because eps_t is independent of
d_{t-1}. So phi_hat tends to phi.

**Convention (not from the book).** sigma is the sample standard deviation of the residuals
y_t - phi_hat x_t with ddof = 1. The residual mean is not forced to zero. The half-life of a
fitted phi is `ou_half_life(phi)` for 0 < phi < 1, `inf` for phi >= 1, and `nan` for phi <= 0.

**Checked by test.** `test_fit_ou_closed_form_on_three_prices` checks the hand computation for
prices (1, 2, 0) and forecast 0. Lagged values are (1, 2) and current values are (2, 0), so
phi = 2 / 5 = 0.4. The residuals are (1.6, -0.8), with ddof = 1 variance 2.88.
`test_fit_ou_recovers_parameters_on_seeded_path` simulates n = 20000 with phi = 0.9 and
sigma = 1.5 (seed 20261013). Its tolerances are 0.02 for phi and 0.05 for sigma. The standard
errors behind them are about 0.0031 and 0.0075. These come from the usual asymptotic formulas,
sqrt((1 - phi^2) / n) and sigma / sqrt(2n), and were not measured. The tolerances are about six
standard errors each. `test_fit_ou_half_life_is_inf_when_phi_at_or_above_one` and
`test_fit_ou_half_life_is_nan_when_phi_not_positive` check the half-life conventions.

## 4. Holding rule: exit test and holding cap

A path closes at the first update where cP > pt sigma, or cP < -sl sigma, or hp > max_hp. Here
cP = p - start, and hp counts the updates so far.

**Proved here (termination and cap).** hp rises by one at each update, so the test hp > max_hp
holds at the latest when hp = max_hp + 1. Every path therefore ends after at most max_hp + 1
updates. With max_hp = 1 and no barrier hit, a path holds for exactly two updates.

**Proved here (strict barriers).** Both barrier comparisons are strict. A value equal to a
barrier does not close the path.

**Proved here (pt = 0).** With pt = 0, a path closes at the first update where cP > 0, unless the
stop or the cap closes it first. If the stop and the cap are not reached first, every exit is
positive.

**Proved here (one-update closed form).** With max_hp = 0 and barriers that no path reaches in one
update, the single update gives p_1 = (1 - phi) f + phi start + sigma g, with g standard normal.
The exit is p_1 - start.

**Proved here (scale).** With forecast = start = 0, each update is linear in sigma, and the
barriers are pt sigma and sl sigma. Each exit is therefore proportional to sigma, and so are its
mean and standard deviation.

**Checked by test.** `test_exit_test_is_strict_and_holding_cap_by_hand` (a barrier equal to the
value does not close, and a hand-computed three-update path exits at 2.375).
`test_max_hp_one_holds_two_updates_by_hand`, `test_max_hp_zero_closes_after_one_update_by_hand`,
`test_max_hp_one_kernel_uses_two_updates_per_path`, `test_pt_zero_exits_on_first_positive_move`,
`test_one_step_holding_matches_closed_form`, `test_sigma_scales_exit_values`.

## 5. Kernel, transcription and determinism

**Proved here (determinism).** The exit values of pair k depend only on phi, forecast, sigma,
start, pt_k sigma, sl_k sigma, max_hp, n_iter, and seed_k. Here seed_k is the first uint32 of
`SeedSequence(seed, spawn_key=(k,))`. The kernel call for pair k seeds numba's generator on
entry, draws only from that stream, and writes only its own output slots. No pair reads another
pair's stream, so the output does not depend on the thread count or on the order in which the
threads run the pairs. This argument relies on numba keeping generator state per thread and running
each prange iteration on a single thread. That behaviour is not verified here against numba's
documentation. The tests below check the consequence.

**Checked by test.** `test_parallel_result_equals_serial_and_other_thread_counts` (exact equality at
thread counts 1, 2 and the default), `test_same_seed_gives_identical_frames`, and
`test_extending_the_grid_keeps_existing_rows` (pair 0 of a larger grid equals the 1 x 1 grid).

**Proved here (kernel equals transcription).** `_exits_one_pair` and `_otr_exits_python` evaluate
p = (1 - phi) f + phi p + sigma g and cP = p - start in the same order, on the same normals. In
IEEE double arithmetic their outputs are then equal. This assumes numba does not fuse the
multiply-add. Fastmath is off. The test checks the equality on the platform that ran it.

**Checked by test.** `test_kernel_matches_python_transcription` (exact equality for four parameter
sets, including zero barriers and max_hp = 0). `test_grid_rows_match_python_transcription_with_their_pair_seed`
(mean and std of every row, relative tolerance 1e-12, because NumPy and numba sum in different
orders).

**Proved here (constant exits give nan Sharpe).** Take half_life = 1e-3 and forecast = start =
1e20. Then phi = 2^(-1000), so phi p is about 1e-281. The term sigma g is of order 1e-3. Half an
ulp of 1e20 is 8192, so a normal draw would need |g| above about 8e6 to move the sum. The seeded
draws in the test do not reach that size. Each update therefore rounds to 1e20, every cP is 0,
std is 0, and the Sharpe ratio is nan. **Checked by test:**
`test_sharpe_is_nan_when_exits_are_constant`.

## 6. Grid procedure and the book's results

**Claimed from the book only (not verified against the text).** AFML 13.5 (Snippet 13.2)
simulates the exit value for each (profit-taking, stop-loss) pair and summarises the exits by
their Sharpe ratio. The book shows heat maps of that Sharpe ratio over the grid and draws
conclusions from them. This document does not reproduce the heat maps, and the conclusions are
not verified here.

**Claimed from the book only (not verified against the text).** Snippet 13.1 is the OU estimation
that `fit_ou` follows.

**Convention (not from the book).** The default grid is 21 levels from 0 to 10 in units of sigma,
and the default n_iter is 100,000. Both are choices of this code. The output columns `pt` and
`sl` hold the grid values, not the barriers.

## 7. Measured

**Measured (`benchmarks/bench_trading_rules.py`, not a test).** Command:
`PYTHONPATH=src python3 benchmarks/bench_trading_rules.py`, run three times on 2026-10-14, on a
shared 4-vCPU machine. The load average was about 1.0 to 1.4 during the runs. Settings: 5 x 5 grid,
n_iter = 2000, max_hp = 100, half_life = 5, sigma = 1. The numba time is the best of 3 calls, and the
kernel is compiled before timing. The naive loop is timed once, on normals drawn before timing, so its
time leaves out the draws.

| run | naive loop (s) | numba (s) | speedup | means agree |
|---|---|---|---|---|
| 1 | 0.1659 | 0.0222 | 7.5x | yes |
| 2 | 0.1250 | 0.0197 | 6.3x | yes |
| 3 | 0.1290 | 0.0191 | 6.8x | yes |

The speedup is a lower bound for a naive loop that draws its own normals, because the timed naive
loop does not include those draws.

**Measured (docs page example, cold numba cache, 2026-10-14, a shared 4-vCPU machine).** The code block
alone, after the imports, took 1.62 s, 1.65 s and 2.03 s over three cold-cache runs, and 0.38 s on a warm
cache. The whole process, including the imports of numba, pandas and quantowl, took 2.35 s, 2.38 s and 2.40 s.
The slowest cold run is above the 2 s target.

---

## 8. Float64 range

**Proved here (fit_ou is equivariant in the price scale).** Scaling every price and the forecast by c > 0
scales the deviations x and y by c. Then phi = sum(x y) / sum(x^2) is unchanged, and the residual standard
deviation scales by c. The module divides the deviations by their largest magnitude before the sums, so the
sums stay finite and nonzero for prices near 1e170 and 1e-170.

**Checked by test.** `test_fit_ou_is_equivariant_in_price_scale_away_from_float_limits` (scales 1e170 and 1e-170),
`test_fit_ou_closed_form_on_three_prices`.

**Proved here (otr_grid is scale-free in sigma).** With forecast = start = 0, the update p = phi p + sigma g
and the barriers pt sigma and sl sigma are all linear in sigma. Every exit, mean and standard deviation
therefore scales by sigma, and the Sharpe ratio does not change. The kernel divides the exits by their largest
magnitude before it squares the deviations.

**Checked by test.** `test_otr_grid_is_scale_free_at_extreme_sigma` (sigma 1e-300 and 1e300),
`test_sigma_scales_exit_values`.

**Convention (not from the book): refusals.** A result that float64 cannot represent raises ValueError
naming the argument:

- sigma times the grid overflows, or a simulated mean or standard deviation is not finite:
  `test_otr_grid_refuses_barriers_that_overflow_float64`, `test_otr_grid_refuses_exits_that_overflow_float64`;
- n_iter or max_hp above 2**63 - 1: `test_otr_grid_refuses_counts_beyond_int64_by_name`;
- a scalar beyond the float64 range: `test_scalars_beyond_float64_range_are_refused_by_name`.

ou_phi returns the endpoints 0 and 1 for half-lives beyond the rounding thresholds, and otr_grid simulates
both cases: `test_ou_phi_rounds_to_endpoints_for_extreme_half_lives`.

---

## 9. Not covered

- The estimation error of phi and sigma is not carried into the grid results.
- Transaction costs are not modelled.
- Paths start at `start`, not from the stationary distribution (section 2).
