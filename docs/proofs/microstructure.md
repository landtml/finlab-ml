# finlab: Proofs and Checks for `finlab.microstructure` (AFML ch. 19)

Component: [`src/finlab/microstructure.py`](../../src/finlab/microstructure.py).
Tests: [`tests/test_microstructure.py`](../../tests/test_microstructure.py).

Every result is tagged with one of three statuses:

- **[proved here]**: derived in this document from stated assumptions.
- **[checked by test]**: verified numerically by the test suite against a naive
  reference or a simulation. This is evidence, not a proof.
- **[claimed from the book only]**: stated in AFML or the cited paper and used
  as given. It is not re-derived here, and the tests check at most a consequence.

---

## 1. Tick rule (AFML 19.3.1)

**Definition.** With prices `p_0, ..., p_T` and `dp_t = p_t - p_{t-1}`,
`b_0 = 1` and

    b_t = +1 if dp_t > 0,   -1 if dp_t < 0,   b_{t-1} if dp_t = 0.

**Proposition 1.1 [proved here].** For every `t >= 0`, `b_t` is in `{-1, +1}`, and
if `t >= 1` and some `s` in `1..t` has `dp_s != 0`, then `b_t` equals the sign of
`dp_s` for the largest such `s`. If no such `s` exists, `b_t = b_0 = 1`.

*Proof.* Induction on `t`. The base case `b_0 = 1` is given. For `t >= 1`, if
`dp_t != 0` then `b_t = sign(dp_t)` by definition, which is the claim with
`s = t`. If `dp_t = 0`, then `b_t = b_{t-1}`, and by the induction hypothesis
`b_{t-1}` is the sign of the last nonzero move at or before `t-1`, which is also
the last nonzero move at or before `t`. Both cases give values in `{-1, +1}`. ∎

**[checked by test]** Hand-made path `10, 10.5, 10.5, 10.2, 10.2, 10.3` gives
`[1, 1, 1, -1, -1, 1]`. Randomized paths on a tick grid match the naive loop.

---

## 2. Roll model (AFML 19.3.2)

**Model.** Mid-price `m_t = m_{t-1} + u_t`, with `u_t` i.i.d. `N(0, sigma_u^2)`.
Observed price `p_t = m_t + c b_t`, where `b_t` is in `{-1, +1}`, i.i.d.,
`P(b_t = 1) = 1/2`, and independent of `u`.

**Proposition 2.1 [proved here].** `Var(dp_t) = sigma_u^2 + 2c^2` and
`Cov(dp_t, dp_{t-1}) = -c^2`.

*Proof.* Write `dp_t = u_t + c(b_t - b_{t-1})`. Since `E[b_t] = 0`, `Var(b_t) = 1`,
and `b` is serially uncorrelated, `Var(b_t - b_{t-1}) = 2`. Independence of
`u` and `b` removes the cross term, so `Var(dp_t) = sigma_u^2 + 2c^2`.

Next, `dp_{t-1} = u_{t-1} + c(b_{t-1} - b_{t-2})`. The innovation `u_t` is
uncorrelated with everything in `dp_{t-1}`, so
`Cov(dp_t, dp_{t-1}) = c^2 Cov(b_t - b_{t-1}, b_{t-1} - b_{t-2})
= c^2 (-Var(b_{t-1})) = -c^2`. ∎

**Corollary 2.2 [proved here].** `c = sqrt(-Cov(dp_t, dp_{t-1}))` and
`sigma_u^2 = Var(dp_t) + 2 Cov(dp_t, dp_{t-1})`. The full spread is `2c`.

`roll_measure` returns `2c = 2 sqrt(max(0, -Cov))`, using the sample covariance
of the lagged pairs. It returns the full spread, not the half spread the book
writes as `c`.

**[checked by test]** The estimator matches a loop-based implementation to
machine precision. On a simulated bounce series with `c = 0.05` and 200,000
observations, the estimate is within 5% of `2c`.

**[claimed from the book only]** Consistency of the sample covariance under
stationarity and finite fourth moments (law of large numbers). The book gives
no sampling distribution.

---

## 3. Parkinson constants and high-low volatility (AFML 19.3.3)

For driftless geometric Brownian motion with volatility `sigma` observed over a
bar, the book states

    E[ln(H/L)^2] = k1 sigma^2,    E[ln(H/L)] = k2 sigma,
    k1 = 4 ln 2,    k2 = sqrt(8/pi).                    [claimed from the book only]

These are Parkinson (1980). We did not re-derive them. Only `k2` enters
`becker_parkinson_volatility`, through the Corwin-Schultz system below.

## 4. Corwin-Schultz spread (AFML 19.3.4, Snippets 19.1 and 19.2)

Let `h_s = ln(H_s / L_s)`, `beta_t = E[h_{t-1}^2 + h_t^2]` (sl = 1; the code
averages `sl` such sums), and `gamma_t = ln(max(H_{t-1}, H_t) / min(L_{t-1}, L_t))^2`.

**Definition (estimator).** With `D = 3 - 2 sqrt(2)`,

    alpha_t = (sqrt(2) - 1) sqrt(beta_t) / D  -  sqrt(gamma_t / D),
    alpha_t <- max(alpha_t, 0),
    S_t = 2 (e^{alpha_t} - 1) / (1 + e^{alpha_t}).

**Proposition 4.1 [proved here].** `S_t >= 0` for every `t`, and `S_t` is strictly
increasing in `alpha_t` on `[0, inf)`.

*Proof.* After clipping, `alpha_t >= 0`, so `e^{alpha_t} >= 1` and both numerator
and denominator are non-negative with a positive denominator. The derivative of
`(e^a - 1)/(e^a + 1)` is `2 e^a / (e^a + 1)^2 > 0`. ∎

**Derivation of the alpha formula [claimed from the book only].** The formula
comes from the model of Corwin and Schultz (2012, p. 727), in which the two-bar
range splits into a volatility part, which scales with the square root of time,
and a spread part. We did not re-derive it. Snippet 19.1 is implemented exactly as
written, with the rolling windows re-derived in closed form.

**[checked by test]** Vectorized output matches a loop-based implementation of the
same formula for `sl` in {1, 2, 4} on random bars (rtol 1e-10). Every output is
non-negative. A gapped path with alpha < 0 gives exactly 0, and a constant-range
path gives a strictly positive spread.

**Becker-Parkinson volatility (Snippet 19.2).**

    sigma_t = (2^{-1/2} - 1) sqrt(beta_t) / (k2 D) + sqrt(gamma_t / (k2^2 D)),
    clipped at 0.                                         [claimed from the book only]

We cannot derive this from the model. Two checks were made. (a) On simulated
GBM bars (sigma = 0.02, 20,000 bars, 400 monitoring steps per bar), the snippet's
coefficient `(2^{-1/2} - 1)` gives an average estimate of 0.0206. The alternative
`(2^{1/2} - 1)` gives 0.138, which is off by a factor of about seven, so it is
rejected. The residual 3% is the downward bias of discrete monitoring, not an
error in the formula. The test asserts agreement within 10%. (b) The vectorized
output matches the formula computed in loops.

## 5. Kyle's lambda (AFML 19.4.1)

**Model.** `dp_t = lambda x_t + e_t`, with `x_t = b_t V_t` (signed volume). The
regression has no intercept, as in the book.

**Proposition 5.1 [proved here].** The OLS slope through the origin is
`lambda_hat = sum(x_t dp_t) / sum(x_t^2)`, and `t = lambda_hat / se` with
`se = sqrt(SSR / ((n - 1) sum x_t^2))`.

*Proof.* Minimizing `S(lambda) = sum (dp_t - lambda x_t)^2` gives `S'(lambda) = 0`,
hence `lambda_hat = sum x dp / sum x^2` provided `sum x^2 > 0`. The residual variance
under one estimated parameter and no intercept has `n - 1` degrees of freedom,
so `Var(lambda_hat) = sigma^2 / sum x^2`, which is estimated by
`SSR / ((n - 1) sum x^2)`. ∎

**Note on the data.** Element `t` of `signed_volume` is paired with `p_t - p_{t-1}`.
This is why element 0 of the regressor is discarded.

**[checked by test]** An exact linear path returns `lambda = 2.5` to 1e-12, with
`t = inf` (zero residual). A noisy path matches `numpy.linalg.lstsq` to 1e-10 and
recovers `0.8` within 0.05.

**[claimed from the book only]** Kyle's (1985) equilibrium, `lambda = (1/2) sqrt(Sigma_0 / sigma_u^2)`
and the profit expression `E[pi] = (v - p_0)^2 / (4 lambda)`. The book gives these
in section 19.4.1 and we did not re-derive them.

## 6. Amihud's lambda (AFML 19.4.2)

**Regression form (default, book's definition).** `|r_tau| = lambda DV_tau + e_tau`,
with `DV_tau` the bar dollar volume and no intercept.

**Proposition 6.1 [proved here].** `lambda_hat = sum(|r| DV) / sum(DV^2)`, by the
same argument as Proposition 5.1.

**Ratio form (classic Amihud 2002).** `ILLIQ = mean(|r_tau| / DV_tau)`.
The two are different statistics. We return the regression form by default
(`method="regression"`), and the ratio form with `method="ratio"`.

**[checked by test]** Both forms match loop-based sums to 1e-12.

## 7. Bulk volume classification and VPIN (AFML 19.5.2)

**Note on sources.** The book's text gives VPIN and its bucketed estimator, but
it does not give a bulk-volume classifier. The BVC formula below is from Easley,
Lopez de Prado and O'Hara (2012). It is not part of AFML's text.

**BVC.** With `z_t = dp_t / sigma`, the buy fraction is `phi_t = Phi(z_t)`, where
`Phi` is the standard normal CDF, `Phi(0) = 1/2`, and `sigma` is the standard
deviation of price changes. For observation `t`, the buy volume is `V_t phi_t`
and the sell volume is `V_t (1 - phi_t)`.

**Proposition 7.1 [proved here].** `phi_t` is in `[0, 1]`. If the distribution of
`dp_t` is symmetric about zero, `E[phi_t] = 1/2`.

*Proof.* `Phi` takes values in `[0, 1]`. If `dp` is symmetric, `z` and `-z` have
the same law, so `E[Phi(z)] = E[Phi(-z)] = E[1 - Phi(z)]`, which gives
`E[Phi(z)] = 1/2`. ∎

**Volume buckets.** Buckets have fixed volume `V` (`bucket_volume`). An
observation that straddles a boundary is split pro rata, so each bucket holds
exactly `V` units.

**Proposition 7.2 (VPIN identity and bound) [proved here].** Let `B_tau` and
`S_tau` be the buy and sell volume in bucket `tau`, with `B_tau + S_tau = V`. Then

    |B_tau - S_tau| = V |2 v_tau - 1|,   v_tau = B_tau / V,

and

    VPIN_tau = (1/(nV)) sum_{k = tau-n+1}^{tau} |B_k - S_k| in [0, 1].

*Proof.* `|B - S| = |2B - V| = V |2 v - 1|` since `S = V - B`. Each term
satisfies `0 <= |B_k - S_k| <= B_k + S_k = V`, since `B_k, S_k >= 0`. Summing `n`
terms and dividing by `nV` gives a value in `[0, 1]`. ∎

**Derivation of the book's identity [claimed from the book only].** The book states
`E|V^B - V^S| ~ alpha mu` (Easley et al. 2012a, 2012b) and that
`VPIN = E|2 v - 1| = alpha mu / (alpha mu + 2 eps)` under a volume clock. The
Poisson mixture underlying these is not re-derived here.

**[checked by test]**
- The bucketed VPIN from the kernel equals a trade-by-trade reference
  (`naive_vpin`, which expands each unit of volume) to 1e-9, for integer volumes and
  bucket size. This checks the straddle split and the window.
- On random data, every VPIN value lies in `[0, 1]`.
- With every price rise far above `sigma`, every buy fraction tends to 1 and
  VPIN tends to 1 once the first bucket (which has no prior price, so `phi = 1/2`)
  has left the window.

**Causality.** The default `sigma` is the expanding-window standard deviation of
the price changes strictly before each bar, so the default classification uses no
future prices (checked by `test_bvc_default_scale_is_causal`). An explicit float
`sigma` is applied to every bar, and the caller is then responsible for how it was
estimated.
The straddle split is an approximation, because the buy fraction of a single
observation is applied to every bucket it touches.

## 8. Summary

| Result | Status |
|---|---|
| Tick rule signs (Prop. 1.1) | proved here; checked by test |
| Roll covariance identities (Prop. 2.1, Cor. 2.2) | proved here; bounce recovery checked by test |
| Roll consistency | claimed from the book only |
| Parkinson constants `k1`, `k2` | claimed from the book only |
| Corwin-Schultz alpha derivation | claimed from the book only |
| CS spread non-negativity, monotonicity (Prop. 4.1) | proved here; checked by test |
| Becker-Parkinson coefficient `(2^{-1/2}-1)` | claimed from the book only; checked by simulation (within 4%) |
| Kyle OLS and t-statistic (Prop. 5.1) | proved here; checked by test |
| Kyle equilibrium `lambda = (1/2) sqrt(Sigma/sigma_u^2)` | claimed from the book only |
| Amihud regression closed form (Prop. 6.1) | proved here; checked by test |
| BVC buy fraction in `[0,1]`, mean 1/2 (Prop. 7.1) | proved here; source is Easley et al. (2012), not AFML |
| VPIN identity and bound in `[0,1]` (Prop. 7.2) | proved here; checked by test |
| VPIN `E|V^B - V^S| ~ alpha mu` | claimed from the book only |
