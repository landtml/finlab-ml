# Proof notes: finlab.bet_sizing (AFML ch. 10)

Module: `src/finlab/bet_sizing.py`. Tests: `tests/test_bet_sizing.py`.

Notation: `Phi` is the standard normal CDF, `m` a bet size, `w > 0` the sigmoid
width, `x = f - p` the divergence between forecast `f` and market price `p`.

## Proved here

**P1 (two-class sizing is antisymmetric and zero at 1/2).** For `p in (0,1)`, let
`z(p) = (p - 1/2)/sqrt(p(1-p))` and `m(p) = 2 Phi(z(p)) - 1`. Then
`z(1-p) = -z(p)`, so `m(1-p) = -m(p)`; and `m(1/2) = 0`. Hence the sign of the bet
is the side of the predicted probability, and `|m| < 1` for `p` in `(0,1)`.
*Proof.* `p - 1/2` changes sign under `p -> 1-p`, the denominator is symmetric, and
`Phi(-z) = 1 - Phi(z)` gives `2 Phi(-z) - 1 = -(2 Phi(z) - 1)`. Since `Phi` maps
the reals into `(0,1)`, `|m| < 1`. ∎

**P2 (sigmoid sizing is bounded, odd, strictly increasing in x).** For `w > 0`,
`m(w,x) = x / sqrt(w + x^2)`.
(a) `|m| < 1`, because `x^2 < w + x^2`.
(b) `m(w,-x) = -m(w,x)`.
(c) `dm/dx = w (w + x^2)^(-3/2) > 0`.
*Proof.* (a) and (b) are immediate. (c) is the quotient-rule derivative. ∎

**P3 (calibration).** For `x != 0` and `0 < m* < 1`, `m(w,x) = m*` iff
`w = x^2 (m*^-2 - 1)`.
*Proof.* `m^2 = x^2/(w + x^2)` so `w = x^2/m^2 - x^2`. Taking the sign of `m`
to match `x` gives the formula in `calibrate_sigmoid_width`. ∎

**P4 (breakeven price is the sigmoid inverse).** Solving `m = (f - p)/sqrt(w + (f-p)^2)`
for `p` gives `p = f - m sqrt(w / (1 - m^2))`, for `|m| < 1`. The map `m -> L(f,w,m)`
is strictly decreasing (its derivative in `m` is `-sqrt(w) (1 - m^2)^(-3/2) < 0`), so
larger traded sizes have lower breakeven prices, and the breakeven for a block of
orders is the average of `L` over the traded sizes, as the book states.
*Proof.* Square both sides: `m^2 (w + (f-p)^2) = (f-p)^2`, so
`(f-p)^2 = m^2 w / (1 - m^2)`, and the sign of `f-p` equals the sign of `m`. ∎

**P5 (discretization error).** `discretize_signal(m, d)` returns `round(m/d)*d`,
so `|m - m*| <= d/2` for `d` in `(0,1]`.
*Proof.* `round` moves a real to a nearest integer, error at most `1/2`; multiply by `d`. ∎

**P6 (target position is bounded).** `|target_position| <= max_position`, because
`int(m * Q)` with `|m| < 1` truncates toward zero to a value of absolute value at
most `Q`.

## Checked by test (`test_book_sigmoid_calibration_and_targets`, `test_book_limit_price`, `test_bet_sizing_calibrated_width`, `test_average_active_signals_matches_naive_loop`, `test_two_class_probability_mapping`)

- Book worked examples: calibrating `m* = 0.95` at `x = 10` gives `w = 10.8033` (pinned in `tests/test_doc_claims.py`); at
  forecast 110 the target is 95; at forecast 115 the target is 97, and the breakeven
  limit price for the order of 97 from flat is 112.3657 (AFML 10.6).
- `average_active_signals` matches a naive loop over all timestamps (exact to 1e-12).
- Probability endpoints `p = 0, 1` give bets `-1, +1`.

## Claimed from the book only

- The mixture-of-Gaussians budgeting approach and the budgeting bet size (AFML
  10.2) are not implemented.
- The economic rationale for sizing (avoiding overtrading) is the book's; it is not
  tested.
