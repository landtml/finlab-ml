# `finlab.strategy_risk`: Binomial Sharpe, implied precision and frequency, failure probability (AFML ch. 15)

For a strategy that makes a fixed number of IID bets a year, these functions give the Sharpe
ratio of the bet sequence, the hit rate or bet frequency that reaches a target Sharpe ratio, the
probability that a return sample falls short of a target, and a two-Gaussian mixture sampler.

## Example

Runs as written; `tests/test_doc_examples.py` executes it on every test run.

```python
from finlab.strategy_risk import binomial_sharpe, implied_frequency, implied_precision, mixture_returns, probability_of_failure

print(round(binomial_sharpe(0.55, 52), 4))
print(round(implied_precision(-1.0, 1.0, 52, 1.0), 4))
print(round(implied_frequency(-1.0, 1.0, 0.6, 1.0), 2))

returns = mixture_returns(0.05, -0.1, 0.05, 0.1, 0.75, 2600, seed=0)
print(round(probability_of_failure(returns, 260, 2.0), 4))
```

A bet wins `pt` (positive) with probability `p` and loses `sl` (negative) otherwise. `freq` is
the number of bets a year. `binomial_sharpe(p, freq, pt, sl)` is the annualised Sharpe ratio of
that sequence. With the defaults `pt = 1` and `sl = -1` it is `(2p - 1) sqrt(freq) / (2 sqrt(p(1-p)))`.

`implied_precision` and `implied_frequency` invert `binomial_sharpe` for `p` and `freq`.
`implied_precision` always returns a hit rate in (0, 1). `implied_frequency` returns `nan` when
no positive frequency gives the target: the target is 0, the mean bet return is 0, or the mean
bet return and the target have opposite signs.

## Notes

Proof: [strategy_risk.md](../proofs/strategy_risk.md). Scope: AFML chapter 15, Snippets 15.1
to 15.5. The bets are IID with two outcomes.

`probability_of_failure` passes `p * (1 - p)` as the scale of `norm.cdf`, as Snippet 15.5 does.
The sampling standard error of `p` is `sqrt(p * (1 - p) / n)`, which is about 24 times smaller
for the example above (2600 returns, `p` near 0.66). The module keeps the book's argument. The
proof note, section 4, gives both values.

`implied_precision` needs `sl < 0 < pt`. For a negative `target_sharpe` it returns the smaller
root of the same quadratic, the hit rate that gives that negative Sharpe ratio. The larger root,
which the module uses for non-negative targets, would give the mirror value `-target_sharpe`.

`mixture_returns` rounds the first-component count, `n_obs * prob1`. Truncation would give 1819
for `2600 * 0.7`, which is slightly below 1820 in floating point.

Full signatures: [API reference](../API.md).
