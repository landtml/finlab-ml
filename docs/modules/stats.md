# `quantowl.stats`: Backtest statistics: Sharpe, PSR, DSR, minTRL, bets and drawdown (AFML ch. 14)

Sharpe ratio, probabilistic Sharpe ratio (adjusted for skew and kurtosis), deflated
Sharpe ratio (corrected for the number of trials), and minimum track record length.
Bet timing, holding period, return concentration and drawdown with time under water
(AFML snippets 14.1 to 14.4) are in the second half of the page.

## Example

Runs as written; `tests/test_doc_examples.py` executes it on every test run.

```python
import numpy as np
from quantowl.stats import sharpe_ratio, probabilistic_sharpe_ratio, deflated_sharpe, deflated_sharpe_ratio, min_track_record_length

rng = np.random.default_rng(8)
r = rng.normal(0.0005, 0.01, 500)
sr = sharpe_ratio(r, annualize=False)
print(round(probabilistic_sharpe_ratio(sr, 0.0, 500), 3))
print(round(deflated_sharpe_ratio(sr, n_trials=50, var_sr=0.01, n_obs=500), 3))
res = deflated_sharpe(sr, n_trials=50, var_sr=0.01, n_obs=500)
print(round(res.dsr, 3), round(res.sr_star, 3))
print(round(min_track_record_length(sr, 0.0), 1))
```

`deflated_sharpe` returns a `DeflatedSharpeResult` with `dsr`, `sr_star` (the expected
maximum benchmark) and the inputs. `deflated_sharpe_ratio` returns only the `dsr` float, with
the same value. Counts must be integers (`n_trials >= 1`, `n_obs >= 2`); bool is rejected.

`sr_hat` and `sr_benchmark` are per-period Sharpe ratios. Compute `sr_hat` with `sharpe_ratio(r, annualize=False)`. The annualised default is sqrt(252), about 15.9, times larger. On the example data above, passing it instead gives a PSR of 1.000 instead of 0.920, and a DSR of 1.000 instead of 0.0001.

## Bets, concentration and drawdown

Snippets 14.1 to 14.4. `bet_timing` and `holding_period` take a Series of target positions
(positive long, negative short, 0 flat) on a strictly increasing DatetimeIndex.
`return_concentration` takes a Series of returns; `drawdown_time_under_water` takes a
Series of pnl or an equity curve.

```python
import numpy as np
import pandas as pd
from quantowl.stats import bet_timing, holding_period, return_concentration, drawdown_time_under_water

days = pd.date_range("2024-01-01", periods=7, freq="D")
positions = pd.Series([1, 1, -1, -1, 0, 1, 1], index=days, dtype=float)
print(bet_timing(positions).strftime("%Y-%m-%d").tolist())
print(round(holding_period(positions), 2))

rng = np.random.default_rng(3)
returns = pd.Series(rng.normal(0.0005, 0.01, 260), index=pd.bdate_range("2024-01-01", periods=260))
conc = return_concentration(returns)
print(round(conc.positive, 3), round(conc.negative, 3), round(conc.time, 3))

pnl = pd.Series(np.exp(np.cumsum(rng.normal(0.0002, 0.01, 260))), index=pd.bdate_range("2024-01-01", periods=260))
dd, tuw = drawdown_time_under_water(pnl)
print(round(dd.max(), 3), round(float(dd.mean()), 3), round(float(tuw.mean() * 365.25), 1))
```

`bet_timing` returns the timestamps where a bet ends: a flattening, a sign flip, or the last
row. `holding_period` averages the days each closed unit was held, weighted by its size, in
float days; a position still open at the end adds nothing. `hhi_concentration` returns the
Herfindahl-Hirschman index of a return series, 0 for equal returns and 1 for a single
non-zero return. `return_concentration` applies it to the positive returns, the negative
returns and the number of returns per calendar period (month end by default; pass `freq`
for another period). `drawdown_time_under_water` returns the drawdown of each high-water
mark that is followed by a loss (`dd`, as a fraction or in pnl units), and the time under
water between consecutive such marks in years of 365.25 days (`tuw`).

## Notes

Proof: [stats.md](../proofs/stats.md). Scope: minTRL is cited to Bailey and Lopez de Prado (2012), not AFML.

Time under water runs from a high-water mark to the next one that has a drawdown, not to the
next one. It can therefore exceed the true recovery time when a mark with no later loss lies in
between; the docstring of `drawdown_time_under_water` gives an example. The last open drawdown
has a `dd` entry and no `tuw` entry.

The expected-maximum benchmark is an approximation. Against exact numerical integration it is
about 8% below the true maximum for 2 trials and about 0.4% above it for 1000 trials (proof note,
section 6). The 0.95 level used for a decision is a common convention, not a book value.

Full signatures: [API reference](../API.md).
