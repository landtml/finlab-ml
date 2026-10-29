# `quantowl.trading_rules`: Optimal trading rules on an Ornstein-Uhlenbeck process (AFML ch. 13)

Half-life conversions and fitting for a mean-reverting (Ornstein-Uhlenbeck) price,
and the batch simulation that scores profit-taking and stop-loss levels on it:
`ou_half_life`, `ou_phi`, `fit_ou` and `otr_grid`.

## Example

Runs as written; `tests/test_doc_examples.py` executes it on every test run.

```python
import numpy as np
from quantowl.trading_rules import fit_ou, otr_grid, ou_half_life, ou_phi

rng = np.random.default_rng(13)
phi = ou_phi(5.0)  # half-life of 5 steps
x = np.zeros(500)
for t in range(1, x.size):
    x[t] = phi * x[t - 1] + rng.normal()  # forecast 0

fit = fit_ou(x, forecast=0.0)
print(round(ou_half_life(phi), 3), round(fit.half_life, 1))

levels = np.linspace(0.0, 4.0, 5)
grid = otr_grid(0.0, half_life=5.0, profit_taking=levels, stop_loss=levels, n_iter=2000, seed=0)
print(grid.shape)
best = grid.loc[grid["sharpe"].idxmax()]
print(best["pt"], best["sl"])
```

`otr_grid` returns a DataFrame with one row per (profit-taking, stop-loss) pair and the
columns `pt`, `sl`, `mean`, `std` and `sharpe`. The `pt` and `sl` values are in units of
sigma, so the barriers are `pt * sigma` and `sl * sigma`.

## Notes

Proof: [trading_rules.md](../proofs/trading_rules.md). Scope: one holding rule on a
fixed-forecast OU price. The module has no transaction costs, no uncertainty in the
fitted parameters, and no search beyond the grid the caller passes.

Conventions to check before reading results:

- `fit_ou` returns `half_life = inf` when phi >= 1 (no mean reversion) and `nan` when
  phi <= 0 (no half-life). `sigma` is the innovation scale, not the stationary standard
  deviation.
- A path closes at the first update where its holding count exceeds `max_hp`, so it
  lasts at most `max_hp + 1` updates.
- `sharpe` is `mean / std`, with `std` at ddof=0. It is `nan` when `std` is zero.
- The paths start at `start`, not from the stationary distribution of the process.
- The draws come from numba's random generator, not from `numpy.random.default_rng`.
  The result is the same for a given seed at any thread count.

The snippet and section numbers come from the book and were not checked against its text.

Full signatures: [API reference](../API.md).
