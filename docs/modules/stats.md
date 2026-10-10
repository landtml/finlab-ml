# `finlab.stats`: Backtest statistics: Sharpe, PSR, DSR, minTRL (AFML ch. 14)

Sharpe ratio, probabilistic Sharpe ratio (adjusted for skew and kurtosis), deflated
Sharpe ratio (corrected for the number of trials), and minimum track record length.

## Example

Runs as written; `tests/test_doc_examples.py` executes it on every test run.

```python
import numpy as np
from finlab.stats import sharpe_ratio, probabilistic_sharpe_ratio, deflated_sharpe, deflated_sharpe_ratio, min_track_record_length

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

## Notes

Proof: [stats.md](../proofs/stats.md). Scope: minTRL is cited to Bailey and Lopez de Prado (2012), not AFML.

The expected-maximum benchmark is an approximation. Against exact numerical integration it is
about 8% below the true maximum for 2 trials and about 0.4% above it for 1000 trials (proof note,
section 6). The 0.95 level used for a decision is a common convention, not a book value.

Full signatures: [API reference](../API.md).
