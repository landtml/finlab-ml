# `finlab.pbo`: Probability of backtest overfitting via CSCV (AFML ch. 11-12)

Given returns of many candidate strategies, estimates how often the in-sample best
strategy ranks below the median out of sample. Under pure noise the expectation is 1/2
for an even number of strategies, so PBO near 1 is not a noise result.

## Example

Runs as written; `tests/test_doc_examples.py` executes it on every test run.

```python
import numpy as np
from finlab.pbo import probability_of_backtest_overfitting

rng = np.random.default_rng(7)
returns = rng.normal(0, 0.01, size=(320, 20))   # rows = time, columns = strategies
res = probability_of_backtest_overfitting(returns, n_partitions=8)
print(round(res.pbo, 3), res.n_combinations)
```

## Notes

Proof: [pbo.md](../proofs/pbo.md).

Full signatures: [API reference](../API.md).
