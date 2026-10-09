# `finlab.microstructure`: Microstructure features (AFML ch. 19)

Spread and liquidity estimators from prices and volumes: tick rule, Roll and Corwin-Schultz
spreads, Kyle lambda, Amihud illiquidity, and VPIN order-flow toxicity.

## Example

Runs as written; `tests/test_doc_examples.py` executes it on every test run.

```python
import numpy as np
from finlab.microstructure import tick_rule, roll_measure, corwin_schultz_spread, kyle_lambda, vpin

rng = np.random.default_rng(11)
prices = 100 + np.cumsum(rng.normal(0, 0.2, 400))
volume = rng.integers(1, 20, 400).astype(float)

print(tick_rule(prices)[:5])
print(round(roll_measure(prices), 4))
high = prices + rng.uniform(0, 0.3, 400)
low = prices - rng.uniform(0, 0.3, 400)
print(np.nanmean(corwin_schultz_spread(high, low)).round(4))
print(round(kyle_lambda(prices, volume * np.sign(np.diff(prices, prepend=prices[0]))), 5))
print(np.nanmean(vpin(prices, volume, bucket_volume=200.0, n_window=10)).round(3))
```

## Notes

Proof: [microstructure.md](../proofs/microstructure.md). Scope: the default VPIN scale is causal; the Corwin-Schultz and Kyle results are claimed from the book only.

Full signatures: [API reference](../API.md).
