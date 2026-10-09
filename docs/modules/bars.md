# `finlab.bars`: Information-driven bars and the CUSUM event filter (AFML ch. 2)

Sampling by activity rather than clock time. Tick, volume and dollar bars close after a
fixed amount of activity; imbalance and run bars close when the order flow becomes
unbalanced. `cusum_filter` picks event times where cumulative change exceeds a threshold.

## Example

Runs as written; `tests/test_doc_examples.py` executes it on every test run.

```python
import numpy as np
from finlab.bars import tick_bars, dollar_bars, cusum_filter

rng = np.random.default_rng(1)
prices = 100 + np.cumsum(rng.normal(0, 0.1, 2000))
volumes = rng.integers(1, 50, 2000).astype(float)

bars = tick_bars(prices, threshold=100)
print(bars.shape, list(bars.columns)[:4])

db = dollar_bars(prices, volumes, threshold=50_000.0)
events = cusum_filter(prices, threshold=1.0)
print(len(db), "dollar bars,", len(events), "CUSUM events")
```

## Notes

Proof: [bars.md](../proofs/bars.md). Scope: imbalance and run bars estimate their expected imbalance from the first `init_T` ticks, which is a warm-up look-ahead (see the README).

Full signatures: [API reference](../API.md).
