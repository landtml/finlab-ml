# `quantowl.microstructure`: Microstructure features (AFML ch. 19)

Spread and liquidity estimators from prices and volumes: tick rule, Roll and Corwin-Schultz
spreads, Kyle and Hasbrouck price-impact lambdas, Amihud illiquidity, and VPIN order-flow toxicity.

## Example

Runs as written; `tests/test_doc_examples.py` executes it on every test run.

```python
import numpy as np
from quantowl.microstructure import (
    tick_rule, roll_measure, corwin_schultz_spread, kyle_lambda, vpin,
    hasbrouck_lambda, signed_root_dollar_volume,
)

rng = np.random.default_rng(11)
prices = 100 + np.cumsum(rng.normal(0, 0.2, 400))
volume = rng.integers(1, 20, 400).astype(float)

print(tick_rule(prices)[:5])
print(round(roll_measure(prices), 4))
high = prices + rng.uniform(0, 0.3, 400)
low = prices - rng.uniform(0, 0.3, 400)
print(np.nanmean(corwin_schultz_spread(high, low)).round(4))
signs = np.sign(np.diff(prices, prepend=prices[0]))
print(round(kyle_lambda(prices, volume * signs), 5))
print(np.nanmean(vpin(prices, volume, bucket_volume=200.0, n_window=10)).round(3))
bar = np.arange(400) // 4  # four trades per bar, so 100 bars
srdv = signed_root_dollar_volume(prices, volume, signs, bar)
print(round(hasbrouck_lambda(prices[3::4], srdv), 6))
```

## Notes

Proof: [microstructure.md](../proofs/microstructure.md). Scope: the default VPIN scale is causal; the Corwin-Schultz and Kyle results are claimed from the book only. The Hasbrouck regression form is claimed from the book only (section 19.4.3, not verified against the text).

`hasbrouck_lambda` takes one closing price and one per-bar sum for each bar. The caller chooses the bars. `signed_root_dollar_volume` adds each trade into the bar number it is given, and it does not check that those numbers match the closes.

Full signatures: [API reference](../API.md).
