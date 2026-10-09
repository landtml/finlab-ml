# `finlab.bet_sizing`: Bet sizing from probabilities and forecasts (AFML ch. 10)

Turns model output into a position in [-1, 1] or an integer target position. Includes the
sigmoid sizing with its calibration, breakeven limit prices, averaging of overlapping
signals, and discretization to limit overtrading.

## Example

Runs as written; `tests/test_doc_examples.py` executes it on every test run.

```python
import pandas as pd
from finlab.bet_sizing import prob_bet_size, calibrate_sigmoid_width, target_position, limit_price, discretize_signal

print(prob_bet_size([0.7, 0.5, 0.1]).round(3))    # probability to bet size

w = calibrate_sigmoid_width(10.0, 0.95)           # size 0.95 at a divergence of 10
print(target_position(w, forecast=115, market_price=100, max_position=100))   # 97
print(round(limit_price(97, 0, 115.0, w, 100), 4))                             # 112.3657
print(discretize_signal([0.33, -0.61], 0.2))
```

## Notes

Proof: [bet_sizing.md](../proofs/bet_sizing.md). Scope: the mixture-of-Gaussians budgeting approach is not implemented.

Full signatures: [API reference](../API.md).
