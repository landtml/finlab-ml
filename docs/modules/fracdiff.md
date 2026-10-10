# `finlab.fracdiff`: Fractionally differentiated features (AFML ch. 5)

Differencing of order d in [0, 1] makes a price series stationary while keeping memory.
Fixed-width differencing (`frac_diff_ffd`) uses a window of weights; `frac_diff` is the
expanding-window version. The search `find_min_d` returns the smallest d whose
fixed-width series passes a stationarity test you supply.

## Example

Runs as written; `tests/test_doc_examples.py` executes it on every test run.

```python
import numpy as np
from finlab.fracdiff import frac_diff, get_weights

rng = np.random.default_rng(3)
log_price = np.cumsum(rng.normal(0, 0.01, 500))

print(get_weights(0.5, 4).round(4))     # first weights for d = 0.5
x = frac_diff(log_price, d=0.4)
print(np.corrcoef(log_price[-300:], x[-300:])[0, 1].round(3))
```

## Notes

Proof: [fracdiff.md](../proofs/fracdiff.md). Scope: no ADF test is bundled; `find_min_d` needs one injected.

Full signatures: [API reference](../API.md).
