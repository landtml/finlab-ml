# `finlab.structural_breaks`: Structural breaks: CUSUM and SADF (AFML ch. 17)

CUSUM tests detect a change in mean or regression parameters. SADF is the supremum of
augmented Dickey-Fuller statistics over expanding windows, which flags explosive behaviour.

## Example

Runs as written; `tests/test_doc_examples.py` executes it on every test run.

```python
import numpy as np
from finlab.structural_breaks import cusum_test, chu_stinchcombe_white, sadf

rng = np.random.default_rng(9)
x = np.concatenate([rng.normal(0, 1, 200), rng.normal(2, 1, 200)])
res = cusum_test(x)
print(type(res).__name__)
print(chu_stinchcombe_white(x, reference=100).__class__.__name__)
s = sadf(np.cumsum(rng.normal(size=300)), min_length=40)
print(np.isfinite(np.asarray(s)).any())
```

## Notes

Proof: [structural_breaks.md](../proofs/structural_breaks.md). Scope: no p-values for CUSUM; the Chu-Stinchcombe-White critical value is read from an ambiguous book radical.

Full signatures: [API reference](../API.md).
