# `finlab.hrp`: Hierarchical risk parity (AFML ch. 16)

Allocates a portfolio by clustering assets on correlation, quasi-diagonalizing the
covariance matrix, and bisecting it with inverse-variance weights. It needs no matrix
inversion.

## Example

Runs as written; `tests/test_doc_examples.py` executes it on every test run.

```python
import numpy as np
import pandas as pd
from finlab.hrp import hrp_weights

rng = np.random.default_rng(12)
returns = pd.DataFrame(rng.normal(size=(250, 5)) * [0.01, 0.02, 0.01, 0.015, 0.03],
                       columns=list("ABCDE"))
w = hrp_weights(returns.cov())
print(w.round(3).to_dict(), round(float(w.sum()), 6))
```

## Notes

Proof: [hrp.md](../proofs/hrp.md). Scope: out-of-sample HRP results are claimed from the book only.

Full signatures: [API reference](../API.md).
