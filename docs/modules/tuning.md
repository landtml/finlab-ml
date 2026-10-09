# `finlab.tuning`: Hyper-parameter search with purged CV (AFML ch. 9)

Grid and randomized search where each candidate is scored by a leakage-free splitter.
The reported best score is optimistically biased; nested CV is needed for an unbiased
estimate and is not implemented.

## Example

Runs as written; `tests/test_doc_examples.py` executes it on every test run.

```python
import numpy as np
from finlab.cv import PurgedKFold
from finlab.tuning import grid_search

class Threshold:
    def __init__(self, threshold=0.0):
        self.threshold = threshold
    def fit(self, X, y, sample_weight=None):
        return self
    def predict(self, X):
        return (X[:, 0] > self.threshold).astype(int)

rng = np.random.default_rng(6)
X = rng.normal(size=(300, 1))
y = (X[:, 0] > 0.5).astype(int)
res = grid_search(Threshold, {"threshold": [-1.0, 0.0, 0.5, 2.0]}, X, y, PurgedKFold(5))
print(res.best_params, round(res.best_score, 3))
```

## Notes

Proof: [tuning.md](../proofs/tuning.md).

Full signatures: [API reference](../API.md).
