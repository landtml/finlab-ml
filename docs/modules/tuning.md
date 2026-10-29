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

## Pipeline weights

A model factory may return a scikit-learn `Pipeline`. A Pipeline refuses a bare `sample_weight`
keyword, so when weights are given the search passes them to the final step as
`<step>__sample_weight`, the form AFML Snippet 9.2 uses. The final step must be an estimator; a
Pipeline ending in `None` or `"passthrough"` raises `ValueError`. With scikit-learn metadata
routing enabled, the weights go as `sample_weight` instead, and every step that takes them
must set its request, for example `StandardScaler().set_fit_request(sample_weight=False)`.
`tests/test_tuning.py` checks both forms against a manual loop.

## Notes

Proof: [tuning.md](../proofs/tuning.md).

Full signatures: [API reference](../API.md).
