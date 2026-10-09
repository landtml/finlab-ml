# `finlab.ensemble`: Bagging with sequential bootstrap (AFML ch. 6)

Ordinary bagging draws overlapping labels redundantly. `SequentialBootstrapBagging` draws
each bag with sequential bootstrap, so the trees see more distinct information.

## Example

Runs as written; `tests/test_doc_examples.py` executes it on every test run.

```python
import numpy as np
import pandas as pd
from finlab.cv import make_t1
from finlab.ensemble import SequentialBootstrapBagging

class Centroid:
    def fit(self, X, y, sample_weight=None):
        self.classes_ = np.unique(y)
        self.c_ = np.stack([X[y == c].mean(axis=0) for c in self.classes_])
        return self
    def predict(self, X):
        return self.classes_[((X[:, None] - self.c_[None]) ** 2).sum(2).argmin(1)]

rng = np.random.default_rng(4)
idx = pd.date_range("2020-01-01", periods=200, freq="D")
y = rng.integers(0, 2, 200)
X = rng.normal(size=(200, 2)) + np.where(y[:, None] == 1, 1.5, -1.5)

bag = SequentialBootstrapBagging(Centroid, n_estimators=10, seed=0).fit(X, y, t1=make_t1(idx, 10))
print("train accuracy", (bag.predict(X) == y).mean().round(3))
```

## Notes

Proof: [ensemble.md](../proofs/ensemble.md). Scope: out-of-bag estimation is not implemented.

Full signatures: [API reference](../API.md).
