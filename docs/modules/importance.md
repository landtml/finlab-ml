# `finlab.importance`: Feature importance: MDI, MDA, SFI, orthogonal features (AFML ch. 8)

MDI reads tree impurity importances you already have. MDA and SFI measure out-of-sample
score loss from permuting or isolating a feature under a purged CV. `orthogonal_features`
rotates correlated features into uncorrelated principal components.

## Example

Runs as written; `tests/test_doc_examples.py` executes it on every test run.

```python
import numpy as np
from finlab.cv import PurgedKFold
from finlab.importance import mda, orthogonal_features, mdi_from_matrix

class Centroid:
    def fit(self, X, y, sample_weight=None):
        self.classes_ = np.unique(y)
        self.c_ = np.stack([X[y == c].mean(axis=0) for c in self.classes_])
        return self
    def predict(self, X):
        return self.classes_[((X[:, None] - self.c_[None]) ** 2).sum(2).argmin(1)]

rng = np.random.default_rng(5)
y = rng.integers(0, 2, 300)
X = np.column_stack([rng.normal(size=300) + 2 * (y - 0.5), rng.normal(size=(300, 2))])
baseline, drops = mda(Centroid, X, y, PurgedKFold(n_splits=4))
print(drops["mean"].round(3).tolist())   # feature 0 should show the largest drop

print(mdi_from_matrix(rng.random((20, 3)), names=["a", "b", "c"]).shape)
P, eigvals, eigvecs = orthogonal_features(X)
```

## Notes

Proof: [importance.md](../proofs/importance.md). Scope: MDA and SFI purge only if you pass a purged splitter; MDI is in-sample.

Full signatures: [API reference](../API.md).
