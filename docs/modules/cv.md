# `finlab.cv`: Purged and embargoed cross-validation (AFML ch. 7)

Splitters for label-overlapping time series. Training observations whose labels
overlap the test window are purged, and an embargo removes a buffer after each test
block. `CombinatorialPurgedCV` yields every combination of test groups; `PurgedKFold`
yields one path. Both duck-type the sklearn splitter protocol without importing sklearn.

## Example

Runs as written; `tests/test_doc_examples.py` executes it on every test run.

```python
import numpy as np
import pandas as pd
from finlab.cv import CombinatorialPurgedCV, PurgedKFold, make_t1

idx = pd.date_range("2020-01-01", periods=120, freq="B")
t1 = make_t1(idx, horizon=5)            # each label resolves 5 bars later
X = pd.DataFrame(np.random.default_rng(0).normal(size=(120, 3)), index=idx)  # same index as t1

cpcv = CombinatorialPurgedCV(n_groups=6, n_test_groups=2, embargo_pct=0.01, t1=t1)
print(cpcv.get_n_splits(), "splits,", cpcv.get_n_paths(), "paths")
for train_idx, test_idx in cpcv.split(X):
    assert set(train_idx).isdisjoint(test_idx)

for train_idx, test_idx in PurgedKFold(n_splits=4, t1=t1).split(X):
    pass
```

## Notes

Proof: [cpcv.md](../proofs/cpcv.md). Scope: the guarantee covers the split boundary given the `t1` you supply, not leakage inside your own features.

Full signatures: [API reference](../API.md).
