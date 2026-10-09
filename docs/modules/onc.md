# `finlab.onc`: Optimal number of clusters on a correlation matrix (ONC)

Clusters variables on a correlation matrix and chooses the number of clusters by mean
silhouette. ONC is from Lopez de Prado, Lewis and Boudt (2019), not AFML. This version
omits the paper's repair step.

## Example

Runs as written; `tests/test_doc_examples.py` executes it on every test run.

```python
import numpy as np
from finlab.onc import onc

rng = np.random.default_rng(13)
data = np.vstack([rng.normal(size=(100, 1)) + rng.normal(0, 0.3, (100, 4)) for _ in range(2)]).T
res = onc(np.corrcoef(data), seed=0)
print(res.n_clusters, round(res.silhouette, 3))
```

## Notes

Proof: [onc.md](../proofs/onc.md). Scope: the published repair step and quality-ratio ranking are not implemented.

Full signatures: [API reference](../API.md).
