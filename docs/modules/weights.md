# `finlab.weights`: Concurrency, uniqueness, sample weights and sequential bootstrap (AFML ch. 4)

When labels overlap in time, observations are not independent. Average uniqueness measures
how much information each label adds. Sequential bootstrap draws labels that are more
unique than an ordinary bootstrap would.

## Example

Runs as written; `tests/test_doc_examples.py` executes it on every test run.

```python
import pandas as pd
from finlab.cv import make_t1
from finlab.weights import num_co_events, average_uniqueness, indicator_matrix, sequential_bootstrap

idx = pd.date_range("2020-01-01", periods=60, freq="D")
t1 = make_t1(idx, horizon=10)
c_t = num_co_events(idx, t1)
u = average_uniqueness(idx, t1, c_t)
print("mean uniqueness", round(float(u.mean()), 3))

draws = sequential_bootstrap(indicator_matrix(idx, t1), seed=0)
print(len(draws), "draws")
```

## Notes

Proof: [weights.md](../proofs/weights.md). Scope: the sequential-bootstrap uniqueness gain is modest in measured settings. For a Monte Carlo comparison of standard and sequential bootstrap uniqueness, see [`finlab.monte_carlo`](monte_carlo.md).

Full signatures: [API reference](../API.md).
