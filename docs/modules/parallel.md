# `finlab.parallel`: Multiprocessing helpers (AFML ch. 20)

`mp_pandas_obj` applies a function to contiguous chunks of an index, serially or in a
process pool, and concatenates the results in order. `lin_parts` and `nested_parts`
compute the chunk boundaries.

## Example

Runs as written; `tests/test_doc_examples.py` executes it on every test run.

```python
import pandas as pd
from finlab.parallel import mp_pandas_obj, lin_parts

def square(molecule, offset=0):
    return pd.Series(molecule.astype(float) ** 2 + offset, index=molecule)

idx = pd.RangeIndex(1000)
out = mp_pandas_obj(square, idx, num_threads=1, offset=1)
print(len(out), out.iloc[3])
print(lin_parts(10, 3))
```

## Notes

Proof: [parallel.md](../proofs/parallel.md). Scope: functions passed with `num_threads > 1` must be picklable, meaning defined at module level.

Full signatures: [API reference](../API.md).
