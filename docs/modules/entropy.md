# `finlab.entropy`: Entropy features (AFML ch. 18)

Estimators of how predictable a symbol sequence is. Encoders map returns to symbols by
quantile, by sigma bins, or by sign, so the entropy estimators can be applied to returns.

## Example

Runs as written; `tests/test_doc_examples.py` executes it on every test run.

```python
import numpy as np
from finlab.entropy import plug_in_entropy, kontoyiannis_entropy, encode_quantile, encode_binary

rng = np.random.default_rng(10)
returns = rng.normal(size=500)
symbols = encode_quantile(returns, n_bins=4)
print(round(plug_in_entropy(symbols), 3))
print(round(kontoyiannis_entropy(encode_binary(returns)), 3))
print(plug_in_entropy([0, 0, 0, 0]))   # a constant string has zero entropy
```

## Notes

Proof: [entropy.md](../proofs/entropy.md). Scope: the Lempel-Ziv normalisation is the standard LZ78 one, not from AFML.

Full signatures: [API reference](../API.md).
