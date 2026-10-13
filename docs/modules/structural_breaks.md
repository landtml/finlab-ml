# `finlab.structural_breaks`: Structural breaks: CUSUM, SADF and Chow-type DF (AFML ch. 17)

CUSUM tests detect a change in mean or regression parameters. SADF is the supremum of
augmented Dickey-Fuller statistics over expanding windows, which flags explosive behaviour.
The Chow-type Dickey-Fuller statistic (`chow_dfc`) tests for explosive behaviour after a
break at a chosen date. `sdfc` takes its supremum over admissible break dates.

## Example

Runs as written; `tests/test_doc_examples.py` executes it on every test run.

```python
import numpy as np
from finlab.structural_breaks import cusum_test, chow_dfc, chu_stinchcombe_white, sadf, sdfc

rng = np.random.default_rng(9)
x = np.concatenate([rng.normal(0, 1, 200), rng.normal(2, 1, 200)])
res = cusum_test(x)
print(type(res).__name__)
print(chu_stinchcombe_white(x, reference=100).__class__.__name__)
walk = np.cumsum(rng.normal(size=300))
s = sadf(walk, min_length=40)
print(np.isfinite(np.asarray(s)).any())
print(np.isfinite(chow_dfc(walk, 0.5)))
print(np.isfinite(sdfc(walk, tau0=0.15).statistic))
```

## Chow-type Dickey-Fuller

`chow_dfc(series, tau)` regresses `dy_t` on `y_{t-1} * D_t`, with no intercept and no lags,
where `D_t = 0` before the break and `D_t = 1` after it. The break is the first observation
at or after `ceil(tau * T)`. It returns the t-statistic of the slope.

`sdfc(series, tau0)` takes the maximum of that statistic over the breaks in
`[tau0, 1 - tau0]`. It returns an `SDFCResult` with the maximum, the argmax break index
(a position in the series, not a date), and the full path over the admissible breaks.

## Notes

Proof: [structural_breaks.md](../proofs/structural_breaks.md). Scope: no p-values for CUSUM,
SADF or the Chow-type statistics. The Chu-Stinchcombe-White critical value is read from an
ambiguous book radical.

Full signatures: [API reference](../API.md).
