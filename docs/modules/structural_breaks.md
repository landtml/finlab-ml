# `finlab.structural_breaks`: Structural breaks: CUSUM, SADF, QADF, SMT and Chow-type DF (AFML ch. 17)

CUSUM tests detect a change in mean or regression parameters. SADF is the supremum of
augmented Dickey-Fuller statistics over expanding windows, which flags explosive behaviour.
`qadf` and `cadf` replace that supremum with a quantile and a conditional mean of the same
statistics. `smt` takes the supremum of a curvature or growth coefficient over start points.
`sadf_null_quantiles` simulates critical values for `sadf` under a random-walk null.
The Chow-type Dickey-Fuller statistic (`chow_dfc`) tests for explosive behaviour after a
break at a chosen date. `sdfc` takes its supremum over admissible break dates.

## Example

Runs as written; `tests/test_doc_examples.py` executes it on every test run.

```python
import numpy as np
from finlab.structural_breaks import cusum_test, chow_dfc, chu_stinchcombe_white, qadf, sadf, sdfc, smt

rng = np.random.default_rng(9)
x = np.concatenate([rng.normal(0, 1, 200), rng.normal(2, 1, 200)])
res = cusum_test(x)
print(type(res).__name__)
print(chu_stinchcombe_white(x, reference=100).__class__.__name__)
walk = np.cumsum(rng.normal(size=300))
s = sadf(walk, min_length=40)
print(np.isfinite(np.asarray(s)).any())
print(np.isfinite(np.asarray(qadf(walk, min_length=40, lag=1, q=0.95))).any())
print(np.isfinite(chow_dfc(walk, 0.5)))
print(np.isfinite(sdfc(walk, tau0=0.15).statistic))
growth = np.exp(0.01 * np.arange(200))
print(np.isfinite(np.asarray(smt(growth, min_length=30, kind="exp"))).any())
```

## Simulated critical values for SADF

`sadf` returns the statistic only. `sadf_null_quantiles(n_obs, min_length, lags, constant,
quantiles, n_sim, seed)` simulates `n_sim` random walks of length `n_obs` with N(0, 1)
increments, runs `sadf` on each with the same `min_length`, `lags` and `constant`, and returns
two objects. `pointwise` is a DataFrame with one row per end point (the index matches the
numpy output of `sadf`) and one column per quantile level: entry `(t, q)` is the `q`-quantile
of `SADF_t` over the walks. `maximum` is a Series with the quantiles of `max_t SADF_t`, the
statistic that flags an explosive episode anywhere in the sample.

```python
import numpy as np
from finlab.structural_breaks import sadf, sadf_null_quantiles

pointwise, maximum = sadf_null_quantiles(200, min_length=40, n_sim=200, seed=1)
boom = np.exp(0.02 * np.arange(200)) * (1 + 0.01 * np.random.default_rng(2).normal(size=200))
print(pointwise.shape, list(maximum.index))
print(bool(np.nanmax(sadf(boom, min_length=40)) > maximum[0.95]))
```

Use the walk length of the series you test, and the same `min_length`, `lags` and `constant`.
`constant='nc'` is refused: its statistic depends on the starting level, so no single null
applies. Pass `seed` to reproduce the output. Each call costs one `sadf` run per simulated
walk: about 12 s at `n_obs = 200`, `min_length = 20`, `n_sim = 1000` on the machine used for
the proof note (section 3.6). The quantile estimates have Monte Carlo error that falls as
`1 / sqrt(n_sim)`; the proof note gives the formula and the measured spread.

## Quantile and conditional ADF

For each end point, the ADF statistics over the same start points that `sadf` maximises form
a set `A_t`. `qadf(series, min_length, lag, q)` returns the `q`-quantile of `A_t`, linear
interpolation as in `np.quantile`. Passing `v` also returns the dispersion
`QADF(q + v) - QADF(q - v)`. `cadf` returns the mean of the elements of `A_t` at or above
`QADF(q)`. At `q = 1`, both equal `sadf`.

## Sub- and super-martingale test

`smt(series, min_length, kind, phi)` fits, on every window `[t0, t]` of at least
`min_length` levels, one of four regressions. The statistic is `|b| / (se(b) (t - t0)^phi)`,
and `SMT_t` is its supremum over `t0`. The kinds are `'poly1'` (levels, quadratic in time),
`'poly2'` (logs, quadratic in time), `'exp'` (logs, linear in time) and `'power'` (logs,
linear in log time). Time is the 1-based position in the full series. The log kinds need
strictly positive values, and `smt` refuses others by name.

## Chow-type Dickey-Fuller

`chow_dfc(series, tau)` regresses `dy_t` on `y_{t-1} * D_t`, with no intercept and no lags,
where `D_t = 0` before the break and `D_t = 1` after it. The break is the first observation
at or after `ceil(tau * T)`. It returns the t-statistic of the slope.

`sdfc(series, tau0)` takes the maximum of that statistic over the breaks in
`[tau0, 1 - tau0]`. It returns an `SDFCResult` with the maximum, the argmax break index
(a position in the series, not a date), and the full path over the admissible breaks.

## Notes

Proof: [structural_breaks.md](../proofs/structural_breaks.md). Scope: no p-values for any statistic.
Simulated critical values exist for SADF only (`sadf_null_quantiles`); CUSUM, QADF, CADF, SMT
and the Chow-type statistics have none. The Chu-Stinchcombe-White critical value
is read from an ambiguous book radical.

Full signatures: [API reference](../API.md).
