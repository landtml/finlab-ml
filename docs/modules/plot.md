# `quantowl.plot`: Plotly figures for PBO, deflated Sharpe and Monte Carlo results (optional)

Three functions that draw results quantowl has already computed. Each returns a plotly
`Figure`. The functions bin, group and draw the numbers they receive and compute no
statistics: the PBO, deflated Sharpe and uniqueness values on the figures are read from the
result objects as they are. Needs the `plot` extra: `pip install 'quantowl[plot]'`.

## Example

Runs as written; `tests/test_doc_examples.py` executes it on every test run. Without plotly
it prints the install hint instead of drawing.

```python
import numpy as np
from quantowl import plot
from quantowl.pbo import probability_of_backtest_overfitting
from quantowl.stats import deflated_sharpe

rng = np.random.default_rng(0)
res = probability_of_backtest_overfitting(rng.normal(0.0, 0.01, size=(64, 6)), n_partitions=8)
dsr = deflated_sharpe(sr_hat=0.21, n_trials=40, var_sr=0.02, n_obs=252)

try:
    fig = plot.pbo_distribution(res)
    comparison = plot.deflated_sharpe_comparison(dsr)
except ImportError as exc:   # plotly is the optional "plot" extra
    print(exc)
else:
    print(len(fig.data), len(comparison.data))
```

## Functions

- **`pbo_distribution(result, *, title=None)`** draws one histogram of `result.logits`, the
  CSCV split logits from `probability_of_backtest_overfitting`. A dashed line marks logit 0,
  and an annotation shows `result.pbo` and `result.n_combinations`
  (`test_pbo_distribution_histogram_matches_logits`). The annotation shows the stored PBO,
  not a value recomputed from the logits (`test_pbo_distribution_shows_stored_pbo_not_a_recomputed_one`).
  Default title: "CSCV split logits".
- **`deflated_sharpe_comparison(results, *, title=None)`** draws grouped bars, one group per
  result: `sr_hat` and `sr_star` from a `DeflatedSharpeResult`. Each group's label carries that
  result's `dsr` (`test_deflated_sharpe_comparison_bars_match_result_fields`). It takes one
  result or a sequence (`test_deflated_sharpe_comparison_accepts_a_single_result`). An empty
  sequence raises `ValueError` (`test_deflated_sharpe_comparison_rejects_empty_sequence`).
- **`monte_carlo_histogram(frame, columns=("std_u", "seq_u"), *, title=None)`** overlays one
  semi-transparent histogram per named column of a frame from `bootstrap_uniqueness_mc`, one
  row per trial (`test_monte_carlo_histogram_overlays_both_columns`). A column the frame lacks
  raises `ValueError` (`test_monte_carlo_histogram_rejects_missing_column`). A value that is not
  a pandas DataFrame raises `TypeError` (`test_monte_carlo_histogram_rejects_non_pandas_frame`).

## Without plotly

`import quantowl.plot` works without plotly (`test_module_imports_without_plotly`). Calling any
of the three functions then raises `ImportError` with the install hint
(`test_each_function_raises_install_hint_without_plotly`). The plotting tests in
`tests/test_plot.py` skip when plotly is missing.

## Notes

No proof note: the module computes no statistic. The static guard
`test_plot_module_source_has_no_statistics_calls` checks by call name that `plot.py` does not
call a statistics function. It is a name check, so it cannot see an indirect call.

Tests: `tests/test_plot.py`. Full signatures: [API reference](../API.md).
