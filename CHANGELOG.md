# Changelog

All notable changes to this project are documented in this file. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

## [Unreleased]

## [0.1.4] - 2026-10-11

### Added

- The MCP tool reply carries a `warnings` list when a library call emits a warning. The key
  is present only when the list is not empty.
- `DatetimeIndex` inputs may carry one timezone or none, at any single resolution (seconds,
  microseconds or nanoseconds). The same instants at different resolutions or zones are
  aligned by instant in `get_events`, `average_active_signals`, the bar builders, and the
  `sample_weight` arguments of `importance`, `ensemble` and `tuning`.
- `PBO` warns when its input looks like cumulative equity curves rather than returns.
- `get_bins` warns when it drops every event and returns nothing.
- `README.md` lists the supported input forms under "Scope and limitations".

### Changed

- A vertical barrier that falls between bars is priced at the last bar at or before it, the
  bar the path scan uses. Labels whose barrier falls between bars change.
- A naive `DatetimeIndex` whose repeated label is not adjacent is refused by name before the
  bar builders and `cusum_filter` sort it. Unambiguous naive input and aware input are
  unchanged.
- `get_bins(meta=True)` refuses a NaN `side` by name, with the count of affected events.
- `average_uniqueness` refuses a NaN or non-positive concurrency inside a label span.
- `SequentialBootstrapBagging` refuses a non-integer target, a `None` label and a `NaT` in
  `t1` by name. It reads `pd.NA` features as NaN and passes `sample_weight` only when weights
  are given. `predict` refuses non-finite averaged probabilities.
- `tuning` scorers return one finite or NaN number per fold. A per-class output is refused by
  name. Grid ties keep the first candidate.
- `importance` (`mda`, `sfi`) passes `sample_weight` only when weights are given. Orthogonal
  features read `pd.NA` as NaN.
- `prob_bet_size` aligns a labelled `side` to a labelled probability by label, and refuses a
  label that matches nothing. Array inputs keep positional behaviour.
- `encode_quantile` warns when tied values straddle a bin edge. The codes do not change.
- `plug_in_entropy`, `lempel_ziv` and `kontoyiannis` refuse NaN and infinite list entries
  before coercion. Entropy messages refuse `NaT` and `pd.NA` by name.
- A named DataFrame predicted after a fit on an array warns that the names are not used.
- `purged` cross-validation and the other splitters refuse a `t1` whose end is before its start,
  and a `t1` of non-timestamp values, by name.
- The MCP server refuses a tool call whose argument keys the schema does not declare.
- The MCP server keeps the request id in the reply to a bare `NaN` or `Infinity` literal.
- `time_decay` refuses weights that are not one-dimensional.
- The per-period unit of `probabilistic_sharpe_ratio`, `deflated_sharpe_ratio` and
  `minTRL` is stated in their docstrings and in `docs/modules/stats.md`. `sharpe_ratio`
  annualizes by default, so pass `annualize=False` before feeding its output to these.
- `time_decay` divides by the whole-series total. Weights built on train and test together
  therefore carry test-period information into training weights. The docstring says so.

### Fixed

- Vertical-barrier expiry rounds up to the grid unit before the search.
- The embargo product is rounded before the ceiling is taken.
- The ADF regression is centred before the fit, so `tau` does not depend on the level of the
  series.
- Indicator matrices refuse NaN instead of counting it as active.
- Timezone-aware object-dtype datetime indexes are read for their zone, in `get_events`,
  `add_vertical_barrier` and the purged splitters. Object-dtype `t1` of datetime values is
  accepted by purged cross-validation.
- Same-instant volumes are accepted by the bar builders across units and zones.
- Fitted column names are kept when a refit is refused.
- `tuning` passes `sample_weight` to a fit only when weights are given, so estimators without
  the argument work.
- `importance` and `ensemble` align `sample_weight` by instant across zones.
- A scorer that returns more than one number per fold is refused.
- Library warnings reach the MCP client in the tool reply.

### Known limitations

- Supported input is a monotonic `DatetimeIndex` with one timezone or none, at any single
  resolution, with the same awareness across inputs. Object-dtype datetime indexes, mixed
  zones in one index, strings or integers used as timestamps, and non-monotonic indexes are
  refused.
- `get_daily_vol` on daily bars uses the last bar strictly before `t` minus one day, which is
  two trading days back for most bars. This matches AFML Snippet 3.1.
- The expanding-window `frac_diff` mask depends on the sample length, following AFML Snippet
  5.2. Outputs change with the sample length.
- `prob_bet_size` on an `(n, 2)` array returns `(n, 2)` opposite bets and emits no warning.
- `time_decay` normalises over the whole series, so a factor changes when a later weight
  changes.
- `probabilistic_sharpe_ratio`, `deflated_sharpe_ratio` and `minTRL` take a per-period
  Sharpe ratio and accept moments that no distribution can produce.
- `PBO` on identical trials gives 0.0, because ties favour the in-sample winner.
- `encode_quantile` keeps the codes for tied values at a bin edge and warns when they straddle
  it.
- `roll_measure`, `kyle_lambda` and `amihud_illiquidity` accept returns or prices in place of
  each other, and `sharpe_ratio` accepts price levels. These inputs are not checked.
- The MCP server accepts `NaN`, `Infinity` and string integers such as `"3"` where the schema
  says integer.
- The sequential bootstrap measures uniqueness on the event-start grid, not the bar grid.
- `get_events` does not warn when the target is above 1.
- `add_vertical_barrier` and `get_daily_vol` offset by 24 hours on the index. Across a
  daylight-saving change this can differ from 24 elapsed hours.

## [0.1.3] - 2026-10-11

### Changed

- `TrialRecord` and `TrialRegistry` compare by value. `TrialRecord` stays hashable.
  `TrialRegistry` is not hashable, because it is mutable. Parameter values compare by type,
  so `10` and `10.0` are different.
- `mp_pandas_obj`, and the functions built on it such as `run_trials`, start worker processes
  with `forkserver`, or with `spawn` where `forkserver` is unavailable. A script that uses
  `num_threads > 1` must guard its entry point with `if __name__ == "__main__":`. Without the
  guard, the script raises `RuntimeError` when it starts.
- The package declares its license with the SPDX expression `MIT`. Building it requires
  setuptools 77.0.3 or newer.
- The MCP server replies to a JSON-RPC batch with one error, code `-32600`, and does not reply
  to a message that is a client's response.

## [0.1.2] - 2026-10-10

### Changed

- `discretize_signal` raises `ValueError` for infinite input.
- `quasi_diagonalize` raises `ValueError` for an invalid or cyclic linkage matrix, and for a
  single-row link.
- `prob_bet_size` applies `side` when it is passed with two classes. Omitting `side` keeps the
  long bet.
- The MCP server returns a parse error for oversized integer literals and for `NaN` and
  `Infinity` literals. Boolean arguments must be JSON booleans. Nested arrays are rejected with
  the argument name and element index. Non-object `params` return error `-32602`.
- The VPIN bucket cap is 10,000,000. A larger value raises `ValueError`.
- Degenerate input raises `ValueError` or `TypeError` with the argument name, or emits a
  `UserWarning` that names the cause.

### Fixed

- `limit_price` returns NaN at the position cap. It averages over signed fills, so orders that
  cross zero are priced correctly, and it rejects a negative or infinite `w`.
- `chu_stinchcombe_white` returns NaN when volatility is zero.
- `grid_search` and `randomized_search` rank NaN candidates below finite ones. A string value in
  a tuning grid raises `TypeError`.
- `cv_score` passes a pandas `X` to the splitter, so a datetime-indexed `t1` resolves. A
  generator passed as `cv` works for every candidate. Zero-weight test folds emit a warning.
- `PurgedKFold` and `get_bins` emit a warning when they skip purging or labelling.
- `average_active_signals` drops events whose end is before their start, and excludes NaN
  signals from the average.
- `NaT` in `t1` raises `ValueError`. An infinite signal is confined to its active span.
- `sigmoid_bet_size` handles very large forecasts. `prob_bet_size` rejects probabilities below
  1/k. `calibrate_sigmoid_width` and `discretize_signal` reject inputs outside their domain.
- `sharpe_ratio` returns 0 for all-zero returns and NaN with a warning for other constant
  returns. It warns with fewer than two observations. A zero-return trial does not block the
  deflated Sharpe ratio.
- `probabilistic_sharpe_ratio` rejects a NaN or infinite `n_obs`. `sr_hat` must be finite for
  the probabilistic Sharpe ratio and the minimum track record length.
- `TrialRecord` rejects strings and bools as returns.
- `hrp_weights` checks the correlation shape for a single asset. `quasi_diagonalize` returns
  `[0]` for a single row.
- `PBO` warns when rows are out of time order. Its callable path accepts infinite scores and
  rejects NaN.
- `frac_diff_ffd` emits a warning when the window is longer than the series.
- `time_bars` and tick bars sort unsorted input. `time_bars` warns about a trailing unclosed
  interval.
- `get_events` raises `ValueError` when the target and close timezones differ, and warns when a
  vertical barrier falls between bars. `add_vertical_barrier` and `get_events` keep the index
  timezone on `t1`.
- Timezone-aware `y` and `X` indexes compare as instants.
- `get_daily_vol` accepts leading NaN from a panel, names duplicate timestamps, and rejects
  infinite input. `cusum_filter` sorts a Series input.
- `CUSUM` raises for non-finite sigma. When residuals overflow or underflow, the error names the
  series magnitude.
- `VPIN` rejects non-finite prices and volume.
- Non-integer lags and window sizes raise `ValueError`. Integer arguments keep their exact value,
  and oversized ones are reported by name.
- `mdi_from_matrix` rejects infinite input. A failed `fit` leaves no partial ensemble.
- `importance` and `ensemble` raise `ValueError` when the index of a pandas `y` or sample weight
  differs from the index of `X`.
- Tuning rejects a `y` of the wrong length, and a scalar or `None` `y`. Sample-weight length
  mismatches raise `ValueError`.
- `CPCV.split()` raises its `min_train_size` error when called.
- The MCP stdio server keeps running after malformed lines, deeply nested JSON and tool-side
  exceptions. The string `"false"` is rejected as a boolean argument.

## [0.1.1] - 2026-10-10

Changes since 0.1.0.

### Added

- Trial records (`finlab.trials`, not an AFML construct): `TrialRecord` and `TrialRegistry`
  count the trials, feed the performance matrix to PBO and the number of trials to the
  deflated Sharpe ratio, and round-trip through dict and JSON.
- `DeflatedSharpeResult` and `deflated_sharpe`, with validated inputs. PBO and deflated
  Sharpe results render as HTML tables in notebooks.
- Plotting (`finlab.plot`, optional `plot` extra): plotly figures for PBO, the deflated
  Sharpe ratio and the Monte Carlo experiment. `import finlab.plot` works without plotly.
- Worked notebooks in `examples/` for PBO, the deflated Sharpe ratio with trial records,
  the Monte Carlo experiment and fractional differentiation. Each has a fixed seed and a
  test of its key numbers, and CI executes them.
- Gallery figures in `docs/images/`, shown in the README.
- Ruff linting, `plot` and `notebooks` extras, and CI jobs that install the built wheel
  with and without the extras.

### Changed

- The Monte Carlo trial draws use numpy directly. End to end, one trial takes 54.0 us
  against 207.0 us for a dense NumPy trial with the same seeds. Measured on a shared
  4-vCPU machine; see `docs/proofs/monte_carlo.md`.
- The Monte Carlo experiment follows AFML Snippets 4.7 and 4.8.
- The README benchmark table is re-measured, and each row names its script in
  `benchmarks/`.

### Fixed

- `probability_of_backtest_overfitting` rejects non-integer and boolean `n_partitions`
  before use, and detects zero-variance columns exactly.

### Known limitations

- Some statements in the proof notes are claimed from the book only and have not been
  checked against the printed pages. Each one is labelled in its note.
- The numba kernel `_sample_uniqueness` (`finlab.monte_carlo`) does not check that span ends
  are below the bar count. Its public caller keeps them in range.
- `import finlab.plot` succeeds without plotly; calling a plotting function then raises an
  `ImportError` with the install command.
- The Monte Carlo notebook counts a trial as "higher" only if the difference exceeds 1e-12.
  The fraction is 0.7015 with that rule and 0.712 without it.
- The PBO notebook's edge of 0.003 was chosen on one seed. Over seeds 0 to 19 the mean PBO
  at that edge is 0.021.
- In the deflated Sharpe notebook, adding 40 pure-noise trials lowers the ratio for the
  pinned seed. Over seeds 1000 to 1199 it falls in 133 of 200 seeds.
- The approximation of the expected maximum of N Sharpe ratios differs from the numerical
  value by -7.9% at N = 2 and +0.4% at N = 1000.
- Benchmark timings are single runs on a shared 4-vCPU machine and vary between runs.
- `tests/test_cpcv.py` skips its live-data integration test unless `yfinance` is installed.

## [0.1.0] - 2026-10-09

First tagged release. No changelog was kept for it. Its source is at the `v0.1.0` tag.

[Unreleased]: https://github.com/landtml/finlab-ml/compare/v0.1.4...HEAD
[0.1.4]: https://github.com/landtml/finlab-ml/compare/v0.1.3...v0.1.4
[0.1.3]: https://github.com/landtml/finlab-ml/compare/v0.1.2...v0.1.3
[0.1.2]: https://github.com/landtml/finlab-ml/compare/v0.1.1...v0.1.2
[0.1.1]: https://github.com/landtml/finlab-ml/compare/v0.1.0...v0.1.1
[0.1.0]: https://github.com/landtml/finlab-ml/releases/tag/v0.1.0
