# Changelog

All notable changes to this project are documented in this file. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

## [Unreleased]

## [0.1.2] - 2026-10-10

Changes since 0.1.1: a QA pass over the library, and a change to how releases are published.

### Changed

- A merge to `main` that sets a new version in `pyproject.toml` is tagged and published
  by the Release workflow once CI passes on it, with the sdist and wheel attached.
  The release notes come from `.github/releases/vX.Y.Z.md`, or from this changelog
  section when that file is absent.
- `RELEASING.md` describes the release pull request and the steps that follow it.
- Degenerate or invalid input now raises a `ValueError` or `TypeError` that names the
  argument, or emits a `UserWarning` that names the cause. Valid-input results are unchanged,
  except where listed under Fixed.
- `prob_bet_size` applies `side` when it is passed with two classes. Omitting `side` keeps
  the long bet, as before.
- `discretize_signal` raises on infinite input; it used to warn.
- `quasi_diagonalize` raises `ValueError` for an invalid or cyclic linkage matrix, and for a
  single-row link.
- The MCP server rejects oversized integer literals, `NaN` and `Infinity` literals, and
  non-boolean values for boolean arguments. Nested arrays are rejected with the argument name
  and element index. Non-object `params` return error `-32602`.
- The VPIN bucket cap is 10,000,000.

### Added

- `tools/check_attribution.py` and a CI job that rejects AI attribution in new commit messages
  and pull request bodies.
- `tools/prune_branches.py` and a branch-cleanup workflow.
- The release workflow repairs a tag that has no sdist or wheel asset.

### Fixed

- `limit_price` returns NaN at the position cap instead of dividing by zero. It averages over
  signed fills, so orders that cross zero are correct, and it signs short levels correctly.
- `limit_price` rejects a negative or infinite `w`.
- `chu_stinchcombe_white` returns NaN when volatility is zero, and no longer returns silent
  NaN or zero at extreme magnitudes.
- `grid_search` and `randomized_search` rank NaN candidates below finite ones. A `str` value in
  a tuning grid raises `TypeError`.
- `cv_score` passes a pandas `X` to the splitter, so a datetime-indexed `t1` resolves.
  A generator passed as `cv` works for every candidate. Zero-weight test folds warn.
- `PurgedKFold` and `get_bins` warn where they used to skip purging or labelling silently.
- `average_active_signals` drops events whose end is before their start, and a NaN signal is
  excluded from the average instead of making every later value NaN.
- `NaT` in `t1` raises `ValueError`. An infinite signal is confined to its active span.
- `sigmoid_bet_size` no longer returns 0 for very large forecasts. `prob_bet_size` rejects
  probabilities below 1/k. `calibrate_sigmoid_width` and `discretize_signal` reject inputs
  outside their domain.
- `sharpe_ratio` returns 0 for all-zero returns, and NaN with a warning for other constant
  returns. It warns with fewer than two observations. A zero-return trial no longer blocks the
  deflated Sharpe ratio.
- `probabilistic_sharpe_ratio` rejects a NaN or infinite `n_obs`, and `sr_hat` must be finite
  for PSR and minimum track record length.
- `TrialRecord` rejects strings and bools as returns.
- `hrp_weights` checks the correlation shape for a single asset. `quasi_diagonalize` returns
  `[0]` for a single row.
- `PBO` warns when rows are out of time order. Its callable path accepts infinite scores and
  rejects NaN.
- `frac_diff_ffd` warns when the window is longer than the series, instead of returning all NaN.
- `time_bars` and tick bars sort unsorted input first. `time_bars` warns about a trailing
  unclosed interval.
- `get_events` raises `ValueError` when the target and close timezones differ, and warns when a
  vertical barrier falls between bars. `add_vertical_barrier` and `get_events` keep the index
  timezone on `t1`.
- Timezone-aware `y` and `X` indexes compare as instants.
- `get_daily_vol` allows leading NaN from a panel, names duplicate timestamps, and rejects
  infinite input. `cusum_filter` sorts a Series input.
- `CUSUM` raises for non-finite sigma, and names the series magnitude when residuals overflow
  or underflow. It no longer blames collinear regressors for series at extreme magnitudes.
- `VPIN` rejects non-finite prices and volume. Its bucket cap raises `ValueError` above the
  limit, instead of a raw `MemoryError`.
- Non-integer lags and window sizes raise `ValueError`. Integer arguments stay exact, and
  oversized ones are reported by name.
- `mdi_from_matrix` rejects infinite input. A failed `fit` no longer leaves a partial ensemble.
- `importance` and `ensemble` raise `ValueError` when the index of a pandas `y` or sample weight
  differs from `X`'s, instead of matching labels by position.
- Tuning rejects a `y` of the wrong length, and a scalar or `None` `y`, with a clear error.
  Sample-weight length mismatches raise `ValueError`.
- `CPCV` `split()` raises its `min_train_size` error eagerly.
- The MCP stdio server survives malformed lines, deep JSON nesting and tool-side exceptions.
  It replies with a parse error for oversized integers, and the string `"false"` is rejected
  rather than read as `true`.

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
- The test suite gives each run a fresh numba cache, because a stale cache can keep an
  old cross-module callee.
- The proof notes correct the runs-bar Jensen condition, the HRP bisection cost bound and
  the minTRL check. The minTRL check now uses a tolerance of 1e-9. Each "checked by test"
  label names a test.
- The HRP distance docstring (a metric only on standardized series) and the zero-variance
  note in `TrialRegistry` match the code.

### Known limitations

- Some statements in the proof notes are claimed from the book only and have not been
  checked against the printed pages. Each one is labelled in its note.
- The numba kernels `_sample_uniqueness` (`finlab.monte_carlo`) and `_concurrency_sweep`
  (`finlab.weights`) do not check that span ends are below the bar count. The public callers
  keep them in range.
- With zero draws, the Monte Carlo kernel raises `ZeroDivisionError` and the dense reference
  raises `ValueError`.
- On Python 3.12 and later, the default `fork` start method emits a `DeprecationWarning` from
  the process pool in `finlab.parallel`.
- `setuptools` warns about the `license` table and the license classifier in
  `pyproject.toml`. The package still builds and passes `twine check`.
- numba's cache key covers only the file that defines a function, so a cached caller can
  keep a stale callee from another module after an edit. `tests/conftest.py` therefore gives
  every test run a fresh cache directory.
- `TrialRecord` and `TrialRegistry` compare by identity, not by value.
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

[Unreleased]: https://github.com/landtml/finlab-ml/compare/v0.1.2...HEAD
[0.1.2]: https://github.com/landtml/finlab-ml/compare/v0.1.1...v0.1.2
[0.1.1]: https://github.com/landtml/finlab-ml/compare/v0.1.0...v0.1.1
[0.1.0]: https://github.com/landtml/finlab-ml/releases/tag/v0.1.0
