# Changelog

All notable changes to this project are documented in this file. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

## [Unreleased]

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

[Unreleased]: https://github.com/landtml/finlab-ml/compare/v0.1.1...HEAD
[0.1.1]: https://github.com/landtml/finlab-ml/compare/v0.1.0...v0.1.1
[0.1.0]: https://github.com/landtml/finlab-ml/releases/tag/v0.1.0
