# Changelog

All notable changes to this project are documented in this file. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

## [Unreleased]

## [0.1.0] - 2026-10-10

First release.

### Added

- Cross-validation for overlapping labels (`finlab.cv`, chapter 7): `PurgedKFold`,
  `CombinatorialPurgedCV` with path stitching, and `make_t1`, with purging and embargo.
- Financial data structures (`finlab.bars`, chapter 2): time, tick, volume, dollar,
  imbalance and run bars, and the CUSUM filter.
- Labelling (`finlab.labeling`, chapter 3): daily volatility, triple-barrier events, bin
  labels, meta-labels, rare-label dropping, and trend-scanning labels (not in AFML).
- Sample weights (`finlab.weights`, chapter 4): concurrency, average uniqueness, the
  sequential bootstrap, return attribution and time decay.
- The chapter 4 Monte Carlo experiment (`finlab.monte_carlo`): a seeded trial runner
  (`run_trials`), random label sets (`random_t1`) and the standard against sequential
  bootstrap uniqueness experiment (`bootstrap_uniqueness_mc`), following Snippets 4.7 and 4.8.
- Fractional differentiation (`finlab.fracdiff`, chapter 5): fixed-width and
  expanding-window weights, and a minimum-`d` search with an injected ADF test.
- Ensembles (`finlab.ensemble`, chapter 6): `SequentialBootstrapBagging`.
- Feature importance (`finlab.importance`, chapter 8): MDI, MDA, SFI and orthogonal
  features.
- Hyper-parameter search scored by purged CV (`finlab.tuning`, chapter 9).
- Bet sizing (`finlab.bet_sizing`, chapter 10): probability and sigmoid sizing, target
  positions, limit prices and discretisation.
- Backtest statistics: the probability of backtest overfitting by CSCV (`finlab.pbo`,
  chapter 11); Sharpe, probabilistic and deflated Sharpe ratios and the minimum track record
  length (`finlab.stats`, chapter 14), with result objects `PBOResult` and
  `DeflatedSharpeResult` that render as HTML tables in notebooks.
- Trial records (`finlab.trials`, not an AFML construct): `TrialRecord` and `TrialRegistry`
  count the trials and feed the performance matrix to PBO and the number of trials to the
  deflated Sharpe ratio. Both round-trip through dict and JSON.
- Portfolio construction: hierarchical risk parity (`finlab.hrp`, chapter 16) and ONC
  correlation clustering without the repair step (`finlab.onc`).
- Structural breaks (`finlab.structural_breaks`, chapter 17): CUSUM tests,
  Chu-Stinchcombe-White and SADF.
- Entropy (`finlab.entropy`, chapter 18): plug-in, Lempel-Ziv and Kontoyiannis estimators
  and encoders.
- Microstructure (`finlab.microstructure`, chapter 19): tick rule, Roll, Corwin-Schultz,
  Becker-Parkinson, Kyle lambda, Amihud and VPIN with bulk-volume classification.
- Multiprocessing helpers (`finlab.parallel`, chapter 20): molecule partitioning and
  `mp_pandas_obj` on `concurrent.futures`.
- Plotting (`finlab.plot`, optional `plot` extra): plotly figures for PBO, the deflated
  Sharpe ratio and the Monte Carlo experiment.
- An MCP server (`finlab-mcp`, `finlab.agent`) that exposes selected computations as tools
  with JSON input and output; see `docs/AGENTS.md`.
- Numba kernels for the hot loops, most of them tested against a plain NumPy or Python
  reference, and benchmarks for them in `benchmarks/`.
- Worked notebooks in `examples/` for PBO, the deflated Sharpe ratio with trial records,
  the Monte Carlo experiment and fractional differentiation, each with a fixed seed and a
  test that checks its key numbers. CI executes them.
- Module pages in `docs/modules/`, an API reference in `docs/API.md`, and proof notes in
  `docs/proofs/` that label each claim as proved, checked by test, measured, or claimed
  from the book only.
- CI on Python 3.10 to 3.13: lint, the test suite, and the suite against the installed
  wheel with and without optional extras.

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

[Unreleased]: https://github.com/landtml/finlab-ml/compare/v0.1.0...HEAD
[0.1.0]: https://github.com/landtml/finlab-ml/releases/tag/v0.1.0
