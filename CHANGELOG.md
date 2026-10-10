# Changelog

All notable changes to this project are documented in this file. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

## [Unreleased]

### Added

- Randomized equivalence test for the two uniqueness implementations
  (`tests/test_uniqueness_equivalence.py`: 200 fixed seeds, duplicate starts, identical
  spans, length-one spans, spans ending at the last bar, repeated draws).
- `tests/test_doc_claims.py`: deterministic tests for the numeric claims in the proof notes.
- Optional extras `plot` (plotly) and `notebooks` (jupyter, nbconvert, jupytext, ipykernel)
  in `pyproject.toml`.
- A ruff lint step (rules E9 and F) in `pyproject.toml` and in CI.
- A CI job that builds the wheel, installs it into a fresh venv outside the checkout and runs
  the suite there.

### Changed

- CI runs on every branch and on pull requests. The test job lints before it runs the suite.
- The `Changelog` project URL points at `CHANGELOG.md` on `main`.
- The README module map, benchmark table, limitations and contributing text match the code.
  Each benchmark row names its script. The "naive reference" claim covers most methods, not
  all. Trend-scanning labels are marked as not in AFML. ONC clustering is listed as
  implemented without the repair step. Fixed-width and expanding-window fractional
  differentiation are described separately.
- PBO is cited to AFML chapter 11 only, in the README, the module page, `docs/AGENTS.md`
  and the module docstring.
- Numeric proof-note claims now have tests or are labelled as measurements or book claims.
  Corrections: minTRL hand value 13.174943 to 13.1749455; Gaussian entropy in bits 2.048 to
  2.047; SADF explosive value 14.6 to 14.55; Becker-Parkinson mean 0.0206 to 0.01893 (about
  5% low at the test seed, within 10%); HRP square-matrix range now N = 2 to 24.
- The Roll test uses c = 0.5 and increment sd 0.05, which hold across seeds. The
  Becker-Parkinson test runs on consecutive bars. The two fixtures changed; no tolerance
  was loosened.
- Monte Carlo, weights and microstructure tests check exact values where a value was only
  checked for shape or sign (Kyle t, Corwin-Schultz, BVC, time decay).

### Fixed

- Removed an unused local in the agent CPCV tool and an unused `pjit` import in `weights.py`
  that ruff flagged.
- Test cleanup (eleven modules plus three Monte Carlo, weights and microstructure files):
  tautological and vacuous checks replaced; bare `pytest.raises` calls now match the message;
  zip-truncated comparisons now check lengths; the ensemble vote, the parallel results,
  the CPCV embargo anchor and the tuning tie order are compared with exact references; the
  entropy check uses the exact bin count. A test that could not run under its name was
  renamed to what it checks.
- The structural-breaks documentation example is marked slow.
- Removed the unused `CPCVPaths` import in `tests/test_cpcv.py` and the unused loop variable
  in `tests/test_structural_breaks.py`.

### Notes

- **Sample average uniqueness has two implementations, on purpose.** The dense reference
  `finlab.weights.sample_average_uniqueness` and the numba kernel `_sample_uniqueness` used by
  the Monte Carlo path are kept. `tests/test_uniqueness_equivalence.py` requires them to agree
  at rtol 1e-12 over 200 seeds. Largest observed relative difference: 5.4e-16.
- **Numba stale cache (root cause of the cross-module failure, partly reproduced).** numba's
  cache key for a function covers the bytecode of that function and a stamp of the file that
  defines it. A callee in another module is linked into a cached caller, so an edit to the
  callee does not invalidate the caller's cache entry. Reproduced in a repo copy: after an injected bug in `weights._span_sums`, a warm cache reported 6 failing
  tests, while a fresh cache reported 31 (25 of the missed failures were in
  `test_monte_carlo.py`). The NRT error `'descr' is NULL` reproduced only in a two-module toy,
  where two cached entries share a process-global environment name. It did not reproduce in
  the repo, and the logs of the original 16 to 20 failures are not available. Consequence:
  `tests/conftest.py` gives every run a fresh cache directory. Reusing
  `FINLAB_TEST_NUMBA_CACHE` after a source edit is unsafe; it is only valid for unchanged
  source.
- **Known limitation, not changed.** The njit kernels `_sample_uniqueness` (monte_carlo) and
  `_concurrency_sweep` (weights) do not check that span ends are below the bar count. The only
  caller keeps them in range.
- **Known inconsistency, not changed.** With zero draws, the kernel raises `ZeroDivisionError`
  and the dense reference raises `ValueError`.
- **Known warning, not changed.** On Python 3.13 the default fork start method emits
  `DeprecationWarning` from the multiprocessing pool used by `finlab.parallel`. Warnings
  appear in `tests/test_parallel.py` and in the Monte Carlo tests.
- **Build warnings, not changed.** `setuptools` warns about the `license` table and the
  `License` classifier in `pyproject.toml`. The package still builds and passes `twine check`.
- **One skipped test.** `tests/test_cpcv.py` skips its integration test when `yfinance` is
  not installed. It is marked `integration` and no extra installs it.

## [0.1.0]

### Added

- Monte Carlo experiment for AFML chapter 4 in `finlab.monte_carlo`: a seeded trial runner
  (`run_trials`), random label sets (`random_t1`) and the standard and sequential bootstrap
  uniqueness experiment (`bootstrap_uniqueness_mc`), following Snippets 4.7 and 4.8.
- `sample_average_uniqueness` in `finlab.weights` (Snippet 4.4), the dense reference
  implementation.
- Monte Carlo benchmark (`benchmarks/bench_monte_carlo.py`) and proof note
  (`docs/proofs/monte_carlo.md`).
- Project icon (`assets/icon.svg`) and an animated owl mascot (`assets/owl.svg`) in the README.
- Search keywords, classifiers and project URLs in `pyproject.toml`.

### Changed

- The README was restructured with a module map, install and quick-start sections.
