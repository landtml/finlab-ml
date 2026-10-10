# Changelog

All notable changes to this project are documented in this file. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

## [Unreleased]

### Added

- `finlab.trials`: `TrialRecord` (frozen, validated name, JSON-scalar parameters and finite
  returns) and `TrialRegistry`, which counts the trials and feeds the performance matrix to
  PBO and the number of trials N to the deflated Sharpe ratio. Both round-trip through dict
  and JSON with exact float equality.
- `finlab.stats.deflated_sharpe` and the frozen `DeflatedSharpeResult` (dsr, sr_hat, sr_star,
  n_trials, n_obs, skew, kurt), with validated inputs.
- `_repr_html_` on `PBOResult` and `DeflatedSharpeResult`, built by `finlab._display`
  (escaped values, inline styles, no script). Importing finlab prints nothing.
- `finlab.plot`: `pbo_distribution`, `deflated_sharpe_comparison` and
  `monte_carlo_histogram`, which return plotly figures from stored results. plotly is
  imported on first use; without it, the call raises an ImportError with the install command.
- Worked notebooks, each a Jupytext percent source with a paired `.ipynb` that has no
  committed outputs and a fixed seed: `examples/pbo`, `examples/deflated_sharpe` (with
  trial records), `examples/monte_carlo` (chapter 4) and `examples/fracdiff` (chapter 5).
  Each runs in a few seconds and has a test that asserts its key numbers.
- `.github/workflows/notebooks.yml`: requires at least four example pairs, checks that each
  `.py` regenerates its `.ipynb` and that committed notebooks have no outputs, executes each
  notebook under nbconvert with a time limit, uploads the executed copies, and runs the plot
  tests with and without plotly.
- A Gallery section in the README with three figures from the example notebooks, saved as
  PNG files in `docs/images/`. `tools/export_figures.py` regenerates them from the notebooks'
  fixed seeds.
- A deflated Sharpe comparison plot (40 and 80 trials) in `examples/deflated_sharpe`.
- Tests for the PBO hand-computed and closed-form cases (`tests/test_pbo.py`), the deflated
  Sharpe numerical reference and hand values (`tests/test_stats.py`), the trial records
  (`tests/test_trials.py`), the plotting functions and their no-plotly path
  (`tests/test_plot.py`), the display helper (`tests/test_display.py`) and the notebooks
  (`tests/test_example_*.py`).
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
- `deflated_sharpe_ratio` validates its inputs (rejects booleans, non-integral trial counts
  and non-finite values, which previously returned NaN) and returns the same float.
- `probability_of_backtest_overfitting` checks `n_partitions` as an integer and raises a
  ValueError for other values.
- The README benchmark table was measured again. Each row names its script
  in `benchmarks/`; the old speedups are replaced by the measured ones, which differ by up to
  an order of magnitude for some rows.
- The README module map lists `finlab.trials` and `finlab.plot`, and the install block names
  the `plot` and `notebooks` extras.
- `docs/API.md` is regenerated; the generator lists the trial and plotting modules and labels
  PBO with chapter 11 only.
- The README module map, limitations and contributing text match the code. Each benchmark row
  names its script. The "naive reference" claim covers most methods, not all. Trend-scanning
  labels are marked as not in AFML. ONC clustering is listed as implemented without the
  repair step. Fixed-width and expanding-window fractional differentiation are described
  separately.
- PBO is cited to AFML chapter 11 only, in the README, the module page, `docs/AGENTS.md`
  and the module docstring.
- Numeric proof-note claims now have tests or are labelled as measurements or book claims.
  Corrections: minTRL hand value 13.174943 to 13.1749455; Gaussian entropy in bits 2.048 to
  2.047; SADF explosive value 14.6 to 14.55; Becker-Parkinson mean 0.0206 to 0.01893 (about
  5% low at the test seed, within 10%); HRP square-matrix range now N = 2 to 24.
- The Roll test uses c = 0.5 and increment sd 0.05, which hold across seeds. The
  Becker-Parkinson test runs on consecutive bars. The two fixtures changed; no tolerance
  was loosened.
- Every "checked by test" label in `docs/proofs/` names the test function that asserts the
  claim (129 labels). Two labels are qualified where the test covers only part of a claim.
- The README module map lists `finlab.onc` and `finlab.agent`. The docs-page statement excludes
  `finlab.trials` and `finlab.plot`, which are described in the API reference.
- `finlab.trials` is documented as not an AFML construct: it supplies the inputs that
  chapters 11 and 14 need.
- Proof notes for PBO and deflated Sharpe: the minTRL tolerance statement now gives the
  tolerances the test uses; the default `prob = 0.95` has its own test
  (`test_stats_default_prob_is_0_95`); the pure-noise rank probability is checked by enumeration
  for N = 2 to 9 (`test_pbo_pure_noise_rank_probability_by_enumeration`); and the Gaussian
  Sharpe-ratio variance is checked by simulation at SR = 0.1 and T = 2000
  (`test_stats_sharpe_variance_formula_by_simulation`, +1.09% against the formula).
- `docs/API.md` lists `finlab.onc`, which the generator had omitted; the README's link to the
  full signature list now covers every public module.
- Monte Carlo, weights and microstructure tests check exact values where a value was only
  checked for shape or sign (Kyle t, Corwin-Schultz, BVC, time decay).

### Fixed

- PBO zero-variance columns: a constant column gave a Sharpe near 7e15 from rounding; zero
  variance is now detected by exact equality, so the result is 0 for a zero value and plus or
  minus infinity otherwise, as the module documents.
- PBO `n_partitions`: a non-integral value such as 4.5 failed later with an unclear error;
  it is rejected up front.
- Runs-bar Jensen remark in `docs/proofs/bars.md`: the equality condition was wrong. Corrected,
  with an equality example and a strict example, each checked by exact expectation and
  simulation.
- HRP proof notes: the correlation distance is zero exactly when the correlation is 1
  (a metric only on standardized series); the bisection cost bound is N^2 ceil(log2 N), not
  N^3/3, because the recursive bisection splits at the midpoint.
- Docstrings that disagreed with the code or the tests: the Becker-Parkinson accuracy
  (`finlab.microstructure`), the Lempel-Ziv normalisation (`finlab.entropy`), and the
  distance metric statement (`finlab.hrp`).
- Removed an unused local in the agent CPCV tool and an unused `pjit` import in `weights.py`
  that ruff flagged.
- Test cleanup (eleven modules plus three Monte Carlo, weights and microstructure files):
  tautological and vacuous checks replaced; bare `pytest.raises` calls now match the message;
  zip-truncated comparisons now check lengths; the ensemble vote, the parallel results,
  the CPCV embargo anchor and the tuning tie order are compared with exact references; the
  entropy check uses the exact bin count. A test that could not run under its name was
  renamed to what it checks.
- The structural-breaks documentation example is marked slow.
- The CI install job and the lint step now include `examples/`. Without it, the example tests
  could not find their notebooks when the suite ran against the installed wheel.

### Removed

- The unused `CPCVPaths` import in `tests/test_cpcv.py` and the unused loop variable in
  `tests/test_structural_breaks.py`.
- The README entry that listed ONC clustering as not implemented.
- The README attribution of the distance-of-distances clustering to Bailey and López de Prado
  (the code names no author).
- The README wording "tests carried over from purgedcv" (replaced by the local test file).
- The statement that 0.95 is "the book's suggested threshold" for the deflated Sharpe ratio.
  It is a common convention, labelled as such.
- The superseded benchmark speedups (for example trend scanning at about 991x) and the
  statement that SciPy linkage dominates at large N for HRP.

### Notes

- **Sample average uniqueness has two implementations, on purpose.** The dense
  reference `finlab.weights.sample_average_uniqueness` and the numba kernel
  `_sample_uniqueness` used by the Monte Carlo path are kept. `tests/test_uniqueness_equivalence.py`
  requires them to agree at rtol 1e-12 over 200 seeds. Largest observed relative difference:
  5.4e-16.
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
  source. The suite passes on a fresh and on a warm cache with the same counts.
- **Monte Carlo "higher" uses a tie tolerance.** 37 of the 2,000 trials differ between the
  standard and sequential draws only in the last bit. The notebook counts a trial as higher
  only if the difference exceeds 1e-12. The fraction is 0.7015 with that rule and 0.712
  without it.
- **PBO edge size was chosen after one sweep.** The notebook's edge of 0.003 was picked after
  a sweep on seed 1. Over seeds 0 to 19 the mean PBO at that edge is 0.021; the notebook says
  it is a single seed.
- **Deflated Sharpe: adding trials lowers the value for the pinned seed only.** In the
  notebook, the deflated Sharpe ratio falls when 40 more pure-noise trials are added, for the
  pinned seed. Over seeds 1000 to 1199 it falls in 133 of 200 seeds. The notebook says it can
  also rise.
- **Expected-maximum approximation.** The Bailey and López de Prado approximation of the
  expected maximum of N normal Sharpe ratios differs from the numerical value by -7.9% at N = 2
  and by +0.4% at N = 1000. The tests assert these measured bounds.
- **Trial records use identity equality.** `TrialRecord` and `TrialRegistry` do not define
  value equality, so two equal records compare unequal. Round-trip tests compare fields.
- **Large trial counts.** Integral floats as large as 1e300 are accepted as `n_trials`, and
  the deflated Sharpe ratio is then 0. This is correct but unusual; no upper bound is set.
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
- **Skipped tests.** `tests/test_cpcv.py` skips its integration test when `yfinance` is not
  installed; it is marked `integration` and no extra installs it. Without the `plot` extra,
  the plotting tests skip; without the `notebooks` extra, the notebook-format test skips.
- **Plotting imports lazily.** `import finlab.plot` succeeds without plotly. Calling any
  plotting function without plotly raises an ImportError with the install command. This is
  the intended lazy-import design; a bare import is not an error.
- **Tolerances in the new tests.** The notebook tests compare key numbers with `1e-9` against
  the values measured on the committed seeds (`tests/test_example_*.py`). The plotting and
  notebook tests use the same tolerances as the rest of the suite. The minTRL constant in
  `tests/test_stats.py` was tightened from `1e-5` to `1e-9` after its exact value was checked.
- **Timings.** The benchmark table is one run per script on a 4-vCPU machine shared with other
  jobs; the numbers vary between runs.

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
