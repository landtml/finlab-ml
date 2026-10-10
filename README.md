<h1 align="center">
  <img src="assets/icon.svg" width="40" alt="" align="absmiddle"> finlab
</h1>

<p align="center">
  <img src="assets/owl.svg" width="460" alt="finlab mascot: an owl that blinks and glances at candlestick charts">
</p>

<p align="center">
  <a href="https://github.com/landtml/finlab-ml/actions/workflows/ci.yml"><img alt="CI" src="https://github.com/landtml/finlab-ml/actions/workflows/ci.yml/badge.svg"></a>
  <a href="https://github.com/landtml/finlab-ml/tags"><img alt="Latest tag" src="https://img.shields.io/github/v/tag/landtml/finlab-ml?label=version&color=2f5d8c"></a>
  <img alt="Python 3.10 to 3.13" src="https://img.shields.io/badge/python-3.10%20to%203.13-3776ab">
  <a href="LICENSE"><img alt="MIT license" src="https://img.shields.io/badge/license-MIT-2a9d8f"></a>
</p>

A toolkit for financial machine learning that implements the methods of
Marcos López de Prado's *Advances in Financial Machine Learning* (AFML),
organised by chapter. Hot loops are compiled to machine code with
[numba](https://numba.pydata.org/); there is no scikit-learn dependency.

Most methods come with tests that cross-check them against a naive reference, and
where the book states a result, a proof note under `docs/proofs/` that marks each
claim as proved here, checked by test, measured, or claimed from the book only.

## Install

```bash
pip install -e .            # core library
pip install -e ".[dev]"     # + pytest and ruff, to run the test suite and lint
pip install -e ".[plot]"    # + plotly, for the figures in finlab.plot
pip install -e ".[notebooks]"  # + Jupyter tooling, to run the notebooks in examples/
pytest                      # run the suite
```

To install a tagged release instead:

```bash
pip install "git+https://github.com/landtml/finlab-ml@v0.1.0"
```

Requires Python 3.10 to 3.13.

## Quick start

These examples run as written (checked against the current code).

```python
import numpy as np
import pandas as pd
from finlab.cv import CombinatorialPurgedCV, make_t1
from finlab.stats import sharpe_ratio, deflated_sharpe_ratio
from finlab.bet_sizing import prob_bet_size, discretize_signal
from finlab.hrp import hrp_weights

rng = np.random.default_rng(0)
idx = pd.date_range("2020-01-01", periods=500, freq="B")

# 1. Leakage-free cross-validation for 10-bar forward labels (AFML ch. 7)
t1 = make_t1(idx, horizon=10)
cv = CombinatorialPurgedCV(n_groups=6, n_test_groups=2, embargo_pct=0.01, t1=t1)
X = pd.DataFrame(rng.normal(size=(500, 3)), index=idx)
print(sum(1 for _ in cv.split(X)), "purged splits")        # 15 purged splits

# 2. Backtest statistics: Sharpe and deflated Sharpe over 50 trials (AFML ch. 14)
r = pd.Series(rng.normal(0.0005, 0.01, 500), index=idx)
sr = sharpe_ratio(r, annualize=False)
print(round(deflated_sharpe_ratio(sr, n_trials=50, var_sr=0.01, n_obs=500,
                                  skew=0, kurt=3), 4))

# 3. Bet sizes from predicted probabilities, discretised (AFML ch. 10)
print(prob_bet_size([0.7, 0.2]).round(3), discretize_signal([0.33], 0.2))

# 4. Hierarchical risk parity weights (AFML ch. 16)
cov = pd.DataFrame(np.cov(rng.normal(size=(200, 5)).T),
                   columns=list("abcde"), index=list("abcde"))
print(hrp_weights(cov).round(3).to_dict())
```

## For AI agents

`finlab-mcp` is a local MCP server that exposes the main computations as typed
tools with compact JSON output. See [`docs/AGENTS.md`](docs/AGENTS.md).

## Modules

<details>
<summary><strong>Module map</strong>: chapters 2 to 20 (click to expand)</summary>

| Module | Chapter | What it provides |
|---|---|---|
| `finlab.cv` | 7 | `CombinatorialPurgedCV`, `PurgedKFold`, `make_t1` (purge and embargo) |
| `finlab.bars` | 2 | time, tick, volume, dollar, imbalance and run bars; CUSUM filter |
| `finlab.labeling` | 3 | daily volatility, triple-barrier events, bin labels, meta-labels, rare-label dropping, trend-scanning labels (not in AFML) |
| `finlab.weights` | 4 | concurrency, average uniqueness, sequential bootstrap, return attribution, time decay |
| `finlab.monte_carlo` | 4 | seeded Monte Carlo trial runner (`run_trials`), random label sets, standard vs sequential bootstrap uniqueness experiment |
| `finlab.fracdiff` | 5 | fixed-width and expanding-window fractional differentiation, minimum-d search |
| `finlab.ensemble` | 6 | `SequentialBootstrapBagging` |
| `finlab.importance` | 8 | MDI, MDA, SFI, orthogonal features |
| `finlab.tuning` | 9 | grid and randomized search scored by purged CV |
| `finlab.bet_sizing` | 10 | probability and sigmoid bet sizing, target positions, limit prices, discretisation |
| `finlab.pbo` | 11 | probability of backtest overfitting by CSCV |
| `finlab.trials` | 11, 14 | trial registry (`TrialRegistry`) that feeds PBO and deflated Sharpe the number of trials and the performance matrix |
| `finlab.stats` | 14 | Sharpe, probabilistic and deflated Sharpe, minimum track record length |
| `finlab.plot` | optional | plotly figures for PBO, deflated Sharpe and the Monte Carlo experiment (needs the `plot` extra) |
| `finlab.hrp` | 16 | hierarchical risk parity |
| `finlab.structural_breaks` | 17 | CUSUM tests, Chu-Stinchcombe-White, SADF |
| `finlab.entropy` | 18 | plug-in, Lempel-Ziv and Kontoyiannis entropy; encoders |
| `finlab.microstructure` | 19 | tick rule, Roll, Corwin-Schultz, Becker-Parkinson, Kyle lambda, Amihud, VPIN |
| `finlab.parallel` | 20 | molecule partitioning and `mp_pandas_obj` on `concurrent.futures` |

Each module has its own page with a runnable example in
[`docs/modules/`](docs/modules/) (the examples are executed by
`tests/test_doc_examples.py`). The full signature list is in
[`docs/API.md`](docs/API.md), generated from the code by `tools/gen_api.py`.

</details>

## Validation status

<details>
<summary><strong>Validation status</strong>: what is proved, tested or only claimed (click to expand)</summary>

| Area | Status |
|---|---|
| CPCV purge and no-leakage property | Proved in [`docs/proofs/cpcv.md`](docs/proofs/cpcv.md); tested in `tests/test_cpcv.py`. The leakage checker `tools/verify_leakage.py` is ported from `purgedcv`. The live-data certificate in that file is the `purgedcv` run, not a `finlab` run. |
| Purged K-fold, tuning, bet sizing, parallel helpers | Tested. Book worked examples for bet sizing (sigmoid calibration, target 97, limit price 112.3657) are tests. |
| PBO via CSCV | Tested against a naive transcription. Under i.i.d. noise E[PBO] = 1/2 for even N (proved in `docs/proofs/pbo.md`), so a PBO near 1 is not a noise result. |
| Deflated Sharpe, PSR, minTRL | Tested on hand-checked values. minTRL is cited to Bailey and López de Prado (2012), not AFML. |
| VPIN and bulk-volume classification | Tested against a trade-level reference. The default scale is causal (expanding window), so no future prices enter the classification. The bulk-volume classifier is from Easley et al. (2012), not AFML. |
| Kyle lambda, Corwin-Schultz, Roll | Tested on synthetic data; the Corwin-Schultz and Kyle results are claimed from the book only. |
| SADF and CUSUM | Tested; the Chu-Stinchcombe-White critical value is read from an ambiguous radical in the book (see `docs/proofs/structural_breaks.md`). |
| Entropy | Tested on known cases. The Lempel-Ziv normalisation is the standard LZ78 one, not from AFML. |
| Trend-scanning labels | Implemented from the literal definition. Not in AFML ch. 3; the t-threshold of 1.96 is a design choice and the citation is not verified against a primary source. |
| Time decay | Proved and tested: the oldest observation's weight is `c + (1-c)·w1/T`, not exactly `c`. |
| Monte Carlo trials and bootstrap uniqueness (`finlab.monte_carlo`) | Tested against a dense reference (rtol 1e-12), with worker-count invariance and determinism checked by test; proofs and measurements in [`docs/proofs/monte_carlo.md`](docs/proofs/monte_carlo.md). Follows the printed listings of Snippets 4.7-4.8, checked by reading the book, not by a test against its output. Random numbers come from a numpy Generator, so runs are not the book's. The statistical gap is measured, not proved. |


</details>

## Benchmarks

<details>
<summary><strong>Benchmark table</strong> (click to expand)</summary>

Measured on 2026-10-10 on a shared machine (4 vCPU, Python 3.13.16, numpy 2.5.3, numba 0.68.0), one run per script with other jobs sharing the machine, so numbers vary between runs. Each row names its script in `benchmarks/`.

| Hot path | Naive reference | Jitted | Speedup |
|---|---|---|---|
| `get_events` (triple barrier; naive extrapolated from 1k events; `bench_labeling.py`) | 1915.1 ms | 6.92 ms | 276.6x |
| `trend_scanning_labels` (`bench_labeling.py`) | 6715.7 ms | 0.68 ms | 9870.9x |
| `sequential_bootstrap` (`bench_weights.py`) | 1248.1 ms | 1.05 ms | 1185.7x (the naive version is a dense loop, so this overstates the gain) |
| `num_co_events` (`bench_weights.py`) | 62.0 ms | 5.02 ms | 12.3x |
| SADF, n=300 (`bench_structural_breaks.py`) | 5.5516 s | 0.111449 s | 50x |
| Kontoyiannis entropy, n=300 (`bench_entropy.py`) | 0.0711 s | 0.000151 s | 471x |
| CUSUM test, n=2000 (`bench_structural_breaks.py`) | 0.0459 s | 0.000090 s | 511x |
| VPIN, n=20k (`bench_microstructure.py`) | 17517.37 ms (trade-expanded reference) | 1.045 ms | 16771.0x |
| `tick_rule`, n=200k (`bench_microstructure.py`) | 28.95 ms | 1.380 ms | 21.0x |
| PBO, T=1200, N=40, S=12 (`bench_pbo.py`) | 1.0137 s | 0.0633 s | 16.0x |
| HRP weights, n=200 (`bench_hrp.py`) | 5.26 ms | 0.806 ms | 6.5x (SciPy linkage is not jitted and is inside the timed call; in the same run the speedup is 8.1x at n=50 and 3.9x at n=500) |


</details>

## Scope and limitations

<details>
<summary><strong>Scope and limitations</strong> (click to expand)</summary>

- **Implementation.** Numba (LLVM machine code) is used for the hot loops. Rust and
  JAX are not used.
- **Parallel engine.** Snippet 4.9 runs through the book's `mpEngine` (Chapter 20).
  `finlab.monte_carlo` uses `mp_pandas_obj` in `finlab.parallel` instead.
- **Not implemented:** the Chow DFC/SDFC and
  quantile ADF tests, the distance-of-distances clustering variant of AFML section 16.4.1,
  and out-of-bag estimation for bagging. ONC clustering is in `finlab.onc`, without the
  published repair step.
- **No purging inside CSCV or importance.** Passing a purged splitter is the
  caller's responsibility for MDA and SFI; CSCV follows the book without purging.
- **Leakage beyond the splitter.** The CV guarantee covers the split boundary given
  the `t1` you supply. Leakage in your own feature construction is not detected.
- **Bar warm-up look-ahead.** Imbalance and runs bars start their expected-imbalance
  estimates from the first `init_T` ticks, so the first bars close on a threshold that
  uses later data. After that the estimates are EWMA updates from prior bars and ticks.
  If the expected imbalance is 0 the threshold is 0 and every tick closes a bar, so
  zero-drift data gives very short bars. Runs bars use the book's
  `max{E[buy], E[sell]}` threshold, which is not `E[max]` (Jensen); the proof note
  shows this.
- **Python versions.** CI runs 3.10 to 3.13. Locally the suite was run on 3.13.
- **Sample size.** Average uniqueness and sequential-bootstrap gains are modest in
  practice (one measured setting gave 0.1963 vs 0.1913 over 200 seeds; that run is
  recorded in `docs/proofs/weights.md`, and no script in the repo reproduces it). The
  Monte Carlo experiment in `finlab.monte_carlo` follows Snippets 4.7-4.8 at the Snippet
  4.9 sizes. Its paired gap is about 0.084 (`bootstrap_uniqueness_mc(n_iter=2000, seed=0)`,
  see `docs/proofs/monte_carlo.md` section 4). Its medians of 0.6 and 0.7 at 20,000
  trials match the medians the book states after Figure 4.2 (p. 68, as quoted in that
  proof note; the page reference is not verified) to one decimal. The
  0.1963 vs 0.1913 setting uses a different design, so the two numbers are not
  comparable. All of these are reported as measured, not as a book reproduction.

</details>

## Contributing

Each module has a test file in `tests/` and a page in `docs/modules/`. Most also have a
benchmark in `benchmarks/` and a proof note in `docs/proofs/` (no benchmark for
`finlab.stats` and `finlab.tuning`; no proof note for `finlab.agent`). Keep claims in the
proof notes to four kinds: proved here, checked by test, measured, or claimed from the
book only.

## Credits

The methods implemented here come from Marcos López de Prado, *Advances in
Financial Machine Learning* (Wiley, 2018; ISBN 978-1-119-48208-6). The book's
text is not included in this repository; buy or borrow a copy to read the
derivations. The proof notes cite chapters and snippets by number.

Other sources cited in the code and docs include: López de Prado, Lewis and Boudt
(2019) for ONC clustering, and Bailey and López de Prado (2012) for the minimum
track record length. The purged cross-validation code and its proof come from
the [`purgedcv`](https://github.com/landtml/purgedcv) project.

## License

MIT; see [`LICENSE`](LICENSE). Parts of the CPCV implementation and its proof are ported from
[`purgedcv`](https://github.com/landtml/purgedcv), also MIT; see
`NOTICE_purgedcv_LICENSE.txt`.
