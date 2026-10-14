# Benchmarks

Each benchmark times a hot path against a naive reference and prints the speedup.
They are not run in CI (timings are machine-dependent). Run one with:

    PYTHONPATH=src python3 benchmarks/bench_<name>.py

| Module | Benchmark | What is compared |
|---|---|---|
| `finlab.bars` | `bench_bars.py` | jitted bar boundaries vs a Python tick loop |
| `finlab.labeling` | `bench_labeling.py` | triple-barrier and trend-scanning kernels vs pandas loops |
| `finlab.weights` | `bench_weights.py` | concurrency sweep and sequential bootstrap vs dense loops |
| `finlab.fracdiff` | `bench_fracdiff.py` | fixed-window FFD kernel vs naive dot products |
| `finlab.entropy` | `bench_entropy.py` | Kontoyiannis and Lempel-Ziv kernels vs naive loops |
| `finlab.structural_breaks` | `bench_structural_breaks.py` | SADF, QADF, SMT, CUSUM and CSW kernels vs naive loops |
| `finlab.microstructure` | `bench_microstructure.py` | tick rule, Corwin-Schultz, VPIN, Hasbrouck lambda vs naive loops |
| `finlab.hrp` | `bench_hrp.py` | HRP weights vs a naive bisection (SciPy linkage is not jitted) |
| `finlab.importance` | `bench_importance.py` | MDI column statistics vs a loop |
| `finlab.pbo` | `bench_pbo.py` | CSCV rank and Sharpe kernel vs a naive loop over splits |
| `finlab.trading_rules` | `bench_trading_rules.py` | numba OTR grid vs a pure-Python path loop on the same normals |
| `finlab.stats` | `bench_stats.py` | holding-period kernel vs a Python loop (100,000 rows) |
| `finlab.tuning` | `bench_tuning.py` | wall time of `grid_search` over a NumPy toy model (times only, no reference) |
| `finlab.ensemble` | `bench_ensemble.py` | sequential-bootstrap draws vs a naive loop (different RNG path, same probabilities); fit time with and without `oob_score` |
| `finlab.onc` | `bench_onc.py` | silhouette kernel vs a per-variable loop |
| `finlab.cv` | `bench_cv.py` | CPCV split generation vs a per-observation purge loop |
| `finlab.bet_sizing` | `bench_bet_sizing.py` | overlapping-signal averaging (difference-array sweep) vs a per-timestamp loop |
| `finlab.parallel` | `bench_parallel.py` | `mp_pandas_obj` serial vs process pool (depends on core count) |
| `finlab.agent` | `bench_agent.py` | one tool call via JSON-RPC vs the direct library call (overhead) |
| `finlab.monte_carlo` | `bench_monte_carlo.py` | bootstrap-uniqueness trial kernel vs a dense NumPy path (same draws); first-call compile time; trials/s of `bootstrap_uniqueness_mc` at 1 and N threads |

## Modules without a benchmark, and why

- **`finlab.strategy_risk`**: closed-form formulas. Each call is a handful of
  floating-point operations; there is no loop to accelerate.
- **`finlab.tuning`** has a timing script but no speedup figure: its time is spent in the
  user's `fit`/`predict`, so a speedup would measure the user's model rather than this code.
- **`finlab.bet_sizing.sigmoid_bet_size`, `target_position`, `limit_price`**: scalar
  or vectorised NumPy expressions, with no loop over observations (`limit_price`
  loops over at most `max_position` traded sizes).

## Reproductions

- `repro_uniqueness.py` reruns the sequential versus standard bootstrap uniqueness
  comparison quoted in `docs/proofs/weights.md` and prints both means. It is a
  reproduction, not a benchmark.
