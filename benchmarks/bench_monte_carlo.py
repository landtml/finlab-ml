"""Benchmark finlab.monte_carlo: the bootstrap-uniqueness trial and its throughput.

Not run by CI. Usage (from the repository root):

    python benchmarks/bench_monte_carlo.py

Three measurements, all at the default experiment size (n_obs=10, n_bars=100,
max_h=5):

1. Per trial: the jitted ``_trial_uniqueness`` against a dense NumPy path. Both
   start from the same spans, draws and uniforms, so the label generator
   (``random_t1``) and the pandas wrapper are outside both timings. The dense
   path builds the full bar-by-label indicator matrix and computes the
   sequential draws and both average uniquenesses with matrix operations.
2. Compile time: the first call of ``_trial_uniqueness`` in a fresh Python
   process whose numba cache directory is empty. This is a cold start, one
   measurement per run.
3. Throughput of ``bootstrap_uniqueness_mc`` at ``N_ITER`` trials, one timed run
   each, with ``num_threads=1`` and with ``num_threads=min(4, os.cpu_count())``.
   A warm-up call runs first, so numba compilation or cache loading is not
   timed. Each parallel run includes creating its process pool.

Per-trial timings are the best of ``REPEATS`` runs over ``N_KERNEL_TRIALS``
trials, divided by the number of trials.
"""

from __future__ import annotations

import os
import subprocess
import sys
import tempfile
import time

import numpy as np

from _timing import best_of
from finlab.monte_carlo import _trial_uniqueness, bootstrap_uniqueness_mc, random_t1
from finlab.weights import sample_average_uniqueness

N_OBS, N_BARS, MAX_H = 10, 100, 5
N_KERNEL_TRIALS = 5_000
N_ITER = 20_000
REPEATS = 3
FIRST_CALL_TAG = "FIRST_CALL_S="


def make_trials(n: int, seed: int = 0) -> list[tuple[np.ndarray, ...]]:
    """Inputs for ``n`` trials, drawn in the order documented for bootstrap_uniqueness_trial."""
    rng = np.random.default_rng(seed)
    trials = []
    for _ in range(n):
        t1 = random_t1(N_OBS, N_BARS, MAX_H, rng)
        starts = t1.index.to_numpy(dtype=np.int64)
        ends = t1.to_numpy(dtype=np.int64)
        std_draws = rng.integers(0, N_OBS, size=N_OBS)
        uniforms = rng.random(N_OBS)
        trials.append((starts, ends, std_draws, uniforms))
    return trials


def dense_indicator(starts: np.ndarray, ends: np.ndarray) -> np.ndarray:
    """Full bar-by-label indicator matrix, shape (N_BARS, n_labels)."""
    bars = np.arange(N_BARS)
    return ((bars[:, None] >= starts) & (bars[:, None] <= ends)).astype(np.float64)


def dense_sequential_draws(m: np.ndarray, uniforms: np.ndarray) -> np.ndarray:
    """Sequential bootstrap on a dense matrix, with the kernel's inverse-CDF rule.

    Every label covers at least one bar here (ends >= starts), so ``colsum > 0``.
    """
    colsum = m.sum(axis=0)
    c = np.zeros(m.shape[0])
    draws = np.empty(len(uniforms), dtype=np.int64)
    for k, u in enumerate(uniforms):
        avg = (m.T @ (1.0 / (1.0 + c))) / colsum
        cum = np.cumsum(avg)
        pick = int(np.searchsorted(cum, u * cum[-1], side="right"))
        if pick == len(avg):
            pick = int(np.flatnonzero(avg > 0.0)[-1])
        draws[k] = pick
        c += m[:, pick]
    return draws


def naive_trial(starts, ends, std_draws, uniforms) -> tuple[float, float]:
    """Dense path for one trial: indicator matrix, dense draws, public dense uniqueness."""
    m = dense_indicator(starts, ends)
    seq_draws = dense_sequential_draws(m, uniforms)
    return sample_average_uniqueness(m, std_draws), sample_average_uniqueness(m, seq_draws)


def first_call_seconds() -> float:
    """Time the first call of the jitted trial. Runs in the child process only."""
    starts, ends, std_draws, uniforms = make_trials(1)[0]
    t0 = time.perf_counter()
    _trial_uniqueness(N_BARS, starts, ends, std_draws, uniforms)
    return time.perf_counter() - t0


def cold_first_call_seconds() -> float:
    """First-call time in a fresh process whose numba cache directory is empty."""
    with tempfile.TemporaryDirectory() as cache_dir:
        env = dict(os.environ, NUMBA_CACHE_DIR=cache_dir)
        proc = subprocess.run(
            [sys.executable, os.path.abspath(__file__), "--first-call"],
            env=env,
            capture_output=True,
            text=True,
            check=True,
        )
    for line in proc.stdout.splitlines():
        if line.startswith(FIRST_CALL_TAG):
            return float(line[len(FIRST_CALL_TAG):])
    raise RuntimeError(f"no timing in child output: {proc.stdout!r}")


def compare_per_trial(trials) -> None:
    worst_std = worst_seq = 0.0
    for starts, ends, std_draws, uniforms in trials[:200]:
        k_std, k_seq = _trial_uniqueness(N_BARS, starts, ends, std_draws, uniforms)
        n_std, n_seq = naive_trial(starts, ends, std_draws, uniforms)
        worst_std = max(worst_std, abs(k_std - n_std))
        worst_seq = max(worst_seq, abs(k_seq - n_seq))
    if max(worst_std, worst_seq) > 1e-9:
        raise SystemExit(f"kernel and dense path disagree: std_u {worst_std:.3g}, seq_u {worst_seq:.3g}")

    def run_kernel() -> None:
        for starts, ends, std_draws, uniforms in trials:
            _trial_uniqueness(N_BARS, starts, ends, std_draws, uniforms)

    def run_naive() -> None:
        for starts, ends, std_draws, uniforms in trials:
            naive_trial(starts, ends, std_draws, uniforms)

    t_jit = best_of(run_kernel, repeats=REPEATS) / len(trials)
    t_naive = best_of(run_naive, repeats=REPEATS) / len(trials)
    print(f"check             : max |kernel - dense| over 200 trials: "
          f"std_u {worst_std:.1e}, seq_u {worst_seq:.1e}")
    print(f"per trial         : naive (dense NumPy) {t_naive * 1e6:8.1f} us | "
          f"jitted kernel {t_jit * 1e6:7.2f} us | speedup {t_naive / t_jit:6.1f}x")


def timed_run(num_threads: int) -> float:
    t0 = time.perf_counter()
    out = bootstrap_uniqueness_mc(
        n_obs=N_OBS, n_bars=N_BARS, max_h=MAX_H, n_iter=N_ITER, seed=0, num_threads=num_threads
    )
    elapsed = time.perf_counter() - t0
    assert len(out) == N_ITER
    return elapsed


def main() -> None:
    if "--first-call" in sys.argv:
        print(f"{FIRST_CALL_TAG}{first_call_seconds():.6f}")
        return

    print(f"setup             : n_obs={N_OBS}, n_bars={N_BARS}, max_h={MAX_H}, "
          f"cpu_count={os.cpu_count()}")

    t_first = cold_first_call_seconds()
    print(f"compile           : first call of _trial_uniqueness, fresh process, empty numba cache "
          f"(includes compilation): {t_first * 1e3:.1f} ms")

    trials = make_trials(N_KERNEL_TRIALS)
    compare_per_trial(trials)

    bootstrap_uniqueness_mc(n_obs=N_OBS, n_bars=N_BARS, max_h=MAX_H, n_iter=200, num_threads=1)
    workers = min(4, os.cpu_count() or 1)
    t_serial = timed_run(1)
    t_par = timed_run(workers)
    print(f"throughput        : bootstrap_uniqueness_mc, n_iter={N_ITER}, one timed run each "
          f"(after a warm-up call)")
    print(f"  num_threads=1  : {t_serial:7.2f} s | {N_ITER / t_serial:9.0f} trials/s")
    verdict = "faster" if t_par < t_serial else "slower"
    print(f"  num_threads={workers}  : {t_par:7.2f} s | {N_ITER / t_par:9.0f} trials/s | "
          f"{verdict} than num_threads=1 ({t_serial / t_par:.2f}x)")


if __name__ == "__main__":
    main()
