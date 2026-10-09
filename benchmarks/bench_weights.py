"""Benchmark finlab.weights against brute-force references.

Not run by CI. Usage:

    PYTHONPATH=src python3 (from the repository root) benchmarks/bench_weights.py

Each jitted function is called once before timing so numba compilation is not
measured. Each timing is the best of ``REPEATS`` runs.
"""

from __future__ import annotations

import time

import numpy as np
import pandas as pd

from finlab.weights import indicator_matrix, num_co_events, sequential_bootstrap

REPEATS = 3


def best_time(fn, repeats: int = REPEATS) -> float:
    best = float("inf")
    for _ in range(repeats):
        t0 = time.perf_counter()
        fn()
        best = min(best, time.perf_counter() - t0)
    return best


def naive_concurrency(n_bars, starts, ends):
    """Snippet 4.1 style: one numpy slice increment per label."""
    c = np.zeros(n_bars, dtype=np.int64)
    for s, e in zip(starts, ends):
        c[s:e + 1] += 1
    return c


def naive_sequential(m: np.ndarray, uniforms: np.ndarray) -> np.ndarray:
    """Dense Snippet 4.5 loop: recompute every candidate's average uniqueness per draw."""
    n_bars, n_ev = m.shape
    colsum = (m != 0).sum(axis=0)
    c = np.zeros(n_bars)
    draws = np.empty(len(uniforms), dtype=np.int64)
    for k, u in enumerate(uniforms):
        avg = np.zeros(n_ev)
        for j in range(n_ev):
            if colsum[j] > 0:
                s = 0.0
                for t in range(n_bars):
                    if m[t, j] != 0:
                        s += 1.0 / (1.0 + c[t])
                avg[j] = s / colsum[j]
        total = avg.sum()
        target = u * total
        acc, pick, last = 0.0, -1, -1
        for j in range(n_ev):
            if avg[j] > 0.0:
                last = j
                acc += avg[j]
                if acc > target:
                    pick = j
                    break
        if pick < 0:
            pick = last
        draws[k] = pick
        c[m[:, pick] != 0] += 1.0
    return draws


def main() -> None:
    rng = np.random.default_rng(0)

    # Concurrency: 500k bars, 50k labels, lifespans up to 200 bars.
    n_bars = 500_000
    index = pd.RangeIndex(n_bars)
    starts = np.sort(rng.choice(n_bars - 201, size=50_000, replace=False))
    ends = np.minimum(starts + rng.integers(0, 200, size=50_000), n_bars - 1)
    t1 = pd.Series(ends, index=starts)
    num_co_events(index, t1.iloc[:10])
    t_jit = best_time(lambda: num_co_events(index, t1))
    t_naive = best_time(lambda: naive_concurrency(n_bars, starts, ends))
    print(f"num_co_events     : naive (slice per label) {t_naive * 1e3:9.1f} ms | "
          f"jitted sweep {t_jit * 1e3:8.2f} ms | speedup {t_naive / t_jit:7.1f}x")

    # Sequential bootstrap: 400 bars, 150 labels, lifespans up to 40 bars.
    n_bars, n_lab = 400, 150
    s2 = np.sort(rng.choice(n_bars - 41, size=n_lab, replace=False))
    e2 = np.minimum(s2 + rng.integers(0, 40, size=n_lab), n_bars - 1)
    t1s = pd.Series(pd.RangeIndex(n_bars)[e2], index=pd.RangeIndex(n_bars)[s2])
    m_df = indicator_matrix(pd.RangeIndex(n_bars), t1s)
    m = m_df.to_numpy()
    sequential_bootstrap(m_df, seed=0)
    t_jit = best_time(lambda: sequential_bootstrap(m_df, seed=1))
    uniforms = np.random.default_rng(1).random(n_lab)
    t_naive = best_time(lambda: naive_sequential(m, uniforms), repeats=1)
    print(f"sequential_boot.  : naive (dense loops) {t_naive * 1e3:9.1f} ms | "
          f"jitted kernel {t_jit * 1e3:8.2f} ms | speedup {t_naive / t_jit:7.1f}x")


if __name__ == "__main__":
    main()
