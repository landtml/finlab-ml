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

from finlab import weights
from finlab.weights import indicator_matrix, num_co_events, sequential_bootstrap

REPEATS = 3

# (label, bars, labels, lifespans drawn below max_span bars)
SEQ_SIZES = (
    ("small", 400, 150, 40),
    ("medium", 1500, 500, 80),
    ("large", 4000, 1500, 120),
)


def best_time(fn, repeats: int = REPEATS) -> float:
    best = float("inf")
    for _ in range(repeats):
        t0 = time.perf_counter()
        fn()
        best = min(best, time.perf_counter() - t0)
    return best


def sequential_case(n_bars: int, n_lab: int, max_span: int, seed: int) -> pd.DataFrame:
    """Indicator matrix of random labels with spans of up to max_span bars."""
    rng = np.random.default_rng(seed)
    grid = pd.RangeIndex(n_bars)
    starts = np.sort(rng.choice(n_bars - max_span - 1, size=n_lab, replace=False))
    ends = np.minimum(starts + rng.integers(0, max_span, size=n_lab), n_bars - 1)
    return indicator_matrix(grid, pd.Series(grid[ends], index=grid[starts]))


def csc_of(m: np.ndarray):
    """CSC arrays (ptr, rows, colsum) of a 0/1 matrix, built as sequential_bootstrap builds them."""
    j_idx, t_idx = np.nonzero((m != 0.0).T)
    colsum = np.bincount(j_idx, minlength=m.shape[1]).astype(np.float64)
    ptr = np.zeros(m.shape[1] + 1, dtype=np.int64)
    ptr[1:] = np.cumsum(colsum.astype(np.int64))
    return ptr, t_idx.astype(np.int64), colsum


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

    # Sequential bootstrap at three sizes. The full-recompute kernel is the reference that
    # sequential_bootstrap must match draw for draw; both are timed on the same CSC arrays.
    print("sequential bootstrap, full-recompute kernel vs sequential_bootstrap (seed 1):")
    for label, n_bars_s, n_lab_s, max_span in SEQ_SIZES:
        m_df = sequential_case(n_bars_s, n_lab_s, max_span, seed=2)
        ptr, rows, colsum = csc_of(m_df.to_numpy())
        uniforms_s = np.random.default_rng(1).random(n_lab_s)
        full_draws = weights._seq_bootstrap_kernel(n_bars_s, ptr, rows, colsum, uniforms_s)
        if not np.array_equal(full_draws, sequential_bootstrap(m_df, seed=1)):
            raise SystemExit(f"sequential_bootstrap draws differ from the reference at size {label}")
        t_full = best_time(lambda: weights._seq_bootstrap_kernel(n_bars_s, ptr, rows, colsum, uniforms_s))
        t_pub = best_time(lambda: sequential_bootstrap(m_df, seed=1))
        print(f"  {label:<7} bars {n_bars_s:>5}  labels {n_lab_s:>5}  ones {rows.size:>8}  "
              f"full kernel {t_full * 1e3:9.2f} ms | sequential_bootstrap {t_pub * 1e3:9.2f} ms | "
              f"speedup {t_full / t_pub:6.2f}x")


if __name__ == "__main__":
    main()
