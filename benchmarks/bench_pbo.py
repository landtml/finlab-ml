"""Benchmark: naive Python CSCV loop vs the numba kernel in finlab.pbo.

Not run by CI. Usage:
    PYTHONPATH=src python3 benchmarks/bench_pbo.py
"""

from __future__ import annotations

import itertools
import math
import time

import numpy as np

from finlab.pbo import probability_of_backtest_overfitting


def naive_cscv_logits(M: np.ndarray, S: int) -> np.ndarray:
    T, N = M.shape
    L = T // S
    blocks = [M[s * L : (s + 1) * L] for s in range(S)]
    out = []
    for J in itertools.combinations(range(S), S // 2):
        Jbar = [s for s in range(S) if s not in J]
        tr = np.vstack([blocks[s] for s in J])
        te = np.vstack([blocks[s] for s in Jbar])
        R = [tr[:, n].mean() / tr[:, n].std(ddof=1) for n in range(N)]
        ns = int(np.argmax(R))
        Rb = [te[:, n].mean() / te[:, n].std(ddof=1) for n in range(N)]
        rank = sum(1 for n in range(N) if Rb[n] <= Rb[ns])
        w = rank / (N + 1)
        out.append(np.log(w / (1 - w)))
    return np.array(out)


def _time(fn, repeats: int = 3) -> float:
    best = float("inf")
    for _ in range(repeats):
        t0 = time.perf_counter()
        fn()
        best = min(best, time.perf_counter() - t0)
    return best


def main() -> None:
    # JIT warm-up: compile the kernel once before timing.
    warm = np.random.default_rng(0).normal(size=(16, 3))
    probability_of_backtest_overfitting(warm, n_partitions=4)

    print(f"{'T':>6} {'N':>4} {'S':>3} {'splits':>7} {'naive (s)':>10} {'numba (s)':>10} {'speedup':>8}")
    for T, N, S in [(640, 30, 8), (1200, 40, 12), (1600, 50, 16)]:
        M = np.random.default_rng(T + N + S).normal(0.0, 0.01, size=(T, N))
        fast_t = _time(lambda: probability_of_backtest_overfitting(M, n_partitions=S))
        if S <= 12:
            slow_t = _time(lambda: naive_cscv_logits(M, S), repeats=1)
            speed = f"{slow_t / fast_t:8.1f}x"
            slow_s = f"{slow_t:10.4f}"
        else:
            slow_s = f"{'skipped':>10}"
            speed = f"{'-':>8}"
        print(f"{T:>6} {N:>4} {S:>3} {math.comb(S, S // 2):>7} {slow_s} {fast_t:10.4f} {speed}")


if __name__ == "__main__":
    main()
