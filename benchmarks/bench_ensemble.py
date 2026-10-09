"""Benchmark sequential-bootstrap draws for bagging: vectorised vs a naive loop.

The naive reference recomputes every label's average uniqueness from scratch at
each draw, which is the textbook O(n^2 * draws) implementation. Both implement the same probabilities
but consume random numbers differently, so the draws differ; the speedup compares
the algorithms, not identical outputs.

Run: PYTHONPATH=src python3 benchmarks/bench_ensemble.py
"""
import time

import numpy as np
import pandas as pd

from finlab.cv import make_t1
from finlab.weights import indicator_matrix, sequential_bootstrap


def naive_sequential_bootstrap(ind, n_draws, seed):
    rng = np.random.default_rng(seed)
    dense = np.asarray(ind, dtype=float)
    T, I = dense.shape
    drawn = []
    c = np.zeros(T)
    for _ in range(n_draws):
        avg_u = np.zeros(I)
        for j in range(I):
            tj = dense[:, j] > 0
            avg_u[j] = np.mean(1.0 / (1.0 + c[tj])) if tj.any() else 0.0
        p = avg_u / avg_u.sum()
        j = rng.choice(I, p=p)
        drawn.append(j)
        c += dense[:, j]
    return np.array(drawn)


def best_of(fn, repeats=3):
    best = float("inf")
    for _ in range(repeats):
        t0 = time.perf_counter()
        fn()
        best = min(best, time.perf_counter() - t0)
    return best


def main():
    for n in (100, 300):
        idx = pd.date_range("2020-01-01", periods=n, freq="D")
        t1 = make_t1(idx, horizon=15)
        ind = indicator_matrix(idx, t1)
        sequential_bootstrap(ind, seed=0)  # warm-up
        fast = best_of(lambda: sequential_bootstrap(ind, seed=0))
        slow = best_of(lambda: naive_sequential_bootstrap(ind.to_numpy(), n, 0), repeats=1)
        print(f"n={n}: naive {slow:.4f} s, jitted {fast:.6f} s, speedup {slow / fast:.0f}x")


if __name__ == "__main__":
    main()
