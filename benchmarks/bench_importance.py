"""Benchmark: naive loop MDI vs the jitted MDI column-statistics kernel.

Not run by CI. Usage:

    PYTHONPATH=src python3 benchmarks/bench_importance.py
"""

from __future__ import annotations

import time

import numpy as np

from finlab.importance import mdi_from_matrix


def naive_mdi(imp: list[list[float]]) -> tuple[list[float], list[float]]:
    """Pure-Python reference: skip zeros and NaN, mean, standard error, normalize."""
    n_trees = len(imp)
    p = len(imp[0])
    means, ses = [], []
    for j in range(p):
        vals = []
        for t in range(n_trees):
            v = imp[t][j]
            if v != 0.0 and v == v:
                vals.append(v)
        c = len(vals)
        m = sum(vals) / c
        ss = 0.0
        for v in vals:
            ss += (v - m) ** 2
        se = (ss / (c - 1)) ** 0.5 / c**0.5 if c > 1 else float("nan")
        means.append(m)
        ses.append(se)
    total = sum(means)
    return [m / total for m in means], [s / total for s in ses]


def best_of(fn, repeats: int = 3) -> float:
    best = float("inf")
    for _ in range(repeats):
        t0 = time.perf_counter()
        fn()
        best = min(best, time.perf_counter() - t0)
    return best


def main() -> None:
    rng = np.random.default_rng(0)
    # Warm up the JIT (compile happens once; cache=True reuses it on later runs).
    warm = rng.random((10, 4))
    warm[warm < 0.3] = 0.0
    mdi_from_matrix(warm)

    shapes = [(500, 50), (2000, 200), (5000, 400)]
    print(f"{'shape (trees x feats)':>24} {'naive (s)':>12} {'jitted (s)':>12} {'speedup':>10}")
    for n_trees, n_feats in shapes:
        imp = rng.random((n_trees, n_feats))
        imp[rng.random(imp.shape) < 0.5] = 0.0  # max_features=1 style sparsity
        as_list = imp.tolist()

        mdi_from_matrix(imp)  # ensure this shape path is warm
        t_naive = best_of(lambda: naive_mdi(as_list), repeats=1)
        t_jit = best_of(lambda: mdi_from_matrix(imp))
        print(
            f"{f'{n_trees} x {n_feats}':>24} {t_naive:>12.4f} {t_jit:>12.5f} "
            f"{t_naive / t_jit:>9.1f}x"
        )


if __name__ == "__main__":
    main()
