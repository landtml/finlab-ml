"""Benchmark sequential-bootstrap draws for bagging: vectorised vs a naive loop.

The naive reference recomputes every label's average uniqueness from scratch at
each draw, which is the textbook O(n^2 * draws) implementation. Both implement the same probabilities
but consume random numbers differently, so the draws differ; the speedup compares
the algorithms, not identical outputs.

The second section times SequentialBootstrapBagging.fit with and without
oob_score=True, to show what the out-of-bag pass adds to a fit.

Run: PYTHONPATH=src python3 benchmarks/bench_ensemble.py
"""
import time
import warnings

import numpy as np
import pandas as pd

from quantowl.cv import make_t1
from quantowl.ensemble import SequentialBootstrapBagging
from quantowl.weights import indicator_matrix, sequential_bootstrap


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


class Centroid:
    """Nearest-centroid base with predict_proba, cheap enough that the out-of-bag pass shows."""

    def fit(self, X, y, sample_weight=None):
        self.classes_ = np.unique(y)
        self.c_ = np.stack([X[y == c].mean(axis=0) for c in self.classes_])
        return self

    def predict_proba(self, X):
        s = np.exp(-((X[:, None, :] - self.c_[None]) ** 2).sum(axis=2))
        return s / s.sum(axis=1, keepdims=True)


def oob_overhead(n: int = 1000, n_estimators: int = 50) -> None:
    idx = pd.date_range("2020-01-01", periods=n, freq="D")
    t1 = make_t1(idx, horizon=15)
    rng = np.random.default_rng(0)
    y = rng.integers(0, 2, n)
    X = rng.normal(size=(n, 2)) + np.where(y[:, None] == 1, 1.5, -1.5)

    def fit(oob: bool, sequential: bool) -> SequentialBootstrapBagging:
        with warnings.catch_warnings():
            # Rows drawn by every bag warn on each fit; the timing does not need the warning.
            warnings.simplefilter("ignore", UserWarning)
            return SequentialBootstrapBagging(
                Centroid, n_estimators=n_estimators, sequential=sequential, seed=0, oob_score=oob
            ).fit(X, y, t1=t1)

    fit(False, True)  # warm-up: compiles the jitted draws
    for sequential in (True, False):
        off = best_of(lambda: fit(False, sequential))
        on = best_of(lambda: fit(True, sequential))
        draws = "sequential" if sequential else "iid"
        print(
            f"oob_score overhead, {draws} bags, n={n}, bags={n_estimators}: fit off {off:.4f} s, "
            f"fit on {on:.4f} s, ratio {on / off:.2f}x"
        )


if __name__ == "__main__":
    main()
    oob_overhead()
