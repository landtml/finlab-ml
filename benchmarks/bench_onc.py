"""Benchmark ONC's silhouette kernel: vectorised vs a naive per-variable loop.

Run: PYTHONPATH=src python3 benchmarks/bench_onc.py
"""
import time

import numpy as np

from finlab.onc import silhouette_scores


def naive_silhouette(d, labels):
    n = len(labels)
    out = np.zeros(n)
    for i in range(n):
        own = [j for j in range(n) if labels[j] == labels[i] and j != i]
        if not own:
            continue
        a = np.mean([d[i, j] for j in own])
        b = min(np.mean([d[i, j] for j in range(n) if labels[j] == c])
                for c in set(labels.tolist()) if c != labels[i])
        out[i] = (b - a) / max(a, b) if max(a, b) > 0 else 0.0
    return out


def best_of(fn, repeats=3):
    best = float("inf")
    for _ in range(repeats):
        t0 = time.perf_counter()
        fn()
        best = min(best, time.perf_counter() - t0)
    return best


def main():
    rng = np.random.default_rng(0)
    for n in (200, 800):
        corr = np.corrcoef(rng.normal(size=(n, 40)))
        d = np.sqrt(0.5 * (1 - corr))
        labels = rng.integers(0, 6, n)
        silhouette_scores(d, labels)  # warm-up
        fast = best_of(lambda: silhouette_scores(d, labels))
        slow = best_of(lambda: naive_silhouette(d, labels), repeats=1)
        assert np.allclose(silhouette_scores(d, labels), naive_silhouette(d, labels))
        print(f"n={n}: naive {slow:.4f} s, vectorised {fast:.6f} s, speedup {slow / fast:.0f}x")


if __name__ == "__main__":
    main()
