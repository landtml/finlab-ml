"""Benchmark finlab.fracdiff: naive per-observation dot products vs the numba kernels.

Not run by CI. Run with:  PYTHONPATH=src python3 benchmarks/bench_fracdiff.py

Naive and compiled versions run on the same input at the same size. Nothing is
extrapolated. The naive references loop in Python and call ``np.dot`` once per
observation (the book's snippet style). The JIT is warmed up first.
"""

from __future__ import annotations

import time

import numpy as np

from finlab.fracdiff import frac_diff, frac_diff_ffd, get_weights, get_weights_ffd


def naive_ffd(x: np.ndarray, w_lag: np.ndarray) -> np.ndarray:
    w_book = w_lag[::-1]
    width = len(w_lag) - 1
    out = np.full(len(x), np.nan)
    for t in range(width, len(x)):
        out[t] = np.dot(w_book, x[t - width : t + 1])
    return out


def naive_expanding(x: np.ndarray, d: float, thres: float) -> np.ndarray:
    n = len(x)
    w_lag = get_weights(d, n)
    a = np.abs(w_lag)
    total = a.sum()
    lam = (total - np.cumsum(a)) / total
    first = int(np.flatnonzero(lam <= thres)[0])
    out = np.full(n, np.nan)
    for t in range(first, n):
        out[t] = np.dot(w_lag[: t + 1][::-1], x[: t + 1])
    return out


def timed(fn, repeat: int = 3) -> float:
    best = float("inf")
    for _ in range(repeat):
        t0 = time.perf_counter()
        fn()
        best = min(best, time.perf_counter() - t0)
    return best


def main() -> None:
    rng = np.random.default_rng(0)
    frac_diff_ffd(rng.normal(size=500), 0.4, thres=1e-3)  # JIT warm-up
    frac_diff(rng.normal(size=200), 0.4, thres=1e-2)

    d, thres = 0.4, 1e-5
    w = get_weights_ffd(d, thres)
    n_ffd = 300_000
    x_ffd = np.cumsum(rng.normal(0, 0.01, n_ffd))
    print(f"FFD d={d}, thres={thres:g}: window width={len(w) - 1}, n={n_ffd:,}.")
    print(f"{'benchmark':<34}{'naive (s)':>12}{'jit (s)':>12}{'speedup':>10}")
    t_naive = timed(lambda: naive_ffd(x_ffd, w), repeat=1)
    t_fast = timed(lambda: frac_diff_ffd(x_ffd, d, thres=thres))
    print(f"{'frac_diff_ffd':<34}{t_naive:>12.4f}{t_fast:>12.4f}{t_naive / t_fast:>9.1f}x")

    n_exp = 6_000
    x_exp = np.cumsum(rng.normal(0, 0.01, n_exp))
    print(f"\nExpanding window d={d}, thres=0.01, n={n_exp:,} (O(n^2) in both).")
    t_naive = timed(lambda: naive_expanding(x_exp, d, 0.01), repeat=1)
    t_fast = timed(lambda: frac_diff(x_exp, d, thres=0.01))
    print(f"{'frac_diff (expanding)':<34}{t_naive:>12.4f}{t_fast:>12.4f}{t_naive / t_fast:>9.1f}x")


if __name__ == "__main__":
    main()
