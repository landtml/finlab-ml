"""Benchmark quantowl.structural_breaks against literal naive loops (not run by CI).

Usage: PYTHONPATH=src python3 benchmarks/bench_structural_breaks.py
"""

from __future__ import annotations

import math
import time

import numpy as np

from quantowl.structural_breaks import adf_stat, cadf, chu_stinchcombe_white, cusum_test, qadf, sadf, smt


def best_of(fn, repeat: int = 3) -> float:
    times = []
    for _ in range(repeat):
        t0 = time.perf_counter()
        fn()
        times.append(time.perf_counter() - t0)
    return min(times)


def naive_adf(y: np.ndarray, lag: int) -> float:
    rows, targets = [], []
    for t in range(lag + 1, len(y)):
        rows.append([y[t - 1]] + [y[t - l] - y[t - l - 1] for l in range(1, lag + 1)] + [1.0])
        targets.append(y[t] - y[t - 1])
    x, yy = np.array(rows), np.array(targets)
    beta, *_ = np.linalg.lstsq(x, yy, rcond=None)
    resid = yy - x @ beta
    cov = (resid @ resid / (len(yy) - x.shape[1])) * np.linalg.inv(x.T @ x)
    return beta[0] / math.sqrt(cov[0, 0])


def naive_sadf(y: np.ndarray, min_length: int, lag: int) -> np.ndarray:
    out = np.full(len(y), np.nan)
    for e in range(min_length - 1, len(y)):
        out[e] = max(naive_adf(y[s : e + 1], lag) for s in range(0, e - min_length + 2))
    return out


def naive_qadf(y: np.ndarray, min_length: int, lag: int, q: float) -> np.ndarray:
    out = np.full(len(y), np.nan)
    for e in range(min_length - 1, len(y)):
        taus = [naive_adf(y[s : e + 1], lag) for s in range(0, e - min_length + 2)]
        out[e] = np.quantile(taus, q)
    return out


def naive_smt_poly1(y: np.ndarray, min_length: int) -> np.ndarray:
    out = np.full(len(y), np.nan)
    positions = np.arange(1, len(y) + 1, dtype=float)
    for e in range(min_length - 1, len(y)):
        best = -np.inf
        for s in range(0, e - min_length + 2):
            pos = positions[s : e + 1]
            yy = y[s : e + 1]
            x = np.column_stack([np.ones_like(pos), pos, pos * pos])
            beta, *_ = np.linalg.lstsq(x, yy, rcond=None)
            resid = yy - x @ beta
            cov = (resid @ resid / (len(pos) - 3)) * np.linalg.inv(x.T @ x)
            best = max(best, abs(beta[2]) / math.sqrt(cov[2, 2]))
        out[e] = best
    return out


def naive_cusum(y: np.ndarray) -> np.ndarray:
    w = np.full(len(y), np.nan)
    for t in range(1, len(y)):
        xp = np.ones((t, 1))
        beta = np.linalg.lstsq(xp, y[:t], rcond=None)[0]
        f = 1.0 + 1.0 / t
        w[t] = (y[t] - beta[0]) / math.sqrt(f)
    return w


def naive_csw_sup(y: np.ndarray) -> np.ndarray:
    out = np.full(len(y), np.nan)
    d = np.diff(y, prepend=y[0])
    for t in range(1, len(y)):
        sig = math.sqrt(np.sum(d[1 : t + 1] ** 2) / t)
        out[t] = max((y[t] - y[n]) / (sig * math.sqrt(t - n)) for n in range(t))
    return out


def main() -> None:
    rng = np.random.default_rng(0)

    # Warm up every jitted path (compilation is excluded from the timings).
    warm = rng.standard_normal(60).cumsum()
    adf_stat(warm, 1)
    sadf(warm, 20, 1)
    qadf(warm, 20, 1, q=0.95)
    cadf(warm, 20, 1, q=0.95)
    smt(np.exp(0.01 * np.arange(60) + warm / 10), 12, "poly1")
    cusum_test(warm)
    chu_stinchcombe_white(warm)
    chu_stinchcombe_white(warm, reference=3)

    print("structural_breaks: naive reference vs numba kernels (best of 3)")
    print(f"{'routine':<28}{'n':>6}{'naive s':>12}{'jit s':>12}{'speedup':>10}")

    y = rng.standard_normal(300).cumsum()
    t_naive = best_of(lambda: naive_sadf(y, 30, 1), repeat=1)
    t_jit = best_of(lambda: sadf(y, 30, 1))
    print(f"{'sadf (lag=1, c)':<28}{300:>6}{t_naive:>12.4f}{t_jit:>12.6f}{t_naive / t_jit:>9.0f}x")

    t_naive = best_of(lambda: naive_qadf(y, 30, 1, 0.95), repeat=1)
    t_jit = best_of(lambda: qadf(y, 30, 1, q=0.95))
    print(f"{'qadf (lag=1, c, q=0.95)':<28}{300:>6}{t_naive:>12.4f}{t_jit:>12.6f}{t_naive / t_jit:>9.0f}x")

    y = np.exp(0.01 * np.arange(200) + 0.05 * rng.standard_normal(200))
    t_naive = best_of(lambda: naive_smt_poly1(y, 30), repeat=1)
    t_jit = best_of(lambda: smt(y, 30, "poly1"))
    print(f"{'smt (poly1, phi=0)':<28}{200:>6}{t_naive:>12.4f}{t_jit:>12.6f}{t_naive / t_jit:>9.0f}x")

    y = rng.standard_normal(2000)
    t_naive = best_of(lambda: naive_cusum(y), repeat=1)
    t_jit = best_of(lambda: cusum_test(y))
    print(f"{'cusum_test (mean model)':<28}{2000:>6}{t_naive:>12.4f}{t_jit:>12.6f}{t_naive / t_jit:>9.0f}x")

    y = rng.standard_normal(1500).cumsum()
    t_naive = best_of(lambda: naive_csw_sup(y), repeat=1)
    t_jit = best_of(lambda: chu_stinchcombe_white(y))
    print(f"{'chu_stinchcombe_white sup':<28}{1500:>6}{t_naive:>12.4f}{t_jit:>12.6f}{t_naive / t_jit:>9.0f}x")

    print("\njit-only timing for larger inputs (no naive run):")
    y = rng.standard_normal(1000).cumsum()
    print(f"sadf n=1000 (lag=1, c): {best_of(lambda: sadf(y, 50, 1), repeat=1):.3f} s")
    print(f"qadf n=1000 (lag=1, c, q=0.95): {best_of(lambda: qadf(y, 50, 1, q=0.95), repeat=1):.3f} s")
    y = np.exp(0.01 * np.arange(1000) + 0.05 * rng.standard_normal(1000))
    print(f"smt n=1000 (poly1, phi=0, min_length=50): {best_of(lambda: smt(y, 50, 'poly1'), repeat=1):.3f} s")


if __name__ == "__main__":
    main()
