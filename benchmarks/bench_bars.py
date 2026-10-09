"""Benchmark finlab.bars: naive per-tick Python loops vs the numba kernels.

Not run by CI. Run with:  PYTHONPATH=src python3 benchmarks/bench_bars.py

Kernel rows compare the boundary-detection step (the part the book's rules
define) with a plain Python loop on the same input at the same size. The
end-to-end row times the public function, which also builds the OHLCV frame;
the naive loop has no equivalent for that step, so it is not compared. The
JIT is warmed up first. Nothing is extrapolated.
"""

from __future__ import annotations

import time

import numpy as np

from finlab.bars import _cum_boundaries, _imbalance_ends, _tick_rule, imbalance_bars, tick_bars


def naive_tick_ends(x: np.ndarray, threshold: float) -> list[int]:
    ends, acc = [], 0.0
    for t in range(len(x)):
        acc += x[t]
        if acc >= threshold:
            ends.append(t)
            acc = 0.0
    return ends


def naive_imbalance_ends(p: np.ndarray, init_T: int, span_bars: float, span_ticks: float, b0: float) -> list[int]:
    """Tick rule and imbalance bars with recursive EWMA updates, in pure Python."""
    n = len(p)
    a_b = 2.0 / (span_bars + 1.0)
    a_t = 2.0 / (span_ticks + 1.0)
    b = [b0] * n
    for t in range(1, n):
        dp = p[t] - p[t - 1]
        b[t] = b[t - 1] if dp == 0 else (1.0 if dp > 0 else -1.0)
    m = min(init_T, n)
    e_T = float(init_T)
    e_s = sum(b[:m]) / m
    ends, start, theta = [], 0, 0.0
    thr = e_T * abs(e_s)
    for t in range(n):
        theta += b[t]
        if abs(theta) >= thr:
            ends.append(t)
            e_T = a_b * (t - start + 1) + (1 - a_b) * e_T
            for j in range(start, t + 1):
                e_s = a_t * b[j] + (1 - a_t) * e_s
            start, theta = t + 1, 0.0
            thr = e_T * abs(e_s)
    return ends


def timed(fn, repeat: int = 3) -> float:
    best = float("inf")
    for _ in range(repeat):
        t0 = time.perf_counter()
        fn()
        best = min(best, time.perf_counter() - t0)
    return best


def main() -> None:
    rng = np.random.default_rng(0)
    # JIT warm-up (compilation excluded from all timings).
    wp = 100 + np.cumsum(rng.normal(0, 0.01, 2_000))
    wv = rng.integers(1, 100, 2_000).astype(float)
    _cum_boundaries(np.ones(2_000), 50.0)
    _tick_rule(wp, 1.0)
    _imbalance_ends(np.ones(2_000), 20, 0.1, 0.01)
    tick_bars(wp, 50, volumes=wv)
    imbalance_bars(wp, kind="tick", init_T=20)

    n = 300_000
    p = 100 + np.cumsum(rng.normal(0, 0.01, n))
    v = rng.integers(1, 100, n).astype(float)
    print(f"Same input for naive and jit, n={n:,} ticks.")
    print(f"{'benchmark':<42}{'naive (s)':>12}{'jit (s)':>12}{'speedup':>10}")

    t_naive = timed(lambda: naive_tick_ends(np.ones(n), 100.0), repeat=1)
    t_fast = timed(lambda: _cum_boundaries(np.ones(n), 100.0))
    print(f"{'tick bar boundaries (kernel)':<42}{t_naive:>12.4f}{t_fast:>12.4f}{t_naive / t_fast:>9.1f}x")

    t_naive = timed(lambda: naive_imbalance_ends(p, 100, 20.0, 1000.0, 1.0), repeat=1)

    def fast_imb() -> None:
        b = _tick_rule(p, 1.0)
        _imbalance_ends(b, 100, 2.0 / 21.0, 2.0 / 1001.0)

    t_fast = timed(fast_imb)
    print(f"{'tick imbalance boundaries (kernel)':<42}{t_naive:>12.4f}{t_fast:>12.4f}{t_naive / t_fast:>9.1f}x")

    t_e2e = timed(lambda: tick_bars(p, 100, volumes=v), repeat=1)
    print(f"\nEnd-to-end tick_bars (jit kernel + OHLCV frame): {t_e2e:.4f} s")
    t_e2e = timed(lambda: imbalance_bars(p, kind="tick", init_T=100), repeat=1)
    print(f"End-to-end imbalance_bars, tick kind (incl. frame): {t_e2e:.4f} s")


if __name__ == "__main__":
    main()
