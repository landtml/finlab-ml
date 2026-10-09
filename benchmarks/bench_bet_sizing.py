"""Averaging overlapping bet signals: difference-array sweep vs a per-timestamp loop.

Run: PYTHONPATH=src python3 benchmarks/bench_bet_sizing.py
"""
import numpy as np
import pandas as pd

from _timing import best_of
from finlab.bet_sizing import average_active_signals


def naive_average(signals, t1, grid):
    vals = signals.to_numpy()
    s0 = signals.index.to_numpy()
    s1 = t1.to_numpy()
    out = np.zeros(len(grid))
    for k, t in enumerate(grid):
        active = [vals[i] for i in range(len(vals)) if s0[i] <= t <= s1[i]]
        out[k] = np.mean(active) if active else 0.0
    return out


def main():
    rng = np.random.default_rng(0)
    for n in (2000, 8000):
        idx = pd.date_range("2000-01-01", periods=n, freq="h")
        sig = pd.Series(rng.uniform(-1, 1, n), index=idx)
        lengths = rng.integers(1, 30, n)
        t1 = pd.Series([idx[min(i + L, n - 1)] for i, L in enumerate(lengths)], index=idx)
        average_active_signals(sig, t1)  # warm-up
        fast = best_of(lambda: average_active_signals(sig, t1))
        slow = best_of(lambda: naive_average(sig, t1, idx), repeats=1)
        print(f"n={n}: naive {slow:.4f} s, sweep {fast:.5f} s, speedup {slow / fast:.0f}x")


if __name__ == "__main__":
    main()
