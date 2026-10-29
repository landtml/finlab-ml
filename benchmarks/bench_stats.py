"""Benchmark: naive Python loop vs the numba kernel behind quantowl.stats.holding_period.

Not run by CI. Usage:
    PYTHONPATH=src python3 benchmarks/bench_stats.py
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from _timing import best_of
from quantowl.stats import holding_period


def naive_holding_period(positions: list[float], days: list[float]) -> float:
    closed: list[tuple[float, float]] = []
    t_entry = 0.0
    for i in range(1, len(positions)):
        p_diff = positions[i] - positions[i - 1]
        if p_diff * positions[i - 1] >= 0:
            if positions[i] != 0:
                t_entry = (t_entry * positions[i - 1] + days[i] * p_diff) / positions[i]
        elif positions[i] * positions[i - 1] < 0:
            closed.append((days[i] - t_entry, abs(positions[i - 1])))
            t_entry = days[i]
        else:
            closed.append((days[i] - t_entry, abs(p_diff)))
    total = sum(w for _, w in closed)
    return sum(dt * w for dt, w in closed) / total if total > 0 else float("nan")


def main() -> None:
    rng = np.random.default_rng(14)
    # JIT warm-up: compile the kernel once before timing.
    warm = pd.Series([1.0, 2.0, 0.0], index=pd.date_range("2024-01-01", periods=3))
    holding_period(warm)

    n = 100_000
    steps = rng.integers(-1, 2, size=n)
    positions = np.clip(np.cumsum(steps), -5, 5).astype(float)
    index = pd.date_range("2000-01-01", periods=n, freq="D")
    series = pd.Series(positions, index=index)
    days = ((index - index[0]) / pd.Timedelta(days=1)).tolist()
    pos_list = positions.tolist()

    fast_t = best_of(lambda: holding_period(series))
    slow_t = best_of(lambda: naive_holding_period(pos_list, days), repeats=1)
    diff = abs(holding_period(series) - naive_holding_period(pos_list, days))
    print(f"{'rows':>9} {'naive (s)':>10} {'numba (s)':>10} {'speedup':>8} {'|diff|':>10}")
    print(f"{n:>9} {slow_t:10.4f} {fast_t:10.4f} {slow_t / fast_t:7.1f}x {diff:10.2e}")

if __name__ == "__main__":
    main()
