"""Benchmark finlab.labeling against brute-force references.

Not run by CI. Usage:

    PYTHONPATH=src python3 (from the repository root) benchmarks/bench_labeling.py

Each jitted function is called once before timing so numba compilation is not
measured. Each timing is the best of ``REPEATS`` runs.
"""

from __future__ import annotations

import time

import numpy as np
import pandas as pd
from scipy import stats

from finlab.labeling import add_vertical_barrier, get_events, trend_scanning_labels

REPEATS = 3


def best_time(fn, repeats: int = REPEATS) -> float:
    best = float("inf")
    for _ in range(repeats):
        t0 = time.perf_counter()
        fn()
        best = min(best, time.perf_counter() - t0)
    return best


def naive_events(close, t_events, pt_sl, target, vert):
    """Snippet 3.2 style: one pandas path slice per event."""
    out = []
    for t0 in t_events:
        trg = target.loc[t0]
        t_v = vert.loc[t0]
        end = close.index[-1] if pd.isna(t_v) else t_v
        path = close.loc[t0:end]
        rel = path / close.loc[t0] - 1.0
        t_pt = rel[rel > pt_sl[0] * trg].index.min()
        t_sl = rel[rel < -pt_sl[1] * trg].index.min()
        if not pd.isna(t_sl) and (pd.isna(t_pt) or t_sl <= t_pt):
            out.append(t_sl)
        elif not pd.isna(t_pt):
            out.append(t_pt)
        else:
            out.append(t_v)
    return out


def naive_trend(close, windows, threshold):
    y = np.log(close.to_numpy(dtype=float))
    n = len(y)
    for i in range(n):
        best, bt = -1.0, 0.0
        for length in windows:
            if i + length > n:
                continue
            res = stats.linregress(np.arange(length), y[i:i + length])
            t = res.slope / res.stderr
            if abs(t) > best:
                best, bt = abs(t), t
        _ = (bt, threshold)


def main() -> None:
    rng = np.random.default_rng(0)

    # Triple barrier: 200k bars, 5k events, vertical barrier 40 calendar days.
    n_bars = 200_000
    idx = pd.date_range("2000-01-01", periods=n_bars, freq="h")
    close = pd.Series(100.0 * np.exp(np.cumsum(rng.normal(0, 0.001, n_bars))), index=idx)
    t_events = idx[np.sort(rng.choice(n_bars - 2000, size=5_000, replace=False))]
    target = pd.Series(rng.uniform(0.002, 0.01, n_bars), index=idx)
    vert = add_vertical_barrier(t_events, close, num_days=40 / 24)
    pt_sl = (1.5, 1.5)

    get_events(close, t_events[:10], pt_sl, target, 0.0, vertical_barrier_times=vert[t_events[:10]])
    t_jit = best_time(lambda: get_events(close, t_events, pt_sl, target, 0.0,
                                         vertical_barrier_times=vert))
    sub = t_events[:1_000]
    t_naive_1k = best_time(lambda: naive_events(close, sub, pt_sl, target, vert), repeats=1)
    t_naive = t_naive_1k * (len(t_events) / len(sub))
    print(f"get_events        : naive (pandas loop, extrapolated from 1k events) "
          f"{t_naive * 1e3:9.1f} ms | jitted {t_jit * 1e3:8.2f} ms | "
          f"speedup {t_naive / t_jit:7.1f}x")

    # Trend scanning: 5k bars, windows 10..40.
    m = 5_000
    close_s = close.iloc[:m]
    windows = [10, 20, 40]
    trend_scanning_labels(close_s, windows)
    t_jit = best_time(lambda: trend_scanning_labels(close_s, windows))
    t_naive = best_time(lambda: naive_trend(close_s, windows, 1.96), repeats=1)
    print(f"trend_scanning    : naive (linregress per bar and window) "
          f"{t_naive * 1e3:9.1f} ms | jitted {t_jit * 1e3:8.2f} ms | "
          f"speedup {t_naive / t_jit:7.1f}x")


if __name__ == "__main__":
    main()
