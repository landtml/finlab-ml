"""Benchmark: naive reference vs jitted kernels in finlab.microstructure.

Not run by CI. Usage::

    PYTHONPATH=src python3 benchmarks/bench_microstructure.py

The naive references are the loop-based functions in tests/test_microstructure.py.
Each jitted kernel is called once before timing, so compilation is excluded.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tests"))

from finlab.microstructure import corwin_schultz_spread, tick_rule, vpin  # noqa: E402
from test_microstructure import naive_corwin_schultz, naive_tick_rule, naive_vpin  # noqa: E402


def best_of(fn, repeats: int = 3) -> float:
    best = float("inf")
    for _ in range(repeats):
        t0 = time.perf_counter()
        fn()
        best = min(best, time.perf_counter() - t0)
    return best


def report(name: str, t_naive: float, t_fast: float, n: int) -> None:
    print(
        f"{name:<22} n={n:<9} naive={t_naive * 1e3:10.2f} ms  "
        f"jit={t_fast * 1e3:9.3f} ms  speedup={t_naive / t_fast:8.1f}x"
    )


def main() -> None:
    rng = np.random.default_rng(0)

    # Tick rule
    n = 200_000
    prices = np.round(100 + np.cumsum(rng.normal(size=n)), 2)
    tick_rule(prices[:10])  # warm up JIT
    t_fast = best_of(lambda: tick_rule(prices))
    t_naive = best_of(lambda: naive_tick_rule(prices.tolist()), repeats=1)
    report("tick_rule", t_naive, t_fast, n)

    # Corwin-Schultz (sl=1)
    n = 200_000
    mid = 100 * np.exp(np.cumsum(rng.normal(scale=0.01, size=n)))
    high = mid * (1 + np.abs(rng.normal(scale=0.01, size=n)))
    low = mid * (1 - np.abs(rng.normal(scale=0.01, size=n)))
    corwin_schultz_spread(high[:10], low[:10])  # warm up JIT
    t_fast = best_of(lambda: corwin_schultz_spread(high, low))
    hl, ll = high.tolist(), low.tolist()
    t_naive = best_of(lambda: naive_corwin_schultz(hl, ll, 1), repeats=1)
    report("corwin_schultz", t_naive, t_fast, n)

    # VPIN (per-bucket fill kernel). The naive reference expands trade units
    # and is much slower, so use a smaller n.
    n = 20_000
    p = 100 + np.cumsum(rng.normal(scale=0.3, size=n))
    vol = rng.integers(1, 9, size=n).astype(float)
    vpin(p[:50], vol[:50], bucket_volume=25.0, n_window=4)  # warm up JIT
    t_fast = best_of(lambda: vpin(p, vol, bucket_volume=25.0, n_window=4))
    t_naive = best_of(
        lambda: naive_vpin(p.tolist(), vol.astype(int).tolist(), 25, 4), repeats=1
    )
    report("vpin", t_naive, t_fast, n)


if __name__ == "__main__":
    main()
