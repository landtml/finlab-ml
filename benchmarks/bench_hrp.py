"""Benchmark: naive recursive HRP vs jitted hrp_weights in finlab.hrp.

Not run by CI. Usage::

    PYTHONPATH=src python3 benchmarks/bench_hrp.py

The naive reference is the list-based recursion in tests/test_hrp.py. The
jitted version is called once before timing, so compilation is excluded.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tests"))

from finlab.hrp import hrp_weights  # noqa: E402
from test_hrp import naive_hrp  # noqa: E402


def best_of(fn, repeats: int = 3) -> float:
    best = float("inf")
    for _ in range(repeats):
        t0 = time.perf_counter()
        fn()
        best = min(best, time.perf_counter() - t0)
    return best


def main() -> None:
    rng = np.random.default_rng(0)
    for n in (50, 200, 500):
        X = rng.normal(size=(4 * n, n))
        C = np.cov(X, rowvar=False)
        hrp_weights(C[:4, :4])  # warm up JIT
        t_fast = best_of(lambda: hrp_weights(C))
        t_naive = best_of(lambda: naive_hrp(C), repeats=1)
        print(
            f"hrp_weights            n={n:<9} naive={t_naive * 1e3:10.2f} ms  "
            f"jit={t_fast * 1e3:9.3f} ms  speedup={t_naive / t_fast:8.1f}x"
        )


if __name__ == "__main__":
    main()
