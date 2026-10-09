"""mp_pandas_obj: serial vs process pool on a CPU-bound per-molecule function.

Speed-up depends on the core count and on pickling cost; this reports the measured
ratio on the machine that runs it, not a general claim.

Run: PYTHONPATH=src python3 benchmarks/bench_parallel.py
"""
import os

import numpy as np
import pandas as pd

from _timing import best_of
from finlab.parallel import mp_pandas_obj


def heavy(molecule, reps=200):
    x = np.asarray(molecule, dtype=float)
    out = np.zeros_like(x)
    for _ in range(reps):
        out += np.sin(x) * np.cos(x) + np.sqrt(np.abs(x) + 1.0)
    return pd.Series(out, index=molecule)


def main():
    idx = pd.RangeIndex(200_000)
    workers = min(4, os.cpu_count() or 1)
    serial = best_of(lambda: mp_pandas_obj(heavy, idx, num_threads=1), repeats=1)
    par = best_of(lambda: mp_pandas_obj(heavy, idx, num_threads=workers), repeats=1)
    print(f"cores={os.cpu_count()}, workers={workers}: serial {serial:.3f} s, "
          f"parallel {par:.3f} s, speedup {serial / par:.2f}x")


if __name__ == "__main__":
    main()
