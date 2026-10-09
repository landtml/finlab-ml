"""CPCV purge and embargo: vectorised train mask vs a per-observation Python loop.

Run: PYTHONPATH=src python3 benchmarks/bench_cv.py
"""
import numpy as np
import pandas as pd

from _timing import best_of
from finlab.cv import CombinatorialPurgedCV, make_t1


def naive_split(n, t1_pos, n_groups, combo, embargo):
    """Reference: test a training observation against every test observation in Python."""
    groups = np.arange(n) // (n // n_groups)
    groups[groups >= n_groups] = n_groups - 1
    test = [i for i in range(n) if groups[i] in combo]
    train = []
    for i in range(n):
        if groups[i] in combo:
            continue
        bad = False
        for j in test:
            if t1_pos[i] >= j and i <= t1_pos[j]:  # interval overlap
                bad = True
                break
        if not bad and embargo:
            # forward embargo after each test block
            for j in test:
                if 0 < i - j <= embargo:
                    bad = True
                    break
        if not bad:
            train.append(i)
    return np.array(train)


def main():
    n, groups, k, horizon = 600, 6, 2, 10
    idx = pd.RangeIndex(n)
    t1 = make_t1(idx, horizon)
    cv = CombinatorialPurgedCV(n_groups=groups, n_test_groups=k, embargo_pct=0.01, t1=t1)
    list(cv.split(idx))  # warm-up
    fast = best_of(lambda: list(cv.split(idx)))
    t1_pos = np.minimum(np.arange(n) + horizon, n - 1)
    combos = [(0, 1), (2, 5), (3, 4)]
    slow = best_of(lambda: [naive_split(n, t1_pos, groups, c, int(np.ceil(0.01 * n)))
                            for c in combos], repeats=1)
    per_fast = fast / cv.get_n_splits()
    print(f"full CPCV split ({cv.get_n_splits()} splits, n={n}): {fast:.4f} s, "
          f"{per_fast * 1e3:.2f} ms per split")
    print(f"naive per-observation purge, 3 splits: {slow:.4f} s "
          f"(~{slow / 3 * cv.get_n_splits():.2f} s extrapolated for all splits)")
    print(f"speedup per split: {slow / 3 / per_fast:.0f}x")


if __name__ == "__main__":
    main()
