"""Benchmark: wall time of quantowl.tuning.grid_search over a small grid.

Not run by CI. Usage:
    PYTHONPATH=src python3 benchmarks/bench_tuning.py

The estimator is a nearest-centroid classifier written in NumPy, so no scikit-learn is
needed. The script has no reference implementation, so it prints times and no speedup.
Each time covers the search loop, the purged splits from quantowl.cv, the model's fit and
predict, and the accuracy scorer, all for this one cheap model. A heavier model changes
the time per fit and predict, so these numbers do not carry over to it.
Each timing is the best of 3 runs; the search is warmed up once before timing.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from _timing import best_of
from quantowl.cv import PurgedKFold, make_t1
from quantowl.tuning import grid_search

SHRINK_GRID = {"shrink": [0.0, 0.25, 0.5, 0.9]}
N_SPLITS = 5
HORIZON = 5
ROW_COUNTS = [1_000, 10_000, 50_000]


class NearestCentroid:
    """One mean per class; a row gets the class of the nearest mean.

    ``shrink`` in [0, 1) multiplies the means by ``1 - shrink``.
    """

    def __init__(self, shrink: float = 0.0):
        self.shrink = shrink

    def fit(self, X, y, sample_weight=None):
        self.classes_ = np.unique(y)
        means = np.vstack([X[y == c].mean(axis=0) for c in self.classes_])
        self.centroids_ = means * (1.0 - self.shrink)
        return self

    def predict(self, X):
        d = ((X[:, None, :] - self.centroids_[None, :, :]) ** 2).sum(axis=-1)
        return self.classes_[np.argmin(d, axis=1)]


def make_data(n_rows: int, seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n_rows, 5))
    y = (X[:, 0] + 0.5 * rng.normal(size=n_rows) > 0).astype(int)
    return X, y


def make_cv(n_rows: int) -> PurgedKFold:
    t1 = make_t1(pd.RangeIndex(n_rows), HORIZON)
    return PurgedKFold(n_splits=N_SPLITS, t1=t1, embargo_pct=0.01)


def main() -> None:
    X, y = make_data(500)
    grid_search(NearestCentroid, SHRINK_GRID, X, y, make_cv(500))  # warm-up, not timed

    n_candidates = len(SHRINK_GRID["shrink"])
    n_candidate_folds = n_candidates * N_SPLITS
    header = f"{'rows':>8} {'candidates':>11} {'folds':>6} {'grid_search (s)':>16}"
    print(f"{header} {'per candidate-fold (ms)':>24}")
    for n_rows in ROW_COUNTS:
        X, y = make_data(n_rows)
        cv = make_cv(n_rows)
        t = best_of(lambda: grid_search(NearestCentroid, SHRINK_GRID, X, y, cv))
        per_fold_ms = 1000 * t / n_candidate_folds
        print(f"{n_rows:>8} {n_candidates:>11} {N_SPLITS:>6} {t:>16.4f} {per_fold_ms:>24.3f}")


if __name__ == "__main__":
    main()
