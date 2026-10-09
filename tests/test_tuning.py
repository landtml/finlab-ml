"""Tests for finlab.tuning (AFML ch. 9)."""
import numpy as np
import pytest

from finlab.cv import PurgedKFold
from finlab.tuning import cv_score, grid_search, randomized_search


class Threshold:
    """Predicts 1 when x > threshold; tuned by the search."""

    def __init__(self, threshold=0.0):
        self.threshold = threshold

    def fit(self, X, y, sample_weight=None):
        return self

    def predict(self, X):
        return (X[:, 0] > self.threshold).astype(int)


def _data(n=300, seed=0):
    rng = np.random.default_rng(seed)
    x = rng.normal(size=(n, 1))
    y = (x[:, 0] > 0.5).astype(int)
    return x, y


def test_grid_search_recovers_best_threshold():
    X, y = _data()
    cv = PurgedKFold(n_splits=5)
    res = grid_search(Threshold, {"threshold": [-1.0, 0.0, 0.5, 2.0]}, X, y, cv)
    assert res.best_params == {"threshold": 0.5}
    assert res.best_score == pytest.approx(1.0)
    assert len(res.results) == 4


def test_grid_search_ties_prefer_first_candidate():
    X, y = _data()
    cv = PurgedKFold(n_splits=3)
    res = grid_search(Threshold, {"threshold": [0.5, 0.5]}, X, y, cv)
    assert res.best_params == {"threshold": 0.5}


def test_randomized_search_is_reproducible():
    X, y = _data()
    cv = PurgedKFold(n_splits=4)
    grid = {"threshold": [-1.0, 0.0, 0.5, 2.0]}
    a = randomized_search(Threshold, grid, 6, X, y, cv, seed=7)
    b = randomized_search(Threshold, grid, 6, X, y, cv, seed=7)
    assert a.best_params == b.best_params
    assert a.results.equals(b.results)


def test_cv_score_matches_manual_loop():
    X, y = _data(120)
    cv = PurgedKFold(n_splits=4)
    mean, std = cv_score(Threshold, {"threshold": 0.0}, X, y, cv)
    manual = []
    for tr, te in cv.split(X):
        m = Threshold(0.0).fit(X[tr], y[tr])
        manual.append((m.predict(X[te]) == y[te]).mean())
    assert mean == pytest.approx(np.mean(manual))
    assert std == pytest.approx(np.std(manual))


def test_empty_grid_rejected():
    X, y = _data(60)
    with pytest.raises(ValueError):
        grid_search(Threshold, {}, X, y, PurgedKFold(3))
