"""Tests for finlab.ensemble (AFML ch. 6)."""
import numpy as np
import pandas as pd
import pytest

from finlab.cv import make_t1
from finlab.ensemble import SequentialBootstrapBagging
from finlab.weights import average_uniqueness, num_co_events


class NearestCentroid:
    """Two-class nearest-centroid classifier with predict_proba."""

    def fit(self, X, y, sample_weight=None):
        self.classes_ = np.unique(y)
        self.c_ = np.stack([X[y == c].mean(axis=0) for c in self.classes_])
        return self

    def predict_proba(self, X):
        d = ((X[:, None, :] - self.c_[None]) ** 2).sum(axis=2)
        s = np.exp(-d)
        return s / s.sum(axis=1, keepdims=True)

    def predict(self, X):
        return self.classes_[self.predict_proba(X).argmax(axis=1)]


class NoProba:
    """Same classifier without predict_proba, to exercise majority voting."""

    def fit(self, X, y, sample_weight=None):
        self._inner = NearestCentroid().fit(X, y)
        self.classes_ = self._inner.classes_
        return self

    def predict(self, X):
        return self._inner.predict(X)


def _data(n=200, seed=0, horizon=10):
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2020-01-01", periods=n, freq="D")
    y = rng.integers(0, 2, n)
    X = rng.normal(size=(n, 2)) + np.where(y[:, None] == 1, 1.5, -1.5)
    return X, y, make_t1(idx, horizon)


def test_bagging_learns_separable_signal():
    X, y, t1 = _data(300, seed=1)
    bag = SequentialBootstrapBagging(NearestCentroid, n_estimators=15, seed=3).fit(X, y, t1=t1)
    assert (bag.predict(X) == y).mean() > 0.85
    proba = bag.predict_proba(X)
    np.testing.assert_allclose(proba.sum(axis=1), 1.0)


def test_same_seed_gives_identical_bags():
    X, y, t1 = _data(120, seed=2)
    a = SequentialBootstrapBagging(NearestCentroid, 5, seed=9).fit(X, y, t1=t1)
    b = SequentialBootstrapBagging(NearestCentroid, 5, seed=9).fit(X, y, t1=t1)
    for da, db in zip(a.draws_, b.draws_):
        np.testing.assert_array_equal(da, db)


def test_sequential_bags_have_higher_average_uniqueness_than_iid():
    """Each bag's mean per-draw average uniqueness, the quantity the book optimises."""
    X, y, t1 = _data(200, seed=4, horizon=20)
    seq = SequentialBootstrapBagging(NearestCentroid, 20, sequential=True, seed=5).fit(X, y, t1=t1)
    iid = SequentialBootstrapBagging(NearestCentroid, 20, sequential=False, seed=5).fit(X, y, t1=t1)
    u = average_uniqueness(t1.index, t1, num_co_events(t1.index, t1)).to_numpy()

    def bag_quality(draws):
        return np.mean([u[d].mean() for d in draws])

    assert bag_quality(seq.draws_) > bag_quality(iid.draws_)


def test_majority_vote_when_no_predict_proba():
    X, y, t1 = _data(150, seed=6)
    bag = SequentialBootstrapBagging(NoProba, 7, seed=1).fit(X, y, t1=t1)
    assert bag.predict(X).shape == (150,)


def test_sequential_requires_t1():
    X, y, _ = _data(40)
    with pytest.raises(ValueError):
        SequentialBootstrapBagging(NearestCentroid, 3).fit(X, y)
