"""Tests for finlab.tuning (AFML ch. 9)."""
import numpy as np
import pandas as pd
import pytest

from finlab.cv import PurgedKFold, make_t1
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
    # Both thresholds lie above every x, so they predict all zeros and score identically.
    # Identical candidates would not show which one wins, so the order is swapped to check it.
    X, y = _data()
    cv = PurgedKFold(n_splits=3)
    res = grid_search(Threshold, {"threshold": [100.0, 200.0]}, X, y, cv)
    assert res.results["mean_score"].nunique() == 1
    assert res.best_params == {"threshold": 100.0}
    res = grid_search(Threshold, {"threshold": [200.0, 100.0]}, X, y, cv)
    assert res.best_params == {"threshold": 200.0}


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


def _nan_when_all_zero(y_true, y_pred, sample_weight=None):
    # A candidate that predicts only zeros scores NaN, as a zero-weight test fold would.
    if np.asarray(y_pred).sum() == 0:
        return float("nan")
    return float((np.asarray(y_true) == np.asarray(y_pred)).mean())


def test_nan_first_candidate_loses_to_finite_later_candidate():
    X, y = _data()
    cv = PurgedKFold(n_splits=3)
    res = grid_search(Threshold, {"threshold": [100.0, 0.5]}, X, y, cv,
                      scoring=_nan_when_all_zero)
    assert np.isnan(res.results["mean_score"].iloc[0])
    assert res.best_params == {"threshold": 0.5}
    assert res.best_score == pytest.approx(1.0)


def test_all_nan_grid_returns_first_candidate_without_raising():
    X, y = _data()
    cv = PurgedKFold(n_splits=3)
    res = grid_search(Threshold, {"threshold": [100.0, 200.0]}, X, y, cv,
                      scoring=_nan_when_all_zero)
    assert res.results["mean_score"].isna().all()
    assert res.best_params == {"threshold": 100.0}
    assert np.isnan(res.best_score)


def test_all_finite_grid_ranking_unchanged():
    X, y = _data()
    cv = PurgedKFold(n_splits=5)
    res = grid_search(Threshold, {"threshold": [2.0, -1.0, 0.5, 0.0]}, X, y, cv)
    assert res.best_params == {"threshold": 0.5}
    means = res.results["mean_score"].to_numpy()
    assert res.best_score == means.max()
def _datetime_frame(n=300, seed=0):
    """The same data as ``_data``, carried on a DatetimeIndex."""
    x, y = _data(n, seed)
    idx = pd.date_range("2020-01-01", periods=n, freq="D")
    return pd.DataFrame(x, index=idx, columns=["x"]), pd.Series(y, index=idx, name="y")


def test_grid_search_datetime_index_matches_numpy_index():
    # A datetime-indexed t1 must resolve against the DataFrame's own index. Splitting
    # a numpy copy of X would give a RangeIndex and fail to compare with the dates.
    n, horizon = 300, 5
    X_np, y_np = _data(n)
    X_df, y_df = _datetime_frame(n)
    grid = {"threshold": [-1.0, 0.0, 0.5, 2.0]}
    res_dt = grid_search(
        Threshold, grid, X_df, y_df,
        PurgedKFold(n_splits=5, t1=make_t1(X_df.index, horizon)),
    )
    res_np = grid_search(
        Threshold, grid, X_np, y_np,
        PurgedKFold(n_splits=5, t1=make_t1(pd.RangeIndex(n), horizon)),
    )
    assert res_dt.best_params == res_np.best_params
    assert res_dt.best_score == pytest.approx(res_np.best_score)
    np.testing.assert_allclose(
        res_dt.results["mean_score"].to_numpy(), res_np.results["mean_score"].to_numpy()
    )


def test_randomized_search_datetime_index_runs():
    X_df, y_df = _datetime_frame(200)
    cv = PurgedKFold(n_splits=4, t1=make_t1(X_df.index, 3))
    res = randomized_search(Threshold, {"threshold": [-1.0, 0.0, 0.5]}, 4, X_df, y_df, cv, seed=1)
    assert len(res.results) == 4
    assert res.best_score >= 0.0


def test_cv_score_dataframe_range_index_matches_numpy():
    # A plain RangeIndex DataFrame must score exactly like the numpy array it wraps.
    X_np, y_np = _data(120)
    X_df = pd.DataFrame(X_np)
    cv = PurgedKFold(n_splits=4)
    assert cv_score(Threshold, {"threshold": 0.5}, X_df, y_np, cv) == pytest.approx(
        cv_score(Threshold, {"threshold": 0.5}, X_np, y_np, cv)
    )
