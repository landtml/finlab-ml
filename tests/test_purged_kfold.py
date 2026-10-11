"""Direct tests for PurgedKFold (single-path purged, embargoed K-fold)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from finlab.cv import PurgedKFold, make_t1


def _frame(n: int) -> pd.DataFrame:
    idx = pd.bdate_range("2015-01-01", periods=n)
    rng = np.random.default_rng(0)
    return pd.DataFrame({"f": rng.standard_normal(n)}, index=idx)


@pytest.mark.parametrize("n_splits,horizon,embargo_pct", [(5, 5, 0.02), (4, 21, 0.05), (6, 1, 0.01)])
def test_no_test_index_inside_training_purge_window(n_splits, horizon, embargo_pct):
    X = _frame(300)
    t1 = make_t1(X.index, horizon)
    # Position of each sample's event end, from the exact timestamps in t1.
    end_pos = X.index.get_indexer(t1.values)
    assert (end_pos >= 0).all()
    cv = PurgedKFold(n_splits=n_splits, t1=t1, embargo_pct=embargo_pct)

    for train_idx, test_idx in cv.split(X):
        for i in train_idx:
            # Purge window of training sample i is [i, end_pos[i]].
            inside = test_idx[(test_idx >= i) & (test_idx <= end_pos[i])]
            assert inside.size == 0, f"test index {inside[0]} inside purge window of train {i}"


@pytest.mark.parametrize("n_splits,embargo_pct", [(5, 0.0), (5, 0.03), (3, 0.1)])
def test_train_and_test_never_overlap(n_splits, embargo_pct):
    X = _frame(240)
    cv = PurgedKFold(n_splits=n_splits, t1=make_t1(X.index, 7), embargo_pct=embargo_pct)
    for train_idx, test_idx in cv.split(X):
        assert np.intersect1d(train_idx, test_idx).size == 0
        assert len(np.unique(train_idx)) == len(train_idx)
        assert len(np.unique(test_idx)) == len(test_idx)


@pytest.mark.parametrize("n_splits", [2, 3, 5, 8])
def test_number_of_folds_matches_n_splits(n_splits):
    X = _frame(200)
    cv = PurgedKFold(n_splits=n_splits, t1=make_t1(X.index, 4), embargo_pct=0.02)
    assert cv.get_n_splits() == n_splits
    folds = list(cv.split(X))
    assert len(folds) == n_splits
    # Each observation is in exactly one test fold: the folds tile the timeline.
    all_test = np.concatenate([te for _, te in folds])
    assert np.array_equal(np.sort(all_test), np.arange(len(X)))
