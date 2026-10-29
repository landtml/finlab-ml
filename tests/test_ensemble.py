"""Tests for finlab.ensemble (AFML ch. 6)."""
import numpy as np
import pandas as pd
import pytest

from finlab.cv import PurgedKFold, make_t1
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
    assert len(a.draws_) == len(b.draws_) == 5
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
    pred = bag.predict(X)
    assert pred.shape == (150,)
    assert set(np.unique(pred)) <= {0, 1}
    # Exact majority of the 7 fitted members (odd count, so no ties): class 1 wins with 4+ votes.
    votes = np.stack([e.predict(X) for e in bag.estimators_])
    expected = np.where((votes == 1).sum(axis=0) * 2 > len(bag.estimators_), 1, 0)
    np.testing.assert_array_equal(pred, expected)
    # Same separable data as the predict_proba test, so the vote should be accurate too.
    assert (pred == y).mean() > 0.85


def test_sequential_requires_t1():
    X, y, _ = _data(40)
    with pytest.raises(ValueError):
        SequentialBootstrapBagging(NearestCentroid, 3).fit(X, y)


@pytest.mark.parametrize("length", [10, 400])
def test_sample_weight_length_must_match_rows(length):
    X, y, t1 = _data()
    bag = SequentialBootstrapBagging(NearestCentroid, n_estimators=2)
    with pytest.raises(ValueError, match="sample_weight.*200"):
        bag.fit(X, y, t1=t1, sample_weight=np.ones(length))


def test_pandas_y_with_mismatched_index_is_rejected():
    X, y, t1 = _data(200, seed=1)
    Xdf = pd.DataFrame(X)
    y_reversed = pd.Series(y, index=Xdf.index[::-1])
    bag = SequentialBootstrapBagging(NearestCentroid, n_estimators=2)
    with pytest.raises(ValueError, match="index"):
        bag.fit(Xdf, y_reversed, t1=t1)


def test_pandas_sample_weight_with_mismatched_index_is_rejected():
    X, y, t1 = _data(200, seed=1)
    Xdf = pd.DataFrame(X)
    w_reversed = pd.Series(np.ones(200), index=Xdf.index[::-1])
    bag = SequentialBootstrapBagging(NearestCentroid, n_estimators=2)
    with pytest.raises(ValueError, match="index"):
        bag.fit(Xdf, y, t1=t1, sample_weight=w_reversed)


def test_aligned_pandas_y_and_weight_still_fit():
    X, y, t1 = _data(300, seed=1)
    Xdf = pd.DataFrame(X)
    bag = SequentialBootstrapBagging(NearestCentroid, n_estimators=15, seed=3).fit(
        Xdf,
        pd.Series(y, index=Xdf.index),
        t1=t1,
        sample_weight=pd.Series(np.ones(len(y)), index=Xdf.index),
    )
    assert (bag.predict(X) == y).mean() > 0.85


def test_failed_fit_leaves_no_partial_ensemble():
    X, y, t1 = _data()
    made = []

    def factory():
        made.append(1)
        if len(made) == 2:
            raise RuntimeError("bag 2 failed")
        return NearestCentroid()

    bag = SequentialBootstrapBagging(factory, n_estimators=4)
    with pytest.raises(RuntimeError, match="bag 2 failed"):
        bag.fit(X, y, t1=t1)
    assert not hasattr(bag, "classes_")
    assert not hasattr(bag, "estimators_")
    with pytest.raises(RuntimeError, match="not fitted"):
        bag.predict(X)


def test_pandas_y_same_instants_in_another_timezone_is_accepted():
    X, y, _ = _data(200, seed=1)
    utc = pd.date_range("2020-01-01", periods=200, freq="D", tz="UTC")
    t1 = make_t1(utc, 10)
    Xdf = pd.DataFrame(X, index=utc)
    y_taipei = pd.Series(y, index=utc.tz_convert("Asia/Taipei"))
    bag = SequentialBootstrapBagging(NearestCentroid, n_estimators=3, seed=2).fit(Xdf, y_taipei, t1=t1)
    assert len(bag.estimators_) == 3


def test_pandas_y_shifted_instants_across_timezones_is_rejected():
    X, y, _ = _data(200, seed=1)
    utc = pd.date_range("2020-01-01", periods=200, freq="D", tz="UTC")
    t1 = make_t1(utc, 10)
    Xdf = pd.DataFrame(X, index=utc)
    y_shifted = pd.Series(y, index=(utc + pd.Timedelta(days=1)).tz_convert("Asia/Taipei"))
    bag = SequentialBootstrapBagging(NearestCentroid, n_estimators=2)
    with pytest.raises(ValueError, match="index"):
        bag.fit(Xdf, y_shifted, t1=t1)


class _ProbaNoClasses:
    """Duck-typed estimator: predict_proba columns follow the bag's labels, no classes_ attribute."""

    def fit(self, X, y, sample_weight=None):
        self.c = np.unique(y)
        self.p = np.array([np.mean(y == c) for c in self.c])
        return self

    def predict(self, X):
        return np.repeat(self.c[np.argmax(self.p)], len(X))

    def predict_proba(self, X):
        return np.tile(self.p, (len(X), 1))


class _ProbaWithClasses(_ProbaNoClasses):
    def fit(self, X, y, sample_weight=None):
        super().fit(X, y, sample_weight)
        self.classes_ = self.c
        return self


def test_bags_missing_a_class_without_classes_attr_fail_at_fit_with_a_named_error() -> None:
    # A bag that missed class 2 returned 2 columns and predict_proba failed with a raw broadcast error
    # far from the cause. The mismatch is now named at fit time.
    from finlab.ensemble import SequentialBootstrapBagging

    X = np.arange(40.0).reshape(-1, 1)
    y = np.array([0] * 38 + [1, 2])
    with pytest.raises(ValueError, match="classes_"):
        SequentialBootstrapBagging(_ProbaNoClasses, n_estimators=10, sequential=False).fit(X, y)


def test_bags_with_classes_attr_missing_a_class_still_predict() -> None:
    from finlab.ensemble import SequentialBootstrapBagging

    X = np.arange(40.0).reshape(-1, 1)
    y = np.array([0] * 38 + [1, 2])
    m = SequentialBootstrapBagging(_ProbaWithClasses, n_estimators=10, sequential=False).fit(X, y)
    p = m.predict_proba(X)
    assert p.shape == (40, 3)
    np.testing.assert_allclose(p.sum(axis=1), 1.0)


def test_sequential_bagging_rejects_t1_with_mismatched_index() -> None:
    # t1 was only length-checked, so its own labels drove the uniqueness weights.
    from finlab.cv import make_t1
    from finlab.ensemble import SequentialBootstrapBagging

    class Majority:
        def fit(self, X, y, sample_weight=None):
            self.label = pd.Series(y).mode()[0]
            return self

        def predict(self, X):
            return [self.label] * len(X)

    idx = pd.date_range("2020-01-01", periods=40, freq="D")
    X = pd.DataFrame({"a": range(40)}, index=idx)
    y = pd.Series([0, 1] * 20, index=idx)
    t1 = pd.Series(make_t1(idx, 3).to_numpy(), index=pd.date_range("2000", periods=40))
    with pytest.raises(ValueError, match="t1 index"):
        SequentialBootstrapBagging(Majority, n_estimators=1, seed=1).fit(X, y, t1=t1)


class _Majority:
    def fit(self, X, y, sample_weight=None):
        return self

    def predict(self, X):
        return np.zeros(len(X), int)


def test_ensemble_refuses_fractional_max_samples() -> None:
    # max_samples=2.7 was truncated to 2 with no message. The integer rule of make_t1 applies here
    # too: an integral value such as 2.0 is accepted, 2.7 is refused.
    from finlab.ensemble import SequentialBootstrapBagging

    idx = pd.date_range("2020", periods=12)
    X = np.arange(24.0).reshape(12, 2)
    y = np.arange(12) % 2
    t1 = pd.Series(idx + pd.Timedelta(days=2), index=idx)
    with pytest.raises(ValueError, match="max_samples must be an integer"):
        SequentialBootstrapBagging(_Majority, n_estimators=1, max_samples=2.7).fit(X, y, t1=t1)
    m = SequentialBootstrapBagging(_Majority, n_estimators=1, max_samples=2.0).fit(X, y, t1=t1)
    assert m.draws_[0].shape[0] == 2



def test_ensemble_refuses_t1_that_is_not_a_series() -> None:
    # An ndarray t1 reached t1.index and failed with an AttributeError.
    from finlab.ensemble import SequentialBootstrapBagging

    idx = pd.date_range("2020", periods=12)
    with pytest.raises(ValueError, match="t1 must be a pandas Series"):
        SequentialBootstrapBagging(_Majority, n_estimators=1).fit(
            np.arange(24.0).reshape(12, 2), np.arange(12) % 2, t1=idx.values
        )


def test_sequential_bagging_checks_t1_against_a_shifted_range_index() -> None:
    # The t1 check skipped any pd.RangeIndex on X, so a RangeIndex starting at 5 accepted t1 labelled by
    # dates. Only the default 0..n-1 index is unlabelled.
    from finlab.ensemble import SequentialBootstrapBagging

    class Const:
        def fit(self, X, y, sample_weight=None):
            self.classes_ = np.unique(y)
            return self

        def predict(self, X):
            return np.zeros(len(X), dtype=int)

    ix = pd.date_range("2021-01-01", periods=60, freq="D")
    X = pd.DataFrame(np.random.default_rng(0).normal(size=(60, 2)), index=pd.RangeIndex(5, 65))
    t1 = pd.Series(ix + pd.Timedelta(days=3), index=ix)
    with pytest.raises(ValueError, match="t1 index does not match"):
        SequentialBootstrapBagging(Const, 1, seed=1).fit(X, pd.Series(0, index=X.index), t1=t1)


def test_sequential_bagging_refuses_string_max_samples() -> None:
    # max_samples='7' was read as 7 through float(), while make_t1 refuses strings.
    from finlab.ensemble import SequentialBootstrapBagging

    class Const:
        def fit(self, X, y, sample_weight=None):
            self.classes_ = np.unique(y)
            return self

        def predict(self, X):
            return np.zeros(len(X), dtype=int)

    ix = pd.date_range("2021-01-01", periods=8, freq="D")
    X = pd.DataFrame(np.random.default_rng(0).normal(size=(8, 2)), index=ix)
    t1 = pd.Series(ix + pd.Timedelta(days=3), index=ix)
    with pytest.raises(ValueError, match="max_samples must be an integer"):
        SequentialBootstrapBagging(Const, 1, seed=0, max_samples="7").fit(X, pd.Series(0, index=ix), t1=t1)


def test_sequential_bagging_refuses_nat_end_times() -> None:
    # A NaT end time in t1 was read as "spans to the end of the grid", so the event took about six
    # bars instead of four. It is refused, as average_active_signals refuses it.
    from finlab.ensemble import SequentialBootstrapBagging

    class Const:
        def fit(self, X, y, sample_weight=None):
            self.classes_ = np.unique(y)
            return self

        def predict(self, X):
            return np.zeros(len(X), dtype=int)

    ix = pd.date_range("2021-01-01", periods=8, freq="D")
    X = pd.DataFrame(np.random.default_rng(0).normal(size=(8, 2)), index=ix)
    t1 = pd.Series(ix + pd.Timedelta(days=3), index=ix)
    t1.iloc[2] = pd.NaT
    with pytest.raises(ValueError, match="NaT"):
        SequentialBootstrapBagging(Const, 1, seed=0).fit(X, pd.Series(0, index=ix), t1=t1)


class _ConstLabel:
    def fit(self, X, y, sample_weight=None):
        self.classes_ = np.unique(y)
        return self

    def predict(self, X):
        return np.zeros(len(X), dtype=int)


def test_sequential_bagging_refuses_series_x_with_shifted_y_index() -> None:
    # The y and sample_weight index checks ran only for a DataFrame X, so a Series X with y shifted by
    # one day was used by position. Tuning and importance refuse it.
    from finlab.ensemble import SequentialBootstrapBagging

    ix = pd.date_range("2021-01-01", periods=40)
    xs = pd.Series(np.random.default_rng(0).normal(size=40), index=ix)
    ysh = pd.Series(np.arange(40) % 2, index=ix + pd.Timedelta(days=1))
    with pytest.raises(ValueError, match="y index does not match"):
        SequentialBootstrapBagging(_ConstLabel, n_estimators=2, sequential=False).fit(xs, ysh)
    # Aligned labels on a Series X still fit.
    yal = pd.Series(np.arange(40) % 2, index=ix)
    SequentialBootstrapBagging(_ConstLabel, n_estimators=2, sequential=False).fit(xs, yal)


def test_sequential_bagging_refuses_labelled_range_t1_against_default_x() -> None:
    # The t1 check was skipped for any default X, so t1 labelled 5..44 was matched to rows 0..39 by
    # position. A labelled t1 is checked against X, as a labelled X is.
    from finlab.ensemble import SequentialBootstrapBagging

    X = pd.DataFrame(np.random.default_rng(0).normal(size=(40, 2)))
    y = pd.Series(np.arange(40) % 2)
    t1 = pd.Series(np.minimum(np.arange(40) + 3, 39) + 5, index=pd.RangeIndex(5, 45))
    with pytest.raises(ValueError, match="t1 index does not match"):
        SequentialBootstrapBagging(_ConstLabel, n_estimators=2, seed=0).fit(X, y, t1=t1)
    # A date-labelled t1 against a default X is still matched by position.
    ix = pd.date_range("2021-01-01", periods=40)
    t1_dates = pd.Series(ix[np.minimum(np.arange(40) + 3, 39)], index=ix)
    SequentialBootstrapBagging(_ConstLabel, n_estimators=2, seed=0).fit(X, y, t1=t1_dates)


def test_nat_t1_refusal_is_documented_for_get_events_and_bagging() -> None:
    # get_events leaves t1 NaT for unresolved events, and bagging refuses NaT. The refusal is
    # documented on both functions, so a caller knows to drop those events first.
    from finlab.ensemble import SequentialBootstrapBagging
    from finlab.labeling import get_events

    assert "NaT" in (SequentialBootstrapBagging.fit.__doc__ or "") + (SequentialBootstrapBagging.__doc__ or "")
    assert "NaT" in (get_events.__doc__ or "")


def test_bagging_accepts_unlabelled_y_against_date_x_as_tuning_does() -> None:
    # Tuning matches an unlabelled y to a date-indexed X by position, and the ensemble used to refuse
    # it. The ensemble now agrees; a reversed labelled y is still refused.
    from finlab.ensemble import SequentialBootstrapBagging

    ix = pd.date_range("2021-01-01", periods=40)
    X = pd.DataFrame(np.random.default_rng(0).normal(size=(40, 2)), index=ix)
    yu = pd.Series(np.arange(40) % 2)
    SequentialBootstrapBagging(_ConstLabel, n_estimators=2, sequential=False).fit(X, yu)
    with pytest.raises(ValueError, match="y index does not match"):
        SequentialBootstrapBagging(_ConstLabel, n_estimators=2, sequential=False).fit(
            X, pd.Series(np.arange(40) % 2, index=ix[::-1])
        )


def test_t1_int64_index_labels_follow_the_value_rule() -> None:
    # An Int64 Index with labels 0..n-1 is the same unlabelled index as a RangeIndex, so it is matched
    # by position to a default X. Labels 5..44 are named and refused for both index classes, instead
    # of the class deciding the answer.
    from finlab.ensemble import SequentialBootstrapBagging

    X = pd.DataFrame(np.random.default_rng(0).normal(size=(40, 2)))
    y = pd.Series(np.arange(40) % 2)
    t1_plain = pd.Series(np.minimum(np.arange(40) + 3, 39), index=pd.Index(np.arange(40), dtype="int64"))
    SequentialBootstrapBagging(_ConstLabel, n_estimators=2, seed=0).fit(X, y, t1=t1_plain)
    shifted = pd.Index(np.arange(5, 45), dtype="int64")
    t1_shift = pd.Series(np.minimum(np.arange(40) + 3, 39) + 5, index=shifted)
    with pytest.raises(ValueError, match="t1 index does not match"):
        SequentialBootstrapBagging(_ConstLabel, n_estimators=2, seed=0).fit(X, y, t1=t1_shift)
    t1_range = pd.Series(np.minimum(np.arange(40) + 3, 39) + 5, index=pd.RangeIndex(5, 45))
    with pytest.raises(ValueError, match="t1 index does not match"):
        SequentialBootstrapBagging(_ConstLabel, n_estimators=2, seed=0).fit(X, y, t1=t1_range)


def test_predict_aligns_reordered_columns_by_name() -> None:
    X, y, t1 = _data(200, seed=1)
    df = pd.DataFrame(np.column_stack([X, X[:, 0] - X[:, 1]]), columns=list("abc"), index=t1.index)
    bag = SequentialBootstrapBagging(NearestCentroid, n_estimators=5, seed=3).fit(df, y, t1=t1)
    flipped = df[["c", "b", "a"]]
    np.testing.assert_array_equal(bag.predict(flipped), bag.predict(df))
    np.testing.assert_allclose(bag.predict_proba(flipped), bag.predict_proba(df))
    np.testing.assert_array_equal(bag.predict(df.to_numpy()), bag.predict(df))
    with pytest.raises(ValueError, match="missing.*'c'"):
        bag.predict(df[["a", "b"]])
    with pytest.raises(ValueError, match="extra.*'z'"):
        bag.predict_proba(df.rename(columns={"c": "z"}))


def _fit_bag(X, y, t1):
    return SequentialBootstrapBagging(NearestCentroid, n_estimators=2, seed=0).fit(X, y, t1=t1)


def test_t1_with_a_default_index_is_refused_by_name() -> None:
    X, y, t1 = _data(60, seed=3)
    with pytest.raises(ValueError, match="indexed by event start times"):
        _fit_bag(X, y, t1.reset_index(drop=True))
    with pytest.raises(ValueError, match="indexed by event start times"):
        _fit_bag(pd.DataFrame(X), y, t1.reset_index(drop=True))


def test_t1_ending_before_its_start_is_refused_and_equal_is_allowed() -> None:
    X, y, t1 = _data(60, seed=3)
    bad = t1.copy()
    bad.iloc[5] = bad.index[5] - pd.Timedelta(days=3)
    with pytest.raises(ValueError, match="precedes.*2020-01-06"):
        _fit_bag(X, y, bad)
    same = t1.copy()
    same.iloc[5] = same.index[5]
    _fit_bag(X, y, same)


def test_duplicate_or_unsorted_event_starts_name_the_start_times() -> None:
    X, y, t1 = _data(60, seed=3)
    dup = t1.copy()
    dup.index = dup.index[:5].append(dup.index[4:5]).append(dup.index[6:])
    with pytest.raises(ValueError, match="duplicate or unsorted.*sequential=False"):
        _fit_bag(X, y, dup)
    shuffled = t1.iloc[::-1]
    with pytest.raises(ValueError, match="duplicate or unsorted.*sequential=False"):
        _fit_bag(X[::-1], y[::-1], shuffled)


def _named(X):
    return pd.DataFrame(X, columns=["a", "b"])


def test_refused_refit_keeps_the_earlier_column_names() -> None:
    # columns_ used to be set before the row-count check, so a refused refit left the new names behind.
    X, y, _ = _data(100, seed=4)
    bag = SequentialBootstrapBagging(NearestCentroid, n_estimators=3, seed=1, sequential=False)
    bag.fit(_named(X), y)
    before = bag.predict_proba(_named(X))
    with pytest.raises(ValueError):
        bag.fit(pd.DataFrame(X, columns=["c", "d"]), y[:-1])
    np.testing.assert_allclose(bag.predict_proba(_named(X)), before)
    with pytest.raises(ValueError, match="do not match"):
        bag.predict_proba(pd.DataFrame(X, columns=["c", "d"]))


def test_named_frame_after_numpy_fit_warns_and_keeps_positional_values() -> None:
    X, y, _ = _data(100, seed=5)
    bag = SequentialBootstrapBagging(NearestCentroid, n_estimators=3, seed=1, sequential=False).fit(X, y)
    expected = bag.predict_proba(X)
    with pytest.warns(UserWarning, match="fitted without them"):
        got = bag.predict_proba(_named(X))
    np.testing.assert_allclose(got, expected)


def test_sample_weight_same_instants_in_another_timezone_is_accepted():
    X, y, _ = _data(200, seed=1)
    utc = pd.date_range("2020-01-01", periods=200, freq="D", tz="UTC")
    Xdf = pd.DataFrame(X, index=utc)
    bag = SequentialBootstrapBagging(NearestCentroid, n_estimators=2, seed=2, sequential=False)
    w = np.ones(200)
    bag.fit(Xdf, y, sample_weight=pd.Series(w, index=utc.tz_convert("Asia/Taipei")))
    assert len(bag.estimators_) == 2
    shifted = pd.Series(w, index=(utc + pd.Timedelta(days=1)).tz_convert("Asia/Taipei"))
    with pytest.raises(ValueError, match="sample_weight index"):
        bag.fit(Xdf, y, sample_weight=shifted)


def test_bagging_refuses_continuous_target():
    X, _, t1 = _data(120, seed=7)
    y = 3.0 + 2.0 * X[:, 0] + 0.1 * np.sin(np.arange(120))
    bag = SequentialBootstrapBagging(NearestCentroid, n_estimators=3, seed=1)
    with pytest.raises(ValueError, match="classification labels only"):
        bag.fit(X, y, t1=t1)
    assert not hasattr(bag, "classes_")
    assert not hasattr(bag, "estimators_")


def test_bagging_accepts_integer_valued_float_labels():
    X, y, t1 = _data(120, seed=8)
    bag = SequentialBootstrapBagging(NearestCentroid, n_estimators=3, seed=1).fit(X, y.astype(float), t1=t1)
    np.testing.assert_array_equal(bag.classes_, np.array([0.0, 1.0]))
    assert set(np.unique(bag.predict(X))) <= {0.0, 1.0}


class _NoSampleWeightArg:
    """Base whose fit has no sample_weight parameter, as KNeighborsClassifier does."""

    def fit(self, X, y):
        self.classes_ = np.unique(y)
        self.n_fit_ = len(y)
        return self

    def predict(self, X):
        return np.full(len(X), self.classes_[0])


def test_bagging_base_without_sample_weight_fits_without_weights():
    X, y, t1 = _data(120, seed=9)
    bag = SequentialBootstrapBagging(_NoSampleWeightArg, n_estimators=3, seed=1, sequential=False)
    bag.fit(X, y)
    assert [e.n_fit_ for e in bag.estimators_] == [120, 120, 120]


def test_bagging_kneighbors_base_fits_without_weights():
    pytest.importorskip("sklearn")
    from sklearn.neighbors import KNeighborsClassifier

    X, y, t1 = _data(120, seed=10)
    bag = SequentialBootstrapBagging(lambda: KNeighborsClassifier(n_neighbors=3), n_estimators=3, seed=1)
    bag.fit(X, y, t1=t1)
    assert (bag.predict(X) == y).mean() > 0.85


class _RecordingBase:
    """Records the feature arrays it is fitted and predicted on."""

    def fit(self, X, y, sample_weight=None):
        self.fit_X_ = X
        self.classes_ = np.unique(y)
        return self

    def predict(self, X):
        self.predict_X_ = X
        return np.full(len(X), self.classes_[0])


@pytest.mark.parametrize("dtype", ["Float64", "Int64"])
def test_bagging_nullable_features_with_na_act_as_nan(dtype):
    n = 60
    rng = np.random.default_rng(3)
    X = pd.DataFrame({"a": pd.array(np.arange(n), dtype=dtype), "b": rng.normal(size=n)})
    X.loc[5, "a"] = pd.NA
    y = np.arange(n) % 2
    bags = []

    def factory():
        est = _RecordingBase()
        bags.append(est)
        return est

    bag = SequentialBootstrapBagging(factory, n_estimators=2, seed=1, sequential=False).fit(X, y)
    for est in bags:
        assert est.fit_X_.dtype.kind == "f"
    bag.predict(X)
    assert bags[0].predict_X_.dtype.kind == "f"
    assert np.isnan(bags[0].predict_X_[5, 0])
    assert np.isnan(bags[0].predict_X_).sum() == 1


def test_bagging_refuses_none_label_by_name():
    X, _, t1 = _data(60, seed=11)
    y = np.array([0, 1, None, 1] * 15, dtype=object)
    calls = []

    def factory():
        calls.append(1)
        return NearestCentroid()

    bag = SequentialBootstrapBagging(factory, n_estimators=2, seed=1)
    with pytest.raises(ValueError, match="None"):
        bag.fit(X, y, t1=t1)
    assert calls == []
    assert not hasattr(bag, "classes_")


class _NaNRowBase(NearestCentroid):
    """Nearest-centroid base whose probabilities are NaN for rows marked by a first value above 1e5."""

    def predict_proba(self, X):
        marked = X[:, 0] > 1e5
        out = np.full((len(X), len(self.classes_)), np.nan)
        out[~marked] = super().predict_proba(X[~marked])
        return out


def test_predict_refuses_rows_with_non_finite_base_probabilities() -> None:
    X, y, _ = _data(n=120)
    m = SequentialBootstrapBagging(_NaNRowBase, n_estimators=5, sequential=False).fit(X, y)
    Xt = X[:4].copy()
    Xt[2, 0] = 1e6
    assert np.isnan(m.predict_proba(Xt)[2]).all()
    with pytest.raises(ValueError, match="non-finite probabilities for 1 of 4 rows"):
        m.predict(Xt)
    assert m.predict(X[:4]).shape == (4,)


class _ConstantProba:
    """Gives every row the same class-1 probability, so each bag's vote is one number."""

    def __init__(self, p1):
        self.p1 = p1

    def fit(self, X, y, sample_weight=None):
        self.classes_ = np.unique(y)
        return self

    def predict_proba(self, X):
        return np.tile([1.0 - self.p1, self.p1], (len(X), 1))


def _factory_from(values):
    it = iter(values)
    return lambda: _ConstantProba(next(it))


def _fixed_draws(monkeypatch, draws):
    it = iter(draws)
    monkeypatch.setattr(
        "finlab.ensemble.sequential_bootstrap", lambda ind, n_samples, seed: np.array(next(it))
    )


def test_oob_hand_case_with_fixed_draws(monkeypatch):
    # Bag draws, chosen so the out-of-bag sets are read off by eye:
    #   row 0 is out of bag only in bag 1; row 1 is drawn by every bag; row 2 is out of bag only in bag 0.
    _fixed_draws(monkeypatch, [[0, 0, 1], [1, 2, 2], [0, 1, 2]])
    X = np.arange(6, dtype=float).reshape(3, 2)
    y = np.array([0, 1, 0])
    t1 = make_t1(pd.date_range("2020-01-01", periods=3, freq="D"), 2)

    with pytest.warns(UserWarning, match="1 of 3 samples"):
        bag = SequentialBootstrapBagging(_factory_from([0.2, 0.6, 0.9]), n_estimators=3, oob_score=True)
        bag.fit(X, y, t1=t1)
    np.testing.assert_allclose(bag.oob_decision_function_, [[0.4, 0.6], [np.nan, np.nan], [0.8, 0.2]])
    # Row 0 is predicted 1 (wrong), row 2 is predicted 0 (right): accuracy 1/2.
    assert bag.oob_score_ == pytest.approx(0.5)

    # Weights 1, 100, 3: the row drawn by every bag carries weight 100 but must not count.
    _fixed_draws(monkeypatch, [[0, 0, 1], [1, 2, 2], [0, 1, 2]])
    with pytest.warns(UserWarning, match="1 of 3 samples"):
        weighted = SequentialBootstrapBagging(_factory_from([0.2, 0.6, 0.9]), n_estimators=3, oob_score=True)
        weighted.fit(X, y, t1=t1, sample_weight=[1.0, 100.0, 3.0])
    # (1 * 0 + 3 * 1) / (1 + 3) = 0.75.
    assert weighted.oob_score_ == pytest.approx(0.75)


def test_oob_with_no_out_of_bag_row_gives_nan_score(monkeypatch):
    _fixed_draws(monkeypatch, [[0, 1, 2], [2, 0, 1]])
    X = np.arange(6, dtype=float).reshape(3, 2)
    y = np.array([0, 1, 0])
    t1 = make_t1(pd.date_range("2020-01-01", periods=3, freq="D"), 2)
    with pytest.warns(UserWarning) as record:
        bag = SequentialBootstrapBagging(_factory_from([0.3, 0.7]), n_estimators=2, oob_score=True)
        bag.fit(X, y, t1=t1)
    assert np.isnan(bag.oob_decision_function_).all()
    assert np.isnan(bag.oob_score_)
    messages = [str(r.message) for r in record]
    assert any("3 of 3 samples" in m for m in messages)
    assert any("oob_score_ is NaN" in m for m in messages)


@pytest.mark.filterwarnings("ignore:.*every bag:UserWarning")
@pytest.mark.parametrize("sequential", [True, False])
def test_oob_decision_function_matches_slow_reference(sequential):
    X, y, t1 = _data(120, seed=15, horizon=10)
    bag = SequentialBootstrapBagging(
        NearestCentroid, n_estimators=8, sequential=sequential, seed=4, oob_score=True
    ).fit(X, y, t1=t1)
    n, k = len(y), len(bag.classes_)
    # Reference written from the formula: for each row, average the bags that did not draw it.
    ref = np.full((n, k), np.nan)
    for i in range(n):
        rows = []
        for est, draws in zip(bag.estimators_, bag.draws_):
            if i in draws:
                continue
            full = np.zeros((1, k))
            full[:, np.searchsorted(bag.classes_, est.classes_)] = est.predict_proba(X[i : i + 1])
            rows.append(full[0])
        if rows:
            ref[i] = np.mean(rows, axis=0)
    assert bag.oob_decision_function_.shape == (n, k)
    np.testing.assert_allclose(bag.oob_decision_function_, ref)
    has = ~np.isnan(ref[:, 0])
    pred = bag.classes_[ref[has].argmax(axis=1)]
    assert bag.oob_score_ == pytest.approx((pred == y[has]).mean())


@pytest.mark.filterwarnings("ignore:.*every bag:UserWarning")
def test_oob_score_leaves_bags_and_predictions_unchanged():
    X, y, t1 = _data(120, seed=13)
    Xt = _data(20, seed=14)[0]
    off = SequentialBootstrapBagging(NearestCentroid, 6, seed=21).fit(X, y, t1=t1)
    on = SequentialBootstrapBagging(NearestCentroid, 6, seed=21, oob_score=True).fit(X, y, t1=t1)
    for a, b in zip(off.draws_, on.draws_):
        np.testing.assert_array_equal(a, b)
    np.testing.assert_array_equal(off.predict_proba(Xt), on.predict_proba(Xt))
    np.testing.assert_array_equal(off.predict(Xt), on.predict(Xt))
    assert not hasattr(off, "oob_score_")
    assert not hasattr(off, "oob_decision_function_")


@pytest.mark.filterwarnings("ignore:.*every bag:UserWarning")
def test_refit_without_oob_drops_the_earlier_oob_attributes():
    X, y, t1 = _data(120, seed=18)
    bag = SequentialBootstrapBagging(NearestCentroid, 4, seed=2, oob_score=True).fit(X, y, t1=t1)
    assert hasattr(bag, "oob_score_")
    bag.oob_score = False
    bag.fit(X, y, t1=t1)
    assert not hasattr(bag, "oob_score_")
    assert not hasattr(bag, "oob_decision_function_")


def test_oob_refuses_base_without_predict_proba_by_name():
    X, y, t1 = _data(60, seed=16)
    bag = SequentialBootstrapBagging(NoProba, n_estimators=3, seed=1, oob_score=True)
    with pytest.raises(ValueError, match="oob_score=True.*NoProba"):
        bag.fit(X, y, t1=t1)
    assert not hasattr(bag, "estimators_")


class _NaNProba(NearestCentroid):
    def predict_proba(self, X):
        return np.full((len(X), len(self.classes_)), np.nan)


def test_oob_refuses_non_finite_out_of_bag_probabilities():
    X, y, t1 = _data(60, seed=17)
    bag = SequentialBootstrapBagging(_NaNProba, n_estimators=3, seed=1, oob_score=True)
    with pytest.raises(ValueError, match="non-finite probabilities"):
        bag.fit(X, y, t1=t1)
    assert not hasattr(bag, "oob_decision_function_")


@pytest.mark.parametrize("value", ["yes", 1, None])
def test_oob_score_must_be_a_bool(value):
    X, y, t1 = _data(40, seed=1)
    with pytest.raises(TypeError, match="oob_score"):
        SequentialBootstrapBagging(NearestCentroid, oob_score=value).fit(X, y, t1=t1)


# scikit-learn compatibility: the class has get_params, set_params, score and a classifier tag,
# so clone, cross_val_score and GridSearchCV accept it without importing sklearn.


def test_get_params_returns_the_constructor_arguments_by_name() -> None:
    bag = SequentialBootstrapBagging(
        NearestCentroid, n_estimators=4, sequential=False, max_samples=50, seed=7, oob_score=True
    )
    assert bag.get_params() == {
        "base_factory": NearestCentroid,
        "n_estimators": 4,
        "sequential": False,
        "max_samples": 50,
        "seed": 7,
        "oob_score": True,
    }
    assert bag.get_params(deep=False) == bag.get_params()
    assert bag.get_params()["base_factory"] is NearestCentroid


def test_set_params_sets_each_argument_and_returns_self() -> None:
    bag = SequentialBootstrapBagging(NearestCentroid)
    assert bag.set_params(n_estimators=3, seed=None, oob_score=True) is bag
    assert (bag.n_estimators, bag.seed, bag.oob_score) == (3, None, True)
    params = bag.get_params()
    assert SequentialBootstrapBagging(_Majority).set_params(**params).get_params() == params


def test_set_params_refuses_unknown_name_and_changes_nothing() -> None:
    bag = SequentialBootstrapBagging(NearestCentroid, n_estimators=3)
    with pytest.raises(ValueError, match="n_estmators"):
        bag.set_params(n_estmators=5)
    with pytest.raises(ValueError, match="bogus"):
        bag.set_params(n_estimators=5, bogus=1)
    assert bag.n_estimators == 3


def test_constructor_accepts_bad_arguments_and_fit_names_them() -> None:
    # scikit-learn clones by calling the constructor again, so validation belongs in fit.
    X, y, t1 = _data(40, seed=1)
    bag = SequentialBootstrapBagging(NearestCentroid, n_estimators=0)
    with pytest.raises(ValueError, match="n_estimators must be >= 1, got 0"):
        bag.fit(X, y, t1=t1)
    with pytest.raises(TypeError, match="oob_score must be a bool"):
        SequentialBootstrapBagging(NearestCentroid, oob_score="yes").fit(X, y, t1=t1)


def test_score_is_the_mean_accuracy_of_predict() -> None:
    X, y, t1 = _data(160, seed=5)
    bag = SequentialBootstrapBagging(NearestCentroid, n_estimators=5, seed=1).fit(X, y, t1=t1)
    assert bag.score(X, y) == pytest.approx(float((bag.predict(X) == y).mean()))


def test_score_weights_the_accuracy_by_sample_weight() -> None:
    X, y, t1 = _data(160, seed=5)
    bag = SequentialBootstrapBagging(NearestCentroid, n_estimators=5, seed=1).fit(X, y, t1=t1)
    w = np.random.default_rng(8).uniform(0.5, 2.0, size=len(y))
    correct = (bag.predict(X) == y).astype(float)
    assert bag.score(X, y, sample_weight=w) == pytest.approx(float((w * correct).sum() / w.sum()))


def test_score_refuses_sample_weight_of_the_wrong_length() -> None:
    X, y, t1 = _data(60, seed=5)
    bag = SequentialBootstrapBagging(NearestCentroid, n_estimators=2, seed=1).fit(X, y, t1=t1)
    with pytest.raises(ValueError, match="sample_weight has shape"):
        bag.score(X, y, sample_weight=np.ones(59))


@pytest.mark.parametrize("sequential", [True, False])
def test_fit_refuses_float_array_t1_and_names_sample_weight(sequential: bool) -> None:
    X, y, _ = _data(60, seed=3)
    bag = SequentialBootstrapBagging(NearestCentroid, n_estimators=2, sequential=sequential)
    with pytest.raises(ValueError, match=r"^t1 must be a pandas Series.*sample_weight=") as e:
        bag.fit(X, y, np.ones(60))
    assert "float64" in str(e.value)


def test_fit_refuses_float_series_t1_that_is_really_weights() -> None:
    # A pandas weight Series passed positionally used to fail deep in weights.py with a TypeError.
    X, y, t1 = _data(60, seed=3)
    weights = pd.Series(np.ones(60), index=t1.index)
    with pytest.raises(ValueError, match="t1 values must be datetimes.*float64.*sample_weight="):
        SequentialBootstrapBagging(NearestCentroid, n_estimators=2).fit(X, y, weights)


def test_sample_weight_still_accepted_as_fourth_positional_argument() -> None:
    X, y, t1 = _data(60, seed=3)
    w = np.linspace(0.5, 1.5, 60)
    positional = SequentialBootstrapBagging(NearestCentroid, n_estimators=3).fit(X, y, t1, w)
    keyword = SequentialBootstrapBagging(NearestCentroid, n_estimators=3).fit(X, y, t1=t1, sample_weight=w)
    np.testing.assert_array_equal(positional.predict(X), keyword.predict(X))


def test_bagging_is_a_classifier_to_scikit_learn() -> None:
    pytest.importorskip("sklearn")
    from sklearn.base import is_classifier

    assert is_classifier(SequentialBootstrapBagging(NearestCentroid))


def test_clone_fits_to_the_same_predictions_for_the_same_seed() -> None:
    pytest.importorskip("sklearn")
    from sklearn.base import clone

    X, y, t1 = _data(150, seed=6)
    est = SequentialBootstrapBagging(NearestCentroid, n_estimators=6, seed=4)
    copy = clone(est)
    assert copy is not est and copy.get_params() == est.get_params()
    a = est.fit(X, y, t1=t1).predict_proba(X)
    b = copy.fit(X, y, t1=t1).predict_proba(X)
    np.testing.assert_array_equal(a, b)


def _sk_param_keyword() -> str:
    """scikit-learn 1.4 and later take per-fit values as params=; older releases use fit_params=."""
    import inspect

    from sklearn.model_selection import cross_val_score

    return "params" if "params" in inspect.signature(cross_val_score).parameters else "fit_params"


def test_cross_val_score_with_purged_folds_equals_manual_loop() -> None:
    pytest.importorskip("sklearn")
    from sklearn.model_selection import cross_val_score

    X, y, t1 = _data(180, seed=7)
    Xdf = pd.DataFrame(X, index=t1.index)
    bag = SequentialBootstrapBagging(NearestCentroid, n_estimators=6, seed=4)
    scores = cross_val_score(
        bag, Xdf, y, cv=PurgedKFold(n_splits=3, t1=t1), **{_sk_param_keyword(): {"t1": t1}}
    )
    manual = []
    for train, test in PurgedKFold(n_splits=3, t1=t1).split(Xdf, y):
        fold = SequentialBootstrapBagging(NearestCentroid, n_estimators=6, seed=4)
        fold.fit(Xdf.iloc[train], y[train], t1=t1.iloc[train])
        manual.append(fold.score(Xdf.iloc[test], y[test]))
    assert len(scores) == 3
    np.testing.assert_array_equal(scores, manual)


def test_grid_search_over_n_estimators_routes_t1_to_every_fit() -> None:
    pytest.importorskip("sklearn")
    from sklearn.model_selection import GridSearchCV, cross_val_score

    X, y, t1 = _data(180, seed=7)
    Xdf = pd.DataFrame(X, index=t1.index)
    search = GridSearchCV(
        SequentialBootstrapBagging(NearestCentroid, seed=4),
        {"n_estimators": [2, 6]},
        cv=PurgedKFold(n_splits=3, t1=t1),
    ).fit(Xdf, y, t1=t1)
    assert search.best_params_["n_estimators"] in (2, 6)
    assert len(search.best_estimator_.estimators_) == search.best_params_["n_estimators"]
    for value, mean in zip([2, 6], search.cv_results_["mean_test_score"]):
        bag = SequentialBootstrapBagging(NearestCentroid, n_estimators=value, seed=4)
        expected = cross_val_score(
            bag, Xdf, y, cv=PurgedKFold(n_splits=3, t1=t1), **{_sk_param_keyword(): {"t1": t1}}
        ).mean()
        assert mean == pytest.approx(expected)


def test_t1_of_object_timestamps_fits_like_datetimes_and_strings_are_refused() -> None:
    # Object Timestamps are datetimes and always worked; string dates are not, so they are refused.
    X, y, t1 = _data(60, seed=3)
    objects = pd.Series(t1.to_numpy(dtype=object), index=t1.index, dtype=object)
    as_dates = SequentialBootstrapBagging(NearestCentroid, n_estimators=2, seed=0).fit(X, y, t1=t1)
    as_objects = SequentialBootstrapBagging(NearestCentroid, n_estimators=2, seed=0).fit(
        X, y, t1=objects
    )
    for a, b in zip(as_dates.draws_, as_objects.draws_):
        np.testing.assert_array_equal(a, b)
    strings = pd.Series(t1.astype(str).to_numpy(), index=t1.index)
    with pytest.raises(ValueError, match="t1 values must be datetimes"):
        SequentialBootstrapBagging(NearestCentroid, n_estimators=2).fit(X, y, t1=strings)
