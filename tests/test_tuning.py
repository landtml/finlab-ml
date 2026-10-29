"""Tests for quantowl.tuning (AFML ch. 9)."""
import numpy as np
import pandas as pd
import pytest

from quantowl.cv import PurgedKFold, make_t1
from quantowl.tuning import cv_score, grid_search, randomized_search


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


def test_all_nan_grid_raises_instead_of_returning_first_candidate():
    # An all-NaN grid used to return the first grid point as best_params with a NaN score. There is
    # no highest-scoring candidate in that case, so the search now raises ValueError.
    X, y = _data()
    cv = PurgedKFold(n_splits=3)
    with pytest.raises(ValueError, match="finite"):
        grid_search(Threshold, {"threshold": [100.0, 200.0]}, X, y, cv,
                    scoring=_nan_when_all_zero)


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


def test_generator_cv_is_usable_by_every_candidate():
    # A generator is consumed by the first candidate unless the splits are materialized once.
    X, y = _data()
    cv = PurgedKFold(n_splits=4)
    expected = grid_search(Threshold, {"threshold": [-1.0, 0.5]}, X, y, list(cv.split(X)))
    res = grid_search(Threshold, {"threshold": [-1.0, 0.5]}, X, y, cv.split(X))
    assert len(res.results) == 2
    assert res.results["mean_score"].tolist() == pytest.approx(expected.results["mean_score"].tolist())


def test_sample_weight_length_must_match_rows():
    X, y = _data()
    with pytest.raises(ValueError, match="sample_weight.*300"):
        grid_search(Threshold, {"threshold": [0.0]}, X, y, PurgedKFold(3), sample_weight=np.ones(10))
    with pytest.raises(ValueError, match="sample_weight.*300"):
        cv_score(Threshold, {"threshold": 0.0}, X, y, PurgedKFold(3), sample_weight=np.ones(301))


def test_cv_score_warns_on_zero_weight_test_fold():
    X, y = _data()
    cv = PurgedKFold(n_splits=3)
    first_test = next(iter(cv.split(X)))[1]
    w = np.ones(len(y))
    w[first_test] = 0.0
    with pytest.warns(UserWarning, match="fold 0"):
        mean, _ = cv_score(Threshold, {"threshold": 0.0}, X, y, cv, sample_weight=w)
    assert np.isnan(mean)


def test_y_length_must_match_rows():
    X, y = _data()
    y_long = np.concatenate([y, y[:100]])  # 400 labels for 300 rows
    with pytest.raises(ValueError, match="y.*300"):
        grid_search(Threshold, {"threshold": [0.0]}, X, y_long, PurgedKFold(3))
    with pytest.raises(ValueError, match="y.*300"):
        cv_score(Threshold, {"threshold": 0.0}, X, y_long, PurgedKFold(3))


@pytest.mark.parametrize("bad", ["ab", b"ab"])
def test_string_grid_value_rejected(bad):
    X, y = _data()
    with pytest.raises(TypeError, match="threshold"):
        grid_search(Threshold, {"threshold": bad}, X, y, PurgedKFold(3))
    with pytest.raises(TypeError, match="threshold"):
        randomized_search(Threshold, {"threshold": bad}, 2, X, y, PurgedKFold(3))


def test_non_string_iterable_grid_values_still_accepted():
    X, y = _data()
    res = grid_search(Threshold, {"threshold": (0.0, 0.5)}, X, y, PurgedKFold(3))
    assert len(res.results) == 2
    res = grid_search(Threshold, {"threshold": range(2)}, X, y, PurgedKFold(3))
    assert len(res.results) == 2
    res = randomized_search(Threshold, {"threshold": np.array([0.0, 0.5])}, 3, X, y, PurgedKFold(3))
    assert len(res.results) == 3


@pytest.mark.parametrize("call", ["grid", "cv"])
def test_y_none_raises_value_error_naming_y(call):
    X, _ = _data()
    with pytest.raises(ValueError, match="y"):
        if call == "grid":
            grid_search(Threshold, {"threshold": [0.0]}, X, None, PurgedKFold(3))
        else:
            cv_score(Threshold, {"threshold": 0.0}, X, None, PurgedKFold(3))


def test_grid_search_rejects_parameter_named_like_a_result_column() -> None:
    # A parameter called mean_score was overwritten by the score in the table.
    from quantowl.tuning import grid_search

    class C:
        def __init__(self, mean_score=0):
            self.m = mean_score

        def fit(self, X, y, sample_weight=None):
            return self

        def predict(self, X):
            return np.zeros(len(X), int)

    X = np.arange(60.0).reshape(30, 2)
    y = np.array([0] * 20 + [1] * 10)
    folds = [(np.arange(10, 30), np.arange(0, 10)), (np.arange(0, 20), np.arange(20, 30))]
    with pytest.raises(ValueError, match="mean_score"):
        grid_search(lambda mean_score: C(mean_score), {"mean_score": [5]}, X, y, folds)


def test_grid_search_rejects_y_with_mismatched_index() -> None:
    # A reversed y was used by position, so the selected threshold changed.
    x, labels = _data()
    X = pd.DataFrame(x, index=pd.date_range("2020-01-01", periods=len(x)))
    y = pd.Series(labels, index=X.index)
    with pytest.raises(ValueError, match="index"):
        grid_search(Threshold, {"threshold": [0.0, 0.5]}, X, y.iloc[::-1], PurgedKFold(n_splits=5))


def test_grid_search_rejects_sample_weight_with_mismatched_index() -> None:
    x, labels = _data()
    X = pd.DataFrame(x, index=pd.date_range("2020-01-01", periods=len(x)))
    y = pd.Series(labels, index=X.index)
    w = pd.Series(np.ones(len(x)), index=X.index[::-1])
    with pytest.raises(ValueError, match="index"):
        grid_search(Threshold, {"threshold": [0.0, 0.5]}, X, y, PurgedKFold(n_splits=5), sample_weight=w)


def test_cv_score_accepts_unlabelled_y_by_position() -> None:
    # A pandas y whose index differs from a labelled X was refused, even when y has no labels (a
    # default RangeIndex). The sibling rule in ensemble matches such an index by position, so tuning
    # does too. A reversed or otherwise labelled index is still refused.
    class Majority:
        def fit(self, X, y, sample_weight=None):
            self.v = np.bincount(np.asarray(y).ravel()).argmax()
            return self

        def predict(self, X):
            return np.full(len(X), self.v)

    from quantowl.tuning import cv_score

    Xd = pd.DataFrame(np.arange(12.0).reshape(6, 2), index=pd.date_range("2020", periods=6))
    folds = [(np.arange(0, 3), np.arange(3, 6))]
    got = cv_score(Majority, {}, Xd, pd.Series([0, 1, 1, 0, 1, 1]), folds)
    assert got[0] == pytest.approx(2 / 3)
    with pytest.raises(ValueError, match="index"):
        cv_score(Majority, {}, Xd, pd.Series([0, 1, 1, 0, 1, 1], index=Xd.index[::-1]), folds)


def test_cv_score_one_column_dataframe_label_matches_series() -> None:
    # A (n, 1) label stayed 2-D, so the accuracy compared every row's label with every prediction and
    # a perfect model scored 26/36. A one-column label is a 1-D label.
    from quantowl.tuning import cv_score

    class Echo:
        def fit(self, X, y, sample_weight=None):
            return self

        def predict(self, X):
            return (np.asarray(X)[:, 0] > 0).astype(int)

    X = pd.DataFrame(np.array([[-1, 0], [1, 0], [-1, 0], [-1, 0], [-1, 0], [-1, 0]], float))
    y = pd.Series([0, 1, 0, 0, 0, 0])
    folds = [(np.array([], int), np.arange(6))]
    assert cv_score(Echo, {}, X, y, folds) == pytest.approx((1.0, 0.0))
    assert cv_score(Echo, {}, X, pd.DataFrame({"a": y}), folds) == pytest.approx((1.0, 0.0))


def test_unlabelled_y_does_not_match_a_shifted_integer_index_by_position() -> None:
    # Any default RangeIndex y used to be exempt, so 0..59 was matched by position to X's labels
    # 5..64. An integer X index names the rows, so the exemption does not apply to it. A date-labelled
    # X still matches an unlabelled y by position.
    class Majority:
        def fit(self, X, y, sample_weight=None):
            self.v = np.bincount(np.asarray(y).ravel()).argmax()
            return self

        def predict(self, X):
            return np.full(len(X), self.v)

    from quantowl.tuning import cv_score

    X = pd.DataFrame(np.arange(120.0).reshape(60, 2), index=pd.RangeIndex(5, 65))
    y = pd.Series(np.repeat([0, 1], 30), index=pd.RangeIndex(0, 60))
    folds = [(np.arange(0, 40), np.arange(40, 60))]
    with pytest.raises(ValueError, match="y index does not match"):
        cv_score(Majority, {}, X, y, folds)


def test_randomized_search_refuses_an_empty_value_list() -> None:
    # An empty list reached rng.integers(0) and failed with "high <= 0".
    from quantowl.tuning import randomized_search

    class Thr:
        def __init__(self, t=0.5):
            self.t = t

        def fit(self, X, y, sample_weight=None):
            self.classes_ = np.unique(y)
            return self

        def predict(self, X):
            return np.zeros(len(X), dtype=int)

    X = np.random.default_rng(0).normal(size=(30, 2))
    y = np.zeros(30, dtype=int)
    with pytest.raises(ValueError, match="must not be empty"):
        randomized_search(Thr, {"t": []}, 2, X, y, [(np.arange(20), np.arange(20, 30))])


def test_tuning_refuses_a_two_column_label_by_name() -> None:
    # Only a one-column y was squeezed, so a two-column y gave a numpy broadcast error. It is refused
    # by name, as the one-column and 0-d cases are.
    class Majority:
        def fit(self, X, y, sample_weight=None):
            self.v = 0
            return self

        def predict(self, X):
            return np.zeros(len(X), dtype=int)

    from quantowl.tuning import cv_score

    X = np.random.default_rng(0).normal(size=(20, 2))
    y2 = np.c_[np.arange(20) % 2, np.arange(20) % 2]
    folds = [(np.arange(0, 10), np.arange(10, 20))]
    with pytest.raises(ValueError, match="one column"):
        cv_score(Majority, {}, X, y2, folds)


def test_unlabelled_int64_index_matches_date_x_by_position_as_rangeindex() -> None:
    # An Int64 Index 0..n-1 was refused against a date-indexed X, while a RangeIndex 0..n-1 was
    # matched by position. The two are the same unlabelled index.
    from quantowl.tuning import _matches_by_position

    dates = pd.date_range("2021-01-01", periods=60)
    assert _matches_by_position(pd.RangeIndex(60), dates, 60)
    assert _matches_by_position(pd.Index(np.arange(60), dtype="int64"), dates, 60)
    # An integer X index still names its rows, so a shifted 0..n-1 index is not matched.
    assert not _matches_by_position(pd.Index(np.arange(60), dtype="int64"), pd.RangeIndex(5, 65), 60)


class _ColumnThreshold(Threshold):
    def predict(self, X):
        return super().predict(X).reshape(-1, 1)


class _WideThreshold(Threshold):
    def predict(self, X):
        return np.stack([super().predict(X)] * 2, axis=1)


def test_cv_score_squeezes_a_column_vector_prediction():
    X, y = _data()
    cv = list(PurgedKFold(n_splits=3).split(X))
    assert cv_score(_ColumnThreshold, {"threshold": 0.5}, X, y, cv) == cv_score(
        Threshold, {"threshold": 0.5}, X, y, cv
    )
    with pytest.raises(ValueError, match=r"prediction.*shape \(\d+, 2\)"):
        cv_score(_WideThreshold, {"threshold": 0.5}, X, y, cv)


def test_nan_sample_weight_is_refused_by_name():
    X, y = _data()
    w = np.ones(len(y))
    w[3] = np.nan
    cv = PurgedKFold(n_splits=3)
    with pytest.raises(ValueError, match="NaN"):
        grid_search(Threshold, {"threshold": [0.0, 0.5]}, X, y, cv, sample_weight=w)
    with pytest.raises(ValueError, match="NaN"):
        cv_score(Threshold, {"threshold": 0.5}, X, y, cv, sample_weight=w)


def test_grid_search_accepts_sample_weight_over_the_same_instants_in_another_zone() -> None:
    x, labels = _data()
    utc = pd.date_range("2020-01-01", periods=len(x), tz="UTC")
    X = pd.DataFrame(x, index=utc)
    y = pd.Series(labels, index=utc)
    cv = PurgedKFold(n_splits=5)
    w = np.ones(len(x))
    res = grid_search(Threshold, {"threshold": [0.0, 0.5]}, X, y, cv,
                      sample_weight=pd.Series(w, index=utc.tz_convert("Asia/Taipei")))
    assert res.best_params == {"threshold": 0.5}
    shifted = pd.Series(w, index=(utc + pd.Timedelta(days=1)).tz_convert("Asia/Taipei"))
    with pytest.raises(ValueError, match="sample_weight index"):
        grid_search(Threshold, {"threshold": [0.0, 0.5]}, X, y, cv, sample_weight=shifted)


def _per_class_scorer(y_true, y_pred, sample_weight=None):
    # One value per class, as f1_score(average=None) returns.
    return np.array([0.25, 0.75])


def test_grid_search_refuses_non_scalar_scorer_output():
    X, y = _data()
    with pytest.raises(ValueError, match="one number per fold"):
        grid_search(Threshold, {"threshold": [0.0, 0.5]}, X, y, PurgedKFold(3),
                    scoring=_per_class_scorer)


def test_cv_score_accepts_numpy_scalar_scorer_output():
    X, y = _data()

    def half(y_true, y_pred, sample_weight=None):
        return np.float32(0.5)

    mean, std = cv_score(Threshold, {"threshold": 0.0}, X, y, PurgedKFold(3), scoring=half)
    assert mean == pytest.approx(0.5)
    assert std == pytest.approx(0.0)


class _FitWithoutSampleWeight:
    """Base whose fit takes no sample_weight keyword, as KNeighborsClassifier does."""

    def __init__(self, threshold=0.0):
        self.threshold = threshold

    def fit(self, X, y):
        return self

    def predict(self, X):
        return (X[:, 0] > self.threshold).astype(int)


def test_grid_search_fits_base_without_sample_weight_keyword_when_no_weights():
    X, y = _data()
    res = grid_search(_FitWithoutSampleWeight, {"threshold": [0.0, 0.5]}, X, y, PurgedKFold(3))
    assert res.best_params == {"threshold": 0.5}


def test_cv_score_fits_kneighbors_base_when_no_weights():
    pytest.importorskip("sklearn")
    from sklearn.neighbors import KNeighborsClassifier

    X, y = _data()
    mean, _ = cv_score(lambda: KNeighborsClassifier(n_neighbors=3), {}, X, y, PurgedKFold(3))
    assert 0.0 <= mean <= 1.0


class _PipelineName:
    pass


# sklearn.pipeline.Pipeline's name and module, which is what quantowl checks.
_PipelineName.__name__ = "Pipeline"
_PipelineName.__module__ = "sklearn.pipeline"


class _FakePipeline(_PipelineName):
    """Stand-in for sklearn.pipeline.Pipeline: steps, and a fit that refuses sample_weight.

    quantowl recognises a Pipeline by class name and module, so the stand-in declares that module.
    """

    def __init__(self, threshold=0.0):
        self.threshold = threshold
        self.steps = [("scale", Threshold()), ("thresh", Threshold(threshold))]
        self.named_steps = dict(self.steps)
        self.fit_params = None

    def fit(self, X, y, **fit_params):
        if "sample_weight" in fit_params:
            raise ValueError("Pipeline.fit does not accept the sample_weight parameter.")
        self.fit_params = fit_params
        return self

    def predict(self, X):
        return (X[:, 0] > self.threshold).astype(int)


def test_pipeline_gets_sample_weight_on_its_final_step():
    X, y = _data(120)
    w = np.random.default_rng(3).uniform(0.5, 1.5, len(y))
    cv = PurgedKFold(n_splits=3)
    made = []

    def factory(**params):
        model = _FakePipeline(**params)
        made.append(model)
        return model

    grid_search(factory, {"threshold": [0.0]}, X, y, cv, sample_weight=w)
    expected = [w[train] for train, _ in cv.split(X)]
    assert len(made) == len(expected)
    for model, want in zip(made, expected):
        assert list(model.fit_params) == ["thresh__sample_weight"]
        np.testing.assert_array_equal(model.fit_params["thresh__sample_weight"], want)


def test_pipeline_without_sample_weight_is_fit_with_no_fit_params():
    X, y = _data(120)
    made = []

    def factory(**params):
        model = _FakePipeline(**params)
        made.append(model)
        return model

    grid_search(factory, {"threshold": [0.0]}, X, y, PurgedKFold(n_splits=3))
    assert all(model.fit_params == {} for model in made)


@pytest.mark.parametrize("final", ["passthrough", None])
def test_pipeline_ending_in_passthrough_or_none_is_refused_when_weighted(final):
    class _Ends(_FakePipeline):
        def __init__(self, threshold=0.0):
            super().__init__(threshold)
            self.steps = [("scale", Threshold()), ("thresh", final)]
            self.named_steps = dict(self.steps)

    X, y = _data(120)
    w = np.ones(len(y))
    with pytest.raises(ValueError, match="final step"):
        grid_search(_Ends, {"threshold": [0.0]}, X, y, PurgedKFold(n_splits=3), sample_weight=w)


class _StepsButNotPipeline:
    """A user model that happens to have steps and named_steps; it takes sample_weight directly."""

    def __init__(self):
        self.steps = [("a", Threshold()), ("b", Threshold())]
        self.named_steps = dict(self.steps)
        self.fit_params = None

    def fit(self, X, y, **fit_params):
        self.fit_params = fit_params
        return self

    def predict(self, X):
        return (X[:, 0] > 0).astype(int)


def test_model_with_steps_that_is_not_a_pipeline_gets_sample_weight_directly():
    X, y = _data(120)
    w = np.random.default_rng(4).uniform(0.5, 1.5, len(y))
    made = []

    def factory(k=0):
        model = _StepsButNotPipeline()
        made.append(model)
        return model

    grid_search(factory, {"k": [0]}, X, y, PurgedKFold(n_splits=3), sample_weight=w)
    assert made and all(list(m.fit_params) == ["sample_weight"] for m in made)


def _sklearn_pipeline_factory():
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler

    def factory(C):
        return Pipeline([("scale", StandardScaler()), ("lr", LogisticRegression(C=C))])

    return factory


def _sklearn_data(n=300, seed=1):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, 3))
    y = (X[:, 0] + 0.5 * rng.normal(size=n) > 0).astype(int)
    w = rng.uniform(0.5, 1.5, n)
    return X, y, w


def _manual_pipeline_fold_scores(factory, X, y, w, cv, key):
    """Weighted accuracy per fold, fitting the pipeline with its weight under ``key``."""
    scores = []
    for train, test in cv.split(X):
        model = factory(C=1.0)
        model.fit(X[train], y[train], **{key: w[train]})
        hit = (model.predict(X[test]) == y[test]).astype(float)
        scores.append(float(np.sum(hit * w[test]) / np.sum(w[test])))
    return scores


def test_grid_search_with_sklearn_pipeline_matches_manual_weighted_loop():
    pytest.importorskip("sklearn")
    X, y, w = _sklearn_data()
    cv = PurgedKFold(n_splits=4)
    factory = _sklearn_pipeline_factory()
    res = grid_search(factory, {"C": [1.0]}, X, y, cv, sample_weight=w)
    manual = _manual_pipeline_fold_scores(factory, X, y, w, cv, "lr__sample_weight")
    assert res.best_score == np.mean(manual)


def test_grid_search_with_nested_sklearn_pipeline_routes_through_each_final_step():
    pytest.importorskip("sklearn")
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler

    def factory(C):
        inner = Pipeline([("lr", LogisticRegression(C=C))])
        return Pipeline([("scale", StandardScaler()), ("model", inner)])

    X, y, w = _sklearn_data()
    cv = PurgedKFold(n_splits=4)
    res = grid_search(factory, {"C": [1.0]}, X, y, cv, sample_weight=w)
    manual = _manual_pipeline_fold_scores(factory, X, y, w, cv, "model__lr__sample_weight")
    assert res.best_score == np.mean(manual)


def test_grid_search_with_sklearn_pipeline_and_metadata_routing_matches_manual_loop():
    pytest.importorskip("sklearn")
    from sklearn import config_context
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler

    X, y, w = _sklearn_data()
    cv = PurgedKFold(n_splits=4)

    def factory(C):
        # Routing needs an explicit request from every step that takes sample_weight, the scaler too.
        return Pipeline(
            [
                ("scale", StandardScaler().set_fit_request(sample_weight=False)),
                ("lr", LogisticRegression(C=C).set_fit_request(sample_weight=True)),
            ]
        )

    with config_context(enable_metadata_routing=True):
        res = grid_search(factory, {"C": [1.0]}, X, y, cv, sample_weight=w)
        manual = _manual_pipeline_fold_scores(factory, X, y, w, cv, "sample_weight")
    assert res.best_score == np.mean(manual)
