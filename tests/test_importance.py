"""Tests for finlab.importance (AFML chapter 8).

No sklearn. The model under test is a tiny nearest-centroid classifier written in
numpy. Each fast implementation is cross-checked against a naive, loop-based
reference on seeded random inputs.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from finlab.cv import PurgedKFold
from finlab.importance import (
    OrthogonalFeatures,
    mda,
    mdi_from_matrix,
    orthogonal_features,
    sfi,
)


class NearestCentroid:
    """Minimal classifier implementing the duck-typed protocol."""

    def fit(self, X, y, sample_weight=None):
        X = np.asarray(X, dtype=float)
        y = np.asarray(y)
        w = np.ones(len(y)) if sample_weight is None else np.asarray(sample_weight, dtype=float)
        self.classes_ = np.unique(y)
        self.centroids_ = np.vstack(
            [np.average(X[y == c], axis=0, weights=w[y == c]) for c in self.classes_]
        )
        return self

    def predict(self, X):
        X = np.asarray(X, dtype=float)
        d = ((X[:, None, :] - self.centroids_[None, :, :]) ** 2).sum(axis=-1)
        return self.classes_[np.argmin(d, axis=1)]


def make_data(n: int = 600, p: int = 5, seed: int = 0):
    """Feature 0 carries the label (class means -2 and +2); the rest is noise."""
    rng = np.random.default_rng(seed)
    y = rng.integers(0, 2, n)
    X = rng.normal(size=(n, p))
    X[:, 0] += 2.0 * (2 * y - 1)
    return X, y


# ---------------------------------------------------------------------------
# Naive references
# ---------------------------------------------------------------------------


def naive_mdi(imp, names=None):
    """Loop-based MDI: skip zeros and NaN, mean and standard error, normalize."""
    n_trees = len(imp)
    p = len(imp[0])
    means, ses = [], []
    for j in range(p):
        vals = []
        for t in range(n_trees):
            v = imp[t][j]
            if v != 0.0 and v == v:  # v == v is False only for NaN
                vals.append(v)
        c = len(vals)
        m = sum(vals) / c
        if c > 1:
            ss = 0.0
            for v in vals:
                ss += (v - m) ** 2
            se = (ss / (c - 1)) ** 0.5 / c**0.5
        else:
            se = float("nan")
        means.append(m)
        ses.append(se)
    total = sum(means)
    return [m / total for m in means], [s / total for s in ses]


def naive_mda(model_factory, X, y, folds, seed=0, n_repeats=1):
    """Loop-based MDA that draws permutations in the same order as the module."""
    X = np.asarray(X, dtype=float)
    y = np.asarray(y)
    rng = np.random.default_rng(seed)
    p = X.shape[1]
    baselines, drops = [], []
    for train, test in folds:
        model = model_factory().fit(X[train], y[train], sample_weight=None)
        base = float(np.mean(model.predict(X[test]) == y[test]))
        baselines.append(base)
        fold = np.zeros(p)
        for _ in range(n_repeats):
            for j in range(p):
                perm = rng.permutation(len(test))
                Xp = X[test].copy()
                Xp[:, j] = X[test][perm, j]
                acc = float(np.mean(model.predict(Xp) == y[test]))
                fold[j] += base - acc
        drops.append(fold / n_repeats)
    return np.array(baselines), np.array(drops)


def naive_correlation(X):
    """Correlation matrix by explicit double loop (no numpy linear algebra)."""
    n, p = len(X), len(X[0])
    mu = [sum(X[i][j] for i in range(n)) / n for j in range(p)]
    sd = []
    for j in range(p):
        ss = sum((X[i][j] - mu[j]) ** 2 for i in range(n))
        sd.append((ss / (n - 1)) ** 0.5)
    C = np.zeros((p, p))
    for a in range(p):
        for b in range(p):
            s = 0.0
            for i in range(n):
                s += ((X[i][a] - mu[a]) / sd[a]) * ((X[i][b] - mu[b]) / sd[b])
            C[a, b] = s / (n - 1)
    return C


# ---------------------------------------------------------------------------
# MDI
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_mdi_matches_naive_reference(seed):
    rng = np.random.default_rng(seed)
    imp = rng.random((50, 7))
    imp[rng.random(imp.shape) < 0.4] = 0.0  # feature not drawn
    imp[rng.random(imp.shape) < 0.05] = np.nan
    imp[:, 3] = np.where(np.arange(50) == 0, 0.5, 0.0)  # one non-zero value -> std is NaN

    out = mdi_from_matrix(imp)
    ref_mean, ref_se = naive_mdi(imp.tolist())
    np.testing.assert_allclose(out["mean"].to_numpy(), ref_mean, rtol=1e-10, atol=1e-12, equal_nan=True)
    np.testing.assert_allclose(out["std"].to_numpy(), ref_se, rtol=1e-10, atol=1e-12, equal_nan=True)


def test_mdi_normalizes_to_one_and_is_bounded():
    rng = np.random.default_rng(3)
    imp = rng.random((40, 6)) * (rng.random((40, 6)) > 0.3)
    out = mdi_from_matrix(imp, names=list("abcdef"))
    assert list(out.index) == list("abcdef")
    assert out["mean"].sum() == pytest.approx(1.0)
    assert ((out["mean"] >= 0) & (out["mean"] <= 1)).all()


def test_mdi_dataframe_names_and_zero_exclusion():
    imp = pd.DataFrame({"x": [0.0, 0.2, 0.4], "y": [0.0, 0.0, 0.0], "z": [0.1, 0.0, 0.3]})
    out = mdi_from_matrix(imp[["x", "z"]])
    assert list(out.index) == ["x", "z"]
    assert out["mean"].sum() == pytest.approx(1.0)
    with pytest.raises(ValueError):  # no positive importance anywhere
        mdi_from_matrix(imp[["y"]])


# ---------------------------------------------------------------------------
# MDA
# ---------------------------------------------------------------------------


def test_mda_informative_feature_has_largest_drop_and_noise_is_near_zero():
    X, y = make_data(n=600, p=5, seed=0)
    cv = PurgedKFold(n_splits=5)
    baseline, imp = mda(NearestCentroid, X, y, cv, n_repeats=3, seed=0)

    assert baseline.shape == (5,)
    assert imp.shape == (5, 2)
    assert list(imp.columns) == ["mean", "std"]
    assert imp["mean"].idxmax() == 0
    assert imp.loc[0, "mean"] > 0.2
    noise = imp.loc[1:, "mean"].to_numpy()
    assert np.all(np.abs(noise) < 0.03), noise


def test_mda_matches_naive_reference_exactly():
    X, y = make_data(n=300, p=4, seed=5)
    cv = PurgedKFold(n_splits=4)
    folds = list(cv.split(X))
    base_ref, drops_ref = naive_mda(NearestCentroid, X, y, folds, seed=7, n_repeats=2)

    baseline, imp = mda(NearestCentroid, X, y, cv, seed=7, n_repeats=2)
    np.testing.assert_allclose(baseline, base_ref)
    np.testing.assert_allclose(imp["mean"].to_numpy(), drops_ref.mean(axis=0), atol=1e-12)
    np.testing.assert_allclose(
        imp["std"].to_numpy(), drops_ref.std(axis=0, ddof=1) / np.sqrt(len(folds)), atol=1e-12
    )


def test_mda_custom_scoring_and_dataframe_input():
    X, y = make_data(n=300, p=3, seed=1)
    Xdf = pd.DataFrame(X, columns=["signal", "n1", "n2"])

    def weighted_acc(y_true, y_pred, w):
        w = np.ones(len(y_true)) if w is None else w
        return float(np.sum(w * (y_true == y_pred)) / np.sum(w))

    cv = PurgedKFold(n_splits=3)
    base_a, imp_a = mda(NearestCentroid, Xdf, y, cv, scoring=weighted_acc, seed=0)
    base_b, imp_b = mda(NearestCentroid, X, y, cv, scoring="accuracy", seed=0)
    np.testing.assert_allclose(base_a, base_b)
    assert list(imp_a.index) == ["signal", "n1", "n2"]
    assert imp_a["mean"].idxmax() == "signal"


def test_mda_rejects_unknown_scoring():
    X, y = make_data(n=100, p=2)
    with pytest.raises(ValueError):
        mda(NearestCentroid, X, y, PurgedKFold(n_splits=2), scoring="f1")


# ---------------------------------------------------------------------------
# SFI
# ---------------------------------------------------------------------------


def test_sfi_ranks_informative_feature_first():
    X, y = make_data(n=600, p=5, seed=2)
    baseline, imp = sfi(NearestCentroid, X, y, PurgedKFold(n_splits=5))
    assert baseline.shape == (5,)
    assert imp.shape == (5, 2)
    assert imp["mean"].idxmax() == 0
    assert imp.loc[0, "mean"] > 0.9
    assert np.all(imp.loc[1:, "mean"] < 0.65)
    # Each single-feature score should be close to the chance rate on noise.
    assert abs(imp.loc[1:, "mean"].mean() - 0.5) < 0.1


def test_sfi_with_sample_weight_and_dataframe_names():
    X, y = make_data(n=300, p=3, seed=4)
    Xdf = pd.DataFrame(X, columns=list("abc"))
    w = np.random.default_rng(0).uniform(0.5, 1.5, len(y))
    _, imp = sfi(NearestCentroid, Xdf, y, PurgedKFold(n_splits=3), sample_weight=w)
    assert list(imp.index) == list("abc")
    assert imp["mean"].idxmax() == "a"


# ---------------------------------------------------------------------------
# Orthogonal features
# ---------------------------------------------------------------------------


def correlated_data(n: int = 500, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    base = rng.normal(size=(n, 3))
    mix = np.array(
        [
            [1.0, 0.8, 0.1, 0.0, 0.3],
            [0.0, 0.5, 1.0, 0.2, 0.0],
            [0.4, 0.0, 0.0, 1.0, 0.9],
        ]
    )
    return base @ mix + 0.1 * rng.normal(size=(n, 5)) + np.array([3.0, -1.0, 0.0, 2.0, 5.0])


def test_orthogonal_features_are_uncorrelated_and_match_eigenvalues():
    X = correlated_data()
    P, eigenvalues, eigenvectors = orthogonal_features(X, variance_threshold=1.0)
    n = X.shape[0]
    cov = np.cov(P, rowvar=False)
    off = cov - np.diag(np.diag(cov))
    assert np.max(np.abs(off)) < 1e-9 * np.max(np.abs(np.diag(cov)))
    # Book: P'P = Lambda, so the variance of component k is Lambda_k / (n - 1).
    np.testing.assert_allclose(np.diag(cov), eigenvalues.to_numpy() / (n - 1), rtol=1e-9)
    # Eigenvectors are orthonormal.
    W = eigenvectors.to_numpy()
    np.testing.assert_allclose(W.T @ W, np.eye(W.shape[1]), atol=1e-10)
    assert np.all(np.diff(eigenvalues.to_numpy()) <= 0)


def test_orthogonal_features_eigenvalues_agree_with_naive_correlation():
    X = correlated_data(n=200, seed=3)
    C = naive_correlation(X.tolist())  # correlation matrix, explicit loops
    ref = np.sort(np.linalg.eigvals(C).real)[::-1]
    _, eigenvalues, _ = orthogonal_features(X, variance_threshold=1.0)
    # Z'Z = (n - 1) * C, so Lambda / (n - 1) = eig(C).
    np.testing.assert_allclose(eigenvalues.to_numpy() / (X.shape[0] - 1), ref, rtol=1e-8, atol=1e-10)


def test_transform_reproduces_fitted_components_and_applies_to_new_data():
    X = correlated_data(n=400, seed=7)
    model = OrthogonalFeatures(variance_threshold=0.95).fit(X)
    P_train = model.transform(X)
    P_fit = model.fit_transform(X)
    np.testing.assert_allclose(P_train, P_fit, atol=1e-12)
    P_full, _, _ = orthogonal_features(X, variance_threshold=0.95)
    np.testing.assert_allclose(P_train, np.asarray(P_full), atol=1e-12)

    # New data is standardized with the training mean and sd, not refit.
    X_new = correlated_data(n=50, seed=8)
    Z_new = (X_new - model.mean_) / model.std_
    np.testing.assert_allclose(model.transform(X_new), Z_new @ model.eigenvectors_, atol=1e-12)


def test_variance_threshold_selects_minimal_components():
    X = correlated_data(n=300, seed=9)
    model = OrthogonalFeatures(variance_threshold=0.9).fit(X)
    k = model.n_components_
    ratio = model.explained_variance_ratio_
    assert ratio.sum() >= 0.9
    if k > 1:
        assert ratio[: k - 1].sum() < 0.9  # the component that crosses the threshold is the last kept
    assert model.eigenvectors_.shape == (5, k)


def test_orthogonal_features_dataframe_round_trip_and_constant_column_rejected():
    X = correlated_data(n=100, seed=10)
    Xdf = pd.DataFrame(X, index=pd.RangeIndex(100, 200), columns=list("abcde"))
    P, eigenvalues, eigenvectors = orthogonal_features(Xdf)
    assert isinstance(P, pd.DataFrame)
    assert list(P.index) == list(Xdf.index)
    assert list(eigenvalues.index) == list(P.columns)
    assert list(eigenvectors.index) == list("abcde")
    with pytest.raises(ValueError):
        orthogonal_features(np.column_stack([X, np.ones(100)]))
    with pytest.raises(RuntimeError):
        OrthogonalFeatures().transform(X)


@pytest.mark.parametrize("length", [10, 400])
def test_mda_and_sfi_reject_sample_weight_of_wrong_length(length):
    rng = np.random.default_rng(0)
    X = rng.normal(size=(300, 2))
    y = (X[:, 0] > 0).astype(int)
    cv = PurgedKFold(n_splits=3)
    with pytest.raises(ValueError, match="sample_weight.*300"):
        mda(NearestCentroid, X, y, cv, sample_weight=np.ones(length))
    with pytest.raises(ValueError, match="sample_weight.*300"):
        sfi(NearestCentroid, X, y, cv, sample_weight=np.ones(length))


def test_mda_and_sfi_reject_pandas_y_with_mismatched_index():
    rng = np.random.default_rng(0)
    Xdf = pd.DataFrame(rng.normal(size=(300, 2)), columns=["a", "b"])
    y_reversed = pd.Series((Xdf["a"] > 0).astype(int).to_numpy(), index=Xdf.index[::-1])
    cv = PurgedKFold(n_splits=3)
    with pytest.raises(ValueError, match="index"):
        mda(NearestCentroid, Xdf, y_reversed, cv)
    with pytest.raises(ValueError, match="index"):
        sfi(NearestCentroid, Xdf, y_reversed, cv)


def test_mda_and_sfi_reject_pandas_weight_with_mismatched_index():
    rng = np.random.default_rng(0)
    Xdf = pd.DataFrame(rng.normal(size=(300, 2)), columns=["a", "b"])
    y = pd.Series((Xdf["a"] > 0).astype(int).to_numpy(), index=Xdf.index)
    w_reversed = pd.Series(np.ones(300), index=Xdf.index[::-1])
    cv = PurgedKFold(n_splits=3)
    with pytest.raises(ValueError, match="index"):
        mda(NearestCentroid, Xdf, y, cv, sample_weight=w_reversed)
    with pytest.raises(ValueError, match="index"):
        sfi(NearestCentroid, Xdf, y, cv, sample_weight=w_reversed)


def test_sfi_accepts_pandas_y_and_weight_aligned_to_x():
    rng = np.random.default_rng(4)
    Xdf = pd.DataFrame(rng.normal(size=(300, 3)), columns=list("abc"))
    y = (Xdf["a"] > 0).astype(int).to_numpy()
    w = np.random.default_rng(0).uniform(0.5, 1.5, len(y))
    base_np, imp_np = sfi(NearestCentroid, Xdf, y, PurgedKFold(n_splits=3), sample_weight=w)
    base_pd, imp_pd = sfi(
        NearestCentroid,
        Xdf,
        pd.Series(y, index=Xdf.index),
        PurgedKFold(n_splits=3),
        sample_weight=pd.Series(w, index=Xdf.index),
    )
    np.testing.assert_allclose(np.asarray(base_pd), np.asarray(base_np))
    np.testing.assert_allclose(imp_pd.to_numpy(), imp_np.to_numpy())


def test_mdi_rejects_infinite_impurity():
    imp = np.array([[np.inf, 1.0], [1.0, 1.0]])
    with pytest.raises(ValueError, match="infinite"):
        mdi_from_matrix(imp, ["a", "b"])


def test_mda_warns_on_zero_weight_test_fold_and_keeps_values():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(300, 2))
    y = (X[:, 0] > 0).astype(int)
    cv = PurgedKFold(n_splits=3)
    first_test = next(iter(cv.split(X)))[1]
    w = np.ones(300)
    w[first_test] = 0.0
    with pytest.warns(UserWarning, match="fold 0"):
        baseline, _ = mda(NearestCentroid, X, y, cv, sample_weight=w)
    assert np.isnan(baseline[0])
    assert np.all(np.isfinite(baseline[1:]))


def test_check_xy_accepts_same_instants_in_another_timezone():
    from finlab.importance import _check_xy

    utc = pd.date_range("2020-01-01", periods=10, freq="D", tz="UTC")
    X = pd.DataFrame(np.random.default_rng(0).normal(size=(10, 2)), index=utc)
    y = pd.Series(np.arange(10), index=utc.tz_convert("Asia/Taipei"))
    _, y_arr = _check_xy(X, y)
    np.testing.assert_array_equal(y_arr, np.arange(10))


def test_check_xy_rejects_shifted_instants_across_timezones():
    from finlab.importance import _check_xy

    utc = pd.date_range("2020-01-01", periods=10, freq="D", tz="UTC")
    X = pd.DataFrame(np.random.default_rng(0).normal(size=(10, 2)), index=utc)
    y = pd.Series(np.arange(10), index=(utc + pd.Timedelta(days=1)).tz_convert("Asia/Taipei"))
    with pytest.raises(ValueError, match="index"):
        _check_xy(X, y)


def test_mdi_from_matrix_rejects_negative_impurity_decreases() -> None:
    # A negative decrease was averaged in and credited to a feature.
    from finlab.importance import mdi_from_matrix

    with pytest.raises(ValueError, match="negative"):
        mdi_from_matrix(np.array([[0.5, -0.1], [0.4, 0.2]]))


def test_orthogonal_features_transform_rejects_reordered_columns() -> None:
    # transform used positional columns, so a reordered frame gave silently wrong features.
    X = pd.DataFrame(np.random.default_rng(0).normal(size=(40, 2)), columns=["a", "b"])
    model = OrthogonalFeatures(0.95).fit(X)
    with pytest.raises(ValueError, match="column"):
        model.transform(X[["b", "a"]])
    # Same names in the same order still work.
    assert model.transform(X).shape[0] == 40


def test_mda_one_column_dataframe_label_matches_series() -> None:
    # A (n, 1) label stayed 2-D, so the accuracy compared every row with every prediction by
    # broadcasting and the importances came out 0. A one-column label is a 1-D label.
    from finlab.importance import mda

    rng = np.random.default_rng(3)
    X = pd.DataFrame(rng.normal(size=(60, 3)), columns=["a", "b", "c"])
    y = pd.Series((X["a"] > 0).astype(int).to_numpy())
    cv = PurgedKFold(n_splits=3)

    def mk():
        return NearestCentroid()

    want = mda(mk, X, y, cv)[1]["mean"].to_numpy()
    got = mda(mk, X, pd.DataFrame({"y": y}), cv)[1]["mean"].to_numpy()
    np.testing.assert_allclose(got, want)
    assert got[0] > 0


def test_mdi_from_matrix_normalizes_when_the_sum_overflows() -> None:
    # Two finite decreases whose sum overflows to inf made every normalized mean 0.0 with no error.
    # The means are scaled by their peak before they are summed.
    from finlab.importance import mdi_from_matrix

    got = mdi_from_matrix(np.array([[1e308, 1e308]]))["mean"].tolist()
    assert got == pytest.approx([0.5, 0.5])


def test_mdi_from_matrix_accepts_rounding_level_negative() -> None:
    # A -1e-17 (rounding of a zero decrease) was refused by the strict sign rule. A rounding-level
    # negative is now taken as the zero it is, and the feature still counts as drawn, so the mean is
    # [1/3, 2/3].
    from finlab.importance import mdi_from_matrix

    out = mdi_from_matrix(np.array([[-1e-17, 1.0], [1.0, 1.0]]))
    assert out["mean"].to_numpy() == pytest.approx([1 / 3, 2 / 3], rel=1e-12)


def test_mdi_from_matrix_does_not_overflow_large_impurities() -> None:
    # Only the across-column sum was scaled; the within-column sum overflowed.
    # All the inputs are finite and positive, so each result must be finite.
    from finlab.importance import mdi_from_matrix

    out = mdi_from_matrix(np.array([[1e308, 1e308], [1e308, 1e308]]))
    assert out["mean"].tolist() == pytest.approx([0.5, 0.5])
    out = mdi_from_matrix(np.array([[1.7e308, 1.0], [1.7e308, 1.0]]))
    assert out["mean"].iloc[0] == pytest.approx(1.0)
    assert np.isfinite(out["std"]).all()
    out = mdi_from_matrix(np.array([[1e200], [0.5e200]]))
    assert np.isfinite(out["std"]).all() and out["mean"].iloc[0] == pytest.approx(1.0)


@pytest.mark.parametrize("scale", [1e14, 1e16, 1e300])
def test_mdi_rounding_negative_keeps_draw_at_large_scale(scale: float) -> None:
    # The rounding-level negative was set to tiny before scaling by the largest impurity, so at scale
    # 1e16 it underflowed to 0 and the feature lost a draw. The means are ratios and must not depend on
    # the scale.
    from finlab.importance import mdi_from_matrix

    M = np.array([[scale, -1e-16 * scale], [scale, scale]])
    mean = mdi_from_matrix(M)["mean"].to_numpy()
    assert mean == pytest.approx([2 / 3, 1 / 3], rel=1e-9)


def test_mdi_refuses_a_real_negative_of_1e_minus_12_relative() -> None:
    # The rounding tolerance was 1e-12 of the largest impurity, about 4500 times float64 rounding, so
    # -1e-12 was read as a zero decrease. Only rounding-level negatives are set to zero now (1e-14); a
    # larger one is refused as a negative decrease.
    from finlab.importance import mdi_from_matrix

    with pytest.raises(ValueError, match="negative"):
        mdi_from_matrix(np.array([[1.0, -1e-12], [1.0, 1.0]]))
    with pytest.raises(ValueError, match="negative"):
        mdi_from_matrix(np.array([[1.0, -1e-11], [1.0, 1.0]]))
    # A rounding-level negative, -1e-17, is still a zero decrease and gives the means [2/3, 1/3].
    mean = mdi_from_matrix(np.array([[1.0, -1e-17], [1.0, 1.0]]))["mean"].to_numpy()
    assert mean == pytest.approx([2 / 3, 1 / 3], rel=1e-9)


def test_mda_accepts_unlabelled_y_against_date_x_as_tuning_does() -> None:
    # Tuning matches an unlabelled 0..n-1 y to a date-indexed X by position, but mda used to refuse
    # it. The importance check now agrees. A reversed, labelled y is still refused.
    from finlab.cv import PurgedKFold

    X, y = make_data(n=300, p=3, seed=0)
    X = pd.DataFrame(np.asarray(X), index=pd.date_range("2021-01-01", periods=len(X)))
    y_unlabelled = pd.Series(np.asarray(y))
    baseline, imp = mda(NearestCentroid, X, y_unlabelled, PurgedKFold(n_splits=5), n_repeats=1, seed=0)
    assert imp.shape == (3, 2)
    with pytest.raises(ValueError, match="y index does not match"):
        mda(NearestCentroid, X, pd.Series(np.asarray(y), index=X.index[::-1]), PurgedKFold(n_splits=5))


def test_mdi_positive_impurity_below_peak_stays_drawn() -> None:
    # Dividing by the peak made a positive impurity of 1e-320 against a peak of 1e300 divide to 0.0,
    # which was read as "not drawn", so the feature's mean was NaN. It is drawn in every tree, so its
    # share is about 1e-620 (0.0 after rounding) and must not be NaN.
    from finlab.importance import mdi_from_matrix

    m = np.array([[1e-320, 1e300], [1e-320, 1e300], [1e-320, 1e300]])
    out = mdi_from_matrix(m)
    assert not np.isnan(out["mean"].to_numpy()).any()
    assert out["mean"].tolist() == pytest.approx([0.0, 1.0], abs=1e-12)
    assert out["std"].tolist() == pytest.approx([0.0, 0.0], abs=1e-12)
    # A column at 1e308 next to a 1e308 peak stays finite and sums to one.
    big = mdi_from_matrix(np.array([[1e308, 1e308], [1e308, 0.5e308]]))
    assert np.isfinite(big["mean"].to_numpy()).all()
    assert big["mean"].sum() == pytest.approx(1.0)


class _ColumnPredict(NearestCentroid):
    def predict(self, X):
        return super().predict(X).reshape(-1, 1)


class _WidePredict(NearestCentroid):
    def predict(self, X):
        return np.stack([super().predict(X)] * 2, axis=1)


def test_mda_squeezes_a_column_vector_prediction() -> None:
    X, y = make_data(n=300, p=3, seed=2)
    cv = PurgedKFold(n_splits=3)
    flat, _ = mda(NearestCentroid, X, y, cv, n_repeats=1, seed=0)
    column, _ = mda(_ColumnPredict, X, y, cv, n_repeats=1, seed=0)
    np.testing.assert_allclose(column, flat)
    with pytest.raises(ValueError, match=r"prediction.*shape \(\d+, 2\)"):
        mda(_WidePredict, X, y, cv, n_repeats=1, seed=0)


def test_mda_and_sfi_refuse_nan_sample_weight() -> None:
    X, y = make_data(n=300, p=3, seed=2)
    w = np.ones(300)
    w[7] = np.nan
    cv = PurgedKFold(n_splits=3)
    with pytest.raises(ValueError, match="NaN"):
        mda(NearestCentroid, X, y, cv, sample_weight=w)
    with pytest.raises(ValueError, match="NaN"):
        sfi(NearestCentroid, X, y, cv, sample_weight=w)


class _FixedSplit:
    def __init__(self, folds, mask=False):
        self.folds = folds
        self.mask = mask

    def split(self, X):
        for tr, te in self.folds:
            yield (np.isin(np.arange(len(X)), tr) if self.mask else tr), te


def test_mda_boolean_mask_splitter_matches_index_splitter() -> None:
    X, y = make_data(n=240, p=3, seed=4)
    folds = list(PurgedKFold(n_splits=3).split(X))
    base_i, imp_i = mda(NearestCentroid, X, y, _FixedSplit(folds), n_repeats=2, seed=1)
    base_m, imp_m = mda(NearestCentroid, X, y, _FixedSplit(folds, mask=True), n_repeats=2, seed=1)
    np.testing.assert_allclose(base_m, base_i)
    np.testing.assert_allclose(imp_m.to_numpy(), imp_i.to_numpy())


def test_mda_accepts_sample_weight_over_the_same_instants_in_another_zone():
    X, y = make_data(n=120, p=3, seed=2)
    utc = pd.date_range("2020-01-01", periods=120, freq="D", tz="UTC")
    Xdf = pd.DataFrame(X, index=utc)
    cv = PurgedKFold(n_splits=3)
    w = np.ones(120)
    base_utc, _ = mda(NearestCentroid, Xdf, y, cv, sample_weight=pd.Series(w, index=utc))
    base_tpe, _ = mda(NearestCentroid, Xdf, y, cv, sample_weight=pd.Series(w, index=utc.tz_convert("Asia/Taipei")))
    np.testing.assert_allclose(base_tpe, base_utc)
    shifted = pd.Series(w, index=(utc + pd.Timedelta(days=1)).tz_convert("Asia/Taipei"))
    with pytest.raises(ValueError, match="sample_weight index"):
        mda(NearestCentroid, Xdf, y, cv, sample_weight=shifted)


class _NoWeightCentroid(NearestCentroid):
    """A model whose fit takes no sample_weight keyword, as KNeighborsClassifier does."""

    def fit(self, X, y):
        return super().fit(X, y)


def test_mda_fits_without_sample_weight() -> None:
    X, y = make_data(n=300, p=3, seed=2)
    baseline, imp = mda(_NoWeightCentroid, X, y, PurgedKFold(n_splits=3))
    assert baseline.shape == (3,)
    assert imp.shape[0] == 3


def test_sfi_fits_without_sample_weight() -> None:
    X, y = make_data(n=300, p=3, seed=2)
    baseline, imp = sfi(_NoWeightCentroid, X, y, PurgedKFold(n_splits=3))
    assert baseline.shape == (3,)
    assert imp.shape[0] == 3


def test_orthogonal_features_treats_pd_na_as_nan() -> None:
    cols = {
        "a": pd.array([1.0, 2.0, None, 4.0, 5.0, 3.0], dtype="Float64"),
        "b": pd.array([2.0, 1.0, 3.0, 0.5, 4.0, 6.0], dtype="Float64"),
    }
    X = pd.DataFrame(cols)
    X_nan = X.astype(float)
    with pytest.raises(ValueError, match="non-finite"):
        OrthogonalFeatures().fit(X_nan)
    with pytest.raises(ValueError, match="non-finite"):
        OrthogonalFeatures().fit(X)
    with pytest.raises(ValueError, match="non-finite"):
        orthogonal_features(X)


def test_orthogonal_transform_maps_pd_na_to_a_nan_row() -> None:
    rng = np.random.default_rng(3)
    clean = pd.DataFrame(rng.normal(size=(50, 2)), columns=["a", "b"]).astype("Float64")
    model = OrthogonalFeatures().fit(clean)
    new = clean.iloc[:3].copy()
    new.iloc[1, 0] = pd.NA
    out = np.asarray(model.transform(new), dtype=float)
    assert np.isnan(out[1]).all()
    assert np.isfinite(out[0]).all() and np.isfinite(out[2]).all()


class _PipelineName:
    pass


# sklearn.pipeline.Pipeline's name and module, which is what finlab checks.
_PipelineName.__name__ = "Pipeline"
_PipelineName.__module__ = "sklearn.pipeline"


class _FakePipeline(NearestCentroid, _PipelineName):
    """Stand-in for sklearn.pipeline.Pipeline: steps, and a fit that refuses sample_weight.

    finlab recognises a Pipeline by class name and module, so the stand-in declares that module.
    """

    def __init__(self):
        self.steps = [("scale", NearestCentroid()), ("clf", NearestCentroid())]
        self.named_steps = dict(self.steps)
        self.fit_params = None

    def fit(self, X, y, **fit_params):
        if "sample_weight" in fit_params:
            raise ValueError("Pipeline.fit does not accept the sample_weight parameter.")
        self.fit_params = fit_params
        return super().fit(X, y)


def test_mda_and_sfi_route_sample_weight_to_pipeline_final_step() -> None:
    X, y = make_data(n=300, p=3, seed=6)
    w = np.random.default_rng(6).uniform(0.5, 1.5, len(y))
    cv = PurgedKFold(n_splits=3)
    for run in (mda, sfi):
        made: list[_FakePipeline] = []

        def factory() -> _FakePipeline:
            model = _FakePipeline()
            made.append(model)
            return model

        run(factory, X, y, cv, sample_weight=w)
        # sfi fits the full-feature model, then one single-feature model per feature, in each fold.
        per_fold = 1 + X.shape[1] if run is sfi else 1
        expected = [w[train] for train, _ in cv.split(X) for _ in range(per_fold)]
        assert len(made) == len(expected)
        for model, want in zip(made, expected):
            assert list(model.fit_params) == ["clf__sample_weight"]
            np.testing.assert_array_equal(model.fit_params["clf__sample_weight"], want)


@pytest.mark.parametrize("run", [mda, sfi])
@pytest.mark.parametrize("final", ["passthrough", None])
def test_mda_and_sfi_refuse_pipeline_ending_in_passthrough_or_none(run, final) -> None:
    class _Ends(_FakePipeline):
        def __init__(self) -> None:
            super().__init__()
            self.steps = [("scale", NearestCentroid()), ("clf", final)]
            self.named_steps = dict(self.steps)

    X, y = make_data(n=120, p=3, seed=7)
    with pytest.raises(ValueError, match="final step"):
        run(_Ends, X, y, PurgedKFold(n_splits=3), sample_weight=np.ones(len(y)))


def test_mda_refuses_pipeline_ending_in_passthrough_when_weighted() -> None:
    class _Ends(_FakePipeline):
        def __init__(self) -> None:
            super().__init__()
            self.steps = [("scale", NearestCentroid()), ("clf", "passthrough")]
            self.named_steps = dict(self.steps)

    X, y = make_data(n=120, p=3, seed=7)
    with pytest.raises(ValueError, match="final step"):
        mda(_Ends, X, y, PurgedKFold(n_splits=3), sample_weight=np.ones(len(y)))


def _sklearn_pipeline_factory():
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler

    return lambda: Pipeline([("scale", StandardScaler()), ("lr", LogisticRegression())])


def test_mda_and_sfi_with_sklearn_pipeline_match_manual_weighted_loop() -> None:
    pytest.importorskip("sklearn")
    X, y = make_data(n=300, p=3, seed=8)
    w = np.random.default_rng(8).uniform(0.5, 1.5, len(y))
    cv = PurgedKFold(n_splits=4)
    factory = _sklearn_pipeline_factory()

    manual_full: list[float] = []
    manual_single = np.zeros((4, 3))
    for fold, (train, test) in enumerate(cv.split(X)):
        model = factory().fit(X[train], y[train], lr__sample_weight=w[train])
        hit = (model.predict(X[test]) == y[test]).astype(float)
        manual_full.append(float(np.sum(hit * w[test]) / np.sum(w[test])))
        for j in range(3):
            single = factory().fit(X[train][:, [j]], y[train], lr__sample_weight=w[train])
            hit = (single.predict(X[test][:, [j]]) == y[test]).astype(float)
            manual_single[fold, j] = float(np.sum(hit * w[test]) / np.sum(w[test]))

    baseline, _ = mda(factory, X, y, cv, sample_weight=w)
    np.testing.assert_array_equal(baseline, manual_full)
    baseline, imp = sfi(factory, X, y, cv, sample_weight=w)
    np.testing.assert_array_equal(baseline, manual_full)
    np.testing.assert_allclose(imp["mean"].to_numpy(), manual_single.mean(axis=0), rtol=0, atol=1e-12)
