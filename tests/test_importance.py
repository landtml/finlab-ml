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
