"""Feature importance: MDI, MDA, SFI and orthogonal features.

AFML chapter 8 ("Feature Importance").

* :func:`mdi_from_matrix` -- mean decrease impurity from a caller-supplied
  ``(n_trees, n_features)`` matrix of per-tree importances. AFML section 8.3.1,
  snippet 8.2. In-sample (IS) by construction.
* :func:`mda` -- mean decrease accuracy: permute one column of the *test* fold at
  a time on a purged CV and measure the loss of score. AFML section 8.3.2,
  snippet 8.3. Out-of-sample (OOS).
* :func:`sfi` -- single feature importance: a model fit on one feature at a time,
  scored OOS with the same CV. AFML section 8.4.1, snippet 8.4.
* :class:`OrthogonalFeatures` / :func:`orthogonal_features` -- standardize, then
  eigendecompose ``Z'Z``; keep the smallest number of components that explain at
  least ``variance_threshold`` of the variance. AFML section 8.4.2, snippet 8.5.

The estimator protocol is duck-typed and sklearn is never imported. A model is
any object with ``fit(X, y, sample_weight=None)`` and ``predict(X)``. A
``model_factory`` is a zero-argument callable that returns a fresh, unfitted
estimator for every CV fold. Scoring defaults to accuracy on ``predict`` output;
a custom ``scoring(y_true, y_pred, sample_weight) -> float`` may be supplied.
Log-loss and F1 need probabilities or a different definition, so they must be
passed as a custom callable that closes over the model (not supported directly).

Conventions:

* ``std`` in every returned table is the standard error of the mean, computed as
  the sample standard deviation divided by ``sqrt(n)``. This follows the book's
  ``std * n**-0.5``. For MDI, ``n`` is the number of non-missing (non-zero)
  trees, not the total tree count.
* MDA reports the raw drop in score (baseline minus permuted), not the book's
  improvement divided by the maximal score. Ranking is unchanged.

Not covered: the weighted Kendall's tau between importances and PCA ranks
(AFML snippet 8.6), the parallelized-vs-stacked importance design of section
8.5, the cvScore helper from chapter 7, and any tree-model fitting. MDI needs the
caller to extract ``feature_importances_`` from their own forest. Substitution
effects are only diluted through orthogonalization, never corrected.
"""

from __future__ import annotations

import math
import warnings
from typing import Any, Callable, Iterator, Protocol, Sequence

import numpy as np
import numpy.typing as npt
import pandas as pd

from finlab._checks import _same_index, tz_mismatch_message
from finlab._jit import jit

__all__ = [
    "Estimator",
    "OrthogonalFeatures",
    "mda",
    "mdi_from_matrix",
    "orthogonal_features",
    "sfi",
]

Scorer = Callable[..., float]  # scoring(y_true, y_pred, sample_weight) -> float


class Estimator(Protocol):
    """Duck-typed estimator: ``fit(X, y, sample_weight=None)`` and ``predict(X)``."""

    def fit(self, X: Any, y: Any, sample_weight: Any = None) -> Any: ...

    def predict(self, X: Any) -> Any: ...


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------


def _n_features(X: Any) -> int:
    arr_shape = X.shape if hasattr(X, "shape") else np.asarray(X).shape
    if len(arr_shape) != 2:
        raise ValueError(f"X must be 2-dimensional, got shape {arr_shape}.")
    return int(arr_shape[1])


def _feature_names(X: Any, p: int) -> list[Any]:
    if isinstance(X, pd.DataFrame):
        return list(X.columns)
    return list(range(p))


def _take_rows(obj: Any, idx: npt.NDArray[np.int_]) -> Any:
    """Row subset that keeps pandas objects as pandas (so column names survive)."""
    if isinstance(obj, (pd.Series, pd.DataFrame)):
        return obj.iloc[idx]
    return np.asarray(obj)[idx]


def _select_column(X: Any, j: int) -> Any:
    if isinstance(X, pd.DataFrame):
        return X.iloc[:, [j]]
    return np.asarray(X)[:, [j]]


def _permute_column(X_test: Any, j: int, perm: npt.NDArray[np.int_]) -> Any:
    """Copy of ``X_test`` whose column ``j`` is reordered by ``perm``."""
    if isinstance(X_test, pd.DataFrame):
        out = X_test.copy()
        out.iloc[:, j] = X_test.iloc[perm, j].to_numpy()
        return out
    out = np.array(X_test, copy=True)
    out[:, j] = X_test[perm, j]
    return out


def _prediction_vector(y_pred: Any, n: int) -> npt.NDArray[Any]:
    """A (n, 1) prediction as (n,); any other shape is refused.

    Left as (n, 1), it broadcast against y into an n by n comparison and scored near 0.5.
    """
    pred = np.asarray(y_pred)
    if pred.ndim == 2 and pred.shape[1] == 1:
        pred = pred[:, 0]
    if pred.shape != (n,):
        raise ValueError(f"model.predict returned a prediction of shape {pred.shape}; expected ({n},).")
    return pred


def _score(
    scoring: str | Scorer,
    y_true: npt.NDArray[Any],
    y_pred: npt.NDArray[Any],
    sample_weight: npt.NDArray[np.float64] | None,
) -> float:
    if isinstance(scoring, str):
        if scoring != "accuracy":
            raise ValueError(
                f"Unknown scoring {scoring!r}; use 'accuracy' or a callable "
                "scoring(y_true, y_pred, sample_weight) -> float."
            )
        hit = (np.asarray(y_true) == _prediction_vector(y_pred, len(y_true))).astype(np.float64)
        if sample_weight is None:
            return float(hit.mean())
        return float(np.sum(sample_weight * hit) / np.sum(sample_weight))
    return float(scoring(y_true, y_pred, sample_weight))


def _fit(
    model_factory: Callable[[], Estimator],
    X: Any,
    y: Any,
    sample_weight: npt.NDArray[np.float64] | None,
) -> Estimator:
    model = model_factory()
    if sample_weight is None:
        # Some models (KNeighborsClassifier) take no sample_weight keyword at all.
        model.fit(X, y)
    else:
        model.fit(X, y, sample_weight=sample_weight)
    return model


def _positions(idx: Any) -> npt.NDArray[np.intp]:
    arr = np.asarray(idx)
    if arr.dtype == np.bool_:
        return np.flatnonzero(arr).astype(np.intp)
    return arr.astype(np.intp)


def _folds(cv: Any, X: Any) -> Iterator[tuple[npt.NDArray[np.int_], npt.NDArray[np.int_]]]:
    for train, test in cv.split(X):
        # A boolean mask cast to intp became the positions 0 and 1; take the True positions instead.
        yield _positions(train), _positions(test)


def _warn_if_zero_weight(w_te: npt.NDArray[np.float64] | None, fold: int) -> None:
    """Warn when a test fold has no weight, since its weighted accuracy is 0/0 = NaN."""
    if w_te is not None and not np.any(w_te):
        warnings.warn(
            f"fold {fold} has zero total sample weight in its test rows; its score is NaN.",
            UserWarning,
            stacklevel=3,
        )


def _summarize(per_fold: npt.NDArray[np.float64], names: list[Any]) -> pd.DataFrame:
    """Mean and standard error across folds (rows) for each feature (columns)."""
    n = per_fold.shape[0]
    mean = per_fold.mean(axis=0)
    if n > 1:
        se = per_fold.std(axis=0, ddof=1) / math.sqrt(n)
    else:
        se = np.full(per_fold.shape[1], np.nan)
    return pd.DataFrame({"mean": mean, "std": se}, index=names)


def _matches_unlabelled(index: pd.Index, x_index: pd.Index, n: int) -> bool:
    """True for a default 0..n-1 RangeIndex that is matched to a non-integer X index by position."""
    # Value and integer dtype, not the RangeIndex class, decide it.
    return (
        pd.api.types.is_integer_dtype(index)
        and index.equals(pd.RangeIndex(n))
        and not pd.api.types.is_integer_dtype(x_index)
    )


def _check_xy(X: Any, y: Any, sample_weight: Any = None) -> tuple[int, npt.NDArray[Any]]:
    p = _n_features(X)
    y_arr = np.asarray(y)
    if y_arr.ndim == 2 and y_arr.shape[1] == 1:
        # A one-column label is a 1-D label. Kept 2-D, the accuracy broadcast against
        # every prediction and the importances came out zero.
        y_arr = y_arr[:, 0]
    n = np.shape(X)[0]
    if y_arr.shape[0] != n:
        raise ValueError(
            f"X and y disagree on the number of samples: {n} vs {y_arr.shape[0]}."
        )
    if sample_weight is not None and np.shape(sample_weight) != (n,):
        # Checked here because a longer array would be silently truncated by the fold indexing.
        raise ValueError(
            f"sample_weight has shape {np.shape(sample_weight)} but X has {n} rows; "
            f"expected length {n}."
        )
    if sample_weight is not None and np.isnan(np.asarray(sample_weight, dtype=np.float64)).any():
        # A NaN weight on a test row made the weighted score NaN with no message.
        raise ValueError("sample_weight contains NaN; drop or fill those rows before calling.")
    # Labels must line up too: a pandas y or weight is otherwise used by position.
    if isinstance(X, (pd.DataFrame, pd.Series)):
        # A default 0..n-1 y or weight is matched by position against a non-integer X index,
        # as tuning does; an integer X index must match it exactly.
        unlabelled_y = isinstance(y, (pd.DataFrame, pd.Series)) and _matches_unlabelled(y.index, X.index, n)
        if isinstance(y, (pd.DataFrame, pd.Series)) and not _same_index(y.index, X.index) and not unlabelled_y:
            raise ValueError(
                tz_mismatch_message(y.index, X.index, "y", "X.index")
                or "y index does not match X.index; align y to X before calling."
            )
        if isinstance(sample_weight, (pd.DataFrame, pd.Series)) and not _same_index(
            sample_weight.index, X.index
        ) and not _matches_unlabelled(sample_weight.index, X.index, n):
            raise ValueError(
                tz_mismatch_message(sample_weight.index, X.index, "sample_weight", "X.index")
                or "sample_weight index does not match X.index; align sample_weight to X before calling."
            )
    return p, y_arr


# ---------------------------------------------------------------------------
# MDI (AFML 8.3.1, snippet 8.2)
# ---------------------------------------------------------------------------


@jit
def _mdi_column_stats(imp: npt.NDArray[np.float64]) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    """Per-column mean and standard error, skipping zeros and NaNs.

    A zero means the feature was not drawn at that tree (``max_features=1``), so it
    is treated as missing (AFML 8.3.1, note 1b). Two passes give a numerically
    stable variance.
    """
    n, p = imp.shape
    mean = np.empty(p)
    se = np.empty(p)
    for j in range(p):
        total = 0.0
        count = 0
        for i in range(n):
            v = imp[i, j]
            if v != 0.0 and not np.isnan(v):
                total += v
                count += 1
        if count == 0:
            mean[j] = np.nan
            se[j] = np.nan
            continue
        m = total / count
        ss = 0.0
        for i in range(n):
            v = imp[i, j]
            if v != 0.0 and not np.isnan(v):
                d = v - m
                ss += d * d
        mean[j] = m
        if count > 1:
            se[j] = math.sqrt(ss / (count - 1)) / math.sqrt(count)
        else:
            se[j] = np.nan
    return mean, se


def mdi_from_matrix(imp: Any, names: Sequence[Any] | None = None) -> pd.DataFrame:
    """Mean decrease impurity from a per-tree importance matrix.

    Parameters
    ----------
    imp : array-like or pd.DataFrame, shape (n_trees, n_features)
        ``imp[t, j]`` is the impurity decrease attributed to feature ``j`` by tree
        ``t`` (for example ``tree.feature_importances_`` of each fitted tree). Zeros
        are treated as "feature not drawn" and excluded. NaN is also excluded.
    names : sequence, optional
        Feature names. Defaults to the DataFrame columns, else ``0..n_features-1``.

    Returns
    -------
    pd.DataFrame
        Indexed by feature, with columns ``mean`` and ``std``. Both are normalized
        so that the means sum to 1 over features with a finite mean. ``std`` is the
        standard error of the normalized mean.

    Notes
    -----
    MDI is in-sample (IS). The importances are computed on the data the trees were
    fit on, so every feature gets some importance even if it has no predictive
    power. Do not use MDI alone as evidence of out-of-sample relevance. MDI is also
    diluted by substitute features (identical features each get about half the
    importance). Strobl et al. (2007) show a bias toward high-cardinality features.
    """
    if isinstance(imp, pd.DataFrame):
        names = list(imp.columns) if names is None else list(names)
        arr = imp.to_numpy(dtype=np.float64)
    else:
        arr = np.asarray(imp, dtype=np.float64)
    if arr.ndim != 2:
        raise ValueError(f"imp must be 2-dimensional (n_trees, n_features), got {arr.shape}.")
    p = arr.shape[1]
    names = list(range(p)) if names is None else list(names)
    if len(names) != p:
        raise ValueError(f"Got {len(names)} names for {p} features.")
    # Infinity would turn its column into NaN and zero out the others after normalizing.
    if np.isinf(arr).any():
        raise ValueError("imp contains infinite values; impurity decreases must be finite.")
    finite_vals = arr[~np.isnan(arr)]
    absmax = float(np.max(np.abs(finite_vals))) if finite_vals.size else 0.0
    # A negative within float rounding of the largest value (1e-14 relative, about 45 ulp) is a zero
    # decrease from rounding, not a real negative. It is set to the
    # smallest positive normal, so the feature still counts as drawn, as before, and the means match
    # main to rounding.
    rounding = (arr < 0) & (arr >= -1e-14 * absmax)
    if ((arr < 0) & ~rounding).any():
        # A tree's impurity decrease cannot be negative; averaging one in credits the feature wrongly.
        raise ValueError("imp contains negative impurity decreases; they must be >= 0.")

    # Work in units of the largest impurity. The column sums of raw values overflowed for
    # impurities near 1e308, and the ratios that give the result are unchanged by the scaling.
    # The rounding-level zero is set after the scaling: set before it, a tiny value divided by
    # a large scale underflowed to 0, which dropped the feature's draw and shifted the means.
    scale = absmax if absmax > 0.0 else 1.0
    scaled = np.where(rounding, np.finfo(np.float64).tiny, arr / scale)
    # A positive impurity far below the peak can divide to 0.0, which _mdi_column_stats reads as
    # "not drawn". It was drawn, so it keeps the smallest positive normal, as a rounding-level zero does.
    drawn = (arr != 0.0) & ~np.isnan(arr)
    scaled = np.where(drawn & (scaled == 0.0), np.finfo(np.float64).tiny, scaled)
    mean, se = _mdi_column_stats(np.ascontiguousarray(scaled))
    # Scale by the peak before summing: a sum of finite means can overflow to inf, which made
    # every normalized mean 0.0. The ratios are unchanged by the scaling.
    finite_mean = mean[np.isfinite(mean)]
    peak = float(finite_mean.max()) if finite_mean.size else 0.0
    if not peak > 0.0:
        raise ValueError("No positive MDI importance found; cannot normalize.")
    unit = mean / peak
    total = float(np.nansum(unit))
    return pd.DataFrame({"mean": unit / total, "std": (se / peak) / total}, index=names)


# ---------------------------------------------------------------------------
# MDA (AFML 8.3.2, snippet 8.3)
# ---------------------------------------------------------------------------


def mda(
    model_factory: Callable[[], Estimator],
    X: Any,
    y: Any,
    cv: Any,
    sample_weight: Any = None,
    scoring: str | Scorer = "accuracy",
    n_repeats: int = 1,
    seed: int = 0,
) -> tuple[npt.NDArray[np.float64], pd.DataFrame]:
    """Mean decrease accuracy (permutation importance) over a CV.

    For each fold, the model is fit on the training rows. Its score on the test rows
    is the baseline. Then each feature column of the test rows is permuted on its own
    (``n_repeats`` times) and the score drop is recorded.

    Parameters
    ----------
    model_factory : callable
        Zero-argument callable returning a fresh unfitted estimator.
    X : array-like or pd.DataFrame, shape (n_samples, n_features)
    y : array-like or pd.Series, shape (n_samples,)
    cv : splitter
        Any object with ``split(X)`` yielding ``(train_idx, test_idx)``, such as
        :class:`finlab.cv.PurgedKFold` or :class:`finlab.cv.CombinatorialPurgedCV`.
        Use a purged, embargoed splitter (AFML chapter 7).
    sample_weight : array-like, optional
        Per-sample weights, used in fitting and in weighted scoring.
    scoring : {'accuracy'} or callable, default 'accuracy'
        ``'accuracy'`` is the weighted fraction of correct ``predict`` outputs. A
        callable is ``scoring(y_true, y_pred, sample_weight) -> float`` and is
        higher-is-better.
    n_repeats : int, default 1
        Number of independent permutations per feature per fold. Their drops are
        averaged within each fold.
    seed : int, default 0
        Seed for ``numpy.random.default_rng``. Draws happen in the order fold, then
        repeat, then feature, so results are reproducible.

    Returns
    -------
    baseline : np.ndarray, shape (n_folds,)
        Unpermuted OOS score per fold.
    importance : pd.DataFrame
        Indexed by feature, with columns ``mean`` (average score drop, baseline minus
        permuted; positive means the feature helps) and ``std`` (standard error across
        folds).

    Notes
    -----
    MDA is OOS, so it can conclude that all features are unimportant. It is also
    susceptible to substitution: two identical features make each look irrelevant,
    because permuting one leaves the other intact. Use :func:`orthogonal_features`
    first to reduce this. The drop is a ranking, not a calibrated effect size, when
    the scoring callable is not a metric.
    """
    if n_repeats < 1:
        raise ValueError(f"n_repeats must be >= 1, got {n_repeats}.")
    p, y_arr = _check_xy(X, y, sample_weight)
    w_arr = None if sample_weight is None else np.asarray(sample_weight, dtype=np.float64)
    names = _feature_names(X, p)
    rng = np.random.default_rng(seed)

    baseline: list[float] = []
    drops: list[npt.NDArray[np.float64]] = []
    for fold, (train, test) in enumerate(_folds(cv, X)):
        X_tr, y_tr = _take_rows(X, train), y_arr[train]
        X_te, y_te = _take_rows(X, test), y_arr[test]
        w_tr = None if w_arr is None else w_arr[train]
        w_te = None if w_arr is None else w_arr[test]
        _warn_if_zero_weight(w_te, fold)

        model = _fit(model_factory, X_tr, y_tr, w_tr)
        base = _score(scoring, y_te, np.asarray(model.predict(X_te)), w_te)
        baseline.append(base)

        fold_drop = np.zeros(p, dtype=np.float64)
        for _ in range(n_repeats):
            for j in range(p):
                perm = rng.permutation(len(test))
                X_perm = _permute_column(X_te, j, perm)
                permuted = _score(scoring, y_te, np.asarray(model.predict(X_perm)), w_te)
                fold_drop[j] += base - permuted
        drops.append(fold_drop / n_repeats)

    if not drops:
        raise ValueError("cv produced no folds.")
    return np.asarray(baseline, dtype=np.float64), _summarize(np.vstack(drops), names)


# ---------------------------------------------------------------------------
# SFI (AFML 8.4.1, snippet 8.4)
# ---------------------------------------------------------------------------


def sfi(
    model_factory: Callable[[], Estimator],
    X: Any,
    y: Any,
    cv: Any,
    sample_weight: Any = None,
    scoring: str | Scorer = "accuracy",
) -> tuple[npt.NDArray[np.float64], pd.DataFrame]:
    """Single feature importance: OOS score of a model using one feature at a time.

    Parameters
    ----------
    model_factory, X, y, cv, sample_weight, scoring
        As in :func:`mda`.

    Returns
    -------
    baseline : np.ndarray, shape (n_folds,)
        OOS score per fold of the model that uses *all* features. It is the reference
        point for judging the single-feature scores.
    importance : pd.DataFrame
        Indexed by feature, with columns ``mean`` (average OOS score of the
        single-feature model) and ``std`` (standard error across folds). Higher means
        more informative on its own. These are scores, not drops.

    Notes
    -----
    SFI has no substitution effects, since each model sees only one feature. It
    misses joint and hierarchical effects. A feature that helps only in combination
    with another can score at chance. A score at chance is not evidence the feature
    is useless.
    """
    p, y_arr = _check_xy(X, y, sample_weight)
    w_arr = None if sample_weight is None else np.asarray(sample_weight, dtype=np.float64)
    names = _feature_names(X, p)

    baseline: list[float] = []
    per_feature: list[npt.NDArray[np.float64]] = []
    for fold, (train, test) in enumerate(_folds(cv, X)):
        y_tr, y_te = y_arr[train], y_arr[test]
        w_tr = None if w_arr is None else w_arr[train]
        w_te = None if w_arr is None else w_arr[test]
        _warn_if_zero_weight(w_te, fold)

        X_tr_all, X_te_all = _take_rows(X, train), _take_rows(X, test)
        full = _fit(model_factory, X_tr_all, y_tr, w_tr)
        baseline.append(_score(scoring, y_te, np.asarray(full.predict(X_te_all)), w_te))

        row = np.empty(p, dtype=np.float64)
        for j in range(p):
            X_tr_j = _select_column(X_tr_all, j)
            X_te_j = _select_column(X_te_all, j)
            model = _fit(model_factory, X_tr_j, y_tr, w_tr)
            row[j] = _score(scoring, y_te, np.asarray(model.predict(X_te_j)), w_te)
        per_feature.append(row)

    if not per_feature:
        raise ValueError("cv produced no folds.")
    return np.asarray(baseline, dtype=np.float64), _summarize(np.vstack(per_feature), names)


# ---------------------------------------------------------------------------
# Orthogonal features (AFML 8.4.2, snippet 8.5)
# ---------------------------------------------------------------------------


def _float_values(X: Any) -> npt.NDArray[np.float64]:
    """Float values of X, with pd.NA read as NaN so the NaN refusal applies."""
    if isinstance(X, (pd.DataFrame, pd.Series)):
        return X.to_numpy(dtype=np.float64, na_value=np.nan)
    return np.asarray(X, dtype=np.float64)


class OrthogonalFeatures:
    """PCA orthogonalization that can be fit on one sample and applied to another.

    Parameters
    ----------
    variance_threshold : float, default 0.95
        Keep the smallest number of components whose cumulative explained-variance
        ratio reaches this value. The component that crosses the threshold is kept.
        Must lie in (0, 1].

    Attributes
    ----------
    mean_, std_ : np.ndarray, shape (n_features,)
        Training mean and sample standard deviation (ddof=1) per feature.
    eigenvalues_ : np.ndarray, shape (n_components,)
        Eigenvalues of ``Z'Z`` in descending order (AFML's Lambda). The variance of
        component ``k`` in the training sample is ``eigenvalues_[k] / (n - 1)``.
    eigenvectors_ : np.ndarray, shape (n_features, n_components)
        Orthonormal columns (AFML's W, truncated).
    explained_variance_ratio_ : np.ndarray, shape (n_components,)
        Share of total variance explained by each kept component.
    n_components_ : int
        Number of kept components.

    Notes
    -----
    Standardization uses the training mean and sample standard deviation. The same
    transform is therefore applied to new data without refitting. Constant columns
    have zero standard deviation and are rejected, because they cannot be scaled.
    """

    def __init__(self, variance_threshold: float = 0.95) -> None:
        if not 0.0 < variance_threshold <= 1.0:
            raise ValueError(f"variance_threshold must be in (0, 1], got {variance_threshold!r}.")
        self.variance_threshold = variance_threshold

    def fit(self, X: Any, y: Any = None) -> OrthogonalFeatures:
        arr = _float_values(X)
        if arr.ndim != 2:
            raise ValueError(f"X must be 2-dimensional, got shape {arr.shape}.")
        n, p = arr.shape
        if n < 2:
            raise ValueError("At least 2 samples are required.")
        mean = arr.mean(axis=0)
        std = arr.std(axis=0, ddof=1)
        if not np.all(np.isfinite(std)) or np.any(std <= 0.0):
            raise ValueError("Constant or non-finite feature columns cannot be standardized.")
        Z = (arr - mean) / std
        dot = Z.T @ Z
        eigval, eigvec = np.linalg.eigh(dot)
        order = np.argsort(eigval)[::-1]
        eigval = eigval[order]
        eigvec = eigvec[:, order]

        clipped = np.clip(eigval, 0.0, None)  # tiny negatives are rounding noise
        total = clipped.sum()
        cum = np.cumsum(clipped) / total
        k = min(int(np.searchsorted(cum, self.variance_threshold, side="left")) + 1, p)

        self.mean_ = mean
        self.std_ = std
        self.eigenvalues_ = eigval[:k].copy()
        self.eigenvectors_ = eigvec[:, :k].copy()
        self.explained_variance_ratio_ = clipped[:k] / total
        self.n_components_ = k
        self.feature_names_in_ = _feature_names(X, p)
        self._names_from_frame = isinstance(X, pd.DataFrame)
        return self

    def transform(self, X: Any) -> Any:
        """Project ``X`` onto the fitted components.

        Returns a DataFrame with ``PC_1..PC_k`` columns if ``X`` is a DataFrame,
        otherwise an ndarray.
        """
        if not hasattr(self, "eigenvectors_"):
            raise RuntimeError("OrthogonalFeatures is not fitted; call fit first.")
        arr = _float_values(X)
        if arr.ndim != 2 or arr.shape[1] != self.mean_.shape[0]:
            raise ValueError(f"Expected {self.mean_.shape[0]} features, got shape {arr.shape}.")
        # Columns are matched by position, so a reordered frame would give silently
        # wrong features. When the fit saw named columns, the names must match in the same order.
        if isinstance(X, pd.DataFrame) and getattr(self, "_names_from_frame", False):
            if list(X.columns) != list(self.feature_names_in_):
                raise ValueError("X columns must match the fitted feature names, in the same order.")
        P = ((arr - self.mean_) / self.std_) @ self.eigenvectors_
        if isinstance(X, pd.DataFrame):
            cols = [f"PC_{k + 1}" for k in range(P.shape[1])]
            return pd.DataFrame(P, index=X.index, columns=cols)
        return P

    def fit_transform(self, X: Any, y: Any = None) -> Any:
        return self.fit(X, y).transform(X)


def orthogonal_features(
    X: Any, variance_threshold: float = 0.95
) -> tuple[Any, pd.Series, pd.DataFrame]:
    """Orthogonalize features with PCA on standardized data (AFML snippet 8.5).

    Parameters
    ----------
    X : array-like or pd.DataFrame, shape (n_samples, n_features)
    variance_threshold : float, default 0.95
        See :class:`OrthogonalFeatures`.

    Returns
    -------
    P : pd.DataFrame or np.ndarray, shape (n_samples, n_components)
        Orthogonal features ``P = Z W``. Same kind as ``X``.
    eigenvalues : pd.Series
        Kept eigenvalues of ``Z'Z``, indexed ``PC_1..PC_k``.
    eigenvectors : pd.DataFrame
        Kept eigenvectors, indexed by feature (for DataFrame input) with columns
        ``PC_1..PC_k``.

    Notes
    -----
    Proved in ``docs/proofs/importance.md``: ``P'P = Lambda``, so the columns of
    ``P`` are uncorrelated, and the kept components explain at least the threshold.
    """
    model = OrthogonalFeatures(variance_threshold).fit(X)
    P = model.transform(X)
    k = model.n_components_
    pcs = [f"PC_{i + 1}" for i in range(k)]
    names = _feature_names(X, model.mean_.shape[0])
    eigenvalues = pd.Series(model.eigenvalues_, index=pcs)
    eigenvectors = pd.DataFrame(model.eigenvectors_, index=names, columns=pcs)
    return P, eigenvalues, eigenvectors
