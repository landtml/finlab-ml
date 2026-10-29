"""Hyper-parameter tuning with purged cross-validation (AFML ch. 9).

Grid and randomized search over a model factory, where every candidate is scored
by a purged, embargoed splitter (see :mod:`quantowl.cv`) rather than by a plain
K-fold. Using the leakage-free splitter inside the search prevents the tuner from
selecting hyper-parameters that only look good because of label overlap.

The model protocol is duck-typed: ``model_factory(**params)`` returns an object
with ``fit(X, y, sample_weight=None)`` and ``predict(X)``. Scoring defaults to
accuracy and accepts any ``scoring(y_true, y_pred, sample_weight) -> float``.

Scope
-----
Each candidate is evaluated over every split the CV yields and the mean score is
reported. The sample weights are passed to ``fit`` and to the scorer, so
concurrency-aware weights from :mod:`quantowl.weights` are respected.

Not covered
-----------
Nested CV for an unbiased estimate of the tuned model's performance (AFML ch. 9
discusses it; run an outer splitter yourself), and Bayesian search.
"""

from __future__ import annotations

import itertools
import warnings
from dataclasses import dataclass
from typing import Any, Callable, Iterable, Mapping, Sequence

import numpy as np
import numpy.typing as npt
import pandas as pd

from ._checks import _same_index, tz_mismatch_message
from ._fit import fit_model
from .importance import _prediction_vector

__all__ = ["cv_score", "grid_search", "randomized_search", "SearchResult"]

Scorer = Callable[[npt.NDArray, npt.NDArray, "npt.NDArray | None"], float]


def _accuracy(y_true: npt.NDArray, y_pred: npt.NDArray, sample_weight=None) -> float:
    hit = (np.asarray(y_true) == _prediction_vector(y_pred, len(y_true))).astype(float)
    if sample_weight is None:
        return float(hit.mean())
    w = np.asarray(sample_weight, dtype=float)
    return float((hit * w).sum() / w.sum())


@dataclass(frozen=True)
class SearchResult:
    """Outcome of a hyper-parameter search.

    Attributes
    ----------
    best_params : dict
        Parameters of the highest mean-scoring candidate.
    best_score : float
        Its mean score over the CV splits.
    results : pd.DataFrame
        One row per candidate: its parameters, ``mean_score``, ``std_score``.
    """

    best_params: dict[str, Any]
    best_score: float
    results: pd.DataFrame


def _unlabelled(index: pd.Index, n: int) -> bool:
    """True for an unlabelled 0..n-1 index, which carries no labels to compare.

    The test is on the values and the integer dtype, not the class: an Int64 Index 0..n-1 is the same
    unlabelled index as a RangeIndex.
    """
    return pd.api.types.is_integer_dtype(index) and index.equals(pd.RangeIndex(n))


def _matches_by_position(index: pd.Index, x_index: pd.Index, n: int) -> bool:
    """True when an unlabelled ``index`` may be matched to X by position.

    That holds for a default RangeIndex on y or weights against a non-integer X index, such as dates
    An integer X index names its rows, so an unlabelled 0..n-1 index must match it exactly:
    a shifted RangeIndex X is refused.
    """
    return _unlabelled(index, n) and not pd.api.types.is_integer_dtype(x_index)


def _as_arrays(X, y, sample_weight):
    Xa = X.to_numpy() if isinstance(X, (pd.DataFrame, pd.Series)) else np.asarray(X)
    ya = y.to_numpy() if isinstance(y, (pd.DataFrame, pd.Series)) else np.asarray(y)
    if ya.ndim == 2 and ya.shape[1] == 1:
        # A one-column label is a 1-D label. Kept 2-D, every row's label was compared with
        # every prediction by broadcasting, which changed the selected parameters.
        ya = ya[:, 0]
    if ya.ndim == 2:
        # A multi-column label gave a numpy broadcast error; it is refused by name.
        raise ValueError(f"y must be one column (1-D labels), got shape {ya.shape}.")
    # Labels and weights are used by position in the CV loop, so a pandas y or weight
    # whose index differs from X.index would silently change the selected parameters.
    if isinstance(X, (pd.DataFrame, pd.Series)):
        # A default 0..n-1 RangeIndex carries no labels, so it is matched by position when X is labelled
        # by dates or other non-integer labels. Integer labels, including a shifted RangeIndex,
        # must match X.index. ensemble refuses an unlabelled t1 against a labelled X.
        n_rows = len(Xa)
        if (
            isinstance(y, (pd.DataFrame, pd.Series))
            and not _same_index(y.index, X.index)
            and not _matches_by_position(y.index, X.index, n_rows)
        ):
            raise ValueError(
                tz_mismatch_message(y.index, X.index, "y", "X.index")
                or "y index does not match X.index; align y to X before calling."
            )
        if (
            isinstance(sample_weight, (pd.DataFrame, pd.Series))
            and not _same_index(sample_weight.index, X.index)
            and not _matches_by_position(sample_weight.index, X.index, n_rows)
        ):
            raise ValueError(
                tz_mismatch_message(sample_weight.index, X.index, "sample_weight", "X.index")
                or "sample_weight index does not match X.index; align sample_weight to X before calling."
            )
    # None or a scalar has no rows to compare, so reject it before the shape check.
    if ya.ndim == 0:
        raise ValueError(f"y must be array-like with one label per row, got {y!r}.")
    # A longer y would be silently truncated by the row indexing in the CV loop.
    if ya.shape[0] != Xa.shape[0]:
        raise ValueError(
            f"y has {ya.shape[0]} labels but X has {Xa.shape[0]} rows; expected length {Xa.shape[0]}."
        )
    if sample_weight is not None:
        sample_weight = np.asarray(sample_weight, dtype=float)
        # A longer array would be silently truncated by the row indexing below.
        if sample_weight.shape != (Xa.shape[0],):
            raise ValueError(
                f"sample_weight has shape {sample_weight.shape} but X has {Xa.shape[0]} rows; "
                f"expected length {Xa.shape[0]}."
            )
        if np.isnan(sample_weight).any():
            # A NaN weight on a test row made every score NaN, and the search error did not say why.
            raise ValueError("sample_weight contains NaN; drop or fill those rows before calling.")
    return Xa, ya, sample_weight


def _materialize_splits(cv, X) -> list[tuple[npt.NDArray, npt.NDArray]]:
    """Index pairs of ``cv`` as a list, so a one-shot iterator can be reused.

    Pandas input goes to the splitter unchanged so a datetime-indexed t1 resolves
    against the real index; numpy input keeps its positional RangeIndex.
    """
    if hasattr(cv, "split"):
        split_input = X if isinstance(X, (pd.DataFrame, pd.Series)) else np.asarray(X)
        return list(cv.split(split_input))
    return list(cv)


def _fold_score(value: Any, fold: int) -> float:
    """One fold's score as a float. The scorer must return one number per fold.

    NaN is allowed: the built-in accuracy gives NaN for a zero-weight test fold.
    """
    if isinstance(value, (str, bytes)):
        raise ValueError(f"the scorer must return one number per fold; fold {fold} returned {type(value).__name__}")
    try:
        arr = np.asarray(value, dtype=float)
    except (TypeError, ValueError):
        raise ValueError(
            f"the scorer must return one number per fold; fold {fold} returned {type(value).__name__}"
        ) from None
    if arr.ndim != 0:
        raise ValueError(
            f"the scorer must return one number per fold; fold {fold} returned an array of shape {arr.shape}"
        )
    if np.isinf(arr):
        raise ValueError(f"the scorer must return one number per fold; fold {fold} returned {float(arr)}")
    return float(arr)


def cv_score(
    model_factory: Callable[..., Any],
    params: Mapping[str, Any],
    X,
    y,
    cv,
    sample_weight=None,
    scoring: Scorer | None = None,
) -> tuple[float, float]:
    """Mean and standard deviation of one candidate's score over ``cv`` splits.

    Parameters
    ----------
    model_factory : callable
        ``model_factory(**params)`` returns a fresh, unfitted model.
    params : mapping
        Hyper-parameters for this candidate.
    X, y : array-like or pandas
        Features and labels.
    cv : iterable of (train_idx, test_idx)
        Any purged splitter from :mod:`quantowl.cv`, or a list of index pairs.
    sample_weight : array-like, optional
        Per-sample weights used for fitting and scoring.
    scoring : callable, optional
        ``scoring(y_true, y_pred, sample_weight)``; defaults to accuracy.
    """
    Xa, ya, w = _as_arrays(X, y, sample_weight)
    score_fn = scoring or _accuracy
    scores = []
    for fold, (train_idx, test_idx) in enumerate(_materialize_splits(cv, X)):
        model = model_factory(**params)
        w_tr = None if w is None else w[train_idx]
        fit_model(model, Xa[train_idx], ya[train_idx], w_tr)
        pred = model.predict(Xa[test_idx])
        w_te = None if w is None else w[test_idx]
        if w_te is not None and not np.any(w_te):
            warnings.warn(
                f"fold {fold} has zero total sample weight in its test rows; its score is NaN.",
                UserWarning,
                stacklevel=2,
            )
        scores.append(_fold_score(score_fn(ya[test_idx], pred, w_te), fold))
    arr = np.asarray(scores, dtype=float)
    if arr.size == 0:
        raise ValueError("the cross-validator yielded no splits.")
    return float(arr.mean()), float(arr.std(ddof=0))


def _improves(mean: float, incumbent: float) -> bool:
    """Whether ``mean`` should replace the current best score.

    A NaN score never replaces a finite one, since ``x > NaN`` is always False and a
    leading NaN would otherwise stay best forever. A finite score replaces a NaN one.
    Ties keep the earlier candidate.
    """
    if np.isnan(mean):
        return False
    if np.isnan(incumbent):
        return True
    return mean > incumbent


def _search(
    model_factory: Callable[..., Any],
    candidates: Iterable[dict[str, Any]],
    X,
    y,
    cv,
    sample_weight,
    scoring,
) -> SearchResult:
    # Materialized once so a generator or iterator cv is shared by every candidate.
    splits = _materialize_splits(cv, X)
    rows = []
    best: tuple[float, dict[str, Any]] | None = None
    for params in candidates:
        clash = sorted({"mean_score", "std_score"} & set(params))
        if clash:
            # The result columns would overwrite this parameter's value in the table.
            raise ValueError(f"parameter name {clash[0]!r} collides with a result column; rename it.")
        mean, std = cv_score(model_factory, params, X, y, splits, sample_weight, scoring)
        rows.append({**params, "mean_score": mean, "std_score": std})
        if best is None or _improves(mean, best[0]):
            best = (mean, params)
    if best is None:
        raise ValueError("the parameter space is empty.")
    if np.isnan(best[0]):
        # Every candidate scored NaN, so there is no "highest mean-scoring" candidate to return.
        raise ValueError("no candidate has a finite mean score; check the scoring and the folds.")
    frame = pd.DataFrame(rows)
    return SearchResult(best_params=dict(best[1]), best_score=best[0], results=frame)


def _check_value_lists(mapping: Mapping[str, Any]) -> None:
    """Reject a str or bytes value, which would otherwise be iterated character by character."""
    for key, values in mapping.items():
        if isinstance(values, (str, bytes)):
            raise TypeError(
                f"values for {key!r} must be a list, tuple or other non-string sequence, "
                f"not {type(values).__name__}; wrap a single string as [{values!r}]."
            )


def grid_search(
    model_factory: Callable[..., Any],
    param_grid: Mapping[str, Sequence[Any]],
    X,
    y,
    cv,
    sample_weight=None,
    scoring: Scorer | None = None,
) -> SearchResult:
    """Exhaustive search over the Cartesian product of ``param_grid``.

    Ties are resolved in favour of the candidate that appears first in the grid.
    """
    if not param_grid:
        raise ValueError("param_grid must not be empty.")
    _check_value_lists(param_grid)
    keys = list(param_grid)
    combos = (dict(zip(keys, vals)) for vals in itertools.product(*(param_grid[k] for k in keys)))
    return _search(model_factory, combos, X, y, cv, sample_weight, scoring)


def randomized_search(
    model_factory: Callable[..., Any],
    param_distributions: Mapping[str, Sequence[Any]],
    n_iter: int,
    X,
    y,
    cv,
    sample_weight=None,
    scoring: Scorer | None = None,
    seed: int | None = 0,
) -> SearchResult:
    """Sample ``n_iter`` candidates uniformly from each parameter's list of values.

    Candidates are drawn with replacement, so ``n_iter`` may exceed the grid size.
    """
    if n_iter < 1:
        raise ValueError(f"n_iter must be >= 1, got {n_iter!r}.")
    _check_value_lists(param_distributions)
    for key, values in param_distributions.items():
        if len(values) == 0:
            # rng.integers(0) would fail with "high <= 0", so an empty list is refused here.
            raise ValueError(f"param_distributions[{key!r}] must not be empty.")
    rng = np.random.default_rng(seed)
    keys = list(param_distributions)

    def draws():
        for _ in range(n_iter):
            yield {k: param_distributions[k][rng.integers(len(param_distributions[k]))]
                   for k in keys}

    return _search(model_factory, draws(), X, y, cv, sample_weight, scoring)
