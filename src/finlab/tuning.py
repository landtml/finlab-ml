"""Hyper-parameter tuning with purged cross-validation (AFML ch. 9).

Grid and randomized search over a model factory, where every candidate is scored
by a purged, embargoed splitter (see :mod:`finlab.cv`) rather than by a plain
K-fold. Using the leakage-free splitter inside the search prevents the tuner from
selecting hyper-parameters that only look good because of label overlap.

The model protocol is duck-typed: ``model_factory(**params)`` returns an object
with ``fit(X, y, sample_weight=None)`` and ``predict(X)``. Scoring defaults to
accuracy and accepts any ``scoring(y_true, y_pred, sample_weight) -> float``.

Scope
-----
Each candidate is evaluated over every split the CV yields and the mean score is
reported. The sample weights are passed to ``fit`` and to the scorer, so
concurrency-aware weights from :mod:`finlab.weights` are respected.

Not covered
-----------
Nested CV for an unbiased estimate of the tuned model's performance (AFML ch. 9
discusses it; run an outer splitter yourself), and Bayesian search.
"""

from __future__ import annotations

import itertools
from dataclasses import dataclass
from typing import Any, Callable, Iterable, Mapping, Sequence

import numpy as np
import numpy.typing as npt
import pandas as pd

__all__ = ["cv_score", "grid_search", "randomized_search", "SearchResult"]

Scorer = Callable[[npt.NDArray, npt.NDArray, "npt.NDArray | None"], float]


def _accuracy(y_true: npt.NDArray, y_pred: npt.NDArray, sample_weight=None) -> float:
    hit = (np.asarray(y_true) == np.asarray(y_pred)).astype(float)
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


def _as_arrays(X, y, sample_weight):
    Xa = X.to_numpy() if isinstance(X, (pd.DataFrame, pd.Series)) else np.asarray(X)
    ya = y.to_numpy() if isinstance(y, (pd.DataFrame, pd.Series)) else np.asarray(y)
    if sample_weight is not None:
        sample_weight = np.asarray(sample_weight, dtype=float)
    return Xa, ya, sample_weight


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
        Any purged splitter from :mod:`finlab.cv`, or a list of index pairs.
    sample_weight : array-like, optional
        Per-sample weights used for fitting and scoring.
    scoring : callable, optional
        ``scoring(y_true, y_pred, sample_weight)``; defaults to accuracy.
    """
    Xa, ya, w = _as_arrays(X, y, sample_weight)
    score_fn = scoring or _accuracy
    scores = []
    for train_idx, test_idx in cv.split(Xa) if hasattr(cv, "split") else cv:
        model = model_factory(**params)
        w_tr = None if w is None else w[train_idx]
        model.fit(Xa[train_idx], ya[train_idx], sample_weight=w_tr)
        pred = model.predict(Xa[test_idx])
        w_te = None if w is None else w[test_idx]
        scores.append(score_fn(ya[test_idx], pred, w_te))
    arr = np.asarray(scores, dtype=float)
    if arr.size == 0:
        raise ValueError("the cross-validator yielded no splits.")
    return float(arr.mean()), float(arr.std(ddof=0))


def _search(
    model_factory: Callable[..., Any],
    candidates: Iterable[dict[str, Any]],
    X,
    y,
    cv,
    sample_weight,
    scoring,
) -> SearchResult:
    rows = []
    best: tuple[float, dict[str, Any]] | None = None
    for params in candidates:
        mean, std = cv_score(model_factory, params, X, y, cv, sample_weight, scoring)
        rows.append({**params, "mean_score": mean, "std_score": std})
        if best is None or mean > best[0]:
            best = (mean, params)
    if best is None:
        raise ValueError("the parameter space is empty.")
    frame = pd.DataFrame(rows)
    return SearchResult(best_params=dict(best[1]), best_score=best[0], results=frame)


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
    rng = np.random.default_rng(seed)
    keys = list(param_distributions)

    def draws():
        for _ in range(n_iter):
            yield {k: param_distributions[k][rng.integers(len(param_distributions[k]))]
                   for k in keys}

    return _search(model_factory, draws(), X, y, cv, sample_weight, scoring)
