"""Bagging with sequential bootstrap for overlapping labels (AFML ch. 6).

Standard bagging draws observations i.i.d., so when labels overlap in time the
bootstrap samples are highly redundant and the ensemble's trees are almost
identical. Sequential bootstrap (Snippet 4.5, see :mod:`finlab.weights`) draws
each observation with probability proportional to its uniqueness given the
observations already drawn, so the bags are more diverse and the out-of-bag
estimates are less optimistic.

:class:`SequentialBootstrapBagging` wraps any estimator that follows the duck-typed
protocol ``fit(X, y, sample_weight=None)`` and ``predict(X)`` (plus optionally
``predict_proba(X)``). There is no scikit-learn dependency.

Scope
-----
Classification with labels as arbitrary hashable values. Each bag is trained on
the rows drawn for it; no sample weights are applied inside the bag unless the
caller passes them to :meth:`fit`.

Not covered
-----------
Random-forest-specific tricks from AFML ch. 6 (e.g. the number of features per
split), and the book's analytic calculation of bagging accuracy (Snippet 6.1), which
is not a Monte Carlo study.
"""

from __future__ import annotations

from typing import Any, Callable

import numpy as np
import numpy.typing as npt
import pandas as pd

from .weights import indicator_matrix, sequential_bootstrap

__all__ = ["SequentialBootstrapBagging"]


class SequentialBootstrapBagging:
    """Bagged ensemble whose bags are drawn by sequential bootstrap.

    Parameters
    ----------
    base_factory : callable
        ``base_factory()`` returns a fresh, unfitted estimator.
    n_estimators : int, default 10
        Number of bags.
    sequential : bool, default True
        Use sequential bootstrap (needs ``t1``). If False, use ordinary i.i.d.
        bootstrap, which is the baseline the book compares against.
    max_samples : int, optional
        Draws per bag. Defaults to the number of observations, as in the book's
        bagging. Set it to ``round(n * mean(average_uniqueness))`` to reproduce the
        book's advice of subsampling to the effective sample size.
    seed : int, optional
        Seed for reproducible bags.

    Attributes
    ----------
    estimators_ : list
        Fitted base estimators, one per bag.
    draws_ : list of np.ndarray
        Row positions drawn for each bag.
    classes_ : np.ndarray
        Sorted unique labels seen in ``fit``.
    """

    def __init__(
        self,
        base_factory: Callable[[], Any],
        n_estimators: int = 10,
        sequential: bool = True,
        max_samples: int | None = None,
        seed: int | None = 0,
    ) -> None:
        if n_estimators < 1:
            raise ValueError(f"n_estimators must be >= 1, got {n_estimators!r}.")
        self.base_factory = base_factory
        self.n_estimators = n_estimators
        self.sequential = sequential
        self.max_samples = max_samples
        self.seed = seed

    def fit(
        self,
        X: npt.ArrayLike | pd.DataFrame,
        y: npt.ArrayLike | pd.Series,
        t1: pd.Series | None = None,
        sample_weight: npt.ArrayLike | None = None,
    ) -> "SequentialBootstrapBagging":
        """Fit ``n_estimators`` bags.

        Parameters
        ----------
        X, y : array-like or pandas
            Features and labels, rows aligned.
        t1 : pd.Series, optional
            Event-end time per row, indexed by row start time (see
            :func:`finlab.cv.make_t1`). Required when ``sequential=True``.
        sample_weight : array-like, optional
            Weights passed to each base estimator's ``fit``, sliced per bag.
        """
        Xa = X.to_numpy() if isinstance(X, pd.DataFrame) else np.asarray(X)
        ya = y.to_numpy() if isinstance(y, pd.Series) else np.asarray(y)
        n = Xa.shape[0]
        if ya.shape[0] != n:
            raise ValueError("X and y must have the same number of rows.")
        w = None if sample_weight is None else np.asarray(sample_weight, dtype=float)
        rng = np.random.default_rng(self.seed)
        size = n if self.max_samples is None else int(self.max_samples)
        if size < 1:
            raise ValueError("max_samples must be >= 1.")

        if self.sequential:
            if t1 is None:
                raise ValueError("t1 is required for sequential bootstrap.")
            if len(t1) != n:
                raise ValueError("t1 must have one entry per row of X.")
            ind = indicator_matrix(t1.index, t1)

        self.classes_ = np.unique(ya)
        self.estimators_: list[Any] = []
        self.draws_: list[np.ndarray] = []
        for _ in range(self.n_estimators):
            seed_b = int(rng.integers(0, 2**31 - 1))
            if self.sequential:
                draws = sequential_bootstrap(ind, n_samples=size, seed=seed_b)
            else:
                draws = np.random.default_rng(seed_b).integers(0, n, size=size)
            est = self.base_factory()
            w_b = None if w is None else w[draws]
            est.fit(Xa[draws], ya[draws], sample_weight=w_b)
            self.estimators_.append(est)
            self.draws_.append(draws)
        return self

    def predict_proba(self, X: npt.ArrayLike | pd.DataFrame) -> np.ndarray:
        """Average the bags' class probabilities; columns follow ``classes_``.

        Requires every base estimator to implement ``predict_proba``; estimators
        whose ``predict_proba`` columns follow their own ``classes_`` attribute
        are re-aligned to ``self.classes_``.
        """
        Xa = X.to_numpy() if isinstance(X, pd.DataFrame) else np.asarray(X)
        total = np.zeros((Xa.shape[0], self.classes_.shape[0]))
        for est in self.estimators_:
            if not hasattr(est, "predict_proba"):
                raise AttributeError("base estimator has no predict_proba.")
            p = np.asarray(est.predict_proba(Xa))
            est_classes = getattr(est, "classes_", self.classes_)
            cols = np.searchsorted(self.classes_, est_classes)
            total[:, cols] += p
        return total / self.n_estimators

    def predict(self, X: npt.ArrayLike | pd.DataFrame) -> np.ndarray:
        """Predict by averaged probability if available, else by majority vote."""
        if all(hasattr(e, "predict_proba") for e in self.estimators_):
            return self.classes_[self.predict_proba(X).argmax(axis=1)]
        Xa = X.to_numpy() if isinstance(X, pd.DataFrame) else np.asarray(X)
        votes = np.stack([np.asarray(e.predict(Xa)) for e in self.estimators_])
        out = np.empty(Xa.shape[0], dtype=self.classes_.dtype)
        for i in range(Xa.shape[0]):
            vals, counts = np.unique(votes[:, i], return_counts=True)
            out[i] = vals[counts.argmax()]
        return out
