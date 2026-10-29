"""Bagging with sequential bootstrap for overlapping labels (AFML ch. 6).

Standard bagging draws observations i.i.d., so when labels overlap in time the
bootstrap samples are highly redundant and the ensemble's trees are almost
identical. Sequential bootstrap (Snippet 4.5, see :mod:`quantowl.weights`) draws
each observation with probability proportional to its uniqueness given the
observations already drawn, so the bags are more diverse and the out-of-bag
estimates are less optimistic.

:class:`SequentialBootstrapBagging` wraps any estimator that follows the duck-typed
protocol ``fit(X, y, sample_weight=None)`` and ``predict(X)`` (plus optionally
``predict_proba(X)``). There is no scikit-learn dependency. The class has ``get_params``,
``set_params`` and ``score`` with scikit-learn's meaning, so ``clone``, ``cross_val_score`` and
``GridSearchCV`` accept it when scikit-learn is installed.

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

import inspect
import numbers
import warnings

import numpy as np
import numpy.typing as npt
import pandas as pd

from ._checks import _same_index, tz_mismatch_message
from ._fit import fit_model
from .weights import indicator_matrix, sequential_bootstrap

__all__ = ["SequentialBootstrapBagging"]


def _unlabelled_by_position(index: pd.Index, x_index: pd.Index, n: int) -> bool:
    """A default 0..n-1 RangeIndex may be matched to a non-integer X index by position."""
    # Value and integer dtype, not the RangeIndex class, decide it.
    return (
        pd.api.types.is_integer_dtype(index)
        and index.equals(pd.RangeIndex(n))
        and not pd.api.types.is_integer_dtype(x_index)
    )


def _frame_values(X: pd.DataFrame) -> np.ndarray:
    """Values for the bases. pd.NA in a mixed frame becomes object and reaches sklearn as a non-number."""
    # Only a frame with an NA takes na_value: pandas refuses NaN as a value for int columns.
    # A mixed nullable frame needs dtype=float, or na_value=np.nan alone still gives object.
    if X.isna().to_numpy().any():
        if all(pd.api.types.is_numeric_dtype(t) for t in X.dtypes):
            return X.to_numpy(dtype=float, na_value=np.nan)
        return X.to_numpy(na_value=np.nan)
    return X.to_numpy()


def _check_t1_values(t1: Any) -> None:
    """Refuse a t1 that is not a Series of end times, before any other check reads it."""
    # Integer values stay accepted as positional end bars. Float weights passed as t1 are the
    # usual mistake, so the message points at sample_weight when the dtype is numeric.
    dtype = getattr(t1, "dtype", None)
    numeric = (
        dtype is not None
        and pd.api.types.is_numeric_dtype(dtype)
        and not pd.api.types.is_bool_dtype(dtype)
    )
    hint = (
        " If these are sample weights, pass them as sample_weight=...; the third argument of fit"
        " is t1."
        if numeric
        else ""
    )
    if not isinstance(t1, pd.Series):
        got = type(t1).__name__ if dtype is None else f"{type(t1).__name__} of dtype {dtype}"
        raise ValueError(
            "t1 must be a pandas Series indexed by event start, with event end times as values; "
            f"got {got}.{hint}"
        )
    # An object Series of Timestamps or of ints is accepted, as the bags have always accepted it.
    inferred = pd.api.types.infer_dtype(t1, skipna=False)
    is_datetimes = pd.api.types.is_datetime64_any_dtype(t1) or inferred == "datetime"
    is_positions = pd.api.types.is_integer_dtype(t1) or inferred == "integer"
    if not (is_datetimes or is_positions):
        raise ValueError(
            "t1 values must be datetimes (event end times) or integer positions; "
            f"got a Series of dtype {t1.dtype}.{hint}"
        )


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
    oob_score : bool, default False
        Compute the out-of-bag estimate in :meth:`fit`, scoring each row with the
        bags that did not draw it (see Notes). Needs ``predict_proba`` on the base
        estimator. With False, fit draws the same bags and the same predictions.

    Attributes
    ----------
    estimators_ : list
        Fitted base estimators, one per bag.
    draws_ : list of np.ndarray
        Row positions drawn for each bag.
    classes_ : np.ndarray
        Sorted unique labels seen in ``fit``.
    oob_decision_function_ : np.ndarray
        Only with ``oob_score=True``. Shape ``(n_samples, n_classes)``: the mean
        ``predict_proba`` over the bags whose draws exclude each row. A row that
        every bag drew is NaN.
    oob_score_ : float
        Only with ``oob_score=True``. Accuracy of ``classes_[argmax(oob_decision_function_)]``
        on the rows that have an estimate, weighted by ``sample_weight`` when weights
        were given. NaN when no such row has positive weight.

    Notes
    -----
    Labels that overlap in time leak between in-bag and out-of-bag samples, so the
    OOB score is optimistic for overlapping labels and is no substitute for purged
    cross-validation. Claimed from the book only (not verified against the text).
    Rows that every bag drew have no out-of-bag estimate. They are left out of
    ``oob_score_``, and ``fit`` warns with their count.
    """

    _estimator_type = "classifier"

    def __init__(
        self,
        base_factory: Callable[[], Any],
        n_estimators: int = 10,
        sequential: bool = True,
        max_samples: int | None = None,
        seed: int | None = 0,
        oob_score: bool = False,
    ) -> None:
        # Arguments are stored unchanged and checked in fit: scikit-learn's clone calls the
        # constructor with the stored values, so a check here would run again on every clone.
        self.base_factory = base_factory
        self.n_estimators = n_estimators
        self.sequential = sequential
        self.max_samples = max_samples
        self.seed = seed
        self.oob_score = oob_score

    def get_params(self, deep: bool = True) -> dict[str, Any]:
        """Constructor arguments by name, as scikit-learn's ``get_params``.

        Parameters
        ----------
        deep : bool, default True
            Accepted for scikit-learn compatibility. The arguments hold no nested
            estimators, so the result is the same either way.

        Returns
        -------
        dict
            One entry per ``__init__`` argument, holding the value currently set.
        """
        return {name: getattr(self, name) for name in self._param_names()}

    def set_params(self, **params: Any) -> "SequentialBootstrapBagging":
        """Set constructor arguments by name, as scikit-learn's ``set_params``.

        Values are checked when :meth:`fit` runs, not here.

        Parameters
        ----------
        **params
            Names from ``get_params``.

        Returns
        -------
        SequentialBootstrapBagging
            ``self``.

        Raises
        ------
        ValueError
            If a name is not a constructor argument. No argument is changed in that case.
        """
        valid = self._param_names()
        unknown = [name for name in params if name not in valid]
        if unknown:
            raise ValueError(
                f"Invalid parameter(s) {unknown} for SequentialBootstrapBagging; "
                f"valid parameters are {valid}."
            )
        for name, value in params.items():
            setattr(self, name, value)
        return self

    @classmethod
    def _param_names(cls) -> list[str]:
        return [name for name in inspect.signature(cls.__init__).parameters if name != "self"]

    def score(
        self,
        X: npt.ArrayLike | pd.DataFrame,
        y: npt.ArrayLike | pd.Series,
        sample_weight: npt.ArrayLike | None = None,
    ) -> float:
        """Mean accuracy of :meth:`predict`, as scikit-learn's ``ClassifierMixin.score``.

        Parameters
        ----------
        X : array-like or pandas
            Features, rows aligned with ``y``.
        y : array-like or pandas
            True labels.
        sample_weight : array-like, optional
            Weight per row. The accuracy is the weighted mean of the per-row hits.

        Returns
        -------
        float
        """
        pred = np.asarray(self.predict(X))
        y_true = np.asarray(y)
        if y_true.ndim > 1 and y_true.size == pred.size:
            # A single-column y, (n, 1), would otherwise broadcast to an (n, n) comparison.
            y_true = y_true.reshape(pred.shape)
        if y_true.shape != pred.shape:
            raise ValueError(
                f"y has shape {y_true.shape} but predict gives {pred.shape}; pass one label per row of X."
            )
        correct = pred == y_true
        if sample_weight is None:
            return float(correct.mean())
        w = np.asarray(sample_weight, dtype=float)
        if w.shape != correct.shape:
            raise ValueError(
                f"sample_weight has shape {w.shape} but X has {correct.shape[0]} rows; "
                f"expected length {correct.shape[0]}."
            )
        return float(np.average(correct, weights=w))

    def __sklearn_tags__(self) -> Any:
        # Imported here: scikit-learn calls this only when it is installed, and this module
        # never imports it. The tags mirror ClassifierMixin.
        from sklearn.base import BaseEstimator
        from sklearn.utils import ClassifierTags

        tags = BaseEstimator.__sklearn_tags__(self)
        tags.estimator_type = "classifier"
        tags.classifier_tags = ClassifierTags()
        tags.target_tags.required = True
        return tags

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
            :func:`quantowl.cv.make_t1`). Its values must be datetimes, or integers when
            they are positional end bars; any other dtype is refused. Required when
            ``sequential=True``. A NaT end time is refused; :func:`quantowl.labeling.get_events`
            leaves NaT for unresolved events, so drop those rows before fitting.
        sample_weight : array-like, optional
            Weights passed to each base estimator's ``fit``, sliced per bag.

        Raises
        ------
        ValueError
            If ``n_estimators`` is below 1, or ``t1`` is not a Series of end times.
        TypeError
            If ``oob_score`` is not a bool.
        """
        if self.n_estimators < 1:
            raise ValueError(f"n_estimators must be >= 1, got {self.n_estimators!r}.")
        if not isinstance(self.oob_score, (bool, np.bool_)):
            raise TypeError(f"oob_score must be a bool, got {self.oob_score!r}.")
        if t1 is not None:
            _check_t1_values(t1)
        Xa = _frame_values(X) if isinstance(X, pd.DataFrame) else np.asarray(X)
        ya = y.to_numpy() if isinstance(y, pd.Series) else np.asarray(y)
        n = Xa.shape[0]
        if ya.shape[0] != n:
            raise ValueError("X and y must have the same number of rows.")
        w = None if sample_weight is None else np.asarray(sample_weight, dtype=float)
        if w is not None and w.shape != (n,):
            # A longer array would be silently truncated when the bags index it.
            raise ValueError(
                f"sample_weight has shape {w.shape} but X has {n} rows; expected length {n}."
            )
        # Labels must line up too: a pandas y or weight is otherwise used by position.
        # A Series X is labelled too, so its y and sample_weight are checked as a DataFrame X is.
        if isinstance(X, (pd.DataFrame, pd.Series)):
            if (
                isinstance(y, (pd.DataFrame, pd.Series))
                and not _same_index(y.index, X.index)
                and not _unlabelled_by_position(y.index, X.index, n)
            ):
                raise ValueError(
                    tz_mismatch_message(y.index, X.index, "y", "X.index")
                    or "y index does not match X.index; align y to X before calling."
                )
            if (
                isinstance(sample_weight, (pd.DataFrame, pd.Series))
                and not _same_index(sample_weight.index, X.index)
                and not _unlabelled_by_position(sample_weight.index, X.index, n)
            ):
                raise ValueError(
                    tz_mismatch_message(sample_weight.index, X.index, "sample_weight", "X.index")
                    or "sample_weight index does not match X.index; align sample_weight to X before calling."
                )
        rng = np.random.default_rng(self.seed)
        if self.max_samples is None:
            size = n
        else:
            ms = self.max_samples
            # Same integer rule as make_t1: 2.7 must not be truncated to 2 with no message.
            if isinstance(ms, (str, bytes, bool, np.bool_)) or not isinstance(ms, numbers.Real) or not float(ms).is_integer():
                raise ValueError(f"max_samples must be an integer >= 1, got {ms!r}.")
            size = int(ms)
        if size < 1:
            raise ValueError("max_samples must be >= 1.")

        if self.sequential:
            if t1 is None:
                raise ValueError("t1 is required for sequential bootstrap.")
            if len(t1) != n:
                raise ValueError("t1 must have one entry per row of X.")
            if t1.isna().any():
                # A NaT end time was read as "spans to the end of the grid". average_active_signals
                # refuses NaT the same way.
                raise ValueError("t1 contains NaT; every event needs an end time.")
            if pd.api.types.is_datetime64_any_dtype(t1) and not isinstance(t1.index, pd.DatetimeIndex):
                # A reset index kept the end times and lost the start times, which then failed in a
                # raw comparison of ints with timestamps.
                raise ValueError("t1 must be indexed by event start times (a DatetimeIndex).")
            # The bags index X by position, but the uniqueness weights come from t1's
            # own labels. A labelled X whose index differs from t1's attached the weights to the wrong
            # rows. A default RangeIndex on X carries no labels, so t1 is matched by position there.
            # Only the default 0..n-1 index is unlabelled; any other RangeIndex names its rows.
            # A labelled t1 (a RangeIndex other than 0..n-1) is checked even against a default X.
            # A date-labelled t1 with a default X is still matched by position.
            t1_labelled_range = pd.api.types.is_integer_dtype(t1.index) and not t1.index.equals(pd.RangeIndex(n))
            x_labelled = isinstance(X, (pd.DataFrame, pd.Series)) and not X.index.equals(pd.RangeIndex(n))
            if (
                isinstance(X, (pd.DataFrame, pd.Series))
                and (x_labelled or t1_labelled_range)
                and not _same_index(t1.index, X.index)
            ):
                raise ValueError("t1 index does not match X.index; align t1 to X before calling.")
            # .values compares instants, so a tz-aware t1 stays vectorised.
            if pd.api.types.is_datetime64_any_dtype(t1) and (t1.values < t1.index.values).any():
                # An end before its start gives an empty span, so the row was never drawn.
                early = t1[t1.values < t1.index.values]
                shown = ", ".join(f"{a} -> {b}" for a, b in zip(early.index[:3], early.to_numpy()[:3]))
                raise ValueError(f"t1 precedes the event start for {len(early)} rows; first: {shown}.")
            if not (t1.index.is_unique and t1.index.is_monotonic_increasing):
                # The event starts double as the bar grid here, which must be strictly increasing.
                raise ValueError(
                    "t1 has duplicate or unsorted event start times; the sequential bootstrap needs "
                    "strictly increasing starts. Sort and de-duplicate them, or pass sequential=False."
                )
            ind = indicator_matrix(t1.index, t1)

        if ya.dtype.kind == "f":
            # predict takes a majority vote over the bags' values, so a continuous target gives
            # a vote-winning value, not the mean. Refuse it here rather than return that.
            fractional = ~(np.isfinite(ya) & (ya == np.trunc(ya)))
            if fractional.any():
                raise ValueError(
                    "SequentialBootstrapBagging supports classification labels only, but y has "
                    f"non-integer values (first: {ya[fractional][0]!r}). Bin a continuous target "
                    "into class labels before fitting."
                )
        if ya.dtype == object and any(v is None for v in ya):
            # np.unique sorts the labels, and sorting None against a label fails with a TypeError
            # that names neither the label nor the row.
            raise ValueError(
                "y contains None, which is not a class label; every row needs a label. "
                "Drop or fill the None rows before fitting."
            )
        # Kept so predict can align a reordered frame by name; set only after validation so a
        # refused refit leaves the earlier names in place.
        self.columns_ = list(X.columns) if isinstance(X, pd.DataFrame) else None
        self.classes_ = np.unique(ya)
        self.estimators_: list[Any] = []
        self.draws_: list[np.ndarray] = []
        # Out-of-bag values from an earlier fit must not outlive a refit that did not ask for them.
        self.__dict__.pop("oob_decision_function_", None)
        self.__dict__.pop("oob_score_", None)
        try:
            for _ in range(self.n_estimators):
                seed_b = int(rng.integers(0, 2**31 - 1))
                if self.sequential:
                    draws = sequential_bootstrap(ind, n_samples=size, seed=seed_b)
                else:
                    draws = np.asarray(np.random.default_rng(seed_b).integers(0, n, size=size))
                est = self.base_factory()
                if self.oob_score and not hasattr(est, "predict_proba"):
                    # Checked before the first bag trains, so a base that cannot score costs no fit.
                    raise ValueError(
                        "oob_score=True needs predict_proba on the base estimator, but "
                        f"{type(est).__name__} has none. Set oob_score=False or use a base with predict_proba."
                    )
                fit_model(est, Xa[draws], ya[draws], None if w is None else w[draws])
                if hasattr(est, "predict_proba") and not hasattr(est, "classes_"):
                    # Without classes_ the columns are taken to be self.classes_, which holds only
                    # if the bag saw every class. Name the problem here, not at predict time.
                    if not np.array_equal(np.unique(ya[draws]), self.classes_):
                        raise ValueError(
                            "a bootstrap bag saw only some classes, and the base estimator has "
                            "predict_proba but no classes_ attribute, so its columns cannot be "
                            "aligned. Expose classes_ on the estimator."
                        )
                self.estimators_.append(est)
                self.draws_.append(draws)
            if self.oob_score:
                self._compute_oob(Xa, ya, w)
        except Exception:
            # A half-built ensemble would let predict run silently on a subset of bags.
            for name in (
                "classes_",
                "estimators_",
                "draws_",
                "columns_",
                "oob_decision_function_",
                "oob_score_",
            ):
                self.__dict__.pop(name, None)
            raise
        return self

    def _compute_oob(self, Xa: np.ndarray, ya: np.ndarray, w: np.ndarray | None) -> None:
        """Set ``oob_decision_function_`` and ``oob_score_`` from the bags' own draws."""
        n = Xa.shape[0]
        k = self.classes_.shape[0]
        total = np.zeros((n, k))
        n_bags = np.zeros(n, dtype=int)
        for b, (est, draws) in enumerate(zip(self.estimators_, self.draws_)):
            in_bag = np.zeros(n, dtype=bool)
            in_bag[draws] = True
            oob = np.flatnonzero(~in_bag)
            if oob.size == 0:
                continue
            p = np.asarray(est.predict_proba(Xa[oob]))
            bad = ~np.isfinite(p).all(axis=1)
            if bad.any():
                # A NaN would reach the average and be scored as a class, so refuse it as predict does.
                raise ValueError(
                    f"bag {b} returned non-finite probabilities for {int(bad.sum())} out-of-bag rows; "
                    "oob_decision_function_ cannot average them."
                )
            block = np.zeros((oob.size, k))
            block[:, self._proba_columns(est)] = p
            total[oob] += block
            n_bags[oob] += 1
        has = n_bags > 0
        odf = np.full((n, k), np.nan)
        odf[has] = total[has] / n_bags[has, None]
        n_missing = int((~has).sum())
        if n_missing:
            warnings.warn(
                f"{n_missing} of {n} samples were in every bag, so they have no out-of-bag estimate. "
                "Their oob_decision_function_ rows are NaN and oob_score_ leaves them out.",
                UserWarning,
                stacklevel=3,
            )
        pred = self.classes_[odf[has].argmax(axis=1)]
        correct = (pred == ya[has]).astype(float)
        weights = np.ones(correct.shape[0]) if w is None else w[has]
        denom = weights.sum()
        if denom > 0:
            score = float((weights * correct).sum() / denom)
        else:
            score = float("nan")
            warnings.warn(
                "oob_score_ is NaN: no sample has an out-of-bag estimate with positive weight.",
                UserWarning,
                stacklevel=3,
            )
        self.oob_decision_function_ = odf
        self.oob_score_ = score

    def _proba_columns(self, est: Any) -> np.ndarray:
        """Positions in ``self.classes_`` of the columns ``est.predict_proba`` returns."""
        return np.searchsorted(self.classes_, getattr(est, "classes_", self.classes_))

    def _check_fitted(self) -> None:
        if not getattr(self, "estimators_", None):
            raise RuntimeError("SequentialBootstrapBagging is not fitted; call fit first.")

    def _predict_array(self, X: npt.ArrayLike | pd.DataFrame) -> np.ndarray:
        if not isinstance(X, pd.DataFrame):
            return np.asarray(X)
        cols = getattr(self, "columns_", None)
        if cols is not None and list(X.columns) != cols:
            missing = [c for c in cols if c not in X.columns]
            extra = [c for c in X.columns if c not in cols]
            if missing or extra or X.columns.has_duplicates:
                raise ValueError(
                    f"X columns do not match the fitted columns: missing {missing}, extra {extra}."
                )
            X = X[cols]
        elif cols is None and not X.columns.equals(pd.RangeIndex(X.shape[1])):
            # No fitted names exist to align by; sklearn warns in the same case.
            warnings.warn(
                "X has column names but the model was fitted without them, so the column "
                "order is taken as given.",
                UserWarning,
                stacklevel=3,
            )
        return _frame_values(X)

    def predict_proba(self, X: npt.ArrayLike | pd.DataFrame) -> np.ndarray:
        """Average the bags' class probabilities; columns follow ``classes_``.

        Requires every base estimator to implement ``predict_proba``; estimators
        whose ``predict_proba`` columns follow their own ``classes_`` attribute
        are re-aligned to ``self.classes_``.
        """
        self._check_fitted()
        Xa = self._predict_array(X)
        total = np.zeros((Xa.shape[0], self.classes_.shape[0]))
        for est in self.estimators_:
            if not hasattr(est, "predict_proba"):
                raise AttributeError("base estimator has no predict_proba.")
            p = np.asarray(est.predict_proba(Xa))
            total[:, self._proba_columns(est)] += p
        # The fitted bags, not n_estimators, which set_params can change after fit.
        return total / len(self.estimators_)

    def predict(self, X: npt.ArrayLike | pd.DataFrame) -> np.ndarray:
        """Predict by averaged probability if available, else by majority vote."""
        self._check_fitted()
        if all(hasattr(e, "predict_proba") for e in self.estimators_):
            proba = self.predict_proba(X)
            bad = ~np.isfinite(proba).all(axis=1)
            if np.any(bad):
                # argmax reads NaN as the largest value, which would label such a row classes_[0].
                raise ValueError(
                    f"a base returned non-finite probabilities for {int(bad.sum())} of "
                    f"{bad.shape[0]} rows; predict cannot choose a class for them."
                )
            return self.classes_[proba.argmax(axis=1)]
        Xa = self._predict_array(X)
        votes = np.stack([np.asarray(e.predict(Xa)) for e in self.estimators_])
        out = np.empty(Xa.shape[0], dtype=self.classes_.dtype)
        for i in range(Xa.shape[0]):
            vals, counts = np.unique(votes[:, i], return_counts=True)
            out[i] = vals[counts.argmax()]
        return out
