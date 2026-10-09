"""Purged K-fold cross-validation (AFML snippet 7.3).

A single-path alternative to :class:`finlab.cv.CombinatorialPurgedCV`: the
timeline is cut into ``n_splits`` contiguous folds, each fold is the test set
once, and the same purge + forward embargo as CPCV is applied to the training
side.
"""

from __future__ import annotations

from typing import Iterator

import numpy as np
import numpy.typing as npt
import pandas as pd

from ._cpcv import _end_positions, _make_group_labels, _sample_index, _train_mask

__all__ = ["PurgedKFold"]


class PurgedKFold:
    """Purged, embargoed K-fold splitter for label-overlapping time series.

    Parameters
    ----------
    n_splits : int, default 5
        Number of contiguous folds (``>= 2``).
    t1 : pd.Series, optional
        Event-end time per observation (see :func:`finlab.cv.make_t1`).
        Without it every observation is treated as a one-bar event.
    embargo_pct : float, default 0.0
        Forward embargo as a fraction of the sample length.

    Notes
    -----
    Duck-types the sklearn splitter protocol (``split``, ``get_n_splits``), so
    it can be passed to sklearn helpers without importing sklearn.
    """

    def __init__(
        self,
        n_splits: int = 5,
        t1: pd.Series | None = None,
        embargo_pct: float = 0.0,
    ) -> None:
        if n_splits < 2:
            raise ValueError(f"n_splits must be >= 2, got {n_splits!r}.")
        if not 0.0 <= embargo_pct < 1.0:
            raise ValueError(f"embargo_pct must be in [0, 1), got {embargo_pct!r}.")
        self.n_splits = n_splits
        self.t1 = t1
        self.embargo_pct = embargo_pct

    def get_n_splits(self, X=None, y=None, groups=None) -> int:
        return self.n_splits

    def split(
        self, X, y=None, groups=None, t1: pd.Series | None = None
    ) -> Iterator[tuple[npt.NDArray[np.int_], npt.NDArray[np.int_]]]:
        t1 = self.t1 if t1 is None else t1
        index = _sample_index(X)
        n = len(index)
        if n < self.n_splits:
            raise ValueError(f"n_samples ({n}) must be >= n_splits ({self.n_splits}).")
        end_pos = _end_positions(index, t1)
        return self._iter(n, end_pos)

    def _iter(self, n: int, end_pos: npt.NDArray[np.int_]):
        labels = _make_group_labels(n, self.n_splits)
        start_pos = np.arange(n)
        embargo = int(np.ceil(self.embargo_pct * n)) if self.embargo_pct else 0
        all_idx = np.arange(n)
        for k in range(self.n_splits):
            test_mask = labels == k
            test_pos = all_idx[test_mask]
            blocks = [(int(test_pos[0]), int(test_pos[-1]))]
            train_mask = _train_mask(start_pos, end_pos, blocks, embargo, "label_end")
            train_mask &= ~test_mask
            yield all_idx[train_mask], test_pos
