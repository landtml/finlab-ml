"""Purged and embargoed cross-validation for overlapping financial labels.

* :class:`CombinatorialPurgedCV` -- CPCV, AFML ch. 7 (all C(N,k) simulations
  and the backtest path map).
* :class:`PurgedKFold` -- single-path purged K-fold, AFML snippet 7.3.
* :func:`make_t1` -- builds the event-end series both splitters need.
"""

from ._cpcv import CombinatorialPurgedCV, CPCVPaths, make_t1
from .purged_kfold import PurgedKFold

__all__ = ["CombinatorialPurgedCV", "CPCVPaths", "PurgedKFold", "make_t1"]
