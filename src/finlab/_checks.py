"""Argument checks shared by the pandas-only helpers.

These helpers index their inputs by label (``reindex``, ``.index``,
``.columns``). A list or NumPy array fails deep inside pandas with an
AttributeError that does not name the argument, so the check runs first.
"""

from __future__ import annotations

import datetime
from typing import Any

import numpy as np
import pandas as pd


def _require_type(value: Any, name: str, kind: type) -> None:
    """Raise TypeError unless ``value`` is an instance of ``kind``.

    The exception stays a TypeError, not an AttributeError subclass, so a
    caller that catches AttributeError does not hide a wrong input type.
    """
    if not isinstance(value, kind):
        raise TypeError(
            f"{name} must be a pandas {kind.__name__}, got {type(value).__name__}."
        )


def _require_real_values(name: str, raw: np.ndarray) -> None:
    """Refuse strings, bytes, bools and None in a numeric container.

    ``np.asarray(..., dtype=float)`` turns ``'0.1'`` into 0.1, ``True`` into 1.0 and ``None`` into NaN,
    so the bad entry is named here before that coercion. Real numbers, including NaN and inf, pass.
    """
    if raw.dtype.kind in "USb":
        bad = raw.ravel()[0]
        raise ValueError(f"{name} must be real numbers, got {type(bad).__name__} {bad!r}.")
    if raw.dtype.kind == "O":
        for value in raw.ravel():
            if value is None or isinstance(value, (str, bytes, bool, np.bool_)):
                raise ValueError(f"{name} must be real numbers, got {type(value).__name__} {value!r}.")


def _holds_datetimes(index: pd.Index) -> bool:
    """True for an object index whose every non-null element is a datetime (pd.Timestamp included)."""
    if index.dtype != object or len(index) == 0:
        return False
    return all(v is None or isinstance(v, datetime.datetime) for v in index)


def as_datetime_index(index: pd.Index) -> pd.Index:
    """Return ``index`` as a DatetimeIndex when it holds datetimes, so its zone and unit are visible.

    An object index of Timestamps has no ``.tz``. A mix of aware zones is converted to UTC, which
    names the same instants. Naive-and-aware mixes, strings and other values come back unchanged.
    """
    if isinstance(index, pd.DatetimeIndex) or not _holds_datetimes(index):
        return index
    try:
        return pd.DatetimeIndex(index)
    except ValueError:
        pass
    values = [v for v in index if v is not None and v is not pd.NaT]
    if values and all(v.tzinfo is not None for v in values):
        return pd.DatetimeIndex(pd.to_datetime(list(index), utc=True))
    return index


def index_tz(index: pd.Index) -> Any:
    """Timezone of a datetime index (object indexes of datetimes included), or None when naive."""
    return getattr(as_datetime_index(index), "tz", None)


def _same_index(a: pd.Index, b: pd.Index) -> bool:
    """True when two indexes are equal, or two tz-aware ones name the same instants in different zones."""
    a, b = as_datetime_index(a), as_datetime_index(b)
    if isinstance(a, pd.DatetimeIndex) and isinstance(b, pd.DatetimeIndex):
        if a.tz is not None and b.tz is not None:
            a, b = a.tz_convert("UTC"), b.tz_convert("UTC")
        # The resolution (s, us, ns) is not part of the instant, so compare at one unit.
        return a.as_unit("ns").equals(b.as_unit("ns"))
    return a.equals(b)


def tz_mismatch_message(a: pd.Index, b: pd.Index, a_name: str, b_name: str) -> str | None:
    # Naive against aware never compares equal, and the plain "indexes do not match" hides the cause.
    a_tz, b_tz = index_tz(a), index_tz(b)
    if (a_tz is None) == (b_tz is None):
        return None
    if a_tz is None:
        return (
            f"{a_name} is tz-naive and {b_name} is tz-aware ({b_tz}); "
            f"localize or convert {a_name} before calling."
        )
    return (
        f"{a_name} is tz-aware ({a_tz}) and {b_name} is tz-naive; "
        f"localize or convert {b_name} before calling."
    )
