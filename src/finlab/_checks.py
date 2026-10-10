"""Argument checks shared by the pandas-only helpers.

These helpers index their inputs by label (``reindex``, ``.index``,
``.columns``). A list or NumPy array fails deep inside pandas with an
AttributeError that does not name the argument, so the check runs first.
"""

from __future__ import annotations

from typing import Any


def _require_type(value: Any, name: str, kind: type) -> None:
    """Raise TypeError unless ``value`` is an instance of ``kind``.

    The exception stays a TypeError, not an AttributeError subclass, so a
    caller that catches AttributeError does not hide a wrong input type.
    """
    if not isinstance(value, kind):
        raise TypeError(
            f"{name} must be a pandas {kind.__name__}, got {type(value).__name__}."
        )
