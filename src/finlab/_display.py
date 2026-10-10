"""HTML display helpers for finlab result objects in notebooks.

Notebook front ends call ``_repr_html_`` on the last expression of a cell. The
helpers here build a small, self-contained HTML fragment: inline ``style``
attributes only, with no ``<style>``, ``<script>`` or ``<link>`` elements and no
external resources. Every label, value and title is passed through
:func:`html.escape`.

Values are formatted as follows: floats with six significant digits
(``"%.6g"``), integers exactly, booleans as ``True`` / ``False``, and anything
else with ``str``.
"""

from __future__ import annotations

import html
import numbers
from collections.abc import Iterable

import numpy as np

__all__ = ["html_table", "format_value"]

_FLOAT_FORMAT = "%.6g"

_WRAPPER_STYLE = "font-family:sans-serif;font-size:13px;margin:4px 0;"
_TITLE_STYLE = "font-weight:bold;margin-bottom:4px;"
_TABLE_STYLE = "border-collapse:collapse;"
_LABEL_STYLE = "text-align:left;font-weight:normal;padding:2px 16px 2px 0;"
_VALUE_STYLE = "text-align:right;font-family:monospace;padding:2px 0;"


def format_value(value: object) -> str:
    """Return the display text of a single value (not yet HTML-escaped).

    Parameters
    ----------
    value : object
        Any value. ``bool`` and NumPy booleans give ``True`` / ``False``;
        integers (Python or NumPy) give their exact decimal form; floats (Python
        or NumPy) give six significant digits; anything else gives ``str(value)``.

    Returns
    -------
    str
        The text to show in a table cell.
    """
    if isinstance(value, (bool, np.bool_)):
        return str(bool(value))
    if isinstance(value, numbers.Integral):
        return str(int(value))
    if isinstance(value, (float, np.floating)):
        return _FLOAT_FORMAT % float(value)
    return str(value)


def html_table(title: str, rows: Iterable[tuple[object, object]]) -> str:
    """Return a self-contained HTML table of ``(label, value)`` rows.

    Parameters
    ----------
    title : str
        Heading shown above the table. Escaped.
    rows : iterable of (label, value) pairs
        One table row per pair. The label is shown in the left column and the
        value, formatted by :func:`format_value`, in the right column. Labels
        and formatted values are escaped.

    Returns
    -------
    str
        An HTML fragment: a ``<div>`` holding the title and a ``<table>``. It
        uses inline ``style`` attributes only and loads nothing external.

    Notes
    -----
    The fragment contains no ``<script>``, ``<style>`` or ``<link>`` element, so
    a notebook can display it without running or fetching anything.
    """
    body = "".join(
        f'<tr><th style="{_LABEL_STYLE}">{html.escape(str(label))}</th>'
        f'<td style="{_VALUE_STYLE}">{html.escape(format_value(value))}</td></tr>'
        for label, value in rows
    )
    return (
        f'<div style="{_WRAPPER_STYLE}">'
        f'<div style="{_TITLE_STYLE}">{html.escape(str(title))}</div>'
        f'<table style="{_TABLE_STYLE}">{body}</table>'
        "</div>"
    )
