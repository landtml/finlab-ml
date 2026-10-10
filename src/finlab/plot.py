"""Plotly figures for finlab results (optional extra: ``pip install 'finlab[plot]'``).

Each function takes a result that finlab has already computed and returns a
:class:`plotly.graph_objects.Figure`. The functions only bin, group and draw
the numbers they receive. They compute no statistics: a value such as the PBO
or the deflated Sharpe ratio is read from the result object and shown as is.

plotly is imported inside each function, so this module imports without it.
A missing plotly raises ImportError only when a figure is requested.

Functions
---------
pbo_distribution
    Histogram of the CSCV split logits, with the PBO value annotated.
deflated_sharpe_comparison
    Grouped bars of the observed and the benchmark Sharpe ratio per result.
monte_carlo_histogram
    Overlaid histograms of two uniqueness columns from a Monte Carlo run.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:
    import pandas as pd
    import plotly.graph_objects as go

    from finlab.pbo import PBOResult
    from finlab.stats import DeflatedSharpeResult

__all__ = ["pbo_distribution", "deflated_sharpe_comparison", "monte_carlo_histogram"]

_INSTALL_HINT = "plotly is required for finlab.plot. Install it with: pip install 'finlab[plot]'"


def _require_plotly() -> Any:
    """Return ``plotly.graph_objects``, or raise ImportError with the install hint."""
    try:
        import plotly.graph_objects as go
    except ImportError as exc:
        raise ImportError(_INSTALL_HINT) from exc
    return go


def pbo_distribution(result: PBOResult, *, title: str | None = None) -> go.Figure:
    """Histogram of the CSCV split logits, with the PBO value annotated.

    Parameters
    ----------
    result : finlab.pbo.PBOResult
        Output of :func:`finlab.pbo.probability_of_backtest_overfitting`. Its
        ``logits``, ``pbo`` and ``n_combinations`` attributes are read and drawn
        as they are.
    title : str, optional
        Figure title. Defaults to "CSCV split logits".

    Returns
    -------
    plotly.graph_objects.Figure
        One histogram of the logits, a dashed vertical line at 0, and an
        annotation with the PBO and the number of splits.

    Raises
    ------
    ImportError
        If plotly is not installed.
    """
    go = _require_plotly()
    logits = np.asarray(result.logits, dtype=float)
    fig = go.Figure(go.Histogram(x=logits, name="splits"))
    fig.add_vline(x=0.0, line_dash="dash", line_color="black")
    fig.add_annotation(
        xref="paper",
        yref="paper",
        x=0.99,
        y=0.97,
        xanchor="right",
        yanchor="top",
        showarrow=False,
        text=f"PBO = {result.pbo:.3f}<br>splits = {result.n_combinations}",
    )
    fig.update_layout(
        title=title or "CSCV split logits",
        xaxis_title="logit",
        yaxis_title="splits",
    )
    return fig


def deflated_sharpe_comparison(
    results: DeflatedSharpeResult | Sequence[DeflatedSharpeResult],
    *,
    title: str | None = None,
) -> go.Figure:
    """Grouped bars of the observed and the benchmark Sharpe ratio per result.

    Parameters
    ----------
    results : finlab.stats.DeflatedSharpeResult or sequence of them
        Output of :func:`finlab.stats.deflated_sharpe`. For each result the bars
        show ``sr_hat`` and ``sr_star``. The label under each group shows the
        ``dsr`` value of that result.
    title : str, optional
        Figure title. Defaults to "Deflated Sharpe ratio: observed vs expected maximum".

    Returns
    -------
    plotly.graph_objects.Figure
        Two bar traces, "sr_hat" and "sr_star", in grouped mode.

    Raises
    ------
    ImportError
        If plotly is not installed.
    ValueError
        If ``results`` is an empty sequence.
    """
    go = _require_plotly()
    items: list[Any] = [results] if hasattr(results, "dsr") else list(results)
    if not items:
        raise ValueError("results must contain at least one DeflatedSharpeResult")
    labels = [f"#{i + 1} (DSR {r.dsr:.3f})" for i, r in enumerate(items)]
    fig = go.Figure(
        [
            go.Bar(x=labels, y=[float(r.sr_hat) for r in items], name="sr_hat"),
            go.Bar(x=labels, y=[float(r.sr_star) for r in items], name="sr_star"),
        ]
    )
    fig.update_layout(
        barmode="group",
        title=title or "Deflated Sharpe ratio: observed vs expected maximum",
        yaxis_title="Sharpe ratio (non-annualised)",
    )
    return fig


def monte_carlo_histogram(
    frame: pd.DataFrame,
    columns: Sequence[str] = ("std_u", "seq_u"),
    *,
    title: str | None = None,
) -> go.Figure:
    """Overlaid histograms of two uniqueness columns from a Monte Carlo run.

    Parameters
    ----------
    frame : pandas.DataFrame
        Output of :func:`finlab.monte_carlo.bootstrap_uniqueness_mc`, one row per
        trial.
    columns : sequence of str, default ("std_u", "seq_u")
        Columns to draw, one histogram each.
    title : str, optional
        Figure title. Defaults to "Bootstrap uniqueness: standard vs sequential".

    Returns
    -------
    plotly.graph_objects.Figure
        One histogram per column, in overlay mode with partial opacity.

    Raises
    ------
    ImportError
        If plotly is not installed.
    ValueError
        If a requested column is not in ``frame``.
    """
    go = _require_plotly()
    missing = [c for c in columns if c not in frame.columns]
    if missing:
        raise ValueError(f"frame has no column(s): {missing}")
    fig = go.Figure()
    for col in columns:
        fig.add_trace(go.Histogram(x=frame[col].to_numpy(dtype=float), name=col, opacity=0.6))
    fig.update_layout(
        barmode="overlay",
        title=title or "Bootstrap uniqueness: standard vs sequential",
        xaxis_title="average uniqueness per bootstrap sample",
        yaxis_title="trials",
    )
    return fig
