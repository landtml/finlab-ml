"""Tests for finlab.plot (optional plotly figures of PBO, deflated Sharpe and Monte Carlo results).

The plotting tests need plotly and skip without it. The tests that check the
missing-plotly error, the module import and the static guard always run.
"""

from __future__ import annotations

import ast
import dataclasses
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

import finlab.plot as plot
from finlab.monte_carlo import bootstrap_uniqueness_mc
from finlab.pbo import probability_of_backtest_overfitting
from finlab.stats import deflated_sharpe

INSTALL_HINT = "plotly is required for finlab.plot. Install it with: pip install 'finlab[plot]'"


@pytest.fixture(scope="module")
def pbo_result():
    rng = np.random.default_rng(0)
    matrix = rng.normal(0.0, 1.0, size=(64, 6))
    return probability_of_backtest_overfitting(matrix, n_partitions=8)


@pytest.fixture(scope="module")
def dsr_results():
    return [
        deflated_sharpe(sr_hat=0.21, n_trials=40, var_sr=0.02, n_obs=252, skew=-0.3, kurt=4.5),
        deflated_sharpe(sr_hat=0.10, n_trials=5, var_sr=0.01, n_obs=120),
    ]


@pytest.fixture(scope="module")
def mc_frame():
    return bootstrap_uniqueness_mc(n_iter=50, seed=0)


@pytest.fixture
def go():
    """plotly.graph_objects, or skip the test when plotly is not installed."""
    return pytest.importorskip("plotly.graph_objects")


@pytest.fixture
def no_plotly(monkeypatch):
    monkeypatch.setitem(sys.modules, "plotly", None)
    monkeypatch.setitem(sys.modules, "plotly.graph_objects", None)


# ---------------------------------------------------------------------------
# Always run: missing plotly and the import without plotly.
# ---------------------------------------------------------------------------
def test_each_function_raises_install_hint_without_plotly(
    no_plotly, pbo_result, dsr_results, mc_frame
):
    calls = [
        lambda: plot.pbo_distribution(pbo_result),
        lambda: plot.deflated_sharpe_comparison(dsr_results),
        lambda: plot.monte_carlo_histogram(mc_frame),
    ]
    for call in calls:
        with pytest.raises(ImportError) as excinfo:
            call()
        assert str(excinfo.value) == INSTALL_HINT


def test_module_imports_without_plotly():
    code = (
        "import sys; sys.modules['plotly'] = None; sys.modules['plotly.graph_objects'] = None; "
        "import finlab.plot as p; assert callable(p.pbo_distribution)"
    )
    proc = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, timeout=120)
    assert proc.returncode == 0, proc.stderr


# ---------------------------------------------------------------------------
# Plotting tests: skipped without plotly.
# ---------------------------------------------------------------------------
def test_pbo_distribution_histogram_matches_logits(go, pbo_result):
    fig = plot.pbo_distribution(pbo_result)
    assert isinstance(fig, go.Figure)
    hists = [t for t in fig.data if isinstance(t, go.Histogram)]
    assert len(hists) == 1
    np.testing.assert_array_equal(np.asarray(hists[0].x), pbo_result.logits)
    lines = [s for s in fig.layout.shapes if s.type == "line" and s.x0 == 0 and s.x1 == 0]
    assert len(lines) == 1
    texts = [a.text for a in fig.layout.annotations]
    assert any(
        f"PBO = {pbo_result.pbo:.3f}" in t and f"splits = {pbo_result.n_combinations}" in t
        for t in texts
    )


def test_pbo_distribution_shows_stored_pbo_not_a_recomputed_one(go, pbo_result):
    stored_value = 0.4567
    recomputed = float(np.mean(pbo_result.logits <= 0.0))
    # The check below must be able to tell the stored and the recomputed value apart.
    assert f"{recomputed:.3f}" != f"{stored_value:.3f}"
    stored = dataclasses.replace(pbo_result, pbo=stored_value)
    fig = plot.pbo_distribution(stored, title="Custom")
    assert fig.layout.title.text == "Custom"
    assert any(f"PBO = {stored_value:.3f}" in a.text for a in fig.layout.annotations)


def test_deflated_sharpe_comparison_bars_match_result_fields(go, dsr_results):
    fig = plot.deflated_sharpe_comparison(dsr_results)
    assert isinstance(fig, go.Figure)
    assert fig.layout.barmode == "group"
    assert all(isinstance(t, go.Bar) for t in fig.data)
    bars = {t.name: t for t in fig.data}
    assert set(bars) == {"sr_hat", "sr_star"}
    np.testing.assert_array_equal(np.asarray(bars["sr_hat"].y), [r.sr_hat for r in dsr_results])
    np.testing.assert_array_equal(np.asarray(bars["sr_star"].y), [r.sr_star for r in dsr_results])
    labels = list(bars["sr_hat"].x)
    assert len(labels) == len(dsr_results)
    for r, label in zip(dsr_results, labels):
        assert f"DSR {r.dsr:.3f}" in label


def test_deflated_sharpe_comparison_accepts_a_single_result(go, dsr_results):
    fig = plot.deflated_sharpe_comparison(dsr_results[0], title="One")
    assert fig.layout.title.text == "One"
    assert len(fig.data) == 2
    assert all(len(t.x) == 1 for t in fig.data)


def test_deflated_sharpe_comparison_rejects_empty_sequence(go):
    with pytest.raises(ValueError, match="at least one"):
        plot.deflated_sharpe_comparison([])


def test_monte_carlo_histogram_overlays_both_columns(go, mc_frame):
    fig = plot.monte_carlo_histogram(mc_frame)
    assert isinstance(fig, go.Figure)
    assert fig.layout.barmode == "overlay"
    assert [t.name for t in fig.data] == ["std_u", "seq_u"]
    for t in fig.data:
        assert isinstance(t, go.Histogram)
        np.testing.assert_array_equal(np.asarray(t.x), mc_frame[t.name].to_numpy())
        assert t.opacity is not None and 0 < t.opacity < 1


def test_monte_carlo_histogram_rejects_missing_column(go, mc_frame):
    with pytest.raises(ValueError, match="no column"):
        plot.monte_carlo_histogram(mc_frame, columns=("std_u", "missing"))


# ---------------------------------------------------------------------------
# Static guard (always runs). It is a guard, not a proof: it looks only at
# call names in the AST, so it does not see a statistic computed by an
# indirect route such as a stored function reference.
# ---------------------------------------------------------------------------
_FORBIDDEN_CALLS = frozenset(
    {
        "probability_of_backtest_overfitting",
        "deflated_sharpe",
        "deflated_sharpe_ratio",
        "probabilistic_sharpe_ratio",
        "sharpe_ratio",
        "expected_max_sharpe",
        "expected_max",
        "percentile",
        "mean",
    }
)


def _called_names(source: str) -> set[str]:
    names: set[str] = set()
    for node in ast.walk(ast.parse(source)):
        if isinstance(node, ast.Call):
            func = node.func
            if isinstance(func, ast.Name):
                names.add(func.id)
            elif isinstance(func, ast.Attribute):
                names.add(func.attr)
    return names


def test_plot_module_source_has_no_statistics_calls():
    """Guard: plot.py must not call any function that computes a statistic."""
    source = Path(plot.__file__).read_text(encoding="utf-8")
    assert _called_names(source) & _FORBIDDEN_CALLS == set()


def test_static_guard_flags_a_forbidden_call():
    """The guard must catch a forbidden call, or the check above proves nothing."""
    source = "x = np.mean(y)\nz = deflated_sharpe(1.0, 2, 0.1, 10)\nw = plot_title(x)\n"
    assert _called_names(source) & _FORBIDDEN_CALLS == {"mean", "deflated_sharpe"}
