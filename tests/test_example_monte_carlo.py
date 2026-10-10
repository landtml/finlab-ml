"""Tests for examples/monte_carlo.py, the chapter 4 bootstrap uniqueness notebook.

The script is run with runpy and its KEY dictionary is checked. The histogram cell falls back to
a text summary without plotly, so the main checks do not need plotly. The plotly check is skipped
when plotly is not installed.
"""

from __future__ import annotations

import json
import pathlib
import runpy
import sys

import numpy as np
import pytest

from finlab.monte_carlo import bootstrap_uniqueness_mc

ROOT = pathlib.Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "examples" / "monte_carlo.py"
NOTEBOOK = ROOT / "examples" / "monte_carlo.ipynb"

KEYS = {"mean_gap", "se_gap", "frac_seq_higher", "median_std", "median_seq"}

# Measured by running the script (n_obs=10, n_bars=100, max_h=5, n_iter=2000, seed=0) on the
# development machine. Each run is seeded, so these values are reproducible. The tolerance for the
# exact checks is 1e-9 absolute. A different floating-point summation order moves a value by about
# 1e-16, far below this. Estimator choices do not hide behind it: ddof=0 for the standard error
# moves se_gap by about 7e-7, and a strict comparison with no tie tolerance gives 0.712 instead of
# 0.7015 for frac_seq_higher.
EXACT_TOL = 1e-9
MEASURED = {
    "mean_gap": 0.08435739225088185,
    "se_gap": 0.0028074872265062972,
    "frac_seq_higher": 0.7015,  # 1403 of 2000 trials, with ties (within 1e-12) not counted
    "median_std": 0.6,
    "median_seq": 0.7,
}


@pytest.fixture(scope="module")
def ns():
    """Globals of one run of the script, shared by the checks that do not change plotly."""
    return runpy.run_path(str(SCRIPT), run_name="example")


def test_key_has_the_documented_keys(ns):
    assert set(ns["KEY"]) == KEYS


def test_gap_exceeds_four_standard_errors(ns):
    # Same criterion as the existing check in tests/test_monte_carlo.py:
    # test_sequential_bootstrap_uniqueness_exceeds_standard_by_four_se.
    key = ns["KEY"]
    assert key["mean_gap"] > 4 * key["se_gap"]


def test_sequential_is_higher_in_most_trials(ns):
    assert ns["KEY"]["frac_seq_higher"] > 0.5


def test_key_matches_measured_values(ns):
    for name, expected in MEASURED.items():
        assert abs(ns["KEY"][name] - expected) <= EXACT_TOL, name


def test_key_matches_independent_recomputation(ns):
    # The same quantities computed here from a direct call, with this test's own formulas.
    df = bootstrap_uniqueness_mc(n_obs=10, n_bars=100, max_h=5, n_iter=2000, seed=0)
    d = (df["seq_u"] - df["std_u"]).to_numpy()
    expected = {
        "mean_gap": d.mean(),
        "se_gap": d.std(ddof=1) / np.sqrt(len(d)),
        "frac_seq_higher": (d > 1e-12).mean(),
        "median_std": df["std_u"].median(),
        "median_seq": df["seq_u"].median(),
    }
    for name, value in expected.items():
        assert abs(ns["KEY"][name] - value) <= EXACT_TOL, name


def test_runs_without_plotly(monkeypatch, capsys):
    # Block plotly even when it is installed, so the fallback path runs.
    monkeypatch.setitem(sys.modules, "plotly", None)
    monkeypatch.setitem(sys.modules, "plotly.graph_objects", None)
    run = runpy.run_path(str(SCRIPT), run_name="example")
    out = capsys.readouterr().out
    assert "pip install 'finlab[plot]'" in out
    assert "Text summary" in out
    assert run["fig"] is None
    assert abs(run["KEY"]["mean_gap"] - MEASURED["mean_gap"]) <= EXACT_TOL


def test_histogram_cell_builds_figure_with_plotly(ns):
    go = pytest.importorskip("plotly.graph_objects")
    fig = ns["fig"]
    assert isinstance(fig, go.Figure)
    assert len(fig.data) == 2
    xs = [np.asarray(trace.x) for trace in fig.data]
    for col in ("std_u", "seq_u"):
        assert any(np.array_equal(x, ns["frame"][col].to_numpy()) for x in xs), col


def test_notebook_has_no_committed_outputs():
    nb = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
    code_cells = [cell for cell in nb["cells"] if cell["cell_type"] == "code"]
    assert code_cells
    for cell in code_cells:
        assert cell["outputs"] == []
        assert cell["execution_count"] is None


def test_notebook_is_paired_with_script():
    jupytext = pytest.importorskip("jupytext")
    from_script = jupytext.read(SCRIPT)
    from_notebook = jupytext.read(NOTEBOOK)
    assert [c.cell_type for c in from_notebook.cells] == [c.cell_type for c in from_script.cells]
    assert [c.source for c in from_notebook.cells] == [c.source for c in from_script.cells]
