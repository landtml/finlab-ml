"""Run examples/deflated_sharpe.py and check the values in its KEY dict."""

from __future__ import annotations

import json
import pathlib
import runpy
import time

import pytest

ROOT = pathlib.Path(__file__).resolve().parents[1]
EXAMPLE = ROOT / "examples" / "deflated_sharpe.py"
NOTEBOOK = ROOT / "examples" / "deflated_sharpe.ipynb"

# Measured on seed 20261010 with numpy 2.5.3, scipy 1.18.1 and Python 3.13.16.
# The value is pinned to 1e-9.
EXPECTED_DSR = 0.5903803322029791
MAX_SECONDS = 60.0


@pytest.fixture(scope="module")
def run():
    t0 = time.perf_counter()
    ns = runpy.run_path(str(EXAMPLE))
    return ns["KEY"], time.perf_counter() - t0


def test_example_registers_40_trials(run):
    key, _ = run
    assert key["n_trials"] == 40


def test_dsr_is_below_naive_psr_against_zero(run):
    key, _ = run
    assert key["dsr"] < key["psr_vs_zero"]


def test_dsr_after_80_trials_is_below_dsr(run):
    key, _ = run
    assert key["dsr_after_80"] < key["dsr"]


def test_dsr_matches_measured_value(run):
    key, _ = run
    assert abs(key["dsr"] - EXPECTED_DSR) <= 1e-9


def test_json_round_trip_is_equal(run):
    key, _ = run
    assert key["roundtrip_ok"] is True


def test_example_runs_under_limit(run):
    _, elapsed = run
    assert elapsed < MAX_SECONDS


def test_notebook_has_no_committed_outputs():
    nb = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
    code_cells = [c for c in nb["cells"] if c["cell_type"] == "code"]
    assert code_cells, "notebook has no code cells"
    assert all(c["outputs"] == [] for c in code_cells)
    assert all(c["execution_count"] is None for c in code_cells)
