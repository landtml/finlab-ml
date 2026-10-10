"""Run examples/pbo.py end to end and check its key numbers.

The example is run as a script with runpy (run_name="example"), so its
``if __name__`` blocks are not needed. The plot cell runs only when plotly is
installed. The no-plotly test forces the ImportError path, so the module-level
checks never need plotly.

Measured values with the seeds in the example (T=320, N=60, S=8):
noise_pbo = 45/70 = 0.6428571428571429 (seed 0), edge_pbo = 0.0 (seed 1).
Both PBO values are k/70 for an integer k, so the 1e-9 absolute tolerance
used below is far below the grid spacing (1/70).
"""

from __future__ import annotations

import json
import pathlib
import runpy
import sys

import numpy as np
import pytest

ROOT = pathlib.Path(__file__).resolve().parents[1]
EXAMPLE = ROOT / "examples" / "pbo.py"
NOTEBOOK = ROOT / "examples" / "pbo.ipynb"

MEASURED_NOISE_PBO = 45 / 70  # seed 0; C(8, 4) = 70 splits
MEASURED_EDGE_PBO = 0.0  # seed 1
TOL = 1e-9


def _run_example() -> dict:
    return runpy.run_path(str(EXAMPLE), run_name="example")


@pytest.fixture(scope="module")
def example_globals() -> dict:
    return _run_example()


@pytest.fixture(scope="module")
def key(example_globals: dict) -> dict:
    return example_globals["KEY"]


def test_key_has_expected_fields(key: dict) -> None:
    assert set(key) == {"noise_pbo", "edge_pbo", "n_combinations"}


def test_n_combinations_is_binomial_8_choose_4(key: dict) -> None:
    # C(8, 4) = 70. Integer comparison, with the 1e-9 tolerance the notebook tests share.
    assert key["n_combinations"] == pytest.approx(70, abs=TOL)


def test_noise_pbo_is_near_half(key: dict) -> None:
    # Pure noise with even N gives E[PBO] = 1/2 (docs/proofs/pbo.md, section 3).
    # One fixed-seed draw should sit in a generous band around that.
    assert 0.3 <= key["noise_pbo"] <= 0.7
    assert key["noise_pbo"] == pytest.approx(MEASURED_NOISE_PBO, abs=TOL)


def test_edge_pbo_is_near_zero(key: dict) -> None:
    assert key["edge_pbo"] < 0.05
    assert key["edge_pbo"] == pytest.approx(MEASURED_EDGE_PBO, abs=TOL)


def test_example_runs_without_plotly(monkeypatch: pytest.MonkeyPatch) -> None:
    # A None entry in sys.modules makes the import raise ImportError, which is
    # the same failure as a missing plotly. The plot cell must catch it and the
    # key numbers must still be computed.
    monkeypatch.setitem(sys.modules, "plotly", None)
    monkeypatch.setitem(sys.modules, "plotly.graph_objects", None)
    g = _run_example()
    assert g["fig"] is None
    assert g["KEY"]["n_combinations"] == pytest.approx(70, abs=TOL)
    assert g["KEY"]["noise_pbo"] == pytest.approx(MEASURED_NOISE_PBO, abs=TOL)
    assert g["KEY"]["edge_pbo"] == pytest.approx(MEASURED_EDGE_PBO, abs=TOL)


def test_plot_cell_builds_figure_when_plotly_present(example_globals: dict) -> None:
    go = pytest.importorskip("plotly.graph_objects")
    fig = example_globals["fig"]
    assert isinstance(fig, go.Figure)
    assert len(fig.data) >= 1
    # The histogram is drawn from the noise logits, unchanged.
    np.testing.assert_allclose(
        np.asarray(fig.data[0].x, dtype=float),
        example_globals["noise_result"].logits,
        rtol=0,
        atol=0,
    )


def test_notebook_has_no_committed_outputs() -> None:
    nb = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
    code_cells = [c for c in nb["cells"] if c["cell_type"] == "code"]
    assert code_cells, "notebook has no code cells"
    for cell in code_cells:
        assert cell.get("outputs") == []
        assert cell.get("execution_count") is None


def test_notebook_is_paired_with_script() -> None:
    jupytext = pytest.importorskip("jupytext")
    py_nb = jupytext.read(EXAMPLE)
    ipynb = jupytext.read(NOTEBOOK)
    py_cells = [(c.cell_type, c.source) for c in py_nb.cells]
    nb_cells = [(c.cell_type, c.source) for c in ipynb.cells]
    assert nb_cells == py_cells
