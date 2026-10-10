"""Runs examples/fracdiff.py and checks the values in its KEY dict.

The percent-format script is the source of truth for the example. Running it
here keeps the closed-form checks (weight recursion, d = 1 first difference)
in the test suite.
"""

from __future__ import annotations

import runpy
from pathlib import Path

import pytest

EXAMPLE = Path(__file__).resolve().parents[1] / "examples" / "fracdiff.py"

# Measured with SEED=20260101, N_OBS=3000, THRES=1e-5 on d=0.4: the FFD window
# width is 1457, so the number of finite outputs is 3000 - 1457 = 1543.
N_OUT_D04_MEASURED = 1543


@pytest.fixture(scope="module")
def key() -> dict:
    ns = runpy.run_path(str(EXAMPLE), run_name="example")
    return ns["KEY"]


def test_weights_match_recursion(key: dict) -> None:
    assert key["weights_match_recursion"] is True
    assert key["max_abs_w_err"] < 1e-12


def test_d1_matches_first_difference(key: dict) -> None:
    assert key["d1_matches_first_difference"] is True


def test_n_out_d04_matches_measured(key: dict) -> None:
    assert key["n_out_d04"] == N_OUT_D04_MEASURED
