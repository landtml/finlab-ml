"""Tests for finlab._display and the _repr_html_ methods of PBOResult and DeflatedSharpeResult."""

from __future__ import annotations

import html
import re
import subprocess
import sys

import numpy as np

from finlab._display import html_table
from finlab.pbo import PBOResult, probability_of_backtest_overfitting
from finlab.stats import DeflatedSharpeResult, deflated_sharpe


def _cell(page: str, label: str) -> str:
    """Return the escaped value text of the table row whose label is ``label``."""
    pattern = (
        r'<th style="[^"]*">' + re.escape(html.escape(label)) + r"</th>"
        r'<td style="[^"]*">([^<]*)</td>'
    )
    match = re.search(pattern, page)
    assert match is not None, f"no row labelled {label!r} in {page!r}"
    return match.group(1)


def _assert_self_contained(page: str) -> None:
    lowered = page.lower()
    for token in ("<script", "<link", "<style", "href=", "src="):
        assert token not in lowered, f"found {token!r} in {page!r}"
    for tag in ("div", "table", "tr", "th", "td"):
        assert page.count(f"<{tag}") == page.count(f"</{tag}>"), tag


def test_html_table_formats_floats_to_six_significant_digits() -> None:
    page = html_table("t", [("pi", 3.14159265358979), ("big", 1234567.0), ("tiny", 1.5e-7)])
    assert _cell(page, "pi") == "3.14159"
    assert _cell(page, "big") == "1.23457e+06"
    assert _cell(page, "tiny") == "1.5e-07"


def test_html_table_shows_integers_exactly_and_bools_as_words() -> None:
    page = html_table("t", [("count", 2704156), ("np_count", np.int64(184756)), ("flag", True)])
    assert _cell(page, "count") == "2704156"
    assert _cell(page, "np_count") == "184756"
    assert _cell(page, "flag") == "True"


def test_html_table_escapes_title_labels_and_values() -> None:
    page = html_table("<script>alert(1)</script>", [("<script>", '<b>"x" & y</b>')])
    assert "<script>" not in page
    assert "<b>" not in page
    assert "&lt;script&gt;" in page
    assert _cell(page, "<script>") == "&lt;b&gt;&quot;x&quot; &amp; y&lt;/b&gt;"
    _assert_self_contained(page)


def test_html_table_has_no_external_or_script_elements() -> None:
    page = html_table("t", [("a", 1.0), ("b", "text")])
    _assert_self_contained(page)
    assert page.startswith("<div")
    assert 'style="' in page


def test_pbo_repr_html_reads_values_from_the_object() -> None:
    # Constructed directly so that a recomputed PBO would differ: all logits are
    # positive, which gives PBO = 0 if it were recalculated.
    result = PBOResult(pbo=0.123456789, logits=np.array([0.5, 1.0, 2.0]), n_combinations=12870)
    page = result._repr_html_()
    assert _cell(page, "PBO") == "0.123457"
    assert _cell(page, "n_combinations = C(S, S/2)") == "12870"
    assert _cell(page, "n_logits") == "3"
    _assert_self_contained(page)


def test_pbo_repr_html_matches_a_computed_result() -> None:
    rng = np.random.default_rng(7)
    matrix = rng.normal(0.0005, 0.01, size=(64, 5))
    result = probability_of_backtest_overfitting(matrix, n_partitions=8)
    page = result._repr_html_()
    assert result.n_combinations == 70
    assert len(result.logits) == 70
    assert _cell(page, "PBO") == f"{result.pbo:.6g}"
    assert _cell(page, "n_combinations = C(S, S/2)") == "70"
    assert _cell(page, "n_logits") == "70"


def test_dsr_repr_html_shows_dsr_benchmark_and_inputs() -> None:
    result = deflated_sharpe(sr_hat=0.1234567, n_trials=40, var_sr=0.01, n_obs=252)
    page = result._repr_html_()
    assert _cell(page, "dsr") == f"{result.dsr:.6g}"
    assert _cell(page, "sr_hat") == "0.123457"
    assert _cell(page, "sr_star") == f"{result.sr_star:.6g}"
    assert _cell(page, "n_trials") == "40"
    assert _cell(page, "n_obs") == "252"
    _assert_self_contained(page)


def test_dsr_repr_html_reads_dsr_from_the_object() -> None:
    result = DeflatedSharpeResult(
        dsr=0.9876543,
        sr_hat=0.5,
        sr_star=0.25,
        n_trials=3,
        n_obs=100,
        skew=0.0,
        kurt=3.0,
    )
    page = result._repr_html_()
    assert _cell(page, "dsr") == "0.987654"
    assert _cell(page, "sr_star") == "0.25"
    assert _cell(page, "n_trials") == "3"


def test_repr_is_still_the_dataclass_repr() -> None:
    pbo = PBOResult(pbo=0.25, logits=np.array([-1.0, 1.0]), n_combinations=2)
    text = repr(pbo)
    assert text.startswith("PBOResult(pbo=0.25, logits=")
    assert text.endswith("n_combinations=2)")
    assert "<" not in text

    dsr = DeflatedSharpeResult(
        dsr=0.5, sr_hat=0.1, sr_star=0.0, n_trials=40, n_obs=252, skew=0.0, kurt=3.0
    )
    assert repr(dsr) == (
        "DeflatedSharpeResult(dsr=0.5, sr_hat=0.1, sr_star=0.0, n_trials=40, "
        "n_obs=252, skew=0.0, kurt=3.0)"
    )


def test_importing_public_modules_prints_nothing() -> None:
    proc = subprocess.run(
        [sys.executable, "-c", "import finlab, finlab.pbo, finlab.stats, finlab.trials"],
        capture_output=True,
        text=True,
        timeout=300,
    )
    assert proc.returncode == 0, proc.stderr
    assert proc.stdout == ""
    assert proc.stderr == ""
