"""Run every ```python block in docs/modules/*.md; each page's example must execute."""
import pathlib
import re

import pytest

PAGES = sorted((pathlib.Path(__file__).resolve().parents[1] / "docs" / "modules").glob("*.md"))
BLOCK = re.compile(r"```python\n(.*?)```", re.S)
# Pages whose example takes over 2 s to run (measured with --durations=0).
SLOW_PAGES = {"structural_breaks"}
PARAMS = [pytest.param(p, marks=pytest.mark.slow) if p.stem in SLOW_PAGES else p for p in PAGES]


@pytest.mark.parametrize("page", PARAMS, ids=lambda p: p.stem)
def test_module_page_example_runs(page):
    blocks = BLOCK.findall(page.read_text(encoding="utf-8"))
    assert blocks, f"{page.name} has no runnable example"
    for code in blocks:
        exec(compile(code, str(page), "exec"), {"__name__": "__doc_example__"})
