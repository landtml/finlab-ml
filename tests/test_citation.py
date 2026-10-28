"""CITATION.cff carries the required CFF 1.2.0 keys and the package version."""

from __future__ import annotations

import re
from pathlib import Path

try:
    import tomllib
except ModuleNotFoundError:  # Python 3.10
    tomllib = None

ROOT = Path(__file__).resolve().parents[1]


def _top_level_keys(text: str) -> dict[str, str]:
    keys = {}
    for line in text.splitlines():
        m = re.match(r"^([a-z][a-z-]*):\s*(.*)$", line)
        if m:
            keys[m.group(1)] = m.group(2).strip().strip('"')
    return keys


def _pyproject_version() -> str:
    text = (ROOT / "pyproject.toml").read_text(encoding="utf-8")
    if tomllib is not None:
        return tomllib.loads(text)["project"]["version"]
    m = re.search(r'^version\s*=\s*"([^"]+)"', text, flags=re.MULTILINE)
    assert m is not None
    return m.group(1)


def test_citation_has_required_keys() -> None:
    keys = _top_level_keys((ROOT / "CITATION.cff").read_text(encoding="utf-8"))
    assert keys["cff-version"] == "1.2.0"
    for required in ("message", "authors", "title"):
        assert required in keys


def test_citation_version_matches_pyproject() -> None:
    keys = _top_level_keys((ROOT / "CITATION.cff").read_text(encoding="utf-8"))
    assert keys["version"] == _pyproject_version()
