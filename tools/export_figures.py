"""Export the example figures to docs/images/ as PNG files.

Runs three examples with their fixed seeds and saves the figure each one ends with
(the variable ``fig``):

* examples/pbo.py -> docs/images/pbo_logits.png
* examples/monte_carlo.py -> docs/images/monte_carlo_uniqueness.png
* examples/deflated_sharpe.py -> docs/images/deflated_sharpe_comparison.png

Static export needs plotly and kaleido::

    pip install -e ".[plot]" kaleido

kaleido looks for a Chrome build. ``plotly_get_chrome`` installs one, or set
``BROWSER_PATH`` to an existing Chrome or Chromium binary.

Run from the repository root::

    python tools/export_figures.py
"""

from __future__ import annotations

import runpy
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "docs" / "images"
EXAMPLES = {
    "pbo": "pbo_logits.png",
    "monte_carlo": "monte_carlo_uniqueness.png",
    "deflated_sharpe": "deflated_sharpe_comparison.png",
}


def main() -> None:
    import plotly.io as pio

    OUT.mkdir(parents=True, exist_ok=True)
    for name, filename in EXAMPLES.items():
        namespace = runpy.run_path(str(ROOT / "examples" / f"{name}.py"), run_name="example")
        fig = namespace.get("fig")
        if fig is None:
            raise SystemExit(f"examples/{name}.py produced no figure; is plotly installed?")
        path = OUT / filename
        pio.write_image(fig, path, format="png", width=960, height=540, scale=2)
        print(f"wrote {path.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
