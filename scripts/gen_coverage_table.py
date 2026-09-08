#!/usr/bin/env python3
"""Regenerate `docs/reference/coverage.md` from the registry in the package.

The table is code (`spexial._src.registry`), not prose. This renders it between
the markers in the page; `tests/unit/test_registry.py` fails if the committed
page and the registry disagree, so the two cannot drift.

Usage: ``uv run scripts/gen_coverage_table.py``
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from spexial.registry import render_markdown

START = "<!-- BEGIN GENERATED TABLE -->"
END = "<!-- END GENERATED TABLE -->"
PAGE = Path(__file__).resolve().parent.parent / "docs" / "reference" / "coverage.md"


def apply(text: str) -> str:
    """Splice the rendered table into ``text`` between the markers."""
    before, _, rest = text.partition(START)
    _, _, after = rest.partition(END)
    return f"{before}{START}\n\n{render_markdown()}\n{END}{after}"


if __name__ == "__main__":
    PAGE.write_text(apply(PAGE.read_text()))
    print(f"wrote {PAGE}")
