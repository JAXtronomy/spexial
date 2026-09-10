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
    """Splice the rendered table into ``text`` between the markers.

    Both markers must be present, and their absence is an error rather than
    something to work around. `str.partition` returns the whole string as its
    *first* element when the separator is missing, so the obvious unguarded
    version fails in two different silent ways: without `START` it appends a
    second table at the end of the file, and without `END` it deletes
    everything that followed the marker. This script exists to keep the page
    and the registry in agreement, so a page it cannot parse is exactly the
    case that must stop rather than produce something plausible.
    """
    if START not in text or END not in text:
        missing = [m for m in (START, END) if m not in text]
        msg = f"{PAGE} is missing the generated-table marker(s): {', '.join(missing)}"
        raise ValueError(msg)
    before, _, rest = text.partition(START)
    _, _, after = rest.partition(END)
    return f"{before}{START}\n\n{render_markdown()}\n{END}{after}"


if __name__ == "__main__":
    # Explicit UTF-8 both ways: the page contains typographic characters, and
    # `Path.read_text` defaults to the locale encoding, which is not UTF-8 on
    # every platform this might be run from.
    PAGE.write_text(apply(PAGE.read_text(encoding="utf-8")), encoding="utf-8")
    print(f"wrote {PAGE}")
