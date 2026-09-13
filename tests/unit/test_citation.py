"""`CITATION.cff` says things that are also said in `pyproject.toml`.

Two copies of a fact drift, and every other twice-written fact in this
repository has drifted at least once. This is what notices.

**The CFF schema is not checked here.** The Citation File Format project ships
its own validator against its own schema, and CI runs it -- see the
``Validate CITATION.cff`` step in ``.github/workflows/ci.yml``. Re-deriving
"which keys are required" in this file would be a second, worse copy of a
specification somebody else maintains. What is left below is only what is
specific to *this* repository, which no external validator can know.
"""

import tomllib
from pathlib import Path

import pytest
import yaml

import spexial as sp
from spexial.registry import REGISTRY

ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture(scope="module")
def citation() -> dict:
    """`CITATION.cff`, parsed."""
    return yaml.safe_load((ROOT / "CITATION.cff").read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def project() -> dict:
    """The `[project]` table of `pyproject.toml`."""
    text = (ROOT / "pyproject.toml").read_text(encoding="utf-8")
    return tomllib.loads(text)["project"]


def test_the_title_and_licence_match_the_package(citation, project):
    """The same two facts, stated in two files."""
    assert citation["title"] == project["name"]
    assert citation["license"] == project["license"]


def test_the_urls_match_the_package(citation, project):
    """`repository-code` and `url` are `Homepage` and `Documentation` again."""
    urls = project["urls"]
    assert citation["repository-code"].rstrip("/") == urls["Homepage"].rstrip("/")
    assert citation["url"].rstrip("/") == urls["Documentation"].rstrip("/")


def test_the_version_if_present_matches_the_package(citation):
    """`version` is set at release time; when it is set, it must be right.

    Absent between releases rather than stale, which is why this is
    conditional: a `CITATION.cff` claiming a version the package does not have
    would be worse than one claiming none.
    """
    if "version" not in citation:
        pytest.skip("no `version` key; set at release time")
    assert citation["version"] == sp.__version__


def test_the_abstract_does_not_outlive_its_measurements(citation):
    """The abstract quotes a memory figure; it has to be one the registry holds.

    The 268x is `polylog`'s, and it is correct -- but it was asserted in prose
    and checked nowhere, which is the exact shape of every documentation defect
    this project has had to correct.
    """
    ratios = {
        round(1 / row.cost.memory)
        for row in REGISTRY.values()
        if row.cost is not None and row.cost.memory
    }
    assert 268 in ratios, f"abstract quotes 268x; registry has {sorted(ratios)}"
    assert "268x" in citation["abstract"]
