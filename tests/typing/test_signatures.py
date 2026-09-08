"""Static-typing guard for the public API.

This fixture is the *only* thing pyright, ty and mypy are pointed at (see
``[tool.pyright]``, ``[tool.ty.src]`` and ``[tool.mypy]`` in ``pyproject.toml``).
It exists so the public signatures cannot silently regress, without asking the
whole -- not yet checker-clean -- source tree to pass. Widen the scoping in
``pyproject.toml`` as more of the tree becomes clean.
"""

import spexial


def test_version_is_a_string() -> None:
    """`spexial.__version__` is typed as `str`."""
    version: str = spexial.__version__
    assert isinstance(version, str)
