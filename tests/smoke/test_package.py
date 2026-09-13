"""Test the package itself."""

import importlib.metadata

import spexial as sp


def test_version():
    """The installed distribution version matches `spexial.__version__`."""
    assert importlib.metadata.version("spexial") == sp.__version__


def test_version_is_a_string():
    """`__version__` is a `str`, and is exported."""
    assert isinstance(sp.__version__, str)
    assert "__version__" in sp.__all__
