"""Unit tests for the runtime-typechecking switch.

`spexial.setup_package` reads `SPEXIAL_ENABLE_RUNTIME_TYPECHECKING` once, at
import. The test environment sets it to `beartype.beartype`, so the other
spellings are only reachable by re-importing the module under a different
environment -- which is what these do, restoring the original afterwards.
"""

import contextlib
import importlib

import pytest

from spexial import setup_package


@pytest.fixture(autouse=True)
def _restore_module():
    """Reload under the ambient environment after each test."""
    yield
    importlib.reload(setup_package)


def _reload_with(monkeypatch, value):
    if value is None:
        monkeypatch.delenv("SPEXIAL_ENABLE_RUNTIME_TYPECHECKING", raising=False)
    else:
        monkeypatch.setenv("SPEXIAL_ENABLE_RUNTIME_TYPECHECKING", value)
    return importlib.reload(setup_package)


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        (None, False),
        ("False", False),
        ("None", None),
        ("beartype.beartype", "beartype.beartype"),
    ],
)
def test_env_var_selects_the_typechecker(monkeypatch, value, expected):
    """Each spelling maps to the documented value; unset means off."""
    assert expected == _reload_with(monkeypatch, value).RUNTIME_TYPECHECKER


@pytest.mark.parametrize(
    ("value", "installs"), [("False", False), ("beartype.beartype", True)]
)
def test_hook_is_installed_only_when_asked(monkeypatch, value, installs):
    """Off gives a `nullcontext`; a named checker gives the jaxtyping hook."""
    hook = _reload_with(monkeypatch, value).install_import_hook("spexial")
    assert isinstance(hook, contextlib.nullcontext) is not installs
    with hook:  # both forms must be usable as a context manager
        pass


def test_the_import_hook_is_not_a_top_level_attribute():
    """`install_import_hook` is documented on `spexial.setup_package` only.

    It is imported into `spexial/__init__.py` to wrap the package's own
    imports, and deleted afterwards: leaving it bound made
    `spexial.install_import_hook` a name users could come to depend on by
    accident. Keeping it out of `__all__` is not enough, since that governs
    `import *` and nothing else.
    """
    package = importlib.import_module("spexial")
    assert not hasattr(package, "install_import_hook")
    assert not hasattr(package, "_install_import_hook")
    assert hasattr(setup_package, "install_import_hook")
