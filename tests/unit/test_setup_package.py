"""Unit tests for the runtime-typechecking switch.

`spexial.setup_package` reads `SPEXIAL_ENABLE_RUNTIME_TYPECHECKING` once, at
import. The test environment sets it to `beartype.beartype`, so the other two
arms of that `match` are only reachable by re-importing the module under a
different environment -- which is what these tests do, restoring the original
module afterwards so nothing else in the session sees a rebound
`RUNTIME_TYPECHECKER`.
"""

import contextlib
import importlib

import pytest

from spexial import setup_package


@pytest.fixture(autouse=True)
def _restore_module():
    """Reload the module under the ambient environment after each test."""
    yield
    importlib.reload(setup_package)


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        ("False", False),
        ("None", None),
        ("beartype.beartype", "beartype.beartype"),
        ("typeguard.typechecked", "typeguard.typechecked"),
    ],
)
def test_env_var_selects_the_typechecker(monkeypatch, value, expected):
    """Each spelling of the variable maps to the documented value."""
    monkeypatch.setenv("SPEXIAL_ENABLE_RUNTIME_TYPECHECKING", value)
    reloaded = importlib.reload(setup_package)
    assert expected == reloaded.RUNTIME_TYPECHECKER


def test_unset_defaults_to_disabled(monkeypatch):
    """Runtime typechecking is off unless asked for."""
    monkeypatch.delenv("SPEXIAL_ENABLE_RUNTIME_TYPECHECKING", raising=False)
    reloaded = importlib.reload(setup_package)
    assert reloaded.RUNTIME_TYPECHECKER is False


def test_disabled_hook_is_a_nullcontext(monkeypatch):
    """With checking off, the hook must not install anything."""
    monkeypatch.setenv("SPEXIAL_ENABLE_RUNTIME_TYPECHECKING", "False")
    reloaded = importlib.reload(setup_package)
    hook = reloaded.install_import_hook("spexial")
    assert isinstance(hook, contextlib.nullcontext)


def test_enabled_hook_is_the_jaxtyping_one(monkeypatch):
    """With a checker named, the real jaxtyping hook is returned."""
    monkeypatch.setenv("SPEXIAL_ENABLE_RUNTIME_TYPECHECKING", "beartype.beartype")
    reloaded = importlib.reload(setup_package)
    hook = reloaded.install_import_hook("spexial")
    assert not isinstance(hook, contextlib.nullcontext)
    with hook:
        pass
