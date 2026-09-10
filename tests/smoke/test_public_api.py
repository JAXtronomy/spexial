"""Every name in `spexial.__all__` is importable and really re-exported."""

import importlib

import pytest

import spexial as sp

# name -> the private module it is defined in
_ORIGINS = {
    "k0": "spexial._src.kn",
    "k0e": "spexial._src.kn",
    "k1": "spexial._src.kn",
    "k1e": "spexial._src.kn",
    "k2": "spexial._src.kn",
    "k2e": "spexial._src.kn",
    "polylog": "spexial._src.polylog",
    "comb": "spexial._src.comb",
    "eval_gegenbauer": "spexial._src.gegenbauer",
    "eval_gegenbauers": "spexial._src.gegenbauer",
    "gamma": "spexial._src.gamma",
    "sph_harm_y": "spexial._src.sph_harm",
    "sph_harm_y_cart": "spexial._src.sph_harm",
    "sph_harm_y_cart_all": "spexial._src.sph_harm",
    "sph_harm_y_cart_all_terms": "spexial._src.sph_harm",
    "sph_legendre_p": "spexial._src.sph_harm",
    "spence": "spexial._src.spence",
    "zeta": "spexial._src.zeta",
}

# The uppercase spellings `spexial` shipped before it followed `scipy.special`
# on case. They stay in `__all__` until their removal release -- see the
# schedule in `AGENTS.md` -- so they are listed here rather than exempted, and
# this set is what gets deleted when they go.
_DEPRECATED_ALIASES = {"K0", "K1", "K2", "K0e", "K1e", "K2e", "Li"}


def test_all_is_unique():
    """`__all__` is free of duplicates.

    Ordering is not asserted here: ruff's `RUF022` already enforces it, and it
    uses a natural sort (``k0, k1, k2, k0e``) that deliberately differs from
    `sorted` (``k0, k0e, k1``). Two rules disagreeing about the same list is
    worse than one.
    """
    assert len(set(sp.__all__)) == len(sp.__all__)


def test_all_covers_every_origin():
    """No public function is defined in `_src` but missing from `__all__`."""
    assert set(_ORIGINS) | _DEPRECATED_ALIASES | {"__version__"} == set(sp.__all__)


def test_every_alias_points_at_a_real_export():
    """Each deprecated spelling names a function that is itself exported."""
    replacements = {
        "K0": "k0",
        "K1": "k1",
        "K2": "k2",
        "K0e": "k0e",
        "K1e": "k1e",
        "K2e": "k2e",
        "Li": "polylog",
    }
    assert set(replacements) == _DEPRECATED_ALIASES
    for alias, real in replacements.items():
        assert alias in sp.__all__
        assert real in _ORIGINS


@pytest.mark.parametrize("name", sp.__all__)
def test_export_is_importable(name):
    """Every `__all__` entry is actually an attribute of the package."""
    assert getattr(sp, name, None) is not None


@pytest.mark.parametrize(("name", "module"), sorted(_ORIGINS.items()))
def test_export_is_the_src_object(name, module):
    """Each export is the very object defined in its `_src` module."""
    assert getattr(sp, name) is getattr(importlib.import_module(module), name)


def test_src_modules_declare_all():
    """Every `_src` module carries a docstring and an `__all__`."""
    for module in sorted(set(_ORIGINS.values())):
        mod = importlib.import_module(module)
        assert mod.__doc__
        assert isinstance(mod.__all__, list)


def test_module_docstring_flags_the_non_scipy_exports():
    """The package docstring names the two exports scipy has no counterpart for."""
    assert sp.__doc__ is not None
    assert "`polylog`" in sp.__doc__
    assert "`eval_gegenbauers`" in sp.__doc__
