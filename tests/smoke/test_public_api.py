"""Every name in `spexial.__all__` is importable and really re-exported."""

import importlib

import pytest

import spexial as sp

# name -> the private module it is defined in
_ORIGINS = {
    "K0": "spexial._src.kn",
    "K0e": "spexial._src.kn",
    "K1": "spexial._src.kn",
    "K1e": "spexial._src.kn",
    "K2": "spexial._src.kn",
    "K2e": "spexial._src.kn",
    "Li": "spexial._src.polylog",
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


def test_all_is_unique():
    """`__all__` is free of duplicates.

    Ordering is not asserted here: ruff's `RUF022` already enforces it, and it
    uses a natural sort (``K0, K1, K2, K0e``) that deliberately differs from
    `sorted` (``K0, K0e, K1``). Two rules disagreeing about the same list is
    worse than one.
    """
    assert len(set(sp.__all__)) == len(sp.__all__)


def test_all_covers_every_origin():
    """No public function is defined in `_src` but missing from `__all__`."""
    assert set(_ORIGINS) | {"__version__"} == set(sp.__all__)


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
    assert "`Li`" in sp.__doc__
    assert "`eval_gegenbauers`" in sp.__doc__
