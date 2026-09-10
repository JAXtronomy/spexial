"""The uppercase spellings still work, and say they are deprecated.

`spexial` follows `scipy.special` for names, and `scipy.special` spells the
modified Bessel functions lowercase. The uppercase aliases are kept working for
the three-release window `AGENTS.md` specifies; these tests are what guarantee
"kept working" is true rather than assumed.
"""

import warnings

import jax
import jax.numpy as jnp
import pytest

import spexial as sp

PAIRS = [
    ("K0", "k0"),
    ("K1", "k1"),
    ("K2", "k2"),
    ("K0e", "k0e"),
    ("K1e", "k1e"),
    ("K2e", "k2e"),
]


@pytest.mark.parametrize(("old", "new"), PAIRS)
def test_uppercase_bessel_warns_and_matches(old, new):
    """The alias returns exactly what the real name does, and warns once."""
    argument = jnp.asarray(1.5)
    expected = float(getattr(sp, new)(argument))
    with pytest.warns(DeprecationWarning, match=f"`spexial.{old}` is deprecated"):
        got = float(getattr(sp, old)(argument))
    assert got == expected


def test_uppercase_polylog_warns_and_matches():
    """`Li` is `polylog`, not `li` -- `li` is the logarithmic integral."""
    expected = float(sp.polylog(2, 0.5))
    with pytest.warns(DeprecationWarning, match="use `spexial.polylog` instead"):
        got = float(sp.Li(2, 0.5))
    assert got == expected


@pytest.mark.parametrize(("old", "new"), PAIRS)
def test_the_alias_survives_the_transforms(old, new):
    """`jit`, `vmap` and `grad` must all still work through the alias.

    `deprecated` wraps the object it is given, and these are `jax.custom_jvp`
    instances -- so the wrapper has to stay transparent to every transform, not
    merely return the right number when called directly.
    """
    alias, real = getattr(sp, old), getattr(sp, new)
    argument = jnp.asarray(1.5)
    pattern = f"`spexial.{old}` is deprecated"
    # A fresh `pytest.warns` per block: the context manager is not reusable,
    # and reusing one silently stops checking after the first `with`.
    with pytest.warns(DeprecationWarning, match=pattern):
        assert float(jax.jit(alias)(argument)) == float(jax.jit(real)(argument))
    with pytest.warns(DeprecationWarning, match=pattern):
        assert float(jax.grad(alias)(argument)) == float(jax.grad(real)(argument))
    with pytest.warns(DeprecationWarning, match=pattern):
        batched = jax.vmap(alias)(jnp.asarray([1.5, 2.5]))
    assert batched.tolist() == jax.vmap(real)(jnp.asarray([1.5, 2.5])).tolist()


@pytest.mark.parametrize(("old", "new"), [*PAIRS, ("Li", "polylog")])
def test_both_spellings_are_exported(old, new):
    """Until the removal release, `__all__` carries both."""
    assert old in sp.__all__
    assert new in sp.__all__


def test_the_lowercase_names_do_not_warn():
    """The real names must be usable without tripping `filterwarnings = error`."""
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        assert float(sp.k0(1.5)) > 0.0
        assert float(sp.polylog(2, 0.5)) > 0.0
