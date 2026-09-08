"""Unit tests for `spexial.gamma`."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from scipy.special import digamma, gamma as scipy_gamma

import spexial as sp


@pytest.mark.parametrize(
    ("x", "expected"),
    [(1.0, 1.0), (2.0, 1.0), (5.0, 24.0), (0.5, np.sqrt(np.pi))],
)
def test_known_values(x, expected):
    """Textbook values."""
    np.testing.assert_allclose(sp.gamma(x), expected, rtol=1e-12)


def test_array_input_broadcasts():
    """REGRESSION: `lax.cond` needs a scalar predicate, so arrays used to fail.

    The old body was ``lax.cond(x < 0.5, reflect, lanczos, x)``, which raises
    for any non-scalar ``x``.
    """
    x = jnp.asarray([0.5, 1.0, 2.0, 5.0, -0.5, -1.5])
    got = sp.gamma(x)
    assert got.shape == x.shape
    np.testing.assert_allclose(got, scipy_gamma(np.asarray(x)), rtol=1e-11)


def test_array_input_2d():
    """Shape is preserved for more than one dimension."""
    x = jnp.asarray([[1.0, 2.0], [3.0, 4.0]])
    assert sp.gamma(x).shape == (2, 2)
    np.testing.assert_allclose(sp.gamma(x), [[1.0, 1.0], [2.0, 6.0]], rtol=1e-11)


@pytest.mark.parametrize("x", [0.0, -1.0, -2.0, -10.0])
def test_poles_are_inf(x):
    """Non-positive integers are poles and give `inf`, as scipy does."""
    assert jnp.isinf(sp.gamma(x))
    assert jnp.isinf(scipy_gamma(x))


def test_reflection_branch():
    """Below 0.5 the reflection formula is used and still matches scipy."""
    x = np.linspace(-4.9, 0.49, 97)
    np.testing.assert_allclose(sp.gamma(x), scipy_gamma(x), rtol=1e-11)


def test_large_argument_does_not_overflow_early():
    """The log-space Lanczos evaluation reaches scipy's own overflow point.

    A direct ``t ** (z - 0.5)`` overflows above x ~ 142, well below the true
    overflow at ~171.6.
    """
    x = jnp.asarray([150.0, 170.0, 171.0])
    np.testing.assert_allclose(sp.gamma(x), scipy_gamma(np.asarray(x)), rtol=1e-11)
    assert jnp.isinf(sp.gamma(172.0))


def test_complex_input_is_rejected_by_the_type_checker():
    """`gamma` is documented as real-only; complex input is a type error.

    The runtime type checker is on under pytest (see ``[tool.pytest_env]``), so
    this is the behaviour a user sees rather than a silent wrong answer.
    """
    with pytest.raises(Exception, match=r"(?i)typecheck|complex"):
        sp.gamma(jnp.asarray(1.0 + 2.0j))


def test_dtype_is_float_for_integer_input():
    """Integer input is promoted to float."""
    assert jnp.issubdtype(sp.gamma(jnp.asarray([1, 2, 3])).dtype, jnp.floating)


def test_jit():
    """`gamma` is jittable."""
    np.testing.assert_allclose(jax.jit(sp.gamma)(5.0), 24.0, rtol=1e-12)


def test_vmap():
    """`gamma` is vmappable."""
    got = jax.vmap(sp.gamma)(jnp.asarray([1.0, 2.0, 5.0]))
    np.testing.assert_allclose(got, [1.0, 1.0, 24.0], rtol=1e-11)


@pytest.mark.parametrize("x", [1.5, 5.0, -0.5, -2.5])
def test_grad(x):
    """gamma'(x) == gamma(x) * digamma(x)."""
    got = jax.grad(sp.gamma)(x)
    np.testing.assert_allclose(got, scipy_gamma(x) * digamma(x), rtol=1e-9)
