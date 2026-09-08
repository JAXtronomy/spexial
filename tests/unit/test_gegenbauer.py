"""Unit tests for `spexial.eval_gegenbauer` and `spexial.eval_gegenbauers`."""

from functools import partial

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from scipy.special import eval_gegenbauer as scipy_eval_gegenbauer

import spexial as sp
from spexial._src.gegenbauer import C0


def test_c0_broadcasts():
    """REGRESSION: `C0` returned the Python float 1.0, ignoring `x`.

    Order 0 is constant, but it still has to carry the shape of ``x``;
    ``eval_gegenbauer(0, alpha, x)`` used to collapse an array to a scalar.
    """
    x = jnp.asarray([0.1, 0.2, 0.3])
    assert C0(x).shape == x.shape
    got = sp.eval_gegenbauer(0, 1.0, x)
    assert got.shape == x.shape
    np.testing.assert_array_equal(got, jnp.ones(3))


def test_c0_dtype_is_float_for_integer_input():
    """`C0` promotes integer input, so degree 0 is not an integer array."""
    assert jnp.issubdtype(C0(jnp.asarray([1, 2])).dtype, jnp.floating)


def test_eval_gegenbauer_broadcasts_over_x():
    """Higher degrees are elementwise in `x` too."""
    x = jnp.asarray([-1.0, 0.0, 0.5, 1.0])
    got = sp.eval_gegenbauer(3, 1.0, x)
    assert got.shape == x.shape
    np.testing.assert_allclose(
        got, scipy_eval_gegenbauer(3, 1.0, np.asarray(x)), rtol=1e-12, atol=1e-12
    )


@pytest.mark.parametrize("n", range(6))
def test_eval_gegenbauers_length(n):
    """`eval_gegenbauers(n, ...)` returns exactly n + 1 values."""
    got = sp.eval_gegenbauers(n, 1.5, 0.3)
    assert got.shape == (n + 1,)


@pytest.mark.parametrize("n", range(6))
def test_eval_gegenbauers_agrees_with_scalar_calls(n):
    """The stacked result is the same as calling `eval_gegenbauer` per degree."""
    got = sp.eval_gegenbauers(n, 1.5, 0.3)
    expected = [float(sp.eval_gegenbauer(i, 1.5, 0.3)) for i in range(n + 1)]
    np.testing.assert_allclose(got, expected, rtol=1e-12, atol=1e-12)


def test_alpha_zero_vanishes_above_degree_zero():
    """C_n^(0) is 0 for n >= 1, as in scipy."""
    got = sp.eval_gegenbauers(4, 0.0, 0.3)
    np.testing.assert_allclose(got, [1.0, 0.0, 0.0, 0.0, 0.0], atol=1e-15)


def test_jit():
    """Both entry points are jittable (degree is a static argument)."""
    got = jax.jit(sp.eval_gegenbauer, static_argnums=(0,))(3, 1.0, 0.5)
    np.testing.assert_allclose(got, -1.0, rtol=1e-12, atol=1e-12)


def test_vmap_over_x():
    """`eval_gegenbauers` vmaps over the evaluation point."""
    x = jnp.asarray([0.1, 0.5])
    got = jax.vmap(partial(sp.eval_gegenbauers, 3, 1.0))(x)
    assert got.shape == (2, 4)
    expected = [[float(sp.eval_gegenbauer(i, 1.0, v)) for i in range(4)] for v in x]
    np.testing.assert_allclose(got, expected, rtol=1e-12, atol=1e-12)


@pytest.mark.parametrize("x", [-0.7, 0.0, 0.5])
def test_grad(x):
    """d/dx C_n^(alpha)(x) == 2 alpha C_{n-1}^(alpha+1)(x)."""
    got = jax.grad(partial(sp.eval_gegenbauer, 3, 1.0))(x)
    expected = 2 * 1.0 * scipy_eval_gegenbauer(2, 2.0, x)
    np.testing.assert_allclose(got, expected, rtol=1e-10, atol=1e-12)


def test_grad_at_degree_zero_is_zero():
    """Degree 0 is constant, so its gradient is 0 -- and defined at all."""
    assert float(jax.grad(partial(sp.eval_gegenbauer, 0, 1.0))(0.5)) == 0.0
