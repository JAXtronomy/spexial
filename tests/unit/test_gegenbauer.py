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
    """C_n^(0) is 0 for n >= 1, and C_0^(0) is 1.

    scipy agrees up to 1.14; from 1.18 it returns 0.0 for every order at exactly
    alpha == 0, including order 0. The generating function gives 1, so this
    asserts our own value.
    """
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


@pytest.mark.parametrize("n", [0, 1, 2, 5])
def test_integer_alpha_and_integer_x(n):
    """REGRESSION: integer `alpha` *and* integer `x` used to raise.

    `C0` promoted to float while `C1` (`2 * alpha * x`) stayed int64, so from
    n >= 2 `lax.scan` rejected the carry as having mismatched types. scipy
    promotes integer input, so we do too.
    """
    got = sp.eval_gegenbauer(n, 1, jnp.asarray([0, 1]))
    expected = scipy_eval_gegenbauer(n, 1, np.array([0, 1]))
    assert jnp.issubdtype(got.dtype, jnp.floating)
    np.testing.assert_allclose(got, expected, rtol=1e-12, atol=1e-13)


@pytest.mark.parametrize("n", [0, 1, 2, 3, 5])
@pytest.mark.parametrize(
    ("alpha", "x"),
    [
        (jnp.asarray([0.5, 1.0]), 0.3),  # alpha wider than x
        (jnp.asarray([0.5, 1.0, 2.0]), jnp.asarray([0.3])),  # both arrays, broadcasting
        (jnp.asarray([0.5, 1.0]), jnp.asarray([0.3, 0.4])),  # same shape
    ],
)
def test_alpha_broadcasts_against_x(n, alpha, x):
    """REGRESSION: an `alpha` wider than `x` died in `lax.scan` from n >= 2.

    `C0` follows `x`'s shape while `C1` follows the broadcast of both, so the
    scan carry had one shape going in and another coming out -- surfacing as
    "carry input and carry output must have equal types", which names neither
    this function nor the argument at fault. Orders 0 and 1 never reach the scan
    and so appeared to work, which made the break look arbitrary.
    `scipy.special.eval_gegenbauer` broadcasts here, so this matches it.
    """
    got = sp.eval_gegenbauer(n, alpha, x)
    expected = scipy_eval_gegenbauer(n, np.asarray(alpha), np.asarray(x))
    np.testing.assert_allclose(got, expected, rtol=1e-10, atol=1e-12)
