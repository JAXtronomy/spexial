"""Unit tests for `spexial.K0`, `K1` and `K2`."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from scipy.special import kn as scipy_kn

import spexial as sp

# The series cross over at z = 9; ~3e-8 relative there, far better away from it.
RTOL = 1e-6

ORDERS = [(0, sp.K0), (1, sp.K1), (2, sp.K2)]


@pytest.mark.parametrize(("order", "func"), ORDERS)
def test_array_input_is_elementwise(order, func):
    """REGRESSION: the series `jnp.sum` collapsed the caller's axis.

    ``jnp.sum(...)`` without an ``axis`` reduced over *every* axis, so an array
    argument silently produced a single (wrong) scalar instead of an
    elementwise result.
    """
    z = jnp.asarray([0.5, 1.0, 5.0, 20.0])
    got = func(z)
    assert got.shape == z.shape
    np.testing.assert_allclose(got, scipy_kn(order, np.asarray(z)), rtol=RTOL)


@pytest.mark.parametrize("func", [sp.K0, sp.K1, sp.K2])
def test_array_input_matches_scalar_calls(func):
    """Elementwise output equals looping over the same points."""
    z = jnp.asarray([0.2, 3.0, 9.0, 50.0])
    np.testing.assert_allclose(func(z), [float(func(v)) for v in z], rtol=1e-14)


def test_array_input_spans_both_branches():
    """A single array may straddle the z = 9 cross-over."""
    z = jnp.asarray([[1.0, 8.0], [10.0, 30.0]])
    np.testing.assert_allclose(sp.K0(z), scipy_kn(0, np.asarray(z)), rtol=RTOL)


def test_k0_at_zero_is_inf():
    """K0(0) diverges, as in scipy."""
    assert jnp.isinf(sp.K0(0.0))


@pytest.mark.parametrize("func", [sp.K0, sp.K1, sp.K2])
def test_negative_argument_is_nan(func):
    """Negative z is outside the (real) domain."""
    assert jnp.isnan(func(-1.0))


def test_k1_underflows_above_700():
    """`i0` overflows past ~700, so K1 is taken to have underflowed to 0.

    `scipy.special.kn` also returns 0 there, so this is not a divergence.
    """
    assert float(sp.K1(750.0)) == 0.0
    assert scipy_kn(1, 750.0) == 0.0


@pytest.mark.parametrize("func", [sp.K0, sp.K1, sp.K2])
def test_jit(func):
    """The Bessel functions are jittable."""
    np.testing.assert_allclose(jax.jit(func)(2.0), float(func(2.0)), rtol=1e-14)


@pytest.mark.parametrize("func", [sp.K0, sp.K1, sp.K2])
def test_vmap(func):
    """The Bessel functions are vmappable."""
    z = jnp.asarray([0.5, 4.0, 12.0])
    np.testing.assert_allclose(jax.vmap(func)(z), func(z), rtol=1e-14)


@pytest.mark.parametrize("z", [0.5, 2.0, 5.0, 20.0])
def test_grad_k0_is_minus_k1(z):
    """K0'(z) == -K1(z)."""
    np.testing.assert_allclose(jax.grad(sp.K0)(z), -scipy_kn(1, z), rtol=1e-6)


@pytest.mark.parametrize("func", [sp.K0, sp.K1, sp.K2])
def test_dtype_is_float_for_integer_input(func):
    """Integer input is promoted to float."""
    assert jnp.issubdtype(func(jnp.asarray([1, 2])).dtype, jnp.floating)
