"""Unit tests for `spexial.zeta`."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from scipy.special import zeta as scipy_zeta

import spexial as sp


@pytest.mark.parametrize(
    ("n", "expected"),
    [
        (2.0, np.pi**2 / 6),
        (4.0, np.pi**4 / 90),
        (0.0, -0.5),
        (-1.0, -1 / 12),
        (-3.0, 1 / 120),
        (-2.0, 0.0),
        (-10.0, 0.0),
    ],
)
def test_known_values(n, expected):
    """Textbook values on both halves of the line."""
    np.testing.assert_allclose(sp.zeta(n), expected, rtol=1e-12, atol=1e-15)


def test_float_input_does_not_raise():
    """REGRESSION: the Bernoulli lookup used to be indexed with a float.

    ``bernoulli_ary[-n + 1]`` is a float index whenever ``n`` is a float, which
    JAX rejects outright.
    """
    np.testing.assert_allclose(sp.zeta(-3.0), 1 / 120, rtol=1e-12)


@pytest.mark.parametrize("n", [-61.0, -101.0, -1001.0])
def test_odd_negatives_past_the_table_are_nan(n):
    """REGRESSION: out-of-range indices used to be silently clamped.

    ``bernoulli_ary[-n + 1]`` past the end of the table is clamped by JAX to
    the last element, so ``zeta(-61)`` quietly returned ``zeta(-59)``'s
    Bernoulli number instead of failing.
    """
    assert jnp.isnan(sp.zeta(n))


@pytest.mark.parametrize("n", [-60.0, -100.0, -1000.0])
def test_even_negatives_are_zero_at_any_magnitude(n):
    """The trivial zeros need no table lookup, so they hold beyond -60."""
    assert float(sp.zeta(n)) == 0.0


def test_last_supported_odd_negative():
    """-59 is the deepest supported odd negative integer."""
    np.testing.assert_allclose(sp.zeta(-59.0), scipy_zeta(-59.0), rtol=1e-12)


def test_pole_at_one():
    """The point n = 1 is the pole."""
    assert jnp.isinf(sp.zeta(1.0))


@pytest.mark.parametrize("n", [0.1, 0.5, 0.9])
def test_critical_strip_is_unsupported(n):
    """0 < n <= 1 is `nan`: `jax.scipy.special.zeta` does not cover it.

    `scipy.special.zeta` does; this is a genuine gap, not a rounding issue.
    """
    assert jnp.isnan(sp.zeta(n))
    assert np.isfinite(scipy_zeta(n))


@pytest.mark.parametrize("n", [-0.5, -1.5, -2.5])
def test_non_integer_negatives_are_unsupported(n):
    """The functional equation used here needs an integer, so these are `nan`."""
    assert jnp.isnan(sp.zeta(n))
    assert np.isfinite(scipy_zeta(n))


def test_array_input():
    """Arrays are handled elementwise, mixing all the branches."""
    n = jnp.asarray([2.0, 0.0, -1.0, -2.0, -3.0])
    got = sp.zeta(n)
    assert got.shape == n.shape
    np.testing.assert_allclose(
        got, [np.pi**2 / 6, -0.5, -1 / 12, 0.0, 1 / 120], rtol=1e-12, atol=1e-15
    )


def test_jit():
    """`zeta` is jittable."""
    np.testing.assert_allclose(jax.jit(sp.zeta)(2.0), np.pi**2 / 6, rtol=1e-12)


def test_vmap():
    """`zeta` is vmappable."""
    n = jnp.asarray([2.0, -1.0, -3.0])
    np.testing.assert_allclose(jax.vmap(sp.zeta)(n), sp.zeta(n), rtol=1e-14)


@pytest.mark.parametrize("n", [2.0, 3.0, 10.0])
def test_grad_is_finite_and_correct_for_n_gt_1(n):
    """`jax.grad` works on the half-line that delegates to JAX's own zeta."""
    got = jax.grad(sp.zeta)(n)
    step = 1e-5
    expected = (float(sp.zeta(n + step)) - float(sp.zeta(n - step))) / (2 * step)
    np.testing.assert_allclose(got, expected, rtol=1e-6)


@pytest.mark.parametrize("n", [-1.0, -2.0, -3.0])
def test_grad_is_finite_but_meaningless_on_the_negative_line(n):
    """Documented caveat: the negative branch is a table lookup.

    `jax.grad` must at least stay finite there (the old ``(-1.0) ** k`` made
    the *whole* function's gradient `nan`), but the number it reports is not
    zeta'.
    """
    assert np.isfinite(jax.grad(sp.zeta)(n))


def test_dtype_is_float_for_integer_input():
    """Integer input is promoted to float."""
    assert jnp.issubdtype(sp.zeta(jnp.asarray([2, 3])).dtype, jnp.floating)
