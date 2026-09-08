"""Unit tests for `spexial.K0`, `K1` and `K2`."""

import jax
import jax.numpy as jnp
import mpmath as mp
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


@pytest.mark.parametrize(("func", "order"), [(sp.K0, 0), (sp.K1, 1), (sp.K2, 2)])
def test_all_orders_at_zero_are_inf(func, order):
    """REGRESSION: every K_n diverges at 0, and scipy returns `inf` for each.

    `K1`'s closed form is `1/z - i1(z) * K0(z)`, which at 0 evaluates to
    `inf - 0 * inf == nan`; `K2` inherited that through the recurrence. Both
    returned `nan` where scipy returns `inf`.
    """
    assert jnp.isinf(func(0.0))
    assert jnp.isinf(scipy_kn(order, 0.0))


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
    # 1e-12, not bit-equality: XLA may fuse and reassociate the series
    # differently from the eager path, and JAX does not promise the two agree to
    # the last ulp. On the oldest supported jax they differ by 1.5e-14. This is
    # still four orders tighter than the function's own documented accuracy, so
    # it remains a real check that `jit` does not change the answer.
    np.testing.assert_allclose(jax.jit(func)(2.0), float(func(2.0)), rtol=1e-12)


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


@pytest.mark.parametrize("z", [699.5, 701.0, 705.0])
def test_k2_keeps_its_recurrence_term_in_the_subnormal_tail(z):
    """REGRESSION: `(2/z) * K1` was flushed to zero, silently dropping 0.3%.

    `K2 = K0 + (2/z) K1` computed directly puts that second term into the
    subnormal range around z = 699, and XLA on CPU flushes subnormal results to
    zero -- so `K2` returned exactly `K0` while still looking plausible, a 2.85e-3
    relative error against a documented 1e-6. Evaluating as
    `K0 * (1 + (2/z)(K1/K0))` keeps every intermediate normal.
    """
    # mpmath, not `scipy.kn`: the reference itself underflows to 0 at z ~ 698,
    # which is exactly the region under test.
    with mp.workdps(40):
        expected = float(mp.besselk(2, z))
    assert float(sp.K2(z)) > float(sp.K0(z))
    np.testing.assert_allclose(sp.K2(z), expected, rtol=1e-6)


@pytest.mark.parametrize("z", [0.5, 2.0, 8.5, 9.5, 20.0])
def test_grad_k1_is_minus_half_k0_plus_k2(z):
    """K1'(z) = -(K0(z) + K2(z)) / 2, via the custom JVP."""
    got = jax.grad(sp.K1)(z)
    np.testing.assert_allclose(got, -0.5 * (sp.K0(z) + sp.K2(z)), rtol=1e-12)


@pytest.mark.parametrize("z", [0.5, 2.0, 8.5, 9.5, 20.0])
def test_grad_k2_matches_the_recurrence(z):
    """K2'(z) = -K1(z) - (2/z) K2(z), via the custom JVP."""
    got = jax.grad(sp.K2)(z)
    np.testing.assert_allclose(got, -sp.K1(z) - 2.0 / z * sp.K2(z), rtol=1e-12)


@pytest.mark.parametrize("func", [sp.K0, sp.K1, sp.K2])
def test_custom_jvp_agrees_with_differentiating_the_series(func):
    """The analytic derivative must match what autodiff would have produced.

    A custom JVP silently replaces the true derivative, so this pins it against
    a finite-difference estimate rather than against itself.
    """
    z = 3.0
    h = 1e-6
    analytic = float(jax.grad(func)(z))
    numeric = float((func(z + h) - func(z - h)) / (2 * h))
    np.testing.assert_allclose(analytic, numeric, rtol=1e-7)


@pytest.mark.parametrize("func", [sp.K0, sp.K1, sp.K2])
def test_vjp_works(func):
    """`jax.vjp` is derived from the custom JVP, so it must work too."""
    z = jnp.asarray([1.0, 4.0])
    out, pullback = jax.vjp(func, z)
    (cotangent,) = pullback(jnp.ones_like(out))
    assert cotangent.shape == z.shape
    np.testing.assert_allclose(
        cotangent, jax.grad(lambda a: func(a).sum())(z), rtol=1e-12
    )
