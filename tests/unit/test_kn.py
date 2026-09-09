"""Unit tests for `spexial.K0`, `K1` and `K2`."""

import jax
import jax.numpy as jnp
import mpmath as mp
import numpy as np
import pytest
from scipy.special import (
    k0 as scipy_k0,
    k0e as scipy_k0e,
    k1 as scipy_k1,
    k1e as scipy_k1e,
    kn as scipy_kn,
    kve as scipy_kve,
)

import spexial as sp

# The series cross over at z = 9; ~9e-8 relative there, far better away from it.
RTOL = 1e-6

ORDERS = [(0, sp.K0), (1, sp.K1), (2, sp.K2)]

SCALED = [
    (sp.K0e, scipy_k0e),
    (sp.K1e, scipy_k1e),
    (sp.K2e, lambda z: scipy_kve(2, z)),
]

ALL_FUNCS = [sp.K0, sp.K1, sp.K2, sp.K0e, sp.K1e, sp.K2e]


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


@pytest.mark.parametrize("func", ALL_FUNCS)
def test_negative_argument_is_nan(func):
    """Negative z is outside the (real) domain."""
    assert jnp.isnan(func(-1.0))


def test_k1_underflows_above_705():
    """Past z = 705.5 the true value is subnormal, and XLA flushes it to 0.

    This is the float64 floor, not a limit of the algorithm: `K1e` is exact at
    the same z (see `test_scaled_survives_where_unscaled_underflows`), and
    `scipy.special.kn` also returns 0 here.
    """
    assert float(sp.K1(750.0)) == 0.0
    assert scipy_kn(1, 750.0) == 0.0


@pytest.mark.parametrize("func", ALL_FUNCS)
def test_jit(func):
    """The Bessel functions are jittable."""
    # 1e-12, not bit-equality: XLA may fuse and reassociate the series
    # differently from the eager path, and JAX does not promise the two agree to
    # the last ulp. On the oldest supported jax they differ by 1.5e-14. This is
    # still four orders tighter than the function's own documented accuracy, so
    # it remains a real check that `jit` does not change the answer.
    np.testing.assert_allclose(jax.jit(func)(2.0), float(func(2.0)), rtol=1e-12)


@pytest.mark.parametrize("func", ALL_FUNCS)
def test_vmap(func):
    """The Bessel functions are vmappable."""
    z = jnp.asarray([0.5, 4.0, 12.0])
    np.testing.assert_allclose(jax.vmap(func)(z), func(z), rtol=1e-14)


@pytest.mark.parametrize("z", [0.5, 2.0, 5.0, 20.0])
def test_grad_k0_is_minus_k1(z):
    """K0'(z) == -K1(z)."""
    np.testing.assert_allclose(jax.grad(sp.K0)(z), -scipy_kn(1, z), rtol=1e-6)


@pytest.mark.parametrize("func", ALL_FUNCS)
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


@pytest.mark.parametrize("func", ALL_FUNCS)
def test_custom_jvp_agrees_with_differentiating_the_series(func):
    """The analytic derivative must match what autodiff would have produced.

    A custom JVP silently replaces the true derivative, so this pins it against
    a finite-difference estimate rather than against itself.
    """
    z = 3.0
    # h = 1e-4, not 1e-6: the scaled forms vary ~7x more slowly relative to
    # their own size (|f'/f| is 0.16 for `K0e` against 1.16 for `K0`), so a
    # smaller step puts the difference into the functions' own noise floor.
    h = 1e-4
    analytic = float(jax.grad(func)(z))
    numeric = float((func(z + h) - func(z - h)) / (2 * h))
    np.testing.assert_allclose(analytic, numeric, rtol=1e-7)


@pytest.mark.parametrize("func", ALL_FUNCS)
def test_vjp_works(func):
    """`jax.vjp` is derived from the custom JVP, so it must work too."""
    z = jnp.asarray([1.0, 4.0])
    out, pullback = jax.vjp(func, z)
    (cotangent,) = pullback(jnp.ones_like(out))
    assert cotangent.shape == z.shape
    np.testing.assert_allclose(
        cotangent, jax.grad(lambda a: func(a).sum())(z), rtol=1e-12
    )


@pytest.mark.parametrize(("func", "reference"), SCALED)
@pytest.mark.parametrize("z", [0.01, 0.5, 2.0, 8.9, 9.0, 9.1, 50.0, 700.0, 1e5])
def test_scaled_matches_scipy(func, reference, z):
    """`K0e`/`K1e`/`K2e` equal scipy's scaled Bessel K, at any z.

    Unlike the unscaled forms there is no ceiling to work around: `e^z K_n(z)`
    decays only as `1/sqrt(z)`, so scipy stays a valid reference out to 1e5.
    """
    np.testing.assert_allclose(func(z), reference(z), rtol=RTOL)


@pytest.mark.parametrize(
    ("unscaled", "scaled"), [(sp.K0, sp.K0e), (sp.K1, sp.K1e), (sp.K2, sp.K2e)]
)
@pytest.mark.parametrize("z", [706.0, 800.0, 5000.0])
def test_scaled_survives_where_unscaled_underflows(unscaled, scaled, z):
    """The scaled form is exact exactly where the unscaled one has no value.

    Past z = 705.5 `K_n(z)` is smaller than the smallest normal double, so XLA
    returns 0 and nothing can be done about it. `e^z K_n(z)` is order 1e-2 and
    accurate to machine precision, which is the whole reason these three exist.
    """
    assert float(unscaled(z)) == 0.0
    assert float(scaled(z)) > 0.0


@pytest.mark.parametrize(("order", "func"), [(0, sp.K0e), (1, sp.K1e), (2, sp.K2e)])
@pytest.mark.parametrize("z", [706.0, 800.0, 5000.0, 1e5])
def test_scaled_is_exact_in_the_tail(order, func, z):
    """Against mpmath, the scaled forms are at machine precision past z = 705."""
    with mp.workdps(40):
        expected = float(mp.exp(z) * mp.besselk(order, z))
    np.testing.assert_allclose(func(z), expected, rtol=1e-13)


@pytest.mark.parametrize(("order", "func"), [(1, sp.K1), (2, sp.K2)])
@pytest.mark.parametrize("z", [699.0, 700.0, 703.0, 705.0])
def test_grad_keeps_its_recurrence_term_in_the_subnormal_tail(order, func, z):
    """REGRESSION: the derivative lost a term to the subnormal flush.

    `K2'(z) = -K1(z) - (2/z) K2(z)` formed directly puts the second term into
    the subnormal range from z ~ 699, where XLA on CPU flushes it to zero -- so
    `grad(K2)` was wrong by 2.86e-3 against a documented 1e-6, while still
    returning a plausible number. Exactly the bug that `K2`'s *value* had, left
    behind in its *derivative*. Both are now summed in the scaled variables and
    scaled down once.
    """
    with mp.workdps(40):
        expected = float(mp.diff(lambda t: mp.besselk(order, t), z))
    np.testing.assert_allclose(jax.grad(func)(z), expected, rtol=1e-6)


@pytest.mark.parametrize(("order", "func"), [(0, sp.K0e), (1, sp.K1e), (2, sp.K2e)])
@pytest.mark.parametrize("z", [0.5, 8.5, 9.5, 50.0, 800.0])
def test_scaled_grad_matches_mpmath(order, func, z):
    """The scaled custom JVPs, against mpmath's derivative of `e^z K_n(z)`.

    `rtol` is 1e-5, not the 1e-6 the values hold to: `(e^z K_n)' = e^z(K_n -
    K_{n+1})` subtracts two nearly equal numbers, and the cancellation costs
    about a decade of relative accuracy near the z = 9 cross-over (measured
    worst 1.7e-6). Away from it the derivative is good to 5e-13.
    """
    with mp.workdps(40):
        expected = float(mp.diff(lambda t: mp.exp(t) * mp.besselk(order, t), z))
    np.testing.assert_allclose(jax.grad(func)(z), expected, rtol=1e-5)


@pytest.mark.parametrize("func", [sp.K0e, sp.K1e, sp.K2e])
def test_scaled_at_zero_is_inf(func):
    """Every K_n diverges at 0, and scaling by e^0 = 1 does not change that."""
    assert jnp.isinf(func(0.0))


@pytest.mark.parametrize(
    ("unscaled", "scaled"), [(sp.K0, sp.K0e), (sp.K1, sp.K1e), (sp.K2, sp.K2e)]
)
@pytest.mark.parametrize("z", [0.5, 3.0, 8.999, 9.001, 20.0, 100.0])
def test_scaled_and_unscaled_agree(unscaled, scaled, z):
    """`K_n(z) == e^-z * (e^z K_n(z))`, across both branches."""
    np.testing.assert_allclose(unscaled(z), float(scaled(z)) * np.exp(-z), rtol=1e-13)


@pytest.mark.parametrize("func", [sp.K0e, sp.K1e, sp.K2e])
def test_scaled_array_input_is_elementwise(func):
    """Array input spans both branches and stays elementwise."""
    z = jnp.asarray([[0.5, 8.0], [10.0, 900.0]])
    got = func(z)
    assert got.shape == z.shape
    np.testing.assert_allclose(
        got, [[float(func(v)) for v in row] for row in z], rtol=1e-14
    )


@pytest.mark.parametrize(
    ("func", "reference"),
    [
        (sp.K0, scipy_k0),
        (sp.K1, scipy_k1),
        (sp.K2, lambda z: scipy_kn(2, z)),
        (sp.K0e, scipy_k0e),
        (sp.K1e, scipy_k1e),
        (sp.K2e, lambda z: scipy_kve(2, z)),
    ],
)
def test_positive_infinity_is_zero(func, reference):
    """REGRESSION: `+inf` gave `nan`; every K_n tends to 0 there.

    `_K0e_large` divides by `i0e(z)`, and `i0e(inf) == 0`, so the quotient was
    `inf * 0 == nan`. `K_n(inf) = 0` is a well-defined limit and `scipy.k0`,
    `k1`, `kn`, `k0e` and `k1e` all return it. (`scipy.kve(2, inf)` is `nan`,
    which is scipy being inconsistent with its own `k0e`/`k1e`.)
    """
    assert float(func(jnp.inf)) == 0.0
    expected = reference(np.inf)
    if not np.isnan(expected):
        assert expected == 0.0


@pytest.mark.parametrize("func", ALL_FUNCS)
def test_negative_infinity_is_nan(func):
    """`-inf` is outside the domain, like any negative argument."""
    assert jnp.isnan(func(-jnp.inf))
