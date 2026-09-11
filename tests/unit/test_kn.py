"""Unit tests for `spexial.k0`, `k1` and `k2`."""

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

ORDERS = [(0, sp.k0), (1, sp.k1), (2, sp.k2)]

SCALED = [
    (sp.k0e, scipy_k0e),
    (sp.k1e, scipy_k1e),
    (sp.k2e, lambda z: scipy_kve(2, z)),
]

ALL_FUNCS = [sp.k0, sp.k1, sp.k2, sp.k0e, sp.k1e, sp.k2e]


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


@pytest.mark.parametrize("func", [sp.k0, sp.k1, sp.k2])
def test_array_input_matches_scalar_calls(func):
    """Elementwise output equals looping over the same points."""
    z = jnp.asarray([0.2, 3.0, 9.0, 50.0])
    np.testing.assert_allclose(func(z), [float(func(v)) for v in z], rtol=1e-14)


def test_array_input_spans_both_branches():
    """A single array may straddle the z = 9 cross-over."""
    z = jnp.asarray([[1.0, 8.0], [10.0, 30.0]])
    np.testing.assert_allclose(sp.k0(z), scipy_kn(0, np.asarray(z)), rtol=RTOL)


@pytest.mark.parametrize(("func", "order"), [(sp.k0, 0), (sp.k1, 1), (sp.k2, 2)])
def test_all_orders_at_zero_are_inf(func, order):
    """REGRESSION: every K_n diverges at 0, and scipy returns `inf` for each.

    `k1`'s closed form is `1/z - i1(z) * k0(z)`, which at 0 evaluates to
    `inf - 0 * inf == nan`; `k2` inherited that through the recurrence. Both
    returned `nan` where scipy returns `inf`.
    """
    assert jnp.isinf(func(0.0))
    assert jnp.isinf(scipy_kn(order, 0.0))


def test_k0_at_zero_is_inf():
    """k0(0) diverges, as in scipy."""
    assert jnp.isinf(sp.k0(0.0))


@pytest.mark.parametrize("func", ALL_FUNCS)
def test_negative_argument_is_nan(func):
    """Negative z is outside the (real) domain."""
    assert jnp.isnan(func(-1.0))


def test_k1_underflows_above_705():
    """Past z = 705.5 the true value is subnormal, and XLA flushes it to 0.

    This is the float64 floor, not a limit of the algorithm: `k1e` is exact at
    the same z (see `test_scaled_survives_where_unscaled_underflows`), and
    `scipy.special.kn` also returns 0 here.
    """
    assert float(sp.k1(750.0)) == 0.0
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
    """k0'(z) == -k1(z).

    `rtol` is 1e-10, not the 1e-6 these functions are documented to: none of
    these four points is near the z = 9 cross-over, so the true error is 4.9e-13
    and a 1e-6 gate would sit ~190,000x above it -- passing a 10,000x
    regression in `grad(k0)` without noticing.
    """
    np.testing.assert_allclose(jax.grad(sp.k0)(z), -scipy_kn(1, z), rtol=1e-10)


@pytest.mark.parametrize("func", ALL_FUNCS)
def test_dtype_is_float_for_integer_input(func):
    """Integer input is promoted to float."""
    assert jnp.issubdtype(func(jnp.asarray([1, 2])).dtype, jnp.floating)


@pytest.mark.parametrize("z", [699.5, 701.0, 705.0])
def test_k2_keeps_its_recurrence_term_in_the_subnormal_tail(z):
    """REGRESSION: `(2/z) * k1` was flushed to zero, silently dropping 0.3%.

    `k2 = k0 + (2/z) k1` computed directly puts that second term into the
    subnormal range around z = 699, and XLA on CPU flushes subnormal results to
    zero -- so `k2` returned exactly `k0` while still looking plausible, a 2.85e-3
    relative error against a documented 1e-6. `k2` is now `k2e(z) * exp(-z)`:
    the recurrence is summed where both terms are order 1e-2 and scaled down
    once, so no intermediate is ever subnormal.

    `rtol` is 1e-14 for the same reason as the derivative test below: the value
    is exact here (measured 1.8e-16), so the documented 1e-6 would not notice a
    four-order regression of the very mechanism this test exists to guard.
    """
    # mpmath, not `scipy.kn`: the reference itself underflows to 0 at z ~ 698,
    # which is exactly the region under test.
    with mp.workdps(40):
        expected = float(mp.besselk(2, z))
    assert float(sp.k2(z)) > float(sp.k0(z))
    np.testing.assert_allclose(sp.k2(z), expected, rtol=1e-14)


@pytest.mark.parametrize("z", [0.5, 2.0, 8.5, 9.5, 20.0])
def test_grad_k1_is_minus_half_k0_plus_k2(z):
    """k1'(z) = -(k0(z) + k2(z)) / 2, via the custom JVP."""
    got = jax.grad(sp.k1)(z)
    np.testing.assert_allclose(got, -0.5 * (sp.k0(z) + sp.k2(z)), rtol=1e-12)


@pytest.mark.parametrize("z", [0.5, 2.0, 8.5, 9.5, 20.0])
def test_grad_k2_matches_the_recurrence(z):
    """k2'(z) = -k1(z) - (2/z) k2(z), via the custom JVP."""
    got = jax.grad(sp.k2)(z)
    np.testing.assert_allclose(got, -sp.k1(z) - 2.0 / z * sp.k2(z), rtol=1e-12)


@pytest.mark.parametrize("func", ALL_FUNCS)
def test_custom_jvp_agrees_with_differentiating_the_series(func):
    """The analytic derivative must match what autodiff would have produced.

    A custom JVP silently replaces the true derivative, so this pins it against
    a finite-difference estimate rather than against itself.
    """
    z = 3.0
    # h = 1e-4, not 1e-6: the scaled forms vary ~7x more slowly relative to
    # their own size (|f'/f| is 0.16 for `k0e` against 1.16 for `k0`), so a
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
    """`k0e`/`k1e`/`k2e` equal scipy's scaled Bessel K, at any z.

    Unlike the unscaled forms there is no ceiling to work around: `e^z K_n(z)`
    decays only as `1/sqrt(z)`, so scipy stays a valid reference out to 1e5.
    """
    np.testing.assert_allclose(func(z), reference(z), rtol=RTOL)


@pytest.mark.parametrize(
    ("unscaled", "scaled"), [(sp.k0, sp.k0e), (sp.k1, sp.k1e), (sp.k2, sp.k2e)]
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


@pytest.mark.parametrize(("order", "func"), [(0, sp.k0e), (1, sp.k1e), (2, sp.k2e)])
@pytest.mark.parametrize("z", [706.0, 800.0, 5000.0, 1e5])
def test_scaled_is_exact_in_the_tail(order, func, z):
    """Against mpmath, the scaled forms are at machine precision past z = 705."""
    with mp.workdps(40):
        expected = float(mp.exp(z) * mp.besselk(order, z))
    np.testing.assert_allclose(func(z), expected, rtol=1e-15)


@pytest.mark.parametrize(("order", "func"), [(1, sp.k1), (2, sp.k2)])
@pytest.mark.parametrize("z", [699.0, 700.0, 703.0, 705.0])
def test_grad_keeps_its_recurrence_term_in_the_subnormal_tail(order, func, z):
    """REGRESSION: the derivative lost a term to the subnormal flush.

    `k2'(z) = -k1(z) - (2/z) k2(z)` formed directly puts the second term into
    the subnormal range from z ~ 699, where XLA on CPU flushes it to zero -- so
    `grad(k2)` was wrong by 2.86e-3 against a documented 1e-6, while still
    returning a plausible number. Exactly the bug that `k2`'s *value* had, left
    behind in its *derivative*. Both are now summed in the scaled variables and
    scaled down once.

    `rtol` is 1e-14, not the 1e-6 these functions are documented to: the scaled
    sum is exact here (measured 3.1e-16), so a 1e-6 gate would still pass if the
    same mechanism regressed by four orders of magnitude.
    """
    with mp.workdps(40):
        expected = float(mp.diff(lambda t: mp.besselk(order, t), z))
    np.testing.assert_allclose(jax.grad(func)(z), expected, rtol=1e-14)


@pytest.mark.parametrize(("order", "func"), [(0, sp.k0e), (1, sp.k1e), (2, sp.k2e)])
@pytest.mark.parametrize("z", [0.5, 8.5, 8.999, 9.5, 50.0, 800.0])
def test_scaled_grad_matches_mpmath(order, func, z):
    """The scaled custom JVPs, against mpmath's derivative of `e^z K_n(z)`.

    `rtol` is 1e-5, not the 1e-6 the values hold to: `(e^z K_n)' = e^z(K_n -
    K_{n+1})` subtracts two nearly equal numbers, and the cancellation costs
    about a decade of relative accuracy near the z = 9 cross-over (measured
    worst 2.3e-6, at z = 8.999, which is one of the points below). Away from
    the cross-over the derivative is good to 5e-13.
    """
    with mp.workdps(40):
        expected = float(mp.diff(lambda t: mp.exp(t) * mp.besselk(order, t), z))
    np.testing.assert_allclose(jax.grad(func)(z), expected, rtol=1e-5)


@pytest.mark.parametrize("func", [sp.k0e, sp.k1e, sp.k2e])
def test_scaled_at_zero_is_inf(func):
    """Every K_n diverges at 0, and scaling by e^0 = 1 does not change that."""
    assert jnp.isinf(func(0.0))


@pytest.mark.parametrize(
    ("unscaled", "scaled"), [(sp.k0, sp.k0e), (sp.k1, sp.k1e), (sp.k2, sp.k2e)]
)
@pytest.mark.parametrize("z", [0.5, 3.0, 8.999, 9.001, 20.0, 100.0])
def test_scaled_and_unscaled_agree(unscaled, scaled, z):
    """`K_n(z) == e^-z * (e^z K_n(z))`, across both branches."""
    np.testing.assert_allclose(unscaled(z), float(scaled(z)) * np.exp(-z), rtol=1e-13)


@pytest.mark.parametrize("func", [sp.k0e, sp.k1e, sp.k2e])
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
        (sp.k0, scipy_k0),
        (sp.k1, scipy_k1),
        (sp.k2, lambda z: scipy_kn(2, z)),
        (sp.k0e, scipy_k0e),
        (sp.k1e, scipy_k1e),
        (sp.k2e, lambda z: scipy_kve(2, z)),
    ],
)
def test_positive_infinity_is_zero(func, reference):
    """REGRESSION: `+inf` gave `nan`; every K_n tends to 0 there.

    `_k0e_large` divides by `i0e(z)`, and `i0e(inf) == 0`, so the quotient was
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


@pytest.mark.parametrize("func", ALL_FUNCS)
def test_negative_zero_is_the_same_pole_as_positive_zero(func):
    """REGRESSION: `k2(-0.0)` and `k2e(-0.0)` were `nan`; scipy gives `inf`.

    `k1e`'s pole guard is `z == 0.0`, which is `True` for `-0.0`, so it returned
    `+inf` -- but `2.0 / -0.0` is `-inf`, and `k2e = k0e + (2/z) k1e` became
    `inf - inf == nan`. `k0`, `k1`, `k0e` and `k1e` were all correct at `-0.0`,
    so the library was inconsistent at a single point that ordinary arithmetic
    reaches: `jnp.asarray(0.0) * -1` is `-0.0`.
    """
    assert jnp.isinf(func(-0.0))
    assert jnp.isinf(func(jnp.asarray(0.0) * -1))


@pytest.mark.parametrize("func", ALL_FUNCS)
def test_derivative_at_the_pole_is_minus_infinity(func):
    """REGRESSION: the scaled rules gave `nan` where the unscaled gave `-inf`.

    `(e^z K_0)' = e^z (K_0 - K_1)` is `inf - inf` at 0. The true one-sided limit
    is `-inf`, which the unscaled rules already returned because each has a
    single divergent term -- so the two families disagreed at the pole.
    """
    assert float(jax.grad(func)(0.0)) == -np.inf
    assert float(jax.grad(func)(-0.0)) == -np.inf


@pytest.mark.parametrize(
    ("func", "reference"),
    [
        (sp.k0e, scipy_k0e),
        (sp.k1e, scipy_k1e),
        # `scipy.kve(2, z)` is `nan` this far out. `k2e = k0e + (2/z) k1e`, and that
        # second term is ~1e-462 here -- genuinely below the float64 range, not lost
        # information -- so `k0e` is the correct reference to a full 16 digits.
        (sp.k2e, scipy_k0e),
    ],
)
@pytest.mark.parametrize("z", [8.99e307, 1.0e308, 1.7e308])
def test_scaled_has_no_practical_ceiling(func, reference, z):
    """REGRESSION: the scaled forms returned 0 above z = 8.99e307.

    `_k0e_large` divided by `2.0 * z * i0e(z)`, which forms `2 * z` first --
    overflowing to `inf` above DBL_MAX/2 and sending the quotient to 0. Grouped
    as `2 * (z * i0e(z))` nothing overflows, because `z * i0e(z)` is
    ~sqrt(z / 2pi). `k1e` had a second instance of the same fault: the
    Wronskian's `1/z` term is itself subnormal above z = 4.5e307, so it returned
    exactly 0; multiplying numerator and denominator through by `z` keeps every
    term normal.
    """
    assert float(func(z)) > 0.0
    np.testing.assert_allclose(func(z), reference(z), rtol=1e-13)


# ---------------------------------------------------------------------------
# Low precision. The whole suite runs with `JAX_ENABLE_X64=True` (pinned in
# `pyproject.toml`), so without these the dtype-dependent cross-over -- the fix
# for the most serious bug found in this package -- is never evaluated at all.
# 100% branch coverage does not help: a one-line conditional `return` is not a
# branch as far as coverage.py is concerned.


@pytest.mark.parametrize("dtype", ["float32", "float16", "bfloat16"])
@pytest.mark.parametrize("func", ALL_FUNCS)
def test_low_precision_is_never_the_wrong_sign(dtype, func):
    """REGRESSION: `k0(float32(8.5))` was **negative** (true +8.6e-5).

    `_SMALL_Z` was 9.0 at every dtype. The ascending series cancels two terms of
    size e^z/sqrt(z) down to a result of size e^-z, costing ~2z/ln(10) decimal
    digits: float64 has 16 and still holds 8 at z = 9, float32 has 7 and had
    none. `float16`/`bfloat16` have no working cross-over at any z, so they are
    computed in float32 and rounded back.
    """
    z = jnp.asarray(np.linspace(0.5, 12.0, 60), dtype=dtype)
    got = np.asarray(func(z), dtype=np.float64)
    assert np.all(got > 0.0), f"{dtype}: {func.__name__} is not positive"


@pytest.mark.parametrize("dtype", ["float32", "float16", "bfloat16"])
@pytest.mark.parametrize(
    ("func", "reference"),
    [
        (sp.k0, scipy_k0),
        (sp.k1, scipy_k1),
        (sp.k0e, scipy_k0e),
        (sp.k1e, scipy_k1e),
    ],
)
def test_low_precision_accuracy(dtype, func, reference):
    """Each dtype gets roughly what it can represent, and no more is claimed.

    Measured worst over a dense sweep: 7.1e-3 for float32 (at the z = 4.65
    cross-over), 6.7e-3 for float16 and 1.9e-2 for bfloat16 -- the last two
    dominated by the quantisation of `z` itself, not by the algorithm.
    """
    z32 = np.linspace(0.5, 12.0, 120)
    z = np.asarray(
        jnp.asarray(z32, dtype=dtype), dtype=np.float64
    )  # as the dtype sees it
    got = np.asarray(func(jnp.asarray(z32, dtype=dtype)), dtype=np.float64)
    expected = reference(z)
    usable = expected > np.finfo(np.float32).tiny
    tol = {"float32": 1e-2, "float16": 2e-2, "bfloat16": 6e-2}[dtype]
    np.testing.assert_allclose(got[usable], expected[usable], rtol=tol)


@pytest.mark.parametrize("dtype", ["float16", "bfloat16"])
@pytest.mark.parametrize("func", ALL_FUNCS)
def test_narrow_dtypes_are_preserved(dtype, func):
    """Computing in float32 must not silently widen the caller's dtype.

    Returning float32 from a bfloat16 input would break a `lax.scan` carry, and
    JAX's own `i0`/`log` preserve these dtypes.
    """
    assert func(jnp.asarray(2.0, dtype=dtype)).dtype == jnp.dtype(dtype)


def test_integer_input_still_promotes():
    """Integer input has no narrow dtype to go back to, so it stays promoted."""
    assert jnp.issubdtype(sp.k0(jnp.asarray([1, 2])).dtype, jnp.floating)


@pytest.mark.parametrize(("order", "func"), [(0, sp.k0), (1, sp.k1)])
@pytest.mark.parametrize("z", [2.3e-308, 3e-308, 4.45e-308, 1e-300, 1e-100])
def test_tiny_but_normal_argument(order, func, z):
    """REGRESSION: `k0(3e-308)` was `inf` and `k1(3e-308)` was `nan`.

    `_k0_small` computed `log(z / 2)`. Halving a `z` that is small but
    perfectly *normal* lands in the subnormal range, XLA on CPU flushes it to
    zero, and `log(0)` is `-inf` -- so the whole band 2.2e-308 <= z < 4.45e-308
    returned `inf`/`nan` where the true values are ~708 and ~3e307. Note this is
    the opposite of the documented subnormal limitation, which is about
    subnormal *outputs*: here both input and output are normal.
    """
    reference = (scipy_k0, scipy_k1)[order]
    np.testing.assert_allclose(func(z), reference(z), rtol=RTOL)


@pytest.mark.parametrize(("order", "func"), [(0, sp.k0), (1, sp.k1), (2, sp.k2)])
@pytest.mark.parametrize("z", [694.0, 700.0, 703.0, 705.0])
def test_second_derivative_in_the_subnormal_tail(order, func, z):
    """REGRESSION: `grad(grad(k1))` was 7.2e-4 low from z ~ 694.

    The first derivative was fixed to sum in scaled variables, but it was an
    *expression* inside the JVP, so differentiating it again formed
    `d(1/z) * k1 * e^-z` -- about 4e-312 at z = 700, which XLA flushes. Exactly
    the bug the first derivative was fixed for, one order up, and invisible
    because no test went past the first derivative here. `k1'` and `k2'` are now
    named functions carrying their own rules, so the second derivative is summed
    scaled too.
    """
    with mp.workdps(50):
        expected = float(mp.diff(lambda t: mp.besselk(order, t), z, 2))
    np.testing.assert_allclose(jax.grad(jax.grad(func))(z), expected, rtol=1e-13)


@pytest.mark.parametrize("dtype", ["float32", "float16", "bfloat16"])
@pytest.mark.parametrize("func", ALL_FUNCS)
def test_narrow_dtypes_survive_differentiation(dtype, func):
    """REGRESSION: `grad(k2)` *raised* on bfloat16, and `jvp` widened the primal.

    `_k2_jvp` narrowed its primal with `_cast_like` but left the tangent at the
    internal width, so JAX rejected the rule outright ("must produce primal and
    tangent outputs with corresponding ... dtypes"). The other five narrowed
    neither, so `jax.jvp` quietly returned a wider primal than the plain call.
    Value, `grad` and `jvp` must all agree with each other.
    """
    z = jnp.asarray(2.0, dtype=dtype)
    tangent = jnp.asarray(1.0, dtype=dtype)
    primal, tangent_out = jax.jvp(func, (z,), (tangent,))
    assert func(z).dtype == jnp.dtype(dtype)
    assert jax.grad(func)(z).dtype == jnp.dtype(dtype)
    assert primal.dtype == jnp.dtype(dtype)
    assert tangent_out.dtype == jnp.dtype(dtype)


@pytest.mark.parametrize("func", ALL_FUNCS)
def test_float32_is_not_silently_widened(func):
    """REGRESSION: float32 input returned float64 whenever x64 was enabled.

    `jnp.arange(1.0, n)` defaults to float64 under x64, so the series constants
    promoted the whole computation. Nothing caught it: `_cast_like` narrowed
    only float16/bfloat16, and every test runs in float64. This is the same
    hazard the narrow-dtype test guards -- a widened output breaks a `lax.scan`
    carry -- one dtype up.
    """
    assert func(jnp.asarray([2.0], dtype=jnp.float32)).dtype == jnp.float32


@pytest.mark.parametrize("func", ALL_FUNCS)
def test_second_derivative_at_the_pole_is_positive_infinity(func):
    """REGRESSION: the scaled family gave `nan` where the unscaled gave `inf`.

    Every K_n'' diverges to `+inf` at 0 (the `1/z^2` term dominates), but the
    closed forms are `inf - inf` there, and `2/|z|` has derivative `sign(0) = 0`
    so the product was `-inf * 0`. `test_derivative_at_the_pole_is_minus_infinity`
    stops the two families disagreeing at first order; this does the same at
    second order.
    """
    assert float(jax.grad(jax.grad(func))(0.0)) == np.inf
    assert float(jax.grad(jax.grad(func))(-0.0)) == np.inf


@pytest.mark.parametrize(("order", "func"), [(0, sp.k0e), (1, sp.k1e), (2, sp.k2e)])
@pytest.mark.parametrize("z", [0.5, 3.0, 50.0, 700.0])
def test_scaled_second_derivative(order, func, z):
    """The scaled second derivatives, against mpmath.

    `rtol` is 1e-9, looser than the values' own 2.0e-7 would suggest is needed,
    because `(e^z K_0)'' = 2G0 - 2G1 + G1/z` subtracts two nearly equal numbers:
    at z = 8.9 the cancellation costs about two decades, and it worsens with z
    (measured 3e-6 at z = 1e5). The absolute error stays at machine precision.
    Points near the cross-over are therefore excluded and covered by the
    documented figure instead.
    """
    with mp.workdps(50):
        expected = float(mp.diff(lambda t: mp.exp(t) * mp.besselk(order, t), z, 2))
    np.testing.assert_allclose(jax.grad(jax.grad(func))(z), expected, rtol=1e-9)


@pytest.mark.parametrize(("order", "func"), [(1, sp.k1), (2, sp.k2)])
@pytest.mark.parametrize("derivative_order", [1, 2])
def test_derivatives_are_exact_to_second_order_in_the_tail(
    order, func, derivative_order
):
    """Orders 1 and 2 are exact at z = 700; order 3 is documented as not.

    Pinned separately from `test_second_derivative_in_the_subnormal_tail` so
    that the *boundary* of the guarantee is explicit: if a future change makes
    order 3 exact too, the companion test below fails and the docs get updated.
    """
    g = func
    for _ in range(derivative_order):
        g = jax.grad(g)
    with mp.workdps(50):
        expected = float(
            mp.diff(lambda t: mp.besselk(order, t), 700.0, derivative_order)
        )
    np.testing.assert_allclose(g(700.0), expected, rtol=1e-13)


def test_third_derivative_in_the_tail_is_a_documented_limitation():
    """Order 3 loses 7.2e-4 above z ~ 690, and that is stated rather than fixed.

    Each autodiff pass expands a Leibniz product of a scaled quantity with
    `e^-z`; the individual terms go subnormal even though their sum does not
    (`(4/z**3) * k1e * e^-z` is 5.4e-314 at z = 700). Naming a third derivative
    function would move the wall to order 4 rather than remove it -- orders 3
    and 4 are both 7.2e-4 low today -- so the ceiling is documented instead.
    This test fails if that ever stops being true, which is the point.
    """
    with mp.workdps(50):
        expected = float(mp.diff(lambda t: mp.besselk(1, t), 700.0, 3))
    got = float(jax.grad(jax.grad(jax.grad(sp.k1)))(700.0))
    assert 1e-4 < abs(got / expected - 1) < 1e-2, (
        "grad^3 in the tail changed; re-measure and update the docs"
    )


@pytest.mark.parametrize("z", [1e-310, 1e-320, 5e-324])
def test_subnormal_argument(z):
    """A subnormal `z` must not come back as `inf`.

    XLA on CPU flushes a subnormal *input* to zero, so `jnp.log(z)` was `-inf`
    and every `k0` below `tiny` was `inf` where the true value is an ordinary
    number near 700. This is a different fault from the `log(z) - log(2)`
    rearrangement, which only stops a *normal* `z` being halved into the
    subnormal range.
    """
    expected = float(mp.besselk(0, z))
    np.testing.assert_allclose(sp.k0(z), expected, rtol=1e-13)


@pytest.mark.parametrize("z", [1e-38, 1e-40, 1e-45])
def test_subnormal_argument_float32(z):
    """The same band in float32, where it starts at an ordinary 1.2e-38."""
    argument = jnp.asarray(z, dtype=jnp.float32)
    got = sp.k0(argument)
    assert got.dtype == jnp.float32
    # Against the float32 value actually held, not the decimal literal: the
    # smallest subnormals have a bit or two of mantissa, so `float32(1e-45)` is
    # 1.401e-45, and `k0` of the two differs in the third digit.
    np.testing.assert_allclose(got, float(mp.besselk(0, float(argument))), rtol=1e-6)


@pytest.mark.parametrize("z", [-1.0, -1e-310, -1e-320])
def test_negative_subnormal_is_nan_for_k0(z):
    """Sign has to be read off the bits.

    XLA compares a subnormal as if it were zero, so `z > 0` is False for
    exactly the positive values the subnormal branch exists to catch, and
    `z < tiny` is True for every negative. Reading the sign bit separates them;
    getting this wrong turned every negative argument into `inf`.
    """
    assert jnp.isnan(sp.k0(z))


def test_signed_zero_is_still_the_pole():
    """`-0.0` is the one negative bit pattern that is not out of domain."""
    assert jnp.isinf(sp.k0(-0.0))
    assert jnp.isinf(sp.k0(0.0))


@pytest.mark.parametrize("z", [1e-38, 5e-39, 3e-39])
def test_k1_in_the_subnormal_band_where_one_over_z_still_fits(z):
    """`k1(z) -> 1/z`, and in float32 that is representable below `tiny`.

    The Wronskian form divides by `z * i0e(z)`, which flushes to zero for a
    subnormal `z` and made the quotient `inf`. The band where this matters is a
    factor of about two wide -- `tiny * max` is ~2 in any IEEE format -- but in
    float32 it is the reachable 2.9e-39 to 1.2e-38.
    """
    argument = jnp.asarray(z, dtype=jnp.float32)
    got = sp.k1(argument)
    assert got.dtype == jnp.float32
    np.testing.assert_allclose(got, float(mp.besselk(1, float(argument))), rtol=1e-5)


@pytest.mark.parametrize("z", [2e-39, 1e-40, 1e-310])
def test_k1_is_infinite_only_where_one_over_z_overflows(z):
    """Past the band the true value exceeds the dtype, and `inf` is correct."""
    dtype = jnp.float32 if z > 1e-45 else jnp.float64
    assert jnp.isinf(sp.k1(jnp.asarray(z, dtype=dtype)))


@pytest.mark.parametrize("func", [sp.k0e, sp.k1e, sp.k2e])
def test_negative_subnormal_gradient_is_nan(func):
    """The pole guard used `z >= 0.0`, which XLA reads as True for these.

    So a negative subnormal -- out of domain, and `nan` in the value -- was
    handed the pole limit and came back `-inf` in the gradient.
    """
    z = jnp.asarray(-1e-320)
    assert jnp.isnan(func(z))
    assert jnp.isnan(jax.grad(func)(z))


@pytest.mark.parametrize("wrap", [lambda f: f, jax.jit], ids=["eager", "jit"])
@pytest.mark.parametrize("z", [2.2e-308, 1.112537e-308, 6e-309])
def test_k1_subnormal_band_is_the_same_jitted(wrap, z):
    """The subnormal band must give the same answer compiled as interpreted.

    It did not. Separating the pole from the subnormals with a bit test is
    correct in isolation and wrong once XLA fuses the select into the same
    kernel, so `jit(k1)` returned `inf` across the whole band while eager
    returned the right value -- and every test here called it eagerly.
    """
    got = float(wrap(sp.k1)(jnp.asarray(z)))
    np.testing.assert_allclose(got, float(mp.besselk(1, z)), rtol=1e-13)


@pytest.mark.parametrize("wrap", [lambda f: f, jax.jit], ids=["eager", "jit"])
@pytest.mark.parametrize("func", [sp.k0, sp.k1, sp.k0e, sp.k1e, sp.k2, sp.k2e])
def test_negative_subnormal_is_out_of_domain_jitted(wrap, func):
    """A negative subnormal is out of domain, compiled or not.

    Under `jit` the fused bit test read it as the pole and returned `inf`.
    """
    assert jnp.isnan(wrap(func)(jnp.asarray(-1.112537e-308)))


@pytest.mark.parametrize("f", [sp.k0, sp.k1, sp.k2, sp.k0e, sp.k1e, sp.k2e])
@pytest.mark.parametrize("sign", [1.0, -1.0], ids=["nan", "-nan"])
def test_derivative_of_nan_is_nan(f, sign):
    """REGRESSION: the pole guard fired on a `nan` argument.

    `_at_pole` keyed on the derivative being non-finite and on ``z`` not being
    negative. A `nan` makes the derivative `nan` too and has an unset sign bit,
    so the scaled family handed back ``-inf`` for the first derivative and
    ``+inf`` for the second -- a definite answer for an argument whose *value*
    is `nan`, and one that flipped to `nan` for ``-nan``, i.e. turned on a bit
    that carries no meaning. `scipy.special.kve(0, nan)` is `nan`.
    """
    argument = jnp.asarray(sign * np.nan)
    assert np.isnan(float(f(argument)))
    assert np.isnan(float(jax.grad(f)(argument)))
    assert np.isnan(float(jax.grad(jax.grad(f))(argument)))


@pytest.mark.parametrize("z", [1e-154, 1e-120, 1e-104, 0.0])
def test_second_derivative_at_the_pole_agrees_between_modes(z):
    """REGRESSION: ``jit(grad(grad(k2e)))`` was ``-inf`` where eager was `nan`.

    Both stand for the same ``inf - inf``, and which one it comes out as is a
    property of the graph, not of the arithmetic: XLA reassociates the sum under
    `jit`. Keying the guard on `isnan` therefore missed it in exactly the mode
    users run, and returned the pole's infinity with the sign flipped --
    ``(e^z K_2)'' ~ 12/z**4`` is ``+inf``.
    """
    second = jax.grad(jax.grad(sp.k2e))
    argument = jnp.asarray(z)
    assert float(second(argument)) == np.inf
    assert float(jax.jit(second)(argument)) == np.inf
    assert float(jax.jit(jax.vmap(second))(jnp.asarray([z]))[0]) == np.inf


@pytest.mark.parametrize(
    ("dtype", "ceilings"),
    [
        (jnp.float64, (705.34, 705.34, 705.35)),
        (jnp.float32, (85.34, 85.34, 85.36)),
        (jnp.float16, (16.15, 16.18, 16.27)),
    ],
    ids=["float64", "float32", "float16"],
)
def test_underflow_ceiling_per_dtype(dtype, ceilings):
    """Where each unscaled `K` reaches zero, which is a property of the *dtype*.

    The docstrings quoted float64's 705 for every width; float16 dies at 16.
    """
    for f, ceiling in zip((sp.k0, sp.k1, sp.k2), ceilings, strict=True):
        assert float(f(jnp.asarray(ceiling * 0.99, dtype=dtype))) > 0.0
        assert float(f(jnp.asarray(ceiling * 1.01, dtype=dtype))) == 0.0
