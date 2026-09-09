"""Unit tests for `spexial.zeta`."""

import jax
import jax.numpy as jnp
import mpmath as mp
import numpy as np
import pytest
from scipy.special import zeta as scipy_zeta

import spexial as sp


def test_float_input_does_not_raise():
    """REGRESSION: the Bernoulli lookup used to be indexed with a float.

    ``bernoulli_ary[-n + 1]`` is a float index whenever ``n`` is a float, which
    JAX rejects outright.
    """
    np.testing.assert_allclose(sp.zeta(-3.0), 1 / 120, rtol=1e-12)


@pytest.mark.parametrize("n", [-61.0, -101.0, -1001.0])
def test_odd_negatives_past_the_table(n):
    """Beyond the Bernoulli table these come from the functional equation.

    They used to be `nan`, which was better than the silent index clamping it
    replaced -- `zeta(-61)` once returned `zeta(-59)`'s Bernoulli number -- but
    still a gap SciPy did not have. The reflection formula needs only
    `gamma` and `zeta(1 - n)` with `1 - n > 1`, both of which are available, so
    the whole negative half-line is now reachable.
    """
    with mp.workdps(30):
        expected = float(mp.zeta(n))
    np.testing.assert_allclose(sp.zeta(n), expected, rtol=1e-12)


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
def test_critical_strip(n):
    """`0 < n <= 1` comes from the eta series; upstream returns `nan` there.

    `jax.scipy.special.zeta` does not implement the Riemann zeta on the strip,
    and the functional equation does not help because it maps the strip onto
    itself. The alternating series converges there but far too slowly to use
    directly, so Borwein's acceleration supplies it in 32 terms.
    """
    with mp.workdps(30):
        expected = float(mp.zeta(n))
    np.testing.assert_allclose(sp.zeta(n), expected, rtol=1e-12)
    np.testing.assert_allclose(sp.zeta(n), scipy_zeta(n), rtol=1e-12)


@pytest.mark.parametrize("n", [-0.5, -1.5, -2.5])
def test_non_integer_negatives(n):
    """The reflection formula covers these; the Bernoulli one could not.

    `zeta(-k) = (-1)^k B_{k+1}/(k+1)` needs `(-1)^-n`, which is undefined for a
    non-integer, so these were `nan`. The functional equation has no such
    restriction.
    """
    with mp.workdps(30):
        expected = float(mp.zeta(n))
    np.testing.assert_allclose(sp.zeta(n), expected, rtol=1e-12)
    np.testing.assert_allclose(sp.zeta(n), scipy_zeta(n), rtol=1e-12)


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


@pytest.mark.parametrize("n", [54.0, 55.0, 100.0, 1e3, 1e15, 1e16, 1e300, np.inf])
def test_large_argument_is_exactly_one(n):
    """REGRESSION: `zeta(1e16)` was `nan`, inherited from JAX.

    `zeta(n) - 1 ~ 2**-n` drops below half an eps of 1 once `n > 53`, so every
    double-precision value from there up *is* `1.0` -- returning the constant is
    exact, not an approximation, and it steps around `jax.scipy.special.zeta`
    giving `nan` above n ~ 1e15. The docs previously recorded that `nan` as a
    permanent limitation; nothing tested it, which is how the claim went stale.
    """
    assert float(sp.zeta(n)) == 1.0
    if np.isfinite(n):
        assert scipy_zeta(n) == 1.0


def test_the_constant_branch_starts_where_it_is_exact():
    """Just below the cut-off the value is still computed, and still right."""
    # 1 + 2**-53 is representable, so zeta(53) must not be clamped to 1.0.
    assert float(sp.zeta(53.0)) == pytest.approx(float(scipy_zeta(53.0)), rel=1e-15)
    assert float(sp.zeta(20.0)) == pytest.approx(1.0000009539620338, rel=1e-15)


def test_large_argument_has_zero_derivative():
    """`zeta'(n) ~ -2**-n log 2`, which is 0 in float64 past the cut-off."""
    assert float(jax.grad(sp.zeta)(1e16)) == 0.0


def test_float32_is_not_silently_widened():
    """REGRESSION: a float32 argument came back float64.

    The Bernoulli table is float64 by construction (exact `Fraction`
    arithmetic), so indexing it widened the result. It is cast to the argument's
    dtype instead.
    """
    assert sp.zeta(jnp.asarray([2.0], dtype=jnp.float32)).dtype == jnp.float32


@pytest.mark.parametrize("n", [-(10.0**-k) for k in range(1, 21)])
def test_just_below_zero(n):
    """`zeta` stays accurate as `n` approaches 0 from below.

    The reflection forms ``1 - n``, which rounds to exactly ``1`` once ``|n|``
    falls under half an eps -- feeding the pole of ``zeta(1)`` into a formula
    whose answer is a finite ``-0.5``, and returning ``-inf``. The eta series
    covers a window below zero for this reason.
    """
    with mp.workdps(30):
        expected = float(mp.zeta(n))
    assert np.isfinite(sp.zeta(n))
    np.testing.assert_allclose(sp.zeta(n), expected, rtol=1e-12)


@pytest.mark.parametrize("n", [1.0 - 10.0**-k for k in range(1, 17)])
def test_approaching_the_pole_from_below(n):
    """The eta denominator must not cancel as `n` approaches 1.

    ``1 - 2**(1-n)`` loses every digit there and is exactly 0 half an eps below
    1, which made the value `inf` with the wrong sign. ``expm1`` of the same
    quantity keeps full precision, which is what the diverging value needs.
    """
    with mp.workdps(30):
        expected = float(mp.zeta(n))
    np.testing.assert_allclose(sp.zeta(n), expected, rtol=1e-12)


def test_nan_propagates():
    """A `nan` argument must not come back as a plausible number.

    ``nan > 0`` is False, so `nan` falls down the negative branch and picks up
    whatever placeholder the unselected branches are fed -- which came back as
    ``zeta(-0.5)``, indistinguishable from a real answer.
    """
    assert np.isnan(sp.zeta(np.nan))
    got = sp.zeta(jnp.asarray([np.nan, 2.0, -np.inf]))
    assert np.isnan(got[0])
    np.testing.assert_allclose(got[1], scipy_zeta(2.0), rtol=1e-12)
    assert np.isnan(got[2])


@pytest.mark.parametrize("k", [1, 2, 5, 25, 50])
@pytest.mark.parametrize("j", [6, 9, 12, 14])
@pytest.mark.parametrize("sign", [-1, 1])
def test_near_the_trivial_zeros(k, j, sign):
    """Just off a negative even integer, where the reflection's sine is small.

    ``sin(pi * n / 2)`` has to be evaluated by reducing the argument first: the
    product ``pi * n / 2`` carries an absolute rounding error larger than the
    sine itself there, which cost every digit -- 71% relative error at
    ``n = -102 + 4e-15``. SciPy has the same defect, so this compares against
    mpmath rather than against SciPy.
    """
    n = -2.0 * k + sign * 10.0**-j
    with mp.workdps(40):
        expected = float(mp.zeta(n))
    np.testing.assert_allclose(sp.zeta(n), expected, rtol=1e-11)


@pytest.mark.parametrize("n", [-0.25, -1.5, -3.5, -20.5, 0.5, 2.5])
def test_grad_matches_mpmath_off_the_integers(n):
    """Away from the negative integers the gradient is genuine.

    The functional equation differentiates, where the Bernoulli table cannot;
    only the tabulated integers keep the artefact documented above.
    """
    with mp.workdps(30):
        expected = float(mp.diff(mp.zeta, n))
    np.testing.assert_allclose(jax.grad(sp.zeta)(n), expected, rtol=1e-8)
