"""Unit tests for `spexial.Li`."""

from functools import partial

import jax
import jax.numpy as jnp
import mpmath as mp
import numpy as np
import pytest

import spexial as sp
from spexial._src.polylog import _bernoulli_poly


def reference(n, z):
    """Real part of mpmath's polylogarithm."""
    return complex(mp.polylog(n, z)).real


def test_li_at_one_is_zeta():
    """Li_n(1) == zeta(n)."""
    for n in (2, 3, 4):
        np.testing.assert_allclose(sp.Li(n, 1.0), float(sp.zeta(float(n))), rtol=1e-11)


@pytest.mark.parametrize("n", [1, 2, 3, 4])
@pytest.mark.parametrize("z", [-2.0, 2.0])
def test_branch_boundary_at_abs_z_equals_two(n, z):
    """REGRESSION: |z| == 2 fell through every branch and returned 0.

    The branches were ``|z| <= 0.5``, ``0.5 < |z| < 2`` and ``|z| > 2``, which
    leaves ``|z| == 2`` uncovered; `jnp.where` then produced the 0.0 default.
    """
    got = float(sp.Li(n, z))
    assert got != 0.0
    np.testing.assert_allclose(got, reference(n, z), rtol=1e-11, atol=1e-12)


@pytest.mark.parametrize("z", [0.25, 0.75, 3.0])
def test_high_order_does_not_overflow_int64(z):
    """`j ** n` over a traced integer `j` overflows int64 for n >= 12."""
    got = float(sp.Li(20, z))
    assert np.isfinite(got)
    np.testing.assert_allclose(got, reference(20, z), rtol=1e-11, atol=1e-12)


@pytest.mark.parametrize("n", [0, -1])
def test_non_positive_order_is_rejected(n):
    """Only n >= 1 is implemented; say so instead of returning nonsense."""
    with pytest.raises(ValueError, match="n >= 1"):
        sp.Li(n, 0.25)


def test_array_input_is_rejected():
    """`Li` is documented as scalar-only; the type checker enforces it."""
    with pytest.raises(Exception, match=r"(?i)typecheck"):
        sp.Li(2, jnp.asarray([0.1, 0.2]))


def test_vmap_is_the_supported_way_to_batch():
    """`jax.vmap` gives the elementwise behaviour `Li` itself does not."""
    z = jnp.asarray([0.1, 0.3, 0.9, 3.0])
    got = jax.vmap(partial(sp.Li, 2))(z)
    expected = [reference(2, float(v)) for v in z]
    np.testing.assert_allclose(got, expected, rtol=1e-11, atol=1e-12)


def test_jit():
    """`Li` is jittable (order is a static argument)."""
    np.testing.assert_allclose(sp.Li(2, 0.3), reference(2, 0.3), rtol=1e-11)


@pytest.mark.parametrize("z", [0.3, 0.75, 3.0])
def test_grad(z):
    """d/dz Li_n(z) == Li_{n-1}(z) / z."""
    got = jax.grad(partial(sp.Li, 3))(z)
    np.testing.assert_allclose(got, reference(2, z) / z, rtol=1e-9)


def test_bernoulli_polynomial():
    """B_n(x) helper: B_2(x) == x^2 - x + 1/6."""
    for x in (0.0, 1.0, 3.0):
        np.testing.assert_allclose(
            _bernoulli_poly(2, jnp.asarray(x)), x**2 - x + 1 / 6, rtol=1e-12, atol=1e-15
        )


@pytest.mark.parametrize("n", [1, 2, 3])
@pytest.mark.parametrize("delta", [5e-9, 1e-10, 1e-12])
def test_near_z_equals_one_is_not_snapped_to_the_pole(n, delta):
    """REGRESSION: `jnp.isclose` swallowed a 1e-8 neighbourhood of z = 1.

    The branch used `jnp.isclose(z - 1.0, 0.0)`, whose default `atol` is 1e-8,
    so every `z` within that of 1 was treated as *exactly* 1 and returned
    `zeta(n)`. For n = 1 that is wrong by 100% (2.5e-9 instead of 19.1); for
    n = 2 by 6e-8, which is 6000x the parity suite's own tolerance.
    """
    z = 1.0 - delta
    with mp.workdps(30):
        expected = float(complex(mp.polylog(n, z)).real)
    np.testing.assert_allclose(sp.Li(n, z), expected, rtol=1e-11)


def test_li1_at_one_is_the_pole():
    """Li_1(1) diverges; it used to return 0.0."""
    assert jnp.isinf(sp.Li(1, 1.0))


@pytest.mark.parametrize("n", [2, 5, 20])
def test_li_at_one_is_zeta_for_higher_orders(n):
    """Li_n(1) == zeta(n) for n >= 2, where the pole is absent."""
    np.testing.assert_allclose(sp.Li(n, 1.0), sp.zeta(float(n)), rtol=1e-13)


@pytest.mark.parametrize("n", [61, 62, 70])
def test_order_past_the_bernoulli_table_is_nan(n):
    """REGRESSION: the inversion branch silently reused the last Bernoulli number.

    `_bernoulli_poly` indexes the table up to `n`, and an out-of-bounds index is
    *clamped* under `jit` rather than raising, so `Li(62, 3.0)` returned 0.979
    where the true value is 3.0. Orders past the table now say so.
    """
    assert jnp.isnan(sp.Li(n, 3.0))


@pytest.mark.parametrize("n", [61, 70, 150])
def test_high_order_still_works_below_the_inversion_branch(n):
    """Only |z| >= 2 needs the Bernoulli table; the other branches are unaffected."""
    with mp.workdps(30):
        expected = float(complex(mp.polylog(n, 0.5)).real)
    np.testing.assert_allclose(sp.Li(n, 0.5), expected, rtol=1e-11)


@pytest.mark.parametrize("n", [1, 2, 3, 5, 12])
@pytest.mark.parametrize("z", [-2.5, -0.3, 0.2, 0.45, 0.9])
def test_custom_jvp_matches_a_finite_difference(n, z):
    """The analytic derivative must be the real one, not merely self-consistent.

    A wrong `custom_jvp` leaves every value correct and every gradient silently
    wrong, so this is pinned against a central difference rather than against
    the identity it was derived from.
    """
    h = 1e-6
    analytic = float(jax.grad(lambda a: sp.Li(n, a))(z))
    numeric = float((sp.Li(n, z + h) - sp.Li(n, z - h)) / (2 * h))
    np.testing.assert_allclose(analytic, numeric, rtol=1e-6)


@pytest.mark.parametrize("n", [2, 4, 7])
def test_derivative_is_the_lower_order_polylog(n):
    """d/dz Li_n(z) = Li_{n-1}(z) / z for n >= 2."""
    z = 0.35
    got = jax.grad(lambda a: sp.Li(n, a))(z)
    np.testing.assert_allclose(got, sp.Li(n - 1, z) / z, rtol=1e-12)


def test_order_one_derivative_is_the_special_case():
    """Li_1(z) = -log(1-z), so its derivative is 1/(1-z).

    The general identity would need `Li_0`, which `Li` refuses to compute.
    """
    z = 0.35
    np.testing.assert_allclose(
        jax.grad(lambda a: sp.Li(1, a))(z), 1 / (1 - z), rtol=1e-12
    )
