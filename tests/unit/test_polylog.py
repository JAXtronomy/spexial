"""Unit tests for `spexial.polylog`."""

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
        np.testing.assert_allclose(
            sp.polylog(n, 1.0), float(sp.zeta(float(n))), rtol=1e-11
        )


@pytest.mark.parametrize("n", [1, 2, 3, 4])
@pytest.mark.parametrize("z", [-2.0, 2.0])
def test_branch_boundary_at_abs_z_equals_two(n, z):
    """REGRESSION: |z| == 2 fell through every branch and returned 0.

    The branches were ``|z| <= 0.5``, ``0.5 < |z| < 2`` and ``|z| > 2``, which
    leaves ``|z| == 2`` uncovered; `jnp.where` then produced the 0.0 default.
    """
    got = float(sp.polylog(n, z))
    assert got != 0.0
    np.testing.assert_allclose(got, reference(n, z), rtol=1e-11, atol=1e-12)


@pytest.mark.parametrize("z", [0.25, 0.75, 3.0])
def test_high_order_does_not_overflow_int64(z):
    """`j ** n` over a traced integer `j` overflows int64 for n >= 12."""
    got = float(sp.polylog(20, z))
    assert np.isfinite(got)
    np.testing.assert_allclose(got, reference(20, z), rtol=1e-11, atol=1e-12)


@pytest.mark.parametrize("n", [0, -1])
def test_non_positive_order_is_rejected(n):
    """Only n >= 1 is implemented; say so instead of returning nonsense."""
    with pytest.raises(ValueError, match="n >= 1"):
        sp.polylog(n, 0.25)


def test_array_input_is_rejected():
    """`polylog` is documented as scalar-only; the type checker enforces it."""
    with pytest.raises(Exception, match=r"(?i)typecheck"):
        sp.polylog(2, jnp.asarray([0.1, 0.2]))


def test_vmap_is_the_supported_way_to_batch():
    """`jax.vmap` gives the elementwise behaviour `polylog` itself does not."""
    z = jnp.asarray([0.1, 0.3, 0.9, 3.0])
    got = jax.vmap(partial(sp.polylog, 2))(z)
    expected = [reference(2, float(v)) for v in z]
    np.testing.assert_allclose(got, expected, rtol=1e-11, atol=1e-12)


def test_jit():
    """`polylog` is jittable (order is a static argument)."""
    np.testing.assert_allclose(sp.polylog(2, 0.3), reference(2, 0.3), rtol=1e-11)


@pytest.mark.parametrize("z", [0.3, 0.75, 3.0])
def test_grad(z):
    """d/dz Li_n(z) == Li_{n-1}(z) / z."""
    got = jax.grad(partial(sp.polylog, 3))(z)
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
    np.testing.assert_allclose(sp.polylog(n, z), expected, rtol=1e-11)


def test_li1_at_one_is_the_pole():
    """Li_1(1) diverges; it used to return 0.0."""
    assert jnp.isinf(sp.polylog(1, 1.0))


@pytest.mark.parametrize("n", [2, 5, 20])
def test_li_at_one_is_zeta_for_higher_orders(n):
    """Li_n(1) == zeta(n) for n >= 2, where the pole is absent."""
    np.testing.assert_allclose(sp.polylog(n, 1.0), sp.zeta(float(n)), rtol=1e-13)


@pytest.mark.parametrize("n", [61, 62, 70])
def test_order_past_the_bernoulli_table_is_nan(n):
    """REGRESSION: the inversion branch silently reused the last Bernoulli number.

    `_bernoulli_poly` indexes the table up to `n`, and an out-of-bounds index is
    *clamped* under `jit` rather than raising, so `polylog(62, 3.0)` returned 0.979
    where the true value is 3.0. Orders past the table now say so.
    """
    assert jnp.isnan(sp.polylog(n, 3.0))


@pytest.mark.parametrize("n", [61, 70, 150])
def test_high_order_still_works_below_the_inversion_branch(n):
    """Only |z| >= 2 needs the Bernoulli table; the other branches are unaffected."""
    with mp.workdps(30):
        expected = float(complex(mp.polylog(n, 0.5)).real)
    np.testing.assert_allclose(sp.polylog(n, 0.5), expected, rtol=1e-11)


@pytest.mark.parametrize("n", [1, 2, 3, 5, 12])
@pytest.mark.parametrize("z", [-2.5, -0.3, 0.2, 0.45, 0.9])
def test_custom_jvp_matches_a_finite_difference(n, z):
    """The analytic derivative must be the real one, not merely self-consistent.

    A wrong `custom_jvp` leaves every value correct and every gradient silently
    wrong, so this is pinned against a central difference rather than against
    the identity it was derived from.
    """
    h = 1e-6
    analytic = float(jax.grad(lambda a: sp.polylog(n, a))(z))
    numeric = float((sp.polylog(n, z + h) - sp.polylog(n, z - h)) / (2 * h))
    np.testing.assert_allclose(analytic, numeric, rtol=1e-6)


@pytest.mark.parametrize("n", [2, 4, 7])
def test_derivative_is_the_lower_order_polylog(n):
    """d/dz Li_n(z) = Li_{n-1}(z) / z for n >= 2."""
    z = 0.35
    got = jax.grad(lambda a: sp.polylog(n, a))(z)
    np.testing.assert_allclose(got, sp.polylog(n - 1, z) / z, rtol=1e-12)


def test_order_one_derivative_is_the_special_case():
    """Li_1(z) = -log(1-z), so its derivative is 1/(1-z).

    The general identity would need `Li_0`, which `polylog` refuses to compute.
    """
    z = 0.35
    np.testing.assert_allclose(
        jax.grad(lambda a: sp.polylog(1, a))(z), 1 / (1 - z), rtol=1e-12
    )


@pytest.mark.parametrize("n", [2.0, 1.0, 2.5, -0.5, "2"])
def test_non_integer_order_is_rejected(n):
    """A non-integer order is a `TypeError`, and says so clearly.

    The docs promised a `ValueError` here, which was wrong twice over: the body
    only validates `n < 1`, and the `n: int` annotation gets there first under
    the runtime type checker. What a caller actually sees is a
    `jaxtyping.TypeCheckError` -- a `TypeError` subclass -- naming the parameter
    and the expected type, which is the right error for the wrong type. Pinned
    so the documented exception cannot drift from the real one again.
    """
    with pytest.raises(TypeError, match="n"):
        sp.polylog(n, 0.3)


@pytest.mark.parametrize("z", [0.3 + 0.1j, 3.0 + 1.0j])
def test_complex_argument_is_rejected(z):
    """REGRESSION: complex `z` silently lost its imaginary part in two branches.

    The `|z| <= 1/2` series returned the true complex value, while `expansion`
    and `inversion` take `jnp.real` of a complex intermediate -- exact for real
    `z`, where the imaginary parts cancel, but for complex `z` it returned a
    plausible number with the imaginary part discarded (`polylog(2, 3+1j)` gave
    1.3459 + 0j against mpmath's 1.3459 + 3.3651j).
    """
    with pytest.raises(ValueError, match="real z"):
        sp.polylog(2, z)


@pytest.mark.parametrize("dtype", ["float32", "float64"])
def test_narrow_scalar_does_not_raise(dtype):
    """REGRESSION: `polylog(2, float32(0.5))` died inside `lax.fori_loop`.

    The carry was seeded at the argument's dtype but the body multiplied by the
    float64 Bernoulli table, widening it -- which `fori_loop` rejects, with an
    error naming neither `polylog` nor the dtype. `z` is documented as scalar, and a
    float32 scalar is a scalar.
    """
    got = float(sp.polylog(2, jnp.asarray(0.5, dtype=dtype)))
    np.testing.assert_allclose(got, reference(2, 0.5), rtol=1e-6)


@pytest.mark.parametrize("n", [1, 2, 3, 5, 12])
def test_derivative_at_zero(n):
    """REGRESSION: `grad(Li_n)(0)` was `nan` for every order n >= 2.

    The JVP is `Li_{n-1}(z) / z`, which is 0/0 at the origin -- while the
    *value* `polylog(n, 0)` was correctly 0, so only the gradient broke. `n = 1` has
    its own `1/(1-z)` branch and was unaffected, which made the break look
    selective. The limit is 1 for every order, since `Li_{n-1}(z) = z + O(z^2)`.

    The second derivative is checked too, because the obvious guard --
    `where(z == 0, 1.0, ratio)` -- fixes the first order and silently breaks the
    second, a constant differentiating to 0 where `Li_n''(0) = 2**(1-n)`. The
    ratio is instead rewritten as the series it equals, evaluated by Horner so
    that it is analytic at 0 to every order.
    """
    assert float(jax.grad(partial(sp.polylog, n))(0.0)) == pytest.approx(1.0)
    assert float(jax.grad(partial(sp.polylog, n))(-0.0)) == pytest.approx(1.0)
    second = float(jax.grad(jax.grad(partial(sp.polylog, n)))(0.0))
    assert second == pytest.approx(2.0 ** (1 - n))


@pytest.mark.parametrize("n", [60, 61, 62])
def test_value_and_gradient_agree_about_the_order_cap(n):
    """Above the Bernoulli table the value is `nan`; the gradient must be too.

    The derivative needs only ``Li_{n-1}``, so at exactly one order past the cap
    it stayed finite and correct while the value gave up -- a caller guarding on
    `isnan` got opposite answers depending on which it checked.
    """
    z = jnp.asarray(3.0)
    value = sp.polylog(n, z)
    grad = jax.grad(partial(sp.polylog, n))(z)
    assert bool(jnp.isnan(value)) == bool(jnp.isnan(grad))


@pytest.mark.parametrize("order", [1, 2, 3])
@pytest.mark.parametrize("n", [61, 62])
def test_every_derivative_order_respects_the_cap(order, n):
    """Past the Bernoulli table nothing is defined, derivatives included.

    Marking the tangent `nan` with a `where` fixed the first derivative and
    left the second at `0.0`, because differentiating that `where` again
    differentiates a constant. A multiplicative `nan` mask survives every
    order without poisoning the branch that was not taken.
    """
    f = partial(sp.polylog, n)
    for _ in range(order):
        f = jax.grad(f)
    assert jnp.isnan(f(jnp.asarray(3.0)))


def test_second_derivative_below_the_cap_is_untouched():
    """The mask must not leak into orders that are perfectly well defined."""
    for order, expected in ((2, 0.5), (3, 0.25)):
        second = jax.grad(jax.grad(partial(sp.polylog, order)))
        got = float(second(jnp.asarray(0.0)))
        np.testing.assert_allclose(got, expected, rtol=1e-12)


@pytest.mark.parametrize("dtype", ["float16", "bfloat16", "float32", "float64"])
def test_order_one_differentiates_at_every_width(dtype):
    """The `n == 1` tangent must carry the primal's dtype, not the argument's.

    It builds `1/(1 - z)` directly while the primal comes back promoted, so the
    `custom_jvp` contract was violated and `grad` raised for any dtype narrower
    than the promoted one. Orders from 2 up route through `_li_core` and were
    promoted already, which is what hid it.
    """
    dt = jnp.dtype(dtype)
    got = jax.grad(partial(sp.polylog, 1))(jnp.asarray(0.25, dt))
    assert got.dtype == dt
    np.testing.assert_allclose(float(got), 4.0 / 3.0, rtol=8 * float(jnp.finfo(dt).eps))
