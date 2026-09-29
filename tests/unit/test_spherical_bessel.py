"""Unit tests for `spherical_jn` and `spherical_jn_all`."""

from math import factorial

import jax
import jax.numpy as jnp
import numpy as np
import pytest

import spexial as sp


def _closed_forms(x):
    """j_0 ... j_3 in closed form (DLMF 10.49.3)."""
    s, c = np.sin(x), np.cos(x)
    return [
        s / x,
        s / x**2 - c / x,
        (3 / x**2 - 1) * s / x - 3 * c / x**2,
        (15 / x**3 - 6 / x) * s / x - (15 / x**2 - 1) * c / x,
    ]


def test_matches_the_closed_forms():
    """Orders 0 to 3 against their closed forms."""
    x = np.linspace(5.0, 60.0, 400)
    np.testing.assert_allclose(
        sp.spherical_jn_all(3, jnp.asarray(x)),
        np.stack(_closed_forms(x)),
        rtol=1e-12,
        atol=1e-15,
    )


def test_j1_small_argument():
    """`j_1` on [1e-3, 1] against its power series (DLMF 10.53.1), relatively."""
    x = np.geomspace(1e-3, 1.0, 200)
    series = x * sum(
        (-x * x / 2) ** k / (factorial(k) * np.prod(np.arange(2 * k + 3, 0, -2.0)))
        for k in range(20)
    )
    np.testing.assert_allclose(sp.spherical_jn(1, jnp.asarray(x)), series, rtol=1e-13)


def _assert_agrees(a, b, z, n):
    """Two routes to the same values agree, to the accuracy the docstring claims.

    Not bit for bit. Separate `jax.jit` programs may contract the same
    arithmetic differently: on macOS the single-order call and the row of the
    table differ by 1 ulp. Below the turning point only the absolute size is
    meaningful, so that part is compared against the peak of the order.
    """
    a, b, z = np.asarray(a), np.asarray(b), np.asarray(z)
    assert np.abs(a - b).max() <= 1e-6 * np.abs(b).max()
    above = np.abs(z) >= n
    np.testing.assert_allclose(a[above], b[above], rtol=1e-12, atol=1e-15)


@pytest.mark.parametrize("n", [0, 1, 2, 57])
def test_single_order_is_the_row_of_the_table(n):
    """`spherical_jn` is the matching row of `spherical_jn_all`."""
    x = jnp.linspace(0.0, 200.0, 2001)
    _assert_agrees(sp.spherical_jn(n, x), sp.spherical_jn_all(max(n, 3), x)[n], x, n)


def test_special_values():
    """Exact at the origin, 0 at infinity, and odd or even in the order."""
    np.testing.assert_array_equal(sp.spherical_jn_all(4, 0.0), [1.0, 0, 0, 0, 0])
    table = np.asarray(sp.spherical_jn_all(4, jnp.asarray([jnp.inf, -jnp.inf])))
    np.testing.assert_array_equal(table, 0.0)
    assert np.all(np.isnan(sp.spherical_jn_all(4, jnp.nan)))
    z = jnp.linspace(0.0, 40.0, 201)
    for n in range(4):
        np.testing.assert_array_equal(
            sp.spherical_jn(n, -z), (-1.0) ** n * sp.spherical_jn(n, z)
        )


def test_upward_zeroes_small_values():
    """Upward recurrence alone sets values far below the turning point to 0."""
    assert (
        float(sp.spherical_jn(100, 10.0, recurrence=sp.SphericalJnRecurrence.UP)) == 0.0
    )


def test_default_keeps_small_values():
    """The default is right there instead: scipy gives 5.832040182006039e-90."""
    got = float(sp.spherical_jn(100, 10.0))
    assert got == pytest.approx(5.832040182006039e-90, rel=1e-12)


@pytest.mark.parametrize("dtype", ["float32", "float64"])
def test_tiny_argument_is_finite(dtype):
    """Down to z ~ tiny, where one step once grew the float64 carry by 2**535."""
    tiny = float(jnp.finfo(dtype).tiny)
    z = jnp.asarray([tiny, 1e-30, 1e-21, 1e-15], dtype)
    for n in (2, 5, 60):
        for d in (False, True):
            assert np.all(np.isfinite(sp.spherical_jn(n, z, derivative=d)))
    # The leading term of the series, x**2 / 15, where it is representable.
    rtol = 1e-12 if dtype == "float64" else 1e-5
    np.testing.assert_allclose(sp.spherical_jn(2, z[1:]), z[1:] ** 2 / 15, rtol=rtol)


def test_downward_is_nan_above_the_turning_point():
    """`SphericalJnRecurrence.DOWN` cannot be right at ``|z| >= n``, and says so."""
    z = jnp.asarray([0.5, 9.9, 10.0, -10.0, 30.0])
    got = np.asarray(sp.spherical_jn(10, z, recurrence=sp.SphericalJnRecurrence.DOWN))
    np.testing.assert_array_equal(np.isnan(got), [False, False, True, True, True])
    both = np.asarray(sp.spherical_jn(10, z[:2]))
    np.testing.assert_allclose(got[:2], both, rtol=1e-14)
    table = sp.spherical_jn_all(10, 12.0, recurrence=sp.SphericalJnRecurrence.DOWN)
    assert np.all(np.isnan(table[2:]))


@pytest.mark.parametrize("n", [7, 60])
def test_one_sided_batches_match_a_mixed_one(n):
    """A batch wholly above or below the turning point skips one recurrence.

    It must still give what the same points give inside a mixed batch, which
    runs both -- including through `jax.grad` and `jax.vmap`, which see the
    `lax.cond` differently.
    """
    above = jnp.linspace(n + 1.0, 4.0 * n, 97)  # n + 1 keeps the derivative's
    below = jnp.linspace(0.0, 0.9 * n, 97)  # order n + 1 on the same side
    mixed = jnp.concatenate([below, above])
    for f in (
        lambda z: sp.spherical_jn(n, z),
        lambda z: sp.spherical_jn(n, z, derivative=True),
        lambda z: sp.spherical_jn_all(n, z),
        jax.vmap(jax.grad(lambda t: sp.spherical_jn(n, t))),
    ):
        whole = np.asarray(f(mixed))
        split = np.concatenate([np.asarray(f(below)), np.asarray(f(above))], -1)
        np.testing.assert_allclose(split, whole, rtol=1e-13, atol=1e-16)
    per_point = jax.vmap(lambda t: sp.spherical_jn(n, t))(above)
    np.testing.assert_allclose(per_point, sp.spherical_jn(n, above), rtol=1e-13)


@pytest.mark.parametrize("n", [0, 1])
def test_downward_orders_0_and_1_are_exact_everywhere(n):
    """Closed forms under `DOWN` too, derivative included, past ``|z| = n + 1``.

    The derivative rule needs order ``n + 1``; taking that from the downward
    recurrence made ``j_1'`` `nan` for every ``|z| >= 2``.
    """
    z = np.linspace(0.5, 40.0, 81)
    down = sp.SphericalJnRecurrence.DOWN
    j0, j1, j2, _ = _closed_forms(z)
    want = -j1 if n == 0 else (j0 - 2.0 * j2) / 3.0
    got = sp.spherical_jn(n, jnp.asarray(z), derivative=True, recurrence=down)
    np.testing.assert_allclose(got, want, rtol=1e-12, atol=1e-15)
    grad = jax.vmap(jax.grad(lambda t: sp.spherical_jn(n, t, recurrence=down)))
    np.testing.assert_allclose(grad(jnp.asarray(z)), want, rtol=1e-12, atol=1e-15)


@pytest.mark.parametrize("n", [2, 5, 30])
def test_downward_value_and_derivative_are_nan_together(n):
    """Under `DOWN`, j_n and j_n' are `nan` from exactly ``|z| = n``, and agree.

    The derivative's neighbour, order ``n + 1``, is finite up to ``n + 1``,
    which once left a finite `jax.grad` of a `nan` value on ``[n, n + 1)``.
    """
    down = sp.SphericalJnRecurrence.DOWN
    z = jnp.asarray([n - 0.5, n - 1e-9, n, n + 0.5, n + 1.0 - 1e-9, -(n + 0.5)])
    beyond = np.abs(np.asarray(z)) >= n
    value = np.asarray(sp.spherical_jn(n, z, recurrence=down))
    slope = np.asarray(sp.spherical_jn(n, z, derivative=True, recurrence=down))
    grad = jax.vmap(jax.grad(lambda t: sp.spherical_jn(n, t, recurrence=down)))
    for got in (value, slope, np.asarray(grad(z))):
        np.testing.assert_array_equal(np.isnan(got), beyond)
    table = sp.spherical_jn_all(n, z, derivative=True, recurrence=down)
    np.testing.assert_array_equal(np.isnan(table), np.broadcast_to(beyond, table.shape))


@pytest.mark.parametrize("n", [5, 60])
def test_both_splits_exactly_at_the_order(n):
    """`BOTH` is `DOWN` on ``[n - 1, n)`` and `UP` from ``n``, bit for bit.

    Pins the switch: upward recurrence is still accurate just below the
    turning point, so no accuracy test notices it moving there.
    """
    R = sp.SphericalJnRecurrence
    below = jnp.linspace(n - 1.0, n - 1e-6, 17)
    above = jnp.linspace(float(n), n + 1.0, 17)
    np.testing.assert_array_equal(
        sp.spherical_jn(n, below), sp.spherical_jn(n, below, recurrence=R.DOWN)
    )
    np.testing.assert_array_equal(
        sp.spherical_jn(n, above), sp.spherical_jn(n, above, recurrence=R.UP)
    )


def test_tiny_argument_series_every_order():
    """Below the Miller floor, each order is ``z**l / (2l + 1)!!`` to rounding."""
    z = np.asarray([1e-24, 1e-30, 1e-40])
    table = np.asarray(sp.spherical_jn_all(12, jnp.asarray(z)))
    for order in range(2, 13):
        double_factorial = float(np.prod(np.arange(2 * order + 1, 0, -2.0)))
        want = z**order / double_factorial
        kept = want > 1e-300  # past this the true value is subnormal
        np.testing.assert_allclose(table[order][kept], want[kept], rtol=1e-13)


def test_recurrence_accepts_its_string_values():
    """``"up"`` is `SphericalJnRecurrence.UP`, and an unknown name raises."""
    z = jnp.linspace(0.1, 30.0, 7)
    np.testing.assert_array_equal(
        sp.spherical_jn(5, z, recurrence="up"),
        sp.spherical_jn(5, z, recurrence=sp.SphericalJnRecurrence.UP),
    )
    # `ValueError` from `SphericalJnRecurrence(...)`, or `TypeError` first where the
    # runtime type checker is on, as it is under the test suite.
    with pytest.raises((ValueError, TypeError)):
        sp.spherical_jn(5, z, recurrence="sideways")


def test_tiny_argument():
    """The limit 1/3 survives, where scipy's j_1' returns 1 once j_1 underflows."""
    assert float(sp.spherical_jn(1, 1e-300, derivative=True)) == pytest.approx(1 / 3)


@pytest.mark.parametrize("n", [0, 1, 2, 3])
def test_derivatives_at_the_origin(n):
    """First three derivatives at z = 0, from j_n = z^n / (2n+1)!! (1 - ...)."""

    def double_factorial(k):
        return 1 if k <= 0 else k * double_factorial(k - 2)

    def taylor(k):
        if k == n:
            return factorial(n) / double_factorial(2 * n + 1)
        if k == n + 2:
            return -factorial(n + 2) / (2 * (2 * n + 3) * double_factorial(2 * n + 1))
        return 0.0

    fs = [lambda z: sp.spherical_jn(n, z)]
    for _ in range(3):
        fs.append(jax.grad(fs[-1]))
    got = [float(f(0.0)) for f in fs]
    np.testing.assert_allclose(got, [taylor(k) for k in range(4)], atol=1e-16)


@pytest.mark.parametrize("n", [0, 4, 30])
def test_derivative_flag_is_the_gradient(n):
    """``derivative=True`` agrees with `jax.grad`."""
    z = jnp.linspace(0.0, 80.0, 401)
    _assert_agrees(
        jax.vmap(jax.grad(lambda t: sp.spherical_jn(n, t)))(z),
        sp.spherical_jn(n, z, derivative=True),
        z,
        n,
    )


def test_jit_vmap_and_reverse_mode():
    """Composes with `jit` and `vmap`; forward and reverse mode agree."""
    z = jnp.linspace(0.1, 30.0, 7)
    direct = sp.spherical_jn(5, z)
    _assert_agrees(jax.jit(lambda t: sp.spherical_jn(5, t))(z), direct, z, 5)
    _assert_agrees(jax.vmap(lambda t: sp.spherical_jn(5, t))(z), direct, z, 5)
    fwd = jax.jacfwd(lambda t: sp.spherical_jn_all(6, t))(z)
    rev = jax.jacrev(lambda t: sp.spherical_jn_all(6, t))(z)
    np.testing.assert_allclose(fwd, rev, rtol=1e-14, atol=1e-16)


def test_shapes():
    """Elementwise in ``z``; the table adds a leading axis."""
    z = jnp.ones((3, 4))
    assert sp.spherical_jn(3, 2.0).shape == ()
    assert sp.spherical_jn(3, z).shape == (3, 4)
    assert sp.spherical_jn_all(3, 2.0).shape == (4,)
    assert sp.spherical_jn_all(3, z, derivative=True).shape == (4, 3, 4)


@pytest.mark.parametrize("fn", [sp.spherical_jn, sp.spherical_jn_all])
def test_rejects_bad_input(fn):
    """Negative or non-integer orders and complex ``z`` raise."""
    with pytest.raises(ValueError, match="n must be >= 0"):
        fn(-1, 1.0)
    with pytest.raises(TypeError):
        fn(2.0, 1.0)
    with pytest.raises(ValueError, match="only real z"):
        fn(2, jnp.asarray(1.0 + 1.0j))
