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


def test_small_values_are_zero():
    """Far below the turning point the value is exactly zero."""
    assert float(sp.spherical_jn(100, 10.0)) == 0.0


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
