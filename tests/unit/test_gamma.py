"""Unit tests for `spexial.gamma`."""

import jax
import jax.numpy as jnp
import jax.scipy.special as jss
import mpmath as mp
import numpy as np
import pytest
from scipy.special import digamma, gamma as scipy_gamma, polygamma

import spexial as sp


def test_array_input_broadcasts():
    """REGRESSION: `lax.cond` needs a scalar predicate, so arrays used to fail.

    The old body was ``lax.cond(x < 0.5, reflect, lanczos, x)``, which raises
    for any non-scalar ``x``.
    """
    x = jnp.asarray([0.5, 1.0, 2.0, 5.0, -0.5, -1.5])
    got = sp.gamma(x)
    assert got.shape == x.shape
    np.testing.assert_allclose(got, scipy_gamma(np.asarray(x)), rtol=1e-11)


def test_array_input_2d():
    """Shape is preserved for more than one dimension."""
    x = jnp.asarray([[1.0, 2.0], [3.0, 4.0]])
    assert sp.gamma(x).shape == (2, 2)
    np.testing.assert_allclose(sp.gamma(x), [[1.0, 1.0], [2.0, 6.0]], rtol=1e-11)


def test_pole_at_zero_is_inf():
    """`gamma(0)` is `inf`, as in both JAX and scipy."""
    assert jnp.isinf(sp.gamma(0.0))


@pytest.mark.parametrize("x", [-1.0, -2.0, -10.0])
def test_negative_integer_poles_are_nan(x):
    """The negative integers give `nan`, matching JAX and scipy >= 1.18.

    Gamma has a pole at each of them and the two-sided limit does not exist, so
    `nan` is the defensible value; C99 `tgamma` agrees. This changed when
    `gamma` began delegating to `jax.scipy.special.gamma` -- before that
    `spexial` returned `inf` here, which matched scipy 1.14 but not 1.18.
    """
    assert jnp.isnan(sp.gamma(x))
    assert jnp.isnan(jss.gamma(jnp.asarray(x)))


def test_large_argument_does_not_overflow_early():
    """The log-space Lanczos evaluation reaches scipy's own overflow point.

    A direct ``t ** (z - 0.5)`` overflows above x ~ 142, well below the true
    overflow at ~171.6.
    """
    x = jnp.asarray([150.0, 170.0, 171.0])
    np.testing.assert_allclose(sp.gamma(x), scipy_gamma(np.asarray(x)), rtol=1e-11)
    assert jnp.isinf(sp.gamma(172.0))


def _jax_gamma_supports_complex() -> bool:
    """Probe rather than version-compare: capability, not a number to maintain."""
    try:
        jss.gamma(jnp.asarray(1 + 2j))
    except Exception:  # noqa: BLE001
        return False
    return True


@pytest.mark.skipif(
    not _jax_gamma_supports_complex(),
    reason="jax.scipy.special.gamma gained complex support in 0.10.2; below that "
    "it branches on floor(x) and raises. spexial delegates, so it inherits the "
    "limitation rather than papering over it.",
)
def test_complex_input_is_supported():
    """Complex input works, and matches scipy.

    `spexial.gamma` was real-only while it carried its own Lanczos series, whose
    reflection branch needed an elementwise ``x < 0.5`` test. Delegating removed
    that constraint, so the restriction went with it.
    """
    z = jnp.asarray([1 + 2j, -1.5 + 0.5j])
    got = np.asarray(sp.gamma(z))
    expected = scipy_gamma(np.asarray([1 + 2j, -1.5 + 0.5j]))
    np.testing.assert_allclose(got, expected, rtol=1e-12)


def test_dtype_is_float_for_integer_input():
    """Integer input is promoted to float."""
    assert jnp.issubdtype(sp.gamma(jnp.asarray([1, 2, 3])).dtype, jnp.floating)


def test_jit():
    """`gamma` is jittable."""
    np.testing.assert_allclose(jax.jit(sp.gamma)(5.0), 24.0, rtol=1e-12)


def test_vmap():
    """`gamma` is vmappable."""
    got = jax.vmap(sp.gamma)(jnp.asarray([1.0, 2.0, 5.0]))
    np.testing.assert_allclose(got, [1.0, 1.0, 24.0], rtol=1e-11)


@pytest.mark.parametrize("x", [1.5, 5.0, -0.5, -2.5])
def test_grad(x):
    """gamma'(x) == gamma(x) * digamma(x)."""
    got = jax.grad(sp.gamma)(x)
    np.testing.assert_allclose(got, scipy_gamma(x) * digamma(x), rtol=1e-9)


def test_positive_infinity_is_infinity():
    """`gamma(+inf)` is `inf`; the Lanczos series alone gives `nan`."""
    assert jnp.isinf(sp.gamma(jnp.inf))
    assert sp.gamma(jnp.inf) > 0


def test_negative_infinity_is_nan():
    """Deliberate divergence: scipy returns -inf, but the limit does not exist.

    Gamma has a pole at every negative integer, so there is no limit at -inf to
    return. Documented in `docs/reference/accuracy-and-domains.md`.
    """
    assert jnp.isnan(sp.gamma(-jnp.inf))


@pytest.mark.parametrize("x", [0.5, 2.5, 5.0])
def test_second_derivative_is_correct_on_the_positive_axis(x):
    """Gamma''(x) = Gamma(x) * (digamma(x)**2 + polygamma(1, x))."""
    got = float(jax.grad(jax.grad(sp.gamma))(x))
    expected = scipy_gamma(x) * (digamma(x) ** 2 + polygamma(1, x))
    np.testing.assert_allclose(got, expected, rtol=1e-11)


@pytest.mark.parametrize("x", [-10.5, -20.5])
def test_second_derivative_is_broken_on_the_negative_axis_like_jax(x):
    """The second derivative is unusable for x < ~-7.5, and that is upstream.

    `gamma`'s JVP is `g * digamma(x)`, so differentiating twice goes through
    `jax.scipy.special.digamma`'s own derivative -- and JAX's trigamma diverges
    from `scipy.special.polygamma(1, .)` on the negative axis, wrongly by 1e11
    at -10.5 and with the wrong *sign* at -20.5. `spexial` is no worse than
    JAX here, which is what this pins: if JAX fixes it, this test fails and the
    docs get corrected rather than quietly staying pessimistic.
    """
    ours = float(jax.grad(jax.grad(sp.gamma))(x))
    theirs = float(jax.grad(jax.grad(jss.gamma))(x))
    truth = scipy_gamma(x) * (digamma(x) ** 2 + polygamma(1, x))
    np.testing.assert_allclose(ours, theirs, rtol=1e-12)
    assert abs(ours / truth - 1) > 0.1, "JAX's trigamma looks fixed; update the docs"


@pytest.mark.skipif(
    not _jax_gamma_supports_complex(),
    reason="jax.scipy.special.gamma gained complex support in 0.10.2; below that "
    "it branches on floor(x) and raises, so there is nothing to differentiate",
)
@pytest.mark.parametrize("z", [1.5 + 2j, -0.5 + 0.25j])
def test_complex_input_differentiates(z):
    """The custom rule must not be a regression on the function it wraps.

    `jax.scipy.special.digamma` rejects complex input, so the analytic rule
    raised `TypeError` where `jax.scipy.special.gamma` -- whose value `gamma`
    merely forwards -- differentiates complex input perfectly well. A
    `custom_jvp` cannot decline to apply, so the complex case is handed back to
    the wrapped function.
    """
    _, tangent = jax.jvp(sp.gamma, (jnp.asarray(z),), (jnp.asarray(1.0 + 0j),))
    with mp.workdps(30):
        w = mp.mpc(z.real, z.imag)
        expected = complex(mp.gamma(w) * mp.digamma(w))
    assert abs(complex(tangent) - expected) <= 1e-12 * abs(expected)
