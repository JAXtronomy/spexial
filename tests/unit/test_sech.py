"""Unit tests for `spexial.sech`."""

import jax
import jax.numpy as jnp
import mpmath as mp
import numpy as np
import pytest

import spexial as sp


@pytest.fixture(autouse=True)
def _fifty_digits():
    """Reference values at 50 digits, without changing mpmath for other tests."""
    with mp.workdps(50):
        yield


def _true(x):
    """`sech` and its derivative at `x`, at 50 digits."""
    xm = mp.mpf(float(x))
    return mp.sech(xm), -mp.sech(xm) * mp.tanh(xm)


def _rel(got, true):
    return float(abs((mp.mpf(float(got)) - true) / true))


@pytest.mark.parametrize("dtype", ["float16", "bfloat16", "float32", "float64"])
def test_dtype_is_preserved(dtype):
    """A floating argument keeps its own width."""
    assert sp.sech(jnp.asarray([0.5, 3.0], dtype=dtype)).dtype == jnp.dtype(dtype)


@pytest.mark.parametrize("x", [jnp.asarray([0, 1, 3]), jnp.asarray([True, False]), 2])
def test_integer_and_bool_input_gives_a_strong_float(x):
    """An integer or bool becomes a real float, not a truncated or weak one.

    `-jnp.abs` raises on a bool, so the conversion is what makes it work, and a
    weakly typed result would silently take the dtype of whatever it meets next.
    """
    got = sp.sech(x)
    assert jnp.issubdtype(got.dtype, jnp.floating)
    assert not got.weak_type
    want = [float(mp.sech(mp.mpf(float(v)))) for v in np.ravel(np.asarray(x))]
    np.testing.assert_allclose(np.ravel(got), want, rtol=1e-14)


@pytest.mark.parametrize(("dtype", "rtol"), [("float16", 5e-3), ("bfloat16", 2e-2)])
def test_narrow_dtypes_value_and_gradient(dtype, rtol):
    """Float16 and bfloat16 are not widened; both value and gradient are right.

    Measured worst case is 7e-4 and 9e-4 for float16, 4e-3 and 8.5e-3 for
    bfloat16.
    """
    for x in (0.25, 1.0, 3.0, -2.0):
        xj = jnp.asarray(x, dtype=dtype)
        sech_true, grad_true = _true(xj)
        assert _rel(sp.sech(xj), sech_true) < rtol
        assert _rel(jax.grad(sp.sech)(xj), grad_true) < rtol


@pytest.mark.parametrize(
    ("x", "value", "grad"),
    [(jnp.inf, 0.0, 0.0), (-jnp.inf, 0.0, 0.0), (jnp.nan, jnp.nan, jnp.nan)],
)
def test_non_finite_arguments(x, value, grad):
    """The limits at infinity are 0, in value and gradient, and `nan` stays.

    `tanh` saturates, so the gradient at infinity is 0, not `nan`.
    """
    np.testing.assert_array_equal(sp.sech(x), value)
    np.testing.assert_array_equal(jax.grad(sp.sech)(x), grad)


def test_complex_input_is_rejected():
    """`|x|` is not analytic: complex input would return sech of the modulus."""
    with pytest.raises(ValueError, match="real x"):
        sp.sech(1.0 + 1.0j)


# ---------------------------------------------------------------------------
# The reason this function exists: the gradient.


def test_naive_reciprocal_cosh_gradient_is_nan():
    """The defect `sech` exists to avoid, asserted against the naive form.

    A test of the *reason* for the function, which cannot rot: if JAX ever
    makes `1 / cosh` differentiate cleanly here, this fails and the docstring
    and registry row need revisiting.
    """
    naive = jax.grad(lambda x: 1 / jnp.cosh(x))
    assert jnp.isnan(naive(800.0))
    assert jnp.isnan(naive(jnp.float32(100.0)))
    # ...and it is wrong well before it is `nan`: the denominator has
    # overflowed, so the gradient is 0 where the true value is 1e-154.
    assert float(naive(400.0)) == 0.0


@pytest.mark.parametrize(
    ("dtype", "x"),
    [
        ("float64", 710.0),
        ("float64", 1e4),
        ("float64", 1e300),
        ("float32", 100.0),
        ("float32", 1e3),
        ("float32", 1e30),
    ],
)
def test_gradient_is_finite_beyond_where_cosh_overflows(dtype, x):
    """REGRESSION: `1 / cosh` is `nan` from ~710 (float64) and ~90 (float32)."""
    xj = jnp.asarray(x, dtype=dtype)
    assert jnp.isfinite(jax.grad(sp.sech)(xj))
    assert jnp.isfinite(jax.grad(sp.sech)(-xj))


@pytest.mark.parametrize(
    "x", [1e-12, 1e-8, 1e-6, 1e-3, 0.1, 1.0, 5.0, 30.0, 300.0, 400.0, 700.0]
)
def test_gradient_keeps_its_relative_accuracy_float64(x):
    """The hand-written rule has no cancellation near zero.

    Differentiating the stable value form directly gives errors of order 1e-4 at
    1e-12 and 1e-10 at 1e-6; this holds the rule to 1e-14 everywhere. Measured
    worst case is 9.4e-16.
    """
    sech_true, grad_true = _true(x)
    assert _rel(sp.sech(x), sech_true) < 1e-14
    assert _rel(jax.grad(sp.sech)(x), grad_true) < 1e-14
    assert _rel(jax.grad(sp.sech)(-x), -grad_true) < 1e-14


@pytest.mark.parametrize("x", [1e-8, 1e-6, 1e-3, 0.1, 1.0, 5.0, 30.0, 80.0])
def test_gradient_keeps_its_relative_accuracy_float32(x):
    """Measured worst case is 3.9e-7 over float32's whole range."""
    xj = jnp.asarray(x, dtype=jnp.float32)
    sech_true, grad_true = _true(xj)
    assert _rel(sp.sech(xj), sech_true) < 3e-6
    assert _rel(jax.grad(sp.sech)(xj), grad_true) < 3e-6
    assert _rel(jax.grad(sp.sech)(-xj), -grad_true) < 3e-6


@pytest.mark.parametrize("order", [2, 3, 4])
@pytest.mark.parametrize("x", [0.0, 1e-6, 1e-3, 0.3, 1.0, 4.0])
def test_higher_derivatives(x, order):
    """The rule is written in terms of `sech`, so every order is right.

    `x = 0` is the point that matters: a `where(x == 0, ...)` guard in the rule
    would give the right gradient and a wrong second derivative there, and
    AGENTS.md asks for the second and third at any guarded point. The odd orders
    vanish at zero, so the comparison is absolute there.
    """
    f = sp.sech
    for _ in range(order):
        f = jax.grad(f)
    true = float(mp.diff(mp.sech, mp.mpf(x), order))
    np.testing.assert_allclose(f(x), true, rtol=1e-12, atol=1e-12)


def test_jit_vmap_grad_agree():
    """The usual transforms compose and agree with the eager value."""
    x = jnp.linspace(-30.0, 30.0, 61)
    eager = sp.sech(x)
    np.testing.assert_allclose(jax.jit(sp.sech)(x), eager, rtol=1e-14)
    np.testing.assert_allclose(jax.vmap(sp.sech)(x), eager, rtol=1e-14)
    g = jax.vmap(jax.grad(sp.sech))(x)
    np.testing.assert_allclose(g, -eager * np.tanh(np.asarray(x)), rtol=1e-14)


# ---------------------------------------------------------------------------
# The documented ceiling.


@pytest.mark.parametrize(("dtype", "x"), [("float64", 708.7), ("float32", 87.7)])
def test_flushed_band_just_above_the_smallest_normal(dtype, x):
    """In a band of width ln 2 the value may be flushed to 0.

    `exp(-|x|)` is subnormal there and XLA on CPU flushes it, while the true
    value is still between `tiny` and `2 tiny`. The result is therefore either
    right or 0, and the absolute error is at most `2 tiny`; this is a ceiling,
    recorded in `docs/reference/accuracy-and-domains.md`, not a tolerance
    widened to pass.
    """
    xj = jnp.asarray(x, dtype=dtype)
    tiny = float(np.finfo(dtype).tiny)
    true = float(_true(xj)[0])
    assert tiny <= true < 2 * tiny
    assert abs(float(sp.sech(xj)) - true) <= 2 * tiny


@pytest.mark.parametrize(("dtype", "x"), [("float64", 1e-310), ("float32", 1e-40)])
def test_gradient_at_a_subnormal_argument_is_flushed(dtype, x):
    """The gradient at a subnormal `x` may be `-0.0` where the truth is `-x`.

    XLA flushes the subnormal operand. The absolute error is below `tiny`, and
    this is recorded as a ceiling in `docs/reference/accuracy-and-domains.md`.
    """
    xj = jnp.asarray(x, dtype=dtype)
    assert abs(float(jax.grad(sp.sech)(xj)) + x) <= np.finfo(dtype).tiny
