"""Unit tests for `spexial.sech`."""

import os
import subprocess
import sys

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
    """A floating argument keeps its own width, and a Python float stays weak."""
    assert sp.sech(jnp.asarray([0.5, 3.0], dtype=dtype)).dtype == jnp.dtype(dtype)
    assert sp.sech(0.5).weak_type


@pytest.mark.parametrize(
    "x",
    [
        jnp.asarray([0, 1, 3], dtype="int8"),
        jnp.asarray([0, 1, 3], dtype="uint8"),
        jnp.asarray([0, 1, 3], dtype="int32"),
        jnp.asarray([0, 1, 3], dtype="int64"),
        jnp.asarray([True, False, True]),
        2,
    ],
)
def test_integer_and_bool_input_gives_a_strong_float(x):
    """An integer or bool becomes a real float, not a truncated or weak one.

    `-jnp.abs` raises on a bool, so the conversion is what makes it work, and a
    weakly typed result would silently take the dtype of whatever it meets next.
    """
    got = sp.sech(x)
    assert got.dtype == jnp.result_type(float)
    assert not got.weak_type
    want = [float(mp.sech(mp.mpf(float(v)))) for v in np.ravel(np.asarray(x))]
    np.testing.assert_allclose(np.ravel(got), want, rtol=1e-14)


def test_integer_input_without_x64_is_a_strong_float32():
    """With x64 off, integer and bool input give a strong float32, with no warning.

    Runs in a subprocess, because the suite pins `JAX_ENABLE_X64`. A hard-coded
    float64 would be identical under the suite and wrong here.
    """
    script = (
        "import warnings\n"
        "warnings.simplefilter('error')\n"
        "import jax.numpy as jnp, spexial as sp\n"
        "for x in (2, jnp.asarray([0, 1, 3]), jnp.asarray([True, False])):\n"
        "    out = sp.sech(x)\n"
        "    print(out.dtype, out.weak_type)\n"
    )
    env = {k: v for k, v in os.environ.items() if k != "JAX_ENABLE_X64"}
    out = subprocess.run(  # noqa: S603
        [sys.executable, "-c", script],
        capture_output=True,
        text=True,
        env=env,
        check=True,
    )
    assert out.stdout.split("\n")[:3] == ["float32 False"] * 3, out.stdout


def test_shapes():
    """Empty and N-d arrays keep their shape and are evaluated elementwise."""
    assert sp.sech(jnp.zeros((0, 3))).shape == (0, 3)
    x = jnp.linspace(-3.0, 3.0, 24).reshape(2, 3, 4)
    np.testing.assert_allclose(sp.sech(x), 1 / np.cosh(np.asarray(x)), rtol=1e-14)


@pytest.mark.parametrize(("dtype", "rtol"), [("float16", 5e-3), ("bfloat16", 2e-2)])
def test_narrow_dtypes_value_and_gradient(dtype, rtol):
    """Float16 and bfloat16 are not widened; both value and gradient are right.

    Measured worst case, over every argument where the result is at least
    `2 tiny`, is about 1.4e-3 for float16 and 1e-2 for bfloat16, gradient
    included.
    """
    for x in (0.25, 1.0, 3.0, -2.0, 10.0):
        xj = jnp.asarray(x, dtype=dtype)
        sech_true, grad_true = _true(xj)
        assert _rel(sp.sech(xj), sech_true) < rtol
        assert _rel(jax.grad(sp.sech)(xj), grad_true) < rtol


@pytest.mark.parametrize(
    ("x", "value", "grad"),
    [
        (-0.0, 1.0, 0.0),
        (jnp.inf, 0.0, 0.0),
        (-jnp.inf, 0.0, 0.0),
        (jnp.nan, jnp.nan, jnp.nan),
    ],
)
def test_special_arguments(x, value, grad):
    """`sech(-0.0)` is 1, the limits at infinity are 0, and `nan` stays.

    `tanh` saturates, so the gradient at infinity is 0, not `nan`.
    """
    np.testing.assert_array_equal(sp.sech(x), value)
    np.testing.assert_array_equal(jax.grad(sp.sech)(x), grad)


@pytest.mark.parametrize(
    ("x", "dtype"),
    [
        (jnp.asarray([1.0 + 1.0j, 2.0], dtype="complex64"), "complex64"),
        (jnp.asarray([1.0 + 1.0j, 2.0], dtype="complex128"), "complex128"),
        (1.0 + 1.0j, "complex128"),
    ],
)
def test_complex_input_is_rejected(x, dtype):
    """`|x|` is not analytic: complex input would return sech of the modulus."""
    with pytest.raises(ValueError, match=f"real x, got dtype {dtype}"):
        sp.sech(x)


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


@pytest.mark.parametrize(
    ("dtype", "x"),
    [
        ("float64", 709.5),
        ("float64", 1e4),
        ("float64", 1e300),
        ("float32", 88.5),
        ("float32", 1e3),
        ("float32", 1e30),
    ],
)
def test_beyond_underflow_value_and_gradient_vanish(dtype, x):
    """Past where the value is subnormal, value and gradient are 0, not just finite.

    `1 / cosh` has a `nan` gradient from ~710 (float64) and ~89 (float32).
    """
    tiny = float(np.finfo(dtype).tiny)
    for sign in (1.0, -1.0):
        xj = jnp.asarray(sign * x, dtype=dtype)
        assert abs(float(sp.sech(xj))) <= tiny
        assert abs(float(jax.grad(sp.sech)(xj))) <= tiny


@pytest.mark.parametrize(
    "x",
    [1e-12, 1e-8, 1e-6, 1e-3, 0.1, 1.0, 5.0, 30.0, 300.0, 400.0, 700.0, 708.0, 709.0],
)
def test_gradient_keeps_its_relative_accuracy_float64(x):
    """The hand-written rule has no cancellation near zero.

    Differentiating the value form directly gives errors of order 1e-4 at 1e-12
    and 1e-10 at 1e-6; this holds the rule to 4e-15 everywhere. Measured worst
    case is 1.05e-15.
    """
    sech_true, grad_true = _true(x)
    assert _rel(sp.sech(x), sech_true) < 4e-15
    assert _rel(jax.grad(sp.sech)(x), grad_true) < 4e-15
    assert _rel(jax.grad(sp.sech)(-x), -grad_true) < 4e-15


@pytest.mark.parametrize(
    "x", [1e-8, 1e-6, 1e-3, 0.1, 1.0, 5.0, 30.0, 80.0, 85.0, 87.0, 88.0]
)
def test_gradient_keeps_its_relative_accuracy_float32(x):
    """Measured worst case is 4.4e-7 over float32's whole range."""
    xj = jnp.asarray(x, dtype=jnp.float32)
    sech_true, grad_true = _true(xj)
    assert _rel(sp.sech(xj), sech_true) < 1e-6
    assert _rel(jax.grad(sp.sech)(xj), grad_true) < 1e-6
    assert _rel(jax.grad(sp.sech)(-xj), -grad_true) < 1e-6


# Per dtype: the range in |x| to test over (points whose true result is below
# `2 tiny` are skipped), and the tolerance, a few times the measured worst case.
_GRIDS = {
    "float64": (1e-300, 709.0, 4e-15),
    "float32": (1e-30, 88.0, 1e-6),
    "float16": (1e-4, 17.0, 5e-3),
    "bfloat16": (1e-4, 88.0, 2e-2),
}


@pytest.mark.parametrize("dtype", list(_GRIDS))
def test_value_and_gradient_over_the_whole_range(dtype):
    """A dense grid, both signs, so an error confined to a sub-range cannot hide."""
    lo, hi, rtol = _GRIDS[dtype]
    tiny = float(jnp.finfo(dtype).tiny)
    pos = np.concatenate([np.geomspace(lo, 1.0, 40), np.linspace(1.0, hi, 300)])
    xj = jnp.asarray(np.concatenate([pos, -pos]), dtype=dtype)
    value = sp.sech(xj)
    grad = jax.vmap(jax.grad(sp.sech))(xj)
    for xi, v, g in zip(xj, value, grad, strict=True):
        sech_true, grad_true = _true(xi)
        if abs(sech_true) >= 2 * tiny:
            assert _rel(v, sech_true) < rtol, (dtype, float(xi))
        if abs(grad_true) >= 2 * tiny:
            assert _rel(g, grad_true) < rtol, (dtype, float(xi))


_DERIVATIVE_CASES = [
    (dtype, tol, order)
    for dtype, tol, orders in (
        ("float64", 8e-15, (2, 3, 4, 5)),
        ("float32", 1e-6, (2, 3, 4)),
    )
    for order in orders
]


@pytest.mark.parametrize(("dtype", "tol", "order"), _DERIVATIVE_CASES)
@pytest.mark.parametrize(
    "x", [-4.0, -1.0, -0.3, -1e-3, 0.0, 1e-12, 1e-6, 1e-3, 0.3, 1.0, 4.0]
)
def test_higher_derivatives(x, dtype, tol, order):
    """The rule is written in terms of `sech`, so every order is right.

    `x = 0` is the point that matters: a `where(x == 0, ...)` guard in the rule
    would give the right gradient and a wrong second derivative there, and
    AGENTS.md asks for the second and third at any guarded point. The odd orders
    vanish at zero, so the comparison is absolute there. Measured worst case is
    2.7e-15 in float64 and 2.1e-7 in float32.
    """
    xj = jnp.asarray(x, dtype=dtype)
    f = sp.sech
    for _ in range(order):
        f = jax.grad(f)
    true = float(mp.diff(mp.sech, mp.mpf(float(xj)), order))
    np.testing.assert_allclose(f(xj), true, rtol=tol, atol=tol)


def test_forward_mode_matches_reverse():
    """`jvp` with a non-unit tangent agrees with `grad`.

    Reverse mode transposes away any term of the rule that does not depend on
    the tangent, so only forward mode can see one.
    """
    x = jnp.linspace(-30.0, 30.0, 61)
    t = jnp.linspace(0.5, 2.0, 61)
    _, tangent = jax.jvp(sp.sech, (x,), (t,))
    reverse = jax.vmap(jax.grad(sp.sech))(x) * t
    np.testing.assert_allclose(tangent, reverse, rtol=1e-14)


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


@pytest.mark.parametrize(
    ("dtype", "x", "rtol"),
    [
        ("float64", 708.7, 4e-15),
        ("float64", 709.0, 4e-15),
        ("float32", 87.7, 1e-6),
        ("float32", 88.0, 1e-6),
    ],
)
def test_value_just_above_the_smallest_normal_is_not_flushed(dtype, x, rtol):
    """The true value is between `tiny` and `2 tiny` here, and is returned.

    A form built on `exp(-|x|)` has a subnormal intermediate there, which XLA
    flushes to give 0; `2 / (e + 1/e)` with `e = exp(|x|)` has none.
    """
    xj = jnp.asarray(x, dtype=dtype)
    tiny = float(np.finfo(dtype).tiny)
    sech_true, grad_true = _true(xj)
    assert tiny <= sech_true < 2 * tiny
    assert _rel(sp.sech(xj), sech_true) < rtol
    assert _rel(jax.grad(sp.sech)(xj), grad_true) < rtol


@pytest.mark.parametrize(("dtype", "x"), [("float64", 1e-310), ("float32", 1e-40)])
def test_gradient_at_a_subnormal_argument_is_flushed(dtype, x):
    """The gradient at a subnormal `x` may be `-0.0` where the truth is `-x`.

    XLA flushes the subnormal operand. The absolute error is below `tiny`, and
    this is recorded as a ceiling in `docs/reference/accuracy-and-domains.md`.
    """
    xj = jnp.asarray(x, dtype=dtype)
    assert abs(float(jax.grad(sp.sech)(xj)) + x) <= np.finfo(dtype).tiny
