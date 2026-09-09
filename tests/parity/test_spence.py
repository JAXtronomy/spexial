"""Parity for `spexial.spence` against `scipy.special`, and its derivative.

The value is checked against scipy, which is the reference. The *derivative* is
checked against the closed form rather than against
`jax.scipy.special.spence` -- JAX's gradient is `nan` across roughly
``1 < x < 2`` (see `test_gradient_is_finite_where_jax_is_nan`), so it cannot
serve as a reference there.
"""

import os
import subprocess
import sys

import jax
import jax.numpy as jnp
import jax.scipy.special as jss
import numpy as np
import pytest
from hypothesis import example, given, settings, strategies as st
from scipy.special import spence as scipy_spence

from spexial import spence


@given(
    st.floats(min_value=0, max_value=10),
    st.floats(min_value=0, max_value=2 * np.pi),
)
@example(1.0, 0.0)
@example(5, 0)
@settings(deadline=1000)
def test_spence_matches_scipy(x, phi):
    r"""`spexial.spence` matches `scipy.special.spence`, real and complex.

    Complex arguments are built in polar form deliberately: there is a branch
    point on the negative real axis, so sampling :math:`-1 \pm \epsilon`
    directly would straddle it and compare two different branches.
    """
    z = x * np.exp(1j * phi)
    # rtol was 1e-5 here, eight orders looser than the measured worst case
    # of 3.4e-14 -- slack that large would let the function be 1e8x wrong
    # and still pass.
    np.testing.assert_allclose(spence(z), scipy_spence(z), rtol=1e-12, atol=1e-13)
    np.testing.assert_allclose(spence(x), scipy_spence(x), rtol=1e-12, atol=1e-13)


@given(st.floats(min_value=0.05, max_value=10))
@settings(deadline=1000)
def test_gradient_matches_the_closed_form(x):
    r"""The derivative is the integrand of the definition, :math:`\log z/(1-z)`.

    Checked against that closed form rather than against JAX: `spexial` matched
    it to zero relative error over a 400-point sweep, while JAX returns `nan`
    on part of the same range.
    """
    if abs(x - 1.0) < 1e-6:  # covered by `test_gradient_at_the_removable_singularity`
        return
    np.testing.assert_allclose(
        jax.grad(spence)(x), np.log(x) / (1 - x), rtol=1e-10, atol=1e-12
    )


@pytest.mark.parametrize("x", [0.3, 0.75, 1.0, 1.5, 2.5, 6.0])
def test_gradient_matches_a_central_difference(x):
    """REGRESSION: the derivative was compared only against its own formula.

    `test_gradient_matches_the_closed_form` asserts `grad(spence)(x)` equals
    `log(x)/(1-x)` -- which is literally the expression the JVP evaluates, so it
    cannot fail on a wrong formula, only on an unwired rule. This differentiates
    `spence`'s own *values* instead, which is what caught the `z = 1` bug below.
    """
    h = 1e-5
    numeric = float((spence(x + h) - spence(x - h)) / (2 * h))
    np.testing.assert_allclose(jax.grad(spence)(x), numeric, rtol=1e-8, atol=1e-9)


def test_gradient_at_the_removable_singularity():
    """REGRESSION: the derivative at ``z = 1`` was 0; the true value is -1.

    `log(z)/(1-z)` is `0/0` at `z = 1`, and the guard replaced it with `0.0`.
    But the singularity is *removable*, not a zero: `spence(1 + h) = -h +
    O(h^2)`, so the limit is `-1`. The old value was a plausible-looking wrong
    answer at exactly the point a user is most likely to evaluate, and it
    survived `jit`, `vmap` and array input -- one element equal to 1.0 in an
    otherwise-correct array silently got a zero gradient.
    """
    assert float(jax.grad(spence)(1.0)) == pytest.approx(-1.0)
    # ...and it is continuous, which is what makes 0.0 indefensible.
    assert float(jax.grad(spence)(1.0 - 1e-9)) == pytest.approx(-1.0)
    assert float(jax.grad(spence)(1.0 + 1e-9)) == pytest.approx(-1.0)


def test_gradient_at_zero_is_minus_infinity():
    """REGRESSION: `z = 0` returned 0; `log(0)/(1-0)` is `-inf`, the true limit."""
    assert float(jax.grad(spence)(0.0)) == -np.inf


@pytest.mark.parametrize("x", [1.2, 1.5, 1.8, 1.98])
def test_gradient_is_finite_where_jax_is_nan(x):
    """One of the reasons this module exists rather than re-exporting JAX.

    `jax.scipy.special.spence` differentiates to `nan` across roughly
    ``1 < x < 2`` -- 67 of 400 points sampled over ``[0.05, 6]``. `spexial`'s
    analytic rule is finite and exact there.
    """
    assert jnp.isnan(jax.grad(jss.spence)(x))
    np.testing.assert_allclose(jax.grad(spence)(x), np.log(x) / (1 - x), rtol=1e-12)


def test_series_denominator_does_not_overflow_int32():
    """REGRESSION: `(n(n+1)(n+2))**2` was computed in integer dtype.

    JAX is int32 unless x64 is enabled, and that degree-6 polynomial overflows
    int32 from n = 35 -- 235 of the 499 terms went to garbage, many negative,
    for a 3.8e-5 error at z = 2. That is ~300x worse than float32 rounding
    alone, so it is corruption rather than graceful degradation, and it was
    invisible to the rest of the suite because `pyproject.toml` pins
    `JAX_ENABLE_X64=True` for every test. Runs in a subprocess to get a JAX
    without x64.
    """
    script = "import spexial as sp;print(float(sp.spence(2.0)), float(sp.spence(1.3)))"
    env = {k: v for k, v in os.environ.items() if k != "JAX_ENABLE_X64"}
    out = subprocess.run(  # noqa: S603
        [sys.executable, "-c", script],
        capture_output=True,
        text=True,
        env=env,
        check=True,
    )
    at_two, at_one_three = (float(v) for v in out.stdout.split())
    # float32 eps is 1.2e-7; allow a decade of accumulation over the 499 terms.
    np.testing.assert_allclose(at_two, -0.8224670334241132, rtol=1e-6)
    np.testing.assert_allclose(at_one_three, -0.2800743338009109, rtol=1e-5)


@pytest.mark.parametrize(
    ("dtype", "value"),
    [
        ("float32", 2.0),
        ("float64", 2.0),
        ("complex64", 2.0 + 1.0j),
        ("complex128", 2.0 + 1.0j),
    ],
)
@pytest.mark.parametrize("z", [0.2, 0.6, 1.0, 1.5, 3.0])
def test_dtype_is_preserved(dtype, value, z):
    """A narrow argument must not come back widened, on any branch.

    `spence` stitches three series together and both the series-about-zero and
    the reflected branch carry a `pi**2 / 6` constant. `np.pi` is a Python
    `float`, so that constant is *weakly* typed and follows the argument --
    but a `np.float64(...)` in its place would promote float32 to float64
    silently, which is exactly the kind of widening that breaks a `lax.scan`
    carry downstream. Every branch is exercised: `z` spans |z| <= 1/2,
    |1 - z| <= 1 and the reflected region.
    """
    arg = complex(z, value.imag) if isinstance(value, complex) else z
    got = spence(jnp.asarray(arg, dtype=dtype))
    assert got.dtype == jnp.dtype(dtype)
