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
import mpmath as mp
import numpy as np
import pytest
from hypothesis import example, given, strategies as st
from scipy.special import spence as scipy_spence

from spexial import spence


@given(
    st.floats(min_value=0, max_value=10),
    st.floats(min_value=0, max_value=2 * np.pi),
)
@example(1.0, 0.0)
@example(5, 0)
def test_spence_matches_scipy(x, phi):
    r"""`spexial.spence` matches `scipy.special.spence`, real and complex.

    Complex arguments are built in polar form deliberately: there is a branch
    point on the negative real axis, so sampling :math:`-1 \pm \epsilon`
    directly would straddle it and compare two different branches.
    """
    z = x * np.exp(1j * phi)
    # SciPy is not a valid reference near 3 +- sqrt(3): its *complex* spence is
    # the series this module was translated from, and shares the removable 0/0
    # that `_series_about_one` now guards -- `scipy.special.spence(3 - sqrt(3) +
    # 0j)` returns 0.01125 against a true -0.25186. Its *real* path is Cephes
    # and is correct, so only the complex comparison is skipped. `mpmath` covers
    # the excluded neighbourhood in `test_spence_at_the_removable_singularity`
    # and in the mpmath parity suite.
    if min(abs(z - (3 - np.sqrt(3))), abs(z - (3 + np.sqrt(3)))) > 0.05:
        np.testing.assert_allclose(spence(z), scipy_spence(z), rtol=1e-12, atol=1e-13)
    # rtol was 1e-5 here, seven orders looser than the measured worst case of
    # 2.7e-14 -- slack that large would let the function be 1e7x wrong and still
    # pass. (Before the root fix the true worst was 1.9e-12, so this assertion
    # was latently failing, not merely slack.)
    np.testing.assert_allclose(spence(x), scipy_spence(x), rtol=1e-12, atol=1e-13)


@given(st.floats(min_value=0.05, max_value=10))
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


@pytest.mark.parametrize("z", [3 - np.sqrt(3), 3 + np.sqrt(3)])
@pytest.mark.parametrize("delta", [0.0, 1e-8, -1e-8, 1e-4, -1e-4, 1e-2])
def test_spence_at_the_removable_singularity(z, delta):
    """REGRESSION: `spence(3 - sqrt(3))` was **59% wrong**.

    `_series_about_one`'s accelerated form divides by `1 + 4t + t**2` with
    `t = 1 - z`, which vanishes at `t = -2 + sqrt(3)`. The numerator vanishes
    too, so the quotient is 0/0 and the relative error grows as ~1e-16/|denom|:
    5.9e-1 at the root, and still 1.0e-8 a whole 1e-8 away. The second point,
    `3 + sqrt(3)`, reaches the same root through the reflected branch, since
    `z / (z - 1)` maps it onto `3 - sqrt(3)`.

    Checked against mpmath, not SciPy: SciPy's *complex* spence is the code this
    was translated from and returns 0.01125 here.
    """
    point = z + delta
    with mp.workdps(40):
        expected = float(mp.re(mp.polylog(2, 1 - mp.mpf(point))))
    np.testing.assert_allclose(spence(point), expected, rtol=1e-13)


@pytest.mark.parametrize("z", [0.3, 0.75, 1.0, 1.5, 2.5, 6.0])
def test_second_derivative_matches_mpmath(z):
    """REGRESSION: `spence''(1)` was 0; the true value is 1/2.

    `_spence_gradient` guards the removable singularity at z = 1 by returning
    the constant -1, which is right for the first derivative and differentiates
    to **0** for the second. Only that single point was wrong -- 1 +- 1e-7 was
    already correct to seven digits -- which is what made it a plausible number
    rather than an obvious one. Precisely the defect the first derivative had at
    this same point, one order up, so it is now a rule rather than arithmetic.
    """
    with mp.workdps(40):
        expected = float(mp.re(mp.diff(lambda t: mp.polylog(2, 1 - t), z, 2)))
    np.testing.assert_allclose(jax.grad(jax.grad(spence))(z), expected, rtol=1e-9)


def test_second_derivative_at_the_pole_is_positive_infinity():
    """Both signed zeros agree, and on the true limit.

    `spence''(z) = 1/(z(1-z)) + log(z)/(1-z)**2` tends to `+inf` as z -> 0,
    since 1/z outruns log z. The closed form is `inf - inf` at `+0.0` and
    committed to `-inf` at `-0.0`, so the two zeros disagreed with each other
    as well as with the limit.
    """
    assert float(jax.grad(jax.grad(spence))(0.0)) == np.inf
    assert float(jax.grad(jax.grad(spence))(-0.0)) == np.inf


@pytest.mark.parametrize("order", [1, 2, 3, 4, 5])
def test_derivatives_at_the_removable_point_to_every_order(order):
    """REGRESSION: each order in turn was 0 at z = 1, one round after the last.

    The first derivative was 0 there (true -1), then the second was 0 (true
    1/2), then the third was 0 (true -2/3) -- three separate findings, one
    defect: a `jnp.where` substituting a *constant* at the removable point, and
    a constant differentiates to zero. Patching one more order would only have
    moved the boundary again.

    `log(z)/(1-z)` is analytic at z = 1, so it is now evaluated there as the
    series it equals. Differentiating a polynomial is right at every order, so
    this asserts the whole chain rather than the next rung of it.
    """
    f = spence
    for _ in range(order):
        f = jax.grad(f)
    with mp.workdps(40):
        expected = float(mp.re(mp.diff(lambda t: mp.polylog(2, 1 - t), 1, order)))
    np.testing.assert_allclose(f(1.0), expected, rtol=1e-9, atol=1e-12)
