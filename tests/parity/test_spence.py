"""Parity for `spexial.spence` against `scipy.special`, and its derivative.

The value is checked against scipy, which is the reference. The *derivative* is
checked against the closed form rather than against
`jax.scipy.special.spence` -- JAX's gradient is `nan` across roughly
``1 < x < 2`` (see `test_gradient_is_finite_where_jax_is_nan`), so it cannot
serve as a reference there.
"""

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
    np.testing.assert_allclose(spence(z), scipy_spence(z), rtol=1e-5, atol=1e-8)
    np.testing.assert_allclose(spence(x), scipy_spence(x), rtol=1e-5, atol=1e-8)


@given(st.floats(min_value=0.05, max_value=10))
@settings(deadline=1000)
def test_gradient_matches_the_closed_form(x):
    r"""The derivative is the integrand of the definition, :math:`\log z/(1-z)`.

    Checked against that closed form rather than against JAX: `spexial` matched
    it to zero relative error over a 400-point sweep, while JAX returns `nan`
    on part of the same range.
    """
    if abs(x - 1.0) < 1e-6:  # the removable singularity; both terms vanish
        return
    np.testing.assert_allclose(
        jax.grad(spence)(x), np.log(x) / (1 - x), rtol=1e-10, atol=1e-12
    )


def test_gradient_at_the_special_points():
    """`0` and `1` are special-cased in the derivative to avoid a divergence."""
    assert float(jax.grad(spence)(0.0)) == 0.0
    assert float(jax.grad(spence)(1.0)) == 0.0


@pytest.mark.parametrize("x", [1.2, 1.5, 1.8, 1.98])
def test_gradient_is_finite_where_jax_is_nan(x):
    """One of the reasons this module exists rather than re-exporting JAX.

    `jax.scipy.special.spence` differentiates to `nan` across roughly
    ``1 < x < 2`` -- 67 of 400 points sampled over ``[0.05, 6]``. `spexial`'s
    analytic rule is finite and exact there.
    """
    assert jnp.isnan(jax.grad(jss.spence)(x))
    np.testing.assert_allclose(jax.grad(spence)(x), np.log(x) / (1 - x), rtol=1e-12)
