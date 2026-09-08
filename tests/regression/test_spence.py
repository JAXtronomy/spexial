"""Test `spexial.spence` matches `scipy.special`."""

import jax
import numpy as np
from hypothesis import example, given, settings, strategies as st
from scipy.special import spence as scipy_spence

from spexial import spence


@given(st.floats(min_value=0, max_value=10), st.floats(min_value=0, max_value=2 * np.pi))
@example(1.0, 0.0)
@example(5, 0)
@settings(deadline=1000)  # Set the maximum time for this test to 1000ms
def test_eval_spence(x, phi):
    r"""
    Test `spexial.spence` matches `scipy.special`.

    There is a branch point on the negative real axis and so tests at, e.g.,
    :math:`-1 + \epsilon` and :math:`-1 - \epsilon` give very different results
    so I took this somewhat roundabout method to get complex numbers.
    """
    z = x * np.exp(1j * phi)
    result = spence(z)
    expected = scipy_spence(z)
    np.testing.assert_allclose(result, expected, rtol=1e-5, atol=1e-8)
    result = spence(x)
    expected = scipy_spence(x)
    np.testing.assert_allclose(result, expected, rtol=1e-5, atol=1e-8)


@given(st.floats(min_value=0.1, max_value=10))
@example(0.0)
@settings(deadline=1000)  # Set the maximum time for this test to 1000ms
def test_grad_spence_real(x):
    r"""
    Test the gradient of `spexial.spence` matches `jax.scipy.special`.

    For some reason, the gradient of the jax implementation returns nan
    between one and two and has numerical issues for small x.
    """
    if abs(x - 1.5) < 0.5:
        x += 1
    print(x)
    result = jax.grad(spence)(x)
    expected = jax.grad(jax.scipy.special.spence)(x)
    np.testing.assert_allclose(result, expected, rtol=1e-5, atol=1e-8)
