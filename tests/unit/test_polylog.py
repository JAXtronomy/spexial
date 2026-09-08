"""Unit tests for `spexial.Li`."""

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
        np.testing.assert_allclose(sp.Li(n, 1.0), float(sp.zeta(float(n))), rtol=1e-11)


@pytest.mark.parametrize("n", [1, 2, 3, 4])
@pytest.mark.parametrize("z", [-2.0, 2.0])
def test_branch_boundary_at_abs_z_equals_two(n, z):
    """REGRESSION: |z| == 2 fell through every branch and returned 0.

    The branches were ``|z| <= 0.5``, ``0.5 < |z| < 2`` and ``|z| > 2``, which
    leaves ``|z| == 2`` uncovered; `jnp.where` then produced the 0.0 default.
    """
    got = float(sp.Li(n, z))
    assert got != 0.0
    np.testing.assert_allclose(got, reference(n, z), rtol=1e-11, atol=1e-12)


@pytest.mark.parametrize("z", [0.25, 0.75, 3.0])
def test_high_order_does_not_overflow_int64(z):
    """`j ** n` over a traced integer `j` overflows int64 for n >= 12."""
    got = float(sp.Li(20, z))
    assert np.isfinite(got)
    np.testing.assert_allclose(got, reference(20, z), rtol=1e-11, atol=1e-12)


@pytest.mark.parametrize("n", [0, -1])
def test_non_positive_order_is_rejected(n):
    """Only n >= 1 is implemented; say so instead of returning nonsense."""
    with pytest.raises(ValueError, match="n >= 1"):
        sp.Li(n, 0.25)


def test_array_input_is_rejected():
    """`Li` is documented as scalar-only; the type checker enforces it."""
    with pytest.raises(Exception, match=r"(?i)typecheck"):
        sp.Li(2, jnp.asarray([0.1, 0.2]))


def test_vmap_is_the_supported_way_to_batch():
    """`jax.vmap` gives the elementwise behaviour `Li` itself does not."""
    z = jnp.asarray([0.1, 0.3, 0.9, 3.0])
    got = jax.vmap(partial(sp.Li, 2))(z)
    expected = [reference(2, float(v)) for v in z]
    np.testing.assert_allclose(got, expected, rtol=1e-11, atol=1e-12)


def test_jit():
    """`Li` is jittable (order is a static argument)."""
    np.testing.assert_allclose(sp.Li(2, 0.3), reference(2, 0.3), rtol=1e-11)


@pytest.mark.parametrize("z", [0.3, 0.75, 3.0])
def test_grad(z):
    """d/dz Li_n(z) == Li_{n-1}(z) / z."""
    got = jax.grad(partial(sp.Li, 3))(z)
    np.testing.assert_allclose(got, reference(2, z) / z, rtol=1e-9)


def test_bernoulli_polynomial():
    """B_n(x) helper: B_2(x) == x^2 - x + 1/6."""
    for x in (0.0, 1.0, 3.0):
        np.testing.assert_allclose(
            _bernoulli_poly(2, jnp.asarray(x)), x**2 - x + 1 / 6, rtol=1e-12, atol=1e-15
        )
