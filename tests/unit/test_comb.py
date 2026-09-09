"""Unit tests for `spexial.comb`."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from scipy.special import comb as scipy_comb

import spexial as sp


@pytest.mark.parametrize(
    ("n", "k"),
    [
        (5, 6),  # k > N
        (5, -1),  # k < 0
        (0, 1),
        (2, 100),
        (-5, 2),  # N < 0: the old code gave `nan` here
        # Non-integer arguments out of range: the old code gave a wrong,
        # plausible-looking *non-zero* value, because `gammaln` of a negative
        # non-integer is finite.
        (5.0, 6.5),
        (5.5, 7.0),
        (4.5, -0.5),
        (-2.5, 1.0),
    ],
)
def test_out_of_range_is_zero(n, k):
    """REGRESSION: out-of-range arguments must give 0, as `scipy.special.comb` does.

    The old implementation was a bare
    ``exp(gammaln(N+1) - gammaln(k+1) - gammaln(N-k+1))``. For integer ``k > N``
    that happened to land on 0, but ``N < 0`` gave ``inf - inf == nan``, and any
    *non-integer* out-of-range pair gave a finite, wrong, non-zero answer.
    """
    result = sp.comb(n, k)
    assert not jnp.isnan(result)
    assert float(result) == 0.0
    assert float(result) == scipy_comb(n, k)


def test_out_of_range_elementwise():
    """The zero-fill is elementwise, not all-or-nothing."""
    got = sp.comb(jnp.asarray([5, 5, 5, 5]), jnp.asarray([-1, 2, 5, 6]))
    np.testing.assert_allclose(got, [0.0, 10.0, 1.0, 0.0], rtol=1e-12)


def test_broadcasting():
    """`N` and `k` broadcast against each other."""
    got = sp.comb(jnp.arange(5)[:, None], jnp.arange(5)[None, :])
    expected = scipy_comb(np.arange(5)[:, None], np.arange(5)[None, :])
    assert got.shape == (5, 5)
    np.testing.assert_allclose(got, expected, rtol=1e-12)


def test_dtype_is_float_for_integer_input():
    """Integer input still gives a float result -- this is the inexact variant."""
    assert jnp.issubdtype(
        sp.comb(jnp.asarray([5]), jnp.asarray([2])).dtype, jnp.floating
    )


def test_jit():
    """`comb` is jittable."""
    np.testing.assert_allclose(jax.jit(sp.comb)(5.0, 2.0), 10.0, rtol=1e-12)


def test_vmap():
    """`comb` is vmappable."""
    got = jax.vmap(sp.comb, in_axes=(None, 0))(10.0, jnp.arange(4.0))
    np.testing.assert_allclose(got, [1.0, 10.0, 45.0, 120.0], rtol=1e-12)


def test_grad():
    """d/dN comb(N, 2) at N=5 is comb(5,2) * (psi(6) - psi(4)) == 4.5."""
    got = jax.grad(lambda n: sp.comb(n, 2.0))(5.0)
    np.testing.assert_allclose(got, 4.5, rtol=1e-10)


def test_infinite_n_is_infinity():
    """`comb(inf, k)` is `inf`, as in scipy; `gammaln` alone gives `inf - inf`."""
    assert jnp.isinf(sp.comb(jnp.inf, 2.0))
    assert float(sp.comb(jnp.inf, 2.0)) == scipy_comb(np.inf, 2)


def test_infinite_n_with_zero_k_is_one():
    """REGRESSION: `comb(inf, 0)` returned `inf`; C(N, 0) = 1 for every N.

    The `isinf(N) -> inf` override was unconditional in `k`, throwing away the
    masked `gammaln` path that already computes `exp(0) == 1` correctly.
    """
    assert float(sp.comb(jnp.inf, 0.0)) == 1.0
    assert scipy_comb(np.inf, 0) == 1.0
    # k >= 1 is still infinite, as before.
    assert jnp.isinf(sp.comb(jnp.inf, 1.0))
    assert jnp.isinf(sp.comb(jnp.inf, 2.0))
