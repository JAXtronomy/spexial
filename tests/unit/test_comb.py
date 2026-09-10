"""Unit tests for `spexial.comb`."""

import math

import jax
import jax.numpy as jnp
import mpmath as mp
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


@pytest.mark.parametrize("exponent", range(18))
def test_n_choose_one_is_n(exponent):
    """``C(N, 1) == N`` is an identity, so it needs no reference implementation.

    It is the cheapest probe of the large-`N` branch: the log-gamma difference
    collapsed to ``1.0`` here once the two terms converged.
    """
    n = 10.0**exponent
    np.testing.assert_allclose(sp.comb(n, 1.0), n, rtol=1e-12)


@pytest.mark.parametrize("n", [999.0, 1000.0, 1001.0])
@pytest.mark.parametrize("k", [1.0, 2.0, 3.0, 30.0, 60.0])
def test_the_crossover_is_continuous(n, k):
    """The two formulas must agree where they are stitched together.

    Below ``N = 1000`` the log-gamma difference is the accurate one and above it
    the Beta form is; they are within a few times 1e-13 of each other at the
    join, which is what keeps the function continuous there.
    """
    np.testing.assert_allclose(sp.comb(n, k), scipy_comb(n, k), rtol=1e-11)


@pytest.mark.parametrize(
    "n", [8.988465674311582e307, 1e308, 1.5e308, float(np.finfo(np.float64).max)]
)
def test_n_choose_one_past_the_subnormal_cliff(n):
    """`C(N, 1) == N` must hold above ``N = 2**1023``.

    `jax.scipy.special.betaln` divides its smaller argument by its larger one,
    and XLA flushes that quotient to zero the moment it is subnormal -- which
    drops a term worth exactly the smaller argument, making `comb` a factor of
    ``e**-2`` low. 86% wrong, silently, at the very top of the range.

    The tolerance is 1e-12 rather than the ulp because the value goes through
    ``exp`` of a logarithm near 709, and that round trip alone costs about
    ``709 * eps``, i.e. 1.6e-13. Measured worst here is 3.1e-14.
    """
    np.testing.assert_allclose(sp.comb(n, 1.0), n, rtol=1e-12)
    np.testing.assert_allclose(sp.comb(n, 1.0), scipy_comb(n, 1.0), rtol=1e-12)


@pytest.mark.parametrize(
    ("k", "expected"), [(0.0, 1.0), (1.0, 1e308), (2.0, np.inf), (1e308, 1.0)]
)
def test_the_whole_subnormal_band(k, expected):
    """Every `k` that gives a finite answer at ``N = 1e308``, and one that does not.

    ``k >= 2`` overflows there because ``C(N, 2) ~ N**2 / 2``, so `inf` is the
    right answer rather than a failure.
    """
    got = sp.comb(1e308, k)
    if np.isinf(expected):
        assert np.isinf(got)
    else:
        np.testing.assert_allclose(got, expected, rtol=1e-12)
    assert np.isinf(scipy_comb(1e308, k)) or np.allclose(
        got, scipy_comb(1e308, k), rtol=1e-12
    )


@pytest.mark.parametrize("dtype", ["float32", "float16", "bfloat16"])
@pytest.mark.parametrize("n", [20.0, 50.0, 100.0, 500.0])
def test_low_precision_is_within_its_own_dtype(dtype, n):
    """Narrow dtypes must get what their dtype can represent.

    The log-gamma difference has lost every digit by ``N = 20`` in bfloat16 --
    ``comb(100, 2)`` was 345% high and ``comb(500, 2)`` returned ``1.0`` for a
    true 124750. Two things fix it: computing one width up, and moving the
    branch cross-over, which was measured in float64 and is 100x too high for
    anything narrower.
    """
    dt = jnp.dtype(dtype)
    expected = float(mp.binomial(int(n), 2))
    if expected > float(jnp.finfo(dt).max):
        pytest.skip(f"{expected:g} overflows {dtype}")
    got = sp.comb(jnp.asarray(n, dt), jnp.asarray(2.0, dt))
    assert got.dtype == dt
    # 16 eps, not one: the value is `exp` of a logarithm, and that round trip
    # alone costs about `log(comb) * eps` -- ~7 eps at these sizes. Before the
    # fix, bfloat16 was 345% out and float16 returned `inf`.
    np.testing.assert_allclose(float(got), expected, rtol=16 * float(jnp.finfo(dt).eps))


@pytest.mark.parametrize(
    ("n_dtype", "k_dtype"),
    [
        ("float32", "float64"),
        ("float64", "float32"),
        ("float16", "float32"),
        ("float16", "float16"),
        ("bfloat16", "bfloat16"),
    ],
)
def test_result_follows_promotion_of_both_arguments(n_dtype, k_dtype):
    """`comb` is the only two-argument entry point, and both arguments count.

    The result used to be narrowed to `N`'s dtype alone, so a
    `(float32, float64)` call returned float32 -- the *narrower* of the two,
    the opposite of what JAX's promotion entitles the caller to.
    """
    n_dt, k_dt = jnp.dtype(n_dtype), jnp.dtype(k_dtype)
    got = sp.comb(jnp.asarray(20.0, n_dt), jnp.asarray(3.0, k_dt))
    assert got.dtype == jnp.promote_types(n_dt, k_dt)
    np.testing.assert_allclose(
        float(got), 1140.0, rtol=32 * float(jnp.finfo(got.dtype).eps)
    )


def test_a_python_int_does_not_pick_the_wrong_crossover():
    """The branch cross-over must come from the promoted dtype, not from `N`.

    `as_float(1000, keep_weak=True)` is a *weak* float64, which defers to a
    float32 `k` for every subsequent operation -- so reading `eps` off `N`
    alone ran the float64-tuned cross-over in float32, and this call was 205x
    worse than the same one with both arguments float32.
    """
    mixed = sp.comb(1000, jnp.float32(6))
    both = sp.comb(jnp.float32(1000), jnp.float32(6))
    assert mixed.dtype == both.dtype == jnp.float32
    np.testing.assert_allclose(float(mixed), float(both), rtol=1e-6)
    np.testing.assert_allclose(float(mixed), float(math.comb(1000, 6)), rtol=1e-5)


@pytest.mark.parametrize("bad", [-1e-320, -5e-324])
def test_negative_subnormals_are_out_of_domain(bad):
    """XLA compares a subnormal as zero, so `k >= 0` was True for `k = -1e-40`.

    The guard then let it through as `C(N, 0) = 1`, where SciPy gives 0.
    """
    assert float(sp.comb(5.0, bad)) == 0.0
    assert float(sp.comb(bad, 0.0)) == 0.0
    # `-0.0` is not negative for this purpose, and stays in the domain
    assert float(sp.comb(5.0, -0.0)) == 1.0


@pytest.mark.parametrize("exponent", [2, 10, 16, 20, 100, 153, 154, 200, 300])
def test_derivative_of_n_choose_one_is_one(exponent):
    """``d/dN C(N, 1) == 1`` is an identity, so it needs no reference.

    It is the cheapest probe of the derivative at large `N`, where
    `jax.scipy.special.betaln`'s own gradient is exactly twice the truth from
    `a` of about 1e154 and `nan` above. Through
    ``comb = exp(-betaln(N, 2) - log(N + 1))`` that arrived as
    ``C * (4/N - 1/N) = 3`` -- a silent 200% error while the value stayed
    correct. The Beta branch's asymptotic form has no `betaln` in it and now
    covers the whole band where that defect lives.
    """
    n = jnp.asarray(10.0**exponent)
    one = jnp.asarray(1.0)
    np.testing.assert_allclose(float(jax.grad(sp.comb, 0)(n, one)), 1.0, rtol=1e-12)
    np.testing.assert_allclose(float(jax.jacfwd(sp.comb, 0)(n, one)), 1.0, rtol=1e-12)


@pytest.mark.parametrize(("k", "exponent"), [(2, 103), (3, 78), (5, 52)])
def test_derivative_is_finite_where_the_value_is(k, exponent):
    """Value and gradient must agree about the domain.

    The `nan` onset moved with `k` -- 1e154 at k = 1, 1e103 at 2, 1e77 at 3 --
    so a caller guarding on `isnan(value)` was safe where one guarding on
    `isnan(grad)` was not.
    """
    n, kk = jnp.asarray(10.0**exponent), jnp.asarray(float(k))
    assert jnp.isfinite(sp.comb(n, kk))
    assert jnp.isfinite(jax.grad(sp.comb, 0)(n, kk))
