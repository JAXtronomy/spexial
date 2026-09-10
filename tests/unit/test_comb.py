"""Unit tests for `spexial.comb`."""

import math
import os
import subprocess
import sys

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


@pytest.mark.parametrize(
    ("exponent", "k"), [(306, 1.0), (307, 1.0), (154, 2.0), (78, 3.0)]
)
def test_reverse_and_forward_gradients_agree_at_large_n(exponent, k):
    """`jnp.minimum`'s transpose multiplies the cotangent by a 0/1 selector.

    At large `N` that cotangent has already overflowed to `inf`, so `0 * inf`
    made `jax.grad` `nan` from about `N = 1e306` -- and from `1e154` at
    `k = 2` -- while the value and `jacfwd` were both correct. A `where`
    transposes as a select and never forms the product.
    """
    n, kk = jnp.asarray(10.0**exponent), jnp.asarray(k)
    reverse = float(jax.grad(sp.comb, 0)(n, kk))
    forward = float(jax.jacfwd(sp.comb, 0)(n, kk))
    assert jnp.isfinite(reverse)
    np.testing.assert_allclose(reverse, forward, rtol=1e-12)


@pytest.mark.parametrize("wrap", [lambda f: f, jax.jit], ids=["eager", "jit"])
@pytest.mark.parametrize(
    ("k", "expected"),
    [(0.0, 1.0), (-0.0, 1.0), (5e-324, 1.0), (1e-320, 1.0), (1.0, np.inf)],
)
def test_infinite_n_agrees_between_modes(wrap, k, expected):
    """REGRESSION: the ``k = 0`` limit at ``N = inf`` must not depend on `jit`.

    `dtype.exactly_zero` read the bits so that a subnormal ``k`` would not be
    mistaken for zero, and got ``inf`` -- eagerly. Under `jit` the operand
    reaching the bitcast has already been flushed, so the same call returned
    ``1.0``: one value disagreeing with itself between the two modes. The
    flushed comparison gives the whole subnormal band the ``k = 0`` answer,
    which is a documented floor rather than the limit, but it is one answer.
    """
    got = wrap(sp.comb)(jnp.asarray(jnp.inf), jnp.asarray(k))
    assert float(got) == expected


@pytest.mark.parametrize("n", [11.0, 100.0, 1000.0])
def test_derivative_at_k_zero_float32(n):
    """REGRESSION: the asymptotic branch was taken at ``k = 0`` for every ``N``.

    Its dropped correction is ``small (small - 1) / (2 big)``, which vanishes
    identically at ``k = 0`` (where ``small`` is 1) but whose *derivative* is
    ``1 / (2 (N + 1))``. Testing the value alone therefore let the branch fire
    across the whole float32 Beta range and left ``jax.grad`` low by exactly
    that -- 1.4e-2 relative at ``N = 11``, in the most ordinary call there is.
    """
    mp.mp.dps = 60
    got = float(jax.grad(lambda k: sp.comb(jnp.float32(n), k))(jnp.float32(0.0)))
    expect = float(
        mp.binomial(mp.mpf(n), 0) * (mp.digamma(mp.mpf(n) + 1) - mp.digamma(1))
    )
    assert got == pytest.approx(expect, rel=1e-6)


@pytest.mark.parametrize("n", [1001.0, 1e4, 1e6, 1e16, 1e100, 1e300])
def test_derivative_at_the_ends_of_the_k_range(n):
    """The same defect in float64, at both ``k = 0`` and ``k = N``."""
    mp.mp.dps = 300
    expect = float(
        mp.binomial(mp.mpf(n), 0) * (mp.digamma(mp.mpf(n) + 1) - mp.digamma(1))
    )
    at_zero = float(jax.grad(lambda k: sp.comb(jnp.float64(n), k))(jnp.float64(0.0)))
    at_n = float(jax.grad(lambda m: sp.comb(m, jnp.float64(n)))(jnp.float64(n)))
    assert at_zero == pytest.approx(expect, rel=1e-13)
    assert at_n == pytest.approx(expect, rel=1e-13)


def test_derivative_is_continuous_across_the_asymptotic_join():
    """No step in ``d comb / dk`` where the asymptotic branch takes over.

    The old predicate put the join at ``k = 2 big eps``, and the derivative
    jumped by 6.7e-5 relative across it -- a reachable interval in float32.
    """
    mp.mp.dps = 60
    for k in (1e-5, 1e-4, 2.0e-4, 2.4e-4, 1e-3):
        got = float(
            jax.grad(lambda kk: sp.comb(jnp.float32(1000.0), kk))(jnp.float32(k))
        )
        kk = mp.mpf(float(np.float32(k)))
        expect = float(
            mp.binomial(mp.mpf(1000), kk)
            * (mp.digamma(mp.mpf(1000) - kk + 1) - mp.digamma(kk + 1))
        )
        assert got == pytest.approx(expect, rel=1e-5)


@pytest.mark.parametrize(
    ("n", "still_negative"),
    [(1e150, True), (1e153, True), (1e154, False), (1e200, False)],
)
def test_second_derivative_sign_is_the_documented_floor(n, still_negative):
    """`comb`'s second derivative w.r.t. ``N`` flips sign past ``N = 1.3e154``.

    Both surviving terms are of size ``1/N**2``, which is *subnormal* there, and
    XLA flushes subnormals: `jax.grad(jax.grad(jnp.log))(1e154)` is ``-0.0``
    where the true ``-1e-308`` is a representable denormal. The dominant
    negative term vanishes and the positive one is left, so the magnitude is
    right and the sign inverted. Nothing recovers it short of a rule that never
    enters log space, which `comb` cannot have. Pinned so the boundary cannot
    move unnoticed.
    """
    hessian = jax.hessian(lambda v: sp.comb(v[0], v[1]))
    got = float(hessian(jnp.asarray([n, 0.5]))[0, 0])
    expect = -0.25 / math.gamma(1.5) * n**-1.5
    assert abs(got) == pytest.approx(abs(expect), rel=1e-10)
    assert (got < 0.0) is still_negative, "the sign floor moved"


@pytest.mark.parametrize("dtype", [jnp.float32, jnp.float64])
def test_comb_n_choose_zero_is_one(dtype):
    """``C(N, 0) = C(N, N) = 1`` exactly, to within the documented trade.

    Round 14 routed these two points through `betaln` to make the *derivative*
    right, which cost the value 28 ulps in float64 for ``1000 < N < 3e6``. That
    is the trade; this pins how large it is allowed to be.
    """
    for n in (5.0, 1001.0, 1e6, 1e100):
        if dtype is jnp.float32 and n > 1e30:
            continue
        assert float(sp.comb(jnp.asarray(n, dtype), jnp.asarray(0.0, dtype))) == (
            pytest.approx(1.0, rel=1e-6 if dtype is jnp.float32 else 1e-13)
        )
        assert float(sp.comb(jnp.asarray(n, dtype), jnp.asarray(n, dtype))) == (
            pytest.approx(1.0, rel=1e-6 if dtype is jnp.float32 else 1e-13)
        )


@pytest.mark.parametrize("wrap", [lambda f: f, jax.jit], ids=["eager", "jit"])
def test_infinite_n_with_subnormal_k_is_inf_in_float16(wrap):
    """float16 is the one width where the subnormal-`k` floor does not apply.

    `as_float` widens float16 to float32 before anything else, and a float16
    subnormal (3.05e-5) is an entirely normal float32, so it survives the flush
    and `comb(inf, k)` returns the mathematically correct `inf`. The documented
    `1.0` floor is real at every other width; this pins that float16 escapes it.
    """
    subnormal = jnp.asarray(3.0517578125e-05, dtype=jnp.float16)
    got = wrap(sp.comb)(jnp.asarray(jnp.inf, dtype=jnp.float16), subnormal)
    assert float(got) == np.inf


def test_comb_n_choose_zero_error_structure_in_float32():
    """`comb(N, 0)` falls at most 8 ulps short of 1 without x64, at every `N`.

    `jax.scipy.special.gammaln(float32(1.0))` is ``2**-21`` rather than 0, and
    ``comb(N, 0)`` reduces to ``exp(-gammaln(1))``, so the whole shortfall is
    upstream's. Pinned as a *bound* rather than as a list of exceptional `N`:
    the shortfall is 0, 4, 6 or 8 ulps depending where the two branch
    cross-overs and `gammaln`'s rounding fall, and an earlier attempt to name
    the individual cases got three of five wrong. Runs in a subprocess, because
    the suite pins `JAX_ENABLE_X64` and this is a float32-only defect.
    """
    script = (
        "import jax.numpy as jnp, numpy as np, spexial as sp\n"
        "sp1 = np.spacing(np.float32(1.0), dtype=np.float32)\n"
        "rng = np.random.default_rng(0)\n"
        "xs = np.concatenate([np.arange(0, 200, dtype=float),\n"
        "                     np.exp(rng.uniform(np.log(200), np.log(3e38), 2000))])\n"
        "worst = 0\n"
        "for x in xs:\n"
        "    for k in (0.0, float(x)):\n"
        "        v = float(sp.comb(jnp.asarray(x, jnp.float32),\n"
        "                          jnp.asarray(k, jnp.float32)))\n"
        "        worst = min(worst, round((v - 1.0) / sp1))\n"
        "print('WORST', worst)\n"
        "print('BIG', float(sp.comb(jnp.asarray(1e8, jnp.float32),\n"
        "                           jnp.asarray(0.0, jnp.float32))))\n"
    )
    env = {k: v for k, v in os.environ.items() if k != "JAX_ENABLE_X64"}
    out = subprocess.run(  # noqa: S603
        [sys.executable, "-c", script],
        capture_output=True,
        text=True,
        env=env,
        check=True,
    )
    worst = int(out.stdout.split("WORST")[1].split()[0])
    assert -8 <= worst <= 0, out.stdout
    # Above the asymptotic cross-over the shortfall is gone entirely.
    assert float(out.stdout.split("BIG")[1].strip()) == 1.0
