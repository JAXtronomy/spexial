"""Unit tests for `spexial.incomplete_beta` outside the binomial series' range.

Large ``b`` and large ``a`` take a continued fraction (GH-64) or `_large_a`;
these tests cover those regimes, the switches between them, and the
regressions fixed alongside (GH-66, GH-67, GH-68). Split from `test_beta.py`
so the two files run on separate workers under ``--dist=loadfile``: together
they were one serialized ~20-minute file.
"""

import jax
import jax.numpy as jnp
import mpmath as mp
import numpy as np
import pytest

import spexial as sp
from spexial._src.beta import _CF_FROM, _branch, _branch_index

# `a > 0` always. `b` is any real: the negative and zero values are the reason
# this exists, and come up as ordinary slopes in double power-law profiles.
AS = [0.25, 0.5, 1.0, 1.31, 2.0, 4.5]
BS = [-2.5, -1.0, -0.1, 0.0, 0.3, 1.31, 4.0]
# Straddles the z = 1/2 branch switch deliberately, and both endpoints.
ZS = [1e-6, 1e-3, 0.1, 0.3, 0.5, 0.5 + 1e-7, 0.7, 0.9, 0.99, 1 - 1e-6]


def _mp_incomplete_beta(a, b, z):
    """Unregularized ``B(a, b, z)`` by mpmath, robust for large ``b > 0``.

    ``mp.betainc`` sums a hypergeometric series that converges too slowly for
    large ``b`` above the switch point -- mpmath 1.3.0, the supported floor,
    raises `NoConvergence` at ``a = 0.1, b = 1e5``. There the complement
    ``B(a, b) - B(b, a, 1 - z)`` converges fast, as it does for the code under
    test.
    """
    a, b, z = mp.mpf(a), mp.mpf(b), mp.mpf(z)
    if b > 0 and z >= (a + 1) / (a + b + 2):
        return mp.beta(a, b) - mp.betainc(b, a, 0, 1 - z)
    return mp.betainc(a, b, 0, z)


# ============================================================================
# Large b: the continued fraction (GH-64)

LARGE_BS = [np.nextafter(10.0, 99), 20.0, 50.0, 100.0, 300.0, 1e3, 1e4]


@pytest.mark.parametrize("a", [*AS, 8.0, 50.0])
@pytest.mark.parametrize("b", LARGE_BS)
def test_large_b_matches_mpmath(a, b):
    """REGRESSION: the series lost every digit above ``b ~ 50`` (GH-64).

    Its coefficients ``(1-b)_k / k!`` alternate and grow like ``b^k / k!``: at
    ``b = 300, z = 1/2`` it returned ``-1.1e44`` for ``1.1e-5``. Above
    `_CF_FROM` the continued fraction takes over.

    The reference is mpmath, not ``beta * betainc``: SciPy's regularized form
    is itself only good to ~1e-11 at ``b = 1e4``.

    ``a`` reaches 50, the top of the documented range for the continued
    fraction and a regime where the series fails even at small ``b``.

    Tolerances are the measured worst with headroom: 2e-13 up to ``b = 100``,
    then 8e-12 out to ``b = 1e5``, where 32 steps of the fraction stop short of
    full convergence (`_CF_STEPS`).
    """
    z = np.array([1e-8, *ZS[1:], 1.0])
    got = np.asarray(sp.incomplete_beta(a, b, jnp.asarray(z)))
    with mp.workdps(40):
        expect = np.array([float(_mp_incomplete_beta(a, b, zi)) for zi in z])
    np.testing.assert_allclose(got, expect, rtol=1e-12 if b <= 100 else 1e-11)


@pytest.mark.parametrize("a", [0.5, 2.0, 8.0])
def test_continuous_across_the_large_b_switch(a):
    """Series below `_CF_FROM`, continued fraction above: they must meet."""
    z = jnp.asarray([0.05, 0.5, 0.95])
    below = sp.incomplete_beta(a, np.nextafter(_CF_FROM, 0), z)
    above = sp.incomplete_beta(a, np.nextafter(_CF_FROM, 99), z)
    np.testing.assert_allclose(np.asarray(below), np.asarray(above), rtol=1e-13)


@pytest.mark.parametrize(("a", "b", "z"), [(2.0, 50.0, 0.01), (0.5, 300.0, 0.5)])
def test_large_b_parameter_derivatives(a, b, z):
    """``a``/``b`` tangents through the continued fraction, in both modes.

    Against mpmath's derivative, not a finite difference: the ``b``-derivative
    here is ~1e-4 of the value, so a difference quotient's own rounding is
    already a 1e-6 relative error, where autodiff is good to 1e-15.
    """
    zz = jnp.asarray(z)
    refs = (
        lambda t: _mp_incomplete_beta(t, b, z),
        lambda t: _mp_incomplete_beta(a, t, z),
    )
    for argnum, ref in enumerate(refs):
        with mp.workdps(40):
            expect = float(mp.diff(ref, (a, b)[argnum]))
        for mode in (jax.jacfwd, jax.jacrev):
            got = mode(sp.incomplete_beta, argnum)(a, b, zz)
            np.testing.assert_allclose(float(got), expect, rtol=1e-12)


def test_vmap_over_b_across_the_switch_has_finite_gradients():
    """Under `vmap` the `cond` becomes a `select` and both branches run.

    Each is handed a ``b`` it is safe for, so the unselected one -- the series
    at ``b = 1e4``, or the reflection at ``b <= 0`` -- cannot put a non-finite
    value where autodiff multiplies it by a zero cotangent.
    """
    bs = jnp.asarray([-2.0, 0.0, 5.0, 10.0, 10.5, 300.0, 1e4])
    f = lambda b: sp.incomplete_beta(2.0, b, jnp.asarray(0.3))
    batched = jax.vmap(f)(bs)
    single = jnp.stack([f(b) for b in bs])
    # Batched, `b` is traced and takes the `cond`; one at a time it is concrete
    # and takes its branch statically -- two compilations of the same
    # arithmetic, which XLA fuses differently: 2.1e-14 apart at b = 1e4, well
    # inside the accuracy bound. Not a measure of accuracy, which is tested
    # against mpmath elsewhere.
    np.testing.assert_allclose(np.asarray(batched), np.asarray(single), rtol=5e-14)
    assert bool(jnp.all(jnp.isfinite(jax.vmap(jax.grad(f))(bs))))


@pytest.mark.parametrize("z", [1e-8, 1e-12])
@pytest.mark.parametrize("b", [1.5, 20.0])
def test_mixed_partial_at_small_z(z, b):
    """REGRESSION (GH-68): ``d/db d/dz B`` to full precision as ``z -> 0``.

    ``d/dz B = z^(a-1) (1-z)^(b-1)``, so ``d/db`` of it is that times
    ``log(1-z)``. The z-rule's power must carry ``log1p(-z)`` when ``b`` is
    traced; ``log`` of a rounded ``1 - z`` kept only ~1e-16 / z of it: 5e-9
    relative at ``z = 1e-8`` and 2e-5 at ``1e-12``. Both the series and the
    continued-fraction regimes share the rule, hence both ``b``.
    """
    a = 2.0
    got = jax.grad(jax.grad(lambda bb, zz: sp.incomplete_beta(a, bb, zz), 1), 0)
    expect = z ** (a - 1) * np.log1p(-z) * np.exp((b - 1) * np.log1p(-z))
    np.testing.assert_allclose(float(got(b, jnp.asarray(z))), expect, rtol=1e-14)


def test_z_derivative_with_traced_b_at_z_eq_1():
    """With ``b`` traced, the z-rule's clamp keeps ``(1-z)^(b-1)`` exact at z = 1."""
    one = jnp.asarray(1.0)
    dz = jax.jit(jax.grad(lambda zz, bb: sp.incomplete_beta(2.0, bb, zz)))
    assert float(dz(one, 1.0)) == 1.0
    assert float(dz(one, 2.0)) == 0.0
    assert float(dz(one, 0.5)) == np.inf


# ============================================================================
# Large a


@pytest.mark.parametrize("a", [12.0, 16.0, 24.0, 50.0])
@pytest.mark.parametrize("b", [0.1, 0.5, 4.0, 10.0])
def test_large_a_matches_mpmath(a, b):
    """Above ``a = 8`` with ``b >= 0.1`` the continued fraction takes over.

    `_large_z` expands ``(1-u)^(a-1)``, whose coefficients cancel just as
    ``(1-b)_k`` does for large ``b``: the series was 4e-11 at ``a = 16``,
    2e-8 at ``24``, and wrong from about 50. Measured worst here 5e-13.
    """
    z = np.array([1e-6, 0.01, 0.3, 0.7, 0.9, 0.99, 1 - 1e-6])
    got = np.asarray(sp.incomplete_beta(a, b, jnp.asarray(z)))
    with mp.workdps(40):
        expect = np.array([float(_mp_incomplete_beta(a, b, zi)) for zi in z])
    np.testing.assert_allclose(got, expect, rtol=1e-12)


@pytest.mark.parametrize("b", [0.1, 1.0, 5.0])
def test_continuous_across_the_large_a_switch(b):
    """Series at or below ``a = 8``, continued fraction above: they must meet."""
    z = jnp.asarray([0.05, 0.5, 0.95])
    below = sp.incomplete_beta(8.0, b, z)
    above = sp.incomplete_beta(np.nextafter(8.0, 99), b, z)
    np.testing.assert_allclose(np.asarray(below), np.asarray(above), rtol=1e-13)


def _mp_hyp2f1_beta(a, b, z):
    """``B(a, b, z)`` by DLMF 8.17.7 at 40 digits: valid for any real ``b``."""
    with mp.workdps(40):
        return float(mp.mpf(z) ** a / a * mp.hyp2f1(a, 1 - b, a + 1, z))


@pytest.mark.parametrize("a", [12.0, 24.0, 50.0, 100.0])
@pytest.mark.parametrize("b", [-50.0, -2.5, -1.0, -0.3, -1e-4, 0.0, 0.05, 0.099])
def test_large_a_small_b(a, b):
    """Large ``a`` with ``b < 0.1``, including ``b <= 0``: `_large_a`.

    The series was 1e-9 off at ``a = 24`` and had no correct digits by 50; the
    continued fraction's reflection needs ``B(a, b)``, which diverges at
    ``b <= 0``. Measured worst here 8e-14.
    """
    z = np.array([0.01, 0.3, 0.5, 0.7, 0.9, 0.95, 0.99, 1 - 1e-6, 1 - 1e-12])
    got = np.asarray(sp.incomplete_beta(a, b, jnp.asarray(z)))
    expect = np.array([_mp_hyp2f1_beta(a, b, zi) for zi in z])
    ok = np.isfinite(expect) & (expect != 0)
    np.testing.assert_allclose(got[ok], expect[ok], rtol=5e-13)


@pytest.mark.parametrize("b", [-2.5, -0.3, 0.0, 0.05])
def test_continuous_across_the_large_a_switch_small_b(b):
    """Series at or below ``a = 8``, `_large_a` above: they must meet."""
    z = jnp.asarray([0.05, 0.5, 0.95, 0.999])
    below = sp.incomplete_beta(8.0, b, z)
    above = sp.incomplete_beta(np.nextafter(8.0, 99), b, z)
    np.testing.assert_allclose(np.asarray(below), np.asarray(above), rtol=1e-13)


@pytest.mark.parametrize("a", [2.0, 24.0])
def test_pole_at_z_eq_1_large_and_small_a(a):
    """``+inf`` at ``z = 1`` for ``b <= 0`` on the series and on `_large_a`."""
    for b in (-2.5, -0.5, 0.0):
        assert float(sp.incomplete_beta(a, b, jnp.asarray(1.0))) == np.inf


@pytest.mark.parametrize("a", [2.0, 24.0])
@pytest.mark.parametrize("b", [1e-6, 1e-4, 5e-4])
def test_z_eq_1_with_tiny_positive_b(a, b):
    """REGRESSION: ``B(a, b, 1)`` for ``0 < b < 1e-3`` was ``nan``.

    ``b`` sits in the near-pole band, whose expansion carries ``log(1 - z)``;
    at ``z = 1`` that is ``-inf``, though there is nothing to cancel there --
    ``(1 - z)^b`` is 0 -- so the direct form is used.
    """
    with mp.workdps(40):
        expect = float(mp.beta(a, b))
    got = float(sp.incomplete_beta(a, b, jnp.asarray(1.0)))
    np.testing.assert_allclose(got, expect, rtol=1e-12)


@pytest.mark.parametrize("b", [-1e-4, -1 + 5e-4, 2e-4])
def test_near_pole_expansion_close_to_z_eq_1(b):
    """REGRESSION: the near-pole expansion truncates like ``s^K L^(K+1)``.

    With 4 terms, ``b = -1e-4`` at ``z = 1 - 1e-15`` (``L ~ -35``) was 1.2e-12
    off; with `_POLE_TERMS` = 8 the truncation is ~1e-25.
    """
    z = np.array([0.9, 1 - 1e-8, 1 - 1e-15])
    got = np.asarray(sp.incomplete_beta(2.0, b, jnp.asarray(z)))
    expect = np.array([_mp_hyp2f1_beta(2.0, b, zi) for zi in z])
    np.testing.assert_allclose(got, expect, rtol=5e-14)


def test_vmap_over_a_through_large_a_has_finite_gradients():
    """Under `vmap` over ``a`` the `switch` runs every branch, on safe values."""
    a_s = jnp.asarray([0.5, 4.0, 8.0, 8.5, 16.0, 50.0])
    f = lambda a: sp.incomplete_beta(a, -0.5, jnp.asarray(0.97))
    batched = jax.vmap(f)(a_s)
    single = jnp.stack([f(a) for a in a_s])
    np.testing.assert_allclose(np.asarray(batched), np.asarray(single), rtol=5e-14)
    assert bool(jnp.all(jnp.isfinite(jax.vmap(jax.grad(f))(a_s))))


def test_vmap_over_a_across_the_switch_has_finite_gradients():
    """As for ``b``: under `vmap` over ``a`` both branches run, on safe values."""
    a_s = jnp.asarray([0.5, 4.0, 8.0, 8.5, 16.0, 50.0])
    f = lambda a: sp.incomplete_beta(a, 0.5, jnp.asarray(0.3))
    batched = jax.vmap(f)(a_s)
    single = jnp.stack([f(a) for a in a_s])
    np.testing.assert_allclose(np.asarray(batched), np.asarray(single), rtol=1e-14)
    assert bool(jnp.all(jnp.isfinite(jax.vmap(jax.grad(f))(a_s))))


@pytest.mark.parametrize("b", [2.5, 3.0, 4.0])
def test_higher_z_derivatives_at_z_eq_1_with_traced_b(b):
    """Traced and concrete ``b`` agree at ``z = 1`` to the third z-derivative.

    The traced path writes ``(1-z)^(b-1)`` through ``log1p(-z)`` (GH-68), which
    is ``-inf`` at ``z = 1``. A clamp there differentiates as ``0 * -inf`` and
    made the second and third derivatives ``nan``; `pow` takes the endpoint.
    """
    one = jnp.asarray(1.0)
    f = lambda bb, zz: sp.incomplete_beta(2.0, bb, zz)
    d1 = jax.grad(f, 1)
    d2 = jax.grad(d1, 1)
    d3 = jax.grad(d2, 1)
    for d in (d1, d2, d3):
        traced, concrete = float(jax.jit(d)(b, one)), float(d(b, one))
        assert traced == concrete, (b, traced, concrete)
        assert not np.isnan(traced)


@pytest.mark.parametrize("b", [1e3, 1e4, 1e5])
def test_large_b_small_a(b):
    """``a = 0.1``: the complete beta's second fraction converges slowest here.

    At 32 steps it was 1e-11 off; it has its own 64 (`_CF_COMPLETE_STEPS`).
    """
    a = 0.1
    z = np.array(
        [1e-6, 0.5 * (a + 1) / (a + b + 2), 2 * (a + 1) / (a + b + 2), 0.5, 1.0]
    )
    got = np.asarray(sp.incomplete_beta(a, b, jnp.asarray(z)))
    with mp.workdps(40):
        expect = np.array([float(_mp_incomplete_beta(a, b, zi)) for zi in z])
    np.testing.assert_allclose(got, expect, rtol=1e-12)


@pytest.mark.parametrize(("a", "b"), [(301.0, 1000.0), (300.0, 1000.0), (50.0, 2000.0)])
def test_large_b_no_premature_underflow(a, b):
    """A tiny front factor must not flush to zero while the value is normal.

    ``B(301, 1000) ~ 3.9e-307``: the front factor alone underflows, and was
    multiplied by the fraction after the fact, so this returned 0.
    """
    with mp.workdps(40):
        expect = float(mp.beta(a, b))
    got = float(sp.incomplete_beta(a, b, jnp.asarray(1.0)))
    np.testing.assert_allclose(got, expect, rtol=1e-12)


@pytest.mark.parametrize("z", [0.0, 1.0])
def test_large_b_parameter_gradients_at_the_endpoints(z):
    """``a``/``b`` gradients at ``z = 0`` and ``z = 1``, with ``b > 10``.

    The endpoint term is a genuine zero; evaluated, ``log(0)`` puts
    ``0 * -inf = nan`` into its parameter derivatives.
    """
    a, b = 2.0, 50.0
    zz = jnp.asarray(z)
    refs = (lambda t: mp.beta(t, b), lambda t: mp.beta(a, t))
    for argnum, ref in enumerate(refs):
        got = float(jax.grad(sp.incomplete_beta, argnum)(a, b, zz))
        if z == 0.0:
            expect = 0.0
        else:
            with mp.workdps(40):
                expect = float(mp.diff(ref, (a, b)[argnum]))
        np.testing.assert_allclose(got, expect, rtol=1e-11, atol=1e-300)


def test_large_b_float32():
    """The continued fraction in float32: about float32 precision, still float32."""
    z = jnp.asarray([1e-3, 0.01, 0.3, 0.9], jnp.float32)
    for b in (50.0, 1e4):
        got = sp.incomplete_beta(2.0, b, z)
        assert got.dtype == jnp.float32
        with mp.workdps(40):
            expect = np.array([float(_mp_incomplete_beta(2, b, float(zi))) for zi in z])
        np.testing.assert_allclose(np.asarray(got, np.float64), expect, rtol=5e-6)


@pytest.mark.parametrize("b", [1.5, 50.0])
@pytest.mark.parametrize("wrap", [np.float64, jnp.float64], ids=["numpy", "jax"])
def test_float32_z_with_strong_float64_b(b, wrap):
    """REGRESSION (GH-66): float32 ``z`` with a strong float64 ``b``.

    The `scan` carry started in float32 and the first step promoted it to
    float64, which `scan` rejects with a `TypeError`. ``z`` now promotes to the
    common type, as in any elementwise op -- on both the series (b = 1.5) and
    the continued fraction (b = 50).
    """
    z = jnp.asarray([0.3, 0.9], jnp.float32)
    eager = sp.incomplete_beta(2.0, wrap(b), z)
    jitted = jax.jit(lambda bb: sp.incomplete_beta(2.0, bb, z))(wrap(b))
    expect = sp.incomplete_beta(2.0, b, jnp.asarray(z, jnp.float64))
    for got in (eager, jitted):
        assert got.dtype == jnp.float64
        np.testing.assert_allclose(np.asarray(got), np.asarray(expect), rtol=1e-14)
    # A weak (Python) b does not promote.
    assert sp.incomplete_beta(2.0, b, z).dtype == jnp.float32


# Two values -- one negative, and zero -- because each compiles the a- and
# b-Jacobians afresh (~35 s apiece); `test_pole_at_z_eq_1_large_and_small_a`
# covers more `b` for the value alone.
@pytest.mark.parametrize("b", [-1.0, 0.0])
def test_pole_at_z_eq_1_for_non_positive_b(b):
    """REGRESSION (GH-67): ``B(a, b, 1) = +inf`` for ``b <= 0``, not ``nan``.

    And the series must not leak its ``nan`` there into the gradients of the
    other points through the `where` that picks the pole.
    """
    z = jnp.asarray([0.3, 0.9, 1.0])
    got = np.asarray(sp.incomplete_beta(2.0, b, z))
    assert got[-1] == np.inf
    assert np.all(np.isfinite(got[:-1]))
    for argnum in (0, 1):
        jac = np.asarray(
            jax.jacrev(lambda *ab: sp.incomplete_beta(*ab, z), argnum)(2.0, b)
        )
        assert np.all(np.isfinite(jac[:-1])), jac


@pytest.mark.parametrize("a", [0.5, 8.0, np.nextafter(8.0, 99), 50.0])
@pytest.mark.parametrize(
    "b", [-1.0, 0.0, np.nextafter(0.1, 0), 0.1, 5.0, 10.0, np.nextafter(10.0, 99)]
)
def test_static_and_traced_branch_choice_agree(a, b):
    """`_branch` (concrete, plain Python) mirrors `_branch_index` (traced)."""
    assert _branch(a, b) == int(_branch_index(a, b))
