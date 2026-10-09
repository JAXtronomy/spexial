"""Unit tests for `spexial.incomplete_beta`.

Two fixed-length series rather than a call into `hyp2f1`, so the values are
checked against three independent references over the whole domain: the
DLMF 8.17.7 hypergeometric identity, the regularized form where that is
defined, and direct quadrature where it is not.

The `b <= 0` cases are the point of the function and get their own coverage --
that is exactly where `beta(a, b) * betainc(a, b, z)` returns `nan`.
"""

import jax
import jax.numpy as jnp
import jax.scipy.special as jsp
import numpy as np
import pytest
from scipy.integrate import quad
from scipy.special import beta as scipy_beta, betainc as scipy_betainc, hyp2f1

import spexial as sp
from spexial._src.beta import _EXPRL_BAND, _incomplete_beta_core, _is_static_one

# `a > 0` always. `b` is any real: the negative and zero values are the reason
# this exists, and come up as ordinary slopes in double power-law profiles.
AS = [0.25, 0.5, 1.0, 1.31, 2.0, 4.5]
BS = [-2.5, -1.0, -0.1, 0.0, 0.3, 1.31, 4.0]
# Straddles the z = 1/2 branch switch deliberately, and both endpoints.
ZS = [1e-6, 1e-3, 0.1, 0.3, 0.5, 0.5 + 1e-7, 0.7, 0.9, 0.99, 1 - 1e-6]


# ============================================================================
# Agreement with independent references


@pytest.mark.parametrize("a", AS)
@pytest.mark.parametrize("b", BS)
def test_matches_the_hyp2f1_identity(a, b):
    """B(a,b,z) = z^a/a * 2F1(a, 1-b; a+1; z), DLMF 8.17.7.

    The only reference that covers `b <= 0` as well as `b > 0`, which is why it
    carries the whole parameter grid.
    """
    z = np.array(ZS)
    got = np.asarray(sp.incomplete_beta(a, b, jnp.asarray(z)))
    expect = z**a / a * hyp2f1(a, 1.0 - b, a + 1.0, z)
    np.testing.assert_allclose(got, expect, rtol=1e-11)


@pytest.mark.parametrize("a", AS)
@pytest.mark.parametrize("b", [b for b in BS if b > 0])
def test_matches_the_regularized_form(a, b):
    """For ``b > 0``, ``beta(a, b) * betainc(a, b, z)`` is a valid reference.

    It is not for ``b <= 0``; see `test_finite_where_the_complete_beta_diverges`.
    """
    z = np.array(ZS)
    got = np.asarray(sp.incomplete_beta(a, b, jnp.asarray(z)))
    expect = scipy_beta(a, b) * scipy_betainc(a, b, z)
    np.testing.assert_allclose(got, expect, rtol=1e-11)


@pytest.mark.parametrize("a", [0.5, 1.0, 2.0])
@pytest.mark.parametrize("b", [-2.5, -1.0, 0.0])
def test_matches_quadrature_where_b_is_not_positive(a, b):
    """Direct quadrature of the definition, over the domain no closed form covers.

    Slow and independent of every series in the module, which is what makes it
    worth having for exactly the parameters the function exists to serve.
    """
    for z in (0.1, 0.5, 0.9):
        expect = quad(lambda t: t ** (a - 1) * (1 - t) ** (b - 1), 0, z, limit=400)[0]
        got = float(sp.incomplete_beta(a, b, jnp.asarray(z)))
        np.testing.assert_allclose(got, expect, rtol=1e-9)


# ============================================================================
# The domain upstream cannot reach


@pytest.mark.parametrize("b", [-2.5, -1.0, -0.1, 0.0])
def test_finite_where_the_complete_beta_diverges(b):
    """REGRESSION GUARD ON UPSTREAM: ``beta * betainc`` is `nan` for ``b <= 0``.

    B(a, b) has a pole there; the *product* does not. This asserts both halves
    -- that the reconstruction fails, and that we do not -- so it says
    something if either ever changes.
    """
    z = jnp.asarray([0.1, 0.5, 0.9, 1 - 1e-6])
    assert bool(jnp.all(jnp.isnan(jsp.beta(2.0, b) * jsp.betainc(2.0, b, z))))
    assert bool(jnp.all(jnp.isfinite(sp.incomplete_beta(2.0, b, z))))


@pytest.mark.parametrize("a", AS)
@pytest.mark.parametrize("b", BS)
def test_branch_switch_is_continuous(a, b):
    """The two series must agree where they meet, at ``z = 1/2``.

    Each is clamped to the side it is valid on and a `where` picks between
    them, so a step here would be invisible to every other test -- they sample
    both sides but never the seam.

    The tolerance is the *measured* step, not machine precision: the two
    series truncate differently, so they meet to about 1e-11 relative rather
    than exactly. Worst over this grid is 2.9e-11, at ``a = 4.5, b = -2.5``,
    which is comfortably inside the 1e-11 rtol the reference tests assert
    against `hyp2f1` -- the seam is not the accuracy-limiting feature.
    """
    below = float(sp.incomplete_beta(a, b, jnp.asarray(0.5 - 1e-12)))
    above = float(sp.incomplete_beta(a, b, jnp.asarray(0.5 + 1e-12)))
    np.testing.assert_allclose(below, above, rtol=5e-11)


@pytest.mark.parametrize("b", [-1.0, -1e-4, 0.0, 1e-4])
def test_near_the_b_plus_m_pole(b):
    """`_large_z` expands around ``b + m == 0``, where the direct form cancels.

    ``b`` at and either side of an integer is where the switch fires; the
    expansion has to be accurate there, not merely finite.
    """
    z = jnp.asarray([0.6, 0.8, 0.95])
    got = np.asarray(sp.incomplete_beta(2.0, b, z))
    expect = np.asarray(z) ** 2.0 / 2.0 * hyp2f1(2.0, 1.0 - b, 3.0, np.asarray(z))
    np.testing.assert_allclose(got, expect, rtol=1e-10)


# ============================================================================
# Endpoints and derivatives


@pytest.mark.parametrize("a", [0.5, 2.0])
@pytest.mark.parametrize("b", [1.0, 2.5])
def test_endpoints(a, b):
    """B(a,b,0) = 0, and for ``b > 0``, B(a,b,1) is the complete beta function."""
    got = sp.incomplete_beta(a, b, jnp.asarray([0.0, 1.0]))
    np.testing.assert_allclose(float(got[0]), 0.0, atol=1e-300)
    np.testing.assert_allclose(float(got[1]), scipy_beta(a, b), rtol=1e-11)


@pytest.mark.parametrize("a", [0.5, 1.0, 2.0])
@pytest.mark.parametrize("b", [-1.0, 0.0, 1.5])
def test_z_derivative_is_the_integrand(a, b):
    """The custom JVP must equal ``z^(a-1) (1-z)^(b-1)``, exactly.

    A `custom_jvp` silently replaces the true derivative: get it wrong and
    every value stays right while every gradient is quietly wrong. By Leibniz
    this one is just the integrand at the endpoint, so it can be checked in
    closed form rather than against a difference.
    """
    for z in (0.1, 0.49, 0.51, 0.9):
        got = jax.grad(lambda zz: sp.incomplete_beta(a, b, zz))(jnp.asarray(z))
        expect = z ** (a - 1.0) * (1.0 - z) ** (b - 1.0)
        np.testing.assert_allclose(float(got), expect, rtol=1e-10)


@pytest.mark.parametrize("a", [0.5, 2.0])
@pytest.mark.parametrize("b", [-1.0, 0.5])
def test_parameter_derivatives(a, b):
    """The ``a``/``b`` tangents fall back to autodiff of the series; check them.

    There is no cheap closed form for these, so the rule differentiates the
    implementation -- but only when asked, via `symbolic_zeros`. Finite
    differences of the primal are the reference.
    """
    z = jnp.asarray(0.7)
    eps = 1e-6
    for argnum, (da, db) in enumerate([(eps, 0.0), (0.0, eps)]):
        got = jax.grad(sp.incomplete_beta, argnums=argnum)(a, b, z)
        expect = (
            sp.incomplete_beta(a + da, b + db, z)
            - sp.incomplete_beta(a - da, b - db, z)
        ) / (2 * eps)
        np.testing.assert_allclose(float(got), float(expect), rtol=1e-5)


def test_second_derivative_composes():
    """`custom_jvp`, not `custom_vjp`, so ``jacfwd(jacrev(...))`` still works.

    A `custom_vjp` can never be forward-differentiated, which would make the
    hessian of anything downstream raise unconditionally.
    """
    f = lambda zz: sp.incomplete_beta(2.0, 1.5, zz)
    assert bool(jnp.isfinite(jax.jacfwd(jax.jacrev(f))(jnp.asarray(0.3))))


def test_forward_and_reverse_agree():
    """`jacfwd` and `jacrev` must give the same answer.

    The two modes disagreeing is the signature of a `where` transposing a zero
    cotangent into a branch holding an infinity -- the family of bug that bit
    `eval_gegenbauer` at ``x = inf``. Both series here are masked, so this pins
    that the masking did not create one.
    """
    f = lambda zz: sp.incomplete_beta(1.31, -1.0, zz)
    for z in (0.2, 0.5, 0.85):
        fwd = float(jax.jacfwd(f)(jnp.asarray(z)))
        rev = float(jax.jacrev(f)(jnp.asarray(z)))
        np.testing.assert_allclose(fwd, rev, rtol=1e-12)


# ============================================================================
# JAX transformations and shapes


@pytest.mark.parametrize("shape", [(), (5,), (2, 3), (2, 1, 4)])
def test_broadcasts_over_z_at_every_rank(shape):
    """``z`` may be any shape; ``a`` and ``b`` are scalars."""
    z = jnp.full(shape, 0.35)
    assert sp.incomplete_beta(2.0, 0.5, z).shape == shape


def test_jit_and_vmap_compose():
    """`jit` and `vmap` agree with the unwrapped call.

    `vmap` in particular is the reason this is two fixed-length `scan`s rather
    than `hyp2f1`: upstream's `while_loop` makes every lane pay the worst
    lane's trip count.
    """
    z = jnp.asarray(ZS)
    direct = sp.incomplete_beta(2.0, -1.0, z)
    jitted = jax.jit(lambda zz: sp.incomplete_beta(2.0, -1.0, zz))(z)
    mapped = jax.vmap(lambda zz: sp.incomplete_beta(2.0, -1.0, zz))(z)
    np.testing.assert_allclose(np.asarray(jitted), np.asarray(direct), rtol=1e-14)
    np.testing.assert_allclose(np.asarray(mapped), np.asarray(direct), rtol=1e-14)


def test_integer_arguments_are_promoted():
    """Integer ``a``/``b``/``z`` promote to float, as everywhere else here."""
    got = sp.incomplete_beta(2, 1, jnp.asarray(1))
    assert jnp.issubdtype(got.dtype, jnp.floating)
    np.testing.assert_allclose(float(got), scipy_beta(2.0, 1.0), rtol=1e-12)


# ============================================================================
# The a == 1 closed form


@pytest.mark.parametrize("b", BS)
def test_a_eq_1_closed_form_matches_the_series(b):
    """A static ``a == 1`` takes `_a_eq_1`; it must agree with the series it skips."""
    z = jnp.asarray(ZS)
    np.testing.assert_allclose(
        np.asarray(sp.incomplete_beta(1.0, b, z)),
        np.asarray(_incomplete_beta_core(1.0, b, z)),
        rtol=1e-11,
    )


def test_a_eq_1_traced_takes_the_series():
    """A traced ``a`` cannot be inspected, so it takes the series -- same answer."""
    z = jnp.asarray(ZS)
    traced = jax.jit(lambda a: sp.incomplete_beta(a, -1.0, z))(1.0)
    static = sp.incomplete_beta(1.0, -1.0, z)
    np.testing.assert_allclose(np.asarray(traced), np.asarray(static), rtol=1e-11)


@pytest.mark.parametrize("z", [0.1, 0.5, 0.9, 1 - 1e-6])
def test_a_eq_1_b_derivatives_at_the_removable_singularity(z):
    """``b = 0`` is removable; every order of the ``b``-derivative must be right.

    ``B(1, b, z) = -L sum_k (bL)^k / (k+1)!`` with ``L = log(1-z)``, so the
    n-th ``b``-derivative at ``b = 0`` is ``-L^(n+1) / (n+1)``. A `where` that
    substituted ``-L`` at ``b = 0`` would get the value right and all of these
    wrong.
    """
    L = np.log1p(-z)
    f = lambda b: sp.incomplete_beta(1.0, b, jnp.asarray(z))
    d1, d2, d3 = jax.grad(f), jax.grad(jax.grad(f)), jax.grad(jax.grad(jax.grad(f)))
    np.testing.assert_allclose(float(f(0.0)), -L, rtol=1e-14)
    np.testing.assert_allclose(float(d1(0.0)), -(L**2) / 2, rtol=1e-13)
    np.testing.assert_allclose(float(d2(0.0)), -(L**3) / 3, rtol=1e-13)
    np.testing.assert_allclose(float(d3(0.0)), -(L**4) / 4, rtol=1e-13)


@pytest.mark.parametrize("side", [-1, 1])
def test_a_eq_1_continuous_across_the_series_band(side):
    """Value and ``b``-derivative agree with the reference on both sides of the band.

    `_a_eq_1` switches from the series to ``expm1(x) / x`` at
    ``|x| = _EXPRL_BAND``; a step there would be invisible to tests that never
    sample the seam.
    """
    z = 0.9
    L = np.log1p(-z)
    for x in (_EXPRL_BAND * (1 - 1e-9), _EXPRL_BAND * (1 + 1e-9)):
        b = side * x / abs(L)
        got = float(sp.incomplete_beta(1.0, b, jnp.asarray(z)))
        np.testing.assert_allclose(got, -np.expm1(b * L) / b, rtol=1e-14)
        grad = float(
            jax.grad(lambda bb: sp.incomplete_beta(1.0, bb, jnp.asarray(z)))(b)
        )
        expect = (np.expm1(b * L) - b * L * np.exp(b * L)) / b**2
        np.testing.assert_allclose(grad, expect, rtol=1e-12)


def test_a_eq_1_forward_and_reverse_agree():
    """The masked branches must not leak a ``0 * inf`` into either mode."""
    f = lambda bb: sp.incomplete_beta(1.0, bb, jnp.asarray(0.7))
    for b in (-1.0, 0.0, 1e-9, 0.3):
        fwd = float(jax.jacfwd(f)(b))
        rev = float(jax.jacrev(f)(b))
        np.testing.assert_allclose(fwd, rev, rtol=1e-14)


@pytest.mark.parametrize("b", [0.5, 1.0, 2.5, 300.0])
def test_a_eq_1_at_z_eq_1(b):
    """``B(1, b, 1) = 1/b`` for ``b > 0``, and ``d/db`` there is ``-1/b^2``.

    The ``-L expm1(x)/x`` form is ``inf * 0`` here, since ``L = log(1-z) = -inf``.
    Both differentiation modes are checked: the masked branch holds an infinity.
    """
    one = jnp.asarray(1.0)
    np.testing.assert_allclose(
        float(sp.incomplete_beta(1.0, b, one)), 1 / b, rtol=1e-15
    )
    for d in (jax.jacfwd, jax.jacrev):
        got = float(d(lambda bb: sp.incomplete_beta(1.0, bb, one))(b))
        np.testing.assert_allclose(got, -1 / b**2, rtol=1e-14)


@pytest.mark.parametrize("b", [-2.5, -1e-300, 0.0])
def test_a_eq_1_diverges_at_z_eq_1(b):
    """For ``b <= 0`` the integral diverges at ``z = 1``: ``+inf``, not ``nan``.

    ``b = 0`` is the case that needs care: ``b * log(1-z)`` is ``0 * -inf``.
    """
    assert float(sp.incomplete_beta(1.0, b, jnp.asarray(1.0))) == np.inf


@pytest.mark.parametrize(
    "z",
    [
        jnp.asarray([0.3, 0.9], jnp.float32),
        jnp.asarray([0.3, 0.9], jnp.bfloat16),
        jnp.asarray([0.3, 0.9], jnp.float16),
        jnp.asarray([0.3, 0.9]),
        0.3,
    ],
    ids=["float32", "bfloat16", "float16", "float64", "weak"],
)
def test_a_eq_1_keeps_the_input_dtype(z):
    """The closed form returns what the series would: same dtype, same weak type.

    Otherwise the output dtype would depend on whether ``a`` happens to be 1.
    """
    got, series = sp.incomplete_beta(1.0, 0.5, z), _incomplete_beta_core(1.0, 0.5, z)
    assert (got.dtype, got.weak_type) == (series.dtype, series.weak_type)


@pytest.mark.parametrize(
    ("a", "expect"),
    [
        (1.0, True),
        (1, True),
        (np.float32(1), True),
        (jnp.asarray(1.0), True),
        (np.array([1.0]), False),
        (1 + 1e-12, False),
    ],
)
def test_static_one_dispatch(a, expect):
    """Only a concrete real scalar exactly equal to 1 takes the closed form."""
    assert _is_static_one(a) is expect


@pytest.mark.parametrize(("b", "z"), [(20.0, 0.9), (2.5, 1 - 1e-10), (2.5, 0.999)])
def test_a_eq_1_mixed_partial_where_exp_x_is_tiny(b, z):
    """``d/dz d/db B = log(1-z) (1-z)^(b-1)``, even where ``exp(bL)`` is tiny.

    JAX differentiates ``expm1(x)`` as ``expm1(x) + 1``, which is exactly zero
    below ``x ~ -37`` -- so the far branch uses ``1 - exp(x)``, whose derivative
    is ``exp(x)`` itself.
    """
    got = jax.grad(jax.grad(lambda bb, zz: sp.incomplete_beta(1.0, bb, zz), 0), 1)
    expect = np.log1p(-z) * np.exp((b - 1) * np.log1p(-z))
    np.testing.assert_allclose(float(got(b, jnp.asarray(z))), expect, rtol=1e-12)


@pytest.mark.parametrize("z", [1e-8, 1e-12])
def test_a_eq_1_mixed_partial_at_small_z(z):
    """``d/db d/dz B = log(1-z) (1-z)^(b-1)`` to full precision as ``z -> 0``.

    The z-rule's ``b``-derivative must carry ``log1p(-z)``, not ``log(1 - z)``,
    which keeps only ``1e-16 / z`` relative precision.
    """
    got = jax.grad(jax.grad(lambda bb, zz: sp.incomplete_beta(1.0, bb, zz), 1), 0)
    expect = np.log1p(-z) * np.exp(-np.log1p(-z))
    np.testing.assert_allclose(float(got(0.0, jnp.asarray(z))), expect, rtol=1e-14)


def _a_eq_1_ref(b, z, n=0):
    """``d^n/db^n B(1, b, z)`` at 50 digits, from the closed form itself."""
    import mpmath as mp  # noqa: PLC0415

    with mp.workdps(50):
        L = mp.log1p(-mp.mpf(z))
        f = lambda t: -L if t == 0 else -mp.expm1(t * L) / t
        return float(f(mp.mpf(b)) if n == 0 else mp.diff(f, mp.mpf(b), n))


@pytest.mark.parametrize("side", [-1, 1])
@pytest.mark.parametrize("n", [2, 3])
def test_a_eq_1_higher_b_derivatives_across_the_series_band(side, n):
    """The 2nd and 3rd ``b``-derivatives are right on both sides of the band.

    The direct quotient's derivatives cancel as ``|x| -> 0``; `_EXPRL_BAND` is
    where they are still accurate (3e-15 / 8e-14 at ``|x| = 1/2``). Narrowed to
    0.05 they were 2e-12 / 2e-10, which the first derivative alone did not show.
    """
    z = 0.9
    L = np.log1p(-z)
    d = lambda bb: sp.incomplete_beta(1.0, bb, jnp.asarray(z))
    for _ in range(n):
        d = jax.grad(d)
    for x in (_EXPRL_BAND * (1 - 1e-9), _EXPRL_BAND * (1 + 1e-9)):
        b = side * x / abs(L)
        np.testing.assert_allclose(float(d(b)), _a_eq_1_ref(b, z, n), rtol=1e-12)


@pytest.mark.parametrize("b", [2.5, 3.0, 4.0])
def test_a_eq_1_higher_z_derivatives_at_z_eq_1_with_traced_b(b):
    """Traced and concrete ``b`` agree at ``z = 1`` to the third z-derivative.

    A traced ``b`` writes ``(1-z)^(b-1)`` via ``log(1-z)``, ``-inf`` at z = 1;
    clamping it differentiated as ``0 * -inf`` and made these ``nan``.
    """
    one = jnp.asarray(1.0)
    d1 = jax.grad(lambda bb, zz: sp.incomplete_beta(1.0, bb, zz), 1)
    d2 = jax.grad(d1, 1)
    d3 = jax.grad(d2, 1)
    for d in (d1, d2, d3):
        traced, concrete = float(jax.jit(d)(b, one)), float(d(b, one))
        assert traced == concrete, (b, traced, concrete)
        assert not np.isnan(traced)


@pytest.mark.parametrize("b", [-500.0, -50.0, 50.0])
def test_a_eq_1_in_the_log1p_band(b):
    """REGRESSION: XLA's ``log1p`` is ~240 ulp off for ``z`` in (0.3, 0.42].

    The closed form multiplies that by ``|bL|``: 7e-12 at ``b = -500``. `_log1m`
    switches to ``log(1 - z)`` there, which is exact enough.
    """
    z = np.linspace(0.3, 0.42, 25)
    got = np.asarray(sp.incomplete_beta(1.0, b, jnp.asarray(z)))
    expect = np.array([_a_eq_1_ref(b, zi) for zi in z])
    np.testing.assert_allclose(got, expect, rtol=1e-13)


def test_a_eq_1_infinite_b():
    """``b = -inf`` diverges for ``z > 0`` (was ``nan``); ``b = +inf`` is 0."""
    z = jnp.asarray([0.0, 0.3, 1.0])
    np.testing.assert_array_equal(
        np.asarray(sp.incomplete_beta(1.0, -np.inf, z)), [0.0, np.inf, np.inf]
    )
    np.testing.assert_array_equal(
        np.asarray(sp.incomplete_beta(1.0, np.inf, z)), [0.0, 0.0, 0.0]
    )


def test_a_eq_1_rejects_an_array_b():
    """``b`` is a scalar; the closed form must not quietly broadcast it.

    The series path cannot (it raises inside its `scan`), so neither may this,
    or the behaviour would depend on ``a``. Under pytest the jaxtyping hook
    rejects it first; without it, the wrapper's own check does. Both are a
    `TypeError`.
    """
    with pytest.raises(TypeError):
        sp.incomplete_beta(1.0, np.array([0.0, 1.0]), jnp.asarray(0.3))
