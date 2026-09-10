"""Unit tests for the spherical harmonics and spherical Legendre functions.

Two of the four functions here exist because `jax.scipy.special.sph_harm_y` is
*wrong*, not merely narrow, so a substantial part of this file pins the exact
defects: the broadcasting one that returns silently incorrect values, and the
pole derivatives. Those tests are worth more than the implementations they
guard -- the day JAX fixes either, they say so.
"""

import jax
import jax.numpy as jnp
import jax.scipy.special as jsp
import numpy as np
import pytest
import scipy.special as sps
from jaxtyping import TypeCheckError

import spexial as sp
from spexial._src.sph_harm import _reduced, _seed

# A spread of angles that deliberately includes both poles and a point a
# nanoradian off one, since that is where every interesting failure lives.
THETA = np.concatenate(
    [
        np.linspace(0.07, np.pi - 0.07, 11),
        [0.0, np.pi, 1e-9, np.pi - 1e-9],
    ]
)
PHI = np.linspace(0.0, 2 * np.pi, THETA.size, endpoint=False)
UVEC = np.stack(
    [np.sin(THETA) * np.cos(PHI), np.sin(THETA) * np.sin(PHI), np.cos(THETA)], -1
)


def orders(n_max):
    """Every valid ``(n, m)`` with ``|m| <= n <= n_max``, negatives included."""
    return [(n, m) for n in range(n_max + 1) for m in range(-n, n + 1)]


# ============================================================================
# Agreement with SciPy


@pytest.mark.parametrize(("n", "m"), orders(8))
def test_sph_legendre_p_matches_scipy(n, m):
    """The value is `scipy.special.sph_legendre_p`, to near machine precision.

    This is also what pins the **Condon-Shortley phase**: SciPy's
    `sph_legendre_p` and `lpmv` and GSL's ``gsl_sf_legendre_sphPlm`` all carry
    it, so agreeing with SciPy at ``m >= 1`` is agreeing about the sign.
    ``m = 0`` cannot pin it -- the phase is :math:`(-1)^m` -- which is why the
    parametrization runs over every order rather than a representative few.
    """
    got = np.asarray(sp.sph_legendre_p(n, m, jnp.asarray(THETA)))
    expect = np.asarray(sps.sph_legendre_p(n, m, THETA)).reshape(-1)
    np.testing.assert_allclose(got, expect, rtol=1e-12, atol=1e-14)


@pytest.mark.parametrize(("n", "m"), orders(8))
def test_sph_harm_y_matches_scipy(n, m):
    """The full complex harmonic agrees with `scipy.special.sph_harm_y`."""
    got = np.asarray(sp.sph_harm_y(n, m, jnp.asarray(THETA), jnp.asarray(PHI)))
    expect = np.asarray(sps.sph_harm_y(n, m, THETA, PHI)).reshape(-1)
    np.testing.assert_allclose(got, expect, rtol=1e-12, atol=1e-14)


@pytest.mark.parametrize(("n", "m"), orders(8))
def test_sph_harm_y_cart_matches_scipy(n, m):
    """The Cartesian form is the same function, evaluated from a unit vector."""
    got = np.asarray(sp.sph_harm_y_cart(n, m, jnp.asarray(UVEC)))
    expect = np.asarray(sps.sph_harm_y(n, m, THETA, PHI)).reshape(-1)
    np.testing.assert_allclose(got, expect, rtol=1e-12, atol=1e-14)


@pytest.mark.parametrize(("n", "m"), [(0, 0), (1, 1), (3, 0), (4, 2), (6, 6)])
def test_sph_harm_y_cart_all_matches_scipy_layout(n, m):
    """Shape *and* index layout follow `scipy.special.sph_harm_y_all`.

    The layout is the part worth pinning: negative orders live at the far end
    of the second axis, so ``Y[l, -k]`` reaches them by ordinary Python
    negative indexing. Getting the values right with the columns transposed
    would be a silent disaster for every consumer.
    """
    got = np.asarray(sp.sph_harm_y_cart_all(n, m, jnp.asarray(UVEC)))
    expect = np.asarray(sps.sph_harm_y_all(n, m, THETA, PHI))
    assert got.shape == expect.shape
    np.testing.assert_allclose(got, expect, rtol=1e-12, atol=1e-14)


@pytest.mark.parametrize(("n", "m"), [(0, 0), (2, 1), (5, 5), (7, 3)])
def test_sph_harm_y_cart_all_matches_the_one_pair_function(n, m):
    """The batched table is the per-pair function, entry by entry.

    `sph_harm_y_cart_all` shares both recurrences across the table instead of
    restarting them; this is the direct statement of what that optimisation has
    to preserve.
    """
    table = sp.sph_harm_y_cart_all(n, m, jnp.asarray(UVEC))
    for deg in range(n + 1):
        for k in range(-m, m + 1):
            expect = (
                sp.sph_harm_y_cart(deg, k, jnp.asarray(UVEC))
                if abs(k) <= deg
                else jnp.zeros(THETA.size, dtype=table.dtype)
            )
            np.testing.assert_allclose(
                np.asarray(table[deg, k]), np.asarray(expect), rtol=1e-13, atol=1e-15
            )


@pytest.mark.parametrize(("n", "m"), [(0, 0), (1, 1), (4, 2), (6, 6), (7, 0)])
def test_terms_and_stacked_table_are_the_same_values(n, m):
    """`sph_harm_y_cart_all_terms` and `sph_harm_y_cart_all` differ only in container.

    Same values, same indexing, same layout -- ``terms[i][j]`` against
    ``table[i, j]``, negative orders included. The container is a performance
    decision, not a semantic one, and this is the statement of that.
    """
    uvec = jnp.asarray(UVEC)
    terms = sp.sph_harm_y_cart_all_terms(n, m, uvec)
    table = sp.sph_harm_y_cart_all(n, m, uvec)
    assert len(terms) == n + 1
    assert all(len(row) == 2 * m + 1 for row in terms)
    for i in range(n + 1):
        for j in range(-m, m + 1):
            np.testing.assert_allclose(
                np.asarray(terms[i][j]), np.asarray(table[i, j]), rtol=1e-13, atol=1e-15
            )


def test_terms_is_not_jitted_so_it_can_fuse():
    """REGRESSION: the terms form must stay un-`jit`ted, and that is load-bearing.

    A `jax.jit`-wrapped function returning a pytree materializes each leaf as
    its own output buffer at the call boundary -- exactly the fusion this form
    exists to preserve. Every other public function here is jitted; this one
    must not be, and a well-meaning decorator would silently undo the whole
    point without changing a single value.
    """
    assert not isinstance(sp.sph_harm_y_cart_all_terms, jax.stages.Wrapped)
    assert not hasattr(sp.sph_harm_y_cart_all_terms, "lower")
    # It still traces and composes: `jit` belongs around it, not on it.
    uvec = jnp.asarray(UVEC)
    fused = jax.jit(
        lambda v: sum(
            row[j].real
            for i, row in enumerate(sp.sph_harm_y_cart_all_terms(3, 3, v))
            for j in range(i + 1)
        )
    )(uvec)
    assert np.all(np.isfinite(np.asarray(fused)))


@pytest.mark.parametrize(("n", "m"), [(2, 3), (-1, 0)])
def test_terms_validates_its_orders(n, m):
    """The same eager validation as the rest of the family."""
    with pytest.raises(ValueError, match="0 <= "):
        sp.sph_harm_y_cart_all_terms(n, m, jnp.asarray([0.0, 0.0, 1.0]))


# ============================================================================
# The upstream defects


def test_upstream_sph_harm_y_does_not_broadcast():
    """REGRESSION GUARD ON UPSTREAM: `jax.scipy.special.sph_harm_y` is wrong.

    It indexes its Legendre table with ``jnp.arange(len(n))``, pairing ``n[i]``
    with ``theta[i]`` rather than broadcasting, so a length-1 degree against a
    batch of angles is right only at index 0. This asserts the *defect*, not
    our behaviour: when JAX fixes it, this test fails and the row in the
    coverage registry can be revisited. Ours is checked against SciPy above.
    """
    theta, phi = jnp.asarray(THETA[:4]), jnp.asarray(PHI[:4])
    upstream = np.asarray(
        jsp.sph_harm_y(jnp.asarray([2]), jnp.asarray([1]), theta, phi, n_max=2)
    )
    expect = np.asarray(sps.sph_harm_y(2, 1, THETA[:4], PHI[:4]))
    # Index 0 is right; the rest are not.
    assert abs(upstream[0] - expect[0]) < 1e-12
    assert np.max(np.abs(upstream - expect)) > 1e-3
    # Ours is right at every index.
    ours = np.asarray(sp.sph_harm_y(2, 1, theta, phi))
    np.testing.assert_allclose(ours, expect, rtol=1e-12, atol=1e-14)


def test_upstream_sph_harm_y_rejects_rank_0_input():
    """REGRESSION GUARD ON UPSTREAM: ``len()`` of a 0-d array raises.

    Ours takes scalars, since the degree and order are static and there is
    nothing to take the length of.
    """
    with pytest.raises(TypeError):
        jsp.sph_harm_y(
            jnp.asarray(1), jnp.asarray(0), jnp.asarray(0.5), jnp.asarray(0.5), n_max=1
        )
    assert np.isfinite(complex(sp.sph_harm_y(1, 0, 0.5, 0.5)).real)


@pytest.mark.parametrize(("n", "m"), [(1, 0), (2, 0), (2, 1), (3, 3)])
@pytest.mark.parametrize("pole", [0.0, np.pi])
def test_upstream_pole_derivative_is_nan_and_ours_is_finite(n, m, pole):
    r"""REGRESSION GUARD ON UPSTREAM: every ``n >= 1`` derivative is `nan` at a pole.

    Upstream differentiates :math:`\sqrt{1 - \cos^2\theta}`, whose slope is
    infinite there; the surviving ``0 * inf`` is `nan`. The true derivative is
    finite. Building :math:`\sin^m\theta` as an integer power never forms the
    square root.

    ``n = m = 0`` is deliberately excluded: upstream used to be `nan` there too
    and has since been fixed, so including it would make this test fail for the
    wrong reason.
    """

    def upstream(t):
        return jsp.sph_harm_y(
            jnp.asarray([n]),
            jnp.asarray([m]),
            jnp.asarray([t]),
            jnp.asarray([0.0]),
            n_max=n,
        ).real[0]

    # `nan` at almost every pair, but `-inf` at (n=2, m=1, theta=pi): the
    # `0 * inf` sometimes survives as the infinity instead of collapsing.
    # Either way it is unusable, which is the claim being made.
    assert not np.isfinite(float(jax.grad(upstream)(pole)))
    ours = float(jax.grad(lambda t: sp.sph_legendre_p(n, m, t))(pole))
    assert np.isfinite(ours)


@pytest.mark.parametrize(("n", "m"), [(1, 0), (2, 1), (3, 2), (4, 4)])
def test_pole_derivative_matches_a_one_sided_difference(n, m):
    r"""The finite pole derivative is also the *right* number.

    Checked one-sided, from inside the domain: SciPy's `sph_legendre_p` carries
    :math:`(1-u^2)^{m/2}`, which is :math:`\lvert\sin\theta\rvert` and
    therefore even about ``theta = 0``, so a central difference through the pole
    reports zero for every odd ``m`` regardless of the truth. Ours uses
    :math:`\sin^m\theta`, the analytic continuation, which is odd for odd
    ``m`` -- the two agree on ``[0, pi]``, which is SciPy's whole domain.
    """
    h = 1e-6
    f = lambda t: sp.sph_legendre_p(n, m, t)
    onesided = (float(f(2 * h)) - 4 * float(f(h)) + 3 * float(f(0.0))) / (-2 * h)
    got = float(jax.grad(f)(0.0))
    np.testing.assert_allclose(got, onesided, rtol=2e-5, atol=1e-9)


def test_cartesian_gradient_on_the_z_axis_is_not_zero():
    r"""The reason `sph_harm_y_cart` exists at all.

    In :math:`(\theta, \phi)` the Cartesian gradient of any :math:`m \ge 1`
    term is exactly ``0.0`` on the axis, because neither angle has a
    directional derivative there. The true limit is not zero. This checks the
    autodiff gradient against a finite difference taken along ``x`` from the
    pole, which is the only honest reference for a limit the coordinates
    cannot express.
    """
    axis = jnp.asarray([0.0, 0.0, 1.0])
    grad = np.asarray(jax.grad(lambda v: sp.sph_harm_y_cart(2, 1, v).real)(axis))

    eps = 1e-6
    off = jnp.asarray([eps, 0.0, np.sqrt(1 - eps**2)])
    fd = float(
        (sp.sph_harm_y_cart(2, 1, off).real - sp.sph_harm_y_cart(2, 1, axis).real) / eps
    )

    assert abs(grad[0]) > 0.1  # not the spurious zero
    np.testing.assert_allclose(grad[0], fd, rtol=1e-5)


# ============================================================================
# Numerical behaviour


@pytest.mark.parametrize("m", [40, 90, 118, 150])
def test_high_order_seed_does_not_overflow(m):
    """REGRESSION: :math:`p_m^m = (-1)^m (2m-1)!!` overflows float64 near m = 90.

    The normalization is folded into the recurrence *seed*, in log space, so
    the astronomically large unnormalized quantity is never materialized. Built
    the obvious way -- recurrence first, ``N_lm`` afterwards -- everything from
    ``m = 90`` up is ``inf * 0 = nan``.
    """
    got = np.asarray(sp.sph_legendre_p(m, m, jnp.asarray(THETA)))
    assert np.all(np.isfinite(got))
    expect = np.asarray(sps.sph_legendre_p(m, m, THETA)).reshape(-1)
    np.testing.assert_allclose(got, expect, rtol=1e-11, atol=1e-13)


def test_reduced_legendre_is_order_one_at_high_m():
    """The *reduced*, normalized quantity stays O(1) where the raw one overflows."""
    u = jnp.linspace(-0.99, 0.99, 21)
    for m in (0, 30, 90, 150):
        q = np.asarray(_reduced(m, m, u))
        assert np.all(np.isfinite(q))
        assert np.max(np.abs(q)) < 10.0
    assert np.all(np.isfinite(np.asarray(_seed(200, u))))


def test_zero_vector_is_finite_and_has_a_finite_hessian():
    r"""A zero direction is well defined here, and stays differentiable twice.

    `sph_harm_y_cart` deliberately does not normalize, which is what makes this
    work: :math:`Y_0^0` is its constant and every :math:`m \ge 1` term is zero,
    where ``uvec / |uvec|`` would be ``0/0``. Consumers that floor the radius
    themselves depend on the second derivative surviving too.
    """
    zero = jnp.zeros(3)
    assert float(sp.sph_harm_y_cart(0, 0, zero).real) == pytest.approx(
        1 / np.sqrt(4 * np.pi)
    )
    assert complex(sp.sph_harm_y_cart(2, 1, zero)) == 0
    hess = jax.hessian(lambda v: sp.sph_harm_y_cart(2, 1, v).real)(zero)
    assert np.all(np.isfinite(np.asarray(hess)))


@pytest.mark.parametrize("shape", [(), (5,), (2, 3), (2, 1, 4)])
def test_broadcasts_at_every_rank(shape):
    """Angles broadcast at any rank, including 0-d, which upstream rejects."""
    theta = jnp.full(shape, 0.7)
    uvec = jnp.broadcast_to(jnp.asarray([0.0, 0.6, 0.8]), (*shape, 3))
    assert sp.sph_legendre_p(3, 2, theta).shape == shape
    assert sp.sph_harm_y(3, 2, theta, 0.3).shape == shape
    assert sp.sph_harm_y_cart(3, 2, uvec).shape == shape
    assert sp.sph_harm_y_cart_all(3, 2, uvec).shape == (4, 5, *shape)


def test_negative_order_identities():
    r"""Negative orders follow SciPy's two reflection identities.

    :math:`\bar{P}_n^{-m} = (-1)^m \bar{P}_n^m` for the Legendre function, and
    :math:`Y_n^{-m} = (-1)^m \overline{Y_n^m}` for the harmonic -- the latter
    carrying the conjugation of the azimuth as well as the sign.
    """
    theta, phi = jnp.asarray(THETA), jnp.asarray(PHI)
    for n in range(5):
        for m in range(1, n + 1):
            np.testing.assert_allclose(
                np.asarray(sp.sph_legendre_p(n, -m, theta)),
                (-1.0) ** m * np.asarray(sp.sph_legendre_p(n, m, theta)),
                rtol=1e-13,
            )
            np.testing.assert_allclose(
                np.asarray(sp.sph_harm_y(n, -m, theta, phi)),
                (-1.0) ** m * np.conj(np.asarray(sp.sph_harm_y(n, m, theta, phi))),
                rtol=1e-13,
                atol=1e-15,
            )


def test_orthonormality_over_the_sphere():
    r""":math:`\int Y_l^m \overline{Y_{l'}^{m'}} d\Omega = \delta`.

    Gauss-Legendre in :math:`\cos\theta` and the trapezium rule in
    :math:`\phi`, which is spectrally accurate for a periodic integrand. This
    is the one check that does not go through SciPy at all -- it tests the
    normalization against its own definition.
    """
    n_max = 4
    u, w = np.polynomial.legendre.leggauss(2 * n_max + 4)
    n_phi = 4 * n_max + 4
    phi = np.arange(n_phi) * 2 * np.pi / n_phi
    theta = np.arccos(u)
    tt, pp = np.meshgrid(theta, phi, indexing="ij")
    weight = np.broadcast_to(w[:, None], tt.shape) * (2 * np.pi / n_phi)

    pairs = [(deg, k) for deg in range(n_max + 1) for k in range(-deg, deg + 1)]
    values = {
        (deg, k): np.asarray(sp.sph_harm_y(deg, k, jnp.asarray(tt), jnp.asarray(pp)))
        for deg, k in pairs
    }
    for left in pairs:
        for right in pairs:
            integral = np.sum(values[left] * np.conj(values[right]) * weight)
            expect = 1.0 if left == right else 0.0
            np.testing.assert_allclose(integral.real, expect, atol=1e-12)
            np.testing.assert_allclose(integral.imag, 0.0, atol=1e-12)


# ============================================================================
# Static-argument validation


@pytest.mark.parametrize(("n", "m"), [(-1, 0), (2, 3), (2, -3), (0, 1)])
def test_invalid_degree_order_raises(n, m):
    """``|m| > n`` is rejected eagerly, not returned as `nan`.

    Both are static Python `int`s, which `spexial`'s conventions say to
    validate: nothing here is traced. Silently returning the seed -- what an
    empty recurrence gives -- would be a wrong number, not a missing one.
    """
    with pytest.raises(ValueError, match="0 <= "):
        sp.sph_legendre_p(n, m, 0.5)
    with pytest.raises(ValueError, match="0 <= "):
        sp.sph_harm_y(n, m, 0.5, 0.5)


@pytest.mark.parametrize("func", ["sph_legendre_p", "sph_harm_y"])
def test_traced_degree_is_rejected(func):
    """A traced degree is refused, since it cannot be static.

    Three mechanisms can fire depending on the environment -- `jax.jit`
    complains that a static argument is unhashable, the runtime type checker
    rejects it against `int`, or the eager guard raises. Any refusal will do;
    what must not happen is a value silently baked in.
    """
    args = (
        (jnp.asarray(2), 1, 0.5)
        if func == "sph_legendre_p"
        else (jnp.asarray(2), 1, 0.5, 0.5)
    )
    with pytest.raises((TypeError, ValueError, TypeCheckError)):
        getattr(sp, func)(*args)


def test_uvec_needs_a_trailing_axis_of_three():
    """A direction that is not 3-dimensional is a shape error, stated as one."""
    with pytest.raises(ValueError, match="trailing axis of length 3"):
        sp.sph_harm_y_cart(1, 0, jnp.asarray([0.0, 1.0]))


# ============================================================================
# JAX transformations


@pytest.mark.parametrize(("n", "m"), [(0, 0), (2, 1), (4, -3)])
def test_jit_and_vmap_compose(n, m):
    """`jit` and `vmap` both work, and agree with the unwrapped call."""
    theta, phi = jnp.asarray(THETA), jnp.asarray(PHI)
    direct = sp.sph_harm_y(n, m, theta, phi)
    jitted = jax.jit(lambda t, p: sp.sph_harm_y(n, m, t, p))(theta, phi)
    mapped = jax.vmap(lambda t, p: sp.sph_harm_y(n, m, t, p))(theta, phi)
    np.testing.assert_allclose(np.asarray(jitted), np.asarray(direct), rtol=1e-14)
    np.testing.assert_allclose(np.asarray(mapped), np.asarray(direct), rtol=1e-14)


@pytest.mark.parametrize(("n", "m"), [(1, 0), (2, 1), (3, -2)])
def test_forward_and_reverse_gradients_agree(n, m):
    """`jacfwd` and `jacrev` give the same Cartesian gradient, off the axis.

    The two modes disagreeing is the signature of a `where` transposing a zero
    cotangent into an ``inf`` branch, which is how the same family of bugs
    showed up in `eval_gegenbauer`.
    """
    v = jnp.asarray([0.3, -0.5, 0.8]) / jnp.linalg.norm(jnp.asarray([0.3, -0.5, 0.8]))
    f = lambda u: sp.sph_harm_y_cart(n, m, u).real
    np.testing.assert_allclose(
        np.asarray(jax.jacfwd(f)(v)), np.asarray(jax.jacrev(f)(v)), rtol=1e-12
    )
