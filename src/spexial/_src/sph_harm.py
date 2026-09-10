r"""Spherical harmonics and the spherical Legendre functions beneath them.

Note that this module is NOT public API; only the names re-exported from the
top-level `spexial` namespace are.

Everything here is built on one quantity, the *reduced* spherical Legendre
function

.. math::

    q_l^m(u) = N_{lm}\, p_l^m(u),
    \qquad p_l^m(u) = \frac{P_l^m(u)}{(1 - u^2)^{m/2}},
    \qquad N_{lm} = \sqrt{\frac{2l+1}{4\pi}\frac{(l-m)!}{(l+m)!}}

which is a *polynomial* in ``u``. The factor of :math:`(1-u^2)^{m/2}` that
`sph_legendre_p` carries and :math:`q_l^m` does not is where every one of this
module's reasons for existing lives: it is :math:`\sin^m\theta`, it is the only
non-smooth part of the harmonic, and both upstream defects below are a
derivative of it taken in the wrong variable.

Why not `jax.scipy.special.sph_harm_y`
--------------------------------------

Two measured defects, both present in JAX 0.11.1.

**It pairs its arguments element-wise rather than broadcasting.**
``sph_harm_y(n, m, theta, phi, n_max=...)`` indexes its internal Legendre table
with ``jnp.arange(len(n))``, so ``n[i]`` is matched to ``theta[i]``. A length-1
``n``/``m`` against a length-N ``theta`` is therefore correct at index 0 and
silently wrong everywhere else -- measured against `scipy.special.lpmv` over
four positions, up to **1.18 absolute** for :math:`l \le 3`. ``len()`` also
rejects 0-d input outright. Here ``n`` and ``m`` are static Python `int`s, as
they are for `eval_gegenbauer`, so there is no pairing to get wrong and
``theta``/``phi`` broadcast at any rank including 0.

**Every derivative is `nan` at both poles for** :math:`l \ge 1`. Upstream
differentiates :math:`\sqrt{1 - \cos^2\theta}`, whose slope is infinite at
:math:`\theta = 0, \pi`, and the surviving :math:`0 \cdot \infty` is `nan` --
even though :math:`\partial_\theta Y_l^m` is perfectly finite there. Building
:math:`\sin^m\theta` as an integer power of :math:`\sin\theta` instead never
forms the square root, and the derivative comes out exact. (The
:math:`l = m = 0` case, where the function is constant, was `nan` upstream too
and has since been fixed; :math:`l \ge 1` has not.)

The z-axis, and why there is a Cartesian form
---------------------------------------------

A third problem is not upstream's fault and cannot be fixed in
:math:`(\theta, \phi)` at all. The *Cartesian* gradient of any :math:`m \ge 1`
harmonic is routed by the chain rule through :math:`\theta` and :math:`\phi`,
neither of which has a directional derivative on the z-axis, so it evaluates to
exactly ``0.0`` there against a non-zero true limit.

`sph_harm_y_cart` sidesteps it by never introducing the angles:

.. math::

    \sin^m\theta \, e^{im\phi} = \left(\frac{x + iy}{r}\right)^m
    \quad\Longrightarrow\quad
    Y_l^m = q_l^m(z/r) \left(\frac{x + iy}{r}\right)^m

whose right-hand side is polynomial in :math:`x` and :math:`y` and therefore
smooth on the axis.

"""

__all__ = [
    "sph_harm_y",
    "sph_harm_y_cart",
    "sph_harm_y_cart_all",
    "sph_legendre_p",
]

import math
from functools import partial

import jax
import jax.numpy as jnp
from jax import lax

from .custom_types import AnyArray, ComplexArray, RealArrayLike
from .dtype import as_float


def _seed(m: int, u: AnyArray, /) -> AnyArray:
    r"""Start the reduced recurrence at :math:`l = m` with :math:`q_m^m`.

    :math:`p_m^m = (-1)^m (2m-1)!!`, which overflows float64 near
    :math:`m = 90`; the normalized :math:`q_m^m = N_{mm} p_m^m` is O(1) at every
    order. Multiplying the two afterwards would materialize the overflow, so the
    product is formed in log space and only then exponentiated. That is the
    whole reason the normalization is folded into the recurrence rather than
    applied at the end -- and it also avoids losing roughly two digits to
    cancellation at moderate ``m``.

    Both factors are Python floats -- ``m`` is static -- so this is arithmetic
    the tracer never sees, and the array it returns is a constant.
    """
    log_seed = (
        0.5 * math.log((2 * m + 1) / (4 * math.pi))
        + 0.5 * math.lgamma(2 * m + 1)
        - m * math.log(2)
        - math.lgamma(m + 1)
    )
    return jnp.full_like(u, (-1.0) ** m * math.exp(log_seed))


def _step(deg: int, m: int, /) -> tuple[float, float]:
    r"""Give the ``(a, b)`` of one degree step of the reduced recurrence.

    The step is :math:`q_l = a(u q_{l-1} - b q_{l-2})`, with ``deg`` the ``l``
    being computed. ``b`` is zero for the first step off the seed, where there
    is no :math:`q_{l-2}` to subtract and its formula would divide by zero.
    """
    a = math.sqrt((4 * deg * deg - 1) / (deg * deg - m * m))
    b = (
        math.sqrt(((deg - 1) ** 2 - m * m) / (4 * (deg - 1) ** 2 - 1))
        if deg >= 2
        else 0.0
    )
    return a, b


def _reduced(n: int, m: int, u: AnyArray, /) -> AnyArray:
    r""":math:`q_n^m(u) = N_{nm} P_n^m(u) / (1-u^2)^{m/2}`, for :math:`0 \le m \le n`.

    ``n`` and ``m`` are static, so the loop unrolls at trace time and the
    recurrence costs no traced control flow.
    """
    q_prev, q_cur = jnp.zeros_like(u), _seed(m, u)
    for deg in range(m + 1, n + 1):
        a, b = _step(deg, m)
        q_prev, q_cur = q_cur, a * (u * q_cur - b * q_prev)
    return q_cur


def _check_degree_order(n: int, m: int, /) -> None:
    r"""Reject a degree/order pair outside :math:`\lvert m \rvert \le n`, eagerly.

    Both are static Python `int`s, so this is the case `spexial`'s conventions
    say to validate rather than return `nan` for: nothing here is traced, and a
    ``|m| > n`` would otherwise produce a silently empty recurrence returning
    the seed instead of the zero that harmonic actually is.
    """
    if not isinstance(n, int) or not isinstance(m, int):
        msg = (
            "n and m must be Python ints (they are static under jit), not "
            f"{type(n).__name__} and {type(m).__name__}"
        )
        raise TypeError(msg)
    if n < 0 or abs(m) > n:
        msg = f"require 0 <= |m| <= n, got n={n}, m={m}"
        raise ValueError(msg)


def _uvec_components(uvec: RealArrayLike, /) -> tuple[AnyArray, AnyArray, AnyArray]:
    """Split a ``(..., 3)`` direction into its three components."""
    u = as_float(uvec)
    if u.shape[-1] != 3:
        msg = f"uvec must have a trailing axis of length 3, got shape {u.shape}"
        raise ValueError(msg)
    return u[..., 0], u[..., 1], u[..., 2]


@partial(jax.jit, static_argnums=(0, 1))
def sph_legendre_p(n: int, m: int, theta: RealArrayLike, /) -> AnyArray:
    r"""Spherical Legendre function :math:`\bar{P}_n^m(\cos\theta)`.

    .. math::

        \bar{P}_n^m(\cos\theta) = \sqrt{\frac{2n+1}{4\pi}
            \frac{(n-m)!}{(n+m)!}}\; P_n^m(\cos\theta)

    the normalization for which :math:`Y_n^m(\theta,\phi) =
    \bar{P}_n^m(\cos\theta)\,e^{im\phi}`. Includes the Condon-Shortley phase,
    matching `scipy.special.sph_legendre_p`, `scipy.special.lpmv` and GSL's
    ``gsl_sf_legendre_sphPlm``.

    `jax.scipy.special` has no counterpart at any version -- the normalized
    Legendre function is reachable only through `jax.scipy.special.sph_harm_y`,
    and then only as part of a complex harmonic.

    Parameters
    ----------
    n, m
        Degree and order, with ``abs(m) <= n``. Static Python `int`s, as for
        `eval_gegenbauer`; unlike `scipy.special.sph_legendre_p` they may not
        be arrays, and the returned shape is therefore ``theta``'s own rather
        than a broadcast against them.
    theta
        Polar angle in radians, of any shape.

    Returns
    -------
    Array
        Shaped like ``theta``.

    See Also
    --------
    sph_harm_y : the full complex harmonic.
    scipy.special.sph_legendre_p : the SciPy counterpart.

    Examples
    --------
    >>> import jax.numpy as jnp
    >>> import spexial as sp

    :math:`\bar{P}_0^0 = 1/\sqrt{4\pi}` everywhere:

    >>> round(float(sp.sph_legendre_p(0, 0, 0.7)), 10)
    0.2820947918

    Negative orders follow SciPy,
    :math:`\bar{P}_n^{-m} = (-1)^m \bar{P}_n^{m}`:

    >>> bool(jnp.isclose(sp.sph_legendre_p(2, -1, 0.3), -sp.sph_legendre_p(2, 1, 0.3)))
    True

    Unlike upstream's, the derivative is finite at the pole:

    >>> import jax
    >>> float(jax.grad(lambda t: sp.sph_legendre_p(1, 0, t))(0.0))
    -0.0

    """
    _check_degree_order(n, m)
    theta_arr = as_float(theta)
    order = abs(m)
    # `sin(theta)**order`, an *integer* power, is the whole point: upstream
    # forms `sqrt(1 - cos(theta)**2)`, whose infinite slope at the poles is what
    # makes every derivative there `nan`. `order` is static, so this compiles to
    # `lax.integer_pow`, whose derivative at zero is exact.
    value = _reduced(n, order, jnp.cos(theta_arr)) * jnp.sin(theta_arr) ** order
    # P_n^{-m} = (-1)^m (n-m)!/(n+m)! P_n^m, and the two factorial ratios in the
    # spherical normalization cancel it down to the sign alone.
    return value if m >= 0 else (-1.0) ** order * value


@partial(jax.jit, static_argnums=(0, 1))
def sph_harm_y(
    n: int, m: int, theta: RealArrayLike, phi: RealArrayLike, /
) -> ComplexArray:
    r"""Spherical harmonic :math:`Y_n^m(\theta, \phi)`.

    .. math::

        Y_n^m(\theta, \phi) = \bar{P}_n^m(\cos\theta)\, e^{im\phi}

    with `sph_legendre_p` supplying :math:`\bar{P}_n^m`. Same convention as
    `scipy.special.sph_harm_y` and `jax.scipy.special.sph_harm_y`: ``theta`` is
    the polar angle and ``phi`` the azimuth, and the Condon-Shortley phase is
    included.

    This differs from `jax.scipy.special.sph_harm_y` in two ways that are the
    reason it exists, both measured on JAX 0.11.1. Upstream pairs ``n[i]`` with
    ``theta[i]`` positionally instead of broadcasting, so a scalar degree
    against a batch of angles is right only at index 0 -- up to 1.18 absolute
    error for :math:`n \le 3` -- and it rejects 0-d input. And upstream's
    derivatives are `nan` at :math:`\theta = 0, \pi` for every :math:`n \ge 1`.
    Here ``n`` and ``m`` are static, so there is nothing to mispair, and
    ``theta``/``phi`` broadcast against each other at any rank.

    .. note::

        The **Cartesian** gradient of an :math:`m \ge 1` harmonic is still
        exactly zero on the z-axis if it is obtained by differentiating through
        :math:`\theta` and :math:`\phi`, which have no directional derivative
        there. That is a property of the coordinates, not of any
        implementation. Use `sph_harm_y_cart` when the gradient on the axis
        matters.

    Parameters
    ----------
    n, m
        Degree and order, with ``abs(m) <= n``. Static Python `int`s; unlike
        `scipy.special.sph_harm_y` they may not be arrays.
    theta
        Polar angle in radians.
    phi
        Azimuthal angle in radians. Broadcast against ``theta``.

    Returns
    -------
    Array
        Complex, of the broadcast shape of ``theta`` and ``phi``.

    See Also
    --------
    sph_harm_y_cart : the same harmonic from a Cartesian direction.
    scipy.special.sph_harm_y : the SciPy counterpart.

    Examples
    --------
    >>> import jax.numpy as jnp
    >>> import spexial as sp

    >>> complex(sp.sph_harm_y(0, 0, 0.4, 1.2)).real
    0.28209479177387814

    It broadcasts, which upstream does not:

    >>> theta = jnp.asarray([0.3, 1.1, 2.0])
    >>> sp.sph_harm_y(2, 1, theta, 0.0).real.round(6).tolist()
    [-0.218107, -0.312301, 0.292333]

    """
    _check_degree_order(n, m)
    theta_arr, phi_arr = as_float(theta), as_float(phi)
    return sph_legendre_p(n, m, theta_arr) * jnp.exp(1j * m * phi_arr)


def _azimuth_power(m: int, ux: AnyArray, uy: AnyArray, /) -> tuple[AnyArray, AnyArray]:
    r"""Real and imaginary parts of :math:`((x + iy)/r)^m`, by repeated multiply.

    ``m`` is static, so this unrolls. Formed as a real pair rather than as a
    complex power because `jax.numpy.power` of a complex number routes through
    ``exp(m log z)``, and ``log(0)`` on the z-axis is exactly where the answer
    has to stay finite.
    """
    cos_mphi, sin_mphi = jnp.ones_like(ux), jnp.zeros_like(ux)
    for _ in range(m):
        cos_mphi, sin_mphi = (
            cos_mphi * ux - sin_mphi * uy,
            cos_mphi * uy + sin_mphi * ux,
        )
    return cos_mphi, sin_mphi


def _conjugate_for_negative_m(
    m: int, real: AnyArray, imag: AnyArray, /
) -> tuple[AnyArray, AnyArray]:
    r""":math:`Y_n^{-m} = (-1)^m \overline{Y_n^m}`, matching SciPy.

    Applied to the *harmonic*, not to the Legendre factor, so it carries the
    conjugation of the azimuth as well as the sign.
    """
    if m >= 0:
        return real, imag
    sign = (-1.0) ** abs(m)
    return sign * real, -sign * imag


@partial(jax.jit, static_argnums=(0, 1))
def sph_harm_y_cart(n: int, m: int, uvec: RealArrayLike, /) -> ComplexArray:
    r"""Spherical harmonic :math:`Y_n^m` from a Cartesian **unit** direction.

    .. math::

        Y_n^m = q_n^m(\hat{z}) \left(\hat{x} + i\hat{y}\right)^m,
        \qquad q_n^m(u) = N_{nm} \frac{P_n^m(u)}{(1-u^2)^{m/2}}

    Mathematically identical to ``sph_harm_y(n, m, arccos(uz), arctan2(uy,
    ux))``, and numerically better behaved in the one place it matters: the
    right-hand side is a *polynomial* in :math:`\hat{x}` and :math:`\hat{y}`,
    so it is smooth on the z-axis, where :math:`\theta` and :math:`\phi` are
    singular.

    That singularity is not cosmetic. Differentiating the :math:`(\theta,
    \phi)` form through the chain rule gives a Cartesian gradient of **exactly
    zero** for every :math:`m \ge 1` term on the axis -- neither angle has a
    directional derivative there -- against a non-zero true limit. There is no
    counterpart to this function in SciPy or JAX; the name is
    `scipy.special.sph_harm_y` plus the argument convention that distinguishes
    it.

    Parameters
    ----------
    n, m
        Degree and order, with ``abs(m) <= n``. Static Python `int`s.
    uvec
        Cartesian direction, shape ``(..., 3)``. Assumed **already normalized**;
        this deliberately does not normalize, both to avoid repeating a caller's
        own work and because the caller owns the policy at :math:`r = 0`. A zero
        vector is well defined and finite here -- it gives :math:`Y_0^0 =
        N_{00}` and zero for every :math:`m \ge 1` -- which normalizing
        internally would turn into ``nan``.

    Returns
    -------
    Array
        Complex, shaped like ``uvec`` without its trailing axis.

    See Also
    --------
    sph_harm_y : the same harmonic in spherical coordinates.
    sph_harm_y_cart_all : every ``(l, m)`` up to a maximum degree, in one sweep.

    Examples
    --------
    >>> import jax, jax.numpy as jnp
    >>> import spexial as sp

    >>> uvec = jnp.asarray([0.0, 0.0, 1.0])
    >>> round(complex(sp.sph_harm_y_cart(1, 0, uvec)).real, 10)
    0.4886025119

    On the z-axis the gradient of an ``m = 1`` term is finite and non-zero,
    where the spherical form gives exactly zero:

    >>> g = jax.grad(lambda v: sp.sph_harm_y_cart(1, 1, v).real)(uvec)
    >>> round(float(g[0]), 6)
    -0.345494

    """
    _check_degree_order(n, m)
    ux, uy, uz = _uvec_components(uvec)
    order = abs(m)
    q = _reduced(n, order, uz)
    cos_mphi, sin_mphi = _azimuth_power(order, ux, uy)
    real, imag = _conjugate_for_negative_m(m, q * cos_mphi, q * sin_mphi)
    return lax.complex(real, imag)


@partial(jax.jit, static_argnums=(0, 1))
def sph_harm_y_cart_all(n: int, m: int, uvec: RealArrayLike, /) -> ComplexArray:
    r"""Every :math:`Y_l^k` with :math:`l \le n` and :math:`\lvert k \rvert \le m`.

    The same values `sph_harm_y_cart` returns pair by pair, but with both
    recurrences carried *across* the table instead of restarted for each entry:
    one pass per order advances :math:`((x+iy)/r)^k` by a single complex
    multiply and walks the Legendre recurrence up in :math:`l` from its seed at
    :math:`l = k`. That makes the table :math:`O(n^2)` rather than cubic.

    Since the degree and order are static the saving is in *traced* operations,
    so it shows up as a smaller HLO -- and hence faster tracing and compilation
    -- rather than as faster execution, which XLA's fusion had already largely
    recovered. Measured on the equivalent code in ``galax``, tracing was 3-4.6x
    faster for :math:`n \le 20` with run time flat.

    Layout and argument order follow `scipy.special.sph_harm_y_all`, which has
    no JAX counterpart; the difference is only that the direction is Cartesian.

    .. warning::

        Reach for this when you want the **table**. If instead you are about
        to index it and reduce -- summing :math:`\\sum_{lm} c_{lm} Y_l^m`,
        say -- call `sph_harm_y_cart` per pair and fold each term into the sum
        as it is produced. Indexing a stacked table defeats XLA's fusion, so
        the whole thing is materialized: measured on a multipole expansion at
        :math:`n = 12` over a million directions, the table form ran in 17.7 s
        against 10 ms for per-pair calls, for identical values. The saving
        here is in *traced* operations -- roughly 2.3x less tracing and
        compiling at :math:`n = 20` -- which is worth having only when the
        table itself is the thing you need.

    Parameters
    ----------
    n
        Maximum degree. Static: it sets the first axis.
    m
        Maximum order, ``0 <= m <= n``. Static: it sets the second axis.
    uvec
        Cartesian direction, shape ``(..., 3)``, assumed normalized. See
        `sph_harm_y_cart` on why this does not normalize.

    Returns
    -------
    Array
        Complex, shape ``(n + 1, 2 * m + 1, ...)``, where entry ``[i, j]`` is
        :math:`Y_i^j` for :math:`0 \le i \le n` and :math:`-m \le j \le m` --
        so negative orders live at the *end* of the second axis, reachable by
        ordinary negative indexing, exactly as in SciPy. Entries with
        :math:`\lvert j \rvert > i` are zero, since no such harmonic exists.

    Examples
    --------
    >>> import jax.numpy as jnp
    >>> import spexial as sp

    >>> uz = jnp.asarray([0.0, 0.0, 1.0])
    >>> Y = sp.sph_harm_y_cart_all(2, 2, uz)
    >>> Y.shape
    (3, 5)

    Entries with ``|j| > i`` are zero, since no such harmonic exists:

    >>> bool((Y[0, 1] == 0) & (Y[1, 2] == 0))
    True

    Each entry matches the one-pair function, negative orders included:

    >>> bool(jnp.isclose(Y[2, -1], sp.sph_harm_y_cart(2, -1, uz)))
    True

    """
    _check_degree_order(n, m)
    if m < 0:
        msg = f"require m >= 0 for the table's maximum order, got {m}"
        raise ValueError(msg)
    ux, uy, uz = _uvec_components(uvec)

    shape = (n + 1, 2 * m + 1, *jnp.shape(uz))
    real = jnp.zeros(shape, dtype=uz.dtype)
    imag = jnp.zeros(shape, dtype=uz.dtype)

    cos_kphi, sin_kphi = jnp.ones_like(ux), jnp.zeros_like(ux)
    for k in range(m + 1):
        if k > 0:  # advance ((x + iy)/r)^k by one complex multiply
            cos_kphi, sin_kphi = (
                cos_kphi * ux - sin_kphi * uy,
                cos_kphi * uy + sin_kphi * ux,
            )
        q_prev, q_cur = jnp.zeros_like(uz), _seed(k, uz)
        for deg in range(k, n + 1):
            if deg > k:
                a, b = _step(deg, k)
                q_prev, q_cur = q_cur, a * (uz * q_cur - b * q_prev)
            re, im = q_cur * cos_kphi, q_cur * sin_kphi
            real = real.at[deg, k].set(re)
            imag = imag.at[deg, k].set(im)
            if k > 0:  # the -k column, at the far end of the axis as in SciPy
                neg_re, neg_im = _conjugate_for_negative_m(-k, re, im)
                real = real.at[deg, -k].set(neg_re)
                imag = imag.at[deg, -k].set(neg_im)
    return lax.complex(real, imag)
