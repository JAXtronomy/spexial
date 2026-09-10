r"""The incomplete beta function.

Note that this module is NOT public API; only the name re-exported from the
top-level `spexial` namespace is.

The *unregularized* incomplete beta function

.. math::

    B(a, b, z) = \int_0^z t^{a-1}(1-t)^{b-1}\,\mathrm{d}t

(DLMF 8.17.1). SciPy and JAX both provide only the **regularized** one,
:math:`I_z(a, b)`, under the name ``betainc``; the two differ by a factor of
the complete beta function :math:`B(a, b)`.

Why not just multiply
---------------------

Because ``beta(a, b) * betainc(a, b, z)`` is `nan` for every :math:`b \le 0`.
:math:`B(a, b)` has a pole there while the *product* is perfectly finite, so
the reconstruction loses a domain the function itself has no trouble with --
and :math:`b \le 0` is not exotic, it is an ordinary slope in the double
power-law density profiles this was written for.

>>> import jax.numpy as jnp
>>> import jax.scipy.special as jsp
>>> bool(jnp.isnan(jsp.beta(2.0, 0.0) * jsp.betainc(2.0, 0.0, 0.5)))
True

`jax.scipy.special.hyp2f1` *can* express it for any ``b`` (DLMF 8.17.7), but it
is a `jax.lax.while_loop` whose trip count depends on its data: under
`jax.vmap` every lane pays the worst lane's iteration count, and its derivative
runs a second such loop. What is here instead is two fixed-length,
geometrically convergent series with an exact O(1) derivative rule.

Contributed from `galax <https://github.com/GalacticDynamics/galax>`_, where it
was written for the Zhao (1996) family of density profiles (Eq. 43).

"""

__all__ = ["incomplete_beta"]

import functools as ft

import jax
import jax.numpy as jnp
from jax.custom_derivatives import SymbolicZero

from .custom_types import AnyArray, AnyArrayLike, ScalarLike

_NTERMS = 64
"""Series length. Both series below converge like 2^-k, so this is ~1e-19.

Not worth trimming: measured over a in [0.2, 8], b in [-2.5, 6] and z up to
1 - 1e-8, dropping to 48 terms costs an order of magnitude of accuracy at
moderate `a` (5e-14 -> 6e-13) and 40 breaks the 1e-11 the tests assert, to
save a fraction of a loop that `_UNROLL` already cut four-fold. Above
a ~ 16 the error stops improving with term count at all -- it is cancellation
between large alternating terms, not truncation -- so more terms would not
help there either.
"""

_UNROLL = 16
"""How far to unroll the series loops.

Worth 4x: at 1e5 points the small-z series goes 10.0 ms -> 2.5 ms, for +0.1 s
of compile time. Full unrolling (64) buys a further 15% for 2x the compile,
and a trace-time Python loop is no faster than this while compiling worse.
"""

_POLE_BAND = 1e-3
"""Half-width of the band around `b + m == 0` where `_large_z` expands.

Below this the direct form cancels; above it the expansion's O(s^4 L^5)
truncation shows, for `L = log(1-z)`.
"""


def _small_z(a: ScalarLike, b: ScalarLike, z: AnyArrayLike) -> AnyArray:
    r"""$B(a, b, z)$ for $z \leq 1/2$, by the defining Taylor series.

    Expanding $(1-t)^{b-1}$ binomially and integrating term by term,

    .. math::

        B(a, b, z) = z^a \sum_{k=0}^\infty \frac{(1-b)_k}{k!\,(a+k)} z^k

    The terms fall off like $z^k \leq 2^{-k}$, hence the fixed term count.

    Summed into a `jax.lax.scan` carry: the terms are batched over `z`, so
    materializing them all at once would cost a ``(*batch, 64)`` temporary and
    make this memory- rather than flop-bound.

    `a`, `b` and `z` are closed over rather than carried. Threading them
    through the carry (or `xs`) instead measures the same to within noise and
    returns bit-identical values -- JAX turns a closed-over tracer into a
    constant of the scan's jaxpr, so there is nothing there to hoist.
    """

    def step(
        carry: tuple[AnyArray, AnyArray, ScalarLike], k: ScalarLike
    ) -> tuple[tuple[AnyArray, AnyArray, ScalarLike], None]:
        total, z_pow, coeff = carry  # coeff = (1-b)_k / k!
        total = total + coeff / (a + k) * z_pow
        return (total, z_pow * z, coeff * (k + 1.0 - b) / (k + 1.0)), None

    init = (jnp.zeros_like(z), jnp.ones_like(z), jnp.ones_like(a))
    (total, _, _), _ = jax.lax.scan(step, init, jnp.arange(_NTERMS), unroll=_UNROLL)
    return z**a * total  # type: ignore[no-any-return]


def _large_z(a: ScalarLike, b: ScalarLike, z: AnyArrayLike) -> AnyArray:
    r"""$B(a, b, z)$ for $z > 1/2$, by reflecting about $t = 1/2$.

    Substituting $t = 1-u$ and splitting the range at $u = 1/2$,

    .. math::

        B(a, b, z) = \int_{w}^{1} u^{b-1}(1-u)^{a-1} du
                   = B(a, b, 1/2) + \int_w^{1/2} u^{b-1}(1-u)^{a-1} du

    with $w = 1 - z$. Expanding $(1-u)^{a-1}$ binomially (legitimate since
    $u \leq 1/2$ on the remaining range) and integrating term by term,

    .. math::

        B(a, b, z) = B(a, b, 1/2)
            + \sum_{m=0}^\infty \frac{(1-a)_m}{m!}
              \frac{(1/2)^{b+m} - w^{b+m}}{b+m}

    where the $b + m \to 0$ term is $\ln(1/(2w))$. Both pieces converge like
    $2^{-m}$.

    This branch is what makes $b \leq 0$ work: it never forms the complete beta
    function $B(a, b)$, which is what diverges there, while keeping the genuine
    $z \to 1$ divergence of $B(a, b, z)$ itself exact (as $w^b$, or $-\ln w$
    when $b = 0$).

    Summed into a `jax.lax.scan` carry, for the same reason as `_small_z`.

    Each term is $(e^{sL_1} - e^{sL_2})/s$ with $s = b + m$, $L_1 = \ln(1/2)$
    and $L_2 = \ln w$, which the running powers below evaluate directly. That
    cancels catastrophically as $s \to 0$, so within `_POLE_BAND` of zero it
    switches to the expansion of the same expression,

    .. math::

        \frac{e^{sL_1} - e^{sL_2}}{s}
            = \sum_{k \geq 1} \frac{L_1^k - L_2^k}{k!} s^{k-1}

    whose $L$-powers are loop-invariant, so this costs a few extra FMAs rather
    than transcendentals. Carrying the $O(s)$ term also makes the
    $b$-derivative right *at* $s = 0$, which integer slopes do hit (e.g.
    $\gamma = 2$ in Zhao's Eq. 7).
    """
    w = 1.0 - z
    log_half, log_w = jnp.log(0.5), jnp.log(w)
    # L1^k - L2^k over k!, for the near-pole expansion.
    d1 = log_half - log_w
    d2 = (log_half**2 - log_w**2) / 2
    d3 = (log_half**3 - log_w**3) / 6
    d4 = (log_half**4 - log_w**4) / 24

    def step(
        carry: tuple[AnyArray, AnyArray, ScalarLike, ScalarLike], m: ScalarLike
    ) -> tuple[tuple[AnyArray, AnyArray, ScalarLike, ScalarLike], None]:
        total, w_pow, half_pow, coeff = carry  # coeff = (1-a)_m / m!
        s = b + m
        near_pole = jnp.abs(s) < _POLE_BAND
        s_safe = jnp.where(near_pole, 1.0, s)
        term = jnp.where(
            near_pole,
            d1 + s * (d2 + s * (d3 + s * d4)),
            (half_pow - w_pow) / s_safe,
        )
        total = total + coeff * term
        carry = (total, w_pow * w, half_pow * 0.5, coeff * (m + 1.0 - a) / (m + 1.0))
        return carry, None

    # `_small_z(a, b, 1/2)` below looks like a scalar being recomputed per
    # point, but computing it as a scalar and broadcasting is *slower*: XLA
    # already folds the constant-array input, and doing it by hand breaks the
    # fusion (measured 0.80x).
    init = (jnp.zeros_like(w), w**b, 0.5**b, jnp.ones_like(a))
    (total, _, _, _), _ = jax.lax.scan(step, init, jnp.arange(_NTERMS), unroll=_UNROLL)
    return _small_z(a, b, jnp.full_like(w, 0.5)) + total  # type: ignore[no-any-return]


def _incomplete_beta_impl(a: ScalarLike, b: ScalarLike, z: AnyArrayLike) -> AnyArray:
    r"""$B(a, b, z)$ for $a > 0$, any real $b$, and $z \in [0, 1]$."""
    z = jnp.asarray(z)
    # Both branches are evaluated, so clamp each one's input to the range where
    # it is well behaved; `where` then discards the unused value.
    return jnp.where(  # type: ignore[no-any-return]
        z <= 0.5,
        _small_z(a, b, jnp.minimum(z, 0.5)),
        _large_z(a, b, jnp.maximum(z, 0.5)),
    )


@jax.custom_jvp
def incomplete_beta(a: ScalarLike, b: ScalarLike, z: AnyArrayLike, /) -> AnyArray:
    r"""Unregularized incomplete beta function :math:`B(a, b, z)`.

    .. math::

        B(a, b, z) = \int_0^z t^{a-1}(1-t)^{b-1}\,\mathrm{d}t

    the DLMF 8.17.1 form. There is no `scipy.special` counterpart: SciPy's
    ``betainc`` is the *regularized* :math:`I_z(a, b)`, and the two differ by
    the complete beta function :math:`B(a, b)`.

    Reconstructing this as ``beta(a, b) * betainc(a, b, z)`` works only for
    :math:`b > 0`. At :math:`b \le 0` the complete beta function has a pole
    while the product does not, so that route returns `nan` over a domain this
    function handles without difficulty.

    Parameters
    ----------
    a
        First parameter. Must be positive, and scalar.
    b
        Second parameter. **Any real value**, including zero and negative --
        which is the reason this function exists. Scalar.
    z
        Upper limit of integration, in :math:`[0, 1]`. Any shape.

    Returns
    -------
    Array
        Shaped like ``z``.

    See Also
    --------
    scipy.special.betainc : the regularized form, :math:`I_z(a, b)`.
    jax.scipy.special.hyp2f1 : expresses this via DLMF 8.17.7, at the cost of
        a data-dependent `jax.lax.while_loop`.

    Notes
    -----
    Two fixed-length series, switched at :math:`z = 1/2`, each converging like
    :math:`2^{-k}`; see `_small_z` and `_large_z`. The
    :math:`z`-derivative is supplied by a `jax.custom_jvp` and is exact and
    O(1) -- by Leibniz it is just the integrand at the endpoint,
    :math:`z^{a-1}(1-z)^{b-1}` -- rather than differentiating through 64 terms.
    It is a `custom_jvp` rather than a `custom_vjp` so that `jax.hessian`'s
    ``jacfwd(jacrev(...))`` still composes.

    Examples
    --------
    >>> import jax.numpy as jnp
    >>> import jax.scipy.special as jsp
    >>> import spexial as sp

    It agrees with the regularized form wherever that is defined, including
    close to the :math:`z \to 1` endpoint:

    >>> a, b = 2.0, 1.5
    >>> z = jnp.asarray([0.3, 0.999])
    >>> bool(
    ...     jnp.allclose(
    ...         sp.incomplete_beta(a, b, z), jsp.beta(a, b) * jsp.betainc(a, b, z)
    ...     )
    ... )
    True

    But unlike that product it stays finite for ``b <= 0``:

    >>> round(float(sp.incomplete_beta(2.0, 0.0, jnp.asarray(0.5))), 8)
    0.19314718

    >>> bool(jnp.isnan(jsp.beta(2.0, 0.0) * jsp.betainc(2.0, 0.0, 0.5)))
    True

    """
    return _incomplete_beta_impl(a, b, z)


@ft.partial(incomplete_beta.defjvp, symbolic_zeros=True)
def _incomplete_beta_jvp(
    primals: tuple[ScalarLike, ScalarLike, AnyArray],
    tangents: tuple[
        ScalarLike | SymbolicZero, ScalarLike | SymbolicZero, AnyArray | SymbolicZero
    ],
) -> tuple[AnyArray, AnyArray]:
    a, b, z = primals
    a_dot, b_dot, z_dot = tangents

    primal_out = _incomplete_beta_impl(a, b, z)
    tangent_out = jnp.zeros_like(primal_out)

    # d/dz B(a, b, z) = z^(a-1) (1-z)^(b-1): the integrand at the endpoint.
    if not isinstance(z_dot, SymbolicZero):
        tangent_out = tangent_out + z ** (a - 1.0) * (1.0 - z) ** (b - 1.0) * z_dot

    # The a/b tangents are only needed when the parameters are themselves
    # differentiated; there is no cheap closed form, so fall back to autodiff of
    # the series. Skipped entirely in the common case, where the differentiation
    # is with respect to position at fixed a and b.
    a_zero, b_zero = isinstance(a_dot, SymbolicZero), isinstance(b_dot, SymbolicZero)
    if not (a_zero and b_zero):
        _, ab_tangent = jax.jvp(
            lambda aa, bb: _incomplete_beta_impl(aa, bb, z),
            (a, b),
            (
                jnp.zeros_like(a) if a_zero else a_dot,
                jnp.zeros_like(b) if b_zero else b_dot,
            ),
        )
        tangent_out = tangent_out + ab_tangent

    return primal_out, tangent_out
