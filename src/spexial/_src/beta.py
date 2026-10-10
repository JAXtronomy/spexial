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
import math

import jax
import jax.numpy as jnp
import numpy as np
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


_EXPRL_BAND = 0.5
"""Below this ``|x|``, `_a_eq_1` sums ``expm1(x) / x`` as a series.

Above it the direct quotient is accurate to working precision in value and in
its first three derivatives; below it, those derivatives cancel.
"""

_EXPRL_COEFFS = tuple(1.0 / math.factorial(k + 1) for k in reversed(range(18)))
"""Taylor coefficients of ``expm1(x) / x = sum_k x^k / (k+1)!``, highest first.

At ``|x| < 1/2`` the first omitted term is ``0.5^18 / 19! ~ 3e-23``, far below
float64 resolution. Python floats, not an array, so that Horner's rule in
`_a_eq_1` keeps the input's dtype and weak type rather than promoting both to
the default float.
"""


_LOG1P_UPTO = 0.3
"""Below this ``z``, `_log1m` uses ``log1p(-z)``; above it, ``log(1 - z)``.

XLA's CPU ``log1p`` is off by up to ~240 ulp (2.7e-14) for arguments in
``(-0.42, -0.3)``, and the closed form multiplies that by ``|b L|``. From
``z = 0.3`` up, ``1 - z`` is exact to within half an ulp, so ``log`` of it is
good to 3e-16; below, ``log1p`` is. Either way about 2 ulp. (In float32 the
switch buys nothing -- there ``log1p`` is the better of the two, 1.6 against
2.6 ulp on that band -- but costs nothing either.)
"""


def _log1m(z: AnyArray) -> AnyArray:
    """``log(1 - z)`` to about 2 ulp, ``-inf`` at ``z = 1``; see `_LOG1P_UPTO`."""
    small = z <= _LOG1P_UPTO
    # Double `where`: each side only ever sees arguments it is accurate and finite
    # on, so neither puts a non-finite derivative under the other's mask.
    via_log1p = jnp.log1p(-jnp.where(small, z, 0.0))
    via_log = jnp.log(1.0 - jnp.where(small, 0.5, z))
    return jnp.where(small, via_log1p, via_log)  # type: ignore[no-any-return]


def _a_eq_1(b: ScalarLike, z: AnyArrayLike) -> AnyArray:
    r"""$B(1, b, z)$, in closed form.

    .. math::

        B(1, b, z) = \int_0^z (1-t)^{b-1}\,\mathrm{d}t
                   = \frac{1 - (1-z)^b}{b}
                   = \frac{1 - e^{x}}{b}
                   = -L\,\frac{\operatorname{expm1}(x)}{x},
        \qquad L = \ln(1-z),\ x = bL

    The removable singularity at $b = 0$ (where it is $-\ln(1-z)$) is handled
    by evaluating ``expm1(x) / x`` as its Taylor series near $x = 0$, by
    Horner. That changes the *formula* there rather than substituting a value,
    so every derivative in $b$ is right at $b = 0$ too, not just the value.
    Away from it, ``(1 - exp(x)) / b`` divides by $b$ rather than $x$, so that
    at $z = 1$ -- where $L = -\infty$ -- it is exactly $1/b$ for $b > 0$ and
    $+\infty$ for $b < 0$, instead of $\infty \cdot 0$. It is ``exp``, not
    ``expm1``, there: with $\lvert x \rvert \ge 1/2$ nothing cancels, and JAX
    differentiates ``expm1(x)`` as ``expm1(x) + 1``, which rounds $e^x$ to zero
    below $x \approx -37$ and so zeroes the mixed derivative
    $\partial_z \partial_b B$.
    """
    z = jnp.asarray(z)
    # `L` is clamped at z = 1 only, to the dtype's most negative finite value --
    # not for z > 1, where `log(1 - z)` is `nan` and must stay so -- and by a
    # double `where`, so that neither `0 * -inf` (at b = 0) nor the derivative
    # of a clamp at `-inf` reaches a value or a cotangent.
    at_1 = z == 1.0
    L = jnp.where(at_1, float(jnp.finfo(z.dtype).min), _log1m(jnp.where(at_1, 0.0, z)))
    x = b * L
    near = jnp.abs(x) < _EXPRL_BAND
    # `|x| >= 1/2` implies `b != 0`, so these placeholders only keep the
    # unselected branch finite, so that no `0 * inf` reaches a cotangent.
    x_far = jnp.where(near, 0.0, x)
    b_far = jnp.where(near, 1.0, b)
    # Far from x = 0 the closed form is (1 - e^x) / b. For b < 0, x > 0 and e^x
    # overflows at x ~ 709.8 (88.7 in float32) while the value, ~e^x / |b|, need
    # not: it is written e^(x - log|b|) - 1/|b| there, with nothing to cancel
    # since x >= 1/2.
    negative = b < 0
    abs_b = jnp.abs(b_far)
    far_negative = jnp.exp(x_far - jnp.log(abs_b)) - 1.0 / abs_b
    far_positive = (1.0 - jnp.exp(jnp.where(negative, 0.0, x_far))) / b_far
    far = jnp.where(negative, far_negative, far_positive)
    x_near = jnp.where(near, x, 0.0)
    series = _EXPRL_COEFFS[0]
    for c in _EXPRL_COEFFS[1:]:
        series = series * x_near + c
    # At z = 1 the series branch is taken only for |b| below ~1e-308, where the
    # value is +inf either way (a pole for b <= 0; 1/b overflows for b > 0). It
    # is returned directly rather than as `-L * series` with an infinite `L`,
    # which would multiply a masked branch by `-inf` in the mixed derivatives.
    near_value = jnp.where(at_1, jnp.inf, -L * series)
    out = jnp.where(near, near_value, far)
    # b = -inf: x = +inf and `far` is inf / -inf. The integral of (1-t)^(-inf)
    # diverges for any z > 0; at z = 0 it is empty.
    infinite_b = jnp.where(z > 0, jnp.where(b > 0, 0.0, jnp.inf), 0.0)
    return jnp.where(jnp.isinf(b), infinite_b, out)  # type: ignore[no-any-return]


@jax.custom_jvp
def _a_eq_1_core(b: ScalarLike, z: AnyArrayLike) -> AnyArray:
    """`_a_eq_1`, with the same exact O(1) `z`-derivative as the series path."""
    return _a_eq_1(b, z)


@ft.partial(_a_eq_1_core.defjvp, symbolic_zeros=True)
def _a_eq_1_jvp(
    primals: tuple[ScalarLike, AnyArray],
    tangents: tuple[ScalarLike | SymbolicZero, AnyArray | SymbolicZero],
) -> tuple[AnyArray, AnyArray]:
    b, z = primals
    b_dot, z_dot = tangents

    primal_out = _a_eq_1(b, z)
    tangent_out = jnp.zeros_like(primal_out)

    # The integrand at the endpoint, as for the series -- here (1-z)^(b-1).
    if not isinstance(z_dot, SymbolicZero):
        if isinstance(b, jax.core.Tracer):
            # `b` may itself be differentiated, so write the power so that its
            # `b`-derivative carries an accurate `log(1 - z)` (`_log1m`), not
            # `log` of a rounded `1 - z`, which loses digits as z -> 0. At z = 1
            # itself `pow` takes over, by a double `where`: the logarithm is
            # `-inf` there, and any clamp of it differentiates as `0 * -inf`,
            # making the second and third z-derivatives `nan`.
            z_arr = jnp.asarray(z)
            inside = z_arr != 1.0
            via_log = jnp.exp((b - 1.0) * _log1m(jnp.where(inside, z_arr, 0.0)))
            integrand = jnp.where(inside, via_log, (1.0 - z_arr) ** (b - 1.0))
        else:
            # A concrete `b` has no derivative to get wrong, and as a constant
            # exponent XLA can simplify `pow` (to a reciprocal at b = 0, a square
            # root at b = 3/2), which `exp(... log1p)` would hide from it.
            integrand = (1.0 - z) ** (b - 1.0)
        tangent_out = tangent_out + integrand * z_dot

    # Elementary, so autodiff of the closed form is cheap and exact; the series
    # near b = 0 keeps every order right at the removable singularity.
    if not isinstance(b_dot, SymbolicZero):
        _, b_tangent = jax.jvp(lambda bb: _a_eq_1(bb, z), (b,), (b_dot,))
        tangent_out = tangent_out + b_tangent

    return primal_out, tangent_out


def _is_static_one(a: object) -> bool:
    """Whether ``a`` is a concrete scalar equal to 1, decidable at trace time."""
    return (
        not isinstance(a, jax.core.Tracer)
        and np.ndim(a) == 0
        and np.isrealobj(a)
        and bool(np.equal(a, 1))
    )


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
    The derivative rules are `jax.custom_jvp`, not `jax.custom_vjp`, so that
    `jax.hessian`'s ``jacfwd(jacrev(...))`` still composes. They sit on inner
    cores: `incomplete_beta` itself is a plain function that dispatches between
    them, so it has no ``.fun`` or ``.defjvp`` of its own.

    When ``a`` is a concrete 1 (a Python or NumPy scalar, or a concrete 0-d
    array; not a traced value) the integral is elementary,
    :math:`(1 - (1-z)^b)/b`, and that closed form replaces the series; see
    `_a_eq_1`. It carries the same exact
    :math:`z`-derivative rule, and its ``b``-derivative is autodiff of the
    closed form, which is elementary. A traced ``a`` cannot be inspected and
    takes the series even at 1, which holds its accuracy only for
    :math:`\lvert b \rvert \lesssim 10`; keep ``a`` concrete for the closed
    form's wider domain.

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

    With ``a`` a literal 1 the integral is elementary, and that closed form is
    used instead of the series:

    >>> round(float(sp.incomplete_beta(1.0, 0.0, jnp.asarray(0.5))), 8)
    0.69314718

    """
    # Decided at trace time, so it costs nothing when it does not apply. A traced
    # `a` -- e.g. one being differentiated or `vmap`ped -- takes the series.
    if _is_static_one(a):
        if np.ndim(b) != 0:
            # The series path cannot broadcast `b` either (it raises inside its
            # `scan`); say so here too, rather than answer for some `a` only.
            msg = f"incomplete_beta: `b` must be a scalar, got shape {np.shape(b)}"
            raise TypeError(msg)
        # An integer `z` promotes to float, as on the series path; against `b`
        # the closed form's own arithmetic promotes it as any elementwise op.
        z = jnp.asarray(z)
        if not jnp.issubdtype(z.dtype, jnp.inexact):
            z = z.astype(jnp.promote_types(z.dtype, float))
        return _a_eq_1_core(b, z)
    return _incomplete_beta_core(a, b, z)


@jax.custom_jvp
def _incomplete_beta_core(a: ScalarLike, b: ScalarLike, z: AnyArrayLike) -> AnyArray:
    """`incomplete_beta` by the series, with an exact O(1) `z`-derivative rule."""
    return _incomplete_beta_impl(a, b, z)


@ft.partial(_incomplete_beta_core.defjvp, symbolic_zeros=True)
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
