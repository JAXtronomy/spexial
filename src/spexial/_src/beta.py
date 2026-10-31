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
from collections.abc import Callable

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

_NEAR = 0.5
"""Half-width of the band around ``b + m == 0`` handled apart in `_expand_from`.

At most one step ``m`` has ``|b + m| < 1/2``. Its term,
``(w0^s - w^s) / s``, cancels for small ``s``; it is evaluated once, after the
scan, in a form with no cancellation for any ``s`` or dtype. Every other step
has ``|s| >= 1/2``, where the direct quotient loses nothing worth counting.
An earlier band of 1e-3, with an expansion in ``s`` inside it, left
``1e-3 <= |s| < ~0.1`` to the direct form: 5.8e-12 at ``a = 50, b = -0.999``,
and 3.5e-4 in float32.
"""

_EXPRL_COEFFS = tuple(1.0 / math.factorial(k + 1) for k in reversed(range(18)))
"""Taylor coefficients of ``expm1(y) / y = sum_k y^k / (k+1)!``, highest first.

At ``|y| < 1/2`` the first omitted term is ``0.5^18 / 19! ~ 3e-23``. Python
floats, so Horner's rule keeps the argument's dtype and weak type.
"""


def _expm1_over(y: AnyArray, /) -> AnyArray:
    """``expm1(y) / y``, 1 at ``y = 0``, with exact derivatives of every order there.

    By its Taylor series for ``|y| < 1/2`` (a polynomial, so autodiff is exact
    at 0), and by the quotient outside it, where nothing cancels.
    """
    small = jnp.abs(y) < 0.5
    y_small = jnp.where(small, y, 0.0)
    y_large = jnp.where(small, 1.0, y)
    series = _EXPRL_COEFFS[0]
    for c in _EXPRL_COEFFS[1:]:
        series = series * y_small + c
    return jnp.where(small, series, jnp.expm1(y_large) / y_large)  # type: ignore[no-any-return]


def _small_z(a: ScalarLike, b: ScalarLike, z: AnyArrayLike, /) -> AnyArray:
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
        carry: tuple[AnyArray, AnyArray, ScalarLike], k: ScalarLike, /
    ) -> tuple[tuple[AnyArray, AnyArray, ScalarLike], None]:
        total, z_pow, coeff = carry  # coeff = (1-b)_k / k!
        total = total + coeff / (a + k) * z_pow
        return (total, z_pow * z, coeff * (k + 1.0 - b) / (k + 1.0)), None

    init = (jnp.zeros_like(z), jnp.ones_like(z), jnp.ones_like(a))
    (total, _, _), _ = jax.lax.scan(step, init, jnp.arange(_NTERMS), unroll=_UNROLL)
    return z**a * total  # type: ignore[no-any-return]


def _large_z(a: ScalarLike, b: ScalarLike, z: AnyArrayLike, /) -> AnyArray:
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

    The terms are summed by `_expand_from`, which also handles the one term with
    $b + m$ near zero, where the direct form cancels.
    """
    w = 1.0 - z
    # `_small_z(a, b, 1/2)` below looks like a scalar being recomputed per
    # point, but computing it as a scalar and broadcasting is *slower*: XLA
    # already folds the constant-array input, and doing it by hand breaks the
    # fusion (measured 0.80x).
    return _small_z(a, b, jnp.full_like(w, 0.5)) + _expand_from(a, b, w, 0.5)  # type: ignore[no-any-return]


def _expand_from(
    a: ScalarLike, b: ScalarLike, w: AnyArray, w0: AnyArrayLike, /
) -> AnyArray:
    r"""$\int_w^{w_0} u^{b-1}(1-u)^{a-1}\,du$, expanding $(1-u)^{a-1}$ binomially.

    .. math::

        \sum_{m=0}^\infty \frac{(1-a)_m}{m!} \frac{w_0^{b+m} - w^{b+m}}{b+m}

    with the :math:`b + m \to 0` term :math:`\ln(w_0 / w)`. The terms fall off
    like :math:`((a - 1) w_0)^m / m!`, so this is the piece of `_large_z` above
    :math:`z = 1/2` (:math:`w_0 = 1/2`), and of `_large_a` above its split
    (:math:`w_0 = 2/(a+2)`, where :math:`(a-1) w_0 < 2` however large ``a``).

    Each term is :math:`(w_0^s - w^s)/s` with :math:`s = b + m`. That cancels as
    :math:`s \to 0`; at most one step has :math:`|s| <` `_NEAR`, and it is
    evaluated once, after the scan, as

    .. math::

        \frac{w_0^s - w^s}{s} = w^s\,\Delta\,
            \frac{\operatorname{expm1}(s\Delta)}{s\Delta},
        \qquad \Delta = \ln(w_0 / w),

    which cancels for no :math:`s` and in no dtype; `_expm1_over` is a
    polynomial near 0, so every :math:`b`-derivative is exact at :math:`s = 0`,
    which integer slopes do hit (e.g. :math:`\gamma = 2` in Zhao's Eq. 7).

    The sum is carried divided by :math:`w_0^b` and scaled back at the end,
    through logarithms only if :math:`w_0^b` itself would overflow -- so a very
    negative :math:`b` cannot overflow an intermediate while the result is
    representable.

    At :math:`w = 0` (:math:`z = 1`) there is nothing to cancel -- :math:`w^s`
    is 0 for :math:`s > 0` -- so the direct form is used there whatever
    :math:`s`; :math:`s \le 0` at :math:`w = 0` is the pole, which the callers
    handle. :math:`\ln w` is taken of a safe value there, so that no
    :math:`-\infty` sits under the mask.

    Summed into a `jax.lax.scan` carry: the terms are batched over ``w``, so
    materializing them all at once would make this memory- rather than
    flop-bound.
    """
    # An array, weakly typed for a Python scalar: Python float arithmetic on a
    # concrete `b` (0.5 ** -1100) would raise `OverflowError`, not return inf.
    w0 = jnp.asarray(w0)
    at_0 = w == 0
    w_safe = jnp.where(at_0, w0, w)
    log_w0 = jnp.log(w0)
    delta = log_w0 - jnp.log(w_safe)  # log(w0 / w) >= 0
    # The sum is carried divided by w0^b, which is applied at the end: each term
    # w0^(b+m) - w^(b+m) is w0^b (w0^m - r^b w^m), r = w / w0 <= 1. Unscaled, a
    # very negative b overflowed w0^b in an intermediate while the result -- a
    # *difference* of such terms -- was representable.
    r_b = jnp.where(at_0, 0.0, jnp.exp(-b * delta))  # r^b; 0^b = 0 for b > 0

    # The one step with |s| < `_NEAR` is skipped here and added afterwards; the
    # scan only records its coefficient and `s` (scalars). At w = 0 there is
    # nothing to cancel -- w^s = 0 for s > 0 -- so that step is taken directly
    # (s <= 0 at w = 0 is the pole, which the callers handle).
    def step(
        carry: tuple[AnyArray, AnyArray, AnyArray, ScalarLike, ScalarLike, ScalarLike],
        m: ScalarLike,
        /,
    ) -> tuple[
        tuple[AnyArray, AnyArray, AnyArray, ScalarLike, ScalarLike, ScalarLike], None
    ]:
        total, w0_m, w_m, coeff, near_coeff, near_s = carry  # coeff = (1-a)_m/m!
        s = b + m
        in_band = jnp.abs(s) < _NEAR
        skip = in_band & ~at_0
        s_safe = jnp.where(skip | (s == 0), 1.0, s)
        total = total + coeff * jnp.where(skip, 0.0, (w0_m - w_m) / s_safe)
        near_coeff = jnp.where(in_band, coeff, near_coeff)
        near_s = jnp.where(in_band, s, near_s)
        next_coeff = coeff * (m + 1.0 - a) / (m + 1.0)
        return (total, w0_m * w0, w_m * w, next_coeff, near_coeff, near_s), None

    zero = jnp.zeros_like(log_w0 * b)  # floating, weakly typed if both are
    init = (jnp.zeros_like(w), jnp.ones_like(w), r_b, jnp.ones_like(a), zero, zero)
    (total, _, _, _, near_coeff, near_s), _ = jax.lax.scan(
        step, init, jnp.arange(_NTERMS), unroll=_UNROLL
    )
    # The skipped step, scaled likewise: (w0^s - w^s) / s / w0^b
    # = r^b w^m* (e^(s delta) - 1) / s = r^b w^m* delta expm1(s delta) / (s delta),
    # which cancels for no s; m* = s - b. At w = 0 it was taken in the scan.
    w_m_near = r_b * jnp.exp((near_s - b) * jnp.log(w_safe))
    near = near_coeff * w_m_near * delta * _expm1_over(near_s * delta)
    scaled = total + jnp.where(at_0, 0.0, near)
    # Apply w0^b: directly while it is finite, and through logarithms only when
    # it would overflow (b log w0 > ~700), so as not to spend |log| * eps of
    # accuracy on a round trip the common case does not need.
    log_scale = b * log_w0
    big = log_scale > 700.0
    direct = scaled * jnp.exp(jnp.where(big, 0.0, log_scale))
    via_log = jnp.exp(
        jnp.where(big, log_scale, 0.0)
        + jnp.log(jnp.where(big & (scaled > 0), scaled, 1.0))
    )
    return jnp.where(  # type: ignore[no-any-return]
        big, jnp.where(scaled > 0, via_log, 0.0), direct
    )


_CF_FROM = 10.0
r"""Above this ``b``, `_large_b` replaces the two series.

Both series expand :math:`(1-t)^{b-1}` binomially, and its coefficients
:math:`(1-b)_k / k!` alternate and grow like :math:`b^k / k!`: at
:math:`z = 1/2` the sum is a near-total cancellation, worst relative error
:math:`2\times10^{-15}` at :math:`b = 10`, :math:`10^{-11}` at 30, and
:math:`10^{7}` at 100. The continued fraction is :math:`5\times10^{-14}` from
:math:`b = 0.1` up, so the switch can sit wherever the series is still exact.
"""

_CF_UNROLL = 1
"""Not unrolled, unlike the series.

A traced ``a`` or ``b`` goes through a `lax.switch`, which compiles every
branch, so this is compiled whenever either is traced, whatever the values
(a concrete pair compiles only its own branch). Unrolled 16-fold it made a
second ``b``-derivative take 200 s to compile for no runtime gain (measured
39 vs 41 ms at 1e5 points); unrolled once, 11 s.
"""

_CF_STEPS = 64
r"""Double steps of the continued fraction (each an even and an odd term).

The slowest case is the reflected fraction -- large first parameter -- just
above its switch point, at small ``a``: at :math:`a = 0.1, b = 10^3` it is
:math:`10^{-11}` at 32 steps, :math:`5\times10^{-14}` at 48 and
:math:`6\times10^{-15}` at 64, past which nothing changes. What remains is
rounding, growing like :math:`b\,\epsilon`: worst over
:math:`a \in [0.1, 50]`, :math:`4\times10^{-12}` at :math:`b = 10^5`,
:math:`5\times10^{-10}` at :math:`10^7`.
"""


def _continued_fraction(
    a: AnyArrayLike, b: AnyArrayLike, x: AnyArray, /, steps: int | None = None
) -> AnyArray:
    r"""Evaluate the DLMF 8.17.22 continued fraction by the modified Lentz method.

    .. math::

        B(a, b, x) = \frac{x^a (1-x)^b}{a} \cdot
            \cfrac{1}{1 + \cfrac{d_1}{1 + \cfrac{d_2}{1 + \cdots}}}

    with :math:`d_{2m+1} = -\frac{(a+m)(a+b+m)x}{(a+2m)(a+2m+1)}` and
    :math:`d_{2m} = \frac{m(b-m)x}{(a+2m-1)(a+2m)}`. Returns the fraction,
    :math:`1/(1 + d_1/(1 + \cdots))`. It converges fast for
    :math:`x < (a+1)/(a+b+2)`, and its terms are bounded for any :math:`b > 0`,
    so -- unlike the series -- nothing cancels as :math:`b` grows. A fixed
    number of steps, as for the series, so `vmap` lanes do not wait on each
    other.
    """
    # Lentz's guard against a zero denominator: the dtype's smallest normal, as
    # a Python float so that it does not promote the carry.
    tiny = float(jnp.finfo(x.dtype).tiny)
    guard = lambda v: jnp.where(jnp.abs(v) < tiny, tiny, v)  # noqa: E731

    def step(
        carry: tuple[AnyArray, AnyArray, AnyArray], m: ScalarLike, /
    ) -> tuple[tuple[AnyArray, AnyArray, AnyArray], None]:
        c, d, h = carry
        even = m * (b - m) * x / ((a + 2.0 * m - 1.0) * (a + 2.0 * m))
        odd = -(a + m) * (a + b + m) * x / ((a + 2.0 * m) * (a + 2.0 * m + 1.0))
        for coeff in (even, odd):
            d = 1.0 / guard(1.0 + coeff * d)
            c = guard(1.0 + coeff / c)
            h = h * d * c
        return (c, d, h), None

    d = 1.0 / guard(1.0 - (a + b) * x / (a + 1.0))
    init = (jnp.ones_like(x), d, d)
    (_, _, h), _ = jax.lax.scan(
        step,
        init,
        jnp.arange(1, (_CF_STEPS if steps is None else steps) + 1),
        unroll=_CF_UNROLL,
    )
    return h  # type: ignore[no-any-return]


def _complete_beta(a: ScalarLike, b: ScalarLike, /) -> AnyArray:
    r"""$\log B(a, b)$ for $a, b > 0$, as the two fractions at the switch point.

    :math:`B(a, b) = B(a, b, x_0) + B(b, a, 1 - x_0)`, two positive terms, so no
    cancellation. Not ``betaln``: `jax.scipy.special.betaln` is only good to
    :math:`3\times10^{-7}` at :math:`(a, b) = (8, 10)`. Returned as a logarithm,
    so that a complete beta function far below the smallest normal number --
    :math:`B(301, 1000) \approx 4\times10^{-307}` -- does not round to zero before
    it is used. Computed once per call, not per point, so `vmap` over ``z`` does
    not repeat it.
    """
    x0 = (a + 1.0) / (a + b + 2.0)
    # The front factor x^a (1-x)^b is the same for both terms, and is formed from
    # x0 itself: `log1p(-x0)`, not `log` of a rounded `1 - x0` scaled by b.
    log_front = a * jnp.log(x0) + b * jnp.log1p(-x0)
    h = _continued_fraction(
        jnp.stack([a, b]),
        jnp.stack([b, a]),
        jnp.stack([x0, 1.0 - x0]),
    )
    return log_front + jnp.log(h[0] / a + h[1] / b)  # type: ignore[no-any-return]


def _large_b(a: ScalarLike, b: ScalarLike, z: AnyArray, /) -> AnyArray:
    r"""$B(a, b, z)$ for $b > 0$, by the continued fraction and reflection.

    Below the switch point :math:`x_0 = (a+1)/(a+b+2)` the fraction converges
    as it stands; above it, by the reflection
    :math:`B(a, b, z) = B(a, b) - B(b, a, 1-z)`, where it converges for the
    swapped arguments. One fraction per point, with the arguments swapped where
    reflected; the complete :math:`B(a, b)` comes from `_complete_beta`.

    Each term is :math:`\exp(\log(\text{front}) + \log(h / p))`, with
    :math:`p` the fraction's first parameter: the front factor
    :math:`z^a (1-z)^b` is the same either way round and is formed from ``z``,
    never from a rounded ``1 - z`` scaled by ``b``; and the fraction is folded
    into the exponent, so a front factor below the smallest normal number does
    not flush to zero while the term itself is representable.
    """
    if jnp.dtype(z.dtype).itemsize < 4:
        # Half precision has too few bits for Lentz's recurrence to settle; work
        # in float32 and round the result once.
        f32 = jnp.float32
        return _large_b(jnp.asarray(a, f32), jnp.asarray(b, f32), z.astype(f32)).astype(
            z.dtype
        )
    x0 = (a + 1.0) / (a + b + 2.0)
    swap = z >= x0
    # z = 0 and z = 1 make the term exactly zero (the direct one at 0, the
    # reflected one at 1). Evaluated, `log(0)` would put `0 * -inf` into the
    # a- and b-derivatives; it is a genuine zero for every a, b > 0, so its
    # parameter derivatives are zero too.
    endpoint = (z == 0) | (z == 1)
    zs = jnp.where(endpoint, 0.5, z)
    log_front = a * jnp.log(zs) + b * jnp.log1p(-zs)
    first = jnp.where(swap, b, a)
    h = _continued_fraction(first, jnp.where(swap, a, b), jnp.where(swap, 1.0 - zs, zs))
    term = jnp.where(endpoint, 0.0, jnp.exp(log_front + jnp.log(h / first)))
    reflected = jnp.exp(_complete_beta(a, b)) - term
    out = jnp.where(swap, reflected, term)
    # Past b ~ 1/eps the switch point x0 is below eps, `1 - x0` rounds to 1, and
    # the reflected fraction's first denominator is exactly zero. Long before
    # that the b*eps rounding has taken every digit (0.1 relative by b ~ 1e15),
    # so say so with `nan` rather than return a confident `inf` or `0`.
    return jnp.where(1.0 - x0 == 1.0, jnp.nan, out)  # type: ignore[no-any-return]


def _series(a: ScalarLike, b: ScalarLike, z: AnyArray, /) -> AnyArray:
    """$B(a, b, z)$ by `_small_z` and `_large_z`, switched at $z = 1/2$."""
    # For b <= 0 the integral diverges at z = 1: a genuine pole, +inf (GH-67).
    # `_large_z` cannot see it -- at w = 1 - z = 0 its expansion is `inf - inf`
    # -- so the pole is returned directly, and the series is handed a harmless
    # z there, so that its `nan` cannot reach a cotangent through the `where`.
    pole = (z == 1.0) & (b <= 0)
    z = jnp.where(pole, 0.5, z)
    # Both branches are evaluated, so clamp each one's input to the range where
    # it is well behaved; `where` then discards the unused value. Clamped by
    # `where`, not `minimum`/`maximum`: those split the derivative at a tie,
    # and at exactly z = 1/2 the series' z-derivative came out halved.
    lower = z <= 0.5
    value = jnp.where(
        lower,
        _small_z(a, b, jnp.where(lower, z, 0.5)),
        _large_z(a, b, jnp.where(lower, 0.5, z)),
    )
    return jnp.where(pole, jnp.inf, value)  # type: ignore[no-any-return]


_CF_A_FROM = 8.0
r"""Above this ``a``, the series give way to `_large_b` or `_large_a`.

`_large_z` expands :math:`(1-u)^{a-1}` over :math:`u \in [w, 1/2]`, and its
coefficients :math:`(1-a)_m/m!` cancel just as :math:`(1-b)_k/k!` do for
large :math:`b`: measured worst over :math:`b \in [-50, 10]`,
:math:`6\times10^{-14}` at :math:`a = 8`, :math:`4\times10^{-11}` at 16,
:math:`2\times10^{-8}` at 24, and wrong from about 50.
"""

_CF_A_B_MIN = 0.1
r"""For large ``a``: at or above this ``b``, `_large_b`; below it, `_large_a`.

`_large_b`'s reflection subtracts from the complete :math:`B(a, b) \sim 1/b`,
so it loses about :math:`\log_{10}(1/b)` digits as :math:`b \to 0^+`
(:math:`10^{-8}` at :math:`b = 10^{-8}`), and has no meaning at
:math:`b \le 0`, where :math:`B(a, b)` diverges. From 0.1 it is
:math:`4\times10^{-13}` or better to :math:`a = 50`; below it `_large_a`, which
needs no reflection, takes over.
"""


def _large_a(a: ScalarLike, b: ScalarLike, z: AnyArray, /) -> AnyArray:
    r"""$B(a, b, z)$ for large $a$ and $b < $ `_CF_A_B_MIN`, including $b \le 0$.

    Split at :math:`s = a/(a+2)`. Below it, the continued fraction directly:
    that is at or below its switch point :math:`x_0 = (a+1)/(a+b+2)` whenever
    :math:`b \le 1 + 2/a` -- so for every :math:`b < 0.1` routed here -- and
    beyond :math:`x_0` it still converges in this range, only more slowly
    (checked against mpmath; for :math:`-(a+2) < b \le -1`, :math:`x_0 > 1`,
    and below that :math:`x_0 < 0`, where the textbook criterion says
    nothing). Above it,

    .. math::

        B(a, b, z) = B(a, b, s) + \int_{1-z}^{2/(a+2)} u^{b-1}(1-u)^{a-1}\,du,

    the second term by `_expand_from` with :math:`w_0 = 2/(a+2)`. That is the
    expansion `_large_z` uses from :math:`w_0 = 1/2`, where its terms grow like
    :math:`(a/2)^m/m!` and cancel; here they fall like :math:`2^m/m!`
    (:math:`(a-1) w_0 < 2`), costing under a digit. Measured against mpmath over
    :math:`a \in [12, 100]`, :math:`b \in [-50, 0.1)` and :math:`z` to
    :math:`1 - 10^{-12}`: :math:`2\times10^{-14}`, where the series was
    :math:`10^{-9}` at :math:`a = 24` and :math:`10^{22}` at 100.

    At :math:`z = 1` with :math:`b \le 0` the integral diverges: the pole is
    returned as ``inf``, with the expansion handed a harmless ``w`` there. At
    :math:`z = 0` the term is a genuine zero, guarded as in `_large_b`.
    """
    if jnp.dtype(z.dtype).itemsize < 4:
        f32 = jnp.float32
        return _large_a(jnp.asarray(a, f32), jnp.asarray(b, f32), z.astype(f32)).astype(
            z.dtype
        )
    # Arrays, weakly typed when ``a`` is a Python scalar, so `z` sets the dtype.
    split = jnp.asarray(a / (a + 2.0))
    w0 = jnp.asarray(2.0 / (a + 2.0))
    below = z < split
    pole = (z == 1.0) & (b <= 0)
    at_0 = z == 0
    # Below the split: the fraction at z itself (z = 0 guarded, as in
    # `_large_b`); above it: the fraction at the split, a scalar, plus the
    # expansion out to z.
    xs = jnp.where(below & ~at_0, z, split)
    log_front = a * jnp.log(xs) + b * jnp.log1p(-xs)
    direct = jnp.exp(log_front + jnp.log(_continued_fraction(a, b, xs) / a))
    direct = jnp.where(at_0, 0.0, direct)
    log_front_s = a * jnp.log(split) + b * jnp.log(w0)
    at_split = jnp.exp(log_front_s + jnp.log(_continued_fraction(a, b, split) / a))
    # At z = 1 (the pole when b <= 0) `_expand_from` takes w = 0 safely itself.
    w = jnp.where(below, w0, 1.0 - z)
    above = at_split + _expand_from(a, b, w, w0)
    return jnp.where(  # type: ignore[no-any-return]
        pole, jnp.inf, jnp.where(below, direct, above)
    )


_NEG_B_FROM = -5.0
r"""Below this ``b`` (for ``a`` up to `_CF_A_FROM`), `_negative_b` replaces the series.

`_small_z`'s terms carry :math:`(1-b)_k / k!`, which for very negative
:math:`b` grows like :math:`(|b| z)^k / k!` before it decays: 64 terms cover
:math:`|b| z \lesssim 10`. Measured worst over :math:`a \in [0.5, 8]`:
:math:`2\times10^{-15}` at :math:`b = -5`, :math:`6\times10^{-12}` at -10,
:math:`2\times10^{-7}` at -20, 8% at -50, no correct digits by -200.
"""

_NEG_B_STEPS = 128
"""Double steps of the continued fraction in `_negative_b`.

At very negative ``b`` and small ``a`` the fraction converges more slowly than
in `_large_b`: 64 steps leave 5e-9 at ``a = 0.5, b = -1100``; 128 give 7e-15,
and 256 change nothing.
"""


def _negative_b(a: ScalarLike, b: ScalarLike, z: AnyArray, /) -> AnyArray:
    r"""$B(a, b, z)$ for very negative $b$, by the continued fraction directly.

    No reflection -- it would need :math:`B(a, b)`, which diverges for
    :math:`b \le 0` -- and none needed: for :math:`b` this negative the fraction
    converges at every :math:`z \in [0, 1)` (checked against mpmath for
    :math:`a \in [0.01, 8]`, :math:`b \in [-1100, -5]`: worst
    :math:`6\times10^{-14}`). The term is formed in log space, as in
    `_large_b`, so a huge value that is still representable does not overflow
    on the way. :math:`z = 0` is a genuine zero; :math:`z = 1` the pole.
    """
    if jnp.dtype(z.dtype).itemsize < 4:
        f32 = jnp.float32
        return _negative_b(
            jnp.asarray(a, f32), jnp.asarray(b, f32), z.astype(f32)
        ).astype(z.dtype)
    endpoint = (z == 0) | (z == 1)
    zs = jnp.where(endpoint, 0.5, z)
    h = _continued_fraction(a, b, zs, steps=_NEG_B_STEPS)
    term = jnp.exp(a * jnp.log(zs) + b * jnp.log1p(-zs) + jnp.log(h / a))
    return jnp.where(  # type: ignore[no-any-return]
        z == 1, jnp.inf, jnp.where(z == 0, 0.0, term)
    )


_SERIES, _CONTINUED_FRACTION, _LARGE_A, _NEGATIVE_B, _EITHER = 0, 1, 2, 3, 4
"""Which evaluation `_incomplete_beta_core` runs; static, so it is compiled alone.

`_EITHER` is for a traced ``a`` or ``b``, whose branch is only known at run time:
`_dispatch` then switches between the others.
"""


def _branch_index(a: ScalarLike, b: ScalarLike, /) -> ScalarLike:
    """Return the branch index (`_SERIES`, `_LARGE_A`, ...) for ``a`` and ``b``."""
    large_a = jnp.where(b >= _CF_A_B_MIN, _CONTINUED_FRACTION, _LARGE_A)
    small_a = jnp.where(b < _NEG_B_FROM, _NEGATIVE_B, _SERIES)
    return jnp.where(  # type: ignore[no-any-return]
        b > _CF_FROM, _CONTINUED_FRACTION, jnp.where(a > _CF_A_FROM, large_a, small_a)
    )


def _branch(a: ScalarLike, b: ScalarLike, /) -> int:
    """Pick the static branch for concrete ``a`` and ``b``, else `_EITHER`.

    Decided in the public wrapper, before `jax.custom_jvp` lifts every argument
    to a tracer: inside it, even a literal ``b`` is traced under `jit`, so a
    choice made there would always compile every branch.
    """
    if isinstance(a, jax.core.Tracer) or isinstance(b, jax.core.Tracer):
        return _EITHER
    # Plain Python, mirroring `_branch_index`: `jnp` ops here would be staged
    # into an enclosing `jit` trace even on concrete inputs.
    if b > _CF_FROM:
        return _CONTINUED_FRACTION
    if a > _CF_A_FROM:
        return _CONTINUED_FRACTION if b >= _CF_A_B_MIN else _LARGE_A
    return _NEGATIVE_B if b < _NEG_B_FROM else _SERIES


def _incomplete_beta_impl(
    a: ScalarLike, b: ScalarLike, z: AnyArrayLike, branch: int, /
) -> AnyArray:
    r"""$B(a, b, z)$ for $a > 0$, any real $b$, and $z \in [0, 1]$."""
    z = jnp.asarray(z)
    if not jnp.issubdtype(z.dtype, jnp.inexact):
        # An integer `z` would otherwise carry its dtype into the continued
        # fraction's switch point, (a+1)/(a+b+2), and truncate it to zero.
        z = z.astype(jnp.promote_types(z.dtype, float))
    # Promote `z` to the common type of all three, as any elementwise op would
    # (GH-66): a float32 `z` with a strong float64 `b` otherwise starts a `scan`
    # carry in float32 that the first step promotes to float64, which `scan`
    # rejects. Weak (Python) scalars do not promote, so the common case keeps
    # `z`'s own dtype and weak type.
    dtype = jnp.result_type(a, b, z)
    if dtype != z.dtype:
        z = z.astype(dtype)
    return (_series, _large_b, _large_a, _negative_b)[branch](a, b, z)


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
    :math:`2^{-k}`; see `_small_z` and `_large_z`. Outside :math:`a \le 8`,
    :math:`-5 \le b \le 10` their binomial coefficients cancel or grow, and
    three fixed-length methods take over: the continued fraction with
    reflection (`_large_b`: :math:`b > 10`, or :math:`a > 8` with
    :math:`b \ge 0.1`), `_large_a` (:math:`a > 8`, :math:`b < 0.1`), and the
    continued fraction directly (`_negative_b`: :math:`a \le 8`,
    :math:`b < -5`). The
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
    # The branch is chosen from the arguments as given -- before the promotion
    # below, whose `jnp` ops would turn a literal into a staged value under an
    # enclosing `jit` and so lose the static choice.
    branch = _branch(a, b)
    a, b, z = _promote(a, b, z)
    if branch != _EITHER:
        return _incomplete_beta_core(a, b, z, branch)
    return _dispatch(a, b, z)


def _dispatch(a: ScalarLike, b: ScalarLike, z: AnyArray, /) -> AnyArray:
    """Pick the branch at run time, for a traced ``a`` or ``b``.

    The `switch` sits outside the custom JVP, each branch its own static
    `_incomplete_beta_core`: inside, the rule's `jvp` of a batched `switch`
    leaves a `stop_gradient` in the linearized program that JAX cannot
    transpose, so ``grad`` of a `vmap` over ``a`` or ``b`` raised.
    """
    index = _branch_index(a, b)
    # `a` and `b` are scalars, so `switch` runs one branch. Under `vmap` it
    # becomes a `select` that runs every branch, on every lane. An unselected
    # branch may then return `nan` or `inf` -- the series at large `b`, the
    # reflection at `b <= 0` -- and autodiff multiplies its Jacobian by the
    # zero cotangent `select` gives it, `0 * inf = nan`. What stops that
    # reaching `a` and `b` is the `where(mine, ...)` on each branch's inputs,
    # whose own transpose zeroes the unselected lane's cotangent. The
    # substituted values matter only in that they keep the unselected branches
    # cheap and finite; any value would do for correctness. `z` needs no mask:
    # its cotangent is selected per lane after each branch's transpose, so an
    # unselected branch's infinite (1-z)^(b-1) at z = 1 is discarded.
    safe = (
        (1.0, 1.0),  # series
        (1.0, 2.0 * _CF_FROM),  # continued fraction
        (2.0 * _CF_A_FROM, 0.0),  # large a
        (1.0, 2.0 * _NEG_B_FROM),  # negative b
    )

    def run(k: int, /) -> Callable[[], AnyArray]:
        def go() -> AnyArray:
            mine = index == k
            ak = jnp.where(mine, a, safe[k][0])
            bk = jnp.where(mine, b, safe[k][1])
            return _incomplete_beta_core(ak, bk, z, k)  # type: ignore[no-any-return]

        return go

    return jax.lax.switch(index, [run(k) for k in range(len(safe))])  # type: ignore[no-any-return]


def _promote(
    a: ScalarLike, b: ScalarLike, z: AnyArrayLike, /
) -> tuple[ScalarLike, ScalarLike, AnyArray]:
    """Make ``a``, ``b`` and ``z`` floating, and the strong ones of one dtype.

    Integer ``z`` becomes float (otherwise it carries its dtype into the
    continued fraction's switch point and truncates it to zero), then takes
    ``result_type(a, b, z)`` (GH-66). Integer ``a`` and ``b`` become float too
    -- an integer seeding a `scan` carry that a float then updates is a type
    error, and an integer primal has a ``float0`` tangent the custom JVP
    cannot build -- and a *strong* ``a`` or ``b`` of another dtype is cast to
    the common one, for the same reason (float32 ``a`` with float64 ``b``).
    Weak parameters -- Python floats -- are left alone: they never change a
    carry's type, and casting them down to a float32 ``z`` would compute the
    parameter-only quantities (the switch point, the complete beta function)
    in float32 for nothing, at ``b * eps`` cost.
    """
    z = jnp.asarray(z)
    if not jnp.issubdtype(z.dtype, jnp.inexact):
        z = z.astype(jnp.promote_types(z.dtype, float))
    dtype = jnp.result_type(a, b, z)
    if dtype != z.dtype:
        z = z.astype(dtype)

    def match(p: ScalarLike, /) -> ScalarLike:
        if isinstance(p, int | float) and not isinstance(p, bool):
            return float(p)  # a weak float
        p = jnp.asarray(p)
        if p.weak_type:
            if jnp.issubdtype(p.dtype, jnp.inexact):
                return p
            return p * 1.0  # a weak integer becomes a weak float
        return p.astype(dtype)

    return match(a), match(b), z


@jax.custom_jvp
def _pow(x: AnyArrayLike, y: ScalarLike, /) -> AnyArray:
    """``x ** y`` for ``x >= 0``, with every x-derivative exact at ``x = 0``.

    JAX differentiates a float-exponent power as ``y x^(y-1)``, which at
    ``x = 0`` is ``0 * inf`` once differentiation has brought ``y`` to 0 --
    upstream's deliberate reading of ``**`` on floats as the real function
    ``x^y``, not differentiable at (0, 0) (jax-ml/jax#14397). For an integer
    ``y = n >= 0`` that made the z-rule's derivatives from order ``n + 2`` on
    ``nan`` at the endpoint (GH-73). Here the x-derivative of ``x^0`` at
    ``x = 0`` is 0 exactly, and the rule recurses through `_pow`, so every order
    meets that case in turn.
    """
    return jnp.asarray(x) ** y


@_pow.defjvp
def _pow_jvp(
    primals: tuple[AnyArrayLike, ScalarLike],
    tangents: tuple[AnyArrayLike, ScalarLike],
    /,
) -> tuple[AnyArray, AnyArray]:
    x, y = primals
    x_dot, y_dot = tangents
    out = _pow(x, y)
    # Only at x = 0: guarding y = 0 alone would also zero the y-derivative of
    # this, y x^(y-1) -> x^(y-1), at every x -- the bug that removed JAX's own
    # special case (jax-ml/jax#12033). Doubly, so the masked 0 * inf cannot
    # reach a cotangent.
    corner = (x == 0) & (y == 0)
    dx = jnp.where(corner, 0.0, y * _pow(jnp.where(corner, 1.0, x), y - 1.0))
    # d/dy x^y = log(x) x^y, which is 0 at x = 0 for y > 0; `log` of a safe
    # value there, as JAX does.
    dy = jnp.log(jnp.where(x == 0, 1.0, x)) * out
    return out, dx * x_dot + dy * y_dot


@ft.partial(jax.custom_jvp, nondiff_argnums=(3,))
def _incomplete_beta_core(
    a: ScalarLike, b: ScalarLike, z: AnyArrayLike, branch: int, /
) -> AnyArray:
    """`incomplete_beta` for a given static branch, with its derivative rule."""
    return _incomplete_beta_impl(a, b, z, branch)


@ft.partial(_incomplete_beta_core.defjvp, symbolic_zeros=True)
def _incomplete_beta_jvp(
    branch: int,
    primals: tuple[ScalarLike, ScalarLike, AnyArray],
    tangents: tuple[
        ScalarLike | SymbolicZero, ScalarLike | SymbolicZero, AnyArray | SymbolicZero
    ],
    /,
) -> tuple[AnyArray, AnyArray]:
    a, b, z = primals
    a_dot, b_dot, z_dot = tangents

    # The a/b tangents are only needed when the parameters are themselves
    # differentiated; there is no cheap closed form, so fall back to autodiff of
    # the implementation. Skipped entirely in the common case, where the
    # differentiation is with respect to position at fixed a and b. When it does
    # run, the primal comes from the same `jvp`, not a second evaluation.
    #
    # It is checkpointed. Reverse-mode through `_dispatch`'s `switch` makes
    # every branch emit the same set of residuals, zero-filling the others', so
    # the series branch would allocate the continued fraction's per-step carries
    # (64 x the points) just to discard them. Checkpointed, its residuals are
    # its inputs. Measured on the a/b gradient at 1e5 points, b = 1.5: 63 -> 46
    # ms (main, with no branching: 40), and the second-derivative compile 4.9 ->
    # 3.4 s. Only this `jvp`: checkpointing the whole core also recomputed the
    # primal in a z-gradient, 4.3 -> 5.6 ms for `value_and_grad`.
    a_zero, b_zero = isinstance(a_dot, SymbolicZero), isinstance(b_dot, SymbolicZero)
    if a_zero and b_zero:
        primal_out = _incomplete_beta_impl(a, b, z, branch)
        tangent_out = jnp.zeros_like(primal_out)
    else:
        primal_out, tangent_out = jax.jvp(
            jax.checkpoint(lambda aa, bb: _incomplete_beta_impl(aa, bb, z, branch)),
            (a, b),
            (
                jnp.zeros_like(a) if a_zero else a_dot,
                jnp.zeros_like(b) if b_zero else b_dot,
            ),
        )

    # d/dz B(a, b, z) = z^(a-1) (1-z)^(b-1): the integrand at the endpoint.
    if not isinstance(z_dot, SymbolicZero):
        if isinstance(b, jax.core.Tracer):
            # `b` may itself be differentiated, so write (1-z)^(b-1) so that its
            # `b`-derivative carries `log1p(-z)`, not `log(1 - z)` of a rounded
            # `1 - z`, which keeps only ~1e-16 / z relative precision (GH-68).
            # At z = 1 itself `pow` takes over, by a double `where`: `log1p(-z)` is
            # `-inf` there, and any clamp of it differentiates as `0 * -inf`,
            # making the second and third z-derivatives `nan`. `pow` has exact
            # derivatives at a zero base, as on the concrete path.
            z_arr = jnp.asarray(z)
            inside = z_arr < 1.0
            z_in = jnp.where(inside, z_arr, 0.0)
            via_log1p = jnp.exp((b - 1.0) * jnp.log1p(-z_in))
            one_minus_z_pow = jnp.where(inside, via_log1p, _pow(1.0 - z_arr, b - 1.0))
        else:
            # A concrete `b` has no derivative to get wrong, and as a constant
            # exponent XLA can simplify `pow` (to a reciprocal at b = 0, a square
            # root at b = 3/2), which `exp(... log1p)` would hide from it.
            one_minus_z_pow = _pow(1.0 - z, b - 1.0)
        tangent_out = tangent_out + _pow(z, a - 1.0) * one_minus_z_pow * z_dot

    return primal_out, tangent_out
