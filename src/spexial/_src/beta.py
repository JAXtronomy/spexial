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
runs a second such loop. What is here instead is fixed-length evaluation
throughout -- two geometrically convergent series, and continued fractions with
a fixed number of steps where the series cancel -- with an exact O(1)
derivative rule.

Contributed from `galax <https://github.com/GalacticDynamics/galax>`_, where it
was written for the Zhao (1996) family of density profiles (Eq. 43).

"""

__all__ = ["incomplete_beta"]

import functools as ft
import math
from collections.abc import Callable
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
from jax.custom_derivatives import SymbolicZero

from .custom_types import AnyArray, AnyArrayLike, ScalarLike

_NTERMS = 64
"""Series length. Both series below converge like 2^-k, so this is ~1e-19.

Not worth trimming: measured over a in [0.2, 8], b in [-2.5, 6] and z up to
1 - 1e-8, dropping to 48 terms costs about an order of magnitude of accuracy
at moderate `a`, and 40 breaks the 1e-11 the tests assert, to
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
floats, so Horner's rule keeps the argument's dtype and weak type. Shared by
`_expm1_over` and `_a_eq_1`.
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
    tail = _expand_from(a, b, w, 0.5, wide_b=False)
    return _small_z(a, b, jnp.full_like(w, 0.5)) + tail  # type: ignore[no-any-return]


def _expand_from(
    a: ScalarLike, b: ScalarLike, w: AnyArray, w0: AnyArrayLike, /, *, wide_b: bool
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

    The sum is carried divided by a scale and multiplied back at the end. For
    `_large_z` (:math:`-5 \le b \le 10`, :math:`w_0 = 1/2`, in float32 or
    float64, since `_series` promotes half precision) the scale is the scalar
    :math:`w_0^b`. With ``wide_b`` (`_split_at`, for `_large_a` and
    `_negative_b`, where :math:`b` is unbounded below) it is per point, the
    larger of :math:`w_0^b` and :math:`w^b`, so that every term is at most 1 and
    the sum, about :math:`1/|b|`, can neither overflow nor sink to the bottom of
    the float range; the scale is then applied through `_exp_of_log_term`, whose
    reverse mode stays finite up to the top of the float range. (Scaled by
    :math:`w_0^b` alone, :math:`(w/w_0)^b` overflowed inside the sum for very
    negative :math:`b`, and reverse mode multiplied the sum's intermediates by
    :math:`w_0^b` before anything brought them back down: ``nan`` gradients from
    values of about :math:`10^{297}`.)

    At :math:`w = 0` (:math:`z = 1`) there is nothing to cancel -- :math:`w^s`
    is 0 for :math:`s > 0` -- so the skipped term there is
    :math:`w_0^s / s`; :math:`s \le 0` at :math:`w = 0` is the pole, which the
    callers handle. :math:`\ln w` is taken of a safe value there, so that no
    :math:`-\infty` sits under the mask.

    Summed into a `jax.lax.scan` carry: the terms are batched over ``w``, so
    materializing them all at once would make this memory- rather than
    flop-bound.
    """
    if wide_b:
        parts = ft.partial(_expand_parts, per_point=True)
        return _exp_of_log_term(parts, a, b, (w, jnp.asarray(w0)))
    log_scale, scaled = _expand_parts(a, b, (w, w0), per_point=False)
    # w0^b = 2^-b for b >= -5: nowhere near overflow.
    return scaled * jnp.exp(log_scale)  # type: ignore[no-any-return]


def _expand_parts(
    a: AnyArrayLike,
    b: AnyArrayLike,
    xs: tuple[AnyArray, AnyArrayLike],
    /,
    *,
    per_point: bool,
) -> tuple[AnyArray, AnyArray]:
    """`_expand_from`'s sum and the log of its scale: the value is ``sum * e^log``."""
    w, w0 = xs
    # An array, weakly typed for a Python scalar: Python float arithmetic on a
    # concrete `b` (0.5 ** -1100) would raise `OverflowError`, not return inf.
    w0 = jnp.asarray(w0)
    log_w0 = jnp.log(w0)
    at_0 = w == 0
    # log w of a safe value at w = 0, so that no -inf sits under the masks.
    delta = log_w0 - jnp.log(jnp.where(at_0, w0, w))  # log(w0 / w) >= 0
    log_r_b = -b * delta  # log (w / w0)^b
    # Each term w0^(b+m) - w^(b+m) is scale * (A w0^m - B w^m), with
    # A = w0^b / scale and B = w^b / scale; 0^b = 0 for b > 0 at w = 0.
    if per_point:
        # The larger of w0^b and w^b, so that A, B <= 1. The choice cancels out
        # of the value and of every derivative, so the tie at w = w0 or b = 0
        # does not matter.
        up = jnp.maximum(log_r_b, 0.0)
        log_scale = b * log_w0 + up
        w0_factor: AnyArray | None = jnp.exp(-up)
        w_start = jnp.where(at_0, 0.0, jnp.exp(log_r_b - up))
    else:
        log_scale = b * log_w0
        w0_factor = None
        w_start = jnp.where(at_0, 0.0, jnp.exp(log_r_b))

    # The one step with |s| < `_NEAR` is skipped here and added afterwards; the
    # scan only records its coefficient, `s` and w0^m (scalars). Everything but
    # `total` and `w_m` is a scalar (and `w0_factor`, per point), so a step costs
    # each point a subtraction, a division by a scalar, a multiply-add and a
    # multiply (and one more multiply per point).
    Carry = tuple[AnyArrayLike, ...]

    def step(carry: Carry, m: ScalarLike, /) -> tuple[Carry, None]:
        # coeff = (1-a)_m / m!
        total, w_m, w0_m, coeff, near_coeff, near_s, near_w0_m = carry
        s = b + m
        in_band = jnp.abs(s) < _NEAR
        w0_term = w0_m if w0_factor is None else w0_factor * w0_m
        # Divided per point, not folded into one scalar coeff / s: reverse mode
        # would then form the cotangent ct * (w0^m - w_m) before the 1 / s^2
        # that tames it, and overflow where the value is finite but huge
        # (5.7e306 at a = 12, b = -200, z = 0.97).
        in_coeff = jnp.where(in_band, 0.0, coeff)
        total = total + in_coeff * ((w0_term - w_m) / jnp.where(in_band, 1.0, s))
        near_coeff = jnp.where(in_band, coeff, near_coeff)
        near_s = jnp.where(in_band, s, near_s)
        near_w0_m = jnp.where(in_band, w0_m, near_w0_m)
        next_coeff = coeff * (m + 1.0 - a) / (m + 1.0)
        carry = (total, w_m * w, w0_m * w0, next_coeff, near_coeff, near_s, near_w0_m)
        return carry, None

    zero = jnp.zeros_like(log_w0 * b)  # floating, weakly typed if both are
    one = jnp.ones_like(log_w0 * b)
    init = (jnp.zeros_like(w), w_start, one, jnp.ones_like(a), zero, zero, zero)
    (total, _, _, _, near_coeff, near_s, near_w0_m), _ = jax.lax.scan(
        step, init, jnp.arange(_NTERMS), unroll=_UNROLL
    )
    # The skipped step, relative to w0^b: with y = -s delta,
    # (w0^s - w^s) / s / w0^b = w0^m (1 - e^y) / s = w0^m delta expm1(y) / y,
    # which cancels for no s. At w = 0 there is nothing to cancel -- w^s = 0 for
    # s > 0 -- so it is w0^m / s, a scalar (s <= 0 at w = 0 is the pole, which
    # the callers handle; `s` is guarded only to keep `nan` out of the mask).
    # The scale there is w0^b, since w^b = 0.
    near = near_coeff * near_w0_m * delta * _expm1_over(-near_s * delta)
    if w0_factor is not None:
        near = w0_factor * near
    near_at_0 = near_coeff * near_w0_m / jnp.where(near_s == 0, 1.0, near_s)
    scaled = total + jnp.where(at_0, near_at_0, near)
    # At w = w0 the integral is exactly 0, but the sum leaves a rounding residue
    # (3e-28 at a = 50, b = -250), which the scale (e^814 there) overflowed.
    return log_scale, jnp.where(w == w0, 0.0, scaled)


_CF_FROM = 10.0
r"""Above this ``b``, `_large_b` replaces the two series.

Both series expand :math:`(1-t)^{b-1}` binomially, and its coefficients
:math:`(1-b)_k / k!` alternate and grow like :math:`b^k / k!`: at
:math:`z = 1/2` the sum is a near-total cancellation. Over :math:`a \le 8` the
series is still accurate at :math:`b = 10`
(`test_continuous_across_the_large_b_switch` holds it to the continued fraction
there at rtol :math:`10^{-13}`), loses about half its digits by :math:`b = 30`,
and all of them by 100. The continued fraction is accurate from :math:`b = 0.1` up
(tested to rtol :math:`10^{-12}` from :math:`b = 0.1` at large ``a`` in
`test_large_a_matches_mpmath`, and above :math:`b = 10` in
`test_large_b_matches_mpmath`), so the switch can sit wherever the series is
still accurate.
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
about :math:`10^{-11}` off at 32 steps and still short of rounding at 48; at 64
it is within the :math:`2\times10^{-14}` that
`test_continued_fraction_converged_near_x0` asserts, and more steps change
nothing. What remains is rounding, growing like :math:`b\,\epsilon`: tested to
rtol :math:`10^{-11}` out to :math:`b = 10^4` (`test_large_b_matches_mpmath`)
and :math:`10^{-12}` at :math:`a = 0.1` out to :math:`10^5`
(`test_large_b_small_a`).
"""


def _continued_fraction(
    a: AnyArrayLike, b: AnyArrayLike, x: AnyArray, /, steps: int = _CF_STEPS
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
        jnp.arange(1, steps + 1),
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
    # x0 itself, by `_log1m`, not `log` of a rounded `1 - x0` scaled by b.
    log_front = a * jnp.log(x0) + b * _log1m(jnp.asarray(x0))
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
    :math:`z^a (1-z)^b` is the same either way round and is formed from ``z``
    by `_log1m`, never from a rounded ``1 - z`` scaled by ``b``, nor from XLA's
    ``log1p`` where it is inaccurate; and the fraction is folded
    into the exponent, so a front factor below the smallest normal number does
    not flush to zero while the term itself is representable.
    """
    x0 = (a + 1.0) / (a + b + 2.0)
    swap = z >= x0
    # z = 0 and z = 1 make the term exactly zero (the direct one at 0, the
    # reflected one at 1). Evaluated, `log(0)` would put `0 * -inf` into the
    # a- and b-derivatives; it is a genuine zero for every a, b > 0, so its
    # parameter derivatives are zero too. The substitute is on the side the
    # endpoint takes, where that fraction converges: a fixed 1/2 ran the direct
    # fraction far above x0 at large b, where it went negative (-27.9 at
    # (2, 1000)), and its `log` made the masked gradients `nan`.
    endpoint = (z == 0) | (z == 1)
    zs = jnp.where(z == 0, 0.5 * x0, jnp.where(z == 1, 0.5 * (1.0 + x0), z))
    log_front = a * jnp.log(zs) + b * _log1m(zs)
    first = jnp.where(swap, b, a)
    h = _continued_fraction(first, jnp.where(swap, a, b), jnp.where(swap, 1.0 - zs, zs))
    log_term = log_front + jnp.log(h / first)
    term = jnp.where(endpoint, 0.0, jnp.exp(log_term))
    # B(a, b) - term, as B(a, b) (1 - term / B(a, b)) with the ratio from the
    # logarithms: near the bottom of the float range the term alone is
    # subnormal, which XLA flushes to 0, and B(301, 1000, 0.255) came back as
    # B(301, 1000), 2.4% high. The endpoint's term is a genuine 0.
    log_complete = _complete_beta(a, b)
    ratio = jnp.where(endpoint, -jnp.inf, log_term - log_complete)
    reflected = -jnp.exp(log_complete) * jnp.expm1(ratio)
    out = jnp.where(swap, reflected, term)
    # Past b ~ 1/eps the switch point x0 is below eps, `1 - x0` rounds to 1, and
    # the reflected fraction's first denominator is exactly zero. Long before
    # that the b*eps rounding has taken every digit (all of them by b ~ 1e15),
    # so say so with `nan` rather than return a confident `inf` or `0`.
    # Judged in z's dtype: a float64 x0 from Python-float a and b let float32
    # past 1/eps32 through, -1.5e-16 at (2, 1e8, 3.3e-8) where it is 8.4e-17.
    too_large = jnp.asarray(x0, z.dtype) < jnp.finfo(z.dtype).eps
    return jnp.where(too_large, jnp.nan, out)  # type: ignore[no-any-return]


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
large :math:`b`. Over :math:`b \in [-5, 10]`, the series' range below the
switch, its worst error sits just above the :math:`z = 1/2` seam: below
:math:`10^{-11}` at :math:`a = 8`, about :math:`10^{-8}` at 16, about
:math:`10^{-4}` at 24, and wrong from about 50. (Below :math:`b = -5` it is
`_negative_b`'s range: there the series is around :math:`10^{-11}` off at
:math:`b = -10` and wrong by percents at :math:`b = -50`.)
"""

_CF_A_B_MIN = 0.1
r"""For large ``a``: at or above this ``b``, `_large_b`; below it, `_large_a`.

`_large_b`'s reflection subtracts from the complete :math:`B(a, b) \sim 1/b`,
so it loses about :math:`\log_{10}(1/b)` digits as :math:`b \to 0^+`, and has
no meaning at :math:`b \le 0`, where :math:`B(a, b)` diverges. From 0.1 it is
tested to rtol :math:`10^{-12}` up to :math:`a = 50`
(`test_large_a_matches_mpmath`); below it `_large_a`, which needs no
reflection, takes over.
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
    (:math:`(a-1) w_0 < 2`), costing under a digit. Tested against mpmath to
    rtol :math:`5\times10^{-13}` over :math:`a \in [12, 100]`,
    :math:`b \in [-50, 0.1)` and :math:`z` to :math:`1 - 10^{-12}`
    (`test_large_a_small_b`), and to :math:`2\times10^{-13}` just above the
    split for :math:`b` down to -381.57 (`test_large_a_just_above_the_split`),
    where the series was about 9% off at :math:`(24, -50, 1/2)` and wrong by
    orders of magnitude at 100.

    At :math:`z = 1` with :math:`b \le 0` the integral diverges: the pole is
    returned as ``inf``, with the expansion handed a harmless ``w`` there. At
    :math:`z = 0` the term is a genuine zero, guarded as in `_large_b`.
    """
    # Arrays, weakly typed when ``a`` is a Python scalar, so `z` sets the dtype.
    # The result does not depend on where the split sits, so it carries no
    # a-derivative: differentiated, the two pieces' derivatives through it cancel
    # at the size of the value, leaving 3e-5 relative error in d/da at
    # (21.16, -5, 1 - 1e-12).
    split = jax.lax.stop_gradient(jnp.asarray(a / (a + 2.0)))
    # 1 - split, not 2/(a+2): the identity needs w0 = 1 - s exactly, and the
    # two roundings missed by an ulp, which |b| (a+2)/2 then amplified: 3.7e-13
    # just above the split at (100, -164.29), 5.9e-11 at a = 1e5.
    w0 = 1.0 - split
    return _split_at(a, b, z, split, w0)


_NEG_B_FROM = -5.0
r"""Below this ``b`` (for ``a`` up to `_CF_A_FROM`), `_negative_b` replaces the series.

`_small_z`'s terms carry :math:`(1-b)_k / k!`, which for very negative
:math:`b` grows like :math:`(|b| z)^k / k!` before it decays: 64 terms cover
:math:`|b| z \lesssim 10`. Over :math:`a \in [0.5, 8]` the series is still
accurate at :math:`b = -5` (`test_continuous_across_the_negative_b_switch`
holds it to the continued fraction there at rtol :math:`10^{-13}`); its worst
error, just above :math:`z = 1/2`, is around :math:`10^{-11}` at -10, about
half the digits by -20, wrong by percents at -50, and no correct digits by
-200.
"""

_NEG_B_STEPS = 128
"""Double steps of the continued fraction in `_negative_b`.

At very negative ``b`` and small ``a`` the fraction converges more slowly than
in `_large_b`: at ``a = 0.5, b = -1100``, 64 steps leave 5e-9 at ``z = 0.3``
and 2.4e-5 at ``z = 0.45``. 128 converge there, and 256 change nothing;
`test_very_negative_b` covers ``a = 0.5, b = -1100`` at rtol 2e-12.
"""


def _negative_b(a: ScalarLike, b: ScalarLike, z: AnyArray, /) -> AnyArray:
    r"""$B(a, b, z)$ for very negative $b$, by the continued fraction and the tail.

    No reflection -- it would need :math:`B(a, b)`, which diverges for
    :math:`b \le 0` -- and none needed: for :math:`b` this negative the fraction
    converges at every :math:`z \in [0, 1)` (tested against mpmath to rtol
    :math:`2\times10^{-12}` for :math:`a \in [0.01, 8]`,
    :math:`b \in [-1100, -5.5]` in `test_very_negative_b`). It is used up to
    :math:`z = 1/2`, and above that `_split_at` adds the binomial tail from
    :math:`w_0 = 1/2`, as `_large_z` does: the value is no better for it, but
    the fraction's :math:`a`-derivative lost up to :math:`1.6\times10^{-5}` near
    :math:`z = 1` (:math:`(0.16, -8.716, 1 - 10^{-9})`), and the tail's is tested
    to rtol :math:`2\times10^{-13}` there
    (`test_parameter_gradients_large_a_and_negative_b`). Below :math:`z = 1/2`
    the fraction's :math:`a`-derivative still loses digits as :math:`1/a` grows,
    several of them by :math:`a = 0.01`. The terms are formed in log space, as
    in `_large_b`, so a huge value that is still representable does not
    overflow on the way. :math:`z = 0` is a genuine zero; :math:`z = 1` the pole.
    """
    half = jnp.asarray(0.5)
    return _split_at(a, b, z, half, half, steps=_NEG_B_STEPS)


def _fraction_parts(
    a: AnyArrayLike, b: AnyArrayLike, x: AnyArray, /, steps: int = _CF_STEPS
) -> tuple[AnyArray, AnyArray]:
    """``B(a, b, x)`` by the continued fraction as ``(log z^a (1-z)^b, h / a)``."""
    log_front = a * jnp.log(x) + b * _log1m(x)
    return log_front, _continued_fraction(a, b, x, steps=steps) / a


Parts = Callable[[AnyArrayLike, AnyArrayLike, Any], tuple[AnyArray, AnyArray]]


def _log_of(log_scale: AnyArray, mant: AnyArray, /) -> AnyArray:
    """``log(mant * e^log_scale)``: ``-inf`` where ``mant`` is 0, `nan` below."""
    pos = mant > 0
    log_mant = jnp.log(jnp.where(pos, mant, 1.0))
    return jnp.where(  # type: ignore[no-any-return]
        pos, log_scale + log_mant, jnp.where(mant == 0, -jnp.inf, jnp.nan)
    )


def _value_of(log_scale: AnyArray, mant: AnyArray, /) -> AnyArray:
    """``mant * e^log_scale``: as a product while the scale is moderate.

    Through ``exp(log)`` only beyond that, so as not to spend ``|log| * eps``
    of accuracy on a round trip the common case does not need.
    """
    half_range = 0.5 * math.log(float(jnp.finfo(jnp.result_type(mant)).max))
    moderate = jnp.abs(log_scale) < half_range
    product = mant * jnp.exp(jnp.where(moderate, log_scale, 0.0))
    return jnp.where(moderate, product, jnp.exp(_log_of(log_scale, mant)))  # type: ignore[no-any-return]


@ft.partial(jax.custom_jvp, nondiff_argnums=(0,))
def _exp_of_log_term(
    parts: Parts, a: ScalarLike, b: ScalarLike, xs: Any, /
) -> AnyArray:
    r"""``mant * exp(log_scale)`` for ``parts(a, b, xs)``, with a safe reverse mode.

    Autodiff of :math:`e^L` hands :math:`L`'s graph the cotangent :math:`e^L`
    itself, which the fraction's recurrence or the binomial sum then amplifies:
    near the top of the float range, from about :math:`10^{297}`, the
    reverse-mode :math:`a`- and :math:`b`-gradients came out ``nan`` while the
    value and forward mode were fine. The tangent here is
    :math:`e^{L/2} \cdot \dot L(e^{L/2} \dot a, e^{L/2} \dot b)`, with
    :math:`L` the log of the value -- the same number, by linearity -- so
    neither side of the transposed map sees more than :math:`e^{L/2}`.

    Where the value is past the float range it is ``inf``, and its tangent is 0,
    as at the pole; that is also what keeps such a lane from making another
    lane's reverse-mode gradient ``nan`` (``0 * inf``) through the shared
    :math:`a` and :math:`b`.
    """
    return _value_of(*parts(a, b, xs))


@_exp_of_log_term.defjvp
def _exp_of_log_term_jvp(
    parts: Parts,
    primals: tuple[ScalarLike, ScalarLike, Any],
    tangents: tuple[ScalarLike, ScalarLike, Any],
    /,
) -> tuple[AnyArray, AnyArray]:
    a, b, xs = primals
    a_dot, b_dot, xs_dot = tangents
    log_scale, mant = parts(a, b, xs)
    log_value = _log_of(log_scale, mant)
    log_max = math.log(float(jnp.finfo(log_value.dtype).max))
    # False past the float range (and where the value is `nan`); True at -inf,
    # where the root is 0.
    finite = log_value <= log_max
    root = jnp.exp(jnp.where(finite, log_value, 0.0) / 2.0)

    def lane(
        primal: AnyArrayLike, tangent: AnyArrayLike, /
    ) -> tuple[AnyArray, AnyArray]:
        # Per lane, so that each lane's tangent can be scaled by its own root;
        # in the dtype the primal computation gives it, which for a weak
        # (Python) parameter is the other arguments', so that a float32 root
        # cannot meet a float64 primal (a `TypeError` in `jax.jvp`).
        p = primal + jnp.zeros_like(log_value)
        return p, (tangent * root).astype(p.dtype)

    a_l, a_t = lane(a, a_dot)
    b_l, b_t = lane(b, b_dot)
    pairs = jax.tree.map(lane, xs, xs_dot)
    xs_l = jax.tree.map(lambda _, pt: pt[0], xs, pairs)
    xs_t = jax.tree.map(lambda _, pt: pt[1], xs, pairs)
    _, log_dot = jax.jvp(
        lambda aa, bb, xx: _log_of(*parts(aa, bb, xx)),
        (a_l, b_l, xs_l),
        (a_t, b_t, xs_t),
    )
    tangent = jnp.where(finite, root * log_dot, 0.0)
    return _value_of(log_scale, mant), tangent.astype(log_value.dtype)


def _split_at(
    a: ScalarLike,
    b: ScalarLike,
    z: AnyArray,
    split: AnyArray,
    w0: AnyArray,
    /,
    steps: int = _CF_STEPS,
) -> AnyArray:
    r"""Sum the fraction below ``split``; above it, the fraction there and the tail.

    .. math::

        B(a, b, z) = B(a, b, s) + \int_{1-z}^{w_0} u^{b-1}(1-u)^{a-1}\,du,
        \qquad w_0 = 1 - s,

    the tail by `_expand_from`. For `_large_a` and `_negative_b`. Near
    :math:`z = 1` the fraction's :math:`a`-derivative is a small remainder of
    :math:`1/a` against :math:`\partial_a \ln h`, and lost 1e-5 of it at
    :math:`(0.1, -5.01, 1 - 10^{-9})`; the tail's has no such cancellation.

    Each lane's fraction is at ``z`` at or below the split and at the split above
    it, so no scalar is shared between lanes. A shared :math:`B(a, b, s)` that
    overflowed, as it does for :math:`b` below about :math:`-218` at
    :math:`a = 50`, put ``0 * inf`` into the reverse-mode :math:`a`- and
    :math:`b`-gradients of the lanes below the split, whose values are finite.

    At :math:`z = 1` with :math:`b \le 0` the integral diverges: the pole is
    returned as ``inf``, and `_expand_from` takes :math:`w = 0` safely itself.
    At :math:`z = 0` the term is a genuine zero; ``log(0)`` is kept out of it,
    so that no ``0 * -inf`` reaches the parameter derivatives.
    """
    pole = (z == 1.0) & (b <= 0)
    above = z > split
    w = jnp.where(above, 1.0 - z, w0)
    tail = _expand_from(a, b, w, w0, wide_b=True)
    # A point whose tail is not positive goes to the fraction, at z itself:
    # the first ulps above the split, where the tail's alternating sum rounds
    # to 0 or below. There its value and tangent are 0, while the fraction is
    # held at the split, so d/dz d/da and d/dz d/db came out 0 -- an ulp above
    # 12/14 at (12, -0.5). The fraction converges there: past the split, but
    # below its own switch point. Not the pole (z = 1, b <= 0), which keeps its
    # own handling below: at the fraction its log(1 - z) put `nan` into the
    # batch's shared a/b gradients. The tail is the one computed for the sum.
    above = above & ((jax.lax.stop_gradient(tail) > 0) | pole)
    at_0 = z == 0
    # At z = 0, a point below the split where z^a (1-z)^b cannot overflow
    # (|b| x <= 1/2): at the split itself (1 - s)^b overflowed for very negative
    # b, and its masked `inf` made the gradients `nan`.
    x_at_0 = 0.5 * split / (1.0 + jnp.abs(b))
    xs = jnp.where(at_0, x_at_0, jnp.where(above, split, z))
    term = _exp_of_log_term(ft.partial(_fraction_parts, steps=steps), a, b, xs)
    direct = jnp.where(at_0, 0.0, term)
    total = direct + jnp.where(above, tail, 0.0)
    # Past the float range the value is `inf` and its tangent 0, as at the pole.
    # The tail's own tangent is 0 there, but the fraction's, at the split, is
    # finite and passed through the sum: the a- and b-gradients at (2, -300,
    # 0.999) were those at z = 1/2, -2.4e87.
    total = jnp.where(jnp.isinf(total), jax.lax.stop_gradient(total), total)
    return jnp.where(pole, jnp.inf, total)  # type: ignore[no-any-return]


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
    # Plain Python floats, mirroring `_branch_index`: comparing a concrete jax
    # array would be staged into an enclosing `jit` trace and give a tracer.
    a, b = float(a), float(b)
    if b > _CF_FROM:
        return _CONTINUED_FRACTION
    if a > _CF_A_FROM:
        return _CONTINUED_FRACTION if b >= _CF_A_B_MIN else _LARGE_A
    return _NEGATIVE_B if b < _NEG_B_FROM else _SERIES


def _incomplete_beta_impl(
    a: ScalarLike, b: ScalarLike, z: AnyArrayLike, branch: int, /
) -> AnyArray:
    r"""$B(a, b, z)$ for $a > 0$, any real $b$, and $z \in [0, 1]$.

    The arguments are already promoted (`_promote`).
    """
    z = jnp.asarray(z)
    method = (_series, _large_b, _large_a, _negative_b)[branch]
    if jnp.dtype(z.dtype).itemsize < 4:
        # Half precision has too few bits for Lentz's recurrence or the series'
        # cancellations (8% off at (4, 9.5, 0.9) in float16), and too little
        # range for `_large_z`'s rescale (nan at (2, -5)): work in float32 and
        # round the result once.
        f32 = jnp.float32
        out = method(jnp.asarray(a, f32), jnp.asarray(b, f32), z.astype(f32))
        return out.astype(z.dtype)  # type: ignore[no-any-return]
    return method(a, b, z)  # type: ignore[no-any-return]


_EXPRL_BAND = 0.5
"""Below this ``|x|``, `_a_eq_1` sums ``expm1(x) / x`` as a series.

Above it the direct quotient is accurate in value and in its first three
derivatives (tested to rtol 1e-12 either side of the band in
`test_a_eq_1_higher_b_derivatives_across_the_series_band`); below it, those
derivatives cancel.
"""


_LOG1P_UPTO = 0.3
"""Below this ``z``, `_log1m` uses ``log1p(-z)``; above it, ``log(1 - z)``.

XLA's CPU ``log1p`` is off by up to ~240 ulp (2.7e-14) for arguments in
``(-0.42, -0.3)``, and the closed form multiplies that by ``|b L|``. From
``z = 0.3`` up, ``1 - z`` is exact to within half an ulp, and `_log1m` adds
back that half ulp, so ``log`` of it is good to about 1 ulp; below, ``log1p``
is. Either way about 2 ulp. (In float32 the
switch buys nothing -- there ``log1p`` is the better of the two, 1.6 against
2.6 ulp on that band -- but costs nothing either.)
"""


def _log1m(z: AnyArray, /) -> AnyArray:
    """``log(1 - z)`` to about 2 ulp, ``-inf`` at ``z = 1``; see `_LOG1P_UPTO`."""
    small = z <= _LOG1P_UPTO
    # Double `where`: each side only ever sees arguments it is accurate and finite
    # on, so neither puts a non-finite derivative under the other's mask.
    via_log1p = jnp.log1p(-jnp.where(small, z, 0.0))
    z_big = jnp.where(small, 0.5, z)
    # `1 - z` rounds by up to half an ulp below z = 1/2, and the closed form
    # multiplies that by |b|. `(1 - w) - z` is that rounding error, exactly
    # (both subtractions are exact by Sterbenz), and `log(w + d) = log(w) + d/w`
    # to far below an ulp. It keeps the closed form at a = 1 within the 1e-13
    # that `test_a_eq_1_in_the_log1p_band` asserts on (0.3, 0.42] at b = -500;
    # without it the error there was 1.1e-13, and 1.5e-13 once jax 0.12's `log`
    # lost a little accuracy. From z = 1/2 up `1 - z` is exact and `d` is 0; its
    # divisor is then 1, so z = 1 stays `-inf` rather than `-inf + 0/0`.
    w = 1.0 - z_big
    d = (1.0 - w) - z_big
    via_log = jnp.log(w) + d / jnp.where(d == 0.0, 1.0, w)
    return jnp.where(small, via_log1p, via_log)  # type: ignore[no-any-return]


def _a_eq_1(b: ScalarLike, z: AnyArrayLike, /) -> AnyArray:
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
def _a_eq_1_core(b: ScalarLike, z: AnyArrayLike, /) -> AnyArray:
    """`_a_eq_1`, with the same exact O(1) `z`-derivative as the series path."""
    return _a_eq_1(b, z)


@ft.partial(_a_eq_1_core.defjvp, symbolic_zeros=True)
def _a_eq_1_jvp(
    primals: tuple[ScalarLike, AnyArray],
    tangents: tuple[ScalarLike | SymbolicZero, AnyArray | SymbolicZero],
    /,
) -> tuple[AnyArray, AnyArray]:
    b, z = primals
    b_dot, z_dot = tangents

    # Through the custom JVP, so that differentiated again the primal takes
    # this rule too, not autodiff of the closed form's (1-z)^b (as in
    # `_incomplete_beta_jvp`).
    primal_out = _a_eq_1_core(b, z)
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
            integrand = jnp.where(inside, via_log, _pow(1.0 - z_arr, b - 1.0))
        else:
            # A concrete `b` has no derivative to get wrong, and as a constant
            # exponent XLA can simplify `pow` (to a reciprocal at b = 0, a square
            # root at b = 3/2), which `exp(... log1p)` would hide from it.
            # `_pow`, not `**`: its z-derivatives at z = 1 are exact for an
            # integer b - 1 >= 0, where `**` made them `nan` (GH-73).
            integrand = _pow(1.0 - z, b - 1.0)
        tangent_out = tangent_out + integrand * z_dot

    # Elementary, so autodiff of the closed form is cheap and exact; the series
    # near b = 0 keeps every order right at the removable singularity.
    if not isinstance(b_dot, SymbolicZero):
        _, b_tangent = jax.jvp(lambda bb: _a_eq_1(bb, z), (b,), (b_dot,))
        tangent_out = tangent_out + b_tangent

    return primal_out, tangent_out


def _is_static_one(a: object, /) -> bool:
    """Whether ``a`` is a concrete scalar equal to 1, decidable at trace time."""
    return (
        not isinstance(a, jax.core.Tracer)
        and np.ndim(a) == 0
        and np.isrealobj(a)
        and bool(np.equal(a, 1))
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
    :math:`2^{-k}`; see `_small_z` and `_large_z`. Outside :math:`a \le 8`,
    :math:`-5 \le b \le 10` their binomial coefficients cancel or grow, and
    three fixed-length methods take over: the continued fraction with
    reflection (`_large_b`: :math:`b > 10`, or :math:`a > 8` with
    :math:`b \ge 0.1`), `_large_a` (:math:`a > 8`, :math:`b < 0.1`), and the
    continued fraction without reflection (`_negative_b`: :math:`a \le 8`,
    :math:`b < -5`).

    The tests check those three methods against mpmath in float64: to rtol
    :math:`10^{-12}` for :math:`b \in (10, 100]` and for :math:`a \in (8, 50]`
    with :math:`b \ge 0.1`, :math:`5 \times 10^{-13}` for
    :math:`a \in [12, 100]` with :math:`b \in [-50, 0.1)`,
    :math:`2 \times 10^{-12}` for :math:`a \le 8` with
    :math:`b \in [-1100, -5)`, and :math:`10^{-11}` for :math:`b` up to
    :math:`10^4`, beyond which rounding grows like :math:`b\,\epsilon`. In the
    series regime they assert :math:`10^{-11}`, about how closely the two
    series meet at :math:`z = 1/2`. The full table is in the
    accuracy-and-domains reference page.

    The :math:`z`-derivative is supplied by a `jax.custom_jvp` and is exact and
    O(1) -- by Leibniz it is just the integrand at the endpoint,
    :math:`z^{a-1}(1-z)^{b-1}` -- rather than differentiating through 64 terms.
    It is a `custom_jvp` rather than a `custom_vjp` so that `jax.hessian`'s
    ``jacfwd(jacrev(...))`` still composes. The rules sit on inner cores:
    `incomplete_beta` itself is a plain function that dispatches between them,
    so it has no ``.fun`` or ``.defjvp`` of its own.

    When ``a`` is a concrete 1 (a Python or NumPy scalar, or a concrete 0-d
    array; not a traced value) the integral is elementary,
    :math:`(1 - (1-z)^b)/b`, and that closed form replaces the series or
    continued fraction; see `_a_eq_1`. It carries the same exact
    :math:`z`-derivative rule, and its ``b``-derivative is autodiff of the
    closed form, which is elementary. A traced ``a`` cannot be inspected and
    takes the branch dispatch below even at 1: as accurate there as the branch
    it takes, but slower, so keep ``a`` concrete for the closed form.

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
    used instead:

    >>> round(float(sp.incomplete_beta(1.0, 0.0, jnp.asarray(0.5))), 8)
    0.69314718

    """
    # Decided at trace time, so it costs nothing when it does not apply. A
    # traced `a` -- e.g. one being differentiated or `vmap`ped -- falls through
    # to the branch dispatch below.
    if _is_static_one(a):
        if np.ndim(b) != 0:
            # The branch dispatch cannot broadcast `b` either (it raises inside
            # its `scan`); say so here too, rather than answer for some `a` only.
            msg = f"incomplete_beta: `b` must be a scalar, got shape {np.shape(b)}"
            raise TypeError(msg)
        # Promoted as on the other paths, so a strong lower-precision `b` is
        # cast up rather than running the closed form in its precision.
        _, b, z = _promote(a, b, z)
        out_dtype = z.dtype
        if jnp.dtype(out_dtype).itemsize < 4:
            # Half precision in float32, rounded once, as on the other paths:
            # in bfloat16 the closed form was 4% off at (-10, 0.6).
            f32 = jnp.float32
            b32 = b if isinstance(b, int | float) else jnp.asarray(b, f32)
            return _a_eq_1_core(b32, z.astype(f32)).astype(out_dtype)
        return _a_eq_1_core(b, z)
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
    # Under `vmap` the `switch` becomes a `select` that runs every branch on
    # every lane; the batched `switch` selects each lane's cotangent after the
    # branches' transposes, so an unselected branch's `nan` does not reach the
    # gradients (`test_vmap_over_b_across_the_switch_has_finite_gradients`).
    branches = [ft.partial(_incomplete_beta_core, a, b, z, k) for k in range(4)]
    return jax.lax.switch(index, branches)  # type: ignore[no-any-return]


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
        # Through the custom JVP itself, not the implementation: differentiated
        # again, the primal then takes this rule too. Through the implementation
        # it differentiated the raw series -- 10x the flops of a traced-b second
        # z-derivative -- and its z**a at z = 0, a = 2 is `0 * inf` by the fifth,
        # which jax 0.12 no longer pruned inside `_dispatch`'s `switch`: `nan`.
        primal_out = _incomplete_beta_core(a, b, z, branch)
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
        tangent_out = tangent_out + _integrand(a, b, z) * z_dot

    return primal_out, tangent_out


def _integrand(a: ScalarLike, b: ScalarLike, z: AnyArrayLike, /) -> AnyArray:
    """``z^(a-1) (1-z)^(b-1)``, the z-derivative of ``B(a, b, z)``.

    For a traced ``a`` or ``b``, or exponents for which one factor could leave
    the float range while the product does not, inside the interval as one
    exponential,
    ``exp((a-1) log z + (b-1) log(1-z))``:
    formed apart, one factor underflowed and the other overflowed at large ``a``
    and very negative ``b``, and their product was ``0 * inf = nan`` where the
    derivative is 4.0, at (1100, -1100, 1/2). ``log(1-z)`` is `_log1m`, so the
    ``b``-derivative keeps full precision as ``z -> 0`` (GH-68). At ``z = 0`` and
    ``z = 1`` `_pow` takes over, whose derivatives are exact at a zero base
    (GH-73). Each side is handed exponents of 0 where it is not selected, so
    neither puts an ``inf`` under the other's mask. Half precision is evaluated
    in float32 and rounded once: at float16 ``z = 0.95``, ``(1-z)^-4`` alone is
    past float16's range while the derivative is not.
    """
    z = jnp.asarray(z)
    if jnp.dtype(z.dtype).itemsize < 4:
        f32 = jnp.float32
        return _integrand(  # type: ignore[no-any-return]
            jnp.asarray(a, f32), jnp.asarray(b, f32), z.astype(f32)
        ).astype(z.dtype)
    # Concrete a and b keep the plain product of powers where it is safe, as
    # its constant exponents XLA simplifies (to z * rsqrt(1 - z) at (2, 1.5)):
    # the exponential costs 13x the flops of that z-gradient. Safe means either
    # exponents of one sign -- both factors <= 1, or both >= 1, so a factor
    # leaves the float range only where the product does too -- or exponents
    # too small for a factor to leave it where the other differs from 1: the
    # nearest z gets to 1 is an ulp, 2^-(nmant+1), so |q| <= maxexp / (nmant+1)
    # keeps (1-z)^q in range (19 in float64, 5 in float32), and z^p with |p|
    # that small leaves it only below 2^-(nmant+1), where 1 - z is exactly 1. A
    # bound of log(max)/log 2 instead left (1-z)^-1023 at z = 0.502 to overflow
    # while z^1023 could not bring it back: inf where the derivative is 3583.
    if not isinstance(a, jax.core.Tracer) and not isinstance(b, jax.core.Tracer):
        p, q = float(a) - 1.0, float(b) - 1.0
        info = jnp.finfo(z.dtype)
        small = info.maxexp / (info.nmant + 1)
        if p * q >= 0 or (abs(p) <= small and abs(q) <= small):
            return _pow(z, a - 1.0) * _pow(1.0 - z, b - 1.0)  # type: ignore[no-any-return]
    end = (z == 0) | (z == 1)
    inner_z = jnp.where(end, 0.5, z)
    inner = jnp.exp(
        jnp.where(end, 0.0, a - 1.0) * jnp.log(inner_z)
        + jnp.where(end, 0.0, b - 1.0) * _log1m(inner_z)
    )
    end_z = jnp.where(end, z, 0.5)
    at_end = _pow(end_z, jnp.where(end, a - 1.0, 0.0)) * _pow(
        1.0 - end_z, jnp.where(end, b - 1.0, 0.0)
    )
    return jnp.where(end, at_end, inner)  # type: ignore[no-any-return]
