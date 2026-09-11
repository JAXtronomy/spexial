"""Modified Bessel functions of the second kind, integer order."""

__all__ = ["k0", "k0e", "k1", "k1e", "k2", "k2e"]

from typing import Any, Final

import jax
import jax.numpy as jnp
from jax.scipy.special import gammaln, i0, i0e, i1e

from .custom_types import AnyArray, RealArrayLike
from .dtype import (
    as_float as _as_float,
    cast_like as _cast_like,
    is_negative,
    log_no_flush as _log_no_flush,
)

_EULER_GAMMA: Final = 0.57721566490153286061
"""The Euler-Mascheroni constant."""

_LN2: Final = 0.6931471805599453
"""log(2), subtracted rather than dividing `z` by 2; see `_k0_small`."""


_SMALL_Z: Final = 9.0
"""Cross-over between the ascending series and the asymptotic expansion, in
float64. See `_SMALL_Z`; float32 has to hand off far earlier."""

_SMALL_Z_LOW_PRECISION: Final = 4.65
"""Cross-over in float32, chosen by measurement rather than scaled from 9.

The ascending series evaluates `-(log(z/2) + gamma) I0(z) + sum(...)`, whose two
terms are both ~e^z/sqrt(z) and cancel down to a result of ~e^-z -- a loss of
roughly `2z/ln(10)` decimal digits. float64 has 16 to spend, so it still has 8
left at z = 9. float32 has 7, and at z = 9 it has *none*: the result came out
**negative**. 4.65 is where the two branches' float32 errors cross, capping the
worst at 7.1e-3 over the whole domain (measured over 12,000 points, 6,000 of
them in [4.0, 5.2])."""

_N_SMALL: Final = 30
"""Terms in the ascending series; enough for ~1e-8 relative accuracy at z < 9."""

_N_LARGE: Final = 10
"""Terms in the asymptotic series; enough for ~1e-8 relative accuracy at z > 9."""


def _k0_small(z: AnyArray) -> AnyArray:
    """Ascending series for `k0`; see Zhang & Jin, *Special Functions* (1996)."""
    # `dtype=z.dtype`: a bare `arange(1.0, n)` is float64 whenever x64 is on, which
    # promoted the whole series and returned float64 from a float32 argument.
    k = jnp.arange(1.0, _N_SMALL + 1.0, dtype=z.dtype)
    harmonic = jnp.cumsum(1.0 / k)
    # `log(z) - log(2)`, never `log(z / 2)`: halving a z that is merely small --
    # but perfectly normal -- lands in the subnormal range, which XLA on CPU
    # flushes to zero, and `log(0)` is `-inf`. That turned the whole finite band
    # 2.2e-308 <= z < 4.45e-308 into `inf` (and `k1` into `nan`) where the true
    # values are ~708 and ~3e307. Subtracting instead touches no small number.
    log_half_z = _log_no_flush(z) - _LN2
    # `z[..., None]` sums over a *trailing* axis: without it `jnp.sum` collapses
    # the caller's own axis and an array argument silently yields one scalar.
    log_term = 2.0 * k * log_half_z[..., None] - 2.0 * gammaln(k + 1.0)
    return -(log_half_z + _EULER_GAMMA) * i0(z) + jnp.sum(
        harmonic * jnp.exp(log_term), axis=-1
    )


def _k0e_large(z: AnyArray) -> AnyArray:
    """Asymptotic expansion for :math:`e^z K_0(z)`, via ``1 / (2 z I0e(z))``.

    Written against `jax.scipy.special.i0e` -- the exponentially scaled
    :math:`I_0` -- rather than `i0`, which overflows just above z = 709.78 and
    used to cap these functions there. Nothing in the scaled form overflows or
    underflows, at any z.
    """
    # `i0e(inf) == 0`, so the quotient would be `inf * 0 == nan`; every K_n
    # tends to 0 at +inf and every scipy counterpart returns that.
    at_inf = z == jnp.inf
    z = jnp.where(at_inf, 1.0, z)
    k = jnp.arange(1.0, _N_LARGE + 1.0, dtype=z.dtype)
    prod = jnp.cumprod(-(2.0 * k - 1.0) / (2.0 * k) * (2.0 * k - 1.0) ** 2.0)
    series = 1.0 + jnp.sum(
        (-1.0) ** k * prod / (2.0 * z[..., None]) ** (2.0 * k), axis=-1
    )
    # Grouped as `2 * (z * i0e(z))`, not `2 * z * i0e(z)`: the latter forms
    # `2 * z` first, which overflows to `inf` above z = DBL_MAX/2 and sent the
    # whole quotient to 0 from z = 8.99e307. `z * i0e(z)` is ~sqrt(z / 2pi) and
    # overflows nowhere.
    return jnp.where(at_inf, 0.0, series / (2.0 * (z * i0e(z))))


def _two_over(z: AnyArray) -> AnyArray:
    """``2 / z``, with `-0.0` treated as the same pole as `+0.0`.

    `2 / -0.0` is `-inf`, which turns `k2`'s `k0e + (2/z) k1e` into
    `inf - inf == nan` -- at a point ordinary arithmetic reaches, since
    `jnp.asarray(0.0) * -1` is `-0.0`. Taking `abs` first is correct over the
    whole domain and costs nothing: for `z > 0` it is the identity, at either
    zero it gives `+inf`, and for `z < 0` -- the only place the sign could
    matter -- `k0e` and `k1e` are already `nan`, so the result is `nan` either
    way.
    """
    return 2.0 / jnp.abs(z)


def _split(z: RealArrayLike) -> tuple[AnyArray, AnyArray, AnyArray, AnyArray]:
    """Both branch arguments, each already made safe for the other's domain."""
    z_arr = _as_float(z)
    # Keyed on `eps`, not on the dtype name, so any low-precision type gets the
    # conservative cross-over rather than inheriting one chosen for float64.
    eps = jnp.finfo(z_arr.dtype).eps
    cut = _SMALL_Z if eps < 1e-10 else _SMALL_Z_LOW_PRECISION
    small = z_arr < cut
    return z_arr, small, jnp.where(small, z_arr, 1.0), jnp.where(small, cut, z_arr)


@jax.custom_jvp
def k0e(z: RealArrayLike, /) -> AnyArray:
    r"""Compute the exponentially scaled :math:`e^z K_0(z)`.

    Equivalent to ``scipy.special.k0e(z)``, which has no JAX counterpart. This
    is the form to reach for beyond ``z = 705``, where :math:`K_0(z)` itself is
    smaller than any normal double and unrepresentable; :math:`e^z K_0(z)`
    decays only as :math:`1/\sqrt{z}` and stays accurate at any ``z``.

    Parameters
    ----------
    z
        Real positive argument, of any shape. Evaluated elementwise. ``z == 0``
        is the pole and gives ``inf``; ``z < 0`` is outside the domain and gives
        `nan`; ``z = inf`` gives 0, the limit.

    Returns
    -------
    Array
        Value(s) of :math:`e^z K_0(z)`, accurate to ~2.0e-7 relative
        (worst at z = 8.9984, just below the ``z = 9`` cross-over; ~8e-9 out to z = 15,
        ~2e-13 to z = 30, and ~1e-15 beyond). Those are float64
        figures: in float32 the cross-over moves to 4.65 and the worst error is
        ~7e-3 (see `_SMALL_Z`). `float16` and `bfloat16` are computed in
        float32 and rounded back, so they get what their dtype can hold.

    Examples
    --------
    >>> import spexial as sp

    >>> round(float(sp.k0e(1.0)), 8)
    1.14446308

    Where `k0` has underflowed to zero, the scaled form is still exact:

    >>> float(sp.k0(800.0))
    0.0
    >>> round(float(sp.k0e(800.0)), 10)
    0.0443044275

    """
    _, small, z_small, z_large = _split(z)
    out = jnp.where(small, _k0_small(z_small) * jnp.exp(z_small), _k0e_large(z_large))
    return _cast_like(out, z)


@jax.custom_jvp
def k1e(z: RealArrayLike, /) -> AnyArray:
    r"""Compute the exponentially scaled :math:`e^z K_1(z)`.

    Equivalent to ``scipy.special.k1e(z)``, which has no JAX counterpart.
    Obtained from `k0e` through the Wronskian
    :math:`I_0(z) K_1(z) + I_1(z) K_0(z) = 1/z`, in the scaled variables.

    Parameters
    ----------
    z
        Real positive argument, of any shape. Evaluated elementwise. ``z == 0``
        is the pole and gives ``inf``; ``z < 0`` is outside the domain and gives
        `nan`; ``z = inf`` gives 0, the limit.

    Returns
    -------
    Array
        Value(s) of :math:`e^z K_1(z)`, accurate to ~1.8e-7 relative
        (worst at z = 8.9984, just below the ``z = 9`` cross-over; ~8e-9 out to z = 15,
        ~2e-13 to z = 30, and ~1e-15 beyond). Those are float64
        figures: in float32 the cross-over moves to 4.65 and the worst error is
        ~7e-3 (see `_SMALL_Z`). `float16` and `bfloat16` are computed in
        float32 and rounded back, so they get what their dtype can hold.

    Examples
    --------
    >>> import spexial as sp

    >>> round(float(sp.k1e(1.0)), 8)
    1.63615349
    >>> round(float(sp.k1e(800.0)), 10)
    0.0443321091

    """
    z_arr = _as_float(z)
    # k1 diverges at 0, but the closed form evaluates to
    # `1/0 - i1e(0) * k0e(0) == inf - 0 * inf == nan` there. At +inf it is
    # `(0 - 0 * 0) / 0 == nan` for the same reason `k0e` needs a guard.
    # Substitute both limits.
    at_inf = z_arr == jnp.inf
    # `z_arr` itself, not a substituted `z_safe`. The degenerate points are
    # overwritten by the `where` below, so the substitution bought nothing --
    # and it cost a great deal: `k0e(z_safe)` is a *different* subgraph from the
    # `k0e(z_arr)` its callers evaluate, so `k2`, `k2e` and every JVP that needs
    # both ran the 30-term series twice with no CSE available.
    z_safe = z_arr
    # The Wronskian gives `(1/z - i1e k0e) / i0e`; this is that identity with
    # numerator and denominator both multiplied by z. Algebraically the same,
    # but every term stays normal: `1/z` alone goes subnormal above z = 4.5e307
    # (as does `i1e * k0e`, which is also ~1/2z), and both flushed to zero, so
    # the unmultiplied form returned exactly 0 from there up. Multiplied
    # through, the numerator tends to 1/2 and the denominator to sqrt(z/2pi).
    k1e = (1.0 - z_safe * i1e(z_safe) * k0e(z_safe)) / (z_safe * i0e(z_safe))
    # Below `tiny` the denominator flushes to zero and the quotient is `inf`,
    # where the true value is `1/z` -- still representable for the factor of
    # about two between `tiny` and `1/max`, which in float32 is the reachable
    # band 2.9e-39 to 1.2e-38. `e^z K_1(z) -> 1/z` there to relative order
    # `z**2`, and the logarithm is the one form that can read a flushed
    # argument at all. Past that band `1/z` overflows and `inf` is correct.
    # `z_arr == 0.0` is the *flushed* comparison, on purpose: it is True for
    # every argument at or below `tiny`, which is exactly the set this branch
    # should own, and it behaves identically eager and under `jit`. Separating
    # the pole from the subnormals with a bit test does not survive XLA's
    # fusion -- `exactly_zero` is correct in isolation and wrong when the select
    # is its only consumer, which is how `k1` came to return `inf` across the
    # whole band under `jit` while eager was right. Nothing here needs the
    # distinction anyway: `e^z K_1(z) -> 1/z`, and `1/0` is the pole's `inf`.
    #
    # `log_no_flush` supplies all three answers from the one expression: `-inf`
    # at zero, so `exp` gives `inf`; the true logarithm across the subnormal
    # band, so `exp` gives `1/z`; and `nan` for a negative argument, which is
    # out of domain. `-0.0` is not negative and correctly gives `inf`.
    small = z_arr == 0.0
    k1e = jnp.where(small, jnp.exp(-_log_no_flush(z_arr)), k1e)
    out = jnp.where(at_inf, 0.0, k1e)
    return _cast_like(out, z)


@jax.custom_jvp
def k2e(z: RealArrayLike, /) -> AnyArray:
    r"""Compute the exponentially scaled :math:`e^z K_2(z)`.

    Equivalent to ``scipy.special.kve(2, z)``, which has no JAX counterpart.
    Obtained from the recurrence :math:`K_2(z) = K_0(z) + (2/z) K_1(z)`, which
    the scaling leaves unchanged.

    Parameters
    ----------
    z
        Real positive argument, of any shape. Evaluated elementwise. ``z == 0``
        is the pole and gives ``inf``; ``z < 0`` is outside the domain and gives
        `nan`; ``z = inf`` gives 0, the limit.

    Returns
    -------
    Array
        Value(s) of :math:`e^z K_2(z)`, accurate to ~1.3e-7 relative
        (worst at z = 8.9984, just below the ``z = 9`` cross-over; ~8e-9 out to z = 15,
        ~2e-13 to z = 30, and ~1e-15 beyond). Those are float64
        figures: in float32 the cross-over moves to 4.65 and the worst error is
        ~7e-3 (see `_SMALL_Z`). `float16` and `bfloat16` are computed in
        float32 and rounded back, so they get what their dtype can hold.

    Examples
    --------
    >>> import spexial as sp

    >>> round(float(sp.k2e(1.0)), 8)
    4.41677005
    >>> round(float(sp.k2e(800.0)), 10)
    0.0444152578

    """
    z_arr = _as_float(z)
    return _cast_like(k0e(z_arr) + _two_over(z_arr) * k1e(z_arr), z)


@jax.custom_jvp
def k0(z: RealArrayLike, /) -> AnyArray:
    """Compute the modified Bessel function of the second kind of order 0.

    Equivalent to ``scipy.special.kn(0, z)``. See Zhang and Jin,
    ``SPECIAL_FUNCTIONS`` in FORTRAN77, for the algorithm: an ascending series
    below ``z = 9`` and an asymptotic expansion above it.

    Parameters
    ----------
    z
        Real positive argument, of any shape. Evaluated elementwise. ``z == 0``
        is the pole and gives ``inf``; ``z < 0`` is outside the domain and gives
        `nan`; ``z = inf`` gives 0, the limit.

    Returns
    -------
    Array
        Value(s) of :math:`K_0(z)`, accurate to ~2.0e-7 relative
        (worst at z = 8.9984, just below the ``z = 9`` cross-over; ~8e-9 out to z = 15,
        ~2e-13 to z = 30, and ~1e-15 beyond). Those are float64
        figures: in float32 the cross-over moves to 4.65 and the worst error is
        ~7e-3 (see `_SMALL_Z`). `float16` and `bfloat16` are computed in
        float32 and rounded back, so they get what their dtype can hold. Underflows to 0
        where the true value falls below the *dtype's* smallest normal, which is a
        different place in each: 705.3 in float64, 85.3 in float32,
        85.2 in bfloat16 and 16.1 in float16. Use `k0e` above it.

    Examples
    --------
    >>> import jax.numpy as jnp
    >>> import spexial as sp

    >>> round(float(sp.k0(1.0)), 8)
    0.42102444

    Array input is evaluated elementwise, spanning both branches:

    >>> [round(float(k), 8) for k in sp.k0(jnp.asarray([0.5, 5.0, 20.0]))]
    [0.92441907, 0.0036911, 0.0]

    """
    _, small, z_small, z_large = _split(z)
    out = jnp.where(small, _k0_small(z_small), _k0e_large(z_large) * jnp.exp(-z_large))
    return _cast_like(out, z)


@jax.custom_jvp
def k1(z: RealArrayLike, /) -> AnyArray:
    """Compute the modified Bessel function of the second kind of order 1.

    Obtained from `k0` through the Wronskian
    :math:`I_0(z) K_1(z) + I_1(z) K_0(z) = 1/z`.

    Parameters
    ----------
    z
        Real positive argument, of any shape. Evaluated elementwise. ``z == 0``
        is the pole and gives ``inf``; ``z < 0`` is outside the domain and gives
        `nan`; ``z = inf`` gives 0, the limit.

    Returns
    -------
    Array
        Value(s) of :math:`K_1(z)`, accurate to ~1.8e-7 relative
        (worst at z = 8.9984, just below the ``z = 9`` cross-over; ~8e-9 out to z = 15,
        ~2e-13 to z = 30, and ~1e-15 beyond). Those are float64
        figures: in float32 the cross-over moves to 4.65 and the worst error is
        ~7e-3 (see `_SMALL_Z`). `float16` and `bfloat16` are computed in
        float32 and rounded back, so they get what their dtype can hold. Underflows to 0
        where the true value falls below the *dtype's* smallest normal, which is a
        different place in each: 705.3 in float64, 85.3 in float32,
        85.2 in bfloat16 and 16.2 in float16. Use `k1e` above it.

    Examples
    --------
    >>> import jax.numpy as jnp
    >>> import spexial as sp

    >>> round(float(sp.k1(1.0)), 8)
    0.60190723

    >>> [round(float(k), 8) for k in sp.k1(jnp.asarray([0.5, 5.0, 20.0]))]
    [1.65644112, 0.00404461, 0.0]

    """
    z_arr = _as_float(z)
    return _cast_like(k1e(z_arr) * jnp.exp(-z_arr), z)


@jax.custom_jvp
def k2(z: RealArrayLike, /) -> AnyArray:
    """Compute the modified Bessel function of the second kind of order 2.

    Obtained from the recurrence :math:`K_2(z) = K_0(z) + (2/z) K_1(z)`.

    Parameters
    ----------
    z
        Real positive argument, of any shape. Evaluated elementwise. ``z == 0``
        is the pole and gives ``inf``; ``z < 0`` is outside the domain and gives
        `nan`; ``z = inf`` gives 0, the limit.

    Returns
    -------
    Array
        Value(s) of :math:`K_2(z)`, accurate to ~1.3e-7 relative
        (worst at z = 8.9984, just below the ``z = 9`` cross-over; ~8e-9 out to z = 15,
        ~2e-13 to z = 30, and ~1e-15 beyond). Those are float64
        figures: in float32 the cross-over moves to 4.65 and the worst error is
        ~7e-3 (see `_SMALL_Z`). `float16` and `bfloat16` are computed in
        float32 and rounded back, so they get what their dtype can hold. Underflows to 0
        where the true value falls below the *dtype's* smallest normal, which is a
        different place in each: 705.3 in float64, 85.4 in float32,
        85.2 in bfloat16 and 16.3 in float16. Use `k2e` above it.

    Examples
    --------
    >>> import jax.numpy as jnp
    >>> import spexial as sp

    >>> round(float(sp.k2(1.0)), 8)
    1.6248389

    >>> [round(float(k), 8) for k in sp.k2(jnp.asarray([0.5, 5.0, 20.0]))]
    [7.55018355, 0.00530894, 0.0]

    """
    # The recurrence is applied in the *scaled* variables and undone once.
    # Evaluated directly, the `(2/z) k1` term drops into the subnormal range
    # around z = 699 -- where XLA on CPU flushes it to zero, silently losing a
    # 0.3% contribution (2850x the documented tolerance) while still returning a
    # plausible number. `k0e` and `k1e` are order 1e-2 there, so the sum is
    # formed entirely in normal arithmetic and only the result is scaled down.
    z_arr = _as_float(z)
    return _cast_like(k2e(z_arr) * jnp.exp(-z_arr), z)


# Analytic derivatives. Letting JAX differentiate through the 30-term ascending
# series and the 10-term asymptotic expansion works, but costs about twice as
# much as evaluating the closed form -- measured 637us -> 310us for `grad` over
# 1000 points. Each identity below was checked against the autodiff result to
# ~1e-8, well inside these functions' own ~1e-6 accuracy.
#
# Standard recurrence Kv'(z) = -K_{v-1}(z) - (v/z) K_v(z), which at v = 0, 1, 2
# gives k0' = -k1, k1' = -k0 - k1/z and k2' = -k1 - (2/z) k2. The scaled forms
# pick up the extra `+ Kn e` term from differentiating the `e^z` factor.


# Each rule returns `_cast_like(..., z)` for the primal *and* for the derivative
# factor. Without it the two disagree: `_k2_jvp` narrowed only its primal, so
# `grad(k2)` on a bfloat16 argument raised outright ("Custom JVP rule must
# produce primal and tangent outputs with corresponding ... dtypes"), while the
# other rules narrowed neither and quietly handed `jax.jvp` a wider primal than
# the plain call returns.


@k0.defjvp
def _k0_jvp(primals: tuple[Any], tangents: tuple[Any]) -> tuple[AnyArray, AnyArray]:
    """k0'(z) = -k1(z)."""
    (z,), (dz,) = primals, tangents
    return k0(z), _cast_like(-k1(z), z) * dz


@jax.custom_jvp
def _dk1(z: AnyArray) -> AnyArray:
    """k1'(z) = -k0(z) - k1(z)/z, summed scaled.

    A named function with its own rule rather than an expression inside
    `_k1_jvp`, so that differentiating it *again* also gets a scaled sum.
    Left as raw arithmetic, `grad(grad(k1))` formed `d(1/z) * k1 * e^-z`, which
    is ~4e-312 at z = 700 -- subnormal, so XLA flushed it and the second
    derivative came out 7.2e-4 low. Exactly the bug the first derivative was
    fixed for, one order up.
    """
    return -(k0e(z) + 0.5 * _two_over(z) * k1e(z)) * jnp.exp(-z)


@_dk1.defjvp
def _dk1_jvp(primals: tuple[Any], tangents: tuple[Any]) -> tuple[AnyArray, AnyArray]:
    """k1''(z) = k1(z) + k0(z)/z + 2 k1(z)/z^2."""
    (z,), (dz,) = primals, tangents
    two_over = _two_over(z)
    second = (k1e(z) + 0.5 * two_over * k0e(z) + 0.5 * two_over**2 * k1e(z)) * jnp.exp(
        -z
    )
    return _dk1(z), second * dz


@jax.custom_jvp
def _dk2(z: AnyArray) -> AnyArray:
    """k2'(z) = -k1(z) - (2/z) k2(z), summed scaled. See `_dk1`."""
    two_over = _two_over(z)
    g0, g1 = k0e(z), k1e(z)
    return -(g1 + two_over * (g0 + two_over * g1)) * jnp.exp(-z)


@_dk2.defjvp
def _dk2_jvp(primals: tuple[Any], tangents: tuple[Any]) -> tuple[AnyArray, AnyArray]:
    """k2''(z) = k0(z) + 3 k1(z)/z + 6 k2(z)/z^2."""
    (z,), (dz,) = primals, tangents
    two_over = _two_over(z)
    second = (k0e(z) + 1.5 * two_over * k1e(z) + 1.5 * two_over**2 * k2e(z)) * jnp.exp(
        -z
    )
    return _dk2(z), second * dz


@k1.defjvp
def _k1_jvp(primals: tuple[Any], tangents: tuple[Any]) -> tuple[AnyArray, AnyArray]:
    """k1'(z) = -k0(z) - k1(z) / z, summed scaled in `_dk1`."""
    (z,), (dz,) = primals, tangents
    return k1(z), _cast_like(_dk1(_as_float(z)), z) * dz


@k2.defjvp
def _k2_jvp(primals: tuple[Any], tangents: tuple[Any]) -> tuple[AnyArray, AnyArray]:
    """k2'(z) = -k1(z) - (2/z) k2(z), summed scaled in `_dk2`."""
    (z,), (dz,) = primals, tangents
    return k2(z), _cast_like(_dk2(_as_float(z)), z) * dz


# At z = 0 each of these is a difference of two infinities, so the closed form
# gives `nan` where the true one-sided limit is `-inf` -- which is what the
# unscaled `k0`/`k1`/`k2` rules already return, since theirs have a single
# divergent term. Substituted so the two families agree at the pole.


def _at_pole(z: AnyArray, deriv: AnyArray, limit: float = -jnp.inf) -> AnyArray:
    """`limit` wherever the closed form degenerates on the non-negative axis.

    At ``z = 0`` each of these rules is a difference of two infinities. So is
    the whole band ``0 < z <~ 6.7e-155``, where the scaled values themselves
    overflow to `inf` and `k2e - k1e - (2/z) k2e` becomes `inf - inf` -- a
    guard on ``z == 0`` alone left `grad(k2e)` returning `nan` there while
    `grad(k2)` and `grad(k1e)` both returned the true limit. Keyed on the
    result not being *finite* rather than on a magnitude threshold, so it
    cannot go stale. Negative `z` keeps its `nan`: that is outside the domain,
    not a pole, and so does a `nan` argument.

    Not-finite rather than `nan`, because which of the two an ``inf - inf``
    comes out as is a property of the graph and not of the arithmetic: XLA
    reassociates the sum under `jit`, so ``jit(grad(grad(k2e)))`` was `-inf`
    across ``3.2e-154 <~ z <~ 5.6e-103`` where eager was `nan` and the true
    limit is ``+inf`` -- the same value disagreeing with itself between the two
    modes, and with the sign flipped in the mode users actually run. Every
    ``e^z K_n`` decreases on ``z > 0``, so the only non-finite derivative
    available on the domain is the pole's own, and substituting it is a no-op
    wherever the closed form already found it.

    First derivatives tend to `-inf` there and second derivatives to `+inf`, the
    `1/z^2` term dominating, so the three second-derivative rules pass
    ``limit=jnp.inf``.
    """
    # `~is_negative(z)`, not `z >= 0.0`: XLA compares a subnormal equal to
    # zero, so a *negative* subnormal satisfied `z >= 0.0` and was handed the
    # pole limit -- the scaled gradients returned `-inf` for an argument whose
    # value is `nan`. This function's own docstring says negative `z` keeps its
    # `nan`, and that is exactly what failed.
    # `~jnp.isnan(z)`, because a `nan` argument makes `deriv` `nan` too and so
    # walked straight into the pole substitution: `grad(k0e)(nan)` was `-inf`
    # and the second derivative `+inf`, for an argument whose *value* is `nan`
    # and which SciPy's `kve` also calls `nan`. Worse, `-nan` came back `nan`,
    # so the answer turned on a sign bit that carries no meaning.
    at_pole = ~is_negative(z) & ~jnp.isnan(z) & ~jnp.isfinite(deriv)
    return jnp.where(at_pole, limit, deriv)


@jax.custom_jvp
def _dk0e(z: AnyArray) -> AnyArray:
    """(e^z k0)' = e^z (k0 - k1). See `_dk1` for why this is a named function."""
    return _at_pole(z, k0e(z) - k1e(z))


@_dk0e.defjvp
def _dk0e_jvp(primals: tuple[Any], tangents: tuple[Any]) -> tuple[AnyArray, AnyArray]:
    """(e^z k0)'' = 2 e^z k0 - 2 e^z k1 + e^z k1 / z."""
    (z,), (dz,) = primals, tangents
    g0, g1 = k0e(z), k1e(z)
    second = 2.0 * g0 - 2.0 * g1 + 0.5 * _two_over(z) * g1
    return _dk0e(z), _at_pole(z, second, jnp.inf) * dz


@jax.custom_jvp
def _dk1e(z: AnyArray) -> AnyArray:
    """(e^z k1)' = e^z k1 - e^z k0 - e^z k1 / z."""
    return _at_pole(z, k1e(z) - k0e(z) - 0.5 * _two_over(z) * k1e(z))


@_dk1e.defjvp
def _dk1e_jvp(primals: tuple[Any], tangents: tuple[Any]) -> tuple[AnyArray, AnyArray]:
    """(e^z k1)'' = 2G1 - 2G0 - 2G1/z + G0/z + 2G1/z^2, with Gn = e^z K_n."""
    (z,), (dz,) = primals, tangents
    g0, g1, half = k0e(z), k1e(z), 0.5 * _two_over(z)
    second = 2.0 * g1 - 2.0 * g0 - 2.0 * half * g1 + half * g0 + 2.0 * half**2 * g1
    return _dk1e(z), _at_pole(z, second, jnp.inf) * dz


@jax.custom_jvp
def _dk2e(z: AnyArray) -> AnyArray:
    """(e^z k2)' = e^z k2 - e^z k1 - (2/z) e^z k2."""
    return _at_pole(z, k2e(z) - k1e(z) - _two_over(z) * k2e(z))


@_dk2e.defjvp
def _dk2e_jvp(primals: tuple[Any], tangents: tuple[Any]) -> tuple[AnyArray, AnyArray]:
    """(e^z k2)'' = G0 - 2G1 + G2 + 3G1/z - 4G2/z + 6G2/z^2."""
    (z,), (dz,) = primals, tangents
    g0, g1, g2, half = k0e(z), k1e(z), k2e(z), 0.5 * _two_over(z)
    second = g0 - 2.0 * g1 + g2 + 3.0 * half * g1 - 4.0 * half * g2 + 6.0 * half**2 * g2
    return _dk2e(z), _at_pole(z, second, jnp.inf) * dz


@k0e.defjvp
def _k0e_jvp(primals: tuple[Any], tangents: tuple[Any]) -> tuple[AnyArray, AnyArray]:
    """(e^z k0)' = e^z (k0 - k1)."""
    (z,), (dz,) = primals, tangents
    z_arr = _as_float(z)
    return _cast_like(k0e(z_arr), z), _cast_like(_dk0e(z_arr), z) * dz


@k1e.defjvp
def _k1e_jvp(primals: tuple[Any], tangents: tuple[Any]) -> tuple[AnyArray, AnyArray]:
    """(e^z k1)' = e^z k1 - e^z k0 - e^z k1 / z."""
    (z,), (dz,) = primals, tangents
    z_arr = _as_float(z)
    return _cast_like(k1e(z_arr), z), _cast_like(_dk1e(z_arr), z) * dz


@k2e.defjvp
def _k2e_jvp(primals: tuple[Any], tangents: tuple[Any]) -> tuple[AnyArray, AnyArray]:
    """(e^z k2)' = e^z k2 - e^z k1 - (2/z) e^z k2."""
    (z,), (dz,) = primals, tangents
    z_arr = _as_float(z)
    return _cast_like(k2e(z_arr), z), _cast_like(_dk2e(z_arr), z) * dz
