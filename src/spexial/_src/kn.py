"""Modified Bessel functions of the second kind, integer order."""

__all__ = ["K0", "K1", "K2", "K0e", "K1e", "K2e"]

from math import log
from typing import Any, Final

import jax
import jax.numpy as jnp
from jax import lax
from jax.scipy.special import gammaln, i0, i0e, i1e

from .custom_types import AnyArray, RealArrayLike
from .dtype import as_float as _as_float, cast_like as _cast_like

_EULER_GAMMA: Final = 0.57721566490153286061
"""The Euler-Mascheroni constant."""

_LN2: Final = 0.6931471805599453
"""log(2), subtracted rather than dividing `z` by 2; see `_K0_small`."""

_INT_OF_WIDTH: Final = {2: jnp.int16, 4: jnp.int32, 8: jnp.int64}
"""Signed integer of the same width as each float dtype, for `_log_no_flush`."""


def _positive_subnormal(z: AnyArray) -> AnyArray:
    """Mask of the arguments XLA has flushed to zero but that are not zero.

    The float tests cannot do this. XLA compares a subnormal as if it were
    zero, so ``z > 0`` is False for exactly these values and ``z == 0.0`` is
    True for them -- which is how a subnormal argument reached `K1`'s pole
    guard and came back ``inf``.
    """
    bits = lax.bitcast_convert_type(z, _INT_OF_WIDTH[jnp.dtype(z.dtype).itemsize])
    return (bits > 0) & (z < jnp.finfo(z.dtype).tiny)


def _exactly_zero(z: AnyArray) -> AnyArray:
    """``z == 0.0`` done on the bits, so a subnormal is not mistaken for zero.

    Only two bit patterns are zero, ``+0.0`` and ``-0.0``; the latter is the
    single integer more negative than every other float.
    """
    bits = lax.bitcast_convert_type(z, _INT_OF_WIDTH[jnp.dtype(z.dtype).itemsize])
    return (bits == 0) | (bits == jnp.iinfo(bits.dtype).min)


def _log_no_flush(z: AnyArray) -> AnyArray:
    """``log(z)``, including where ``z`` is subnormal and XLA has flushed it.

    XLA on CPU flushes a subnormal *input* to zero, so `jnp.log` returns
    ``-inf`` for every ``z`` below ``finfo(dtype).tiny`` -- and `K0` then
    returned ``inf`` where the true value is an ordinary number near 700. In
    float32 that band starts at 1.2e-38, an entirely reachable magnitude.

    A subnormal's bit pattern still holds its mantissa; only arithmetic on it
    flushes. Reading the bits as an integer therefore recovers it, and a
    subnormal is exactly ``mantissa * tiny / 2**nmant``, so its logarithm is
    ``log(mantissa)`` plus a constant. `jnp.frexp` is not an alternative -- it
    flushes too, and reports the same exponent for every subnormal.

    Note this is distinct from the `_LN2` subtraction in `_K0_small`, which
    stops a *normal* ``z`` being halved into the subnormal range. That fix does
    nothing when the argument arrives subnormal already.
    """
    info = jnp.finfo(z.dtype)
    bits = lax.bitcast_convert_type(z, _INT_OF_WIDTH[jnp.dtype(z.dtype).itemsize])
    mantissa = jnp.bitwise_and(bits, (1 << info.nmant) - 1).astype(z.dtype)
    # `bits > 0` is the sign test, done on the integer because the float one
    # cannot be: XLA compares a subnormal as if it were zero, so `z > 0` is
    # False for exactly the values this branch exists to catch. It is also why
    # the magnitude test has to be `z < tiny` rather than `abs(z) < tiny` --
    # and why, without the sign test, every negative argument took this branch
    # and came back `inf` instead of `nan`.
    subnormal = (bits > 0) & (z < info.tiny)
    # Every negative except `-0.0`, whose bit pattern is the one integer more
    # negative than all of them. A negative *subnormal* cannot be recognised any
    # other way -- it compares equal to zero, so `jnp.log` returned `-inf` for
    # it and `K0` came back `inf` where the argument is simply out of domain.
    negative = (bits < 0) & (bits != jnp.iinfo(bits.dtype).min)
    from_bits = jnp.log(mantissa) + (log(float(info.tiny)) - info.nmant * _LN2)
    # Every branch evaluates, so keep `log` off the flushed value.
    plain = jnp.log(jnp.where(subnormal, info.tiny, z))
    return jnp.where(negative, jnp.nan, jnp.where(subnormal, from_bits, plain))


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


def _K0_small(z: AnyArray) -> AnyArray:
    """Ascending series for `K0`; see Zhang & Jin, *Special Functions* (1996)."""
    # `dtype=z.dtype`: a bare `arange(1.0, n)` is float64 whenever x64 is on, which
    # promoted the whole series and returned float64 from a float32 argument.
    k = jnp.arange(1.0, _N_SMALL + 1.0, dtype=z.dtype)
    harmonic = jnp.cumsum(1.0 / k)
    # `log(z) - log(2)`, never `log(z / 2)`: halving a z that is merely small --
    # but perfectly normal -- lands in the subnormal range, which XLA on CPU
    # flushes to zero, and `log(0)` is `-inf`. That turned the whole finite band
    # 2.2e-308 <= z < 4.45e-308 into `inf` (and `K1` into `nan`) where the true
    # values are ~708 and ~3e307. Subtracting instead touches no small number.
    log_half_z = _log_no_flush(z) - _LN2
    # `z[..., None]` sums over a *trailing* axis: without it `jnp.sum` collapses
    # the caller's own axis and an array argument silently yields one scalar.
    log_term = 2.0 * k * log_half_z[..., None] - 2.0 * gammaln(k + 1.0)
    return -(log_half_z + _EULER_GAMMA) * i0(z) + jnp.sum(
        harmonic * jnp.exp(log_term), axis=-1
    )


def _K0e_large(z: AnyArray) -> AnyArray:
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

    `2 / -0.0` is `-inf`, which turns `K2`'s `K0e + (2/z) K1e` into
    `inf - inf == nan` -- at a point ordinary arithmetic reaches, since
    `jnp.asarray(0.0) * -1` is `-0.0`. Taking `abs` first is correct over the
    whole domain and costs nothing: for `z > 0` it is the identity, at either
    zero it gives `+inf`, and for `z < 0` -- the only place the sign could
    matter -- `K0e` and `K1e` are already `nan`, so the result is `nan` either
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
def K0e(z: RealArrayLike, /) -> AnyArray:
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

    >>> round(float(sp.K0e(1.0)), 8)
    1.14446308

    Where `K0` has underflowed to zero, the scaled form is still exact:

    >>> float(sp.K0(800.0))
    0.0
    >>> round(float(sp.K0e(800.0)), 10)
    0.0443044275

    """
    _, small, z_small, z_large = _split(z)
    out = jnp.where(small, _K0_small(z_small) * jnp.exp(z_small), _K0e_large(z_large))
    return _cast_like(out, z)


@jax.custom_jvp
def K1e(z: RealArrayLike, /) -> AnyArray:
    r"""Compute the exponentially scaled :math:`e^z K_1(z)`.

    Equivalent to ``scipy.special.k1e(z)``, which has no JAX counterpart.
    Obtained from `K0e` through the Wronskian
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

    >>> round(float(sp.K1e(1.0)), 8)
    1.63615349
    >>> round(float(sp.K1e(800.0)), 10)
    0.0443321091

    """
    z_arr = _as_float(z)
    # K1 diverges at 0, but the closed form evaluates to
    # `1/0 - i1e(0) * K0e(0) == inf - 0 * inf == nan` there. At +inf it is
    # `(0 - 0 * 0) / 0 == nan` for the same reason `K0e` needs a guard.
    # Substitute both limits.
    at_zero, at_inf = _exactly_zero(z_arr), z_arr == jnp.inf
    # `z_arr` itself, not a substituted `z_safe`. The degenerate points are
    # overwritten by the `where` below, so the substitution bought nothing --
    # and it cost a great deal: `K0e(z_safe)` is a *different* subgraph from the
    # `K0e(z_arr)` its callers evaluate, so `K2`, `K2e` and every JVP that needs
    # both ran the 30-term series twice with no CSE available.
    z_safe = z_arr
    # The Wronskian gives `(1/z - i1e K0e) / i0e`; this is that identity with
    # numerator and denominator both multiplied by z. Algebraically the same,
    # but every term stays normal: `1/z` alone goes subnormal above z = 4.5e307
    # (as does `i1e * K0e`, which is also ~1/2z), and both flushed to zero, so
    # the unmultiplied form returned exactly 0 from there up. Multiplied
    # through, the numerator tends to 1/2 and the denominator to sqrt(z/2pi).
    k1e = (1.0 - z_safe * i1e(z_safe) * K0e(z_safe)) / (z_safe * i0e(z_safe))
    # Below `tiny` the denominator flushes to zero and the quotient is `inf`,
    # where the true value is `1/z` -- still representable for the factor of
    # about two between `tiny` and `1/max`, which in float32 is the reachable
    # band 2.9e-39 to 1.2e-38. `e^z K_1(z) -> 1/z` there to relative order
    # `z**2`, and the logarithm is the one form that can read a flushed
    # argument at all. Past that band `1/z` overflows and `inf` is correct.
    subnormal = _positive_subnormal(z_arr)
    k1e = jnp.where(subnormal, jnp.exp(-_log_no_flush(z_arr)), k1e)
    out = jnp.where(at_zero, jnp.inf, jnp.where(at_inf, 0.0, k1e))
    return _cast_like(out, z)


@jax.custom_jvp
def K2e(z: RealArrayLike, /) -> AnyArray:
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

    >>> round(float(sp.K2e(1.0)), 8)
    4.41677005
    >>> round(float(sp.K2e(800.0)), 10)
    0.0444152578

    """
    z_arr = _as_float(z)
    return _cast_like(K0e(z_arr) + _two_over(z_arr) * K1e(z_arr), z)


@jax.custom_jvp
def K0(z: RealArrayLike, /) -> AnyArray:
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
        above ``z = 705.5``, where the true value is subnormal; use `K0e` there.

    Examples
    --------
    >>> import jax.numpy as jnp
    >>> import spexial as sp

    >>> round(float(sp.K0(1.0)), 8)
    0.42102444

    Array input is evaluated elementwise, spanning both branches:

    >>> [round(float(k), 8) for k in sp.K0(jnp.asarray([0.5, 5.0, 20.0]))]
    [0.92441907, 0.0036911, 0.0]

    """
    _, small, z_small, z_large = _split(z)
    out = jnp.where(small, _K0_small(z_small), _K0e_large(z_large) * jnp.exp(-z_large))
    return _cast_like(out, z)


@jax.custom_jvp
def K1(z: RealArrayLike, /) -> AnyArray:
    """Compute the modified Bessel function of the second kind of order 1.

    Obtained from `K0` through the Wronskian
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
        above ``z = 705.5``, where the true value is subnormal; use `K1e` there.

    Examples
    --------
    >>> import jax.numpy as jnp
    >>> import spexial as sp

    >>> round(float(sp.K1(1.0)), 8)
    0.60190723

    >>> [round(float(k), 8) for k in sp.K1(jnp.asarray([0.5, 5.0, 20.0]))]
    [1.65644112, 0.00404461, 0.0]

    """
    z_arr = _as_float(z)
    return _cast_like(K1e(z_arr) * jnp.exp(-z_arr), z)


@jax.custom_jvp
def K2(z: RealArrayLike, /) -> AnyArray:
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
        above ``z = 705.5``, where the true value is subnormal; use `K2e` there.

    Examples
    --------
    >>> import jax.numpy as jnp
    >>> import spexial as sp

    >>> round(float(sp.K2(1.0)), 8)
    1.6248389

    >>> [round(float(k), 8) for k in sp.K2(jnp.asarray([0.5, 5.0, 20.0]))]
    [7.55018355, 0.00530894, 0.0]

    """
    # The recurrence is applied in the *scaled* variables and undone once.
    # Evaluated directly, the `(2/z) K1` term drops into the subnormal range
    # around z = 699 -- where XLA on CPU flushes it to zero, silently losing a
    # 0.3% contribution (2850x the documented tolerance) while still returning a
    # plausible number. `K0e` and `K1e` are order 1e-2 there, so the sum is
    # formed entirely in normal arithmetic and only the result is scaled down.
    z_arr = _as_float(z)
    return _cast_like(K2e(z_arr) * jnp.exp(-z_arr), z)


# Analytic derivatives. Letting JAX differentiate through the 30-term ascending
# series and the 10-term asymptotic expansion works, but costs about twice as
# much as evaluating the closed form -- measured 637us -> 310us for `grad` over
# 1000 points. Each identity below was checked against the autodiff result to
# ~1e-8, well inside these functions' own ~1e-6 accuracy.
#
# Standard recurrence Kv'(z) = -K_{v-1}(z) - (v/z) K_v(z), which at v = 0, 1, 2
# gives K0' = -K1, K1' = -K0 - K1/z and K2' = -K1 - (2/z) K2. The scaled forms
# pick up the extra `+ Kn e` term from differentiating the `e^z` factor.


# Each rule returns `_cast_like(..., z)` for the primal *and* for the derivative
# factor. Without it the two disagree: `_K2_jvp` narrowed only its primal, so
# `grad(K2)` on a bfloat16 argument raised outright ("Custom JVP rule must
# produce primal and tangent outputs with corresponding ... dtypes"), while the
# other rules narrowed neither and quietly handed `jax.jvp` a wider primal than
# the plain call returns.


@K0.defjvp
def _K0_jvp(primals: tuple[Any], tangents: tuple[Any]) -> tuple[AnyArray, AnyArray]:
    """K0'(z) = -K1(z)."""
    (z,), (dz,) = primals, tangents
    return K0(z), _cast_like(-K1(z), z) * dz


@jax.custom_jvp
def _dK1(z: AnyArray) -> AnyArray:
    """K1'(z) = -K0(z) - K1(z)/z, summed scaled.

    A named function with its own rule rather than an expression inside
    `_K1_jvp`, so that differentiating it *again* also gets a scaled sum.
    Left as raw arithmetic, `grad(grad(K1))` formed `d(1/z) * K1 * e^-z`, which
    is ~4e-312 at z = 700 -- subnormal, so XLA flushed it and the second
    derivative came out 7.2e-4 low. Exactly the bug the first derivative was
    fixed for, one order up.
    """
    return -(K0e(z) + 0.5 * _two_over(z) * K1e(z)) * jnp.exp(-z)


@_dK1.defjvp
def _dK1_jvp(primals: tuple[Any], tangents: tuple[Any]) -> tuple[AnyArray, AnyArray]:
    """K1''(z) = K1(z) + K0(z)/z + 2 K1(z)/z^2."""
    (z,), (dz,) = primals, tangents
    two_over = _two_over(z)
    second = (K1e(z) + 0.5 * two_over * K0e(z) + 0.5 * two_over**2 * K1e(z)) * jnp.exp(
        -z
    )
    return _dK1(z), second * dz


@jax.custom_jvp
def _dK2(z: AnyArray) -> AnyArray:
    """K2'(z) = -K1(z) - (2/z) K2(z), summed scaled. See `_dK1`."""
    two_over = _two_over(z)
    k0e, k1e = K0e(z), K1e(z)
    return -(k1e + two_over * (k0e + two_over * k1e)) * jnp.exp(-z)


@_dK2.defjvp
def _dK2_jvp(primals: tuple[Any], tangents: tuple[Any]) -> tuple[AnyArray, AnyArray]:
    """K2''(z) = K0(z) + 3 K1(z)/z + 6 K2(z)/z^2."""
    (z,), (dz,) = primals, tangents
    two_over = _two_over(z)
    second = (K0e(z) + 1.5 * two_over * K1e(z) + 1.5 * two_over**2 * K2e(z)) * jnp.exp(
        -z
    )
    return _dK2(z), second * dz


@K1.defjvp
def _K1_jvp(primals: tuple[Any], tangents: tuple[Any]) -> tuple[AnyArray, AnyArray]:
    """K1'(z) = -K0(z) - K1(z) / z, summed scaled in `_dK1`."""
    (z,), (dz,) = primals, tangents
    return K1(z), _cast_like(_dK1(_as_float(z)), z) * dz


@K2.defjvp
def _K2_jvp(primals: tuple[Any], tangents: tuple[Any]) -> tuple[AnyArray, AnyArray]:
    """K2'(z) = -K1(z) - (2/z) K2(z), summed scaled in `_dK2`."""
    (z,), (dz,) = primals, tangents
    return K2(z), _cast_like(_dK2(_as_float(z)), z) * dz


# At z = 0 each of these is a difference of two infinities, so the closed form
# gives `nan` where the true one-sided limit is `-inf` -- which is what the
# unscaled `K0`/`K1`/`K2` rules already return, since theirs have a single
# divergent term. Substituted so the two families agree at the pole.


def _at_pole(z: AnyArray, deriv: AnyArray, limit: float = -jnp.inf) -> AnyArray:
    """`limit` wherever the closed form degenerates on the non-negative axis.

    At ``z = 0`` each of these rules is a difference of two infinities. So is
    the whole band ``0 < z <~ 6.7e-155``, where the scaled values themselves
    overflow to `inf` and `K2e - K1e - (2/z) K2e` becomes `inf - inf` -- a
    guard on ``z == 0`` alone left `grad(K2e)` returning `nan` there while
    `grad(K2)` and `grad(K1e)` both returned the true limit. Keyed on the
    result being `nan` rather than on a magnitude threshold, so it cannot go
    stale. Negative `z` keeps its `nan`: that is outside the domain, not a pole.

    First derivatives tend to `-inf` there and second derivatives to `+inf`, the
    `1/z^2` term dominating, so the three second-derivative rules pass
    ``limit=jnp.inf``.
    """
    return jnp.where((z >= 0.0) & jnp.isnan(deriv), limit, deriv)


@jax.custom_jvp
def _dK0e(z: AnyArray) -> AnyArray:
    """(e^z K0)' = e^z (K0 - K1). See `_dK1` for why this is a named function."""
    return _at_pole(z, K0e(z) - K1e(z))


@_dK0e.defjvp
def _dK0e_jvp(primals: tuple[Any], tangents: tuple[Any]) -> tuple[AnyArray, AnyArray]:
    """(e^z K0)'' = 2 e^z K0 - 2 e^z K1 + e^z K1 / z."""
    (z,), (dz,) = primals, tangents
    k0e, k1e = K0e(z), K1e(z)
    second = 2.0 * k0e - 2.0 * k1e + 0.5 * _two_over(z) * k1e
    return _dK0e(z), _at_pole(z, second, jnp.inf) * dz


@jax.custom_jvp
def _dK1e(z: AnyArray) -> AnyArray:
    """(e^z K1)' = e^z K1 - e^z K0 - e^z K1 / z."""
    return _at_pole(z, K1e(z) - K0e(z) - 0.5 * _two_over(z) * K1e(z))


@_dK1e.defjvp
def _dK1e_jvp(primals: tuple[Any], tangents: tuple[Any]) -> tuple[AnyArray, AnyArray]:
    """(e^z K1)'' = 2G1 - 2G0 - 2G1/z + G0/z + 2G1/z^2, with Gn = e^z K_n."""
    (z,), (dz,) = primals, tangents
    k0e, k1e, half = K0e(z), K1e(z), 0.5 * _two_over(z)
    second = 2.0 * k1e - 2.0 * k0e - 2.0 * half * k1e + half * k0e + 2.0 * half**2 * k1e
    return _dK1e(z), _at_pole(z, second, jnp.inf) * dz


@jax.custom_jvp
def _dK2e(z: AnyArray) -> AnyArray:
    """(e^z K2)' = e^z K2 - e^z K1 - (2/z) e^z K2."""
    return _at_pole(z, K2e(z) - K1e(z) - _two_over(z) * K2e(z))


@_dK2e.defjvp
def _dK2e_jvp(primals: tuple[Any], tangents: tuple[Any]) -> tuple[AnyArray, AnyArray]:
    """(e^z K2)'' = G0 - 2G1 + G2 + 3G1/z - 4G2/z + 6G2/z^2."""
    (z,), (dz,) = primals, tangents
    k0e, k1e, k2e, half = K0e(z), K1e(z), K2e(z), 0.5 * _two_over(z)
    second = (
        k0e
        - 2.0 * k1e
        + k2e
        + 3.0 * half * k1e
        - 4.0 * half * k2e
        + 6.0 * half**2 * k2e
    )
    return _dK2e(z), _at_pole(z, second, jnp.inf) * dz


@K0e.defjvp
def _K0e_jvp(primals: tuple[Any], tangents: tuple[Any]) -> tuple[AnyArray, AnyArray]:
    """(e^z K0)' = e^z (K0 - K1)."""
    (z,), (dz,) = primals, tangents
    z_arr = _as_float(z)
    return _cast_like(K0e(z_arr), z), _cast_like(_dK0e(z_arr), z) * dz


@K1e.defjvp
def _K1e_jvp(primals: tuple[Any], tangents: tuple[Any]) -> tuple[AnyArray, AnyArray]:
    """(e^z K1)' = e^z K1 - e^z K0 - e^z K1 / z."""
    (z,), (dz,) = primals, tangents
    z_arr = _as_float(z)
    return _cast_like(K1e(z_arr), z), _cast_like(_dK1e(z_arr), z) * dz


@K2e.defjvp
def _K2e_jvp(primals: tuple[Any], tangents: tuple[Any]) -> tuple[AnyArray, AnyArray]:
    """(e^z K2)' = e^z K2 - e^z K1 - (2/z) e^z K2."""
    (z,), (dz,) = primals, tangents
    z_arr = _as_float(z)
    return _cast_like(K2e(z_arr), z), _cast_like(_dK2e(z_arr), z) * dz
