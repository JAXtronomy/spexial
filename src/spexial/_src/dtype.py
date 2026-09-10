"""Shared dtype plumbing: promote narrow inputs, then give the caller its own back.

Note that this module is NOT public API nor are any of its contents.
Stability is NOT guaranteed.

Two functions need this, for the same underlying reason: `float16` and
`bfloat16` carry 3.3 and 2.4 decimal digits, which is not enough to hold the
intermediate cancellation in either a 30-term ascending series (`kn`) or a
difference of log-gammas (`comb`). Both compute one width up and round back, so
the caller keeps its dtype and gets the accuracy that dtype can represent.

"""

__all__: tuple[str, ...] = ()

from functools import partial
from math import log
from typing import Any, Final

import jax
import jax.numpy as jnp
from jax import lax

from .custom_types import AnyArray, AnyArrayLike


def as_float(x: AnyArrayLike, /, *, keep_weak: bool = False) -> AnyArray:
    """Promote to at least float32, without disturbing a subnormal.

    The promotion is not cosmetic. In `kn`, `float16` and `bfloat16` have no
    working series/asymptotic cross-over at all -- `k0` in bfloat16 was wrong by
    16x and *negative* over part of its range. In `comb`, the log-gamma
    difference has lost every digit by ``N = 20`` in bfloat16, returning `1.0`
    for a true 124750 by ``N = 500``.

    Floats are converted with `astype` rather than multiplied by ``1.0``. The
    multiply looks like a no-op and is not: XLA on CPU flushes a subnormal
    operand to zero, so ``z * 1.0`` silently zeroed every `kn` argument below
    ``tiny``, which then came back as `inf` from ``log(0)``. `astype` is a
    conversion, not arithmetic, and leaves the value alone. Integers have no
    subnormal to lose and still need the multiply to become floats at all.

    ``keep_weak`` chooses between two things that cannot both be had, and the
    two callers need opposite ones.

    `comb` needs ``True``. `polylog` calls it with plain Python ``int``
    arguments from inside a `lax.scan`, and a *weakly* typed float64 is what
    stops those promoting the loop carry from complex64 to complex128 -- which
    it did, with an error naming neither function.

    `kn` needs ``False``. Its `custom_jvp` rules are annotated `AnyArray`, and
    a weak scalar reaches them still wrapped as a Python float, which the
    runtime type checker rejects. Only an actual conversion unwraps it, and
    `astype` is skipped when it would not widen -- so the weak-preserving path
    has to be the one that is asked for, not the default.

    This deliberately does *not* normalise ``-0.0``. It used to, with
    ``where(x == 0.0, 0.0, x)`` -- which cost a select on every call, and,
    because XLA's comparison treats subnormals as zero, silently mapped every
    subnormal argument to an exact zero as well.
    """
    x_arr = jnp.asarray(x)
    if not jnp.issubdtype(x_arr.dtype, jnp.floating):
        x_arr = x_arr * 1.0
    target = jnp.promote_types(x_arr.dtype, jnp.float32)
    if keep_weak and x_arr.dtype == target:
        return x_arr
    return x_arr.astype(target)


def promote_integers(x: AnyArrayLike, /) -> AnyArray:
    """Make integer and boolean input floating, and leave everything else alone.

    The narrower cousin of `as_float`, for callers that want nothing except
    integer promotion -- `gamma` delegates to a `jax.scipy.special` function
    that already handles every float width itself, so widening `float16` would
    only throw away the caller's dtype.

    The point is what it does *not* do. ``x * 1.0`` is the obvious spelling and
    is wrong for a float: on XLA it flushes a subnormal to zero, which is the
    hazard this module exists to document and the one that cost `kn.k0` its
    whole subnormal band. Integers have no subnormal to lose, so they can take
    the multiply -- and need it, since that is what makes them floats at all.
    """
    x_arr = jnp.asarray(x)
    if jnp.issubdtype(x_arr.dtype, jnp.inexact):
        return x_arr
    return x_arr * 1.0


def cast_like(out: AnyArray, x: AnyArrayLike, /) -> AnyArray:
    """Return the caller's own floating dtype, whatever width we computed in.

    Covers two separate widenings: the deliberate one `as_float` performs, and
    the accidental one where a series constant defaults to float64 under x64 and
    promotes a float32 argument. Integer input has no float dtype to return to
    and stays promoted.
    """
    dtype = jnp.asarray(x).dtype
    if not jnp.issubdtype(dtype, jnp.floating):
        return out  # integer input has no float dtype to go back to
    narrower = jnp.finfo(dtype).bits < jnp.finfo(out.dtype).bits
    return out.astype(dtype) if narrower else out


_LN2: Final = 0.6931471805599453
"""log(2)."""

INT_OF_WIDTH: Final = {2: jnp.int16, 4: jnp.int32, 8: jnp.int64}
"""Signed integer of the same width as each float dtype, for reading its bits."""


def positive_subnormal(z: AnyArray) -> AnyArray:
    """Mask of the arguments XLA has flushed to zero but that are not zero.

    The float tests cannot do this. XLA compares a subnormal as if it were
    zero, so ``z > 0`` is False for exactly these values and ``z == 0.0`` is
    True for them -- which is how a subnormal argument reached `kn.k1`'s pole
    guard and came back ``inf``.
    """
    bits = lax.bitcast_convert_type(z, INT_OF_WIDTH[jnp.dtype(z.dtype).itemsize])
    return (bits > 0) & (z < jnp.finfo(z.dtype).tiny)


def exactly_zero(z: AnyArray) -> AnyArray:
    """``z == 0.0`` done on the bits, so a subnormal is not mistaken for zero.

    Only two bit patterns are zero, ``+0.0`` and ``-0.0``; the latter is the
    single integer more negative than every other float.
    """
    bits = lax.bitcast_convert_type(z, INT_OF_WIDTH[jnp.dtype(z.dtype).itemsize])
    return (bits == 0) | (bits == jnp.iinfo(bits.dtype).min)


def is_negative(z: AnyArray) -> AnyArray:
    """``z < 0`` done on the bits, so a negative subnormal is not read as zero.

    Every negative float has the sign bit set; ``-0.0`` is the single integer
    more negative than all of them, and is not "negative" for this purpose.
    """
    bits = lax.bitcast_convert_type(z, INT_OF_WIDTH[jnp.dtype(z.dtype).itemsize])
    return (bits < 0) & (bits != jnp.iinfo(bits.dtype).min)


@partial(jax.custom_jvp, nondiff_argnums=(1,))
def ldexp_no_flush(z: AnyArray, exponent: int, /) -> AnyArray:
    """``z * 2**exponent``, including where XLA has flushed a subnormal ``z``.

    `jax.numpy.ldexp` is no use here for the same reason `jnp.frexp` is no use
    to `log_no_flush`: it does arithmetic on the flushed operand. The bits are
    still there, though, and a subnormal is exactly
    ``mantissa * 2**(minexp - nmant)``, so the scaled value is a plain product
    of two perfectly normal numbers.

    Only for lifting a subnormal *up*: a normal ``z`` takes the ordinary
    multiply and will overflow if ``exponent`` is large enough to send it past
    the dtype's maximum.
    """
    info = jnp.finfo(z.dtype)
    bits = lax.bitcast_convert_type(z, INT_OF_WIDTH[jnp.dtype(z.dtype).itemsize])
    mantissa = jnp.bitwise_and(bits, (1 << info.nmant) - 1).astype(z.dtype)
    sign = jnp.where(bits < 0, -1.0, 1.0)
    from_bits = sign * mantissa * float(2.0 ** (info.minexp - info.nmant + exponent))
    # The flushed comparison, deliberately: it is True for every subnormal and
    # also for zero, whose zero mantissa sends `from_bits` to zero as well --
    # the same answer by either route, so the two need not be told apart.
    return jnp.where(jnp.abs(z) < info.tiny, from_bits, z * float(2.0**exponent))


@ldexp_no_flush.defjvp
def _ldexp_no_flush_jvp(
    exponent: int, primals: tuple[Any], tangents: tuple[Any]
) -> tuple[AnyArray, AnyArray]:
    """``d/dz (z * 2**k) = 2**k``, which the bit path does not supply by itself.

    `lax.bitcast_convert_type` carries an identically **zero** tangent, so the
    reconstructed branch differentiated to 0 -- and `_log_complex_no_flush` sits
    on top of it, which made `log_no_flush` non-holomorphic wherever that branch
    fires: the imaginary-direction derivative was 0 across the band, against a
    true ``1/z`` of 4.5e307 at ``z = tiny`` -- a representable number, not an
    overflow. This function is ``z * 2**k`` everywhere, on both branches, so its
    derivative is the constant and saying so is exact rather than a patch.
    """
    (z,), (dz,) = primals, tangents
    return ldexp_no_flush(z, exponent), dz * jnp.asarray(
        2.0**exponent, dtype=jnp.asarray(dz).dtype
    )


def _log_complex_no_flush(z: AnyArray, /) -> AnyArray:
    """`log_no_flush` for a complex argument, which cannot be bitcast whole.

    `lax.bitcast_convert_type` is undefined for a complex dtype, but
    `jax.numpy.real` and `jax.numpy.imag` hand back ordinary floats *with the
    subnormal bits intact* -- in both modes, which is the part that matters --
    so the components can be lifted out of the band one at a time and the
    logarithm taken of the scaled number.

    Whenever **either** component is inside the band, not only when both are.
    An earlier revision required both, on the reasoning that a subnormal beside
    a normal component is 292 decades below it and contributes nothing. That is
    true at the top of the normal range and false at the bottom of it: with
    ``real`` exactly ``tiny`` and ``imaginary`` just under it the two are within
    a factor of two, the flush took the argument to ``0`` instead of ``pi/4``,
    and the imaginary part of the logarithm was not approximately wrong but
    *entirely absent* -- a 0.79 radian error on `spence`'s complex derivative.

    And **only** where the whole number is small, which is the other half of
    the predicate and was missing for one round. ``0.0 < tiny`` is true, so
    "either component subnormal" is true of every ``z`` on either axis -- the
    entire ordinary plane. Those all went down the scaled branch, where
    ``log(z * 2**k) - k*log2`` subtracts two nearly equal numbers as soon as
    ``|z| ~ 1``: it cost three decimal digits in complex64 (1.08% relative at
    ``z = 1.0001``) and was *worse* than the plain logarithm it replaced.

    The bound is the magnitude below which a subnormal component can still
    change the answer at all. A component ``s`` moves ``hypot`` or ``atan2``
    only when ``s / magnitude`` exceeds an eps, so ``magnitude < tiny / eps``,
    i.e. ``tiny * 2**nmant`` -- which is `tiny / eps` bit for bit, since
    ``eps == 2**-nmant`` at both widths. Below that bound the subtraction
    cannot cancel: ``log(z * 2**k)`` is negative there and ``k * log2`` is
    positive, so the two magnitudes add rather than destroy each other. Above
    it the subnormal component is beneath the result's own last bit and
    `jnp.log` is already right. It doubles as the
    overflow guard the previous revision used, since scaling a number that
    small can never reach the dtype's maximum.
    """
    real, imaginary = jnp.real(z), jnp.imag(z)
    info = jnp.finfo(real.dtype)
    # Twice the mantissa width lands every subnormal in the normal range with
    # room to spare, and is itself an exactly representable power of two.
    exponent = 2 * info.nmant
    magnitude = jnp.maximum(jnp.abs(real), jnp.abs(imaginary))
    reachable = float(info.tiny) * float(2.0**info.nmant)
    band = ((jnp.abs(real) < info.tiny) | (jnp.abs(imaginary) < info.tiny)) & (
        magnitude < reachable
    )
    scaled = lax.complex(
        ldexp_no_flush(real, exponent), ldexp_no_flush(imaginary, exponent)
    )
    # Both branches evaluate, and each would poison the other: the scaled form
    # overflows for an ordinary argument, and the plain one is `-inf` for a
    # subnormal. Mask the operand on each side, not just the result.
    one = jnp.ones((), z.dtype)
    from_bits = jnp.log(jnp.where(band, scaled, one)) - exponent * _LN2
    plain = jnp.log(jnp.where(band, one, z))
    return jnp.where(band, from_bits, plain)


def log_no_flush(z: AnyArray, /, *, dtype: Any = None) -> AnyArray:
    """``log(z)``, including where ``z`` is subnormal and XLA has flushed it.

    XLA on CPU flushes a subnormal *input* to zero, so `jnp.log` returns
    ``-inf`` for every ``z`` below ``finfo(dtype).tiny`` -- and `kn.k0` then
    returned ``inf`` where the true value is an ordinary number near 700. In
    float32 that band starts at 1.2e-38, an entirely reachable magnitude.

    A subnormal's bit pattern still holds its mantissa; only arithmetic on it
    flushes. Reading the bits as an integer therefore recovers it, and a
    subnormal is exactly ``mantissa * tiny / 2**nmant``, so its logarithm is
    ``log(mantissa)`` plus a constant. `jnp.frexp` is not an alternative -- it
    flushes too, and reports the same exponent for every subnormal.

    Note this is distinct from the `_LN2` subtraction in `_k0_small`, which
    stops a *normal* ``z`` being halved into the subnormal range. That fix does
    nothing when the argument arrives subnormal already.

    Complex input goes to `_log_complex_no_flush`, which takes the two
    components apart rather than bitcasting the pair. Callers therefore need no
    dtype test of their own; `spence` carried one at two sites, and it was the
    reason its *complex* derivative stayed `nan` across the whole band -- an
    ordinary magnitude in complex64 -- after the real one had been fixed.
    """
    if jnp.issubdtype(z.dtype, jnp.complexfloating):
        # Not silently ignored. `dtype` widens the arithmetic for a caller whose
        # own width cannot hold the answer -- bfloat16, where a logarithm near
        # -87 has nowhere to go -- and there is no complex bfloat16, so no
        # caller needs the combination. Raising says so rather than returning a
        # narrower answer than was asked for.
        if dtype is not None:
            msg = "log_no_flush: `dtype` widening is not supported for complex input"
            raise NotImplementedError(msg)
        return _log_complex_no_flush(z)
    info = jnp.finfo(z.dtype)
    bits = lax.bitcast_convert_type(z, INT_OF_WIDTH[jnp.dtype(z.dtype).itemsize])
    # The bits must be read at the argument's own width, but the arithmetic on
    # them need not be done there. `dtype` widens that half: a logarithm near
    # -87 has no room in bfloat16, where the spacing is 0.5, so the caller that
    # exponentiates it back gets a factor of `e**0.25` for free. Widening is not
    # automatic because `kn` wants its result in the width it asked for.
    arithmetic = jnp.dtype(dtype) if dtype is not None else z.dtype
    mantissa = jnp.bitwise_and(bits, (1 << info.nmant) - 1).astype(arithmetic)
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
    # it and `kn.k0` came back `inf` where the argument is simply out of domain.
    negative = (bits < 0) & (bits != jnp.iinfo(bits.dtype).min)
    from_bits = jnp.log(mantissa) + (log(float(info.tiny)) - info.nmant * _LN2)
    # Every branch evaluates, so keep `log` off the flushed value.
    plain = jnp.log(jnp.where(subnormal, info.tiny, z).astype(arithmetic))
    return jnp.where(negative, jnp.nan, jnp.where(subnormal, from_bits, plain))
