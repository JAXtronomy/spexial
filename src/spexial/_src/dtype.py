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

from math import log
from typing import Any, Final

import jax.numpy as jnp
from jax import lax

from .custom_types import AnyArray, AnyArrayLike


def as_float(x: AnyArrayLike, /, *, keep_weak: bool = False) -> AnyArray:
    """Promote to at least float32, without disturbing a subnormal.

    The promotion is not cosmetic. In `kn`, `float16` and `bfloat16` have no
    working series/asymptotic cross-over at all -- `K0` in bfloat16 was wrong by
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
    hazard this module exists to document and the one that cost `kn.K0` its
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
    True for them -- which is how a subnormal argument reached `kn.K1`'s pole
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


def mul_no_flush(a: AnyArray, x: AnyArray, /) -> AnyArray:
    """Multiply ``a`` by an ``x`` that may be subnormal, without losing it.

    The multiply is the problem, not the value: a subnormal *operand* is flushed
    on XLA-CPU, so `eval_gegenbauer(1, 1e300, 5e-324)` -- whose answer is an
    ordinary 9.9e-24 -- came out exactly 0. Neither `ldexp` nor a wider dtype
    recovers it; bfloat16 and float32 share an exponent range, and float64 has
    nothing above it.

    A subnormal is ``mantissa * tiny / 2**nmant`` with the mantissa an integer,
    so the product can be regrouped as ``(a * mantissa) * tiny * 2**-nmant``
    where every operand is normal. Order matters: the two small factors go last,
    after ``a`` has absorbed the mantissa.

    What this cannot do is make a subnormal *result* survive -- the final
    multiply underflows and XLA flushes that too. So it recovers exactly the
    cases that were visibly wrong, where ``a`` is large enough that ``a * x`` is
    a normal number, and leaves the rest at the platform's floor.
    """
    info = jnp.finfo(x.dtype)
    bits = lax.bitcast_convert_type(x, INT_OF_WIDTH[jnp.dtype(x.dtype).itemsize])
    mantissa = jnp.bitwise_and(bits, (1 << info.nmant) - 1).astype(x.dtype)
    negative = (bits < 0) & (bits != jnp.iinfo(bits.dtype).min)
    # `-0.0` has a non-zero bit pattern (it is `iinfo.min`) but is not
    # subnormal; without excluding it the reconstruction runs and returns
    # `+0.0`, losing the sign IEEE gives `a * -0.0`.
    subnormal = (jnp.abs(x) < info.tiny) & (bits != 0) & ~exactly_zero(x)
    signed = jnp.where(negative, -mantissa, mantissa)
    # `optimization_barrier` between the steps, because the grouping *is* the
    # fix and XLA will otherwise reassociate the chain straight back into
    # something that flushes. Without the barriers this is correct eagerly and
    # returns zero again under `jit` -- which is how the callers run.
    absorbed = lax.optimization_barrier(a * signed)
    scaled = lax.optimization_barrier(absorbed * info.tiny)
    return jnp.where(subnormal, scaled * (2.0**-info.nmant), a * x)


def log_no_flush(z: AnyArray, /, *, dtype: Any = None) -> AnyArray:
    """``log(z)``, including where ``z`` is subnormal and XLA has flushed it.

    XLA on CPU flushes a subnormal *input* to zero, so `jnp.log` returns
    ``-inf`` for every ``z`` below ``finfo(dtype).tiny`` -- and `kn.K0` then
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
    # it and `kn.K0` came back `inf` where the argument is simply out of domain.
    negative = (bits < 0) & (bits != jnp.iinfo(bits.dtype).min)
    from_bits = jnp.log(mantissa) + (log(float(info.tiny)) - info.nmant * _LN2)
    # Every branch evaluates, so keep `log` off the flushed value.
    plain = jnp.log(jnp.where(subnormal, info.tiny, z).astype(arithmetic))
    return jnp.where(negative, jnp.nan, jnp.where(subnormal, from_bits, plain))
