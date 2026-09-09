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

import jax.numpy as jnp

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
