"""The Gegenbauer (ultraspherical) polynomials."""

__all__ = ["eval_gegenbauer", "eval_gegenbauers"]

from functools import partial
from typing import TypeAlias

import jax
import jax.numpy as jnp

from .custom_types import AnyArray, AnyArrayLike, ScalarLike, Vector
from .dtype import exactly_zero, is_negative, promote_integers

_Carry: TypeAlias = tuple[AnyArray, AnyArray, AnyArray, AnyArray]


def C0(x: AnyArrayLike, /) -> AnyArray:
    """Return the Gegenbauer polynomial of order 0, which is 1.

    It still has to broadcast: returning the Python scalar ``1.0`` would drop
    the shape (and dtype) of ``x`` on the floor.

    Written as ``x * 0 + 1`` rather than `jax.numpy.ones_like` so that a weakly
    typed ``x`` stays weakly typed, matching every higher degree.

    Examples
    --------
    >>> import jax.numpy as jnp
    >>> from spexial._src.gegenbauer import C0
    >>> C0(jnp.asarray([0.1, 0.2, 0.3])).tolist()
    [1.0, 1.0, 1.0]

    """
    x_arr = jnp.asarray(x)
    # `x * 0 + 1` is `nan` at +-inf. `C_0 == 1` at every x including the
    # infinities, which is what `scipy.special.eval_gegenbauer` returns, and a
    # `nan` x still gives `nan`. This fixes orders 0 to 2 only: from n = 3 the
    # recurrence itself forms `inf - inf` and returns `nan` where the true value
    # is +-inf. That is outside the supported |x| <= 1 and is documented on the
    # accuracy page; SciPy is not self-consistent there either.
    return jnp.where(jnp.isinf(x_arr), 1.0, x_arr * 0.0 + 1.0)


def C1(alpha: AnyArrayLike, x: AnyArrayLike, /) -> AnyArray:
    r"""Return the Gegenbauer polynomial of order 1, :math:`2 \alpha x`.

    Examples
    --------
    >>> import jax.numpy as jnp
    >>> from spexial._src.gegenbauer import C1
    >>> C1(1.5, jnp.asarray([0.1, 0.2])).tolist()
    [0.30000000000000004, 0.6000000000000001]

    """
    # `promote_integers` rather than `* 1.0`: both promote, but the multiply
    # also flushes a subnormal `x` to zero. With integer `alpha` *and* integer
    # `x` the product would otherwise stay int64,
    # while `C0` is always float, and `lax.scan` then rejects the carry as having
    # mismatched types. scipy promotes integer input to float, so we do too.
    # A plain multiply, which loses a subnormal `x` to XLA's flush. That is
    # recoverable -- `dtype.mul_no_flush` regroups the product around the
    # mantissa bits and gets `eval_gegenbauer(1, 1e300, 5e-324)` exactly right
    # -- but it costs two things this function is not willing to pay. It needs
    # `lax.optimization_barrier` to survive XLA's reassociation, which inside
    # the recurrence blocks fusion (1.58x slower over 128 points), and its
    # `where` over bitcast integers returns a *strongly* typed result, so
    # `eval_gegenbauer(3, 0.5, 0.25)` stopped being weakly typed -- the property
    # `C0` is written `x * 0 + 1` to preserve, and which changes how every
    # ordinary result promotes downstream.
    #
    # Both costs land on every caller; the loss lands only where `alpha` is
    # around 1e285, since below that the lost quantity is itself denormal. So
    # the flush stays, documented as the platform floor it is.
    return 2 * promote_integers(alpha) * promote_integers(x)


def _unify_dtypes(alpha: AnyArrayLike, x: AnyArrayLike, /) -> tuple[AnyArray, AnyArray]:
    """Put `alpha` and `x` on their common dtype, leaving their shapes alone.

    `jnp.broadcast_arrays` unifies *shapes* but not *dtypes*, and the recurrence
    needs both: `C0` follows `x` while `C1` follows the promotion of the two, so
    a float64 `alpha` against a float32 `x` gives the `lax.scan` carry one dtype
    going in and another coming out. That surfaces as "scan body function carry
    input and carry output must have equal types", naming neither the function
    nor the argument at fault, and only from ``n >= 2`` -- orders 0 and 1 never
    reach the scan, so the break looks arbitrary.

    Separate from `_seed` because `eval_gegenbauers` needs the dtype half
    without the shape half: its documented return shape is ``(n + 1,)``, which a
    broadcast `alpha` would silently change.

    Only when the dtypes actually differ: an unconditional `astype` strips
    *weak* typing, which is deliberately preserved here -- see `C0`, written as
    ``x * 0 + 1`` rather than `ones_like` for exactly that reason. Casting
    regardless turned ``eval_gegenbauer(3, 0.5, 0.25)`` from a weak float64 into
    a strong one, changing how the result promotes downstream.
    """
    # `promote_integers`, not `* 1.0`: the multiply promotes integers, which is
    # what it is for, but it also flushes a subnormal `x` to zero on XLA. That
    # took `eval_gegenbauer(1, 1e300, 5e-324)` -- an ordinary 9.9e-24 -- to
    # exactly 0, inside the documented `|x| <= 1`.
    alpha_arr = promote_integers(alpha)
    x_arr = promote_integers(x)
    if alpha_arr.dtype != x_arr.dtype:
        dtype = jnp.result_type(alpha_arr, x_arr)
        alpha_arr, x_arr = alpha_arr.astype(dtype), x_arr.astype(dtype)
    return alpha_arr, x_arr


def _seed(alpha: AnyArrayLike, x: AnyArrayLike, /) -> tuple[AnyArray, AnyArray]:
    """Broadcast `alpha` against `x` so the recurrence carry is shape-stable.

    `C0` follows `x`'s shape while `C1` follows the broadcast of both, so an
    `alpha` wider than `x` gave the `lax.scan` carry one shape going in and
    another coming out. That surfaced as "scan body function carry input and
    carry output must have equal types" -- naming neither this function nor the
    argument at fault -- but only from ``n >= 2``, since orders 0 and 1 never
    reach the scan. `scipy.special.eval_gegenbauer` broadcasts here, so rather
    than reject it, match it -- in `eval_gegenbauer` only. `eval_gegenbauers`
    keeps `alpha` scalar: its documented return shape is ``(n + 1,)``, which an
    array `alpha` would silently change.
    """
    # `jnp.broadcast_arrays` unifies *shapes* but not *dtypes*, which left the
    # dtype half of the same bug: a strongly-typed float64 `alpha` against a
    # float32 `x` gave `C0` float32 and the recurrence float64, so the carry
    # changed dtype and the scan raised -- same message, same "n <= 1 works,
    # n >= 2 dies" signature as the shape mismatch above. Promote both to their
    # common type first. Only strong dtypes trigger it; a weakly-typed Python
    # float follows `x`, which is why it went unnoticed.
    # `jnp.broadcast_arrays` returns a list, which the runtime type checker
    # rejects against the annotation, hence `tuple(...)`.
    return tuple(jnp.broadcast_arrays(*_unify_dtypes(alpha, x)))


def _at_infinity(n: int, alpha: AnyArray, x: AnyArray, value: AnyArray) -> AnyArray:
    """Substitute the analytic limit where `x` is infinite.

    The recurrence forms `2(n + a)x C_{n-1} - (n + 2a - 2) C_{n-2}`, which is
    `inf - inf` from n = 3 once `x` is infinite, so every order from there was
    `nan` -- while orders 0 to 2 happened to come out right, which made the
    break look arbitrary.

    The limit is set by the leading coefficient, `2^n (a)_n / n!`. For the
    supported `a > -1/2` every factor of the Pochhammer symbol after the first
    is positive, so its sign is just `sign(a)`, and

        C_n(+inf) = sign(a) * inf,   C_n(-inf) = sign(a) * (-1)^n * inf.

    At `a = 0` the polynomial is identically zero for n >= 1, so the limit is 0
    rather than an infinity -- which also keeps `sign(a) == 0` from producing
    `0 * inf == nan`.

    This is outside the documented `|x| <= 1` domain, where SciPy is not
    self-consistent either (it returns `inf` at `+inf` and `nan` at `-inf`).
    """
    if n == 0:
        return value
    # The sign of `alpha` from its bits, not from `jnp.sign`: XLA reports 0 for
    # a subnormal, so `eval_gegenbauer(1, 5e-324, inf)` took the `sign == 0`
    # branch and returned 0 where the limit is `+inf`.
    # `nan` first: a `nan` alpha has `bits > 0`, so the bit tests below would
    # read it as positive and hand back a definite `+inf` for an argument whose
    # limit does not exist. `jnp.sign` propagated `nan` for free; replacing it
    # with bit tests dropped that, and this puts it back explicitly.
    alpha_sign = jnp.where(
        jnp.isnan(alpha),
        jnp.nan,
        jnp.where(exactly_zero(alpha), 0.0, jnp.where(is_negative(alpha), -1.0, 1.0)),
    )
    sign = alpha_sign * jnp.where(x > 0, 1.0, (-1.0) ** n)
    limit = jnp.where(sign == 0, 0.0, sign * jnp.inf)
    return jnp.where(jnp.isinf(x), limit, value)


def _C_n_plus_1(carry: _Carry, n: AnyArray) -> tuple[_Carry, AnyArray]:
    """Apply the three-term Gegenbauer recurrence once."""
    alpha, x, Cn, Cn_minus_1 = carry
    # Deliberately a plain multiply, unlike `C1`. `mul_no_flush` would keep a
    # subnormal `x` alive here too, but it needs `lax.optimization_barrier` to
    # stop XLA reassociating its grouping, and a barrier inside the scan body
    # blocks the fusion the recurrence depends on: measured 1.58x slower on
    # `eval_gegenbauer` over 128 points. That is not a trade worth making for a
    # corner that needs `alpha` around 1e285 before the loss is even visible, so
    # orders from 2 up sit at the platform's floor and the docs say so.
    # `(n - 1) + 2a`, never `n + 2a - 1`. They are the same number and not the
    # same computation: at n = 1 the second spells `1 + 2a - 1`, which
    # annihilates `2a` entirely once it falls below an eps of 1, and the
    # recurrence then carries the error up through every higher degree. At
    # alpha = 1e-20 that took `C_3` to exactly 0 against a true -6.7e-21 and
    # flipped the sign of `C_5`. Grouped this way the small term has nothing to
    # cancel against; from n = 2 up it is a legitimate rounding, since there
    # `2a` really is negligible beside `n - 1`.
    coefficient = (n - 1) + 2 * alpha
    accumulate = (2 * (n + alpha) * x * Cn - coefficient * Cn_minus_1) / (n + 1)
    return (alpha, x, accumulate, Cn), accumulate


# TODO: support n non-integer
@partial(jax.jit, static_argnums=(0,))
def eval_gegenbauers(n: int, alpha: ScalarLike, x: ScalarLike, /) -> Vector:
    r"""Return the Gegenbauer polynomial of degree ``n`` and all lower ones.

    There is no `scipy.special` counterpart; it is the by-product of the
    three-term recurrence that `eval_gegenbauer` runs anyway, and is much
    cheaper than ``n + 1`` separate calls.

    The Gegenbauer polynomials can be defined via the Gauss hypergeometric
    function :math:`{}_2F_1` as

    .. math::

        C_n^{(\alpha)} = \frac{(2\alpha)_n}{\Gamma(n + 1)}
          {}_2F_1(-n, 2\alpha + n; \alpha + 1/2; (1 - z)/2).

    When :math:`n` is an integer the result is a polynomial of degree
    :math:`n`. See 22.5.46 in [AS]_ for details.

    Parameters
    ----------
    n
        Degree of the polynomial. Must be a static Python `int`; non-integer
        degrees are not supported yet.
    alpha
        Parameter.
    x
        Scalar point at which to evaluate the polynomials. Use
        ``jax.vmap(..., in_axes=(None, None, 0))`` for many points.

    Returns
    -------
    Array[float, (n + 1,)]
        Values of :math:`C_i^{(\alpha)}(x)` for ``i = 0 ... n``.

    References
    ----------
    .. [AS] Milton Abramowitz and Irene A. Stegun, eds.
        Handbook of Mathematical Functions with Formulas,
        Graphs, and Mathematical Tables. New York: Dover, 1972.

    Examples
    --------
    >>> import spexial as sp

    >>> sp.eval_gegenbauers(3, 1.0, 0.5).tolist()
    [1.0, 1.0, 0.0, -1.0]

    The degree-0 case is a single value, not two:

    >>> sp.eval_gegenbauers(0, 1.0, 0.5).tolist()
    [1.0]

    """
    alpha, x = _unify_dtypes(alpha, x)
    C0_val = C0(x)
    if n == 0:
        return jnp.atleast_1d(C0_val)

    C1_val = C1(alpha, x)
    if n == 1:
        return jnp.hstack([C0_val, C1_val])

    carry = (alpha, x, C1_val, C0_val)
    n_values = jnp.arange(1, n)  # starts at 1: 0 is already initialized above
    _, C_values = jax.lax.scan(_C_n_plus_1, carry, n_values)

    orders = jnp.hstack([C0_val, C1_val, C_values])
    # Every order from 3 up is `inf - inf` when `x` is infinite; substitute each
    # one's limit. `C_0` is 1 there and `C_1 = 2 a x` is already right except at
    # a = 0, so the whole vector goes through `_at_infinity` order by order.
    return jnp.stack(
        [
            _at_infinity(k, jnp.asarray(alpha), jnp.asarray(x), orders[k])
            for k in range(n + 1)
        ]
    )


# TODO: support n non-integer
@partial(jax.jit, static_argnums=(0,))
def eval_gegenbauer(n: int, alpha: AnyArrayLike, x: AnyArrayLike, /) -> AnyArray:
    r"""Evaluate the Gegenbauer polynomial :math:`C_n^{(\alpha)}(x)`.

    The Gegenbauer polynomials can be defined via the Gauss hypergeometric
    function :math:`{}_2F_1` as

    .. math::

        C_n^{(\alpha)} = \frac{(2\alpha)_n}{\Gamma(n + 1)}
          {}_2F_1(-n, 2\alpha + n; \alpha + 1/2; (1 - z)/2).

    When :math:`n` is an integer the result is a polynomial of degree
    :math:`n`. See 22.5.46 in [AS]_ for details.

    Parameters
    ----------
    n
        Degree of the polynomial. Must be a static Python `int`; non-integer
        degrees are not supported yet.
    alpha
        Parameter.
    x
        Point(s) at which to evaluate the polynomial. Evaluated elementwise.

    Returns
    -------
    Array
        Values of :math:`C_n^{(\alpha)}(x)`.

    See Also
    --------
    eval_gegenbauers : the same, plus every lower degree.
    scipy.special.roots_gegenbauer : roots and quadrature weights.
    jax.scipy.special.hyp2f1 : Gauss hypergeometric function.

    References
    ----------
    .. [AS] Milton Abramowitz and Irene A. Stegun, eds.
        Handbook of Mathematical Functions with Formulas,
        Graphs, and Mathematical Tables. New York: Dover, 1972.

    Examples
    --------
    >>> import jax.numpy as jnp
    >>> import spexial as sp

    >>> float(sp.eval_gegenbauer(3, 1.0, 0.5))
    -1.0

    It broadcasts over ``x``:

    >>> sp.eval_gegenbauer(2, 1.0, jnp.asarray([-1.0, 0.0, 1.0])).tolist()
    [3.0, -1.0, 3.0]

    """
    # Seeded before the early returns too: at n == 0 the result is `C0(x)`,
    # whose shape follows `x` alone, so an `alpha` wider than `x` came back the
    # wrong shape instead of broadcasting as scipy does.
    alpha_arr, x_arr = _seed(alpha, x)
    if n == 0:
        return C0(x_arr)
    if n == 1:
        # `2 * alpha * x` is `nan` at alpha = 0 with x infinite, so this early
        # return needs the same limit substitution as the scan below.
        return _at_infinity(1, alpha_arr, x_arr, C1(alpha_arr, x_arr))

    carry = (alpha_arr, x_arr, C1(alpha_arr, x_arr), C0(x_arr))
    n_values = jnp.arange(1, n)  # 0 is already done
    _, C_values = jax.lax.scan(_C_n_plus_1, carry, n_values)
    return _at_infinity(n, alpha_arr, x_arr, C_values[-1])
