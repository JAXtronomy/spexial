"""The Gegenbauer (ultraspherical) polynomials."""

__all__ = ["eval_gegenbauer", "eval_gegenbauers"]

from functools import partial
from typing import TypeAlias

import jax
import jax.numpy as jnp

from .custom_types import AnyArray, AnyArrayLike, ScalarLike, Vector

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
    # `x * 0 + 1` is `nan` at +-inf, and that `nan` then propagates through the
    # recurrence to every order. `C_0 == 1` at every x including the infinities,
    # which is what `scipy.special.eval_gegenbauer` returns. A `nan` x still
    # gives `nan`.
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
    # `* 1.0` promotes: with integer `alpha` *and* integer `x` this stays int64,
    # while `C0` is always float, and `lax.scan` then rejects the carry as having
    # mismatched types. scipy promotes integer input to float, so we do too.
    return 2 * jnp.asarray(alpha) * jnp.asarray(x) * 1.0


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
    alpha_arr = jnp.asarray(alpha) * 1.0
    x_arr = jnp.asarray(x) * 1.0
    # Only when they actually differ: an unconditional `astype` strips *weak*
    # typing, and weak types are deliberately preserved here -- see `C0`, which
    # is written as `x * 0 + 1` rather than `ones_like` for exactly that reason.
    # Casting regardless turned `eval_gegenbauer(3, 0.5, 0.25)` from a weak
    # float64 into a strong one, which changes how the result promotes against
    # anything narrower downstream.
    if alpha_arr.dtype != x_arr.dtype:
        dtype = jnp.result_type(alpha_arr, x_arr)
        alpha_arr = alpha_arr.astype(dtype)
        x_arr = x_arr.astype(dtype)
    # Unpacked into a real tuple: `jnp.broadcast_arrays` returns a list, which
    # the runtime type checker rejects against the annotation.
    alpha_arr, x_arr = jnp.broadcast_arrays(alpha_arr, x_arr)
    return alpha_arr, x_arr


def _C_n_plus_1(carry: _Carry, n: AnyArray) -> tuple[_Carry, AnyArray]:
    """Apply the three-term Gegenbauer recurrence once."""
    alpha, x, Cn, Cn_minus_1 = carry
    accumulate = (2 * (n + alpha) * x * Cn - (n + 2 * alpha - 1) * Cn_minus_1) / (n + 1)
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
    C0_val = C0(x)
    if n == 0:
        return jnp.atleast_1d(C0_val)

    C1_val = C1(alpha, x)
    if n == 1:
        return jnp.hstack([C0_val, C1_val])

    carry = (alpha, x, C1_val, C0_val)
    n_values = jnp.arange(1, n)  # starts at 1: 0 is already initialized above
    _, C_values = jax.lax.scan(_C_n_plus_1, carry, n_values)

    return jnp.hstack([C0_val, C1_val, C_values])


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
        return C1(alpha_arr, x_arr)

    carry = (alpha_arr, x_arr, C1(alpha_arr, x_arr), C0(x_arr))
    n_values = jnp.arange(1, n)  # 0 is already done
    _, C_values = jax.lax.scan(_C_n_plus_1, carry, n_values)
    return C_values[-1]
