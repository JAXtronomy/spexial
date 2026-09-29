"""Spherical Bessel functions of the first kind."""

__all__ = ["SphericalJnRecurrence", "spherical_jn", "spherical_jn_all"]

import operator
from enum import StrEnum
from functools import partial
from typing import Any, Final, Literal, TypeAlias

import jax
import jax.numpy as jnp
import numpy as np
from jax import lax

from .custom_types import AnyArray, AnyArrayLike
from .dtype import as_float, cast_like

_CUTOFF: Final = 2e-10
"""Threshold of the `_xmin` estimate."""

_UNROLL: Final = 8
"""Recurrence steps per loop iteration. Against 1: 4-15x faster on an A100 and
2-4x on CPU, for about 30% more compile time per ``n``."""

_RESCALE_AT: Final = 2.0**40
"""`_downward` rescales its carry once it passes this."""

_SHIFT: Final = 100.0
"""... by ``2**-_SHIFT``, exactly, keeping the exponent apart."""

_Carry: TypeAlias = tuple[AnyArray, AnyArray]


class SphericalJnRecurrence(StrEnum):
    """Which recurrence `spherical_jn` and `spherical_jn_all` evaluate.

    Upward recurrence is stable above the turning point ``|z| ~ n`` and unstable
    below it; Miller's downward recurrence is the reverse. `BOTH` runs each on
    its own side, which is right everywhere and costs both.
    """

    BOTH = "both"
    """Upward for ``|z| >= n``, downward below. Accurate for every ``z``."""

    UP = "up"
    """Upward only: the fastest, and exact above the turning point.

    Below it, values under ~1e-6 of the peak are noise in float64 -- wrong in
    sign and magnitude, or set to zero -- and in float32 the whole region is
    wrong, by up to 80x the peak.
    """

    DOWN = "down"
    """Downward only, for arguments known to be below the turning point.

    Accurate, to the smallest values, wherever ``|z| < n``, and `nan` at
    ``|z| >= n``, where it cannot be. Orders 0 and 1 alone are closed forms,
    and exact everywhere.
    """


SphericalJnRecurrenceLike: TypeAlias = (
    SphericalJnRecurrence | Literal["both", "up", "down"]
)
"""A `SphericalJnRecurrence`, or its string value."""


def _xmin(orders: np.ndarray, cutoff: float) -> np.ndarray:
    """Argument below which `j_l` is set to zero, from the Debye expansion."""
    nu = orders + 0.5
    lhs = np.log(cutoff * nu) / nu
    alpha = (
        -0.4
        * lhs
        * (1.0 + 2.0 * np.cosh(np.arccosh(1.0 + 375.0 / (16.0 * lhs * lhs)) / 3.0))
    )
    return nu / np.cosh(alpha)


def _j0(x: AnyArray, /) -> AnyArray:
    safe = jnp.where(x == 0.0, 1.0, x)
    return jnp.where(x == 0.0, 1.0, jnp.sin(safe) / safe)


def _j1(x: AnyArray, /) -> AnyArray:
    x2 = x * x
    small = x2 < 1e-2  # `sin x / x - cos x` cancels; use the series
    safe = jnp.where(small, 1.0, x)
    series = (
        x
        / 3.0
        * (1.0 - x2 / 10.0 + x2**2 / 280.0 - x2**3 / 15120.0 + x2**4 / 1330560.0)
    )
    return jnp.where(small, series, (jnp.sin(safe) / safe - jnp.cos(safe)) / safe)


def _rows(lo: int, hi: int, recurrence: SphericalJnRecurrence, x: AnyArray) -> AnyArray:
    """Orders ``lo ... hi`` at finite ``x >= 0``.

    Upward recurrence is stable above the turning point. Below it, the CLASS
    cutoff keeps its noise to ``eps / _CUTOFF``: ~1e-7 of the peak in float64,
    and ~600x the peak in float32, which no cutoff fixes (spexial#51, #53).
    Miller's downward recurrence is stable there instead, so
    `SphericalJnRecurrence.BOTH` takes each on its own side of ``x = hi``.

    Under `BOTH`, each recurrence sits in its own `lax.cond` and runs only if
    some ``x`` needs it, so a batch wholly on one side pays for one: 1.4-3.6x
    faster on CPU (``spherical_jn_all(500)`` above the turning point, 54 ->
    15 ms). A mixed batch runs both, up to 14% slower than a plain `where`, as
    the `cond` boundaries stop XLA fusing the select into the recurrences. One
    `cond` each, not a three-way `lax.switch`: that compiled each recurrence
    twice, and made every new ``n`` 55% slower to compile. Under `jax.vmap`
    the predicates are batched, and JAX turns both into selects.
    """
    if hi <= 1 or recurrence is SphericalJnRecurrence.UP:
        return _upward(lo, hi, x)
    if recurrence is SphericalJnRecurrence.DOWN:
        return jnp.where(x < hi, _downward(lo, hi, x), jnp.nan)
    above = x >= hi

    def skipped(v: AnyArray) -> AnyArray:
        return jnp.zeros((hi - lo + 1, *v.shape), v.dtype)

    up = lax.cond(jnp.any(above), partial(_upward, lo, hi), skipped, x)
    down = lax.cond(jnp.any(~above), partial(_downward, lo, hi), skipped, x)
    return jnp.where(above, up, down)


def _upward(lo: int, hi: int, x: AnyArray) -> AnyArray:
    """Orders ``lo ... hi`` by upward recurrence, zeroed below `_xmin`."""
    j0, j1 = _j0(x), _j1(x)
    if hi <= 1:
        return jnp.stack([j0, j1][lo : hi + 1])

    orders = jnp.arange(2, hi + 1, dtype=x.dtype)
    xmin = jnp.asarray(_xmin(np.arange(2.0, hi + 1), _CUTOFF), dtype=x.dtype)
    inv_x = 1.0 / jnp.where(x < xmin[0], 1.0, x)  # every row is 0 there

    def step(
        carry: _Carry, order_xmin: tuple[AnyArray, AnyArray]
    ) -> tuple[_Carry, AnyArray]:
        prev, cur = carry
        order, xmin_l = order_xmin
        keep = x >= xmin_l
        nxt = jnp.where(keep, (2.0 * order + 1.0) * inv_x * cur - prev, 0.0)
        return (cur, nxt), jnp.where(keep, cur, 0.0)

    carry = (j1, 3.0 * inv_x * j1 - j0)
    k = max(lo - 2, 0)
    if k:  # advance to `lo` without keeping the rows
        carry, _ = lax.scan(
            lambda c, s: (step(c, s)[0], None),
            carry,
            (orders[:k], xmin[:k]),
            unroll=_UNROLL,
        )
    _, rows = lax.scan(step, carry, (orders[k:], xmin[k:]), unroll=_UNROLL)
    if lo >= 2:
        return rows
    return jnp.concatenate([jnp.stack([j0, j1][lo:]), rows])


def _inverse_power_of_two(
    flag: AnyArray, dtype: Any, *, from_bits: bool = False
) -> AnyArray:
    """``2**-_SHIFT`` where `flag` is set and ``1`` elsewhere, exactly.

    ``from_bits`` writes a float64's exponent field directly instead of calling
    `exp2`, which is a full transcendental there. That made `_downward`'s
    non-emitting passes 2.8x faster, but its row-emitting pass 1.7x *slower*,
    as fused by XLA on CPU, so it is chosen per pass. float32's `exp2` beats the
    integer path either way.
    """
    if not from_bits or jnp.finfo(dtype).bits < 64:
        return jnp.exp2(-_SHIFT * flag.astype(dtype))
    exponent = 1023 - int(_SHIFT) * flag.astype(jnp.int64)
    return lax.bitcast_convert_type(exponent << 52, dtype)


def _downward(lo: int, hi: int, x: AnyArray) -> AnyArray:
    """Orders ``lo ... hi``, ``hi >= 2``, by Miller's downward recurrence.

    Starts from ``f_{N+1} = 0, f_N = 1`` far enough above ``max(hi, x)`` that
    the seed has decayed below an eps by order ``hi``, recurs down to 0, and
    normalises on whichever of `j_0` and `j_1` is larger, so a zero of either
    costs nothing. The carry is rescaled by a power of two whenever it grows
    past `_RESCALE_AT`, with the exponent kept apart, since ``j_0 / j_N``
    overflows any float.
    Accurate only for ``x < hi``; `_rows` takes `_upward` above that.
    """
    top = hi + int(6.0 * hi ** (1.0 / 3.0)) + 16  # past the turning point
    first = max(lo, 2)  # j_0 and j_1 come from their closed forms
    # Above this, one step grows the carry by at most `(2 top + 1) / x`, under
    # 2**80. Below it the leading term of the series is exact to rounding: the
    # next is `x**2` smaller, and `x < 1e-15` for any n < 10**7.
    floor = (2.0 * top + 1.0) * 2.0**-80
    inv_x = 1.0 / jnp.maximum(x, floor)

    def step(
        carry: tuple[AnyArray, AnyArray, AnyArray],
        order: AnyArray,
        *,
        from_bits: bool = False,
    ) -> tuple[tuple[AnyArray, AnyArray, AnyArray], _Carry]:
        above, cur, exponent = carry  # f_{l+1}, f_l, both times 2**-exponent
        below = (2.0 * order + 1.0) * inv_x * cur - above
        # A carry under 2**40 grows by under 2**80 (see `floor`), to at most
        # 2**120, inside float32's 2**128; one `_SHIFT` of 100 brings it back
        # under 2**40. (A floor at `sqrt(tiny)` let float64 grow by 2**535 a
        # step, faster than the rescale, and overflowed to `nan`.) The factor
        # is computed from a 0/1 flag, not selected with a `where`: under
        # `_UNROLL` XLA fuses the selects and recomputes them, 7x slower.
        big = jnp.abs(below) > _RESCALE_AT
        factor = _inverse_power_of_two(big, x.dtype, from_bits=from_bits)
        shift = _SHIFT * big.astype(x.dtype)
        return (cur * factor, below * factor, exponent + shift), (cur, exponent)

    def skip(
        carry: tuple[AnyArray, AnyArray, AnyArray], order: AnyArray
    ) -> tuple[tuple[AnyArray, AnyArray, AnyArray], None]:
        return step(carry, order, from_bits=True)[0], None

    def orders(start: int, stop: int) -> AnyArray:
        return jnp.arange(start, stop - 1, -1, dtype=x.dtype)

    carry = (jnp.zeros_like(x), jnp.ones_like(x), jnp.zeros_like(x))
    carry, _ = lax.scan(skip, carry, orders(top, hi + 1), unroll=_UNROLL)
    carry, (rows, exponents) = lax.scan(step, carry, orders(hi, first), unroll=_UNROLL)
    (f1, f0, e0), _ = lax.scan(skip, carry, orders(first - 1, 1), unroll=_UNROLL)

    j0, j1 = _j0(x), _j1(x)
    on_j0 = jnp.abs(j0) >= jnp.abs(j1)
    # Both the rows and the normaliser can sit anywhere in [2**-60, 2**40], so
    # fold their exponents into the one power of two. Otherwise `exp2` drops
    # into the subnormals, which XLA flushes, before the mantissas lift it back.
    mantissa, shift = jnp.frexp(jnp.where(on_j0, f0, f1))
    scale = jnp.where(on_j0, j0, j1) / mantissa
    row_mantissas, row_shifts = jnp.frexp(rows[::-1])
    power = exponents[::-1] + row_shifts - e0 - shift
    rows = row_mantissas * scale * jnp.exp2(power)
    # Below `floor`, `j_l(x) = x**l / (2l + 1)!!` to rounding, formed in logs so
    # that it underflows to 0 rather than overflowing on the way there.
    orders_np = np.arange(first, hi + 1)
    log_double_factorial = np.cumsum(np.log(2.0 * np.arange(hi + 1) + 1.0))
    shape = (-1,) + (1,) * x.ndim
    series = jnp.exp(
        jnp.asarray(orders_np, x.dtype).reshape(shape) * jnp.log(x)
        - jnp.asarray(log_double_factorial[orders_np], x.dtype).reshape(shape)
    )
    rows = jnp.where(x < floor, series, rows)
    if lo >= 2:
        return rows
    return jnp.concatenate([jnp.stack([j0, j1][lo:]), rows])


@partial(jax.custom_jvp, nondiff_argnums=(0, 1, 2))
def _band(
    lo: int, hi: int, recurrence: SphericalJnRecurrence, z: AnyArrayLike
) -> AnyArray:
    """Orders ``lo ... hi`` at real ``z``, stacked on a leading axis."""
    z_arr = as_float(z)
    finite = jnp.isfinite(z_arr)
    rows = _rows(lo, hi, recurrence, jnp.where(finite, jnp.abs(z_arr), 1.0))
    odd = (np.arange(lo, hi + 1) % 2 == 1).reshape((-1,) + (1,) * z_arr.ndim)
    rows = jnp.where(odd & (z_arr < 0.0), -rows, rows)
    rows = jnp.where(finite, rows, jnp.where(jnp.isnan(z_arr), jnp.nan, 0.0))
    return cast_like(rows, z)


def _derivative(
    lo: int, hi: int, recurrence: SphericalJnRecurrence, z: AnyArrayLike
) -> tuple[AnyArray, AnyArray]:
    """Values and derivatives, ``j_l' = (l j_{l-1} - (l+1) j_{l+1}) / (2l+1)``."""
    wide = _band(max(lo - 1, 0), hi + 1, recurrence, z)
    if lo == 0:  # j_{-1} enters with coefficient l = 0
        wide = jnp.concatenate([jnp.zeros_like(wide[:1]), wide])
    order = jnp.arange(lo, hi + 1, dtype=wide.dtype)
    order = order.reshape((-1,) + (1,) * (wide.ndim - 1))
    deriv = (order * wide[:-2] - (order + 1.0) * wide[2:]) / (2.0 * order + 1.0)
    return wide[1:-1], deriv


@_band.defjvp
def _band_jvp(
    lo: int,
    hi: int,
    recurrence: SphericalJnRecurrence,
    primals: tuple[Any],
    tangents: tuple[Any],
) -> tuple[AnyArray, AnyArray]:
    (z,), (dz,) = primals, tangents
    value, deriv = _derivative(lo, hi, recurrence, z)
    return value, deriv * dz


@partial(jax.jit, static_argnums=(0, 1), static_argnames=("derivative", "recurrence"))
def _evaluate(
    lo: int,
    hi: int,
    z: AnyArrayLike,
    *,
    derivative: bool,
    recurrence: SphericalJnRecurrence,
) -> AnyArray:
    if derivative:
        return _derivative(lo, hi, recurrence, z)[1]
    return _band(lo, hi, recurrence, z)


def _validate(n: int, z: AnyArrayLike) -> int:
    n = operator.index(n)
    if n < 0:
        msg = f"order n must be >= 0, got {n}"
        raise ValueError(msg)
    if jnp.iscomplexobj(z):
        msg = f"only real z is supported, got dtype {jnp.asarray(z).dtype}"
        raise ValueError(msg)
    return n


def spherical_jn(
    n: int,
    z: AnyArrayLike,
    derivative: bool = False,  # noqa: FBT001, FBT002 -- scipy's signature
    *,
    recurrence: SphericalJnRecurrenceLike = SphericalJnRecurrence.BOTH,
) -> AnyArray:
    r"""Compute the spherical Bessel function of the first kind, :math:`j_n(z)`.

    Equivalent to ``scipy.special.spherical_jn`` for real ``z``. Computed by
    recurrence from :math:`j_0` and :math:`j_1`, and ``recurrence`` chooses
    which (see `SphericalJnRecurrence`). Upward recurrence is stable above the
    turning point :math:`|z| \approx n` and unstable below it; Miller's
    downward recurrence is the reverse.

    - `SphericalJnRecurrence.BOTH`, the default, runs each on its own side. It
      is right everywhere, down to the smallest values, and is the choice
      unless you know where your arguments are. It pays for that: on CPU it is
      2-5x slower than `SphericalJnRecurrence.UP` and ~3x slower to
      compile.
    - `SphericalJnRecurrence.UP` is the fastest, and exact wherever
      :math:`|z| \ge n`. Use it when every argument is above the turning point,
      or when only values near the peak matter. Below the turning point it is
      noise: in float64, values under ~1e-6 of the peak are wrong in sign and
      magnitude or set to exactly zero, and in float32 the whole region is
      wrong, by up to 80x the peak.
    - `SphericalJnRecurrence.DOWN` is for arguments known to be below the
      turning point, e.g. small :math:`kr` in a large multipole. It is as
      accurate as `SphericalJnRecurrence.BOTH` there, skips the upward pass,
      and returns `nan` at :math:`|z| \ge n` (:math:`|z| \ge n + 1` for the
      derivative).

    Each ``n`` compiles separately, and so does each ``recurrence``. For many
    orders at the same ``z``, use `spherical_jn_all`, which computes them all at
    once.

    Parameters
    ----------
    n
        Order. Must be a static, non-negative Python `int`.
    z
        Real argument, of any shape. Evaluated elementwise.
    derivative
        If `True`, return :math:`j_n'(z)` instead.
    recurrence
        Which recurrence to run: a `SphericalJnRecurrence`, or its string value
        (``"both"``, ``"up"``, ``"down"``). Static.

    Returns
    -------
    Array
        Value(s) of :math:`j_n(z)` or :math:`j_n'(z)`. For :math:`|z| \ge n` the
        error is below 1e-12 of :math:`\sqrt{j_n^2 + y_n^2}` for
        :math:`n \le 10^4`, growing roughly as :math:`n` beyond (in float32,
        :math:`\max(10^{-5}, 10^{-7} n)` of the peak). Below the turning point,
        `SphericalJnRecurrence.BOTH` and `SphericalJnRecurrence.DOWN` are
        accurate *relatively*: to 5e-13 in float64 and 5e-5 in float32 for
        :math:`n \le 1000`, down to the smallest representable values.
        `SphericalJnRecurrence.UP` is accurate
        there only absolutely, to 1e-6 of :math:`\max_z |j_n(z)|` in float64
        for :math:`n \le 10^4` (3e-6 for the derivative).

    References
    ----------
    .. [1] J. Lesgourgues and T. Tram, "Fast and accurate CMB computations in
        non-flat FLRW universes", JCAP 09 (2014) 032, arXiv:1312.2697.
    .. [2] T. Tram, "Computation of hyperspherical Bessel functions",
        Communications in Computational Physics 22 (2017) 852, arXiv:1311.0839.

    Examples
    --------
    >>> import spexial as sp

    >>> round(float(sp.spherical_jn(2, 3.0)), 12)
    0.298637497076
    >>> round(float(sp.spherical_jn(1, 0.0, derivative=True)), 12)
    0.333333333333

    Far below the turning point the default is right, and upward recurrence
    alone returns noise, here with the wrong sign:

    >>> f"{float(sp.spherical_jn(5, 0.1)):.4e}"
    '9.6163e-10'
    >>> float(sp.spherical_jn(5, 0.1, recurrence=sp.SphericalJnRecurrence.UP)) < 0
    True

    """
    n = _validate(n, z)
    return _evaluate(
        n,
        n,
        z,
        derivative=bool(derivative),
        recurrence=SphericalJnRecurrence(recurrence),
    )[0]


def spherical_jn_all(
    n: int,
    z: AnyArrayLike,
    derivative: bool = False,  # noqa: FBT001, FBT002 -- as `spherical_jn`
    *,
    recurrence: SphericalJnRecurrenceLike = SphericalJnRecurrence.BOTH,
) -> AnyArray:
    r"""Return :math:`j_l(z)` for every order ``l = 0 ... n``.

    There is no `scipy.special` counterpart. The orders come from the same
    recurrences as `spherical_jn`, chosen the same way by ``recurrence``, with
    the same accuracy and cost. Each order has its own turning point, and the
    split is made at the highest one, ``|z| = n``: with
    `SphericalJnRecurrence.BOTH`, every row at ``|z| < n`` comes from the
    downward recurrence, and `SphericalJnRecurrence.DOWN` is `nan` for the
    whole column at ``|z| >= n``.

    Parameters
    ----------
    n
        Highest order. Must be a static, non-negative Python `int`.
    z
        Real argument, of any shape.
    derivative
        If `True`, return :math:`j_l'(z)` instead.
    recurrence
        As `spherical_jn`.

    Returns
    -------
    Array[float, (n + 1, ...)]
        The orders ``0 ... n`` stacked on a new leading axis.

    Examples
    --------
    >>> import spexial as sp

    >>> [round(v, 12) for v in sp.spherical_jn_all(3, 2.0).tolist()]
    [0.454648713413, 0.43539777498, 0.198447949057, 0.060722097663]

    """
    n = _validate(n, z)
    return _evaluate(
        0,
        n,
        z,
        derivative=bool(derivative),
        recurrence=SphericalJnRecurrence(recurrence),
    )
