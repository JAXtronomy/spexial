"""Spherical Bessel functions of the first kind."""

__all__ = ["spherical_jn", "spherical_jn_all"]

import operator
from functools import partial
from typing import Any, Final, TypeAlias

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


def _rows(lo: int, hi: int, x: AnyArray) -> AnyArray:
    """Orders ``lo ... hi`` at finite ``x >= 0``.

    Upward recurrence is stable above the turning point, and in float64 the
    CLASS cutoff keeps its noise below it to ~1e-7 of the peak. That noise is
    ``eps / _CUTOFF``, so in float32 it is ~600x the peak, and no cutoff fixes
    it: one large enough zeroes most of `j_n` for ``n >~ 100``. Below the
    turning point float32 therefore uses Miller's downward recurrence instead
    (spexial#51), which is stable there.
    """
    if hi <= 1 or jnp.finfo(x.dtype).bits >= 64:
        return _upward(lo, hi, x)
    return jnp.where(x >= hi, _upward(lo, hi, x), _downward(lo, hi, x))


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
    # Below this `j_2 ~ x^2 / 15` is under `tiny`, and `(2l + 1) / x` would
    # overflow the carry.
    floor = float(np.sqrt(15.0 * jnp.finfo(x.dtype).tiny))
    inv_x = 1.0 / jnp.maximum(x, floor)

    def step(
        carry: tuple[AnyArray, AnyArray, AnyArray], order: AnyArray
    ) -> tuple[tuple[AnyArray, AnyArray, AnyArray], _Carry]:
        above, cur, exponent = carry  # f_{l+1}, f_l, both times 2**-exponent
        below = (2.0 * order + 1.0) * inv_x * cur - above
        # One step grows by at most `(2 top + 1) / floor`, under 2**87 for any
        # n < 10**7, so a carry kept below 2**40 cannot overflow float32's
        # 2**128. The factor
        # is an `exp2` of a 0/1 flag, not a `where`: under `_UNROLL` XLA fuses
        # the selects and recomputes them, which made this loop 7x slower.
        shift = _SHIFT * (jnp.abs(below) > _RESCALE_AT).astype(x.dtype)
        factor = jnp.exp2(-shift)
        return (cur * factor, below * factor, exponent + shift), (cur, exponent)

    def skip(
        carry: tuple[AnyArray, AnyArray, AnyArray], order: AnyArray
    ) -> tuple[tuple[AnyArray, AnyArray, AnyArray], None]:
        return step(carry, order)[0], None

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
    rows = jnp.where(x < floor, 0.0, rows)
    if lo >= 2:
        return rows
    return jnp.concatenate([jnp.stack([j0, j1][lo:]), rows])


@partial(jax.custom_jvp, nondiff_argnums=(0, 1))
def _band(lo: int, hi: int, z: AnyArrayLike) -> AnyArray:
    """Orders ``lo ... hi`` at real ``z``, stacked on a leading axis."""
    z_arr = as_float(z)
    finite = jnp.isfinite(z_arr)
    rows = _rows(lo, hi, jnp.where(finite, jnp.abs(z_arr), 1.0))
    odd = (np.arange(lo, hi + 1) % 2 == 1).reshape((-1,) + (1,) * z_arr.ndim)
    rows = jnp.where(odd & (z_arr < 0.0), -rows, rows)
    rows = jnp.where(finite, rows, jnp.where(jnp.isnan(z_arr), jnp.nan, 0.0))
    return cast_like(rows, z)


def _derivative(lo: int, hi: int, z: AnyArrayLike) -> tuple[AnyArray, AnyArray]:
    """Values and derivatives, ``j_l' = (l j_{l-1} - (l+1) j_{l+1}) / (2l+1)``."""
    wide = _band(max(lo - 1, 0), hi + 1, z)
    if lo == 0:  # j_{-1} enters with coefficient l = 0
        wide = jnp.concatenate([jnp.zeros_like(wide[:1]), wide])
    order = jnp.arange(lo, hi + 1, dtype=wide.dtype)
    order = order.reshape((-1,) + (1,) * (wide.ndim - 1))
    deriv = (order * wide[:-2] - (order + 1.0) * wide[2:]) / (2.0 * order + 1.0)
    return wide[1:-1], deriv


@_band.defjvp
def _band_jvp(
    lo: int, hi: int, primals: tuple[Any], tangents: tuple[Any]
) -> tuple[AnyArray, AnyArray]:
    (z,), (dz,) = primals, tangents
    value, deriv = _derivative(lo, hi, z)
    return value, deriv * dz


@partial(jax.jit, static_argnums=(0, 1), static_argnames=("derivative",))
def _evaluate(lo: int, hi: int, z: AnyArrayLike, *, derivative: bool) -> AnyArray:
    return _derivative(lo, hi, z)[1] if derivative else _band(lo, hi, z)


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
) -> AnyArray:
    r"""Compute the spherical Bessel function of the first kind, :math:`j_n(z)`.

    Equivalent to ``scipy.special.spherical_jn`` for real ``z``. Computed by
    upward recurrence from :math:`j_0` and :math:`j_1`. The recurrence is
    unstable below the turning point :math:`|z| \approx n`, so there values
    smaller than about :math:`10^{-6}` of the peak (for :math:`n \le 10^4`,
    growing roughly as :math:`n^{5/6}` beyond) are unreliable in sign and
    magnitude, and those far enough below it are returned as exactly zero.
    That describes float64. In float32 and narrower, the part below the turning
    point comes from Miller's downward recurrence instead, which is stable
    there: the error is below 1e-5 of the peak, with no noise band.

    Each ``n`` compiles separately. For many orders at the same ``z``, use
    `spherical_jn_all`, which computes them all at once.

    Parameters
    ----------
    n
        Order. Must be a static, non-negative Python `int`.
    z
        Real argument, of any shape. Evaluated elementwise.
    derivative
        If `True`, return :math:`j_n'(z)` instead.

    Returns
    -------
    Array
        Value(s) of :math:`j_n(z)` or :math:`j_n'(z)`. For :math:`|z| \ge n` the
        error is below 1e-12 of :math:`\sqrt{j_n^2 + y_n^2}` for
        :math:`n \le 10^4`, growing roughly as :math:`n` beyond. Below the turning
        point it is absolute: below 1e-6 of :math:`\max_z |j_n(z)|` for
        :math:`n \le 10^4`, and 3e-6 of :math:`\max_z |j_n'(z)|` for the
        derivative. In float32 the error is below 1e-5 of the peak below the
        turning point, and above it grows roughly as :math:`10^{-7} n` of the
        peak, from rounding accumulated over the recurrence.

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

    """
    n = _validate(n, z)
    return _evaluate(n, n, z, derivative=bool(derivative))[0]


def spherical_jn_all(
    n: int,
    z: AnyArrayLike,
    derivative: bool = False,  # noqa: FBT001, FBT002 -- as `spherical_jn`
) -> AnyArray:
    r"""Return :math:`j_l(z)` for every order ``l = 0 ... n``.

    There is no `scipy.special` counterpart. The orders come from the same
    recurrence as `spherical_jn`, with the same accuracy.

    Parameters
    ----------
    n
        Highest order. Must be a static, non-negative Python `int`.
    z
        Real argument, of any shape.
    derivative
        If `True`, return :math:`j_l'(z)` instead.

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
    return _evaluate(0, n, z, derivative=bool(derivative))
