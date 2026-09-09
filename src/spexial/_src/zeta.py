"""The Riemann zeta function."""

__all__ = ["zeta"]

from typing import Final

import jax.numpy as jnp
from jax.scipy.special import zeta as _hurwitz_zeta

from .bernoulli import ORDER, bernoulli_numbers
from .custom_types import AnyArray, RealArrayLike

_UNIT: Final = 54.0
"""At and above this, :math:`\\zeta(n)` is exactly 1 in float64.

:math:`\\zeta(n) - 1 \\approx 2^{-n}`, which falls below half an eps of 1 once
``n > 53``, so every double-precision value from here up is ``1.0`` --
``zeta(54)`` included, which is why the guard is ``>=`` and not ``>``.
Returning
the constant is not an approximation, and it sidesteps
`jax.scipy.special.zeta`, which gives `nan` above ``n`` of about ``1e15``.
"""


def zeta(n: RealArrayLike, /) -> AnyArray:
    r"""Compute the Riemann zeta function :math:`\zeta(n)`.

    Differs from `jax.scipy.special.zeta` in that negative arguments are
    supported, through the functional equation
    :math:`\zeta(-k) = (-1)^k B_{k+1} / (k+1)`.

    Reference:
    https://docs.scipy.org/doc/scipy/reference/generated/scipy.special.zeta.html

    Parameters
    ----------
    n
        Real argument, of any shape. Evaluated elementwise.

    Returns
    -------
    Array
        Value(s) of :math:`\zeta(n)`, or `nan` outside the supported domain
        (see below). ``n = 1`` is the pole and gives ``inf``.

    Notes
    -----
    The negative half-line is only supported where the functional equation can
    be evaluated from the tabulated Bernoulli numbers:

    * ``n > 1`` -- delegated to `jax.scipy.special.zeta`, except at and above
      ``n = 54`` where the exact double-precision value is ``1.0``. Taking that
      constant also avoids `jax.scipy.special.zeta` returning `nan` for ``n``
      above roughly ``1e15``.
    * ``0 < n <= 1`` -- `nan`. `jax.scipy.special.zeta` does not implement the
      critical strip, and this function does not paper over that; use
      `scipy.special.zeta` on the host if you need it.
    * ``n`` a negative *even* integer -- exactly 0, at any magnitude.
    * ``n`` a negative *odd* integer with ``n > -60`` -- from :math:`B_{1-n}`.
    * ``n <= -60`` and odd -- `nan`; the Bernoulli table stops at
      :math:`B_{60}`.
    * ``n < 0`` and *not* an integer -- `nan`. The functional equation used here
      needs :math:`(-1)^{-n}`, which is undefined for non-integers, so unlike
      `scipy.special.zeta` this implementation does not cover, e.g.,
      ``zeta(-0.5)``.

    `jax.grad` is only meaningful for ``n > 1``. On the negative half-line the
    value comes out of a Bernoulli *table*, which carries no information about
    how :math:`\zeta` varies between the integers, so the derivative reported
    there is an artefact -- finite, but not :math:`\zeta'`.

    Examples
    --------
    >>> import jax.numpy as jnp
    >>> import spexial as sp

    >>> round(float(sp.zeta(2.0)), 10)
    1.6449340668

    Negative integers use the functional equation:

    >>> [round(float(z), 12) for z in sp.zeta(jnp.asarray([0.0, -1.0, -2.0, -3.0]))]
    [-0.5, -0.083333333333, 0.0, 0.008333333333]

    ``zeta(-3) == 1 / 120``:

    >>> float(1 / 120)
    0.008333333333333333

    Large arguments are exactly 1, where `jax.scipy.special.zeta` gives `nan`:

    >>> float(sp.zeta(1e16))
    1.0

    """
    # `* 1.0` promotes integers; the Bernoulli table is float64 by construction
    # (exact `Fraction` arithmetic), so it is cast down to `n_arr`'s dtype below
    # rather than being allowed to widen a float32 argument to float64.
    n_arr = jnp.asarray(n) * 1.0
    positive = n_arr > 0
    k = -n_arr  # zeta(-k)
    # Kept in float throughout: `astype(int)` canonicalises to int32 unless x64
    # is on, which would overflow the parity and range tests around 2.1e9
    # instead of the 9.2e18 the docs claim. Float is exact to 2^53 either way.
    k_round = jnp.round(k)

    is_integer = k == k_round
    # Clip *before* the cast, so the index cannot overflow whatever width `int`
    # happens to be: `k + 1` is negative for n > -1 and past the end of the
    # table for n <= -60. Under `jax.jit` an out-of-bounds index is silently
    # clamped rather than raising, so the guard has to be explicit.
    index = jnp.clip(k_round + 1.0, 0.0, ORDER).astype(int)
    # `(-1) ** k` would be `nan` under `jax.grad` (it differentiates through
    # `log(-1)`); take the sign off the parity of k instead.
    sign = jnp.where(jnp.mod(k_round, 2.0) == 0.0, 1.0, -1.0)
    # Likewise keep the denominator away from 0: k == -1 (i.e. n == 1) is the
    # pole, and belongs to the `positive` branch.
    denom = jnp.where(positive, 1.0, k + 1.0)
    reflected = jnp.where(
        ~is_integer | (k_round + 1.0 > ORDER),
        jnp.nan,
        sign * bernoulli_numbers().astype(n_arr.dtype)[index] / denom,
    )

    # `_hurwitz_zeta` is `nan` for n above ~1e15, and is exactly 1.0 for every n
    # past `_UNIT` anyway, so it is only ever called on the range it handles.
    unit = n_arr >= _UNIT
    return jnp.where(
        positive,
        jnp.where(
            unit, 1.0, _hurwitz_zeta(jnp.where(positive & ~unit, n_arr, 2.0), 1.0)
        ),
        jnp.where(
            (n_arr < 0) & is_integer & (jnp.mod(k_round, 2.0) == 0.0), 0.0, reflected
        ),
    )
