"""The Riemann zeta function."""

__all__ = ["zeta"]

from fractions import Fraction
from functools import cache
from math import factorial
from typing import Final

import jax.numpy as jnp
from jax.scipy.special import gammaln, zeta as _hurwitz_zeta

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


_ETA_TERMS: Final = 32
"""Terms in the Borwein acceleration used on the critical strip.

32 gives ~1e-15 over most of `0 < n < 1`, degrading to ~9e-13 as `n` approaches
the pole at 1, where the value itself is diverging.
"""


@cache
def _eta_coefficients() -> tuple[float, ...]:
    r"""Borwein's :math:`d_k`, built from exact integer arithmetic.

    .. math::

        d_k = N \sum_{i=0}^{k} \frac{(N+i-1)!\,4^i}{(N-i)!\,(2i)!}

    Exact `Fraction` arithmetic for the same reason `bernoulli_numbers` uses it:
    the terms span many orders of magnitude and a floating-point recurrence
    loses digits that the accelerated sum then cannot recover. The table is
    small and fixed, so it is built once.
    """
    n = _ETA_TERMS
    coefficients = []
    total = Fraction(0)
    for i in range(n + 1):
        total += Fraction(
            factorial(n + i - 1) * 4**i, factorial(n - i) * factorial(2 * i)
        )
        coefficients.append(float(n * total))
    return tuple(coefficients)


def _by_eta(n: AnyArray) -> AnyArray:
    r""":math:`\zeta(n)` on the critical strip, through the eta function.

    .. math::

        \zeta(s) = \frac{\eta(s)}{1 - 2^{1-s}},
        \qquad \eta(s) = \sum_{k \ge 1} \frac{(-1)^{k-1}}{k^s}

    `jax.scipy.special.zeta` does not implement `0 < n <= 1` and returns `nan`
    there, and the functional equation does not help: it maps the strip onto
    itself. The alternating series does converge, just far too slowly to use
    directly, so Borwein's acceleration supplies the answer in 32 terms.

    At ``n = 1`` the denominator is exactly 0 and the result is `inf`, which is
    the pole.
    """
    coefficients = jnp.asarray(_eta_coefficients(), dtype=n.dtype)
    last = coefficients[_ETA_TERMS]
    k = jnp.arange(1.0, _ETA_TERMS + 1.0, dtype=n.dtype)
    # `n[..., None]` puts the 32 terms on a *trailing* axis and sums over that
    # one only. Without it an array argument broadcasts against the term axis
    # and the shapes collide -- the same mistake `_K0_small` once made with a
    # bare `jnp.sum`, which silently collapsed the caller's own axis instead.
    weights = (-1.0) ** (k - 1.0) * (coefficients[:_ETA_TERMS] - last)
    eta = -jnp.sum(weights / k ** jnp.asarray(n)[..., None], axis=-1) / last
    return eta / (1.0 - 2.0 ** (1.0 - n))


def _by_reflection(n: AnyArray) -> AnyArray:
    r""":math:`\zeta(n)` for negative `n`, via the functional equation.

    .. math::

        \zeta(s) = 2^s \pi^{s-1} \sin(\pi s/2)\, \Gamma(1-s)\, \zeta(1-s)

    Every piece is already available: :math:`1 - s > 1` there, which is exactly
    the range `jax.scipy.special.zeta` covers, and `gammaln` supplies the rest.
    That makes the whole negative half-line reachable -- non-integers included,
    and integers of any magnitude -- where the Bernoulli functional equation
    reaches only the integers, and only as far as the table.

    Evaluated in log space. :math:`\Gamma(1-s)` overflows a double from
    :math:`s \approx -170.6`, while the *result* stays finite far beyond that
    (:math:`\zeta(-171) \approx 1.3\times10^{172}`), so forming the product
    directly would throw away a domain that is perfectly representable.
    """
    sine = jnp.sin(jnp.pi * n / 2)
    log_magnitude = (
        n * jnp.log(2.0)
        + (n - 1.0) * jnp.log(jnp.pi)
        + jnp.log(jnp.abs(sine))
        + gammaln(1.0 - n)
        + jnp.log(_hurwitz_zeta(1.0 - n, 1.0))
    )
    return jnp.sign(sine) * jnp.exp(log_magnitude)


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
    # The Bernoulli table is exact where it reaches -- 0 ulp against mpmath at
    # the negative odd integers, better than SciPy -- so it is kept for those.
    # Everything else on the negative half-line goes through the functional
    # equation, which used to be `nan`: non-integers, and odd integers past the
    # table. `1 - n` is safe there by construction, but the reflection is also
    # evaluated on the unselected positive branch, so feed it a negative
    # argument to keep `gammaln` and `log` off their own edges.
    from_table = sign * bernoulli_numbers().astype(n_arr.dtype)[index] / denom
    in_table = is_integer & (k_round + 1.0 <= ORDER)
    reflected = jnp.where(
        in_table, from_table, _by_reflection(jnp.where(n_arr < 0, n_arr, -0.5))
    )

    # `_hurwitz_zeta` is `nan` for n above ~1e15, and is exactly 1.0 for every n
    # past `_UNIT` anyway, so it is only ever called on the range it handles.
    unit = n_arr >= _UNIT
    # `0 < n <= 1` is the critical strip, which upstream does not implement; the
    # eta series covers it. Above 1, delegate as before. Both branches are
    # evaluated, so each gets an argument the other's domain can survive.
    strip = positive & (n_arr <= 1.0)
    above = jnp.where(positive & ~unit & ~strip, n_arr, 2.0)
    return jnp.where(
        positive,
        jnp.where(
            strip,
            _by_eta(jnp.where(strip, n_arr, 0.5)),
            jnp.where(unit, 1.0, _hurwitz_zeta(above, 1.0)),
        ),
        jnp.where(
            (n_arr < 0) & is_integer & (jnp.mod(k_round, 2.0) == 0.0), 0.0, reflected
        ),
    )
