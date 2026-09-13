"""The Riemann zeta function."""

__all__ = ["zeta"]

from fractions import Fraction
from functools import cache
from math import factorial
from typing import Final

import jax.numpy as jnp
from jax.scipy.special import gammaln, zeta as _hurwitz_zeta

from .bernoulli import ORDER, bernoulli_fractions
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

32 holds to ~1e-15 across the whole strip, including right up against the pole
at 1, and keeps that accuracy a short way below zero as well -- 2e-15 at
``n = -0.5``. It decays from there (2e-9 by ``n = -5``), which is what
`_ETA_FLOOR` is for.
"""

_ETA_FLOOR: Final = -0.5
"""Below this the reflection takes over from the eta series.

The series is needed below zero because the reflection forms ``1 - n``, which
loses ``n`` as it shrinks: the reflection is 1.1e-13 by ``n = -1e-3``, 8.3e-8 by
``-1e-9``, and ``-inf`` once ``|n|`` drops under half an eps, where ``1 - n``
rounds to exactly ``1`` and puts the pole of :math:`\\zeta(1)` into a formula
whose answer is a finite ``-0.5``.

The floor is *not* the point where the two measure equal -- below about
``-0.05`` they interleave, both at 1e-15, with the reflection ahead as often as
not. It sits at ``-0.5`` because that is comfortably inside the region where the
reflection is established and well clear of where it collapses; nothing between
``-0.5`` and ``-0.05`` distinguishes them.
"""


@cache
def _eta_weights() -> tuple[float, ...]:
    r"""Borwein's :math:`d_k`, normalised by :math:`d_n` and offset.

    .. math::

        d_k = n \sum_{i=0}^{k} \frac{(n+i-1)!\,4^i}{(n-i)!\,(2i)!},
        \qquad w_k = \frac{d_k - d_n}{d_n}

    Exact `Fraction` arithmetic for the same reason `bernoulli_numbers` uses it:
    the terms span many orders of magnitude and a floating-point recurrence
    loses digits the accelerated sum cannot recover.

    Returning the *ratio* rather than the raw :math:`d_k` is what makes this
    work at every width. The largest :math:`d_k` here is 1.6e24, which overflows
    `float16` on the cast to the argument's dtype -- taking `last` to `inf` and
    every value on ``-0.5 < n < 1`` to `nan`, silently. Dividing through first
    puts every entry in ``[-1, 0]``, where no float dtype can overflow, and the
    common factor :math:`n` cancels on the way.
    """
    n = _ETA_TERMS
    partial = []
    total = Fraction(0)
    for i in range(n + 1):
        total += Fraction(
            factorial(n + i - 1) * 4**i, factorial(n - i) * factorial(2 * i)
        )
        partial.append(total)
    last = partial[n]
    return tuple(float((partial[k] - last) / last) for k in range(n))


@cache
def _zeta_at_negative_integers() -> tuple[float, ...]:
    r"""``B_i / i`` for :math:`i = 0 \ldots` `ORDER`, rounded once rather than twice.

    :math:`\zeta(-k) = (-1)^k B_{k+1} / (k+1)`, and doing that division in
    floating point costs a second rounding: `bernoulli_numbers` has already
    rounded :math:`B_{k+1}`, so ``float(B) / (k + 1)`` is a rounding of a
    rounding and lands one ulp off the correctly-rounded answer at
    ``n = -11, -13, -23, -27, -33``. The coverage table calls this column
    "exact (0 ulp)", so one ulp is one too many.

    Dividing inside `Fraction` and converting once makes the claim true. The
    tabulated value keeps ``B_i``'s own alternating sign; the caller's
    ``(-1)**k`` factor is separate from it.
    """
    exact = bernoulli_fractions()
    # Indexed by the caller's clipped ``k + 1``, so entry ``i`` holds
    # ``|B_i / i|``. Entry 0 is unreachable -- ``k + 1 <= 0`` means ``n >= 1``,
    # which the positive branch owns -- and exists only to keep the offsets
    # lined up.
    return (0.0, *(float(exact[i] / i) for i in range(1, ORDER + 1)))


def _by_eta(n: AnyArray) -> AnyArray:
    r""":math:`\zeta(n)` on the critical strip, through the eta function.

    .. math::

        \zeta(s) = \frac{\eta(s)}{1 - 2^{1-s}},
        \qquad \eta(s) = \sum_{k \ge 1} \frac{(-1)^{k-1}}{k^s}

    `jax.scipy.special.zeta` does not implement `0 < n <= 1` and returns `nan`
    there, and the functional equation does not help: it maps the strip onto
    itself. The alternating series does converge, just far too slowly to use
    directly, so Borwein's acceleration supplies the answer in 32 terms.

    The same series continues to hold below zero, so `zeta` also uses it down
    to `_ETA_FLOOR` -- see there for why the reflection cannot cover that part.
    """
    weights_table = jnp.asarray(_eta_weights(), dtype=n.dtype)
    k = jnp.arange(1.0, _ETA_TERMS + 1.0, dtype=n.dtype)
    # `n[..., None]` puts the 32 terms on a *trailing* axis and sums over that
    # one only. Without it an array argument broadcasts against the term axis
    # and the shapes collide -- the same mistake `_k0_small` once made with a
    # bare `jnp.sum`, which silently collapsed the caller's own axis instead.
    weights = (-1.0) ** (k - 1.0) * weights_table
    eta = -jnp.sum(weights / k ** jnp.asarray(n)[..., None], axis=-1)
    # `1 - 2**(1-n)` cancels to nothing as `n` approaches 1 -- it is exactly 0
    # half an eps below it, and only ~4 digits survive by `1 - 1e-12`. `expm1`
    # of the same quantity carries every digit, which matters because the pole
    # it is dividing by is what makes the value large in the first place.
    return eta / -jnp.expm1((1.0 - n) * jnp.log(2.0))


def _sin_pi_half(n: AnyArray) -> AnyArray:
    r"""Evaluate :math:`\sin(\pi n / 2)` accurately near its zeros.

    ``jnp.sin(jnp.pi * n / 2)`` loses the answer near an even integer, where the
    sine is small but ``pi * n / 2`` is not: at :math:`n = -102 + 4\times10^{-15}`
    the product carries an absolute rounding error of about ``3e-14`` while the
    true sine is ``1e-14``, so nothing survives. Reducing the argument first
    keeps the small residue exact -- ``half - nearest`` is a subtraction of two
    nearby values, so it is exact in floating point -- and the sine is then
    evaluated where it has full relative precision.
    """
    half = n / 2.0
    nearest = jnp.round(half)
    parity = jnp.where(jnp.mod(nearest, 2.0) == 0.0, 1.0, -1.0)
    return parity * jnp.sin(jnp.pi * (half - nearest))


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
    sine = _sin_pi_half(n)
    log_magnitude = (
        n * jnp.log(2.0)
        + (n - 1.0) * jnp.log(jnp.pi)
        + jnp.log(jnp.abs(sine))
        + gammaln(1.0 - n)
        # `1 - n` runs past 1e15 for n below about -1e15, where
        # `jax.scipy.special.zeta` returns `nan` -- and zeta is exactly 1 from
        # `_UNIT` up, so clamp to the range it can answer. The positive branch
        # has always done this; the reflection's own argument was missed, and
        # returned `nan` where SciPy gives `±inf`.
        + jnp.log(_hurwitz_zeta(jnp.minimum(1.0 - n, _UNIT), 1.0))
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
    Every real ``n`` is covered, by whichever of four methods is accurate there:

    * ``n >= 54`` -- exactly ``1.0``. :math:`\zeta(n) - 1 \approx 2^{-n}` is
      below half an eps of 1 from there up, so this is the exact
      double-precision value rather than an approximation. It also avoids
      `jax.scipy.special.zeta`, which returns `nan` for ``n`` above about
      ``1e15``.
    * ``n > 1`` -- delegated to `jax.scipy.special.zeta`.
    * ``n = 1`` -- the pole, ``inf``.
    * ``-0.5 < n < 1`` -- Borwein's acceleration of the eta series. This is the
      critical strip, which `jax.scipy.special.zeta` does not implement, plus a
      little below zero where the reflection below cannot be used.
    * ``n`` a negative *even* integer -- exactly ``0``, at any magnitude.
    * ``n`` a negative integer down to ``-59`` -- from the tabulated
      :math:`B_{1-n}`, which is exact to the ulp. ``-60`` is covered too, but
      as an even integer rather than by the table, which ends at
      :math:`B_{60}`.
    * every other ``n <= -0.5`` -- the functional equation
      :math:`\zeta(s) = 2^s \pi^{s-1} \sin(\pi s/2) \Gamma(1-s) \zeta(1-s)`,
      evaluated in log space so that :math:`\Gamma(1-s)` overflowing at
      :math:`s \approx -170.6` does not cost a domain the result is finite on.

    `jax.grad` is genuine wherever the eta series or the functional equation
    supplies the value, which is everywhere except three sets, all of which
    report a finite number that is not :math:`\zeta'`: the tabulated integers
    ``0 >= n >= -59``, where a *table* carries no information about how
    :math:`\zeta` varies between its entries; the negative *even* integers at
    any magnitude, which are a constant ``0``; and ``n >= 54``, where the value
    is the constant ``1.0`` and the reported derivative is ``0`` against a true
    :math:`\zeta'(54) = -3.8\times10^{-17}`. The odd integers past the table
    are fine -- ``grad`` at ``n = -101`` matches :math:`\zeta'` to 6e-14 --
    because those go through the functional equation, which differentiates.

    Accuracy is ``6e-16`` for ``n > 1``, ``2.3e-15`` on the critical strip
    itself and ``1e-14`` on the window below zero the same series covers, where
    it works hardest right against `_ETA_FLOOR`. On the
    negative line it degrades with ``|n|``, because ``gammaln(1 - n)`` grows and
    the exponential of it carries that magnitude's rounding: ``9e-15`` out to
    ``n = -10``, ``2.2e-13`` by ``-100`` and ``6e-13`` by ``-260``, past which
    the true value exceeds ``DBL_MAX`` and the answer is ``±inf`` -- as it is in
    SciPy -- so only the trivial zeros and their neighbours are finite. Just off a
    negative even integer, where the sine of the functional equation is near a
    zero of its own, it is a few times ``1e-13`` -- still better than SciPy,
    which is ``2e-4`` there.

    Those are the *scalar* figures. XLA re-associates the 32-term eta sum
    differently once there is a batch axis, so an array or `jax.jit` argument
    can differ from the scalar one by up to ``1.5e-14`` on the strip -- and by
    more on the window below zero the same series covers, where a batched call
    measures ``2.7e-14`` against the scalar ``8.2e-15``. The caveat belongs to
    the eta series, not to the strip alone.

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
    # The Bernoulli table is exact where it reaches -- 0 ulp against mpmath at
    # the negative odd integers, better than SciPy -- so it is kept for those.
    # Everything else on the negative half-line goes through the functional
    # equation, which used to be `nan`: non-integers, and odd integers past the
    # table. `1 - n` is safe there by construction, but the reflection is also
    # evaluated on the unselected positive branch, so feed it a negative
    # argument to keep `gammaln` and `log` off their own edges.
    # Built at float64 and *then* narrowed, never constructed at the caller's
    # dtype: entries reach 1e32, and asking numpy for a float16 array of those
    # emits an overflow warning that `filterwarnings = ["error"]` turns into a
    # failure -- even on the strip, where this branch is not selected at all.
    table = jnp.asarray(_zeta_at_negative_integers()).astype(n_arr.dtype)
    from_table = sign * table[index]
    in_table = is_integer & (k_round + 1.0 <= ORDER)

    # `_hurwitz_zeta` is `nan` for n above ~1e15, and is exactly 1.0 for every n
    # past `_UNIT` anyway, so it is only ever called on the range it handles.
    unit = n_arr >= _UNIT
    # The eta series takes the critical strip, which upstream does not
    # implement, and continues below zero as far as `_ETA_FLOOR`. The integers
    # it would otherwise cover -- only ``n = 0`` lies in the window -- stay with
    # the table, which is exact there.
    by_eta = (n_arr < 1.0) & (n_arr > _ETA_FLOOR) & ~in_table
    # Every branch is evaluated whatever the argument, so each gets one its own
    # domain can survive when it is not the branch being selected.
    above = jnp.where(positive & ~unit & (n_arr > 1.0), n_arr, 2.0)
    eta_arg = jnp.where(by_eta, n_arr, 0.5)
    trivial_zero = (n_arr < 0) & is_integer & (jnp.mod(k_round, 2.0) == 0.0)
    # The trivial zeros are excluded from the reflection's argument as well as
    # from its result. Reducing the sine's argument makes it exactly 0 there
    # rather than the 1.2e-16 an unreduced `sin(pi * n / 2)` leaves behind, so
    # `log(|sin|)` is `-inf` and its derivative `nan` -- and `jnp.where` takes
    # the `nan` from the branch it did not select.
    reflect_arg = jnp.where((n_arr <= _ETA_FLOOR) & ~trivial_zero, n_arr, -1.5)
    result = jnp.where(
        unit,
        1.0,
        jnp.where(
            n_arr > 1.0,
            _hurwitz_zeta(above, 1.0),
            jnp.where(
                n_arr == 1.0,
                jnp.inf,  # the pole, not a guard around one
                jnp.where(
                    by_eta,
                    _by_eta(eta_arg),
                    jnp.where(
                        trivial_zero,
                        0.0,
                        jnp.where(in_table, from_table, _by_reflection(reflect_arg)),
                    ),
                ),
            ),
        ),
    )
    # `nan > 0` is False, so a `nan` argument would otherwise fall down the
    # negative side and come back as whatever the placeholder above evaluates
    # to -- a real-looking number in place of the `nan` that went in.
    return jnp.where(jnp.isnan(n_arr), jnp.nan, result)
