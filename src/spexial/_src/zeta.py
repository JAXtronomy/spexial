"""The Riemann zeta function."""

__all__ = ["zeta"]

import jax.numpy as jnp
from jax.scipy.special import zeta as _hurwitz_zeta

from .bernoulli import ORDER, bernoulli_numbers
from .custom_types import AnyArray, RealArrayLike


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

    * ``n > 1`` -- delegated to `jax.scipy.special.zeta`.
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

    """
    n_arr = jnp.asarray(n) * 1.0
    positive = n_arr > 0
    k = -n_arr  # zeta(-k)
    k_int = jnp.round(k).astype(int)

    is_integer = k == k_int
    # Clip before indexing: `k + 1` is negative for n > -1 and past the end of
    # the table for n <= -60. Under `jax.jit` an out-of-bounds index is
    # silently clamped rather than raising, so the guard has to be explicit.
    index = jnp.clip(k_int + 1, 0, ORDER)
    # `(-1) ** k` would be `nan` under `jax.grad` (it differentiates through
    # `log(-1)`); take the sign off the *integer* k instead.
    sign = jnp.where(k_int % 2 == 0, 1.0, -1.0)
    # Likewise keep the denominator away from 0: k == -1 (i.e. n == 1) is the
    # pole, and belongs to the `positive` branch.
    denom = jnp.where(positive, 1.0, k + 1.0)
    reflected = jnp.where(
        ~is_integer | (k_int + 1 > ORDER),
        jnp.nan,
        sign * bernoulli_numbers()[index] / denom,
    )

    return jnp.where(
        positive,
        _hurwitz_zeta(jnp.where(positive, n_arr, 2.0), 1.0),
        jnp.where((n_arr < 0) & is_integer & (k_int % 2 == 0), 0.0, reflected),
    )
