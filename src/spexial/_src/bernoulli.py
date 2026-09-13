"""The Bernoulli numbers :math:`B_n`.

Note that this module is NOT public API nor are any of its contents.
Stability is NOT guaranteed.

Both `spexial._src.zeta` and `spexial._src.polylog` need the same table. It is
built once, lazily, from exact `fractions.Fraction` arithmetic -- importing
`spexial` therefore neither duplicates the table nor touches a JAX device.

Using exact arithmetic is not gold-plating: `jax.scipy.special.bernoulli` (and
`scipy.special.bernoulli`) evaluate the numbers by a numerically unstable route
that loses ~7 digits on :math:`B_4` and ~6 on :math:`B_6`, which propagates
straight into ``zeta(-3)`` and ``polylog(4, ...)``.

"""

__all__: tuple[str, ...] = ()

from fractions import Fraction
from functools import cache
from math import comb
from typing import Final

import jax.numpy as jnp

from .custom_types import Vector

ORDER: Final = 60
"""Highest index in the table, i.e. the table holds :math:`B_0 \\ldots B_{60}`."""


@cache
def bernoulli_fractions() -> tuple[Fraction, ...]:
    r"""Build :math:`B_0 \ldots B_{60}` exactly, as `fractions.Fraction`.

    Exposed unrounded because `zeta` needs :math:`B_{k+1}/(k+1)`, and doing
    that division after rounding is a rounding of a rounding -- one ulp off at
    five of the negative integers.
    """
    bs = [Fraction(1)] + [Fraction(0)] * ORDER
    for m in range(1, ORDER + 1):
        bs[m] = -sum(comb(m + 1, k) * bs[k] for k in range(m)) / (m + 1)
    return tuple(bs)


@cache
def _bernoulli_floats() -> tuple[float, ...]:
    r"""Build :math:`B_0 \ldots B_{60}` exactly, then round to `float`."""
    return tuple(float(b) for b in bernoulli_fractions())


def bernoulli_numbers() -> Vector:
    r"""Return :math:`B_0, \ldots, B_{60}` as a float array.

    Uses the convention :math:`B_1 = -1/2`, matching `scipy.special.bernoulli`.

    Only the Python-level table is cached: caching the `jax.Array` itself would
    capture a tracer if the first call happened inside a `jax.jit` trace.
    Re-wrapping the cached tuple is trace-time-only work.

    Returns
    -------
    Array[float, (61,)]
        The Bernoulli numbers, exactly rounded to the default float dtype.

    Examples
    --------
    >>> from spexial._src.bernoulli import bernoulli_numbers
    >>> [float(b) for b in bernoulli_numbers()[:5]]
    [1.0, -0.5, 0.16666666666666666, 0.0, -0.03333333333333333]

    """
    return jnp.asarray(_bernoulli_floats())
