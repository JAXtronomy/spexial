"""The gamma function."""

__all__ = ["gamma"]

from typing import Final

import jax.numpy as jnp

from .custom_types import AnyArray, RealArrayLike

_LANCZOS_G: Final = 7.0
"""The ``g`` parameter of the Lanczos approximation."""

_LANCZOS_P: Final = (
    0.99999999999980993,
    676.5203681218851,
    -1259.1392167224028,
    771.32342877765313,
    -176.61502916214059,
    12.507343278686905,
    -0.13857109526572012,
    9.9843695780195716e-6,
    1.5056327351493116e-7,
)
"""Lanczos coefficients for ``g = 7``, ``n = 9``, good to ~15 digits."""


def _lanczos(z: AnyArray) -> AnyArray:
    """Evaluate the Lanczos approximation, valid for ``Re(z) >= 0.5``."""
    p = jnp.asarray(_LANCZOS_P)
    zm1 = z - 1.0
    # `[..., None]` broadcasts the coefficient sum over a *trailing* axis so
    # that `z` of any shape is handled elementwise.
    series = p[0] + jnp.sum(p[1:] / (zm1[..., None] + jnp.arange(1, p.size)), axis=-1)
    t = zm1 + _LANCZOS_G + 0.5
    # Evaluated in log space: `t ** (zm1 + 0.5)` overflows for `z` above ~142,
    # well before the true gamma function does (at ~171.6).
    return jnp.sqrt(2 * jnp.pi) * jnp.exp((zm1 + 0.5) * jnp.log(t) - t) * series


def gamma(x: RealArrayLike, /) -> AnyArray:
    """Compute the gamma function using the Lanczos approximation.

    .. warning::

        Unlike `scipy.special.gamma`, this implementation supports **real
        arguments only**. The reflection formula needs an elementwise ``x <
        0.5`` test, which is not defined for complex input. Use
        `jax.scipy.special.gammaln` (or unwrap the Lanczos series yourself) if
        you need the complex gamma function.

    Reference:
    https://docs.scipy.org/doc/scipy/reference/generated/scipy.special.gamma.html

    Parameters
    ----------
    x
        Real-valued argument, of any shape. Evaluated elementwise.

    Returns
    -------
    Array
        Value(s) of the gamma function. Poles (``x`` a non-positive integer)
        give ``+inf``, matching `scipy.special.gamma`.

    Examples
    --------
    >>> import jax.numpy as jnp
    >>> import spexial as sp

    >>> round(float(sp.gamma(5.0)), 10)
    24.0

    It broadcasts over arrays, using the reflection formula below ``0.5``:

    >>> [round(float(g), 10) for g in sp.gamma(jnp.asarray([0.5, 1.0, -0.5]))]
    [1.7724538509, 1.0, -3.5449077018]

    Poles give ``inf``:

    >>> float(sp.gamma(0.0))
    inf

    """
    x_arr = jnp.asarray(x) * 1.0
    reflect = x_arr < 0.5
    # Feed the series a value it is valid for, then select; this keeps the
    # untaken branch free of `nan`, so `jax.grad` works either side of 0.5.
    lanczos = _lanczos(jnp.where(reflect, 1.0 - x_arr, x_arr))
    out = jnp.where(reflect, jnp.pi / (jnp.sin(jnp.pi * x_arr) * lanczos), lanczos)
    # `sin(pi * x)` is only ~1e-16, not 0, at the negative integers, so the
    # poles need to be put in by hand.
    is_pole = (x_arr <= 0) & (x_arr == jnp.floor(x_arr))
    return jnp.where(is_pole, jnp.inf, out)
