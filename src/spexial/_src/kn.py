"""Modified Bessel functions of the second kind, integer order."""

__all__ = ["K0", "K1", "K2"]

from typing import Final

import jax.numpy as jnp
from jax.scipy.special import gammaln, i0, i1

from .custom_types import AnyArray, RealArrayLike

_EULER_GAMMA: Final = 0.57721566490153286061
"""The Euler-Mascheroni constant."""

_SMALL_Z: Final = 9.0
"""Cross-over between the ascending series and the asymptotic expansion."""

_MAX_Z: Final = 700.0
"""Above this `jax.scipy.special.i0` overflows, so `K1` is taken to underflow.

`scipy.special.kn` also returns 0 from ~700 upwards, so this matches it.
"""

_N_SMALL: Final = 30
"""Terms in the ascending series; enough for ~1e-8 relative accuracy at z < 9."""

_N_LARGE: Final = 10
"""Terms in the asymptotic series; enough for ~1e-8 relative accuracy at z > 9."""


def _K0_small(z: AnyArray) -> AnyArray:  # noqa: N802
    """Ascending series for `K0`; see Zhang & Jin, *Special Functions* (1996)."""
    k = jnp.arange(1.0, _N_SMALL + 1.0)
    harmonic = jnp.cumsum(1.0 / k)
    # `z[..., None]` sums over a *trailing* axis: without it `jnp.sum` collapses
    # the caller's own axis and an array argument silently yields one scalar.
    log_term = 2.0 * k * jnp.log(z[..., None] / 2.0) - 2.0 * gammaln(k + 1.0)
    return -(jnp.log(z / 2.0) + _EULER_GAMMA) * i0(z) + jnp.sum(
        harmonic * jnp.exp(log_term), axis=-1
    )


def _K0_large(z: AnyArray) -> AnyArray:  # noqa: N802
    """Asymptotic expansion for `K0`, via the ``1 / (2 z I0(z))`` form."""
    k = jnp.arange(1.0, _N_LARGE + 1.0)
    prod = jnp.cumprod(-(2.0 * k - 1.0) / (2.0 * k) * (2.0 * k - 1.0) ** 2.0)
    series = 1.0 + jnp.sum(
        (-1.0) ** k * prod / (2.0 * z[..., None]) ** (2.0 * k), axis=-1
    )
    return series / (2.0 * z * i0(z))


def K0(z: RealArrayLike, /) -> AnyArray:  # noqa: N802
    """Compute the modified Bessel function of the second kind of order 0.

    Equivalent to ``scipy.special.kn(0, z)``. See Zhang and Jin,
    ``SPECIAL_FUNCTIONS`` in FORTRAN77, for the algorithm: an ascending series
    below ``z = 9`` and an asymptotic expansion above it.

    Parameters
    ----------
    z
        Real positive argument, of any shape. Evaluated elementwise. ``z <= 0``
        is outside the domain and gives ``inf`` (at 0) or `nan`.

    Returns
    -------
    Array
        Value(s) of :math:`K_0(z)`, accurate to ~3e-8 relative (worst near the
        ``z = 9`` cross-over; ~1e-10 or better away from it).

    Examples
    --------
    >>> import jax.numpy as jnp
    >>> import spexial as sp

    >>> round(float(sp.K0(1.0)), 8)
    0.42102444

    Array input is evaluated elementwise, spanning both branches:

    >>> [round(float(k), 8) for k in sp.K0(jnp.asarray([0.5, 5.0, 20.0]))]
    [0.92441907, 0.0036911, 0.0]

    """
    z_arr = jnp.asarray(z) * 1.0
    small = z_arr < _SMALL_Z
    return jnp.where(
        small,
        _K0_small(jnp.where(small, z_arr, 1.0)),
        _K0_large(jnp.where(small, _SMALL_Z, z_arr)),
    )


def K1(z: RealArrayLike, /) -> AnyArray:  # noqa: N802
    """Compute the modified Bessel function of the second kind of order 1.

    Obtained from `K0` through the Wronskian
    :math:`I_0(z) K_1(z) + I_1(z) K_0(z) = 1/z`.

    Parameters
    ----------
    z
        Real positive argument, of any shape. Evaluated elementwise. ``z <= 0``
        is outside the domain and gives `nan`.

    Returns
    -------
    Array
        Value(s) of :math:`K_1(z)`, accurate to ~3e-8 relative. Underflows to 0
        for ``z >= 700``, where `jax.scipy.special.i0` overflows.

    Examples
    --------
    >>> import jax.numpy as jnp
    >>> import spexial as sp

    >>> round(float(sp.K1(1.0)), 8)
    0.60190723

    >>> [round(float(k), 8) for k in sp.K1(jnp.asarray([0.5, 5.0, 20.0]))]
    [1.65644112, 0.00404461, 0.0]

    """
    z_arr = jnp.asarray(z) * 1.0
    finite = z_arr < _MAX_Z
    z_safe = jnp.where(finite, z_arr, 1.0)
    k1 = (1.0 / z_safe - i1(z_safe) * K0(z_safe)) / i0(z_safe)
    return jnp.where(finite, k1, 0.0)


def K2(z: RealArrayLike, /) -> AnyArray:  # noqa: N802
    """Compute the modified Bessel function of the second kind of order 2.

    Obtained from the recurrence :math:`K_2(z) = K_0(z) + (2/z) K_1(z)`.

    Parameters
    ----------
    z
        Real positive argument, of any shape. Evaluated elementwise. ``z <= 0``
        is outside the domain and gives `nan`.

    Returns
    -------
    Array
        Value(s) of :math:`K_2(z)`, accurate to ~2e-8 relative.

    Examples
    --------
    >>> import jax.numpy as jnp
    >>> import spexial as sp

    >>> round(float(sp.K2(1.0)), 8)
    1.6248389

    >>> [round(float(k), 8) for k in sp.K2(jnp.asarray([0.5, 5.0, 20.0]))]
    [7.55018355, 0.00530894, 0.0]

    """
    z_arr = jnp.asarray(z) * 1.0
    return K0(z_arr) + 2.0 / z_arr * K1(z_arr)
