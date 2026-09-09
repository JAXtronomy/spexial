"""Modified Bessel functions of the second kind, integer order."""

__all__ = ["K0", "K1", "K2", "K0e", "K1e", "K2e"]

from typing import Any, Final

import jax
import jax.numpy as jnp
from jax.scipy.special import gammaln, i0, i0e, i1e

from .custom_types import AnyArray, RealArrayLike

_EULER_GAMMA: Final = 0.57721566490153286061
"""The Euler-Mascheroni constant."""

_SMALL_Z: Final = 9.0
"""Cross-over between the ascending series and the asymptotic expansion."""

_N_SMALL: Final = 30
"""Terms in the ascending series; enough for ~1e-8 relative accuracy at z < 9."""

_N_LARGE: Final = 10
"""Terms in the asymptotic series; enough for ~1e-8 relative accuracy at z > 9."""


def _K0_small(z: AnyArray) -> AnyArray:
    """Ascending series for `K0`; see Zhang & Jin, *Special Functions* (1996)."""
    k = jnp.arange(1.0, _N_SMALL + 1.0)
    harmonic = jnp.cumsum(1.0 / k)
    # `z[..., None]` sums over a *trailing* axis: without it `jnp.sum` collapses
    # the caller's own axis and an array argument silently yields one scalar.
    log_term = 2.0 * k * jnp.log(z[..., None] / 2.0) - 2.0 * gammaln(k + 1.0)
    return -(jnp.log(z / 2.0) + _EULER_GAMMA) * i0(z) + jnp.sum(
        harmonic * jnp.exp(log_term), axis=-1
    )


def _K0e_large(z: AnyArray) -> AnyArray:
    """Asymptotic expansion for :math:`e^z K_0(z)`, via ``1 / (2 z I0e(z))``.

    Written against `jax.scipy.special.i0e` -- the exponentially scaled
    :math:`I_0` -- rather than `i0`, which overflows just above z = 709.78 and
    used to cap these functions there. Nothing in the scaled form overflows or
    underflows, at any z.
    """
    # `i0e(inf) == 0`, so the quotient would be `inf * 0 == nan`; every K_n
    # tends to 0 at +inf and every scipy counterpart returns that.
    at_inf = z == jnp.inf
    z = jnp.where(at_inf, 1.0, z)
    k = jnp.arange(1.0, _N_LARGE + 1.0)
    prod = jnp.cumprod(-(2.0 * k - 1.0) / (2.0 * k) * (2.0 * k - 1.0) ** 2.0)
    series = 1.0 + jnp.sum(
        (-1.0) ** k * prod / (2.0 * z[..., None]) ** (2.0 * k), axis=-1
    )
    return jnp.where(at_inf, 0.0, series / (2.0 * z * i0e(z)))


def _split(z: RealArrayLike) -> tuple[AnyArray, AnyArray, AnyArray, AnyArray]:
    """Both branch arguments, each already made safe for the other's domain."""
    z_arr = jnp.asarray(z) * 1.0
    small = z_arr < _SMALL_Z
    return z_arr, small, jnp.where(small, z_arr, 1.0), jnp.where(small, _SMALL_Z, z_arr)


@jax.custom_jvp
def K0e(z: RealArrayLike, /) -> AnyArray:
    r"""Compute the exponentially scaled :math:`e^z K_0(z)`.

    Equivalent to ``scipy.special.k0e(z)``, which has no JAX counterpart. This
    is the form to reach for beyond ``z = 705``, where :math:`K_0(z)` itself is
    smaller than any normal double and unrepresentable; :math:`e^z K_0(z)`
    decays only as :math:`1/\sqrt{z}` and stays accurate at any ``z``.

    Parameters
    ----------
    z
        Real positive argument, of any shape. Evaluated elementwise. ``z == 0``
        is the pole and gives ``inf``; ``z < 0`` is outside the domain and gives
        `nan`; ``z = inf`` gives 0, the limit.

    Returns
    -------
    Array
        Value(s) of :math:`e^z K_0(z)`, accurate to ~1.2e-7 relative
        (worst just below the ``z = 9`` cross-over; ~8e-9 out to z = 15,
        ~2e-13 to z = 30, and ~1e-15 beyond).

    Examples
    --------
    >>> import spexial as sp

    >>> round(float(sp.K0e(1.0)), 8)
    1.14446308

    Where `K0` has underflowed to zero, the scaled form is still exact:

    >>> float(sp.K0(800.0))
    0.0
    >>> round(float(sp.K0e(800.0)), 10)
    0.0443044275

    """
    _, small, z_small, z_large = _split(z)
    return jnp.where(small, _K0_small(z_small) * jnp.exp(z_small), _K0e_large(z_large))


@jax.custom_jvp
def K1e(z: RealArrayLike, /) -> AnyArray:
    r"""Compute the exponentially scaled :math:`e^z K_1(z)`.

    Equivalent to ``scipy.special.k1e(z)``, which has no JAX counterpart.
    Obtained from `K0e` through the Wronskian
    :math:`I_0(z) K_1(z) + I_1(z) K_0(z) = 1/z`, in the scaled variables.

    Parameters
    ----------
    z
        Real positive argument, of any shape. Evaluated elementwise. ``z == 0``
        is the pole and gives ``inf``; ``z < 0`` is outside the domain and gives
        `nan`; ``z = inf`` gives 0, the limit.

    Returns
    -------
    Array
        Value(s) of :math:`e^z K_1(z)`, accurate to ~1.0e-7 relative
        (worst just below the ``z = 9`` cross-over; ~8e-9 out to z = 15,
        ~2e-13 to z = 30, and ~1e-15 beyond).

    Examples
    --------
    >>> import spexial as sp

    >>> round(float(sp.K1e(1.0)), 8)
    1.63615349
    >>> round(float(sp.K1e(800.0)), 10)
    0.0443321091

    """
    z_arr = jnp.asarray(z) * 1.0
    # K1 diverges at 0, but the closed form evaluates to
    # `1/0 - i1e(0) * K0e(0) == inf - 0 * inf == nan` there. At +inf it is
    # `(0 - 0 * 0) / 0 == nan` for the same reason `K0e` needs a guard.
    # Substitute both limits.
    at_zero, at_inf = z_arr == 0.0, z_arr == jnp.inf
    z_safe = jnp.where(at_zero | at_inf, 1.0, z_arr)
    k1e = (1.0 / z_safe - i1e(z_safe) * K0e(z_safe)) / i0e(z_safe)
    return jnp.where(at_zero, jnp.inf, jnp.where(at_inf, 0.0, k1e))


@jax.custom_jvp
def K2e(z: RealArrayLike, /) -> AnyArray:
    r"""Compute the exponentially scaled :math:`e^z K_2(z)`.

    Equivalent to ``scipy.special.kve(2, z)``, which has no JAX counterpart.
    Obtained from the recurrence :math:`K_2(z) = K_0(z) + (2/z) K_1(z)`, which
    the scaling leaves unchanged.

    Parameters
    ----------
    z
        Real positive argument, of any shape. Evaluated elementwise. ``z == 0``
        is the pole and gives ``inf``; ``z < 0`` is outside the domain and gives
        `nan`; ``z = inf`` gives 0, the limit.

    Returns
    -------
    Array
        Value(s) of :math:`e^z K_2(z)`, accurate to ~7.4e-8 relative
        (worst just below the ``z = 9`` cross-over; ~8e-9 out to z = 15,
        ~2e-13 to z = 30, and ~1e-15 beyond).

    Examples
    --------
    >>> import spexial as sp

    >>> round(float(sp.K2e(1.0)), 8)
    4.41677005
    >>> round(float(sp.K2e(800.0)), 10)
    0.0444152578

    """
    z_arr = jnp.asarray(z) * 1.0
    return K0e(z_arr) + 2.0 / z_arr * K1e(z_arr)


@jax.custom_jvp
def K0(z: RealArrayLike, /) -> AnyArray:
    """Compute the modified Bessel function of the second kind of order 0.

    Equivalent to ``scipy.special.kn(0, z)``. See Zhang and Jin,
    ``SPECIAL_FUNCTIONS`` in FORTRAN77, for the algorithm: an ascending series
    below ``z = 9`` and an asymptotic expansion above it.

    Parameters
    ----------
    z
        Real positive argument, of any shape. Evaluated elementwise. ``z == 0``
        is the pole and gives ``inf``; ``z < 0`` is outside the domain and gives
        `nan`; ``z = inf`` gives 0, the limit.

    Returns
    -------
    Array
        Value(s) of :math:`K_0(z)`, accurate to ~1.2e-7 relative
        (worst just below the ``z = 9`` cross-over; ~8e-9 out to z = 15,
        ~2e-13 to z = 30, and ~1e-15 beyond). Underflows to 0
        above ``z = 705.5``, where the true value is subnormal; use `K0e` there.

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
    _, small, z_small, z_large = _split(z)
    return jnp.where(small, _K0_small(z_small), _K0e_large(z_large) * jnp.exp(-z_large))


@jax.custom_jvp
def K1(z: RealArrayLike, /) -> AnyArray:
    """Compute the modified Bessel function of the second kind of order 1.

    Obtained from `K0` through the Wronskian
    :math:`I_0(z) K_1(z) + I_1(z) K_0(z) = 1/z`.

    Parameters
    ----------
    z
        Real positive argument, of any shape. Evaluated elementwise. ``z == 0``
        is the pole and gives ``inf``; ``z < 0`` is outside the domain and gives
        `nan`; ``z = inf`` gives 0, the limit.

    Returns
    -------
    Array
        Value(s) of :math:`K_1(z)`, accurate to ~1.0e-7 relative
        (worst just below the ``z = 9`` cross-over; ~8e-9 out to z = 15,
        ~2e-13 to z = 30, and ~1e-15 beyond). Underflows to 0
        above ``z = 705.5``, where the true value is subnormal; use `K1e` there.

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
    return K1e(z_arr) * jnp.exp(-z_arr)


@jax.custom_jvp
def K2(z: RealArrayLike, /) -> AnyArray:
    """Compute the modified Bessel function of the second kind of order 2.

    Obtained from the recurrence :math:`K_2(z) = K_0(z) + (2/z) K_1(z)`.

    Parameters
    ----------
    z
        Real positive argument, of any shape. Evaluated elementwise. ``z == 0``
        is the pole and gives ``inf``; ``z < 0`` is outside the domain and gives
        `nan`; ``z = inf`` gives 0, the limit.

    Returns
    -------
    Array
        Value(s) of :math:`K_2(z)`, accurate to ~7.4e-8 relative
        (worst just below the ``z = 9`` cross-over; ~8e-9 out to z = 15,
        ~2e-13 to z = 30, and ~1e-15 beyond). Underflows to 0
        above ``z = 705.5``, where the true value is subnormal; use `K2e` there.

    Examples
    --------
    >>> import jax.numpy as jnp
    >>> import spexial as sp

    >>> round(float(sp.K2(1.0)), 8)
    1.6248389

    >>> [round(float(k), 8) for k in sp.K2(jnp.asarray([0.5, 5.0, 20.0]))]
    [7.55018355, 0.00530894, 0.0]

    """
    # The recurrence is applied in the *scaled* variables and undone once.
    # Evaluated directly, the `(2/z) K1` term drops into the subnormal range
    # around z = 699 -- where XLA on CPU flushes it to zero, silently losing a
    # 0.3% contribution (2850x the documented tolerance) while still returning a
    # plausible number. `K0e` and `K1e` are order 1e-2 there, so the sum is
    # formed entirely in normal arithmetic and only the result is scaled down.
    z_arr = jnp.asarray(z) * 1.0
    return K2e(z_arr) * jnp.exp(-z_arr)


# Analytic derivatives. Letting JAX differentiate through the 30-term ascending
# series and the 10-term asymptotic expansion works, but costs about twice as
# much as evaluating the closed form -- measured 637us -> 310us for `grad` over
# 1000 points. Each identity below was checked against the autodiff result to
# ~1e-8, well inside these functions' own ~1e-6 accuracy.
#
# Standard recurrence Kv'(z) = -K_{v-1}(z) - (v/z) K_v(z), which at v = 0, 1, 2
# gives K0' = -K1, K1' = -K0 - K1/z and K2' = -K1 - (2/z) K2. The scaled forms
# pick up the extra `+ Kn e` term from differentiating the `e^z` factor.


@K0.defjvp
def _K0_jvp(primals: tuple[Any], tangents: tuple[Any]) -> tuple[AnyArray, AnyArray]:
    """K0'(z) = -K1(z)."""
    (z,), (dz,) = primals, tangents
    return K0(z), -K1(z) * dz


@K1.defjvp
def _K1_jvp(primals: tuple[Any], tangents: tuple[Any]) -> tuple[AnyArray, AnyArray]:
    """K1'(z) = -K0(z) - K1(z) / z, summed scaled for the same reason as `K2`.

    Formed directly, the `K1(z) / z` term is subnormal from z = 700 and XLA
    flushes it, dropping 0.2% of the derivative.
    """
    (z,), (dz,) = primals, tangents
    z_arr = jnp.asarray(z) * 1.0
    deriv = -(K0e(z_arr) + K1e(z_arr) / z_arr) * jnp.exp(-z_arr)
    return K1(z_arr), deriv * dz


@K2.defjvp
def _K2_jvp(primals: tuple[Any], tangents: tuple[Any]) -> tuple[AnyArray, AnyArray]:
    """K2'(z) = -K1(z) - (2/z) K2(z), summed scaled as in `K2` itself."""
    (z,), (dz,) = primals, tangents
    z_arr = jnp.asarray(z) * 1.0
    deriv = -(K1e(z_arr) + 2.0 / z_arr * K2e(z_arr)) * jnp.exp(-z_arr)
    return K2(z_arr), deriv * dz


@K0e.defjvp
def _K0e_jvp(primals: tuple[Any], tangents: tuple[Any]) -> tuple[AnyArray, AnyArray]:
    """(e^z K0)' = e^z (K0 - K1)."""
    (z,), (dz,) = primals, tangents
    return K0e(z), (K0e(z) - K1e(z)) * dz


@K1e.defjvp
def _K1e_jvp(primals: tuple[Any], tangents: tuple[Any]) -> tuple[AnyArray, AnyArray]:
    """(e^z K1)' = e^z K1 - e^z K0 - e^z K1 / z."""
    (z,), (dz,) = primals, tangents
    z_arr = jnp.asarray(z) * 1.0
    return K1e(z_arr), (K1e(z_arr) - K0e(z_arr) - K1e(z_arr) / z_arr) * dz


@K2e.defjvp
def _K2e_jvp(primals: tuple[Any], tangents: tuple[Any]) -> tuple[AnyArray, AnyArray]:
    """(e^z K2)' = e^z K2 - e^z K1 - (2/z) e^z K2."""
    (z,), (dz,) = primals, tangents
    z_arr = jnp.asarray(z) * 1.0
    return K2e(z_arr), (K2e(z_arr) - K1e(z_arr) - 2.0 / z_arr * K2e(z_arr)) * dz
