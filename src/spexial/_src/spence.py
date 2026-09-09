"""Spence's function (the dilogarithm), for complex argument.

A translation of scipy's Cython implementation into JAX, contributed by Colm
Talbot. Adapted from
https://github.com/scipy/scipy/blob/8971d5e9b72931987b7d3c5a25da1a8e7e5485d0/scipy/special/_spence.pxd

`jax.scipy.special.spence` exists at every JAX version `spexial` supports, but
is real-only -- it raises on complex input. That, plus the analytic derivative
below, is why this module is here; see `spexial.registry`.
"""

__all__ = ["spence"]

from typing import Any, Final

import jax
import jax.numpy as jnp
import numpy as np

from .custom_types import AnyArray, AnyArrayLike

_MAXITER: Final = 500
"""Terms taken in each series branch."""


@jnp.vectorize
def _series_about_zero(z: AnyArray) -> AnyArray:
    r"""Evaluate the series about :math:`z = 0`, for :math:`|z| < 1/2`.

    The third term diverges for :math:`|z| \rightarrow 0` so we special case
    it.
    """
    # Real dtype matching `z`: a bare `arange(1.0, n)` is float64 whenever x64
    # is on, which widened a float32 (or complex64) argument to float64.
    nn = jnp.arange(1.0, _MAXITER, dtype=_real_dtype(z))
    temp = z**nn / nn
    sum1 = jnp.sum(temp / nn)
    sum2 = jnp.sum(temp)

    return jnp.where(z == 0, np.pi**2 / 6, np.pi**2 / 6 - sum1 + jnp.log(z) * sum2)


def _real_dtype(z: AnyArray) -> Any:
    """Give the real floating dtype to build series constants in.

    `jnp.finfo(z.dtype)` is not enough: it raises on integer input, and `z` may
    be complex, in which case the constants want the *real* component's width.
    Multiplying by 1.0 promotes an integer to the default float and leaves
    everything else alone.
    """
    return jnp.asarray(jnp.real(z) * 1.0).dtype


@jnp.vectorize
def _series_about_one(z: AnyArray) -> AnyArray:
    """Evaluate the expansion about :math:`z = 1`.

    Used for :math:`|z| > 1/2` where the reflected form does not apply.
    """
    z = 1 - z
    # Float, not `arange(1, ...)`: the denominator is a degree-6 integer
    # polynomial that overflows int32 from n = 35, and JAX is int32 unless x64
    # is on. Under float32 that silently corrupted 235 of the 499 terms (many
    # going negative) for a 3.8e-5 error at z = 2 -- 300x worse than float32
    # rounding alone. Compare `polylog.py`, which guards the same hazard.
    nn = jnp.arange(1.0, _MAXITER, dtype=_real_dtype(z))
    res = jnp.sum(z**nn / (nn * (nn + 1) * (nn + 2)) ** 2)

    res *= 4 * z**2
    res += 4 * z + 5.75 * z**2 + 3 * (1 - z**2) * jnp.log1p(-z)
    res /= 1 + 4 * z + z**2
    return res


def _series_reflected(z: AnyArray) -> AnyArray:
    """Evaluate the reflected form, for :math:`|z| > 1/2` and :math:`|1 - z| > 1`.

    Divides by :math:`z - 1`, so it is `nan` at ``z = 1``. That is harmless:
    ``z = 1`` is never selected into this branch, `jax.lax.select` takes the
    chosen operand elementwise, and the gradient comes from a `jax.custom_jvp`
    that does not go through here at all.
    """
    return -_series_about_one(z / (z - 1)) - np.pi**2 / 6 - jnp.log(z - 1) ** 2 / 2


@jax.jit
def spence(z: AnyArrayLike, /) -> AnyArray:
    r"""Compute Spence's function -- the dilogarithm -- for real or complex input.

    `jax.scipy.special.spence` covers the real case at every JAX version
    `spexial` supports, and raises on complex input. This accepts both, and
    carries an analytic derivative: differentiating the series instead costs
    444x the time and 1534x the residual memory (23.9 MB against 16 kB over
    2000 points).

    .. math::

        \int_{1}^{z} dt \frac{\log(t)}{1 - t}

    See Also
    --------
    scipy.special.spence: the reference implementation in scipy
    jax.scipy.special.spence: jax implementation for real inputs

    """
    return jax.lax.select(
        abs(z) < 0.5,
        _series_about_zero(z),
        jax.lax.select(
            abs(1 - z) > 1,
            _series_reflected(z),
            _series_about_one(z),
        ),
    )


def _spence_gradient(z: AnyArrayLike) -> AnyArray:
    r"""Analytic derivative of Spence's function.

    `spence` is defined by an integral, so the derivative is its integrand:

    .. math::

        \frac{\log(z)}{1 - z}

    The derivative is `-1` at ``z = 1`` (a removable singularity) and `-inf` at
    ``z = 0``, both of which are the true limits.
    """
    # `z = 1` is a *removable* singularity, not a zero: log(z)/(1-z) is 0/0
    # there and the limit is -1, since spence(1 + h) = -h + O(h^2). Returning 0
    # would be a plausible-looking wrong answer at exactly one point, and the
    # one place a user is most likely to evaluate. `z = 0` needs no special
    # case: log(0)/(1-0) is -inf, which is the true derivative.
    at_one = z == 1
    z_safe = jnp.where(at_one, 2.0, z)
    return jnp.where(at_one, -1.0, jnp.log(z_safe) / (1 - z_safe))


def _spence_jvp(primals: tuple[Any], tangents: tuple[Any]) -> tuple[AnyArray, AnyArray]:
    """JVP rule built from the analytic derivative."""
    (z,) = primals
    (z_dot,) = tangents
    dspence = _spence_gradient(z)
    return spence(z), z_dot * dspence


spence = jax.custom_jvp(spence)
spence.defjvp(_spence_jvp)
