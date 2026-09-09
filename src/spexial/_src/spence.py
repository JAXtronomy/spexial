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

_PLAIN_TERMS: Final = 60
"""Terms in the plain `sum z**n / n**2` fallback near the removable root."""


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


# `np.pi` is a Python `float`, not a `np.float64` scalar, so `np.pi**2 / 6` is
# weakly typed and follows the argument's dtype rather than forcing float64.
# Keep it that way: a `np.float64(...)` constant here would silently promote a
# float32 or complex64 argument, undoing `_real_dtype`. Locked by
# `test_dtype_is_preserved`.


def _real_dtype(z: AnyArray) -> Any:
    """Give the real floating dtype to build series constants in.

    `jnp.finfo(z.dtype)` is not enough, because `z` may be complex and the
    constants want the *real* component's width. Integer input never reaches
    here -- `spence` promotes it once at entry, precisely so that every branch
    agrees on a dtype -- so this only has to strip the imaginary part.
    """
    return jnp.asarray(jnp.real(z)).dtype


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

    # The accelerated form divides by `1 + 4z + z**2`, which vanishes at
    # `z = -2 +- sqrt(3)`. Only the first root is reachable -- both callers pass
    # |z| < 1 -- and it sits at z = -0.2679, i.e. Spence argument 3 - sqrt(3),
    # and again at 3 + sqrt(3) through the reflected branch. The numerator
    # vanishes there too, so the quotient is 0/0 and the relative error grows as
    # ~1e-16 / |denominator|: at the root itself `spence(3 - sqrt(3))` was
    # **59% wrong**, and still 1e-8 wrong a whole 1e-8 away. `scipy.special`'s
    # *complex* spence -- which this was translated from -- has the same defect,
    # so a complex parity test against SciPy agrees on the wrong answer; its
    # real path is Cephes and is unaffected, which is why real parity tests only
    # caught it as a tolerance overshoot.
    #
    # Near the root, fall back to the defining series `Li_2(z) = sum z^n / n^2`.
    # It is exact there for a different reason than the accelerated form is fast
    # elsewhere: |z| <= 0.42 wherever the denominator is small, so 499 terms
    # converge far below machine precision.
    denom = 1 + 4 * z + z**2
    near_root = jnp.abs(denom) < 0.5
    # 60 terms, not the full `_MAXITER`: the fallback only runs where the
    # denominator is below 0.5, which bounds |z| by 0.42, and 0.42**60 is 1e-23.
    # Both branches are evaluated under the `where`, so the shorter sum is the
    # difference between costing 11% on the forward path and costing ~1%.
    plain = jnp.sum(z ** nn[:_PLAIN_TERMS] / nn[:_PLAIN_TERMS] ** 2)
    return jnp.where(near_root, plain, res / jnp.where(near_root, 1.0, denom))


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
    444x the time and 1652x the residual memory (26.4 MB against 16 kB over
    2000 points; the `near_root` fallback added ~2 MB to the differentiated
    series, which is exactly the cost a custom rule avoids).

    .. math::

        \int_{1}^{z} dt \frac{\log(t)}{1 - t}

    See Also
    --------
    scipy.special.spence: the reference implementation in scipy
    jax.scipy.special.spence: jax implementation for real inputs

    """
    # Promoted once, here, rather than inside each branch: `_series_reflected`
    # forms `z / (z - 1)`, and JAX's integer division yields float32 for int32
    # input while the other two branches build float64 constants -- so
    # `lax.select` got two dtypes and raised. Integer widths all land on the
    # default float; float and complex arguments are untouched, which is what
    # keeps `_cast_like`-style dtype preservation intact.
    z = jnp.asarray(z)
    if not jnp.issubdtype(z.dtype, jnp.inexact):
        z = z.astype(jnp.asarray(0.0).dtype)
    return jax.lax.select(
        abs(z) < 0.5,
        _series_about_zero(z),
        jax.lax.select(
            abs(1 - z) > 1,
            _series_reflected(z),
            _series_about_one(z),
        ),
    )


@jax.custom_jvp
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


@_spence_gradient.defjvp
def _spence_gradient_jvp(
    primals: tuple[Any], tangents: tuple[Any]
) -> tuple[AnyArray, AnyArray]:
    r"""Second derivative: :math:`1/(z(1-z)) + \log z/(1-z)^2`.

    A rule rather than plain arithmetic, because the `jnp.where` above returns a
    *constant* at ``z = 1`` -- correct for the value, but its derivative is then
    `0`, where the true ``spence''(1)`` is `1/2`. Substituting `z = 1 + h` in the
    closed form and expanding gives `1/2 - 2h/3 + O(h^2)`, so the limit is `1/2`
    and the neighbourhood was already right: only the single guarded point was
    wrong, which is what made it a plausible number rather than an obvious one.
    Exactly the defect the *first* derivative had at this same point, one order
    up.

    At ``z = 0`` both terms diverge: the closed form is `inf - inf` at `+0.0`
    and commits to `-inf` at `-0.0`, while the true limit is `+inf` from either
    side, since `1/z` outruns `log z`.
    """
    (z,), (dz,) = primals, tangents
    at_one = z == 1
    at_zero = z == 0
    z_safe = jnp.where(at_one | at_zero, 2.0, z)
    second = 1.0 / (z_safe * (1 - z_safe)) + jnp.log(z_safe) / (1 - z_safe) ** 2
    second = jnp.where(at_one, 0.5, jnp.where(at_zero, jnp.inf, second))
    return _spence_gradient(z), second * dz


def _spence_jvp(primals: tuple[Any], tangents: tuple[Any]) -> tuple[AnyArray, AnyArray]:
    """JVP rule built from the analytic derivative."""
    (z,) = primals
    (z_dot,) = tangents
    dspence = _spence_gradient(z)
    return spence(z), z_dot * dspence


spence = jax.custom_jvp(spence)
spence.defjvp(_spence_jvp)
