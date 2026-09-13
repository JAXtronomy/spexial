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
from jax.scipy.special import spence as _jax_spence

from .custom_types import AnyArray, AnyArrayLike
from .dtype import as_float, cast_like, is_negative, log_no_flush

_MAXITER: Final = 500
"""Terms taken in each series branch."""

_PLAIN_TERMS: Final = 60
"""Terms in the plain `sum z**n / n**2` fallback near the removable root."""

_GRADIENT_RADIUS: Final = 0.1
"""How close to `z = 1` the derivative switches to its series form."""

_GRADIENT_TERMS: Final = 20
"""Terms in that series. `0.1**20` is 1e-20, so the truncation is invisible.

The radius is 0.1 rather than 0.5 because both branches of the `where` are
evaluated on every call, and a 60-term Horner chain is a long serial dependency:
at 0.5 it cost 2.7x on `grad(spence)`. Just outside 0.1 the closed form is still
good to ~1e-15, so nothing is given up by switching later."""


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

    There is no upper limit on a real argument. Above ``z = 1/tiny`` the value
    comes from the inversion formula rather than from upstream, whose opening
    ``1/z`` is subnormal there and is flushed into the wrong branch; see the
    comment on that branch. ``spence(inf)`` is `nan`, as in SciPy.

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
    # Real input is handed straight to JAX, which is the same choice `gamma`
    # makes and for the same reason: a delegated value cannot drift from
    # upstream, and upstream is more accurate here (2.0e-15 against 2.4e-14 for
    # the series below, measured over [1e-6, 50] against mpmath at 40 digits).
    # It also sidesteps the removable 0/0 the series inherited from SciPy's
    # Cython at Spence argument 3 +- sqrt(3).
    #
    # The series is kept for complex input, which `jax.scipy.special.spence`
    # rejects outright, and `spexial` supplies the derivative in both cases --
    # JAX's own gradient is `nan` across roughly 1 < z < 2.
    if not jnp.issubdtype(z.dtype, jnp.complexfloating):
        # `jax.scipy.special.spence` supports float32 and float64 only, and
        # raised on the two narrow types rather than returning anything. They
        # are computed one width up and rounded back, which is what `kn` and
        # `comb` do -- `keep_weak` so that an already-wide argument is handed
        # over untouched and keeps its weak typing.
        promoted = as_float(z, keep_weak=True)
        # A negative subnormal reaches upstream's `z == 0` branch, because XLA
        # compares one as if it were zero, and comes back as `pi**2/6` -- a
        # perfectly ordinary number for an argument that is out of domain.
        # SciPy returns `nan`; the sign has to be read off the bits, since
        # `z < 0` is False for exactly these values.
        out = _jax_spence(promoted)
        # Upstream's first move for `x > 2` is `x -> 1/x`, and XLA flushes that
        # reciprocal as soon as it is subnormal. From `z = 1/tiny` up it is
        # exactly zero, the `x < 0.5` branch is taken instead of the reflected
        # one, and `log(0) * 0` makes the answer `nan` -- across the top two
        # binades of every width, where SciPy is finite and this module's own
        # complex path returns the right number. Reported as spexial#28.
        #
        # There the asymptotic form is not an approximation. With `w = 1 - z`,
        # `spence(z) = Li_2(w)` and the inversion `Li_2(w) = -pi**2/6 -
        # log(-w)**2/2 - Li_2(1/w)` leaves a correction of order `1/z`, which
        # below `tiny` is 1e-313 of the value -- 300 orders under an eps. It
        # agrees with SciPy to the last bit at `1/tiny`, `1e308` and `DBL_MAX`.
        #
        # `isfinite`, so that `z = inf` keeps the `nan` SciPy gives it rather
        # than the `-inf` the limit would suggest.
        huge = jnp.isfinite(promoted) & (
            promoted >= 1.0 / jnp.finfo(promoted.dtype).tiny
        )
        out = jnp.where(huge, -(np.pi**2) / 6 - jnp.log(promoted) ** 2 / 2, out)
        return cast_like(jnp.where(is_negative(promoted), jnp.nan, out), z)
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
    # Near z = 1 this is evaluated as the power series it equals, not as the
    # quotient with a constant patched in at the singular point. `log(z)/(1-z)`
    # is *itself* analytic there: with u = z - 1,
    #
    #     log(1 + u) / (-u) = -1 + u/2 - u^2/3 + ... = sum_j (-1)^(j+1) u^j/(j+1)
    #
    # so a `where(z == 1, -1.0, ...)` gets the value right and every derivative
    # wrong -- a constant differentiates to zero. That cost `spence''(1)` (0
    # against 1/2) and then `spence'''(1)` (0 against -2/3), each found one
    # round after the last. The series has no such floor: differentiating a
    # polynomial is correct at every order, so this closes the whole chain
    # rather than one more rung of it.
    near_one = jnp.abs(z - 1) < _GRADIENT_RADIUS
    u = jnp.where(near_one, z - 1, 0.0)
    j = jnp.arange(_GRADIENT_TERMS, dtype=_real_dtype(jnp.asarray(z)))
    series = jnp.polyval(((-1.0) ** (j + 1) / (j + 1))[::-1], u)
    z_safe = jnp.where(near_one, 2.0, z)
    # `log_no_flush`, because XLA flushes a subnormal argument and `jnp.log`
    # then reports `-inf` for the whole band below `tiny`, where the true
    # derivative is an ordinary number -- -713.8 at `z = 1e-310`. It now covers
    # complex input too, so there is no dtype test here any more: routing
    # complex to a plain `jnp.log` to keep the bitcast away from it left the
    # complex derivative `nan` across that same band, which in complex64 starts
    # at 1.18e-38.
    #
    # No pole guard. `log_no_flush(0)` is already `-inf`, so the quotient gives
    # the pole its own answer; a subnormal gets its true logarithm; and a
    # negative argument gets `nan`. A `where` on top of that bought nothing and
    # did not survive XLA's fusion -- `exactly_zero` is correct in isolation and
    # collapses when a select is its only consumer, which is how
    # `jit(grad(spence))` came to be `-inf` across the band while eager was
    # right.
    logarithm = log_no_flush(z_safe)
    # Below `tiny` the denominator is exactly 1, so dividing by it is a no-op --
    # except in complex arithmetic, where `(-inf + 0j) / (1 + 0j)` forms
    # `-inf * 0` in the imaginary part and hands back `nan`. Skipping the
    # division changes no value and removes that, at the pole and across the
    # band alike. The comparison is the flushed one, which is exactly the set
    # where `1 - z == 1`.
    negligible = jnp.abs(z_safe) < jnp.finfo(_real_dtype(jnp.asarray(z))).tiny
    closed = jnp.where(negligible, logarithm, logarithm / (1 - z_safe))
    # `z = 0` is a pole and the limit is `-inf`, which the real path gets for
    # free from `log(0) / 1`. The *complex* path does not: `(-inf + 0j)` divided
    # by `(1 + 0j)` leaves `0 - (-inf * 0)` in the imaginary part, i.e. `nan`,
    # so the branch this module exists to provide disagreed with the real one at
    # the one point both can reach.
    # `exactly_zero`, not `z == 0`: XLA compares every subnormal equal to zero,
    # so the pole guard fired across the whole subnormal band and returned
    # `-inf` for arguments that have a perfectly finite derivative. That is the
    # defect `dtype.exactly_zero` exists for, and it was reintroduced here by
    # the fix for the *complex* pole one round earlier.
    return jnp.where(near_one, series, closed)


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
    # Same treatment as the gradient itself, and for the same reason: term-by-
    # term differentiation of that series, `sum_m (m+1)(-1)^m u^m/(m+2)`, which
    # is again a polynomial and therefore right to every further order.
    near_one = jnp.abs(z - 1) < _GRADIENT_RADIUS
    u = jnp.where(near_one, z - 1, 0.0)
    m = jnp.arange(_GRADIENT_TERMS - 1, dtype=_real_dtype(jnp.asarray(z)))
    series = jnp.polyval(((m + 1) * (-1.0) ** m / (m + 2))[::-1], u)
    # `z == 0` is the *flushed* comparison and so covers the whole subnormal
    # band, not just the pole. That is a ceiling rather than the right answer --
    # `spence''` is finite over most of the band (4.5e307 at `z = 2.2e-308`) --
    # but separating the two needs a bit test, and a bit test does not survive
    # XLA's fusion here: `exactly_zero` is correct in isolation and collapses
    # when a select is its only consumer. A magnitude test flushes for exactly
    # the same reason. Both alternatives were tried; the second lost `z = 0`
    # its `inf` as well, which is worse than the ceiling. Documented instead.
    #
    # `log_no_flush` handles complex input itself now, so there is no dtype
    # test here any more. It matters at second order for the same reason it did
    # at first: `at_zero` compares a *complex* `z` componentwise, so a real part
    # of exactly `tiny` beside a subnormal imaginary part is not "at zero" and
    # reached the logarithm with one component already flushed.
    at_zero = z == 0
    z_safe = jnp.where(near_one | at_zero, 2.0, z)
    logarithm = log_no_flush(z_safe)
    # **Neither** quotient is fused, and both halves matter. Differentiating
    # `a / b` forms `a * db / b**2`, so a squared denominator here becomes a
    # *fourth* power in the third derivative: `1 / (z*(1-z))` squared `z(1-z)`
    # and made the third derivative `nan` below `z = 1.5e-154`, and
    # `log(z) / (1-z)**2` squares `(1-z)**2` into `(1-z)**4`, which overflows
    # from `z = 8.2e76` -- dropping the dominant `2 log z / (1-z)**3` term and
    # leaving the third derivative with the **wrong sign**, `+3/z**3`, in every
    # mode at once. In float32 that starts at `z = 3.1e9`. Splitting the second
    # term was the round-14 fix; splitting only one of the two left the same
    # defect standing at the other end of the range, which is why they are now
    # written the same way.
    closed = (1.0 / z_safe) / (1 - z_safe) + (logarithm / (1 - z_safe)) / (1 - z_safe)
    # `z = 0` is a genuine pole, not a removable point: every order diverges,
    # so a constant is the only answer available and orders past this one are
    # 0 there. `z = 1` needs no such guard any more.
    second = jnp.where(near_one, series, jnp.where(at_zero, jnp.inf, closed))
    return _spence_gradient(z), second * dz


def _spence_jvp(primals: tuple[Any], tangents: tuple[Any]) -> tuple[AnyArray, AnyArray]:
    """JVP rule built from the analytic derivative."""
    (z,) = primals
    (z_dot,) = tangents
    dspence = _spence_gradient(z)
    return spence(z), z_dot * dspence


spence = jax.custom_jvp(spence)
spence.defjvp(_spence_jvp)
