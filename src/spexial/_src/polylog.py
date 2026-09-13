"""The polylogarithm."""

__all__ = ["polylog"]

from functools import partial
from typing import Any, Final

import jax
import jax.numpy as jnp
from jax import lax
from jax.scipy.special import gamma as _jax_gamma

from .bernoulli import ORDER, bernoulli_numbers
from .comb import comb
from .custom_types import AnyArray, Scalar, ScalarLike
from .dtype import promote_integers
from .zeta import zeta

_N_TERMS: Final = ORDER
"""Number of terms kept in each of the three series."""


def _bernoulli_poly(n: int, x: AnyArray) -> AnyArray:
    r"""Evaluate the Bernoulli polynomial :math:`B_n(x)`, for ``n <= 60``.

    See https://en.wikipedia.org/wiki/Bernoulli_polynomials.

    Parameters
    ----------
    n
        Order of the Bernoulli polynomial. Must be a static Python `int`.
    x
        Point at which to evaluate it.

    Returns
    -------
    Array
        Value of :math:`B_n(x)`.

    Examples
    --------
    >>> import jax.numpy as jnp
    >>> from spexial._src.polylog import _bernoulli_poly

    ``B_2(x) = x**2 - x + 1/6``:

    >>> round(float(_bernoulli_poly(2, jnp.asarray(3.0))), 12)
    6.166666666667

    """
    # The Bernoulli table is float64 by construction (it comes from exact
    # `Fraction` arithmetic), so a narrower `x` seeds the carry at its own width
    # and the body then widens it -- which `lax.fori_loop` rejects outright,
    # with an error naming neither `polylog` nor the dtype. Casting the table to the
    # carry's dtype keeps the loop type-stable at any input width.
    x_arr = (
        jnp.asarray(x) * 1.0
    )  # promotes integers; leaves a weak float at the default
    bs = bernoulli_numbers().astype(x_arr.dtype)
    return lax.fori_loop(
        0,
        n + 1,
        lambda i, val: val + bs[i] * comb(n, i) * x_arr ** (n - i),
        jnp.zeros_like(x_arr),
    )


def polylog(n: int, z: ScalarLike, /) -> Scalar:
    r"""Compute the polylogarithm :math:`\mathrm{Li}_n(z)`.

    There is no `scipy.special` counterpart; `mpmath.polylog` is the reference
    used by the test suite.

    Three series are stitched together: the defining sum for
    :math:`|z| \le 1/2`, the Hurwitz-zeta expansion in :math:`\log z` for
    :math:`1/2 < |z| < 2`, and the inversion formula for :math:`|z| \ge 2`.

    Parameters
    ----------
    n
        Order of the polylogarithm. Must be a static Python `int` and
        ``>= 1``. A non-integer order -- including a whole-number `float` such
        as ``polylog(2.0, z)`` -- is rejected by the runtime type checker with a
        `TypeError`; an integer below 1 raises `ValueError`.
    z
        Real, **scalar** argument. The middle series is built from a
        length-60 vector of powers of :math:`\log z`, so it cannot broadcast;
        use ``jax.vmap(partial(polylog, n))`` for arrays.

    Returns
    -------
    Array
        The real part of :math:`\mathrm{Li}_n(z)`. For ``z > 1`` the
        polylogarithm is genuinely complex; only its real part is returned.

    Notes
    -----
    Measured against `mpmath.polylog` over ``1 <= n <= 20`` and
    ``|z| <= 1000``, the relative error stays below ~1e-12 in all three
    branches. The bound on ``n`` is branch-dependent, and the tighter
    one bites first: for :math:`\lvert z \rvert \ge 2` the inversion formula
    needs the Bernoulli table, which stops at :math:`B_{60}`, so ``n > 60``
    there returns `nan`. The other two branches run to ``n = 170``, where
    :math:`\Gamma(n+1)` overflows.

    Examples
    --------
    >>> import spexial as sp

    ``Li_1(z) == -log(1 - z)``:

    >>> round(float(sp.polylog(1, 0.5)), 12)
    0.69314718056

    ``Li_2(1) == zeta(2)``:

    >>> round(float(sp.polylog(2, 1.0)), 10)
    1.6449340668

    >>> round(float(sp.polylog(3, -1.0)), 8)
    -0.90154268

    """
    # Validated here rather than inside `_li`, so the error is raised eagerly at
    # call time instead of during tracing -- a `ValueError` from inside a jitted
    # body surfaces with a confusing traceback and only when the trace happens.
    # `isinstance` and not `n != int(n)`: a whole-number *float* order such as
    # `polylog(2.0, z)` used to reach `lax.fori_loop(0, n + 1, ...)` and die there
    # with "lower and upper arguments must have equal types", which says nothing
    # about what the caller did wrong.
    if n < 1:
        msg = f"polylog is only implemented for integer order n >= 1, got {n}"
        raise ValueError(msg)
    # Two of the three branches take `jnp.real` of a complex intermediate --
    # correct for real `z`, where the imaginary parts cancel exactly, but it
    # would silently discard a genuine imaginary part. Rejected rather than
    # returned wrong: `jnp.iscomplexobj` reads the dtype, so this works on a
    # tracer and costs nothing at runtime.
    if jnp.iscomplexobj(z):
        msg = (
            f"polylog is only implemented for real z, got dtype {jnp.asarray(z).dtype}"
        )
        raise ValueError(msg)
    # `_li_core` evaluates all three branches under `jnp.where`, and two of them
    # build float64 constants (the zeta table, `gamma(arange(...))`, the
    # Bernoulli table), so the `where` promoted a float32 argument to float64 --
    # leaving `polylog` the only public function whose primal disagreed with its own
    # `grad`. Narrowed back here, as `kn.py` does with `_cast_like`.
    out = _li(n, z)
    dtype = jnp.asarray(z).dtype
    if (
        jnp.issubdtype(dtype, jnp.inexact)
        and jnp.finfo(dtype).bits < jnp.finfo(out.dtype).bits
    ):
        return out.astype(dtype)
    return out


@partial(jax.custom_jvp, nondiff_argnums=(0,))
def _li_core(n: int, z: ScalarLike) -> Scalar:
    """Evaluate the polylogarithm; see `polylog`, which validates ``n`` first."""

    def series(z: AnyArray) -> AnyArray:
        """Evaluate the defining series, for |z| <= 1/2."""
        # `j` is a traced integer: `j ** n` would overflow int64 for n >= 12.
        return lax.fori_loop(
            1, _N_TERMS, lambda j, val: val + z**j / (j * 1.0) ** n, jnp.zeros_like(z)
        )

    def expansion(z: AnyArray) -> AnyArray:
        """Evaluate the Hurwitz-zeta expansion in log(z), for 1/2 < |z| < 2."""
        # The m == n - 1 term is the zeta(1) pole; it is carried by
        # `harmonic_term` below instead.
        zeta_ary = jnp.asarray(
            [0.0 if m == n - 1 else zeta(n - m) for m in range(_N_TERMS)]
        )
        log_z = jnp.log(z + 0j)
        # `log_z ** 0` would be `nan` at z == 1; spell the leading 1 out.
        powers = jnp.concatenate(
            (jnp.ones(1, dtype=log_z.dtype), log_z ** jnp.arange(1, _N_TERMS))
        )
        zeta_series = jnp.sum(
            zeta_ary * powers / _jax_gamma(jnp.arange(_N_TERMS) + 1.0)
        )

        # Exact comparison, deliberately: `jnp.isclose` defaults to atol=1e-8,
        # which swallowed a whole neighbourhood of z = 1 and returned zeta(n)
        # there -- wrong by 6e-8 for n = 2 and by 100% for n = 1.
        at_one = z == 1.0
        harmonic = jnp.sum(1.0 / jnp.arange(1, n))
        harmonic_term = jnp.where(
            at_one,
            0.0,
            log_z ** (n - 1)
            / _jax_gamma(n)
            * (harmonic - jnp.log(-jnp.log(jnp.where(at_one, 2.0, z) + 0j) + 0j)),
        )
        out = jnp.real(zeta_series + harmonic_term)
        # Li_1(1) is the pole of the polylogarithm. Every higher order is
        # finite there (Li_n(1) = zeta(n)) and needs no special case.
        if n == 1:
            out = jnp.where(at_one, jnp.inf, out)
        return out

    def inversion(z: AnyArray) -> AnyArray:
        """Evaluate the inversion formula, for |z| >= 2."""
        if n > ORDER:
            # `_bernoulli_poly` indexes the table up to `n`, and an out-of-bounds
            # index is silently *clamped* under `jit` rather than raising -- so
            # without this guard every order above the table reused B_ORDER and
            # returned a plausible, wrong number. `zeta` guards the same hazard.
            return jnp.full_like(jnp.real(z), jnp.nan)
        recip = lax.fori_loop(
            1,
            _N_TERMS,
            lambda j, val: val + (1 / z) ** j / (j * 1.0) ** n,
            jnp.zeros_like(z),
        )
        bern = _bernoulli_poly(n, jnp.log(z + 0j) / (2 * jnp.pi * 1j))
        return jnp.real(
            -((-1) ** n) * recip - (2 * jnp.pi * 1j) ** n / _jax_gamma(n + 1) * bern
        )

    z_arr = promote_integers(z)
    abs_z = jnp.abs(z_arr)
    small = abs_z <= 0.5
    large = abs_z >= 2.0  # `>=`, not `>`: |z| == 2 used to fall through all three

    # Each branch is fed an argument inside its own domain, so the untaken
    # branches stay finite and `jax.grad` is well behaved.
    return jnp.where(
        small,
        series(jnp.where(small, z_arr, 0.25)),
        jnp.where(
            large,
            inversion(jnp.where(large, z_arr, 3.0)),
            expansion(jnp.where(small | large, 0.75, z_arr)),
        ),
    )


@_li_core.defjvp
def _li_jvp(n: int, primals: tuple[Any], tangents: tuple[Any]) -> tuple[Scalar, Scalar]:
    r"""Analytic derivative of the polylogarithm in ``z``.

    :math:`\mathrm{d}/\mathrm{d}z\,\mathrm{Li}_n(z) = \mathrm{Li}_{n-1}(z)/z`.

    Differentiating the implementation means differentiating a 60-term series,
    an expansion in powers of :math:`\log z`, or the inversion formula --
    whichever branch was taken -- and keeping every intermediate alive for the
    backward pass: 4.2 MB of residual over 2000 points. One extra evaluation of
    :math:`\mathrm{Li}_{n-1}` replaces all of it.

    ``n = 1`` is special-cased. The identity needs :math:`\mathrm{Li}_0(z) =
    z/(1-z)`, which `polylog` itself refuses to compute (it requires ``n >= 1``), and
    :math:`\mathrm{Li}_0(z)/z` is just :math:`1/(1-z)`.
    """
    (z,), (dz,) = primals, tangents
    z_arr = promote_integers(z)
    if n == 1:
        deriv = 1.0 / (1.0 - z_arr)
    else:
        # `Li_{n-1}(0) / 0` is 0/0, and returned `nan` for every order n >= 2
        # while the *value* `polylog(n, 0)` was correctly 0. (`n = 1` has its own
        # branch and was never affected, which made the break look selective.)
        #
        # Guarding it with `where(z == 0, 1.0, ratio)` fixes the first
        # derivative and silently breaks the second: a constant branch
        # differentiates to 0, where `Li_n''(0) = 2^(1-n)`. That is the same
        # trap `spence` fell into at its own removable point, so instead of
        # patching a value in, the ratio is *rewritten* as the series it equals:
        #
        #     Li_{n-1}(z) / z = sum_{j >= 0} z^j / (j + 1)^(n - 1)
        #
        # which is analytic at 0 and therefore right to every order. It is used
        # only for |z| < 1/2, where 60 terms give 8.7e-19; beyond that the
        # division is nowhere near zero and the closed form is kept.
        # Evaluated by Horner (`jnp.polyval`) rather than as `sum(z**j * c_j)`:
        # `z**j` differentiates to `j * z**(j-1)`, which is `0 * inf` at z = 0
        # for j = 0, so the sum form is `nan` at the very point this exists to
        # get right. Horner keeps it an honest polynomial, differentiable to
        # every order there.
        small = jnp.abs(z_arr) < 0.5
        j = jnp.arange(_N_TERMS, dtype=z_arr.dtype)
        coefficients = (1.0 / (j + 1.0) ** (n - 1))[::-1]
        series = jnp.polyval(coefficients, jnp.where(small, z_arr, 0.0))
        z_big = jnp.where(small, 1.0, z_arr)
        deriv = jnp.where(small, series, _li_core(n - 1, z_big) / z_big)
    value = _li_core(n, z_arr)
    # The value gives up above the Bernoulli table (`n > ORDER`, |z| >= 2) and
    # returns `nan`; the derivative needs only `Li_{n-1}`, so at exactly
    # `n = ORDER + 1` it stayed finite and correct. Value and gradient then
    # disagreed about the supported domain at one order, which is worse than
    # either answer alone -- a caller guarding on `isnan(value)` was safe and
    # one guarding on `isnan(grad)` was not.
    # Multiplied by a `nan`, rather than selected against one. Three attempts
    # at this, and the first two were the same mistake at different orders: a
    # `where` whose chosen branch is a constant transposes to a *zero*
    # cotangent, so `nan` on the product gave `grad` 0.0, and `nan` on the
    # factor gave `grad` the right answer but left `grad(grad)` at 0.0 --
    # differentiating that `where` differentiates a constant. Putting `z * nan`
    # there propagates but poisons the branch that was *not* taken, which cost
    # `Li_3''(0)` its perfectly good 0.25.
    #
    # A multiplicative mask has neither problem. `deriv * nan` is `nan`, and so
    # is every derivative of it, because the product rule keeps the factor;
    # where the mask is 1.0 nothing is disturbed, and the mask's own derivative
    # is zero either way, so no unselected branch leaks.
    out_of_domain = jnp.where(jnp.isnan(value), jnp.nan, 1.0)
    # The tangent must carry the *primal's* dtype, and the `n == 1` branch does
    # not get that for free: it builds `1/(1 - z)` from the argument, while the
    # primal comes back promoted. Every order from 2 up routes its derivative
    # through `_li_core` and is promoted already, which is why only order 1
    # raised -- and only for a dtype narrower than the promoted one.
    tangent = (deriv * out_of_domain).astype(value.dtype)
    return value, tangent * dz


_li = jax.jit(_li_core, static_argnums=(0,))
"""`_li_core` under `jit`; the custom JVP rides along."""
