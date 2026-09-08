"""The polylogarithm."""

__all__ = ["Li"]

from functools import partial
from typing import Final

import jax
import jax.numpy as jnp
from jax import lax
from jax.scipy.special import gamma as _jax_gamma

from ._bernoulli import ORDER, bernoulli_numbers
from ._typing import AnyArray, Scalar, ScalarLike
from .comb import comb
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
    bs = bernoulli_numbers()
    return lax.fori_loop(
        0,
        n + 1,
        lambda i, val: val + bs[i] * comb(n, i) * x ** (n - i),
        jnp.zeros_like(x),
    )


@partial(jax.jit, static_argnums=(0,))
def Li(n: int, z: ScalarLike, /) -> Scalar:  # noqa: N802
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
        ``>= 1``.
    z
        Real, **scalar** argument. The middle series is built from a
        length-60 vector of powers of :math:`\log z`, so it cannot broadcast;
        use ``jax.vmap(partial(Li, n))`` for arrays.

    Returns
    -------
    Array
        The real part of :math:`\mathrm{Li}_n(z)`. For ``z > 1`` the
        polylogarithm is genuinely complex; only its real part is returned.

    Notes
    -----
    Measured against `mpmath.polylog` over ``1 <= n <= 20`` and
    ``|z| <= 1000``, the relative error stays below ~1e-12 in all three
    branches. Large ``n`` is bounded by :math:`\Gamma(n+1)`, which overflows
    above ``n = 170``.

    Examples
    --------
    >>> import spexial as sp

    ``Li_1(z) == -log(1 - z)``:

    >>> round(float(sp.Li(1, 0.5)), 12)
    0.69314718056

    ``Li_2(1) == zeta(2)``:

    >>> round(float(sp.Li(2, 1.0)), 10)
    1.6449340668

    >>> round(float(sp.Li(3, -1.0)), 8)
    -0.90154268

    """
    if n < 1:
        msg = f"Li is only implemented for integer order n >= 1, got {n}"
        raise ValueError(msg)

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

        at_one = jnp.isclose(z - 1.0, 0.0)
        harmonic = jnp.sum(1.0 / jnp.arange(1, n))
        harmonic_term = jnp.where(
            at_one,
            0.0,
            log_z ** (n - 1)
            / _jax_gamma(n)
            * (harmonic - jnp.log(-jnp.log(jnp.where(at_one, 2.0, z) + 0j) + 0j)),
        )
        return jnp.real(zeta_series + harmonic_term)

    def inversion(z: AnyArray) -> AnyArray:
        """Evaluate the inversion formula, for |z| >= 2."""
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

    z_arr = jnp.asarray(z) * 1.0
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
