from functools import partial

import jax
from jax import lax, numpy as jnp
from jax.scipy.special import bernoulli, gamma as jax_gamma

from .comb import comb
from .zeta import Riemann_zeta

# Number of terms for Li series.
L = 60

# List of Bernoulli numbers B_n
bernoulli_ary = bernoulli(L)


def Bernoulli(n, x):
    """The Bernoulli polynomial B_n(x), only implemented
    for n < 60. See https://en.wikipedia.org/wiki/Bernoulli_polynomials.

    Parameters
    ----------
    n : int
        Order of Bernoulli polynomial

    x : float
        Input value

    Returns
    -------
    float
        Value of B_n(x)

    """
    return lax.fori_loop(
        0, n + 1, lambda i, val: val + bernoulli_ary[i] * comb(n, i) * x ** (n - i), 0
    )


@partial(jax.jit, static_argnums=(0,))
def Li(n, z):
    """Polylogarithm of order n and argument z.  No scipy.special
    equivalent.

    Parameters
    ----------
    n : int
        Order of the polylogarithm

    z : float
        Argument of polylogarithm.

    Returns
    -------
    float
        Value of Li_n(z)

    """

    def _Li_z_small(z):
        return lax.fori_loop(1, L, lambda j, val: val + z**j / j**n, 0)

    def _Li_z_intermed(z):
        # Oddly enough, the fastest way to do this.
        zeta_ary = jnp.concatenate(
            (
                jnp.array([Riemann_zeta(n - m) for m in jnp.arange(n - 1)]),
                jnp.array([0.0]),
                jnp.array([Riemann_zeta(n - m) for m in jnp.arange(n, L)]),
            )
        )

        zeta_series_term = jnp.sum(
            zeta_ary
            * jnp.concatenate(
                (jnp.array([1.0, jnp.log(z + 0j)]), jnp.log(z + 0j) ** jnp.arange(2, L))
            )
            / jax_gamma(jnp.arange(L) + 1.0)
        )

        H_n_m_1 = jnp.sum(1.0 / jnp.arange(1, n))
        harmonic_term = jnp.where(
            jnp.isclose(z - 1.0, 0),
            0.0,
            jnp.log(z + 0j) ** (n - 1)
            / jax_gamma(n)
            * (
                H_n_m_1
                - jnp.log(
                    -jnp.log(jnp.where(jnp.isclose(z - 1.0, 0), 2.0, z) + 0j) + 0j
                )
            ),
        )

        res = zeta_series_term + harmonic_term

        return jnp.real(res)

    def _Li_z_large(z):
        recip_Li = lax.fori_loop(1, L, lambda j, val: val + (1 / z) ** j / j**n, 0)

        B_n = Bernoulli(n, jnp.log(z + 0j) / (2 * jnp.pi * 1j))

        return jnp.real(
            -((-1) ** n) * recip_Li - (2 * jnp.pi * 1j) ** n / jax_gamma(n + 1) * B_n
        )

    small_range = jnp.abs(z) <= 0.5

    intermed_range = (jnp.abs(z) > 0.5) & (jnp.abs(z) < 2)

    large_range = jnp.abs(z) > 2

    return jnp.where(
        small_range,
        _Li_z_small(jnp.where(small_range, z, 3.0)),
        jnp.where(
            intermed_range,
            _Li_z_intermed(jnp.where(intermed_range, z, 3.0)),
            jnp.where(large_range, _Li_z_large(jnp.where(large_range, z, 3.0)), 0.0),
        ),
    )
