from jax import numpy as jnp
from jax.scipy.special import bernoulli, zeta

# Number of terms for Li series.
L = 60

# List of Bernoulli numbers B_n
bernoulli_ary = bernoulli(L)


def Riemann_zeta(n):
    """Riemann/Hurwitz zeta function. Differs from the
    jax.scipy.special implementation in that it
    supports negative arguments.  Only implemented for
    n > -60.

    See also https://docs.scipy.org/doc/scipy/reference/generated/scipy.special.zeta.html#scipy.special.zeta

    Parameters
    ----------
    n : array_like of float
        Input data

    Returns
    -------
    array_like of float
        Value of zeta(n)

    """
    return jnp.where(
        n > 0,
        zeta(n, 1),
        jnp.where(
            (n < 0) & (-n % 2 == 0),
            0.0,
            (-1.0) ** (-n) * bernoulli_ary[-n + 1] / (-n + 1),
        ),
    )
