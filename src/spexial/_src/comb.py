from jax import numpy as jnp
from jax.scipy.special import gammaln


def comb(N, k):
    """Combinatoric factor N! / (k! (N-k)!), or the
    number of combinations of N things taken k at
    a time.

    This is often expressed as “N choose k”.

    See also https://docs.scipy.org/doc/scipy/reference/generated/scipy.special.comb.html#scipy.special.comb

    Parameters
    ----------
    N : array_like of int
        Number of things.

    k : array_like of int
        Number of elements taken.

    Returns
    -------
    array_like of float
        The total number of combinations.


    """
    return jnp.exp(gammaln(N + 1) - gammaln(k + 1) - gammaln(N - k + 1))
