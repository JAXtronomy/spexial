"""Binomial coefficients."""

__all__ = ["comb"]

import jax.numpy as jnp
from jax.scipy.special import gammaln

from ._typing import AnyArray, AnyArrayLike


def comb(N: AnyArrayLike, k: AnyArrayLike, /) -> AnyArray:
    """Compute the number of combinations of ``N`` things taken ``k`` at a time.

    This is the "N choose k" factor :math:`N! / (k! (N-k)!)`. It is the
    *inexact* variant of `scipy.special.comb` -- the ``exact=False`` one -- and
    is evaluated through `jax.scipy.special.gammaln`, so ``N`` and ``k`` need
    not be integers and the result is a float, not a Python `int`.

    Reference:
    https://docs.scipy.org/doc/scipy/reference/generated/scipy.special.comb.html

    Parameters
    ----------
    N
        Number of things. Broadcast against ``k``.
    k
        Number of elements taken. Broadcast against ``N``.

    Returns
    -------
    Array
        The total number of combinations. Following `scipy.special.comb`, this
        is ``0`` wherever ``k > N``, ``k < 0``, or ``N < 0``.

    Examples
    --------
    >>> import jax.numpy as jnp
    >>> import spexial as sp

    >>> round(float(sp.comb(5, 2)), 12)
    10.0

    Out-of-range ``k`` gives 0, not `nan`:

    >>> [
    ...     round(float(c), 12)
    ...     for c in sp.comb(jnp.asarray([5, 5, 5]), jnp.asarray([2, 6, -1]))
    ... ]
    [10.0, 0.0, 0.0]

    Non-integer arguments are the generalized binomial coefficient:

    >>> round(float(sp.comb(5.5, 2)), 12)
    12.375

    """
    n_arr, k_arr = jnp.asarray(N) * 1.0, jnp.asarray(k) * 1.0
    in_domain = (n_arr >= 0) & (k_arr >= 0) & (k_arr <= n_arr)
    # Mask *before* the gammaln call: gammaln of a non-positive integer is
    # +inf, and inf - inf would give nan rather than the 0 scipy returns.
    n_safe = jnp.where(in_domain, n_arr, 0.0)
    k_safe = jnp.where(in_domain, k_arr, 0.0)
    log_comb = gammaln(n_safe + 1) - gammaln(k_safe + 1) - gammaln(n_safe - k_safe + 1)
    return jnp.where(in_domain, jnp.exp(log_comb), 0.0)
