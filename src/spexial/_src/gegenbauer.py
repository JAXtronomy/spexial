"""Jax implementation of the Gegenbauer polynomials."""

__all__ = ["eval_gegenbauer", "eval_gegenbauers"]

from functools import partial
from typing import Any, TypeAlias

import jax
import jax.numpy as jnp
from jaxtyping import Array, Shaped

Scalar: TypeAlias = Shaped[Array, ""]
Vector: TypeAlias = Shaped[Array, "N"]  # type: ignore[name-defined]
AnyArray: TypeAlias = Shaped[Array, "..."]
Carry: TypeAlias = tuple[float, Scalar, Scalar, Scalar]


@jax.jit
def C0(x: Any, /) -> float:
    """Gegenbauer polynomial of order 0. THis is 1.0."""
    return 1.0


@jax.jit
def C1(alpha: float, x: AnyArray) -> AnyArray:
    """Gegenbauer polynomial of order 1.

    This is :math:`2 * alpha * x`.

    """
    return 2 * alpha * x


@jax.jit
def _C_n_plus_1(carry: Carry, n: int) -> tuple[Carry, Scalar]:
    """Recursive formula for Gegenbauer polynomials."""
    alpha, x, Cn, Cn_minus_1 = carry
    accumulate = (2 * (n + alpha) * x * Cn - (n + 2 * alpha - 1) * Cn_minus_1) / (n + 1)
    return (alpha, x, accumulate, Cn), accumulate


# TODO: support n non-integer
@partial(jax.jit, static_argnums=(0,))
def eval_gegenbauers(n: int, alpha: float, x: Scalar) -> Vector:
    r"""Return Gegenbauer polynomial and all lower at a point.

    The Gegenbauer polynomials can be defined via the Gauss
    hypergeometric function :math:`{}_2F_1` as

    .. math::

        C_n^{(\alpha)} = \frac{(2\alpha)_n}{\\Gamma(n + 1)}
          {}_2F_1(-n, 2\alpha + n; \alpha + 1/2; (1 - z)/2).

    When :math:`n` is an integer the result is a polynomial of degree
    :math:`n`. See 22.5.46 in [AS]_ for details.

    Parameters
    ----------
    n : int
        Degree of the polynomial.

        :note:`non-integers are not supported yet.`

    alpha : float
        Parameter.

        :note:`non-floats are not supported yet.`

    x : Array[Any, ()]
        Point at which to evaluate the Gegenbauer polynomial.

        :note:`use vmap(..., in_axes=(0, None, None)).`

    Returns
    -------
    C : Array[Any, (N,)]
        Values of the (i<=n, alpha) Gegenbauer polynomial

    """
    C0_val = C0(x)
    C1_val = C1(alpha, x)

    carry = (alpha, x, C1_val, C0_val)
    n_values = jnp.arange(1, n)  # starts from 1 since already initialized 0->1 above

    _, C_values = jax.lax.scan(_C_n_plus_1, carry, n_values)

    return jnp.hstack([C0_val, C1_val, C_values])


# TODO: support n non-integer
@partial(jax.jit, static_argnums=(0,))
def eval_gegenbauer(n: int, alpha: float, x: Scalar) -> Scalar:
    r"""Evaluate Gegenbauer polynomial at a point.

    The Gegenbauer polynomials can be defined via the Gauss
    hypergeometric function :math:`{}_2F_1` as

    .. math::

        C_n^{(\alpha)} = \frac{(2\alpha)_n}{\\Gamma(n + 1)}
          {}_2F_1(-n, 2\alpha + n; \alpha + 1/2; (1 - z)/2).

    When :math:`n` is an integer the result is a polynomial of degree
    :math:`n`. See 22.5.46 in [AS]_ for details.

    Parameters
    ----------
    n : int
        Degree of the polynomial.
    alpha : float
        Parameter
    x : Array
        Points at which to evaluate the Gegenbauer polynomial.

    Returns
    -------
    C : Array
        Values of the Gegenbauer polynomial

    See Also
    --------
    roots_gegenbauer : roots and quadrature weights of Gegenbauer
                       polynomials
    gegenbauer : Gegenbauer polynomial object
    hyp2f1 : Gauss hypergeometric function

    References
    ----------
    .. [AS] Milton Abramowitz and Irene A. Stegun, eds.
        Handbook of Mathematical Functions with Formulas,
        Graphs, and Mathematical Tables. New York: Dover, 1972.

    """
    if n == 0:
        return C0(x)
    if n == 1:
        return C1(alpha, x)

    carry = (alpha, x, C1(alpha, x), C0(x))
    n_values = jnp.arange(1, n)  # already done 0
    _, C_values = jax.lax.scan(_C_n_plus_1, carry, n_values)
    return C_values[-1]
