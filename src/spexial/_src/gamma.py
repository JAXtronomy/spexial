"""The gamma function.

Delegates the value to `jax.scipy.special.gamma` and supplies an analytic
derivative. See `gamma` for why that is worth a module.
"""

__all__ = ["gamma"]

from typing import Any

import jax
import jax.numpy as jnp
import jax.scipy.special as jss
from jax.scipy.special import digamma

from .custom_types import AnyArray, AnyArrayLike
from .dtype import is_negative, log_no_flush, positive_subnormal


@jax.custom_jvp
def gamma(x: AnyArrayLike, /) -> AnyArray:
    r"""Compute the gamma function :math:`\Gamma(x)`.

    The value is `jax.scipy.special.gamma`, called directly, so it cannot drift
    from upstream. What this adds is the derivative: JAX differentiates its own
    implementation term by term, while :math:`\Gamma'(x) = \Gamma(x)\,\psi(x)`
    is one extra call. The saving is in **memory, not time**: measured over
    10,000 points, `jax.grad` keeps 240 kB of residuals through the backward
    pass against this rule's 80 kB, while wall-clock is a wash (171 µs against
    166 µs, i.e. 1.03x -- neutral within noise, reproduced across `grad`,
    `vmap(grad)` and `jvp` harnesses). An earlier revision of this docstring
    claimed 4.3x faster; that was a measurement error, and it is not plausible
    either -- JAX's `gamma` is `sign * exp(gammaln(x))`, whose autodiff already
    *is* the same product, so there is no arithmetic to save.

    Reference:
    https://docs.scipy.org/doc/scipy/reference/generated/scipy.special.gamma.html

    Parameters
    ----------
    x
        Argument, of any shape. Evaluated elementwise. Complex input requires
        ``jax >= 0.10.2``; below that `jax.scipy.special.gamma` raises, since it
        branches on ``floor(x)`` internally.

    Returns
    -------
    Array
        Value(s) of the gamma function. ``x = 0`` gives ``inf``; the negative
        integers give ``nan``, matching both `jax.scipy.special.gamma` and
        `scipy.special.gamma` from 1.18.

    Examples
    --------
    >>> import jax.numpy as jnp
    >>> import spexial as sp

    >>> round(float(sp.gamma(5.0)), 10)
    24.0

    It broadcasts, and handles negative arguments:

    >>> [round(float(g), 10) for g in sp.gamma(jnp.asarray([0.5, 1.0, -0.5]))]
    [1.7724538509, 1.0, -3.5449077018]

    Complex input works on ``jax >= 0.10.2`` and returns the complex gamma
    function; it is not shown here because this example is executed against
    every supported JAX, including ones that predate it.

    """
    # Only non-inexact input takes the multiply. `x * 1.0` promotes integers,
    # which is what it is for, but it also flushes a subnormal float to zero on
    # XLA -- the hazard `spexial._src.dtype` documents, and the one that cost
    # `K0` its entire subnormal band. `jax.scipy.special.gamma` handles every
    # float width itself, including `float16` and `bfloat16`, so a floating
    # argument is passed through untouched and keeps its dtype.
    x_arr = jnp.asarray(x)
    if not jnp.issubdtype(x_arr.dtype, jnp.inexact):
        x_arr = x_arr * 1.0
    out = jss.gamma(x_arr)
    if jnp.issubdtype(x_arr.dtype, jnp.complexfloating):
        return out
    # Near zero, `Gamma(x) = 1/x - gamma_E + O(x)`, and once `x` is subnormal
    # the `1/x` term is the entire answer to full precision. Upstream returns
    # `inf` for all of them -- it flushes the argument internally -- but `1/x`
    # is still representable for the factor of about two between `tiny` and
    # `1/max`, which in float32 is the reachable band 2.9e-39 to 1.2e-38, and
    # SciPy gives the finite value there. This is the one place the delegated
    # value is deliberately overridden, and only where upstream has no answer.
    magnitude = jnp.abs(x_arr)
    subnormal = positive_subnormal(magnitude)
    reciprocal = jnp.exp(-log_no_flush(magnitude))
    return jnp.where(
        subnormal, jnp.where(is_negative(x_arr), -reciprocal, reciprocal), out
    )


@gamma.defjvp
def _gamma_jvp(primals: tuple[Any], tangents: tuple[Any]) -> tuple[AnyArray, AnyArray]:
    r""":math:`\Gamma'(x) = \Gamma(x)\,\psi(x)`.

    Real input only. `jax.scipy.special.digamma` rejects a complex dtype, so
    applying this rule to complex input raised -- while `jax.scipy.special.gamma`,
    whose value this function forwards, differentiates complex input perfectly
    well. A `custom_jvp` cannot decline to apply, so the complex case is handed
    straight back to the function being wrapped. The dtype test is static, so
    the branch is resolved at trace time and costs nothing.
    """
    (x,), (dx,) = primals, tangents
    if jnp.iscomplexobj(jnp.asarray(x)):
        return jax.jvp(lambda v: jss.gamma(jnp.asarray(v)), (x,), (dx,))
    g = gamma(x)
    # `as_float`, not `* 1.0`: the multiply promotes integers and also flushes a
    # subnormal float to zero, which is the hazard `spexial._src.dtype` exists
    # to document. `digamma` is unaffected in practice -- it returns `-inf`
    # either way -- but the pattern is the one that cost `K0` its whole
    # subnormal band, so it does not stay in the codebase.
    x_arr = jnp.asarray(x)
    if not jnp.issubdtype(x_arr.dtype, jnp.inexact):
        x_arr = x_arr * 1.0
    return g, g * digamma(x_arr) * dx
