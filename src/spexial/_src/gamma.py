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
from .dtype import (
    exactly_zero,
    is_negative,
    log_no_flush,
    positive_subnormal,
    promote_integers,
)


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
    # Integers only. `jax.scipy.special.gamma` handles every float width
    # itself, `float16` and `bfloat16` included, so a floating argument is
    # passed through untouched and keeps its dtype -- and must be, because
    # multiplying it by 1.0 would flush a subnormal to zero. See
    # `promote_integers`.
    x_arr = promote_integers(x)
    out = jss.gamma(x_arr)
    if jnp.issubdtype(x_arr.dtype, jnp.complexfloating):
        return out
    # Near zero, `Gamma(x) = 1/x - gamma_E + O(x)`, and once `x` is subnormal
    # the `1/x` term is the entire answer to full precision. Upstream returns
    # `inf` for all of them -- it flushes the argument internally -- but `1/x`
    # is still representable for the factor of about two between `tiny` and
    # `1/max`, which in float32 is the reachable band 2.9e-39 to 1.2e-38, and
    # SciPy gives the finite value there. This is the one place the delegated
    # value is deliberately overridden. See the guard below for on what grounds.
    magnitude = jnp.abs(x_arr)
    # The logarithm and its exponential are taken at the default float width,
    # not the caller's. A bfloat16 subnormal is a *float32* subnormal too --
    # the two share an exponent range -- so only float64 gives `log |x| ~ -87`
    # any room, and doing it in bfloat16 came out 36% wrong.
    # The *default* float width, which is float64 only when x64 is enabled --
    # with it off, this is float32 and the band costs ~30 ulp there. Either way
    # it is wider than the caller's for the two narrow types, which is the
    # point: a bfloat16 subnormal is a *float32* subnormal too, the two sharing
    # an exponent range, so `log |x| ~ -87` has no room in a dtype whose
    # spacing there is 0.5.
    wide = jnp.asarray(0.0).dtype
    reciprocal = jnp.exp(-log_no_flush(magnitude, dtype=wide)).astype(x_arr.dtype)
    signed = jnp.where(is_negative(x_arr), -reciprocal, reciprocal)
    # Substituted across the whole subnormal band, upstream finite or not.
    # Delegation is the default, and being *more accurate* is a reason to
    # depart from it: measured against mpmath over all 767 float16 subnormals
    # where `jax.scipy.special.gamma` returns a finite value, this branch is
    # better at 690 of them and worse at 5, worst 4.8e-4 against upstream's
    # 4.2e-3. An earlier version gated on `~isfinite(out)` to "keep upstream's
    # answer where upstream has one", justified by a 60x figure that was
    # measured backwards.
    return jnp.where(positive_subnormal(magnitude), signed, out)


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
    # Same promotion as the value above, through the same helper so the two
    # cannot drift apart -- they already had, once, leaving a comment here that
    # described a spelling the code no longer used.
    x_arr = promote_integers(x)
    # `digamma` is handed a subnormal it will flush, giving `nan` where the
    # derivative is a perfectly definite infinity: with `Gamma(x) ~ 1/x` near
    # zero, `Gamma'(x) ~ -1/x**2`, which overflows every float width in this
    # band and so is `-inf` from either side. That is what the function already
    # returns one ulp above `tiny`; substituting it keeps the derivative
    # continuous across a boundary that is an artefact of the representation,
    # not of the mathematics.
    #
    # A constant, because this is a genuine pole rather than a removable point:
    # every order diverges here, so no reformulation buys the next one.
    # `| exactly_zero` because `positive_subnormal` tests `bits > 0`, which is
    # False at the pole itself -- so the guard covered the whole band *except*
    # the one point most likely to be evaluated, and `gamma(0)` fell through to
    # `inf * digamma(0)` = `inf * nan`.
    diverges = positive_subnormal(jnp.abs(x_arr)) | exactly_zero(x_arr)
    # `digamma` is kept off the flushed argument rather than merely overridden.
    # Left to evaluate, its `nan` transposes into the selected branch and
    # reverse mode returns `nan` where forward mode returns the constant, so
    # the two modes disagreed about the same point.
    safe = jnp.where(diverges, jnp.ones_like(x_arr), x_arr)
    return g, jnp.where(diverges, -jnp.inf, g * digamma(safe)) * dx
