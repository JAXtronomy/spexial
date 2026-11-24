"""The hyperbolic secant, with a gradient that stays finite and accurate.

Neither NumPy, `jax.numpy` nor `scipy.special` has `sech`, so it is usually
written ``1 / jnp.cosh(x)``. That is right in value but not in gradient; see
`sech` for what goes wrong and what this does instead.
"""

__all__ = ["sech"]

from typing import Any

import jax
import jax.numpy as jnp

from .custom_types import AnyArray, AnyArrayLike


@jax.custom_jvp
def sech(x: AnyArrayLike, /) -> AnyArray:
    r"""Compute the hyperbolic secant :math:`\mathrm{sech}(x) = 1/\cosh(x)`.

    Written ``1 / jnp.cosh(x)`` the value is only good to about :math:`5\times10^{-14}`
    in float64 and the gradient is `nan` from :math:`|x| \approx 710` (89 in
    float32), and wrong before that. This evaluates :math:`2/(e + 1/e)` with
    :math:`e = \exp(|x|)`, which loses nothing to the overflow of `e`: it gives
    the correct limit 0. The derivative :math:`-\mathrm{sech}(x)\tanh(x)` is
    supplied by hand, because differentiating the value form directly loses
    the gradient's relative accuracy near zero, where its derivative contains
    a cancellation. See *Accuracy and domains* in the documentation for the
    numbers.

    Real input only; a complex argument raises `ValueError`.

    Reference: https://dlmf.nist.gov/4.28 for the definition and
    https://dlmf.nist.gov/4.34 for the derivative.

    Parameters
    ----------
    x
        Argument, of any shape. Evaluated elementwise. Integers are promoted to
        floating point; a floating argument keeps its dtype.

    Returns
    -------
    Array
        :math:`\mathrm{sech}(x)`, in ``[0, 1]``. ``sech(inf)`` and
        ``sech(-inf)`` are ``0``, and ``nan`` stays ``nan``.

    Examples
    --------
    >>> import jax
    >>> import jax.numpy as jnp
    >>> import spexial as sp

    >>> float(sp.sech(0.0))
    1.0

    The gradient is finite where ``1 / jnp.cosh`` gives `nan`:

    >>> float(jax.grad(lambda x: 1 / jnp.cosh(x))(800.0))
    nan
    >>> float(jax.grad(sp.sech)(800.0))
    -0.0

    """
    x_arr = jnp.asarray(x)
    if not jnp.issubdtype(x_arr.dtype, jnp.inexact):
        # Integers and bools: a *strong* float of the default width. Left to
        # `jnp.exp`, a Python int would stay weakly typed, and bool, int8 and
        # uint8 would give float32 instead of float64.
        x_arr = x_arr.astype(jnp.result_type(float))
    # The dtype is static, so this is resolved at trace time and costs nothing.
    if jnp.iscomplexobj(x_arr):
        msg = f"sech is only implemented for real x, got dtype {x_arr.dtype}"
        raise ValueError(msg)
    e = jnp.exp(jnp.abs(x_arr))
    return 2 / (e + 1 / e)


@sech.defjvp
def _sech_jvp(primals: tuple[Any], tangents: tuple[Any]) -> tuple[AnyArray, AnyArray]:
    r""":math:`\mathrm{sech}'(x) = -\mathrm{sech}(x)\tanh(x)`."""
    (x,), (dx,) = primals, tangents
    s = sech(x)
    return s, -s * jnp.tanh(x) * dx
