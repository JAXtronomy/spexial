"""Shared `jaxtyping` aliases.

Note that this module is NOT public API nor are any of its contents.
Stability is NOT guaranteed.

The ``*Like`` aliases accept anything `jax.numpy.asarray` accepts -- Python
scalars, NumPy arrays, and `jax.Array` -- and are used for *inputs*. The plain
aliases are `jax.Array` only and are used for *returns*, since every public
function in `spexial` returns a `jax.Array`.

"""

__all__: tuple[str, ...] = ()

from typing import TypeAlias

from jaxtyping import Array, ArrayLike, Real, Shaped

Scalar: TypeAlias = Shaped[Array, ""]
"""A 0-dimensional `jax.Array` of any dtype."""

Vector: TypeAlias = Shaped[Array, " N"]
"""A 1-dimensional `jax.Array` of any dtype."""

AnyArray: TypeAlias = Shaped[Array, "..."]
"""A `jax.Array` of any shape and dtype."""

ScalarLike: TypeAlias = Shaped[ArrayLike, ""]
"""Anything array-like that is 0-dimensional."""

AnyArrayLike: TypeAlias = Shaped[ArrayLike, "..."]
"""Anything array-like, of any shape and dtype."""

RealArrayLike: TypeAlias = Real[ArrayLike, "..."]
"""Anything array-like with a real (non-complex, non-bool) dtype."""
