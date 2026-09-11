"""Deprecated uppercase spellings of the public names.

Note that this module is NOT public API nor are any of its contents.
Stability is NOT guaranteed. The *names* it defines are re-exported from
`spexial` and are deprecated there; this module is only where they live until
they are deleted.

`spexial` follows `scipy.special` for names, call signatures and return values.
Seven functions did not: `scipy.special` spells the modified Bessel functions
`k0`, `k1`, `k0e` and `k1e`, and `spexial` spelled them `K0`, `K1`, `K0e` and
`K1e`. `K2`/`K2e` have no direct scipy counterpart -- scipy reaches order two
through ``kn(2, x)`` and ``kve(2, x)`` -- but they belong to the same family and
took the same capital. The polylogarithm has no counterpart in either library,
and its lowercase form `li` is spoken for: `li` is the *logarithmic integral*,
a different function, which `mpmath` ships alongside `mp.polylog`. So that one
is `polylog` rather than `li`.

Removal follows the schedule in ``AGENTS.md``, which is three releases and not
one: the alias is deprecated here, and deleted two releases later. Nothing
breaks on an upgrade.
"""

__all__: tuple[str, ...] = ()

from typing_extensions import deprecated

from .kn import k0, k0e, k1, k1e, k2, k2e
from .polylog import polylog


def _message(old: str, new: str, why: str) -> str:
    """Build the deprecation text, so all seven read identically."""
    return (
        f"`spexial.{old}` is deprecated; use `spexial.{new}` instead. {why} "
        f"The uppercase alias still works and will be removed in a future "
        f"release -- see the removal schedule in AGENTS.md."
    )


_BESSEL = "`spexial` follows `scipy.special`, which spells this one lowercase."
_FAMILY = (
    "`scipy.special` reaches order two through `kn(2, x)` and `kve(2, x)` "
    "rather than a name of its own, so this follows `k0`/`k1` instead."
)
_POLYLOG = (
    "Neither `scipy.special` nor `jax.scipy.special` has a polylogarithm, so "
    "there is no name to follow; `polylog` matches `mpmath.polylog`. It is not "
    "`li`, which is the logarithmic integral and a different function."
)

# Assignments rather than `def`s on purpose: `deprecated` wraps the object it
# is given, and the objects here are `jax.custom_jvp` instances. Wrapping keeps
# `jit`, `vmap` and every order of `grad` working through the alias -- the
# wrapper is a plain call into the same rule -- while the warning fires once,
# at trace time, rather than inside a compiled kernel.
K0 = deprecated(_message("K0", "k0", _BESSEL))(k0)
K1 = deprecated(_message("K1", "k1", _BESSEL))(k1)
K2 = deprecated(_message("K2", "k2", _FAMILY))(k2)
K0e = deprecated(_message("K0e", "k0e", _BESSEL))(k0e)
K1e = deprecated(_message("K1e", "k1e", _BESSEL))(k1e)
K2e = deprecated(_message("K2e", "k2e", _FAMILY))(k2e)
Li = deprecated(_message("Li", "polylog", _POLYLOG))(polylog)
