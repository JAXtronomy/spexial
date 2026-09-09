r"""``spexial``: `scipy.special` in JAX.

``spexial`` implements special functions on top of JAX, so that they compose
with `jax.jit`, `jax.vmap` and `jax.grad`. Where a `scipy.special` counterpart
exists the name, call signature and return values follow it.

Two exports have **no** `scipy.special` counterpart:

- `Li`, the polylogarithm :math:`\mathrm{Li}_n(z)` (compare `mpmath.polylog`).
- `eval_gegenbauers`, which returns the Gegenbauer polynomial of degree ``n``
  *and every lower degree*, as a by-product of the recurrence.

The remaining exports do have one, but are not always drop-in replacements --
`zeta` covers only part of the negative half-line, and `K0`/`K1`/`K2` are
accurate to ~1e-8 rather than to machine precision. Each docstring states its
own domain and accuracy.

Examples
--------
>>> import spexial as sp
>>> float(sp.comb(5, 2))
10.00000000000002

"""

__all__ = [
    "K0",
    "K1",
    "K2",
    "K0e",
    "K1e",
    "K2e",
    "Li",
    "__version__",
    "comb",
    "eval_gegenbauer",
    "eval_gegenbauers",
    "gamma",
    "spence",
    "zeta",
]

from .setup_package import install_import_hook

with install_import_hook("spexial"):
    from ._src.comb import comb
    from ._src.gamma import gamma
    from ._src.gegenbauer import eval_gegenbauer, eval_gegenbauers
    from ._src.kn import K0, K1, K2, K0e, K1e, K2e
    from ._src.polylog import Li
    from ._src.spence import spence
    from ._src.zeta import zeta
    from ._version import version as __version__
