r"""``spexial``: `scipy.special` in JAX.

``spexial`` implements special functions on top of JAX, so that they compose
with `jax.jit`, `jax.vmap` and `jax.grad`. Where a `scipy.special` counterpart
exists the name, call signature and return values follow it.

Two exports have **no** `scipy.special` counterpart:

- `Li`, the polylogarithm :math:`\mathrm{Li}_n(z)` (compare `mpmath.polylog`).
- `eval_gegenbauers`, which returns the Gegenbauer polynomial of degree ``n``
  *and every lower degree*, as a by-product of the recurrence.

The remaining exports do have one, but are not always drop-in replacements --
`K0`/`K1`/`K2` are accurate to ~2e-7 rather than to machine precision, and
`comb` is the ``exact=False`` variant, so it returns a float that is only
close to the integer. Each docstring states its own domain and accuracy.

Examples
--------
>>> import spexial as sp
>>> round(float(sp.comb(5, 2)), 9)
10.0

`comb` is inexact, so the value is near 10 rather than exactly 10:

>>> float(sp.comb(5, 2)) == 10.0
False

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

from .setup_package import install_import_hook as _install_import_hook

with _install_import_hook("spexial"):
    from ._src.comb import comb
    from ._src.gamma import gamma
    from ._src.gegenbauer import eval_gegenbauer, eval_gegenbauers
    from ._src.kn import K0, K1, K2, K0e, K1e, K2e
    from ._src.polylog import Li
    from ._src.spence import spence
    from ._src.zeta import zeta
    from ._version import version as __version__

# The hook is documented as living on `spexial.setup_package`, so it should not
# also be reachable as `spexial.install_import_hook` -- a name users could come
# to depend on by accident. Deleted rather than merely left out of `__all__`,
# which governs `import *` and nothing else.
del _install_import_hook
