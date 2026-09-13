r"""``spexial``: `scipy.special` in JAX.

``spexial`` implements special functions on top of JAX, so that they compose
with `jax.jit`, `jax.vmap` and `jax.grad`. Where a `scipy.special` counterpart
exists the name, call signature and return values follow it.

Two exports have **no** `scipy.special` counterpart:

- `polylog`, the polylogarithm :math:`\mathrm{Li}_n(z)` (compare `mpmath.polylog`).
- `eval_gegenbauers`, which returns the Gegenbauer polynomial of degree ``n``
  *and every lower degree*, as a by-product of the recurrence.

The remaining exports do have one, but are not always drop-in replacements --
`k0`/`k1`/`k2` are accurate to ~2e-7 rather than to machine precision, and
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
    # Deprecated uppercase spellings, kept working until their removal
    # release. See `spexial._src.deprecated`.
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
    "incomplete_beta",
    "k0",
    "k0e",
    "k1",
    "k1e",
    "k2",
    "k2e",
    "polylog",
    "spence",
    "sph_harm_y",
    "sph_harm_y_cart",
    "sph_harm_y_cart_all",
    "sph_harm_y_cart_all_terms",
    "sph_legendre_p",
    "zeta",
]

from .setup_package import install_import_hook as _install_import_hook

with _install_import_hook("spexial"):
    from ._src.beta import incomplete_beta
    from ._src.comb import comb
    from ._src.gamma import gamma
    from ._src.gegenbauer import eval_gegenbauer, eval_gegenbauers
    from ._src.kn import k0, k0e, k1, k1e, k2, k2e
    from ._src.polylog import polylog
    from ._src.spence import spence
    from ._src.sph_harm import (
        sph_harm_y,
        sph_harm_y_cart,
        sph_harm_y_cart_all,
        sph_harm_y_cart_all_terms,
        sph_legendre_p,
    )
    from ._src.zeta import zeta
    from ._version import version as __version__

# Outside the hook: these are assignments, not definitions, so jaxtyping has
# nothing to instrument, and the objects they wrap were already instrumented
# above.
from ._src.deprecated import K0, K1, K2, K0e, K1e, K2e, Li

# The hook is documented as living on `spexial.setup_package`, so it should not
# also be reachable as `spexial.install_import_hook` -- a name users could come
# to depend on by accident. Deleted rather than merely left out of `__all__`,
# which governs `import *` and nothing else.
del _install_import_hook
