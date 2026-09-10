"""Coverage registry: what `jax.scipy.special` and `scipy.special` provide.

The central table of the library -- which functions `spexial` implements, which
are waiting on an upstream floor bump, and where an analytic derivative is worth
writing. See `spexial._src.registry` for the reasoning behind each field.

Examples
--------
>>> from spexial.registry import REGISTRY, Status
>>> sorted(k for k, v in REGISTRY.items() if v.status is Status.UNIQUE)
['K0', 'K0e', 'K1', 'K1e', 'K2', 'K2e', 'Li', 'eval_gegenbauer', 'eval_gegenbauers',
 'sph_harm_y_cart', 'sph_harm_y_cart_all', 'sph_harm_y_cart_all_terms',
 'sph_legendre_p']

"""

__all__ = ["JAX_FLOOR", "REGISTRY", "Coverage", "Status", "Support", "render_markdown"]

from ._src.registry import (
    JAX_FLOOR,
    REGISTRY,
    Coverage,
    Status,
    Support,
    render_markdown,
)
