"""Coverage registry: what `jax.scipy.special` and `scipy.special` provide.

The central table of the library -- which functions `spexial` implements, which
are waiting on an upstream floor bump, and where an analytic derivative is worth
writing. See `spexial._src.registry` for the reasoning behind each field.

Examples
--------
>>> from spexial.registry import REGISTRY, Status
>>> unique = sorted(k for k, v in REGISTRY.items() if v.status is Status.UNIQUE)
>>> unique[:4]
['eval_gegenbauer', 'eval_gegenbauers', 'incomplete_beta', 'k0']
>>> unique[4:9]
['k0e', 'k1', 'k1e', 'k2', 'k2e']
>>> unique[9:]
['polylog', 'sph_harm_y_cart', 'sph_harm_y_cart_all',
 'sph_harm_y_cart_all_terms', 'sph_legendre_p']

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
