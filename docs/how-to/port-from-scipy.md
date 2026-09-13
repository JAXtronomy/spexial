# How to port code from `scipy.special`

Where a function exists in `scipy.special`, `spexial` keeps the same name and the same argument order, so the port is usually an import swap plus JAX arrays.

```pycon
>>> import jax
>>> jax.config.update("jax_enable_x64", True)

>>> import jax.numpy as jnp
>>> import spexial as sp

```

## Swap the import

```python
# before
from scipy.special import eval_gegenbauer

# after
from spexial import eval_gegenbauer
```

Pass `jnp` arrays rather than NumPy ones, and enable double precision first — see [How to enable double precision](enable-double-precision.md).

## Check the three things that differ

Before you trust a ported call, check the function against [Accuracy and domains](../reference/accuracy-and-domains.md) for these:

**Is there a counterpart at all?** `polylog`, `eval_gegenbauers`, `incomplete_beta`, `sph_harm_y_cart` and `sph_harm_y_cart_all` have none — they are additions, not replacements.

Watch `incomplete_beta` in particular: the _name_ has no SciPy counterpart, but `scipy.special.betainc` looks like one and is not. SciPy's is the regularized $I_z(a,b)$; this is the unregularized $B(a,b,z)$, larger by a factor of the complete beta function. Port `beta(a, b) * betainc(a, b, z)` to `incomplete_beta(a, b, z)`, not `betainc` to `incomplete_beta`.

**Is it as accurate?** Not always — the modified Bessel functions in particular are tested to a looser tolerance than SciPy delivers. Check the _Tested to_ column before you assume parity.

**Does it cover the same arguments?** For the most part yes: `gamma` and `zeta` accept everything their SciPy counterparts do. Check the _Supported domain_ column for the exceptions, which are about representable range rather than coverage — `k0`, `k1` and `k2` underflow above $z \approx 705$, where `k0e`, `k1e` and `k2e` keep working.

In the other direction, `spence` accepts complex input, which SciPy's real path does not, and `k0e`/`k1e`/`k2e` stay accurate to `DBL_MAX`, where `scipy.special.kve` returns `nan`.

## The angular functions take static degrees

`sph_legendre_p` and `sph_harm_y` keep SciPy's names and argument order, but their degree and order are **static Python `int`s** rather than arrays:

```pycon
>>> theta = jnp.asarray([0.3, 1.1, 2.0])
>>> sp.sph_legendre_p(2, 1, theta).shape  # not sp.sph_legendre_p([2], [1], theta)
(3,)

```

So a SciPy call that broadcasts over degrees becomes a Python loop over them — or, better, one call to `sph_harm_y_cart_all`, which returns the whole table in the same layout as `scipy.special.sph_harm_y_all` and shares the recurrences across it.

The narrowing is deliberate: it is what makes the values right. `jax.scipy.special.sph_harm_y` accepts array degrees and pairs them positionally with the angles rather than broadcasting, which is a wrong answer rather than a missing feature. See [Accuracy and domains](../reference/accuracy-and-domains.md).

One convention differs outside SciPy's documented domain. For $\theta \notin [0, \pi]$, `sph_legendre_p` carries $\sin^m\theta$ where SciPy carries $\lvert\sin\theta\rvert^m$, so the two differ by a sign for odd $m$. On $[0, \pi]$ they agree.

## Expect `nan` where SciPy raised

SciPy raises or warns on some bad input. Traced JAX code cannot raise, so `spexial` returns `nan` or `inf` instead, and that value propagates silently through `jit`, `vmap` and `grad`. If your SciPy code relied on an exception to catch bad input, replace it with an explicit check:

```pycon
>>> x = sp.k0(-1.0)  # negative argument: outside the domain
>>> bool(jnp.isnan(x))
True

```

The exception is a parameter that is a static Python value rather than an array — `spexial` can and does validate those eagerly:

```pycon
>>> try:
...     sp.polylog(0, 0.5)
... except ValueError as e:
...     print(e)
...
polylog is only implemented for integer order n >= 1, got 0

```

## Keep an eye on `comb`

`spexial.comb` is the inexact variant — it is built on `gammaln`, equivalent to `scipy.special.comb(..., exact=False)`. There is no `exact=True` path, so if your SciPy code relied on exact integer combinatorics, `spexial` is not a drop-in.
