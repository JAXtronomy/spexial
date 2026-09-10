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

**Is there a counterpart at all?** `Li` and `eval_gegenbauers` have none — they are additions, not replacements.

**Is it as accurate?** Not always — the modified Bessel functions in particular are tested to a looser tolerance than SciPy delivers. Check the _Tested to_ column before you assume parity.

**Does it cover the same arguments?** For the most part yes: `gamma` and `zeta` accept everything their SciPy counterparts do. Check the _Supported domain_ column for the exceptions, which are about representable range rather than coverage — `K0`, `K1` and `K2` underflow above $z \approx 705$, where `K0e`, `K1e` and `K2e` keep working.

In the other direction, `spence` accepts complex input, which SciPy's real path does not, and `K0e`/`K1e`/`K2e` stay accurate to `DBL_MAX`, where `scipy.special.kve` returns `nan`.

## Expect `nan` where SciPy raised

SciPy raises or warns on some bad input. Traced JAX code cannot raise, so `spexial` returns `nan` or `inf` instead, and that value propagates silently through `jit`, `vmap` and `grad`. If your SciPy code relied on an exception to catch bad input, replace it with an explicit check:

```pycon
>>> x = sp.K0(-1.0)  # negative argument: outside the domain
>>> bool(jnp.isnan(x))
True

```

The exception is a parameter that is a static Python value rather than an array — `spexial` can and does validate those eagerly:

```pycon
>>> try:
...     sp.Li(0, 0.5)
... except ValueError as e:
...     print(e)
...
Li is only implemented for integer order n >= 1, got 0

```

## Keep an eye on `comb`

`spexial.comb` is the inexact variant — it is built on `gammaln`, equivalent to `scipy.special.comb(..., exact=False)`. There is no `exact=True` path, so if your SciPy code relied on exact integer combinatorics, `spexial` is not a drop-in.
