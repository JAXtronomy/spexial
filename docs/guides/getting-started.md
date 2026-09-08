# Getting started

`spexial` is a library of plain JAX functions. There is no `spexial` object model, no custom array type, and no dispatch layer -- if you know JAX, you already know how to use it.

## The two imports

```pycon
>>> import jax.numpy as jnp
>>> import spexial as sp
```

`jnp` and `sp` are the conventional aliases; the repo's own lint config enforces them. `spexial` functions take and return `jax.Array`, so the inputs you build with `jnp` flow straight in.

## Enable x64 first

JAX computes in 32-bit by default. Special functions -- series, recurrences, reflection formulae -- lose accuracy fast at that width, so turn on double precision before you touch any array:

```pycon
>>> import jax
>>> jax.config.update("jax_enable_x64", True)
```

This must happen before arrays are created, and it is process-global. The alternative is the `JAX_ENABLE_X64=1` environment variable. See [Sharp bits](sharp-bits.md) for what goes wrong if you skip it.

## Composing with JAX transforms

Everything is a normal JAX function, so the usual transforms apply.

`jit` compiles:

```pycon
>>> f = lambda x: sp.eval_gegenbauer(3, 0.5, x)
>>> jax.jit(f)(jnp.array(0.5))
Array(-0.4375, dtype=float64, weak_type=True)
```

`vmap` vectorises over an argument that a function only accepts as a scalar:

```pycon
>>> xs = jnp.linspace(-1.0, 1.0, 5)
>>> jax.vmap(f)(xs)
Array([-1.    ,  0.4375, -0.    , -0.4375,  1.    ], dtype=float64)
```

`grad` differentiates:

```pycon
>>> jax.grad(f)(jnp.array(0.5))
Array(0.375, dtype=float64, weak_type=True)
```

Compose them as you would anywhere else in JAX -- `jax.jit(jax.vmap(jax.grad(f)))` is fine.

## Integer-valued parameters are static

Degrees and orders (the `n` of $C_n^{(\alpha)}$, for instance) are Python `int`s that change the shape of the computation, so they are _static_: passing a traced array where an `int` is expected will fail, and each distinct value triggers its own compilation. Hold them fixed across a `vmap`/`jit` with `in_axes=None` and `static_argnums`, and map over the continuous arguments instead.

## Coming from SciPy

Where a function exists in `scipy.special`, `spexial` keeps the same name and argument order, so the port is normally an import swap plus `jnp` arrays. A few functions have no SciPy counterpart; those are flagged in [Accuracy and domains](accuracy-and-domains.md) and in the [API reference](../api/index.md).
