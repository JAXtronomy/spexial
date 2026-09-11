# How to use spexial with jit, vmap and grad

`spexial` functions are ordinary JAX functions, so the transforms apply directly. Three things about this library change how you reach for them.

```pycon
>>> import jax
>>> jax.config.update("jax_enable_x64", True)

>>> import jax.numpy as jnp
>>> import spexial as sp

```

## Evaluate over an array

Most functions already broadcast over their evaluation point, so you need no transform at all:

```pycon
>>> sp.eval_gegenbauer(3, 0.5, jnp.linspace(-1.0, 1.0, 5))
Array([-1.    ,  0.4375, -0.    , -0.4375,  1.    ], dtype=float64)

```

If you want `vmap` anyway — to map over an axis of a larger computation — it gives the same answer.

## Evaluate `polylog` over an array

`polylog` is the exception: it accepts only a scalar `z`, and an array argument raises a broadcasting `TypeError` rather than mapping elementwise. To evaluate it over an array, use `vmap`:

```pycon
>>> jax.vmap(lambda z: sp.polylog(2, z))(jnp.array([0.25, 0.5]))
Array([0.26765264, 0.58224053], dtype=float64)

```

Do not reach for a Python loop here; `vmap` compiles to a single batched kernel. See [Accuracy and domains](../reference/accuracy-and-domains.md) for which functions carry restrictions.

## Hold an integer parameter fixed

Degrees and orders — the `n` of $C_n^{(\alpha)}$, the `n` of $\mathrm{Li}_n$ — are Python `int`s that change the shape of the computation. They are static: you cannot pass a traced array where one is expected, and each distinct value compiles separately.

To map over the continuous arguments while holding the degree fixed, pin it with `in_axes=None`:

```pycon
>>> f = lambda n, x: sp.eval_gegenbauer(n, 0.5, x)
>>> jax.vmap(f, in_axes=(None, 0))(3, jnp.linspace(-1.0, 1.0, 5))
Array([-1.    ,  0.4375, -0.    , -0.4375,  1.    ], dtype=float64)

```

If you need many degrees at once, prefer a function that returns them all in one call — `eval_gegenbauers` returns every order up to `n` — over sweeping `n` in a Python loop, which pays a compilation per iteration.

## Differentiate

`jax.grad` works directly:

```pycon
>>> float(jax.grad(sp.zeta)(4.0))
-0.06891126589612538

```

One trap: **`zeta` is only differentiable in a meaningful sense for $n > 1$.** On the negative line its value comes from a Bernoulli-number table, and `jax.grad` will return a finite number there that is not $\zeta'$. It will not warn you. See [About domain edges](../explanation/edges.md).

## Compile

```pycon
>>> float(jax.jit(sp.zeta)(4.0))
1.0823232337111384

```

Compose the transforms as you would anywhere else in JAX — `jax.jit(jax.vmap(jax.grad(f)))` is fine.
