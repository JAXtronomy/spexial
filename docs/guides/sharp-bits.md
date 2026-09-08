# Sharp bits

JAX's [own sharp bits](https://docs.jax.dev/en/latest/notebooks/Common_Gotchas_in_JAX.html) all apply here. This page covers the ones that bite hardest in a special-functions library.

## float32 is the default, and it is not enough

JAX creates 32-bit arrays unless told otherwise. Special functions are built out of series, recurrences and reflection formulae, all of which shed significant digits; at float32 you can lose most of them, and a result that is merely _wrong_ -- rather than `nan` -- is easy to miss.

Turn on double precision before creating any array:

```pycon
>>> import jax
>>> jax.config.update("jax_enable_x64", True)
>>> import jax.numpy as jnp
>>> jnp.array(1.0).dtype
dtype('float64')
```

Two things to watch:

- **It is process-global and order-sensitive.** Arrays created before the `update` call stay float32. Set it at the top of your entry point, or use the `JAX_ENABLE_X64=1` environment variable, which applies from interpreter start.
- **NumPy inputs are silently downcast.** A `numpy.float64` array passed into a JAX function becomes float32 without x64 enabled, and no warning is issued.

## `Li` accepts only a scalar `z`

Every function broadcasts over its evaluation point except [`Li`][spexial.Li]. Its intermediate branch builds a length-60 vector of powers of $\\log z$, so an array argument collides with that axis and raises a broadcasting `TypeError` from `pow` rather than mapping elementwise.

The fix is `vmap`, not a Python loop:

```pycon
>>> import spexial as sp
>>> jax.vmap(lambda z: sp.Li(2, z))(jnp.array([0.25, 0.5]))
Array([0.26765264, 0.58224053], dtype=float64)
```

This is the only such restriction, and it is recorded in [Accuracy and domains](accuracy-and-domains.md). It is a property of the algorithm, not a general pattern to expect elsewhere.

## Integer parameters are static, and each value recompiles

Degrees and orders control the _length_ of a recurrence, so they must be Python `int`s known at trace time, not traced arrays. Consequences:

- You cannot `vmap` over a degree. Map over the continuous arguments and hold the degree fixed with `in_axes=None`.
- You cannot compute a degree from data inside a `jit`.
- Every distinct degree triggers a fresh compilation. Sweeping a degree in a Python loop pays a compile per iteration; if you need many degrees at once, reach for a function that returns all of them in one call (for example `eval_gegenbauers`) instead.

## There are no domain errors

Traced code cannot raise. Out-of-domain inputs return `nan` or `inf`, and those propagate quietly through `jit`, `vmap` and `grad` -- often turning an entire gradient into `nan` from a single bad element. Validate inputs before the call, or check the output with `jnp.isnan`.

## Gradients at the edges -- and a gradient that lies

A function can be accurate at a point and still have a `nan` gradient there: branch boundaries, series cutoffs and endpoints of the domain are the usual culprits. If `jax.grad` returns `nan` where the forward pass is fine, suspect a boundary before suspecting your model.

Worse than a `nan` is a plausible number that is not the derivative. `zeta` is evaluated on the negative line by looking up a Bernoulli number, and a table lookup carries no derivative information -- so `jax.grad(sp.zeta)` returns a finite value there that is **not** $\\zeta'$. It will not warn you. Differentiate `zeta` only for $n > 1$.

## Debugging inside `jit`

`print` shows tracers, not values. Use `jax.debug.print` for values, and `jax.disable_jit()` to step through with an ordinary debugger.
