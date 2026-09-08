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

## Not every function accepts arrays

Some implementations branch internally on the value of their argument (`lax.cond` needs a scalar predicate), so they only accept a _scalar_ input. Passing an array raises a `TypeError` about a non-scalar predicate rather than broadcasting.

The fix is `vmap`, not a loop:

```pycon
>>> import spexial as sp
>>> f = lambda x: sp.eval_gegenbauer(3, 0.5, x)
>>> jax.vmap(f)(jnp.linspace(-1.0, 1.0, 5))
Array([-1.    ,  0.4375, -0.    , -0.4375,  1.    ], dtype=float64)
```

Which functions are scalar-only is recorded per function in [Accuracy and domains](accuracy-and-domains.md).

## Integer parameters are static, and each value recompiles

Degrees and orders control the _length_ of a recurrence, so they must be Python `int`s known at trace time, not traced arrays. Consequences:

- You cannot `vmap` over a degree. Map over the continuous arguments and hold the degree fixed with `in_axes=None`.
- You cannot compute a degree from data inside a `jit`.
- Every distinct degree triggers a fresh compilation. Sweeping a degree in a Python loop pays a compile per iteration; if you need many degrees at once, reach for a function that returns all of them in one call (for example `eval_gegenbauers`) instead.

## There are no domain errors

Traced code cannot raise. Out-of-domain inputs return `nan` or `inf`, and those propagate quietly through `jit`, `vmap` and `grad` -- often turning an entire gradient into `nan` from a single bad element. Validate inputs before the call, or check the output with `jnp.isnan`.

## Gradients at the edges

A function can be accurate at a point and still have a `nan` gradient there: branch boundaries, series cutoffs and endpoints of the domain are the usual culprits. If `jax.grad` returns `nan` where the forward pass is fine, suspect a boundary before suspecting your model.

## Debugging inside `jit`

`print` shows tracers, not values. Use `jax.debug.print` for values, and `jax.disable_jit()` to step through with an ordinary debugger.
