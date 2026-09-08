# spexial

`scipy.special` in [JAX](https://docs.jax.dev).

`spexial` implements special functions -- Gegenbauer polynomials, the gamma function, modified Bessel functions, the Riemann zeta function, polylogarithms -- as plain JAX functions. Because they are ordinary JAX code, they compose with the rest of the ecosystem: `jax.jit`, `jax.vmap`, `jax.grad`, and execution on CPU, GPU and TPU.

Where a function has a `scipy.special` counterpart, `spexial` matches its name, its argument order, and its convention. Porting a `scipy.special` call is usually a matter of changing the import.

## Installation

```bash
pip install spexial
```

or

```bash
uv add spexial
```

## A first example

The Gegenbauer polynomial $C_n^{(\alpha)}(x)$, evaluated at a point:

```pycon
>>> import jax
>>> jax.config.update("jax_enable_x64", True)

>>> import jax.numpy as jnp
>>> import spexial as sp

>>> sp.eval_gegenbauer(3, 0.5, jnp.array(0.5))
Array(-0.4375, dtype=float64, weak_type=True)
```

That call is `jit`-compiled, differentiable, and vectorisable like any other JAX function -- see [Getting started](guides/getting-started.md).

!!! warning "Enable x64"

    JAX defaults to 32-bit floats. Special functions are sensitive to precision,
    so enable double precision before doing anything else, as above. See
    [Sharp bits](guides/sharp-bits.md).

## Where to go next

- [Getting started](guides/getting-started.md) -- the JAX idioms `spexial` expects.
- [Accuracy and domains](guides/accuracy-and-domains.md) -- what is supported, where.
- [Sharp bits](guides/sharp-bits.md) -- precision, shapes, and tracing pitfalls.
- [API reference](api/index.md) -- every public function.
- [Conventions](conventions.md) -- naming and argument-order rules.
- [Contributing](contributing.md).
