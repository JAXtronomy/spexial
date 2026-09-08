# Conventions

## Naming

- **A function with a `scipy.special` counterpart keeps SciPy's name.** Porting code should be an import swap.
- **A function without a counterpart keeps the name used in the literature.** These are called out as having no SciPy counterpart in the [API reference](api.md) and in [Accuracy and domains](accuracy-and-domains.md).
- The public surface is the top-level `spexial` namespace, and only what `spexial.__all__` lists. Anything under `spexial._src` is private and may change without notice.

## Signatures

- **Argument order follows SciPy.** Degree/order parameters come first, then function parameters, then the point of evaluation -- e.g. `eval_gegenbauer(n, alpha, x)`.
- **Integer-valued structural parameters (degrees, orders) are Python `int`s** and are static under `jit`. Continuous parameters and evaluation points are arrays.
- **Inputs and outputs are `jax.Array`.** Shapes are annotated with [jaxtyping](https://docs.kidger.site/jaxtyping/).

## Behaviour

- Functions are pure and JAX-transformable: `jit`, `vmap`, `grad`.
- Out-of-domain _array_ inputs return `nan`/`inf` rather than raising -- traced code cannot raise. A static Python parameter can still be validated eagerly, and is: `Li` raises `ValueError` for a non-integer or non-positive order. See [About domain edges](../explanation/edges.md).
- Double precision is assumed. Accuracy claims hold with `jax_enable_x64`.

## Docstrings

NumPy-style, with an `Examples` section. The conventions contributors are held to are in [Contributing](../about/contributing.md).
