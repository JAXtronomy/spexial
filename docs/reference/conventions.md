# Conventions

## Naming

- **A function with a `scipy.special` counterpart keeps SciPy's name.** Porting code should be an import swap.
- **A function without a counterpart keeps the name used in the literature.** These are called out as having no SciPy counterpart in the [API reference](api.md) and in [Accuracy and domains](accuracy-and-domains.md).
- **Case follows SciPy too.** The modified Bessel functions are `k0`, `k1`, `k0e`, `k1e` in `scipy.special`, so they are lowercase here; `k2`/`k2e` follow the same family, though SciPy reaches order two through `kn(2, x)` and `kve(2, x)` rather than by a name of its own.
- The public surface is the top-level `spexial` namespace, and only what `spexial.__all__` lists. Anything under `spexial._src` is private and may change without notice.

### Deprecated spellings

`spexial` originally spelled seven functions with a capital, which broke the first rule above. The lowercase names are the real ones; the old spellings still work, emit a `DeprecationWarning`, and will be removed on the schedule in `AGENTS.md` -- three releases, so nothing breaks on an upgrade.

| Deprecated | Use | Why that name |
| --- | --- | --- |
| `K0` | `k0` | `scipy.special.k0` |
| `K1` | `k1` | `scipy.special.k1` |
| `K2` | `k2` | follows `k0`/`k1`; SciPy uses `kn(2, x)` |
| `K0e` | `k0e` | `scipy.special.k0e` |
| `K1e` | `k1e` | `scipy.special.k1e` |
| `K2e` | `k2e` | follows `k0e`/`k1e`; SciPy uses `kve(2, x)` |
| `Li` | `polylog` | matches `mpmath.polylog`. **Not** `li`, which is the logarithmic integral -- a different function |

## Signatures

- **Argument order follows SciPy.** Degree/order parameters come first, then function parameters, then the point of evaluation -- e.g. `eval_gegenbauer(n, alpha, x)`.
- **Integer-valued structural parameters (degrees, orders) are Python `int`s** and are static under `jit`. Continuous parameters and evaluation points are arrays.
- **Inputs and outputs are `jax.Array`.** Shapes are annotated with [jaxtyping](https://docs.kidger.site/jaxtyping/).

## Behaviour

- Functions are pure and JAX-transformable: `jit`, `vmap`, `grad`.
- Out-of-domain _array_ inputs return `nan`/`inf` rather than raising -- traced code cannot raise. A static Python parameter can still be validated eagerly, and is: `polylog` raises `ValueError` for a non-integer or non-positive order. See [About domain edges](../explanation/edges.md).
- Double precision is assumed. Accuracy claims hold with `jax_enable_x64`.

## Docstrings

NumPy-style, with an `Examples` section. The conventions contributors are held to are in [Contributing](../about/contributing.md).
