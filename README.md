<h1 align='center'> spexial </h1>
<h3 align="center"><code>scipy.special</code> in JAX</h3>

<p align="center">
<a href="https://pypi.org/project/spexial/"><img alt="PyPI version" src="https://img.shields.io/pypi/v/spexial"></a>
<a href="https://pypi.org/project/spexial/"><img alt="PyPI platforms" src="https://img.shields.io/pypi/pyversions/spexial"></a>
<a href="https://github.com/JAXtronomy/spexial/actions"><img alt="Actions Status" src="https://github.com/JAXtronomy/spexial/workflows/CI/badge.svg"></a>
<a href="https://codecov.io/gh/JAXtronomy/spexial"><img alt="codecov" src="https://codecov.io/gh/JAXtronomy/spexial/graph/badge.svg"></a>
<a href="https://jaxtronomy.github.io/spexial"><img alt="Documentation" src="https://img.shields.io/badge/docs-jaxtronomy.github.io-blue"></a>
</p>

`spexial` provides special functions for JAX, following the `scipy.special` API. The implementations are written in terms of JAX primitives, so they compose with `jit`, `grad` and `vmap`, and run on CPU, GPU and TPU.

It exists to fill gaps in `jax.scipy.special`: `zeta` accepts negative integers where JAX returns `nan`, `spence` accepts complex arguments where JAX is real-only, and the modified Bessel `K`, Gegenbauer and polylogarithm functions have no JAX counterpart at any version. Where SciPy already covers a case, `spexial` matches it rather than claiming to exceed it.

## Installation

```bash
pip install spexial
```

or

```bash
uv add spexial
```

## Example

```pycon
>>> import jax
>>> jax.config.update("jax_enable_x64", True)

>>> import jax.numpy as jnp
>>> import spexial as sp

>>> # Gegenbauer polynomial C_n^alpha(x), matching scipy.special.eval_gegenbauer
>>> sp.eval_gegenbauer(3, 0.5, 0.25)
Array(-0.3359375, dtype=float64, weak_type=True)

```

Everything is vectorisable and differentiable in the usual way:

```pycon
>>> xs = jnp.linspace(-1.0, 1.0, 5)
>>> jax.vmap(lambda x: sp.eval_gegenbauer(3, 0.5, x))(xs)
Array([-1.    ,  0.4375, -0.    , -0.4375,  1.    ], dtype=float64)

```

## What is here

| Function            | `scipy.special` counterpart                           |
| ------------------- | ----------------------------------------------------- |
| `comb`              | `comb` (the `exact=False` variant)                    |
| `gamma`             | `gamma` — JAX's value, plus an analytic derivative    |
| `eval_gegenbauer`   | `eval_gegenbauer`                                     |
| `eval_gegenbauers`  | -- returns every order up to `n`                      |
| `k0`, `k1`, `k2`    | `k0`, `k1`, `kn`                                      |
| `k0e`, `k1e`, `k2e` | `k0e`, `k1e`, `kve` — scaled by `e^z`, no upper limit |
| `polylog`           | -- the polylogarithm                                  |
| `spence`            | `spence` — complex too, which JAX rejects             |
| `zeta`              | `zeta` — negative integers, which JAX gives as `nan`  |

**Read [Accuracy and domains](https://jaxtronomy.github.io/spexial/reference/accuracy-and-domains/) before relying on any of these.** It records, per function, the domain each is tested over and the tolerance it actually meets. Some are not machine-precision — the modified Bessel functions are accurate to about `1e-7`, and `zeta` does not implement the critical strip.

## Documentation

<https://jaxtronomy.github.io/spexial>

## Development

We welcome contributions! See [CONTRIBUTING.md](CONTRIBUTING.md).

```bash
uv sync --group dev
uv run nox -s all      # lint -> test -> docs
```

## Citation

If you use `spexial` in work you publish, please cite it — see [CITATION.cff](CITATION.cff). Several routines originate in the [LINX](https://github.com/cgiovanetti/LINX) code.

## License

MIT. See [LICENSE](LICENSE).
