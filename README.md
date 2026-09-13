<h1 align='center'> spexial </h1>
<h3 align="center">`scipy.special` in JAX</h3>

<br>

## Installation

[![PyPI version][pypi-version]][pypi-link]
[![PyPI platforms][pypi-platforms]][pypi-link]

```bash
pip install spexial
```

or

```bash
uv add spexial
```

## Supported functions

`spexial` implements a subset of `scipy.special`, in JAX:

| Function                        | Description                                                                    |
| ------------------------------- | ------------------------------------------------------------------------------ |
| `comb(N, k)`                    | Number of combinations of `N` things taken `k` at a time.                      |
| `gamma(x)`                      | Gamma function; supports complex arguments (unlike `jax.scipy.special.gamma`). |
| `K0(z)`, `K1(z)`, `K2(z)`       | Modified Bessel functions of the second kind, orders 0-2.                      |
| `zeta(n)`                       | Riemann/Hurwitz zeta function; supports negative arguments.                    |
| `eval_gegenbauer(n, alpha, x)`  | Gegenbauer polynomial evaluated at a point.                                    |
| `eval_gegenbauers(n, alpha, x)` | Gegenbauer polynomial and all lower-order polynomials at a point.\*            |
| `Li(n, z)`                      | Polylogarithm of order `n`.\*                                                  |

\* Not available in `scipy.special`.

```python
import spexial as sx

sx.gamma(5.0)
```

## Development

[![codecov][codecov-badge]][codecov-link]
[![Actions Status][actions-badge]][actions-link]

We welcome contributions!

<!-- prettier-ignore-start -->
[actions-badge]:            https://github.com/JAXtronomy/spexial/workflows/CI/badge.svg
[actions-link]:             https://github.com/JAXtronomy/spexial/actions
[codecov-badge]:            https://codecov.io/gh/JAXtronomy/spexial/graph/badge.svg
[codecov-link]:             https://codecov.io/gh/JAXtronomy/spexial
[pypi-link]:                https://pypi.org/project/spexial/
[pypi-platforms]:           https://img.shields.io/pypi/pyversions/spexial
[pypi-version]:             https://img.shields.io/pypi/v/spexial

<!-- prettier-ignore-end -->
