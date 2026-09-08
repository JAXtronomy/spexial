# spexial

`scipy.special` in [JAX](https://docs.jax.dev).

`spexial` implements special functions — Gegenbauer polynomials, the gamma function, modified Bessel functions, the Riemann zeta function, polylogarithms — as plain JAX functions. Because they are ordinary JAX code, they compose with `jax.jit`, `jax.vmap` and `jax.grad`, and run on CPU, GPU and TPU.

Where a function has a `scipy.special` counterpart, `spexial` matches its name, its argument order and its convention. It exists to fill gaps in `jax.scipy.special`, and in a couple of places to go further than SciPy: `gamma` accepts negative reals and `zeta` accepts negative integers.

## Installation

```bash
pip install spexial
```

## Where to start

<div class="grid cards" markdown>

- **[Tutorial](tutorials/index.md)**

  New here? Compute a physical constant from two special functions, and learn the shape of the library along the way.

- **[How-to guides](how-to/index.md)**

  Know what you want to do? Enable double precision, use the JAX transforms, port from SciPy.

- **[Reference](reference/index.md)**

  Every function, its supported domain, and the accuracy it actually delivers.

- **[Explanation](explanation/index.md)**

  Why double precision is mandatory, why bad input returns `nan`, and why one gradient lies.

</div>

## Before you rely on it

Some routines are less accurate than their SciPy counterparts, and `zeta` does not cover every argument SciPy does. [Accuracy and domains](reference/accuracy-and-domains.md) states, per function, the domain it is tested over and the tolerance it meets. Read it before swapping an import in work you care about.

## About

- [Contributing](about/contributing.md)
- [Source on GitHub](https://github.com/JAXtronomy/spexial)
