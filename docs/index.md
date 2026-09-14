# spexial

`scipy.special` in [JAX](https://docs.jax.dev).

`spexial` implements special functions — Gegenbauer polynomials, the gamma function, modified Bessel functions, the Riemann zeta function, polylogarithms, the incomplete beta function, spherical harmonics — as plain JAX functions. Because they are ordinary JAX code, they compose with `jax.jit`, `jax.vmap` and `jax.grad`, and run on CPU, GPU and TPU.

Where a function has a `scipy.special` counterpart, `spexial` matches its name, its argument order and its convention. It exists to fill gaps in `jax.scipy.special`, of three kinds: functions JAX does not have at any version (the modified Bessel `K` functions, the polylogarithm, the Gegenbauer polynomials, the unregularized incomplete beta, the Cartesian spherical harmonics); functions JAX has but gets wrong or supports over less (`zeta` returns `nan` for negative integers, `spence` rejects complex arguments, `sph_harm_y` returns incorrect values for array degrees); and functions whose _derivative_ is the problem, where writing the JVP out by hand is faster, cheaper in memory, or finite where differentiating the implementation gives `nan`.

SciPy covers most of these already; the gap this library fills is on the JAX side, not the SciPy side. [Coverage](reference/coverage.md) states, per function, which gap it fills — and what would have to happen upstream for the row to be deleted.

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
