---
name: spexial
description: Use when writing JAX code that needs special functions - gamma, zeta, polylog, modified Bessel (K0/K1/K2), Gegenbauer polynomials, or binomial coefficients - or when jax.scipy.special lacks a function or a domain you need. Covers which spexial function to reach for, where it goes beyond scipy.special, and the domains where it returns nan or loses precision.
---

# Using `spexial`

`spexial` is `scipy.special` for JAX: JAX-primitive implementations that compose with `jit`, `grad` and `vmap` and run on CPU/GPU/TPU. Names and argument orders follow `scipy.special`.

```python
import jax

jax.config.update("jax_enable_x64", True)  # do this first, see below

import spexial as sp
```

## Which library to reach for

1. **`jax.scipy.special` first.** If it has what you need over the domain you need, use it — it is maintained upstream and generally faster.
2. **`spexial` when `jax.scipy.special` has a gap.** Its `gamma` is real-positive-ish only; `spexial.gamma` handles negative reals. Its `zeta` does not do negative arguments; `spexial.zeta` does. There is no polylogarithm, no `K2`, no Gegenbauer.
3. **`scipy.special` when you are not in JAX.** `spexial` buys you nothing outside a traced context, and is less accurate for the Bessel functions.

## What is available

| `spexial` | `scipy.special` | Notes |
| --- | --- | --- |
| `comb(N, k)` | `comb` | the `exact=False` variant, via `gammaln` |
| `gamma(x)` | `gamma` | **real only**; handles negative `x` |
| `eval_gegenbauer(n, alpha, x)` | `eval_gegenbauer` | `n` is a static integer |
| `eval_gegenbauers(n, alpha, x)` | -- | all orders `0..n` at once |
| `K0(z)`, `K1(z)`, `K2(z)` | `k0`, `k1`, `kn` | modified Bessel, 2nd kind |
| `Li(n, z)` | -- | polylogarithm; **scalar `z` only** |
| `zeta(n)` | `zeta` | handles negative integers |

## Enable x64 before anything else

```python
import jax

jax.config.update("jax_enable_x64", True)
```

These are series and asymptotic expansions. In float32 the accuracy figures below are meaningless — you will lose most of your digits to the accumulation, not the algorithm. Set this at program start, before any array is created.

## The traps

**`nan` does not raise.** JAX has no exceptions inside traced code. Every domain violation below returns `nan` or `inf` and propagates silently through `jit`, `vmap` and `grad`. Check your inputs before the call, or check the output after.

**`zeta` has real gaps.** `nan` on the critical strip (`0 < n <= 1`), for negative non-integers, and for odd `n <= -60`. SciPy handles all three; `spexial` does not.

**`jax.grad(zeta)` is only meaningful for `n > 1`.** The negative line is evaluated from a Bernoulli table, so the derivative reported there is finite but is not `ζ'`. It will not warn you.

**`gamma` is real-only and imprecise near the poles.** Complex input is rejected. Near a non-positive integer the relative error is roughly `1e-17 / distance-to-pole` — about `1e-9` at a distance of `1e-8`. Poles return `inf`, like SciPy.

**`Li` takes scalar `z` only.** It cannot broadcast. Use `jax.vmap`:

```python
import jax
import jax.numpy as jnp

import spexial as sp

zs = jnp.array([0.25, 0.5, 1.5])
result = jax.vmap(lambda z: sp.Li(2, z))(zs)
```

`n` must be an integer `>= 1`, and is bounded above by gamma overflow around 170.

**`K0`/`K1`/`K2` are accurate to ~`1e-7`, not machine precision.** A 30-term ascending series meets a 10-term asymptotic expansion at `z = 9`; the worst relative error is `8e-8`, right at that cross-over. Away from it, ~`1e-10`. If you need full double precision from a modified Bessel function, this is not it. `K1` underflows to 0 above `z ~ 700`.

**`n` in the Gegenbauer functions is static.** It is a Python integer baked into the trace, not a traced value — a different `n` triggers recompilation. Do not `vmap` over it.

## Accuracy summary

Everything below assumes x64. Full detail, including how each was measured, is at <https://jaxtronomy.github.io/spexial/reference/accuracy-and-domains/>.

| Function          | Domain                                  | Accurate to  |
| ----------------- | --------------------------------------- | ------------ |
| `comb`            | `0 <= N, k <= 170`                      | `3e-13`      |
| `gamma`           | real, `\|x\| < 171`, away from poles    | `2e-14`      |
| `eval_gegenbauer` | `n <= 20`, `alpha > -0.5`, `\|x\| <= 1` | `2e-12` atol |
| `K0`/`K1`/`K2`    | `0 < z < 700`                           | `8e-8`       |
| `Li`              | scalar `z`, `n >= 1`                    | `6e-13`      |
| `zeta`            | `n > 1`, or negative integer `> -60`    | `7e-16`      |
