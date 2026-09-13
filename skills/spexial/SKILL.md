---
name: spexial
description: Use when writing JAX code that needs special functions - gamma, zeta, polylog, modified Bessel (k0/k1/k2), Gegenbauer polynomials, or binomial coefficients - or when jax.scipy.special lacks a function or a domain you need. Covers which spexial function to reach for, where it goes beyond scipy.special, and the domains where it returns nan or loses precision.
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
2. **`spexial` when `jax.scipy.special` has a gap.** It has no modified Bessel `K`, no Gegenbauer and no general polylogarithm at any version. Its `zeta` does not take negative arguments; `spexial.zeta` does. Its `spence` is real-only and its gradient is `nan` across roughly `1 < x < 2`. For `gamma` and `spence` the values come _from_ JAX — `spexial` wraps them only to supply a cheaper analytic derivative. The saving is mostly **memory**: 3x less residual for `gamma` (at parity on time) and 38x less for `spence` (which is also 5x faster).
3. **`scipy.special` when you are not in JAX.** `spexial` buys you nothing outside a traced context, and is less accurate for the Bessel functions.

## What is available

| `spexial` | `scipy.special` | Notes |
| --- | --- | --- |
| `comb(N, k)` | `comb` | the `exact=False` variant, via `gammaln` |
| `gamma(x)` | `gamma` | JAX's value, plus an analytic derivative; complex on jax >= 0.10.2 |
| `eval_gegenbauer(n, alpha, x)` | `eval_gegenbauer` | `n` is a static integer |
| `eval_gegenbauers(n, alpha, x)` | -- | all orders `0..n` at once |
| `k0(z)`, `k1(z)`, `k2(z)` | `k0`, `k1`, `kn` | modified Bessel, 2nd kind |
| `k0e(z)`, `k1e(z)`, `k2e(z)` | `k0e`, `k1e`, `kve` | the same, scaled by `e^z`; **the only ones that work past `z = 705`** |
| `polylog(n, z)` | -- | polylogarithm; **scalar `z` only** |
| `zeta(n)` | `zeta` | handles negative integers |

## Enable x64 before anything else

```python
import jax

jax.config.update("jax_enable_x64", True)
```

These are series and asymptotic expansions. In float32 the accuracy figures below are meaningless — you will lose most of your digits to the accumulation, not the algorithm. Set this at program start, before any array is created.

## The traps

**`nan` does not raise.** JAX has no exceptions inside traced code. Every domain violation below returns `nan` or `inf` and propagates silently through `jit`, `vmap` and `grad`. Check your inputs before the call, or check the output after.

**`zeta` covers every real argument**, by four methods stitched together (see the docs table). Worst accuracy `6e-13`, at large `|n|`; past `n ~ -260.2` the true value exceeds `DBL_MAX` and the answer is `±inf`, as in SciPy. `jax.grad` is an artefact only at the tabulated integers `0 >= n >= -59`, at the negative even integers, and at `n >= 54`.

**`jax.grad(zeta)` is only meaningful for `n > 1`.** The negative line is evaluated from a Bernoulli table, so the derivative reported there is finite but is not `ζ'`. It will not warn you.

**`gamma` returns `nan` at the negative integers.** It delegates to `jax.scipy.special.gamma`, so `x = 0` gives `inf` but every negative integer gives `nan` — the two-sided limit does not exist, and this matches JAX and scipy from 1.18. Complex input works on jax >= 0.10.2. Accuracy is ~`4e-13` throughout, including close to the poles.

**`polylog` takes scalar `z` only.** It cannot broadcast. Use `jax.vmap`:

```python
import jax
import jax.numpy as jnp

import spexial as sp

zs = jnp.array([0.25, 0.5, 1.5])
result = jax.vmap(lambda z: sp.polylog(2, z))(zs)
```

`n` must be an integer `>= 1`. For `|z| >= 2` it is capped at **60** by the Bernoulli table the inversion formula needs, and the result is `nan` past that; smaller `|z|` has no ceiling at all (`polylog(500, 1.5)` is exact).

**`k0`/`k1`/`k2` are accurate to ~`1e-7`, not machine precision.** A 30-term ascending series meets a 10-term asymptotic expansion at `z = 9`; the worst relative error is `2.0e-7`, at `z = 8.9984` just below that cross-over. It improves in stages away from it: ~`8e-9` to `z = 15`, ~`1e-15` past `z = 30`. If you need full double precision from a modified Bessel function, this is not it.

**Above `z = 705.3`, use `k0e`/`k1e`/`k2e`.** The unscaled functions underflow to 0 there — the true value is smaller than any normal double, and XLA on CPU flushes it. That ceiling belongs to the **dtype**, not the function: it is `z = 85.3` in float32, `85.2` in bfloat16 and `16.2` in float16. The scaled `e^z K_n(z)` decays only as `1/sqrt(z)` and stays exact to `DBL_MAX`.

**In float32 the cross-over moves to `z = 4.65` and the worst error is `7.1e-3`** — about 2.5 digits. `float16`/`bfloat16` are computed in float32 and rounded back. Enable x64 if you need better.

**`n` in the Gegenbauer functions is static.** It is a Python integer baked into the trace, not a traced value — a different `n` triggers recompilation. Do not `vmap` over it.

## Accuracy summary

Everything below assumes x64. Full detail, including how each was measured, is at <https://jaxtronomy.github.io/spexial/reference/accuracy-and-domains/>.

| Function | Domain | Accurate to |
| --- | --- | --- |
| `comb` | `0 <= k <= N`, to `DBL_MAX` | `3.6e-12` |
| `gamma` | real or complex, `\|x\| < 171` | `4.3e-13` |
| `eval_gegenbauer` | `n <= 20`, `alpha > -0.5`, `\|x\| <= 1` | `2.1e-12` rtol; absolute error scales with the recurrence, up to `9e-5` at `n = 20, alpha = 10` |
| `k0`/`k1`/`k2` | `0 < z < 705.3` (float64; 85.3 in float32, 16.2 in float16) | `2.0e-7` |
| `k0e`/`k1e`/`k2e` | `z > 0`, no upper limit | `2.0e-7` |
| `polylog` | scalar `z`, `n >= 1` | `7.7e-12` |
| `zeta` | `n > 1`, or negative integer `> -60` | `7e-16` |
