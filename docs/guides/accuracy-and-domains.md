# Accuracy and domains

Every `spexial` function is tested against a reference implementation — `scipy.special` where a counterpart exists, `mpmath` where it does not. This page records, per function, which reference it is checked against, over which input domain it is known to be accurate, and anything a caller should know before trusting it outside that domain.

## How to read the table

- **Function** — the `spexial` name, as exported from the top-level namespace.
- **`scipy.special`** — the counterpart whose name, argument order and convention `spexial` follows. `--` means the function has no SciPy counterpart.
- **Supported domain** — the input range over which the implementation is tested and expected to be accurate. Outside it, the result may be inaccurate, `nan`, or `inf` without warning: JAX does not raise on domain errors inside `jit`.
- **Tested to** — the tolerance the parity suite actually asserts, and the worst error measured over the supported domain. Where these differ substantially, the tolerance carries headroom; where they are close, the tolerance is the honest limit of the algorithm.

All entries assume double precision (`jax_enable_x64`); see [Sharp bits](sharp-bits.md).

## Functions

| Function | `scipy.special` | Supported domain | Tested to |
| --- | --- | --- | --- |
| `eval_gegenbauer` | `eval_gegenbauer` | $n \le 20$ integer, $\alpha > -1/2$, $\lvert x \rvert \le 1$ | rtol $10^{-10}$, atol $10^{-9}$; worst measured atol $1.9 \times 10^{-12}$ near polynomial roots |
| `eval_gegenbauers` | -- | as `eval_gegenbauer`; returns all orders $0 \ldots n$ | as above |
| `comb` | `comb` | $0 \le N, k \le 170$ | rtol $10^{-11}$; worst $3.2 \times 10^{-13}$ |
| `gamma` | `gamma` | real $x$, $\lvert x \rvert \lesssim 171$ | rtol $10^{-10}$; worst $1.8 \times 10^{-14}$ for $x \ge 1/2$ |
| `K0` | `k0` | $0 < z \lesssim 710$ | **rtol $10^{-6}$**; worst $8 \times 10^{-8}$ at the $z = 9$ cross-over, ~$10^{-10}$ elsewhere |
| `K1` | `k1` | $0 < z \lesssim 700$ | as `K0` |
| `K2` | `kn` ($n = 2$) | $0 < z \lesssim 710$ | as `K0` |
| `Li` | -- (`mpmath.polylog`) | **scalar** $z$, integer $n \ge 1$ | rtol $10^{-11}$, atol $10^{-12}$; worst $5.7 \times 10^{-13}$ over $1 \le n \le 20$, $\lvert z \rvert \le 1000$ |
| `zeta` | `zeta` | $n > 1$, or $n$ a negative integer $> -60$ | rtol $10^{-12}$; worst $6.7 \times 10^{-16}$ |

## Caveats worth reading before you rely on a function

### `gamma` is real-only, and loses precision near the poles

Complex input is rejected rather than silently mishandled: the reflection formula branches on $x < 1/2$, a test complex input cannot supply. `jax.scipy.special.gamma` is also real-only; use it if you do not need negative arguments.

Near a non-positive integer the reflection formula $\Gamma(x) = \pi / \left(\sin(\pi x)\,\Gamma(1-x)\right)$ loses precision in proportion to the distance from the pole — roughly $10^{-17}/\delta$. At $\delta = 10^{-4}$ the relative error is $6 \times 10^{-12}$; at $\delta = 10^{-8}$ it is $1.5 \times 10^{-9}$. This is inherent to the formula, not a fixable defect. The parity suite keeps $10^{-4}$ clear of the poles and pins the near-pole behaviour separately at rtol $10^{-7}$.

At the poles themselves the result is `inf`, matching SciPy.

### The modified Bessel functions are accurate to ~$10^{-7}$, not machine precision

`K0`/`K1`/`K2` use a 30-term ascending series below $z = 9$ and a 10-term asymptotic expansion above it. The two meet at $z = 9$ with a worst-case relative error of $8 \times 10^{-8}$. Away from the cross-over the error is around $10^{-10}$. If you need full double precision from a modified Bessel function, this implementation will not give it to you.

`K1` underflows to zero for $z \gtrsim 700$; SciPy does the same.

### `Li` takes scalar `z` only

The intermediate branch builds a length-60 vector of powers of $\log z$, so it cannot broadcast over an array argument. The type annotation enforces this. Use `jax.vmap` to map it over an array — this is tested and supported:

```python
import jax
import jax.numpy as jnp

import spexial as sp

zs = jnp.array([0.25, 0.5, 1.5])
print(jax.vmap(lambda z: sp.Li(2, z))(zs))
```

`n` must be an integer $\ge 1$; `Li` raises `ValueError` otherwise. Large `n` is bounded by $\Gamma(n+1)$ overflow above $n \approx 170$.

### `zeta` has real gaps relative to SciPy

These are genuine gaps, not deliberate restrictions:

| Input | `spexial` | `scipy.special.zeta` |
| --- | --- | --- |
| $n > 1$ | accurate | accurate |
| $0 < n \le 1$ (the critical strip) | `nan` | accurate |
| negative integer $> -60$ | exact | accurate |
| negative even integer, any magnitude | exactly `0` | `0` |
| negative odd integer $\le -60$ | `nan` (Bernoulli table ends at $B_{60}$) | accurate |
| negative non-integer | `nan` | accurate |

The negative line is evaluated from a Bernoulli table via the functional equation, so `jax.grad(zeta)` **is only meaningful for $n > 1$**. On the negative line the derivative it reports is finite but is not $\zeta'$; a table lookup carries no derivative information. Do not use it there.

The Bernoulli numbers are computed from exact `fractions.Fraction` arithmetic rather than `jax.scipy.special.bernoulli`, which loses roughly seven digits on $B_4$ and six on $B_6$ — enough to put a $10^{-6}$ error into `Li(4, ·)`, `Li(5, ·)` and `zeta(-3)`.

## Reference implementations

| Reference       | Used for                                 |
| --------------- | ---------------------------------------- |
| `scipy.special` | Every function with a SciPy counterpart. |
| `mpmath`        | `Li`, which has no SciPy counterpart.    |

## Outside the supported domain

JAX has no exceptions inside traced code. A function called outside its domain returns `nan` or `inf` rather than raising, and the failure propagates silently through `jit`, `vmap` and `grad`. If your inputs may leave the supported domain, check them yourself before the call, or check the output for `nan` afterwards.
