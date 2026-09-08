# Accuracy and domains

The supported input domain and measured accuracy of every public function. This page is the canonical home for these facts; other pages link here rather than restating them.

Every function is tested against a reference implementation — `scipy.special` where a counterpart exists, `mpmath` where it does not. All entries assume double precision ([how to enable it](../how-to/enable-double-precision.md)).

## Functions

| Function | `scipy.special` | Supported domain | Tested to |
| --- | --- | --- | --- |
| `eval_gegenbauer` | `eval_gegenbauer` | $n \le 20$ integer, $\alpha > -1/2$, $\lvert x \rvert \le 1$ | rtol $10^{-10}$, atol $10^{-9}$; worst measured atol $1.9 \times 10^{-12}$ |
| `eval_gegenbauers` | -- | as `eval_gegenbauer`; returns all orders $0 \ldots n$ | as `eval_gegenbauer` |
| `comb` | `comb` (`exact=False`) | $0 \le N, k \le 170$ | rtol $10^{-11}$; worst $3.2 \times 10^{-13}$ |
| `gamma` | `gamma` | real $x$, $\lvert x \rvert \lesssim 171$ | rtol $10^{-10}$; worst $1.8 \times 10^{-14}$ for $x \ge 1/2$ |
| `K0` | `k0` | $0 < z \lesssim 710$ | rtol $10^{-6}$; worst $8 \times 10^{-8}$ at $z = 9$, ~$10^{-10}$ elsewhere |
| `K1` | `k1` | $0 < z \lesssim 700$ | as `K0` |
| `K2` | `kn` ($n = 2$) | $0 < z \lesssim 710$ | as `K0` |
| `Li` | -- (`mpmath.polylog`) | scalar $z$, integer $n \ge 1$ | rtol $10^{-11}$, atol $10^{-12}$; worst $5.7 \times 10^{-13}$ for $1 \le n \le 20$, $\lvert z \rvert \le 1000$ |
| `zeta` | `zeta` | $n > 1$, or $n$ a negative integer $> -60$ | rtol $10^{-12}$; worst $6.7 \times 10^{-16}$ |

## Per-function limits

`comb` : Returns `0` for `k > N`, `k < 0` and `N < 0`, matching `scipy.special.comb`. The inexact variant only; there is no `exact=True` path.

`gamma` : Real input only; complex input is rejected by the type annotation. Returns `inf` at the non-positive integers, matching SciPy. Relative error near a pole is approximately $10^{-17}/\delta$, where $\delta$ is the distance to it: $6 \times 10^{-12}$ at $\delta = 10^{-4}$, $1.5 \times 10^{-9}$ at $\delta = 10^{-8}$. The parity suite keeps $10^{-4}$ clear of the poles and pins near-pole behaviour separately at rtol $10^{-7}$.

`K0`, `K1`, `K2` : A 30-term ascending series below $z = 9$ and a 10-term asymptotic expansion above it. Worst relative error is $8 \times 10^{-8}$, at the cross-over. `K1` underflows to `0` for $z \gtrsim 700$; SciPy does the same.

`Li` : Accepts a scalar `z` only. An array argument raises a broadcasting `TypeError`; use `jax.vmap` ([how](../how-to/use-with-jit-vmap-and-grad.md)). Raises `ValueError` for a non-integer order or an order below 1. Large `n` is bounded by $\Gamma(n+1)$ overflow above $n \approx 170$.

`zeta` : Bernoulli numbers are computed from exact `fractions.Fraction` arithmetic, not `jax.scipy.special.bernoulli`. `jax.grad(zeta)` is meaningful only for $n > 1$; on the negative line it returns a finite value that is not $\zeta'$.

### `zeta` coverage against SciPy

| Input                                | `spexial`   | `scipy.special.zeta` |
| ------------------------------------ | ----------- | -------------------- |
| $n > 1$                              | accurate    | accurate             |
| $0 < n \le 1$ (the critical strip)   | `nan`       | accurate             |
| negative integer $> -60$             | exact       | accurate             |
| negative even integer, any magnitude | exactly `0` | `0`                  |
| negative odd integer $\le -60$       | `nan`       | accurate             |
| negative non-integer                 | `nan`       | accurate             |

## Reference implementations

| Reference       | Used for                                 |
| --------------- | ---------------------------------------- |
| `scipy.special` | Every function with a SciPy counterpart. |
| `mpmath`        | `Li`, which has no SciPy counterpart.    |

## Outside the supported domain

Traced JAX code cannot raise. A function called outside its domain returns `nan` or `inf`, and that value propagates through `jit`, `vmap` and `grad`. Validate inputs before the call, or test the output with `jnp.isnan`. For why the library behaves this way, see [About domain edges](../explanation/edges.md).
