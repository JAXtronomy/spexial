# Accuracy and domains

The supported input domain and measured accuracy of every public function. This page is the canonical home for these facts; other pages link here rather than restating them.

Every function is tested against a reference implementation — `scipy.special` where a counterpart exists, `mpmath` where it does not. All entries assume double precision ([how to enable it](../how-to/enable-double-precision.md)).

## Functions

| Function | `scipy.special` | Supported domain | Tested to |
| --- | --- | --- | --- |
| `eval_gegenbauer` | `eval_gegenbauer` | $n \le 20$ integer, $\alpha > -1/2$, $\lvert x \rvert \le 1$ | rtol $10^{-10}$, atol $10^{-9}$; worst measured atol $1.9 \times 10^{-12}$ |
| `eval_gegenbauers` | -- | as `eval_gegenbauer`; returns all orders $0 \ldots n$ | as `eval_gegenbauer` |
| `comb` | `comb` (`exact=False`) | $0 \le N, k \le 170$ | rtol $10^{-11}$; worst $3.2 \times 10^{-13}$ |
| `gamma` | `gamma` | real $x$, $-170 \lesssim x \lesssim 171$ | rtol $10^{-10}$ for $x \ge -30$, $5 \times 10^{-10}$ below; worst $2.9 \times 10^{-13}$ for $x \ge 1/2$, $2.4 \times 10^{-10}$ on $[-170, -30]$ |
| `K0` | `k0` | $0 < z \lesssim 705$ | rtol $10^{-6}$; worst $8 \times 10^{-8}$ at $z = 9$, ~$10^{-16}$ elsewhere |
| `K1` | `k1` | $0 < z \lesssim 700$ | as `K0` |
| `K2` | `kn` ($n = 2$) | $0 < z \lesssim 705$ | as `K0` |
| `Li` | -- (`mpmath.polylog`) | scalar $z$, integer $n \ge 1$ | rtol $10^{-11}$, atol $10^{-12}$; worst $3.4 \times 10^{-12}$ for $1 \le n \le 20$, $\lvert z \rvert \le 1000$ |
| `spence` | `spence` | real or complex $z$ | rtol $10^{-5}$, atol $10^{-8}$ vs scipy; the derivative is exact (closed form) |
| `zeta` | `zeta` | $1 < n \lesssim 10^{15}$, or $n$ a negative integer $> -60$ | rtol $10^{-12}$; worst $8.9 \times 10^{-16}$ |

## Per-function limits

`eval_gegenbauer`, `eval_gegenbauers` : At exactly $\alpha = 0$, `spexial` returns $C_0^{(0)} = 1$ and $C_n^{(0)} = 0$ for $n \ge 1$, which is what the generating function gives. SciPy agrees up to 1.14; from 1.18 it returns `0.0` for _every_ order at exactly $\alpha = 0$, while still returning `1.0` at $\alpha = 10^{-300}$. The parity suite excludes that single point rather than follow it.

`comb` : Returns `0` for `k > N`, `k < 0` and `N < 0`, matching `scipy.special.comb`. The inexact variant only; there is no `exact=True` path.

`gamma` : Real input only; complex input is rejected by the type annotation. Returns `inf` at the non-positive integers. SciPy is not a stable reference here: it returns `inf` everywhere up to 1.14, but from 1.18 returns `nan` at the negative integers while still returning `inf` at 0, matching C99 `tgamma`. Relative error near a pole is approximately $\lvert x \rvert \times 10^{-16}/\delta$, where $\delta$ is the distance to it -- the argument error in $\sin(\pi x)$ grows with $\lvert x \rvert$, so the loss is worse at the deeper poles: $4.3 \times 10^{-8}$ at $x = -7 - 10^{-8}$ but $5.4 \times 10^{-7}$ at $x = -55 - 10^{-8}$. The parity suite keeps $10^{-4}$ clear of the poles and pins near-pole behaviour at several poles separately. Below $x \approx -170.6$ the true value is subnormal and XLA on CPU flushes it to zero, so `spexial` returns `0` where SciPy returns a denormal. `gamma(+inf)` is `inf`, matching SciPy; `gamma(-inf)` is `nan` -- Gamma has a pole at every negative integer, so the limit does not exist, and SciPy's `-inf` is not something to copy.

`K0`, `K1`, `K2` : A 30-term ascending series below $z = 9$ and a 10-term asymptotic expansion above it. Worst relative error is $8 \times 10^{-8}$, at the cross-over; away from it all three are accurate to ~$10^{-16}$. All three underflow to `0` near $z = 706$, bounded by `K0`. SciPy is not uniform here: `scipy.special.kn` underflows at $z \approx 698$, while `k0` and `k1` keep returning denormals out to $z \approx 745$ -- so beyond ~706 `spexial` returns `0` where `k0`/`k1` still have a value. The parity suite therefore stops at $z = 690$, below every one of those cliffs, and references each order against the function named in the table above.

`Li` : Accepts a scalar `z` only. An array argument raises a broadcasting `TypeError`; use `jax.vmap` ([how](../how-to/use-with-jit-vmap-and-grad.md)). Raises `ValueError` for a non-integer order or an order below 1. For $\lvert z \rvert \ge 2$ the order is capped at **60** by the Bernoulli table the inversion formula needs; past that the result is `nan`. Smaller $\lvert z \rvert$ is unaffected, bounded instead by $\Gamma(n+1)$ overflow above $n \approx 170$. `Li(1, 1)` is the pole and returns `inf`.

`zeta` : Bernoulli numbers are computed from exact `fractions.Fraction` arithmetic, not `jax.scipy.special.bernoulli`. Above $n \approx 10^{15}$ the result is `nan`, and that comes from `jax.scipy.special.zeta` rather than from this package; SciPy returns `1.0`. `jax.grad(zeta)` is meaningful only for $n > 1$; on the negative line it returns a finite value that is not $\zeta'$.

`spence` : Accepts real _and_ complex argument; `jax.scipy.special.spence` is real-only and raises on complex. Its derivative, $\log z/(1-z)$, is supplied analytically — which matters beyond speed: JAX's own `spence` differentiates to `nan` across roughly $1 < z < 2$, where `spexial` is exact. Translated from SciPy's Cython implementation.

### `zeta` coverage against SciPy

| Input | `spexial` | `scipy.special.zeta` |
| --- | --- | --- |
| $n > 1$ | accurate | accurate |
| $0 < n \le 1$ (the critical strip) | `nan` | accurate |
| negative integer $> -60$ | exact | accurate |
| negative even integer, to $\sim 9 \times 10^{18}$ | exactly `0` | `0` |
| negative even integer, beyond int64 | `nan` | `0` |
| negative odd integer $\le -60$ | `nan` | accurate |
| negative non-integer | `nan` | accurate |

## Reference implementations

| Reference       | Used for                                 |
| --------------- | ---------------------------------------- |
| `scipy.special` | Every function with a SciPy counterpart. |
| `mpmath`        | `Li`, which has no SciPy counterpart.    |

## Outside the supported domain

Traced JAX code cannot raise. A function called outside its domain returns `nan` or `inf`, and that value propagates through `jit`, `vmap` and `grad`. Validate inputs before the call, or test the output with `jnp.isnan`. For why the library behaves this way, see [About domain edges](../explanation/edges.md).
