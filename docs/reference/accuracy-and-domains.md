# Accuracy and domains

The supported input domain and measured accuracy of every public function. This page is the canonical home for these facts; other pages link here rather than restating them.

Every function is tested against a reference implementation — `scipy.special` where a counterpart exists, `mpmath` where it does not. All entries assume double precision ([how to enable it](../how-to/enable-double-precision.md)).

## Functions

| Function | `scipy.special` | Supported domain | Tested to |
| --- | --- | --- | --- |
| `eval_gegenbauer` | `eval_gegenbauer` | $n \le 20$ integer, $\alpha > -1/2$, $\lvert x \rvert \le 1$ | rtol $10^{-10}$, atol $10^{-11}$; worst measured atol $1.9 \times 10^{-12}$ |
| `eval_gegenbauers` | -- | as `eval_gegenbauer`; returns all orders $0 \ldots n$ | as `eval_gegenbauer` |
| `comb` | `comb` (`exact=False`) | $0 \le N, k \le 170$ | rtol $10^{-11}$; worst $3.2 \times 10^{-13}$ |
| `gamma` | `gamma` | real (and complex, jax $\ge$ 0.10.2), $-170 \lesssim x \lesssim 171$ | rtol $10^{-11}$; worst $3.5\times10^{-13}$ for $x \ge 1/2$, $4.3\times10^{-13}$ on $[-170,-30]$ |
| `K0` | `k0` | $0 < z \lesssim 705.5$ | rtol $10^{-6}$; worst $1.2 \times 10^{-7}$ just below $z = 9$, ~$10^{-15}$ above $z = 30$ |
| `K1` | `k1` | $0 < z \lesssim 705.5$ | as `K0`; worst $1.0 \times 10^{-7}$ |
| `K2` | `kn` ($n = 2$) | $0 < z \lesssim 705.5$ | as `K0`; worst $7.4 \times 10^{-8}$ |
| `K0e` | `k0e` | $0 < z \lesssim 9 \times 10^{307}$ | as `K0`; verified to $z = 10^5$ |
| `K1e` | `k1e` | $0 < z \lesssim 9 \times 10^{307}$ | as `K1`; verified to $z = 10^5$ |
| `K2e` | `kve` ($v = 2$) | $0 < z \lesssim 9 \times 10^{307}$ | as `K2`; verified to $z = 10^5$ |
| `Li` | -- (`mpmath.polylog`) | scalar $z$, integer $n \ge 1$ | rtol $10^{-11}$, atol $10^{-12}$; worst $3.4 \times 10^{-12}$ for $1 \le n \le 20$, $\lvert z \rvert \le 1000$ |
| `spence` | `spence` | real or complex $z$ | rtol $10^{-12}$, atol $10^{-13}$ vs scipy; worst $3.4 \times 10^{-14}$ |
| `zeta` | `zeta` | $n > 1$, or $n$ a negative integer $> -60$ | rtol $10^{-12}$; worst $8.9 \times 10^{-16}$ |

## Per-function limits

`eval_gegenbauer`, `eval_gegenbauers` : At exactly $\alpha = 0$, `spexial` returns $C_0^{(0)} = 1$ and $C_n^{(0)} = 0$ for $n \ge 1$, which is what the generating function gives. SciPy agrees up to 1.14; from 1.18 it returns `0.0` for _every_ order at exactly $\alpha = 0$, while still returning `1.0` at $\alpha = 10^{-300}$. The parity suite excludes that single point rather than follow it.

`comb` : Returns `0` for `k > N`, `k < 0` and `N < 0`, matching `scipy.special.comb`. `comb(inf, 0)` is `1`, `comb(inf, k >= 1)` is `inf` and `comb(inf, inf)` is `nan`, all as in SciPy. The inexact variant only; there is no `exact=True` path. The $N \le 170$ domain is a real ceiling, not a formality: the value comes from $\Gamma\ln(N+1) - \Gamma\ln(N-k+1)$, which cancels catastrophically for large $N$ — `comb(1e10, 2)` is already off by $3.7\times10^{-6}$, and `comb(1e300, 2)` returns `1.0` where SciPy returns `inf`.

`gamma` : Delegates the value to `jax.scipy.special.gamma`, so it cannot drift from upstream; `spexial` supplies only the derivative. Accepts real **and complex** input, the latter from jax 0.10.2 (below that JAX branches on `floor(x)` and raises). Returns `inf` at `x = 0` and `nan` at the negative integers, matching JAX and `scipy.special.gamma` from 1.18 — SciPy is not a stable reference at the poles, returning `inf` everywhere up to 1.14. There is no near-pole blow-up: measured error stays at $10^{-16}$–$8\times10^{-14}$ right up to $10^{-8}$ from a pole, where the previous hand-rolled Lanczos reflection lost precision as $\lvert x\rvert\times10^{-16}/\delta$. Below $x \approx -170.6$ the true value is subnormal and XLA on CPU flushes it to zero, so `spexial` returns `0` where SciPy returns a denormal. The **first** derivative is accurate to $1.6\times10^{-13}$ everywhere tested, but the **second** is not usable on the negative axis: $\Gamma''$ routes through `jax.scipy.special.digamma`'s own derivative, and JAX's trigamma diverges from the truth from about $x = -7.5$ (at $x = -10.5$ it is wrong by $10^{11}$ relative, and at $-20.5$ it has the wrong sign). This is upstream — `jax.grad(jax.grad(jax.scipy.special.gamma))` returns the identical wrong number — but `spexial` inherits it. `gamma(+inf)` is `inf`, matching SciPy; `gamma(-inf)` is `nan` — Gamma has a pole at every negative integer, so the limit does not exist, and SciPy's `-inf` is not something to copy.

`K0`, `K1`, `K2`, `K0e`, `K1e`, `K2e` : A 30-term ascending series below $z = 9$ and a 10-term asymptotic expansion above it. Worst relative error is $1.2 \times 10^{-7}$ for `K0`, $1.0 \times 10^{-7}$ for `K1` and $7.4 \times 10^{-8}$ for `K2`, all just below the cross-over. It falls off in stages rather than immediately: ~$8\times10^{-9}$ out to $z = 15$, ~$2\times10^{-13}$ to $z = 30$, and ~$10^{-15}$ beyond. (An earlier version of this page claimed ~$10^{-16}$ "away from" the cross-over, which only becomes true above $z \approx 30$.) All three underflow to `0` near $z = 705.5$, bounded by `K0` -- that is where the true value falls below the smallest _normal_ double and XLA on CPU flushes it to zero, the same floor `gamma` meets below $x = -170.6$; the series itself tracks `k0` bit-for-bit right up to it. SciPy is not uniform here: `scipy.special.kn` underflows at $z \approx 698$, while `k0` and `k1` keep returning denormals out to $z \approx 745$ -- so beyond ~705.5 `spexial` returns `0` where `k0`/`k1` still have a value. **`K0e`, `K1e` and `K2e` have no such ceiling** — $e^z K_n(z)$ decays only as $1/\sqrt z$, so they stay accurate to ~$10^{-15}$, verified against `scipy.special.k0e`/`k1e`/`kve` out to $z = 10^5$ and against `mpmath` to $z = 10^{300}$. That is what they are for. They do run out eventually, at $z \approx 9 \times 10^{307}$, where $z \cdot I_0^e(z)$ can no longer be doubled without overflowing; there is no float64 argument beyond that worth reaching.

In **float32** the cross-over moves to $z = 4.65$ rather than 9. The ascending series cancels two terms of size $e^z/\sqrt z$ down to a result of size $e^{-z}$, costing roughly $2z/\ln 10$ decimal digits — affordable with float64's 16, but float32 has 7 and at $z = 9$ had none left, returning a **negative** number. At 4.65 the worst float32 error over the whole domain is $4.5\times10^{-3}$. The parity suite therefore stops at $z = 690$, below every one of those cliffs, and references each order against the function named in the table above.

`Li` : Accepts a scalar `z` only. An array argument raises a broadcasting `TypeError`; use `jax.vmap` ([how](../how-to/use-with-jit-vmap-and-grad.md)). Raises `ValueError` for an order below 1, and for complex `z` — two of the three branches take the real part of a complex intermediate, which is exact for real `z` and would silently discard a genuine imaginary part. A non-integer order, including a whole-number `float` such as `Li(2.0, z)`, is a `TypeError` from the runtime type checker rather than a `ValueError`. For $\lvert z \rvert \ge 2$ the order is capped at **60** by the Bernoulli table the inversion formula needs; past that the result is `nan`. Smaller $\lvert z \rvert$ is unaffected, bounded instead by $\Gamma(n+1)$ overflow above $n \approx 170$. `Li(1, 1)` is the pole and returns `inf`.

`zeta` : Bernoulli numbers are computed from exact `fractions.Fraction` arithmetic, not `jax.scipy.special.bernoulli`. Above $n = 54$ the result is the constant `1.0`, which is not an approximation: $\zeta(n) - 1 \approx 2^{-n}$ falls below half an eps of 1 once $n > 53$, so every double-precision value from there up _is_ `1.0`. Taking the constant also steps around `jax.scipy.special.zeta`, which returns `nan` above $n \approx 10^{15}$; `spexial` is correct at every magnitude including `inf`, matching SciPy. `jax.grad(zeta)` is meaningful only for $n > 1$; on the negative line it returns a finite value that is not $\zeta'$.

`spence` : Accepts real _and_ complex argument; `jax.scipy.special.spence` is real-only and raises on complex. Its derivative, $\log z/(1-z)$, is supplied analytically — evaluated as the limit $-1$ at $z = 1$, where the closed form is $0/0$, and $-\infty$ at $z = 0$ — which matters beyond speed: JAX's own `spence` differentiates to `nan` across roughly $1 < z < 2$, where `spexial` is exact. Translated from SciPy's Cython implementation.

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
