# Accuracy and domains

The supported input domain and measured accuracy of every public function. This page is the canonical home for these facts; other pages link here rather than restating them.

Every function is tested against a reference implementation — `scipy.special` where a counterpart exists, `mpmath` where it does not. All entries assume double precision ([how to enable it](../how-to/enable-double-precision.md)).

## Functions

| Function | `scipy.special` | Supported domain | Tested to |
| --- | --- | --- | --- |
| `eval_gegenbauer` | `eval_gegenbauer` | $n \le 20$ integer, $\alpha > -1/2$, $\lvert x \rvert \le 1$ | rtol $10^{-10}$, atol $10^{-13}\times$ recurrence scale; worst $1.5\times10^{-7}$ absolute at $n=20,\ \alpha=10$ |
| `eval_gegenbauers` | -- | as `eval_gegenbauer`; returns all orders $0 \ldots n$ | as `eval_gegenbauer` |
| `comb` | `comb` (`exact=False`) | $0 \le k \le N$, to `DBL_MAX` | rtol $10^{-11}$; worst $1.2 \times 10^{-12}$ |
| `gamma` | `gamma` | real (and complex, jax $\ge$ 0.10.2), $-170 \lesssim x \lesssim 171$ | rtol $10^{-11}$; worst $3.5\times10^{-13}$ for $x \ge 1/2$, $4.3\times10^{-13}$ on $[-170,-30]$ |
| `K0` | `k0` | $0 < z \lesssim 705.5$ | rtol $10^{-6}$; worst $2.0 \times 10^{-7}$ at $z = 8.9984$, ~$10^{-15}$ above $z = 30$ |
| `K1` | `k1` | $0 < z \lesssim 705.5$ | as `K0`; worst $1.8 \times 10^{-7}$ |
| `K2` | `kn` ($n = 2$) | $0 < z \lesssim 705.5$ | as `K0`; worst $1.3 \times 10^{-7}$ |
| `K0e` | `k0e` | $z > 0$, no upper limit | as `K0`; verified to `DBL_MAX` |
| `K1e` | `k1e` | $z > 0$, no upper limit | as `K1`; verified to `DBL_MAX` |
| `K2e` | `kve` ($v = 2$) | $z > 0$, no upper limit | as `K2`; verified to `DBL_MAX` |
| `Li` | -- (`mpmath.polylog`) | scalar $z$, integer $n \ge 1$ | rtol $10^{-11}$, atol $10^{-12}$; worst $3.4 \times 10^{-12}$ for $1 \le n \le 20$, $\lvert z \rvert \le 1000$ |
| `spence` | `spence` | real or complex $z$ | rtol $10^{-12}$, atol $10^{-13}$ vs scipy; worst $1.6 \times 10^{-14}$ real, $2.9 \times 10^{-15}$ complex |
| `zeta` | `zeta` | all real $n$ | rtol $10^{-12}$; worst $3.3 \times 10^{-13}$, near the trivial zeros |

## Per-function limits

`eval_gegenbauer`, `eval_gegenbauers` : At exactly $\alpha = 0$, `spexial` returns $C_0^{(0)} = 1$ and $C_n^{(0)} = 0$ for $n \ge 1$, which is what the generating function gives. SciPy agrees up to 1.14; from 1.18 it returns `0.0` for _every_ order at exactly $\alpha = 0$, while still returning `1.0` at $\alpha = 10^{-300}$.

`eval_gegenbauer` : Error is proportional to the **recurrence's** largest intermediate value, not to the size of the result. At $n = 20,\ \alpha = 10$ the intermediates peak at $9.6\times10^{7}$ on the way to a result of $1.3\times10^{-2}$, and the worst absolute error near a root there is $1.5\times10^{-7}$ — $1.6\times10^{-15}$ of that peak. Expect absolute error of roughly $10^{-15}$ times the peak intermediate, which at large $\alpha$ can be far larger than $10^{-15}$ times the answer. ([Why](../explanation/algorithms.md#gegenbauer-a-recurrence-and-what-that-implies).)

`eval_gegenbauer` : At $x = \pm\infty$ (outside the supported $\lvert x \rvert \le 1$) every order returns its analytic limit. The leading coefficient is $2^n(\alpha)_n/n!$, and for $\alpha > -1/2$ every factor after the first is positive, so its sign is $\operatorname{sign}(\alpha)$ — giving $C_n(+\infty) = \operatorname{sign}(\alpha)\infty$ and $C_n(-\infty) = \operatorname{sign}(\alpha)(-1)^n\infty$, and $0$ for $n \ge 1$ at $\alpha = 0$, where the polynomial vanishes identically. SciPy differs here: it returns `inf` at $+\infty$ but `nan` at $-\infty$. A finite argument large enough to overflow the polynomial gives `nan`.

`comb` : Returns `0` for `k > N`, `k < 0` and `N < 0`, matching `scipy.special.comb`. `comb(inf, 0)` is `1`, `comb(inf, k >= 1)` is `inf` and `comb(inf, inf)` is `nan`, all as in SciPy. The inexact variant only; there is no `exact=True` path. Two formulas are stitched at $N = 1000$ in float64, and at $N = 10$ in float32 and narrower — the log-gamma cancellation grows in units of eps, so it bites $10^9$ times sooner there, while the Beta form's own error is dtype-independent. `float16` and `bfloat16` are computed in float32 and rounded back. The two are: a log-gamma difference below the join, which is the more accurate there but cancels catastrophically above, and a Beta-function form above, which does not cancel. The step in value across the join is a few times $10^{-13}$. Above $N = 2^{1023} \approx 8.99\times10^{307}$ the Beta form is evaluated from its asymptotic form instead, because `jax.scipy.special.betaln` divides its smaller argument by its larger one and XLA flushes that quotient to zero once it is subnormal — which silently cost a factor of $e^{-2}$, 86%, at `comb(N, 1)`. `comb` now holds to $3\times10^{-14}$ out to `DBL_MAX`, the residue being the `exp` of a logarithm near 709 rather than the formula.

`gamma` : Delegates the value to `jax.scipy.special.gamma`, so it cannot drift from upstream; `spexial` supplies only the derivative. Accepts real **and complex** input, the latter from jax 0.10.2 (below that JAX branches on `floor(x)` and raises). Returns `inf` at `x = 0` and `nan` at the negative integers, matching JAX and `scipy.special.gamma` from 1.18 — SciPy is not a stable reference at the poles, returning `inf` everywhere up to 1.14. There is no near-pole blow-up: measured error stays at $10^{-16}$–$8\times10^{-14}$ right up to $10^{-8}$ from a pole. Below $x \approx -170.6$ the true value is subnormal and XLA on CPU flushes it to zero, so `spexial` returns `0` where SciPy returns a denormal. The **first** derivative is accurate to $1.6\times10^{-13}$ everywhere tested, but the **second** is not usable on the negative axis: $\Gamma''$ routes through `jax.scipy.special.digamma`'s own derivative, and JAX's trigamma diverges from the truth from about $x = -7.5$ (at $x = -10.5$ it is wrong by $10^{11}$ relative, and at $-20.5$ it has the wrong sign). This is upstream — `jax.grad(jax.grad(jax.scipy.special.gamma))` returns the identical wrong number — but `spexial` inherits it. `gamma(+inf)` is `inf`, matching SciPy; `gamma(-inf)` is `nan` — Gamma has a pole at every negative integer, so the limit does not exist, and SciPy's `-inf` is not something to copy.

`K0`, `K1`, `K2`, `K0e`, `K1e`, `K2e` : Worst relative error is $2.0 \times 10^{-7}$ for `K0`, $1.8 \times 10^{-7}$ for `K1` and $1.3 \times 10^{-7}$ for `K2`, at $z = 8.998353$. The profile there is **spiky, not smooth** — half a thousandth away, at $z = 8.9932$, the error is only $7.9\times10^{-8}$ — so a coarse grid understates it. Away from that point it falls in stages: ~$8\times10^{-9}$ out to $z = 15$, ~$2\times10^{-13}$ to $z = 30$, ~$10^{-15}$ beyond. ([Why the peak is where it is](../explanation/algorithms.md#modified-bessel-k-two-series-and-a-cross-over).)

**Upper limit.** `K0`, `K1` and `K2` underflow to `0` above $z \approx 705.5$, where the true value is smaller than the smallest normal double. SciPy is not uniform here: `scipy.special.kn` underflows at $z \approx 698$, while `k0` and `k1` return denormals out to $z \approx 745$. **Use `K0e`, `K1e` or `K2e` above $z \approx 705$**: $e^z K_n(z)$ decays only as $1/\sqrt z$ and stays accurate to ~$10^{-15}$ up to `DBL_MAX` ($1.798\times10^{308}$), where `scipy.special.kve` returns `nan`.

**In float32**, the cross-over moves to $z = 4.65$, the worst error rises to $7.1\times10^{-3}$, and the unscaled ceiling drops to $z \approx 85.3$ — the point where $K_0$ falls below float32's smallest normal, $1.18\times10^{-38}$. `float16` and `bfloat16` are computed in float32 and rounded back to the caller's dtype. The scaled forms have no ceiling at any width.

**Derivatives** are exact to second order wherever the values are, the $z = 699$–$705$ tail included, and `grad(grad)` at the pole is `inf` for all six. Three caveats:

- A **subnormal** argument is supported. XLA on CPU flushes one to zero, which made every `K0` below `tiny` return `inf`; the logarithm is taken from the mantissa bits instead, so `K0` matches SciPy across the band and at $5\times10^{-324}$ returns 744.556 where SciPy overflows. `K1` follows $1/z$ there and is exact wherever $1/z$ is representable — a band about a factor of two wide, since $\mathrm{tiny} \times \mathrm{max} \approx 2$ in any IEEE format, which in float32 is $2.9\times10^{-39}$ to $1.2\times10^{-38}$. Past it, and for `K2` throughout, the true value overflows and `inf` is the right answer.
- For `K1` and `K2`, orders **three and above** run about $7\times10^{-4}$ low from $z \approx 686$ (orders 1 and 2 stay at $2\times10^{-16}$). `K0` is unaffected at every order — it holds $1.6\times10^{-16}$ through the same band.
- The **scaled** forms lose relative accuracy to cancellation, since $(e^zK_0)' = G_0 - G_1$ subtracts two nearly equal numbers: the first derivative passes $10^{-6}$ relative at $z \approx 3\times10^{10}$ and is meaningless by $10^{16}$; the second costs about two decades near the cross-over ($3.6\times10^{-5}$ at $z = 8.9$). Absolute error stays at machine precision, and the _values_ remain exact to $1.6\times10^{-16}$ at $10^{300}$ — it is only the ratio that degrades.
- At $z = 0$ the **third** derivative of `K0e`/`K1e`/`K2e` is `nan` under reverse mode and $0$ under forward mode, where the unscaled functions give $-\infty$; the same holds over a band of very small $z$, up to ~$10^{-155}$ for `K1e`. Second order at the pole is guaranteed; third is not.

`Li` : Accepts a scalar `z` only. An array argument raises a broadcasting `TypeError`; use `jax.vmap` ([how](../how-to/use-with-jit-vmap-and-grad.md)). Raises `ValueError` for an order below 1, and for complex `z` — two of the three branches take the real part of a complex intermediate, which is exact for real `z` and would silently discard a genuine imaginary part. A non-integer order, including a whole-number `float` such as `Li(2.0, z)`, is a `TypeError` from the runtime type checker rather than a `ValueError`. For $\lvert z \rvert \ge 2$ the order is capped at **60** by the Bernoulli table the inversion formula needs; past that the result is `nan`. Smaller $\lvert z \rvert$ is unaffected, bounded instead by $\Gamma(n+1)$ overflow above $n \approx 170$. `Li(1, 1)` is the pole and returns `inf`.

`zeta` : Bernoulli numbers are computed from exact `fractions.Fraction` arithmetic, not `jax.scipy.special.bernoulli`. At and above $n = 54$ the result is the constant `1.0`, which is not an approximation: $\zeta(n) - 1 \approx 2^{-n}$ falls below half an eps of 1 once $n > 53$, so every double-precision value from there up _is_ `1.0`. Taking the constant also steps around `jax.scipy.special.zeta`, which returns `nan` above $n \approx 10^{15}$; `spexial` is correct at every magnitude including `inf`, matching SciPy. `jax.grad(zeta)` is meaningful except at two sets of points: the tabulated integers $0 \ge n \ge -60$, where the value is a table lookup that carries no information about how $\zeta$ varies between entries, and the negative even integers at any magnitude, which are a constant `0`. Both report a finite number that is not $\zeta'$. Everywhere else — non-integers, and the odd integers past the table — the functional equation is differentiated and the gradient is genuine to about $10^{-14}$.

`spence` : Accepts real _and_ complex argument; `jax.scipy.special.spence` is real-only and raises on complex. Its derivative, $\log z/(1-z)$, is supplied analytically — evaluated as the limit $-1$ at $z = 1$, where the closed form is $0/0$, and $-\infty$ at $z = 0$ — which matters beyond speed: JAX's own `spence` differentiates to `nan` across roughly $1 < z < 2$, where `spexial` is exact. **Do not compare against SciPy's complex `spence` near $z = 3 \pm \sqrt3$.** It returns `0.01125` at $3 - \sqrt3$ where the true value is $-0.25186$; `spexial` returns the true value. Use `mpmath.polylog(2, 1 - z)` as the reference at those two points. SciPy's _real_ path is unaffected and agrees everywhere.

### `zeta` coverage against SciPy

| Input | `spexial` | how |
| --- | --- | --- |
| $n \ge 54$ | exactly `1` | constant |
| $n > 1$ | $4\times10^{-16}$ | `jax.scipy.special` |
| $n = 1$ | `inf` | the pole |
| $0 < n < 1$ (the critical strip) | $1.8\times10^{-15}$ | eta series |
| $-0.5 < n < 0$ | $8.6\times10^{-15}$ | eta series |
| negative integer $\ge -60$ | exact (0 ulp) | Bernoulli table |
| negative even integer, any magnitude | exactly `0` | trivial zero |
| negative odd integer $< -60$ | $4.2\times10^{-13}$ | functional equation |
| negative non-integer $\le -0.5$ | $3.9\times10^{-13}$ | functional equation |

Every real argument is covered, and SciPy agrees throughout. Accuracy is worst just off a negative even integer, where the $\sin(\pi n/2)$ of the functional equation is near a zero of its own: $1.8\times10^{-13}$ at $n = -99.99$. SciPy is $2\times10^{-4}$ there, so this is the better of the two.

Below about $n = -1000$ the true value exceeds `DBL_MAX` and the result is `±inf`, as it is in SciPy.

## Reference implementations

| Reference       | Used for                                 |
| --------------- | ---------------------------------------- |
| `scipy.special` | Every function with a SciPy counterpart. |
| `mpmath`        | `Li`, which has no SciPy counterpart.    |

## Outside the supported domain

Traced JAX code cannot raise. A function called outside its domain returns `nan` or `inf`, and that value propagates through `jit`, `vmap` and `grad`. Validate inputs before the call, or test the output with `jnp.isnan`. For why the library behaves this way, see [About domain edges](../explanation/edges.md).
