# How to choose between `spexial` and `jax.scipy.special`

Thirteen of the eighteen functions `spexial` exports have no counterpart in JAX at any version, so for those there is nothing to decide. This page is about the other five, where both libraries have something and the right answer depends on what you need.

The short version: **use JAX unless one of the reasons below applies to you** — with one exception, `sph_harm_y`, where JAX's returns wrong values. Fewer dependencies is worth something, and for two of the five the only difference is a gradient you may not be taking.

## The decision, per function

### `gamma` — use either; `spexial` if you differentiate it a lot

`spexial.gamma` calls `jax.scipy.special.gamma` for the value, so the two cannot disagree. What it adds is an analytic derivative, $\Gamma'(x) = \Gamma(x)\psi(x)$, in place of differentiating JAX's implementation term by term.

That is **not** faster — measured at parity, 1.00× — but it keeps **3× less residual memory** through the backward pass: 80 kB against 240 kB over 10,000 points. If you are differentiating `gamma` inside a large `vmap` or a long `scan`, that is the reason to reach for it. If you are not differentiating at all, use JAX's and save the dependency.

### `spence` — use `spexial` if you need complex input or gradients

Two concrete reasons, either of which settles it:

- `jax.scipy.special.spence` is **real-only** and raises on complex input. `spexial.spence` accepts both.
- JAX's gradient is **`nan` across roughly $1 < z < 2$**. `spexial`'s analytic rule is finite and correct there.

It is also 5× faster to differentiate on 38× less residual. This is the clearest case of the four.

### `zeta` — use `spexial` for negative arguments, JAX otherwise

`jax.scipy.special.zeta` is the Hurwitz zeta and returns `nan` for negative arguments. `spexial.zeta` extends it to the negative integers through the functional equation, using Bernoulli numbers computed from exact `fractions.Fraction` arithmetic.

For $n > 1$ `spexial` simply delegates, so there is no accuracy difference and JAX's is marginally faster (1.08× on the gradient). Note the extension is partial: the critical strip $0 < n \le 1$, negative non-integers, and odd $n \le -60$ all return `nan`. If you need those, neither library helps — use `scipy.special.zeta` on the host.

### `sph_harm_y` — use `spexial`; JAX's returns incorrect values

This is the only entry on this page where the choice is not a trade-off.

`jax.scipy.special.sph_harm_y` pairs its arguments element-wise instead of broadcasting them: it indexes its Legendre table with `arange(len(n))`, so `n[i]` goes with `theta[i]`. A scalar degree against a batch of angles is therefore right at index 0 and **silently wrong at every other index** — up to 1.18 absolute for $n \le 3$ — and 0-d input raises from `len()`. Its derivatives are also `nan` at both poles for every $n \ge 1$.

`spexial.sph_harm_y` takes the degree and order as static Python `int`s, so there is nothing to mispair, and it agrees with `scipy.special.sph_harm_y` to $7.6\times10^{-15}$.

If the **Cartesian gradient on the z-axis** matters to you, neither `sph_harm_y` will do — see `sph_harm_y_cart` below, and [Why upstream cannot fix this one](../explanation/why-not-upstream.md#the-case-upstream-cannot-fix-at-all).

### `comb` — use JAX if your floor allows it

`jax.scipy.special.comb` arrived in **jax 0.10.2**. Above that floor the two are equivalent and JAX's is marginally faster. `spexial.comb` exists because this project supports jax from 0.7.2, where JAX has no `comb` at all.

This is the one row in the coverage table marked `redundant-above-floor`: when `spexial`'s minimum jax rises to 0.10.2, `comb` becomes a re-export, then deprecated, then removed. If you are already on 0.10.2 or later, prefer JAX's and you will never notice the transition.

## Functions with no JAX counterpart

There is no decision to make for these — JAX has nothing at any version:

|  |  |
| --- | --- |
| `k0`, `k1`, `k2` | modified Bessel functions of the second kind |
| `k0e`, `k1e`, `k2e` | the same, scaled by $e^z$ — **the only ones that work past $z \approx 705$** |
| `polylog` | the polylogarithm (compare `mpmath.polylog`) |
| `eval_gegenbauer` | Gegenbauer polynomials |
| `eval_gegenbauers` | every order up to `n` in one pass; no counterpart anywhere |
| `sph_legendre_p` | normalized associated Legendre — in `scipy.special` since 1.15, never in JAX |
| `sph_harm_y_cart` | spherical harmonic from a **Cartesian** direction: correct gradients on the z-axis, where the $(\theta, \phi)$ form gives exactly zero |
| `sph_harm_y_cart_all` | the whole $(l, m)$ table in one sweep, laid out as `scipy.special.sph_harm_y_all` |
| `sph_harm_y_cart_all_terms` | the same table unstacked, so a caller's reduction over it stays fused |

`scipy.special` has most of these, but it does not help inside a JAX program: with `SCIPY_ARRAY_API=1` only `gamma` differentiates, `k0`/`k1` return a value but do not, and `kn`, `comb` and `eval_gegenbauer` do not dispatch on JAX arrays at all — they silently convert to NumPy, which breaks under `jit`.

## Checking the current state yourself

The table above is generated from a registry that ships with the package, and it is verified against the _installed_ jax on every test run — so it cannot quietly go stale when JAX adds a function. You can query it directly:

```pycon
>>> from spexial.registry import REGISTRY, Status
>>> unique = sorted(k for k, v in REGISTRY.items() if v.status is Status.UNIQUE)
>>> unique[:4]
['eval_gegenbauer', 'eval_gegenbauers', 'k0', 'k0e']
>>> unique[4:9]
['k1', 'k1e', 'k2', 'k2e', 'polylog']
>>> unique[9:]
['sph_harm_y_cart', 'sph_harm_y_cart_all', 'sph_harm_y_cart_all_terms',
 'sph_legendre_p']

```

To see which functions JAX has caught up on and could be dropped once the floor rises:

```pycon
>>> [k for k, v in REGISTRY.items() if v.status is Status.REDUNDANT_ABOVE_FLOOR]
['comb']

```

And the version each one arrived in upstream:

```pycon
>>> REGISTRY["comb"].jax_since
'0.10.2'

```

The full rendered table, with the measured speed and memory of every gradient, is at [Coverage](../reference/coverage.md).

## If you are unsure

Ask which of these is true of your code:

1. **You need complex input** → `spence` is the only one that offers it.
2. **You differentiate, inside `vmap` or `scan`, and memory is tight** → `spexial` for `gamma` and `spence`.
3. **You need negative arguments to `zeta`** → `spexial`.
4. **You support jax below 0.10.2 and need `comb`** → `spexial`.
5. **None of the above** → JAX's, and one fewer dependency.
