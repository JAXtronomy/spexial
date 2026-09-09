# Coverage: what is here, and why

`spexial` exists to fill gaps in `jax.scipy.special`. Which gaps those are is a moving target — JAX adds functions, and `scipy.special` has recently begun dispatching on JAX arrays — so this page is generated from a registry that lives in the package itself (`spexial.registry`), and a test fails if the two drift.

It is also the roadmap. A function upstream covers everywhere `spexial` supports has no reason to stay: it becomes a re-export, then a deprecation, then a removal.

## How to read it

- **In JAX** — whether `jax.scipy.special` provides it, and from which release. `spexial`'s floor is `jax >= 0.7.2`, so `all >= 0.7.2` means every supported JAX has it and the row is redundant _today_; a specific version means the row is still needed below it.
- **JAX autodiff** — whether the JAX version survives `jax.jvp` and `jax.vjp`.
- **scipy on JAX arrays** — what `scipy.special` delivers on JAX arrays with `SCIPY_ARRAY_API=1`. This is opt-in: with the variable unset, scipy converts to NumPy and raises under `jit`. Verified against scipy 1.18.1; scipy 1.14.1 provides nothing for any row.
- **Custom JVP** — `yes` where `spexial` defines an analytic derivative rather than differentiating through the series; `available` where the closed form is known and writing it is outstanding work.
- **Status** — what should happen to the row.

<!-- BEGIN GENERATED TABLE -->

| Function | In JAX | JAX autodiff | scipy on JAX arrays | Custom JVP | Grad speed | Grad memory | Status |
| --- | --- | --- | --- | --- | --- | --- | --- |
| `K0` | -- | -- | value only | yes | 0.49x (2.0x better) | 0.0149x (67.3x better) | only here |
| `K1` | -- | -- | value only | yes | 1.11x (1.1x worse) | 0.333x (3.0x better) | only here |
| `K2` | -- | -- | -- | yes | 1.06x (1.1x worse) | 0.333x (3.0x better) | only here |
| `K0e` | -- | -- | -- | yes | 0.725x (1.4x better) | 0.0149x (67.3x better) | only here |
| `K1e` | -- | -- | -- | yes | 0.917x (1.1x better) | 0.0901x (11.1x better) | only here |
| `K2e` | -- | -- | -- | yes | 0.99x (1.0x better) | 0.2x (5.0x better) | only here |
| `Li` | -- | -- | -- | yes | 1.73x (1.7x worse) | 0.00373x (268.0x better) | only here |
| `eval_gegenbauer` | -- | -- | -- | available | -- | -- | only here |
| `eval_gegenbauers` | -- | -- | -- | -- | -- | -- | only here |
| `spence` | yes (all >= 0.7.2) | value + autodiff | value + autodiff | yes | 0.2x (5.0x better) | 0.026x (38.5x better) | extends upstream |
| `zeta` | yes (all >= 0.7.2) | value + autodiff | -- | -- | 1.15x (1.1x worse) | -- | extends upstream |
| `comb` | yes (>= 0.10.2) | value + autodiff | -- | -- | 0.97x (1.0x better) | -- | redundant above floor |
| `gamma` | yes (all >= 0.7.2) | value + autodiff | value + autodiff | yes | 1.03x (1.0x worse) | 0.333x (3.0x better) | delegates + our JVP |

## Per-function detail

`K0`
:   JAX has no modified Bessel function of the second kind at any version. scipy's `k0` returns a value under `jit` but raises under `grad`. `spexial` defines the analytic derivative, which measured 2.04x faster than differentiating the 30-term series (579us -> 285us for `grad` over 1000 points).

    Derivative: `-K1(z)`.

    Gradient cost vs differentiating our own series: speed 0.49x (2.0x better), memory 0.0149x (67.3x better).

`K1`
:   As `K0`, but the trade is now the other way round: `K1` is a thin wrapper over `K1e`, whose own rule autodiff already picks up, so the hand-written rule buys 3x less residual for 11% more wall-clock. Kept for the memory column. K1'(z) = -K0(z) - K1(z)/z.

    Derivative: `-K0(z) - K1(z) / z`.

    Gradient cost vs differentiating our own series: speed 1.11x (1.1x worse), memory 0.333x (3.0x better).

`K2`
:   `scipy.special.kn` does not dispatch on JAX arrays at all, even with the array API enabled. As `K1`, kept for the memory column. K2'(z) = -K1(z) - (2/z) K2(z), summed in the scaled variables: formed directly the `(2/z) K2` term is subnormal from z = 699 and XLA flushes it, which cost the derivative 0.29%.

    Derivative: `-K1(z) - (2/z) K2(z)`.

    Gradient cost vs differentiating our own series: speed 1.06x (1.1x worse), memory 0.333x (3.0x better).

`K0e`
:   Exponentially scaled e^z K0(z), matching `scipy.special.k0e`. JAX has no scaled Bessel K at any version, and scipy's does not dispatch on JAX arrays. This is the only form that survives past z = 705.5, where K0 itself is subnormal and XLA flushes it to 0.

    Derivative: `K0e(z) - K1e(z)`.

    Gradient cost vs differentiating our own series: speed 0.725x (1.4x better), memory 0.0149x (67.3x better).

`K1e`
:   As `K0e`; matches `scipy.special.k1e`.

    Derivative: `K1e(z) - K0e(z) - K1e(z) / z`.

    Gradient cost vs differentiating our own series: speed 0.917x (1.1x better), memory 0.0901x (11.1x better).

`K2e`
:   As `K0e`; matches `scipy.special.kve(2, z)`.

    Derivative: `K2e(z) - K1e(z) - (2/z) K2e(z)`.

    Gradient cost vs differentiating our own series: speed 0.99x (1.0x better), memory 0.2x (5.0x better).

`Li`
:   No general polylogarithm anywhere. `jax.scipy.special.spence` is the n = 2 case only, and scipy has no polylog. The custom JVP is the one row where the two cost columns disagree: it keeps 268x less residual (4.2 MB -> 16 kB over 2000 points) but runs 1.7x slower, because Li_{n-1} must be evaluated afresh rather than reusing saved intermediates. Kept for the memory, which is the binding constraint when vmapping over a large batch.

    Derivative: `Li_{n-1}(z) / z`.

    Gradient cost vs differentiating our own series: speed 1.73x (1.7x worse), memory 0.00373x (268.0x better).

`eval_gegenbauer`
:   Absent from JAX. scipy's does not dispatch on JAX arrays. The derivative in x is 2a C_{n-1}^{a+1}(x), but a custom JVP is *not* wired up: `alpha` is traced too and dC/da has no closed form, so a rule supplying only the x-tangent would silently break `grad` with respect to `alpha`. The saving on offer is modest anyway -- 6 residual leaves, against 29 for `Li`.

    Derivative: `2a C_{n-1}^{a+1}(x)`.

`eval_gegenbauers`
:   No counterpart anywhere: returns every order up to n in one pass, which is the point of it.

`spence`
:   JAX has had `spence` since before our floor, but it is real-only and raises on complex input; this accepts both, which is the reason the row exists. It also wins on both cost columns -- 5.0x faster on 38.5x less residual -- because the analytic derivative log(z)/(1-z) is simply the integrand of the definition. The custom JVP is not merely an optimisation here: `lax.select` evaluates every branch, so differentiating the implementation yields `nan` -- and JAX's own `spence` differentiates to `nan` across roughly 1 < x < 2, where ours is exact. Contributed by Colm Talbot, translated from scipy's Cython implementation.

    Derivative: `log(z) / (1 - z)`.

    Gradient cost vs jax.scipy.special.spence: speed 0.2x (5.0x better), memory 0.026x (38.5x better).

`zeta`
:   `jax.scipy.special.zeta` is the Hurwitz form and returns `nan` for negative arguments; `spexial` adds the negative integers via the functional equation. scipy raises `NotImplementedError` for the Riemann form on JAX arrays. No closed form for zeta', so no custom JVP.

    Gradient cost vs jax.scipy.special.zeta, n > 1 only: speed 1.15x (1.1x worse), memory --.

`comb`
:   Added to JAX in 0.10.2, below which `spexial` is still needed. `jax.scipy.special.comb` agrees on every edge case `spexial` handles (k > N, k < 0, N < 0) and differentiates. Re-export once the floor reaches 0.10.2.

    Gradient cost vs jax.scipy.special.comb: speed 0.97x (1.0x better), memory --.

`gamma`
:   The value is `jax.scipy.special.gamma`, called directly, so it cannot drift. What `spexial` adds is the derivative: Gamma'(x) = Gamma(x) psi(x) keeps 3x less residual than differentiating JAX's implementation, at neutral wall-clock (1.03x, i.e. no measurable saving -- the row earns its place on memory alone). Complex input works from jax 0.10.2 -- the same release that added `comb`, and below it `jax.scipy.special.gamma` branches on `floor(x)` and raises. Delegating means inheriting that limit rather than papering over it; the test probes the capability instead of comparing versions.

    Derivative: `gamma(x) psi(x)`.

    Gradient cost vs jax.scipy.special.gamma: speed 1.03x (1.0x worse), memory 0.333x (3.0x better).

<!-- END GENERATED TABLE -->

## Regenerating

The table above is rendered from `spexial._src.registry`:

```bash
uv run scripts/gen_coverage_table.py
```

`tests/unit/test_registry.py` checks every `jax_*` claim against the installed JAX and fails if this page is stale, so neither the table nor the roadmap can go quietly out of date when upstream moves.
