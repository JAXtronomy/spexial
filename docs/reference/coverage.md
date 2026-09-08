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
| `K0` | -- | -- | value only | yes | 0.48x (2.1x better) | 0.01x (67.4x better) | only here |
| `K1` | -- | -- | value only | yes | -- | 0.06x (16.6x better) | only here |
| `K2` | -- | -- | -- | yes | -- | 0.08x (12.1x better) | only here |
| `Li` | -- | -- | -- | available | -- | -- | only here |
| `eval_gegenbauer` | -- | -- | -- | available | -- | -- | only here |
| `eval_gegenbauers` | -- | -- | -- | -- | -- | -- | only here |
| `zeta` | yes (all >= 0.7.2) | value + autodiff | -- | -- | 1.15x (1.1x worse) | -- | extends upstream |
| `comb` | yes (>= 0.10.2) | value + autodiff | -- | -- | 0.97x (1.0x better) | -- | redundant above floor |
| `gamma` | yes (all >= 0.7.2) | value + autodiff | value + autodiff | yes | 0.23x (4.3x better) | 0.33x (3.0x better) | delegates + our JVP |

## Per-function detail

`K0`
:   JAX has no modified Bessel function of the second kind at any version. scipy's `k0` returns a value under `jit` but raises under `grad`. `spexial` defines the analytic derivative, which measured 2.07x faster than differentiating the 30-term series (637us -> 307us for `grad` over 1000 points).

    Derivative: `-K1(z)`.

    Gradient cost vs differentiating our own series: speed 0.48x (2.1x better), memory 0.01x (67.4x better).

`K1`
:   As `K0`. K1'(z) = -(K0(z) + K2(z)) / 2.

    Derivative: `-(K0(z) + K2(z)) / 2`.

    Gradient cost vs differentiating our own series: speed --, memory 0.06x (16.6x better).

`K2`
:   `scipy.special.kn` does not dispatch on JAX arrays at all, even with the array API enabled. K2'(z) = -K1(z) - (2/z) K2(z).

    Derivative: `-K1(z) - (2/z) K2(z)`.

    Gradient cost vs differentiating our own series: speed --, memory 0.08x (12.1x better).

`Li`
:   No general polylogarithm anywhere. `jax.scipy.special.spence` is the n = 2 case only, and scipy has no polylog. d/dz Li_n(z) = Li_{n-1}(z) / z.

    Derivative: `Li_{n-1}(z) / z`.

`eval_gegenbauer`
:   Absent from JAX. scipy's does not dispatch on JAX arrays. d/dx C_n^a(x) = 2a C_{n-1}^{a+1}(x).

    Derivative: `2a C_{n-1}^{a+1}(x)`.

`eval_gegenbauers`
:   No counterpart anywhere: returns every order up to n in one pass, which is the point of it.

`zeta`
:   `jax.scipy.special.zeta` is the Hurwitz form and returns `nan` for negative arguments; `spexial` adds the negative integers via the functional equation. scipy raises `NotImplementedError` for the Riemann form on JAX arrays. No closed form for zeta', so no custom JVP.

    Gradient cost vs jax.scipy.special.zeta, n > 1 only: speed 1.15x (1.1x worse), memory --.

`comb`
:   Added to JAX in 0.10.2, below which `spexial` is still needed. `jax.scipy.special.comb` agrees on every edge case `spexial` handles (k > N, k < 0, N < 0) and differentiates. Re-export once the floor reaches 0.10.2.

    Gradient cost vs jax.scipy.special.comb: speed 0.97x (1.0x better), memory --.

`gamma`
:   The value is `jax.scipy.special.gamma`, called directly, so it cannot drift. What `spexial` adds is the derivative: Gamma'(x) = Gamma(x) psi(x) is 4.3x faster than differentiating JAX's implementation and keeps 3x less residual. Complex input works, since delegating removed the reflection-formula constraint that made the old Lanczos version real-only.

    Derivative: `gamma(x) psi(x)`.

    Gradient cost vs jax.scipy.special.gamma: speed 0.23x (4.3x better), memory 0.33x (3.0x better).

<!-- END GENERATED TABLE -->

## Regenerating

The table above is rendered from `spexial._src.registry`:

```bash
uv run scripts/gen_coverage_table.py
```

`tests/unit/test_registry.py` checks every `jax_*` claim against the installed JAX and fails if this page is stale, so neither the table nor the roadmap can go quietly out of date when upstream moves.
