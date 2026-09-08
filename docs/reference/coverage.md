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

| Function | In JAX | JAX autodiff | scipy on JAX arrays | Custom JVP | Status |
| --- | --- | --- | --- | --- | --- |
| `K0` | -- | -- | value only | yes | only here |
| `K1` | -- | -- | value only | yes | only here |
| `K2` | -- | -- | -- | yes | only here |
| `Li` | -- | -- | -- | available | only here |
| `eval_gegenbauer` | -- | -- | -- | available | only here |
| `eval_gegenbauers` | -- | -- | -- | -- | only here |
| `zeta` | yes (all >= 0.7.2) | value + autodiff | -- | -- | extends upstream |
| `comb` | yes (>= 0.10.2) | value + autodiff | -- | -- | redundant above floor |
| `gamma` | yes (all >= 0.7.2) | value + autodiff | value + autodiff | available | redundant |

## Per-function detail

`K0`
:   JAX has no modified Bessel function of the second kind at any version. scipy's `k0` returns a value under `jit` but raises under `grad`. `spexial` defines the analytic derivative, which measured 2.07x faster than differentiating the 30-term series (637us -> 307us for `grad` over 1000 points).

    Derivative: `-K1(z)`.

`K1`
:   As `K0`. K1'(z) = -(K0(z) + K2(z)) / 2.

    Derivative: `-(K0(z) + K2(z)) / 2`.

`K2`
:   `scipy.special.kn` does not dispatch on JAX arrays at all, even with the array API enabled. K2'(z) = -K1(z) - (2/z) K2(z).

    Derivative: `-K1(z) - (2/z) K2(z)`.

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

`comb`
:   Added to JAX in 0.10.2, below which `spexial` is still needed. `jax.scipy.special.comb` agrees on every edge case `spexial` handles (k > N, k < 0, N < 0) and differentiates. Re-export once the floor reaches 0.10.2.

`gamma`
:   `jax.scipy.special.gamma` is a strict superset: negative reals, complex input, and autodiff, at comparable accuracy. `spexial`'s is real-only. The original implementation existed to add complex support, which JAX now provides -- so this row has no reason to stay. Re-export, then remove.

    Derivative: `gamma(x) psi(x)`.

<!-- END GENERATED TABLE -->

## Regenerating

The table above is rendered from `spexial._src.registry`:

```bash
uv run scripts/gen_coverage_table.py
```

`tests/unit/test_registry.py` checks every `jax_*` claim against the installed JAX and fails if this page is stale, so neither the table nor the roadmap can go quietly out of date when upstream moves.
