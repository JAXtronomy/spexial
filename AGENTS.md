# spexial — Agent Instructions

`spexial` is special functions for JAX, following the `scipy.special` API. It fills gaps in `jax.scipy.special` — which has no modified Bessel `K` at any release, and returns `nan` for `zeta` on the negative line — and supplies analytic derivatives that cut backward-pass memory by up to 268x. It reaches past SciPy in two places: `k0e`/`k1e`/`k2e` stay accurate to `DBL_MAX` where `scipy.special.kve` returns `nan`, and complex `spence` is correct at `3 ± sqrt(3)` where SciPy loses every significant digit. Note both `scipy.special.gamma` and `scipy.special.zeta` already accept negative arguments, so those are _not_ extensions. Everything is written in JAX primitives, so it composes with `jit`, `grad` and `vmap`.

For _using_ `spexial` correctly — which function to reach for, where it differs from SciPy, what is not supported — read [skills/spexial/SKILL.md](skills/spexial/SKILL.md). This file is for working _inside_ this repo.

## Essential commands

```bash
uv sync --group dev                # install everything
uv run nox -s all                  # the full gate: lint -> test -> docs
uv run nox -s lint                 # prek (incl. pyright/ty/mypy guards) + pylint
uv run nox -s test                 # pytest
uv run nox -s docs                 # build the Zensical site into site/
uv run nox -s docs -- --serve      # live preview
uv run nox -s pytest_benchmark     # CodSpeed benchmarks
```

Always go through `uv run`/`nox` — never bare `python`/`pytest`/`ruff`. Re-sync if `uv.lock` moved.

## Layout

```
src/spexial/
  __init__.py       # flat public API mirroring scipy.special; installs the jaxtyping hook
  setup_package.py  # SPEXIAL_ENABLE_RUNTIME_TYPECHECKING -> jaxtyping import hook
  _src/
    custom_types.py # shared Scalar / Vector / AnyArray + *Like input aliases
    bernoulli.py    # the one Bernoulli table
    comb.py gamma.py gegenbauer.py kn.py polylog.py zeta.py
tests/{smoke,unit,parity,benchmark,typing}/
```

The public namespace is deliberately **flat** — `spexial.gamma`, not `spexial.gamma.gamma` — because it mirrors `scipy.special`. Implementations live in `_src/`; `__init__.py` is the only re-export point.

## The rules that will bite you

- **Doctests are load-bearing tests.** Every `Examples` block in a docstring, plus every `python`/`pycon` block in `README.md` and `docs/**`, is executed by [Sybil](https://sybil.readthedocs.io/) (`-p no:doctest` hands the job to Sybil, not stdlib doctest). Output must match exactly. `[tool.pytest_env]` sets `JAX_ENABLE_X64=True`, so array reprs say `dtype=float64` — write examples accordingly.
- **`filterwarnings = ["error"]`.** An unexpected warning is a failure. Check the curated ignore list in `pyproject.toml` before adding to it, and comment why.
- **`xfail_strict = true`.** An `xfail` that starts passing fails the suite.
- **`SPEXIAL_ENABLE_RUNTIME_TYPECHECKING=beartype.beartype` is set under pytest** (off by default at runtime). jaxtyping annotations are enforced at test time, so a wrong shape or dtype annotation is a test failure, not documentation drift.
- **The `ruff-check` hook autofixes** (`--fix --show-fixes`), unlike a bare `ruff check`. Run `uv run nox -s precommit` before assuming you are clean.
- **pyright/ty/mypy are scoped to `tests/typing/` only**, at pinned versions, as local `prek` hooks. They guard the public signatures, not the tree. Widening the scope is welcome; do it deliberately.

## Accuracy is the review question

This is a numerics library. The interesting review question is never "does it run" — it is **"over what domain is this correct, and how do you know?"**

- Every function has a parity test against `scipy.special`, or against `mpmath` where SciPy has no counterpart (`polylog`).
- **Never widen a tolerance to make a test pass.** The tolerances in `tests/parity/` are measured worst-case errors with modest headroom, and several are _not_ machine precision — `k0`/`k1`/`k2` assert rtol `1e-6` because a 30-term series meeting a 10-term asymptotic expansion at `z = 9` delivers `2.0e-7`, and no more — and that is the _float64_ figure; in float32 the cross-over moves to 4.65 and the worst is `7.1e-3`. Loosening one of these silently converts a regression into a pass.
- Where a domain is genuinely unsupported, it is expressed as a domain restriction with a comment, or as an explicit test of the `nan`/degraded behaviour — not as a skip. Keep it that way.
- Every documented domain and tolerance lives in [docs/reference/accuracy-and-domains.md](docs/reference/accuracy-and-domains.md). **A change to numerical behaviour must update that page in the same PR.**

## Guards at a removable singularity

A `jnp.where` that substitutes a **constant** at a singular point makes the value right and every derivative wrong, because a constant differentiates to zero. The wrongness then reappears one order higher each time it is patched: `spence` at `z = 1` was reported three times this way — wrong first derivative, then second, then third — and `polylog` at `z = 0` once.

**Where the function is analytic at the point, change the formula, not the value.** `log(z)/(1-z)` at `z = 1` and `Li_{n-1}(z)/z` at `z = 0` are both `0/0` in their closed forms and both have ordinary power series there, so each is now evaluated as its series — by Horner, `jnp.polyval`, not `sum(z**j * c_j)`, whose term-by-term derivative `j * z**(j-1)` is `0 * inf` at the origin for `j = 0`. Autodiff then differentiates a polynomial and every order is right at once.

**Where the point is a genuine pole**, nothing is finite and no reformulation helps: each order needs its own substituted constant, so fixing order 3 leaves order 4. `kn`'s `_at_pole` is that case, and the ceiling is documented rather than chased. Do not confuse the two — the same symptom has opposite fixes.

When touching any of these, test the second _and_ third derivative at the guarded point, not just the value.

## Known gaps — do not "fix" these by accident

- `gamma` **delegates its value** to `jax.scipy.special.gamma` and supplies only the derivative, so the value cannot drift from upstream. It accepts complex input from jax 0.10.2. Do not reintroduce a hand-rolled Lanczos reflection: the old one lost precision as `|x| * 1e-16 / distance-to-pole`, where the delegated value stays at `1e-16`-`8e-14` right up to `1e-8` from a pole.
- `gamma`'s **second** derivative is unusable on the negative axis, from about `x = -7.5`. `Gamma''` routes through `jax.scipy.special.digamma`'s derivative, and JAX's trigamma is wrong there. It is upstream, not ours, and pinned by a test that fails if JAX fixes it.
- Bugs found in SciPy and JAX themselves are tracked as issues on this repo, labelled `upstream`, each carrying a verified reproduction ready to submit to the project it belongs to: #24, #25, #26 (SciPy) and #27 (JAX). Only #27 is worked around in this codebase, in `comb`; the SciPy three are avoided rather than patched. Do not restate them in `docs/`, which is for people using `spexial`.
- `zeta` covers the whole real line: `jax.scipy.special` above 1, a Borwein eta series on `-0.5 < n < 1`, the Bernoulli table at the integers it reaches, and the functional equation below. Worst accuracy is `6e-13`, at large `|n|` where `gammaln(1 - n)` carries the most magnitude; `1.2e-14` out to `|n| = 10`. Past `n ~ -260.2` the true value exceeds `DBL_MAX` and the answer is `±inf`, as it is in SciPy. `jax.grad` is an artefact at the tabulated integers `0 >= n >= -59` (the table reaches -59, not -60), at the negative even integers of any magnitude, and at `n >= 54` where the value is the constant 1.0; genuine everywhere else, including the odd integers past the table.
- `polylog` takes **scalar `z` only** — the middle branch builds a length-60 vector of powers of `log z`. `jax.vmap` is the supported workaround and is tested.
- `bernoulli.py` builds its table from exact `fractions.Fraction` arithmetic, **not** `jax.scipy.special.bernoulli`, which loses ~7 digits on `B4`. Do not "simplify" it back. Only the Python tuple is cached — caching the `jax.Array` leaks a tracer when the first call happens inside a `jit` trace.
- `gegenbauer.C0` is written `jnp.asarray(x) * 0.0 + 1.0` rather than `ones_like` so weakly-typed input stays weak; `ones_like` changes the repr and breaks doctests.

## Bumping the minimum supported JAX

This is the procedure the library is organised around, and it is driven entirely by the coverage registry in [`src/spexial/_src/registry.py`](src/spexial/_src/registry.py) — rendered to [docs/reference/coverage.md](docs/reference/coverage.md), which is generated, not hand-edited.

**The rule.** A function is removed from `spexial` when upstream covers it _and_ upstream is no worse. "No worse" means all four of:

1. **Available** at the new floor — `jax_since` is `"*"` or at or below it.
2. **Correct over the same domain.** `zeta` fails this: JAX's is the Hurwitz form and returns `nan` on the negative line.
3. **As fast** to differentiate — `cost.speed >= 1.0`.
4. **As lean** to differentiate — `cost.memory >= 1.0`.

Points 3 and 4 are why `gamma` survived a floor at which it was otherwise redundant: JAX computes the value, but differentiating JAX's implementation costs 3x the residual memory of `Gamma'(x) = Gamma(x) psi(x)`, at **no** saving in time (measured 1.0x, i.e. parity — an earlier revision of this file claimed 4.3x faster, which was a measurement error). `gamma` is therefore the clearest case of the general rule: **memory is usually the deciding column, not speed** — a custom JVP replaces a whole series' worth of saved intermediates with one array, and for `k0` that is 68x less residual against a 1.5x speed-up. A row that wins on memory alone still earns its place; a row that wins on neither does not.

If a row fails only 3 or 4, it does not get removed — it becomes `Status.DELEGATES`: call upstream for the value so it cannot drift, and keep our `jax.custom_jvp`. That is strictly better than reimplementing.

### Steps

1. **Re-measure before deciding.** The `cost` numbers are measurements with a shelf life:

   ```bash
   uv run --group bench pytest benchmarks/test_derivatives.py --benchmark-only
   uv run pytest tests/unit/test_registry.py     # memory ratios, asserted exactly
   ```

   Each `spexial` gradient benchmark has an upstream counterpart, so the comparison happens on one runner in one run.

2. **Probe the new floor** for availability and autodiff, rather than trusting a changelog:

   ```bash
   uvx --with "jax==<new-floor>" --with "jaxlib==<new-floor>" python - <<'PY'
   import jax, jax.scipy.special as jss
   print([n for n in ("k0", "kn", "comb", "eval_gegenbauer") if hasattr(jss, n)])
   PY
   ```

3. **Update `pyproject.toml`** (`dependencies`, the `cpu`/`cuda*` extras, and `[tool.ruff]`/`classifiers` if the Python floor moves with it) and `JAX_FLOOR` in the registry. `test_floor_matches_the_declared_dependency` fails if these disagree.

4. **Re-evaluate every `REDUNDANT_ABOVE_FLOOR` row** against the rule above, and move it to `DELEGATES`, `REDUNDANT`, or leave it.

5. **Regenerate and relock:**

   ```bash
   uv run scripts/gen_coverage_table.py
   uv lock
   ```

6. **Run the gate.** `tests/unit/test_registry.py` asserts that rows claiming JAX has no equivalent are still true, so a floor bump that makes one of them wrong fails there rather than silently shipping a duplicate implementation.

### Removing a row

Removal is three releases, never one:

| Release | Action |
| --- | --- |
| N | `Status.DELEGATES` or `REDUNDANT` — re-export upstream, keep the name working |
| N+1 | `DeprecationWarning` on import, docs point at the upstream name |
| N+2 | Remove from `__all__` and delete |

Drop the parity tests only in the last step: while `spexial` still exports the name, it still owes the guarantee.

## Commit style

Conventional commits + gitmoji, enforced by `commitizen` (`cz-conventional-gitmoji`) as a pre-commit hook: `<emoji> <type>(<scope>): <description> (#PR)`.

```
✨ feat(bessel): add K3 via the upward recurrence (#42)
🐛 fix(comb): mask non-integer out-of-range pairs before gammaln (#41)
📝 docs(accuracy): record the measured k0 cross-over error (#43)
```

No `CHANGELOG.md` — deliberate. GitHub Releases are the changelog, per [RELEASING.md](RELEASING.md).

## Release

Single package, tag-driven. A `vX.Y.Z` tag push triggers `.github/workflows/cd.yml` → build + provenance attestation → PyPI trusted publishing. `hatch-vcs` derives the version from tags matching `v*`. Details in [RELEASING.md](RELEASING.md).

## Further reading

- [skills/spexial/SKILL.md](skills/spexial/SKILL.md) — using `spexial` correctly (consumer-facing)
- [.github/skills/code-review/SKILL.md](.github/skills/code-review/SKILL.md) — reviewing PRs here
- [docs/reference/accuracy-and-domains.md](docs/reference/accuracy-and-domains.md), [docs/reference/conventions.md](docs/reference/conventions.md)
- [CONTRIBUTING.md](CONTRIBUTING.md), [RELEASING.md](RELEASING.md)
