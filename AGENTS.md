# spexial — Agent Instructions

`spexial` is special functions for JAX, following the `scipy.special` API. It fills gaps in `jax.scipy.special` and in places extends beyond SciPy (`gamma` and `zeta` accept negative arguments). Everything is written in JAX primitives, so it composes with `jit`, `grad` and `vmap`.

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

- Every function has a parity test against `scipy.special`, or against `mpmath` where SciPy has no counterpart (`Li`).
- **Never widen a tolerance to make a test pass.** The tolerances in `tests/parity/` are measured worst-case errors with modest headroom, and several are _not_ machine precision — `K0`/`K1`/`K2` assert rtol `1e-6` because a 30-term series meeting a 10-term asymptotic expansion at `z = 9` delivers `8e-8`, and no more. Loosening one of these silently converts a regression into a pass.
- Where a domain is genuinely unsupported, it is expressed as a domain restriction with a comment, or as an explicit test of the `nan`/degraded behaviour — not as a skip. Keep it that way.
- Every documented domain and tolerance lives in [docs/reference/accuracy-and-domains.md](docs/reference/accuracy-and-domains.md). **A change to numerical behaviour must update that page in the same PR.**

## Known gaps — do not "fix" these by accident

- `gamma` is **real-only**. The reflection formula branches on `x < 0.5`, which complex input cannot supply. The `Real[...]` annotation is deliberate.
- `gamma` near a non-positive integer loses precision as `1e-17 / distance-to-pole`. Inherent to the reflection formula. The parity suite keeps `1e-4` clear of the poles and pins near-pole behaviour separately.
- `zeta` returns `nan` on the critical strip (`0 < n <= 1`), for negative non-integers, and for odd `n <= -60` (the Bernoulli table ends at `B60`). SciPy handles all three. These are real gaps, documented as such.
- `zeta`'s negative line is a table lookup, so **`jax.grad(zeta)` is only meaningful for `n > 1`**. It returns a finite number on the negative line that is not `ζ'`.
- `Li` takes **scalar `z` only** — the middle branch builds a length-60 vector of powers of `log z`. `jax.vmap` is the supported workaround and is tested.
- `bernoulli.py` builds its table from exact `fractions.Fraction` arithmetic, **not** `jax.scipy.special.bernoulli`, which loses ~7 digits on `B4`. Do not "simplify" it back. Only the Python tuple is cached — caching the `jax.Array` leaks a tracer when the first call happens inside a `jit` trace.
- `gegenbauer.C0` is written `jnp.asarray(x) * 0.0 + 1.0` rather than `ones_like` so weakly-typed input stays weak; `ones_like` changes the repr and breaks doctests.

## Commit style

Conventional commits + gitmoji, enforced by `commitizen` (`cz-conventional-gitmoji`) as a pre-commit hook: `<emoji> <type>(<scope>): <description> (#PR)`.

```
✨ feat(bessel): add K3 via the upward recurrence (#42)
🐛 fix(comb): mask non-integer out-of-range pairs before gammaln (#41)
📝 docs(accuracy): record the measured K0 cross-over error (#43)
```

No `CHANGELOG.md` — deliberate. GitHub Releases are the changelog, per [RELEASING.md](RELEASING.md).

## Release

Single package, tag-driven. A `vX.Y.Z` tag push triggers `.github/workflows/cd.yml` → build + provenance attestation → PyPI trusted publishing. `hatch-vcs` derives the version from tags matching `v*`. Details in [RELEASING.md](RELEASING.md).

## Further reading

- [skills/spexial/SKILL.md](skills/spexial/SKILL.md) — using `spexial` correctly (consumer-facing)
- [.github/skills/code-review/SKILL.md](.github/skills/code-review/SKILL.md) — reviewing PRs here
- [docs/reference/accuracy-and-domains.md](docs/reference/accuracy-and-domains.md), [docs/reference/conventions.md](docs/reference/conventions.md)
- [CONTRIBUTING.md](CONTRIBUTING.md), [RELEASING.md](RELEASING.md)
