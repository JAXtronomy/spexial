---
name: code-review
description: >
  Use when reviewing a pull request or diff in the spexial repository. Covers the spexial-specific defects that generic review misses — a new special function whose signature or naming silently diverges from its `scipy.special` counterpart, a doctest whose printed output was hand-typed rather than run, a `jax.jit` wrapper rebuilt per call, a float32/float64 assumption that only holds because `JAX_ENABLE_X64=True` is set in the test environment, and a `filterwarnings` ignore added to silence a real bug.
---

# spexial code review

- spexial is a JAX reimplementation of `scipy.special`. The contract is numerical and API parity with scipy, verified by `tests/regression/`; most real defects are a quiet divergence from that contract, not a style problem.
- Every `Examples` block in a public docstring, and every `python`/`pycon` block in `README.md` and `docs/**`, is a real, exact-match test run by Sybil (see `conftest.py`). An example that was not actually executed against current JAX is a shipped bug, not documentation.

## Scope of this review

- Don't restate what CI already catches: ruff/prek (`nox -s lint`), pyright/ty/mypy (scoped to `tests/typing/`), pytest + Sybil, CodSpeed. Skip purely mechanical checks that are already enforced.
- No generic security checklist — no network, no deserialization of untrusted data. Argument validation matters as _numerical domain_ validation (a bad `alpha`, an out-of-domain `x`), not as an attack surface.
- Don't re-derive JAX's or scipy's own numerics. Focus on spexial's layer: the recurrences, the domain handling, the jit/vmap structure, and the parity claim.

## What changed → what to check

| Change touches | Read |
| --- | --- |
| A new or renamed public function in `src/spexial/_src/**` or `__init__.py` | [scipy parity](#scipy-parity) |
| A recurrence, series, or asymptotic branch | [Numerics](#numerics) |
| A docstring `Examples` block, or `python`/`pycon` blocks in `README.md`/`docs/**` | [Doctests](#doctests) |
| `jax.jit`, `partial(jax.jit, static_argnums=...)`, `lax.scan`/`cond`/`while_loop` | [JAX structure](#jax-structure) |
| `pyproject.toml` `filterwarnings`, `[tool.pytest_env]` | [Warnings and env](#warnings-and-env) |
| Type annotations on a public function | [Typing](#typing) |
| `tests/**` | [Tests](#tests) |

## scipy parity

- A function that shares a name with `scipy.special` must match it: same argument order, same broadcasting, same branch/domain conventions, same return for edge inputs (`x = ±1`, `n = 0`, negative order). A signature that "improves on" scipy's is a divergence — it belongs behind a new name.
- A function with _no_ `scipy.special` counterpart (`eval_gegenbauers`, `polylog`) must be flagged as such in `__init__.py`'s `__all__` comment, the way the existing entries are, and needs a non-scipy reference for its expected values (mpmath is the established choice — it is a `test` group dependency for exactly this).
- `k0`/`k1`/`k2` and friends deviate from scipy's `kn(n, x)` shape on purpose. A new function should follow the surrounding convention rather than invent a third one; if it must deviate, the docstring should say against what and why.

## Numerics

- A recurrence needs its stability domain stated. Upward recurrence for Gegenbauer/Bessel is fine in one regime and catastrophically wrong in another; a PR extending a parameter range without saying which regime it was checked in is under-reviewed.
- Check the closed-form base cases against the recurrence at the boundary (`n = 0`, `n = 1`) — an off-by-one in the seed of a `lax.scan` reproduces as a plausible-looking wrong answer, not an exception.
- Tolerances in `assert_allclose` are part of the claim. A loosened `rtol`/`atol` in an existing parity test is a regression report, not a test fix, unless the PR explains what changed numerically.

## Doctests

- Every changed or added example must show output that was actually produced by running it. Sybil matches exactly (modulo `ELLIPSIS` and `NORMALIZE_WHITESPACE`, both enabled in `conftest.py`) — including `dtype=float64`, which only appears because `[tool.pytest_env]` sets `JAX_ENABLE_X64=True`. Hand-typed `dtype=float32` output is a giveaway that the block was never run.
- Prefer physically or mathematically meaningful arguments over `x = 1`, and show the scipy value alongside where the point of the example is parity.
- `pyproject.toml`'s `testpaths` includes `README.md`, `docs`, and `src/`, so a broken example anywhere in those fails the suite — there is no "docs only" change that skips testing.

## JAX structure

- **A `jax.jit` (or `partial(jax.jit, ...)`) wrapper constructed inside a function body, loop, or method is a compile-cache miss on every call.** `jax.jit` caches on the identity of the wrapped Python function, not on argument equality. It must be built once, at module scope — which is what the existing `@jax.jit`-decorated helpers in `_src/**` do.
- `static_argnums` on an argument that varies per call recompiles per value. Check a newly-static argument is genuinely a small, closed set (like a polynomial order bounded by the caller), not an arbitrary integer.
- A Python `if`/`for` over a traced value silently fails or over-specializes. Control flow that depends on array data belongs in `lax.cond`/`lax.select` and `lax.scan`.

## Warnings and env

- `filterwarnings = ["error"]` is project-wide by design. A new ignore entry needs a comment naming the emitting library and version and why it is expected — the existing Sybil and jax entries are the model. An ignore added to make a failing test pass is hiding the bug.
- `[tool.pytest_env]` sets `JAX_ENABLE_X64=True` and `SPEXIAL_ENABLE_RUNTIME_TYPECHECKING=beartype.beartype`. Source code must not assume either is set: `setup_package.py` defaults the typechecker to `False`, and a user without `x64` gets float32. A claim about precision that only holds under x64 belongs in the docstring, not implicitly in the test config.

## Typing

- pyright, ty, and mypy are all scoped to `tests/typing/` (see `[tool.pyright]`, `[tool.ty.src]`, `[tool.mypy]`) — they are a _signature guard_, not a whole-tree check. A new public function needs a corresponding call in `tests/typing/test_signatures.py`, or its annotations are unverified.
- jaxtyping annotations (`Shaped[Array, "N"]`) are only enforced at runtime when `SPEXIAL_ENABLE_RUNTIME_TYPECHECKING` names a checker. Treat a shape annotation as documentation that must match the code, since nothing checks it in a normal user's environment.

## Tests

- A new special function needs a `tests/regression/` parity test against `scipy.special` (or mpmath where scipy has no counterpart), driven by hypothesis over the claimed domain — not a handful of hand-picked points.
- Hypothesis's per-example deadline is disabled globally in `conftest.py` because jit compilation blows past it; a PR re-enabling a deadline per test should say why that test is compile-free.
- `tests/smoke/` is the fast gate CI runs before the full matrix. A new public export belongs in the smoke test's API check.
- Benchmarks live in `tests/benchmark` and only run under the `⏱️ Run benchmarks` PR label (or on main). A performance claim without a benchmark case is unverifiable in review.

## Repo conventions

- `uv run nox -s ...`, never bare `pytest`/`ruff`/`python`.
- Conventional commits + gitmoji (commitizen, `cz_gitmoji`).
- The prek `ruff-check` hook runs with `--fix --show-fixes` and can modify files — a "ruff passed" claim from a bare `uv run ruff check` is not the same gate as CI's `prek run --all-files`.
