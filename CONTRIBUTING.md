# Contributing to `spexial`

## Reporting issues

When opening an issue to report a problem, please provide:

- a minimal code example that reproduces it,
- the operating system, Python version, and the versions of `spexial`, `jax` and `jaxlib`,
- and whether you were running with `jax_enable_x64` on or off — for a numerics library this is very often the answer.

## Getting set up

```bash
uv sync --group dev
uv run nox -s all        # the full gate: lint -> test -> docs
```

Always go through `uv run` / `nox` — never a bare `python`, `pytest` or `ruff`. Re-sync whenever `uv.lock` moves.

Useful individual sessions:

```bash
uv run nox -s lint               # prek (incl. the pyright/ty/mypy guards) + pylint
uv run nox -s test               # pytest
uv run nox -s docs -- --serve    # build + preview the Zensical site
uv run nox -s pytest_benchmark   # CodSpeed benchmarks
```

## What a good pull request contains

Open PRs against `main`.

- **Code.** For a new special function, include a reference to the algorithm you implemented — a paper, a book section (e.g. Zhang & Jin, _Computation of Special Functions_), or DLMF. "Ported from X" is a fine reference; "I derived it" needs the derivation.
- **Tests.** A new function needs unit tests for known values and edge cases, a parity test against `scipy.special` (or `mpmath` where scipy has no counterpart), and coverage of `jax.jit` / `jax.vmap` / `jax.grad`.
- **Docstring.** With a runnable `Examples` block — these are executed as tests, see below.
- **Documentation.** New functions get a row in `docs/reference/accuracy-and-domains.md` stating the supported domain and how it differs from `scipy.special`.
- **Benchmarks.** If you are changing performance-sensitive code, add a benchmark under `tests/benchmark/`. A maintainer can run comparative benchmarks; the PR needs the `⏱️ Run benchmarks` label to trigger the workflow.

## Things that will surprise you

- **Doctests are tests.** Every `Examples` block in a docstring, and every `python`/`pycon` block in `README.md` and `docs/**`, is executed by [Sybil](https://sybil.readthedocs.io/). Output must match exactly. Note that the test environment sets `JAX_ENABLE_X64=True`, so array reprs differ from an unconfigured session.
- **Warnings are errors.** `filterwarnings = ["error"]`. An unexpected warning fails the suite. Check the curated ignore list in `pyproject.toml` before adding to it, and say why in a comment.
- **`xfail_strict = true`.** An `xfail` that starts passing is a failure. Good — it means you fixed something and should say so.
- **The `ruff-check` hook autofixes.** It runs `--fix --show-fixes`, so it modifies files, unlike a plain `ruff check` which only reports. Run `uv run nox -s precommit` before assuming you are clean.
- **Type checkers are scoped.** `pyright`, `ty` and `mypy` are each pointed only at `tests/typing/` (see `[tool.pyright]`, `[tool.ty.src]`, `[tool.mypy]`). They guard the public signatures, not the whole tree. Widening that scope is welcome, incrementally.

## Accuracy and honesty about it

This is a numerics library, so the most important review question is: _over what domain is this actually correct, and how do you know?_

Do not widen a test tolerance to make a test pass. If an implementation is inaccurate in some regime, or a domain is unsupported, say so explicitly — mark the test `xfail` with a reason, or restrict the Hypothesis strategy with a comment explaining the bound — and document it in `docs/reference/accuracy-and-domains.md`. A documented limitation is a contribution. A silently loosened `rtol` is a bug with a green checkmark on it.

## Commit style

Conventional commits with gitmoji, enforced by `commitizen` (`cz-conventional-gitmoji`) as a pre-commit hook:

```
<emoji> <type>(<scope>): <description> (#PR)
```

For example:

```
✨ feat(bessel): add K3 via the upward recurrence (#42)
🐛 fix(comb): return 0 rather than nan for k > N (#41)
📝 docs(accuracy): document the zeta domain floor (#43)
```

There is no `CHANGELOG.md` — this is deliberate. GitHub Releases, generated at tag-push time, are the changelog. See [RELEASING.md](RELEASING.md).
