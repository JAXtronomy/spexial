# Contributing

Contributions are welcome. The repo lives at [JAXtronomy/spexial](https://github.com/JAXtronomy/spexial).

This page is the short version — how to get a working checkout and run things. The full policy, including what a reviewable pull request contains and how accuracy claims are expected to be justified, is in [CONTRIBUTING.md](https://github.com/JAXtronomy/spexial/blob/main/CONTRIBUTING.md) in the repository root.

## Setup

```bash
git clone https://github.com/JAXtronomy/spexial
cd spexial
uv sync --group dev
```

## The task runner

Everything is a [nox](https://nox.thea.codes) session, defined in `noxfile.py`:

```bash
uv run nox -l                  # list every session
uv run nox                     # the default: lint, test, docs
uv run nox -s pytest           # tests
uv run nox -s precommit        # prek (the linters)
uv run nox -s pyright          # or -s ty, -s mypy
uv run nox -s docs             # build the site into `site/`
uv run nox -s docs -- --serve  # live preview
```

Always go through `uv run` / `nox` — never a bare `python`, `pytest` or `ruff`.

## What a new function needs

1. An implementation in `src/spexial/_src/`, exported from `src/spexial/__init__.py` via `__all__`.
2. A NumPy-style docstring with a runnable `Examples` section — it is executed as a test.
3. A parity test against `scipy.special` (or `mpmath` where there is no SciPy counterpart).
4. A row in [Accuracy and domains](guides/accuracy-and-domains.md) stating the verified domain.
5. Adherence to [Conventions](conventions.md) — SciPy's name and argument order where one exists.

!!! warning "On tolerances"

    Do not widen a test tolerance to make a test pass. If an implementation is inaccurate in some regime, document the limitation instead — see [CONTRIBUTING.md](https://github.com/JAXtronomy/spexial/blob/main/CONTRIBUTING.md#accuracy-and-honesty-about-it).
