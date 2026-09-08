#!/usr/bin/env -S uv run --script
# /// script
#    dependencies = ["nox", "nox_uv"]
# ///
"""Nox setup."""

import argparse
import shutil
from pathlib import Path

import nox
from nox_uv import session

nox.needs_version = ">=2024.3.2"
nox.options.default_venv_backend = "uv"
# A bare `nox` runs `all`, which notifies lint -> test -> docs.
nox.options.sessions = ["all"]

DIR = Path(__file__).parent.resolve()

# `zensical build` has no output-dir flag; it always writes `site_dir` from
# `mkdocs.yml`. The `docs` session relocates the tree afterwards when asked.
DEFAULT_DOCS_OUTPUT = "site"


# =============================================================================
# Comprehensive sessions


@session(uv_groups=["lint", "test", "docs"], reuse_venv=True)
def all(s: nox.Session, /) -> None:  # noqa: A001
    """Run all default sessions."""
    s.notify("lint")
    s.notify("test")
    s.notify("docs")


# =============================================================================
# Linting


@session(uv_groups=["lint"], reuse_venv=True)
def lint(s: nox.Session, /) -> None:
    """Run the linter."""
    s.notify("precommit")
    s.notify("pylint")


@session(uv_groups=["lint"], reuse_venv=True)
def precommit(s: nox.Session, /) -> None:
    """Run prek (the pre-commit hooks)."""
    s.run("prek", "run", "--all-files", *s.posargs)


@session(uv_groups=["lint"], reuse_venv=True)
def pylint(s: nox.Session, /) -> None:
    """Run PyLint."""
    s.run("pylint", "src/spexial", *s.posargs)


# =============================================================================
# Type checking
#
# All three checkers are scoped by `pyproject.toml` to `tests/typing/` -- the
# smoke fixture that guards the public signatures. That is why each invocation
# below is bare, with no path argument.


@session(uv_groups=["typecheck"], reuse_venv=True)
def pyright(s: nox.Session, /) -> None:
    """Type-check the typing smoke fixture with pyright.

    Scoped (via ``[tool.pyright]`` ``include``) to ``tests/typing`` -- the guard
    on the public function signatures. Not the whole tree, which is not yet
    pyright-clean.
    """
    s.run("pyright", *s.posargs)


@session(uv_groups=["typecheck"], reuse_venv=True)
def ty(s: nox.Session, /) -> None:
    """Type-check the typing smoke fixture with ty.

    Scoped (via ``[tool.ty.src]`` ``include``) to ``tests/typing`` -- the same
    signature guard as the ``pyright`` session. ty (0.0.x) is pinned; expect
    deliberate periodic bumps.
    """
    s.run("ty", "check", *s.posargs)


@session(uv_groups=["typecheck"], reuse_venv=True)
def mypy(s: nox.Session, /) -> None:
    """Type-check the typing smoke fixture with mypy.

    Scoped (via ``[tool.mypy]`` ``files``) to ``tests/typing`` -- the same
    signature guard as the ``pyright`` session, not the whole tree (which mypy
    is not yet clean on).
    """
    s.run("mypy", *s.posargs)


# =============================================================================
# Testing


@session(uv_groups=["test"], reuse_venv=True)
def test(s: nox.Session, /) -> None:
    """Run the unit and regular tests."""
    s.notify("pytest", posargs=s.posargs)


@session(uv_groups=["test"], reuse_venv=True)
def pytest(s: nox.Session, /) -> None:
    """Run the unit and regular tests.

    Paths come from ``[tool.pytest.ini_options]`` ``testpaths``: ``README.md``,
    ``docs``, ``src/`` and ``tests/``.
    """
    s.run("pytest", *s.posargs)


@session(uv_groups=["bench"], reuse_venv=True)
def pytest_benchmark(s: nox.Session, /) -> None:
    """Run the benchmarks under CodSpeed."""
    s.run("pytest", "tests/benchmark", "--codspeed", *s.posargs)


# =============================================================================
# Documentation


@session(uv_groups=["docs"], reuse_venv=True)
def docs(s: nox.Session, /) -> None:
    """Build the docs with Zensical. Pass "--serve" to serve them instead."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--serve", action="store_true", help="Serve after building")
    parser.add_argument(
        "--output-dir",
        dest="output_dir",
        default=DEFAULT_DOCS_OUTPUT,
        help=f"Move the built site here (default: {DEFAULT_DOCS_OUTPUT})",
    )
    parser.add_argument(
        "--no-strict",
        action="store_true",
        help="Do not fail the build on warnings (unresolved cross-references, ...)",
    )
    args, posargs = parser.parse_known_args(s.posargs)

    if args.serve:
        s.run("zensical", "serve", *posargs)
        return

    # `zensical build` exits 0 even when it reports unresolved cross-references,
    # so without `--strict` a broken link is invisible to CI and to this session.
    strict = [] if args.no_strict else ["--strict"]
    s.run("zensical", "build", "--clean", *strict, *posargs)

    built = DIR / DEFAULT_DOCS_OUTPUT
    dest = DIR / args.output_dir
    if dest != built:
        if dest.exists():
            shutil.rmtree(dest)
        shutil.move(str(built), str(dest))
    print(f"Docs built into {dest}")


# =============================================================================
# Packaging


@session(uv_groups=["build"])
def build(s: nox.Session, /) -> None:
    """Build an SDist and wheel."""
    build_path = DIR / "build"
    if build_path.exists():
        shutil.rmtree(build_path)

    s.run("python", "-m", "build")


# =============================================================================

if __name__ == "__main__":
    nox.main()
