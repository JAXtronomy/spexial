"""The coverage registry must describe reality, not intentions.

`spexial._src.registry` records what `jax.scipy.special` provides and what
`spexial` adds on top. That is only useful if it cannot go stale, so these tests
check every claim against the *installed* JAX and against the package itself,
and check that the generated documentation page matches. When upstream adds a
function, the relevant test fails and the row has to be revisited.
"""

import importlib
from pathlib import Path

import jax
import jax.numpy as jnp
import jax.scipy.special as jss
import pytest

import spexial as sp
from spexial.registry import JAX_FLOOR, REGISTRY, Status, Support

ROOT = Path(__file__).resolve().parents[2]


def test_registry_covers_exactly_the_public_api():
    """Every export has a row, and every row is an export."""
    exported = {name for name in sp.__all__ if name != "__version__"}
    assert set(REGISTRY) == exported


@pytest.mark.parametrize("name", list(REGISTRY))
def test_jax_availability_claim_is_true(name):
    """`jax_name`/`jax_since` must match the installed JAX.

    A row claiming JAX has no equivalent, when it now does, is how this table
    would quietly become wrong.
    """
    row = REGISTRY[name]
    if row.jax_name is None:
        assert row.jax_since is None
        # Probe the obvious name too, so a newly added upstream function trips this.
        assert not hasattr(jss, name.lower()), (
            f"jax.scipy.special now has `{name.lower()}`; revisit this row"
        )
    else:
        assert hasattr(jss, row.jax_name), (
            f"registry claims jax.scipy.special.{row.jax_name} exists, but it does not"
        )


@pytest.mark.parametrize("name", [n for n, r in REGISTRY.items() if r.jax_name])
def test_jax_autodiff_claim_is_true(name):
    """`jax_support=AUTODIFF` must actually survive `jvp` and `vjp`."""
    row = REGISTRY[name]
    if row.jax_support is not Support.AUTODIFF:
        pytest.skip("row does not claim autodiff")
    fn = getattr(jss, row.jax_name)
    call = {"zeta": lambda a: fn(a, 1.0), "comb": lambda a: fn(a, 2.0)}.get(
        row.jax_name, fn
    )
    x = jnp.asarray(2.5)
    jax.jvp(call, (x,), (jnp.asarray(1.0),))
    _, pullback = jax.vjp(call, x)
    pullback(jnp.asarray(1.0))


@pytest.mark.parametrize("name", [n for n, r in REGISTRY.items() if r.custom_jvp])
def test_custom_jvp_claim_is_true(name):
    """`custom_jvp=True` must mean a `jax.custom_jvp` is actually installed."""
    assert isinstance(getattr(sp, name), jax.custom_jvp)


@pytest.mark.parametrize("name", [n for n, r in REGISTRY.items() if not r.custom_jvp])
def test_absent_custom_jvp_claim_is_true(name):
    """...and `False` must mean there is not one."""
    assert not isinstance(getattr(sp, name), jax.custom_jvp)


def test_redundant_rows_really_are_redundant():
    """A row marked `REDUNDANT` must have a JAX equivalent at the floor."""
    for name, row in REGISTRY.items():
        if row.status is Status.REDUNDANT:
            assert row.jax_since == "*", (
                f"{name} is marked redundant but is only in JAX from {row.jax_since}"
            )
            assert row.jax_support is Support.AUTODIFF


def test_floor_matches_the_declared_dependency():
    """`JAX_FLOOR` must track `pyproject.toml`, or the roadmap reasons from a lie."""
    pyproject = (ROOT / "pyproject.toml").read_text()
    assert f'"jax>={JAX_FLOOR}"' in pyproject


def test_generated_docs_page_is_current():
    """`docs/reference/coverage.md` must match what the registry renders."""
    spec = importlib.util.spec_from_file_location(
        "gen_coverage_table", ROOT / "scripts" / "gen_coverage_table.py"
    )
    gen = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(gen)
    page = (ROOT / "docs" / "reference" / "coverage.md").read_text()
    assert gen.apply(page) == page, (
        "coverage.md is stale; run `uv run scripts/gen_coverage_table.py`"
    )
