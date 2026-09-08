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
import numpy as np
import pytest

import spexial as sp
from spexial._src.polylog import _li_core
from spexial.registry import JAX_FLOOR, REGISTRY, Status, Support

ROOT = Path(__file__).resolve().parents[2]

# `Li` validates `n` in a plain wrapper and delegates to an inner core, so the
# `jax.custom_jvp` object is not the exported name. Everything else decorates
# the export directly. These probes let the checks below stay strict rather than
# being relaxed to accommodate the difference.
_JVP_OBJECT = {
    "K0": sp.K0,
    "K1": sp.K1,
    "K2": sp.K2,
    "gamma": sp.gamma,
    "Li": _li_core,
}

_PROBES = {
    "K0": (sp.K0, sp.K0.fun),
    "K1": (sp.K1, sp.K1.fun),
    "K2": (sp.K2, sp.K2.fun),
    "gamma": (sp.gamma, sp.gamma.fun),
    "Li": (
        lambda z: jax.vmap(lambda t: _li_core(3, t))(z),
        lambda z: jax.vmap(lambda t: _li_core.fun(3, t))(z),
    ),
}


def test_registry_covers_exactly_the_public_api():
    """Every export has a row, and every row is an export."""
    exported = {name for name in sp.__all__ if name != "__version__"}
    assert set(REGISTRY) == exported


def _version(text: str) -> tuple[int, ...]:
    """Leading numeric components of a version string, for ordering."""
    parts: list[int] = []
    for chunk in text.split("."):
        digits = "".join(c for c in chunk if c.isdigit())
        if not digits:
            break
        parts.append(int(digits))
    return tuple(parts)


JAX_VERSION = _version(jax.__version__)


@pytest.mark.parametrize("name", list(REGISTRY))
def test_jax_availability_claim_is_true(name):
    """`jax_since` must hold against *the installed* JAX, in both directions.

    This is the mechanism the whole registry turns on, so it is checked on
    whatever JAX the job happens to have rather than assumed. The two CI legs
    that matter are `Oldest supported deps` and `Newest supported deps`:

    - On the floor, a row like `comb` must be **absent** -- that absence is the
      reason `spexial` still implements it. A test that merely asserted
      "upstream has it" would fail there, which is backwards.
    - On the newest, a row claiming upstream lacks a function fails the moment
      upstream adds it, which is how a removal candidate gets noticed.
    """
    row = REGISTRY[name]
    if row.jax_name is None:
        assert row.jax_since is None
        # Probe the obvious name, so a newly added upstream function trips this.
        assert not hasattr(jss, name.lower()), (
            f"jax.scipy.special now has `{name.lower()}` (jax {jax.__version__}); "
            "this row is no longer unique to spexial -- re-measure its cost and "
            "consider DELEGATES or REDUNDANT"
        )
        return

    expected = row.jax_since == "*" or _version(row.jax_since) <= JAX_VERSION
    actual = hasattr(jss, row.jax_name)
    if expected and not actual:
        pytest.fail(
            f"registry says jax.scipy.special.{row.jax_name} exists from "
            f"{row.jax_since}, but jax {jax.__version__} does not have it -- "
            "`jax_since` is too low"
        )
    if actual and not expected:
        pytest.fail(
            f"jax {jax.__version__} already has `{row.jax_name}`, but the "
            f"registry says it arrives in {row.jax_since} -- `jax_since` is too "
            "high, and this row may be removable at a lower floor than recorded"
        )


@pytest.mark.parametrize("name", [n for n, r in REGISTRY.items() if r.jax_name])
def test_jax_autodiff_claim_is_true(name):
    """`jax_support=AUTODIFF` must actually survive `jvp` and `vjp`."""
    row = REGISTRY[name]
    if row.jax_support is not Support.AUTODIFF:
        pytest.skip("row does not claim autodiff")
    if not hasattr(jss, row.jax_name):
        pytest.skip(
            f"jax {jax.__version__} predates {row.jax_name} "
            f"(arrives in {row.jax_since}) -- which is why spexial still has it"
        )
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
    assert isinstance(_JVP_OBJECT[name], jax.custom_jvp)


@pytest.mark.parametrize("name", [n for n, r in REGISTRY.items() if not r.custom_jvp])
def test_absent_custom_jvp_claim_is_true(name):
    """...and `False` must mean there is not one."""
    assert name not in _JVP_OBJECT
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


def _residual_bytes(fn, x):
    """Bytes the backward pass must keep alive, via the first-class VJP object.

    `jax.vjp` returns a pytree whose leaves are the saved residuals, so this is
    exact rather than sampled -- unlike wall-clock, it is safe to assert on.
    """
    _, pullback = jax.vjp(fn, x)
    return sum(getattr(leaf, "nbytes", 0) for leaf in jax.tree.leaves(pullback))


@pytest.mark.parametrize(
    "name",
    [n for n, r in REGISTRY.items() if r.custom_jvp and r.cost and r.cost.memory],
)
def test_custom_jvp_really_saves_the_claimed_memory(name):
    """A row claiming a memory saving must actually deliver one.

    This is the column that justifies most of these rows existing, so it is
    checked rather than trusted. `jax.custom_jvp` exposes the undecorated
    implementation as `.fun`, which is what the saving is measured against.
    """
    row = REGISTRY[name]
    custom, plain = _PROBES[name]
    # `Li` only converges for |z| < 1 on its series branch; the others are happy
    # anywhere positive.
    x = (
        jnp.linspace(0.05, 0.45, 2_000)
        if name == "Li"
        else jnp.linspace(0.6, 20.0, 10_000)
    )
    with_jvp = _residual_bytes(custom, x)
    without = _residual_bytes(plain, x)
    assert with_jvp < without, f"{name}: custom JVP saves nothing"
    # The recorded ratio is a measurement, not a contract; allow it to drift by
    # 2x either way before demanding it be re-measured.
    measured = with_jvp / without
    assert measured < row.cost.memory * 2, (
        f"{name}: memory saving has regressed -- recorded {row.cost.memory:.4f}, "
        f"now {measured:.4f}. Re-measure the registry."
    )


@pytest.mark.parametrize("name", [n for n, r in REGISTRY.items() if r.custom_jvp])
def test_custom_jvp_agrees_with_differentiating_the_implementation(name):
    """The analytic derivative must equal what autodiff would have produced.

    A custom JVP silently replaces the true derivative: get it wrong and every
    value stays right while every gradient is quietly wrong. `.fun` is the
    undecorated implementation, so this compares the hand-written rule against
    JAX differentiating the series it replaced.
    """
    custom, plain = _PROBES[name]
    x = jnp.linspace(0.05, 0.45, 40) if name == "Li" else jnp.linspace(0.7, 12.0, 40)
    analytic = jax.grad(lambda a: custom(a).sum())(x)
    autodiff = jax.grad(lambda a: plain(a).sum())(x)
    np.testing.assert_allclose(analytic, autodiff, rtol=1e-6)
