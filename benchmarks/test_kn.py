"""Benchmarks for the modified Bessel functions of the second kind."""

import jax
import jax.numpy as jnp
import pytest
from pytest_codspeed import BenchmarkFixture

from benchmarks.utils import warm

import spexial as sp

# Below the z < 9 cut-off used by the implementation the series expansion for
# small arguments is used, above it the asymptotic one.
Z_SMALL = jnp.asarray(1.5)
Z_LARGE = jnp.asarray(50.0)
Z_VECTOR = jnp.linspace(0.1, 20.0, 500)

FUNCTIONS = {"k0": sp.k0, "k1": sp.k1, "k2": sp.k2}


@pytest.mark.parametrize("name", list(FUNCTIONS))
@pytest.mark.parametrize(("regime", "z"), [("small", Z_SMALL), ("large", Z_LARGE)])
def test_kn_scalar(
    benchmark: BenchmarkFixture,
    name: str,
    regime: str,
    z: jax.Array,
) -> None:
    """Evaluate a modified Bessel function at a single point."""
    del regime  # only used to name the benchmark
    benchmark(warm(FUNCTIONS[name], z))


@pytest.mark.parametrize("name", list(FUNCTIONS))
def test_kn_vector(benchmark: BenchmarkFixture, name: str) -> None:
    """Evaluate a modified Bessel function on 500 points.

    Called directly rather than under ``jax.vmap``: these broadcast over ``z``,
    so this measures the path a caller actually takes. Note that ``Z_VECTOR``
    straddles the ``z = 9`` cut-off, so both branches are exercised in one call.
    """
    benchmark(warm(FUNCTIONS[name], Z_VECTOR))
