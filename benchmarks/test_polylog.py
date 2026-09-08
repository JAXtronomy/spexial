"""Benchmarks for the polylogarithm and the Riemann zeta function."""

import jax
import jax.numpy as jnp
import pytest
from pytest_codspeed import BenchmarkFixture

from benchmarks.utils import warm

import spexial

# The polylogarithm implementation uses a different series in each of these
# three ranges of |z|.
Z_SMALL = jnp.asarray(0.25)
Z_INTERMEDIATE = jnp.asarray(1.25)
Z_LARGE = jnp.asarray(20.0)
Z_VECTOR = jnp.linspace(0.05, 0.5, 200)

N_POSITIVE = jnp.arange(2, 42)
N_NEGATIVE = jnp.arange(-40, 0)


@pytest.mark.parametrize(
    ("regime", "z"),
    [("small", Z_SMALL), ("intermediate", Z_INTERMEDIATE), ("large", Z_LARGE)],
)
def test_polylog_scalar(
    benchmark: BenchmarkFixture,
    regime: str,
    z: jax.Array,
) -> None:
    """Evaluate the polylogarithm of order 3 at a single point."""
    del regime  # only used to name the benchmark
    benchmark(warm(lambda z: spexial.Li(3, z), z))


@pytest.mark.parametrize("n", [3, 10])
def test_polylog_vector(benchmark: BenchmarkFixture, n: int) -> None:
    """Evaluate the polylogarithm on 200 points."""
    fn = jax.vmap(lambda z: spexial.Li(n, z))
    benchmark(warm(fn, Z_VECTOR))


@pytest.mark.parametrize(
    ("sign", "n"),
    [("positive", N_POSITIVE), ("negative", N_NEGATIVE)],
)
def test_zeta_vector(benchmark: BenchmarkFixture, sign: str, n: jax.Array) -> None:
    """Evaluate the Riemann zeta function on 40 integer arguments."""
    del sign  # only used to name the benchmark
    benchmark(warm(jax.vmap(spexial.zeta), n))
