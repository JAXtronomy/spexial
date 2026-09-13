"""Benchmarks for the gamma function and the combinatoric factor."""

import jax
import jax.numpy as jnp
import pytest
from pytest_codspeed import BenchmarkFixture

from benchmarks.utils import warm

import spexial as sp

# The implementation reflects arguments below 0.5 onto the other half plane.
X_REFLECTED = jnp.asarray(-2.7)
X_LANCZOS = jnp.asarray(7.3)
X_VECTOR = jnp.linspace(0.6, 20.0, 500)

N_VECTOR = jnp.arange(1.0, 1_001.0)
K_VECTOR = jnp.floor(N_VECTOR / 2)


@pytest.mark.parametrize(
    ("branch", "x"),
    [("lanczos", X_LANCZOS), ("reflection", X_REFLECTED)],
)
def test_gamma_scalar(benchmark: BenchmarkFixture, branch: str, x: jax.Array) -> None:
    """Evaluate the gamma function at a single point."""
    del branch  # only used to name the benchmark
    benchmark(warm(sp.gamma, x))


def test_gamma_vector(benchmark: BenchmarkFixture) -> None:
    """Evaluate the gamma function on 500 points.

    Called directly rather than under ``jax.vmap``: ``gamma`` broadcasts over its
    argument, so this measures the path a caller actually takes.
    """
    benchmark(warm(sp.gamma, X_VECTOR))


def test_comb_scalar(benchmark: BenchmarkFixture) -> None:
    """Evaluate "N choose k" for a single pair."""
    benchmark(warm(sp.comb, jnp.asarray(64.0), jnp.asarray(17.0)))


def test_comb_vector(benchmark: BenchmarkFixture) -> None:
    """Evaluate "N choose k" for 1000 pairs."""
    benchmark(warm(sp.comb, N_VECTOR, K_VECTOR))
