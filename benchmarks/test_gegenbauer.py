"""Benchmarks for the Gegenbauer polynomials."""

import jax
import jax.numpy as jnp
import pytest
from pytest_codspeed import BenchmarkFixture

from benchmarks.utils import warm

import spexial

ALPHA = 1.5
X_SCALAR = jnp.asarray(0.3)
X_VECTOR = jnp.linspace(-0.99, 0.99, 1_000)


@pytest.mark.parametrize("n", [2, 16, 128])
def test_eval_gegenbauer_scalar(benchmark: BenchmarkFixture, n: int) -> None:
    """Evaluate a single Gegenbauer polynomial at a single point."""
    benchmark(warm(lambda x: spexial.eval_gegenbauer(n, ALPHA, x), X_SCALAR))


@pytest.mark.parametrize("n", [2, 16, 128])
def test_eval_gegenbauer_vector(benchmark: BenchmarkFixture, n: int) -> None:
    """Evaluate a single Gegenbauer polynomial on 1000 points."""
    fn = jax.vmap(lambda x: spexial.eval_gegenbauer(n, ALPHA, x))
    benchmark(warm(fn, X_VECTOR))


@pytest.mark.parametrize("n", [16, 128])
def test_eval_gegenbauers_scalar(benchmark: BenchmarkFixture, n: int) -> None:
    """Evaluate all Gegenbauer polynomials up to degree ``n`` at a point."""
    benchmark(warm(lambda x: spexial.eval_gegenbauers(n, ALPHA, x), X_SCALAR))


def test_eval_gegenbauers_vector(benchmark: BenchmarkFixture) -> None:
    """Evaluate all Gegenbauer polynomials up to degree 16 on 1000 points."""
    fn = jax.vmap(lambda x: spexial.eval_gegenbauers(16, ALPHA, x))
    benchmark(warm(fn, X_VECTOR))


def test_eval_gegenbauer_grad(benchmark: BenchmarkFixture) -> None:
    """Differentiate a Gegenbauer polynomial with respect to its argument."""
    fn = jax.grad(lambda x: spexial.eval_gegenbauer(16, ALPHA, x))
    benchmark(warm(fn, X_SCALAR))
