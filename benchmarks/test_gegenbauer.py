"""Benchmarks for the Gegenbauer polynomials."""

import jax
import jax.numpy as jnp
import pytest
from pytest_codspeed import BenchmarkFixture

from benchmarks.utils import warm

import spexial as sp

ALPHA = 1.5
X_SCALAR = jnp.asarray(0.3)
X_VECTOR = jnp.linspace(-0.99, 0.99, 1_000)


@pytest.mark.parametrize("n", [2, 16, 128])
def test_eval_gegenbauer_scalar(benchmark: BenchmarkFixture, n: int) -> None:
    """Evaluate a single Gegenbauer polynomial at a single point."""
    benchmark(warm(lambda x: sp.eval_gegenbauer(n, ALPHA, x), X_SCALAR))


@pytest.mark.parametrize("n", [2, 16, 128])
def test_eval_gegenbauer_vector(benchmark: BenchmarkFixture, n: int) -> None:
    """Evaluate a single Gegenbauer polynomial on 1000 points.

    Called directly rather than under ``jax.vmap``: ``eval_gegenbauer``
    broadcasts over ``x``.
    """
    benchmark(warm(lambda x: sp.eval_gegenbauer(n, ALPHA, x), X_VECTOR))


@pytest.mark.parametrize("n", [16, 128])
def test_eval_gegenbauers_scalar(benchmark: BenchmarkFixture, n: int) -> None:
    """Evaluate all Gegenbauer polynomials up to degree ``n`` at a point."""
    benchmark(warm(lambda x: sp.eval_gegenbauers(n, ALPHA, x), X_SCALAR))


def test_eval_gegenbauers_vector(benchmark: BenchmarkFixture) -> None:
    """Evaluate all Gegenbauer polynomials up to degree 16 on 1000 points.

    Called directly rather than under `jax.vmap`: `eval_gegenbauers` now
    broadcasts `alpha` against `x` and stacks the orders on a new leading axis.
    """
    benchmark(warm(lambda x: sp.eval_gegenbauers(16, ALPHA, x), X_VECTOR))


def test_eval_gegenbauers_vector_via_vmap(benchmark: BenchmarkFixture) -> None:
    """The same result via `jax.vmap` over the scalar core, for comparison.

    This is what callers had to write before `eval_gegenbauers` broadcast, and
    it is the measurement that decided the signature change: fusing both axes
    into the one scan beat mapping the scalar core over them.
    """
    fn = jax.vmap(lambda x: sp.eval_gegenbauers(16, ALPHA, x))
    benchmark(warm(fn, X_VECTOR))


def test_eval_gegenbauers_alpha_by_x_table(benchmark: BenchmarkFixture) -> None:
    """A column of parameters against a row of points, in one scan.

    The pattern an SCF basis expansion needs -- one ``alpha = 2l + 3/2`` per
    angular order, evaluated at every point -- and the reason the scalar
    signature was widened.
    """
    alpha = (2 * jnp.arange(7.0) + 1.5)[:, None]
    benchmark(warm(lambda a, x: sp.eval_gegenbauers(16, a, x), alpha, X_VECTOR))


def test_eval_gegenbauer_grad(benchmark: BenchmarkFixture) -> None:
    """Differentiate a Gegenbauer polynomial with respect to its argument."""
    fn = jax.grad(lambda x: sp.eval_gegenbauer(16, ALPHA, x))
    benchmark(warm(fn, X_SCALAR))
