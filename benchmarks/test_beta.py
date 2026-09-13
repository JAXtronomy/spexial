"""Benchmarks for the incomplete beta function.

The pair worth watching is `test_grad_incomplete_beta_custom_jvp` against
`test_grad_incomplete_beta_autodiff`: the analytic rule replaces a 64-term
series with the integrand evaluated at the endpoint, and it is the largest
saving in the package.
"""

import jax
import jax.numpy as jnp
import pytest
from pytest_codspeed import BenchmarkFixture

from benchmarks.utils import warm

import spexial as sp

Z_SCALAR = jnp.asarray(0.3)
Z_VECTOR = jnp.linspace(0.001, 0.999, 1_000)
# Below and above the z = 1/2 branch switch, so each series can be timed alone.
Z_SMALL = jnp.linspace(0.001, 0.499, 1_000)
Z_LARGE = jnp.linspace(0.501, 0.999, 1_000)

A, B = 2.0, 1.5


def test_incomplete_beta_scalar(benchmark: BenchmarkFixture) -> None:
    """Evaluate at a single point."""
    benchmark(warm(lambda z: sp.incomplete_beta(A, B, z), Z_SCALAR))


@pytest.mark.parametrize(
    "z", [Z_SMALL, Z_LARGE, Z_VECTOR], ids=["small", "large", "both"]
)
def test_incomplete_beta_vector(benchmark: BenchmarkFixture, z: jax.Array) -> None:
    """Evaluate on 1000 points, on each side of the branch switch and across it.

    ``both`` costs roughly the sum of the other two: `jax.numpy.where` picks
    between the branches, so every point pays for both series.
    """
    benchmark(warm(lambda zz: sp.incomplete_beta(A, B, zz), z))


@pytest.mark.parametrize("b", [-1.0, 0.0])
def test_incomplete_beta_non_positive_b(benchmark: BenchmarkFixture, b: float) -> None:
    """The domain `beta(a, b) * betainc(a, b, z)` cannot reach at all.

    Benchmarked to show it costs nothing extra -- the same two series handle it;
    it is the *reconstruction* that fails there, not the integral.
    """
    benchmark(warm(lambda z: sp.incomplete_beta(A, b, z), Z_VECTOR))


def test_grad_incomplete_beta_custom_jvp(benchmark: BenchmarkFixture) -> None:
    """Differentiate with respect to ``z`` through the analytic rule."""
    fn = jax.grad(lambda z: sp.incomplete_beta(A, B, z).sum())
    benchmark(warm(fn, Z_VECTOR))


def test_grad_incomplete_beta_autodiff(benchmark: BenchmarkFixture) -> None:
    """The same gradient, by differentiating the series the rule replaces.

    `jax.custom_jvp` exposes the undecorated implementation as ``.fun``, so
    this is the honest alternative rather than a proxy for it. The gap between
    this and the benchmark above is what the custom rule buys.
    """
    fn = jax.grad(lambda z: sp.incomplete_beta.fun(A, B, z).sum())
    benchmark(warm(fn, Z_VECTOR))


def test_grad_incomplete_beta_parameters(benchmark: BenchmarkFixture) -> None:
    """Differentiate with respect to ``a`` and ``b``.

    There is no closed form for these, so the rule falls back to autodiff of
    the series -- but only when asked. This is the path `symbolic_zeros` skips
    in the common case, and it is much the more expensive one.
    """
    fn = jax.grad(lambda a, b: sp.incomplete_beta(a, b, Z_VECTOR).sum(), argnums=(0, 1))
    benchmark(warm(fn, A, B))
