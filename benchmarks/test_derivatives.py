"""Gradient benchmarks: the custom JVPs, and how they compare to upstream.

These are the timing half of the registry's `Cost` columns. The memory half is
asserted exactly in `tests/unit/test_registry.py` -- residual bytes are a
property of the traced graph, so they can be tested; wall-clock cannot, and
lives here instead.

Each `spexial` benchmark has an upstream counterpart wherever one exists, so the
comparison that decides whether a row survives a floor bump is measured on the
same runner in the same run, rather than across environments.
"""

import jax
import jax.numpy as jnp
import jax.scipy.special as jss
import pytest
from pytest_codspeed import BenchmarkFixture

from benchmarks.utils import warm

import spexial as sp

X = jnp.linspace(0.6, 20.0, 1_000)
Z = jnp.linspace(0.5, 20.0, 1_000)
N = jnp.linspace(2.0, 40.0, 1_000)


@pytest.mark.parametrize("name", ["k0", "k1", "k2"])
def test_grad_bessel_custom_jvp(benchmark: BenchmarkFixture, name: str) -> None:
    """Gradient through the analytic derivative -- what ships."""
    fn = getattr(sp, name)
    benchmark(warm(jax.grad(lambda a: fn(a).sum()), Z))


@pytest.mark.parametrize("name", ["k0", "k1", "k2"])
def test_grad_bessel_autodiff(benchmark: BenchmarkFixture, name: str) -> None:
    """Gradient through the series instead, for comparison.

    `.fun` is the undecorated implementation behind the `jax.custom_jvp`.
    """
    fn = getattr(sp, name).fun
    benchmark(warm(jax.grad(lambda a: fn(a).sum()), Z))


def test_grad_gamma_spexial(benchmark: BenchmarkFixture) -> None:
    """`spexial.gamma`: JAX's value, our analytic derivative."""
    benchmark(warm(jax.grad(lambda a: sp.gamma(a).sum()), X))


def test_grad_gamma_jax(benchmark: BenchmarkFixture) -> None:
    """`jax.scipy.special.gamma` differentiated by JAX -- the alternative."""
    benchmark(warm(jax.grad(lambda a: jss.gamma(a).sum()), X))


def test_grad_zeta_spexial(benchmark: BenchmarkFixture) -> None:
    """`spexial.zeta`, which also covers the negative line JAX does not."""
    benchmark(warm(jax.grad(lambda a: sp.zeta(a).sum()), N))


def test_grad_zeta_jax(benchmark: BenchmarkFixture) -> None:
    """`jax.scipy.special.zeta` on the half-line where it is defined."""
    benchmark(warm(jax.grad(lambda a: jss.zeta(a, 1.0).sum()), N))


def test_grad_comb_spexial(benchmark: BenchmarkFixture) -> None:
    """`spexial.comb` -- kept only until the JAX floor reaches 0.10.2."""
    benchmark(warm(jax.grad(lambda a: sp.comb(a, 3.0).sum()), N))


@pytest.mark.skipif(
    not hasattr(jss, "comb"),
    reason="jax.scipy.special.comb arrives in 0.10.2; below that spexial's is the "
    "only implementation, which is why the row still exists",
)
def test_grad_comb_jax(benchmark: BenchmarkFixture) -> None:
    """`jax.scipy.special.comb`, the replacement waiting on that floor bump."""
    benchmark(warm(jax.grad(lambda a: jss.comb(a, 3.0).sum()), N))
