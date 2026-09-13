"""CodSpeed benchmarks for spexial."""

import jax.numpy as jnp
import pytest

import spexial


@pytest.fixture
def x():
    return jnp.linspace(0.1, 1.0, 100)


def test_gegenbauer(benchmark, x):
    benchmark(spexial.eval_gegenbauer, 3, 1.5, x)


def test_comb(benchmark):
    benchmark(spexial.comb, 10, 3)


def test_gamma(benchmark, x):
    benchmark(spexial.gamma, x)


def test_k0(benchmark, x):
    benchmark(spexial.K0, x)


def test_zeta(benchmark, x):
    benchmark(spexial.zeta, 2.0, x)
