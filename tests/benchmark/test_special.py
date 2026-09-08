"""Micro-benchmarks for the public API.

Deliberately small: one representative call per function, always on
already-compiled code (`block_until_ready` keeps the async dispatch out of the
timing). Kept out of the default suite -- `pytest-benchmark` lives in the
`bench` dependency group, so this module skips itself when it is absent.
"""

from functools import partial

import jax
import jax.numpy as jnp
import pytest

import spexial as sp

pytest.importorskip("pytest_benchmark", reason="benchmarks run under the `bench` group")

X = jnp.linspace(0.5, 20.0, 1000)
Z = jnp.linspace(0.1, 50.0, 1000)


def _timed(func, *args):
    """Compile once, then return a closure that runs and waits."""
    jitted = jax.jit(func)
    jax.block_until_ready(jitted(*args))
    return lambda: jax.block_until_ready(jitted(*args))


@pytest.mark.parametrize(
    ("name", "func", "args"),
    [
        ("comb", sp.comb, (jnp.arange(1000.0), 3.0)),
        ("gamma", sp.gamma, (X,)),
        ("K0", sp.K0, (Z,)),
        ("K2", sp.K2, (Z,)),
        ("zeta", sp.zeta, (X,)),
        ("eval_gegenbauer", partial(sp.eval_gegenbauer, 10, 1.5), (X,)),
        ("eval_gegenbauers", partial(sp.eval_gegenbauers, 10, 1.5), (0.3,)),
        ("Li", partial(sp.Li, 3), (0.3,)),
    ],
)
def test_benchmark(benchmark, name, func, args):
    """Time one steady-state call of each public function."""
    benchmark.group = name
    benchmark(_timed(func, *args))
