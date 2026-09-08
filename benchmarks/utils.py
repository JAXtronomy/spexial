"""Helpers shared by the benchmarks."""

__all__ = ["warm"]

from collections.abc import Callable
from typing import Any

import jax


def warm(fn: Callable[..., Any], /, *args: Any, **kwargs: Any) -> Callable[[], Any]:
    """Return a zero-argument callable ready to be benchmarked.

    ``fn`` is JIT-compiled and called once so that tracing and compilation --
    which JAX caches -- are not part of the measurement. The returned callable
    blocks until the computation is done, so the measurement covers the whole
    dispatch and execution of the compiled function.

    """
    jitted = jax.jit(fn)
    jax.block_until_ready(jitted(*args, **kwargs))

    def run() -> Any:
        return jax.block_until_ready(jitted(*args, **kwargs))

    return run
