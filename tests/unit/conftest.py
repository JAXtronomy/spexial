"""Shared fixtures for the unit tests."""

import os
import resource
import sys
from pathlib import Path

import jax
import pytest

_RSS_LIMIT = 2 * 1024**3
"""Resident memory above which `_release_compiled_programs` drops JAX's caches."""


def _resident_bytes() -> int:
    """Return resident memory: current where the platform says, else the peak."""
    try:
        pages = Path("/proc/self/statm").read_text().split()[1]  # Linux, as on CI
        return int(pages) * os.sysconf("SC_PAGE_SIZE")
    except OSError:
        pass
    try:
        import psutil  # noqa: PLC0415

        return int(psutil.Process().memory_info().rss)
    except ImportError:
        # Peak, not current: past the limit this clears after every test,
        # which costs only time.
        peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        return peak if sys.platform == "darwin" else peak * 1024


@pytest.fixture(autouse=True)
def _release_compiled_programs():
    """Drop JAX's compilation caches when this process grows past `_RSS_LIMIT`.

    Many unit tests compile large programs of their own -- every branch of a
    four-way `switch`, often with derivatives -- and JAX keeps them all for the
    session: one process peaked at 5.9 GB, and four ``--dist=loadfile``
    workers on a 16 GB CI runner were enough for it to be killed mid-run.
    Clearing after every test held the peak to 1.9 GB but nearly doubled the
    file's runtime, by recompiling programs that later tests reuse; clearing
    only past the limit keeps that reuse until memory is the problem.
    """
    yield
    if _resident_bytes() > _RSS_LIMIT:
        jax.clear_caches()
