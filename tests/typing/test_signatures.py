"""Static-typing guard for the public API.

This fixture is the *only* thing pyright, ty and mypy are pointed at (see
``[tool.pyright]``, ``[tool.ty.src]`` and ``[tool.mypy]`` in ``pyproject.toml``).
It exists so the public signatures cannot silently regress, without asking the
whole -- not yet checker-clean -- source tree to pass. Widen the scoping in
``pyproject.toml`` as more of the tree becomes clean.

Every binding below is annotated explicitly: the assignment is the assertion.
If a public function's parameter or return type drifts, one of these fails to
type check under `mypy --strict`, pyright and ty, even though the runtime
assertions are trivial.
"""

from typing import TypeAlias

import jax.numpy as jnp
from jaxtyping import Array

import spexial as sp

# Everything public returns a `jax.Array`, never a Python scalar.
Out: TypeAlias = Array


def test_version_is_a_string() -> None:
    """`spexial.__version__` is typed as `str`."""
    version: str = sp.__version__
    assert isinstance(version, str)


def test_comb_signature() -> None:
    """`comb(N, k, /)` takes two array-likes positionally and returns an Array."""
    from_scalars: Out = sp.comb(5, 2)
    from_floats: Out = sp.comb(5.5, 2.0)
    from_arrays: Out = sp.comb(jnp.asarray([5, 6]), jnp.asarray([2, 3]))
    assert from_scalars.shape == ()
    assert from_floats.shape == ()
    assert from_arrays.shape == (2,)


def test_gamma_signature() -> None:
    """`gamma(x, /)` takes one real array-like and returns an Array."""
    from_scalar: Out = sp.gamma(5.0)
    from_array: Out = sp.gamma(jnp.asarray([1.0, 2.0]))
    assert from_scalar.shape == ()
    assert from_array.shape == (2,)


def test_bessel_signatures() -> None:
    """`K0`/`K1`/`K2` each take one real array-like and return an Array."""
    k0: Out = sp.K0(1.0)
    k1: Out = sp.K1(jnp.asarray([1.0, 2.0]))
    k2: Out = sp.K2(1.0)
    assert k0.shape == ()
    assert k1.shape == (2,)
    assert k2.shape == ()


def test_zeta_signature() -> None:
    """`zeta(n, /)` takes one real array-like and returns an Array."""
    from_scalar: Out = sp.zeta(2.0)
    from_array: Out = sp.zeta(jnp.asarray([2.0, 3.0]))
    assert from_scalar.shape == ()
    assert from_array.shape == (2,)


def test_polylog_signature() -> None:
    """`Li(n, z, /)` takes a static `int` order and a scalar, returns an Array."""
    order: int = 2
    value: Out = sp.Li(order, 0.25)
    assert value.shape == ()


def test_gegenbauer_signatures() -> None:
    """`eval_gegenbauer(n, alpha, x, /)` and the plural form."""
    order: int = 3
    single: Out = sp.eval_gegenbauer(order, 1.0, 0.5)
    batched: Out = sp.eval_gegenbauer(order, 1.0, jnp.asarray([0.1, 0.2]))
    ladder: Out = sp.eval_gegenbauers(order, 1.0, 0.5)
    assert single.shape == ()
    assert batched.shape == (2,)
    assert ladder.shape == (order + 1,)
