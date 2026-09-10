"""Benchmarks for the spherical harmonics and spherical Legendre functions.

The interesting axis here is not raw speed but *what the table costs*.
`sph_harm_y_cart_all` and `sph_harm_y_cart_all_terms` compute identical values
from identical recurrences and differ only in their container -- one stacked
array against a nested tuple of arrays -- and that difference is worth two
orders of magnitude to a caller that reduces over the table rather than keeping
it. The `reduce_*` benchmarks below are the ones that show it.
"""

import jax
import jax.numpy as jnp
import pytest
from pytest_codspeed import BenchmarkFixture

from benchmarks.utils import warm

import spexial as sp

THETA_SCALAR = jnp.asarray(0.7)
THETA_VECTOR = jnp.linspace(0.01, jnp.pi - 0.01, 1_000)
PHI_VECTOR = jnp.linspace(0.0, 2 * jnp.pi, 1_000, endpoint=False)

_DIRECTIONS = jnp.stack(
    [
        jnp.sin(THETA_VECTOR) * jnp.cos(PHI_VECTOR),
        jnp.sin(THETA_VECTOR) * jnp.sin(PHI_VECTOR),
        jnp.cos(THETA_VECTOR),
    ],
    axis=-1,
)
UVEC_SCALAR = jnp.asarray([0.0, 0.6, 0.8])


@pytest.mark.parametrize("n", [2, 16, 64])
def test_sph_legendre_p_scalar(benchmark: BenchmarkFixture, n: int) -> None:
    """Evaluate one spherical Legendre function at one angle."""
    benchmark(warm(lambda t: sp.sph_legendre_p(n, n // 2, t), THETA_SCALAR))


@pytest.mark.parametrize("n", [2, 16, 64])
def test_sph_legendre_p_vector(benchmark: BenchmarkFixture, n: int) -> None:
    """Evaluate one spherical Legendre function on 1000 angles.

    Called directly rather than under `jax.vmap`: the degree and order are
    static, so ``theta`` broadcasts on its own.
    """
    benchmark(warm(lambda t: sp.sph_legendre_p(n, n // 2, t), THETA_VECTOR))


@pytest.mark.parametrize("n", [2, 16])
def test_sph_harm_y_vector(benchmark: BenchmarkFixture, n: int) -> None:
    """Evaluate one complex harmonic on 1000 directions, in spherical coordinates."""
    benchmark(
        warm(
            lambda t, p: sp.sph_harm_y(n, n // 2, t, p),
            THETA_VECTOR,
            PHI_VECTOR,
        )
    )


@pytest.mark.parametrize("n", [2, 16])
def test_sph_harm_y_cart_vector(benchmark: BenchmarkFixture, n: int) -> None:
    """The same harmonic from a Cartesian direction.

    Against `test_sph_harm_y_vector` this is the cost of the formulation that
    keeps the z-axis gradient finite -- it should be no worse, since it skips
    the trigonometry rather than adding any.
    """
    benchmark(warm(lambda u: sp.sph_harm_y_cart(n, n // 2, u), _DIRECTIONS))


@pytest.mark.parametrize("n", [4, 12])
def test_sph_harm_y_cart_all_table(benchmark: BenchmarkFixture, n: int) -> None:
    """Build the whole stacked table over 1000 directions."""
    benchmark(warm(lambda u: sp.sph_harm_y_cart_all(n, n, u), _DIRECTIONS))


@pytest.mark.parametrize("n", [4, 12])
def test_sph_harm_y_cart_all_terms_table(benchmark: BenchmarkFixture, n: int) -> None:
    """Build the same table unstacked.

    Building alone should be close to the stacked form; the two diverge only
    once a caller indexes and reduces, which is what the pair below measures.
    """

    def build(u: jax.Array) -> jax.Array:
        terms = sp.sph_harm_y_cart_all_terms(n, n, u)
        return jnp.stack([jnp.stack(row) for row in terms])

    benchmark(warm(build, _DIRECTIONS))


def _coefficients(n: int) -> jax.Array:
    """Real expansion coefficients over ``0 <= m <= l <= n``."""
    return jnp.arange(1.0, (n + 1) ** 2 + 1).reshape(n + 1, n + 1) / (n + 1) ** 2


@pytest.mark.parametrize("n", [4, 12])
def test_reduce_over_stacked_table(benchmark: BenchmarkFixture, n: int) -> None:
    """Sum ``c_lm Y_l^m`` by indexing the *stacked* table.

    This is the slow path, benchmarked deliberately. Indexing a stacked array
    stops XLA folding each term into the sum as it is produced, so the whole
    table is materialized first. Keeping it here is what stops the claim in
    `sph_harm_y_cart_all_terms`'s docstring from going stale.
    """
    coefficients = _coefficients(n)

    def reduce(u: jax.Array) -> jax.Array:
        table = sp.sph_harm_y_cart_all(n, n, u)
        return sum(
            coefficients[l, m] * table[l, m].real
            for l in range(n + 1)
            for m in range(l + 1)
        )

    benchmark(warm(reduce, _DIRECTIONS))


@pytest.mark.parametrize("n", [4, 12])
def test_reduce_over_terms(benchmark: BenchmarkFixture, n: int) -> None:
    """The same sum, from the unstacked terms -- the fused path."""
    coefficients = _coefficients(n)

    def reduce(u: jax.Array) -> jax.Array:
        terms = sp.sph_harm_y_cart_all_terms(n, n, u)
        return sum(
            coefficients[l, m] * terms[l][m].real
            for l in range(n + 1)
            for m in range(l + 1)
        )

    benchmark(warm(reduce, _DIRECTIONS))


def test_sph_legendre_p_grad(benchmark: BenchmarkFixture) -> None:
    """Differentiate with respect to the polar angle.

    At an ordinary angle, not a pole -- the pole is where
    `jax.scipy.special.sph_harm_y` is `nan` and this one is finite, which is a
    correctness claim rather than a performance one and is tested, not
    benchmarked.
    """
    fn = jax.grad(lambda t: sp.sph_legendre_p(8, 3, t))
    benchmark(warm(fn, THETA_SCALAR))


def test_sph_harm_y_cart_grad(benchmark: BenchmarkFixture) -> None:
    """Differentiate a Cartesian harmonic with respect to the direction."""
    fn = jax.grad(lambda u: sp.sph_harm_y_cart(8, 3, u).real)
    benchmark(warm(fn, UVEC_SCALAR))
