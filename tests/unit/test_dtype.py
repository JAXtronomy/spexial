"""Unit tests for `spexial._src.dtype`, the shared bit-level helpers.

These exist because XLA on CPU flushes subnormals to zero -- in arithmetic and
in comparisons alike -- so the ordinary float tests cannot see the values these
helpers are written to recover. Every case is checked eagerly *and* under `jit`:
a helper can be correct in isolation and collapse once XLA fuses it into a
surrounding kernel, which is how several of these defects were introduced.
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from spexial._src.dtype import ldexp_no_flush, log_no_flush


@pytest.mark.parametrize("dtype", [jnp.float32, jnp.float64])
def test_ldexp_no_flush_lifts_the_whole_subnormal_band(dtype):
    """The scaled value must be exact, which `z * 2**k` is not once `z` is flushed."""
    info = jnp.finfo(dtype)
    exponent = 2 * info.nmant
    smallest = float(np.nextafter(np.zeros((), dtype), np.ones((), dtype)))
    for value in (
        smallest,
        float(info.tiny) / 2,
        float(info.tiny),
        1.0,
        -smallest,
        0.0,
        -0.0,
    ):
        argument = jnp.asarray(value, dtype=dtype)
        expect = np.float64(value) * np.float64(2.0**exponent)
        # `exponent` is static: closed over rather than traced as an operand.
        for wrap in (lambda f: f, jax.jit):
            got = float(wrap(lambda z: ldexp_no_flush(z, exponent))(argument))
            assert got == pytest.approx(float(expect), rel=1e-6), value


@pytest.mark.parametrize("dtype", [jnp.complex64, jnp.complex128])
def test_log_no_flush_complex_when_one_component_straddles_tiny(dtype):
    """REGRESSION: reconstructing only when *both* components were subnormal.

    The predicate was written on the reasoning that a subnormal beside a normal
    component is 292 decades below it and cannot matter. At the bottom of the
    normal range that is false -- `tiny` and the largest subnormal are one
    binade apart -- and the flush took the *argument* of the logarithm to `0`
    where the answer was `pi/4`, losing the imaginary part outright rather than
    approximately.
    """
    real_dtype = jnp.zeros((), dtype).real.dtype
    tiny = float(jnp.finfo(real_dtype).tiny)
    below = float(np.nextafter(np.asarray(tiny, real_dtype), np.zeros((), real_dtype)))
    for re, im in ((tiny, below), (below, tiny), (tiny, tiny / 2), (-tiny, tiny / 2)):
        argument = jnp.asarray(complex(re, im), dtype=dtype)
        # numpy does not flush on CPU, so it holds the true value of the bits.
        expect = np.log(np.complex128(complex(re, im)))
        for wrap in (lambda f: f, jax.jit):
            got = complex(wrap(log_no_flush)(argument))
            assert got.imag == pytest.approx(float(expect.imag), rel=1e-6), (re, im)
            assert got.real == pytest.approx(float(expect.real), rel=1e-6), (re, im)


def test_log_no_flush_refuses_to_silently_drop_the_dtype_argument():
    """`dtype` widens the arithmetic; the complex path cannot honour it."""
    with pytest.raises(NotImplementedError, match="complex"):
        log_no_flush(jnp.asarray(1 + 1j, dtype=jnp.complex128), dtype=jnp.complex128)


@pytest.mark.parametrize("dtype", [jnp.complex64, jnp.complex128])
@pytest.mark.parametrize(
    "value", [2 + 0j, 0.85 + 0j, 0.5 + 0j, 1.0001 + 0j, 3j, -2 + 0j]
)
def test_log_no_flush_complex_is_never_worse_than_plain_log(dtype, value):
    """REGRESSION: the reconstruction branch swallowed the whole ordinary plane.

    `0.0 < tiny` is true, so a predicate of "either component is subnormal" is
    true of every `z` on either axis -- which is most complex arguments anyone
    passes. Those all took the scaled branch, where ``log(z * 2**k) - k*log2``
    cancels catastrophically once ``|z| ~ 1``: three decimal digits gone in
    complex64, 1.08% relative at ``z = 1.0001``, and *worse* than the `jnp.log`
    it was standing in for. Reconstruction is only ever needed where a
    subnormal component could still move the answer.
    """
    argument = jnp.asarray(value, dtype=dtype)
    for wrap in (lambda f: f, jax.jit):
        assert complex(wrap(log_no_flush)(argument)) == complex(jnp.log(argument))


@pytest.mark.parametrize("dtype", [jnp.complex64, jnp.complex128])
def test_log_no_flush_complex_is_holomorphic(dtype):
    """REGRESSION: the reconstruction branch differentiated to zero.

    `lax.bitcast_convert_type` carries an identically zero tangent, so anything
    rebuilt from bits contributed nothing to the derivative and `log_no_flush`
    was not holomorphic where that branch fired -- the imaginary-direction
    derivative was `0` against a true ``1/z`` of 4.5e307 at ``z = tiny``, which
    is a representable number rather than an overflow. `ldexp_no_flush` now
    declares its own derivative, which it can do exactly: it is ``z * 2**k``.
    """
    real_dtype = jnp.zeros((), dtype).real.dtype
    tiny = float(jnp.finfo(real_dtype).tiny)
    along_real = jnp.asarray(1 + 0j, dtype)
    along_imaginary = jnp.asarray(1j, dtype)
    # One ordinary argument on the axis, one inside the reconstruction band.
    for value in (2 + 0j, complex(tiny, tiny / 4)):
        argument = jnp.asarray(value, dtype=dtype)
        d_real = jax.jvp(log_no_flush, (argument,), (along_real,))[1]
        d_imaginary = jax.jvp(log_no_flush, (argument,), (along_imaginary,))[1]
        assert complex(d_real) == pytest.approx(1 / value, rel=1e-6), value
        assert complex(d_imaginary) == pytest.approx(1j / value, rel=1e-6), value
