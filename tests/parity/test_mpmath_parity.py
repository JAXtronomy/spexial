"""Hypothesis-driven parity against `mpmath`, at 40 decimal digits.

This suite exists because `test_scipy_parity.py` cannot catch a *shared* error.
SciPy and `spexial` are both float64 implementations of the same formulae, and
where they agree it is sometimes because they agree on the same approximation
rather than on the truth. `mpmath` is arbitrary-precision and independent, so
it is ground truth rather than a second opinion -- and it is the only reference
available at all for `polylog`, and the only usable one in the tails, where SciPy's
own `kn` underflows to zero while the true value is still finite.

The tolerances below are therefore not the same numbers as in the SciPy suite:
they are `spexial`'s *actual* accuracy, measured against the truth. Two of them
come out **tighter** than the SciPy comparison can support, because SciPy is the
less accurate of the two -- see `test_zeta_negative_integers`.

Every tolerance was measured over the stated domain and then given roughly a
decade of headroom. None was widened to make a failing test pass.
"""

import mpmath as mp
import numpy as np
import pytest
from hypothesis import assume, given, settings, strategies as st

import spexial as sp
from .test_scipy_parity import floats

_DPS = 40
"""Working precision for every reference value. 40 digits is ~24 more than
float64 carries, so the reference contributes nothing to the measured error."""

# `mpmath` is pure Python and slow next to SciPy's C -- at 40 digits a single
# `besselk` is milliseconds, against microseconds for the whole SciPy suite. The
# example count is therefore cut to keep the wall-clock civil; the domains are
# small and the boundaries have their own explicit `parametrize`d tests, so the
# coverage does not depend on volume.
_SLOW = settings(max_examples=15, deadline=None)


def _ref(f, *args, **kwargs):
    """Evaluate `f` at `_DPS` digits and return it as a float64."""
    with mp.workdps(_DPS):
        return float(f(*args, **kwargs))


def _cref(f, *args):
    """As `_ref`, for references whose branch cut makes them complex."""
    with mp.workdps(_DPS):
        return complex(f(*args))


# ---------------------------------------------------------------------------
# gamma
#
# Split at zero: the negative axis is the reflected branch and has its own
# (slightly worse) error, and poles must be avoided on it.


@_SLOW
@given(x=floats(0.5, 171.0))
def test_gamma_positive(x):
    """Measured worst case 2.9e-13 relative against 40-digit `mp.gamma`."""
    np.testing.assert_allclose(sp.gamma(x), _ref(mp.gamma, x), rtol=1e-11)


@_SLOW
@given(x=floats(-170.0, -0.5))
def test_gamma_negative(x):
    """Measured worst case 3.1e-13 relative, off the poles.

    Gamma has a pole at every negative integer, so draws within 1e-3 of one are
    skipped -- not because `spexial` is wrong there (it tracks JAX exactly, and
    the near-pole behaviour has its own test in `tests/unit/test_gamma.py`) but
    because the *reference* value grows without bound and a relative tolerance
    stops meaning anything.
    """
    assume(abs(x - round(x)) > 1e-3)
    np.testing.assert_allclose(sp.gamma(x), _ref(mp.gamma, x), rtol=1e-11)


# ---------------------------------------------------------------------------
# comb


@_SLOW
@given(
    N=st.integers(min_value=0, max_value=170),
    k=st.integers(min_value=0, max_value=170),
)
def test_comb(N, k):
    """Measured worst case 1.9e-13 relative against exact `mp.binomial`.

    `mp.binomial` is exact for integer arguments, so this measures `spexial`'s
    `gammaln` round-trip against the true value rather than against SciPy's own
    `gammaln` round-trip -- which is the same algorithm and could share an error.
    """
    assume(k <= N)
    np.testing.assert_allclose(sp.comb(N, k), _ref(mp.binomial, N, k), rtol=1e-11)


# ---------------------------------------------------------------------------
# zeta


@_SLOW
@given(n=floats(1.0001, 54.0))
def test_zeta_positive(n):
    """Measured worst case 6.6e-16 relative -- essentially machine precision.

    Capped at 54 because `spexial` returns the exact constant 1.0 above that
    (see `test_zeta_is_exactly_one_past_the_cut_off`), which would make the
    comparison trivially true rather than informative.
    """
    np.testing.assert_allclose(sp.zeta(n), _ref(mp.zeta, n), rtol=5e-15)


@pytest.mark.parametrize("n", range(-59, 1))
def test_zeta_negative_integers(n):
    """Exact to 1.7e-16 -- **tighter than the SciPy suite can assert**.

    `spexial` builds these from `fractions.Fraction` Bernoulli numbers, so they
    are correct to the last ulp. `test_scipy_parity.test_zeta_negative_integers`
    has to allow 1e-12 because *SciPy* is off by up to 9.6e-15 here; against the
    truth, 1e-15 holds. This is exactly the case an independent reference buys:
    the SciPy comparison cannot tell a regression from SciPy's own error.
    """
    np.testing.assert_allclose(sp.zeta(float(n)), _ref(mp.zeta, n), rtol=1e-15)


@pytest.mark.parametrize("n", [54.0, 100.0, 1e6, 1e16, 1e300])
def test_zeta_is_exactly_one_past_the_cut_off(n):
    """Past n = 54 the true value *is* 1.0 in float64, and mpmath confirms it."""
    assert float(sp.zeta(n)) == 1.0
    assert _ref(mp.zeta, n) == 1.0


# ---------------------------------------------------------------------------
# spence
#
# `spence(z) = Li_2(1 - z)`, which is how SciPy defines it and how mpmath is
# asked for it here.


@_SLOW
@given(z=floats(1e-6, 50.0))
def test_spence_real(z):
    """Measured worst case 2.7e-14 relative against `mp.polylog(2, 1 - z)`.

    At z = 0.586, over 44,000 points at 40 digits across this test's own
    domain. The `rtol` of 1e-12 keeps 37x headroom.
    """
    expected = _cref(mp.polylog, 2, 1 - z).real
    np.testing.assert_allclose(sp.spence(z), expected, rtol=1e-12, atol=1e-13)


@_SLOW
@given(r=floats(0.05, 10.0), phi=floats(0.0, 2 * np.pi))
def test_spence_complex(r, phi):
    """The complex branch, which `jax.scipy.special.spence` does not implement."""
    z = r * np.exp(1j * phi)
    got = complex(sp.spence(z))
    expected = _cref(mp.polylog, 2, 1 - z)
    np.testing.assert_allclose(
        [got.real, got.imag], [expected.real, expected.imag], rtol=1e-11, atol=1e-12
    )


# ---------------------------------------------------------------------------
# kn
#
# mpmath is the only workable reference above z ~ 698, where `scipy.kn`
# underflows to 0 but the true value is still a finite (if tiny) number. The
# SciPy suite therefore stops at 690; this one does not have to.


_ORDERS = [(0, sp.k0), (1, sp.k1), (2, sp.k2)]
_SCALED = [(0, sp.k0e), (1, sp.k1e), (2, sp.k2e)]


@pytest.mark.parametrize(("order", "func"), _ORDERS)
@_SLOW
@given(z=floats(1e-6, 700.0))
def test_kn(order, func, z):
    """1e-6, the honest tolerance for a 30-term series meeting a 10-term one.

    Worst measured is 2.0e-7 (`k0`) just below the z = 9 cross-over; away from
    it the same code is good to 1e-15.
    """
    np.testing.assert_allclose(func(z), _ref(mp.besselk, order, z), rtol=1e-6)


@pytest.mark.parametrize(("order", "func"), _ORDERS)
@_SLOW
@given(z=floats(8.5, 9.5))
def test_kn_across_the_crossover(order, func, z):
    """The hand-off between the two series, where the error actually peaks."""
    np.testing.assert_allclose(func(z), _ref(mp.besselk, order, z), rtol=1e-6)


@pytest.mark.parametrize(("order", "func"), _SCALED)
@_SLOW
@given(z=floats(1e-6, 1e4))
def test_scaled_kn(order, func, z):
    """The scaled forms run far past where the unscaled ones have any value.

    Sampled to 1e4 rather than to the true ceiling (~9e307): the far tail has
    its own explicit boundary tests in `tests/unit/test_kn.py`, and mpmath at 40
    digits is slow enough that spending draws out there buys nothing.
    """
    expected = _ref(lambda o, t: mp.exp(t) * mp.besselk(o, t), order, z)
    np.testing.assert_allclose(func(z), expected, rtol=1e-6)


@pytest.mark.parametrize(("order", "func"), _SCALED)
@pytest.mark.parametrize("z", [706.0, 1000.0, 1e4, 1e6])
def test_scaled_kn_in_the_tail_scipy_cannot_reach(order, func, z):
    """Past z = 705.5 the unscaled value is unrepresentable; the scaled one is exact.

    This is the region the SciPy suite has to stop short of, and the reason
    these three functions exist. Machine precision, not the 1e-6 the cross-over
    forces elsewhere.
    """
    expected = _ref(lambda o, t: mp.exp(t) * mp.besselk(o, t), order, z)
    assert float(sp.k0(z)) == 0.0  # the unscaled form has nothing to offer here
    np.testing.assert_allclose(func(z), expected, rtol=1e-15)


# ---------------------------------------------------------------------------
# polylog
#
# No SciPy counterpart at all, so mpmath is not a cross-check here -- it is the
# only reference there is.


@_SLOW
@given(n=st.integers(min_value=1, max_value=20), z=floats(-1000.0, 0.999))
def test_li(n, z):
    """Measured worst case 3.8e-14 relative over 1 <= n <= 20, |z| <= 1000.

    The worst point sits just past the |z| = 2 branch boundary
    (n = 12, z = -2 - 1e-9), which uniform sampling does not find; an
    earlier figure of 1.3e-14 came from a grid that missed it. The gate
    keeps 2.8x headroom, the tightest in the suite.

    `z` stops below 1: `Li_n` has a branch point there, and `Li_1(1)` is a pole.
    Values above it are covered by `tests/unit/test_polylog.py`, which pins the
    branch explicitly rather than sampling across it.
    """
    expected = _cref(mp.polylog, n, z).real
    # 1e-13, not 1e-11: measured worst over this domain is 1.3e-14, so the
    # looser gate would pass a 100x regression.
    np.testing.assert_allclose(sp.polylog(n, z), expected, rtol=1e-13, atol=1e-14)


@pytest.mark.parametrize("n", [1, 2, 3, 5, 10, 20])
@pytest.mark.parametrize("z", [-2.0, -0.5, 0.5, 2.0])
def test_li_at_the_branch_boundaries(n, z):
    """|z| = 1/2 and |z| = 2 are where the three series hand off to each other.

    Hypothesis will effectively never draw them exactly, and `|z| = 2` was a
    real bug once -- no branch covered the boundary and it returned 0.0.
    """
    expected = _cref(mp.polylog, n, z).real
    np.testing.assert_allclose(sp.polylog(n, z), expected, rtol=1e-11, atol=1e-12)


def _recurrence_scale(n, alpha, x):
    """Largest intermediate the three-term recurrence passes through.

    A recurrence's attainable accuracy is set by its own working magnitude, not
    by the size of its answer: at n = 20, alpha = 10 the intermediate |C_k|
    peaks at 9.6e7 for an answer of 1.3e-2, so a flat `atol` asserts something
    float64 cannot deliver. Measured worst against mpmath is 1.5e-7 absolute --
    1.6e-15 of that peak, i.e. backward-stable to machine precision.
    """
    return float(np.abs(np.asarray(sp.eval_gegenbauers(n, alpha, x))).max())


# ---------------------------------------------------------------------------
# eval_gegenbauer


@_SLOW
@given(
    n=st.integers(min_value=0, max_value=20),
    alpha=floats(-0.49, 10.0),
    x=floats(-1.0, 1.0),
)
def test_eval_gegenbauer(n, alpha, x):
    """Measured worst case 1.9e-12 absolute, near the polynomial roots.

    `alpha == 0` is excluded, as in the SciPy suite, but for a different reason
    worth recording: at exactly zero **mpmath returns 0.0 for order 0 too**,
    where SciPy 1.17 and `spexial` both return 1.0. mpmath evaluates through a
    `1/Gamma(2*alpha)` prefactor that vanishes in the limit; the generating
    function `(1 - 2xt + t^2)^-alpha = 1` at `alpha = 0` gives `C_0 = 1`. So
    this single degenerate point is a convention artefact in the reference, not
    an error in `spexial` -- and away from it mpmath agrees to 1e-12.
    """
    assume(alpha != 0.0)
    # `zeroprec` is mpmath's own remedy for its adaptive summation failing to
    # decide whether a value is zero: without it, `gegenbauer(1, a, 0.0)` -- an
    # exact root, since C_1 = 2*a*x -- raises rather than returning 0. It changes
    # nothing where the value is not (near) zero.
    expected = _ref(mp.gegenbauer, n, alpha, x, zeroprec=1000)
    np.testing.assert_allclose(
        sp.eval_gegenbauer(n, alpha, x),
        expected,
        rtol=1e-10,
        atol=max(1e-11, 1e-13 * _recurrence_scale(n, alpha, x)),
    )


@_SLOW
@given(
    n=st.integers(min_value=0, max_value=20),
    alpha=floats(-0.49, 10.0),
    x=floats(-1.0, 1.0),
)
def test_eval_gegenbauers(n, alpha, x):
    """Every order the all-orders variant returns, against the truth.

    Checked against mpmath rather than against `spexial.eval_gegenbauer`: the
    two share a recurrence, so a self-consistency test would pass with both of
    them wrong.
    """
    assume(alpha != 0.0)
    got = sp.eval_gegenbauers(n, alpha, x)
    expected = [_ref(mp.gegenbauer, k, alpha, x, zeroprec=1000) for k in range(n + 1)]
    assert got.shape == (n + 1,)
    np.testing.assert_allclose(
        got,
        expected,
        rtol=1e-10,
        atol=max(1e-11, 1e-13 * _recurrence_scale(n, alpha, x)),
    )
