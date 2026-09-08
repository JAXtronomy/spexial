"""Hypothesis-driven parity against `scipy.special` (and `mpmath` for `Li`).

Every tolerance here was measured, not tuned until the suite went green. Where
an implementation genuinely does not cover a domain, the strategy is restricted
and the comment says why -- see the module-level notes on each function.
"""

import mpmath as mp
import numpy as np
import pytest
from hypothesis import assume, example, given, strategies as st
from scipy.special import (
    comb as scipy_comb,
    eval_gegenbauer as scipy_eval_gegenbauer,
    gamma as scipy_gamma,
    k0 as scipy_k0,
    k1 as scipy_k1,
    kn as scipy_kn,
    zeta as scipy_zeta,
)

import spexial as sp


def floats(lo, hi):
    """Finite float64s in ``[lo, hi]``; bounded so Hypothesis never filters."""
    return st.floats(
        min_value=lo,
        max_value=hi,
        allow_nan=False,
        allow_infinity=False,
        allow_subnormal=False,
        width=64,
    )


# ---------------------------------------------------------------------------
# comb


@given(
    N=st.integers(min_value=0, max_value=170),
    k=st.integers(min_value=-5, max_value=175),
)
def test_comb(N, k):
    """Measured worst case over 0 <= N, k <= 170 is 3.2e-13 relative."""
    np.testing.assert_allclose(sp.comb(N, k), scipy_comb(N, k), rtol=1e-11)


@given(
    N=floats(0.0, 50.0),
    k=floats(0.0, 50.0),
)
def test_comb_non_integer(N, k):
    """The generalized (non-integer) binomial coefficient agrees too."""
    # No `assume` here: over the strategy's own range `scipy_comb` is always
    # finite, so the guard this used to carry filtered nothing and only implied
    # a hazard that does not exist.
    np.testing.assert_allclose(sp.comb(N, k), scipy_comb(N, k), rtol=1e-11)


# ---------------------------------------------------------------------------
# gamma
#
# Real arguments only -- see `spexial.gamma`. Below 0.5 the reflection formula
# divides by `sin(pi x)`, whose relative accuracy degrades like the reciprocal
# of the distance to the nearest pole; `test_gamma_near_a_pole` pins that down.


@given(x=floats(-30.0, 170.0))
@example(x=0.5)
@example(x=1.0)
@example(x=-0.5)
def test_gamma(x):
    """Measured worst case (poles avoided by 1e-4) is 6.1e-12 relative."""
    assume(x >= 0.5 or abs(x - round(x)) > 1e-4)
    np.testing.assert_allclose(sp.gamma(x), scipy_gamma(x), rtol=1e-10)


@given(x=floats(-170.0, -30.0))
def test_gamma_negative_tail(x):
    """Below -30 the reflection formula degrades; 5e-10, not the 1e-10 above.

    The documented domain reaches |x| ~ 171 but the strategy above stops at -30.
    Measured worst case over [-170, -30] is 2.4e-10, so the tighter tolerance
    genuinely does not hold here -- it is stated separately rather than either
    loosened everywhere or left untested.
    """
    assume(abs(x - round(x)) > 1e-4)
    expected = scipy_gamma(x)
    # Below ~1e-300 the true value is subnormal, and XLA on CPU flushes those to
    # zero; see the accuracy docs.
    assume(abs(expected) > 1e-300)
    np.testing.assert_allclose(sp.gamma(x), expected, rtol=5e-10)


@pytest.mark.parametrize("pole", [-7.0, -25.0, -40.0, -55.0])
@pytest.mark.parametrize("distance", [1e-6, 1e-8])
def test_gamma_near_a_pole(pole, distance):
    """Accuracy near a pole is only ~1e-17 / distance, not 1e-10.

    This is inherent to the reflection formula, not a fixable bug: the
    ``sin(pi x)`` denominator loses exactly the digits that ``x`` is close to
    an integer by. Documented rather than papered over.
    """
    # The loss scales as |x| * 1e-16 / distance, not 1e-17 / distance: the
    # argument error in `sin(pi x)` grows with |x|. Pinned at several poles
    # because the old single hard-coded -7.0 passed with 2.3x margin and would
    # have failed at -25 or beyond on its own tolerance.
    x = pole + distance
    np.testing.assert_allclose(sp.gamma(x), scipy_gamma(x), rtol=abs(pole) * 1e-8)


# ---------------------------------------------------------------------------
# eval_gegenbauer


@given(
    n=st.integers(min_value=0, max_value=20),
    # alpha > -1/2 is the classical parameter range; the recurrence is unstable
    # at and below -1/2, where the weight (1-x^2)^(alpha-1/2) stops being
    # integrable.
    alpha=floats(-0.49, 10.0),
    x=floats(-1.0, 1.0),
)
@example(n=0, alpha=1.0, x=1.0)
@example(n=1, alpha=1.0, x=1.0)
@example(n=2, alpha=1.0, x=1.0)
def test_eval_gegenbauer(n, alpha, x):
    """Measured worst case needs atol 1.9e-12 at rtol 1e-10 (values near roots)."""
    # scipy >= 1.18 returns 0.0 for `eval_gegenbauer(n, 0.0, x)` at *exactly*
    # alpha == 0, while returning 1.0 for alpha = 1e-300 and every other value --
    # a discontinuity at its own limit, and a change from 1.14, which returned
    # 1.0. C_0^(0) = 1 follows from the generating function, so `spexial` keeps
    # 1.0 and this one degenerate point is excluded rather than chased.
    assume(alpha != 0.0)
    np.testing.assert_allclose(
        sp.eval_gegenbauer(n, alpha, x),
        scipy_eval_gegenbauer(n, alpha, x),
        rtol=1e-10,
        # 1e-13, not 1e-9: the loose value never fired (max residual beyond rtol
        # is 0 over 420k samples) and would have hidden a 1000x error on any
        # result below 1e-9 -- the near-root values it was meant to protect.
        atol=1e-13,
    )


@given(
    n=st.integers(min_value=0, max_value=20),
    alpha=floats(-0.49, 10.0),
    x=floats(-1.0, 1.0),
)
def test_eval_gegenbauers(n, alpha, x):
    """The all-orders variant agrees with scipy at every order it returns.

    `eval_gegenbauers` has no scipy counterpart as a whole, but each element of
    its output does: entry `k` must equal `eval_gegenbauer(k, alpha, x)`.
    Checking against scipy rather than against our own `eval_gegenbauer` is the
    point -- the two share a recurrence, so a self-consistency test would pass
    with both of them wrong.
    """
    # scipy >= 1.18 returns 0.0 for `eval_gegenbauer(n, 0.0, x)` at *exactly*
    # alpha == 0, while returning 1.0 for alpha = 1e-300 and every other value --
    # a discontinuity at its own limit, and a change from 1.14, which returned
    # 1.0. C_0^(0) = 1 follows from the generating function, so `spexial` keeps
    # 1.0 and this one degenerate point is excluded rather than chased.
    assume(alpha != 0.0)
    got = sp.eval_gegenbauers(n, alpha, x)
    expected = [scipy_eval_gegenbauer(k, alpha, x) for k in range(n + 1)]
    assert got.shape == (n + 1,)
    np.testing.assert_allclose(got, expected, rtol=1e-10, atol=1e-13)


# ---------------------------------------------------------------------------
# kn


_KN_REFERENCE = (scipy_k0, scipy_k1, lambda z: scipy_kn(2, z))


@pytest.mark.parametrize("order", [0, 1, 2])
@given(z=floats(8.0, 10.0))
def test_kn_across_the_crossover(order, z):
    """The z = 9 hand-off between the two series, where the error actually peaks.

    `test_kn` justifies its 1e-6 tolerance by this cross-over and then never
    visits it: over its `floats(1e-30, 690)` range, 5000 draws landed in
    [8.9, 9.1] zero times. Without this test a regression that made the
    asymptotic branch 100x worse would still pass.
    """
    func = (sp.K0, sp.K1, sp.K2)[order]
    np.testing.assert_allclose(func(z), _KN_REFERENCE[order](z), rtol=1e-6)


@pytest.mark.parametrize("order", [0, 1, 2])
@given(z=floats(1e-30, 690.0))
def test_kn(order, z):
    """~8e-8 worst case, at the z = 9 cross-over between the two series.

    That is the accuracy the truncated series can deliver (30 ascending terms,
    10 asymptotic ones), so 1e-6 is the honest tolerance -- not machine
    precision.
    """
    func = (sp.K0, sp.K1, sp.K2)[order]
    np.testing.assert_allclose(func(z), _KN_REFERENCE[order](z), rtol=1e-6)


# ---------------------------------------------------------------------------
# zeta
#
# Restricted to n > 1 and to the negative integers: 0 < n <= 1 is not
# implemented by `jax.scipy.special.zeta`, and negative non-integers are not
# reachable from the Bernoulli functional equation. See `spexial.zeta`.


@given(n=floats(1.0001, 60.0))
def test_zeta_positive(n):
    """Delegated to JAX; measured worst case 6.7e-16 relative."""
    np.testing.assert_allclose(sp.zeta(n), scipy_zeta(n), rtol=1e-12)


@given(n=st.integers(min_value=-59, max_value=0))
def test_zeta_negative_integers(n):
    """From exact Bernoulli numbers; measured worst case 9.6e-15 relative."""
    np.testing.assert_allclose(
        sp.zeta(float(n)), scipy_zeta(float(n)), rtol=1e-12, atol=1e-300
    )


# ---------------------------------------------------------------------------
# Li -- no scipy counterpart, so mpmath supplies the reference values.


@given(
    # Up to 20, matching the range the tolerance below was measured over. The
    # `j ** n` int64 overflow that used to break `Li` starts at n = 12, so a
    # strategy stopping at 8 could not have caught it.
    n=st.integers(min_value=1, max_value=20),
    z=floats(-1000.0, 1000.0),
)
@example(n=1, z=2.0)
@example(n=2, z=-2.0)
@example(n=3, z=0.5)
def test_li(n, z):
    """Measured worst case 5.7e-13 relative over |z| <= 1000, 1 <= n <= 20."""
    assume(not (n == 1 and abs(z - 1.0) < 1e-9))  # Li_1(1) is the pole
    with mp.workdps(30):
        expected = complex(mp.polylog(n, z)).real
    np.testing.assert_allclose(sp.Li(n, z), expected, rtol=1e-11, atol=1e-12)
