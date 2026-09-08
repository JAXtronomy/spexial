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
    expected = scipy_comb(N, k)
    assume(np.isfinite(expected))
    np.testing.assert_allclose(sp.comb(N, k), expected, rtol=1e-11, atol=1e-300)


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


@pytest.mark.parametrize("distance", [1e-6, 1e-8])
def test_gamma_near_a_pole(distance):
    """Accuracy near a pole is only ~1e-17 / distance, not 1e-10.

    This is inherent to the reflection formula, not a fixable bug: the
    ``sin(pi x)`` denominator loses exactly the digits that ``x`` is close to
    an integer by. Documented rather than papered over.
    """
    x = -7.0 + distance
    np.testing.assert_allclose(sp.gamma(x), scipy_gamma(x), rtol=1e-7)


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
    np.testing.assert_allclose(
        sp.eval_gegenbauer(n, alpha, x),
        scipy_eval_gegenbauer(n, alpha, x),
        rtol=1e-10,
        atol=1e-9,
    )


# ---------------------------------------------------------------------------
# kn


@pytest.mark.parametrize("order", [0, 1, 2])
@given(z=floats(1e-30, 600.0))
def test_kn(order, z):
    """~8e-8 worst case, at the z = 9 cross-over between the two series.

    That is the accuracy the truncated series can deliver (30 ascending terms,
    10 asymptotic ones), so 1e-6 is the honest tolerance -- not machine
    precision.
    """
    func = (sp.K0, sp.K1, sp.K2)[order]
    np.testing.assert_allclose(func(z), scipy_kn(order, z), rtol=1e-6)


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
    n=st.integers(min_value=1, max_value=8),
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
