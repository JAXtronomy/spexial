"""Binomial coefficients."""

__all__ = ["comb"]

from typing import Final

import jax.numpy as jnp
from jax.scipy.special import betaln, gammaln

from .custom_types import AnyArray, AnyArrayLike
from .dtype import as_float, cast_like, exactly_zero, is_negative

_BETA_FROM: Final = 1000.0
"""Above this ``N``, the Beta form replaces the log-gamma difference, in float64.

Measured against `mpmath.binomial` over a grid of ``k`` at each ``N``: the
log-gamma difference is worst 1.3e-13 at ``N = 170``, 6.6e-13 at 600 and
1.6e-12 at 1000, while the Beta form improves the other way -- 1.6e-9 at 100,
7.3e-12 at 600, 1.7e-12 at 1000, and ~1e-14 from there up. They cross here, so
the step in value across the crossover is a few times 1e-13.
"""

_BETA_FROM_LOW_PRECISION: Final = 10.0
"""The same cross-over, for float32 and narrower.

It moves because the two error sources scale differently. The Beta form's own
error is a property of `jax.scipy.special.betaln` and is ~1e-9 at ``N = 100``
whatever the dtype -- invisible under float32's 1.2e-7 eps, decisive under
float64's 2.2e-16. The log-gamma cancellation grows with ``N`` in *units of
eps*, so it bites 10^9 times sooner in float32. Measured in float32, the Beta
form is the better of the two from ``N = 10`` up, by 3x at 20 and 400x by 1000,
where the log-gamma difference reaches 4.5e-4 -- four decades worse than the
dtype can do, and the reason a single constant could not serve both.
"""


def comb(N: AnyArrayLike, k: AnyArrayLike, /) -> AnyArray:
    """Compute the number of combinations of ``N`` things taken ``k`` at a time.

    This is the "N choose k" factor :math:`N! / (k! (N-k)!)`. It is the
    *inexact* variant of `scipy.special.comb` -- the ``exact=False`` one -- and
    is evaluated through `jax.scipy.special.gammaln`, so ``N`` and ``k`` need
    not be integers and the result is a float, not a Python `int`.

    Reference:
    https://docs.scipy.org/doc/scipy/reference/generated/scipy.special.comb.html

    Parameters
    ----------
    N
        Number of things. Broadcast against ``k``.
    k
        Number of elements taken. Broadcast against ``N``.

    Returns
    -------
    Array
        The total number of combinations. Following `scipy.special.comb`, this
        is ``0`` wherever ``k > N``, ``k < 0``, or ``N < 0``.

    Examples
    --------
    >>> import jax.numpy as jnp
    >>> import spexial as sp

    >>> round(float(sp.comb(5, 2)), 12)
    10.0

    Out-of-range ``k`` gives 0, not `nan`:

    >>> [
    ...     round(float(c), 12)
    ...     for c in sp.comb(jnp.asarray([5, 5, 5]), jnp.asarray([2, 6, -1]))
    ... ]
    [10.0, 0.0, 0.0]

    Non-integer arguments are the generalized binomial coefficient:

    >>> round(float(sp.comb(5.5, 2)), 12)
    12.375

    """
    # Computed one width up for `float16`/`bfloat16` and rounded back: the
    # log-gamma difference needs more digits than either carries, and returned
    # `1.0` for a true 124750 at bfloat16 `N = 500`. See `spexial._src.dtype`.
    n_arr, k_arr = as_float(N, keep_weak=True), as_float(k, keep_weak=True)
    # The dtype that governs everything below is the one the two arguments
    # *promote to*, not `N`'s. `as_float(1000, keep_weak=True)` is a weak
    # float64, which defers to a float32 `k` for every subsequent operation --
    # so reading `eps` and `tiny` off `n_arr` alone picked float64's constants
    # for a computation running in float32, and `comb(1000, float32(6))` came
    # out 205x worse than the same call with both arguments float32.
    # Two different dtypes, and conflating them is what went wrong before.
    # `compute_dtype` is what the arithmetic actually runs in, and both
    # arguments are widened to it -- otherwise a `(float32, float64)` call
    # evaluated `gammaln` on the float32 half and returned a float64 carrying
    # float32's error. `out_dtype` is what the *caller's* two arguments promote
    # to, which is where the answer has to be narrowed back to; it is not the
    # same, because `as_float` has already widened `float16` to float32.
    #
    # Cast only when it changes something. `astype` strips weak typing even
    # when the dtype already matches, and a strong float64 from `comb(n, i)`
    # with two Python ints promotes `polylog`'s complex64 scan carry to
    # complex128 -- an error naming neither function, and the second time that
    # exact regression has been introduced here.
    compute_dtype = jnp.result_type(n_arr, k_arr)
    if n_arr.dtype != compute_dtype:
        n_arr = n_arr.astype(compute_dtype)
    if k_arr.dtype != compute_dtype:
        k_arr = k_arr.astype(compute_dtype)
    out_dtype = jnp.result_type(jnp.asarray(N), jnp.asarray(k))
    # `is_negative` rather than `< 0`: XLA compares a subnormal as if it were
    # zero, so `k >= 0` was True for `k = -1e-40` and the guard let it through
    # as `C(N, 0) = 1` where SciPy gives 0.
    in_domain = ~is_negative(n_arr) & ~is_negative(k_arr) & (k_arr <= n_arr)
    # Mask *before* the gammaln call: gammaln of a non-positive integer is
    # +inf, and inf - inf would give nan rather than the 0 scipy returns.
    n_safe = jnp.where(in_domain, n_arr, 0.0)
    k_safe = jnp.where(in_domain, k_arr, 0.0)
    # Two forms, because neither is best everywhere, and the crossover is where
    # they measure equal rather than at any round number.
    #
    # The log-gamma difference wins below N of about 1000 -- 1.6e-12 at worst
    # there against the Beta form's 1.6e-9 at N = 100 -- but it cancels
    # catastrophically above it: the two log-gammas converge as N grows, their
    # difference is 2.8e-7 relative by N = 1e8 and exactly 0 by N = 1e16, where
    # the result collapses to a plausible-looking `1.0`.
    #
    # `comb(N, k) = 1 / ((N + 1) B(N - k + 1, k + 1))` has no such subtraction.
    # It is the weaker of the two on small N, where the Beta function's own
    # argument reduction costs digits, and holds at ~1e-14 from N = 1000 to
    # `DBL_MAX`, given the subnormal correction below.
    eps = float(jnp.finfo(compute_dtype).eps)
    beta_from = _BETA_FROM if eps < 1e-10 else _BETA_FROM_LOW_PRECISION
    log_comb = gammaln(n_safe + 1) - gammaln(k_safe + 1) - gammaln(n_safe - k_safe + 1)
    # `jax.scipy.special.betaln` orders its arguments and forms `small / big`.
    # XLA on CPU flushes that quotient to zero as soon as it is subnormal, and
    # the term it feeds, `(big + small - 0.5) * log1p(small / big)`, is worth
    # `small`. Dropping it makes `comb` a factor of `e**-small` low -- 86% for
    # `comb(N, 1)`, where `small` is 2. The quotient goes subnormal once
    # `small < big * tiny`, which for `k = 1` is `N > 2**1023`.
    #
    # That same limit is where `log1p(h) == h` to the last bit, so the whole of
    # `algdiv` collapses to its leading term and `-log B = small * log(big) -
    # lgamma(small)` is exact -- checked against `mpmath` at 400 digits, 0 ulp
    # at `N = DBL_MAX`, integer and non-integer `k` alike.
    left, right = n_safe - k_safe + 1, k_safe + 1
    # `where`, not `jnp.minimum`/`maximum`, and the difference is entirely in
    # the transpose. `min`'s reverse rule multiplies the cotangent by a 0/1
    # selector, and at large `N` that cotangent has already overflowed to `inf`,
    # so `0 * inf` made `jax.grad(comb)` `nan` from about `N = 1e306` -- and
    # from `1e154` at `k = 2` -- while the value and `jacfwd` were both correct.
    # A `where` transposes as a select and never forms the product.
    swap = left < right
    small = jnp.where(swap, left, right)
    big = jnp.where(swap, right, left)
    # The asymptotic is taken wherever it is *exact*, not merely where the
    # subnormal flush forces it. `log B = lgamma(small) - small*log(big)` drops
    # a correction of order `small*(small-1)/(2*big)`, so once that is below an
    # eps the two forms agree bit for bit -- verified from `N = 1e16` up.
    #
    # Widening it that far is what makes the *derivative* right. Differentiating
    # `betaln` inherits a defect in `jax.scipy.special.betaln`'s own gradient,
    # which is exactly twice the truth from `a` of about 1e154 and `nan` above:
    # `jax.grad(comb)(1e154, 1.0)` was `3.0` against a true `1.0`, and `nan`
    # past that, while the value stayed correct to `DBL_MAX`. This branch has no
    # `betaln` in it, so it differentiates cleanly, and it now covers the whole
    # band where that defect lives.
    eps = jnp.finfo(compute_dtype).eps
    flushed = (small < big * jnp.finfo(compute_dtype).tiny) | (
        small * (small - 1) < 2 * big * eps
    )
    neg_log_beta = jnp.where(
        flushed, small * jnp.log(big) - gammaln(small), -betaln(left, right)
    )
    from_beta = neg_log_beta - jnp.log(n_safe + 1)
    out = jnp.where(
        in_domain, jnp.exp(jnp.where(n_arr > beta_from, from_beta, log_comb)), 0.0
    )
    # `gammaln(inf) - gammaln(inf) - ...` is `inf - inf == nan`; the limit is
    # plainly +inf and scipy returns that -- except at k = 0, where C(N, 0) = 1
    # for every N including infinity, as scipy also returns.
    # C(inf, inf) is indeterminate, and `nan` is what scipy returns for it.
    at_infinity = jnp.where(
        exactly_zero(k_arr), 1.0, jnp.where(jnp.isinf(k_arr), jnp.nan, jnp.inf)
    )
    # Narrowed to what `N` and `k` promote to, not to `N`. `comb` is the only
    # two-argument entry point here, and casting to `N` alone returned float32
    # from a `(float32, float64)` call -- the *narrower* of the two, which is
    # the opposite of what JAX's promotion entitles the caller to.
    out = jnp.where(in_domain & jnp.isinf(n_arr), at_infinity, out)
    return cast_like(out, jnp.zeros((), out_dtype))
