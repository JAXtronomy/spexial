r"""The incomplete beta function.

Note that this module is NOT public API; only the name re-exported from the
top-level `spexial` namespace is.

The *unregularized* incomplete beta function

.. math::

    B(a, b, z) = \int_0^z t^{a-1}(1-t)^{b-1}\,\mathrm{d}t

(DLMF 8.17.1). SciPy and JAX both provide only the **regularized** one,
:math:`I_z(a, b)`, under the name ``betainc``; the two differ by a factor of
the complete beta function :math:`B(a, b)`.

Why not just multiply
---------------------

Because ``beta(a, b) * betainc(a, b, z)`` is `nan` for every :math:`b \le 0`.
:math:`B(a, b)` has a pole there while the *product* is perfectly finite, so
the reconstruction loses a domain the function itself has no trouble with --
and :math:`b \le 0` is not exotic, it is an ordinary slope in the double
power-law density profiles this was written for.

>>> import jax.numpy as jnp
>>> import jax.scipy.special as jsp
>>> bool(jnp.isnan(jsp.beta(2.0, 0.0) * jsp.betainc(2.0, 0.0, 0.5)))
True

`jax.scipy.special.hyp2f1` *can* express it for any ``b`` (DLMF 8.17.7), but it
is a `jax.lax.while_loop` whose trip count depends on its data: under
`jax.vmap` every lane pays the worst lane's iteration count, and its derivative
runs a second such loop. What is here instead is two fixed-length,
geometrically convergent series with an exact O(1) derivative rule.

Contributed from `galax <https://github.com/GalacticDynamics/galax>`_, where it
was written for the Zhao (1996) family of density profiles (Eq. 43).

"""

__all__ = ["incomplete_beta"]

import functools as ft

import jax
import jax.numpy as jnp
from jax.custom_derivatives import SymbolicZero

from .custom_types import AnyArray, AnyArrayLike, ScalarLike

_NTERMS = 64
"""Series length. Both series below converge like 2^-k, so this is ~1e-19.

Not worth trimming: measured over a in [0.2, 8], b in [-2.5, 6] and z up to
1 - 1e-8, dropping to 48 terms costs an order of magnitude of accuracy at
moderate `a` (5e-14 -> 6e-13) and 40 breaks the 1e-11 the tests assert, to
save a fraction of a loop that `_UNROLL` already cut four-fold. Above
a ~ 16 the error stops improving with term count at all -- it is cancellation
between large alternating terms, not truncation -- so more terms would not
help there either.
"""

_UNROLL = 16
"""How far to unroll the series loops.

Worth 4x: at 1e5 points the small-z series goes 10.0 ms -> 2.5 ms, for +0.1 s
of compile time. Full unrolling (64) buys a further 15% for 2x the compile,
and a trace-time Python loop is no faster than this while compiling worse.
"""

_POLE_BAND = 1e-3
"""Half-width of the band around `b + m == 0` where `_large_z` expands.

Below this the direct form cancels; above it the expansion's O(s^4 L^5)
truncation shows, for `L = log(1-z)`.
"""


def _small_z(a: ScalarLike, b: ScalarLike, z: AnyArrayLike) -> AnyArray:
    r"""$B(a, b, z)$ for $z \leq 1/2$, by the defining Taylor series.

    Expanding $(1-t)^{b-1}$ binomially and integrating term by term,

    .. math::

        B(a, b, z) = z^a \sum_{k=0}^\infty \frac{(1-b)_k}{k!\,(a+k)} z^k

    The terms fall off like $z^k \leq 2^{-k}$, hence the fixed term count.

    Summed into a `jax.lax.scan` carry: the terms are batched over `z`, so
    materializing them all at once would cost a ``(*batch, 64)`` temporary and
    make this memory- rather than flop-bound.

    `a`, `b` and `z` are closed over rather than carried. Threading them
    through the carry (or `xs`) instead measures the same to within noise and
    returns bit-identical values -- JAX turns a closed-over tracer into a
    constant of the scan's jaxpr, so there is nothing there to hoist.
    """

    def step(
        carry: tuple[AnyArray, AnyArray, ScalarLike], k: ScalarLike
    ) -> tuple[tuple[AnyArray, AnyArray, ScalarLike], None]:
        total, z_pow, coeff = carry  # coeff = (1-b)_k / k!
        total = total + coeff / (a + k) * z_pow
        return (total, z_pow * z, coeff * (k + 1.0 - b) / (k + 1.0)), None

    init = (jnp.zeros_like(z), jnp.ones_like(z), jnp.ones_like(a))
    (total, _, _), _ = jax.lax.scan(step, init, jnp.arange(_NTERMS), unroll=_UNROLL)
    return z**a * total  # type: ignore[no-any-return]


def _large_z(a: ScalarLike, b: ScalarLike, z: AnyArrayLike) -> AnyArray:
    r"""$B(a, b, z)$ for $z > 1/2$, by reflecting about $t = 1/2$.

    Substituting $t = 1-u$ and splitting the range at $u = 1/2$,

    .. math::

        B(a, b, z) = \int_{w}^{1} u^{b-1}(1-u)^{a-1} du
                   = B(a, b, 1/2) + \int_w^{1/2} u^{b-1}(1-u)^{a-1} du

    with $w = 1 - z$. Expanding $(1-u)^{a-1}$ binomially (legitimate since
    $u \leq 1/2$ on the remaining range) and integrating term by term,

    .. math::

        B(a, b, z) = B(a, b, 1/2)
            + \sum_{m=0}^\infty \frac{(1-a)_m}{m!}
              \frac{(1/2)^{b+m} - w^{b+m}}{b+m}

    where the $b + m \to 0$ term is $\ln(1/(2w))$. Both pieces converge like
    $2^{-m}$.

    This branch is what makes $b \leq 0$ work: it never forms the complete beta
    function $B(a, b)$, which is what diverges there, while keeping the genuine
    $z \to 1$ divergence of $B(a, b, z)$ itself exact (as $w^b$, or $-\ln w$
    when $b = 0$).

    Summed into a `jax.lax.scan` carry, for the same reason as `_small_z`.

    Each term is $(e^{sL_1} - e^{sL_2})/s$ with $s = b + m$, $L_1 = \ln(1/2)$
    and $L_2 = \ln w$, which the running powers below evaluate directly. That
    cancels catastrophically as $s \to 0$, so within `_POLE_BAND` of zero it
    switches to the expansion of the same expression,

    .. math::

        \frac{e^{sL_1} - e^{sL_2}}{s}
            = \sum_{k \geq 1} \frac{L_1^k - L_2^k}{k!} s^{k-1}

    whose $L$-powers are loop-invariant, so this costs a few extra FMAs rather
    than transcendentals. Carrying the $O(s)$ term also makes the
    $b$-derivative right *at* $s = 0$, which integer slopes do hit (e.g.
    $\gamma = 2$ in Zhao's Eq. 7).
    """
    w = 1.0 - z
    log_half, log_w = jnp.log(0.5), jnp.log(w)
    # L1^k - L2^k over k!, for the near-pole expansion.
    d1 = log_half - log_w
    d2 = (log_half**2 - log_w**2) / 2
    d3 = (log_half**3 - log_w**3) / 6
    d4 = (log_half**4 - log_w**4) / 24

    def step(
        carry: tuple[AnyArray, AnyArray, ScalarLike, ScalarLike], m: ScalarLike
    ) -> tuple[tuple[AnyArray, AnyArray, ScalarLike, ScalarLike], None]:
        total, w_pow, half_pow, coeff = carry  # coeff = (1-a)_m / m!
        s = b + m
        near_pole = jnp.abs(s) < _POLE_BAND
        s_safe = jnp.where(near_pole, 1.0, s)
        term = jnp.where(
            near_pole,
            d1 + s * (d2 + s * (d3 + s * d4)),
            (half_pow - w_pow) / s_safe,
        )
        total = total + coeff * term
        carry = (total, w_pow * w, half_pow * 0.5, coeff * (m + 1.0 - a) / (m + 1.0))
        return carry, None

    # `_small_z(a, b, 1/2)` below looks like a scalar being recomputed per
    # point, but computing it as a scalar and broadcasting is *slower*: XLA
    # already folds the constant-array input, and doing it by hand breaks the
    # fusion (measured 0.80x).
    init = (jnp.zeros_like(w), w**b, 0.5**b, jnp.ones_like(a))
    (total, _, _, _), _ = jax.lax.scan(step, init, jnp.arange(_NTERMS), unroll=_UNROLL)
    return _small_z(a, b, jnp.full_like(w, 0.5)) + total  # type: ignore[no-any-return]


_CF_FROM = 10.0
r"""Above this ``b``, `_large_b` replaces the two series.

Both series expand :math:`(1-t)^{b-1}` binomially, and its coefficients
:math:`(1-b)_k / k!` alternate and grow like :math:`b^k / k!`: at
:math:`z = 1/2` the sum is a near-total cancellation, worst relative error
:math:`2\times10^{-15}` at :math:`b = 10`, :math:`10^{-11}` at 30, and
:math:`10^{7}` at 100. The continued fraction is :math:`5\times10^{-14}` from
:math:`b = 0.1` up, so the switch can sit wherever the series is still exact.
"""

_CF_UNROLL = 1
"""Not unrolled, unlike the series.

`lax.cond` traces both branches, so this is compiled on every call, series or
not. Unrolled 16-fold it made a second ``b``-derivative take 200 s to compile
for no runtime gain (measured 39 vs 41 ms at 1e5 points); unrolled once, 11 s.
"""

_CF_STEPS = 64
r"""Double steps of the continued fraction (each an even and an odd term).

The slowest case is the reflected fraction -- large first parameter -- just
above its switch point, at small ``a``: at :math:`a = 0.1, b = 10^3` it is
:math:`10^{-11}` at 32 steps, :math:`5\times10^{-14}` at 48 and
:math:`6\times10^{-15}` at 64, past which nothing changes. What remains is
rounding, growing like :math:`b\,\epsilon`: worst over
:math:`a \in [0.1, 50]`, :math:`4\times10^{-12}` at :math:`b = 10^5`,
:math:`5\times10^{-10}` at :math:`10^7`.
"""


def _continued_fraction(a: AnyArrayLike, b: AnyArrayLike, x: AnyArray) -> AnyArray:
    r"""Evaluate the DLMF 8.17.22 continued fraction by the modified Lentz method.

    .. math::

        B(a, b, x) = \frac{x^a (1-x)^b}{a} \cdot
            \cfrac{1}{1 + \cfrac{d_1}{1 + \cfrac{d_2}{1 + \cdots}}}

    with :math:`d_{2m+1} = -\frac{(a+m)(a+b+m)x}{(a+2m)(a+2m+1)}` and
    :math:`d_{2m} = \frac{m(b-m)x}{(a+2m-1)(a+2m)}`. Returns the fraction,
    :math:`1/(1 + d_1/(1 + \cdots))`. It converges fast for
    :math:`x < (a+1)/(a+b+2)`, and its terms are bounded for any :math:`b > 0`,
    so -- unlike the series -- nothing cancels as :math:`b` grows. A fixed
    number of steps, as for the series, so `vmap` lanes do not wait on each
    other.
    """
    # Lentz's guard against a zero denominator: the dtype's smallest normal, as
    # a Python float so that it does not promote the carry.
    tiny = float(jnp.finfo(x.dtype).tiny)
    guard = lambda v: jnp.where(jnp.abs(v) < tiny, tiny, v)  # noqa: E731

    def step(
        carry: tuple[AnyArray, AnyArray, AnyArray], m: ScalarLike
    ) -> tuple[tuple[AnyArray, AnyArray, AnyArray], None]:
        c, d, h = carry
        even = m * (b - m) * x / ((a + 2.0 * m - 1.0) * (a + 2.0 * m))
        odd = -(a + m) * (a + b + m) * x / ((a + 2.0 * m) * (a + 2.0 * m + 1.0))
        for coeff in (even, odd):
            d = 1.0 / guard(1.0 + coeff * d)
            c = guard(1.0 + coeff / c)
            h = h * d * c
        return (c, d, h), None

    d = 1.0 / guard(1.0 - (a + b) * x / (a + 1.0))
    init = (jnp.ones_like(x), d, d)
    (_, _, h), _ = jax.lax.scan(
        step, init, jnp.arange(1, _CF_STEPS + 1), unroll=_CF_UNROLL
    )
    return h  # type: ignore[no-any-return]


def _complete_beta(a: ScalarLike, b: ScalarLike) -> AnyArray:
    r"""$\log B(a, b)$ for $a, b > 0$, as the two fractions at the switch point.

    :math:`B(a, b) = B(a, b, x_0) + B(b, a, 1 - x_0)`, two positive terms, so no
    cancellation. Not ``betaln``: `jax.scipy.special.betaln` is only good to
    :math:`3\times10^{-7}` at :math:`(a, b) = (8, 10)`. Returned as a logarithm,
    so that a complete beta function far below the smallest normal number --
    :math:`B(301, 1000) \approx 4\times10^{-307}` -- does not round to zero before
    it is used. Computed once per call, not per point, so `vmap` over ``z`` does
    not repeat it.
    """
    x0 = (a + 1.0) / (a + b + 2.0)
    # The front factor x^a (1-x)^b is the same for both terms, and is formed from
    # x0 itself: `log1p(-x0)`, not `log` of a rounded `1 - x0` scaled by b.
    log_front = a * jnp.log(x0) + b * jnp.log1p(-x0)
    h = _continued_fraction(
        jnp.stack([a, b]),
        jnp.stack([b, a]),
        jnp.stack([x0, 1.0 - x0]),
    )
    return log_front + jnp.log(h[0] / a + h[1] / b)  # type: ignore[no-any-return]


def _large_b(a: ScalarLike, b: ScalarLike, z: AnyArray) -> AnyArray:
    r"""$B(a, b, z)$ for $b > 0$, by the continued fraction and reflection.

    Below the switch point :math:`x_0 = (a+1)/(a+b+2)` the fraction converges
    as it stands; above it, by the reflection
    :math:`B(a, b, z) = B(a, b) - B(b, a, 1-z)`, where it converges for the
    swapped arguments. One fraction per point, with the arguments swapped where
    reflected; the complete :math:`B(a, b)` comes from `_complete_beta`.

    Each term is :math:`\exp(\log(\text{front}) + \log(h / p))`, with
    :math:`p` the fraction's first parameter: the front factor
    :math:`z^a (1-z)^b` is the same either way round and is formed from ``z``,
    never from a rounded ``1 - z`` scaled by ``b``; and the fraction is folded
    into the exponent, so a front factor below the smallest normal number does
    not flush to zero while the term itself is representable.
    """
    if jnp.dtype(z.dtype).itemsize < 4:
        # Half precision has too few bits for Lentz's recurrence to settle; work
        # in float32 and round the result once.
        return _large_b(a, b, z.astype(jnp.float32)).astype(z.dtype)
    x0 = (a + 1.0) / (a + b + 2.0)
    swap = z >= x0
    # z = 0 and z = 1 make the term exactly zero (the direct one at 0, the
    # reflected one at 1). Evaluated, `log(0)` would put `0 * -inf` into the
    # a- and b-derivatives; it is a genuine zero for every a, b > 0, so its
    # parameter derivatives are zero too.
    endpoint = (z == 0) | (z == 1)
    zs = jnp.where(endpoint, 0.5, z)
    log_front = a * jnp.log(zs) + b * jnp.log1p(-zs)
    first = jnp.where(swap, b, a)
    h = _continued_fraction(first, jnp.where(swap, a, b), jnp.where(swap, 1.0 - zs, zs))
    term = jnp.where(endpoint, 0.0, jnp.exp(log_front + jnp.log(h / first)))
    reflected = jnp.exp(_complete_beta(a, b)) - term
    out = jnp.where(swap, reflected, term)
    # Past b ~ 1/eps the switch point x0 is below eps, `1 - x0` rounds to 1, and
    # the reflected fraction's first denominator is exactly zero. Long before
    # that the b*eps rounding has taken every digit (0.1 relative by b ~ 1e15),
    # so say so with `nan` rather than return a confident `inf` or `0`.
    return jnp.where(1.0 - x0 == 1.0, jnp.nan, out)  # type: ignore[no-any-return]


def _series(a: ScalarLike, b: ScalarLike, z: AnyArray) -> AnyArray:
    """$B(a, b, z)$ by `_small_z` and `_large_z`, switched at $z = 1/2$."""
    # For b <= 0 the integral diverges at z = 1: a genuine pole, +inf (GH-67).
    # `_large_z` cannot see it -- at w = 1 - z = 0 its expansion is `inf - inf`
    # -- so the pole is returned directly, and the series is handed a harmless
    # z there, so that its `nan` cannot reach a cotangent through the `where`.
    pole = (z == 1.0) & (b <= 0)
    z = jnp.where(pole, 0.5, z)
    # Both branches are evaluated, so clamp each one's input to the range where
    # it is well behaved; `where` then discards the unused value.
    value = jnp.where(
        z <= 0.5,
        _small_z(a, b, jnp.minimum(z, 0.5)),
        _large_z(a, b, jnp.maximum(z, 0.5)),
    )
    return jnp.where(pole, jnp.inf, value)  # type: ignore[no-any-return]


_CF_A_FROM = 8.0
r"""Above this ``a`` (with ``b >= _CF_A_B_MIN``), `_large_b` replaces the series too.

`_large_z` expands :math:`(1-u)^{a-1}`, so the same cancellation that `_CF_FROM`
avoids in :math:`b` happens in :math:`a`: measured worst over
:math:`b \in [-2.5, 10]`, :math:`6\times10^{-14}` at :math:`a = 8`,
:math:`4\times10^{-11}` at 16, :math:`2\times10^{-8}` at 24, and wrong from
about 50. The continued fraction is :math:`5\times10^{-13}` or better there,
up to :math:`a = 100`.
"""

_CF_A_B_MIN = 0.1
r"""The smallest ``b`` for which large ``a`` is routed to `_large_b`.

The reflection subtracts from the complete :math:`B(a, b) \sim 1/b`, so it
loses about :math:`\log_{10}(1/b)` digits as :math:`b \to 0^+`: measured
:math:`3\times10^{-12}` at :math:`b = 10^{-3}`, :math:`10^{-8}` at
:math:`10^{-8}`. From 0.1 it is :math:`4\times10^{-13}` or better to
:math:`a = 50`. Below it -- including all :math:`b \le 0`, where there is no
reflection at all -- large ``a`` stays on the series, with its known limit.
"""


def _use_cf(a: ScalarLike, b: ScalarLike) -> ScalarLike:
    """Whether `_large_b` rather than `_series` evaluates ``B(a, b, z)``."""
    return (b > _CF_FROM) | ((a > _CF_A_FROM) & (b >= _CF_A_B_MIN))


_SERIES, _CONTINUED_FRACTION, _EITHER = 0, 1, 2
"""Which evaluation `_incomplete_beta_core` runs; static, so it is compiled alone.

`_EITHER` is for a traced ``a`` or ``b``, whose branch is only known at run time.
"""


def _branch(a: ScalarLike, b: ScalarLike) -> int:
    """Pick the static branch for concrete ``a`` and ``b``, else `_EITHER`.

    Decided in the public wrapper, before `jax.custom_jvp` lifts every argument
    to a tracer: inside it, even a literal ``b`` is traced under `jit`, so a
    choice made there would always compile both branches.
    """
    if isinstance(a, jax.core.Tracer) or isinstance(b, jax.core.Tracer):
        return _EITHER
    return _CONTINUED_FRACTION if bool(_use_cf(a, b)) else _SERIES


def _incomplete_beta_impl(
    a: ScalarLike, b: ScalarLike, z: AnyArrayLike, branch: int
) -> AnyArray:
    r"""$B(a, b, z)$ for $a > 0$, any real $b$, and $z \in [0, 1]$."""
    z = jnp.asarray(z)
    if not jnp.issubdtype(z.dtype, jnp.inexact):
        # An integer `z` would otherwise carry its dtype into the continued
        # fraction's switch point, (a+1)/(a+b+2), and truncate it to zero.
        z = z.astype(jnp.promote_types(z.dtype, float))
    # Promote `z` to the common type of all three, as any elementwise op would
    # (GH-66): a float32 `z` with a strong float64 `b` otherwise starts a `scan`
    # carry in float32 that the first step promotes to float64, which `scan`
    # rejects. Weak (Python) scalars do not promote, so the common case keeps
    # `z`'s own dtype and weak type.
    dtype = jnp.result_type(a, b, z)
    if dtype != z.dtype:
        z = z.astype(dtype)
    if branch == _SERIES:
        return _series(a, b, z)
    if branch == _CONTINUED_FRACTION:
        return _large_b(a, b, z)
    use_cf = _use_cf(a, b)
    # `a` and `b` are scalars, so `cond` runs one branch. Under `vmap` it becomes
    # a `select` that runs both, so each is handed parameters it is safe for --
    # the series is wildly wrong for large `a` or `b`, and the reflection needs
    # `b > 0` -- so that neither puts a non-finite value where autodiff would
    # multiply it by the zero cotangent of the unselected side.
    #
    # Each branch is checkpointed. Reverse-mode through a `cond` makes both
    # branches emit the same set of residuals, zero-filling the other's, so the
    # series branch would allocate the continued fraction's per-step carries
    # (64 x the points) just to discard them. Checkpointed, a branch's residuals
    # are its inputs. Measured on the a/b gradient at 1e5 points, b = 1.5: 63 ->
    # 46 ms (main, with no `cond`: 40), and the second-derivative compile 4.9 ->
    # 3.4 s.
    return jax.lax.cond(  # type: ignore[no-any-return]
        use_cf,
        lambda: jax.checkpoint(_large_b)(
            jnp.where(use_cf, a, 1.0), jnp.where(use_cf, b, 2.0 * _CF_FROM), z
        ),
        lambda: jax.checkpoint(_series)(
            jnp.where(use_cf, 1.0, a), jnp.where(use_cf, 1.0, b), z
        ),
    )


def incomplete_beta(a: ScalarLike, b: ScalarLike, z: AnyArrayLike, /) -> AnyArray:
    r"""Unregularized incomplete beta function :math:`B(a, b, z)`.

    .. math::

        B(a, b, z) = \int_0^z t^{a-1}(1-t)^{b-1}\,\mathrm{d}t

    the DLMF 8.17.1 form. There is no `scipy.special` counterpart: SciPy's
    ``betainc`` is the *regularized* :math:`I_z(a, b)`, and the two differ by
    the complete beta function :math:`B(a, b)`.

    Reconstructing this as ``beta(a, b) * betainc(a, b, z)`` works only for
    :math:`b > 0`. At :math:`b \le 0` the complete beta function has a pole
    while the product does not, so that route returns `nan` over a domain this
    function handles without difficulty.

    Parameters
    ----------
    a
        First parameter. Must be positive, and scalar.
    b
        Second parameter. **Any real value**, including zero and negative --
        which is the reason this function exists. Scalar.
    z
        Upper limit of integration, in :math:`[0, 1]`. Any shape.

    Returns
    -------
    Array
        Shaped like ``z``.

    See Also
    --------
    scipy.special.betainc : the regularized form, :math:`I_z(a, b)`.
    jax.scipy.special.hyp2f1 : expresses this via DLMF 8.17.7, at the cost of
        a data-dependent `jax.lax.while_loop`.

    Notes
    -----
    Two fixed-length series, switched at :math:`z = 1/2`, each converging like
    :math:`2^{-k}`; see `_small_z` and `_large_z`. Above :math:`b = 10`, or
    above :math:`a = 8` with :math:`b \ge 0.1`, their alternating terms cancel,
    and a fixed-length continued fraction is used instead; see `_large_b`. The
    :math:`z`-derivative is supplied by a `jax.custom_jvp` and is exact and
    O(1) -- by Leibniz it is just the integrand at the endpoint,
    :math:`z^{a-1}(1-z)^{b-1}` -- rather than differentiating through 64 terms.
    It is a `custom_jvp` rather than a `custom_vjp` so that `jax.hessian`'s
    ``jacfwd(jacrev(...))`` still composes.

    Examples
    --------
    >>> import jax.numpy as jnp
    >>> import jax.scipy.special as jsp
    >>> import spexial as sp

    It agrees with the regularized form wherever that is defined, including
    close to the :math:`z \to 1` endpoint:

    >>> a, b = 2.0, 1.5
    >>> z = jnp.asarray([0.3, 0.999])
    >>> bool(
    ...     jnp.allclose(
    ...         sp.incomplete_beta(a, b, z), jsp.beta(a, b) * jsp.betainc(a, b, z)
    ...     )
    ... )
    True

    But unlike that product it stays finite for ``b <= 0``:

    >>> round(float(sp.incomplete_beta(2.0, 0.0, jnp.asarray(0.5))), 8)
    0.19314718

    >>> bool(jnp.isnan(jsp.beta(2.0, 0.0) * jsp.betainc(2.0, 0.0, 0.5)))
    True

    """
    return _incomplete_beta_core(a, b, z, _branch(a, b))


@ft.partial(jax.custom_jvp, nondiff_argnums=(3,))
def _incomplete_beta_core(
    a: ScalarLike, b: ScalarLike, z: AnyArrayLike, branch: int
) -> AnyArray:
    """`incomplete_beta` for a given static branch, with its derivative rule."""
    return _incomplete_beta_impl(a, b, z, branch)


@ft.partial(_incomplete_beta_core.defjvp, symbolic_zeros=True)
def _incomplete_beta_jvp(
    branch: int,
    primals: tuple[ScalarLike, ScalarLike, AnyArray],
    tangents: tuple[
        ScalarLike | SymbolicZero, ScalarLike | SymbolicZero, AnyArray | SymbolicZero
    ],
) -> tuple[AnyArray, AnyArray]:
    a, b, z = primals
    a_dot, b_dot, z_dot = tangents

    # The a/b tangents are only needed when the parameters are themselves
    # differentiated; there is no cheap closed form, so fall back to autodiff of
    # the implementation. Skipped entirely in the common case, where the
    # differentiation is with respect to position at fixed a and b. When it does
    # run, the primal comes from the same `jvp`, not a second evaluation: with
    # `b` traced the implementation is a `lax.cond`, and XLA cannot merge two
    # copies of one, so a separate primal ran the whole thing twice (1.8x).
    a_zero, b_zero = isinstance(a_dot, SymbolicZero), isinstance(b_dot, SymbolicZero)
    if a_zero and b_zero:
        primal_out = _incomplete_beta_impl(a, b, z, branch)
        tangent_out = jnp.zeros_like(primal_out)
    else:
        primal_out, tangent_out = jax.jvp(
            lambda aa, bb: _incomplete_beta_impl(aa, bb, z, branch),
            (a, b),
            (
                jnp.zeros_like(a) if a_zero else a_dot,
                jnp.zeros_like(b) if b_zero else b_dot,
            ),
        )

    # d/dz B(a, b, z) = z^(a-1) (1-z)^(b-1): the integrand at the endpoint.
    if not isinstance(z_dot, SymbolicZero):
        if isinstance(b, jax.core.Tracer):
            # `b` may itself be differentiated, so write (1-z)^(b-1) so that its
            # `b`-derivative carries `log1p(-z)`, not `log(1 - z)` of a rounded
            # `1 - z`, which keeps only ~1e-16 / z relative precision (GH-68).
            # At z = 1 itself `pow` takes over, by a double `where`: `log1p(-z)` is
            # `-inf` there, and any clamp of it differentiates as `0 * -inf`,
            # making the second and third z-derivatives `nan`. `pow` has exact
            # derivatives at a zero base, as on the concrete path.
            z_arr = jnp.asarray(z)
            inside = z_arr < 1.0
            z_in = jnp.where(inside, z_arr, 0.0)
            via_log1p = jnp.exp((b - 1.0) * jnp.log1p(-z_in))
            one_minus_z_pow = jnp.where(inside, via_log1p, (1.0 - z_arr) ** (b - 1.0))
        else:
            # A concrete `b` has no derivative to get wrong, and as a constant
            # exponent XLA can simplify `pow` (to a reciprocal at b = 0, a square
            # root at b = 3/2), which `exp(... log1p)` would hide from it.
            one_minus_z_pow = (1.0 - z) ** (b - 1.0)
        tangent_out = tangent_out + z ** (a - 1.0) * one_minus_z_pow * z_dot

    return primal_out, tangent_out
