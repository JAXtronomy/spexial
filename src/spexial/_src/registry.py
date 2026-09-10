"""The coverage registry: what upstream provides, and what `spexial` adds.

This is the central table of the library. Every function `spexial` exports has a
row here recording what `jax.scipy.special` and `scipy.special` provide, from
which version, and whether the upstream version is differentiable. The table
drives three decisions:

1. **Whether a function belongs in `spexial` at all.** A function upstream
   already covers everywhere we support is redundant.
2. **When to stop implementing it.** Once `spexial`'s own floor rises above a
   function's `jax_since`, the implementation can become a re-export, then be
   deprecated, then removed. `Status.REDUNDANT_ABOVE_FLOOR` marks the rows
   waiting on exactly that.
3. **Where a hand-written derivative is worth the code.** `custom_jvp` records
   where `spexial` defines an analytic derivative rather than letting JAX
   differentiate through a series.

The table is *checked*, not asserted: `tests/unit/test_registry.py` verifies
every `jax_*` field against the installed JAX, so a row cannot quietly go stale
when upstream adds a function. `docs/reference/coverage.md` is generated from
here by `scripts/gen_coverage_table.py`, and a test fails if it drifts.

Examples
--------
>>> from spexial.registry import REGISTRY, Status

>>> REGISTRY["k0"].status is Status.UNIQUE
True

>>> REGISTRY["comb"].jax_since
'0.10.2'

"""

__all__ = ["REGISTRY", "Coverage", "Status", "Support", "render_markdown"]

from dataclasses import dataclass
from enum import StrEnum
from types import MappingProxyType
from typing import Final


class Support(StrEnum):
    """How well an upstream implementation works on JAX arrays."""

    NONE = "none"
    """Not provided at all."""

    VALUE = "value"
    """Returns a value under tracing, but does not differentiate."""

    AUTODIFF = "autodiff"
    """Traceable and differentiable: both `jax.jvp` and `jax.vjp` work."""


class Status(StrEnum):
    """What `spexial` should do about a function, now and later."""

    UNIQUE = "unique"
    """No upstream equivalent on JAX. `spexial` owns this one."""

    EXTENDS = "extends"
    """Upstream exists but covers less -- a narrower domain, or no derivative."""

    REDUNDANT_ABOVE_FLOOR = "redundant-above-floor"
    """Upstream covers it, but only above `spexial`'s supported JAX floor.

    Re-export it once the floor rises to `jax_since`; deprecate a release later;
    remove the release after that.
    """

    DELEGATES = "delegates"
    """Upstream computes the value; `spexial` supplies a cheaper derivative.

    The value is obtained by calling upstream directly, so it cannot drift. The
    row earns its place only on the `cost` columns -- if upstream's own gradient
    ever matches on both speed and memory, it becomes `REDUNDANT`.
    """

    REDUNDANT = "redundant"
    """Upstream covers it everywhere we support, at no worse cost. Drop it."""


@dataclass(frozen=True, slots=True)
class Cost:
    """Measured cost of a gradient, relative to the alternative.

    Both are ratios of `spexial` to the alternative, so **below 1 means
    `spexial` wins**. `speed` is wall-clock for `jax.grad` over 10,000 points;
    `memory` is the bytes of residual the backward pass must keep alive,
    obtained from the first-class VJP object's leaves and therefore exact rather
    than sampled.

    Memory is the column that usually decides. A custom JVP replaces a whole
    series' worth of saved intermediates with a single array, and the saving is
    far larger than the speed-up: `k0` differentiates 1.5x faster but keeps 68x
    less residual.
    """

    speed: float | None
    """Gradient wall-clock, `spexial` / alternative. `None` if not comparable."""

    memory: float | None
    """Backward-pass residual bytes, `spexial` / alternative."""

    against: str
    """What the ratio is measured against."""


@dataclass(frozen=True, slots=True)
class Coverage:
    """One row: what upstream provides for a single function."""

    name: str
    """The name `spexial` exports."""

    jax_name: str | None
    """Equivalent in `jax.scipy.special`, or `None` if there is none."""

    jax_since: str | None
    """First JAX release providing it. `None` means never; `"*"` means at or
    before `spexial`'s current floor, so every supported JAX has it."""

    jax_support: Support
    """What the JAX equivalent can do."""

    scipy_array_api: Support
    """What `scipy.special` delivers *on JAX arrays* with `SCIPY_ARRAY_API=1`.

    Verified against scipy 1.18.1; scipy 1.14.1 provides `Support.NONE` for
    every entry, so this capability is recent. It is also opt-in -- with the
    environment variable unset, scipy converts to NumPy and fails under `jit`.
    """

    status: Status
    """What `spexial` should do about it."""

    custom_jvp: bool
    """Whether `spexial` defines a `jax.custom_jvp` today.

    `False` with a non-`None` `derivative` means the identity is known and
    wiring it up is available work, not that none exists.
    """

    derivative: str | None
    """The analytic derivative, where there is a usable closed form.

    Recorded even when `custom_jvp` is `False`, so the opportunity is visible.
    `None` means no closed form worth using (`zeta'` has none elementary).
    """

    cost: Cost | None
    """Measured gradient cost against the best alternative, or `None`.

    For rows with an upstream equivalent this compares against upstream, and is
    what decides whether the row can be dropped. For rows unique to `spexial` it
    compares the custom JVP against differentiating our own implementation,
    which is what justifies writing the derivative by hand.
    """

    notes: str
    """Why the row reads the way it does."""


#: JAX floor this table is reasoned against; keep in sync with `pyproject.toml`.
JAX_FLOOR: Final = "0.7.2"

_ROWS: Final = (
    Coverage(
        name="k0",
        jax_name=None,
        jax_since=None,
        jax_support=Support.NONE,
        scipy_array_api=Support.VALUE,
        status=Status.UNIQUE,
        custom_jvp=True,
        derivative="-k1(z)",
        cost=Cost(speed=0.674, memory=0.0146, against="differentiating our own series"),
        notes=(
            "JAX has no modified Bessel function of the second kind at any "
            "version. scipy's `k0` returns a value under `jit` but raises under "
            "`grad`. `spexial` defines the analytic derivative, which measured 1.5x "
            "faster than differentiating the 30-term series, and keeps 68x "
            "less residual -- the column that actually decides."
        ),
    ),
    Coverage(
        name="k1",
        jax_name=None,
        jax_since=None,
        jax_support=Support.NONE,
        scipy_array_api=Support.VALUE,
        status=Status.UNIQUE,
        custom_jvp=True,
        derivative="-k0(z) - k1(z) / z",
        cost=Cost(speed=0.994, memory=0.333, against="differentiating our own series"),
        notes=(
            "As `k0`. `k1` is a thin wrapper over `k1e`, whose own rule autodiff "
            "already picks up, so the hand-written rule earns its place on the "
            "memory column -- 3x less residual at neutral wall-clock. "
            "k1'(z) = -k0(z) - k1(z)/z."
        ),
    ),
    Coverage(
        name="k2",
        jax_name=None,
        jax_since=None,
        jax_support=Support.NONE,
        scipy_array_api=Support.NONE,
        status=Status.UNIQUE,
        custom_jvp=True,
        derivative="-k1(z) - (2/z) k2(z)",
        cost=Cost(speed=0.989, memory=0.333, against="differentiating our own series"),
        notes=(
            "`scipy.special.kn` does not dispatch on JAX arrays at all, even "
            "with the array API enabled. As `k1`, kept for the memory column. "
            "k2'(z) = -k1(z) - (2/z) k2(z), summed in the scaled variables: "
            "formed directly the `(2/z) k2` term is subnormal from z = 699 and "
            "XLA flushes it, which cost the derivative 0.29%."
        ),
    ),
    Coverage(
        name="k0e",
        jax_name=None,
        jax_since=None,
        jax_support=Support.NONE,
        scipy_array_api=Support.NONE,
        status=Status.UNIQUE,
        custom_jvp=True,
        derivative="k0e(z) - k1e(z)",
        cost=Cost(speed=0.573, memory=0.0146, against="differentiating our own series"),
        notes=(
            "Exponentially scaled e^z k0(z), matching `scipy.special.k0e`. JAX "
            "has no scaled Bessel K at any version, and scipy's does not "
            "dispatch on JAX arrays. This is the only form that survives past "
            "z = 705.5, where k0 itself is subnormal and XLA flushes it to 0."
        ),
    ),
    Coverage(
        name="k1e",
        jax_name=None,
        jax_since=None,
        jax_support=Support.NONE,
        scipy_array_api=Support.NONE,
        status=Status.UNIQUE,
        custom_jvp=True,
        derivative="k1e(z) - k0e(z) - k1e(z) / z",
        cost=Cost(speed=0.875, memory=0.0755, against="differentiating our own series"),
        notes="As `k0e`; matches `scipy.special.k1e`.",
    ),
    Coverage(
        name="k2e",
        jax_name=None,
        jax_since=None,
        jax_support=Support.NONE,
        scipy_array_api=Support.NONE,
        status=Status.UNIQUE,
        custom_jvp=True,
        derivative="k2e(z) - k1e(z) - (2/z) k2e(z)",
        cost=Cost(speed=0.982, memory=0.195, against="differentiating our own series"),
        notes="As `k0e`; matches `scipy.special.kve(2, z)`.",
    ),
    Coverage(
        name="polylog",
        jax_name=None,
        jax_since=None,
        jax_support=Support.NONE,
        scipy_array_api=Support.NONE,
        status=Status.UNIQUE,
        custom_jvp=True,
        derivative="Li_{n-1}(z) / z",
        cost=Cost(
            speed=1.288,
            memory=1 / 268.0,
            against="differentiating our own series",
        ),
        notes=(
            "No general polylogarithm anywhere. `jax.scipy.special.spence` is "
            "the n = 2 case only, and scipy has no polylog. The custom JVP is "
            "the one row where the two cost columns disagree: it keeps 268x "
            "less residual (4.2 MB -> 16 kB over 2000 points) but runs 1.7x "
            "slower, because Li_{n-1} must be evaluated afresh rather than "
            "reusing saved intermediates. Kept for the memory, which is the "
            "binding constraint when vmapping over a large batch."
        ),
    ),
    Coverage(
        name="eval_gegenbauer",
        jax_name=None,
        jax_since=None,
        jax_support=Support.NONE,
        scipy_array_api=Support.NONE,
        status=Status.UNIQUE,
        custom_jvp=False,
        derivative="2a C_{n-1}^{a+1}(x)",
        cost=None,
        notes=(
            "Absent from JAX. scipy's does not dispatch on JAX arrays. The "
            "derivative in x is 2a C_{n-1}^{a+1}(x), but a custom JVP is *not* "
            "wired up: `alpha` is traced too and dC/da has no closed form, so a "
            "rule supplying only the x-tangent would silently break `grad` with "
            "respect to `alpha`. The saving on offer is modest anyway -- 6 "
            "residual leaves, against 29 for `polylog`."
        ),
    ),
    Coverage(
        name="eval_gegenbauers",
        jax_name=None,
        jax_since=None,
        jax_support=Support.NONE,
        scipy_array_api=Support.NONE,
        status=Status.UNIQUE,
        custom_jvp=False,
        derivative=None,
        cost=None,
        notes=(
            "No counterpart anywhere: returns every order up to n in one pass, "
            "which is the point of it."
        ),
    ),
    Coverage(
        name="sph_legendre_p",
        jax_name=None,
        jax_since=None,
        jax_support=Support.NONE,
        scipy_array_api=Support.NONE,
        status=Status.UNIQUE,
        custom_jvp=False,
        derivative=None,
        cost=None,
        notes=(
            "Absent from JAX at any version -- the normalized Legendre function "
            "is reachable only through `jax.scipy.special.sph_harm_y`, and then "
            "only wrapped in a complex harmonic. scipy has had it since 1.15 "
            "but converts JAX arrays to NumPy, so it dies under `jit`. Computed "
            "from the *reduced* function P_l^m/(1-u^2)^(m/2), which is a "
            "polynomial, times an integer power of sin(theta); that is what "
            "makes the derivative finite at the poles, where upstream's "
            "sqrt(1 - cos^2) is not. The normalization is folded into the "
            "recurrence seed in log space, because p_m^m = (2m-1)!! overflows "
            "float64 near m = 90 and multiplying by a tiny N_lm afterwards is "
            "inf * 0. No custom JVP: the derivative identity in theta needs "
            "P_l^{m+1} as well, so it would buy a second recurrence, not save "
            "one."
        ),
    ),
    Coverage(
        name="sph_harm_y",
        jax_name="sph_harm_y",
        jax_since="*",
        jax_support=Support.AUTODIFF,
        scipy_array_api=Support.NONE,
        status=Status.EXTENDS,
        custom_jvp=False,
        derivative=None,
        cost=None,
        notes=(
            "The one row where upstream exists and returns *incorrect values* "
            "rather than merely covering less. `jax.scipy.special.sph_harm_y` "
            "indexes its Legendre table with `arange(len(n))`, so it pairs "
            "n[i] with theta[i] instead of broadcasting: a length-1 degree "
            "against a batch of angles is right at index 0 and wrong "
            "everywhere else, by up to 1.18 absolute for n <= 3, and 0-d input "
            "raises from `len()`. Its derivatives are also `nan` at both poles "
            "for every n >= 1. Here n and m are static Python ints, as for "
            "`eval_gegenbauer`, so there is no pairing to get wrong. "
            "`tests/unit/test_sph_harm.py` asserts both defects against "
            "upstream directly, so the row cannot rot: the day JAX fixes "
            "either, those tests fail and say so."
        ),
    ),
    Coverage(
        name="sph_harm_y_cart",
        jax_name=None,
        jax_since=None,
        jax_support=Support.NONE,
        scipy_array_api=Support.NONE,
        status=Status.UNIQUE,
        custom_jvp=False,
        derivative=None,
        cost=None,
        notes=(
            "No counterpart anywhere. Evaluates Y_n^m from a Cartesian unit "
            "direction as q_n^m(z) ((x + iy))^m, which is polynomial in x and y "
            "and therefore smooth on the z-axis -- where theta and phi are "
            "singular and the chain rule sends the Cartesian gradient of every "
            "m >= 1 term to exactly 0.0 against a non-zero true limit. That is "
            "a property of the coordinates, so it cannot be fixed in the "
            "(theta, phi) form at all, by us or by upstream. Deliberately does "
            "not normalize its argument: a zero vector is then finite and twice "
            "differentiable, which callers that floor the radius themselves "
            "depend on."
        ),
    ),
    Coverage(
        name="sph_harm_y_cart_all",
        jax_name=None,
        jax_since=None,
        jax_support=Support.NONE,
        scipy_array_api=Support.NONE,
        status=Status.UNIQUE,
        custom_jvp=False,
        derivative=None,
        cost=None,
        notes=(
            "Stands to `sph_harm_y_cart` as `eval_gegenbauers` does to "
            "`eval_gegenbauer`: the whole (l, m) table is a by-product of the "
            "recurrences any single entry already runs. Shape and index layout "
            "follow `scipy.special.sph_harm_y_all`, negative orders at the far "
            "end of the second axis included. Since l and m are static the "
            "saving is in *traced* operations -- a smaller HLO, so faster "
            "tracing and compiling, not faster execution, which XLA's fusion "
            "had already recovered. Measured on the equivalent code in `galax`: "
            "tracing 3-4.6x faster up to n = 20, run time flat."
        ),
    ),
    Coverage(
        name="sph_harm_y_cart_all_terms",
        jax_name=None,
        jax_since=None,
        jax_support=Support.NONE,
        scipy_array_api=Support.NONE,
        status=Status.UNIQUE,
        custom_jvp=False,
        derivative=None,
        cost=None,
        notes=(
            "`sph_harm_y_cart_all`'s values, same indexing and same layout, "
            "returned as a nested tuple of separate arrays instead of one "
            "stacked array. The container is the whole point: indexing a "
            "stacked table stops XLA folding each term into a caller's "
            "reduction as it is produced, so the table is materialized. "
            "Measured on a multipole expansion at n = 12 over a million "
            "directions, summing from the stacked form took 17.7 s against "
            "10 ms from these terms. The only function here not wrapped in "
            "`jax.jit`, deliberately: a jitted function returning a pytree "
            "materializes each leaf at the call boundary, which is exactly the "
            "fusion this exists to preserve."
        ),
    ),
    Coverage(
        name="incomplete_beta",
        jax_name=None,
        jax_since=None,
        jax_support=Support.NONE,
        scipy_array_api=Support.NONE,
        status=Status.UNIQUE,
        custom_jvp=True,
        derivative="z^(a-1) (1-z)^(b-1)",
        cost=Cost(speed=0.007, memory=0.0144, against="differentiating our own series"),
        notes=(
            "The *unregularized* B(a, b, z) of DLMF 8.17.1. Neither JAX nor "
            "scipy has one: their `betainc` is the regularized I_z(a, b), and "
            "reconstructing this as `beta(a, b) * betainc(a, b, z)` is `nan` "
            "for every b <= 0 -- B(a, b) has a pole there while the product "
            "does not, and b <= 0 is an ordinary slope in a double power-law "
            "density profile. `jax.scipy.special.hyp2f1` can express it via "
            "DLMF 8.17.7, but it is a `while_loop` whose trip count depends on "
            "its data, so under `vmap` every lane pays the worst lane's count "
            "and its derivative runs a second such loop. Two fixed-length "
            "series instead, switched at z = 1/2. The custom JVP is the "
            "strongest case in this table: by Leibniz the z-derivative is just "
            "the integrand at the endpoint, so it is exact and O(1) against 64 "
            "terms -- 143x faster on 70x less residual. `custom_jvp` rather "
            "than `custom_vjp` so `jacfwd(jacrev(...))` still composes. "
            "Contributed from `galax`."
        ),
    ),
    Coverage(
        name="spence",
        jax_name="spence",
        jax_since="*",
        jax_support=Support.AUTODIFF,
        scipy_array_api=Support.AUTODIFF,
        status=Status.EXTENDS,
        custom_jvp=True,
        derivative="log(z) / (1 - z)",
        cost=Cost(
            speed=55.4 / 277.6,
            memory=80.0 / 3080.0,
            against="jax.scipy.special.spence",
        ),
        notes=(
            "JAX has had `spence` since before our floor, but it is real-only "
            "and raises on complex input; this accepts both, which is the "
            "reason the row exists. It also wins on both cost columns -- 5.0x "
            "faster on 38.5x less residual -- because the analytic derivative "
            "log(z)/(1-z) is simply the integrand of the definition. The "
            "custom JVP is not merely an optimisation here: `lax.select` "
            "evaluates every branch, so differentiating the implementation "
            "yields `nan` -- and JAX's own `spence` differentiates to `nan` "
            "across roughly 1 < x < 2, where ours is exact. "
            "Contributed by Colm Talbot, translated from scipy's Cython "
            "implementation."
        ),
    ),
    Coverage(
        name="zeta",
        jax_name="zeta",
        jax_since="*",
        jax_support=Support.AUTODIFF,
        scipy_array_api=Support.NONE,
        status=Status.EXTENDS,
        custom_jvp=False,
        derivative=None,
        cost=Cost(
            speed=1.078, memory=None, against="jax.scipy.special.zeta, n > 1 only"
        ),
        notes=(
            "`jax.scipy.special.zeta` is the Hurwitz form and returns `nan` for "
            "negative arguments; `spexial` adds the negative integers via the "
            "functional equation. scipy raises `NotImplementedError` for the "
            "Riemann form on JAX arrays. No closed form for zeta', so no "
            "custom JVP."
        ),
    ),
    Coverage(
        name="comb",
        jax_name="comb",
        jax_since="0.10.2",
        jax_support=Support.AUTODIFF,
        scipy_array_api=Support.NONE,
        status=Status.REDUNDANT_ABOVE_FLOOR,
        custom_jvp=False,
        derivative=None,
        cost=Cost(speed=1.067, memory=None, against="jax.scipy.special.comb"),
        notes=(
            "Added to JAX in 0.10.2, below which `spexial` is still needed. "
            "`jax.scipy.special.comb` agrees on every edge case `spexial` "
            "handles (k > N, k < 0, N < 0) and differentiates. Re-export once "
            "the floor reaches 0.10.2."
        ),
    ),
    Coverage(
        name="gamma",
        jax_name="gamma",
        jax_since="*",
        jax_support=Support.AUTODIFF,
        scipy_array_api=Support.AUTODIFF,
        status=Status.DELEGATES,
        custom_jvp=True,
        derivative="gamma(x) psi(x)",
        cost=Cost(speed=0.998, memory=0.333, against="jax.scipy.special.gamma"),
        notes=(
            "The value is `jax.scipy.special.gamma`, called directly, so it "
            "cannot drift. What `spexial` adds is the derivative: "
            "Gamma'(x) = Gamma(x) psi(x) keeps 3x less residual than "
            "differentiating JAX's implementation, at neutral wall-clock (1.03x, "
            "i.e. no measurable saving -- the row earns its place on memory "
            "alone). Complex input "
            "works from jax 0.10.2 -- the same release that added `comb`, and "
            "below it `jax.scipy.special.gamma` branches on `floor(x)` and "
            "raises. Delegating means inheriting that limit rather than "
            "papering over it; the test probes the capability instead of "
            "comparing versions."
        ),
    ),
)

REGISTRY: Final[MappingProxyType[str, Coverage]] = MappingProxyType(
    {row.name: row for row in _ROWS}
)
"""Every exported function, keyed by name."""


_STATUS_LABEL: Final = {
    Status.UNIQUE: "only here",
    Status.EXTENDS: "extends upstream",
    Status.REDUNDANT_ABOVE_FLOOR: "redundant above floor",
    Status.DELEGATES: "delegates + our JVP",
    Status.REDUNDANT: "redundant",
}

_SUPPORT_LABEL: Final = {
    Support.NONE: "--",
    Support.VALUE: "value only",
    Support.AUTODIFF: "value + autodiff",
}


def _ratio(value: float | None) -> str:
    """Format a `Cost` ratio: `0.48x (2.1x better)`, or `--` when unmeasured."""
    if value is None:
        return "--"
    direction = "better" if value < 1 else "worse"
    return f"{value:.3g}x ({max(value, 1 / value):.1f}x {direction})"


def render_markdown() -> str:
    """Render the registry as the Markdown body of the coverage reference page.

    `scripts/gen_coverage_table.py` writes this into
    `docs/reference/coverage.md`, and a test fails if the two disagree -- so the
    published table cannot drift from the code.

    Examples
    --------
    >>> from spexial.registry import render_markdown
    >>> header = render_markdown().splitlines()[0]
    >>> all(c in header for c in ("Function", "In JAX", "Grad speed", "Grad memory"))
    True

    """
    header = (
        "| Function | In JAX | JAX autodiff | scipy on JAX arrays "
        "| Custom JVP | Grad speed | Grad memory | Status |\n"
        "| --- | --- | --- | --- | --- | --- | --- | --- |"
    )
    rows = []
    for row in _ROWS:
        if row.jax_since is None:
            in_jax = "--"
        elif row.jax_since == "*":
            in_jax = f"yes (all >= {JAX_FLOOR})"
        else:
            in_jax = f"yes (>= {row.jax_since})"
        jvp = "yes" if row.custom_jvp else ("available" if row.derivative else "--")
        speed = _ratio(row.cost.speed if row.cost else None)
        memory = _ratio(row.cost.memory if row.cost else None)
        rows.append(
            f"| `{row.name}` | {in_jax} | {_SUPPORT_LABEL[row.jax_support]} "
            f"| {_SUPPORT_LABEL[row.scipy_array_api]} | {jvp} "
            f"| {speed} | {memory} | {_STATUS_LABEL[row.status]} |"
        )
    table = "\n".join([header, *rows])

    details = []
    for row in _ROWS:
        deriv = f"\n\n    Derivative: `{row.derivative}`." if row.derivative else ""
        cost = (
            f"\n\n    Gradient cost vs {row.cost.against}: "
            f"speed {_ratio(row.cost.speed)}, memory {_ratio(row.cost.memory)}."
            if row.cost
            else ""
        )
        details.append(f"`{row.name}`\n:   {row.notes}{deriv}{cost}")
    return table + "\n\n## Per-function detail\n\n" + "\n\n".join(details) + "\n"
