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

>>> REGISTRY["K0"].status is Status.UNIQUE
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
    far larger than the speed-up: `K0` differentiates 2.1x faster but keeps 67x
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

    scipy_name: str | None
    """Equivalent in `scipy.special`, or `None`."""

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
        name="K0",
        jax_name=None,
        jax_since=None,
        jax_support=Support.NONE,
        scipy_name="k0",
        scipy_array_api=Support.VALUE,
        status=Status.UNIQUE,
        custom_jvp=True,
        derivative="-K1(z)",
        cost=Cost(
            speed=1 / 2.07, memory=1 / 67.4, against="differentiating our own series"
        ),
        notes=(
            "JAX has no modified Bessel function of the second kind at any "
            "version. scipy's `k0` returns a value under `jit` but raises under "
            "`grad`. `spexial` defines the analytic derivative, which measured 2.07x "
            "faster than differentiating the 30-term series (637us -> 307us "
            "for `grad` over 1000 points)."
        ),
    ),
    Coverage(
        name="K1",
        jax_name=None,
        jax_since=None,
        jax_support=Support.NONE,
        scipy_name="k1",
        scipy_array_api=Support.VALUE,
        status=Status.UNIQUE,
        custom_jvp=True,
        derivative="-(K0(z) + K2(z)) / 2",
        cost=Cost(
            speed=None, memory=1 / 16.6, against="differentiating our own series"
        ),
        notes="As `K0`. K1'(z) = -(K0(z) + K2(z)) / 2.",
    ),
    Coverage(
        name="K2",
        jax_name=None,
        jax_since=None,
        jax_support=Support.NONE,
        scipy_name="kn",
        scipy_array_api=Support.NONE,
        status=Status.UNIQUE,
        custom_jvp=True,
        derivative="-K1(z) - (2/z) K2(z)",
        cost=Cost(
            speed=None, memory=1 / 12.1, against="differentiating our own series"
        ),
        notes=(
            "`scipy.special.kn` does not dispatch on JAX arrays at all, even "
            "with the array API enabled. K2'(z) = -K1(z) - (2/z) K2(z)."
        ),
    ),
    Coverage(
        name="Li",
        jax_name=None,
        jax_since=None,
        jax_support=Support.NONE,
        scipy_name=None,
        scipy_array_api=Support.NONE,
        status=Status.UNIQUE,
        custom_jvp=True,
        derivative="Li_{n-1}(z) / z",
        cost=Cost(
            speed=1.73,
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
        scipy_name="eval_gegenbauer",
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
            "residual leaves, against 29 for `Li`."
        ),
    ),
    Coverage(
        name="eval_gegenbauers",
        jax_name=None,
        jax_since=None,
        jax_support=Support.NONE,
        scipy_name=None,
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
        name="zeta",
        jax_name="zeta",
        jax_since="*",
        jax_support=Support.AUTODIFF,
        scipy_name="zeta",
        scipy_array_api=Support.NONE,
        status=Status.EXTENDS,
        custom_jvp=False,
        derivative=None,
        cost=Cost(
            speed=1.15, memory=None, against="jax.scipy.special.zeta, n > 1 only"
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
        scipy_name="comb",
        scipy_array_api=Support.NONE,
        status=Status.REDUNDANT_ABOVE_FLOOR,
        custom_jvp=False,
        derivative=None,
        cost=Cost(speed=0.97, memory=None, against="jax.scipy.special.comb"),
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
        scipy_name="gamma",
        scipy_array_api=Support.AUTODIFF,
        status=Status.DELEGATES,
        custom_jvp=True,
        derivative="gamma(x) psi(x)",
        cost=Cost(speed=1 / 4.26, memory=1 / 3.0, against="jax.scipy.special.gamma"),
        notes=(
            "The value is `jax.scipy.special.gamma`, called directly, so it "
            "cannot drift. What `spexial` adds is the derivative: "
            "Gamma'(x) = Gamma(x) psi(x) is 4.3x faster than differentiating "
            "JAX's implementation and keeps 3x less residual. Complex input "
            "works, since delegating removed the reflection-formula constraint "
            "that made the old Lanczos version real-only."
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
    better = 1.0 / value
    direction = "better" if better > 1 else "worse"
    magnitude = max(better, 1 / better)
    return f"{value:.3g}x ({magnitude:.1f}x {direction})"


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
