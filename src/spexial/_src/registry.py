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

    REDUNDANT = "redundant"
    """Upstream covers it everywhere we support. Should be re-exported or dropped."""


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
        custom_jvp=False,
        derivative="Li_{n-1}(z) / z",
        notes=(
            "No general polylogarithm anywhere. `jax.scipy.special.spence` is "
            "the n = 2 case only, and scipy has no polylog. "
            "d/dz Li_n(z) = Li_{n-1}(z) / z."
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
        notes=(
            "Absent from JAX. scipy's does not dispatch on JAX arrays. "
            "d/dx C_n^a(x) = 2a C_{n-1}^{a+1}(x)."
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
        status=Status.REDUNDANT,
        custom_jvp=False,
        derivative="gamma(x) psi(x)",
        notes=(
            "`jax.scipy.special.gamma` is a strict superset: negative reals, "
            "complex input, and autodiff, at comparable accuracy. `spexial`'s "
            "is real-only. The original implementation existed to add complex "
            "support, which JAX now provides -- so this row has no reason to "
            "stay. Re-export, then remove."
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
    Status.REDUNDANT: "redundant",
}

_SUPPORT_LABEL: Final = {
    Support.NONE: "--",
    Support.VALUE: "value only",
    Support.AUTODIFF: "value + autodiff",
}


def render_markdown() -> str:
    """Render the registry as the Markdown body of the coverage reference page.

    `scripts/gen_coverage_table.py` writes this into
    `docs/reference/coverage.md`, and a test fails if the two disagree -- so the
    published table cannot drift from the code.

    Examples
    --------
    >>> from spexial.registry import render_markdown
    >>> render_markdown().splitlines()[0]
    '| Function | In JAX | JAX autodiff | scipy on JAX arrays | Custom JVP | Status |'

    """
    header = (
        "| Function | In JAX | JAX autodiff | scipy on JAX arrays "
        "| Custom JVP | Status |\n"
        "| --- | --- | --- | --- | --- | --- |"
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
        rows.append(
            f"| `{row.name}` | {in_jax} | {_SUPPORT_LABEL[row.jax_support]} "
            f"| {_SUPPORT_LABEL[row.scipy_array_api]} | {jvp} "
            f"| {_STATUS_LABEL[row.status]} |"
        )
    table = "\n".join([header, *rows])

    details = []
    for row in _ROWS:
        deriv = f"\n\n    Derivative: `{row.derivative}`." if row.derivative else ""
        details.append(f"`{row.name}`\n:   {row.notes}{deriv}")
    return table + "\n\n## Per-function detail\n\n" + "\n\n".join(details) + "\n"
