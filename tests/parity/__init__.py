"""Parity tests: value correctness against a reference implementation.

Hypothesis-driven comparison against `scipy.special`, and against `mpmath` for
`polylog`, which has no scipy counterpart. This is the authority on whether a
function returns the right number; `tests/unit/` covers everything else.

Every tolerance here was measured rather than tuned until the suite passed, and
every restricted strategy carries a comment saying which domain is genuinely
unsupported and why.
"""
