"""Unit tests: behaviour the parity suite cannot express.

Value correctness against a reference implementation is
`tests/parity/`'s job -- it checks far more inputs, through Hypothesis, than a
hand-picked table of known values ever could. Duplicating a few of those inputs
here only adds a second place to update when a tolerance moves.

What belongs here instead:

- shapes, broadcasting and elementwise behaviour (parity is scalar);
- dtype promotion;
- transform compatibility -- `jax.jit`, `jax.vmap`, `jax.grad`;
- exact domain guards: `nan`, `inf`, exactly `0`, and raised exceptions, none of
  which a relative-tolerance comparison can assert;
- regression tests naming a specific fixed bug;
- cross-function identities that link two implementations (`polylog(n, 1) == zeta(n)`).
"""
