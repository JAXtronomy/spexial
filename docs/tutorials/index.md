# Tutorials

Lessons that take you through a complete piece of work, step by step. Start here if you are new to `spexial` and want to learn how it fits together rather than look something up.

Every step shows its output, and the output is real — these pages are executed as tests on every commit, so they cannot silently rot.

- [Compute the Stefan–Boltzmann constant](stefan-boltzmann.md) — evaluate $\zeta(4)$ and the polylogarithm, check them against each other and against a closed form, and end with a physical constant matching every published digit. Covers double precision, `jax.grad` and `jax.jit`.
- [Check Gegenbauer orthogonality by quadrature](gegenbauer-orthogonality.md) — evaluate a polynomial family across a whole grid at once, build a quadrature rule, and recover an identity matrix to machine precision. Covers array-valued evaluation, static arguments and `jax.vmap`.

If you already know what you are trying to accomplish, the [how-to guides](../how-to/index.md) will get you there faster.
