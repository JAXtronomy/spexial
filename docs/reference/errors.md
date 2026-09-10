# Errors

What `spexial` raises, when, and what it deliberately does not raise. For the domains themselves see [Accuracy and domains](accuracy-and-domains.md); for the reasoning behind returning `nan` instead of raising, see [About domain edges](../explanation/edges.md).

## The two kinds of failure

`spexial` fails in exactly two ways, and which one you get is not arbitrary.

**A value outside the domain returns `nan` or `inf`.** It does not raise. Under `jax.jit` the value is not available at trace time, so raising is not an option that exists — a check on a traced value cannot run. Returning the IEEE sentinel is the only behaviour that is the same jitted and unjitted.

**A misuse of the _interface_ raises**, because those arguments are static: the order of a polylogarithm, the degree of a Gegenbauer polynomial, the dtype of an argument. These are known while tracing, so an error can be raised at the point of the mistake rather than surfacing as a `nan` three steps later.

## Exceptions raised

| Call | Exception | Message |
| --- | --- | --- |
| `polylog(0, z)`, `polylog(-1, z)` | `ValueError` | `polylog is only implemented for integer order n >= 1, got 0` |
| `polylog(n, z)` with complex `z` | `ValueError` | `polylog is only implemented for real z, got dtype complex128` |
| `polylog(2.0, z)` — non-integer order | `TypeError` | see [below](#a-non-integer-order) |
| `polylog(n, z)` with array `z` | `TypeError` | `pow got incompatible shapes for broadcasting: (2,), (59,)` |
| `eval_gegenbauers(n, alpha, x)` with array `x` | `TypeError` | `Cannot concatenate arrays with different numbers of dimensions` |
| `k0`/`k1`/`k2`/`k0e`/`k1e`/`k2e` with complex `z` | `TypeError` | `lgamma does not accept dtype complex128` |
| `zeta(n)` with complex `n` | `ValueError` | `Clip received a complex value` |

The last four are raised by JAX, not by `spexial`, and their wording is therefore not ours to promise. They are listed because they are what you will actually see, and because each corresponds to a documented restriction: `polylog` takes a scalar `z`, `eval_gegenbauers` takes a scalar `x`, and the Bessel and zeta functions are real-only.

### A non-integer order

`polylog(2.0, z)` — a whole-number `float` where an `int` is meant — is the most likely of these mistakes, and what you see depends on whether runtime type checking is switched on.

With it enabled, the error names the parameter and the expected type:

```
jaxtyping.TypeCheckError: Type-check error whilst checking the parameters of
spexial._src.polylog.polylog.  The problem arose whilst typechecking parameter 'n'.
Actual value: 2.0
Expected type: <class 'int'>.
```

With it disabled — the default — the annotation is not enforced, and `2.0` reaches an internal `lax.fori_loop`, which fails on its own terms:

```
TypeError: lower and upper arguments to fori_loop must have equal types,
got int64 and float64
```

That message names neither `polylog` nor the argument at fault. If you are debugging an opaque error from inside a trace, turning the checker on is usually the fastest way to find out which argument caused it:

```bash
SPEXIAL_ENABLE_RUNTIME_TYPECHECKING=beartype.beartype python your_script.py
```

The variable takes `"False"` (the default), `"None"` to check only `@jaxtyped`-decorated functions, or a typechecker path such as `"beartype.beartype"`. It is a development aid — leave it off in production, where it costs a check on every call.

## What returns a sentinel instead

These are **not** errors. Each is a documented value, and each is what `scipy.special` returns for the same input unless noted.

| Situation | Result |
| --- | --- |
| `k0`/`k1`/`k2` at `z = 0` | `inf` — the pole |
| `k0`/`k1`/`k2` at `z < 0` | `nan` — outside the real domain |
| `k0`/`k1`/`k2` above `z ≈ 705.5` | `0` — the true value is subnormal and XLA flushes it; use `k0e`/`k1e`/`k2e` |
| `gamma` at a non-positive integer | `nan` at the negative integers, `inf` at 0 |
| `gamma` below `x ≈ -170.6` | `0` — SciPy returns a denormal here |
| `zeta` in the critical strip `0 < n ≤ 1` | `nan` — not implemented |
| `zeta` at a negative non-integer, or odd `n ≤ -60` | `nan` — outside the Bernoulli table |
| `polylog(n, z)` with `n > 60` and $\lvert z \rvert \ge 2$ | `nan` — the inversion branch needs $B_{60}$ |
| `comb` with `k > N`, `k < 0` or `N < 0` | `0` |
| Any function given `nan` | `nan`, except `comb(nan, k)`, which is `0` as in SciPy |

A `nan` from one of these will propagate silently through the rest of a computation. If you are getting one and cannot see why, the fastest diagnosis is usually to evaluate the offending function alone on the same input, outside `jit`.

## Gradients that do not raise and are still wrong

Two cases where a derivative is finite, plausible, and not the derivative you want. Neither raises, because neither can be detected at trace time.

`zeta` on the negative half-line takes its value from a table of Bernoulli numbers. A table carries no information about how $\zeta$ varies _between_ the integers, so `jax.grad(zeta)` there returns a finite number that is not $\zeta'$. Only `n > 1` gives a meaningful gradient.

Third and higher derivatives of `k0`/`k1`/`k2` lose about $7\times10^{-4}$ above $z \approx 690$. The first two orders are exact throughout; the loss comes from XLA flushing subnormal intermediates, and no reformulation avoids it. It is measured and pinned by a test rather than hidden.

Both are described in full in [About domain edges](../explanation/edges.md).

## Reporting one

If you find a value that is wrong rather than merely undefined — a plausible number where the truth is different — that is a bug worth reporting, and the most useful report includes the reference you compared against. `mpmath` at high precision is the arbiter this project uses, because SciPy and `spexial` share algorithmic ancestry in places and can agree on the same wrong answer. See [Contributing](../about/contributing.md).
