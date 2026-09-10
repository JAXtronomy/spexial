# Why `spexial` exists alongside `jax.scipy.special`

A reasonable first question about this library is why it is a library at all, rather than a pull request to JAX. The honest answer is that most of it _should_ end up upstream, and the design assumes it will.

## What upstream can and cannot take

`jax.scipy.special` mirrors `scipy.special`. That constraint is deliberate and it is load-bearing: it is what lets a NumPy user move to JAX without relearning a namespace. It also decides what can go in.

Two of the functions here have **no `scipy.special` counterpart at all**. `Li` is the general polylogarithm — `scipy` has only `spence`, the order-2 case at a shifted argument. `eval_gegenbauers` returns every order up to `n` from a single pass of the recurrence, which is a JAX-shaped idea rather than a SciPy one: it exists because computing degrees `0..n` separately repeats the same work `n` times, and because returning a stacked array is natural where returning `n + 1` scalars is not. Neither can be upstreamed into a namespace defined by mirroring SciPy without first changing what that namespace is for.

The rest could go upstream, and some of it should.

## The floor problem

`spexial` supports jax from 0.7.2. `jax.scipy.special.comb` arrived in 0.10.2.

Those two facts are the whole reason `comb` exists here. It is not better than JAX's; above 0.10.2 it is marginally slower. It exists so that code depending on `spexial` works on the versions `spexial` claims to support. A contribution to JAX, however good, does nothing for anyone pinned below the release it lands in — and scientific code is pinned below current releases most of the time.

This is a general shape, not a one-off: a function can be _upstream_ and still _unavailable_. The gap between "JAX has it" and "your environment has it" is measured in years for some projects.

## Becoming redundant on purpose

What makes this defensible rather than merely duplicative is that the redundancy is tracked and has an exit.

Every function carries a `Status` in the [coverage registry](../reference/coverage.md), and one of the values is `REDUNDANT_ABOVE_FLOOR` — meaning upstream covers it, but only above the version floor this package supports:

```pycon
>>> from spexial.registry import REGISTRY, Status
>>> [k for k, v in REGISTRY.items() if v.status is Status.REDUNDANT_ABOVE_FLOOR]
['comb']

```

The procedure for such a row is written down in `AGENTS.md`: when the floor rises to the version in `jax_since`, the implementation becomes a re-export; one release later it is deprecated; the release after that it is removed. Removal takes three releases, so nobody's code breaks on an upgrade.

The registry is checked against the _installed_ JAX on every test run, in both directions — a row claiming JAX lacks a function fails if JAX has it, and vice versa. That is what keeps this from rotting into a permanent shadow library: the moment upstream catches up, a test says so.

## Where it earns its place independently

Two functions here are not waiting for anything, because they do something upstream has chosen not to.

`spence` accepts complex arguments; `jax.scipy.special.spence` is real-only and raises. And JAX's gradient is `nan` across roughly $1 < z < 2$, where this one is exact. Those are different capabilities, not a different implementation of the same capability.

`zeta` extends JAX's Hurwitz zeta to the negative integers through the functional equation. JAX returns `nan` there. The extension is partial — the critical strip and negative non-integers are still `nan` — and the [accuracy page](../reference/accuracy-and-domains.md) says so rather than implying full coverage.

`gamma` is the interesting middle case. The value is JAX's, called directly, so it cannot drift. What `spexial` adds is an analytic derivative that keeps 3× less residual memory through the backward pass at parity on time. That is a real benefit and a narrow one, and the registry records it as `DELEGATES` — a row that earns its place on the cost columns alone, and becomes redundant the moment upstream's own gradient matches on both.

## The case that is none of the above: upstream is wrong

`sph_harm_y` is the one row where `jax.scipy.special` has the function, at every version this package supports, and returns **incorrect values**.

It indexes its internal Legendre table with `arange(len(n))`, which pairs `n[i]` with `theta[i]` positionally instead of broadcasting the four arguments against each other. A length-1 degree against a batch of angles is therefore correct at index 0 and silently wrong at every other index — measured against `scipy.special.lpmv` over four positions, up to **1.18 absolute** for $l \le 3$. Rank-0 input is rejected outright, because `len()` has nothing to measure.

This is not a narrower domain or a missing derivative. It is a wrong answer returned confidently for input its own signature accepts, and it had already reached at least one downstream package's released code, where no test caught it because every fixture evaluated a single position — the one index that is right.

`spexial`'s takes the degree and order as static Python `int`s, as `eval_gegenbauer` does, so there is nothing to mispair and `theta`/`phi` broadcast at any rank.

The part worth more than the implementation is `tests/unit/test_sph_harm.py`, which asserts the **defect** against upstream directly rather than only asserting our own correctness. Such a test cannot rot: the day JAX fixes the broadcasting, or the `nan` pole derivatives that go with it, that test fails and says exactly which claim in the [registry](../reference/coverage.md) needs revisiting.

## The case upstream cannot fix at all

`sph_harm_y_cart` is here for a reason no upstream change would remove.

On the z-axis, $\theta$ and $\phi$ have no directional derivative. Any spherical harmonic evaluated through them therefore has a _Cartesian_ gradient of exactly `0.0` there for every $m \ge 1$ term, against a non-zero true limit — not because of how the function is implemented, but because of the coordinates it is written in. SciPy has the same limitation, as does every implementation that takes angles as its arguments.

Evaluating from a Cartesian unit direction instead,

$$Y_l^m = q_l^m(\hat{z})\,(\hat{x} + i\hat{y})^m,$$

is polynomial in $\hat{x}$ and $\hat{y}$ and so smooth on the axis. That is new capability rather than a repair, which is why it is a separate function under a separate name rather than a flag on `sph_harm_y`.

## The honest summary

This library is three things at once, and it is worth being clear about which part is which:

1. **A staging area** for functions that belong upstream but are not there yet, or are there only above a floor real users have not reached. `comb` today; more of the Bessel functions eventually.
2. **A home for things that do not fit the `scipy.special` mirror** — `Li`, `eval_gegenbauers`, and the Cartesian harmonics.
3. **A small set of genuine improvements** — complex `spence`, negative `zeta`, cheaper gradients — that exist because a focused package can make choices a general one cannot.
4. **A place to record, in executable form, where upstream is wrong** — `sph_harm_y`.

The second category is permanent, and so, unhappily, is the fourth until upstream moves. The first is designed to shrink, and the registry exists to make sure someone notices when it should.
