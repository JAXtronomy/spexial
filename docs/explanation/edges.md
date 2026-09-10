# About domain edges, `nan`, and gradients that lie

_Why does an out-of-domain input return `nan` instead of raising, and why can a gradient be wrong without being obviously wrong?_

Two of `spexial`'s more surprising behaviours have the same root: a function that will be traced, compiled and differentiated by JAX cannot behave like an ordinary Python function at its boundaries.

## Why there are no domain errors

Inside `jax.jit`, your function does not run on values. It runs once on tracers, to build a graph, and the graph is what executes afterwards — possibly on a GPU, possibly batched over inputs you have not seen yet. A Python `raise` at that point has nothing to raise _about_: the tracer has no value to be out of domain.

So the boundary has to be encoded as data rather than control flow, and `nan` is the value the floating-point standard provides for exactly this. It propagates: any arithmetic involving it produces `nan`, so a single bad element quietly poisons everything downstream of it. Under `jax.grad` that is often the whole gradient, from one bad element in one array.

This is a real cost, and it is worth being clear-eyed that it is a cost rather than a design virtue. What you get in exchange is that the function composes — it can be compiled, vectorised and differentiated without a special case for the error path. What you lose is the thing an exception gives you for free: a stack trace pointing at the call that was wrong.

The practical consequence is that validation moves to you, and it moves _earlier_ — before the call, not after it. [Accuracy and domains](../reference/accuracy-and-domains.md) states each function's supported range for that reason: it is not background reading, it is the input contract you are now responsible for enforcing.

The exception proves the rule. `polylog` raises `ValueError` for an order below 1, because the order is a static Python integer, known at trace time and not part of the traced computation at all. A value JAX never sees can be checked the ordinary way.

## Why precision degrades near a pole

`gamma` uses the reflection formula below $x = 1/2$:

$$\Gamma(x) = \frac{\pi}{\sin(\pi x)\,\Gamma(1-x)}$$

At a non-positive integer, $\sin(\pi x)$ is mathematically zero, and the pole is real — $\Gamma$ genuinely diverges there. But floating-point $\sin(\pi x)$ near such a point is not zero; it is a small number carrying an absolute error of order $10^{-16}$, inherited from representing $\pi x$ at all. Dividing by a small number with a fixed absolute error gives a result with a _relative_ error that grows as you approach the pole — roughly $10^{-17}/\delta$ for a distance $\delta$.

No amount of care in the implementation removes this, because the error is already present in the input before `gamma` sees it. It is a property of asking for a value next to a singularity in finite precision. `spexial` reports the measured degradation rather than pretending to a uniform tolerance it cannot deliver.

## Why a gradient can be finite and still wrong

This is the sharpest edge in the library, and the one most likely to cost someone a day.

`zeta` is evaluated two different ways depending on its argument. For $n > 1$ it sums a series — a real computation that JAX can differentiate through, producing a real $\zeta'$. On the negative line it does something else entirely: it looks up a Bernoulli number from a precomputed table and applies the functional equation.

A table lookup carries no derivative information. Differentiating it does not fail; it returns the derivative of the _arithmetic around_ the lookup, which is a perfectly finite number that is not $\zeta'(n)$. There is no `nan` to warn you, because nothing undefined happened.

This is the failure mode worth generalising from. A `nan` is a bad answer that announces itself. A plausible number from a computation that was never differentiable is a bad answer that does not, and it will survive every check you have that tests for `nan`. When you differentiate a special function, it is worth knowing which branch of it you are on.

For which functions and ranges this affects, see [Accuracy and domains](../reference/accuracy-and-domains.md); for the practical rule, see [How to use spexial with jit, vmap and grad](../how-to/use-with-jit-vmap-and-grad.md).
