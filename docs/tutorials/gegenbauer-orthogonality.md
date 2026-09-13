# Check Gegenbauer orthogonality by quadrature

In this tutorial we will evaluate Gegenbauer polynomials on a whole array of points at once, then use them to reproduce a classical result: that they are mutually orthogonal under their own weight function. The answer comes out as an identity matrix, and every off-diagonal entry will be zero to within one part in $10^{16}$ — which is a much sharper check than comparing a few values against a table.

Along the way we will meet the one thing about these functions that surprises people: the degree is a _static_ argument, not a traced one.

You need `spexial` installed and nothing else.

## Step 1: turn on double precision

```pycon
>>> import jax
>>> jax.config.update("jax_enable_x64", True)

>>> import jax.numpy as jnp
>>> import spexial as sp

```

As always, this has to come before any array is created. See [How to enable double precision](../how-to/enable-double-precision.md) if you want the reasons.

## Step 2: evaluate on an array

`eval_gegenbauer(n, alpha, x)` broadcasts over `x`, so you can hand it a whole grid:

```pycon
>>> xs = jnp.linspace(-1.0, 1.0, 5)
>>> [round(float(v), 12) for v in sp.eval_gegenbauer(3, 1.0, xs)]
[-4.0, 1.0, -0.0, -1.0, 4.0]

```

At $\alpha = 1$ these are the Chebyshev polynomials of the second kind, $U_n$, which is why the endpoint values are the integers $\pm 4$.

The degree `n` is different from the other two arguments. It is a Python `int` baked into the trace, not a value JAX tracks — so a different `n` triggers a recompilation, and you cannot `vmap` over it. `alpha` and `x` are ordinary traced values and broadcast against each other normally.

## Step 3: get every order in one pass

Evaluating degrees `0..n` separately repeats the same recurrence `n` times. `eval_gegenbauers` — note the plural — returns all of them from a single pass:

```pycon
>>> [round(float(v), 12) for v in sp.eval_gegenbauers(3, 1.0, 0.5)]
[1.0, 1.0, 0.0, -1.0]

```

This one takes a **scalar** `x` and returns an array of shape `(n + 1,)`. To evaluate it across many points, `vmap` over `x`:

```pycon
>>> rows = jax.vmap(lambda t: sp.eval_gegenbauers(3, 1.0, t))(jnp.asarray([0.0, 0.5]))
>>> rows.shape
(2, 4)

```

Those four numbers satisfy the three-term recurrence the implementation is built on,

$$n\,C_n^{(\alpha)}(x) = 2(n + \alpha - 1)\,x\,C_{n-1}^{(\alpha)}(x) - (n + 2\alpha - 2)\,C_{n-2}^{(\alpha)}(x),$$

which you can confirm directly:

```pycon
>>> c = sp.eval_gegenbauers(3, 1.0, 0.5)
>>> round(float((2 * 3.0 * 0.5 * c[2] - 3.0 * c[1]) / 3), 12)
-1.0

```

## Step 4: build a quadrature rule

Gegenbauer polynomials are orthogonal on $[-1, 1]$ against the weight $(1 - x^2)^{\alpha - 1/2}$. At $\alpha = 1$ that weight is $\sqrt{1 - x^2}$, and the matching quadrature is Gauss–Chebyshev of the second kind:

$$
\int_{-1}^{1} f(x)\,\sqrt{1 - x^2}\,\mathrm{d}x \;\approx\; \sum_{k=1}^{N} w_k\, f(x_k),
\qquad x_k = \cos\frac{k\pi}{N+1},
\qquad w_k = \frac{\pi}{N+1}\sin^2\frac{k\pi}{N+1}.
$$

It is exact for polynomials up to degree $2N - 1$, so with $N = 24$ we have room to spare:

```pycon
>>> N = 24
>>> k = jnp.arange(1, N + 1)
>>> nodes = jnp.cos(k * jnp.pi / (N + 1))
>>> weights = jnp.pi / (N + 1) * jnp.sin(k * jnp.pi / (N + 1)) ** 2

```

A quick sanity check before using it — the weights must sum to $\int_{-1}^{1}\sqrt{1-x^2}\,\mathrm{d}x = \pi/2$:

```pycon
>>> round(float(weights.sum()), 12)
1.570796326795

```

## Step 5: the orthogonality matrix

Now evaluate the first four degrees across all 24 nodes. Each call broadcasts over the whole node array, so this is four calls, not ninety-six:

```pycon
>>> vals = jnp.stack([sp.eval_gegenbauer(n, 1.0, nodes) for n in range(4)])
>>> vals.shape
(4, 24)

```

The Gram matrix $G_{mn} = \int C_m C_n \sqrt{1-x^2}\,\mathrm{d}x$ is then one matrix product, and the theory says it should be $\tfrac{\pi}{2}$ times the identity:

```pycon
>>> gram = (vals * weights) @ vals.T / (jnp.pi / 2)
>>> [round(float(gram[i, i]), 10) for i in range(4)]
[1.0, 1.0, 1.0, 1.0]

```

The diagonal is exactly 1. The real test is the off-diagonal, which should be zero:

```pycon
>>> float(jnp.max(jnp.abs(gram - jnp.eye(4)))) < 1e-15
True

```

Every one of the twelve off-diagonal entries is below $10^{-15}$. That is twelve independent integrals agreeing with theory at machine precision — a far stronger statement than any single value comparison, because an error in the recurrence would have to conspire across all of them to cancel.

## Step 6: differentiate

The derivative with respect to $x$ has a closed form, $\frac{\mathrm{d}}{\mathrm{d}x}C_n^{(\alpha)} = 2\alpha\,C_{n-1}^{(\alpha + 1)}$, and `jax.grad` agrees with it:

```pycon
>>> round(float(jax.grad(lambda t: sp.eval_gegenbauer(3, 1.0, t))(0.5)), 12)
2.0
>>> round(float(2 * 1.0 * sp.eval_gegenbauer(2, 2.0, 0.5)), 12)
2.0

```

Note that this derivative comes from JAX differentiating the recurrence, not from a hand-written rule. `eval_gegenbauer` deliberately has no custom JVP: `alpha` is traced too, and $\partial C / \partial \alpha$ has no closed form, so a rule supplying only the $x$-tangent would silently break `grad` with respect to `alpha`. The reasoning is recorded in the [coverage table](../reference/coverage.md).

## What to remember

- `eval_gegenbauer` broadcasts over `x` and `alpha`; `eval_gegenbauers` takes a scalar `x` and returns every order at once. Use `vmap` to spread the latter over points.
- `n` is static. A new `n` means a new compilation, and `vmap` over it will not work.
- Accuracy is governed by the _recurrence's_ working magnitude, not by the size of the answer. At large $\alpha$ the intermediate values can run many orders above the final one, and the absolute error scales with that peak rather than with the result — see [Accuracy and domains](../reference/accuracy-and-domains.md).

## Where to go next

- [How to use spexial with jit, vmap and grad](../how-to/use-with-jit-vmap-and-grad.md) for the general rules on static arguments and broadcasting.
- [About the algorithms](../explanation/algorithms.md) for why the recurrence is the right choice here and what its error behaviour costs.
