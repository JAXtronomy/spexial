# About precision

_Why does `spexial` insist on double precision when most JAX code is happy in float32?_

JAX defaults to 32-bit floats, and for the workload JAX was built for — training neural networks — that default is right. Gradients are noisy, weights are redundant, and half the value of a GPU comes from moving fewer bytes. Special functions are the opposite kind of computation, and the default is wrong for them.

## Where the digits go

A float32 number carries about seven decimal digits. That sounds like plenty until you look at how these functions are actually evaluated. None of them has a closed form you can just apply; each is a sum, a recurrence, or a reflection:

- `k0` sums thirty terms of an ascending series below $z = 9$, and ten terms of an asymptotic expansion above it.
- `gamma` evaluates a Lanczos series, and for $x < 1/2$ divides by $\sin(\pi x)$ first.
- `polylog` builds a length-60 vector of powers of $\log z$ and contracts it against Bernoulli numbers.

Every one of those operations is an opportunity to lose low-order bits, and the losses accumulate rather than cancel. A thirty-term sum whose terms alternate in sign can shed several digits to cancellation alone. Start with seven and you may finish with three — and you will not be told. That is the part that makes float32 dangerous here rather than merely imprecise: the answer comes back looking exactly like a good answer.

The accuracy figures in [Accuracy and domains](../reference/accuracy-and-domains.md) are all measured in double precision. In float32 they do not merely degrade proportionally; they stop meaning anything, because the error is dominated by the arithmetic rather than by the algorithm the figure describes.

## Why it is a global switch and not an argument

It would be tidier if precision were per-call. It is not, because JAX resolves dtypes at trace time from the arrays it is given, and `jax_enable_x64` changes what `jnp.array(1.0)` _is_. The setting has to be in place before any array exists, which is why [the how-to](../how-to/enable-double-precision.md) is insistent about ordering, and why the environment variable is the more reliable of the two mechanisms — it applies from interpreter start, before an import can get in first.

The consequence worth internalising is that a library cannot turn this on for you. `spexial` could refuse to run in float32, but that would break legitimate uses — someone who genuinely wants three digits fast, inside a larger float32 model, should be allowed to have them. So the choice stays with you, and the documentation states the assumption instead.

## The NumPy trap

A `numpy.float64` array passed into a JAX function with x64 disabled is silently downcast. No warning is issued, because from JAX's point of view nothing went wrong: you asked for its default precision and you got it. This is the most common way a computation ends up in float32 long after someone thought they had turned x64 on — the config call was there, but it ran after an import that had already built an array.

Check the dtype of what comes _out_ of your first `spexial` call, not what goes in.
