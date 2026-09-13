# Compute the Stefan–Boltzmann constant

In this tutorial we will compute the Stefan–Boltzmann constant $\sigma$ — the number that tells you how much power a black body radiates — from nothing but Planck's law and two functions from `spexial`. Along the way we will see two different special functions return bit-for-bit the same answer, and we will differentiate one of them.

You need `spexial` installed and nothing else. No network, no data files.

## Step 1: turn on double precision

Create a file called `sigma.py` and start it like this:

```pycon
>>> import jax
>>> jax.config.update("jax_enable_x64", True)

>>> import jax.numpy as jnp
>>> import spexial as sp

```

Check that it took effect:

```pycon
>>> jnp.zeros(1).dtype
dtype('float64')

```

You should see `dtype('float64')`. If you see `float32`, the `jax.config.update` line ran too late — it has to come before any array is created.

## Step 2: the integral we need

Integrating Planck's law over all frequencies gives the total radiated power, and the whole problem reduces to one dimensionless integral:

$$\int_0^\infty \frac{x^3}{e^x - 1}\,\mathrm{d}x = 6\,\zeta(4)$$

So we need $\zeta(4)$. Ask `spexial` for it:

```pycon
>>> float(sp.zeta(4.0))
1.0823232337111384

```

## Step 3: check it against a value you already know

$\zeta(4)$ has a closed form, $\pi^4/90$. Let's compare:

```pycon
>>> float(jnp.pi**4 / 90)
1.082323233711138

```

Notice that the two agree to fifteen digits, and differ in the last one. That last digit is not a bug in either — it is the ordinary result of doing two different sequences of floating-point operations. This is the level of agreement you should expect from these functions, and `spexial` documents it per function in the reference.

## Step 4: get the same number a completely different way

The polylogarithm $\mathrm{Li}_s(z)$ reduces to the zeta function at $z = 1$: $\mathrm{Li}_s(1) =
\zeta(s)$. `spexial` implements the two with entirely separate code, so this is a real check, not a tautology:

```pycon
>>> float(sp.polylog(4, 1.0))
1.0823232337111384

```

```pycon
>>> import math
>>> math.isclose(float(sp.zeta(4.0)), float(sp.polylog(4, 1.0)), rel_tol=1e-15)
True

```

Two independent implementations — a Borwein-accelerated series and a polylogarithm — agree to the last bit or two. That is a good sign about both. The check above asks for closeness rather than exact equality on purpose: the two happen to land on the same float today, but that is the kind of property a compiler change can alter without either implementation becoming wrong, and a tutorial that fails is worse than one that claims less.

## Step 5: assemble the constant

Now put it together. With Planck's constant $h$, the speed of light $c$ and Boltzmann's constant $k$:

$$\sigma = \frac{2\pi k^4}{c^2 h^3}\int_0^\infty \frac{x^3}{e^x-1}\,\mathrm{d}x$$

```pycon
>>> k = 1.380649e-23  # J/K, exact by SI definition
>>> c = 299792458.0  # m/s, exact by SI definition
>>> h = 6.62607015e-34  # J s, exact by SI definition

>>> integral = 6 * sp.zeta(4.0)
>>> sigma = 2 * jnp.pi * k**4 / (c**2 * h**3) * integral
>>> float(sigma)
5.670374419184433e-08

```

The accepted CODATA value is $5.670374419 \times 10^{-8}\ \mathrm{W\,m^{-2}\,K^{-4}}$. We have matched every published digit.

## Step 6: differentiate it

`spexial` functions are ordinary JAX functions, so `jax.grad` works on them directly. Let's ask for $\zeta'(4)$:

```pycon
>>> float(jax.grad(sp.zeta)(4.0))
-0.06891126589612538

```

Notice we did not write a derivative anywhere. JAX differentiated straight through the series that `spexial` uses to evaluate $\zeta$.

## Step 7: compile it

`jax.jit` works the same way:

```pycon
>>> float(jax.jit(sp.zeta)(4.0))
1.0823232337111384

```

The same value, now going through a compiled kernel.

## What you built

`sigma.py` computes a measured physical constant to nine significant figures from two special functions and three SI definitions. Run it again from a clean interpreter and you will get the same bytes every time.

You have also seen the three things that matter most when using this library: double precision is not optional, agreement with a known value is how you check a special function, and these are plain JAX functions that compose with `grad` and `jit`.

## Where to go next

- [How to enable double precision](../how-to/enable-double-precision.md) — the variations on step 1.
- [How to use spexial with jit, vmap and grad](../how-to/use-with-jit-vmap-and-grad.md) — steps 6 and 7 in general.
- [Accuracy and domains](../reference/accuracy-and-domains.md) — how far you can trust each function.
- [About precision](../explanation/precision.md) — why step 1 exists at all.
