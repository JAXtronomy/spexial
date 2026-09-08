# How to enable double precision

JAX creates 32-bit arrays by default. `spexial`'s accuracy figures all assume 64-bit, so turn double precision on before you create any array.

## In a script or notebook

Set it before anything else runs:

```pycon
>>> import jax
>>> jax.config.update("jax_enable_x64", True)

>>> import jax.numpy as jnp
>>> jnp.zeros(1).dtype
dtype('float64')

```

The setting is process-global and order-sensitive. Arrays created before the call stay float32, so put it above your other imports if any of them build arrays at import time.

## From the environment

If you would rather not touch the code — or you need it to apply from interpreter start, before any import can run:

```bash
JAX_ENABLE_X64=1 python your_script.py
```

To apply it to a whole session, export it:

```bash
export JAX_ENABLE_X64=1
```

## Under pytest

If your project already uses `pytest-env`, set it in `pyproject.toml` so every test runs in 64-bit without a per-test fixture:

```toml
[tool.pytest_env]
JAX_ENABLE_X64 = "True"
```

## Check it worked

Any array will tell you:

```pycon
>>> jnp.zeros(1).dtype
dtype('float64')

```

If you get `float32`, the setting was applied too late.

## Watch for NumPy inputs

Check the dtype of what comes _out_ of your first `spexial` call, not what goes in — a NumPy `float64` array does not guarantee a float64 result. [About precision](../explanation/precision.md) explains why, and why it matters more here than in most JAX code.
