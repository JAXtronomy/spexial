# `betaln(a, b)` is wrong by `min(a, b)` when `min(a, b) / max(a, b)` is subnormal

- **Filed against:** jax
- **Verified with:** jax 0.11.1, jaxlib 0.11.1, CPU backend, x64 enabled
- **Status:** draft, not yet submitted

### Description

`jax.scipy.special.betaln` is exact across most of its range, but returns an answer too large by exactly `min(a, b)` once the ratio of its arguments falls below the smallest normal double.

For `betaln(2, N)` the cutoff is exactly `N = 2**1023`, to the last bit: that is where `2 / N` stops being a normal double. `scipy.special.betaln` is correct throughout.

The consequence downstream is silent and large. `1 / ((N + 1) * B(N - k + 1, k + 1))` is the standard cancellation-free way to evaluate a binomial coefficient, and it comes out a factor of `e**-2` — 86% — low for every `N` above `2**1023`.

### Reproducing code

```python
import jax

jax.config.update("jax_enable_x64", True)
import numpy as np
import scipy.special as ss
from jax.scipy.special import betaln

tiny = np.finfo(np.float64).tiny
cases = [
    (1.0, 1e308),
    (2.0, 1e308),
    (1.5, 1e308),
    (2.0, 8.99e307),
    (2.0, 8.9e307),
    (3.0, 1e308),
    (2.0, 1e300),
]

print(
    f"{'a':>5} {'b':>12} {'a/b':>12} {'subnormal':>10} {'jax':>16} {'scipy':>16} {'err':>7}"
)
for a, b in cases:
    h = a / b
    j = float(betaln(a, b))
    s = float(ss.betaln(a, b))
    print(
        f"{a:>5} {b:>12.3e} {h:>12.4e} {str(0 < h < tiny):>10} {j:>16.8f} {s:>16.8f} {j - s:>+7.2f}"
    )
```

### Output

```
    a            b          a/b  subnormal              jax            scipy     err
  1.0   1.000e+308  1.0000e-308       True    -708.19620864    -709.19620864   +1.00
  2.0   1.000e+308  2.0000e-308       True   -1416.39241728   -1418.39241728   +2.00
  1.5   1.000e+308  1.5000e-308       True   -1062.41509520   -1063.91509520   +1.50
  2.0   8.990e+307  2.2247e-308       True   -1416.17947280   -1418.17947280   +2.00
  2.0   8.900e+307  2.2472e-308      False   -1418.15934965   -1418.15934965   +0.00
  3.0   1.000e+308  3.0000e-308      False   -2126.89547875   -2126.89547875   +0.00
  2.0   1.000e+300  2.0000e-300      False   -1381.55105580   -1381.55105580   +0.00
```

The error is exactly `a` whenever `a / b` is subnormal, and exactly zero otherwise. Verified against `mpmath` at 400 digits (`loggamma(1e308)` is about `7e310`, so the ~1400 being measured needs more than 311 digits of working precision — a lower `dps` silently reports zero error).

### Cause

`jax/_src/third_party/scipy/betaln.py`, in `algdiv`:

```
h = a / b
d = b + (a - 0.5)
...
u = d * lax.log1p(a / b)
```

XLA flushes subnormals to zero on CPU, so once `a / b` is subnormal the quotient becomes `0`, `log1p(0)` is `0`, and `u` — whose true value is `d * (a/b) ≈ b * (a/b) = a` — is dropped entirely. Hence an error of exactly `a`.

The Fortran `algdiv` this is derived from does not flush, which is why `scipy.special.betaln` is unaffected.

### Suggested fix

Where `h` has flushed to zero, `log1p(h)` would have been `h` to the last bit anyway, so `u = d * h = a * (d / b)` — and `d / b` is close to 1, so it cannot underflow. That makes the fix one line, and it changes nothing outside the flushed region because it is only reached when `h == 0`:

```python
u = jnp.where(h == 0, a * (d / b), d * lax.log1p(h))
```

Re-running the table above with that change gives `0.00e+00` error against mpmath for all four subnormal rows, and leaves every other row bit-identical.

### Unrelated observation, from the same measurements

`betaln` is also noticeably less accurate than SciPy's just above the `b >= 8` branch cut — `betaln(2, 8)` is `1.3e-06` off and `betaln(8, 20)` is `8.5e-08` off, where SciPy is exact. That is a separate matter from the flush above and is only mentioned here because the same script measured it.
