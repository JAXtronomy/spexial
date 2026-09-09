# `spence` loses all significance at z = 3 ± √3 on the complex branch

- **Filed against:** scipy
- **Verified with:** scipy 1.17.1, numpy 2.4.6, Python 3.11, macOS arm64
- **Status:** draft, not yet submitted

### Describe your issue

`scipy.special.spence` is accurate to machine precision along the real axis, and its complex branch agrees — except at two points, where the complex branch loses every significant digit while the real branch stays exact.

At `z = 3 − √3` the returned value has the wrong sign and a relative error of **104%**. At `z = 3 + √3` the error is **11.6%**. The error grows smoothly as `z` approaches either point, so a neighbourhood of each is degraded, not just the point itself.

### Reproducing Code Example

```python
import numpy as np
import scipy.special as ss
import mpmath as mp

mp.mp.dps = 40
for z in (3 - np.sqrt(3), 3 + np.sqrt(3)):
    truth = complex(mp.polylog(2, 1 - z))  # spence(z) == Li_2(1 - z)
    print(f"z = {z:.15g}")
    print(f"  spence(complex(z)) = {ss.spence(complex(z))!r}")
    print(f"  spence(float(z))   = {ss.spence(z)!r}")
    print(f"  mpmath             = {truth.real!r}")
```

### Error message

```
z = 1.26794919243112
  spence(complex(z)) = np.complex128(0.01125-0j)      <-- 104% error, wrong sign
  spence(float(z))   = np.float64(-0.2518620186090652)
  mpmath             = -0.2518620186090652

z = 4.73205080756888
  spence(complex(z)) = np.complex128(-2.5233731179845442+0j)   <-- 11.6% error
  spence(float(z))   = np.float64(-2.2602610993754793)
  mpmath             = -2.2602610993754793
```

The degradation is smooth in the distance `d` from `z = 3 − √3`:

| d         | 1e-2  | 1e-4    | 1e-6    | 1e-8    | 1e-10   | 0    |
| --------- | ----- | ------- | ------- | ------- | ------- | ---- |
| rel. err. | 1e-14 | 1.5e-12 | 6.6e-11 | 1.1e-08 | 1.6e-06 | 1.04 |

Everywhere else on the real axis the complex branch is at 1e-16.

### Cause

`cephes/spence.c`'s complex counterpart evaluates the continued-fraction / series form whose denominator contains the factor `1 + 4t + t²`, which vanishes at `t = −2 ± √3`, i.e. exactly `z = 3 ∓ √3`. The quotient is `0/0` there and loses significance in a neighbourhood.
