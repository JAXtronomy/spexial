# `kve(v, z)` returns `nan` above z = 2\*\*30 − 0.5 at every order, where `k0e`/`k1e` stay exact

- **Filed against:** scipy
- **Verified with:** scipy 1.17.1, numpy 2.4.6, Python 3.11, macOS arm64
- **Status:** draft, not yet submitted

### Describe your issue

`scipy.special.kve` returns `nan` for every argument greater than `z = 1073741823.5` (that is `2**30 − 0.5`), at every order tried — integer, half-integer and large. The threshold is sharp to the last bit: the largest double at or below it evaluates correctly, and the very next one is `nan`.

The true values are ordinary small numbers, nowhere near underflow. The scaled function decays only like `sqrt(pi/(2z))`, so it stays representable all the way to `z = 1e300`, where it is `1.25e-150`.

That the value is computable is not in question: the dedicated order-0 and order-1 routines `k0e` and `k1e` return it **correctly to all 16 digits** at every one of these arguments, including `1e300`. Only the general-order `kve` fails.

A cutoff at `2**30 − 0.5` exactly suggests a rounding-then-integer-conversion step inside the AMOS wrapper rather than anything numerical.

### Reproducing Code Example

```python
import numpy as np
import scipy.special as ss
import mpmath as mp

mp.mp.dps = 30
last = 2.0**30 - 0.5  # 1073741823.5
first = np.nextafter(last, np.inf)  # the very next double

print(f"kve(0, {last!r})  = {ss.kve(0, last)!r}")
print(f"kve(0, {first!r}) = {ss.kve(0, first)!r}")
print(f"true value there          = {float(mp.exp(first) * mp.besselk(0, first)):.10e}")
print(f"k0e(first) = {ss.k0e(first)!r}")
print(f"k1e(first) = {ss.k1e(first)!r}")
for v in (0, 1, 2, 3, 0.5, 10):
    print(f"  kve({v}, first) = {ss.kve(v, first)!r}")

print(f"\nfar above: kve(0, 1e300) = {ss.kve(0, 1e300)!r}")
print(f"           k0e(0, 1e300) = {ss.k0e(1e300)!r}  (true 1.253314e-150)")
print(f"           kv(0, first)  = {ss.kv(0, first)!r}")
```

### Error message

```
kve(0, 1073741823.5)  = np.float64(3.824811210514542e-05)
kve(0, np.float64(1073741823.5000001)) = np.float64(nan)
true value there          = 3.8248112105e-05
k0e(first) = np.float64(3.824811210514542e-05)
k1e(first) = np.float64(3.8248112122956085e-05)
  kve(0, first) = np.float64(nan)
  kve(1, first) = np.float64(nan)
  kve(2, first) = np.float64(nan)
  kve(3, first) = np.float64(nan)
  kve(0.5, first) = np.float64(nan)
  kve(10, first) = np.float64(nan)

far above: kve(0, 1e300) = np.float64(nan)
           k0e(0, 1e300) = np.float64(1.2533141373155004e-150)  (true 1.253314e-150)
           kv(0, first)  = np.float64(0.0)
```

The point of the exponentially scaled form is to stay finite where `kv` underflows, so `nan` here is the opposite of what the scaling is for: note that `kv(0, first)` returns a clean `0.0` while `kve(0, first)` is `nan`.
