# The `eval_*` orthogonal polynomials handle infinite arguments inconsistently

- **Filed against:** scipy
- **Verified with:** scipy 1.17.1, numpy 2.4.6, Python 3.11, macOS arm64
- **Status:** draft, not yet submitted

### Describe your issue

Every `eval_*` routine for a classical orthogonal polynomial is, for integer degree, an ordinary polynomial with a limit at ±∞ fixed by its leading term. SciPy returns a mixture of `nan`, `+inf` and `-inf` across the family, with no consistent rule and frequently `nan` where the limit is well defined. The results are also asymmetric in `x` for even-degree polynomials, which are even functions.

This is a family-wide inconsistency rather than a single routine's bug, which is why it is filed as one issue.

### Reproducing Code Example

```python
import numpy as np
import scipy.special as ss

inf = np.inf
rows = [
    ("eval_legendre", lambda n, x: ss.eval_legendre(n, x)),
    ("eval_chebyt", lambda n, x: ss.eval_chebyt(n, x)),
    ("eval_chebyu", lambda n, x: ss.eval_chebyu(n, x)),
    ("eval_hermite", lambda n, x: ss.eval_hermite(n, x)),
    ("eval_laguerre", lambda n, x: ss.eval_laguerre(n, x)),
    ("eval_gegenbauer(a=0.5)", lambda n, x: ss.eval_gegenbauer(n, 0.5, x)),
    ("eval_jacobi(1, 1)", lambda n, x: ss.eval_jacobi(n, 1.0, 1.0, x)),
]
for name, f in rows:
    cells = " ".join(
        f"n={n}: {float(f(n, -inf)):>9} / {float(f(n, inf)):<9}" for n in (2, 3)
    )
    print(f"{name:24s} (-inf / +inf)  {cells}")
```

### Error message

```
eval_legendre            (-inf / +inf)  n=2:       nan / inf       n=3:       nan / inf
eval_chebyt              (-inf / +inf)  n=2:       nan / nan       n=3:       nan / nan
eval_chebyu              (-inf / +inf)  n=2:       nan / nan       n=3:       nan / nan
eval_hermite             (-inf / +inf)  n=2:       inf / inf       n=3:       nan / nan
eval_laguerre            (-inf / +inf)  n=2:       inf / nan       n=3:       inf / nan
eval_gegenbauer(a=0.5)   (-inf / +inf)  n=2:       nan / inf       n=3:       nan / inf
eval_jacobi(1, 1)        (-inf / +inf)  n=2:       nan / inf       n=3:       nan / inf
```

Against the correct limits, read off the leading coefficient:

| call | expected `-inf` / `+inf` | scipy |
| --- | --- | --- |
| `eval_legendre(2, ·)`, `P₂ = (3x²−1)/2` | `+inf` / `+inf` | `nan` / `inf` |
| `eval_chebyt(2, ·)`, `T₂ = 2x²−1` | `+inf` / `+inf` | `nan` / `nan` |
| `eval_hermite(3, ·)`, `H₃ = 8x³−12x` | `-inf` / `+inf` | `nan` / `nan` |
| `eval_laguerre(2, ·)`, `L₂ = (x²−4x+2)/2` | `+inf` / `+inf` | `inf` / `nan` |
| `eval_gegenbauer(2, 0.5, ·) = (3x²−1)/2` | `+inf` / `+inf` | `nan` / `inf` |

`eval_laguerre` is the clearest case: it succeeds at `-inf` and fails at `+inf`, which is the reverse of every other row.

Each of these is finite and correct at large finite arguments — e.g. `eval_gegenbauer(2, 0.5, -1e150)` gives `1.5e300` — so only the infinite input is mishandled. The likely cause is that the hypergeometric and recurrence forms these are evaluated through produce `inf - inf` at an infinite argument.

If infinite arguments are considered out of domain, a uniform `nan` would at least be predictable; the present mixture means callers cannot tell a real result from a failure.
