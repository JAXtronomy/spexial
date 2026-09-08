# Accuracy and domains

Every `spexial` function is tested against a reference implementation -- `scipy.special` where a counterpart exists, `mpmath` where it does not. This page records, per function, which reference it is checked against, over which input domain it is known to be accurate, and anything a caller should know before trusting it outside that domain.

!!! warning "This page is under construction"

    The rows below are scaffolding. The domains and accuracy notes are being
    filled in as the parity suite is built out; a row marked
    `TODO(docs-content)` has **not** been verified and should not be read as a
    guarantee. Until a row is filled in, treat the
    [parity tests](https://github.com/JAXtronomy/spexial/tree/main/tests) as the
    authoritative statement of what is checked.

## How to read the table

- **Function** -- the `spexial` name, as exported from the top-level namespace.
- **`scipy.special`** -- the counterpart whose name, argument order and convention `spexial` follows. `--` means the function has no SciPy counterpart.
- **Supported domain** -- the input range over which the implementation is tested and expected to be accurate. Outside it, the result may be inaccurate, `nan`, or `inf` without warning: JAX does not raise on domain errors inside `jit`.
- **Notes** -- accuracy caveats, scalar-only restrictions, differentiability, and any deliberate deviation from SciPy.

All entries assume double precision (`jax_enable_x64`); see [Sharp bits](sharp-bits.md).

## Functions

| Function | `scipy.special` | Supported domain | Notes |
| --- | --- | --- | --- |
| `eval_gegenbauer` | `eval_gegenbauer` | TODO(docs-content) | TODO(docs-content) |
| `eval_gegenbauers` | -- (no counterpart) | TODO(docs-content) | TODO(docs-content) |
| `comb` | `comb` | TODO(docs-content) | TODO(docs-content) |
| `gamma` | `gamma` | TODO(docs-content) | TODO(docs-content) |
| `K0` | `k0` | TODO(docs-content) | TODO(docs-content) |
| `K1` | `k1` | TODO(docs-content) | TODO(docs-content) |
| `K2` | `kn` (with $n = 2$) | TODO(docs-content) | TODO(docs-content) |
| `Li` | -- (no counterpart) | TODO(docs-content) | TODO(docs-content) |
| `zeta` | `zeta` | TODO(docs-content) | TODO(docs-content) |

!!! note "The row set is provisional"

    The public API is being reworked. Functions will be added to and removed
    from this table as that lands; the columns are stable, the rows are not.

## Reference implementations

| Reference | Used for |
| --- | --- |
| `scipy.special` | Every function with a SciPy counterpart. |
| `mpmath` | Functions with no SciPy counterpart, and arbitrary-precision spot checks. |

## Outside the supported domain

JAX has no exceptions inside traced code. A function called outside its domain returns `nan` or `inf` rather than raising, and the failure propagates silently through `jit`, `vmap` and `grad`. If your inputs may leave the supported domain, check them yourself before the call, or check the output for `nan` afterwards.
