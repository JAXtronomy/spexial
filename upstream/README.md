# Upstream issue drafts

Bugs in SciPy and JAX that `spexial` found while being built and tested against them. Each file is a ready-to-submit issue: a description, a self-contained reproduction, that reproduction's real output, and where known the cause and a suggested fix.

**None of these have been submitted yet.** Each carries its own `Status` line; update it to the issue URL when you file one.

They live here rather than in `docs/`, which is for people using `spexial`.

| Draft | Project | What |
| --- | --- | --- |
| [`scipy-1-spence-complex.md`](scipy-1-spence-complex.md) | SciPy | Complex `spence` loses every significant digit at `z = 3 ± √3` — 104% error, wrong sign. The real branch is exact at the same points. |
| [`scipy-2-kve-large-z.md`](scipy-2-kve-large-z.md) | SciPy | `kve(v, z)` is `nan` above `z = 2**30 − 0.5` at every order, while `k0e`/`k1e` stay accurate to `1e300`. |
| [`scipy-3-eval-poly-at-infinity.md`](scipy-3-eval-poly-at-infinity.md) | SciPy | The `eval_*` orthogonal polynomials return an inconsistent mixture of `nan`/`±inf` at infinite arguments. `eval_laguerre` succeeds at `-inf` and fails at `+inf`. |
| [`jax-1-betaln-subnormal.md`](jax-1-betaln-subnormal.md) | JAX | `betaln(a, b)` is wrong by exactly `min(a, b)` when `min/max` is subnormal, because XLA flushes the quotient. One-line fix suggested. |

Only the last one is worked around in `spexial`, in `comb`; see the comment there. The three SciPy bugs are avoided rather than patched — `spexial` does not route through the affected calls.

## Re-verifying a draft

Every reproduction is a plain script that needs only the packages already in the dev environment. Nothing here is collected by `pytest` (`testpaths` in `pyproject.toml` lists `README.md`, `docs`, `src/` and `tests/`), so these do not run in CI and can go stale. To check one before filing it, extract and run it:

```bash
uv run --frozen python -c "import re,pathlib,subprocess,sys; p=pathlib.Path(sys.argv[1]); c=re.search(r'\`\`\`python\n(.*?)\`\`\`',p.read_text(),re.S).group(1); pathlib.Path('/tmp/repro.py').write_text(c); subprocess.run([sys.executable,'/tmp/repro.py'])" upstream/scipy-2-kve-large-z.md
```

Then compare against the `Output` block in the file. A mismatch means either the bug was fixed upstream — in which case delete the draft — or the versions moved under it, in which case re-measure and update the `Verified with` line.

## A note on references

Two of these were verified against `mpmath`, and both needed care with its working precision. `mp.beta(2, 1e308)` underflows and `mp.loggamma(1e308)` is about `7e310`, so a difference of order `1e3` between two such values needs more than 311 digits — at the default `dps` mpmath silently reports zero error rather than failing. The drafts set `dps` explicitly for this reason.
