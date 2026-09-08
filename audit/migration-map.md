# Migration map — spexial docs

## Summary

- Pages audited: **7**
- Clean (correct mode, correct place): **0**
- Need moves only: **2** (`api/index.md`, `contributing.md`)
- Need splitting or extraction: **4** (`index.md`, `guides/getting-started.md`, `guides/sharp-bits.md`, `guides/accuracy-and-domains.md`)
- Not practitioner documentation: **1** (`contributing.md`)
- New pages required: **8** (1 tutorial, 5 section landing pages, 2 explanation pages)
- **Gaps found and not filled:** see the summary; the headline is that only one tutorial exists after this work, and `reference/api.md` is only as good as the docstrings behind it.

**Verdict.** The doc set has no Diátaxis structure at all: it has a `guides/` folder, which is the classic unsorted overflow bucket, and three of its seven pages are doing two or three jobs each. The single biggest problem is that **there is no tutorial** — `guides/getting-started.md` is titled like one but teaches nothing and builds nothing; it is a mix of one how-to task (enable x64), reference facts (static integer parameters, SciPy correspondence) and generic JAX orientation. The second biggest is that the same handful of facts — `Li` is scalar-only, `zeta`'s gradient lies on the negative line, float32 is not enough — are each asserted on three different pages with no canonical home, so they will drift.

After the restructure a newcomer has an actual lesson that ends with a computed physical constant; a working practitioner has three task-shaped how-tos; every accuracy fact has exactly one home in reference; and the reasoning behind the sharp edges is argued once, in explanation, instead of being sprinkled through procedures.

## Proposed structure

```
docs/
  index.md                              # site landing — orientation only, not a mode
  tutorials/
    index.md
    stefan-boltzmann.md                 # NEW
  how-to/
    index.md
    enable-double-precision.md
    use-with-jit-vmap-and-grad.md
    port-from-scipy.md
  reference/
    index.md
    api.md                              # mkdocstrings
    accuracy-and-domains.md             # the tables, austere
    conventions.md
  explanation/
    index.md
    precision.md
    edges.md
  about/
    contributing.md                     # non-practitioner
```

`about/` holds the one page that is not practitioner documentation. The root `CONTRIBUTING.md`, `RELEASING.md`, `SECURITY.md`, `CODE_OF_CONDUCT.md` and `CITATION.cff` stay at the repository root where GitHub looks for them; `about/contributing.md` remains the short docs-site view that links to them.

## Per-page map

| # | Current path | Words | Dominant mode | Contaminants | Moves | Destination | Redirect | Notes |
| --- | --- | --: | --- | --- | --- | --- | --- | --- |
| 1 | `index.md` | 230 | landing (how-to signals) | "A first example" is tutorial bait; install steps are how-to; the SciPy-porting caveat is how-to | SPLIT, EXTRACT | `index.md` (rewritten as pure orientation) | n/a — path unchanged | Example → `tutorials/stefan-boltzmann.md`; porting caveat → `how-to/port-from-scipy.md`; install stays (one command, it is orientation not a task) |
| 2 | `guides/getting-started.md` | 327 | **none — three things fused** | §"Enable x64 first" is how-to; §"Integer-valued parameters are static" is reference; §"Coming from SciPy" is how-to; §"Composing with JAX transforms" is how-to | SPLIT, DEMOTE, DRAIN, RETITLE | **dissolved** into 3 pages | ✅ | Not a tutorial and never was — nothing is built. §x64 → `how-to/enable-double-precision.md`; §transforms + §static ints → `how-to/use-with-jit-vmap-and-grad.md`; §SciPy → `how-to/port-from-scipy.md`; the static-int _fact_ → `reference/conventions.md` |
| 3 | `guides/sharp-bits.md` | 473 | explanation | §"`Li` accepts only a scalar `z`" is reference + a how-to workaround; §"Integer parameters are static" is reference | SPLIT, DRAIN, EXTRACT, RETITLE | **dissolved** into 3 pages | ✅ | §float32 → `explanation/precision.md`; §"no domain errors" + §"gradients at the edges" → `explanation/edges.md`; the `Li` fact → `reference/accuracy-and-domains.md`, its vmap workaround → `how-to/use-with-jit-vmap-and-grad.md` |
| 4 | `guides/accuracy-and-domains.md` | 910 | reference | §"How to read the table" is how-to; the four "Caveats worth reading" subsections argue _why_ and editorialize ("this implementation will not give it to you") | DRAIN, EXTRACT, MOVE | `reference/accuracy-and-domains.md` | ✅ | Tables and per-function limits stay and become austere; the reasoning (reflection formula degrading as 1/δ, series/asymptotic cross-over, why a table lookup cannot differentiate) → `explanation/edges.md`. **This page is the canonical home for every accuracy fact.** |
| 5 | `conventions.md` | 214 | reference | §"Docstrings" is contributor policy, not product description | MOVE, EXTRACT | `reference/conventions.md` | ✅ | §Docstrings → already stated in root `CONTRIBUTING.md`; LINK rather than duplicate |
| 6 | `contributing.md` | 167 | non-practitioner | — | MOVE | `about/contributing.md` | ✅ | Unchanged apart from link fixes |
| 7 | `api/index.md` | 41 | reference | — | MOVE, RETITLE | `reference/api.md` | ✅ | mkdocstrings directive + the prettier-ignore guard move verbatim |

**Row count check: 7 rows, 7 source files.** No page is left at the root, and no practitioner page sits outside the four mode sections.

## New pages required

| Path | Mode | Why it is needed | Source material | Effort |
| --- | --- | --- | --- | --- |
| `tutorials/stefan-boltzmann.md` | tutorial | **No lesson exists anywhere.** Every current entry path assumes the reader already knows what they want | None reusable — written from scratch | High: written, then executed end to end so every output block is real |
| `explanation/precision.md` | explanation | The _why_ behind x64 is currently wedged into a how-to step and a bullet list | `guides/sharp-bits.md` §float32 | Low |
| `explanation/edges.md` | explanation | The reasoning behind `nan`-not-raise, the pole degradation and the lying gradient is scattered across two pages | `guides/sharp-bits.md` §§4–5, `accuracy-and-domains.md` caveats | Medium |
| `tutorials/index.md`, `how-to/index.md`, `reference/index.md`, `explanation/index.md`, `about/index.md`\* | landing | `navigation.indexes` needs them; they are where a reader works out which mode they are in | New | Low |

\* `about/` has a single page, so it is listed in the nav directly rather than given a landing page — a landing page pointing at one child is noise.

## Redirect map

**None required, and this is not a shortcut.** Zensical has no redirect or alias mechanism (backlog #23), so a restructure would normally need host-level rules or generated meta-refresh stubs. Here neither is needed: `GET /repos/JAXtronomy/spexial/pages` reports `"status": null` — **the site has never been built or deployed**, the docs were created in the still-open PR #22, and `https://jaxtronomy.github.io/spexial/` has never served a page. There are therefore zero inbound links to break.

| Old URL                        | New URL |
| ------------------------------ | ------- |
| _(none — site never deployed)_ |         |

Mechanism: **not applicable this once.** Enforcement: `zensical build --strict` already fails the build on broken internal links and anchors, and runs in CI (`ci.yml` `Docs` job). The moment the site is deployed, any future restructure must generate stubs with `scripts/make_redirect_stubs.py`, because the option to do nothing expires with the first deploy.

## Execution order

Each increment leaves the tree building and the test suite green.

1. Create the section tree and landing pages; rewrite `index.md` as pure orientation.
2. MOVE the two clean pages (`api/index.md`, `contributing.md`) and fix their links.
3. DRAIN `accuracy-and-domains.md` into austere reference; write `explanation/edges.md` from what comes out. **This is the highest-leverage step** — it is the page most read and most cited.
4. Dissolve `getting-started.md` into the three how-tos.
5. Dissolve `sharp-bits.md`; write `explanation/precision.md`.
6. Write `tutorials/stefan-boltzmann.md` and execute it.
7. Update `mkdocs.yml` nav, then every reference from outside `docs/` (`AGENTS.md`, root `CONTRIBUTING.md`, `README.md`, `skills/spexial/SKILL.md`).

None of these are destructive: every source page is dissolved into named destinations, and nothing is DELETEd, so no sign-off is required. Git holds the originals.

## Open questions

- **Is `Li`'s scalar-only restriction permanent?** It is presented as a property of the algorithm. If it is really a to-do, `how-to/use-with-jit-vmap-and-grad.md` should say so.
- **Should `zeta`'s critical-strip gap be a documented non-goal or a tracked issue?** The docs currently state it flatly; a reader cannot tell whether to wait for it.
