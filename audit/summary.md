# Documentation restructure — summary

## What changed

The docs had no Diátaxis structure: a `guides/` folder — the classic unsorted overflow bucket — plus four loose pages, three of which were doing two or three jobs each. The biggest problem was that **there was no tutorial**: `guides/getting-started.md` was titled like one but taught nothing and built nothing. A newcomer now has an actual lesson that ends with a computed physical constant; every accuracy fact has exactly one home; and the reasoning behind the library's sharp edges is argued once, in explanation, rather than sprinkled through procedures.

- Pages before: **7** · after: **15** (2 moved, 3 dissolved into 8 destinations, 8 written new, 0 deleted)
- Sections: tutorials 2 · how-to 4 · reference 4 · explanation 3 · about 1
- Executed doc examples before: **23** · after: **48**. Full suite 367 → **385 passed**.

## Key decisions

- **`guides/getting-started.md` was not a tutorial and never had been** — nothing was built, and it offered choices at three points. Dissolved into three how-tos rather than demoted to one, because its four sections served four different tasks.
- **A real tutorial was written from scratch**: compute the Stefan–Boltzmann constant from $\zeta(4)$ and the polylogarithm. Chosen because it needs nothing but the installed package, produces an artifact the learner can re-run, and has a genuine payoff — it reproduces every published digit of a measured constant.
- **`guides/sharp-bits.md` was dissolved, not moved.** It was explanation with reference facts embedded; keeping it whole would have preserved the duplication that made `Li`-is-scalar-only and the lying `zeta` gradient each appear on three pages.
- **`reference/accuracy-and-domains.md` is declared the canonical home** for every domain and tolerance fact. Its "Caveats worth reading" prose — which argued _why_ and editorialized ("this implementation will not give it to you") — was drained into `explanation/edges.md`.
- **`toc.integrate` was removed from the theme.** It is incompatible with `navigation.indexes`, which the four-mode landing pages require.
- `about/` holds the one non-practitioner page. The root `CONTRIBUTING.md` etc. stay at the repository root where GitHub looks for them; `about/contributing.md` links to them.

## Facts I invented or inferred

**None.** Every number in the new reference pages was moved from the existing docs, which were themselves written from measured test tolerances. Every code block and its output was executed before being written down (see below). The physical constants in the tutorial are the exact SI definitions of $k$, $c$ and $h$, and the CODATA value of $\sigma$ quoted for comparison is the published one.

Two statements were _weakened_ rather than invented, because the source did not support them as written: `how-to/port-from-scipy.md` no longer restates specific tolerances (it points at the reference column instead), removing a second copy of facts that would have drifted.

## Things I claimed to verify, and how

- `python <skill>/scripts/check_links.py docs/` — **OK, 15 files**, every internal link and anchor resolves.
- `python <skill>/scripts/check_repo_refs.py . --old-docs /tmp/pristine/docs --new-docs docs/` — clean apart from `audit/inventory.md` and `audit/migration-map.md`, which reference the old paths _deliberately_ as the historical record. Updated outside `docs/`: `AGENTS.md` (6 refs), `CONTRIBUTING.md` (2), `README.md` (1), `skills/spexial/SKILL.md` (1) — including the published `jaxtronomy.github.io/spexial/guides/...` URLs, which would have 404'd.
- `python <skill>/scripts/docs_inventory.py docs` over the finished tree — no duplicated table rows reported. Prose duplication checked by reading; two real duplications found and fixed (the tolerances restated in `port-from-scipy.md`, and the NumPy-downcast mechanism explained on both `how-to/enable-double-precision.md` and `explanation/precision.md`).
- `uv run pytest` — the project's own gate, as CI runs it: **385 passed** (was 367). `uv run pytest docs README.md` — **48 passed**, up from 23 executed regions.
- `uv run zensical build --clean --strict` — **"No issues found"**. `--strict` matters: a plain build exits 0 on unresolved cross-references.
- `uv run prek run --all-files` — all 27 hooks pass.
- **`tutorials/stefan-boltzmann.md` was executed end to end**, and every "expected output" block is real output from that run — captured by running each expression before writing the page, and re-verified by Sybil, which executes the page as a test on every commit.

## Gaps found and not filled

- **Only one tutorial.** It is entirely scalar; nothing teaches array-valued evaluation or the Gegenbauer polynomials. Listed on `tutorials/index.md`.
- **No errors reference.** Exception types and message strings appear only in the API reference and inline in how-tos. Listed on `reference/index.md`.
- **No page explains why `spexial` exists alongside `jax.scipy.special`**, nor the algorithm choices behind each function (series orders, cross-over points, what higher accuracy would cost). Listed on `explanation/index.md`.
- **No how-to for choosing between `spexial` and `jax.scipy.special`** per function. Listed on `how-to/index.md`.
- `reference/api.md` is only as good as the docstrings behind it; it is not independently audited.

Every one of these also appears under "Not yet written" on its section landing page. **No page was created as a stub** — all 15 have real content.

## Contradictions found in the source

- `docs/conventions.md` stated flatly that out-of-domain input "returns `nan`/`inf` rather than raising", while `Li` in fact raises `ValueError` for a bad order. The page had already been half-corrected; the new wording distinguishes traced array inputs from static Python parameters and gives the reason.

## Links

- **Redirect mechanism: none, and none needed.** Zensical has no redirect support (backlog #23), which would normally force host rules or generated meta-refresh stubs. Here `GET /repos/JAXtronomy/spexial/pages` returns `"status": null` — the site has **never been built or deployed**, the docs were created in the still-open PR #22, and no URL has ever been served. Zero inbound links exist to break.
- **This exemption expires at the first deploy.** Any future restructure must generate stubs with `scripts/make_redirect_stubs.py`.
- Enforcement: `zensical build --strict` fails on broken internal links and anchors, and runs in CI as the `Docs` job in `ci.yml`.

## Open questions for maintainers

- Is `Li`'s scalar-only restriction permanent, or a to-do? The docs present it as a property of the algorithm; a reader cannot tell whether to wait.
- Should `zeta`'s critical-strip gap be a documented non-goal or a tracked issue?
