# Documentation inventory — `docs`

7 pages · 2,362 words

## Mode distribution (heuristic)

| Mode             | Pages |
| ---------------- | ----- |
| tutorial         | 1     |
| how-to           | 1     |
| reference        | 4     |
| explanation      | 0     |
| non-practitioner | 1     |

## Seams — sections that disagree with their page

**Start here.** Each row is a substantial section whose signals point at a different mode from the rest of its page. These are the EXTRACT, DRAIN and SPLIT candidates: the seams where two pieces of documentation have been fused together. Confirm each one with the compass before acting.

| Page | Page reads as | Section | Line | Words | Section reads as |
| --- | --- | --- | --: | --: | --- |
| `guides/getting-started.md` | tutorial | Enable x64 first | 14 | 59 | **how-to** |
| `guides/getting-started.md` | tutorial | Integer-valued parameters are static | 61 | 59 | **reference** |
| `guides/getting-started.md` | tutorial | Coming from SciPy | 65 | 79 | **reference** |
| `index.md` | how-to | spexial | 1 | 134 | **reference** |
| `index.md` | how-to | A first example | 23 | 52 | **tutorial** |
| `index.md` | how-to | Where to go next | 46 | 43 | **tutorial** |
| `guides/accuracy-and-domains.md` | reference | How to read the table | 5 | 122 | **how-to** |

## All pages, ordered by how mixed they are

`mix` is the runner-up mode's score as a fraction of the winner's — 0.80 means the page is doing two jobs almost equally. The `reads as` column is a prompt for the compass, not a verdict; it counts words and cannot know what a user needs.

| Page | Title | Words | Reads as | Runner-up | mix | In-links | Code | Steps | Tables |
| --- | --- | --: | --- | --- | --: | --: | --: | --: | --: |
| `guides/getting-started.md` | Getting started | 327 | tutorial | how-to | 0.81 ⚠️ | 2 | 6 | 0 | 0 |
| `index.md` | spexial | 230 | how-to | tutorial | 0.80 ⚠️ | 0 | 3 | 0 | 0 |
| `guides/sharp-bits.md` | Sharp bits | 473 | reference | how-to | 0.38 | 5 | 2 | 0 | 0 |
| `guides/accuracy-and-domains.md` | Accuracy and domains | 910 | reference | how-to | 0.28 | 7 | 1 | 0 | 23 |
| `api/index.md` | API reference | 41 | reference | tutorial | 0.00 | 2 | 1 | 0 | 0 |
| `conventions.md` | Conventions | 214 | reference | tutorial | 0.00 | 3 | 0 | 0 | 0 |
| `contributing.md` | Contributing | 167 | _n/a — not practitioner docs_ |  |  | 1 | 2 | 5 | 0 |

## Full section breakdown

Apply the compass heading by heading. `⟵` marks a section that disagrees with its page.

### `guides/getting-started.md` — page reads as tutorial (mix 0.81)

| Line | Heading | Words | Reads as |
| --: | --- | --: | --- |
| 1 | Getting started | 34 | tutorial |
| 5 | &nbsp;&nbsp;The two imports | 30 | — |
| 14 | &nbsp;&nbsp;Enable x64 first | 59 | how-to ⟵ |
| 25 | &nbsp;&nbsp;Composing with JAX transforms | 66 | — |
| 61 | &nbsp;&nbsp;Integer-valued parameters are static | 59 | reference ⟵ |
| 65 | &nbsp;&nbsp;Coming from SciPy | 79 | reference ⟵ |

### `index.md` — page reads as how-to (mix 0.80)

| Line | Heading                      | Words | Reads as    |
| ---: | ---------------------------- | ----: | ----------- |
|    1 | spexial                      |   134 | reference ⟵ |
|   23 | &nbsp;&nbsp;A first example  |    52 | tutorial ⟵  |
|   46 | &nbsp;&nbsp;Where to go next |    43 | tutorial ⟵  |

### `guides/sharp-bits.md` — page reads as reference (mix 0.38)

| Line | Heading | Words | Reads as |
| --: | --- | --: | --- |
| 5 | &nbsp;&nbsp;float32 is the default, and it is not enough | 118 | reference |
| 24 | &nbsp;&nbsp;`Li` accepts only a scalar `z` | 76 | reference |
| 38 | &nbsp;&nbsp;Integer parameters are static, and each value recompiles | 96 | reference |
| 46 | &nbsp;&nbsp;There are no domain errors | 43 | — |
| 50 | &nbsp;&nbsp;Gradients at the edges -- and a gradient that lies | 103 | — |

### `guides/accuracy-and-domains.md` — page reads as reference (mix 0.28)

| Line | Heading | Words | Reads as |
| --: | --- | --: | --- |
| 1 | Accuracy and domains | 53 | — |
| 5 | &nbsp;&nbsp;How to read the table | 122 | how-to ⟵ |
| 14 | &nbsp;&nbsp;Functions | 225 | reference |
| 30 | &nbsp;&nbsp;&nbsp;&nbsp;`gamma` is real-only, and loses precision near the poles | 119 | reference |
| 38 | &nbsp;&nbsp;&nbsp;&nbsp;The modified Bessel functions are accurate to ~$10^{-7}$ | 73 | — |
| 44 | &nbsp;&nbsp;&nbsp;&nbsp;`Li` takes scalar `z` only | 61 | reference |
| 60 | &nbsp;&nbsp;&nbsp;&nbsp;`zeta` has real gaps relative to SciPy | 175 | reference |
| 77 | &nbsp;&nbsp;Reference implementations | 31 | reference |
| 84 | &nbsp;&nbsp;Outside the supported domain | 51 | — |

### `conventions.md` — page reads as reference (mix 0.00)

| Line | Heading                | Words | Reads as  |
| ---: | ---------------------- | ----: | --------- |
|    3 | &nbsp;&nbsp;Naming     |    72 | —         |
|    9 | &nbsp;&nbsp;Signatures |    54 | reference |
|   15 | &nbsp;&nbsp;Behaviour  |    56 | reference |
|   21 | &nbsp;&nbsp;Docstrings |    32 | —         |

---

Next: read the flagged pages and fill in the migration map (`assets/audit-template.md`).
