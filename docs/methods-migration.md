# Methods migration: survey and estimate

State: 2026-11-01, after the library and render phases, with a full `make check` and `make fixtures-check` green.
Nothing here has been started. The old `packages/engine/src/methods/` directory is untouched. The walkthrough with the
owner comes first.

Everything under "Survey" was counted from the repository. Everything under "Estimate" is a judgement with one data
source (what the agents cost so far) and roughly a factor of two of uncertainty either way.

## Survey

**The chapter tables** (`packages/engine/src/methods/<area>/README.md`)

| Count | What                                                                                           |
| ----: | ---------------------------------------------------------------------------------------------- |
| 1,284 | method rows across all chapters                                                                |
|   428 | rows with a TODO in their status                                                               |
|   356 | of those 428: every library identifier the row names now exists in `lib/`                      |
|    72 | of those 428: at least one named identifier not found by name (an upper bound on library gaps) |
|   292 | rows whose name matches a `lib/` export directly (a lower bound on rows that are one wrapper)  |
|    34 | rows the status column says are built (the column is stale; no inventory pass has been run)    |

The 72 rows with unfound identifiers cluster in logic and rule learning (9), recommendation and retrieval (5), time
series (5), generative models (4) and reinforcement learning (4). Some are naming mismatches rather than gaps; only
reading each row settles which.

**What the library already provides to a method**
- 210 estimator definitions sit on the iterative, trainer, agent and sweep bases, so each already has `.method` and
  `.session` for free.
- 531 exported `*Algorithm`s, each a stepped routine a session can run.
- Only 10 `defineMethod` uses exist today.

**The old `methods/` directory**: 28 files, 4,569 lines (vision 8 files, linear 6, mixture 3, search 2, dynamics 2, one
each elsewhere). The exemplar page, `pages/01-linear-classifier.tsx`, is 233 lines.

**Render side**: the session layer, panels, overlays, validity greying, slot variants, the settings modal and 187 gallery
recipes (with a coverage test that every record kind is shown) exist, so a method page composes them.

## What a method is now

Because estimators on the bases carry `.method` and `.session`, and the library holds the algorithms, a method is mostly
a declaration: which estimator, which slots (from the tables), which contract paths to show, which panels. That is the
observation behind the proposal below; it is not yet shown on a second chapter.

## Proposal

1. Define a declarative **page spec** (estimator, slots, layout of panels, narrative) and one generic `MethodPage` that
   renders any spec. Convert the exemplar linear-classifier page to it and compare with the 233-line original.
2. Do **one whole chapter** end to end and record the real cost per row. Use that number, not the estimate below, to
   plan the rest.
3. Fill library gaps only when a row needs them, library first.
4. Once pages replace them, remove the old `methods/` files (28 files). This needs the owner's decision.
5. Rebuild the status columns with an inventory script (`scripts/inventory.ts`, not yet written) so the tables stop
   being stale.

## Estimate (tokens)

Reference point: the 35 library and render agents so far spent roughly 14M tokens, about 405k each.

| Route                                      |   Per row |                                             Total |
| ------------------------------------------ | --------: | ------------------------------------------------: |
| Hand-written page per row                  | 10 to 15k |                          13 to 19M for 1,284 rows |
| Declarative spec on a generic `MethodPage` |   4 to 8k | 5 to 10M, plus 0.4 to 0.6M to build the component |
| Library gaps behind the 72 rows            |           |                                           1 to 2M |
| Rows that need a new idea (about a third)  |           |                            2 to 3.5M of the above |
| Main-session coordination                  |           |                                        about 0.5M |

Not included: re-running the gates (they cost wall-clock time, not tokens: about 20 minutes for `make check`).

The step that most reduces uncertainty is step 2. Until one chapter is done, the per-row figure is a guess.

## Open decisions for the owner

- Page-spec approach and permission to replace the old `methods/` directory.
- Rename `lib/sketches` (candidates: `probabilistic`, `probabilistic-structures`).
- Where `lib/exposition` lives (leave it; move it out of `lib/`; or move it out and take record-building out of the
  estimators).
- Whether to push. Main is over 400 commits ahead of origin and nothing is pushed.
