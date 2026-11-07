# Unused exports: the triage of the core exports no other source file uses (P3.9, F-014)

The review's Phase 0 counted 1,952 core exports that no other source file mentions (tests do not count; the site and pocs
do). This is the sort of them into what is deliberate public API, what is a building block that only its own file uses,
and what is dead. Nothing public was deleted except one type (below): the surface is the library's catalogue (D-002), and
a function with a passing `@example` is a documented, executed part of it.

Regenerate: `node scripts/api-surface.ts --full <file>`, then `node scripts/review/unused.mjs <file> [rows.tsv]` (the
script is in the repository; the second argument writes one row per export with the signals below).

## Method

An export is _unused_ when the word occurs in no other source file of the packages and apps (barrels and generated data
files excluded, as `scripts/review/surface.mjs` does). This run reads **1,970** on the tree after P3.8 stage 4 and P3.10
(1,966 by `surface.mjs`: the script here treats the `testing/` helpers as tests, so 4 names only they use appear). The
deterministic signals per export, nothing guessed:
- tests: a test file (unit, property, golden, type test, example test, a `testing/` helper) names it;
- `@concept` or `@index` in its TSDoc;
- docs: `docs/structure.md`, `docs/methods-catalog/` or any other markdown outside `code-review`, `history` and
  `archive` names it;
- uses: how often its own file uses it, comments stripped (a type or interface its own file uses is part of the
  signature of what it configures);
- `@example`: its TSDoc has one (the standard demands it of every core export, and `make examples` runs it).

Classes: **(a)** deliberate public API: any of tests, `@concept`, `@index`, docs, or a type its own file uses; **(b)**
building block: no signal, but its own file uses it (a candidate to move to a private `_` module); **(c)** dead: no
signal and no use anywhere. When unsure it is (a). The classification is by signal, not by judgement of the design.

## Counts

| domain       |    unused | (a) public API | (b) building block | (c) dead | (a) by tests | (a) by @concept | (a) by docs | (a) a type its file uses | functions | constants | interfaces |   types |
| ------------ | --------: | -------------: | -----------------: | -------: | -----------: | --------------: | ----------: | -----------------------: | --------: | --------: | ---------: | ------: |
| `optim`      |       507 |            505 |                  2 |        0 |          192 |              50 |          36 |                      227 |       230 |         2 |        254 |      21 |
| `numerics`   |       329 |            329 |                  0 |        0 |          190 |               4 |          30 |                      105 |       187 |        16 |        100 |      26 |
| `stats`      |       318 |            317 |                  1 |        0 |          183 |               0 |          12 |                      122 |       169 |         2 |        121 |      26 |
| `metrics`    |       250 |            250 |                  0 |        0 |          180 |               0 |           1 |                       69 |        35 |       140 |         65 |      10 |
| `data`       |       126 |            125 |                  0 |        1 |           57 |               0 |          11 |                       57 |        50 |        11 |         54 |      11 |
| `losses`     |       122 |            122 |                  0 |        0 |           90 |               0 |           0 |                       32 |        84 |         3 |         26 |       9 |
| `dists`      |        85 |             84 |                  0 |        1 |           60 |               2 |           3 |                       19 |        59 |         0 |         18 |       8 |
| `kernels`    |        76 |             76 |                  0 |        0 |           36 |               0 |           0 |                       40 |        34 |         0 |         36 |       6 |
| `exposition` |        48 |             48 |                  0 |        0 |            5 |               0 |           1 |                       42 |         5 |         1 |         39 |       3 |
| `trace`      |        28 |             28 |                  0 |        0 |           13 |               0 |           3 |                       12 |         8 |         1 |          9 |      10 |
| `autodiff`   |        25 |             25 |                  0 |        0 |            7 |               6 |           0 |                       12 |         8 |         0 |         14 |       3 |
| `scene`      |        17 |             17 |                  0 |        0 |            9 |               0 |           0 |                        8 |         9 |         0 |          8 |       0 |
| `tensor`     |        16 |             15 |                  0 |        1 |           10 |               1 |           1 |                        3 |        10 |         0 |          3 |       3 |
| `space`      |         9 |              9 |                  0 |        0 |            1 |               0 |           0 |                        8 |         1 |         0 |          7 |       1 |
| `base`       |         6 |              6 |                  0 |        0 |            3 |               0 |           3 |                        0 |         3 |         2 |          0 |       1 |
| `random`     |         5 |              5 |                  0 |        0 |            3 |               0 |           2 |                        0 |         3 |         0 |          2 |       0 |
| `context`    |         3 |              2 |                  0 |        1 |            1 |               0 |           0 |                        1 |         2 |         0 |          1 |       0 |
| **total**    | **1,970** |      **1,963** |              **3** |    **4** |    **1,040** |          **63** |     **103** |                  **757** |   **897** |   **178** |    **757** | **138** |

Counts are of the unused set before the deletion below (`data` is then 125 and `(c)` 3). By the first signal that makes
an export (a): 1,040 are named by a test, 63 carry
`@concept` (50 of them in `optim`), 103 are named in the docs, and 757 are interfaces or types that their own file uses
(the option, state and result records of exported functions: an export that a public signature needs). 665 of those 757
are interfaces. Functions and constants named only by tests (no docs, no `@concept`) are 956: a unit-tested helper is
not dead, but if the surface is ever thinned, this is where to look first (15 of them are also used inside their own
file; listed below).

## (b) Building blocks only their own file uses (3)

Candidates to move to a private `_` module (not done: each is public in a barrel, and a move is a surface change for
the maintainer to approve).

| export              | file                              | uses in its own file | note                               |
| ------------------- | --------------------------------- | -------------------: | ---------------------------------- |
| `scaleBySparseAdam` | `optim/sparse-updates.ts`         |                    2 | has an `@example`; no test, no doc |
| `steppedLineSearch` | `optim/line-search-algorithms.ts` |                    2 | no `@example`; no test, no doc     |
| `mleSmoothProblem`  | `stats/fit.ts`                    |                    2 | has an `@example`; no test, no doc |

## (c) Dead (4)

No test, no `@concept`, no `@index`, no doc mention, no use in its own file.

| export        | file                    | `@example` | action                                                                                                   |
| ------------- | ----------------------- | ---------- | -------------------------------------------------------------------------------------------------------- |
| `RowColumn`   | `data/schema/common.ts` | none       | **deleted**: no test, no example, no mention anywhere in the repository (a type alias nothing refers to) |
| `isContext`   | `context/context.ts`    | yes        | kept: a type guard of the public `Context`, its example runs                                             |
| `supportRank` | `dists/distribution.ts` | yes        | kept: a public helper of `Support`, its example runs                                                     |
| `avalOf`      | `tensor/traced.ts`      | yes        | kept: the abstract value of a tensor, its example runs                                                   |

The rule for acting was "no tests, no examples and no mention anywhere", because every export sits in a public barrel.
Only `RowColumn` met it. The other three stay on the list for the maintainer: they are tiny, correct and executed by
`make examples`, and a deletion is a surface removal.

## (a) With the weakest signal: tested, used inside their file, in no doc (15)

`metrics`: `completenessScore`, `homogeneityScore`, `dnsmos`, `ergas`, `matchErrorRate`, `melCepstralDistortion`,
`relativeAverageSpectralError`, `rougeLsum`, `scaleInvariantSnr`, `wordInformationLost`, `wordInformationPreserved`;
`numerics`: `nufft1`, `nufft2`; `space`: `readKnobs`; `stats`: `yeojohnsonLlf`. All are catalogue members (scipy,
sklearn, jiwer, torchmetrics ports) or the documented entry of a family; kept.

## What this says

- The surface is not carrying dead code: 99.7% of the unused exports are named by a test, a doc, a `@concept`, or are the
  types of a public signature. "Mentioned in no other source file" mostly describes a library: the callers are users.
- The review's question (public API or dead) has the answer "public API" for 1,963 of 1,970; the real cleanup candidates
  are the 3 (b) and 3 (c) above.
- The 757 types are the number to watch if the surface is thinned: an `XOptions` or `XState` is only as public as its
  function.
