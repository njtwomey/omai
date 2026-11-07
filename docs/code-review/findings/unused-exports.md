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


# Render: the same triage for `mdml-render` (RD-19, F19)

The engine's triage above reads the core only. This is the same sort for the 1,127 exports of `packages/render` (the
`--full` surface, the `mdml-render/...` rows), by the same signals and the same classes, with the script adapted to the
render package (`scripts/review/unused.mjs` filters on `mdml-engine`; the copy used here filters on `mdml-render` and
takes the declarations from `packages/render/src`; `_shared.mjs` already has `--package=render`). Unused means: the name
occurs in no other source file of the packages and apps (barrels and generated data excluded, tests not counted).
Taken on 2026-11-07 on the tree after the render accessibility, record-boundary and RD-21 to RD-27 changes.

## Counts

| domain       | unused | (a) public API | (b) building block | (c) dead | (a) by tests | (a) by @index | (a) by docs | (a) a type its file uses | functions | constants | interfaces | types |
| ------------ | -----: | -------------: | -----------------: | -------: | -----------: | ------------: | ----------: | -----------------------: | --------: | --------: | ---------: | ----: |
| `react` | 109 | 96 | 13 | 0 | 23 | 0 | 3 | 70 | 33 | 2 | 65 | 9 |
| `views` | 93 | 76 | 17 | 0 | 35 | 1 | 0 | 40 | 51 | 1 | 35 | 6 |
| `charts` | 41 | 37 | 4 | 0 | 11 | 0 | 1 | 25 | 14 | 1 | 25 | 1 |
| `animation` | 15 | 15 | 0 | 0 | 8 | 0 | 0 | 7 | 7 | 1 | 5 | 2 |
| `producer` | 13 | 13 | 0 | 0 | 3 | 0 | 2 | 8 | 2 | 0 | 9 | 1 |
| `theme` | 12 | 12 | 0 | 0 | 11 | 0 | 0 | 1 | 11 | 0 | 1 | 0 |
| `trace` | 9 | 9 | 0 | 0 | 7 | 0 | 0 | 2 | 7 | 0 | 2 | 0 |
| `maths` | 7 | 6 | 1 | 0 | 4 | 0 | 0 | 2 | 5 | 0 | 2 | 0 |
| `structures` | 7 | 7 | 0 | 0 | 1 | 0 | 0 | 6 | 1 | 0 | 5 | 1 |
| `figure` | 6 | 6 | 0 | 0 | 4 | 0 | 0 | 2 | 3 | 1 | 1 | 1 |
| `layout` | 6 | 6 | 0 | 0 | 1 | 0 | 0 | 5 | 1 | 0 | 5 | 0 |
| `svg` | 5 | 4 | 1 | 0 | 3 | 0 | 0 | 1 | 4 | 0 | 1 | 0 |
| `gym` | 3 | 3 | 0 | 0 | 1 | 0 | 0 | 2 | 1 | 0 | 2 | 0 |
| **total** | **326** | 290 | 36 | 0 | 112 | 1 | 6 | 171 | 140 | 6 | 158 | 21 |

`react` is the folder `src/react` with its subfolders (`controls`, `session`, `views`, `interaction`). The review's
count was 300 of 1,154 exports; the method here is the engine's (one source file mentioning the name, tests not
counted), which also counts the interfaces and types of a signature (171 here), so the totals differ in method, not in
substance.

## (a) Deliberate public API (290)

- **112 are named by a test.** Among them the register's fourteen candidates (`episodeChart`, `messageTimeline`,
  `smoothEma`, `runsChart`, `runTable`, `histogramsOverTime`, `normHeatmap`, `easings`, `tweenTimeline`,
  `sweepTimeline`, `runTimeline`, `structureTimeline`, `interpolateScene`, `rasteriseScene`): each is defined in
  `trace/` or `animation/`, tested, exported from its barrel, and is the documented "traced run to figure" and
  "timeline to GIF" API of `llms.header.md`; none has a consumer in the repository (the site's recipes use other
  functions), which is what a library's API looks like. Thirteen carry `@index` or `@example` now; `easings` carried
  neither and got an `@index` in this change.
- **6 are named in the docs** (`docs/structure.md`, the methods catalogue), 1 carries `@index`.
- **171 are interfaces and types that their own file uses** (the props, option and record types of exported
  components and functions: an export that a public signature needs). 65 of the 158 interfaces are in `react`.
- 77 of the 290 are functions or constants named only by a test (no doc, no `@index`); 60 of them are also used inside
  their own file. A unit-tested helper is not dead: if the surface is ever thinned, this is where to look first.

## (b) Building blocks only their own file uses (36)

Candidates to become module-private (`export` dropped, or a move to a `_` module): no test, no doc, no `@index`, a use
only inside the declaring file. Not done: each is exported through its folder barrel (`export *`), so unexporting is a
surface removal the maintainer approves with `make api-surface`, and none is dead.

| export | file | uses in its own file | note |
| ------ | ---- | -------------------: | ---- |
| `barOffset` | `charts/bars.ts` | 1 | has an `@example`; no test, no doc |
| `distributionRange` | `charts/distributions.ts` | 5 | has an `@example`; no test, no doc |
| `reorderColumns` | `charts/multivariate.ts` | 1 | has an `@example`; no test, no doc |
| `MARKERS` | `charts/points.ts` | 2 | no test, no doc |
| `texErrorMarker` | `maths/tex.ts` | 1 | no test, no doc |
| `datasetTags` | `react/controls/DatasetSelector.tsx` | 1 | no test, no doc |
| `environmentTags` | `react/controls/EnvironmentSelector.tsx` | 1 | no test, no doc |
| `validateNumber` | `react/controls/number.ts` | 1 | no test, no doc |
| `useChannel` | `react/session/RowsPanel.tsx` | 5 | no test, no doc |
| `DEFAULT_PREFERENCES` | `react/session/preferences.ts` | 2 | no test, no doc |
| `fullTensors` | `react/session/trajectory.ts` | 2 | no test, no doc |
| `isComparison` | `react/session/useSession.ts` | 1 | no test, no doc |
| `replayLive` | `react/session/useTape.ts` | 1 | no test, no doc |
| `DEFAULT_COLOUR_MODE` | `react/theme.tsx` | 2 | no test, no doc |
| `usePageColourMode` | `react/theme.tsx` | 2 | no test, no doc |
| `GroupContingencyTables` | `react/views/fairness.tsx` | 1 | no test, no doc |
| `GroupDisparityTable` | `react/views/fairness.tsx` | 1 | no test, no doc |
| `GroupReportWarnings` | `react/views/fairness.tsx` | 1 | no test, no doc |
| `svgToCanvas` | `svg/export.ts` | 5 | no test, no doc |
| `FEATURE_VALUE_RAMP` | `views/attributions.ts` | 2 | no test, no doc |
| `screePanel` | `views/components.ts` | 1 | no test, no doc |
| `loadingsPanel` | `views/components.ts` | 1 | no test, no doc |
| `biplotPanel` | `views/components.ts` | 1 | no test, no doc |
| `topicsView` | `views/components.ts` | 1 | no test, no doc |
| `groupColour` | `views/fairness.ts` | 5 | no test, no doc |
| `groupRatesPanel` | `views/fairness.ts` | 1 | no test, no doc |
| `groupGapsPanel` | `views/fairness.ts` | 1 | no test, no doc |
| `groupRocPanel` | `views/fairness.ts` | 1 | no test, no doc |
| `activeCounts` | `views/paths.ts` | 1 | no test, no doc |
| `regularisationPathPanels` | `views/paths.ts` | 1 | no test, no doc |
| `crossValidationPanel` | `views/paths.ts` | 1 | no test, no doc |
| `learningCurvePanel` | `views/paths.ts` | 1 | no test, no doc |
| `mixtureScores` | `views/steps.ts` | 2 | no test, no doc |
| `surfaceObjective` | `views/surfaces.ts` | 1 | no test, no doc |
| `coefficientIntervals` | `views/tables.ts` | 2 | no test, no doc |
| `nodePrediction` | `views/trees.ts` | 1 | no test, no doc |

The view panel builders (`screePanel`, `loadingsPanel`, `biplotPanel`, `topicsView`, `groupRatesPanel`, `groupGapsPanel`,
`groupRocPanel`, `regularisationPathPanels`, `crossValidationPanel`, `learningCurvePanel`) are pieces of the view that
`views/registry.ts` reaches through the view function. The three `react/views/fairness.tsx` components
(`GroupContingencyTables`, `GroupDisparityTable`, `GroupReportWarnings`) are parts of `GroupReportView` in the same
file, reached through it (no test names them).

## (c) Dead (0)

None: every export without a test, a doc or an `@index` is used by its own file. Nothing in render was deleted and the
public surface did not move for this row (`node scripts/api-surface.ts --compare` shows only what the other rows of the
change did).

## What this says

- Render's surface carries no dead code either: 290 of 326 are public by a signal, the other 36 are building blocks
  that their file uses.
- The review's "37 helpers named nowhere" are the (b) list above (36 now); the register's candidates are public API, as
  `findings/render.md` F19 said.
- The decision left for the maintainer is the one the engine's triage left: whether to drop `export` on the 36 and on the
  roughly 60 helpers only a test names. It is a removal from the digest, so it is not done here.
