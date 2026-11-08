# Plan for 0.1.5: the decksmith gaps

decksmith (a slide library) moved onto omai 0.1.4 and filed 40 gaps (numbered below as in its list). This is the plan to
close them in 0.1.5 and the record of what was decided for each. Every gap is closed in the library (an option, a fix or a
documented rule), with a test, a gallery recipe where a render option is new, TSDoc, and an `@index` tag where the thing is
a main entry. Defaults keep every existing figure as it is, except where an item is a bug (4, 5, 22).

## Decisions where the survey found the gap differs from the filing

- 12 (`NumberField` `format`): `format` is not applied while typing; what is missing is a `parse` and a `draftText`. Add both.
- 28 (`progress` below 1): it reaches exactly 1 when the run is done. What looks stuck is the numerator (`head / steps`, steps
  default 1000) for a run that stops early. Document it, and make a converged run report its own length as the total.
- 27 (the config a run was made with): already readable (`session.config`, `session.seed`) but not documented. Document it and
  expose it on the hook (`madeWith`).
- 31 (states mutated in place): "states must be immutable" cannot be the rule (the library's own searches mutate). The rule to
  document: what a view draws comes from `observe` or `ctx.log` at emit time; `session.state` is the live object; `stateAt(t)` is
  exact through checkpoints. Add an optional `Algorithm.snapshot` that `useLiveRun` uses when it keeps a state.
- `OmaiRoot` (29, 37) is decksmith's own provider. The library side is a `SessionPreferencesProvider` and plain props.
- 16 (bidirectional edge): drawn in `diagram()` with a reversed arrowhead path, no change to the scene contract.
- 20 (autoencoder diagram): `describe` gets a `target` option (an edge from input to output with a label); no new module.
- 36 (importance sampling error): add `importanceEstimate` (ordinary estimate with standard error) and tag the existing
  `importance.ts` functions, which are untagged.
- `slicedWasserstein` exists twice (`losses/divergence.ts`, `optim/wasserstein.ts`) with different conventions. Tag both and say
  which returns the root.

## Work packages

Each is built by one worker in its own worktree (`agent-<id>-<word>`), tested only for its own files (`nice -n 15 env
OMAI_TEST_WORKERS=1 make verify`), and merged by the main session. At most two workers run at once.

| id  | package                                                   | items                                         | main files                                                                                                                                                                   |
| --- | --------------------------------------------------------- | --------------------------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| R1  | Packaging, frames, controls, maths (render/react)         | 2, 3 (chrome, zoom controls), 11, 12, 37, 39  | `packages/render/package.json`, `react/FigureFrame.tsx`, `SceneFrame.tsx`, `StructureView.tsx`, `react/interaction/*`, `react/controls/*`, `react/MathView.tsx`, `maths.css` |
| R2  | Charts: axes, type size, layers, matrices (render/charts) | 4, 5, 6, 7, 8, 9 (regions, plot rect), 10, 40 | `charts/plot.ts`, `compose.ts`, `points.ts`, `bars.ts`, `fields.ts`, `functions.ts`, `annotations.ts`, `tuning.ts`, `matrices.ts`                                            |
| R3  | Sessions and players (render/session, usePlayback)        | 25, 26, 27, 28, 29, 30, 31 (snapshot), 33     | `session/useSession.ts`, `react/usePlayback.ts`, `react/start.ts`, `session/SessionPlayer.tsx`, `preferences.ts`, `RowsPanel.tsx` and the panels, `trace/algorithm.ts`       |
| E1  | Index, docs and small engine functions                    | 1, 31 (docs), 34, 35, 36, migration note      | `stochastic/diffusion-*.ts`, `datasets/{densities,manifolds}.ts`, `optim/schedules.ts`, `sampling/importance.ts`, `CHANGELOG.md`                                             |
| E2  | Structure diagrams                                        | 13 to 20                                      | `scene/structure.ts`, `structures/{diagram,styles}.ts`, `layout/trees.ts`, `nn/describe.ts`                                                                                  |
| E3  | Worlds and belief propagation                             | 21, 22, 23, 24, 32                            | `worlds/drawing.ts`, `gym/worlds.ts`, `environments/classic-control.ts`, `react/SceneView.tsx`, `pgm/inference/belief-propagation.ts`                                        |

Design of the larger ones:
- R2: first `fontSize` plumbing in `plot.ts` and `compose.ts` (4, then 5's colour-bar text and reserved margin), then `adjustable:
  'datalim' | 'box'` (6), `AxisSpec.ticks` (8) and `plotRect` on the projection (9), then the layers: label size and
  position for points (7), `regions` in `annotations` (9), `range` and `ramp` for `vectorField` (10), `maskMatrix` and row and
  column colours (40). The literal 10 px sizes in the layer files come out so labels follow `figure.fontSize`.
- R3: `useSession` gets `active` (30: nothing computes, the tape stays) and `disposed` (26), `useLiveRun` guards its
  rAF and ticker callbacks after dispose (25). `SessionPlayer` gets `fps`, `onFpsChange`, `eagerToggle` and `statistics` (29, 33);
  the panel figure size goes through a context (33).
- E2: new optional fields on the records: `StructureNode.position`, `look`, `StructureEdge.look`, `bidirectional`, `labelAt`;
  `layout: 'manual'`; multi-line labels; shape `'none'` and `'stack'`; bounds that include outside labels; `rankSep` and
  `nodeSep` honoured by the tree layout; group labels widen the group box; edge labels in a layer above the nodes.
- E3: `valueStyle` for world values, non-finite values skipped, a `greedyPath` helper, a `tones` prop on `SceneView` (24),
  `granularity: 'sweep' | 'factor' | 'message'` and `beliefs` in the observation for belief propagation (32).

## Order and conflicts

1. Wave 1: R2 and E2 (disjoint files). The main session does R1 and E1 meanwhile (small, mostly separate files; R1 does item 2 first
   because 39 needs the css shipped).
2. Wave 2: R3 and E3. E3 follows R1 because `SceneFrame` and `SceneView` share the `chrome` and `tones` work.
3. Merge each package as it finishes (`make manifest` is not needed: no fixtures change; belief propagation keeps its default
   output, so its golden fixtures stay).
4. Last, in the main session only: `make barrels` (new files), `make agent-index`, recipes thumbnails (`make thumbs ARGS="--force"`,
   figures change under 4, 5, 6), the `## 0.1.5` CHANGELOG entry (with the migration note), then the full gate (`make check` and
   `make fixtures-check`).
5. Then `make release` (0.1.5), push main and the `v0.1.5` tag, watch ci, pages and pack, and check the release has the three
   tarballs.

Generated files (`llms.txt`, `index/`, barrels, thumbnails) and `CHANGELOG.md` are touched only by the main session at the end, so
workers do not conflict on them.
