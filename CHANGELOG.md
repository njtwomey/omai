# Changelog

Versions are released in lockstep for omai-engine, omai-render and omai-meta. Up to 0.1.3 the packages were published as
mdml-engine, mdml-render and mdml-meta.

## 0.1.9

Follow-ups from decksmith and from the circular statistics page.
- `useLiveRun`: `stepsPerFrame: 'auto'` times the steps as they run and takes as many per frame as fit in `frameBudgetMs` (default
  8, at most `maxStepsPerFrame`, default 16), so a cheap algorithm plays fast and a dear one does not stall the page; no fixed
  number suits a method whose step cost varies with the data. `catchUpMs` (default 40) bounds the time one tick spends computing
  the steps a slow frame owes: before, a long frame was paid back in one tick of up to half a second of frames' steps (a freeze
  of 500 ms with a 4.7 ms step); past the bound the rest of the debt is dropped and the animation runs a little slow. With cheap
  steps nothing changes. `useLiveRun` is not paced like `useSession` (it computes in the frame that draws), so a step dearer than a
  frame still blocks the main thread: use `useSession` with `workerProducer`.
- `omai-engine/worlds`: `policyPath` follows a policy's own actions from a cell (to the goal, stuck where blocked, round a loop,
  or into a hazard, which ends the path), so a policy that has not learnt yet shows as it is; `greedyPath` follows values and can
  show a full path where the policy is stuck.
- Popups: `Select` takes `container` (as `Popover` does; `menuContainer` is kept), its menu is at least as wide as its trigger, and
  Escape closes the popup before any handler of a surrounding card or dialog sees it (the capture phase).
- Structure diagrams, routing (decksmith round 2): `StructureNode.obstacle` (an invisible node, shape `none` with no label, is now
  crossed by orthogonal edges, and a node can be marked as passable); `StructureEdge.fromSide`, `toSide`, `fromAt` and `toAt`
  (which side an edge leaves and enters by, and where along it); edges that look different (colour, width, dash, kind) get
  ports side by side instead of merging into one port and one colour (`DiagramOptions.merge: false` gives every edge its own);
  a back or skip edge leaves a free side (the autoencoder's dashed edge no longer overdraws the solid one); parallel edges
  entering one wide box are spread; `DiagramOptions.labelPlate` (`'page'`, `'none'` or a paint, for a card that is not the page
  colour), `cornerRadius` (rounded right-angle corners; `cornerRadius` on a scene path), ids for edge labels (`edge:<i>:label`
  with its plate `edge:<i>:plate`) and `origin` for manual layouts (so states of a figure line up). Orthogonal diagrams whose
  edges of different looks met at one port now have shifted ports; straight routes are unchanged.
- Structure diagrams, looks: `fillOpacity` and `strokeOpacity` on scene styles and on node and group looks (pale tinted
  regions); edge `labelColour`, `labelSize` and `labelWeight`; `StructureGroupLook.labelWeight`, `labelPosition` and
  `labelSize`; node shape `'pill'` and `fontWeight` and `fontSize` on the node look; `StructureNode.note` may be a record
  (`position` above, below or beside, a solid or dashed `tie`, colour, size and weight); and TeX labels are sized from their
  typeset width (`texExtent` in `omai-render/maths`), so the boxes around them fit: **existing diagrams with TeX labels come
  out smaller**.
- Docs: `active` keeps a playing run playing when a slide is shown again; pause explicitly to stay paused.

## 0.1.8

Live playback is paced, so a playing session no longer spends its frames building a buffer nobody is watching. Measured on the
main thread with a 5 ms step and a 3000-step run (30 steps a second), the median frame went from 23 ms to 8 ms, the buffer
from about 880 steps ahead of the cursor to 15, and the steps computed in 8 s from 1119 to 249, with the cursor at the same
place.
- `useSession` and `useSessions` (main-thread producer, `'play'` start policy): while playing, computing stops once the head is
  `lookahead` steps ahead of the cursor and goes on when half of that is used. By default that is half a second of playback
  at the player's rate (at least 8 steps); `lookahead` in the hook options sets a number of steps, or `false` for the earlier
  behaviour (the whole run as fast as the frame budget allows). A primed or eager session still computes the whole run, at
  4 ms a frame while it plays (it was 12). The view moves when the head moves, so a resting producer never leaves the player
  waiting on a stale head. `LiveHandle.follow(step, fps?)` takes the rate.
- `SessionPlayer` plays a live run with a preroll: after Play, and whenever the clock catches up with the head, it waits for a
  quarter of a second of steps, or at most 250 ms and one step, then plays; the statistics line shows "buffering" and the steps
  ahead. Pressing Play therefore waits up to about 250 ms before the first motion (it was 40 to 120 ms). A single step
  dearer than a frame on the main thread still blocks the page at the step's rate: use a worker.

## 0.1.7

Fixes and additions from decksmith's work on 0.1.6, including orthogonal edge routing for diagrams. Everything is optional and the defaults draw as before, except the
line height of colour-bar labels.
- Sessions and players: `pause()`, `seek()` and `reset()` now win over a tick already queued, in `useLiveRun`,
  `usePlayback` and `SessionPlayer` (a `seek` after a `pause` could be overwritten by a late tick, so a run parked at the
  start or the end could land a few steps off). Every controller verb reads the current position and playing flag, not the
  values of the render it was made in, so a controller kept from an earlier render acts on the present state.
  `usePlayback` takes `active`, as `useLiveRun` and `useSession` do. `useLiveRun` has `madeWith` (the algorithm, config,
  seed, `maxSteps` and `capacity` the current run was made with; the data a slide draws is not captured, and the TSDoc shows
  how to snapshot it). `LiveHandle.playingNow()` is the synchronous playing flag.
- `chrome={false}` on the panel components (`SessionPanels`, `TapeView`, `RowsPanel`, `FieldPanel`, `ZoomPanel` and the record
  panels), and `FigureChromeProvider` and `PanelFigureProvider chrome` to set it for all panels below; a frame's own `chrome`
  wins. `StructureView` follows the provider.
- Charts: the lines of a multi-line colour-bar label are spaced at 1.3 times the type size (they were one type size apart, so a
  second line touched the value under it); `ticks.fontFamily` and `ticks.fontWeight`; point labels take `labelWeight` and
  `labelFamily`; `maskMatrix`'s `on` may be a function of the row and column, to colour cells by part.
- Controls: `Popover` and `Select appearance="menu"` take `container` (draw into a given element, placed in its coordinates and
  corrected for the scale it is drawn at, for a scaled slide), and `PopoverContainer` sets it for all popups below.
- Worlds: `greedyPath` never steps onto hazardous terrain (holes, lava, cliffs) unless `skip` says otherwise; the path of a world
  scene is the node `path`, the walls one group `walls`, and the explored and frontier overlays of `worldView` the groups
  `explored` and `frontier`, so `tones` and `recolour` reach them.
- Structure diagrams: `route: 'orthogonal'` on `diagram()` (default `'straight'`, unchanged) draws edges with right-angle bends:
  a connector leaves the side or bottom facing its target and enters perpendicular to its side with the arrowhead pointing in;
  straight when the boxes overlap on an axis, else an L or a Z (bend at the middle of the gap) that clears the other nodes with
  the fewest bends, else a detour; parallel segments are spread and edges that share a port merge; labels sit on a plate in the
  middle of the longest segment; bounds come from the routed points. The layered layout routes through its rank channels, the
  manual layout takes `route`, and the other layouts are re-routed (`withOrthogonalRoutes`). New: `StructureEdge.via` (waypoints),
  a multi-line muted `detail`, multi-line and TeX edge labels, `StructureGroup.look` (colours, dashed frame), `StructureNode.note`
  (text under a node), and factor and dot labels that avoid edges and nodes. Known limits: overlapping boxes still fall back to a
  slanted line, and the shapes do not minimise crossings. Existing diagrams are unchanged except where a factor or dot label
  crossed an edge or a node, which now moves.
- Belief propagation: the observation's frame lists `sent`, the messages (by variable and factor name) the last step sent.

## 0.1.6

Convolutional sparse coding and dictionary learning, ported from the AIFN engine. Nothing else in the packages changes.
- `omai-engine/optim`: `convolutionalSynthesis` and its adjoint `convolutionalAnalysis`; `convolutionalSparseCode` (ISTA or
  FISTA, built on `proximalGradientAlgorithm` and `proxL1`) and `convolutionalDictionaryLearning`, each with an
  `Algorithm` form that serialises and resumes and logs on typed `cdl/` paths; `matchFilters` (learnt against known
  filters, up to shift and sign). The signal is a vector or an `[N, 1]` column, so a window of a `TimeSeriesData` works,
  including the `electrocardiogram` dataset (the docs show how). With every code map at zero, the filter update no longer
  divides by zero (the AIFN version returned `NaN`). `initialisation: 'energy'` starts the filters on the most energetic
  stretches and recovers planted pulses for every key; the default stays `'random'`.
- `omai-engine/datasets`: `pulseTrain`, a signal of planted pulses with known filters, sparse maps and positions as its truth.
- `omai-engine/decomposition`: `convolutionalDictionaryMethod` and `convolutionalDictionaryProblem`, a session over
  `pulseTrain` with the probes `cdl/filters`, `cdl/codes`, `cdl/reconstruction`, `cdl/planted` and `cdl/recovery`.

## 0.1.5

Closes the gaps found by decksmith while it moved its slides onto omai 0.1.4, and adds the front page to the repository's
website. Everything new is optional and the defaults draw as before, except where a line below says otherwise.

Packaging and the agent index:
- `omai-render` now ships `dist/react/maths.css`: the build copies stylesheets after `tsc` and fails if a `.css` import in
  `dist/` has no file. (0.1.4 imported a file it did not ship.)
- The diffusion kit (`stochastic`: noise schedules, `forwardSample`, `weightedDiffusionLoss`, `ddimSampler`,
  `diffusionTrainer`, `scoreNetwork`, `networkModel`), the 2-D target densities (`datasets/densities`) and importance
  sampling are in `llms.txt` and `index/*.md`, with `@index` tags. Both `slicedWasserstein` functions (`losses`: returns
  the power unless `root`; `optim`: always the root) are indexed and say which is which.

Render:
- `chrome={false}` on `FigureFrame` and `SceneFrame` draws the figure bare (no expand, export or settings buttons);
  `StructureView` takes `chrome={false}` and `zoomControls`.
- Charts: axis labels follow `figure.fontSize` (`cartesian`, `subplots`); the literal 10 and 12 px sizes in the layers
  are gone, so labels that were 10 px are now the figure's size (11 px at the default); give `fontSize` or `labelSize`
  to pin one. Colour bars take `fontSize` and `width` and reserve room for long labels. `adjustable: 'box'` shrinks the
  plot box for `aspect: 'equal'` instead of widening the ranges. `AxisSpec.ticks` (`interval`, `rotate`, `fontSize`).
  `points` labels take `labelSize`, `labelPosition` and `labelColour` and show with `symbol: 'none'`. `annotations`
  takes `regions` (shaded rectangles and bands); `plotRectOf` and `Projection.rect` give the plot rectangle.
  `vectorField` takes `range` and `ramp`. Matrices take `rowColours`, `colColours` and `marks`, and there is a
  `maskMatrix`.
- `DraggableLine` takes `handle` (`bar`, `square` or `circle`; hollow or solid), `DraggablePoints` takes `shape`.
  `NumberField` takes `parse` and `draftText`, so a value can be shown and typed as words. `Select` takes
  `appearance="menu"` (a library popup in place of the browser's list; `DatasetPicker` passes it on). `Tex`, `Equation`
  and `Derivation` take `align`. `SceneView` and `SceneFrame` take `tones` (draw named nodes in other paints).
- Sessions and players: `useSession` and `useLiveRun` take `active` (nothing computes while false; the tape is kept),
  expose `disposed` and, on a session, `madeWith` (the config and seed the current run was made with). A callback queued
  before `dispose()` no longer throws or advances a disposed run. `SessionPlayer` takes `fps`, `onFpsChange`,
  `eagerToggle` and `statistics`; `SessionPreferencesProvider` and `useSetSessionPreferences` set the defaults (the
  stored value still wins; `storage={false}` keeps nothing); `PanelFigureProvider` and `usePanelFigure` set the size and
  type size of every panel; `SessionPanels` takes `figure`. A finished run reports `progress` 1 against its own length.

Structure diagrams (`omai-engine/scene` `structure` and `omai-render/structures`):
- `StructureNode.position` and `layout: 'manual'`; `look` on nodes and edges (colour, stroke, dash, shape) with the
  shapes `none` (text only) and `stack`; multi-line labels (`\n`); `bidirectional` and `labelAt` on edges; `describe`
  takes `target` (an input to output edge, for an autoencoder).
- Edge labels are drawn above the nodes on a small plate and move off nodes they would cover; group boxes are as wide as
  their labels; the scene grows to hold labels below shapes and notes (so a diagram can be a little larger and a label a
  little off where it was). The tree, bipartite, circular and force layouts honour `rankSep` and `nodeSep`.

Engine:
- Worlds: `valueStyle` for `toScene` and `worldView` values (a paint or a colour ramp, `range`, `maxOpacity`); a `NaN`
  among the values no longer switches all shading off; `greedyPath`. Scenes: `recolour`.
- Belief propagation: `granularity: 'factor' | 'message'` takes smaller steps and ends in the same beliefs; every
  variable's belief is in the observation's frame.
- `inverseSqrtDecay` (and an `inverseSqrt` schedule in the table); `swissRoll` takes `dimension: 2`;
  `importanceEstimate` (unbiased estimate with its standard error).
- Traced states: `Algorithm` documents what a view may read (`observe` and `ctx.log` at emit time; `session.state` is the
  live object; `stateAt` is exact through checkpoints) and takes an optional `snapshot`, which `useLiveRun` uses when it
  keeps a state.

Moving from the earlier library, what changes an answer:
- Random draws come from keys, so a toy example's numbers differ from the old streams.
- `linearModel`'s `alpha` is the penalty on the per-sample deviance: an old `l2` on a summed loss is `alpha = l2 / n`.
  Numbers are rounded half to even.
- A training run's loss is the batch loss before the update, so it is one iteration behind the old figure.
- The diffusion forward process takes continuous times `t`, where the old code drew integer levels 1 to 1000.

## 0.1.4

The project is renamed omai (Observable Methods in Machine Learning); the repository is `njtwomey/omai`. Nothing else
changes in what the code does. The renames are the breaking part:
- The packages are `omai-engine`, `omai-render` and `omai-meta` (were `mdml-*`), so every import path changes:
  `omai-engine/tensor` for `mdml-engine/tensor`.
- `MdmlError` is `OmaiError` and `isMdmlError` is `isOmaiError`.
- The environment variables `MDML_*` are `OMAI_*` (`OMAI_TEST_WORKERS`, `OMAI_SITE_BASE`, ...).
- The schema strings of saved files start with `omai.` (`omai.run/1` for a trace, `omai.fixture/` for fixtures). Tapes saved
  by 0.1.3 and earlier carry the old strings and do not load.
- The Python package of the oracle is `omai_oracle`.

## 0.1.3

A structural review of the core (record: `docs/code-review/`, start at `summary.md`). Much of it is behaviour-preserving, but
the public surface changes in the ways listed here, so read this before upgrading from 0.1.2.

**Numerical results that were wrong are now right.** Rank-deficient `lstsq` and `polyfit` return the minimum-norm solution
(they returned garbage of size 1e15); `matmul` no longer turns 0 x NaN into 0 (so neither does `einsum`); `delaunay` on
cocircular points; `svd`, `norm` and the root finders at extreme scales; empty-axis reductions throw as numpy does; the
samplers validate their parameters; NaN no longer reads as a tie in rank tests or as certainty in `chdtr` and `fdtr`;
`permutationTest` with `pairings` enumerated every permutation n! times; `erfinv`, `betainc`, `gammainc`, `tanhSinh`, the
Cauchy `ppf` tails and `floorDiv(0, -1)` at their edges; `vmap(grad(f))` evaluated `f` twice before reporting a missing rule.

**Speed (results unchanged).** `scheduler.fired`, the gamma and count samplers (about 20x), `einsum` (30x), general `eig`
(n = 200: 15.6 s to 0.26 s), `convolve`, `spmv`, OMP, the stationary kernels' Gram matrices (7x to 20x), the elementwise and
reduction kernels, reverse mode (`grad` of a chain 142 ms to 10 ms) and the decision-surface frame (29 ms to 5 ms).

**Breaking: native arrays become tensors (decision D-009).** Public numeric functions take a `TensorLike` (a tensor, a typed
array or nested arrays, so old calls still compile) and return tensors. Done for `random`, `data`, `stats`, `metrics`,
`numerics`, `optim` and `losses` and their result records (about 180 signatures; the full lists are in `docs/structure.md`,
section 2). Examples: `random.dirichlet`, `categorical`, `multinomial`, `aliasTable` (its fields are tensors), `data.balancedWeights`
and `noiseTransition`, `stats.ranksOf`, `metrics.auc` and the curve and confusion records, the polynomial functions (`polymul`,
`polyroots`, ...), `homography`, `GlmProblem.x` (now the `[n, d]` matrix; `n` and `d` are gone), `constraintNullSpace(C)` (no `k`).
Exposition records stay plain data. New: `tensor.toVector`, `tensor.fromVector`.

**Breaking: one name, one meaning.** 59 generic type names that meant different things in different places were renamed to what
they are: `Matrix` (now `ShapedArray`, `ExprMatrix`, `DenseMatrix`, `Rows`), `Box` (`BoundVectors`, `BoxCoords`, `LayoutBox`, `Aabb`),
`Mat` (`ObservationMatrix`), `Bounds`, `Constraint`, `Schedule`, `Selection`, `Direction`, `Kernel`, `Sampler`, `Tree` and others;
`metrics.matrix` is `shapedArray`. No deprecated aliases. `make lint` now forbids redefining the retired names.

**Breaking: removed.** `Context.backend` and `Context.dtype`, `BackendCapabilities`, `supportsDType`, the `not`, `and`, `or` and
`xor` kernels, `avalOf`, `Tolerance`, `defaultTolerance`, `resolveTolerance`, `data.RowColumn`.

**One failure convention.** Result records carry `status` (the one `RunStatus` vocabulary: `converged`, `completed`, `stopped`,
`diverged`, `failed`) and `converged`; non-convergence does not throw (`fixedPoint` now returns a result at its cap instead of
throwing); invalid arguments throw `MdmlError('value')` (`shape` only for dimensions that disagree, `domain` only for a kernel
hyperparameter range). The factorisations, Krylov solvers and nearest-neighbour queries reject NaN and Infinity input with `value`.
`iterate`, `runAlgorithm` and `runFor` share one stepper and agree on the end status. Tape end statuses of the root finders
and eigen iterations at their cap are `completed`.

**Other additions.** `Kernel.psd`; `Primitive.linear`; `RecordBoundary`, `expositionSamples` and `validateScene`; `useTape(file,
registry, onError)`; optional `label` on `Chart`; opt-in `gradTolRel` on the generic solvers; preconditioned `minres`; optional
`CreateOptions` on `arange`, `linspace`, `eye` and the other constructors; `GroupReport.minSize`; `TestResult.status`.

**Tapes.** Checkpoint states are written with the shared JSON codec (typed arrays as `{ $typed, values }`, non-finite numbers and
-0 survive JSON text). Tapes saved by 0.1.2 still load.

**Tooling.** `make layers` checks the tiers of the core and file-level import cycles; `make api-surface-check` (the digest of every
export's signature) runs in `make verify`, `make check` and CI; the render test helpers are no longer in the published package.

## 0.1.2

- mdml-render: a third start policy for sessions, `start: 'first'` (primed): `useSession`, `useSessions`,
  `useLiveRun` and `SessionPlayer` compute the whole run in the background and park the cursor at step 0, paused,
  until play. `start: 'play'` is today's live (the default) and `start: 'end'` today's eager.
- mdml-render: eager no longer blocks the page. An engine `mode: 'eager'` (and `start: 'end'`) is computed in
  animation-frame slices by the hook, and the cursor jumps to the end when the run is done, instead of computing the
  whole run synchronously when the session is made.
- mdml-render: the live handle has `status` (`preparing`, `ready`, `running`), `progress` (0 to 1), `playing`, `play()`,
  `pause()`, `dispose()` and an `onReady` hook option; `dispose` stops the producer, terminates the worker and
  releases the tape, and unmounting calls it. `useLiveRun` returns `phase`, `progress`, `oldest` and `dispose`, takes
  `start` and `onReady`, and keeps its history in a ring buffer.
- mdml-render: the player's clock no longer slows when drawing is slow (it skips steps instead), and
  `SessionPlayer` takes a `playhead` ref that holds the step the clock has reached.
- mdml-render: hand-made handles need `...STILL_HANDLE` for the new members.
- mdml-engine: the linear-programming solver no longer ends on a singular basis for programs where every ratio ties (a
  nearly tied game's correlated equilibrium): a pivot must be large beside its column (`pivotFloor`, default 1e-7),
  `linprog` retries without that floor before the interior point, and a simplex that reaches its default iteration limit
  falls back too. Measured on a nearly-tied generator: 4 failures in 120,000 solves before, 2 in 240,000 after.
- Tests: five flaky property tests fixed (autocorrelation of denormals, the ray cast at a world's boundary, NMS
  thresholds, the spline's finite-difference step, and the text round trip of a negated sum among them); each was a
  false or too-tight claim, not a library bug.

## 0.1.1

- The agent index covers mdml-render as well as mdml-engine: `packages/render/llms.txt` (with a hand-written
  `llms.header.md`), `packages/render/index/` and `index.json`, published with the package. 161 render entries are
  tagged (67 in `react/`, 94 elsewhere), each with a short call to copy, for example `gridSpec(2, 3)`, `bento(...)` or
  `<SessionPlayer live={live} at={at} onScrub={setAt} />`.
- A generated `llms.txt` at the repository root maps the packages (mdml-engine, mdml-render, and mdml-meta as not
  indexed).
- The collector (`mdml-meta` extract, `scripts/agent-index.ts`) indexes several packages: outputs per package, a check
  of stale files per package and of the root file, the same tag vocabulary for all.
- The headers of the engine's and render's `llms.txt` refer to each other.
- mdml-render: removed exports that nothing in the repository used or that the session layer replaced: `EnvView`,
  `EnvViewProps`, `useEnvStepper`, `EnvStepper`, `EpisodePlayer`, `EpisodePlayerProps`, `MethodControls`,
  `MethodControlsProps`, `SessionControls`, `SessionSettings`, `initialSettings`, `logsOf`, `PlaybackSource`,
  `LiveOptions`, `precomputedSource`, `liveSource`, `EpisodeStep`, `episodeSteps`, `stepCaption`, `episodeTimeline`,
  `SunIcon`, `MoonIcon`, `MonitorIcon`, `MenuIcon`, `SceneFigure`, `bestTrialText` and `isProbePending`.
- mdml-render: the player's buttons are grouped (the run, the playback, the frames).
- mdml-engine: the planar flow's inverse works when `w . u_hat` exceeds 1 (the bisection bracket was too narrow).

## 0.1.0

- First release. mdml-engine and mdml-render were released as separate GitHub releases.
