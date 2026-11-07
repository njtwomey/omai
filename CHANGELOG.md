# Changelog

Versions are released in lockstep for mdml-engine, mdml-render and mdml-meta.

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
