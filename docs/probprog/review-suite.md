# Review of the EP methods, suite, pocs side and docs (design review, 2026-11-11)

Read-only review. Scope: everything around the rules and the compiler: the EP inference methods (`probprog/inference`), the suite
(`probprog/ep`, `probprog/validation`), the pocs side (`apps/pocs`), the docs and the public surface. Rules
(`message-passing/`) and the compiler (`probprog/model/factors.ts`) are other reviewers'. Numbers below were measured on `main`
at `0a1c56f9` (atoms, tolerances and coverage by importing the registry and running it; counts by grep/wc).

## 0. The verdict in one paragraph

The core idea is sound and worth keeping: a model is written once as a spec, compiled to a factor graph, run by an inference
method that follows the library's `defineInferenceMethod` convention, and an "atom" is a small model with a reference answer
and a tolerance. The 92 atoms all run and all converge, 27 of 27 benchmark models run, and every atom is checked against an
independent reference. What was not designed is everything that grew around that idea in one day (24 commits on
2026-11-11 touch `probprog/ep`; `ep-plan.md` has 37 commits): the atom record is a fixture, a UI specification and a reference
implementation fused in one hand-written object, 92 times; the "which EP method applies" decision is made in four places;
the EP knobs exist in four places with different defaults; the references are at least eight separate implementations of "a
posterior on a grid"; the suite lives in a folder named `ep` that a library user would take for the EP capability, while
nothing in the engine imports it and 110 of its 134 runtime exports are used by nobody outside it; the pocs have 92 identical
page files; and the docs are an append-only log whose headline numbers are stale by a factor of three. None of it is wrong in
a way that breaks a test. All of it makes the next ten atoms cost about as much as the first ten, and the claims about
coverage unverifiable by reading.

## 1. Findings, ranked by what they cost future work

### F1. The atom is three things in one record (design, high cost)

`AtomSpec`/`AtomProblem` (`ep/ep-atoms.ts:101-232`) carries: (a) a test fixture (model, data, `tolerance`, `recovery`), (b) a
reference implementation (`exact` closures, `grid`, `exactJoint`, `referenceSource`), and (c) the presentation of a page
(`equations`, `figure` with curves and `answer` closures, `mixtures`, `pair`, `hidden`, `pool`, `groupFrom`, `derive`).
Presentation is in the engine, written per atom: 49 of 92 atoms carry a `figure` (37 captions), 35 a `pair`, 15 an
`exactJoint`, 4 `mixtures`. The same mean-plus-minus-2-sd line figure is rebuilt in `ep-atoms-extended-conjugate.ts:81,89`,
`ep-atoms-extended-scales.ts:60,68`, `ep-atoms-extended-gates.ts:138`, `ep-atoms-scales.ts:55` (7 occurrences of
`const x = [0, n + 1]`: `-extended-conjugate.ts:55`, `-extended-gates.ts:73,889`, `-extended-scales.ts:35`, `-extended-likelihoods.ts:112,236,842`). Consequence: a new atom is 80 to 250 lines, most of it not about the factor under test; the engine
carries a chart vocabulary that belongs to render; changing a chart means editing atoms.

What is fine: `defineAtom` validates its inputs (`ep-atoms.ts:245-253`); `atomRun`, `atomReference`, `atomCheck`, `verifyAtom` are a
coherent pipeline with TSDoc and examples; knobs are a `Space` (286 knobs over 92 atoms, rendered by `SpaceControls`).

### F2. "Which EP applies" is decided in four places and the EP knobs live in four places (design, high cost)

Route choice (plain EP, else EP in a grid over hyperparameters, else refuse):
- `ep-atoms.ts:433-444` `atomVerdict` (runs `epSpecMethod(bound).check`, then `epGridMethod(bound).check`),
- `ep-atoms.ts:452-454` `atomRoute` (a second `epSpecMethod(...).check`),
- `ep-atoms.ts:603-` `atomRun` calls `atomRoute`, then `atomVerdict` again (so one run does up to four verdict computations
  and binds the model repeatedly) and wraps the grid method with a cast, `gridMethodOf(...) as never` (`:447`),
- `ep-coverage.ts:84-88` repeats the same two-step for benchmarks (`via: 'ep' | 'ep-grid'`).
The comment on `mp.ts:14` says "Nothing chooses among the three: the user does". In fact the suite and coverage choose
automatically. `epMultiStartMethod` is not on the route at all (only the changepoint page and tests use it), and the
coverage cannot say "runs by EP-multistart".

Knobs:
- `messagePassingKnobs('ep')` (`message-passing/named.ts:324-343`): schedule, granularity, damping, improper, tol (1e-8), maxSweeps (200), restarts.
- `epSpecMethod` adds `power` on top (`mp.ts:129-147`).
- `epGridKnobs` (`ep-grid.ts:90-117`) and `epMultiStartKnobs` (`ep-multistart.ts:79-111`) redeclare `tol`, `maxSweeps`
  (200 and 300), `damping`, and ignore schedule, improper, power and restarts. Grid and multistart cannot be combined with
  fractional EP or a schedule, and nothing says so.
- `EpAtomPage.tsx:26-34` builds a fifth description: a subset of `messagePassingKnobs('ep')` (so `power` is hidden from the
  page while `softmax-regression` sets `power: 0.5` in `atom.ep`).
- `atom.ep` is `Record<string, number|string|boolean>` and unknown keys are accepted silently: `atomRun(p, {dampng: 0.5})`
  runs and reports `ran` (verified). 14 atoms carry overrides (`damping: 0.5`, `tol: 1e-4`, `power`).
The three method files also each hand-roll "bind, `toFactorGraph`, `compileGraph`, try/catch": `_support.ts:190 prepare`,
`ep-grid.ts:356 compileAt`, plus `gridRecord` in `ep-atoms.ts:457` and `evidenceByRows` in `mp.ts:191` (a hidden second EP
run with `maxSweeps: 400` hard coded) which recondition and rerun the model.

Fine: all EP methods follow `defineInferenceMethod`, have a typed contract (`EpPath`, `EpGridPath`, `EpMultiStartPath`), a
`check` returning `Verdict {applies, reason, missing, hint}`, `posterior(state)`, serialisable state, anytime snapshots, and are
registered in `methodKinds` (`models.ts:140-142`) so `methodSpace` offers them as a union with refusals shown. The verdict
reporting is good. CLAUDE.md's "a model is an objective, the solver is a slot" does not apply to EP (no objective is
minimised); it does apply to the neighbours `laplaceMethod`/`svi` and is not violated here. EP/VMP/BP are one `specMethod`
with `rule === 'ep'` branches in six places (`mp.ts:129,134,153,198,249,294`); a small strategy object per rule would be cleaner but
this is polish, not debt.

### F3. Reference implementations: at least eight "posterior on a grid" (duplication, high cost)

| implementation | where | used by |
| --- | --- | --- |
| `gridPosterior` over a program (Laplace box, refinements) | `validation/grid.ts:175` | `atomReference` for `reference: 'grid'` (31 atoms) |
| `quadratureMarginal` (1-D midpoint) | `ep-atoms-quad.ts:87` | scales, mixtures |
| `gridMarginals` (tensor grid in a box) | `ep-atoms-quad.ts:135` | extended-likelihoods (6 call sites), others |
| `hyperPosterior` (1-D log-spaced + `mix`) | `ep-atoms-quad.ts:280` | extended-conjugate (2), extended-scales |
| `posterior3` (3-D, three adaptive passes) | `ep-atoms-covariance-ref.ts:51` | covariance (5 call sites) |
| `clutterPosterior`/`outlierMixturePosterior` (4-D quadrature, `points = 20`) | `ep-atoms-gibbs.ts:269,397` | gates |
| `slopeGrid`, `skillGrid`, weights `gridMarginal`, kernel `logNormalPdf` grid | `ep-atoms-multiclass.ts:63`, `ep-atoms-ordinal.ts:436`, `ep-atoms-weights.ts:49`, `ep-atoms-kernels.ts:35-50` | one family each |
| `spikeSlabPosterior`, `mixtureOfRegressions`, `zeroInflatedPoisson` enumerations | `ep-atoms-gibbs.ts`, `validation/enumeration.ts` | gates, counts |
Plus `epGridMethod` (EP inside a grid) as a ninth. They share no grid code, no normalisation convention, no `ExactMarginal`
adapter except the structural type, and each atom family chooses its own point counts (101, 41, 1500, 34/52/70, 20).
Mislocated: the documentation (`ep/index.ts:5-6`, `structure.md:3855-3857`) says the exact references live in
`probprog/validation`; in fact `ep/` imports only `gridPosterior` and `mixtureOfRegressions` from it (2 imports, `ep-atoms.ts:37`,
`ep-atoms-regression-gates.ts:13`) and holds seven references of its own. `validation/exact.ts` already has `normalNormal`,
`betaBernoulli`, `gammaPoisson`, `dirichletCategorical`, none used by the conjugate atoms, which recompute the closed form by
hand (`ep-atoms-conjugate.ts:41-45,70-75,101-104`). The `ExactMarginal` type (ep) and `ExactPosterior` (validation) are two
types for one idea.

Reference-equals-model risk: the reference density is written a second time by hand (e.g. the mixing-precision-integrated
Student-t log density at `ep-atoms-extended-likelihoods.ts:82-114`). That is a legitimate independence choice, and is the
right call for speed, but it should be a declared `reference: {kind:'density', logDensity, box}` and not an arbitrary
closure on the problem (`get exact()` getters, `once(...)` lazy wrappers).

### F4. Small helpers copied around (duplication, cheap to fix)

- `const lg = (x) => item(lgamma(x))`: 6 copies (`ep-atoms.ts:296`, `-priors.ts:28`, `-gibbs.ts:19`, `-nonparametric.ts:20`,
  `-extended-conjugate.ts:140`, `-extended-likelihoods.ts:31`).
- `const sum = (xs) => xs.reduce(...)`: 4 copies (`-conjugate.ts:21`, `-priors.ts:29`, `-extended-conjugate.ts:141`, `-extended-likelihoods.ts:32`).
  CLAUDE.md says number-array kernels come from `blas.ts` or tensor ops.
- `function once<T>(make)`: 3 identical copies (`-extended-likelihoods.ts:43`, `-ordinal.ts:30`, `-multiclass.ts:39`).
- Normal CDF three times: `normalCdf` (`-quad.ts:39`), `phi` (`-extended-likelihoods.ts:35`, A&S 26.2.17), and `ndtr` in numerics.
- `normalPdf` is exported from `probprog/ep` and used 23 times by atoms; `beliefDensity` and `beliefCredibleInterval`
  (`ep-atoms.ts:278`, `ep-extended.ts:42`) re-derive moment-matched gamma and beta densities and quantiles that
  `omai-engine/dists` already has.
- `groupIndex`, `FIGURE_GROUPS`, a `recoveryOf` helper (`-extended-likelihoods.ts:49`) and a `GAUSSIAN_RECOVERY` constant that
  other files restate; 2 `EXACT` tolerance constants (`-conjugate.ts:15`, `-priors.ts:31`) with the same numbers.
- 20 `as never` casts in `ep/` (10 in `ep-atoms.ts`) around `session`, `posterior`, grid method typing; `ep-grid.ts` and
  `ep-multistart.ts` have none, so the casts are caused by `atomRun` driving three method types through one variable.

### F5. Basic / learned "twins": a convention, not an abstraction (design, medium cost)

63 basic atoms and 29 `extends` ones. Three basics have two twins (`poisson-log`, `student-t`, `prior-normal-weibull-scale`).
Three words for one idea (`learned` in names, `extends` in the field, `extended` in file and function names:
`ep-extended.ts`, `ep-atoms-extended-*.ts`). The twins are written from scratch: `gaussianLearnedProblem` (`-extended-conjugate.ts:35-114`)
shares with `gaussianProblem` (`-conjugate.ts:26-46`) only the idea. Placement is inconsistent: 3 twin pairs are in separate
`extended-` files, but the ordinal, multiclass, kernels and covariance twins sit beside their basic atom. The files are organised by
the work package that added them, not by family: `family` is an untyped string whose values do not match the file names
(`mixtures` file has `family: 'gates'`; `weights` has `'gate weights'`; `multiclass` and `ordinal` have `'likelihoods'`;
`regression-gates` has `'gates'`). `ep-atom-list.ts:114-120` says "in the order of the coverage page: the gates and mixtures
first, then the rest as they are added": the list is a chronological log (92 hand-written imports plus 92 list entries; a new
atom is edited in 3 files here and 2 new files in pocs).

Tolerances are per-atom: 50 distinct `(meanZ, sdRatio, ks)` triples and 75 distinct reasons over 92 atoms (the top triple is used by
8). Recovery specs: 10 different `(seeds, coverage, zrms)` combinations over 29 atoms. CLAUDE.md asks for tiered tolerances
(1e-12 exact ... 1e-6 iterative); a handful of named tiers plus a measured override would say far more than 50 hand-tuned
triples. Honest and useful, though: each tolerance states why (`reason`), and the numbers are measured, not tuned to pass.

### F6. Coverage, gaps and directions: typed in form, prose in substance, and the checks have gone vacuous (high cost for trust)

- `epAtoms` now has 0 `planned` atoms and `epCoverage()` has 0 planned benchmarks (measured: 92/92 atoms, 27/27 benchmarks).
  So `AtomSpec.expected: 'runs'|'planned'`, `AtomSpec.plan` (`ep-atoms.ts:200-203`, "where in ep-plan.md the missing rule is"),
  `CoverageRow.declared`, the second `describe.each` of `ep-atoms.test.ts`, and the "every refused item is named by a gap"
  test (`ep-gaps.test.ts:10-14`) test nothing today. All 13 `refuses` arrays in `ep-gaps.ts` are `[]` (grep: 13 of 13).
  The mechanism is sound (the gaps would be derived from refusals) but the data it reads has run out; what remains is a roadmap
  in prose.
- `refuses` is hand-kept, not derived: a gap lists names of atoms/benchmarks that a refusal reason should have produced.
  Better: tag each `missing` string of the verdict with a gap id in the library (the verdict already has `missing: string[]`),
  and derive `refuses` from the coverage run.
- The gap texts hard-code facts that drift: `ep-gaps.ts:170` says "Variational message passing runs 78 of the 86 atoms" (there are 92);
  `matrix-larger` (`:55-65`) has `status: 'open'` and a summary beginning "Closed 2026-11-11 (`agent-pp191-matrix`)";
  `ordinal-random-cuts` cites `_grid.ts` for a "three-axis version" while `_gridn.ts` exists; 2 lines carry dates and 1 an agent id.
  The `docs-drift` gap (`:176-185`) is the document admitting the problem.
- `EpDirection.pages: ['ep-gate', ...]` (`ep-gaps.ts:245`) are slugs of pocs pages: the engine data points into an app.
- Floors: `ep-coverage.test.ts:20` has a test title that is a changelog (9 of 27 ... 46 of 47 ... 92 of 92 ... 86 of 86 ... 84 of 86 ...)
  and asserts `>= 92`, `>= 27`; the VMP test asserts `>= 78`, `>= 76`, `behindEp <= 8` and carries history in its title (`:34`,
  "36 before ..."). `ep-extended.test.ts:51` asserts `>= 12` extended atoms (there are 29). Floors that sit far below reality
  protect nothing; the real invariant is now `atomsRun === epAtoms.length`. `ep-evidence.test.ts:12-60` is a third hand
  table (atom name, nats tolerance, why) kept apart from the atom's own tolerance.
- `ep-coverage.ts` also holds `vmpCoverage` (`:100-228`): a variational measurement inside the `ep` folder, with its own knob
  constants (`VMP_KNOBS`, `VMP_STEADY_KNOBS`). The folder is really the message-passing suite.

Fine: `epCoverage()` itself is a clean idea (measure, do not remember; 16 s for everything, measured), and `Verdict` carries
machine-readable `missing`.

### F7. The folder is misnamed for what it is and leaks (public surface, medium cost)

`omai-engine/probprog/ep` reads as "the EP capability". What a library user needs (run EP on my model, know whether it applies,
read beliefs, check calibration) is in `probprog/inference` (`epSpecMethod`, `epGridMethod`, `epMultiStartMethod`) and in
`message-passing`. `probprog/ep` is a testbed. Evidence: nothing in `packages/engine` imports it; all 137 importers are pocs
pages and tests; of its 134 runtime exports, 110 are not referenced by name outside the folder (93 are atom constants that
pages reach through `epAtom('name')`, 17 are internal reference helpers: `normalPdf`, `normalCdf`, `posterior3`, `gridMarginals`,
`hyperPosterior`, `mixtureGibbs`, `clutterPosterior`, `outlierMixturePosterior`, `spikeSlabPosterior`, `twoComponentGate`,
`quadratureMarginal`, `momentMatch`, `samplesMarginal`, `logWishart2`, `normalInverseGammaPosterior`, `simulationKnobs`,
`atomBeliefs`). `docs/api-surface.txt` counts 184 exports for `probprog/ep` (types included). The inference and message-passing
folders use the `_x.ts` prefix for non-exported files (`_support.ts`, `_relabel.ts`, `_grid.ts`); `ep/` does not, so `export *`
publishes everything. The agent index lists the 92 atoms as `@index estimator` (92 of about 120 lines of the `probprog/ep`
section of `index/probprog.md`), which tells an agent that searches for estimators that these fixtures are estimators.
`atomRun` is actually the generic "run EP on a bound model through the right route, with a beliefs history and a message record"
that the library lacks as a function and that every program page re-implements.
Naming: atom / problem / program / direction / gap / extended / learned / twin / planned / runs are all in use; `program` means
a closure program in the engine (`toProgram`, `Model`) and a poc page (`ep-program-*`) at the same time.
`structure.md:3855-3865` omits `ep-atoms-covariance*.ts` from its file list.

### F8. pocs: 92 identical pages, programs hand-built, a parallel recovery check (medium cost, needs the user's go-ahead)

- 92 `ep-<atom>/Page.tsx` files differ only by the slug (`<EpAtomPage atom={epAtom('beta-bernoulli')} />`; 92 of 92 share one hash
  after replacing the slug), and 92 `poc.ts` restate `title`/`summary`/`tags` from the atom (`summary` differs from `atom.what` in 7,
  tags omit a factor name in 54). The test (`ep-atoms.test.ts:7-16`) is named "pages generated from the registry" and checks
  only titles. Hand copies were chosen to fit the registry rule ("one folder per poc, nothing else needs editing",
  `registry.ts:4-6`), but an atom page needs no code of its own.
- The stale-file-list trap: `registry.ts:5-6` and `atomSource.ts:5` are `import.meta.glob` lists. The README documents a server
  restart after adding a page for Tailwind's class scan; with 92 hand-made folders every new atom requires touching the pocs
  and a restart; with a registry built from `epAtoms` (below) a new atom appears with engine HMR and needs no new pocs file.
  (I did not test whether Vite re-evaluates a glob when a file is added; the README's documented trap is Tailwind.)
- `atomSource.ts` shows the code that ran by taking `// region <atom.name>` from the engine files and the `// region inference`
  lines of `atomRun`. That honours the "code that ran" rule, but the inference lines shown are `epSpecMethod(bound)`, while the
  run may go through `epGridMethod` (2 benchmarks and the kernel atoms), so the shown code is not always what ran. A route
  function (F2) would make the shown snippet and the run agree.
- The 9 programs (`ep-program-*`, 5 410 lines with tests) repeat one shape: `simulate -> model -> infer -> reference -> check ->
  Page`: each has its own `program.ts` (50-146 lines), `compute.ts` (up to 464), `Page.tsx` (139-419), and often a `reference.ts` with
  a hand-written Gibbs sampler (tracker, changepoint, changepoint-extended). `shared/ProgramPage.tsx` (147) gives the layout,
  not the compute. They are in effect atoms with a sampler reference and no registry.
- `shared/ep/recovery.ts` (`recovery`, `Tolerance {coverage: 0.8, zrms: 1.6}`) re-implements what the engine has as
  `atomRecovery`/`CalibrationRow` (`ep-extended.ts:97,123`) and `recoverySummary` (`ep-recovery.ts:82`); `changepoint-extended`
  has its own `calibration.ts`/`Calibration.tsx`, parallel to `atomCalibration` and `ExtendedParts.tsx`. CLAUDE.md: "anything
  with substance goes into lib/".
- Run mechanism (`run.ts`, `useRun.ts`, `runWorker.ts`, `workerPool.ts`, `atomReference.worker.ts`,
  `atomCalibration.worker.ts`): small, well documented, with a test; the worker protocol (`serveRun`/`workerTask`) is fine. Not a
  finding except that the atom pages call `atomVerdict` and `atomRun` (which verdicts again) on the main thread inside
  `useMemo` (`useAtom.ts:46-60`): up to four binds per knob change.
- Caution: CLAUDE.md says pocs are never edited unasked. Every pocs step below needs the owner's explicit request.

### F9. Docs: an append-only log, stale headline facts, and no design document (high cost for newcomers)

`docs/probprog/ep-plan.md`: 669 lines, 82 840 bytes, 37 commits, written first as a plan ("what we have, what is missing, the plan")
and then extended by every agent with a dated section. Sections are numbered 1-9, then 12, 10, 11, 13, 14, 15, 16, 16, 18 (two
"16", no 17, 12 before 10) and carry agent ids (10 mentions of `agent-pp`). Sections 1-6 (the plan) describe a world that no longer
exists, and sections 7-18 are the changelog. There is no place that says how EP is put together.

Claims checked against the code (each is stale or drifting):
1. `ep-plan.md:11` "The engine has 35 factor operators and eight message families": `defaultOperators()` registers 65 listed
   operators plus `gate` (`message-passing/operators/registry.ts:79-148`); families now include wishart, normal-wishart (10).
2. `ep-plan.md:15` "EP runs on 9" (of 27 benchmark models) and the whole blocker table (`:16-31`) of 18 refusals: now 27 of 27 run
   (`epCoverage()`, measured), 2 of them through `ep-grid`.
3. `ep-plan.md:~131-150` (section 5) "so the number ... (9 of 27 today; target at least 22)": the test floor is 27.
4. `ep-plan.md:152-156` (section 6) "work was paused ... half-done in flight": none is.
5. `ep-priors.md:127-143` "Gaps, in order of value": #3 matrix families (Wishart, normal-Wishart, LKJ-like: built in commit
   `c85ebf69`), #4 random Weibull shape and Student-t df (atoms `weibull-shape-learned`, `student-t-df-learned` run), #5 softmax
   (atoms `softmax-regression*` run), #2 `beta-two-shapes` (atom `prior-beta-two-shapes` runs): four of six gaps are closed.
6. `ep-and-gam.md:7-13` "one is the real remaining gap (an unknown smoothing parameter on a whole vector of coefficients)",
   contradicted by its own "Next steps" (`:132-136`: done).
7. `CHANGELOG.md` `## 0.1.13`, line 8 says `gaussian-gamma-precision` "now has an EP rule", line 232 (same section) says
   "`gaussian-gamma-precision` has no EP rule".
8. `CHANGELOG.md:226-228` "EP runs only in `benchmark/item-response-scoring` and `benchmark/clutter-ep`": 27 of 27 benchmark models run
   under EP in the coverage measurement.
9. `CHANGELOG.md:235` "The `gate` node (mixtures) is not built": the model layer maps `ops.pick`/`where` lookups to the engine
   gate and the gate atoms run (the explicit `gate` node is still reserved, `spec.ts:215`, so this one is half true).
10. `CHANGELOG.md` 0.1.13 does not mention the 92-atom suite, `epGridMethod`, `epMultiStartMethod`, the Wishart families,
    `vmpCoverage` or `epCoverage`: the release entry covers the first of about 20 packages.
11. `structure.md:3855-3857` and `ep/index.ts:5-6`: references are said to be in `probprog/validation`; they are in `ep/` (F3).
12. `ep-atom-list.ts:114-120` "in the order of the coverage page: the gates and mixtures first": chronological.
13. `ep-atoms.ts:200-203`, `plan?:` "where in ep-plan.md the missing rule is": no atom is planned; `ep-plan.md` has no atom index.
14. `ep-gaps.ts:170` "78 of the 86 atoms" (92); `ep-gaps.ts:55` status `open` with a summary that starts "Closed".
15. `mp.ts:14` "Nothing chooses among the three: the user does", while `atomVerdict`/`epCoverage` choose between `ep` and `ep-grid`.
16. `apps/pocs/src/groups.ts` group summary "Problems EP cannot run yet are listed with the reason": none are.
17. `ep-plan.md:279-289` (section 10): "a gap that names something EP now runs is stale" and a test enforces it; but the 13 gaps
    name nothing, so the test cannot go red for any of them.

The generated agent index for this area is accurate but noisy: the `probprog/ep` section is 113 of 347 lines of `index/probprog.md`
(a third of the whole probprog index) and 92 of them are atom constants; the three real entry points
(`epSpecMethod`, `epGridMethod`, `epMultiStartMethod`) are in the inference section as `estimator`, correctly.

What is fine: the three `ep-notes-*.md` are honest, dated, with measured numbers and a "searched and reused" section;
`ep-priors.md` is a useful atlas (what runs, how to write a prior). Their problem is only that status drifts and nothing links
them to code that would fail when they do.

## 2. What is fine and should be kept

- The pipeline: spec -> `toFactorGraph` -> registry of operators -> `epAlgorithm` -> `Posterior`, with a verdict that names the
  node and the missing rule; refusal reasons are specific.
- `defineInferenceMethod` conformance of EP, grid, multistart, VMP and BP; typed tape paths; resumable and snapshot-able state;
  `methodSpace` union. EP respects the Algorithm/session contract and the exposition records (`modelRunFromTape`).
- Atoms are independent of the rule being tested (reference by quadrature or closed form, not by EP), tolerance has a stated
  reason, all 92 converge (verified), the slowest takes 0.75 s (`ordinal-probit-learned`), coverage of everything takes 16 s.
- The recovery/calibration machinery (`atomRecovery`, `atomCalibration`: coverage, bias, zrms, pooled members) is the right
  abstraction and works on any problem that carries truth; it should be reused beyond atoms (F8).
- The run mechanism in pocs (`startRun`, `useRun`, `serveRun`) is small and documented.
- `epAtom('name')` lookup keeps pages independent of atom constants.

## 3. Target structure

Principle: separate (a) the EP capability a user imports, (b) the fixtures that verify it, (c) the references the fixtures use, and
(d) the presentation. Names below are proposals; `docs/structure.md` section 3855 is updated in the same change.

```
lib/message-passing/                      unchanged (rules, engine)
lib/probprog/model/factors.ts             unchanged (compiler)
lib/probprog/inference/
    ep-knobs.ts      NEW  one `epRunKnobs` Space (tol, maxSweeps, damping, + optional schedule, improper, power) with
                          per-method defaults; used by mp.ts, ep-grid.ts, ep-multistart.ts; one `EpKnobs` type
    ep-route.ts      NEW  epRoute(bound) -> { via: 'ep' | 'ep-grid' | 'ep-multistart'?, verdict, method }: the single place that
                          chooses; replaces atomVerdict/atomRoute/gridMethodOf and the benchmark branch of epCoverage
    mp.ts, ep-grid.ts, ep-multistart.ts       use `prepare`/`compileAt` from one `_compile.ts`; the rule strategy in mp.ts
lib/probprog/ep/                          the public part, ~25 exports, "run EP on this model and judge it"
    run.ts           runEp(bound, knobs, {light}) -> EpRun (what atomRun does without atoms: beliefs per sweep, record, evidence)
    beliefs.ts       Belief, beliefDensity, beliefCredibleInterval (built on dists), momentsOf
    check.ts         checkAgainst(reference, beliefs, tolerance), Tolerance tiers
    calibration.ts   calibrate(problemOfSeed, spec) (today's ep-extended.ts, independent of atoms)
    coverage.ts      epCoverage, vmpCoverage (derived), no hand-kept counts
lib/probprog/ep/atoms/                    the fixtures (or `probprog/ep-atlas/`; the name matters less than the split)
    define.ts        defineAtom: { meta, knobs, build(knobs) -> {definition, data, truth, quantities}, reference, tolerance: tier,
                     recovery?: tier, evidenceTolerance?, present?: presentation }
    <family>.ts      ONE file per family, basic and learned together (conjugate, priors, scales, likelihoods, gates,
                     weights, functions, kernels, covariance, ordinal-multiclass, nonparametric, counts): ~12 files
    index.ts         the registry arrays per family; `epAtoms` is their concatenation; atom constants not exported
lib/probprog/ep/atoms/presentation.ts     equations/figure/mixtures builders shared by atoms (lineFit(), band(), groups()), so a
                                          figure is a call, or leave the vocabulary to render and keep only data here
lib/probprog/validation/references/       the reference kernels, one family of them
    grid1.ts (log-spaced and midpoint, was hyperPosterior + quadratureMarginal + weights gridMarginal)
    gridN.ts (box grids: gridMarginals, posterior3, slopeGrid, skillGrid as callers of one adaptive N-D kernel)
    sampler.ts (Gibbs mixture and clutter), closed-form.ts (use validation/exact.ts: normalNormal, betaBernoulli, gammaPoisson)
lib/probprog/ep/roadmap.ts                gaps and directions as data; `refuses` derived from verdict `missing` tags; no counts in prose
docs/probprog/ep.md                       the design doc (about 250 lines)
docs/probprog/ep-status.md                GENERATED by `make ep-report` from epCoverage/epGaps/vmpCoverage/epAtoms (tables)
docs/probprog/ep-changelog.md             ep-plan.md sections 7-18 moved verbatim, then frozen
docs/probprog/notes/                      the ep-notes-*.md and ep-and-gam.md, each with a dated banner
apps/pocs/src/pocs/ep-<program>/          the 9 programs stay hand-made until they are atoms with a sampler reference
apps/pocs/src/registry.ts                 + synthetic entries and one `AtomRoute` for `ep-<atom>` built from epAtoms
```
Alternative for the folder name if the user prefers one top-level import: keep `probprog/ep` for the public part and move atoms
and references to `probprog/ep-atlas` and `probprog/validation/references`. The point that matters: the user-facing surface is
`inference` + `ep/run`, the fixtures are not published as exports of the capability.

Public surface after: `epSpecMethod`, `epGridMethod`, `epMultiStartMethod`, `epRoute`, `runEp`, `checkAgainst`, `calibrate`,
`epCoverage`, `vmpCoverage`, `beliefCredibleInterval`, `recoverySummary`, `mixtureInterval`, `activeSetScore`, `epAtoms`, `epAtom`,
`epGaps`, `epDirections` and their types (about 30). Everything else gets a `_` prefix or leaves the barrel. The agent index then
marks atoms `@index table` (one entry for the registry, one per family file), not 92 `estimator` lines.

On "its own domain": `ep` should not be a new top-level domain. It composes `model`, `inference`, `validation` and `message-passing`
and is imported by none of them; a folder under `probprog` is right. What is wrong is that the capability and its test suite share
one name.

## 4. Stepwise, behaviour-preserving plan

Common proof of "no change" (step 0, once): a script (not committed, or a committed golden test) that for each of the 92 atoms at
default knobs prints: quantities, reference marginals (mean, sd, `pdf` at 7 fixed points, evidence), `atomRun` beliefs at the final sweep
(rounded 1e-9), sweeps, `converged`, check rows, the verdict, plus `epCoverage()` and `vmpCoverage()` rows and the
`atomCalibration` for 3 atoms, and for the pocs the title/summary/tags of each entry. The digest before and after each step must match;
this runs in about 2 minutes. Everything below proves itself against it, plus `make verify`, the barrels, layers and the api surface
(`make api-surface` diff reviewed by hand).

| # | Step | What moves, merges, is generated | Proof | Size / risk | Parallel-safe with |
| --- | --- | --- | --- | --- | --- |
| 0 | Safety net | the snapshot script above | n/a | S / none | all |
| 1 | Helpers | `lg`, `sum`, `once`, `groupIndex`, normal cdf and pdf into one `_util.ts` (or `blas.ts`, `dists`); keep the A&S `phi` as a named fast variant | snapshot identical (bit identical when `phi` stays) | S / low | 2, 3, 8 |
| 2 | One route | `inference/ep-route.ts`; `atomVerdict`, `atomRun`, `epCoverage` call it; delete `gridMethodOf`, `atomRoute`; verdict computed once | snapshot (verdict strings, `via`) | S / low | 1, 3, 8 |
| 3 | One knob space | `inference/ep-knobs.ts`; the three methods import it (per-method defaults kept: 200, 300); `atom.ep` typed `Partial<EpKnobs>`; unknown keys rejected in `methodSpace`/`session`; pocs page reads `epSpecMethod(bound).config`, so `power` is visible | snapshot; one new negative test (`dampng`) | M / medium (a behaviour change: typo now throws, check the 14 overrides and the pocs) | 1, 2, 8 |
| 4 | Consolidate references | build `validation/references/` from `ep-atoms-quad.ts`, `-covariance-ref.ts`, `-gibbs.ts`; adopt `validation/exact.ts` in the 3 conjugate atoms; migrate one family per commit (conjugate, scales, priors, likelihoods, gates, covariance, ordinal/multiclass, weights, kernels) | snapshot of reference marginals to 1e-12 where code only moves; 1e-6 and a note where a kernel is swapped | L / medium | the families are parallel with each other, not with 5 and 6 |
| 5 | Atom record | split `AtomProblem` into build + reference + presentation; `family` becomes a union; tolerance tiers (`exact`, `quadrature`, `moment-matched`, `loopy`, `sampler`) with an optional measured override and reason; `evidenceTolerance` moves from `ep-evidence.test.ts` into the atom; drop `expected`/`plan` or keep only as an escape hatch (all 92 `runs`) | snapshot identical except the removed fields; pocs smoke tests | M / low (mechanical codemod over 22 files, so do it as one sweep with no other atom change in flight) | none: touches every atom file |
| 6 | Re-layout | one file per family, twins beside their basic atom, arrays per family, `epAtoms` concatenated (`make barrels`); `_` prefix for internals; `extended` renamed `learned` everywhere; `ep/` public part separated (`run.ts`, `calibration.ts`, `coverage.ts`); VMP coverage kept in `coverage.ts` | snapshot; api-surface diff shows only removals of internal names and renames; `atomSource` still finds each `// region <atom>` (the pocs test asserts it) | M / low-medium (large diff, small logic) | after 5 only |
| 7 | Derived coverage and tests | `refuses` derived from verdict `missing` tags with a gap id; test asserts `atomsRun === epAtoms.length`, `benchmarksRun === benchmarks.length` and a committed `ep-coverage.snapshot.json` rather than floors; gap prose loses counts; closed gaps removed or marked | tests fail when a rule regresses (mutate one rule to prove) | S-M / low | 5, 6 |
| 8 | Docs | split `ep-plan.md`: `ep.md` (design, how to add a rule, a mapping, an atom, a reference), `ep-changelog.md` (sections 7-18 verbatim, dated), `notes/`; `make ep-report` generates `ep-status.md` and the status column of `ep-priors.md` (closes the `docs-drift` gap); fix `structure.md:3855-3865`, `ep/index.ts`, `mp.ts:14`; write one proper EP entry in CHANGELOG 0.1.13 and correct lines 226-235 | a doc test that the generated tables are current; link check; the 17 stale statements above each resolved or deleted | M / none | everything (docs only), but write after 6 for final paths |
| 9 | pocs atom pages generated (needs the user's explicit request) | delete 92 `ep-<atom>/` dirs; registry adds entries from `epAtoms`; one `AtomRoute`; subgroup `match` derives from `atom.family` instead of slug globs; group summary corrected | page list identical (title, summary, tags, group) before and after; `ep-atoms.test.ts` stays; open 5 pages in a browser | M / low | 6 done |
| 10 | Programs into the library (needs the user's request) | the programs' recovery check goes to `ep/calibration`; their Gibbs/NUTS references to `validation/references/sampler.ts`; `changepoint-extended/calibration.ts` and `Calibration.tsx` replaced by `atomCalibration`/`ExtendedParts`; each program becomes an atom with `reference: {kind:'sampler'}` or a thin file | the 5 `ep-programs*.test.ts` unchanged; check numbers on each page equal before and after | L / medium | after 4 and 9 |

Order: 0 first; then 1, 2, 3 and 8 in parallel (different files); 4 by family in parallel with each other after 1; 5 then 6 as
two sequential mechanical sweeps (no other atom edits meanwhile; a freeze of new atoms for the duration is the real cost of
this plan); 7 alongside 5 and 6; 9 and 10 last and only on request. Total: about 3 to 5 agent-days of work if parallelised as
above; steps 1, 2, 7 and 8 alone (about one day) remove the staleness, the duplicate decisions and the misleading claims
without touching an atom.

Priorities if only three things are done: (i) F6/F9 docs and test honesty (step 8 + 7), because everything else is judged against
what the docs and floors say; (ii) the route and knob unification (steps 2, 3), because it is the only place where two
future features (EP grid with power, EP multistart on a gridded model) cannot be combined; (iii) the family reorganisation of the atoms
with tiered tolerances (steps 5, 6), the one that makes the next 50 atoms cheaper.

## 5. Things not done or not verified

- I did not run the full vitest (read-only, and CLAUDE.md forbids it for agents); counts come from importing the registry and
  from grep. I ran `epCoverage()` (16 s) and every atom's `atomRun` once with Node's type stripping.
- Whether `import.meta.glob` lists go stale on file add under the pocs' Vite version was not tested.
- Rules (`message-passing/operators`, 65 listed operators) and `probprog/model/factors.ts` were not reviewed beyond counting.
- Which of the 29 learned atoms duplicate their basic atom's model text, line by line, was sampled (conjugate, likelihoods), not
  measured across all 29.
