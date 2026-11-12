# Expectation propagation in omai: the design

How EP is put together in this library, what each layer promises the next, and how to add a piece. It is written against the code
as it is; what the three design reviews (`review-compiler.md`, `review-rules.md`, `review-suite.md`) propose and is not built yet is
marked **target**. What runs today, the gaps and every count are in [`ep-status.md`](ep-status.md), generated from the code, so none is
repeated here. What was built when is [`ep-changelog.md`](ep-changelog.md); the rule catalogue and the GAM design are
[`ep-plan.md`](ep-plan.md); the prior atlas is [`ep-priors.md`](ep-priors.md); measured side investigations are in `notes/`.

## 1. What EP is here

A model is written once, as a spec (`defineModel`), and an inference method is a choice. EP is the method that needs the most from the
model: a factor graph of exponential-family messages in which every factor has a rule. The library does not do it by hand for each
model. A **compiler** reads the spec and emits the graph or says exactly why it cannot; the **engine** (`message-passing`) runs EP,
variational message passing (VMP) and belief propagation (BP) on a graph as settings of one engine (projection, schedule, damping,
improper policy); an **inference method** wraps the compile-and-run in the library's `defineInferenceMethod` convention (knobs as a
`Space`, a typed tape, a steppable and resumable state, a verdict from `check`); the **suite** is a set of small models with reference
answers (the atoms), the coverage measurement, the gaps as data and the digest that proves a change moved nothing.

A refusal is information. `Verdict { applies, reason, missing, hint }` names the node and the rule that is missing, so a model EP does
not take is a missing rule or a way of writing it that the compiler does not read, never a silent failure.

## 2. The layers

```
spec (ModelDefinition, bind)                    probprog/model           the user's model, data bound
  -> compiler  toFactorGraph / factorGraphVerdict / factorGraphPosterior      probprog/model/factors.ts
  -> factor graph (CompiledGraph)               message-passing          variables, factors, edges, families
  -> engine + rules  epAlgorithm, FactorOperator  message-passing         one sweep = messages along edges
  -> methods  epSpecMethod, epGridMethod, epMultiStartMethod, epRoute    probprog/inference
  -> suite  atoms, epCoverage, epGaps, calibration                      probprog/ep (a testbed, imported by none of the above)
  -> pages  apps/pocs (one page per atom, programs, the coverage page)  apps/pocs
```

Imports only point downward and `make layers` reports a cycle as its path. Each layer, with its files:

**Model** (`probprog/model/`): `defineModel` builds a `ModelDefinition`; `bind` attaches data and sizes and gives a `BoundModel`
(`spec`, `data`, `options`, `customs`). `analyseSpec` reads the spec statically (the node kinds, what is opaque, discrete, continuous),
which is what `check` reads for the non-EP methods too.

**Compiler** (`probprog/model/factors.ts`, one class of about 4,700 lines today): `toFactorGraph(bound, { plates, packScalars })`
returns a `FactorGraphCompilation` (the graph as plain data, the `SiteLayout` that maps graph variables back to the spec's sites,
and `issues`, each with a node and a reason). A regression on a weight vector over a plate is one plated factor, other plates are
expanded, and scalar normal latents that an observed node reads together are packed into one joint Gaussian vector (`packScalars`:
for EP this keeps their correlation; as separate scalars the loop between them was loopy Gaussian BP whose spreads were two to three
times too small). `factorGraphVerdict(compiled, graph, rule)` asks whether every operator of the graph has a rule under `ep`, `vmp` or
`bp`; `factorGraphPosterior` reads a message state back as a `Posterior` keyed by site name, in the spec's shapes. **Target**
(`review-compiler.md` section 5): a folder `probprog/model/factor-graph/` with the pure leaves (types, refusal, cells), a builder, the
passes made explicit (constants, layout plan, node walk, prune, opaque filter), the idioms as pure functions, an ordered list of
mappings each with `applies`/`emit`/`requires`, and structured refusals (`Refusal(code, node, facts)`); the public surface stays.

**Factor graph and engine** (`message-passing/`): a message is the natural parameters of a family (`gaussian`, `gaussian-mv`, `gamma`,
`beta`, `bernoulli`, `discrete`, `dirichlet`, `normal-gamma`, `wishart`, `normal-wishart`). A factor is an operator of the registry
(`operators/registry.ts`, `defaultOperators()`), compiled against by name. `epAlgorithm(graph, options)` is an `Algorithm` run by
`runAlgorithm` (`trace`), which emits typed events (`EpPath`: the change per sweep, the evidence, `message` events, marginals by
variable id); its state is the messages, so a run serialises, resumes, seeks and forks. Options: schedule (serial, flooding, tree,
residual, custom), granularity (sweep, factor, message), damping, the improper policy (`skip`, `damp`, `reset`, `error`), tolerance, sweep
cap and restarts, plus power (fractional EP) per factor. **Target** (`review-rules.md` section 4): folders `core/`, `numerics/`,
`engine/`, `operators/`, `algorithms/`, an error taxonomy (`ImproperMessageError` against `ParameterError`), one integration layer for
the numeric rules, and a `defineOperator` helper for the numeric `ep` rules; the import path `omai-engine/message-passing` stays.

**Methods** (`probprog/inference/`): section 4.

**Suite** (`probprog/ep/`): section 7. It composes `model`, `inference`, `validation` and `message-passing` and is imported by none of
them. **Target** (`review-suite.md` section 3): separate the public part (run EP on this model and judge it) from the fixtures (atoms,
one file per family) and from the references.

## 3. The contracts

- **Spec to compiler.** A spec is data plus `custom` functions (`customs`); the compiler is given the bound model and must either
  return a graph or `issues`. It never throws for a model it does not take; a throw is a bug (the LDA audit error was one).
- **Compiler to engine.** `FactorSpec { operator, args, params, power?, start? }` names operators by string; `compileGraph(spec,
  registry)` validates arity, port families and each operator's `check`. **Target**: the names become a `const` tuple so a typo fails
  `tsc`, and a test says every operator of the registry is claimed by a mapping or listed as hand-built only.
- **Rules.** A `FactorOperator` has `ports` (with the families each accepts), `support` per mode (`exact`, `bound`, `numeric`), the
  message rules `ep`, `vmp`, `bp` (or `tilted`, the numeric fallback the engine projects), `logEvidence` and `expectedLog` (what the
  evidence and the variational bound need), `check` (can this factor be built), `noRule` and `unsupported` (why a rule is absent, in
  words, for the verdict), `power` (fractional EP), `defines` (an output port that is a deterministic function of the others),
  `readsTarget` and `needsDensity` (how it reads a flat cavity). One policy decides what happens at an improper first message.
- **Methods.** `InferenceMethod<State, Config, Path, Analysis>`: `config` (a `Space`), `check(analysis) -> Verdict`,
  `posterior(state, config) -> Posterior`, a typed `contract`, `make(config)` returning the `Algorithm`. A refusal at `init` carries the
  same reason as `check`, so a page that greys a method out and a run that starts anyway say the same.
- **Atoms.** An atom is a model, its data, the quantities it asks about, a reference, a tolerance with its reason and, for twins, a
  recovery promise. `atomRun` returns `ran` (beliefs per sweep, a message record, the evidence), `refused` (the verdict) or `failed` (the
  error: a bug to report).

## 4. The methods and how one is chosen

| method | what it does | when |
| --- | --- | --- |
| `epSpecMethod(bound)` (`ep`) | plain EP on the compiled graph; a step is a sweep, a factor or a message by `granularity`; also fractional EP by `power` | the model compiles and every operator has an EP rule |
| `epGridMethod(bound)` (`ep-grid`) | EP inside a grid over scalar hyperparameters that a custom function reads (a Gaussian process length scale): each node conditions the model, EP runs, the EP evidence weights the node | plain EP refuses because a custom function reads a random scalar |
| `epMultiStartMethod(bound)` (`ep-multistart`) | EP from several starts of the discrete messages (random, one excluded category each, relabellings), the distinct fixed points weighted by their evidence | a model with discrete latents whose posterior has several readings; chosen by the person |
| `vmpSpecMethod`, `bpSpecMethod` | the variational and the sum-product settings of the same engine | likewise |

`epRoute(bound)` (`inference/ep-route.ts`) is the one place that chooses between plain EP and EP in a grid: it returns `{ via, verdict,
method }` with `via` one of `ep`, `ep-grid` and `none` (refused; the verdict is plain EP's). `atomRun`, `atomVerdict` and `epCoverage` call
it, so the suite, the coverage and the code a page shows agree on what ran. The multi-start method is not on the route: it answers a
different question and is the user's call. No other method chooses for the user.

A method that runs several EP runs (a grid node, a start) shares the compile step with plain EP through `prepare` (`_support.ts`) and
the run knobs below.

## 5. The knobs

Declared once, in `inference/ep-knobs.ts`: `epRunKnobs(method)` is the `Space` of the run knobs, `EpKnobs` the typed record, `checkKnobKeys`
rejects a key the space does not have, `atPower` puts the power of fractional EP on the factors that take one.

| knob | meaning | ep | ep-grid | ep-multistart |
| --- | --- | --- | --- | --- |
| `schedule` | order of a sweep | serial | serial | serial |
| `granularity` | how much of a sweep a step does | sweep | - | - |
| `damping` | weight of the old message | 0 | 0 | 0 |
| `improper` | repair of an update that makes a belief improper | damp | damp | damp |
| `tol` | stop when no belief moves more than this | 1e-8 | 1e-9 | 1e-9 |
| `maxSweeps` | the sweep cap (per run for grid and multi-start) | 200 | 200 | 300 |
| `restarts` | restarts with stronger damping when a run stalls | 1 | 1 | 1 |
| `power` | fractional EP (1 is EP) | 1 | 1 | 1 |

The method-specific knobs (`points`, `tail`, `maxNodes`, `edge` for the grid; `starts`, `jitter`, `exclusions`, `labels`, `merge` for
multi-start) sit beside these. `granularity` is absent from the grid and multi-start methods because their step is a whole EP run.
`atom.ep` is an `EpKnobs`: a key that is not a knob is a type error where the atom is written, an `OmaiError` in `defineAtom` and in
`atomRun`: a misspelled knob used to run on the defaults and report success.

## 6. Refusals, gaps and directions

`epGaps` (`ep/ep-gaps.ts`) is data: each gap has a kind (missing rule, missing mapping, quality, infrastructure), a status, a size, a
plan and **markers**, the phrases its refusals contain. Which items a gap refuses is derived, never kept: `epGapsRefusing(rows)` reads the
reasons and missing tags of the coverage's refused items against the markers, and `unclaimedRefusals` is what no gap claims (the test
holds it empty). Today nothing in the suite is refused; the gaps are shortfalls of quality and reach. `epDirections` is the wider map of
kinds of problem the suite could cover. Neither carries counts or dates; `ep-status.md` does, generated.

## 7. The suite

**Atoms** (`probprog/ep/_atoms/<family>.ts`, registry `ep-atom-list.ts`: `epAtoms`, `epAtom(name)`, `epAtomsOf(family)`). A bare setup of one
factor or family, one file per family (`conjugate`, `priors`, `scales`, `likelihoods`, `ordinal`, `gates`, `weights`, `functions`, `kernels`,
`covariance`, `nonparametric`, `counts`), the basic atoms and their learned twins together; each file ends with its array and the atoms are
not exported one by one. A **basic** atom has known constants; its **learned** twin (`extends`, a name ending `-learned`) makes them unknown
with priors and simulates the data from a stated truth, and promises a recovery (`recovery`: over its seeds the 90% intervals hold the truth
at the stated rate). An atom is declared with `defineAtom` (`_define.ts`) and has three parts, returned by `parts(knobs)`: **build** (the
model, its data and the quantities), **reference** (`exact` closed forms, a `grid` option, the joint of a pair, `referenceSource`) and
**present** (the equations, the figure, the pair and mixtures a page draws); `atom.problem(knobs)` composes them, lazily. The reference
kind is `exact` (closed form), `grid` (the model as a program on a grid, `validation/gridPosterior`) or `truth` (recovery only). The
tolerance is a named rung (`_tolerance.ts`: `tolerance('close', reason)`, `exact` through `lumpiest`, or `measured` with its own numbers
where an atom sits near the edge of every rung) and the recovery one of three (`recovery('standard', reason)`); `evidence` says how close
EP's log evidence is to the grid's where the grid is fast. `family` is the union `AtomFamily`. The records are in `atom-types.ts`, the
functions in `ep-atoms.ts` (`atomReference`, `atomCheck`, `verifyAtom`, `atomVerdict`, `atomRun`) and **`run.ts`**: `runEp(bound, {quantities,
ep, light})` is the generic "run EP on a bound model through `epRoute`, with beliefs per sweep and a record", which `atomRun` calls and a
program page can call on its own model.

**References** are independent of the rule under test: a closed form, a quadrature or a sampler, never EP. They are one family in
`probprog/validation/references/`: `marginal.ts` (the shape `ExactMarginal`; mixtures, `weightedMarginal` with three density readings,
`gaussianMixtureMarginal`, `probabilityMarginal`, draws), `grid.ts` (`quadratureMarginal`, `hyperPosterior`, `gridMarginals`, `logWeights`),
`grid3.ts` (`posterior3`, three coordinates by passes), `conjugate.ts` (gamma, inverse gamma, Student t, beta, the normal-inverse-gamma
posterior, and `conjugateMarginal`, which reads the library's `exact.ts` table), `gate.ts` (the exact two-component switch and spike and
slab), `clutter.ts` (the clutter problem by quadrature) and `sampler.ts` (the Gibbs sampler of a mixture), next to `gridPosterior`,
`mixtureOfRegressions` and `exact.ts`. The conjugate atoms keep their hand-written closed forms and a test checks them against the library's
table (taking the table's turns the check's KS distance 0 into 1e-16, which the bitwise digest sees).

**Coverage** (`ep-coverage.ts`): `epCoverage()` runs every atom and asks the route of every benchmark model; `vmpCoverage()` does the
variational twin. It measures, it does not remember: the tests hold `atomsRun === epAtoms.length` and the like, and the committed digest
is the regression net for the numbers. **Calibration and recovery** (`ep-extended.ts`, `ep-recovery.ts`): `atomCalibration`,
`atomRecovery`, `beliefCredibleInterval`, `recoverySummary`, which work on any problem that carries a truth.

**The pocs** (`apps/pocs`) show the suite: one page per atom (`EpAtomPage`), the programs (`ep-program-*`), the coverage page. A page that
shows a model ends with the code that ran (`atomSource` takes the `// region <atom>` of the atom's file and the `// region inference`
lines of `runEp` (`run.ts`)).

## 8. The proof that a change moved nothing

`make ep-digest-check ARGS="--exact"` (about 2.5 minutes, niced) records, for every atom with its variational twin, every benchmark model
and every EP program, the compiled graphs (hashes), verdicts, beliefs, evidence and check rows, plus the operator registry and the knob
spaces, and compares them to the committed `ep-digest.json`. A refactoring step must report at least EXACT (graphs and counts identical,
numbers to 12 digits; BITS when nothing at all changed). A deliberate change gets `--accept` in a commit of its own, with the reason, so
the baseline's history is the list of intended changes (`docs/structure.md` section 10). `make ep-report` writes `ep-status.md` and the
`status` column of the atlas `ep-priors.md` (each row's `ran by` column names the atom or test that runs it, or says `not yet`;
`scripts/ep-atlas.ts`); `make ep-report-check` fails when `ep-status.md` is stale, when a status word or a reference of the atlas has
drifted from the code, or when the atlas names an atom in prose that does not run; `make ep-check` runs it after the tests. Neither is part of `make verify`.

## 9. How to add a piece

**A message rule for an operator.** Search `message-passing/operators/` first (a numeric likelihood of a linear predictor is a row of a
registry, not an operator). Write the operator with `ports`, `support`, the rule (`ep`, or `tilted` for a numeric one), `logEvidence`
and `check`; add it to the folder's list so `defaultOperators()` has it; test it against a brute-force tilted moment
(`testing/message-passing.ts`) at 1e-9; `unsupported`/`noRule` for what it cannot do. Under VMP give `expectedLog` or say why not.

**A compiler mapping.** In `probprog/model/factors.ts`, in the method for the node kind (latent or observed), add the recognition
and the emission, the `layout` call that maps the graph variable back to the site (forgetting it silently drops the site from the
posterior), and a refusal with a reason where the idiom is close but not it. Add a model test (`model/*-factors.test.ts`). Take care
that order inside the if-chains is semantics. **Target**: add an entry to the ordered mapping list with its `requires`.

**An atom.** Put it in the file of its family (`_atoms/<family>.ts`, next to its twin); wrap the model and data in a `// region <atom-name>` so the page
shows the code that ran; `defineAtom` with `name`, `title`, `what`, `factors`, `family`, `reference`, `tolerance` (with the reason),
`knobs` and `parts` (the build, the reference and the presentation); `ep:` only for knobs of EP the atom needs (`damping`, `tol`, `power`); a learned twin adds `extends` and
`recovery` and carries the `truth` on every quantity. Add it to the array at the end of the family file (the registry concatenates them). A page needs two files in
`apps/pocs/src/pocs/ep-<atom>/` (the owner's call: pocs are added on request). Then `make barrels agent-index api-surface`, the atom's
test through the registry (`ep-atoms.test.ts` runs every atom), and the digest: a new atom adds its item (`--accept` in its own commit).

**A reference.** A closed form from `validation/exact.ts` (`normalNormal`, `betaBernoulli`, `gammaPoisson`, `dirichletCategorical`)
where it exists; else a quadrature in `validation/references` (`quadratureMarginal`, `gridMarginals` with `marginal(k)`, `weightedMarginal`, `logWeights`);
independent of the model's own EP rule. Say in `referenceSource` what it is when it is not a closed form.

**A knob.** In `epRunKnobs` (`ep-knobs.ts`) once, with the per-method default; read it through `epRunSettings`; the three methods and
the atom page take it from there.

**A gap.** In `ep-gaps.ts`: id, kind, status, summary and plan without counts or dates, and `markers` from the real refusal text.

## 10. Where things are

| what | where |
| --- | --- |
| the rules, the engine | `packages/engine/src/lib/message-passing/` (operators in `operators/`) |
| the compiler | `packages/engine/src/lib/probprog/model/factors.ts` |
| the methods, route, knobs | `packages/engine/src/lib/probprog/inference/{mp,ep-grid,ep-multistart,ep-route,ep-knobs}.ts` |
| the suite | `packages/engine/src/lib/probprog/ep/` |
| the digest and the report | `scripts/ep-digest*.ts`, `scripts/ep-report.ts`; `docs/probprog/ep-digest.json`, `ep-status.md` |
| the pages | `apps/pocs/src/pocs/ep-*`, `apps/pocs/src/shared/EpAtomPage.tsx` |
| the reviews with the target structure | `docs/probprog/review-{compiler,rules,suite}.md` |
