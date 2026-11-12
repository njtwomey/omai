# Expectation propagation in omai: the design

How EP is put together in this library, what each layer promises the next, and how to add a piece. It is written against the code
as it is. The three design reviews (`review-compiler.md`, `review-rules.md`, `review-suite.md`) are dated reports of 2026-11-11: their
target structures were built (sections 2 and 7 say where each part is), and what they list as a finding is not necessarily still true.
What runs today, the gaps with their plans and every count are in [`ep-status.md`](ep-status.md), generated from the code, so no count
is repeated here. What was built when is [`ep-changelog.md`](ep-changelog.md); the rule catalogue and the GAM design are
[`ep-plan.md`](ep-plan.md) and [`ep-and-gam.md`](ep-and-gam.md); the prior atlas is [`ep-priors.md`](ep-priors.md); measured side
investigations are in `notes/`; what the compiler maps is [`compiler-mappings.md`](compiler-mappings.md) (generated).

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
  -> compiler  toFactorGraph / factorGraphVerdict / factorGraphPosterior      probprog/model/factors.ts, model/factor-graph/
  -> factor graph (CompiledGraph)               message-passing          variables, factors, edges, families
  -> engine + rules  epAlgorithm, FactorOperator  message-passing         one sweep = messages along edges
  -> methods  epSpecMethod, epGridMethod, epMultiStartMethod, epRoute    probprog/inference
  -> suite  atoms, epCoverage, epGaps, calibration                      probprog/ep (a testbed, imported by none of the above)
  -> pages  apps/pocs (one page per atom, programs, the coverage page), apps/showcase (stories)
```

Imports only point downward and `make layers` reports a cycle as its path. Each layer, with its files:

**Model** (`probprog/model/`): `defineModel` builds a `ModelDefinition`; `bind` attaches data and sizes and gives a `BoundModel`
(`spec`, `data`, `options`, `customs`). `analyseSpec` reads the spec statically (the node kinds, what is opaque, discrete, continuous),
which is what `check` reads for the non-EP methods too. When the evaluator (`model/compile.ts`) gives a discrete latent all its states
(the enumeration handler), the axis of states is a dimension named `enum:<site>` that broadcasts like any other, so an enumerated
latent can be used by a plated observation.

**Compiler** (`probprog/model/factors.ts` is the driver, about 250 lines; the rest is `probprog/model/factor-graph/`):
`toFactorGraph(bound, { plates, packScalars, conjugatePairs })` returns a `FactorGraphCompilation`: the graph as plain data, the
`SiteLayout` that maps graph variables back to the spec's sites, and `issues`, each with a node, a refusal code and a reason. A
compilation runs in passes: the events and scans are unrolled; the constants are found (`_known.ts`); the layout plan is made
(`_plan.ts`: which scalars are packed into vectors, which vectors are tied groups, where one joint site is used); the nodes are walked in
spec order through ordered mapping tables (`_table.ts`: node kinds, latents, observed nodes as a whole, observed rows, observed elements);
dead `apply` factors are pruned; conjugate pairs inside gates are fused when asked; and the opaque nodes are reported. In a table the
first entry whose pattern the node is maps it, so the order is part of the meaning and each entry says why it sits where it does. An
entry that can only tell by trying answers `NOT_MINE` and the graph builder is restored to its checkpoint (`_mapping.ts`), so a failed
recognition leaves nothing in the graph. The mappings are one file per concern in `_mappings/`, over the value algebra of `_cells.ts`
(affine forms over the graph's random scalars), the evaluator of expressions `_symbolic/`, and the recognisers of idioms
`_idioms.ts`. A refusal is a `Refusal` with a code from the closed set `RefusalCode` and a sentence from the catalogue `_reasons.ts`;
the kind of an issue is derived from the code. The compiler names operators by `OperatorName` (`message-passing/operators/names.ts`), so a
misspelt name fails `tsc`, `addFactor` checks each factor with `compileGraph`, and `operator-contract.test.ts` checks that every
operator of the registry is claimed by a mapping or is built by hand only. [`compiler-mappings.md`](compiler-mappings.md) lists every
entry of every table, generated from them (`make ep-compiler-doc`).

What the compiler does with the shapes of a model, in the order a reader meets them. A regression on a weight vector over a plate is one
plated factor, and other plates are expanded. Normal latents that an observed node reads together are packed into one joint Gaussian
vector (`packScalars`: for EP this keeps their correlation; as separate scalars the loop between them was loopy Gaussian BP whose
spreads were two to three times too small). Scalars with known priors have the prior on the vector; a vector over one dimension or a
latent with a prior of its own that the observation reads linearly (the county intercepts of a hierarchy beside the floor effect, the
skills of a tournament that each game reads at two places) is a tied group: made as it is, then tied to its coordinate by an exact
constraint, with a vague starting message on the vector (`findGroups`, `tieToGroup`). Where an observation's scores are a matrix times
one such vector, the site is one joint message on the vector instead of one per score: a categorical's class scores (the class slopes of
up to six classes, the coefficients that the alternatives of a choice share) are one `multiclass-row` per observation, and the two log
shapes of a beta outcome one stacked vector with a `likelihood-pair-row` per outcome (`numerics/_vector-site.ts`), so the correlation
between the scores reaches the vector and the spreads are the exact ones where one message per score gave 0.4 to 0.85 of them.
A latent variable model with random Fourier features (`gplvmFourierModel`: one latent coordinate per point, outputs that are weighted
sums of `cos(omega . x + b)`) is read by the fifth detection of the layout plan (`findGplvm`, `_gplvm.ts`) as one `gplvm-row` per point
on the point's coordinate and the weights of each output: the weights are integrated out in closed form at each node of a zoomed grid
over the coordinate (`numerics/_node-grid.ts`), so the graph is a star around the weights and the loop through the features of a point
is gone. It never converges to `tol` (the change per sweep falls slowly, the coordinates stop moving much earlier) and about one run in
twelve lands in a poor solution because a draw of the random features cannot follow the function; a model with two coordinates per point,
more than 1024 weights or a form written another way compiles by the scalar operators (gap `gplvm-features`; the entry of
`agent-aec1c3-gplvmrow` in the changelog has the measurements). With `conjugatePairs` (on for `ep` and `ep-multistart`, off for VMP and the
grid) the pass `fuseConjugatePairs` (`_fuse.ts`) rewrites a mixture component's mean and precision that form a normal-gamma pair into one
`normal-gamma` variable (section 4).

`factorGraphVerdict(compiled, graph, rule)` asks whether every operator of the graph has a rule under `ep`, `vmp` or `bp`; for EP it
also refuses a graph with the units of a neural-network layer (`networkLayerShape`: an `apply` of an affine map of weights that several
rows share, multiplied by a weight that several rows share). The weights of such a layer are not identified (sign and permutation
symmetries), so one Gaussian per weight has no fixed point that means anything. Measured against a long NUTS run, one sigmoid unit (a
dose-response curve) converges to the sampler's means; one `tanh` unit converges to one of its two modes, and one `cos` or rectifier unit
does not converge. The verdict therefore refuses two or more distinct hidden units and one unit of any activation but the sigmoid, with
a reason that names the cause (gap `network-layer-modes`); the benchmark network is the one refused item. `factorGraphPosterior` reads a message state back as a `Posterior`
keyed by site name, in the spec's shapes.

**Factor graph and engine** (`message-passing/`, import path `omai-engine/message-passing`). A message is the natural parameters of a
family (`gaussian`, `gaussian-mv`, `gamma`, `beta`, `bernoulli`, `discrete`, `dirichlet`, `normal-gamma`, `wishart`, `normal-wishart`).
The folders are `core/` (families, the graph, the operator contract and `defineOperator`, the errors), `numerics/` (the quadratures the
rules share), `engine/` (arguments and cavities, dispatch to an operator, the repair of improper messages, evidence, the verdict, the
restarts, the schedules and the `Algorithm`), `operators/` (the registry and one folder per kind: `gaussian/`, `scales/`, `likelihoods/`,
`links/`, `covariance/`, `discrete/`, `lifts/` for rows and plates, `kernel/` for the variational numeric rules) and `algorithms/` (the
named settings `epAlgorithm`, `vmpAlgorithm`, `bpAlgorithm`, power and alpha EP, stochastic EP, expectation consistent approximation,
regions, approximate message passing, the clutter problem). A factor is an operator of the registry (`operators/registry.ts`,
`defaultOperators()`), compiled against by name; each folder lists its operators in `list.ts` and `operators/names.ts` is the typed
tuple of the names in registry order. `epAlgorithm(graph, options)` is an `Algorithm` run by `runAlgorithm` (`trace`), which emits typed
events (`EpPath`: the change per sweep, the evidence, `message` events, marginals by variable id); its state is the messages, so a run
serialises, resumes, seeks and forks. Options: schedule (serial, flooding, tree, residual, custom), granularity (sweep, factor,
message), damping, the improper policy (`skip`, `damp`, `reset`, `error`), tolerance, sweep cap, restarts and a warm-up, plus power
(fractional EP) per factor.

A parameter that is wrong is a `ParameterError`; a message that is not a proper belief is an `ImproperMessageError`, which the engine
repairs under the improper policy and counts. Only the second is a numerical event. The engine owns what an operator would otherwise
repeat: a numeric rule that returns tilted moments (`tiltedRule`) or a site (`siteRule`) is projected onto the family, the cavity is
subtracted and the result divided by the power in one place (`projectSite`), and a flat Gaussian cavity is shown to an operator that
needs a density as a very wide one (`needsDensity`). The belief of a variable is accumulated in one array where the family's product is
the plain sum of natural parameters (`Family.additive`; not the discrete families, which renormalise), the same additions in the same
order as before, which matters for a variable that has a message from each of hundreds of rows.

**Restarts.** A run that stalls is restarted from the start messages with a stronger damping: the k-th restart is the k-th of 0.5, 0.8
and 0.9 among those stronger than the run's own damping (the last of them for any further restart), and a restart that would repeat the
previous attempt is not made. A run damped by 0.9 or more has none, one damped by 0.5 has two (0.8 and 0.9), and one damped by less than
a half has three. The default is one restart (`restarts`). Four of the EP programs, damped at 0.5 or more and stalled on improper
cavities, now restart and report what the restart reaches, where before they reported convergence.

**Numerics** (`numerics/`). The numeric rules share one integration layer: a panelled grid over one to three axes (`_grid.ts`), the
one-dimensional tilted quadrature with kinks and a zoom (`_tilted.ts`), the window finder for gamma and beta tilts (`_windows.ts`:
`findWindow`, which throws an `ImproperMessageError` when no window holds the mass, and `tiltGrid`), the Bartlett rule of the Wishart
(`_bartlett.ts`, its gamma rules kept in a bounded cache), the node grid of `gplvm-row` (`_node-grid.ts`) and the joint site on a vector
(`_vector-site.ts`, `_product.ts`). Log weights are summed with the largest one removed (`shiftedSum`), variances come from centred
moments (`centredMoments`, and `TiltedMoments.centred` for a Gaussian target whose mean is large against its spread), and a variance that
is not positive and finite is reported as an `ImproperMessageError` instead of being floored (`resolvedVariance`). A tilt that failed is
kept under its key and throws again, so a failing grid pass runs once per sweep. The tilt keys of the covariance groups round to 1e-13 of
the largest argument and include the power and the kinds of the arguments. What the numerics step of the rules review (`review-rules-numerics.md`)
did not change, and why, is listed in that file. The kernels are plain loops over number arrays by decision: `ep-perf-tensor-study.md`
found a tensor version of the grid integrator 3.5 to 4 times slower, the tensor library then got fast paths for broadcast, reductions and
gather and the same study gives 1.15 to 1.5 times, bit for bit, but no speed gain at any size, so converting the kernels stays parked as a
separate decision (`CLAUDE.md`).

**Evidence.** There is one evidence system: each operator's `logEvidence` and `evidenceOf` in `engine/evidence.ts`; the posterior and the
tape give the EP energy of the run (`mp` `evidenceMethod`: `ep-energy`). The log evidence is exact where EP is. Where a cavity is not a
density by itself (the site of a scale mixture, or of a mixture, has a negative precision in some direction) the evidence is taken in
natural parameters where the integral exists: the affine operators and the Gaussian joint (`linear`, `plus`, `difference`,
`times-constant`, `sum`, `gaussian-joint`) and the gate do so, and are finite whenever the joint precision is positive definite. The
energy of an alpha-divergence run is the power EP energy at power alpha for the factors that take a power. A factor whose integral
against its cavities diverges and has no such form has no evidence, and the posterior names it in `diagnostics.evidenceUndefinedAt`
(`EvidenceDefect`: `improper cavity`, `no evidence rule`, `zero mass`); the gap `evidence` lists what is left. The plated regressions
(`probit-plate`, `logistic-plate`) have an evidence only against a proper cavity: where the cavity of the plate is indefinite the plate is
not run there and the evidence is that of the model with a factor per row (`evidenceMethod` `ep-energy (rows expanded)`, kept per state
and per run configuration, `epRunKey`). The engine has an operator trait `improperTarget`, which shows an operator a cavity with no
density but not flat, that the plates set for a short time and no operator sets now: with it the learned logistic regression diverged on
one of the forty seeds of its recovery test and the spreads of the other seeds moved about 4% narrower, so the plates were returned to the rows-expanded
evidence (changelog, `agent-a72b8a-gatefix`). The trait and its test stay for a plate that wants it once that divergence is understood.

**Methods** (`probprog/inference/`): section 4.

**Suite** (`probprog/ep/`): section 7. It composes `model`, `inference`, `validation` and `message-passing` and is imported by none of
them. What `review-suite.md` section 3 asked for is built as follows: the knobs are declared once (`inference/ep-knobs.ts`), the choice of
method is one function (`epRoute`), the atoms are one file per family and are reached through the registry only, the references are one
family in `validation/references/`, and `runEp` is the public "run EP on this model" call. The folder keeps its name, and the suite's
exports are still published with the capability's.

## 3. The contracts

- **Spec to compiler.** A spec is data plus `custom` functions (`customs`); the compiler is given the bound model and must either
  return a graph or `issues`. It never throws for a model it does not take; a throw is a bug (the LDA audit error was one).
- **Compiler to engine.** `FactorSpec { operator, args, params, power?, start? }` names operators by string, because a hand-built graph
  writes them so; the compiler's own factors are typed by `OperatorName`, so a typo fails `tsc`. `compileGraph(spec, registry)` validates
  arity, port families, the parameters against the operator's `params` and each operator's `check`. A test says every operator of the
  registry is claimed by a mapping or is listed as hand-built only (`operator-contract.test.ts`).
- **Rules.** A `FactorOperator` is built with `defineOperator`. It has `ports` (with the families each accepts), `params` (what it reads,
  parsed before anything else), `support` per mode (`exact`, `bound`, `numeric`, derived from the kinds of the rules), the message rules
  `ep`, `vmp`, `bp` (or `tilted`, the numeric fallback the engine projects; a site rule gives the tilt and the engine does the rest),
  `logEvidence` and `expectedLog` (what the evidence and the variational bound need), `check` (can this factor be built), `noRule` and
  `unsupported` (why a rule is absent, in words, for the verdict), `power` (fractional EP), `defines` (an output port that is a
  deterministic function of the others), and the traits `readsTarget` and `needsDensity` (how it reads a flat cavity) and
  `improperTarget` (see the evidence, section 2). One policy decides what happens at an improper first message. Section 1 of
  `review-rules.md` and the trim of the contract in the doc comment of `FactorOperator` say why each member stays.
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
| `epGridMethod(bound)` (`ep-grid`) | EP inside a grid over the hyperparameters that a custom function reads (a Gaussian process length scale, kernel variance and noise sd; the length scale per input of an ARD kernel): each node conditions the model, EP runs, the EP evidence weights the node; over several axes a walk over the region where the posterior has mass | plain EP refuses because a custom function reads a random scalar (or, for an ARD kernel, a plated vector) |
| `epMultiStartMethod(bound)` (`ep-multistart`) | EP from several starts of the discrete messages (random, one excluded category each, relabellings; set on the port of each label's prior and held for three damped sweeps), the distinct fixed points weighted by their evidence (`diagnostics.modeWeights`) | a model with discrete latents whose posterior has several readings; chosen by the person |
| `vmpSpecMethod`, `bpSpecMethod` | the variational and the sum-product settings of the same engine | likewise |

`epRoute(bound)` (`inference/ep-route.ts`) is the one place that chooses between plain EP and EP in a grid: it returns `{ via, verdict,
method }` with `via` one of `ep`, `ep-grid` and `none` (refused; the verdict is plain EP's). `atomRun`, `atomVerdict` and `epCoverage` call
it, so the suite, the coverage and the code a page shows agree on what ran. The multi-start method is not on the route: it answers a
different question and is the user's call. No other method chooses for the user.

**Reporting a multimodal posterior.** One EP run commits to one reading and reports the spread of that reading alone. `epMultiStartMethod`
is the route that reports the others: the posterior is the mixture of the distinct fixed points (runs whose site means agree within
`merge` of the larger sd are one), each weighted by its EP evidence, which for modes that are apart is their share of the mass, and
the evidence of the model is the log sum. It has been held to references where the mode weights are known: the clutter problem's
bimodal location against its quadrature, a two-cluster mean with unequal priors against the enumeration of the assignments, and the
mixture of the benchmarks with symmetric priors (`ep-multistart-modes.test.ts`; the numbers are in the gap `multimodal` and the changelog).
The weights are only as good as the fixed points found: modes that overlap are one fixed point, a regime that few steps visit is not
a relabelling of the reading EP finds, and a symmetry of continuous variables (the sign of a loading) has no discrete message to start from.
The `labels` knob chooses how the labelled components are read. `keep` (the default) is the posterior of the model as written: with
unequal priors it gives the exact moments of each labelled mean (a two-cluster mean against the enumeration: the means within 0.02 sd and the
sds within 1% of the exact ones). `sort` reads the components in order of the first vector of parameters (the low, the middle and the
high regime: what a data-ordered sampler reports, and what a page that names regimes shows). `permute` is `keep` with a start from every
relabelling of each reading found.
Relabelling moves only the component axes of a site (`componentAxesOf` in `_relabel.ts`: the last dimension of the first operand of a
`pick`, the first dimension of a table that an `at` looks up, the category axis of what a categorical reads); a model with a discrete
latent and no lookup the code can read leaves every axis of length `k` as it is, and a model without any keeps the old reading.

Measured on the changepoint program (8 seeds, 3 rates each, 12 starts, the data-ordered Gibbs sampler of the page as the reference; the truth
inside the 90% interval, the spread over the sampler's, mean and median): `keep` 0.958 and 3.2 (median 2.2), `permute` 0.958 and 3.6
(3.2), `sort` 0.917 and 0.72 (0.92), one run 0.778 over 24 seeds earlier (the spread 0.69 of the sampler's, median 0.94); time 75, 57 and 60 s for the
eight seeds (24 starts: 335, 760 and 337 s for 24 seeds earlier, so `permute` costs up to twice as much when the relabellings
are many). `keep` and `permute` are wide because they are the posterior of the labelled model (the labels carry the priors' names and
the posterior of "regime 0" is a mixture over the labellings the priors allow); `sort` is as wide as the sampler where every regime is
visited. A labelling with 2% of the mass (a two-cluster mean with a strong asymmetric prior, exact weight 0.0197) was missed by 12
random starts under `keep` and found, with its exact weight, under `permute`: the relabelling is what makes the weights of the
labellings reliable, the random starts only reach them by chance. Recommendation: keep `keep` as the default (its answer is the exact marginal
of the model as written); use `sort` for a model whose
components are exchangeable up to their parameters and whose page shows "the low, the middle, the high one", and `permute` when the weight
of the less likely labellings matters. Changing the default to `sort` is the owner's choice; it would change what a site named `rate[0]`
means in every multi-start result.

**The symmetric fixed point of a mixture.** A mixture with a label per row (`mu.at(z)` with `z` a latent discrete variable: the compiler
makes one `select` factor per row) has a fixed point where every component is the same: the labels carry no information, the means sit at
the mean of the data, the spread is the spread of the whole data. It exists by symmetry, whatever the data, and it is a clean fixed point
(every update computable, a finite evidence). The clusters' own reading is a different state: the ambiguous rows between two clusters send
a message to the value of the row with a negative precision (the moment-matched mixture of two separate normals is wider than the
likelihood), so their cavities have no density, those updates cannot be computed and are dropped by the improper policy (about a third of
the rows, every sweep), and the evidence is undefined there. From flat messages and no damping, the first sweep commits every row to a side
at its first updates (the shared spread, in the standardised data's units, is 0.2 after the second sweep where the clusters' is 0.55) and the run then drifts into the
symmetric point; on the Gaussian mixture example's defaults (150 rows, two components 3 apart, equal weights) it does so for data seeds
1 and 4 of 8 and for free means on 2 of 3 (the other seeds reach the clusters undamped, with their dropped updates). The separated reading is a fixed point of the damped dynamics and a stable one of the
undamped ones (started from the damped state and let go at damping 0 it stays), so it is the start that decides. What breaks the
symmetry is the first sweeps: `warmup` (a knob of `ep` and `ep-multistart`, an engine setting `EngineSettings.warmup`) damps the first
three sweeps by at least a half and then runs at the run's own damping. `auto`, the default, does it where the graph has a `select` with a
latent index (`hasExchangeableLabels`) and does nothing elsewhere; `on` always, `off` never (the old behaviour). The warm-up is not a
guarantee: it is a start that lets the two components form before the rows commit, and a mixture whose clusters are not apart stays
one fixed point. Because the separated reading has no evidence and the symmetric one has, comparing readings by evidence prefers the
wrong one: `ep-multistart` therefore does not weigh readings when its first run (plain EP, which now reaches the clusters) has no
evidence (`diagnostics.unweighed`): the posterior is the readings that have none, equally weighted. Random and exclusion starts of a
mixture's labels reach the symmetric point (random labels average out to uninformative ones), so they are the readings that have an
evidence and are set aside. A real weighing of the readings of a mixture needs the evidence at the separated fixed point, which needs
an EP rule for the observation of a row that accepts a cavity of negative precision (tried in natural parameters for `gaussian-scale`:
the run diverges, 1e135 in 200 sweeps; the gap `multimodal`).

**Conjugate pairs in a mixture.** A component with a mean and a precision of its own, the mean as wide as its precision allows (`mu ~ N(m, 1 / (kappa tau))`, `tau ~ Gamma`) and its rows the branches of a gate on the label, is the normal-gamma pair, and the compiler (`conjugatePairs`, on for `ep` and `ep-multistart`: `factor-graph/_fuse.ts`) makes it one `normal-gamma` variable with a `gaussian-normal-gamma` branch per row: the messages are closed form instead of a quadrature over the precision (two diagonal groups of 272 rows: 4 sweeps and 0.3 s instead of 10 to 32 sweeps and 7 to 26 s that depended on the order of the rows). Distinct prior centres (`start`) break the symmetry of the labels there, as the warm-up does for a `select`. A precision shared by the components, a centre with another factor on it, or a pair with an independent prior is not fused and runs the quadrature. The gate also takes a vector constant, so a mixture of Gaussian vectors with `mvn-normal-wishart` branches runs when hand-built; the model language and compiler for it are the gap `mixture-vector-components`.

**The grid in more detail.** A method over several axes opens on `order: 'walk'`: a best-first flood fill from the middle of the range
over the lattice neighbours of the nodes within `window` nats of the best, so the cost follows the volume of the posterior and not the
product of the axes. Once some node has an evidence, a node that failed is expanded only within two lattice steps of a node near the best
(`_ep-grid-walk.ts`); while none has, a failed node is expanded first. With `warm`, a node starts from the messages of the nearest settled
node, and keeps its own only when it converged with a finite evidence (`_ep-grid-warm.ts`). The hyperparameter ranges are cached per
`tail`. A node whose EP run has no evidence (an improper cavity of a plated or tied factor) gets no weight; the posterior counts these in
`diagnostics.nodesWithoutEvidence` with a `note` and its status is `completed`, not `converged`, while there are any.

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
| `restarts` | restarts with stronger damping (0.5, 0.8, 0.9, among those stronger than the run's own) when a run stalls | 1 | 1 | 1 |
| `warmup` | the first three sweeps damped by at least a half: `auto` (where labels select components), `on`, `off` | auto | - | auto |
| `power` | fractional EP (1 is EP) | 1 | 1 | 1 |

The method-specific knobs (`points`, `tail`, `maxNodes`, `edge`, `order` (`bisect`, or `walk` from the middle of the range outwards, best first, over any number of axes), `window` (the nats at which a walk stops) and `warm` (each node started from the nearest settled node's messages) for the grid (a method over several axes opens on `walk`, `warm`, 15 points, a window of 7 and a budget of 4000 nodes: `epGridKnobsFor(axes)`); `starts`, `jitter`, `exclusions`, `labels`, `merge` for
multi-start) sit beside these. `granularity` is absent from the grid and multi-start methods because their step is a whole EP run.
`atom.ep` is an `EpKnobs`: a key that is not a knob is a type error where the atom is written, an `OmaiError` in `defineAtom` and in
`atomRun`: a misspelled knob used to run on the defaults and report success.

## 6. Refusals, gaps and directions

`epGaps` (`ep/ep-gaps.ts`) is data: each gap has a kind (missing rule, missing mapping, quality, infrastructure), a status, a size, a
plan and **markers**, the phrases its refusals contain. Which items a gap refuses is derived, never kept: `epGapsRefusing(rows)` reads the
reasons and missing tags of the coverage's refused items against the markers, and `unclaimedRefusals` is what no gap claims (the test
holds it empty). One benchmark model is refused today, the Bayesian neural network, and the gap `network-layer-modes` claims it. The
other gaps are shortfalls of quality and reach. The text of each gap in `ep-gaps.ts` (what is done, what remains, the plan) is the
authority; `ep-status.md` prints the table. The open gaps, by id:

| gap | kind | what is open |
| --- | --- | --- |
| `grid-hyperparameters` | missing mapping | more than about four axes on the grid, priors that are not independent, a likelihood that is not Gaussian at several axes |
| `matrix-larger` | missing rule | a non-conjugate Wishart matrix of dimension 5 and above, an LKJ prior of order 3 and above, a normal-Wishart variable inside a larger model |
| `correlation-strong-few-pairs` | quality | a bivariate correlation near 1 from few pairs has its spread right and its skew wrong |
| `softmax-many-classes` | missing rule | a joint message for the softmax of more than six classes whose cavity is correlated; a Gaussian process per class as a plate of vectors |
| `mixture-vector-components` | missing mapping | a model-language form for a mixture of Gaussian vectors with a covariance matrix per group; the cost of a mixture over many rows |
| `multimodal` | quality | one Gaussian per variable shows one mode; what the multi-start does not reach (overlapping modes, symmetries of continuous variables, an undefined evidence) |
| `network-layer-modes` | quality | hidden units with symmetric weight modes are refused; a fixed point per mode weighted by evidence is needed |
| `loopy-overconfidence` | quality | loops make spreads too small where the vector cannot be packed (more than six classes, a beta whose shape is a gamma variable, other pair likelihoods, non-centred hierarchies) |
| `positive-scale-belief` | quality | the belief of a positive scale is a Gaussian, so its range runs below zero |
| `posterior-sample-correlation` | quality | posterior samples draw each scalar site alone, so correlations between scalar sites are lost |
| `gplvm-features` | quality | a GPLVM by random Fourier features: a draw of features can land in a poor solution, no run settles, two coordinates are left to the scalar route |
| `sv-parameters` | quality | stochastic volatility with an unknown persistence: one EP run is too narrow, the grid is the route |
| `evidence` | infrastructure | no evidence where a factor's integral against its cavities diverges and has no natural-parameter form; the tied-group fallback |
| `variational-rules` | infrastructure | what variational message passing cannot run (hard comparisons, a Weibull with random shape, ordinal outcomes between random cut points, custom functions) and where the mean field is poor |

`epDirections` is the wider map of kinds of problem the suite could cover. Neither carries counts or dates; `ep-status.md` does,
generated.

## 7. The suite

**Atoms** (`probprog/ep/_atoms/<family>.ts`, registry `ep-atom-list.ts`: `epAtoms`, `epAtom(name)`, `epAtomsOf(family)`). A bare setup of one
factor or family, one file per family (`conjugate`, `priors`, `scales`, `likelihoods`, `ordinal`, `gates`, `weights`, `functions`, `kernels`,
`covariance`, `nonparametric`, `counts`; and the composed atoms that put several families in one model: `composed`, `composed-chain`,
`composed-gates`, `composed-pairs`), the basic atoms and their learned twins together; each file ends with its array and the atoms are
not exported one by one. A **basic** atom has known constants; its **learned** twin (`extends`, a name ending `-learned`) makes them unknown
with priors and simulates the data from a stated truth, and promises a recovery (`recovery`: over its seeds the 90% intervals hold the truth
at the stated rate). An atom is declared with `defineAtom` (`_define.ts`) and has three parts, returned by `parts(knobs)`: **build** (the
model, its data and the quantities), **reference** (`exact` closed forms, a `grid` option, the joint of a pair, `referenceSource`) and
**present** (the equations, the figure, the pair and mixtures a page draws); `atom.problem(knobs)` composes them, lazily. The reference
kind is `exact` (closed form), `grid` (the model as a program on a grid, `validation/gridPosterior`) or `truth` (recovery only). The
tolerance is a named rung (`_tolerance.ts`: `tolerance('close', reason)`, `exact` through `lumpiest`, or `measured` with its own numbers
where an atom sits near the edge of every rung) and the recovery one of three (`recovery('standard', reason)`); `evidence` says how close
EP's log evidence is to the grid's where the grid is fast. `family` is the union `AtomFamily` (the file of an atom and its family are not always the same word: the ordinal atoms are in the family
`likelihoods`). The records are in `atom-types.ts`, the
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

**The pocs** (`apps/pocs`, `make pocs`) show the suite. The directory `src/pocs/expectation-propagation/` holds `basic`, `priors` and
`learned` (one page per atom of that section, `EpAtomPage`), `composed-atoms` (the pages of the composed atoms), `programs` (the EP
programs, `ep-program-*`) and `ep-coverage` (the coverage page); `src/pocs/bayesian-examples/` holds the canonical Bayesian examples, one
method at a time. A page that shows a model ends with the code that ran (`atomSource` takes the `// region <atom>` of the atom's file
and the `// region inference` lines of `runEp` (`run.ts`)). Pages are added on request only. The showcase (`apps/showcase`,
`make showcase`) is separate and imports nothing from the pocs: stories in `src/stories/<slug>/` where nothing runs until the reader
presses play and the computation runs in a worker. Several of them run EP (for example `how-expectation-propagation-works`,
`signal-in-the-clutter`, `who-is-better`, `finding-hidden-coordinates` and `gaussian-mixtures`).

## 8. The proof that a change moved nothing

`make ep-digest-check ARGS="--exact"` (about 2.5 minutes, niced) records, for every atom with its variational twin, every benchmark model
and every EP program, the compiled graphs (hashes), verdicts, beliefs, evidence and check rows, plus the operator registry and the knob
spaces, and compares them to the committed `ep-digest.json`. A refactoring step must report at least EXACT (graphs and counts identical,
numbers to 12 digits; BITS when nothing at all changed). A deliberate change gets `--accept` in a commit of its own, with the reason, so
the baseline's history is the list of intended changes (`docs/structure.md` section 10). `ARGS="--only <names or prefixes>"` limits the run
to the items a change can reach (10 to 20 seconds) and `--all` prints every move instead of the first 25. `make ep-report` writes `ep-status.md` and the
`status` column of the atlas `ep-priors.md` (each row's `ran by` column names the atom or test that runs it, or says `not yet`;
`scripts/ep-atlas.ts`); `make ep-report-check` fails when `ep-status.md` is stale, when a status word or a reference of the atlas has
drifted from the code, or when the atlas names an atom in prose that does not run; `make ep-check` runs it after the tests. Neither is part of `make verify`.

### The performance baseline

`make ep-perf` (about 10 minutes, niced, on a quiet machine) measures where the EP code spends its time and writes `ep-perf.json` (the
baseline: medians, spreads, a calibration figure, the machine, the date and commit) and `ep-perf.md` (the tables): the shared grid
integrator per profile (`pair`, `axes`, `triple`, `line`) and the kernels under it, one message call per operator and typical call
signature (recorded from real EP runs and replayed under a timer, grouped by folder), the EP wall time, sweeps, updates and updates per
second of about 26 atoms, models and programs, and the share of each run spent in each operator (the registry's rules are timed from the
harness, never edited). `make ep-perf-check` re-measures and fails only when something is slower than the baseline by more than 1.5x
after dividing by the machine factor (the ratio of the calibration workload, a fixed pure-arithmetic loop timed before and after the run,
so a baseline taken on a busier or another machine still compares); `ARGS="--only item/atom/gaussian,grid/"` limits it and `--ratio 2`
loosens it. Neither is part of `make verify` or `make check`. The tensorisation question they were built for is answered in
`ep-perf-tensor-study.md`: the grid kernels stay plain loops; the tensor library got fast paths (broadcast, reductions, gather) and a
tensor version of the integrator is now 1.15 to 1.5 times the loop, bit for bit, with no speed gain.

## 9. How to add a piece

**A message rule for an operator.** Search `message-passing/operators/` first (a numeric likelihood of a linear predictor is a row of a
registry, not an operator). Write the operator with `defineOperator`: `ports`, `params`, the rule (`messageRule`, or `siteRule` or
`tiltedRule` for a numeric one, so the engine does the projection), `logEvidence` and `check`; add it to its folder's `list.ts` and its
name to `operators/names.ts` (the registry refuses a name without an operator and an operator without a name); test it against a
brute-force tilted moment (`testing/message-passing.ts`) at 1e-9; `unsupported`/`noRule` for what it cannot do. Under VMP give
`expectedLog` or say why not. A new operator changes the registry line of the digest.

**A compiler mapping.** In `probprog/model/factor-graph/_mappings/`, in the file of its concern, write the mapping: a name, a line
saying what it maps, the operators it `requires`, `matches` and `map`. Add it to the table of its kind in `_table.ts` at the place its
order needs and say there why it sits there (the order is part of the meaning); the `layout` call maps the graph variable back to the
site (forgetting it silently drops the site from the posterior). Where the idiom is close but not it, throw a reason from the catalogue
`_reasons.ts` (a new reason is a function there, with a code). Add a model test in `factor-graph/` (`observed-*.test.ts`, `latent-*.test.ts`,
`gates-*.test.ts`), then `make ep-compiler-doc` so `compiler-mappings.md` lists the entry. The graph corpus (`graph-corpus.snap.txt`)
shows what a change did to the graphs of the atoms and models.

**An atom.** Put it in the file of its family (`_atoms/<family>.ts`, next to its twin); wrap the model and data in a `// region <atom-name>` so the page
shows the code that ran; `defineAtom` with `name`, `title`, `what`, `factors`, `family`, `reference`, `tolerance` (with the reason),
`knobs` and `parts` (the build, the reference and the presentation); `ep:` only for knobs of EP the atom needs (`damping`, `tol`, `power`); a learned twin adds `extends` and
`recovery` and carries the `truth` on every quantity. Add it to the array at the end of the family file (the registry concatenates them). A page needs two files in
`apps/pocs/src/pocs/expectation-propagation/<section>/ep-<atom>/` (`basic`, `priors` or `learned`: the directory is the section; the owner's call: pocs are added on request). Then `make barrels agent-index api-surface`, the atom's
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
| the compiler | `packages/engine/src/lib/probprog/model/factors.ts` (the driver), `model/factor-graph/` (tables, mappings, passes, verdict); `docs/probprog/compiler-mappings.md` |
| the methods, route, knobs | `packages/engine/src/lib/probprog/inference/{mp,ep-grid,ep-multistart,ep-route,ep-knobs}.ts` |
| the suite | `packages/engine/src/lib/probprog/ep/` |
| the digest and the report | `scripts/ep-digest*.ts`, `scripts/ep-report.ts`; `docs/probprog/ep-digest.json`, `ep-status.md` |
| the performance baseline | `scripts/ep-perf*.ts`, `scripts/perf/`; `docs/probprog/ep-perf.json`, `ep-perf.md`, `ep-perf-tensor-study.md` |
| the pages | `apps/pocs/src/pocs/expectation-propagation/{basic,priors,learned,composed-atoms,programs,ep-coverage}`, `apps/pocs/src/shared/EpAtomPage.tsx`; the stories in `apps/showcase/src/stories/` |
| the design reviews (2026-11-11, built) | `docs/probprog/review-{compiler,rules,suite}.md`, `review-rules-numerics.md` |
