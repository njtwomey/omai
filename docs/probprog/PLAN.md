# Probabilistic programming in omai: a plan for 0.1.13 and 0.1.14

Status: proposal for the user to read and decide on. Written 2026-11-10 against `main` at `e420394f` (0.1.12 is the
structured-prediction release, finishing). Nothing here is code. The plan is the main document; four files sit beside it:

| file                           | what it holds                                                                                          |
| ------------------------------ | ------------------------------------------------------------------------------------------------------ |
| [inventory.md](inventory.md)   | what exists today, file by file, with an honest assessment, two timings, and the causes of "primitive" |
| [survey.md](survey.md)         | the field: 24 entries, what each does, what to adopt and what to leave, what runs in a browser         |
| [algorithms.md](algorithms.md) | the target repertoire: every algorithm with its needs, knobs, tape, proof, size and priority           |
| this file                      | summary, architecture, validation, roadmap, risks and the open questions                               |

Sections 2 to 4 are short forms of the three files above. Sections 5 to 8 are the plan.

## 1. Executive summary

**What the system will be.** One model, written once, run by any engine that can run it, with the user in control of how.
- A **model layer** (`probprog/model/`): a typed builder in the shape of Stan's blocks and PyMC's named dimensions that produces a
  `ModelSpec`, a JSON-able graph of data, latent, observed, derived and factor nodes with named dimensions (plates), constraints
  and a small registry of operators. A spec compiles to (a) the existing `sample`/`observe` program, so every present engine and
  handler works at once, (b) a factor graph of exponential-family variables for message passing, (c) a static structure
  analysis, (d) a drawing, (e) JSON for workers and docs pages. Existing closure programs keep working and are first-class.
- A **handler layer** extended with the missing Pyro/NumPyro handlers (`reparam`, `collapse` (delayed sampling), `lift`, `scope`,
  `escape`, `equalize`) and Gen-style trace operations (`regenerate`, `update`, `assess`) so users can write their own samplers,
  guides and proposals in a few lines.
- A **unified inference interface**: every method (exact, message passing, variational, MCMC, SMC, nested, likelihood-free) is an
  `InferenceMethod` with a `Space` of knobs, a typed tape contract, a capability check that says why it does or does not apply,
  and a steppable, resumable, traced `Algorithm`; the twelve present engines are wrapped without rewriting them.
- **Composition** (`hybrid`): a plan object, edited by the user or built by the planner, that assigns blocks of sites to methods
  (NUTS for these, an exact conditional for those, EP for this part, collapse the conjugate rest), in the manner of NIMBLE, Turing.jl
  and Bean Machine. Automatic collapse of conjugate and linear-Gaussian structure is the structural leverage.
- A **message-passing engine** (`lib/message-passing/`), the Infer.NET core: exponential-family messages in natural parameters,
  a registry of factor operators, schedules, and slots for the projection (moment matching, KL, alpha-divergence, quadrature,
  Laplace), the region graph (Bethe, Kikuchi), power and damping, stochastic EP. EP, VMP, BP, power EP, alpha-BP, SEP, fractional EP
  and the mixed forms are settings of one engine, not separate code.
- A **validation kernel** that proves engines work: grid and closed-form oracles, simulation-based calibration, Geweke tests, an
  agreement harness across engines, named benchmark models with truth, and NumPyro/BlackJAX/ArviZ/pgmpy/scikit-learn goldens
  where those exist.

**What changes and what does not.** The present probprog is not primitive as a NumPyro port: 12 engines, handlers, auto guides, a
planner, criticism, golden tests against NumPyro. What a user meets as primitive is eleven specific things (inventory section 8):
a model is a closure, not a value; everything is a string name and a raw tensor; data binding is ad hoc; the user picks an engine
and cannot compose or edit inference; only MCMC has a `Space`; structure is used only when the whole program matches; there is no
message passing on programs; ten kernels are unreachable; the MCMC tape is one step; calibration is untested; and nothing shows any
of it. The plan addresses each; it adds layers on top and **replaces nothing** (the old API stays and the new layer compiles to it).

**Recommended order** (details in section 7).
1. Freeze three contracts (spec, method, factor operator) and build the **first vertical slice** (section 7.2): a probit regression
   written in the new builder, run by NUTS (through the compiled program) and by EP (through the new engine), compared with the exact
   grid answer, traced, and shown in a proof of concept page. It proves the architecture end to end in about two weeks of one lane.
2. In parallel, in three lanes with disjoint files: (A) the model layer, then the unified interface and `hybrid`; (B) the
   message-passing engine and its variations; (C) the handlers, then the validation kernel and the first benchmark models.
3. Release **0.1.13** is the architecture and the new capability: model layer, handlers, interface and composition, message
   passing with its variations, the validation kernel and six benchmark models.
4. Release **0.1.14** is breadth and structure: every kernel on programs, SVI extensions (TraceGraph, TraceEnum, structured
   guides, natural gradients, Pathfinder), parallel tempering and the newer samplers, particle MCMC and nested sampling, automatic
   structure exploitation in the planner, diagnostic charts, the full benchmark suite and the SBC campaign.

The user suggested SVI in the first release. I put it second because SVI already works (12 engines, `svi`, five guides, four ELBOs):
what 0.1.13 needs from it is only the contract and `Space` wrapper (inside the interface work). The new SVI algorithms are in
0.1.14. That is a decision for the user (open question 5).

## 2. Inventory in brief

Full text: [inventory.md](inventory.md).

| area                | what is there                                                                                                                                                                                                                           | assessment                                                                                            |
| ------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ----------------------------------------------------------------------------------------------------- |
| programs, handlers  | `sample`, `observe`, `param`, `deterministic`, `factor`, `plate`, `markov`; `trace`, `seed`, `condition`, `substitute`, `replay`, `block`, `scale`, `mask`, `doIntervention`, `enumerate`, `uncondition`, `handle`; `Messenger`, `send` | NumPyro-faithful and golden-tested; missing reparam, collapse, lift, scope, escape, equalize          |
| densities, analysis | `logJoint`, `potentialEnergy` with Jacobians, init strategies; `analyse` from one traced run                                                                                                                                            | strong bridge to samplers; analysis is per-run and not a model value                                  |
| engines (12)        | enumeration, conjugate, gaussian, chain, nuts/hmc, gibbs, smc, importance, laplace, svi, em, abc; ranked by a planner with reasons                                                                                                      | broad; one-shot; whole-program applicability; only MCMC has a `Space` and a typed contract            |
| guides, ELBOs       | autoDelta, autoNormal, MVN, low-rank, Laplace, IAF; Trace, mean-field, Renyi, IWAE                                                                                                                                                      | good NumPyro set; no structured, amortised, TraceGraph, TraceEnum                                     |
| message passing     | `pgm/inference`: discrete BP (four schedules, damping, Bethe), Gaussian BP, discrete EP and Gaussian-latent EP, discrete VMP, mean field, TRW; hand-written TrueSkill EP, GP-classification EP, CAVI Gaussian mixture, variational LDA  | strong in special cases; no general exponential-family engine; `dists/exponential-family.ts` is ready |
| kernels             | `sampling/`: RWM, adaptive, independence, MALA, ULA, SGLD, SVGD, slice, elliptical slice, HMC, NUTS, Riemannian HMC, tempered SMC, AIS; Stan windowed adaptation; R-hat, ESS, MCSE, HDI, BFMI                                           | strong; **only NUTS and HMC are reachable from programs**                                             |
| render, browser     | `Algorithm`/`Method`/`Session`, `defineContract`, worker `producer` by method name with JSON config, `useJob`, message timelines and structure overlays, `import-safe` test                                                             | the machinery to show a message-passing run exists; a model closure cannot cross to a worker          |
| proof, demos        | goldens for density, handlers, ELBO, criticism, em, enumeration, laplace; closed-form checks for the rest. One example (`examples/logistic`), no proof of concept, no diagnostic charts                                                 | no SBC, no Geweke, no agreement harness; nothing shows the system                                     |

Measured (Node, one run): eight-schools NUTS 2 chains x 600 iterations 4.0 s; logistic n = 500, d = 10 NUTS 1 x 400 1.4 s; SVI
0.2 s for 500 to 1,000 steps. Usable in a page for models of this size; the per-evaluation interpretive overhead is the cost to
watch (section 5.8).

## 3. The field in brief

Full text: [survey.md](survey.md) (sources tagged: [I] introspected, [W] web, [M] memory; Pyro, PyMC, TFP, Stan and Infer.NET are not
installed, so those rows are [M] or [W]).

| take from                    | the idea                                                                                                                                 | lands in (plan section) |
| ---------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------- | ----------------------- |
| Stan                         | blocks (data, parameters with constraints, transformed parameters, model, generated quantities); static check; Pathfinder, ADVI, Laplace | 5.1, algorithms C7      |
| PyMC                         | named dimensions and coordinates, swappable data containers, `do`/`observe` model transforms, per-variable step methods                  | 5.1, 5.2, 5.4           |
| Infer.NET                    | operator registry, exponential-family messages, schedule from the graph, ranges and gates, one model with EP, VMP and Gibbs              | 5.5                     |
| Pyro, NumPyro                | handlers (reparam, collapse, lift), guides as programs, TraceGraph and TraceEnum ELBOs, structured autoguides                            | 5.3, algorithms C2, C3  |
| NIMBLE, Turing, Bean Machine | an editable table from nodes to inference methods                                                                                        | 5.4, 5.6                |
| Gen.jl                       | trace operations (`regenerate`, `update`, `assess`) as a toolkit for custom samplers                                                     | 5.3                     |
| Birch                        | delayed sampling: automatic exact marginalisation of conjugate and linear-Gaussian parts inside a running program                        | 5.6, algorithms A5      |
| JAGS                         | automatic sampler selection per node from its full conditional                                                                           | 5.6                     |
| TFP                          | joint distributions as values; automatic structured surrogate posteriors (ASVI)                                                          | 5.1, algorithms C3      |
| BlackJAX                     | kernel contract (already followed), the missing kernels (MCLMC, SGHMC, Barker, ensemble, nested, Pathfinder)                             | algorithms D            |
| WebPPL                       | posteriors as values that feed later models; in-browser demos                                                                            | algorithms A6           |
| Anglican                     | lightweight Metropolis-Hastings by trace regeneration; particle MCMC                                                                     | algorithms D3, E3       |
| Stan SBC, ArviZ              | simulation-based calibration, rank and ECDF plots, power-scaling sensitivity                                                             | section 6               |

Not adopted: Stan's text language and C++ pipeline, universal recursion (WebPPL, Anglican, Church), code generation (Infer.NET),
funsor, TensorFlow graph semantics, GPU back ends.

## 4. The target repertoire in brief

Full tables: [algorithms.md](algorithms.md). About 60 rows in six families, grouped into **seven machines** so the work is
slots, not forty files.

| machine                   | rows     | already there                                           | new in this plan                                                                                             |
| ------------------------- | -------- | ------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------ |
| exact                     | A1 to A8 | enumeration, conjugate (top level), Gaussian, chain     | conjugacy through plates and linear maps, collapse (delayed sampling), analytic pushforward, grid oracle     |
| message passing           | B1 to B9 | discrete BP/EP/VMP/TRW, Gaussian BP, Gaussian-latent EP | one exponential-family engine with EP, VMP, power EP, alpha-BP, SEP, GBP, EC, AMP as settings                |
| ELBO machine (SVI)        | C1 to C9 | `svi`, 6 guides, 4 ELBOs                                | TraceGraph, TraceEnum, structured guides, amortised guides, natural gradient SVI, Pathfinder, nested Laplace |
| kernel driver on a target | D1 to D8 | NUTS, HMC on programs; 10 kernels in `sampling/`        | `kernel:` slot, blocks, HMC-in-Gibbs, parallel tempering, MCLMC, ensemble, SGHMC, reparam                    |
| particle machine          | E1 to E5 | bootstrap PF, tempered SMC, AIS                         | guided proposals, particle MCMC, nested sampling, evidence estimators                                        |
| likelihood-free           | F1 to F4 | ABC, ABC-SMC                                            | synthetic likelihood, neural SBI (stretch), SBC                                                              |
| composition               | D4, A5   | -                                                       | `hybrid`: blocks and mixing patterns                                                                         |

## 5. Core components and architecture

New code goes where `docs/structure.md` says; this section proposes the additions to that document (to be written in the
same change as the first slice).

```
lib/probprog/
  model/            NEW   spec, builder, operators registry, data binding, compilers, serialisation, spec analysis, drawing
  inference/        NEW   the InferenceMethod contract, capabilities, registry, adapters for the 12 engines, hybrid, plan
  validation/       NEW   grid and exact oracles, SBC, Geweke, agreement harness, distances, benchmark runner
  models/           NEW   the named benchmark models, each with its truth (as specs)
  handlers.ts, reparam.ts, collapse.ts, lift.ts   extended handler vocabulary
  engines/          existing; each engine gains a contract and Space; nuts.ts gains the kernel slot
lib/message-passing/ NEW DOMAIN
  messages/  operators/  engine.ts  projections.ts  schedules.ts  regions.ts  energy.ts
lib/sampling/       + MCLMC, SGHMC, Barker, ensemble, parallel tempering, nested sampling (kernels, as today)
lib/exposition/     + records for comparisons, chains, calibration, message traces
packages/render     + views for them; gallery recipes only for new chart capabilities
```

`message-passing` is a new constructed domain: it imports `dists`, `numerics`, `tensor`, `trace`, `space` and `pgm/factor` (for the
discrete tables), and `probprog` imports it, so no cycle (`make layers`). `pgm` must not import it; the existing `pgm/inference`
code stays, and becomes the reference the new engine is tested against. Name is an open question (question 4).

### 5.1 Model definition

**The options.** Compared against what the brief asks (natural arbitrary graphs, little back-end work, inspectable, runs in a
worker, exploitable structure):

| option                                                                                               | natural in TS                       | inspectable and serialisable | crosses to a worker | structure for engines                  | control flow                         | cost to build           |
| ---------------------------------------------------------------------------------------------------- | ----------------------------------- | ---------------------------- | ------------------- | -------------------------------------- | ------------------------------------ | ----------------------- |
| A. Pyro-like closure with handlers (today)                                                           | yes, but tensor-function arithmetic | no                           | no                  | one traced run                         | anything (but analysed once)         | done                    |
| B. Declarative graph and factor DSL (Infer.NET, BUGS)                                                | verbose                             | yes                          | yes                 | complete and static                    | only ranges and gates                | large                   |
| C. Builder with typed variables, named dims, constraints, derived and generated blocks (PyMC/Stan)   | yes with fluent methods             | yes (the builder makes data) | yes                 | complete and static                    | ranges, scans, gates; opaque escape  | medium                  |
| D. Hybrid: C produces a spec that compiles to A; A stays; A can be lifted to C when structure-stable | yes                                 | yes for specs                | yes for specs       | static for specs, one run for programs | everything, with the structure label | medium, and incremental |

**Recommendation: D.** The user writes a spec with a builder; the spec compiles to the existing program, so the 12 engines, the
handlers, the guides and the criticism work on day one; it compiles to a factor graph so message passing works; it analyses
statically so the planner and `hybrid` can see the structure. Closure programs remain for models whose control flow is not
structure-stable (stochastic recursion, data-dependent shapes), and run on the sampling engines as today. The spec is not a new
language: it is a value that the builder returns.

**Sketch** (names are proposals; question 9):

```ts
import { defineModel, ops } from 'omai-engine/probprog/model'
import { normal, halfNormal, bernoulli } from 'omai-engine/dists'

// eight schools, non-centred, written once
const schools = defineModel('eight-schools', (m) => {
  const J = m.range('school') // a named dimension: a plate, a coordinate and a shape
  const y = m.data('y', { dims: [J] })
  const sigma = m.data('sigma', { dims: [J], support: 'positive' })
  const mu = m.latent('mu', normal(0, 5))
  const tau = m.latent('tau', halfNormal(5)) // constraint (positive) comes from the support
  const z = m.latent('z', normal(0, 1), { dims: [J] })
  const theta = m.derived('theta', mu.add(tau.mul(z))) // recorded, like Stan's transformed parameters
  m.observe('y', normal(theta, sigma), y) // vectorised over J: no explicit plate call
  m.generated('yRep', normal(theta, sigma)) // like generated quantities: forward draws for predictive checks
})

const bound = schools.bind({ y: [28, 8, -3, 7, -1, 1, 18, 12], sigma: [15, 10, 16, 11, 9, 11, 10, 18] })
```

What is in the spec, and how each requirement is met:
- **Nodes** (`SpecNode`): `data` (dims, support, role: response, covariate, index, weights, offset, exposure), `latent` (a
  distribution expression, dims, optional extra constraint), `observed` (distribution expression, bound data, mask), `derived`
  (an expression, recorded or not), `factor` (a log-density term, Stan's `target +=`), `generated` (forward only), `param` (a
  non-random learnable, with support), `scan` (a recurrence over a range, for state-space and sequence models), `gate` (select
  among sub-expressions by a discrete variable: mixtures, Infer.NET's `Switch`), `custom` (an opaque TypeScript function: works
  with sampling engines, not with message passing and not in JSON; the escape hatch).
- **Expressions** (`Expr`): a small closed registry of named operators (`add`, `mul`, `div`, `neg`, `exp`, `log`, `sqrt`, `square`,
  `sum` over a dim, `dot`, `matmul`, `at` (gather by an index data node: the group lookup of hierarchical models), `concat`, `where`,
  `logistic`, `probit`, `softmax`, `clip`, ...). Each operator carries: its tensor evaluation (reusing `lib/tensor`), its shape rule
  over named dims, and, optionally, message operators for message passing (5.5). So one registry serves compilation to programs and
  to factor graphs; a user adds an operator once. `Var` objects expose the operators as fluent methods (`w.dot(x).add(b)`); the
  alternative of parsing strings is deferred (it is how a Stan-like text front end would come later, question 3).
- **Distributions** (`DistExpr`): any family name registered in `dists` with parameters as expressions or constants. Families with
  an `ExponentialFamilyForm` are marked, which is what lets message passing and conjugacy detection see them.
- **Plates are named dimensions.** A variable declares `dims`; broadcasting and the plate nesting follow from names (as xarray and
  PyMC `dims`), not from dynamic dimension allocation. On compile to a program, each distinct dim becomes a `plate`; the batch
  dimension numbers are assigned by the compiler, so users never count dimensions. Posteriors are labelled by dims.
- **Constraints and bijectors.** `latent` takes its constraint from the distribution's support (positive, unit interval, simplex,
  ordered, correlation Cholesky, ...) via the existing `bijectorForSupport`, as in the programs; extra constraints (lower and
  upper bounds as in Stan's `<lower=0, upper=1>`, sum-to-zero, ordered) are options that choose a bijector and its Jacobian.
- **Discrete and continuous mix.** A discrete latent is a node with a finite or countable support; `gate` and `at` consume it. The
  compiler marks finite discrete latents `enumerate` (parallel), so NUTS sums them out, and the factor graph treats them as
  Discrete variables with tables.
- **Hierarchies and reuse.** `m.include(sub, { prefix, bind: { x: someVar } })` instantiates a sub-model with a namespace (this is the
  `scope` handler on the program side); hierarchical priors are just latents whose parameters are other latents; the `at` operator
  does group lookups. Models are values: `schools.condition({...})`, `schools.do({...})`, `schools.marginalise(['z'])` (calls
  `collapse`) return new specs (PyMC's `do`/`observe` transformations).
- **Serialisation.** `spec.toJson()` and `defineModel.fromJson(json)` round-trip every spec without `custom` nodes; `custom` nodes
  serialise as a named reference to a function registered in the receiving context (the same "hook by name" device the producer
  protocol already uses for hooks). This is what lets a worker, a docs page and a saved example hold the same model.
- **Compile targets**: `toProgram(bound)` returns a `Model<D>` for the handler engines; `toFactorGraph(bound)` returns the message-passing
  graph (or a report of which nodes are opaque); `analyseSpec(bound)` returns a `ModelAnalysis` of the same shape as today's
  `ProgramAnalysis` but static and complete; `toStructure(spec)` draws it (variables, factors, data nodes, dims as plates), reusing the
  present `Structure` type and the render package.
- **Lifting a program.** `fromProgram(model, data)` best-effort converts a structure-stable closure to a spec of opaque and registered
  nodes by one traced run plus a re-trace with perturbed data and draws to check stability; programs that fail the check stay
  programs. This makes the 30-odd existing test programs usable by message passing without rewriting them. It is optional and
  risky (section 8); the builder is the main path.

**Why not B alone, or A alone.** B alone loses the existing 12 engines and their goldens and cannot express stochastic control flow;
A alone leaves the six problems that make the system feel primitive (closure, string names, no declared dims, no worker, no static
structure). D costs the builder and the compilers, and keeps both worlds.

### 5.2 Data binding

- **Declaration.** A spec declares its data nodes with dims, support and role. `spec.bind(data, options)` checks shapes against named
  dims (sizes come from the data and must agree across nodes), supports (a binary response must be 0 or 1), finiteness, and returns a
  `BoundModel` (spec plus sizes plus data).
- **Covariates and prediction.** `bound.withData(newCovariates)` rebinds the data nodes for new rows without touching the latents;
  `generated` nodes then give predictive draws. This is PyMC's `Data` container, and Stan's `generated quantities` run against new
  data.
- **Index data.** Integer codes with level labels (`school` ids) bind as `index` role and feed `at`. Labels carry into the posterior's
  coordinates.
- **Missing data.** A `NaN` in an observed node means missing: by default the entry becomes a latent imputed value (a real random
  variable in the model); `{ missing: 'ignore' }` drops the entry from the likelihood by a mask (the existing `mask` handler).
- **Batches.** `{ batch: { dim: 'row', size: 64 } }` subsamples the plate and rescales (the existing plate subsampling and `scale`), so
  SVI and SGLD stream; the dataset is never copied.
- **From the library's data.** `bindDataset(spec, dataset, { x: 'covariate', y: 'response' })` binds a `lib/data` dataset (regression,
  classification, time series) by column roles; the dataset's own `truth` (generating program) stays the reference for scoring.
- **Streaming and sequential Bayes.** `posterior.asPrior(site)` returns the site's marginal as an exponential-family distribution
  (when the engine's posterior is one) so the next batch's model uses it as a prior: the pushforward of A6, WebPPL's posterior as a
  value.

### 5.3 The effect-handler layer, and writing your own

Present handlers stay. Added (file in parentheses; each with `@example`s and goldens or exact laws):

| handler                                                    | semantics                                                                                                                                                                                                                 | oracle or law                                                                                                                                                                                      |
| ---------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `reparam(model, { site: reparameteriser })` (`reparam.ts`) | replaces a site by an auxiliary site plus a deterministic transform: `locScale` (non-centred), `transform` (via a bijector), `unitJacobian`, `circular`, `projectedNormal`, `haar`, `dct`, `neutra` (flow-preconditioned) | density of the transformed model equals the original's marginal (property test); NumPyro `reparam` fixtures; funnel divergences vanish (agreement)                                                 |
| `collapse(model, { sites })` (`collapse.ts`)               | delayed sampling: marginalise conjugate and linear-Gaussian sites lazily as the program runs, adding the marginal likelihood factor, and return a `recover` function that draws the removed sites given the rest          | exact marginal equals closed form (conjugate, Gaussian); posterior of the kept sites equals the full model's (A); NumPyro's `collapse` needs funsor, not installed, so closed forms are the oracle |
| `lift(model, priors)` (`lift.ts`)                          | turn `param` sites into `sample` sites with priors (MLE to full Bayes, Pyro's `lift`)                                                                                                                                     | density equals the program with the prior written in                                                                                                                                               |
| `scope(model, prefix)`                                     | namespaces every site name (composition of sub-models)                                                                                                                                                                    | names round-trip                                                                                                                                                                                   |
| `escape`, `equalize`, `mean`                               | Pyro's: hide sites from enclosing handlers, tie sites to share a draw, replace draws by means (plug-in prediction)                                                                                                        | unit tests                                                                                                                                                                                         |
| `inferHint(site, { method, block })`                       | annotate a site with a preferred method or block (a NIMBLE-like sampler assignment stored in `msg.infer`)                                                                                                                 | planner honours it                                                                                                                                                                                 |

**Trace operations as a toolkit** (Gen.jl): small pure functions over a `ProgramTrace` and a `Selection` of site names:
`regenerate(model, trace, selection, key)` (resample the selected sites from the program given the rest, return the new trace and
the log weight), `update(model, trace, changes)`, `assess(model, values)` (log density of a full assignment), `propose`. They are
compositions of present handlers, and they are what a user needs to write a Metropolis-Hastings kernel in a few lines.

**Three levels of user control**, and how the user writes their own at each:
1. _Configure_: choose a method and set its knobs. `infer(bound, { method: 'nuts', numWarmup: 500 })`.
2. _Compose_: assign blocks of sites to methods (5.4, 5.6): `hybrid(bound, [{ sites: ['z'], use: gibbs() }, { sites: ['mu', 'tau'], use: nuts() }])`.
3. _Author_, with four extension points, each a small interface already shaped like something omai has:
   - **a handler**: a `Messenger` (exists) that rewrites messages;
   - **a guide**: any program with `param` sites (exists), now also built from the model's structure by `structuredGuide(bound)`;
   - **a kernel**: an `MCMCKernel` (`init`, `step(key, state, target)`, the existing BlackJAX-shaped contract) where `target` is a **block
     target** giving `logDensity`, `grad`, `regenerate(sites)`, `blanket(site)` and `conditional(site)`; it then runs anywhere a kernel
     is accepted, including inside `hybrid`;
   - **a factor operator or a projection** for message passing (5.5).

   Example of a user-written kernel (a Metropolis step on one block, with a custom proposal), sketched:

   ```ts
   const myRw = defineKernel({
     name: 'rw-mine',
     init: (target, key) => target.initial(key),
     step: (key, state, target) => {
       const proposal = addGaussian(key, state, 0.2)                 // user code
       const accept = Math.exp(target.logDensity(proposal) - target.logDensity(state))
       return uniformDraw(key) < accept ? proposal : state
     },
   })
   hybrid(bound, [{ sites: ['theta'], use: myRw }, { sites: ['sigma'], use: conjugate() }])
   ```

### 5.4 The unified inference interface

All methods implement one contract. The existing `InferenceEngine` is adapted, not rewritten:

```ts
interface InferenceMethod<K extends Space, P extends string> {
  readonly name: string // 'nuts', 'ep', 'svi', 'gibbs', ...
  readonly family:
    | 'exact'
    | 'message-passing'
    | 'variational'
    | 'mcmc'
    | 'smc'
    | 'nested'
    | 'likelihood-free'
    | 'composite'
  readonly space: K // the knobs: controls, tuning and comparison come for free
  readonly contract: Contract<unknown, P> // the typed tape paths
  check(model: ModelAnalysis): Verdict // applies, quality, cost, reason, and what is missing and what would help
  algorithm(model: BoundModel, config: SpaceValue<K>): Algorithm<State> // steppable, resumable, traced
  posterior(state: State): Posterior // the same Posterior as today, plus dims, provenance and evidence
}
```

- **Steppable and resumable.** One step is the unit a page can show or cancel: one iteration across chains (MCMC; today's whole-run step
  is replaced), one sweep or one message batch (message passing), one gradient step (SVI), one temperature or one filtering step
  (SMC), one live-point replacement (nested sampling). State is JSON plus tensors, so `checkpoint`, `seek` and `fork` of the trace
  library work. Methods built this way get `.method` and `.session` and run through the producer in a worker by name.
- **Contracts and Spaces.** Every method declares its Space (`dict` of knobs, `union` with a mode where there are modes) and a
  `defineContract` of paths (algorithms.md). SVI, Gibbs, SMC, EM, Laplace and importance gain these (they are the engines without).
- **Diagnostics are part of the posterior.** `posterior.diagnostics` has, by form: chains (R-hat, bulk and tail ESS, divergences, BFMI,
  tree-depth hits); variational (ELBO curve, the PSIS k-hat of the importance weights, as Yao et al. 2018 [M]); message passing
  (converged, last change, free energy, improper-message count, schedule); particle (ESS history, evidence); nested (logZ error).
- **Capabilities and automatic choice.** `check` returns structured reasons and the features missing. The model analysis is a set of
  features: `continuous latents`, `differentiable`, `finite discrete latents` (with joint size and treewidth), `unbounded discrete`,
  `conjugate sites`, `linear-Gaussian sub-network`, `exponential-family factor graph`, `opaque nodes`, `markov structure`, `has
  likelihood` (versus simulator only). Each method lists the features it requires and provides. `plan` then (1) ranks whole-model
  methods as today, (2) if none is exact, looks for a **decomposition**: collapse the conjugate and Gaussian parts, enumerate the
  low-treewidth discrete part, hand the rest to the best gradient or non-gradient kernel, (3) respects a budget by falling back to
  Laplace, SVI or EP, and (4) returns a `Plan` whose `explain()` says, per block, the method and why, and which methods were rejected
  because of which missing feature. `infer(bound)` runs the plan; the plan is a value the user can print, edit and rerun.
- **Fallback with explanation** keeps today's behaviour (the next engine if one fails at run time) and records every attempt.

### 5.5 Message passing, the Infer.NET core (`lib/message-passing/`)

The aim of the brief: many variations of EP, VMP and friends, cheaply. They are slots in one engine.

**Representation.**
- A **factor graph** of typed variables (each with a message family: Gaussian, Gamma, Beta, Dirichlet, Discrete, Bernoulli, Poisson,
  Wishart, truncated Gaussian, or `Opaque`) and factors (distribution factors, deterministic operators, observations). Plates are
  leading batch dimensions on messages, so a plated model of 10,000 rows is vectorised tensor arithmetic, and `Replicate` and `SumOver`
  operators move between plated and unplated variables. Gates handle mixtures and switches.
- A **message** is an exponential-family distribution in natural parameters (`ExponentialFamilyForm.natural`, existing), with
  division (cavity), power (fractional EP), product (belief), and conversion to moments (`meanParameters`, existing). Discrete
  messages are log tables from `pgm/factor` (division and power exist).
- A **factor operator** is a registry entry keyed by the factor's name, giving per algorithm and per argument a message function,
  plus a log-evidence contribution:

  ```ts
  interface FactorOperator {
    readonly name: string                                   // 'gaussian-plus', 'probit', 'gamma-precision-normal', ...
    readonly ports: readonly Port[]                         // each port has an allowed message family
    ep?(c: EpContext): Message                              // message to port c.to, from the cavity of the others and the factor
    vmp?(c: VmpContext): Message                            // expected-log-factor message
    bp?(c: BpContext): Message                              // discrete sum-product
    logEvidence?(c: EvidenceContext): number
    tilted?(c: EpContext): { moments: readonly Value[]; logZ: number }   // when the projection can be generic
  }
  ```

  A factor with only `tilted` (its moments under cavity times factor) gets EP from a generic projection; a factor with nothing but a
  density gets EP from numerical tilted moments (quadrature, Laplace, importance) through the same slot (algorithms B6). That is
  the answer to "hundreds of operators": a small registry of closed-form operators (about 35 in the first release, listed below), and
  a generic numeric fallback for the rest.

**Slots of the engine** (each an option table with a `Space`, so the variants are configuration):

| slot               | options                                                                                                                                         | gives                                                         |
| ------------------ | ----------------------------------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------- |
| `projection`       | `moment` (EP), `klqp` (VMP), `alpha(a)` (alpha-divergence; alpha = 1 EP, near 0 VMP-like), `laplace`, `quadrature(n)`, `sample(n)`, `unscented` | EP, VMP, alpha-BP, EP with numerical sites                    |
| `schedule`         | `flooding`, `sequential`, `residual`, `tree` (exact on trees), `custom order`                                                                   | convergence behaviour; the existing BP schedules              |
| `regions`          | `bethe`, `kikuchi(size)`, `junction`, `mean-field`, `structured(blocks)`                                                                        | BP, GBP, Kikuchi, mean field, structured mean field           |
| `power`, `damping` | per factor power (fractional EP), damping, adaptive damping on improper messages                                                                | fractional and damped EP                                      |
| `siteGranularity`  | `factor`, `plate` (one averaged site per plate: stochastic EP, memory independent of data size)                                                 | EP, SEP, distributed EP                                       |
| `family`           | per variable: which exponential family approximates it (Gaussian for a positive variable on log scale, Gamma, ...)                              | the approximating family as a choice                          |
| `evidence`         | `bethe`, `ep-energy`, `elbo`                                                                                                                    | the log-evidence or bound that fits the projection            |
| `improper`         | `skip`, `damp`, `reset`, `error`                                                                                                                | what to do with negative precisions (the standard EP failure) |

**Initial operator set** (about 35, with closed-form EP or VMP messages where the literature has them):
Gaussian family (`Gaussian(mean, precision)`, `plus`, `difference`, `times-constant`, `product`, `sum-of-array` and `dot` of Gaussians
with constants, `linear-regression-row`); precision priors (`Gamma-precision`, `Wishart`); probabilities (`Beta-Bernoulli`,
`Dirichlet-Discrete`, `Binomial`, `Poisson-Gamma`); links (`probit` (exact EP moments), `logistic` (Jaakkola bound for VMP,
Knowles-Minka or quadrature for EP), `softmax` (Bouchard bound, quadrature)); truncation (`is-positive`, `greater-than`, `max`, `min`:
truncated-Gaussian moments); structure (`Gate/Switch`, `Mixture`, `Copy`, `Replicate`, `SumOver`, `GetItems/at`, `Concat`); and
discrete tables from `pgm/factor` (`Table`, `And`, `Or`, `Equal`, `Constant`). Each operator ships with the unit test that its
message equals a brute-force numerical integration on a small instance.

**Schedules and convergence.** Schedules come from the graph (leaves inward then outward on trees; sweeps otherwise) with the existing
residual and sequential variants. Each sweep logs the largest message change, the free energy (Bethe for BP, EP energy for EP, ELBO
for VMP), the number of improper messages repaired and, per variable, the current mean and sd. Messages are `message` events (`m`) so
the render package's message timeline and structure overlay show them, and per-sweep marginals make the figure of an approximation
converging.

**Relation to `pgm`.** The discrete BP, Gaussian BP, EP and VMP in `pgm/inference` remain; the new engine, restricted to discrete tables
or Gaussian factors, must reproduce them to tolerance (a test, section 6), and later `pgm` can delegate. `pgm/ratings.ts` (TrueSkill),
`gp/approximations.ts` (probit EP), `mixture/variational.ts` and `decomposition/lda.ts` are the hand-written references; each is a
benchmark model the general engine has to match (two of them against scikit-learn).

### 5.6 The structural side of inference

**What is analysed** (`analyseSpec`, and `analyse` for programs; one `ModelAnalysis`): per latent site, its exponential-family form,
children, the **conjugate pair** with its children _through plates, deterministic linear maps and index lookups_ (today only top-level
pairs), whether its Markov blanket is small, whether it is **collapsible**; globally, the moral graph and its treewidth for the
discrete part, a linear-Gaussian sub-network (maximal subgraph with Gaussian conditionals and affine means), tree or chain structure
of the factor graph, plate-vectorisable structure, and which nodes are opaque.

**How structure is used** (each is a planner rule with an explanation):

| structure found                                         | exploitation                                                                                  | row        |
| ------------------------------------------------------- | --------------------------------------------------------------------------------------------- | ---------- |
| all latents conjugate                                   | closed form                                                                                   | A2         |
| conjugate or Gaussian sub-network inside a bigger model | `collapse` it (Rao-Blackwellise): run the sampler on what remains; draw the rest by `recover` | A5, D4     |
| low-treewidth discrete part                             | enumerate or junction tree, summed out inside NUTS (exists), exact conditionals inside Gibbs  | A1, D4     |
| tree or chain factor graph                              | tree schedule BP is exact; HMM and Kalman special forms                                       | A3, A4, B1 |
| small Markov blankets                                   | blanket-local single-site updates (Bean Machine, JAGS) re-evaluate only the blanket           | D3, D4     |
| plates of exchangeable data                             | vectorise; stochastic EP and minibatch SVI per plate                                          | B5, C1, C5 |
| hierarchical funnel geometry                            | suggest or apply non-centring (`reparam`)                                                     | D8         |

**Mixing message passing, VI and sampling.** The mixes that earn their place, each a `hybrid` pattern:

| pattern                                                  | how                                                                                                                                                              | gain                                                           |
| -------------------------------------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------- | -------------------------------------------------------------- |
| collapse then sample                                     | exact marginalisation of the linear-Gaussian or conjugate block (a message-passing or closed-form computation), NUTS on the remainder                            | Rao-Blackwellisation; fewer dimensions; no divergences from it |
| EP or VMP on the likelihood part, sample hyperparameters | EP gives the site approximations for the latent field; the hyperparameters are sampled or gridded with the EP evidence (EP-in-the-loop, as in GP classification) | the INLA/GP pattern at modest cost                             |
| approximation as warm start                              | Laplace, Pathfinder, EP or VMP result sets the initial point, the mass matrix or the proposal of a later sampler                                                 | faster, more reliable warm-up                                  |
| approximation as proposal                                | EP or VMP posterior as an importance or SMC proposal, with the exact weights                                                                                     | asymptotic exactness with a good start                         |
| local exact, global stochastic                           | SVI on global parameters with exact local updates by message passing (natural-gradient SVI)                                                                      | scales to large data; LDA-style models                         |
| exact discrete, gradient continuous                      | enumerate discrete sites inside NUTS (today), or HMC within Gibbs for discrete sites that cannot be enumerated                                                   | mixed models                                                   |

`hybrid` executes blocks as a Gibbs-style cycle: each block's method targets the block's conditional given the others (the block
target of 5.3). Methods that are not kernels (EP, VMP, SVI, Laplace) enter through the other patterns (collapse, warm start,
proposal), not as blocks; `check` says which patterns each method can take part in.

### 5.7 Outputs: posteriors, exposition records, traces

- **`Posterior`** keeps its five forms and its interface, and gains: `dims` (named, from the spec), `provenance` (method, config, key,
  compiled plan), `evidence` and `evidenceError` where the method has them, `asPrior(site)`, and `draws(site)` with `[chain, draw, ...dims]`
  shape labelled for the render package. Message passing returns `form: 'exact'` or `'variational'` with exponential-family marginals
  (so densities, quantiles and `logProb` exist), plus the final messages.
- **Exposition records** (new, in `lib/exposition/`, each with a constructor, `seal`, a contract-test sample, a view in
  `omai-render/views`): `PosteriorComparison` (per site, several posteriors from different methods and the exact one where known,
  as densities or histograms, with the distance to the reference), `MessageTrace` (per sweep: the approximations of chosen variables
  and the largest change, drawn as converging bells), `ChainDiagnostics` (trace, rank, ECDF-difference and autocorrelation data per
  quantity: the completeness survey's gap number 1), `CalibrationRecord` (SBC rank histogram and ECDF-difference with the envelope),
  `ElboRun` (ELBO curve with k-hat). Most of the picture machinery exists (`structure-steps`, message timelines, posterior
  grids); the rank and ECDF-difference charts are new render capabilities and so get gallery recipes (one per chart), by the
  repository's rule that the gallery shows what render can draw and nothing about methods.
- **The tape.** Each method's contract (algorithms.md). A message-passing run emits `message` events and per-sweep tensors; an
  MCMC run per-iteration scalars per chain (acceptance, step size, energy, divergent) with the draws as a policy-controlled tensor
  channel so a page can watch chains fill in.

### 5.8 Browser constraints

| concern                       | position                                                                                                                                                                                                                                                                                                                                                                                                                                                                      |
| ----------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| compute budget                | measured: about 3 ms per NUTS iteration on 10 to 20 latent dims and 500 rows in Node (inventory section 6; a browser may be 1.5 to 3 times slower, not measured). Target for demos: every poc interactive run below 3 s on a laptop, with a visible progress bar and cancellation. Models above several thousand rows per gradient need the compiled plan below, or minibatch SVI, or message passing                                                                         |
| interpretive overhead         | today every log-density evaluation re-runs the closure through the handler stack and the autodiff transform. A spec makes a **compiled evaluation plan** possible: a topologically sorted list of operator calls over preallocated `Float64Array`s with hand or traced gradients; it is an optimisation behind the same `LogDensity`, justified only when a benchmark shows the closure path is the bottleneck. It is in 0.1.14 as a measured work item, not an assumption    |
| workers                       | what crosses to a worker must be JSON: **a spec is, a closure is not**. Methods are made in a worker by name and configuration (the producer protocol exists); chains and particles can run on several workers with keys folded by chain index so results are identical however many workers run (determinism is part of the contract and a test)                                                                                                                             |
| progress and cancellation     | steps are small (5.4); `runFor` slices work; `useJob` stages and `workerStage` exist. The present MCMC driver's single step is the one thing to change                                                                                                                                                                                                                                                                                                                        |
| determinism                   | every method derives keys by name from the run key; the same seed gives the same chains on any machine and any number of workers                                                                                                                                                                                                                                                                                                                                              |
| bundle size and import safety | `sideEffects: false`, one entry per folder. **The operator registry must not register by import side effect**: operators are exported values collected by an explicit `defaultOperators()` so a page that imports only EP-Gaussian does not pull the rest; `scripts/import-safe.test.ts` covers new entries automatically. New entry points (`omai-engine/message-passing`, `omai-engine/probprog/model`, `.../inference`, `.../validation`) are added to `omai-meta/modules` |
| precision                     | `Float64Array` throughout; WebGPU is f32-only and is not a path for exact gradients                                                                                                                                                                                                                                                                                                                                                                                           |
| what must be tiny             | model construction, analysis, planning, message passing sweeps on plated models, Laplace, small SVI, conjugate and enumeration: all interactive                                                                                                                                                                                                                                                                                                                               |
| what may be heavy             | NUTS on thousands of rows, SMC with thousands of particles, nested sampling beyond a few dimensions, SBC campaigns, neural SBI: with progress, a worker, and a "demo size" default                                                                                                                                                                                                                                                                                            |

## 6. How we prove it

The standing rules apply: parity with the reference as the goal; tiered tolerances; tests run in seconds; Python is only for
fixtures. Every inference method gets, in this order of strength:

| layer                  | what is checked                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                       | reference or oracle                                                                                                                                                              | where                                              |
| ---------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | -------------------------------------------------- |
| model layer            | spec and hand-written program give the same log joint at random points (the shared `MODELS` of `src/testing/probprog.ts` are the programs, written again as specs); JSON round trip is the identity; shape rules; bind errors                                                                                                                                                                                                                                                                                                                                                                                                                         | NumPyro goldens that exist for those programs; property tests (fast-check)                                                                                                       | `probprog/model/*.test.ts`, `*.property.test.ts`   |
| handlers               | reparam preserves the density up to its Jacobian; collapse's marginal equals the closed form and its recovered draws have the right conditional; lift equals writing the prior; scope round trips                                                                                                                                                                                                                                                                                                                                                                                                                                                     | closed forms; NumPyro `reparam` fixtures; `collapse` has no installed oracle (needs funsor), so exact laws                                                                       | `probprog/*.test.ts`, `oracle/cases/probprog/`     |
| exact methods          | closed-form posteriors and evidence; HMM forward algorithm; Kalman; brute-force enumeration; **grid and quadrature posterior** as a generic oracle for models of up to 4 dimensions                                                                                                                                                                                                                                                                                                                                                                                                                                                                   | scipy, hmmlearn, filterpy, pgmpy, pyAgrum (installed); `validation/grid.ts` (new, the most reusable test device)                                                                 | `probprog/validation/`                             |
| message passing        | exact on trees of Gaussians and on discrete trees; on loopy graphs equals the existing `pgm` BP, EP and VMP when restricted to their domains; EP and VMP on the same model reproduce `pgm/ratings` TrueSkill, `gp/approximations` probit EP, scikit-learn's `BayesianGaussianMixture` and `LatentDirichletAllocation` (VMP and natural-gradient SVI); each operator's message equals brute-force integration; properties: VMP's ELBO rises monotonically, schedule does not change a tree's result, alpha = 1 is EP, EP energy is stationary at convergence; the variance underestimate of VMP and the mass-covering of EP are shown against the grid | scikit-learn (installed), pgmpy and pyAgrum for discrete BP, the existing hand-written implementations, grid, and the published numbers of the clutter problem and TrueSkill [M] | `message-passing/*.test.ts`, golden for sklearn    |
| variational            | ELBO at fixed draws against NumPyro (exists); on conjugate models the ELBO at the exact posterior equals the log evidence; full-rank and structured guides recover Gaussian posteriors; estimator unbiasedness (mean of TraceGraph and Trace gradients over many draws agree with the exact gradient on a small model) and variance ordering; Pathfinder against BlackJAX                                                                                                                                                                                                                                                                             | NumPyro, BlackJAX 1.7.1 (installed), closed forms                                                                                                                                | `probprog/elbo*.test.ts`, `oracle/cases/probprog/` |
| MCMC                   | **Geweke joint-distribution test** (alternate drawing data given parameters and a transition given data; the marginal of the parameters must stay the prior); **simulation-based calibration**; moments within Monte Carlo standard errors of the exact posterior; ESS per gradient against NumPyro and BlackJAX on shared targets; step-size adaptation converges to the target acceptance; divergences appear on the centred funnel and vanish when non-centred                                                                                                                                                                                     | closed forms, grid, NumPyro and BlackJAX chains' summary statistics (moments, not draws), ArviZ diagnostics goldens (exist)                                                      | `probprog/validation/`, `sampling/*.test.ts`       |
| SMC, particle, nested  | the evidence estimate is unbiased (mean over repetitions within error) and equals the Kalman and HMM exact evidence; nested sampling returns the analytic evidence of Gaussian targets and shells                                                                                                                                                                                                                                                                                                                                                                                                                                                     | Kalman, forward algorithm, analytic                                                                                                                                              | `sampling/*.test.ts`                               |
| cross-method agreement | `agree(bound, methods, reference)` returns a table per site and method: mean error in posterior sd, KL and 1-D Wasserstein to the reference, 90% interval coverage; thresholds per benchmark model name which methods must agree and which are expected to differ (and how: VMP too narrow, Laplace misses skew, EP mass-covering)                                                                                                                                                                                                                                                                                                                    | the exact or grid answer where it exists, otherwise a long NUTS reference whose **moments** are stored as a fixture (never draws)                                                | `probprog/validation/agreement.ts`                 |
| calibration            | `sbc(bound-generator, method, { replications })`: ranks of the true parameter among posterior draws; uniformity by a chi-square test on the histogram and the ECDF-difference envelope; a deliberately biased method (a too-narrow guide, a wrong Jacobian) **must fail**, which is the test that the test can detect a bug                                                                                                                                                                                                                                                                                                                           | Talts et al. 2018 [M]                                                                                                                                                            | `probprog/validation/sbc.ts`                       |
| predictive             | posterior predictive checks, PSIS-LOO and WAIC (exist, ArviZ goldens), power-scaling sensitivity (new, cheap)                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                         | ArviZ 1.3.0 (installed)                                                                                                                                                          | `probprog/criticism.ts`                            |
| browser                | import-safe for new entries; worker and main thread give identical results for a seed; a cancelled run stops within one step; the compiled plan equals the closure path to 1e-12                                                                                                                                                                                                                                                                                                                                                                                                                                                                      | `scripts/import-safe.test.ts`; producer tests                                                                                                                                    | `render` and `probprog` tests                      |

**Flakiness policy.** Statistical tests use fixed seeds and budgets set once from a documented calculation (the rank test's alpha and
the replications), so a failure means a change, not luck; the unit-test budget is seconds. Heavy campaigns (SBC with several hundred
replications across all methods, the full agreement matrix) are a **script the main session runs on request** and whose result
table is committed to `docs/probprog/` with the commit it was run on; they are not part of `make verify` (question 11).

**Python oracles.** Installed and used: NumPyro 0.22.0 (programs, ELBO, reparam, chains' statistics), BlackJAX 1.7.1 (kernels,
Pathfinder, SMC, adaptation), ArviZ 1.3.0 (diagnostics, LOO, `psense`), pgmpy 1.1.2 and pyAgrum 3.3.0 (discrete exact and BP),
scikit-learn 1.9.1 (VB mixture, LDA), hmmlearn, filterpy, scipy. **Not installed**: Pyro (would give `TraceGraph_ELBO` and `lift`
semantics; it depends on torch, which is installed for `oracle-torch`), PyMC, TensorFlow Probability, any Stan, emcee, dynesty,
UltraNest. Recommendation: add `pyro-ppl` to `oracle-prob` (small, closes a documented TODO in `structure.md` 4.32), nothing else;
Stan and PyMC reference results come from closed forms and from our own long NUTS references (question 7).

### 6.1 The benchmark suite

Named models as specs with their truth, in `probprog/models/` (the LIB.md home for small programs), each carrying: spec, data
generator (or vendored data with licence), `truth` (exact posterior, evidence or reference moments), the list of methods expected
to agree, and the ones expected to fail and how. Expected behaviour below is the qualitative result the suite asserts; numbers
come from the references, not from memory.

| model                                             | tests                                                      | truth                                                                  | expected                                                                                                                                      | release |
| ------------------------------------------------- | ---------------------------------------------------------- | ---------------------------------------------------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------- | ------- |
| beta-binomial, gamma-Poisson, normal-normal       | conjugate machinery, collapse                              | closed form                                                            | conjugate exact; VMP exact for conjugate-exponential; EP exact; all samplers within MCSE                                                      | 0.1.13  |
| Bayesian linear regression (normal-gamma)         | A2, VMP vs exact, EP, Laplace                              | closed form (Student t)                                                | VMP's mean exact, variance too small; EP exact on the Gaussian part                                                                           | 0.1.13  |
| probit and logistic regression                    | EP, Laplace, VMP bound, NUTS, SVI                          | grid for d <= 3, else NUTS reference                                   | EP closest to exact; Laplace good when the posterior is near Gaussian; mean-field too narrow; logistic needs a bound or quadrature for EP/VMP | 0.1.13  |
| clutter problem (Minka 2001)                      | EP vs VB vs Laplace                                        | grid (1-D)                                                             | EP near exact; VB and Laplace worse (the paper's result [M])                                                                                  | 0.1.13  |
| TrueSkill, two players and teams                  | factor graph EP                                            | existing `pgm/ratings`, grid                                           | the general engine equals the hand-written one                                                                                                | 0.1.13  |
| eight schools, centred and non-centred            | funnel, divergences, reparam, SBC                          | NUTS reference moments; NumPyro                                        | centred: divergences, poor tau; non-centred: none; mean-field underestimates tau's spread                                                     | 0.1.13  |
| Gaussian HMM and discrete HMM                     | chain exact, enumeration, Gibbs, SMC, PMCMC                | forward algorithm; hmmlearn                                            | exact methods match; PF evidence unbiased                                                                                                     | 0.1.13  |
| Gaussian mixture, with label switching            | VMP, EM, Gibbs, NUTS with ordered constraint               | sklearn BayesianGaussianMixture; permutation-invariant summaries       | R-hat poor without the ordered constraint; VMP matches sklearn                                                                                | 0.1.13  |
| Neal's funnel, Haario's banana, bimodal Gaussians | geometry, tempering, ensemble                              | analytic                                                               | NUTS fails the centred funnel; Laplace and mean field wrong on the banana; parallel tempering recovers mode weights; VI is mode-seeking       | 0.1.14  |
| hierarchical radon (varying intercepts, slopes)   | hierarchies, collapse of the Gaussian block, reparam, ADVI | NUTS reference; NumPyro                                                | collapse plus NUTS on hyperparameters is faster per ESS; centred is worse                                                                     | 0.1.14  |
| latent GP regression and classification           | EP, Laplace, collapse, INLA-style                          | closed form (regression); `gp/approximations`; grid on hyperparameters | EP vs Laplace on the classification marginal likelihood                                                                                       | 0.1.14  |
| LDA on a small corpus                             | VMP, natural-gradient SVI, collapsed Gibbs                 | sklearn LatentDirichletAllocation perplexity; gensim                   | VMP matches sklearn; collapsed Gibbs agrees within sampling error                                                                             | 0.1.14  |
| stochastic volatility                             | PMCMC, SMC, NUTS with scan                                 | long NUTS reference                                                    | PMMH agrees; bootstrap filter poor on tails                                                                                                   | 0.1.14  |
| Bayesian neural network (small regression)        | SVI, SGLD, ensembles, Laplace                              | HMC reference on a tiny net                                            | uncertainty grows away from data; mean-field overconfident                                                                                    | 0.1.14  |
| item response theory (2PL)                        | EP, VMP, NUTS, SBC                                         | `pgm/irt` marginal ML; NUTS                                            | agree on abilities and items                                                                                                                  | 0.1.14  |
| change point / Dirichlet-process mixture          | discrete mixing, SMC, Gibbs                                | exact for a fixed-K case                                               | Gibbs and SMC agree                                                                                                                           | 0.1.14  |

### 6.2 The demonstrations

By the repository's rules (CLAUDE.md): **proofs of concept and gallery recipes are not added unasked.** The plan therefore lists what
would show the system and the user decides which to ask for.
- _Proofs of concept_ (`apps/pocs`, one page each; a page composes library estimators, a model spec and a `useSession`): (1) the
  vertical slice page (section 7.2); (2) EP, VMP and Laplace against the grid and NUTS on the clutter problem or a probit
  regression, with alpha as a slider; (3) a factor graph with Gaussian messages converging (TrueSkill); (4) the funnel: centred
  against non-centred with divergences, switched by `reparam`; (5) the compositional editor: assign blocks of a model's sites to
  methods and watch the plan and the posterior; (6) SBC rank histograms of three methods, one deliberately wrong; (7) sequential
  Bayes: a posterior passed as the next prior, with the pushforward.
- _Examples_ (`examples/<slug>`, programs with a figure and a tape, for the front page): eight schools four ways; EP versus MCMC on a
  probit regression; a calibration check.
- _Gallery_: only for the new render capabilities (rank and ECDF-difference charts, trace and forest plots, the Gaussian-bell overlay on
  messages if it needs a new overlay type), as the survey already requires.

## 7. Phased roadmap

### 7.1 Work packages

Sizes: S a day or less, M a few days, L a week or more (of one agent). "Owns" lists the files an agent may edit; files generated or shared
(`index.ts` barrels, `llms.txt` and `index/`, `docs/api-surface.txt`, `CHANGELOG.md`, `docs/structure.md`, `omai-meta/modules`)
are **not** edited by lane agents: the main session regenerates barrels and the index and merges the docs, which removes the usual
merge conflicts.

| WP   | work                                                                                                                                                                                                                                                           | size | depends                        | owns                                                                                                               | acceptance                                                                                                                                              | browser-demonstrable                      |
| ---- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ---- | ------------------------------ | ------------------------------------------------------------------------------------------------------------------ | ------------------------------------------------------------------------------------------------------------------------------------------------------- | ----------------------------------------- |
| WP0  | Contracts and structure.md: `ModelSpec` JSON schema, `InferenceMethod`, `FactorOperator`, the new folders; type stubs and TSDoc only                                                                                                                           | S    | -                              | `probprog/model/spec.ts`, `probprog/inference/method.ts`, `message-passing/types.ts`, `docs/structure.md` 4.32/new | reviewed by the user; `tsc` clean; no behaviour                                                                                                         | -                                         |
| WP1  | Model layer: builder, ops registry (30 ops), named dims, constraints, data binding (`bind`, `withData`, missing, batch, `bindDataset`), `toProgram`, JSON, `analyseSpec`, `toStructure`, `include`, `condition`, `do`                                          | L    | WP0                            | `probprog/model/**` (except `factors.ts`)                                                                          | spec and program agree on log joint for the shared `MODELS`; JSON round trip; every existing engine runs a bound spec; structure drawn                  | yes (spec to drawing to NUTS)             |
| WP2  | Handlers: `reparam` (+ reparameterisers), `collapse` (delayed sampling), `lift`, `scope`, `escape`, `equalize`, `mean`, `inferHint`; trace operations `regenerate`, `update`, `assess`; `defineKernel` block target                                            | M-L  | -                              | `probprog/handlers.ts`, `reparam.ts`, `collapse.ts`, `lift.ts`, `trace-ops.ts`, `probprog/inference/kernel.ts`     | laws and goldens of section 6; funnel divergences vanish under non-centring; collapse equals closed form                                                | yes (funnel)                              |
| WP3  | Unified interface: `InferenceMethod`, registry, adapters for the 12 engines, `Space` and contract for SVI, Gibbs, SMC, EM, Laplace, importance; fine-grained MCMC step; capability `check` with reasons; `plan` and `explain` over methods; `infer` runs plans | M-L  | WP0, WP1                       | `probprog/inference/{registry,adapters,plan}.ts`, `probprog/infer.ts`, `probprog/engines/*.ts` (all)               | old `infer` tests pass unchanged; every method runs as a session with automatic controls; MCMC cancels within one iteration; worker and main agree      | yes (any method by name in a worker)      |
| WP4a | Message-passing core: messages, operator registry, the 35 operators, engine with `projection`, `schedule`, `damping`, `improper`; EP, VMP, BP as settings; tape and events                                                                                     | L    | WP0                            | `message-passing/**` (except `regions.ts`, `power.ts`, `stochastic.ts`)                                            | each operator equals numerical integration; EP exact on Gaussian trees; reproduces `pgm` BP, EP and VMP, TrueSkill, probit EP; VMP matches sklearn BGMM | yes (messages converge on a factor graph) |
| WP4b | Variations: power and fractional EP, alpha-divergence, stochastic EP, region graphs (Bethe, Kikuchi, GBP), expectation consistent, numerical projections (quadrature, Laplace, sample), AMP                                                                    | L    | WP4a                           | `message-passing/{regions,power,stochastic,numerical,amp}.ts`                                                      | alpha = 1 is EP, alpha sweep interpolates; Kikuchi beats Bethe on a grid against the junction tree; SEP memory independent of n                         | yes (alpha slider)                        |
| WP4c | Compile spec to factor graph; EP/VMP as method adapters; collapse using message passing; EP posterior as warm start and proposal                                                                                                                               | M    | WP1, WP4a, WP3                 | `probprog/model/factors.ts`, `probprog/inference/mp.ts`                                                            | a bound spec runs on EP and VMP and the planner chooses them with reasons                                                                               | yes                                       |
| WP5  | Validation kernel: grid and quadrature oracle, exact helpers, `agree`, SBC, Geweke, distances (1-D Wasserstein, KS, KL on grids), runner; oracle cases for new goldens                                                                                         | M    | WP0 (programs only; WP1 helps) | `probprog/validation/**`, `oracle/cases/probprog/**`, `src/testing/probprog.ts`, `fixtures/probprog/**`            | SBC passes for NUTS on a conjugate model and fails for a deliberately wrong guide; agreement table for the slice                                        | yes (rank histograms)                     |
| WP6  | Benchmark models as specs with truth (first set 0.1.13, rest 0.1.14); registered with tags like datasets                                                                                                                                                       | M    | WP1, WP5                       | `probprog/models/**`                                                                                               | each model has truth, references and a smoke test                                                                                                       | yes                                       |
| WP7  | SVI breadth: `TraceGraph_ELBO`, `TraceEnum_ELBO`, structured and hierarchical guides, amortised guides, natural-gradient SVI, Pathfinder, control variates, ADVI step sequence                                                                                 | L    | WP3, WP4a (natural gradient)   | `probprog/elbo*.ts`, `guides*.ts`, `engines/svi.ts`, `probprog/engines/pathfinder.ts`                              | estimator unbiasedness and variance; Pathfinder against BlackJAX; natural gradient converges to CAVI                                                    | yes                                       |
| WP8  | MCMC breadth: `kernel:` slot for every `sampling` kernel on specs and programs; blocks in `hybrid`; HMC within Gibbs, MixedHMC; parallel tempering; MCLMC, SGHMC, Barker, ensemble; adaptation (CHEES, low-rank)                                               | L    | WP2, WP3, WP5                  | `sampling/**` (new files), `probprog/engines/nuts.ts`, `mcmc-options.ts`, `probprog/inference/hybrid.ts`           | BlackJAX and NumPyro statistics; PT recovers mode weights; every kernel passes Geweke and SBC                                                           | yes                                       |
| WP9  | Structure exploitation: `analyseSpec` conjugacy through plates and linear maps, blanket and treewidth analysis, collapse planning, hybrid decomposition in `plan`, compiled evaluation plan (if measured)                                                      | L    | WP1 to WP4                     | `probprog/model/analysis*.ts`, `probprog/inference/{plan,decompose}.ts`, `probprog/model/compile-plan.ts`          | planner chooses collapse plus NUTS on the radon model and is faster per ESS; explanations are correct                                                   | yes                                       |
| WP10 | SMC and evidence: guided proposals, PMCMC, nested sampling, evidence estimators slot                                                                                                                                                                           | M-L  | WP3, WP5                       | `sampling/{nested,pmcmc}.ts`, `probprog/engines/smc.ts`                                                            | evidence unbiasedness; nested sampling analytic Gaussian evidence                                                                                       | yes                                       |
| WP11 | Exposition records and views, MCMC diagnostic charts, gallery recipes for new charts only                                                                                                                                                                      | M-L  | WP3, WP5                       | `exposition/inference*.ts`, `render/src/views/inference*.ts`, `render/src/charts/mcmc*.ts`, gallery recipes        | contract and smoke tests; thumbnails regenerated (`make thumbs`)                                                                                        | yes                                       |
| WP12 | Docs: concept pages, TSDoc examples, `@index` tags, `make agent-index`, catalogue rows, structure.md                                                                                                                                                           | M    | all                            | `docs/**` (main session)                                                                                           | `make verify` passes, the docs catalogue builds                                                                                                         | -                                         |
| WP13 | Demonstrations (pocs, examples)                                                                                                                                                                                                                                | M    | on request                     | `apps/pocs/src/pocs/<slug>/`, `examples/<slug>/`                                                                   | the user asked for them                                                                                                                                 | yes                                       |
| WP14 | Stretch: neural SBI, INLA, BNN modules, probabilistic circuits, a Stan-like text front end, decision-diagram exact inference                                                                                                                                   | L    | later                          | -                                                                                                                  | -                                                                                                                                                       | -                                         |

### 7.2 The first vertical slice

**Goal:** one model written in the new DSL, run by two different engines, compared with the exact answer, traced and shown in a page.
It touches WP0, a thin WP1, a thin WP4a, a thin WP5 and one view; it should take one lane about two weeks, and it is the gate for
starting the others (they build on frozen contracts, not guesses).

**The model.** Bayesian probit regression with an intercept and two covariates (three latents), 40 rows from a library dataset with
known truth (`blobs` or the `lineData`-style classification generator).

```ts
const probitModel = defineModel('probit', (m) => {
  const n = m.range('row'),
    d = m.range('feature')
  const X = m.data('X', { dims: [n, d] })
  const y = m.data('y', { dims: [n], support: 'binary' })
  const w = m.latent('w', normal(0, 2), { dims: [d] })
  m.observe('y', bernoulli({ probs: ops.probit(X.matmul(w)) }), y)
})
```

**Steps and acceptance:**
1. Builder and `ModelSpec` for data, latent, observed nodes with named dims and the operators `matmul`, `probit`; `bind`; JSON round trip. _Accept:_ the spec's log joint equals the hand-written program's at 20 random points (1e-12).
2. `toProgram` and the adapter that runs NUTS on it through the existing engine. _Accept:_ matches the existing NUTS on the hand-written program (same seed, same draws).
3. `message-passing` with the Gaussian family, the `matmul-constant`, `Gaussian prior` and `probit` operators, the EP projection and the flooding schedule; `toFactorGraph` for this spec. _Accept:_ EP's marginals within 0.05 posterior sd of the grid answer for the means on this data; the engine reproduces `gp/approximations` probit EP when the model is a GP-classification (a shared test of the same operator).
4. `validation/grid.ts`: adaptive grid posterior in three dimensions with evidence; `agree` over NUTS, EP and Laplace. _Accept:_ NUTS within 3 Monte Carlo standard errors; Laplace and EP reported with KL and sd-errors; EP closer than Laplace on at least the intercept (the expected result; if it is not, that is a finding).
5. The tape: EP emits `message` events and per-sweep marginals; NUTS emits per-iteration scalars. _Accept:_ both run through the producer in a worker by name with the spec as JSON configuration, and the main-thread result equals the worker's.
6. One view: the spec drawn as a factor graph, with the three posteriors overlaid per weight and the EP change curve. This needs a user request for a page (`apps/pocs/src/pocs/probit-two-engines/`) per the repository's rule; without one, the same figure is an `examples/` program.

### 7.3 Lanes and releases

Three agents, each in its own worktree (`agent-<id>-<word>`), merged in order; lanes touch disjoint files (the "owns" column).

| lane | 0.1.13 order                        | 0.1.14 order                                        |
| ---- | ----------------------------------- | --------------------------------------------------- |
| A    | WP0 (with the user), WP1, WP3, WP4c | WP9, WP8's `hybrid`, WP11                           |
| B    | WP4a, WP4b                          | WP7 natural-gradient SVI, B8, WP10                  |
| C    | WP2, WP5, WP6 (first set)           | WP8 kernels, WP6 rest, SBC campaign, WP7 estimators |

**0.1.13: the architecture and message passing.** Lane lengths are about 15 to 20 agent-days each. The cut line, if time presses:
_must_ have WP0, WP1, WP2, WP3, WP4a, WP5 and the slice; _should_ have WP4b (power EP, alpha, SEP, regions), WP4c, WP6 first set;
_moves to 0.1.14_: anything below the line. Rationale: the model layer and the contracts are the user's complaint and everything
else builds on them; message passing is the largest genuinely new capability and is the best browser story; validation must
ship with the engines it validates, never after.

**0.1.14: breadth and structure.** WP7 to WP11 and the rest of WP6, with the SBC campaign and the agreement matrix run and
committed. Rationale: these are additive on stable contracts and mostly independent of one another, so they parallelise well.

The main session alone runs `make verify` or the gate, regenerates barrels and the agent index, and merges; agents test only their
own files (CLAUDE.md), niced and single-worker.

## 8. Risks, non-goals and open questions

### 8.1 Risks

| risk                                                                                | likelihood | what to do                                                                                                                                                                               |
| ----------------------------------------------------------------------------------- | ---------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| the DSL becomes a second language to maintain, with a growing operator list         | medium     | closed registry, `custom` escape hatch, every operator has a tensor evaluation first and message operators only where there is a closed form; cap the list at the first release's        |
| spec and program semantics drift                                                    | medium     | the equivalence test on every benchmark model in CI; the spec compiles to the program, there is no second evaluator (except the optional compiled plan, which has its own equality test) |
| EP numerical fragility (negative variances, oscillation, non-convergence)           | high       | `improper` slot with repair strategies and counts on the tape; damping defaults; `check` warns for known hard factors; tests include a failing case and its repair                       |
| statistical tests that flake or run long                                            | medium     | fixed seeds and budgets, an SBC rank test with a documented alpha, a deliberate-failure test, heavy campaigns as a script                                                                |
| closure interpretive overhead limits browser model size                             | medium     | measure first (WP9); compiled plan if justified; message passing and minibatch SVI as the scalable paths                                                                                 |
| lifting programs to specs is unsound for non-structure-stable programs              | medium     | optional, verified by re-tracing with perturbed values, refuses otherwise; builder is the main path                                                                                      |
| scope creep from the repertoire (60 rows)                                           | high       | the seven machines and the slots; the cut line in 7.3; stretch items marked                                                                                                              |
| API churn for users of the present probprog                                         | low        | nothing is removed; `infer(model, data)` keeps working                                                                                                                                   |
| generated files and shared docs cause merge conflicts between lanes                 | medium     | lane agents never edit barrels, `llms.txt`, `api-surface.txt`, `CHANGELOG.md` or `structure.md`; the main session regenerates and merges                                                 |
| a new domain `message-passing` interacts with `make layers` and `omai-meta/modules` | low        | create it in WP0 with the layering check and the modules entry                                                                                                                           |
| the validation reference (a long NUTS run) is itself wrong                          | low        | prefer closed form and grid; where a NUTS reference is used it is checked against NumPyro's chain moments                                                                                |

### 8.2 Non-goals

Stan language compatibility or its C++ toolchain; universal probabilistic programs with unbounded stochastic recursion (the existing
`importance` and `smc` engines on `markov` programs cover the useful part); GPU or WebAssembly kernels; code generation; trans-dimensional
(reversible jump) inference; full INLA with sparse precision in these two releases; neural SBI in these two releases (listed as
stretch); probabilistic circuits and decision-diagram exact inference in these two releases; automatic differentiation changes;
distributed (multi-machine) EP.

### 8.3 Open questions for the user

Each has a recommendation; the first three change the shape of the work.
1. **Model-definition style.** Recommended: D, the builder producing a spec that compiles to the present programs, with programs kept.
   Alternatives: keep closures only and add a lifting step (cheaper, weaker for worker and message passing); a text language (later).
2. **Keep or replace the old probprog API?** Recommended: keep entirely; the new layer sits on top and `infer` stays. Replacing it would
   throw away the NumPyro goldens for nothing.
3. **How much of Stan's semantics?** Recommended: the block discipline, constraints with Jacobians, derived and generated quantities,
   standalone generated quantities on new data. Not recommended: a Stan text parser (a later option; JAGS or BUGS-style text would
   be cheaper if wanted).
4. **Home of message passing.** Recommended: a new domain `lib/message-passing/` (a constructed domain between `pgm` and `probprog`).
   Alternatives: `pgm/message-passing/` (pgm is about discrete and Gaussian models and would grow a continuous exponential-family
   layer) or inside `probprog`. Name also open: `message-passing`, `messages`, `inference`.
5. **Release split.** Recommended: 0.1.13 architecture and message passing, 0.1.14 breadth (SVI and MCMC additions). You suggested SVI
   in the first release; I moved the new SVI algorithms because SVI already works, so say if you want them earlier.
6. **Scope of simulation-based inference.** Recommended: ABC (exists), synthetic likelihood, SBC as validation in 0.1.14; neural
   posterior and likelihood estimation as stretch.
7. **Oracle environment.** Recommended: add `pyro-ppl` to `oracle-prob`; do not add PyMC, TFP, Stan, dynesty. OK to extend the venv?
8. **Operator set.** Recommended: about 35 closed-form operators plus numeric fallbacks, not the several hundred of Infer.NET. Say if
   a particular family of models (survival, ordinal, count, mixed effects) should be in the first set.
9. **Naming.** Proposed: `defineModel`, `latent`/`observe`/`derived`/`generated`, `bind`, `hybrid`, `InferenceMethod`, `message-passing`.
   `sample` stays the program-level word. Prefer something else?
10. **Proofs of concept.** The slice, and the demonstrations in 6.2, each need your request before any page is added. Which, if any,
    do you want in 0.1.13? (Examples programs need no request.)
11. **Heavy validation.** A script run on request (SBC and the agreement matrix with committed result tables) outside `make verify`.
    Acceptable, or do you want a periodic `make` target?
12. **Browser budget.** Is "every demo under 3 seconds on a laptop, with progress and cancellation" the right bar? It decides whether
    the compiled evaluation plan is in 0.1.14.
13. **Lifting existing programs to specs** (`fromProgram`): worth the risk in 0.1.13, or later?

### 8.4 What was not verified

- The Python packages' current API (Pyro, PyMC, TFP, Stan, Infer.NET, Turing, Gen, Bean Machine, Birch, WebPPL, Anglican, NIMBLE,
  JAGS) is from memory except the three online checks listed in survey.md; maintenance status of TFP, Bean Machine and Figaro is
  uncertain. NumPyro, BlackJAX and ArviZ rows rest on the earlier completeness introspection, not on re-reading the packages.
- The engine tests were not run; the two timings are one run each on this machine; browser speed was not measured.
- The `docs/methods-catalog` status column and the bodies of the larger engines (`gibbs`, `smc`, `em`, `gaussian`, `abc`) were not
  audited beyond headers and signatures.
- Whether `rsample` exists with implicit gradients for gamma, beta and Dirichlet (needed for pathwise gradients of non-Gaussian
  guides) was not checked.
- Sizes and the lane lengths are estimates in the repository's S/M/L scale, not commitments; sizes of the message-passing operators
  are the least certain.
