# Probabilistic programming in omai: a plan for 0.1.13 and 0.1.14

Status: **accepted for execution on 2026-11-10 with the decisions below.** Written against `main` at `e420394f` (0.1.12 is the
structured-prediction release, finishing), then verified against the source, the installed packages and the web the same day
([verification.md](verification.md); its corrections are applied in this file and the three below). Nothing here is code. The plan
is the main document; five files sit beside it:

| file                               | what it holds                                                                                                    |
| ---------------------------------- | ---------------------------------------------------------------------------------------------------------------- |
| [inventory.md](inventory.md)       | what exists today, file by file, with an honest assessment, two timings, and the causes of "primitive"           |
| [survey.md](survey.md)             | the field: 24 entries, what each does, what to adopt and what to leave, what runs in a browser                   |
| [algorithms.md](algorithms.md)     | the target repertoire: every algorithm with its needs, knobs, tape, proof, size and priority                     |
| [verification.md](verification.md) | what was checked and how (source, installed packages, web, runs), the corrections made, what is still unverified |
| this file                          | decisions, summary, architecture, validation, roadmap, risks                                                     |

Sections 2 to 4 are short forms of the three files above. Sections 5 to 8 are the plan.

## Status (2026-11-10, `main` at 613071d9)

What is on `main`, per work package of section 7.1, checked against the code and the result tables the same day. Done means built,
tested and with its campaign table committed where it has one; partly means named parts are missing. The tables are in
`results/` ([README](results/README.md)), the code under `packages/engine/src/lib/` (`probprog/`, `message-passing/`,
`sampling/`, `dynamics/filtering/`), the documentation in `docs/structure.md` 4.32 and 4.32a and the `## 0.1.13` section of
`CHANGELOG.md`. No GitHub release has been made (decision 7).

| WP   | state       | what is there, and what is not                                                                                                                                                                                                                                              | evidence                                                                                                                      |
| ---- | ----------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ----------------------------------------------------------------------------------------------------------------------------- |
| WP0  | done        | the `ModelSpec` contract, the method convention (`defineInferenceMethod`, `runAnytime`), `FactorOperator`, the model registry by name for workers (the route by which a spec reaches a worker)                                                                              | `probprog/model/spec.ts`, `probprog/inference/{method,models}.ts`, `message-passing/types.ts`                                  |
| WP1  | done        | builder, 26 operators, 22 families, named dimensions, constraints, `bind`/`withData`/`bindDataset`, missing values, batches, `toProgram`, JSON, `analyseSpec`, `toStructure`, `include`, `conditionSpec`, `doSpec`. Not built: the `scan` and `gate` node kinds (reserved)   | `probprog/model/**`                                                                                                           |
| WP2  | done        | `reparam` (`locScale`, `transformReparam`, `circular`), `collapse`, `lift`, `scope`, `mean`, `inferHint`, trace operations, `defineKernel`. Not built: `escape` and `equalize` (Pyro-only, lowest priority). Goldens for `reparam`, `lift` against NumPyro and Pyro           | `probprog/{handlers,reparam,collapse,lift,trace-ops}.ts`, `probprog/inference/kernel.ts`, `oracle/cases/probprog/`; funnel: `results/funnel-non-centred.md` |
| WP3  | done        | a method with its own `Space`, tape, `check`, budget and snapshots for every present engine (NUTS fine-grained; SVI, Gibbs, Laplace, EM, importance, tempered SMC, particle filter, ABC-SMC; the one-step exact ones); `plan`, `explainPlan` and `infer` unchanged. There is no universal `infer`, by decision 2 | `probprog/inference/*-method*.ts`, `exact-methods.ts`, `one-step.ts`                                                           |
| WP4a | done        | message passing core: 8 families, 35 operators, the engine with projection, schedule, damping and improper-policy slots; EP, VMP, BP as named settings with typed tapes. Known gaps: VMP has no rule for `greater-than`, `max`, `min`; EP none for `gaussian-gamma-precision`  | `message-passing/{types,families*,engine,named,projections,schedules,graph}.ts`, `operators/`                                  |
| WP4b | done        | power and fractional EP, alpha-divergence, numerical projections, stochastic and averaged EP, region graphs and generalised BP, expectation consistent inference, AMP                                                                                                      | `message-passing/{power,alpha,numerical,stochastic,regions,consistent,amp}.ts`                                                |
| WP4c | partly done | `toFactorGraph` (plated regressions vectorised, other plates expanded), `epSpecMethod`, `vmpSpecMethod`, `bpSpecMethod` with `check`s that say why not. Not built: collapse by message passing, EP as a warm start or proposal for a sampler                                  | `probprog/model/factors.ts`, `probprog/inference/mp.ts`                                                                       |
| WP5  | done        | `gridPosterior`, closed-form oracles, `agree`, SBC, Geweke, runner, campaigns; `make probprog-validate`; first tables (SBC of NUTS and SVI, two wrong guides detected, agreement, funnel, Geweke)                                                                            | `probprog/validation/**`, `scripts/probprog-validate.ts`, `results/{sbc-*,agreement-*,funnel-*,geweke-*}.md`                  |
| WP6  | partly done | 18 benchmark models with truth, tags, generators and method plans, a campaign each. Not in the suite: Bayesian linear regression (normal-gamma), the clutter problem, HMMs, GP regression and classification, LDA, stochastic volatility, a Bayesian neural network, 2PL IRT, change point and Dirichlet-process mixtures | `probprog/models/**`, `results/benchmark-*.md`, `results/structure-benchmarks.md`                                              |
| WP7  | done        | `TraceGraph` and `TraceEnum` ELBOs (baselines), sticking the landing, DReG, structured, Laplace-initialised, amortised, flow and mixture guides, natural-gradient SVI, Pathfinder, ADVI step sequence; a method for each. The variational plans on logistic regression are reported, not asserted (section 8.1) | `probprog/{elbo-*,guides-*}.ts`, `probprog/engines/{svi-natural,pathfinder}.ts`, `optim/{transform,schedules}.ts`, `results/svi-*.md` |
| WP8  | not done    | MCMC breadth (`kernel:` slot for every `sampling` kernel on specs, `hybrid`, HMC within Gibbs, parallel tempering, MCLMC, SGHMC, Barker, ensemble, adaptation). Not on `main` at this date; `defineKernel` (WP2) and the existing kernels are what there is                | -                                                                                                                             |
| WP9  | partly done | structure reports: conjugacy through plates and linear maps, linear-Gaussian blocks, blankets, components, discrete width, minibatch axes, `explain()`, `collapsePlan`/`applyCollapse`, `fromProgram` (built early, decision 4 deferred it). The compiled evaluation plan was measured and not built; `plan`/`decompose` are not goals (decision 2) | `probprog/model/{analysis*,structure-report,collapse-plan,from-program}.ts`, `results/{structure-*,eval-overhead}.md`        |
| WP10 | done        | guided particle proposals, particle MMH and particle Gibbs, nested sampling, evidence estimators (bridge, stepping stone, thermodynamic integration, importance with k-hat; harmonic mean refused), each with a method and a campaign                                            | `sampling/{pmcmc,nested,evidence}.ts`, `dynamics/filtering/{guided,pmcmc}.ts`, `results/evidence-*.md`                        |
| WP11 | done        | `modelRun` and `chainDiagnostics` records, `modelGraph` plate diagrams, `ModelRunPlayer`, trace, rank and R-hat views, gallery recipe `diagrams/16-model-inference`. Files are named `model-run`, `chain-diagnostics` and `model-graph`, not `inference*`                      | `exposition/{model-run*,chain-diagnostics}.ts`, `render/src/{structures,views,react/views}/`, `apps/site/.../16-model-inference.tsx` |
| WP12 | partly done | `docs/structure.md` 4.32 and 4.32a, the changelog, this status. Open: concept pages and how-tos, the catalogue rows of `docs/methods-catalog/probabilistic-inference/` (corrected for the stale rows of verification.md 4)                                                    | this commit                                                                                                                    |
| WP13 | partly done | `apps/pocs` groups and the `probabilistic-programming` group exist (decision 8); the showcase pocs and the `examples/` programs of the plan (probit on two engines, eight schools four ways, a calibration check) are not made                                              | `apps/pocs/src/groups.ts`                                                                                                     |
| WP14 | not done    | stretch (neural SBI, INLA, BNN modules, probabilistic circuits, a text front end, decision diagrams)                                                                                                                                                                        | -                                                                                                                             |

Other facts that moved since the plan: `pyro-ppl` is installed in the `oracle-prob` group (so the Pyro class lists of
verification.md 7 can now be read from the package); a Gibbs sampler, the density and the guides were fixed for constrained
and invalid-parameter sites (`CHANGELOG.md`); the operator count is 26 in the model registry and 35 factor operators in
`message-passing`.

## Decisions (2026-11-10)

The user read the plan and said to execute it. What that settles:
1. **Execute.** Open questions 1 to 11 of section 8.3 are accepted as recommended: the builder producing a spec (D), the old API
   kept, Stan's block discipline without a text parser, `lib/message-passing/`, the release split 0.1.13 and 0.1.14 with the
   SVI breadth in 0.1.14, ABC and synthetic likelihood and SBC in 0.1.14 with neural SBI as stretch, `pyro-ppl` added to the
   `oracle-prob` group, about 35 operators plus numeric fallbacks, the proposed naming, and heavy validation as a script run on
   request (refined by decision 6).
2. **No single entry point and no automatic everything.** "It is absolutely fine to have a bespoke algorithm or methodology for
   each of the scenarios: EP would be different to SVI, would be different to NUTS, and so on. I don't expect there to be a single
   entry point and the system will figure all of that out." So WP3 is re-scoped: each method has its own `Space` of knobs and
   its own typed tape contract, with fine-grained steps (steppable, cancellable, resumable) and shared conventions; the
   registry-driven universal `infer` and the automatic decomposing planner are **not** goals. The existing planner (`plan`,
   `explainPlan`, `infer`) stays exactly as it is, and `hybrid` stays an explicit plan the user builds. A capability check is a
   per-method function that says why a method does not apply, not a selector (section 5.4).
3. **The browser budget is a parameter, and every long-running method gives anytime results.** The user selects a time and/or an
   iteration budget; there is no fixed bar (open question 12 is closed). Every method that can run for more than a moment must
   deliver a periodic snapshot of its current trace and posterior state while it runs, by default about once a second, so a page
   or session can plot the current state live; the contract also has a budget option, a snapshot cadence and cancellation
   (sections 5.4, 5.8, WP3 and its acceptance criteria). The library has no clock, so the time budget and the snapshot cadence are
   realised by the driver that owns one (verification.md 6.3).
4. **Lifting existing programs to specs (`fromProgram`, open question 13): deferred to 0.1.14; the builder is the main path.**
   What the question meant: a closure written with `sample` and `observe` is opaque to a program that wants to read its structure,
   so lifting would run it once under a trace and try to rebuild a spec from what the trace recorded. That is valid only if the
   program's structure does not depend on random values or on the data (the same sites, the same dimensions and the same
   dependencies on every run), and it is unsound otherwise.
5. **No proofs of concept are added without a request** (open question 10). The vertical slice's page waits for the user's
   request; an `examples/` program of the same figure is fine and does not need one.
6. **Heavy validation is periodic, not continuous.** A script and a `make` target that is **not** part of `make verify` or
   `make check`, re-run whenever the probabilistic-programming code changes, with committed result tables (section 6).
7. **Releases.** The incremental 0.1.13 versions are tagged locally only; no GitHub release is made until the
   probabilistic-programming work is complete. (0.1.12 was released as normal.)
8. **Pocs groups, scheduled for the end.** After all lanes are merged and gated: `apps/pocs` gains groups of pocs, and a
   probabilistic-programming group is filled with a set of simple, interesting showcase examples. This overrides decision 5 for
   that set (the user has asked for it) and runs only once everything else is done.
9. **The model description drives pictures.** The spec should be describable so that a visual representation connects directly to
   it: `toStructure` produces plate diagrams (nodes with roles, named and nested plates, observed shading, data and derived nodes),
   and inference tapes (message events, per-site marginals and samples, ELBO and diagnostics per site) address the same stable
   node ids, so a view can overlay live inference state on the picture. Lane A owns the spec side and lists the render gaps;
   WP11 gains a model-graph view (plate diagram with live annotations) in the render package.

## 1. Executive summary

**What the system will be.** One model, written once, run by whichever method the user chooses, each method its own algorithm with its own knobs.
- A **model layer** (`probprog/model/`): a typed builder in the shape of Stan's blocks and PyMC's named dimensions that produces a
  `ModelSpec`, a JSON-able graph of data, latent, observed, derived and factor nodes with named dimensions (plates), constraints
  and a small registry of operators. A spec compiles to (a) the existing `sample`/`observe` program, so every present engine and
  handler works at once, (b) a factor graph of exponential-family variables for message passing, (c) a static structure
  analysis, (d) a drawing, (e) JSON for workers and docs pages. Existing closure programs keep working and are first-class
  (lifting them to specs is deferred to 0.1.14).
- A **handler layer** extended with the missing Pyro/NumPyro handlers (`reparam`, `collapse` (delayed sampling), `lift`, `scope`,
  `infer_config`) and Gen-style trace operations (`regenerate`, `update`, `assess`) so users can write their own samplers,
  guides and proposals in a few lines.
- A **common method convention, not a single entry point**: every method (exact, message passing, variational, MCMC, SMC, nested,
  likelihood-free) is its own bespoke algorithm with its own `Space` of knobs and its own typed tape contract, written as a
  small-step, resumable, cancellable `Algorithm`, with a per-method `check` that says why it does or does not apply and an anytime
  `snapshot` of its current posterior for live plots. The user picks the method; EP, SVI and NUTS are different things. The
  twelve present engines are wrapped without rewriting them, and the existing planner is left as it is.
- **Composition** (`hybrid`): a plan object, **built by the user**, that assigns blocks of sites to methods (NUTS for these, an exact
  conditional for those, EP for this part, collapse the conjugate rest), in the manner of NIMBLE, Turing.jl and Bean Machine.
  Collapse of conjugate and linear-Gaussian structure is the structural leverage, offered as a handler the user applies.
- A **message-passing engine** (`lib/message-passing/`), the Infer.NET core: exponential-family messages in natural parameters,
  a registry of factor operators, schedules, and slots for the projection (moment matching, KL, alpha-divergence, quadrature,
  Laplace), the region graph (Bethe, Kikuchi), power and damping, stochastic EP. EP, VMP, BP, power EP, alpha-BP, SEP, fractional EP
  and the mixed forms are settings of one engine, not separate code.
- A **validation kernel** that proves engines work: grid and closed-form oracles, simulation-based calibration, Geweke tests, an
  agreement harness across engines, named benchmark models with truth, and NumPyro/BlackJAX/ArviZ/pgmpy/scikit-learn goldens
  where those exist.

**What changes and what does not.** The present probprog is not primitive as a NumPyro port: 12 engines, handlers, six auto guides, a
planner, criticism, golden tests against NumPyro. What a user meets as primitive is eleven specific things (inventory section 8):
a model is a closure, not a value; everything is a string name and a raw tensor; data binding is ad hoc; the user picks an engine
and cannot compose or edit inference; only MCMC has a `Space`; structure is used only when the whole program matches; there is no
message passing on programs; ten kernels are unreachable; the MCMC tape is one step; calibration is untested; and nothing shows any
of it. The plan addresses each except "the user chooses an engine, not an inference", which the user has said is fine as it is for
the selection (decision 2); what is added is control inside a method and explicit composition. It adds layers on top and
**replaces nothing** (the old API stays and the new layer compiles to it).

**Recommended order** (details in section 7).
1. Freeze three contracts (spec, method convention, factor operator) and build the **first vertical slice** (section 7.2): a probit
   regression written in the new builder, run by NUTS (through the compiled program) and by EP (through the new engine), compared with
   the exact grid answer, traced, and drawn by an `examples/` program (a proof of concept page only if the user asks). It proves the
   architecture end to end in about two weeks of one lane.
2. In parallel, in three lanes with disjoint files: (A) the model layer, then the method conventions (spaces, contracts, small
   steps, anytime snapshots) and `hybrid`; (B) the message-passing engine and its variations; (C) the handlers, then the validation
   kernel and the first benchmark models.
3. Release **0.1.13** is the architecture and the new capability: model layer, handlers, method conventions and composition, message
   passing with its variations, the validation kernel and six benchmark models.
4. Release **0.1.14** is breadth and structure: every kernel on programs, SVI extensions (TraceGraph, TraceEnum, structured
   guides, natural gradients, Pathfinder), parallel tempering and the newer samplers, particle MCMC and nested sampling,
   structure reports (not automatic decomposition), lifting programs to specs and diagnostic charts, the full benchmark suite and the SBC campaign.

The user suggested SVI in the first release. I put it second because SVI already works (12 engines, `svi`, six guides, four ELBOs):
what 0.1.13 needs from it is only the contract and `Space` wrapper (inside the method-convention work). The new SVI algorithms are in
0.1.14. The user accepted this (decision 1).

## 2. Inventory in brief

Full text: [inventory.md](inventory.md).

| area                | what is there                                                                                                                                                                                                                           | assessment                                                                                                                                                         |
| ------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| programs, handlers  | `sample`, `observe`, `param`, `deterministic`, `factor`, `plate`, `markov`; `trace`, `seed`, `condition`, `substitute`, `replay`, `block`, `scale`, `mask`, `doIntervention`, `enumerate`, `uncondition`, `handle`; `Messenger`, `send` | NumPyro-faithful and golden-tested; missing reparam, collapse, lift, scope, escape, equalize                                                                       |
| densities, analysis | `logJoint`, `potentialEnergy` with Jacobians, init strategies; `analyse` from one traced run                                                                                                                                            | strong bridge to samplers; analysis is per-run and not a model value                                                                                               |
| engines (12)        | enumeration, conjugate, gaussian, chain, nuts/hmc, gibbs, smc, importance, laplace, svi, em, abc; ranked by a planner with reasons                                                                                                      | broad; whole-program applicability; **only `nuts` has a typed contract and `serialise`**; seven are `Algorithm`s, eight are one-shot functions (verification.md 2) |
| guides, ELBOs       | autoDelta, autoNormal, MVN, low-rank, Laplace, IAF; Trace, mean-field, Renyi, IWAE                                                                                                                                                      | good NumPyro set; no structured, amortised, TraceGraph, TraceEnum                                                                                                  |
| message passing     | `pgm/inference`: discrete BP (four schedules, damping, Bethe), Gaussian BP, discrete EP and Gaussian-latent EP, discrete VMP, mean field, TRW; hand-written TrueSkill EP, GP-classification EP, CAVI Gaussian mixture, variational LDA  | strong in special cases; no general exponential-family engine; `dists/exponential-family.ts` is ready                                                              |
| kernels             | `sampling/`: RWM, adaptive, independence, MALA, ULA, SGLD, SVGD, slice, elliptical slice, HMC, NUTS, Riemannian HMC, tempered SMC, AIS; Stan windowed adaptation; R-hat, ESS, MCSE, HDI, BFMI                                           | strong; **ten kernels are unreachable from programs** (NUTS, HMC, and slice and tempered SMC in narrow roles, are reachable)                                       |
| render, browser     | `Algorithm`/`Method`/`Session`, `defineContract`, worker `producer` by method name with JSON config, `useJob`, message timelines and structure overlays, `import-safe` test                                                             | the machinery to show a message-passing run exists; a model closure cannot cross to a worker                                                                       |
| proof, demos        | goldens for density, handlers, ELBO, criticism, em, enumeration, laplace; closed-form checks for the rest (427 tests). One example (`examples/logistic`), no proof of concept, no diagnostic charts                                     | no SBC, no Geweke, no agreement harness; nothing shows the system                                                                                                  |

Measured (Node, three runs each, script `timings.ts`): eight-schools NUTS 2 chains x 600 iterations 2.9 s (2.4 ms per iteration);
logistic n = 500, d = 10 NUTS 1 x 400 1.5 s (3.8 ms); SVI 0.14 to 0.35 s for 500 to 1,000 steps. Usable in a page for models of this
size; the per-evaluation interpretive overhead is the cost to watch (section 5.8). The probprog test folder (20 files, 427 tests)
passes in 36 s.

## 3. The field in brief

Full text: [survey.md](survey.md) (sources tagged: [I] read in the installed package, [W] web, [M] memory; the version and status
facts were re-checked on 2026-11-10, see verification.md sections 7 and 8; Pyro, PyMC, TFP, Stan and Infer.NET are not installed,
so those rows are [W] or, for details not reachable, [M]).

| take from                               | the idea                                                                                                                                                                | lands in (plan section)        |
| --------------------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ------------------------------ |
| Stan (CmdStan 2.40)                     | blocks (data, parameters with constraints, transformed parameters, model, generated quantities); static check; Pathfinder, ADVI, Laplace                                | 5.1, algorithms C7             |
| PyMC                                    | named dimensions and coordinates, swappable data containers, `do`/`observe` model transforms, per-variable step methods                                                 | 5.1, 5.2, 5.4                  |
| Infer.NET                               | operator registry, exponential-family messages, schedule from the graph, ranges and gates, one model with EP, VMP and Gibbs                                             | 5.5                            |
| Pyro, NumPyro                           | handlers (reparam, collapse, lift), guides as programs, TraceGraph and TraceEnum ELBOs, structured autoguides (Pyro only; NumPyro has block Gibbs and ensemble kernels) | 5.3, algorithms C2, C3, D4, D6 |
| NIMBLE, Turing, Bean Machine (archived) | an editable table from nodes to inference methods, written by the user                                                                                                  | 5.4, 5.6                       |
| Gen.jl                                  | trace operations (`regenerate`, `update`, `assess`) as a toolkit for custom samplers                                                                                    | 5.3                            |
| Birch                                   | delayed sampling: automatic exact marginalisation of conjugate and linear-Gaussian parts inside a running program                                                       | 5.6, algorithms A5             |
| JAGS                                    | automatic sampler selection per node from its full conditional                                                                                                          | 5.6                            |
| TFP                                     | joint distributions as values; automatic structured surrogate posteriors (ASVI)                                                                                         | 5.1, algorithms C3             |
| BlackJAX                                | kernel contract (already followed), the missing kernels (MCLMC, SGHMC, Barker, nested slice sampling, Pathfinder)                                                       | algorithms D                   |
| WebPPL                                  | posteriors as values that feed later models; in-browser demos                                                                                                           | algorithms A6                  |
| Anglican                                | lightweight Metropolis-Hastings by trace regeneration; particle MCMC                                                                                                    | algorithms D3, E3              |
| Stan SBC, ArviZ                         | simulation-based calibration, rank and ECDF plots, power-scaling sensitivity                                                                                            | section 6                      |

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
  inference/        NEW   the method convention (`InferenceMethodKit`: Space, contract, check, snapshot), `runAnytime`, wrappers for the 12 engines, hybrid (the user's plan); the old `infer` stays in `probprog/infer.ts`
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

| option                                                                                                                            | natural in TS                       | inspectable and serialisable | crosses to a worker | structure for engines                  | control flow                         | cost to build           |
| --------------------------------------------------------------------------------------------------------------------------------- | ----------------------------------- | ---------------------------- | ------------------- | -------------------------------------- | ------------------------------------ | ----------------------- |
| A. Pyro-like closure with handlers (today)                                                                                        | yes, but tensor-function arithmetic | no                           | no                  | one traced run                         | anything (but analysed once)         | done                    |
| B. Declarative graph and factor DSL (Infer.NET, BUGS)                                                                             | verbose                             | yes                          | yes                 | complete and static                    | only ranges and gates                | large                   |
| C. Builder with typed variables, named dims, constraints, derived and generated blocks (PyMC/Stan)                                | yes with fluent methods             | yes (the builder makes data) | yes                 | complete and static                    | ranges, scans, gates; opaque escape  | medium                  |
| D. Hybrid: C produces a spec that compiles to A; A stays; A can be lifted to C when structure-stable (lifting deferred to 0.1.14) | yes                                 | yes for specs                | yes for specs       | static for specs, one run for programs | everything, with the structure label | medium, and incremental |

**Recommendation: D.** The user writes a spec with a builder; the spec compiles to the existing program, so the 12 engines, the
handlers, the guides and the criticism work on day one; it compiles to a factor graph so message passing works; it analyses
statically so each method's `check`, the structure report and the user's `hybrid` can see the structure. Closure programs remain for models whose control flow is not
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
- **Lifting a program** (`fromProgram`, **deferred to 0.1.14**, decision 4). A closure written with `sample` and `observe` is opaque:
  a program that wants its structure has to run it. Lifting would trace it once and try to rebuild a spec from the trace, which is
  only valid if the program's structure does not depend on random values or on the data; it would check that by re-tracing with
  perturbed draws and data and refuse otherwise. It would let the 30-odd existing test programs reach message passing without being
  rewritten, and it is risky (section 8.1). The builder is the main path; nothing in 0.1.13 depends on lifting.

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

### 5.4 Methods: a common convention, no single entry point

**What the user decided (decision 2).** Each scenario may have its own bespoke algorithm and methodology: EP is not SVI is not
NUTS, and nobody expects one `infer` that works out everything. So this section does not add a registry-driven universal
`infer`, an automatic decomposing planner or a selector. What it adds is what every method needs in order to be steppable, tunable,
watchable and honest about when it applies. The existing `plan`, `explainPlan` and `infer` of `probprog/infer.ts` are **kept as
they are** (their tests pass unchanged); `hybrid` (5.6) is a plan the user writes.

**The convention.** A method is an omai `Method` (`trace/session.ts`: `name`, `config` as a `DictSpace`, `make(config, key)` giving
an `Algorithm`, `contract`, `defaults`, `hooks`) plus three functions that live beside it, by convention and not through a registry
(an app lists the methods it offers in `methodRegistry([...])`, which already exists and registers nothing by import):

```ts
interface InferenceMethodKit<S, C extends Config, P extends string> {
  readonly method: Method<S, C, P> // knobs (Space), typed tape (contract), make() -> Algorithm
  readonly family:
    | 'exact'
    | 'message-passing'
    | 'variational'
    | 'mcmc'
    | 'smc'
    | 'nested'
    | 'likelihood-free'
    | 'composite'
  check(model: ModelAnalysis): Verdict // applies or not, and WHY NOT, what is missing, what would help; never a selector
  posterior(state: S): Posterior // the same Posterior as today, plus dims, provenance and evidence
  snapshot(state: S): Snapshot // the current posterior state, cheap and JSON-able (anytime results)
}
```

- **Its own Space and its own tape.** Each method declares exactly the knobs it has (`dict` of knobs, `union` with a mode where
  there are modes: the kernel slot of MCMC, the projection slot of message passing, the ELBO and guide of SVI) and a
  `defineContract` of typed paths (algorithms.md gives them per row). The existing engines without one gain them: `svi`, `gibbs`,
  `smc` (particle filter), `em`, `abc` (ABC-SMC) are `Algorithm`s already and get a contract and a `serialise`; `laplace`,
  `importance`, tempered SMC, `conjugate`, `gaussian`, `chain` and `enumeration` are one-shot functions with no `Algorithm` (verified,
  verification.md 2): they get a thin wrapper that is one step, and are not made steppable artificially.
- **Fine-grained steps, steppable, cancellable, resumable.** One step is the unit a page can show or cancel: one transition across the
  chains (MCMC; the present driver's whole-run `sample` step is replaced by a composition of the existing `windowAdaptationAlgorithm`
  and `samplerAlgorithm`, so no new kernel code is needed), one sweep or one message batch (message passing), one gradient step
  (SVI), one temperature or one filtering step (SMC), one live-point replacement (nested sampling). State is JSON plus tensors
  (`serialise`/`deserialise`; today only `nuts` has them), so `checkpoint`, `seek`, `fork` and resume of the trace library work.
  Step keys are derived from the run key and `t`, so any pause and resume reproduces the run.
- **Run control: budget, snapshot cadence, cancellation (decision 3).** Every method that can run for more than a moment has these:
  1. _A budget option_ in its Space: an iteration budget (`maxSteps`, in the method's own unit: iterations, sweeps, sweeps of
     messages, particles' time steps). A **time budget** is the user's choice too, but the library has no clock (`scripts/banned.ts`;
     `runFor` takes an injected `clock`), so it is realised by the driver: a small helper `runAnytime(session, { steps, ms, snapshotMs,
     clock, onSnapshot, signal })` (in `probprog/inference/`, clock injected like `runFor`'s; in a worker the clock is the producer's)
     that slices the run with `runFor`, stops at whichever budget is reached first and reports each snapshot.
  2. _A snapshot_: `snapshot(state)` returns the current trace and posterior state in a form a page can plot at once: chains so far
     with running means, sds, R-hat and ESS (MCMC); the guide's parameters with their mean and sd and the ELBO so far (SVI); the
     current marginals and the largest message change (message passing); the weighted particles' summary and the log evidence (SMC);
     the live points and the evidence so far (nested); a single final snapshot for the one-shot exact methods. It costs no more than
     about one step, is JSON, and is also a derived path of the contract (`snapshot/state`) and the step's `frame`, so it rides the
     tape to the page through the producer's tape deltas. **The default cadence is about once a second**: `snapshotMs` 1000 for the
     driver, or, in step terms, a `snapshotEvery` knob (a number of steps) where the page paces itself by steps.
  3. _Cancellation_: between any two steps (a stopped run keeps its tape and its last state and can be resumed or forked); latency is
     one step, so a step must be small enough that one step is well under a second on a browser-size model (an acceptance test: a
     cancelled MCMC run stops within one iteration, a cancelled message-passing run within one sweep).
- **Diagnostics are part of the posterior.** `posterior.diagnostics` has, by form: chains (R-hat, bulk and tail ESS, divergences,
  BFMI, tree-depth hits); variational (ELBO curve, the PSIS k-hat of the importance weights, as Yao et al. 2018 [M]); message passing
  (converged, last change, free energy, improper-message count, schedule); particle (ESS history, evidence); nested (logZ error).
- **The capability check is per method and explains, it does not select.** `check(analysis)` returns a `Verdict` (applies, the
  estimated cost, the reason, the features missing and what would help). The model analysis is a set of features: `continuous
  latents`, `differentiable`, `finite discrete latents` (with joint size and treewidth), `unbounded discrete`, `conjugate sites`,
  `linear-Gaussian sub-network`, `exponential-family factor graph`, `opaque nodes`, `markov structure`, `has likelihood` (versus
  simulator only). A method calls `check` on itself before it runs and throws the verdict's reason as its error; a page can call
  it to grey out a method with the reason ("EP needs an exponential-family factor graph: node `f` is opaque"). The existing
  `InferenceEngine.applies` is the same idea for the present planner and stays.
- **No automatic fallback across families.** The planner's fall-through to the next engine on failure stays as it is, for `infer`
  with `engine: 'auto'`; the new methods do not chain themselves.

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

**How structure is used.** Each row is something a user can apply by hand (a handler, a method choice, a `hybrid` block), and
what `analyseSpec(bound).explain()` reports when it finds the structure ("these three sites are conjugate and could be collapsed").
None of them is applied automatically: the existing planner is not extended into a decomposer (decision 2).

| structure found                                         | exploitation                                                                                  | row        |
| ------------------------------------------------------- | --------------------------------------------------------------------------------------------- | ---------- |
| all latents conjugate                                   | closed form                                                                                   | A2         |
| conjugate or Gaussian sub-network inside a bigger model | `collapse` it (Rao-Blackwellise): run the sampler on what remains; draw the rest by `recover` | A5, D4     |
| low-treewidth discrete part                             | enumerate or junction tree, summed out inside NUTS (exists), exact conditionals inside Gibbs  | A1, D4     |
| tree or chain factor graph                              | tree schedule BP is exact; HMM and Kalman special forms                                       | A3, A4, B1 |
| small Markov blankets                                   | blanket-local single-site updates (Bean Machine, JAGS) re-evaluate only the blanket           | D3, D4     |
| plates of exchangeable data                             | vectorise; stochastic EP and minibatch SVI per plate                                          | B5, C1, C5 |
| hierarchical funnel geometry                            | the report suggests non-centring; the user applies `reparam`                                  | D8         |

**Mixing message passing, VI and sampling.** The mixes that earn their place, each a `hybrid` pattern:

| pattern                                                  | how                                                                                                                                                              | gain                                                           |
| -------------------------------------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------- | -------------------------------------------------------------- |
| collapse then sample                                     | exact marginalisation of the linear-Gaussian or conjugate block (a message-passing or closed-form computation), NUTS on the remainder                            | Rao-Blackwellisation; fewer dimensions; no divergences from it |
| EP or VMP on the likelihood part, sample hyperparameters | EP gives the site approximations for the latent field; the hyperparameters are sampled or gridded with the EP evidence (EP-in-the-loop, as in GP classification) | the INLA/GP pattern at modest cost                             |
| approximation as warm start                              | Laplace, Pathfinder, EP or VMP result sets the initial point, the mass matrix or the proposal of a later sampler                                                 | faster, more reliable warm-up                                  |
| approximation as proposal                                | EP or VMP posterior as an importance or SMC proposal, with the exact weights                                                                                     | asymptotic exactness with a good start                         |
| local exact, global stochastic                           | SVI on global parameters with exact local updates by message passing (natural-gradient SVI)                                                                      | scales to large data; LDA-style models                         |
| exact discrete, gradient continuous                      | enumerate discrete sites inside NUTS (today), or HMC within Gibbs for discrete sites that cannot be enumerated                                                   | mixed models                                                   |

`hybrid` is a plan **written by the user** (`hybrid(bound, [{ sites, use }, ...])`) and executes blocks as a Gibbs-style cycle: each
block's method targets the block's conditional given the others (the block target of 5.3). Methods that are not kernels (EP, VMP, SVI,
Laplace) enter through the other patterns (collapse, warm start, proposal), not as blocks; each method's `check` says why it cannot
take a given role. `hybrid` is itself a method in the convention of 5.4: its own `Space` (the sweep count and order), its own
tape (`hybrid/block`, per-block acceptance), and the same budget, snapshot and cancellation.

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

| concern                       | position                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                           |
| ----------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| compute budget                | measured (three runs, `timings.ts`): 2.4 to 3.8 ms per NUTS iteration on 10 to 20 latent dims and up to 500 rows in Node (a browser may be 1.5 to 3 times slower, not measured). **The budget is a parameter the user selects, a time and/or an iteration budget, not a fixed bar (decision 3)**: every long-running method takes a step budget and the driver takes a time budget (the library has no clock; 5.4), and delivers anytime snapshots about once a second so a page can plot the current state while it runs, with a visible progress indicator and cancellation. Models above several thousand rows per gradient need the compiled plan below, or minibatch SVI, or message passing. Whether the compiled plan is in 0.1.14 is decided by a measurement in the browser, not by a bar |
| interpretive overhead         | today every log-density evaluation re-runs the closure through the handler stack and the autodiff transform. A spec makes a **compiled evaluation plan** possible: a topologically sorted list of operator calls over preallocated `Float64Array`s with hand or traced gradients; it is an optimisation behind the same `LogDensity`, justified only when a benchmark shows the closure path is the bottleneck. It is in 0.1.14 as a measured work item, not an assumption                                                                                                                                                                                                                                                                                                                         |
| workers                       | what crosses to a worker must be JSON: **a spec is, a closure is not**. Methods are made in a worker by name and configuration (the producer protocol exists); but **`Method.config` accepts only `SpaceValue` knobs, so a spec cannot ride in it today** (verification.md 6.3): WP0 decides between a model registry by name in the worker, a JSON `constructables` field in the wire options, or a method made per spec. Chains and particles can run on several workers with keys folded by chain index so results are identical however many workers run (determinism is part of the contract and a test)                                                                                                                                                                                      |
| progress and cancellation     | steps are small (5.4); `runFor` slices work with an injected clock; the worker host paces slices (20 ms) and posts tape deltas (every 40 ms); `useJob` stages and `workerStage` exist. The present MCMC driver's whole-run `sample` step is the one thing to change, by composing the existing `samplerAlgorithm` and `windowAdaptationAlgorithm`. **Anytime snapshots** (5.4) ride the tape as a derived path and a frame, read by the driver about once a second                                                                                                                                                                                                                                                                                                                                 |
| determinism                   | every method derives keys by name from the run key; the same seed gives the same chains on any machine and any number of workers                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                   |
| bundle size and import safety | `sideEffects: false`, one entry per folder. **The operator registry must not register by import side effect**: operators are exported values collected by an explicit `defaultOperators()` so a page that imports only EP-Gaussian does not pull the rest; `scripts/import-safe.test.ts` covers new entries automatically. New entry points (`omai-engine/message-passing`, `omai-engine/probprog/model`, `.../inference`, `.../validation`) are added to `omai-meta/modules`                                                                                                                                                                                                                                                                                                                      |
| precision                     | `Float64Array` throughout; WebGPU is f32-only and is not a path for exact gradients                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                |
| what must be tiny             | model construction, analysis, planning, message passing sweeps on plated models, Laplace, small SVI, conjugate and enumeration: all interactive                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                    |
| what may be heavy             | NUTS on thousands of rows, SMC with thousands of particles, nested sampling beyond a few dimensions, SBC campaigns, neural SBI: with progress, a worker, and a "demo size" default                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                 |

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
| calibration            | `sbc(bound-generator, method, { replications })`: ranks of the true parameter among posterior draws; uniformity by a chi-square test on the histogram and the ECDF-difference envelope; a deliberately biased method (a too-narrow guide, a wrong Jacobian) **must fail**, which is the test that the test can detect a bug                                                                                                                                                                                                                                                                                                                           | Talts et al. 2018 [M]; exact rank distribution; NumPyro `vsbc_diagnostic` for guides                                                                                             | `probprog/validation/sbc.ts`                       |
| predictive             | posterior predictive checks, PSIS-LOO and WAIC (exist, ArviZ goldens), power-scaling sensitivity (new, cheap)                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                         | ArviZ 1.3.0 (installed)                                                                                                                                                          | `probprog/criticism.ts`                            |
| browser                | import-safe for new entries; worker and main thread give identical results for a seed; a cancelled run stops within one step; the compiled plan equals the closure path to 1e-12                                                                                                                                                                                                                                                                                                                                                                                                                                                                      | `scripts/import-safe.test.ts`; producer tests                                                                                                                                    | `render` and `probprog` tests                      |

**Flakiness policy.** Statistical tests use fixed seeds and budgets set once from a documented calculation (the rank test's alpha and
the replications), so a failure means a change, not luck; the unit-test budget is seconds.

**Heavy validation is periodic, not continuous (decision 6).** The heavy campaigns (SBC with several hundred replications across all
methods, the full agreement matrix, the benchmark suite at full size, the timings) are a script, `scripts/probprog-validate.ts`, run
through a `make` target, `make probprog-validate` (optionally `ARGS="--only sbc"` or `ARGS="--only agreement/eight-schools"`), that
is **not** part of `make verify` or `make check`. It is re-run whenever the probabilistic-programming code changes (the sampling,
probprog and message-passing folders), by the main session or on request, and writes **committed result tables** to
`docs/probprog/results/` (one markdown table per campaign, with the commit it was run on, the seeds, the replications and the
time), so a regression is a diff of the tables. A table that is stale against `git log` of those folders is reported by the script
(`--check` prints which campaigns need a re-run) but never fails a gate. The unit-level statistical tests (a few replications, fixed
seeds, seconds) stay in the ordinary test files and in `make verify`. The timings script `docs/probprog/timings.ts` is the first member
of this family.

**Reuse, from verification.md 6.6.** The published `probprog/validation/` code returns numbers and records and imports only library
code: `sampling/diagnostics` (`rhat`, `ess`, `mcse`, `bfmi`, `diagnose`), `stats/tests` (`ks2samp`, `kstest`, `chisquare`,
`wassersteinDistance`, `energyDistance`), `dists/divergences`, and `numerics/quadrature` (`nquad`, `productGaussRule`, `sparseGridRule`)
for the grid oracle. The test-only helpers in `src/testing` (`expectPosteriorMoments`, `expectInvariant` (a ready one-step invariance
test for any kernel), `expectUniformPValues`, `nullPValues`, `expectDistribution`, `coverage`) wrap those in assertions, import vitest,
and are not importable from `lib/`.

**Python oracles.** Installed and used: NumPyro 0.22.0 (programs, ELBO, reparam, `lift`, chains' statistics, the new Gibbs and
ensemble kernels), BlackJAX 1.7.1 (kernels, Pathfinder, SMC, adaptation, nested slice sampling `nss`), ArviZ 1.3.0 (diagnostics,
LOO, `psense`, `bayes_factor`), pgmpy 1.1.2 and pyAgrum 3.3.0 (discrete exact and BP), scikit-learn 1.9.1 (VB mixture, LDA),
hmmlearn, filterpy, scipy. **Not installed**: Pyro, PyMC, TensorFlow Probability, funsor, any Stan, emcee, dynesty, UltraNest, jaxns,
sbi. **Decision 1: add `pyro-ppl` (1.9.2, 2026-10-03, maintained) to `oracle-prob`**; it depends on torch, which is installed for
`oracle-torch`, and gives `TraceGraph_ELBO`, `TraceEnum_ELBO`, `AutoStructured`, `AutoGaussian`, `escape`, `equalize` and the
`poutine` semantics as oracles (`collapse` and `TraceEnum` over funsor would also need `funsor`, 0.4.8; not recommended). Nothing else
is added; Stan and PyMC reference results come from closed forms and from our own long NUTS references. **SBC has no installed
oracle** (ArviZ only plots ranks computed elsewhere; NumPyro has `vsbc_diagnostic` for guides): its oracle is the exact rank
distribution (uniform on 0 to L) and the deliberately biased method that must fail.

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

By the repository's rules (CLAUDE.md) and decision 5: **proofs of concept and gallery recipes are not added unasked.** The user has
not asked for any yet, so none is in the work packages (WP13 waits for a request). The list is what would show the system, for the
user to ask from; programs under `examples/` do not need a request.
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

| WP   | work                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                      | size | depends                                                                     | owns                                                                                                               | acceptance                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                | browser-demonstrable                      |
| ---- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ---- | --------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------ | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ----------------------------------------- |
| WP0  | Contracts and structure.md: `ModelSpec` JSON schema, the method convention of 5.4 (`InferenceMethodKit`, snapshot, run control), `FactorOperator`, the new folders, and **the decision how a spec reaches a worker** (`Method.config` cannot carry it today: a model registry by name, a JSON `constructables` field, or a method per spec; verification.md 6.3); type stubs and TSDoc only                                                                                                                                                                                                                                                                                                                                                                                                                                                                               | S    | -                                                                           | `probprog/model/spec.ts`, `probprog/inference/method.ts`, `message-passing/types.ts`, `docs/structure.md` 4.32/new | reviewed by the user; `tsc` clean; no behaviour                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                           | -                                         |
| WP1  | Model layer: builder, ops registry (30 ops), named dims, constraints, data binding (`bind`, `withData`, missing, batch, `bindDataset`), `toProgram`, JSON, `analyseSpec`, `toStructure`, `include`, `condition`, `do`                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                     | L    | WP0                                                                         | `probprog/model/**` (except `factors.ts`)                                                                          | spec and program agree on log joint for the shared `MODELS`; JSON round trip; every existing engine runs a bound spec; structure drawn                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                    | yes (spec to drawing to NUTS)             |
| WP2  | Handlers: `reparam` (+ reparameterisers), `collapse` (delayed sampling), `lift`, `scope`, `escape`, `equalize` (Pyro-only, lowest priority), `mean`, `inferHint`; trace operations `regenerate`, `update`, `assess`; `defineKernel` block target                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                          | M-L  | -                                                                           | `probprog/handlers.ts`, `reparam.ts`, `collapse.ts`, `lift.ts`, `trace-ops.ts`, `probprog/inference/kernel.ts`     | laws and goldens of section 6; funnel divergences vanish under non-centring; collapse equals closed form                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                  | yes (funnel)                              |
| WP3  | **Method conventions, no single entry point** (decision 2): for each present engine that is an `Algorithm` (`svi`, `gibbs`, particle-filter `smc`, `em`, `abc`-SMC) a `Space` of knobs, a `defineContract` tape and `serialise`; for the one-shot ones (`laplace`, `importance`, tempered SMC, `conjugate`, `gaussian`, `chain`, `enumeration`) a `Space`, a contract and a one-step wrapper; fine-grained MCMC (one transition per step, composed from `samplerAlgorithm` and `windowAdaptationAlgorithm`, replacing the single `sample` step); per-method `check` with reasons; `snapshot` and run control (budget knob, snapshot cadence, cancellation) for every method, with the `runAnytime` driver helper (injected clock). The existing `plan`, `explainPlan`, `infer` and their tests are unchanged; **no registry-driven universal `infer`, no automatic plan** | M-L  | WP0, WP1 (the method conventions need only WP0; the spec adapters need WP1) | `probprog/inference/{method,check,snapshot,anytime}.ts`, `probprog/engines/*.ts` (all; `infer.ts` untouched)       | old `infer` tests pass unchanged; every wrapped method runs as a session with automatic controls and its own typed contract; **run control: a cancelled MCMC run stops within one iteration and a cancelled message-passing run within one sweep; a budgeted run (steps, or `ms` with an injected clock) stops at the budget and a resumed run equals an uninterrupted one (same keys); `snapshot(state)` is JSON, costs about one step, and a run driven by `runAnytime` with `snapshotMs` 1000 and a fake ticking clock reports a snapshot per second of clock time, each equal to the final posterior's form; `check` returns a reason for every non-applicable method on the shared `MODELS`**; worker and main agree | yes (any method by name in a worker)      |
| WP4a | Message-passing core: messages, operator registry, the 35 operators, engine with `projection`, `schedule`, `damping`, `improper`; EP, VMP, BP as settings; tape and events                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                | L    | WP0                                                                         | `message-passing/**` (except `regions.ts`, `power.ts`, `stochastic.ts`)                                            | each operator equals numerical integration; EP exact on Gaussian trees; reproduces `pgm` BP, EP and VMP, TrueSkill, probit EP; VMP matches sklearn BGMM; **run control as WP3: the engine's steps are single sweeps, it takes a sweep budget, serialises its messages and gives a per-sweep snapshot of the marginals**                                                                                                                                                                                                                                                                                                                                                                                                   | yes (messages converge on a factor graph) |
| WP4b | Variations: power and fractional EP, alpha-divergence (for the general engine: the Gaussian-latent case, with damping and Gauss-Hermite sites, already exists in `pgm/inference/ep.ts` and is the reference), stochastic EP, region graphs (Bethe, Kikuchi, GBP), expectation consistent, numerical projections (quadrature, Laplace, sample), AMP                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                        | L    | WP4a                                                                        | `message-passing/{regions,power,stochastic,numerical,amp}.ts`                                                      | alpha = 1 is EP, alpha sweep interpolates; Kikuchi beats Bethe on a grid against the junction tree; SEP memory independent of n                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                           | yes (alpha slider)                        |
| WP4c | Compile spec to factor graph; EP/VMP as method adapters; collapse using message passing; EP posterior as warm start and proposal                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                          | M    | WP1, WP4a, WP3                                                              | `probprog/model/factors.ts`, `probprog/inference/mp.ts`                                                            | a bound spec runs on EP and VMP when the user chooses them, and each method's `check` explains why it applies or not (no automatic choice)                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                | yes                                       |
| WP5  | Validation kernel: grid and quadrature oracle, exact helpers, `agree`, SBC, Geweke, distances (1-D Wasserstein, KS, KL on grids), runner; oracle cases for new goldens; the periodic heavy-validation script `scripts/probprog-validate.ts` and its result tables (decision 6; the `make probprog-validate` target is added by the main session, outside `make verify` and `make check`)                                                                                                                                                                                                                                                                                                                                                                                                                                                                                  | M    | WP0 (programs only; WP1 helps)                                              | `probprog/validation/**`, `oracle/cases/probprog/**`, `src/testing/probprog.ts`, `fixtures/probprog/**`            | SBC passes for NUTS on a conjugate model and fails for a deliberately wrong guide; agreement table for the slice                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                          | yes (rank histograms)                     |
| WP6  | Benchmark models as specs with truth (first set 0.1.13, rest 0.1.14); registered with tags like datasets                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                  | M    | WP1, WP5                                                                    | `probprog/models/**`                                                                                               | each model has truth, references and a smoke test                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                         | yes                                       |
| WP7  | SVI breadth: `TraceGraph_ELBO`, `TraceEnum_ELBO`, structured and hierarchical guides, amortised guides, natural-gradient SVI, Pathfinder, control variates, ADVI step sequence                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                            | L    | WP3, WP4a (natural gradient)                                                | `probprog/elbo*.ts`, `guides*.ts`, `engines/svi.ts`, `probprog/engines/pathfinder.ts`                              | estimator unbiasedness and variance; Pathfinder against BlackJAX; natural gradient converges to CAVI                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                      | yes                                       |
| WP8  | MCMC breadth: `kernel:` slot for every `sampling` kernel on specs and programs; blocks in `hybrid`; HMC within Gibbs, MixedHMC; parallel tempering; MCLMC, SGHMC, Barker, ensemble; adaptation (CHEES, low-rank)                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                          | L    | WP2, WP3, WP5                                                               | `sampling/**` (new files), `probprog/engines/nuts.ts`, `mcmc-options.ts`, `probprog/inference/hybrid.ts`           | BlackJAX and NumPyro statistics; PT recovers mode weights; every kernel passes Geweke and SBC                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                             | yes                                       |
| WP9  | Structure reports: `analyseSpec` conjugacy through plates and linear maps, blanket and treewidth analysis, `explain()` of what could be collapsed or enumerated and by which method, `collapse` planning as a handler the user applies, compiled evaluation plan (if measured in the browser). **No automatic decomposition in `plan`**                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                   | L    | WP1 to WP4                                                                  | `probprog/model/analysis*.ts`, `probprog/inference/{plan,decompose}.ts`, `probprog/model/compile-plan.ts`          | `analyseSpec(bound).explain()` finds the collapsible Gaussian block of the radon model and says so; a user-built `hybrid` of collapse plus NUTS is faster per ESS than plain NUTS; the explanations are correct                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                           | yes                                       |
| WP10 | SMC and evidence: guided proposals, PMCMC, nested sampling, evidence estimators slot                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                      | M-L  | WP3, WP5                                                                    | `sampling/{nested,pmcmc}.ts`, `probprog/engines/smc.ts`                                                            | evidence unbiasedness; nested sampling analytic Gaussian evidence                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                         | yes                                       |
| WP11 | Exposition records and views, MCMC diagnostic charts, gallery recipes for new charts only                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                 | M-L  | WP3, WP5                                                                    | `exposition/inference*.ts`, `render/src/views/inference*.ts`, `render/src/charts/mcmc*.ts`, gallery recipes        | contract and smoke tests; thumbnails regenerated (`make thumbs`)                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                          | yes                                       |
| WP12 | Docs: concept pages, TSDoc examples, `@index` tags, `make agent-index`, catalogue rows, structure.md                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                      | M    | all                                                                         | `docs/**` (main session)                                                                                           | `make verify` passes, the docs catalogue builds                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                           | -                                         |
| WP13 | Demonstrations: `examples/` programs (no request needed); pocs only if the user asks (decision 5)                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                         | M    | a user request (pocs)                                                       | `apps/pocs/src/pocs/<slug>/`, `examples/<slug>/`                                                                   | the user asked for them                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                   | yes                                       |
| WP14 | Stretch: neural SBI, INLA, BNN modules, probabilistic circuits, a Stan-like text front end, decision-diagram exact inference                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                              | L    | later                                                                       | -                                                                                                                  | -                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                         | -                                         |

**Acceptance criteria that apply to every method work package (WP3, WP4a, WP4b, WP7, WP8, WP10), from decisions 2 and 3.** (1) The method has its
own `Space` and `defineContract` tape and no entry point other than its own. (2) A step is small: a cancelled run stops within one step, and a
resumed run equals an uninterrupted one. (3) It has a budget knob (iterations, sweeps or time steps) and works under `runAnytime` with an
injected clock for a time budget. (4) `snapshot(state)` gives the current trace and posterior state, JSON, at about the cost of a step,
and a driver reading it every second of clock time gets a usable plot at each reading (a unit test with a ticking clock; the cadence is
not a property of the algorithm). (5) `check` explains non-applicability. The three browser-facing items are tested with the producer's
fake wire (`producer/_fake.ts`, `tickingClock`), not with a browser.

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
3. `message-passing` with the Gaussian family, the `matmul-constant`, `Gaussian prior` and `probit` operators, the EP projection and the flooding schedule; `toFactorGraph` for this spec. _Accept:_ EP's marginals within 0.05 posterior sd of the grid answer for the means on this data; the engine reproduces `gp/approximations` probit EP when the model is a GP-classification (a shared test of the same operator) and `pgm/inference/ep.ts` `gaussianEp` with a probit site (which already has power, damping and quadrature sites).
4. `validation/grid.ts`: adaptive grid posterior in three dimensions with evidence; `agree` over NUTS, EP and Laplace. _Accept:_ NUTS within 3 Monte Carlo standard errors; Laplace and EP reported with KL and sd-errors; EP closer than Laplace on at least the intercept (the expected result; if it is not, that is a finding).
5. The tape: EP emits `message` events and per-sweep marginals; NUTS emits per-iteration scalars. _Accept:_ both run through the producer in a worker by name with the spec reaching the worker by the route WP0 chooses (verification.md 6.3: `Method.config` cannot carry it as it is), and the main-thread result equals the worker's; both give an anytime snapshot every second of an injected clock (decision 3).
6. One view: the spec drawn as a factor graph, with the three posteriors overlaid per weight and the EP change curve. By decision 5 it is an `examples/probit-two-engines/` program (a figure and a tape, for the front page); the page in `apps/pocs/src/pocs/probit-two-engines/` waits for the user's request.

### 7.3 Lanes and releases

Three agents, each in its own worktree (`agent-<id>-<word>`), merged in order; lanes touch disjoint files (the "owns" column).

| lane | 0.1.13 order              | 0.1.14 order                                        |
| ---- | ------------------------- | --------------------------------------------------- |
| A    | WP0, WP1, WP3, WP4c       | WP9, WP8's `hybrid`, WP11, `fromProgram` (deferred) |
| B    | WP4a, WP4b                | WP7 natural-gradient SVI, B8, WP10                  |
| C    | WP2, WP5, WP6 (first set) | WP8 kernels, WP6 rest, SBC campaign, WP7 estimators |

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

| risk                                                                                | likelihood | what to do                                                                                                                                                                                  |
| ----------------------------------------------------------------------------------- | ---------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| the DSL becomes a second language to maintain, with a growing operator list         | medium     | closed registry, `custom` escape hatch, every operator has a tensor evaluation first and message operators only where there is a closed form; cap the list at the first release's           |
| spec and program semantics drift                                                    | medium     | the equivalence test on every benchmark model in CI; the spec compiles to the program, there is no second evaluator (except the optional compiled plan, which has its own equality test)    |
| EP numerical fragility (negative variances, oscillation, non-convergence)           | high       | `improper` slot with repair strategies and counts on the tape; damping defaults; `check` warns for known hard factors; tests include a failing case and its repair                          |
| statistical tests that flake or run long                                            | medium     | fixed seeds and budgets, an SBC rank test with a documented alpha, a deliberate-failure test, heavy campaigns as a script                                                                   |
| closure interpretive overhead limits browser model size                             | medium     | measure first, in a browser (WP9); compiled plan if justified; message passing and minibatch SVI as the scalable paths; the user's time budget and anytime snapshots keep a slow run usable |
| lifting programs to specs is unsound for non-structure-stable programs              | medium     | deferred to 0.1.14 (decision 4); then optional, verified by re-tracing with perturbed values, refuses otherwise; the builder is the main path                                               |
| scope creep from the repertoire (60 rows)                                           | high       | the seven machines and the slots; the cut line in 7.3; stretch items marked                                                                                                                 |
| API churn for users of the present probprog                                         | low        | nothing is removed; `infer(model, data)` keeps working                                                                                                                                      |
| generated files and shared docs cause merge conflicts between lanes                 | medium     | lane agents never edit barrels, `llms.txt`, `api-surface.txt`, `CHANGELOG.md` or `structure.md`; the main session regenerates and merges                                                    |
| a new domain `message-passing` interacts with `make layers` and `omai-meta/modules` | low        | create it in WP0 with the layering check and the modules entry                                                                                                                              |
| the validation reference (a long NUTS run) is itself wrong                          | low        | prefer closed form and grid; where a NUTS reference is used it is checked against NumPyro's chain moments                                                                                   |

### 8.2 Non-goals

Stan language compatibility or its C++ toolchain; universal probabilistic programs with unbounded stochastic recursion (the existing
`importance` and `smc` engines on `markov` programs cover the useful part); GPU or WebAssembly kernels; code generation; trans-dimensional
(reversible jump) inference; full INLA with sparse precision in these two releases; neural SBI in these two releases (listed as
stretch); probabilistic circuits and decision-diagram exact inference in these two releases; automatic differentiation changes;
distributed (multi-machine) EP.

### 8.3 The open questions and how they were settled

All thirteen were answered on 2026-11-10 (the Decisions section at the top).

| #   | question                             | answer                                                                                                                                                           |
| --- | ------------------------------------ | ---------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| 1   | model-definition style               | accepted as recommended: the builder producing a spec (D), closures kept                                                                                         |
| 2   | keep or replace the old probprog API | keep it entirely; `infer` and the planner stay as they are                                                                                                       |
| 3   | how much of Stan                     | the block discipline, constraints with Jacobians, derived and generated quantities; no text parser                                                               |
| 4   | home of message passing              | `lib/message-passing/`                                                                                                                                           |
| 5   | release split                        | 0.1.13 architecture and message passing, 0.1.14 breadth, SVI additions in 0.1.14                                                                                 |
| 6   | scope of simulation-based inference  | ABC, synthetic likelihood and SBC in 0.1.14; neural SBI as stretch                                                                                               |
| 7   | oracle environment                   | add `pyro-ppl` to `oracle-prob`, nothing else (the main session edits `pyproject.toml` and `uv.lock`)                                                            |
| 8   | operator set                         | about 35 operators plus numeric fallbacks                                                                                                                        |
| 9   | naming                               | as proposed (`defineModel`, `latent`/`observe`/`derived`/`generated`, `bind`, `hybrid`, `message-passing`; the method convention is `InferenceMethodKit` in 5.4) |
| 10  | proofs of concept                    | none added without a request; the slice's page waits; `examples/` programs are fine                                                                              |
| 11  | heavy validation                     | periodic, a script and a `make` target outside `make verify` and `make check`, committed result tables (section 6)                                               |
| 12  | browser budget                       | not a fixed bar: a parameter the user selects, with anytime snapshots from every long-running method (5.4, 5.8)                                                  |
| 13  | lifting programs to specs            | deferred to 0.1.14; the builder is the main path                                                                                                                 |

### 8.4 What was not verified

Verified on 2026-11-10 and no longer open: the larger engines' bodies, the methods catalogue's status column, `rsample` for gamma,
beta and Dirichlet, the engine tests (they pass: 427 tests in 36 s), the two timings (re-measured, three runs), the NumPyro, BlackJAX
and ArviZ class lists (read in the installed packages), and the version and status of the other systems (web). What remains, in
full, is in [verification.md](verification.md) section 10; the main items are:
- Browser speed (no browser run); the 1.5 to 3 times slower than Node figure is still a guess.
- Pyro's MCMC page and the tail of its inference page, PyMC's `nuts_sampler=` option, Infer.NET's exact API signatures, WebPPL's
  browser capability, Anglican's and Birch's details: web pages that could not be reached or do not say.
- Literature claims in the tables (clutter problem results, k-hat, SBC theory, delayed sampling): from memory, none load-bearing for code.
- Sizes and the lane lengths are estimates in the repository's S/M/L scale, not commitments; sizes of the message-passing operators
  are the least certain.
