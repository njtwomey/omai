# Changelog

Versions are released in lockstep for omai-engine, omai-render and omai-meta. Up to 0.1.3 the packages were published as
mdml-engine, mdml-render and mdml-meta.

## 0.1.13

Not yet released on GitHub (no tag has been made and the packages are still 0.1.12): the next GitHub release will cover the
whole probabilistic-programming increment recorded here, under whichever version number it is given. This section records what
is on `main` so far; the plan is `docs/probprog/PLAN.md` (with its status section), the validation tables are in
`docs/probprog/results/`.

Probit classification (omai-engine, `omai-engine/probprog/models`): `probitClassifier(priorScale)` (the probit regression for a design of
any width), `polynomialFeatureMap` (the inputs standardised, every monomial up to a degree, each column standardised, an intercept
first, divided by a `weightScale` so the linear predictor is of order one at the starting weights), `probitPredictive` and
`probitPredictiveGrid` (the posterior predictive from draws of the weights), `heldOutScores`. The `probit` operator of the model
layer now keeps its value in (0, 1) (`1e-300` to `1 - eps/2`): a Bernoulli on it had an infinite logit where the normal CDF rounds
to one, and a gradient method (a variational guide, a leapfrog step) that wandered there ended in a not-a-number gradient. The
proof of concept `probit-lab` (Probit regression lab) runs the methods ticked in Web Workers within a time budget and replays
their frames on one playhead.

Probabilistic programming (omai-engine). The design: a model is written once as a value (a spec), every method is its own
thing with its own knobs and tape, the user chooses (no universal `infer`, no automatic decomposition; the planner `plan`,
`explainPlan` and `infer` is unchanged), and every long-running method gives anytime results.

Model layer (`omai-engine/probprog/model`):
- `defineModel`: a builder with named dimensions (`range`), `data`, `latent` (with constraints), `observe`, `derived`,
  `generated`, `factor`, `param`, `custom` and `include`, producing a JSON-able `ModelSpec` (`Var` handles and `ops`,
  checked as the model is written). A closed registry of 26 operators (`defaultOperators`) and 22 distribution families as
  data (`model/dists`, mapped to the constructors of `omai-engine/dists`). The `gate` node (mixtures) is reserved and not built.
- `scan`, the recurrence node: `m.scan(name, { over, init, step, emit })` writes a time series, a state-space model or a
  hidden Markov chain once (the step reads the previous state, the emissions are observed at each step; one scan is one
  series). It compiles to one `markov` loop (the states are the sites `<scan>[t]`, the emissions `<emission>[t]`), is checked,
  drawn (one node in the plate of its dimension with a `t-1` self-loop), identified (`ids.ts`), analysed (`analyseSpec().scans`,
  `scanLinearGaussian`, a Gaussian chain as one linear-Gaussian block, a structure-report suggestion that names `chain` or the
  particle filter) and read by the chain and particle-filter methods; `unrollScans` writes it out as scalar nodes, which is what
  `toFactorGraph` compiles for message passing.
- Data binding: `bind` (checks arrays against the spec, names the fault), `withData`, `bindDataset`, missing values (NaN in
  an observed node, by policy), batches as subsampled plates, `boundToJSON` and `boundFromJSON`.
- `toProgram` compiles a bound spec to the existing `sample`/`observe` program, so every engine, handler, guide and
  criticism function runs a spec unchanged (the log joint of a spec is `logJoint(toProgram(bound))`; there is no second
  evaluator). `conditionSpec`, `doSpec`, `include` and `fromJSON` compose and carry specs.
- `toStructure` draws a spec in plate notation (or as its factor graph) with ids that name the same nodes in posteriors,
  tapes and factor graphs (`ids.ts`).
- `fromProgram` lifts a `sample`/`observe` program to a spec of its site structure by tracing it (opaque `custom` nodes
  where a parameter moves with other sites; refuses a program whose structure varies).
- Structure analysis, nothing run: `analyseSpec` (dependencies, plates, conjugate pairs, a linear-Gaussian check, nodes
  opaque to message passing), `conjugacyRecords` (through plates, `at` lookups, affine maps, scaled rates, hierarchies
  and joint priors, with the closed form and the conditions), `linearGaussianBlocks`, `graphStructure` (Markov blankets,
  independent components, discrete width, minibatch axes), `structureReport` and `analyseSpec(model).explain()`: advice
  that names the existing method or handler and its knobs, never applied by the library. `collapsePlan` and
  `applyCollapse` (which conjugate sites the `collapse` handler can take out, why not the rest).
- `toFactorGraph` compiles a bound spec to a message-passing factor graph (a regression on a weight vector over a plate is
  one plated factor, other plates are expanded; products, gamma precisions, categorical and Bernoulli gates and maxima map
  to the engine's operators; what cannot be mapped is refused with the node and the reason), and `factorGraphPosterior`
  reads a run back as a `Posterior` by site name.

Inference methods (`omai-engine/probprog/inference`):
- The convention (`defineInferenceMethod`): a `Space` of knobs, a typed tape contract, a steppable, resumable `Algorithm`
  whose state serialises, a capability `check` that says why a method does not apply, an optional `posterior(state)`, and
  the budget and snapshot knobs (`withAnytime`, `runAnytime` with an injected clock; the library reads no clock).
- One method per engine, made from a spec or a program: `nutsMethod` (one iteration of every chain per step),
  `sviMethod`, `gibbsMethod`, `laplaceMethod`, `emMethod`, `importanceMethod`, `temperedSmcMethod`,
  `particleFilterMethod`, `abcSmcMethod`, and one-step methods for the exact engines (`conjugateMethod`,
  `gaussianMethod`, `enumerationMethod`, `chainMethod`, `abcRejectionMethod`). The one-call engines are unchanged.
- EP, VMP and BP on specs (`epSpecMethod`, `vmpSpecMethod`, `bpSpecMethod`).
- The kernels of `omai-engine/sampling` as methods on programs and bound specs, one method each with its own knobs and tape, on
  a shared driver (one transition of every chain per step, warm-up with the library's own adaptation, enumerated discrete sites
  recovered as for NUTS, resumable, published as a posterior): `rwmMethod`, `malaMethod`, `barkerMethod`, `hmcMethod`,
  `sliceMethod`, `ellipticalSliceMethod`, `sgldMethod`, `sghmcMethod`, `mclmcMethod`, the ensemble samplers `aiesMethod` and
  `ensembleSliceMethod`, `cheesMethod` (ChEES-HMC) and `temperingMethod` (parallel tempering over HMC, NUTS, MALA, Barker or a
  random walk, with the ladder, the swap rates and the round trips on the tape); `hybridMethod`, the plan the user writes:
  blocks of latent sites, each with a kernel (`blockKernels`: `hmc`, `nuts`, `mala`, `barker`, `rwm`, `slice`, exact Gibbs
  conditionals, or `custom`), optionally with discrete sites summed out of a block's density. `GibbsOptions.sites` lets the Gibbs
  engine update only the sites it is given. Dense mass matrices take a `lowRank` knob. Not built: MEADS, an adaptive temperature
  ladder, adjusted MCLMC, tempering of the likelihood only, an elliptical slice block in `hybrid`.
- `runAnytimeAsync`: `runAnytime` in slices that hand the thread back to the page between them (the caller supplies the hand-over:
  the library has no timers), so a page keeps painting and answering clicks while runs share the thread within a budget.
- `chainMethod` and `particleFilterMethod` accept a spec with one `scan`, and say why not otherwise.
- A model registry by name for workers (`modelRegistry`, `methodKinds` with 39 kinds, `inferenceMethods` naming each
  method `<kind>/<model>`): a page asks for `nuts/radon` with only knob values.
- Variational methods beyond `sviMethod`, each with its own knobs and tape: `traceGraphSviMethod`, `traceEnumSviMethod`,
  `stickingTheLandingMethod`, `renyiSviMethod`, `structuredSviMethod`, `mixtureSviMethod`, `amortisedSviMethod`,
  `naturalGradientSviMethod`, `conjugateNaturalSviMethod`, `pathfinderMethod`.
- Particle and evidence methods: `pmmhMethod` and `particleGibbsMethod` (particle MCMC; a state-space family, not a spec),
  `stateSpaceFilterMethod` (resampling scheme, ESS threshold and proposal as knobs), `nestedMethod`, and the evidence
  estimators `steppingStoneMethod`, `thermodynamicIntegrationMethod`, `bridgeSamplingMethod`, `importanceEvidenceMethod`
  (`evidenceEstimators` lists what exists and why the harmonic mean is refused).
- `defineKernel`: write a sampler in a few lines on the handlers and trace operations (`resimulationKernel`,
  `randomWalkKernel` as worked examples); it runs as a chain or as an `MCMCKernel` of `sampling`.

Handlers (`omai-engine/probprog`): `reparam` with `locScale` (non-centring), `transformReparam` and `circular`; `collapse`
(delayed sampling of conjugate sites, with `recover`); `lift` (params to latent sites); `scope`; `mean`; `inferConfig` and
`inferHint`; trace operations `assess`, `generate`, `propose`, `regenerate`, `update` (Gen's interface). Not built: Pyro's
`escape` and `equalize`.

SVI breadth (`omai-engine/probprog`): the ELBOs `traceGraphElbo` (Rao-Blackwellised score function with baselines),
`traceEnumElbo` (discrete latents summed out or enumerated), `stlElbo` and `dregElbo`, and `gradientMoments` for the
variance of a gradient estimator; the guides `autoStructured` (follows the model's graph), `laplaceInitialParams`,
`amortisedGuide` (with `amortisationGap`), `autoFlowNormal` (planar, radial and coupling flows) and `autoMixtureNormal`;
natural-gradient SVI (`conjugateNaturalGradientAlgorithm`, `gaussianNaturalGradientAlgorithm`); Pathfinder
(`pathfinderAlgorithm`, a BlackJAX moments fixture); and ADVI's step-size sequence in `omai-engine/optim` (`scaleByAdvi`,
the `advi` optimiser, the `adviDecay` schedule, all three in the option tables).

MCMC kernels (`omai-engine/sampling`): `barkerKernel` (Barker's gradient-skewed proposal), `sghmcKernel` (stochastic-gradient
HMC), `mclmcKernel` with its three-phase tuning `mclmcAdaptationAlgorithm` (microcanonical Langevin Monte Carlo),
`stretchMoveKernel` (the affine-invariant stretch move, or differential evolution) and `ensembleSliceKernel`,
`parallelTemperingKernel` with `temperedTarget` and `temperatureLadder` (swap statistics and round trips in the state),
`cheesKernel` with `cheesAdaptationAlgorithm` (ChEES-HMC), `lowRankInverseMassMatrix` and the `lowRank` option of
`windowAdaptationAlgorithm`, and `chainEssCurves` (the effective sample size by quantile and as the draws accumulate).

SMC, particle MCMC, nested sampling and evidence: `sampling/pmcmc.ts` (`pseudoMarginalMetropolis`), `sampling/nested.ts`
(`nestedAlgorithm`, `nestedSampler`, evidence with an error bar), `sampling/evidence.ts` (`bridgeSampling`,
`thermodynamicIntegration`, `steppingStone`, and `harmonicMeanEvidence`, which refuses by default),
`dynamics/filtering/guided.ts` (`gaussianGuidedModel`, guided proposals through `ParticleModel.guide`, with unbiased
evidence) and `dynamics/filtering/pmcmc.ts` (`pmmhKernel`, `conditionalSmc` with ancestor sampling, `particleGibbs`).

Message passing (`omai-engine/message-passing`, a new domain): a factor-graph engine over exponential-family messages
(Gaussian, gamma, beta, Bernoulli, normal-gamma, multivariate Gaussian, discrete, Dirichlet) with 35 factor operators
(affine, truncation and probit, logistic, conjugate, discrete tables, gates, mixtures, products, `max` and `min`, regression
rows and plates, the spin site), and EP, VMP and BP as named settings of one steppable, resumable algorithm
(`epMethod`, `vmpMethod`, `bpMethod`: projection, schedule and its granularity, damping and the improper-message policy
with counts as slots). Variations, each its own algorithm and method: power and fractional EP, alpha-divergence message
passing, numerical projections (Gauss-Hermite, unscented, Laplace and sample rules, panelled quadrature for numeric
factors), stochastic and averaged EP, region graphs and generalised belief propagation (Bethe, Kikuchi, junction regions),
expectation consistent inference (adaptive TAP) and approximate message passing with its state evolution. Anytime runs
with periodic marginal snapshots. It reproduces the existing `pgm` BP, EP and VMP, the `gp` probit EP, TrueSkill and the
Bayesian Gaussian mixture, matches numerical integration for each operator, and is exact on Gaussian trees.

Validation and benchmarks (`omai-engine/probprog/validation`, `omai-engine/probprog/models`, `make probprog-validate`):
- A validation kernel: `gridPosterior` (evidence and marginals to quadrature), closed-form conjugate oracles, `agree`
  (error in sds and Monte Carlo standard errors, sd ratio, Kolmogorov, Wasserstein and KL against an oracle),
  simulation-based calibration (`sbc`), Geweke's joint distribution test, and a campaign runner.
- Twenty-seven benchmark models as specs with truth, tags, references, a seeded generator and method plans (conjugate
  trio, logistic, probit, Poisson and negative binomial regression, eight schools in both parametrisations, a Gaussian
  mixture with and without label switching, Neal's funnel in both parametrisations, Haario's banana, a two-mode target,
  radon, a three-player TrueSkill, an AR(1) state space as a scan, and nine more: stochastic volatility and a Gaussian
  hidden Markov model as scans, Gaussian process regression centred and non-centred, a small LDA with an exact posterior
  by enumeration, a small Bayesian neural network with a reference run, 2PL item response scoring, a truncated
  stick-breaking mixture and Minka's clutter problem with EP), `benchmarkModels`, `whereBenchmarks`. Simplified from the plan:
  stochastic volatility has known parameters, the IRT items are known, the stick-breaking components are known, the neural
  network has a reference and no closed form. `trueskillTournament` and `trueskillModel` (games added one at a time) and
  `bananaDraws` are generators for the demonstration pages; in `omai-engine/message-passing`, `clutterProblem` (a function,
  not the benchmark model of the same name in `probprog/models`), `clutterGraph`, `clutterPosterior` and `ampProblem` are the
  same for EP and AMP.
- Calibration as a method: `sbcMethod` (one replication per step, the ranks so far as the state, `result(state)` the tests),
  with `sbcReplication` and `sbcSummary` that `sbc` is made of; `posteriorChains` (a posterior's draws regrouped by chain).
- `make probprog-validate` (`scripts/probprog-validate.ts`, not part of `make verify` or `make check`) runs the campaigns and
  writes the committed tables of `docs/probprog/results/`: calibration of NUTS and SVI and of two deliberately wrong guides
  (which must fail), agreement with the grid, the funnel under non-centring, Geweke tests, a campaign per benchmark model,
  the evidence estimators against closed forms, nested sampling's error bar, particle filters and particle MCMC against
  the Kalman filter, the structure report of every model, the variational methods against the truths, the kernel methods
  (`kernels/geweke`; simulation-based calibration of thirteen methods; agreement with the truth on three models; parallel
  tempering against NUTS on the two-mode target) and the later benchmark models with the clutter problem's EP comparison. A
  compiled evaluation plan was measured and not built (`results/eval-overhead.md`).

Engine fixes:
- `potentialEnergy`: a point where the program builds a distribution that refuses its parameter (an overflowing scale or
  rate) has log density -Infinity instead of throwing out of the sampler (`invalidIsImpossible`, default on).
- Gibbs: a site constrained beyond its family's support (`ordered`, a bound on a normal) is updated by slice steps on its
  own scale and starts inside its support (the ordered mixture now agrees with its truth).
- SVI and the guides: a vector support over a plated scalar site (ordered means) is the site's event, so the guides draw it
  outside the plate and SVI runs on the ordered mixture.
- Laplace: `laplaceAt` is the curvature stage, and a mode that is not a point of the posterior (a scale run off to the
  boundary, an ill-conditioned or non-finite curvature) is reported with the coordinates that ran off.
- A posterior summary skips a listed site the posterior holds nothing of; the samplers' and approximations' spec checks
  accept a model with no observed node (a density) and say so.
- `traceView` on a thinned trace places each draw at its own index and marks the divergent transitions at their values
  (`ChainQuantity.atDivergences`), so the markers sit on the line.
- `numerics/special`: `ndtr`, `erf` and `erfc` no longer give NaN at large arguments (0 times Infinity in `expMinusSquare`).

Rendering and exposition (omai-engine, omai-render):
- `modelRun` (beliefs, messages and convergence of inference on a model graph, from a tape or anytime snapshots,
  `modelRunFromTape`) and `chainDiagnostics` (trace, rank and R-hat and ESS data of chains, `chainsFromTape`) exposition
  records, with the views `modelRunScene` and friends, `traceView` and `rankView`, the React `ModelRunPlayer` and
  `ChainDiagnosticsView`, and standard views in the registry.
- Three more records, `sbcRun` (simulation-based calibration: rank histograms with the exact binomial band, the ECDF
  difference with the simultaneous band of Säilynoja et al., the tests and a verdict that says how a method is wrong),
  `elboRun` (a variational fit: the bound, the gradient, the guide's parameters, the beliefs and the final posterior against
  the truth, from the tape of `svi` or Pathfinder or from snapshots, `elboRunFromTape`) and `posteriorComparison` (several
  methods' posteriors against the oracle's as densities, ECDFs or ECDF differences, with the agreement table), which makes 66
  record kinds, with the views `sbcRankView`, `sbcEcdfView`, `elboRunView` and `posteriorComparisonView`, the React `SbcRunView`,
  `ElboRunView` and `PosteriorComparisonView` and standard views in the registry. `chainDiagnostics` gains the charts of ArviZ:
  the ECDF difference with its band (`ecdfBand`), autocorrelation, the effective sample size by quantile and as the draws
  accumulate, the energy with each chain's E-BFMI and the pair plot with the divergent transitions marked (`ecdfView`,
  `autocorrelationView`, `essView`, `energyView`, `pairView`; `ChainDiagnosticsView` offers the charts the record can draw and
  takes an `initial` chart).
- `posteriorPairs` (`omai-engine/exposition`): the corner plot of several methods' posterior draws as a record: each method's
  draws thinned to `maxPoints` (the same points in every pair, with the step each arrived at, the divergent transitions apart),
  the shared axis range and grid of each coordinate, and the density, mean, sd and correlations of the draws up to each of
  `frames` cursor positions, so a view plays the draws arriving with nothing to compute; plain numbers, sized for a Worker
  boundary. The view `posteriorPairsView` (scatter below the diagonal, densities on it, correlations or Gaussian ellipses
  above, the truth marked, divergences in red, a cursor, methods and coordinates to show) and the React `PosteriorPairsView`
  (a switch per method and coordinate, drag to reorder, the upper triangle) and `PosteriorPairsPlayer` (play, pause, scrub, speed;
  the cursor can be the page's: `cursor` and `onCursor`, the hook `usePairsCursor`). `pairView` shares its grid, edge labels
  and thinning with it (`pairGrid`, `pairEdge`, `thinEvenly`).
- Model graphs as plate diagrams: `modelGraph` and `diagram` draw plates that cross (nodes over `[student]`, `[exam]` and
  `[student, exam]`), write sizes as `N = 40`, draw data as dots and unroll vector sites; `StructureNode.groups` and
  `StructureGroup.intersection` in the `Structure` record; `DiagramOptions.decorate` draws a glyph inside each node;
  overlay edges carry a `label`; `StructureView` scales its diagram down to a narrow container. Gallery recipes
  `diagrams/16-model-inference` (extended to the new chain charts), `trust/09-simulation-calibration`,
  `steps/15-variational-run` and `statistical/11-posterior-comparison`.
- `apps/pocs` has groups (`src/groups.ts`, a section per group on the index, a page per group, `groupPocs`); the existing pocs
  joined `probabilistic-programming`, `structured-prediction`, `models` and `graphics-and-fields`. Nine new proofs of concept in
  `probabilistic-programming`: `probit-two-engines`, `ep-alpha-clutter`, `eight-schools-funnel`, `trueskill-messages`,
  `svi-guides`, `tempering-bimodal`, `calibration-sbc`, `evidence-race` and `amp-sparse`, on shared plumbing (`useAnytimeRuns`,
  `RunBar`, `CurveChart`, `useDeferred`) that runs several methods at once within a time budget the user chooses, with a
  snapshot about once a second.

Oracle: `pyro-ppl` joined the `oracle-prob` group (the `lift` and `reparam` goldens use it); new fixtures for `reparam`,
`lift`, Pathfinder (BlackJAX moments), the grid posterior, the exact conjugate oracles and the kernel methods
(`oracle/cases/probprog/inference/kernels.py`: summaries of long chains of NumPyro's AIES and ESS and BlackJAX's MALA, Barker,
MCLMC, SGLD and SGHMC on two programs, compared within the Monte Carlo errors of both).

Docs: `docs/probprog/` (the plan with its status, the inventory, the survey, the algorithm repertoire, the verification
notes and the result tables), `docs/structure.md` sections 4.32 and 4.32a.

Known limits:
- The periodic validation campaigns draw their keys from the campaign's position in the list, so adding or reordering a
  campaign changes every table's draws; a campaign near its tolerance can then flip. The agreement tolerances of the
  slow mixers (`rwm`, `aies`, `ensemble-slice`, and `mala`/`barker` on non-centred eight schools) are therefore `report`
  or carry larger budgets; `rwm` learns its proposal shape from a short warm-up and freezes it, which is the weakness
  behind that.
- The agreement table's mean z uses the sampler's reported ESS, which is overestimated for antithetic chains (HMC on some
  models), so a run with a tiny mean error can still trip `meanZ`; `tempered-smc` shows a mean error larger than its Monte
  Carlo error on item response scoring at 1000 particles (its moves are weak). Both are listed, not fixed.
- EP, VMP and BP are tested against references (the grid for the probit regression, `pgm`, `gp`, TrueSkill, the Bayesian
  mixture). Among the benchmark campaigns EP runs only in `benchmark/item-response-scoring` and `benchmark/clutter-ep`; VMP and
  BP are in none.
- Region-graph (Kikuchi) belief propagation needs damping and does not converge at strong coupling (the tests cover a moderate
  one).
- VMP has no rule for the hard comparison `greater-than`, `max` and `min` are expectation propagation only, and
  `gaussian-gamma-precision` has no EP rule; `check` says so per model. Region graphs and generalised BP are for discrete
  models; AMP is for linear models with iid priors; expectation consistent inference is for a Gaussian times
  one-dimensional sites.
- The `gate` node (mixtures) is not built. Particle MCMC takes a `StateSpaceFamily`, not a spec with a scan (PMMH on scans is
  not done). SMC squared, dynamic nested sampling and Chib's estimator are not built.
- `hybridMethod`, particle MCMC, nested sampling, the evidence methods and the state-space filter are not in `methodKinds`
  (they take inputs other than a model).
- MCMC: no MEADS, no adaptive temperature ladder, no adjusted MCLMC, parallel tempering tempers the whole density (the prior
  too), no elliptical slice block in `hybrid`. MCLMC, SGLD and SGHMC are unadjusted and biased by their step size.
- The nine later benchmark models are simplified: stochastic volatility with known parameters, 2PL scoring with known items,
  the stick-breaking mixture with known components, the neural network with a reference run and no closed form. Not in the
  suite: a normal-gamma linear regression, Gaussian process classification, a change-point model, TrueSkill with teams.
- The 2-column small multiples of the chain and calibration views clip at narrow widths, and the new records have no panel in
  `SessionPanels` (a page builds them from snapshots or a tape: the pocs do).
- Statistical limits recorded in the tables: NUTS diverges on the centred funnel and centred eight schools; a Gaussian guide
  misses the banana and the second mode; Laplace has no mode at tau = 0 of a centred hierarchy; on the logistic regression
  the variational plans are reported, not asserted (the structured guide's means are several Monte Carlo standard errors off,
  sticking the landing's sds are 15 to 20 percent small).
- No browser timings (everything measured in Node), so the decision not to build a compiled evaluation plan rests on Node
  measurements; Pyro's MCMC page and some other web claims of the plan are unverified (`docs/probprog/verification.md`).
- No `examples/` programs for probit on two engines, eight schools or the calibration check (the pocs show them).

Fixes (omai-render), from the notes site's 0.1.12 report:
- A `Chart` no longer makes ECharts warn "has been disposed" when it unmounts (its event handlers were unbound after the
  instance had been disposed).
- `points` `symbolRotate` is a clockwise heading as documented (ECharts turns markers counterclockwise, so the angle is negated).
- A real `Worker` is accepted as a `StageWorker` and a `WorkerLike` without a cast.

## 0.1.12

- `DraggableLine`: the label of a vertical line flips to the left of the line when it would run off the overlay (a line near the
  right edge of the plot, as on a phone).
- `data-omai-state` has a fourth value, `idle`: the run is incomplete and nothing is computing now (a `useLiveRun` that has not
  been played or is paused, a session that is inactive or was halted, a live session never played). `preparing` is work under way
  towards the end, `ready` is the whole run, `failed` is an error. `waitForSettled` treats `idle` like `ready` and `failed`
  (an exporter captures what is shown) and lists the idle roots in its result (`idle`); `live.settled` still resolves only when
  the whole tape exists. A paused live (`'play'`) session stays `preparing`: pausing does not stop its producer, which goes on
  computing the whole run. Before, an unstarted `useLiveRun` read `preparing` for ever, so an exporter waited out its timeout.
- `DraggableRect` and `DraggableLine`: `handleFill` (`'solid'`, `'bg'` or a CSS colour), a visible focus ring drawn in the accent
  colour when focus comes from the keyboard (not from a pointer press), and `keyStep`, the arrow-key step separate from the
  drag snap `step` (default unchanged). `DraggableRect` also takes `drawable` (press outside the rectangle and drag to draw a
  new one, clamped, snapped, `minSize` kept, Escape cancels, a click draws nothing) and `onDrawEnd`.

Structured prediction (omai-engine), all under `pgm/structured/`:
- The CRF family has one home. `crf.ts` and its training moved from `pgm/models` to `pgm/structured` (still exported from
  `omai-engine/pgm`; the golden fixture is now `pgm/structured/crf`). Every model there offers the `StructuredOracle` contract
  (features, expected counts and log Z, an optional `clamped`, cost-augmented `decode`, `loss`), so one table of solvers trains
  them all.
- `structuredSolvers` / `trainStructured`: L-BFGS and every generic smooth solver, `owlqn` (the existing OWL-QN as a CRF
  solver for the L1 penalty, exact zeros, the default when `c1 > 0`; `optim/solvers.ts` gains the reusable `owlqnMode`),
  and the online `perceptron`, `averaged`, `passive-aggressive` and `arow`. `crfSolvers`, `crfSmoothProblem` and `trainCrf` keep
  their signatures and now honour `c1`. Chain data can be compiled once (`compileChainData`, `chainOracle`); `forwardBackward`
  and the CRF attribute lookups are faster.
- Feature extraction reaches the models from all three template languages: `FeatureExtractor` with `crfppExtractor`,
  `wapitiExtractor`, `attributeExtractor` (CRFsuite), `functionExtractor`, `combineExtractors` and `hashedExtractor`,
  `crfModelFor`, and `crfTopFeatures` / `crfTemplateWeights` to read a fitted model back. `text.hashedColumn`, `featureRecord`.
- CRFs beyond the chain: `graphCrf` / `graphOracle` for trees, grids (4 or 8 neighbourhood, directional), skip chains,
  order-k chains and arbitrary graphs with tied weights, exact inference on forests and low-treewidth graphs and loopy or
  tree-reweighted BP otherwise, pseudo-likelihood training, `generalCrfOracle`; `semiMarkovOracle` (semi-Markov CRF);
  `memmSmoothProblem` (the MEMM, with the label-bias example); `partialLabelOracle`, `hiddenStateOracle` (latent-dynamic CRF).
- The CRF estimator in sklearn-crfsuite's shape: `crf(hyper)` / `CRF` (CRFsuite's hyperparameters and feature generation, any solver
  of the slot, a tape of the fit, `predictMarginals`, `stateFeatures`, `constraints` for valid decoding; weights equal
  sklearn-crfsuite's) and `graphCrfEstimator`.
- The neural CRF layer `NeuralCrf` / `neuralCrfLogLikelihood`, `neuralCrfDecode` and `neuralCrfMarginals` (differentiable over tensors,
  masks, start and end transitions; golden-tested against pytorch-crf), and the BiLSTM-CRF `SequenceTagger` with `taggerAlgorithm`
  and `fitTagger`.
- Max-margin: `structuredSvmModes` (subgradient, block-coordinate Frank-Wolfe, one-slack and n-slack cutting planes) and
  `trainStructuredSvm`.
- Decoding: `viterbiKBest`, `constrainedViterbi`, `constrainedPosteriorDecode` and `ChainConstraints` (pgm/inference),
  `allowedTransitions`, `crfDecodeConstrained`, `crfNBest`, `crfPosteriorDecode`, `crfSample`; `sequenceAccuracyScore`.
- MAP: `graphCutMap` (exact binary submodular MAP by min cut, QPBO roof duality for the rest), `alphaExpansion`,
  `alphaBetaSwap`, `ad3Map` and `lpRelaxationMap`; `treeReweighted` also returns `edgeBeliefs`.
- Data: `hyphenationWords` (a 3984-word sample of the public-domain Moby Hyphenator list, one tag per letter),
  `hyphenationTables` and `hyphenationPatternsEnUs` (TeX's US English patterns as the Liang baseline).

From the notes site's last figures (omai-render):
- Fix: `useChartProjection` returns a new projection after a redraw that moved the plotting box, so an overlay held as a stable
  element (`ResponsiveChart`'s `overlay` node) redraws its handles when the chart is resized or its colour bar changes (the
  handles were left 150 to 260 px off after a resize). A test measures `chartProjection` against real ECharts for a heatmap
  with and without a colour bar; a colour bar alone was not found to move the box.
- `DraggablePoints place`: a press on the plot puts the single point (or moves it) and it follows the pointer until release.
  `InteractionOverlay` now keeps one catching rectangle and passes pointer events to every `Backdrop`, so `Probe`, `Brush`,
  `NearestHover` and `DraggablePoints` stack instead of blocking one another.
- `workerStage` and `serveStage` (`omai-render/producer`): a `useJob` stage that runs in a Web Worker with progress and is
  terminated when the job is replaced. `JobContext.onCancel(undo)` runs cleanups when a job is replaced, deactivated or
  unmounted, and a stage still pending then is dropped instead of awaited.
- `image()` takes `Pixels` (RGBA bytes) as well as image data.
- `ControlPanel columns` drops columns on a narrow screen (`minColumnWidth`, default the controls' own minimum).
- The `Player` row wraps its scrubber and stamp onto a second line on a narrow screen instead of clipping the stamp.
- `points` takes `symbol: 'plane'` (new `SYMBOLS` table and `symbolOf`; `cross` moved there from the complex-plane view),
  `symbolRotate` (one angle or one per point) and `labelOffset` (pixels).
- `heatmap cellText(value, row, col, x, y)` gives per-cell hover text.

## 0.1.11

From the notes site's port off its local kit.
- `FigureFrame` can be fluid: `width="fill"` or `height="auto"` makes the content the width of its container and the height its
  content makes, with no fixed box and no clipping, so a figure of several stacked `ResponsiveChart`s, controls and readouts
  fits. It keeps the title and its permalink, purpose, controls, readouts, caption and the export menu; it cannot expand to the
  modal, and it gives no `useFrameSize` (a `ResponsiveChart` inside measures its own container). Numbers behave as before.
- `DraggableLine` with no `span` now runs the plotting box when the projection knows it (`projection.rect`), so the line stops at
  the axes instead of crossing the tick labels; with no known box it still runs the whole overlay.
- `FigureFrame`'s `controls` takes the controls themselves, not a `Controls` row (the frame wraps them): documented.
- Right-angle diagrams, fewer overlaps: `DiagramOptions.separate` (default true) gives a long edge (one that skips a rank) its own
  port and track instead of merging it into a bundle of short edges that share its end and look, and puts runs of different
  edges that end end to end (one reads as the continuation of the other) on separate tracks; `clearance` (pixels, default 18,
  was 6) is the distance a detour or channel track keeps from the nodes, and `trackGap` (default 10, was 8) the spacing of
  parallel tracks. A back edge over a row of nodes now sits clear of their tops, and the scene grows to contain it. Edges
  that share an end and a look and span one rank still merge into one trunk. `separate: false` with `clearance: 6` and
  `trackGap: 8` gives the previous routes exactly; straight routes ignore all three. The gallery entry has controls for them.
- `Player` carries `data-omai-state` from the controller it plays: `useLiveRun`'s controller is `preparing` until the run ends or
  reaches `maxSteps`, then `ready` (`failed` if the run was disposed first); `usePlayback` and `useTimelinePlayback` are `ready` at
  once; a controller with no state gives no attribute. `waitForSettled` on a slide with a player and a chart now waits for the run.
- The control scale (`--omai-control-scale`) also multiplies the default widths of `NumberField`, `Select`, `Choice`, `Toggle`
  and `SeedInput` and the 176 px default of a `Controls` row, so a scaled field is as wide as under CSS zoom. A `width` passed in
  pixels stays literal. Unset, the widths are the same.
- `DraggableRect` (`omai-render/react`): a draggable, resizable rectangle in data coordinates with eight grips (`handles`:
  `all`, `corners` or `edges`), a body that moves it, `min`/`max`, `minSize`, `step`, keyboard nudges (Shift for ten times), Escape
  to cancel a drag, and flipped axes handled. `Handle` is exported and shared with `DraggableLine`.
- `NumberField`'s ladder popup (and `Select` menus of eleven options) no longer clips its last value: the menu was capped at 16 rem,
  and eleven rows need a little more.
- `chartProjection` on ECharts 6: ECharts 6's default `grid.outerBoundsMode: 'auto'` moves the plotting box inside the box a
  chart's option names, by 20 to 40 px, to keep the axis labels and names in the figure; in 0.1.10 the analytic projection assumed
  the box was where the option said, so `project`, `rect` and every overlay on them were that far off. Now a grid that says
  `outerBoundsMode: 'none'` is computed from the option, as before, and any other grid with value axes and numeric limits is
  read from ECharts once per redraw (the pixels of the two corners of the axes, after `finished`) and computed from those, so a
  drag still makes no ECharts call and logs no warning, and `project`, `unproject` and `rect` are where the axes are drawn. The
  charts themselves are unchanged. A test draws real ECharts instances and compares.
- `Chart` takes `ratio` (the magnification: the canvas is made at `devicePixelRatio x ratio`) and `PixelRatioProvider ratio` /
  `usePixelRatio()` set it for every chart below (nestable, the innermost wins); `ResponsiveChart` and `FigureGrid` pass it on. An
  expanded `FigureFrame`'s scale composes with the provider's; an explicit `ratio` wins. Nothing is clamped (the host clamps);
  SVG-rendered charts ignore it.
- Engine, scikit-learn and imbalanced-learn gaps (`docs/completeness/sklearn.md`):
  - Sparse coding and dictionary learning: `sparseCode`/`sparseCoder` (OMP, LARS, lasso-LARS, lasso by coordinate descent,
    thresholding; `positive` where the solver has it), `dictionaryLearning` (batch; the atom update is a slot: coordinate descent,
    MOD or K-SVD; unused atoms are replaced), `miniBatchDictionaryLearning` and `dictLearningOnline` (Mairal et al.), the
    `sparseCodedSignal` generator; all stepped and traced.
  - imbalanced-learn: SMOTEN, SMOTENC, SVM-SMOTE, K-means SMOTE, edited and repeated edited nearest neighbours, AllKNN, condensed
    nearest neighbour, one-sided selection, the neighbourhood cleaning rule, instance hardness threshold, cluster centroids,
    SMOTEENN, SMOTETomek, and the balanced ensembles (balanced bagging, balanced random forest, EasyEnsemble, RUSBoost); the
    deterministic cleaners match imbalanced-learn index for index.
  - Estimators and small items: `kMeans`, `miniBatchKMeans`, `kmeansPlusPlus`, `gaussianMixture`, `bayesianGaussianMixture`,
    `dummyClassifier`, `dummyRegressor`, `fixedThresholdClassifier`, `tunedThresholdClassifierCV`, `truncatedSvd`, `miniBatchNmf`,
    `ardRegression`, `lassoLarsIC`, `classLikelihoodRatios`, `addDummyFeature`, image patches (`extractPatches2d`,
    `reconstructFromPatches2d`, `patchExtractor`, `imgToGraph`), `dictVectorize` and `featureHash`.
  - `latentDirichletAllocation` (batch and online variational Bayes), spectral co-clustering and biclustering with
    `consensusScore`, and `linearSvc` / `linearSvr` (liblinear's model: dual coordinate descent, OWL-QN for L1, the smooth solvers
    for the squared losses).
  - `larsPath` in lasso mode left the optimum on collinear columns and, more rarely, on ordinary designs after a variable left and
    re-entered (30 of 1800 random paths); fixed, and `positive` added.
- Covariance estimation (a new area in `docs/completeness/covariance.md`): the Marchenko-Pastur law (`dists`), spiked-model
  thresholds, `denoiseCorrelation` and `detoneCorrelation`, shrinkage towards diagonal, constant-correlation and single-factor
  targets with analytic intensities and a `shrinkagePath`, Ledoit-Wolf nonlinear shrinkage, `selectCovariance` by held-out
  likelihood, nearest correlation and covariance matrices (Higham with Dykstra; matches statsmodels), `GraphicalLassoCV` with EBIC
  and StARS penalty selection, neighbourhood selection, CLIME, nonparanormal correlations, the adaptive graphical lasso (the
  graphical lasso takes a per-entry penalty matrix and a warm start), and generators with a known covariance or precision graph.
- `Controls` (and so the `controls` of a `FigureFrame`) gives the number fields, selects and choices in it a default width of
  176 pixels (`controlWidth` on `Controls` to change it, a control's own `width` to override), so a row of fields and toggles
  lines up without a `width` on each; a toggle keeps its natural width. Outside a `Controls` row nothing changes.
- `useParams`: values may be nested read-only arrays (a list of points), `set` takes a read-only tuple where the initial value
  was inferred as a mutable array, nested arrays are compared by value, and `reset` can be passed as an event handler
  (`onClick={reset}` resets all).

## 0.1.10

What a notes page needs to use omai directly, with no local kit (the circular statistics page and decksmith).
- `ResponsiveChart` (`omai-render/react`): a chart that fills its container. `build(fig)` makes the option for the measured
  width (rebuilt only when the width changes by a pixel), `narrowAspect` below `narrowBelow` (default 520 px), `square` for a
  square plotting area (`squarePlot` in `omai-render/figure`), and an `overlay` slot that gets `{ projection, size }` so draggable
  handles sit on a fluid chart. An empty (0 by 0, hidden) measurement is ignored and the last size kept. Wrap `build` in
  `useCallback`.
- Colour modes: a layer built with no `mode` is now right in the mode of the chart that draws it. Before, `lines([{ colour: 'fg' }])`
  drawn in a dark `cartesian` had the light foreground (`#0a0a0a`, invisible); every layer builder in `omai-render/charts` (40
  of them) is now redrawn in the chart's mode by `cartesian`, `subplots` and `plotRectOf`. An explicit `mode` on a layer still
  wins, and light output is unchanged. A layer copied or edited as plain data (`{ ...layer, secondary: true }`) keeps the colours
  it had.
- Projection for overlays: `chartProjection` and `useChartProjection` make no ECharts call for a chart whose grid box and value
  axis limits are numbers (the usual omai chart): the map comes from the option. Dragging a handle used to log about 14 ECharts
  warnings ("No coordinate system that supports convertToPixel"); now it logs none. Log, category, time and auto-range axes, a
  `containLabel` grid and any `dataZoom` still use ECharts' converters. `Projection` gained `project(x, y)` and `unproject(px, py)`
  (required members: a hand-built `Projection` needs them), and `useChartProjection` re-renders its host when a redraw moves the box.
- `useJob`: a heavy one-shot computation in named stages with progress (`phase`, `stage`, `stageName`, `progress`, `result`,
  `restart`). Nothing runs until `active`, a change of `inputs` starts a new job and drops the old one's late result, a stage may be
  a generator that yields a fraction and is run in slices of 8 ms so the bar paints. The stage is readable while it runs. A single
  stage that blocks for seconds still blocks the page: only a worker helps.
- `useParams(initial)` (`omai-render/react`): a figure's own knobs as one record (`values`, `set(patch)`, `reset`, `isDefault`).
- Controls: `NumberField`, `Toggle`, `Select` and `Choice` take `size` (`sm`, `md`, `lg`) and `width` (pixels, a CSS length, `fill`
  or `auto`), and their labels wrap instead of truncating (`truncate` brings the old behaviour back). Defaults look as before.
- `FigureFrame`: slots `purpose`, `controls`, `readouts` and `caption`, with small `Readouts` (a label and value list) and `Controls`
  (a wrapping row) components; the title is a permalink (`#id`, copies the address, scrolls to the figure when the page is
  opened at it; `id` and `permalink={false}` to set or turn off). The host-token mapping onto the `--wb-*` variables is documented
  in `react/theme.tsx`.
- Ambient `active`: `ActiveProvider` (`omai-render/react`) says whether everything below it may compute and play; `useSession`,
  `useSessions`, `useLiveRun`, `usePlayback`, `useJob` and `SessionPlayer` read it when their own `active` option is not given
  (an explicit option replaces it). Providers nest: a figure is active only when every provider above it is.
- `ThemeProvider` nests and carries the colour mode, token overrides (`tokens`: a patch, a `{ light, dark }` pair or a function of
  the mode) and the host's CSS variable names (`cssVariables: { fg: '--foreground' }` writes `--wb-fg: var(--foreground)`), so a
  deck can pin a light slide inside a dark deck and stop mapping `--wb-*` by hand. Charts (`Chart`, `ResponsiveChart`), scenes and
  structures follow it; popups in a portal copy its variables. It now renders a `display: contents` wrapper. Colours a builder
  writes from `tokens(mode)` itself stay the mode's (only the ECharts theme follows `tokens`).
- Control scale: one CSS variable `--omai-control-scale` (or `ControlScaleProvider scale={1.3}`) scales the heights, text and
  paddings of the controls and players, popups included, in place of CSS `zoom`; `Player` and `SessionPlayer` take `size`
  (`sm`, `md`, `lg`). Unset, the sizes are the standard ones.
- A settled signal for export: every live figure root carries `data-omai-state` (`preparing`, `ready` or `failed`: `SessionPlayer`,
  `SessionPanels`, `Chart`, `ResponsiveChart`, `SceneView`, `StructureView`); a session handle has `settled`, a promise that never
  rejects and resolves `{ ok: true }` when the whole tape exists, or `{ ok: false, error }` (`'disposed'` when the run was remade or
  dropped first), so an exporter awaits it; `waitForSettled(root, { timeoutMs })` waits for every figure under a root. A run that
  threw on the main thread is now reported (`live.error`, the player's alert) instead of escaping into the page.
- Import safety is tested: every entry point of `omai-engine` and `omai-render` imports in Node with no `window`, `document`,
  `Worker` or similar (`scripts/import-safe.test.ts`).
- Registration is tolerant of a second copy of omai-engine: the same object listed twice in a registry is kept once (two different
  objects under one name still throw), and the default context lives on a `Symbol.for('omai-engine/defaultContext')` holder, so
  a seed set through one copy applies to all.
- `DraggableLine`: `handleShape` (adds `diamond`) and `handleSize`. `NumberField`'s `format`, `parse` and `draftText` are documented
  precisely (format is never applied to a focused draft).

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
