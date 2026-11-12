/**
 * What expectation propagation cannot do yet, and what it does less well than it should, as data the coverage page shows and a test
 * holds to account: every setup and benchmark model that EP refuses is claimed by at least one gap, so a refusal is never an orphan,
 * and every gap says why, how it would be closed, how big that is and where it stands. The prose is written by hand from what the
 * audits, the atoms, the programs and the agents' reports found; which items a gap refuses is not: a gap carries the phrases its
 * refusals contain (`markers`), and `epGapsRefusing` reads the coverage's verdicts against them. When a rule lands the gap is edited
 * or removed; counts and dates do not belong in the prose (`docs/probprog/ep-status.md` is generated from the coverage).
 *
 * @index module The gaps and limits of expectation propagation, each with its plan, size and status, tied to what it refuses.
 */

/** What kind of shortfall: no rule, no compiler mapping, a quality problem of EP itself, an infrastructure task. */
export type EpGapKind = 'missing-rule' | 'missing-mapping' | 'quality' | 'infrastructure'

/** Where a gap stands: queued (next in line, not started), investigating (someone is on it), open (no plan to start yet). */
export type EpGapStatus = 'queued' | 'investigating' | 'open'

/** One shortfall of the EP suite. */
export interface EpGap {
  readonly id: string
  readonly title: string
  readonly kind: EpGapKind
  readonly status: EpGapStatus
  /** What is missing or wrong, in a sentence or two. */
  readonly summary: string
  /** How it would be closed. */
  readonly plan: string
  /** A rough size of the work: small (a day or less), medium (a few days), large (a week or more). */
  readonly size: 'small' | 'medium' | 'large'
  /**
   * Phrases that the reason or the missing tags of a refusal (a verdict, or the error of a run) contain when this gap is what
   * refuses the item; empty for a shortfall that refuses nothing (a quality problem, an infrastructure task).
   */
  readonly markers: readonly string[]
  /** The setups and benchmark models EP refuses because of it, by name: always derived from the coverage (`epGapsRefusing`), empty in `epGaps`. */
  readonly refuses: readonly string[]
}

/**
 * The gaps, most valuable first within each kind.
 *
 * @example
 * ```ts
 * import { epGaps } from 'omai-engine/probprog/ep'
 * epGaps.every((g) => g.plan.length > 0 && g.summary.length > 0) // => true
 * ```
 */
export const epGaps: readonly EpGap[] = gapList().map((g) => ({ ...g, refuses: [] }))

/** The gaps as written: everything but the names they refuse, which the coverage gives. */
function gapList(): readonly Omit<EpGap, 'refuses'>[] {
  return [
    {
      id: 'grid-hyperparameters',
      title: 'Hyperparameters on a grid: a few scalars, one family at a time',
      kind: 'missing-mapping',
      status: 'open',
      summary:
        'A Gaussian process with a random length scale runs by EP inside a grid over the hyperparameters a custom function reads (`epGridMethod`: each node conditions the model, EP runs, the EP evidence weights the node), exact for a Gaussian likelihood. The grid is a product over the hyperparameters, so it serves one to three scalars; an ARD kernel with a length scale per input, a hyperparameter that is a vector or has a prior that depends on another node, and a custom function that is not known or affine once the hyperparameters are set are refused.',
      plan: 'A numeric site over the length scale for the one-dimensional case inside one EP run (the hyperparameter as a latent whose message is a grid), a sparse grid or a Laplace step over several hyperparameters, and a model-layer kernel node (a Gram matrix as a first-class operator on a latent length scale) so that the custom function is not needed.',
      size: 'medium',
      markers: ['hyperparameters', 'opaque to message passing'],
    },
    {
      id: 'matrix-larger',
      title: 'Covariance matrices beyond what the Wishart and the bivariate sd-and-correlation sites reach',
      kind: 'missing-rule',
      status: 'open',
      summary:
        'Done: a Wishart family, the exact conjugate and the Bartlett-quadrature EP rule (`mvn-wishart`, dimensions 2 to 4, the exact update in any dimension), the group form (`mvn-wishart-group`: the vectors of every group, their shared Gaussian mean and the matrix as one factor, one tilt per sweep for all its ports; dimension 3 with eight groups takes about two seconds, dimension 4 about a quarter of a minute, each against an importance-sampling reference to within a few percent of the spread), the normal-Wishart conjugate form, the sd-and-correlation site for a bivariate normal (`mvn-sd-corr`: gamma precisions, an LKJ correlation as a beta variable), and six atoms. What remains: a Wishart matrix of dimension 5 and above in a non-conjugate setting (the quadrature is a product rule with d (d + 1) / 2 coordinates, three nodes each at d = 4 are 59049 nodes and three at d = 5 would be fourteen million, while two nodes per coordinate leave the spread of the matrix 5% to 15% short), an LKJ prior on a correlation matrix of order 3 and above (three correlations are three coordinates of the numeric site, six with the standard deviations, and the product of beta beliefs on the partial correlations loses their dependence), a latent bivariate vector with standard deviations and a correlation of its own (a three-dimensional grid per group: about a second per sweep per group), and a normal-Wishart variable inside a larger model (the compiler maps the Wishart form only, with a random mean as the group form).',
      plan: 'A sparse-grid or Laplace step for the Wishart tilt at dimension 5 and above (the conjugate part of the factor is already taken out, so the remainder is smooth), the canonical-partial-correlation parameterisation with one beta variable per partial correlation and a site over them for the LKJ prior of order 3 (the correlation matrix with known standard deviations first: three coordinates), a group form of the sd-and-correlation site so that a latent bivariate vector per group costs one grid per sweep, and the compiler pattern that reads `mu | L ~ N(m, (kappa L)^-1)` as one normal-Wishart variable.',
      size: 'medium',
      markers: ['the quadrature is built for dimensions up to 4'],
    },
    {
      id: 'correlation-strong-few-pairs',
      title: 'A strong bivariate correlation from few pairs: the beta belief is too narrow',
      kind: 'quality',
      status: 'open',
      summary:
        'A bivariate correlation carried as a beta on (rho + 1) / 2 (a plated factor, so the beta matches the exact marginal on its two log moments) has a spread 0.81 to 0.92 of the exact when the correlation is strong (0.9 to 0.99) and the pairs few; the mean is right to 0.07 sd, and at a mild correlation or a larger concentration the spread is exact to a few percent. The learned covariance of correlated random effects no longer has the error it had: the group vectors, the mean and the matrix are one factor (`mvn-wishart-group`), and with few groups (three to ten, prior degrees of freedom 2.2 to 3) the mean is within a few percent of the exact spread (the atom reference grid is itself a few percent high for it at three groups), the matrix and every group within 1%, the evidence within 0.002 nats, in three sweeps.',
      plan: 'A skew-aware belief for the correlation in place of the beta on (rho + 1) / 2: a Gaussian or a beta on the Fisher transform, with the LKJ prior written on it, so that the projection onto the family keeps the long tail toward the bound.',
      size: 'medium',
      markers: [],
    },
    {
      id: 'softmax-many-classes',
      title: 'Softmax beyond six classes: the probit with a power, correlated classes',
      kind: 'missing-rule',
      status: 'open',
      summary:
        'Done: the multiclass site (`multiclass-site`) integrates any number of classes under EP. Up to six free scores it keeps the adaptive Gauss-Hermite product grid; beyond that (or with `integration: "auxiliary"`) it writes the integral so that the scores do not meet: the softmax normaliser is a gamma integral over one auxiliary variable (`S^-p = int t^(p-1) exp(-t S) dt / Gamma(p)`, so given it the factor is a product over the classes, and the power of fractional EP is the shape), the multinomial probit one integral over the observed score plus its noise with the other scores in closed form. Both cost the number of classes, not its power (a softmax of 20 classes takes a few milliseconds a tilted moment), and agree with a brute-force integral up to four classes, the product grid at six, and, for any number, with the exact identities that the classes\' masses add up to one and their tilted moments to the cavities\' (1e-8 and better). Eight alternatives with two shared coefficients (`softmax-choice`) run, with a grid reference; a multiclass Gaussian process is one dense Gaussian vector over the (input, class) pairs gathered with `ops.at` (`gp-softmax`, two inputs and three classes against quadrature), independent processes per class or any covariance across them. Variational message passing runs any number of classes (`_multiclass-variational.ts`: the expected log of the softmax is exact, an integral over the same auxiliary variable of the product of the transforms of the classes; the multinomial probit has a Jensen bound over the other scores, exact when they are known; `softmax-choice` and its learned twin run, converging in 41 and 46 sweeps, the means within 0.19 grid sds and the spreads 0.65 to 0.88 of the reference grid). What remains: the multinomial probit of more than six scores takes the factor whole (power one); the softmax power may not be below 0.03 (the window of the auxiliary variable); the site sends each score its own Gaussian message, so the correlations the labels put between the classes of one input, and the loop through shared coefficients (the two coefficients of `softmax-choice` have 0.70 to 0.96 of the exact spread, as the class slopes of three classes do), are not carried; and the compiler maps one dense vector over one dimension, so a Gaussian process per class is written as one vector of inputs times classes cells, not as a plate of vectors.',
      plan: 'A joint site message over the K scores of one observation (one Gaussian over the classes: a vector-valued message, which the dense GP prior would then correlate), and a compiler pattern that reads a plate of dense Gaussian vectors (one per class) as one vector so that the cells need not be written out.',
      size: 'small',
      markers: ['more than six scores takes a power of one'],
    },
    {
      id: 'multimodal',
      title: 'One Gaussian per variable cannot show two modes: what the multi-start does not reach',
      kind: 'quality',
      status: 'open',
      summary:
        "One EP run commits to one reading of a bimodal or label-ambiguous posterior and reports the spread of that reading alone: on the changepoint program (24 seeds, 72 intervals) the truth is inside the 90% interval for 0.778 of the rates (the sampler 0.931), on the clutter problem the location is 7.8 +- 0.7 where the posterior is 6.4 +- 2.3. The route that reports the other modes is EP from several starts weighted by evidence (`epMultiStartMethod`): the mixture of the distinct fixed points, each weighted by its EP evidence (`diagnostics.modeWeights`), the model's evidence the log sum. A start is set on the port of each label's prior and held for a few damped sweeps; a label whose only other neighbour is its row's likelihood has no cavity but the prior's message, so a start set anywhere else was divided out and a mixture with independent labels (the clutter problem, a mixture of clusters) had one mode whatever the starts. Held to references where the weights are known: the clutter problem's bimodal location against its quadrature (six data sets: the evidence within 0.006 nats, the mean within 0.03 sd, the sd within 4% of the truth's, the second mode's weight 0.001 to 0.28), a two-cluster mean with unequal priors against the enumeration of the assignments (the mass of the labelling with the first mean below the second 0.559, 0.186 and 0.980, the exact values to four digits; the evidence within 1e-4 nats; the labelled means and sds exact to 0.02 sd and 1%), the mixture of the benchmarks with symmetric priors (equal weights, the sorted means within 0.05 sd of the grid). `labels: 'sort'` reads the components in order of their rates, as a data-ordered sampler does: the changepoint program covers 0.889 of the rates with spreads 0.95 of the sampler's where every regime is visited, and the unvisited regime's rate 0.77 to 1.27 of it (one run: 0.05 to 0.15). What remains: a regime visited by a handful of steps (seeds 4, 7, 10 of the 24: spreads 0.2 to 0.35 of the sampler's, errors 2 to 4.5 sds), where every start reaches the same reading and the other readings are not fixed points of EP (the runs cycle); modes that overlap (ten points between -2 and 2 for two cluster means) are one fixed point whose evidence is the single Gaussian's (-19.07 against the exact -18.75); a symmetry of continuous variables (the sign of a loading, with no discrete label) has no discrete message to start from; the weight of a mode is its EP evidence, as good as the Gaussian approximation of that mode and no better (a mode that is an artefact of a loop weighs what it weighs); and models whose evidence is undefined at the fixed point (the circular mixture's row gates), which cannot be weighted. Power EP, alpha-divergences, damping and a damped retry of the starts do not help (docs/probprog/ep-changelog.md section 11, docs/probprog/notes/ep-notes-quality2.md section 1). The `labels` default (`keep`, the model as written) is the owner's choice; docs/probprog/ep.md section 4 gives the measured recommendation.",
      plan: 'The chain summed out given the shared rates (a forward-algorithm site over the three rates, a numeric message of the rates) so that the loop between a shared rate and a discrete state per step is gone; an evidence for the row gates with an improper cavity (the same expansion as the plated factor) so that the circular mixture can be multi-started; a start of the continuous variables (the mirror of a sign-symmetric loading, a relabelling of a plate of continuous components with no discrete label) so that a symmetry that has no discrete message is multi-started too.',
      size: 'medium',
      markers: [],
    },
    {
      id: 'loopy-overconfidence',
      title: 'Loops make the spreads too small',
      kind: 'quality',
      status: 'open',
      summary:
        'Scalar latents that share a predictor were 2 to 3 times too narrow (packing them into one joint Gaussian fixed the common case, and the packed logistic and probit regressions are calibrated). The vectors are packed too now (a tied group: the county intercepts of a centred hierarchy beside the floor effect, and the skills of a tournament, a vector that each game reads at two places, each tied to its coordinate of one joint vector): the radon floor effect went from 0.71 to 0.91 of the exact sd (24 counties against NUTS: 0.68 to 0.86), the three-player tournament from 0.74 to 0.85 to 0.99 to 1.00 of the exact grid (3 sweeps instead of 28, the evidence 0.0001 nats from the grid), the tournament with a learned spread from 0.78 to 0.97 of NUTS (the truth inside the 90% interval 0.87 to 0.94 over 24 seeds). Remaining: the non-centred hierarchy (a row reads the county intercepts through a product with a random scale, nothing linear in a vector to pack: the floor effect stays at 0.86 of NUTS, and the mean of the hierarchy, which no row reads, at 0.87 to 0.89), the 2PL item response model (the logit is a product of a discrimination and a difference of two vectors, bilinear, so not a Gaussian vector site: 0.04 to 0.7 of the NUTS spread of the discriminations with 30 students and 8 questions), the two shapes of beta outcomes (0.65 to 0.8; the pair site sends one message per shape, so a joint vector gets none that links them; a power below 1 makes it worse, a power near 1.15 gives the exact spread with the mean 0.1 to 0.2 sds off) and the softmax class slopes (0.4 to 0.75 of the exact spread, the truth inside the 90% interval 0.45 to 0.80 of the time against the exact 0.96; the multiclass site likewise sends one message per class). The class slopes are repaired by fractional EP, the `power` knob of `epSpecMethod` (power about 5 divided by the rows: coverage 0.92 to 0.96 with 10 rows, 0.93 with 24, 0.92 with 48, the means 0.1 to 0.2 exact sds off) and the two softmax atoms run at it; the sign of the correction depends on the model, so it is not a default.',
      plan: 'Give the pair site and the multiclass site a vector port (one message in the information form over the shapes, or over the class slopes, so a tied vector receives the off-diagonal terms), which makes the power unnecessary for the slopes and calibrates the beta outcomes; a bilinear site for the 2PL (a rank-one update over a discrimination, an ability and a difficulty); structured EP (regions, the engine has them) for the non-centred hierarchy; keep the calibration panels honest so a regression shows.',
      size: 'medium',
      markers: [],
    },
    {
      id: 'sv-parameters',
      title: 'Stochastic volatility with an unknown persistence: one EP run is too narrow, the grid is the route',
      kind: 'quality',
      status: 'open',
      summary:
        "With the level, the persistence and the innovation precision unknown, one EP run covers the truth for 0.48 of the levels, 0.58 of the persistences and 0.63 of the precisions (30 seeds of 60 returns), the level's sd a fifth to a third of a NUTS run's: the persistence multiplies the state (a `product` factor), so the chain is not linear and its messages are scalar. EP inside a grid over the persistence (`epGridMethod(bound, { over: ['phi'] })`, as a walk from the middle of the range, each node started from its neighbour's messages: `order: 'walk'`, `warm: true`, damping 0, 60 sweeps) is calibrated: with the persistence known the steps of the series are one `gaussian-chain` factor on the states, the level and the precision (the scalar steps sent each state a message from two neighbour cavities taken as independent, which they are not once the level and the precision are shared: the level came out twice as wide as the exact one at a given persistence, 0.47 against 0.25, and the precision's messages cycled at a low persistence), and against a brute-force posterior (a point-mass filter for the likelihood of each parameter triple on a grid; it agrees with NUTS to a fifth of a sd) every one of the 30 seeds has its level, persistence and precision within 0.2, 0.07 and 0.04 reference sds (rms 0.05, 0.05, 0.02), the spreads at 0.96, 1.03 and 0.98 of the reference's at the median (0.88 to 1.07, 1.01 to 1.05, 0.95 to 1.00), the 90% intervals covering the truth for 0.80, 1.00 and 1.00 of the seeds (the reference's own coverage of these 30 draws), in 3 to 6 s a fit with all but the nodes beyond a persistence of one settled (7 s and 0.97, 0.97, 1.00 before). What remains: a single run with the persistence random is still the narrow one; the chain factor needs known coefficients, so only a grid node or a model with a fixed autoregression gets it; the factor's weak prior on a state whose cavity is uninformative (`ridge`, 0.05 of the first state's prior precision, centred at the chain's own estimate) is a heuristic that a return near zero makes necessary (without it the precision runs to zero), and it moves the level and the precision by about 0.05 sd.",
      plan: 'Give the chain factor the persistence as a port (a numeric site over the persistence inside the one factor, the grid of the integral over the precision extended to the pair), so that one EP run, with no outer grid, has the exact conditional and the persistence; and replace the ridge by a likelihood site for a return near zero that is not improper (the mixture approximation of the log chi-squared).',
      size: 'small',
      markers: [],
    },
    {
      id: 'evidence',
      title: 'The EP log evidence is exact where EP is, and has two known limits',
      kind: 'infrastructure',
      status: 'open',
      summary:
        "Every rule of the registry carries a log evidence (each tested against a brute-force integral) and the posterior and the tape (`mp/evidence`) give the model's: against the grid evidence it is within 1e-5 on the conjugate atoms, 1e-3 on the gates, the Student-t, the clutter, the summed-out mixtures, 0.01 to 0.05 on the probit, logistic, Laplace and product atoms, and the numeric priors (half-normal, half-Cauchy, uniform, truncated) are the exact prior, not a projection (uniform exact, half-normal 0.004, half-Cauchy 0.015 nats). Two limits remain: a fixed point with a factor whose cavity is improper has no evidence (the learned logistic regression as one plated factor: the scale mixture's site has a negative precision, so the cavity of the plate is indefinite and its integral diverges; the evidence is then that of the rows-expanded model, named `ep-energy (rows expanded)`, 0.05 nats from the grid; when that does not converge the posterior says which factor in `diagnostics.evidenceUndefinedAt`); a fixed point whose row gates have an improper cavity (the circular mixture, more than a hundred of them) still has none, and an alpha-divergence run has no EP energy.",
      plan: 'Give the gates with an improper cavity an evidence in natural parameters (as the Gaussian, dot and apply rules have), and give the alpha runs the energy of their own divergence.',
      size: 'medium',
      markers: [],
    },
    {
      id: 'variational-rules',
      title: 'Variational message passing: what it cannot run and where the mean field is poor',
      kind: 'infrastructure',
      status: 'open',
      summary:
        'Variational message passing runs most of the atoms (`vmpCoverage` counts them): the numeric likelihoods (the Student-t with a random scale and degrees of freedom, and an ordinal outcome in its first or last category, each checked as a stationary maximum of the brute-force bound), scales, shapes, the softmax and the multinomial probit of any number of classes (the softmax exact, the probit a bound that is exact when the other scores are known), products, maxima, unary functions, gamma products and selects all have a mean-field message. What is left, and what would be sound: the hard comparisons (`greater-than`, `is-between`, `spin`: KL(q || p) is infinite for a Gaussian q, so there is no bound to take; the probit link is a relaxed model with a noise scale, not the same model, and a belief that can be truncated is a family the engine does not have); the Weibull with a random scale and shape (the survival term is a double exponential, whose expectation diverges under a Gaussian belief of the scale: a proof, not a missing rule); an ordinal outcome in a category between two random cut points (zero where they are out of order, so the expected log is minus infinity for every Gaussian belief of them; cut points ordered by construction, a first cut and positive gaps, would have a smooth density and need a compiler pattern); the custom functions of a random value (opaque to message passing); and the mixtures, where the mean field settles on the optimum that explains everything as clutter or as the spike (clutter and spike-and-slab) and the banana posterior of a product that a belief per variable cannot follow.',
      plan: 'Start the mean field of a mixture from the data or from several restarts (as `epMultiStartMethod` does for EP), give the ordinal cut points an order-by-construction form in the model compiler (a first cut and positive gaps), and decide whether the hard comparisons get an opt-in probit relaxation (a different, stated model) in the compiler; keep `vmpCoverage` and the committed digest honest as the count goes up.',
      size: 'small',
      markers: [],
    },
  ]
}

/** What a gap needs to know of an item EP does not run: its name and why (the verdict's reason and missing tags, or the error of the run). */
export interface RefusedItem {
  readonly name: string
  readonly reason: string
  readonly missing: readonly string[]
}

const claims = (g: Omit<EpGap, 'refuses'>, r: RefusedItem): boolean =>
  g.markers.some((m) => r.reason.includes(m) || r.missing.some((t) => t.includes(m)))

/**
 * The gaps with the items each refuses, derived from the verdicts: an item the coverage says EP does not run belongs to every gap
 * one of whose markers its reason or its missing tags contain. Nothing is kept by hand: a rule that lands removes the refusal, and
 * the gap stops naming the item.
 *
 * @param refused - The items EP does not run, with the library's reasons (the `planned` rows of `epCoverage`).
 * @returns The gaps, each with its `refuses`.
 * @example
 * ```ts
 * import { epGapsRefusing } from 'omai-engine/probprog/ep'
 * const gaps = epGapsRefusing([{ name: 'many-classes', reason: 'multiclass-site: the multinomial probit of more than six scores takes a power of one', missing: [] }])
 * gaps.filter((g) => g.refuses.length > 0).map((g) => g.id) // => ['softmax-many-classes']
 * ```
 */
export function epGapsRefusing(refused: readonly RefusedItem[]): readonly EpGap[] {
  return gapList().map((g) => ({ ...g, refuses: refused.filter((r) => claims(g, r)).map((r) => r.name) }))
}

/**
 * The items EP does not run that no gap claims: a refusal nobody has written down. The coverage test holds this empty.
 *
 * @param refused - The items EP does not run, with the library's reasons.
 * @returns The names of the unclaimed ones.
 * @example
 * ```ts
 * import { unclaimedRefusals } from 'omai-engine/probprog/ep'
 * unclaimedRefusals([{ name: 'new', reason: 'something nobody has seen', missing: [] }]) // => ['new']
 * ```
 */
export function unclaimedRefusals(refused: readonly RefusedItem[]): readonly string[] {
  const list = gapList()
  return refused.filter((r) => !list.some((g) => claims(g, r))).map((r) => r.name)
}

/** How far a direction is covered today: runs in the suite, partly (some forms run), being built now, or not started. */
export type EpDirectionStatus = 'covered' | 'partial' | 'in progress' | 'not yet'

/** A kind of model or problem the suite could cover: what it is, how far it is covered and what supporting it needs. */
export interface EpDirection {
  readonly id: string
  readonly title: string
  /** The area it belongs to, for grouping. */
  readonly area:
    | 'time series'
    | 'counts and survival'
    | 'ranking and preference'
    | 'latent structure'
    | 'regression and classification'
    | 'spatial and nonparametric'
    | 'inference quality'
  readonly status: EpDirectionStatus
  /** What the problem is, in a sentence. */
  readonly what: string
  /** What supporting it takes beyond what exists (a rule, a mapping, a family), or what already supports it. */
  readonly needs: string
  readonly size: 'small' | 'medium' | 'large'
  /** The suite's pages (poc slugs) that already show something of it. */
  readonly pages: readonly string[]
}

/**
 * The wider suite: not only what is refused today (the gaps above) but what else the suite could cover and what each would need,
 * written by hand and meant to grow. It is a map of directions for choosing the next programs and rules, not a promise.
 *
 * @example
 * ```ts
 * import { epDirections } from 'omai-engine/probprog/ep'
 * epDirections.every((d) => d.needs.length > 0) // => true
 * ```
 */
export const epDirections: readonly EpDirection[] = [
  {
    id: 'periodic',
    title: 'Periodic and circular data without a circular distribution',
    area: 'latent structure',
    status: 'in progress',
    what: 'Clock times, bearings and phases: a wrapped normal written as three ordinary normals, by shifting the means (the paper) or shifting the data (the post), fitted by EP.',
    needs:
      'Gates and an interval constraint exist; the copy index as a lookup on the data side may need a mapping. Higher dimensions (a torus) are a grid of copies.',
    size: 'medium',
    pages: [],
  },
  {
    id: 'kalman',
    title: 'Linear dynamical systems and Kalman smoothing with vector states',
    area: 'time series',
    status: 'covered',
    what: 'A latent vector state with a linear transition and linear observations: tracking, dynamic regression, random-walk coefficients.',
    needs:
      'Runs: a `scan` whose state is a multivariate normal and whose step is `ops.matvec(F, prev)` is a Gaussian vector per step with a dense `gaussian-transition` between neighbours, and the readings are `ops.dot(h, state)` rows; a position and velocity tracker agrees with the Kalman smoother and the filter log likelihood to rounding (EP is exact on the chain). A random transition matrix, a time-varying one, non-square observations (several readings per step as one vector) and a non-Gaussian reading in a vector scan are not mapped.',
    size: 'medium',
    pages: ['ep-program-tracker', 'ep-program-kalman'],
  },
  {
    id: 'switching',
    title: 'Switching state spaces and regime models',
    area: 'time series',
    status: 'partial',
    what: 'A hidden regime chain over which the dynamics or the observation model switches: hidden Markov models with continuous emissions, switching Kalman filters, change points.',
    needs:
      'Discrete chains and Gaussian emissions run exactly; unknown emission parameters shared across time are loopy (the changepoint program is overconfident when a regime is barely visited), unknown transition rows (a Dirichlet per row, one gate per step) run and agree with Gibbs when every regime is visited, and a gate over a whole sub-chain needs the gate node.',
    size: 'large',
    pages: ['ep-program-changepoint', 'ep-program-changepoint-extended'],
  },
  {
    id: 'volatility',
    title: 'Volatility models beyond one latent chain',
    area: 'time series',
    status: 'partial',
    what: 'Stochastic volatility with inferred persistence runs; leverage effects, jumps and GARCH-style recursions do not.',
    needs:
      'A correlated pair of chains (dense transitions), a jump gate, and for GARCH a deterministic non-Gaussian recursion (`apply` through time).',
    size: 'large',
    pages: ['ep-log-scale'],
  },
  {
    id: 'survival',
    title: 'Survival, reliability and censored data',
    area: 'counts and survival',
    status: 'partial',
    what: 'Waiting times with censoring: Weibull and log-normal regression, competing risks, the Cox model.',
    needs:
      'Weibull and log-normal likelihoods with right censoring are in the registry and a Weibull with a random shape is a pair site (`weibull-shape-learned`); interval censoring, a censoring flag in the model language and the Cox partial likelihood (a ranking over risk sets) are not.',
    size: 'medium',
    pages: [],
  },
  {
    id: 'count-models',
    title: 'Over-dispersed, zero-inflated and hurdle counts',
    area: 'counts and survival',
    status: 'partial',
    what: 'Poisson and negative-binomial regression, zero inflation and hurdle models run; Conway-Maxwell-Poisson and counts with a random exposure do not.',
    needs:
      'The hurdle count (`hurdlePoisson`: a Beta zero probability, a zero-truncated Poisson on a log-linear rate, the coefficients within 0.004 sds of a grid) runs; a count with a random rate shape and a hurdle on a negative binomial (a truncated negative binomial row) are not mapped.',
    size: 'medium',
    pages: ['ep-poisson-log', 'ep-program-zero-inflated', 'ep-hurdle-poisson'],
  },
  {
    id: 'ranking',
    title: 'Rating and ranking systems',
    area: 'ranking and preference',
    status: 'partial',
    what: 'Skills from match results: TrueSkill with draws and teams, Bradley-Terry, Plackett-Luce rankings, time-varying skills.',
    needs:
      'Win/loss and draws (an ordinal probit with the draw margin as cut points) on a skill difference run, teams are sums (affine, supported); a learned draw margin is not mapped, a Plackett-Luce ranking is a softmax over the remaining players (the multiclass site, not yet written as a ranking), drifting skills need a scalar chain per player.',
    size: 'medium',
    pages: ['ep-skill-difference', 'ep-program-skills', 'ep-skill-draws'],
  },
  {
    id: 'factor-models',
    title: 'Matrix factorisation, factor analysis and recommenders',
    area: 'latent structure',
    status: 'not yet',
    what: 'Ratings or measurements as a product of latent user and item vectors (probabilistic matrix factorisation), factor analysis, probabilistic PCA with ARD.',
    needs:
      'The numeric `product` factor exists for scalars; a bilinear factor on whole vectors with packed beliefs, and an ARD precision per factor (the gamma precision rule); many loops, so structured or stochastic EP.',
    size: 'large',
    pages: [],
  },
  {
    id: 'mixed-membership',
    title: 'Mixed membership, topic and community models',
    area: 'latent structure',
    status: 'partial',
    what: 'Topic models, stochastic block models, mixed-membership networks: discrete assignments with Dirichlet-distributed proportions at several levels.',
    needs:
      'Topic models run (a plate of Dirichlet mixtures, a word read through its topic as a gate of conjugate discrete factors: LDA-lite agrees with the exact enumeration within 0.3 sds); a pair of indices (blocks) needs a two-index gate, and a large corpus stochastic EP.',
    size: 'large',
    pages: ['ep-gate-weight-dirichlet', 'ep-topic-model'],
  },
  {
    id: 'mixture-regression',
    title: 'Mixtures of regressions and experts',
    area: 'regression and classification',
    status: 'covered',
    what: 'Different regressions in different regimes of the input or hidden classes: mixtures of linear experts, switching regressions.',
    needs:
      'Runs: `ops.pick(matmul(X, B), z)` is the prediction of the class `z` picks, one gate of Gaussian likelihoods per row (a known index, as in a change point, is the lookup of the branches); a mixture of two lines agrees with the enumeration of the assignments within 0.2 sds. A class that picks a different family (`d.gate(z, [poisson(..), negativeBinomial(..)])`: a count that is a Poisson or an overdispersed one, a regression or a constant) is one gate per row whose branches are the factors each family maps to alone; a branch that is a hurdle or a nested gate, and an index that is a computed random value, are refused. Experts with a class that depends on the input (a gating network) and unknown noise per class are not mapped.',
    size: 'medium',
    pages: ['ep-gate', 'ep-gate-random', 'ep-mixture-regression'],
  },
  {
    id: 'multiclass',
    title: 'Multiclass classification',
    area: 'regression and classification',
    status: 'partial',
    what: 'More than two classes: multinomial probit and logit, multiclass GP classification.',
    needs:
      'The multinomial logit (softmax) and probit run for any number of classes (`multiclass-site`: the product grid up to six, an auxiliary-variable integral beyond, a linear classifier atom with its calibration over seeds and one of eight alternatives), and a multiclass GP as one dense Gaussian vector over inputs and classes (`gp-softmax`); variational message passing stops at six classes, and a joint belief over the classes of an observation is not built.',
    size: 'small',
    pages: ['ep-softmax-regression', 'ep-softmax-regression-learned'],
  },
  {
    id: 'bnn',
    title: 'Bayesian neural networks by probabilistic backpropagation',
    area: 'regression and classification',
    status: 'partial',
    what: 'Weights as Gaussian beliefs, moments pushed through the units.',
    needs:
      '`tanh`, `relu` and the rest compile; a hidden unit can collapse and the posterior is multimodal (sign and permutation symmetries): symmetry breaking in the initial messages and a layer-wise packed form.',
    size: 'large',
    pages: ['ep-tanh'],
  },
  {
    id: 'sparse-recovery',
    title: 'Sparse recovery and compressed sensing',
    area: 'regression and classification',
    status: 'partial',
    what: 'A sparse signal from few linear measurements: spike-and-slab regression runs; approximate message passing, which is EP on a dense linear model, lives in its own domain.',
    needs:
      'Connect the two: express the AMP problem on the compiler and compare EP with the AMP state evolution on the same data.',
    size: 'small',
    pages: ['ep-spike-and-slab', 'ep-program-sparse'],
  },
  {
    id: 'spatial',
    title: 'Spatial and temporal Gaussian processes',
    area: 'spatial and nonparametric',
    status: 'partial',
    what: 'Cox processes, GP classification and regression with learned kernels, sparse GPs with inducing points, GPs on a grid in two dimensions.',
    needs:
      'Fixed-kernel dense GPs run (a one-dimensional Cox process, GP classification); learned kernels (an outer loop), low-rank (inducing point) priors and Kronecker structure for grids.',
    size: 'large',
    pages: ['ep-program-cox', 'ep-gp-probit'],
  },
  {
    id: 'nonparametric',
    title: 'Dirichlet-process and other nonparametric mixtures',
    area: 'spatial and nonparametric',
    status: 'partial',
    what: 'A mixture with an unbounded number of components.',
    needs:
      'The truncated stick-breaking mixture runs (`stickBreaking` as a model operator, a `stick-breaking` factor per assignment: the three breaks of the benchmark within 0.02 sds of the quadrature); a random concentration on the Beta breaks, components with unknown means and other nonparametric priors (Pitman-Yor, the Indian buffet process) are not mapped.',
    size: 'medium',
    pages: ['ep-stick-breaking'],
  },
  {
    id: 'measurement-error',
    title: 'Measurement error, missing data and censored inputs',
    area: 'regression and classification',
    status: 'partial',
    what: 'Regressors observed with noise, missing entries to impute, truncated or censored values.',
    needs:
      'Truncation factors and products of random inputs exist; a Tobit-style truncated-normal observation and a missing-value mechanism (a gate on missingness) need mappings.',
    size: 'medium',
    pages: ['ep-ordered-pair'],
  },
  {
    id: 'calibration',
    title: 'Calibration of EP itself across the whole suite',
    area: 'inference quality',
    status: 'in progress',
    what: "Every setup and program with a calibration panel (many seeds, coverage of the truth by EP's intervals), the cases where it fails collected, and the variants (damping, power and alpha EP, stochastic and structured EP) compared on them.",
    needs:
      'The extended setups and the programs carry the panels; a suite-level table across all of them and the choice of a better default.',
    size: 'medium',
    pages: ['ep-coverage'],
  },
  {
    id: 'scale',
    title: 'Scaling to many observations and large graphs',
    area: 'inference quality',
    status: 'not yet',
    what: 'Thousands of rows and hundreds of latents: the cost of the dense and gridded rules, memory of per-factor messages.',
    needs:
      'Stochastic EP (the engine has it) in the examples, vectorised plates for the likelihood rules, and timing and size limits recorded in the coverage table.',
    size: 'medium',
    pages: [],
  },
]
