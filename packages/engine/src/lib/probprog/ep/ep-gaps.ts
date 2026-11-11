/**
 * What expectation propagation cannot do yet, and what it does less well than it should, as data the coverage page shows and a test
 * holds to account: every setup and benchmark model that EP refuses is named by at least one gap, so a refusal is never an orphan,
 * and every gap says why, how it would be closed, how big that is and where it stands. The list is written by hand from what the
 * audits, the atoms, the programs and the agents' reports found (2026-11-11); when a rule lands the gap is edited or removed and
 * the coverage test says which refused item no longer needs it.
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
  /** The setups and benchmark models EP refuses today because of it (names as in the coverage table). */
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
export const epGaps: readonly EpGap[] = [
  {
    id: 'grid-hyperparameters',
    title: 'Hyperparameters on a grid: a few scalars, one family at a time',
    kind: 'missing-mapping',
    status: 'open',
    summary:
      'A Gaussian process with a random length scale runs by EP inside a grid over the hyperparameters a custom function reads (`epGridMethod`: each node conditions the model, EP runs, the EP evidence weights the node), exact for a Gaussian likelihood. The grid is a product over the hyperparameters, so it serves one to three scalars; an ARD kernel with a length scale per input, a hyperparameter that is a vector or has a prior that depends on another node, and a custom function that is not known or affine once the hyperparameters are set are refused.',
    plan: 'A numeric site over the length scale for the one-dimensional case inside one EP run (the hyperparameter as a latent whose message is a grid), a sparse grid or a Laplace step over several hyperparameters, and a model-layer kernel node (a Gram matrix as a first-class operator on a latent length scale) so that the custom function is not needed.',
    size: 'medium',
    refuses: [],
  },
  {
    id: 'matrix-families',
    title: 'Covariance matrices: Wishart, inverse-Wishart, normal-Wishart, LKJ',
    kind: 'missing-rule',
    status: 'open',
    summary:
      'There is no matrix-valued message family, so a learned covariance (correlated random effects, a multivariate normal with unknown precision) cannot be a latent under EP.',
    plan: 'A Wishart family with an exact normal-Wishart rule for the conjugate case, a model-layer distribution for it, and the Cholesky-with-correlation parameterisation (sds with gamma precisions and a numeric prior on the correlation) as the practical route for 2 by 2 and 3 by 3.',
    size: 'large',
    refuses: [],
  },
  {
    id: 'softmax-many-classes',
    title: 'Softmax beyond six classes, and a multiclass Gaussian process',
    kind: 'missing-rule',
    status: 'open',
    summary:
      'The multiclass site (`multiclass-site`: a softmax or multinomial probit of Gaussian class scores, adaptive Gauss-Hermite over all the free scores) integrates up to six classes, and the classifier atoms use one slope per class on a scalar covariate. More classes, and a Gaussian process whose latent function is a vector over the classes (a joint Gaussian across them), are not mapped.',
    plan: 'A delta-method or Laplace-based site (Knowles and Minka) beyond six classes, where the product grid is too large; a vector-valued belief per observation (one Gaussian over the K scores) for the multiclass GP.',
    size: 'medium',
    refuses: [],
  },
  {
    id: 'ordinal-random-cuts',
    title: 'Ordinal cut points that are random, and a Student-t with random degrees of freedom and scale',
    kind: 'missing-rule',
    status: 'open',
    summary:
      'The ordinal probit and logit and the asymmetric Laplace are model-layer distributions mapped to the registry (`orderedProbit`, `orderedLogistic`, `asymmetricLaplace`, the draw margin of a tournament among them) with the cut points known; random cut points (a prior on increasing thresholds, a learned draw margin) are refused with the reason. The two-variable site takes one random scale or shape beside the location, so a Student-t whose degrees of freedom and scale are both random (three variables) is refused too.',
    plan: 'A three-axis version of the grid in `_grid.ts` (the pair rule with a third coordinate) serves a Student-t with random df and scale; the cut points as a numeric site over the predictor and the cut points on their log gaps (one gap for a draw margin).',
    size: 'medium',
    refuses: [],
  },
  {
    id: 'gate-node',
    title: 'A gate over likelihoods of different families: no explicit gate node',
    kind: 'missing-mapping',
    status: 'open',
    summary:
      'A discrete latent that picks among branch predictions of one likelihood runs: `ops.pick(values, index)` is the gate over Gaussian regressions on a shared design (mixtures of regressions and switching regressions, one engine gate of Gaussian likelihoods per row, agreeing with the enumeration of the class assignments within 0.2 sds), and a hurdle count runs as a family (a conjugate Bernoulli on the zero part, a zero-truncated Poisson site on the positive part). What is not mapped is a gate whose branches are different families chosen by a latent that is not determined by the data (a Poisson against a negative binomial, a regression against a constant), because a distribution of the model language has no nested distributions.',
    plan: 'A `gate` distribution family whose parameters are the index and, per branch, the family and its parameters (flattened into the parameter table so that every analysis still sees the expressions), compiled to the engine gate with one inner operator per branch family; pick and the hurdle count then become special cases.',
    size: 'medium',
    refuses: [],
  },
  {
    id: 'multimodal',
    title: 'One Gaussian per variable cannot show two modes',
    kind: 'quality',
    status: 'investigating',
    summary:
      "On a bimodal or label-ambiguous posterior (clutter, overlapping mixture components, a regime that is hardly visited) one EP run commits to one reading and reports a narrow spread: the changepoint program with an unvisited regime gives rate spreads 0.05 to 0.3 of the reference. Power EP, alpha-divergences, damping and stochastic EP do not restore calibration (docs/probprog/ep-plan.md section 11). EP from several starts weighted by evidence (`epMultiStartMethod`, section 15) does for the regime that nobody visits: the spread of its rate goes from 0.05 to 0.15 of the sampler's to 0.65 to 0.94 on seeds 2, 5 and 8, the means within 0.2 sampler sds. It also finds the relabelled modes of the visited regimes (the broad priors only weakly prefer one labelling, 1.4 to 8 nats), so on seed 1, where one run is exact, the mixture is wider than the sampler that was started in the data-ordered mode (the truth is inside every interval); the spreads of the other rates, which the sampler also gets from label switching, are not recovered on seeds 2 and 8. The same limit shows in a mixture whose components have a mean and a spread each (a gate of `gaussian-scale` branches): the means are within 0.15 sds of a grid but their spreads 0.57 to 0.71 of it, one Gaussian per variable against a posterior that is a mixture over the labels of the ambiguous readings.",
    plan: 'Starts that permute the labels of the regimes (the relabellings of a mode have its evidence, so each adds a copy of the mode) and starts from the data (the regimes ordered by the counts they explain); a mixture-of-Gaussians belief per variable so that one run holds several modes; the explicit gate already reports the probability of each branch.',
    size: 'medium',
    refuses: [],
  },
  {
    id: 'loopy-overconfidence',
    title: 'Loops make the spreads too small',
    kind: 'quality',
    status: 'open',
    summary:
      'Scalar latents that share a predictor were 2 to 3 times too narrow (packing them into one joint Gaussian fixed the common case); players who met in a tournament (75 to 87% of the reference sd), the three class slopes of a softmax classifier that share every observation (0.78 to 0.86 of the exact sd with ten points, 0.54 to 0.59 with 24) and the two shapes of beta outcomes (0.65 to 0.8), hierarchical priors that defeat the packing detector (radon slopes at 0.7 times the truth), and the 2PL item response model (a 17% coverage of the discriminations) remain.',
    plan: 'Pack more structures (plate vectors, hierarchical priors, the class slopes of the multiclass site as one joint vector) and try structured EP (regions, the engine has them) and keep the calibration panels honest so a regression shows. Damping, power and alpha variants make no difference to the tournament skills (measured).',
    size: 'medium',
    refuses: [],
  },
  {
    id: 'sv-parameters',
    title: 'Stochastic volatility with an unknown level, persistence and innovation precision is not calibrated',
    kind: 'quality',
    status: 'open',
    summary:
      "With the three parameters unknown the model compiles and EP runs in seconds, but the truth is inside EP's 90% interval for only 37% of levels, 60% of persistences and 47% of precisions over 30 seeds, with errors of 10 to 40 sds: the level and the persistence are strongly coupled through the whole chain and one Gaussian per parameter cannot hold that.",
    plan: 'A joint rule over the level and the persistence (a block update on the chain), or EP inside a grid over the persistence weighted by evidence; until then the setup is not offered as a learned-parameters page.',
    size: 'large',
    refuses: [],
  },
  {
    id: 'non-centred-scale',
    title: 'A non-centred scale (z / sqrt(precision)) is read through independent coefficient marginals',
    kind: 'quality',
    status: 'open',
    summary:
      'A coefficient written `z / sqrt(g(x))` (the non-centred Bayesian GAM, penalty prior with log-normal hyperpriors, and ARD) runs under EP through the `apply-product` factor, one per coefficient, with the curve within 0.005 response sd of the exact reference and the smoothing within 0.4 reference sd, but the posterior hands back each coefficient as its own marginal, so the credible band is 17% too wide (the dense gamma-smoothing form, whose coefficients are one vector, is within 2%). The log-normal form reaches its fixed point in about 80 sweeps without the tolerance ever being met (quadrature noise), and variational message passing has no rule for the factor (the non-conjugate message of a scale collapses the noise to a point).',
    plan: 'A dense variant: `apply-product` on the whole coefficient vector against a Gaussian log smoothing parameter (the grid of `gaussian-mv-precision-scale` over a Gaussian instead of a gamma cavity), so the posterior keeps the correlations; a tolerance relative to the quadrature accuracy; a variational rule for the factor from the exact conjugate one at a fixed scale.',
    size: 'medium',
    refuses: [],
  },
  {
    id: 'evidence',
    title: 'The EP log evidence is exact where EP is, and has two known limits',
    kind: 'infrastructure',
    status: 'open',
    summary:
      "Every rule of the registry carries a log evidence (each tested against a brute-force integral) and the posterior and the tape (`mp/evidence`) give the model's: against the grid evidence it is within 1e-5 on the conjugate atoms, 1e-3 on the gates, the Student-t, the clutter, the summed-out mixtures, 0.01 to 0.05 on the probit, logistic, Laplace and product atoms, and the numeric priors (half-normal, half-Cauchy, uniform, truncated) are the exact prior, not a projection (uniform exact, half-normal 0.004, half-Cauchy 0.015 nats). Two limits remain: a fixed point with a factor whose cavity is improper has no evidence (the learned logistic regression as one plated factor: the scale mixture's site has a negative precision, so the cavity of the plate is indefinite and its integral diverges; the posterior says which factor in `diagnostics.evidenceUndefinedAt`, and `plates: 'expand'` gives it, 0.05 nats from the grid); an alpha-divergence run has no EP energy.",
    plan: 'Give the plated factors an evidence from a sequential pass over their rows when their cavity is improper (or compile small plates as rows when the weights have a scale mixture), and give the alpha runs the energy of their own divergence.',
    size: 'medium',
    refuses: [],
  },
  {
    id: 'variational-rules',
    title: 'Variational message passing: what it cannot run and where the mean field is poor',
    kind: 'infrastructure',
    status: 'open',
    summary:
      'Variational message passing runs 78 of the 86 atoms (36 before the non-conjugate message by quadrature, the deterministic factors that define their output and the gates): the numeric likelihoods, scales, shapes, softmax, products, maxima, unary functions, gamma products and selects all have a mean-field message, each checked against the brute-force bound. What is left: the hard comparisons (`greater-than`, `is-between`, `spin`: KL(q || p) is infinite for a Gaussian q, the probit link is their variational form), the Weibull with a random shape (its expected log is infinite under a Gaussian or gamma belief), the custom functions of a random value, and the mixtures, where the mean field settles on the optimum that explains everything as clutter or as the spike (clutter and spike-and-slab) and the banana posterior of a product that a belief per variable cannot follow.',
    plan: 'Start the mean field of a mixture from the data or from several restarts (as `epMultiStartMethod` does for EP), give the hard comparisons their probit form in the model compiler, and keep the coverage floor (`vmpCoverage`) going up.',
    size: 'small',
    refuses: [],
  },
  {
    id: 'docs-drift',
    title: 'The prior atlas can drift from the code',
    kind: 'infrastructure',
    status: 'open',
    summary:
      'The atlas tables (`docs/probprog/ep-priors.md`) were written from probes of the library; when a rule lands the table is not updated automatically.',
    plan: 'Generate the status column from the coverage data in a test (fail when the document and the coverage disagree).',
    size: 'small',
    refuses: [],
  },
]

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
      'Runs: `ops.pick(matmul(X, B), z)` is the prediction of the class `z` picks, one gate of Gaussian likelihoods per row (a known index, as in a change point, is the lookup of the branches); a mixture of two lines agrees with the enumeration of the assignments within 0.2 sds. Experts with a class that depends on the input (a gating network) and unknown noise per class are not mapped.',
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
      'The multinomial logit (softmax) and probit of up to six classes run (`multiclass-site`, a linear classifier atom with its calibration over seeds); more classes need a delta-method site and a multiclass GP a vector-valued belief over the classes.',
    size: 'medium',
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
