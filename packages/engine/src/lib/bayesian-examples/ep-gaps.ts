/**
 * What expectation propagation cannot do yet, and what it does less well than it should, as data the coverage page shows and a test
 * holds to account: every setup and benchmark model that EP refuses is named by at least one gap, so a refusal is never an orphan,
 * and every gap says why, how it would be closed, how big that is and where it stands. The list is written by hand from what the
 * audits, the atoms, the programs and the agents' reports found (2026-11-11); when a rule lands the gap is edited or removed and
 * the coverage test says which refused item no longer needs it.
 *
 * @index module The gaps and limits of expectation propagation, each with its plan, size and status, tied to what it refuses.
 */

/** What kind of shortfall: no rule, no compiler mapping, a quality problem of EP itself, an infrastructure or integration task. */
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
 * import { epGaps } from 'omai-engine/bayesian-examples'
 * epGaps.every((g) => g.plan.length > 0 && g.summary.length > 0) // => true
 * ```
 */
export const epGaps: readonly EpGap[] = [
  {
    id: 'kernel-hyperparameters',
    title: 'Kernel hyperparameters as random values',
    kind: 'missing-mapping',
    status: 'queued',
    summary:
      'A Gaussian process whose length scale or amplitude is a latent builds its Gram matrix from a random value, and the Cholesky of it is an opaque custom function: the graph cannot be compiled. With the kernel fixed (a constant covariance) GP classification already runs.',
    plan: 'Run EP inside a grid over the hyperparameters (the log evidence of each EP run weights the grid, as the gp domain does for its marginal likelihood), so the compiler sees a constant Gram matrix at each node; later a numeric site over the length scale if one grid axis is not enough. Needs the EP log evidence of the dense prior and the likelihood sites.',
    size: 'medium',
    refuses: ['gaussian-process-regression', 'gaussian-process-non-centred'],
  },
  {
    id: 'stick-breaking',
    title: 'Stick-breaking and other opaque custom transforms',
    kind: 'missing-mapping',
    status: 'queued',
    summary:
      'The stick-breaking weights of a Dirichlet-process mixture are written as a custom function of Beta sticks, which message passing cannot see through.',
    plan: 'Make the transform first-class in the model layer (a chain of products of Beta sticks and their complements, each a `product`-style factor) or truncate to a Dirichlet; the gate and the discrete operators already take the weights.',
    size: 'medium',
    refuses: ['stick-breaking-mixture'],
  },
  {
    id: 'topic-models',
    title: 'Topic models: Dirichlet arrays and plated categoricals',
    kind: 'missing-mapping',
    status: 'queued',
    summary:
      'The LDA model has a Dirichlet over topics per document and per topic, with a categorical word under a categorical topic. The compiler reports "dirichlet needs a size" (it now refuses cleanly instead of throwing); the conjugate operators exist but the plates of Dirichlets are not mapped.',
    plan: 'Map a plate of Dirichlet latents and a categorical read through a discrete index to `discrete-dirichlet` factors per document; a large graph, so run it with stochastic EP (the engine has it) rather than full EP.',
    size: 'large',
    refuses: ['lda-lite'],
  },
  {
    id: 'two-random-shapes',
    title: 'A Beta or Dirichlet with several random shapes',
    kind: 'missing-rule',
    status: 'open',
    summary:
      'A Beta whose two shapes are separate random variables needs the tilted moments of a two-dimensional integral; only one shared random strength (a gamma or log-normal multiple) is mapped.',
    plan: 'A two-variable numeric rule on a grid, like the negative-binomial dispersion operator; the same rule serves a random Weibull shape and a Student-t with learned degrees of freedom.',
    size: 'medium',
    refuses: ['prior-beta-two-shapes'],
  },
  {
    id: 'learned-df',
    title: 'Random degrees of freedom, a random Weibull shape',
    kind: 'missing-rule',
    status: 'open',
    summary:
      'A Student-t with learned degrees of freedom and a Weibull with a learned shape have no rule; the degrees of freedom stay a fixed number under EP.',
    plan: 'The same two-variable numeric rule as above (the likelihood site gains a second Gaussian or gamma variable).',
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
    id: 'softmax',
    title: 'Softmax and logistic-normal categoricals',
    kind: 'missing-rule',
    status: 'open',
    summary:
      'A categorical with probabilities given by a softmax of Gaussian logits needs a (K-1)-dimensional tilted moment; only Dirichlet probabilities (exact) and binary logistic links are mapped.',
    plan: 'A delta-method or Laplace-based site (Knowles and Minka style) for the softmax, exact quadrature for K = 3.',
    size: 'medium',
    refuses: [],
  },
  {
    id: 'ordinal-asymmetric',
    title: 'Ordinal probit, asymmetric Laplace, a draw margin: no model-layer distribution',
    kind: 'missing-mapping',
    status: 'open',
    summary:
      'The likelihood registry has rules for ordinal probit and the asymmetric Laplace, but no distribution of the model language compiles to them, so they cannot be written in a program (this is also why the tournament has no draw margin).',
    plan: 'Add the distributions to the model layer with reference fixtures, and map them to the registry entries.',
    size: 'small',
    refuses: [],
  },
  {
    id: 'gate-node',
    title: 'A model-layer gate node and gates over non-Gaussian components',
    kind: 'missing-mapping',
    status: 'open',
    summary:
      'The engine gate takes any branch, but the compiler maps lookups and `where` over Gaussian forms only; a gate over a mix of known values and non-Gaussian random components is refused, and there is no explicit `gate` node in the model language.',
    plan: 'A `gate` node with branches that are sub-models, compiled to the engine gate; start with Poisson and Gaussian branches (zero inflation, change points).',
    size: 'medium',
    refuses: [],
  },
  {
    id: 'multimodal',
    title: 'One Gaussian per variable cannot show two modes',
    kind: 'quality',
    status: 'open',
    summary:
      'On a bimodal or label-ambiguous posterior (clutter, overlapping mixture components, a regime that is hardly visited) EP commits to one reading and reports a narrow spread: the changepoint program with an unvisited regime gives rate spreads 0.05 to 0.3 of the reference and 70% regime accuracy. Measured: power EP, alpha-divergences (alpha down to 0.3), damping and stochastic EP do not restore calibration (docs/probprog/ep-plan.md section 11): the posterior over which steps belong to a rarely visited regime is multimodal and one message per variable cannot hold that.',
    plan: 'Several EP runs from different starts weighted by their evidence (needs the log evidence of the rules), or a mixture-of-Gaussians belief per variable; for the label problem a Gibbs or SMC pass as the reference and a warning on the page; the explicit gate already reports the probability of each branch.',
    size: 'large',
    refuses: [],
  },
  {
    id: 'loopy-overconfidence',
    title: 'Loops make the spreads too small',
    kind: 'quality',
    status: 'open',
    summary:
      'Scalar latents that share a predictor were 2 to 3 times too narrow (packing them into one joint Gaussian fixed the common case); players who met in a tournament (75 to 87% of the reference sd), hierarchical priors that defeat the packing detector (radon slopes at 0.7 times the truth), and the 2PL item response model (a 17% coverage of the discriminations) remain.',
    plan: 'Pack more structures (plate vectors, hierarchical priors) and try structured EP (regions, the engine has them) and keep the calibration panels honest so a regression shows. Damping, power and alpha variants make no difference to the tournament skills (measured).',
    size: 'medium',
    refuses: [],
  },
  {
    id: 'improper-start',
    title: 'Improper first messages are patched operator by operator',
    kind: 'quality',
    status: 'open',
    summary:
      "A flat cavity (a variable with no information yet) is read as a very wide Gaussian inside several operators (`cavityOf`, `widened`); start messages replace the compiler's vague priors but the per-operator reads remain.",
    plan: 'One policy in the engine: an operator that cannot read a flat cavity sends nothing and the engine schedules it later, with the start messages as the only initialisation.',
    size: 'small',
    refuses: [],
  },
  {
    id: 'projected-priors',
    title: 'Projected priors are approximations of the prior',
    kind: 'quality',
    status: 'open',
    summary:
      'Half-normal, half-Cauchy, uniform and truncated priors are carried as a Gaussian on a log scale or on the value, so the graph holds the projected prior, not the original (the spread of a variance with a gamma scale hyperprior comes out about 22% too wide).',
    plan: 'Keep the exact prior as a numeric site that is re-projected against the current cavity each sweep (a prior is one factor like any other), instead of projecting it once.',
    size: 'medium',
    refuses: [],
  },
  {
    id: 'evidence',
    title: 'Log evidence for the new numeric rules',
    kind: 'infrastructure',
    status: 'open',
    summary:
      'Several of the new operators (the gamma-precision rule, the likelihood sites, the gates) return moments but not their log evidence, so the evidence of a whole model, hyperparameter learning by maximising it and model comparison are unavailable for them.',
    plan: 'Add `logEvidence` to each rule (the tilted mass is already computed) and test it against the brute-force integral.',
    size: 'medium',
    refuses: [],
  },
  {
    id: 'variational-rules',
    title: 'Variational message passing lags behind EP',
    kind: 'infrastructure',
    status: 'open',
    summary:
      'The new rules (mixture sites, likelihood sites, gates over random components, gamma products) have EP rules only, so the variational comparison is missing for them.',
    plan: 'Add the mean-field rules where they exist in closed form and say so where they do not.',
    size: 'medium',
    refuses: [],
  },
  {
    id: 'integration',
    title: 'The EP suite lives in the examples folder',
    kind: 'infrastructure',
    status: 'queued',
    summary:
      'The rules, the compiler mappings and the model families are in the library proper (`message-passing`, `probprog/model`, `probprog/inference`); the setups, programs, coverage and recovery helpers are in `bayesian-examples`, next to the example pages, where they do not belong.',
    plan: 'Move them to `probprog/ep` (they use the compiler and the methods, so they cannot sit in `message-passing`), keep a re-export while the pages move, and update the structure document; do it once the agents editing the atoms have finished, to avoid conflicts.',
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
 * import { epDirections } from 'omai-engine/bayesian-examples'
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
    status: 'partial',
    what: 'A latent vector state with a linear transition and linear observations: tracking, dynamic regression, random-walk coefficients.',
    needs:
      'Scalar AR(1) chains run; a vector-valued `scan` with matrix transitions needs dense Gaussian transition factors (the dense Gaussian family and the matrix-times-vector operators exist).',
    size: 'medium',
    pages: ['ep-program-tracker'],
  },
  {
    id: 'switching',
    title: 'Switching state spaces and regime models',
    area: 'time series',
    status: 'partial',
    what: 'A hidden regime chain over which the dynamics or the observation model switches: hidden Markov models with continuous emissions, switching Kalman filters, change points.',
    needs:
      'Discrete chains and Gaussian emissions run exactly; unknown emission parameters shared across time are loopy (the changepoint program is overconfident when a regime is barely visited) and a gate over a whole sub-chain needs the gate node.',
    size: 'large',
    pages: ['ep-program-changepoint'],
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
      'Weibull and log-normal likelihoods with right censoring are in the registry; a random shape, interval censoring and the Cox partial likelihood (a ranking over risk sets) are not.',
    size: 'medium',
    pages: [],
  },
  {
    id: 'count-models',
    title: 'Over-dispersed, zero-inflated and hurdle counts',
    area: 'counts and survival',
    status: 'partial',
    what: 'Poisson and negative-binomial regression and zero inflation run; hurdle models, Conway-Maxwell-Poisson and counts with a random exposure do not.',
    needs:
      'A gate over a Poisson and a truncated Poisson branch (the gate node), and a rule for a count with a random rate shape.',
    size: 'medium',
    pages: ['ep-poisson-log', 'ep-program-zero-inflated'],
  },
  {
    id: 'ranking',
    title: 'Rating and ranking systems',
    area: 'ranking and preference',
    status: 'partial',
    what: 'Skills from match results: TrueSkill with draws and teams, Bradley-Terry, Plackett-Luce rankings, time-varying skills.',
    needs:
      'Win/loss on a skill difference runs; draws need the ordinal-probit distribution, teams are sums (affine, supported), a Plackett-Luce ranking needs a softmax factor, drifting skills a scalar chain per player.',
    size: 'medium',
    pages: ['ep-skill-difference', 'ep-program-skills'],
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
    status: 'not yet',
    what: 'Topic models, stochastic block models, mixed-membership networks: discrete assignments with Dirichlet-distributed proportions at several levels.',
    needs:
      'The conjugate discrete-Dirichlet operators exist; plates of Dirichlets and discrete indices through a lookup need the compiler (the LDA gap), and a pair of indices (blocks) a two-index gate.',
    size: 'large',
    pages: ['ep-gate-weight-dirichlet'],
  },
  {
    id: 'mixture-regression',
    title: 'Mixtures of regressions and experts',
    area: 'regression and classification',
    status: 'partial',
    what: 'Different regressions in different regimes of the input or hidden classes: mixtures of linear experts, switching regressions.',
    needs:
      'The gate over Gaussian branches runs; branches that are regressions on a shared design need the gate node over sub-models.',
    size: 'medium',
    pages: ['ep-gate', 'ep-gate-random'],
  },
  {
    id: 'multiclass',
    title: 'Multiclass classification',
    area: 'regression and classification',
    status: 'not yet',
    what: 'More than two classes: multinomial probit and logit, multiclass GP classification.',
    needs:
      'A softmax or multinomial-probit factor (a (K-1)-dimensional tilted moment: quadrature for K = 3, a delta method beyond).',
    size: 'medium',
    pages: [],
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
    status: 'not yet',
    what: 'A mixture with an unbounded number of components.',
    needs:
      'Stick-breaking as a first-class transform (the stick-breaking gap), truncated; the gate and the discrete operators take the weights.',
    size: 'medium',
    pages: [],
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
