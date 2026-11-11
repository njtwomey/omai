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
    status: 'investigating',
    summary:
      'On a bimodal posterior (clutter, mixtures with overlapping components, a regime that is hardly visited) EP commits to one reading and reports a narrow spread: the changepoint program with an unvisited regime gives rate spreads 0.12 to 0.3 of the reference and 70% regime accuracy.',
    plan: 'The explicit gate already reports the probability of each branch; the engine has power EP and alpha-divergences (mass-covering for alpha below one): measure them on the failing programs, expose the one that works as an option and say so on the pages; an EP over mixtures of Gaussians (one component per mode) is the principled longer route.',
    size: 'medium',
    refuses: [],
  },
  {
    id: 'loopy-overconfidence',
    title: 'Loops make the spreads too small',
    kind: 'quality',
    status: 'investigating',
    summary:
      'Scalar latents that share a predictor were 2 to 3 times too narrow (packing them into one joint Gaussian fixed the common case); players who met in a tournament (75 to 87% of the reference sd), hierarchical priors that defeat the packing detector (radon slopes at 0.7 times the truth), and the 2PL item response model (a 17% coverage of the discriminations) remain.',
    plan: 'Pack more structures (plate vectors, hierarchical priors), try structured EP (regions, the engine has them) and damping and power variants on these cases, and keep the calibration panels honest so a regression shows.',
    size: 'medium',
    refuses: [],
  },
  {
    id: 'false-convergence',
    title: 'A degenerate fixed point reported as converged',
    kind: 'quality',
    status: 'investigating',
    summary:
      'Undamped EP on the sparse regression with outliers stopped at a trivial fixed point (every inclusion probability at its prior, the noise precision near zero) and said "converged".',
    plan: 'Report a degenerate or stalled run as such (a status and a reason on the tape) and retry with damping automatically.',
    size: 'small',
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
