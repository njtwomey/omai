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
      title: 'Hyperparameters on a grid: up to about four axes, independent priors',
      kind: 'missing-mapping',
      status: 'open',
      summary:
        'Done: a Gaussian process with random hyperparameters runs by EP inside a grid over them (`epGridMethod`: each node conditions the model, EP runs, the EP evidence weights the node), exact for a Gaussian likelihood. The hyperparameters are the scalar latents a custom function reads and, when the model has no graph without them, the plated vectors it reads (an ARD kernel: one axis per input, one prior for all), so several families sit on one grid: a length scale, a kernel variance and a noise sd (read through a custom function, EP exact in every node). A method over several axes opens on a walk (`order: "walk"`, a best-first flood fill of the lattice neighbours of the nodes within `window` nats of the best, warm starts, 15 points per axis, a window of 7): the cost follows the volume of the posterior and not the product of the axes. Against a brute-force tensor grid of the marginal likelihood: three scalars on a line (`gp-hyperparameters`: 950 nodes, 1 to 2 s) within 3e-4 nats of the evidence and 0.004 reference sds on the means, spreads 0.99 to 1.00; an ARD kernel of two inputs with its variance (three axes, 1160 nodes, 2 s) within 1e-4 nats, means within 0.01 reference sds, spreads 0.99 to 1.02; the same with a noise sd (four axes, 11 points, 2500 nodes, 5 s) within 4e-4 nats, means within 0.03 reference sds, spreads 0.99 to 1.02. What remains: more than four axes (an ARD kernel on five or more inputs: the nodes grow with the volume, ten thousand and more, and a walk steps along the axes only, so a posterior ridge that runs diagonally to the lattice and is thin against its spacing can be cut); a prior that is not the same for every component of a vector, or that depends on another node (a hierarchical hyperparameter); a custom function that is neither known nor affine once the hyperparameters are set; a likelihood that is not Gaussian at several axes (each node is then an EP evidence, whose error is the one of the likelihood; only one axis is checked against a reference, `gp-probit-learned`); and the plated vector joins the grid only when the scalars alone leave no graph (a vector the function is affine in is a random vector for EP).',
      plan: "A numeric site over a hyperparameter inside one EP run for the one-dimensional case (the hyperparameter as a latent whose message is a grid), a sparse grid, a Laplace step or a quasi-random design over five or more axes (the nodes are cheap; the lattice is not), diagonal neighbours or a rotation to the posterior's principal axes for correlated hyperparameters, hierarchical priors on the grid (the log prior of a node is a conditional density given the nodes above it), and a model-layer kernel node (a Gram matrix as a first-class operator on a latent length scale) so that the custom function is not needed.",
      size: 'medium',
      markers: ['hyperparameters', 'opaque to message passing'],
    },
    {
      id: 'matrix-larger',
      title: 'Covariance matrices beyond what the Wishart and the bivariate sd-and-correlation sites reach',
      kind: 'missing-rule',
      status: 'open',
      summary:
        'Done: a Wishart family, the exact conjugate and the Bartlett-quadrature EP rule (`mvn-wishart`, dimensions 2 to 4, the exact update in any dimension), the group form (`mvn-wishart-group`: the vectors of every group, their shared Gaussian mean and the matrix as one factor, one tilt per sweep for all its ports; dimension 3 with eight groups takes about two seconds, dimension 4 about a quarter of a minute, each against an importance-sampling reference to within a few percent of the spread), the normal-Wishart conjugate form, the sd-and-correlation site for a bivariate normal (`mvn-sd-corr`: gamma precisions, an LKJ correlation as a beta variable) and its group form (`mvn-sd-corr-group`: a latent bivariate vector per group with a shared Gaussian mean, gamma precisions and a correlation of their own as one factor, one three-dimensional grid per sweep for all its ports; four groups take under a second, against a dense joint-Gaussian grid the mean, the correlation, the precisions and each vector are matched to within 3% of the spread, the compiler maps a latent `bivariateNormal` over `dims: [group, coef]`), and six atoms. What remains: a Wishart matrix of dimension 5 and above in a non-conjugate setting (the quadrature is a product rule with d (d + 1) / 2 coordinates, three nodes each at d = 4 are 59049 nodes and three at d = 5 would be fourteen million, while two nodes per coordinate leave the spread of the matrix 5% to 15% short), an LKJ prior on a correlation matrix of order 3 and above (three correlations are three coordinates of the numeric site, six with the standard deviations, and the product of beta beliefs on the partial correlations loses their dependence), a bivariate group whose correlation or precisions also enter another factor (the group factor is exact for the tree it is the one non-Gaussian factor of; a second one on the same parameters makes the usual loop), more than two coordinates per vector with standard deviations and a correlation (a correlation matrix of order 3 is the LKJ item above), and a normal-Wishart variable inside a larger model (the compiler maps the Wishart form only, with a random mean as the group form).',
      plan: 'A sparse-grid or Laplace step for the Wishart tilt at dimension 5 and above (the conjugate part of the factor is already taken out, so the remainder is smooth), the canonical-partial-correlation parameterisation with one beta variable per partial correlation and a site over them for the LKJ prior of order 3 (the correlation matrix with known standard deviations first: three coordinates), and the compiler pattern that reads `mu | L ~ N(m, (kappa L)^-1)` as one normal-Wishart variable.',
      size: 'medium',
      markers: ['the quadrature is built for dimensions up to 4'],
    },
    {
      id: 'correlation-strong-few-pairs',
      title: 'A strong bivariate correlation from few pairs: the belief has its spread, not its shape',
      kind: 'quality',
      status: 'open',
      summary:
        'The belief of a bivariate correlation is a beta on (rho + 1) / 2 matched to the tilted mean and variance of the correlation (`mvn-sd-corr`, `match: "moments"`), not to its log moments (the KL projection, still there as `match: "log-moments"`, which had a spread 0.45 to 0.95 of the exact when the correlation is strong and the pairs few): from 2 to 10 pairs, rho 0.3 to 0.99, the mean and the spread are the exact ones (z 0.00, ratio 1.000) and the evidence is exact, in two sweeps. What a beta with two parameters cannot do is the skew: a correlation of 0.99 from five pairs piles against its bound with a long tail toward the interior, and the beta with the right mean and spread has its 5% and 95% points 0.4 of an exact spread off (0.1 to 0.2 at 0.9 and above from other numbers of pairs), and the Kolmogorov distance of its own curve is 0.12 to 0.4 at 0.9 and above where the KL beta had 0.09 to 0.15 (the figure and the atom compare a normal with the belief\'s moments, which for the exact posterior is 0.15 to 0.3 away at 0.9 and above, so the page cannot tell). A Gaussian on the Fisher transform was tried and not kept: its curve is better (0.04 to 0.19), but with a numeric prior site (a beta on its logit) the fixed point of the two sites is off by up to 8% in spread at 0.99 and 0.005 to 0.08 nats in evidence, and takes 10 to 60 sweeps there instead of two.',
      plan: 'A belief for the correlation with a skew parameter that the message algebra can carry (a three-parameter family, or a Gaussian on the Fisher transform whose prior is carried in the factor that reads it rather than as a second site, so that the single-factor case stays exact), matched to the tilted mean, spread and skew.',
      size: 'small',
      markers: [],
    },
    {
      id: 'softmax-many-classes',
      title: 'Softmax beyond six classes: the probit with a power, correlated classes',
      kind: 'missing-rule',
      status: 'open',
      summary:
        'Done: the multiclass site (`multiclass-site`) integrates any number of classes under EP. Up to six free scores it keeps the adaptive Gauss-Hermite product grid; beyond that (or with `integration: "auxiliary"`) it writes the integral so that the scores do not meet: the softmax normaliser is a gamma integral over one auxiliary variable (`S^-p = int t^(p-1) exp(-t S) dt / Gamma(p)`, so given it the factor is a product over the classes, and the power of fractional EP is the shape), the multinomial probit one integral over the observed score plus its noise with the other scores in closed form. Both cost the number of classes, not its power (a softmax of 20 classes takes a few milliseconds a tilted moment), and agree with a brute-force integral up to four classes, the product grid at six, and, for any number, with the exact identities that the classes\' masses add up to one and their tilted moments to the cavities\' (1e-8 and better). Eight alternatives with two shared coefficients (`softmax-choice`) run, with a grid reference; a multiclass Gaussian process is one dense Gaussian vector over the (input, class) pairs gathered with `ops.at` (`gp-softmax`, two inputs and three classes against quadrature), independent processes per class or any covariance across them. Variational message passing runs any number of classes (`_multiclass-variational.ts`: the expected log of the softmax is exact, an integral over the same auxiliary variable of the product of the transforms of the classes; the multinomial probit has a Jensen bound over the other scores, exact when they are known; `softmax-choice` and its learned twin run, converging in 41 and 46 sweeps, the means within 0.19 grid sds and the spreads 0.65 to 0.88 of the reference grid). What remains: the multinomial probit of more than six scores takes the factor whole (power one); the softmax power may not be below 0.03 (the window of the auxiliary variable); the scalar site sends each score its own Gaussian message, so the correlations the labels put between the classes of one input, and the loop through shared coefficients, are not carried where the scores are not a matrix times one vector; where they are (the class slopes of at most six classes, the coefficients that the alternatives of a choice share whatever their number, a multiclass Gaussian process) the compiler makes one `multiclass-row` per observation, a joint message over the scores (the integral runs over the rank of the scores cavity, at most six), and the two coefficients of `softmax-choice` have 0.95 to 0.96 of the exact spread (0.77 to 0.83 before) with the means 0.005 to 0.011 sds off (0.07 to 0.15), the three class slopes 0.995 to 1.000, and the evidence of both 0.03 nats from the grid; the slopes of more than six classes and the auxiliary-variable rule keep one message per class; and the compiler maps one dense vector over one dimension, so a Gaussian process per class is written as one vector of inputs times classes cells, not as a plate of vectors.',
      plan: 'A joint message for the softmax of more than six classes whose cavity is correlated (the auxiliary-variable rule has a product over the classes only for independent cavities), and a compiler pattern that reads a plate of dense Gaussian vectors (one per class) as one vector so that the cells need not be written out.',
      size: 'small',
      markers: ['more than six scores takes a power of one'],
    },
    {
      id: 'multimodal',
      title: 'One Gaussian per variable cannot show two modes: what the multi-start does not reach',
      kind: 'quality',
      status: 'open',
      summary:
        "One EP run commits to one reading of a bimodal or label-ambiguous posterior and reports the spread of that reading alone: on the changepoint program (24 seeds, 72 intervals) the truth is inside the 90% interval for 0.778 of the rates (the sampler 0.931), on the clutter problem the location is 7.8 +- 0.7 where the posterior is 6.4 +- 2.3. The route that reports the other modes is EP from several starts weighted by evidence (`epMultiStartMethod`): the mixture of the distinct fixed points, each weighted by its EP evidence (`diagnostics.modeWeights`), the model's evidence the log sum. A start is set on the port of each label's prior and held for a few damped sweeps; a label whose only other neighbour is its row's likelihood has no cavity but the prior's message, so a start set anywhere else was divided out and a mixture with independent labels (the clutter problem, a mixture of clusters) had one mode whatever the starts. Held to references where the weights are known: the clutter problem's bimodal location against its quadrature (six data sets: the evidence within 0.006 nats, the mean within 0.03 sd, the sd within 4% of the truth's, the second mode's weight 0.001 to 0.28), a two-cluster mean with unequal priors against the enumeration of the assignments (the mass of the labelling with the first mean below the second 0.559, 0.186 and 0.980, the exact values to four digits; the evidence within 1e-4 nats; the labelled means and sds exact to 0.02 sd and 1%), the mixture of the benchmarks with symmetric priors (equal weights, the sorted means within 0.05 sd of the grid). `labels: 'sort'` reads the components in order of their rates, as a data-ordered sampler does: the changepoint program covers 0.889 of the rates with spreads 0.95 of the sampler's where every regime is visited, and the unvisited regime's rate 0.77 to 1.27 of it (one run: 0.05 to 0.15). What remains: a regime visited by a handful of steps (seeds 4, 7, 10 of the 24: spreads 0.2 to 0.35 of the sampler's, errors 2 to 4.5 sds), where every start reaches the same reading and the other readings are not fixed points of EP (the runs cycle); modes that overlap (ten points between -2 and 2 for two cluster means) are one fixed point whose evidence is the single Gaussian's (-19.07 against the exact -18.75); a symmetry of continuous variables (the sign of a loading, with no discrete label) has no discrete message to start from; the weight of a mode is its EP evidence, as good as the Gaussian approximation of that mode and no better (a mode that is an artefact of a loop weighs what it weighs); and models whose evidence is undefined at the fixed point (the circular mixture's likelihood rows, gap `evidence`), which cannot be weighted. A mixture with a label per row (the Gaussian mixture of the pocs, unknown weights, means and spread) has a symmetric fixed point (both components the same, one wide blob: spread 1.8 where the clusters' is 1) that an undamped run from flat messages falls into for data seeds 1 and 4 of 8 (and free means 2 of 3), though the clusters are 3 apart; the damped warm-up (`warmup: auto`, the first three sweeps damped by half where a label selects a component) lets the components form first and the run reaches the clusters (means -1.46 +- 0.14 and 1.52 +- 0.13 against the truth of -1.5 and 1.5, the spread 0.97 against 1, 150 rows). The clusters' fixed point has about a third of the rows with a negative-precision message to the value of the row, so those updates are dropped and its evidence is undefined, while the symmetric one has a finite evidence: a weighing by evidence would pick the wrong one, so `ep-multistart` does not weigh the readings when its first run has no evidence (`diagnostics.unweighed`: the readings without one, equally), and the random and exclusion starts of a mixture, which reach the symmetric point, are set aside. Power EP, alpha-divergences, damping and a damped retry of the starts do not help (docs/probprog/ep-changelog.md section 11, docs/probprog/notes/ep-notes-quality2.md section 1). The `labels` default (`keep`, the model as written) is the owner's choice; docs/probprog/ep.md section 4 gives the measured recommendation.",
      plan: 'An observation of a row (`gaussian-scale`) that accepts a cavity of negative precision for the value of the row (the integral over the value exists when the sum of the precisions is positive; the rule in natural parameters was tried and the run diverged, so it needs a damped or exact treatment of the ambiguous rows, or the labels summed out in a gate with random components) so that the clusters have an evidence and the readings of a mixture can be weighed; the chain summed out given the shared rates (a forward-algorithm site over the three rates, a numeric message of the rates) so that the loop between a shared rate and a discrete state per step is gone; an evidence for the rows of the circular mixture (the `gaussian-gamma-precision` rows whose mean has an indefinite cavity; gap `evidence`) so that it can be multi-started; a start of the continuous variables (the mirror of a sign-symmetric loading, a relabelling of a plate of continuous components with no discrete label) so that a symmetry that has no discrete message is multi-started too.',
      size: 'medium',
      markers: [],
    },
    {
      id: 'network-layer-modes',
      title: 'A network layer: hidden units whose weights have symmetric modes are refused',
      kind: 'quality',
      status: 'open',
      summary:
        "The Bayesian neural network benchmark (three tanh units, sixteen points, five test inputs) is refused by EP, with the reason on the verdict, because a run of it is not a posterior. The weights are not identified (flip the signs of a unit's weights, or permute the units, and the function is the same), so the posterior has many symmetric modes and one Gaussian per weight has no fixed point that means anything. What the run did when it was allowed, each measured against the stored long NUTS run of the function values: at the default knobs 200 sweeps without converging (the beliefs moving by hundreds, thousands of updates dropped as improper, no log evidence, function values that run monotonically away from the data); at damping 0.3 to 0.9, the `reset` policy, the flooding and tree schedules or a power below one it either settled on a fixed point with no information (every weight with mean 0 and the function value 0, with a spread of 1 at damping 0.8 and 0.9, of 0.04 with a power below one: converged, and meaningless) or went on moving with one live unit (an input weight near 2 and an output weight near 0.85) and two collapsed ones, whose function values at the data (-0.9 and 0.8) agree with the long run but whose spreads are 2 to 5 times its spread beside the data and whose extrapolation at the two far inputs is 1.6 to 1.9 spreads off; a start that breaks the symmetry changes which unit lives, not whether the run settles. One unit alone settles. The cause is two things together: the symmetry (the fixed points that exist are the symmetric one and the ones with dead units) and the fine-grained graph (about half of the messages of the `tanh` factors to their inputs had no positive precision at the damped end state, so the affine factor beside such a unit had no cavity and a sixth of the messages of the affine factors were not computable).",
      plan: 'A fixed point per mode and the mixture of them by evidence (the multi-start over the continuous weights named under `multimodal`: a start that gives each unit its own sign and place, the permutation modes related by relabelling), which needs a run per start that settles; for that the whole forward pass of a row as one factor with the weights as the only variables (probabilistic backpropagation: the moments pushed through the layers, one projection per row, so that no hidden variable carries a negative-precision message), or an affine rule that accepts a cavity of negative precision (tried: the messages become computable at damping 0.5 but the sweeps keep moving, so it is not enough alone). The refusal goes when a layer settles on a start of each mode and the mixture is checked against the long NUTS run of the function values.',
      size: 'large',
      markers: ['neural-network layer'],
    },
    {
      id: 'loopy-overconfidence',
      title: 'Loops make the spreads too small',
      kind: 'quality',
      status: 'open',
      summary:
        'Scalar latents that share a predictor were 2 to 3 times too narrow (packing them into one joint Gaussian fixed the common case, and the packed logistic and probit regressions are calibrated). The vectors are packed too (a tied group: the county intercepts of a centred hierarchy beside the floor effect, and the skills of a tournament, a vector that each game reads at two places, each tied to its coordinate of one joint vector): the radon floor effect went from 0.71 to 0.91 of the exact sd (24 counties against NUTS: 0.68 to 0.86), the three-player tournament from 0.74 to 0.85 to 0.99 to 1.00 of the exact grid, the tournament with a learned spread from 0.78 to 0.97 of NUTS. Two sites now send one joint message on the vector their coordinates are made of (`numerics/_vector-site.ts`: the coordinates are a constant matrix times the vector, the integral runs over their rank, the message is a full natural-parameter one). The softmax is `multiclass-row`: the class slopes of a classifier and the coefficients that every alternative of a choice shares are one vector, and plain EP has the exact spreads (three class slopes, ten points: 0.78 to 0.86 of the exact spread at a power of one, 0.95 to 0.97 at the power of 0.5 of the atom with the means 0.07 to 0.21 exact sds off, now 0.995 to 1.000 at either with the means 0.001 to 0.006 off, the evidence 0.03 nats from the grid where it was 0.47; 30 seeds of 24 points with the truth drawn from the prior: the spread 1.00 of the exact one where it was 0.50, the truth inside the 90% interval 0.91 of the time where it was 0.63, the exact posterior 0.91; eight alternatives with two shared coefficients: the spread 0.95 to 0.96 of the exact one where one message per score gave 0.77 to 0.83, and 20 seeds covering 0.900 of the truths against 0.875 before and the exact posterior 0.900), so the power the two softmax atoms still run at is no longer needed (the same numbers at a power of one). The two shapes of a beta outcome are `likelihood-pair-row` on one stacked vector of the two log shapes: 0.47 to 0.65 of the exact spread (the log shapes have a correlation of 0.75 to 0.88) became 1.00 to 1.03, the means 0.08 to 0.16 sds off became exact, in a tenth of the time, and no power is needed (a power of 1.15 was the old workaround). Remaining: the softmax slopes of more than six classes (the vector has more than six coordinates and the integral more than six dimensions: one message per class, 0.4 to 0.75 of the exact spread, still repaired by the `power` knob), a beta whose shape is a gamma variable (not a coordinate of a Gaussian vector), and the other pair likelihoods (a location with a scale, a Student-t, a Weibull shape: the row takes them but each needs its own check before the compiler switches). The non-centred hierarchy (a row reads the county intercepts through a product with a random scale, nothing linear in a vector to pack: the floor effect stays at 0.86 of NUTS, and the mean of the hierarchy, which no row reads, at 0.87 to 0.89). The 2PL item response model was tried and not kept: the logit `a (theta - b)` is a product of two forms of one tied vector (the discriminations, abilities and difficulties, 46 coordinates for 30 students and 8 questions), which `likelihood-pair-row` takes with a logistic-product density; on the hierarchical model the spreads of the discriminations were 0.59 to 0.73 of the NUTS ones and became 0.72 to 0.84 (the difficulties 0.53 to 0.92 and 0.57 to 0.77, the abilities unchanged) in 14 times the time (12 s against 170 s), and with static priors (no hyperparameters) 0.76 to 0.88 became 0.82 to 0.96 (median 0.84 to 0.94, the difficulties 0.84 to 0.85) in 11 times the time: what is left is the loop through the hyperparameters of the hierarchy (the centres 0.7 to 0.8, as in the non-centred case) and the skew of the discriminations, not the bilinear site, so it is not worth that cost and is not in the compiler.',
      plan: 'Structured EP (regions, the engine has them) or a joint vector that includes the hyperparameters for the hierarchies (the non-centred one, the 2PL centres); a joint message for the softmax of more than six classes (the auxiliary-variable rule needs independent cavities, so it needs another form of the integral); switch the other pair likelihoods to the row one at a time against their references; keep the calibration panels honest so a regression shows.',
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
      title:
        "The EP log evidence is exact where EP is, and undefined where a factor's integral against its cavities diverges",
      kind: 'infrastructure',
      status: 'open',
      summary:
        "Every rule of the registry carries a log evidence (each tested against a brute-force integral) and the posterior and the tape (`mp/evidence`) give the model's: against the grid evidence it is within 1e-5 on the conjugate atoms, 1e-3 on the gates, the Student-t, the clutter, the summed-out mixtures, 0.01 to 0.05 on the probit, logistic, Laplace and product atoms, and the numeric priors (half-normal, half-Cauchy, uniform, truncated) are the exact prior, not a projection (uniform exact, half-normal 0.004, half-Cauchy 0.015 nats). A cavity need not be a density by itself (the site of a scale mixture or of a mixture has a negative precision in some direction): the evidence is taken in natural parameters where the integral exists. The affine operators (`linear`, `plus`, `difference`, `times-constant`, `sum`: the output substituted, or an input eliminated when the output is observed) and the Gaussian joint need the cavities proper only together (each against a brute-force two-dimensional integral with an indefinite cavity), the gate takes the cavities of the variables its branches read as they come (the mass of a variable a branch does not read must be proper, and says so), and the plated regression (`probit-plate`, `logistic-plate`) reads its cavity only through the posterior it makes with its rows, so the engine shows it an indefinite one (`improperTarget`), the inner propagation starts from a uniform positive precision on every row, and the plate runs at the fixed point it could not reach before: the learned logistic regression as one plated factor converges in 8 sweeps instead of 35 with an update that could not be computed, its evidence is that of the model with a factor per row to 1e-12 (-26.5201, the grid's -26.4738), and `ep-energy (rows expanded)` is needed no more for it (the beliefs moved to the fixed point of the rows-expanded model: the means by 0.005 to 0.03 sd, the spreads 2% to 3% narrower). The energy of an alpha-divergence run is the power EP energy at the power alpha for the factors that take one (the fixed points of the alpha updates are those of power EP with those factors at alpha): equal to a power EP run's at the same powers to 1e-13, and within 0.003 nats of the quadrature from alpha 1 to 0.1 on the clutter problem. Limits that remain: a factor whose integral against its cavities diverges and has no form in natural parameters has no evidence, and the posterior names it in `diagnostics.evidenceUndefinedAt` (the likelihood of a row with a shared random precision when its mean has an indefinite cavity: the circular mixture's `gaussian-gamma-precision` rows, 23 of 50 in the data form, 7 of 50 in the means form and 3 of 50 when the circle is ignored; the integral over the precision runs to values where the Gaussian one diverges; a product of two Gaussians or a unary function against an indefinite cavity, as in the funnel and the Bayesian neural network, whose integrals are numeric), so those models still have no evidence, and a model with tied groups still falls back on the unpacked model's evidence where the tie has an improper cavity (`mp.ts`).",
      plan: 'Compile a row that reads a mean chosen by a label (the `select` and the `gaussian-gamma-precision` factor) as one gate over the labels, whose cavities are those of the means rather than the indefinite one of the selected value, so that the circular mixture has an evidence and can be weighted by the multi-start; give the numeric product and unary rules an evidence for a cavity that is indefinite but proper together with the factor (a bounded integrand, or the Gaussian integral of its linearisation) where one exists; keep the fallback for tied groups until a tie has an evidence of its own.',
      size: 'small',
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
      '`tanh`, `relu` and the rest run as a unit of a regression (the unary rule); a network layer (units of weights that rows share, multiplied by weights that rows share) is refused, because the posterior is multimodal (sign and permutation symmetries) and the run either settles on the uninformative symmetric fixed point or does not settle (the gap `network-layer-modes`): a fixed point per mode weighted by evidence, and the forward pass of a row as one factor (probabilistic backpropagation) so that no hidden variable carries an improper message.',
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
