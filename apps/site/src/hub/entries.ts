import type { IterativeEstimator } from 'mdml-engine/learn'
import { kmeans } from 'mdml-engine/methods/cluster'
import { kalmanTracking, odeSystems } from 'mdml-engine/methods/dynamics'
import { eigenPower, eigenQr } from 'mdml-engine/methods/linalg'
import { bayesianRegression, lassoPath, logisticRegression, ridge } from 'mdml-engine/methods/linear'
import { bgmm, gmm, moe } from 'mdml-engine/methods/mixture'
import { transformSampling } from 'mdml-engine/methods/probability'
import { mcmc } from 'mdml-engine/methods/sampling'
import { annealing, gridSearch } from 'mdml-engine/methods/search'
import { hmm } from 'mdml-engine/methods/sequence'
import { bayesianOptimisation } from 'mdml-engine/methods/tune'
import type { Method, Subscription } from 'mdml-engine/trace'
import type { Panel } from 'mdml-render/react'

/**
 * The hub's entries are data. What the tape logs and what the page shows are separate: the session may record much
 * more than the layout draws, and only the layout (a list of `mdml-render`'s session panels) decides what appears.
 */
interface EntryBase {
  /** `area/name`, also the page path under /hub. */
  readonly id: string
  readonly title: string
  readonly summary: string
  /** The panels the page draws, in order. */
  readonly layout: readonly Panel[]
  /** The working mode the page starts in (default live). */
  readonly mode?: 'live' | 'eager'
  /** What the session logs (default the method's defaults). */
  readonly logs?: Subscription
  /** The step budget the page starts with (default 300). */
  readonly steps?: number
}

/**
 * A hub entry is data: an id, words, a layout, and what runs. Either an estimator on the contract with the dataset
 * collections it runs on, or a method of nothing but its configuration (a process, an equation, a sampler, a search).
 */
export type HubEntry = EntryBase &
  (
    | {
        // biome-ignore lint/suspicious/noExplicitAny: entries of every particular share one list
        readonly estimator: IterativeEstimator<any, any, any, any>
        /** Ids in `datasetCollections` the entry can be run on. */
        readonly datasets: readonly string[]
      }
    | {
        // biome-ignore lint/suspicious/noExplicitAny: entries of every method share one list
        readonly method: Method<any, any>
      }
  )

/** An area of the hub: the first segment of its entries' ids, with a title and a line for the index. */
export interface HubArea {
  readonly id: string
  readonly title: string
  readonly blurb: string
}

/** The areas, in reading order; an entry whose area is not listed is shown under its id. */
export const AREAS: readonly HubArea[] = [
  { id: 'cluster', title: 'Clustering', blurb: 'Rows grouped by nearness, a step at a time.' },
  {
    id: 'mixture',
    title: 'Mixture models',
    blurb: 'Components fitted by expectation-maximisation or variational inference, and a gate over experts.',
  },
  {
    id: 'linear',
    title: 'Linear models',
    blurb: 'Classifiers, regressors and paths over a design matrix, Bayesian or not.',
  },
  { id: 'sequence', title: 'Sequences', blurb: 'Hidden-state models of a sequence.' },
  { id: 'dynamics', title: 'Dynamics', blurb: 'Equations, processes and filters in time.' },
  { id: 'sampling', title: 'Sampling', blurb: 'Markov chains walking a target.' },
  { id: 'probability', title: 'Probability', blurb: 'The ideas of probability watched.' },
  { id: 'linalg', title: 'Linear algebra', blurb: 'The iterations of numerical linear algebra.' },
  { id: 'search', title: 'Search', blurb: 'Path and local search, expansion by expansion.' },
  { id: 'tune', title: 'Tuning', blurb: 'Choosing parameters trial by trial.' },
]

/** The entries of the hub. */
export const ENTRIES: readonly HubEntry[] = [
  {
    id: 'cluster/kmeans',
    title: 'k-means',
    summary: 'Lloyd’s algorithm: assign rows to the nearest centre, move centres to the means, repeat.',
    estimator: kmeans,
    datasets: ['clustering2d'],
    layout: [
      { kind: 'rows', overlay: 'assignments' },
      { kind: 'curves', paths: ['solver/inertia'] },
    ],
  },
  {
    id: 'mixture/gmm',
    title: 'Gaussian mixture',
    summary:
      'Expectation-maximisation: responsibilities of each component for each row, then new weights, means and covariances.',
    // Random responsibilities to start, so EM is seen doing the work (k-means would hand it the answer).
    estimator: gmm.with({ init: 'random' }),
    datasets: ['clustering2d'],
    layout: [
      { kind: 'rows', overlay: 'components' },
      { kind: 'probe', path: 'model/density', args: { resolution: 40 }, title: 'log density' },
      { kind: 'curves', paths: ['solver/logLikelihood'] },
    ],
  },
  {
    id: 'mixture/bayesian-gmm',
    title: 'Bayesian Gaussian mixture',
    summary:
      'Variational inference with a sparse prior on the weights: eight components start, and the ones the data does not need die off.',
    // From a k-means start: eight pieces of three blobs, merged and switched off by the prior. A random start leaves
    // the components alike, and variational inference then merges two of three separated blobs in nine seeds of ten.
    estimator: bgmm,
    datasets: ['clustering2d'],
    layout: [
      { kind: 'rows', overlay: 'components' },
      { kind: 'series', path: 'solver/weights', title: 'component weights (the unneeded die off)' },
      { kind: 'curves', paths: ['solver/lowerBound'] },
    ],
  },
  {
    id: 'mixture/moe',
    title: 'Mixture of experts',
    summary:
      'Linear experts and a gate that shares the rows among them: the E-step says which expert owns each row, the M-step refits each expert by weighted least squares and moves the gate.',
    estimator: moe,
    datasets: ['regression1d'],
    layout: [
      { kind: 'rows', overlay: 'experts' },
      { kind: 'probe', path: 'model/gating', title: 'gate probabilities' },
      { kind: 'curves', paths: ['solver/logLikelihood'] },
    ],
  },
  {
    id: 'linear/logistic-regression',
    title: 'Logistic regression',
    summary:
      'A linear classifier fitted by iteratively reweighted least squares; the decision surface follows the fit.',
    estimator: logisticRegression,
    datasets: ['classification2d'],
    layout: [
      { kind: 'rows', overlay: 'surface' },
      { kind: 'curves', paths: ['solver/objective'] },
    ],
  },
  {
    id: 'linear/ridge',
    title: 'Ridge regression',
    summary: 'Least squares with an L2 penalty, solved in closed form.',
    estimator: ridge,
    datasets: ['regression1d'],
    layout: [{ kind: 'rows', overlay: 'fit' }],
  },
  {
    id: 'linear/bayesian-regression',
    title: 'Bayesian linear regression',
    summary:
      'A polynomial with a Gaussian prior on its coefficients, one training row per step: the predictive band tightens and the evidence settles.',
    estimator: bayesianRegression,
    datasets: ['regression1d'],
    layout: [
      { kind: 'rows', overlay: 'band' },
      { kind: 'curves', paths: ['solver/logEvidence'] },
      { kind: 'curves', paths: ['solver/meanPredictiveStd'] },
    ],
  },
  {
    id: 'linear/lasso-path',
    title: 'Lasso path',
    summary: 'The lasso along a decreasing penalty grid, each penalty warm-started from the last: play the path.',
    estimator: lassoPath,
    datasets: ['regression1d'],
    layout: [
      { kind: 'projection', name: 'regularisationPath' },
      { kind: 'curves', paths: ['solver/gap', 'solver/alpha'], title: 'duality gap and penalty' },
    ],
  },
  {
    id: 'sampling/mcmc',
    title: 'Markov chain Monte Carlo',
    summary:
      'A chain walks a two-dimensional target by random-walk Metropolis or Hamiltonian Monte Carlo: watch it mix.',
    method: mcmc,
    steps: 2000,
    layout: [
      {
        kind: 'trajectory',
        paths: ['chain/position'],
        points: true,
        over: { path: 'target/density' },
        title: 'the chain over the target',
      },
      { kind: 'series', path: 'chain/position', names: ['x₀', 'x₁'], title: 'trace plot' },
      { kind: 'curves', paths: ['chain/acceptance'], title: 'acceptance rate so far' },
      { kind: 'curves', paths: ['chain/logDensity'] },
    ],
  },
  {
    id: 'probability/transform-sampling',
    title: 'Sampling by transformation',
    summary:
      'Uniform or normal draws pushed through a map become samples of a harder law; the change of variables gives the density they pile up under.',
    method: transformSampling,
    steps: 200,
    layout: [
      { kind: 'probe', path: 'transform/picture', title: 'up to the map, across to the target' },
      { kind: 'curves', paths: ['samples/ks'], title: 'distance to the exact law' },
      { kind: 'curves', paths: ['samples/mean'], title: 'mean of the pushed samples' },
    ],
  },
  {
    id: 'linalg/power-iteration',
    title: 'Power iteration',
    summary:
      'v ← A v / ‖A v‖ on a symmetric matrix: the vector turns to the leading eigenvector, the Rayleigh quotient to its eigenvalue.',
    method: eigenPower,
    steps: 60,
    layout: [
      { kind: 'probe', path: 'matrix/action', title: 'the leading eigenplane: the unit circle, its image, and v' },
      { kind: 'series', path: 'iteration/vector', title: 'the components of v' },
      { kind: 'curves', paths: ['iteration/rayleigh'], title: 'Rayleigh quotient' },
      { kind: 'curves', paths: ['iteration/error'], title: 'error of the eigenvalue estimate' },
    ],
  },
  {
    id: 'linalg/qr-iteration',
    title: 'QR iteration',
    summary:
      'A_{k+1} = R_k Q_k: the matrix converges to a diagonal of its eigenvalues as the off-diagonal mass drains away.',
    method: eigenQr,
    steps: 60,
    layout: [
      { kind: 'tensor', path: 'iteration/matrix', title: 'A_k' },
      { kind: 'series', path: 'iteration/vector', title: 'the diagonal: eigenvalue estimates' },
      { kind: 'curves', paths: ['iteration/offDiagonal'], title: 'off-diagonal norm' },
      { kind: 'curves', paths: ['iteration/error'], title: 'largest eigenvalue error' },
    ],
  },
  {
    id: 'search/grid-search',
    title: 'Path search on a grid',
    summary:
      'Breadth-first, depth-first, uniform-cost, greedy, A* and weighted A* on small maps: watch the frontier flood the grid and the path appear.',
    method: gridSearch,
    steps: 2000,
    layout: [
      { kind: 'projection', name: 'board', title: 'the board: explored, frontier and path' },
      { kind: 'curves', paths: ['frontier', 'explored'] },
      { kind: 'curves', paths: ['pathCost'] },
    ],
  },
  {
    id: 'search/annealing',
    title: 'Local search on a surface',
    summary:
      'Simulated annealing, hill climbing and late acceptance walking over Rastrigin, Himmelblau, Ackley, Rosenbrock and Beale: watch a hot walk cool into a valley, or a greedy one stop in the first it finds.',
    method: annealing,
    steps: 1000,
    layout: [
      {
        kind: 'trajectory',
        paths: ['position'],
        over: { path: 'objective/surface' },
        trail: 200,
        title: 'the walk over the objective',
      },
      { kind: 'curves', paths: ['energy', 'best'] },
      { kind: 'curves', paths: ['temperature'] },
    ],
  },
  {
    id: 'tune/bayesian-optimisation',
    title: 'Bayesian optimisation',
    summary:
      'A Gaussian-process surrogate and an acquisition choose each trial of a one-dimensional function: expected improvement, the lower confidence bound or the probability of improvement.',
    method: bayesianOptimisation,
    steps: 30,
    layout: [
      { kind: 'probe', path: 'tune/posterior', title: 'surrogate and acquisition' },
      { kind: 'curves', paths: ['best'] },
      { kind: 'curves', paths: ['x'] },
    ],
  },
  {
    id: 'dynamics/kalman-tracking',
    title: 'Kalman filter tracking',
    summary:
      'A constant-velocity Kalman filter following a point it only sees through noisy readings: its estimate and 1σ uncertainty against the truth, and how trusting the model too much or too little changes it.',
    method: kalmanTracking,
    steps: 300,
    layout: [
      {
        kind: 'trajectory',
        paths: ['estimate', 'truth'],
        ellipse: 'covariance',
        title: 'filtered position (1σ) against the truth',
      },
      { kind: 'trajectory', paths: ['observation'], points: true, trail: 60, title: 'observations' },
      { kind: 'curves', paths: ['positionError'] },
      { kind: 'curves', paths: ['nis'] },
    ],
  },
  {
    id: 'sequence/hmm',
    title: 'Hidden Markov model',
    summary:
      'Baum-Welch on a sequence with hidden states: forward-backward gives the state posteriors, then the transitions and emissions are re-estimated from the expected counts. The dishonest casino hides which die was rolled; the decoded path is drawn against the truth.',
    estimator: hmm,
    datasets: ['sequences'],
    layout: [
      { kind: 'probe', path: 'model/posterior', title: 'state posteriors and the decoded path' },
      { kind: 'tensor', path: 'solver/transition', title: 'transition matrix' },
      { kind: 'curves', paths: ['solver/logLikelihood'] },
    ],
  },
  {
    id: 'dynamics/ode-systems',
    title: 'ODE systems',
    summary:
      'Predator and prey, Lorenz, Van der Pol, a pendulum or an epidemic, integrated by an adaptive Runge-Kutta method one accepted step at a time: the state against time, the phase portrait, and the step size the controller chose. The panels name the state as y[i].',
    method: odeSystems,
    steps: 1000,
    layout: [
      { kind: 'series', path: 'y', x: 't', title: 'the state against time' },
      { kind: 'trajectory', paths: ['y'], dims: [0, 1], title: 'phase portrait' },
      { kind: 'curves', paths: ['h'], title: 'step size' },
    ],
  },
]
