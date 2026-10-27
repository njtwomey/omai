import type { IterativeEstimator } from 'mdml-engine/learn'
import { kmeans } from 'mdml-engine/methods/cluster'
import { eigenPower, eigenQr } from 'mdml-engine/methods/linalg'
import { lassoPath, logisticRegression, ridge } from 'mdml-engine/methods/linear'
import { gmm } from 'mdml-engine/methods/mixture'
import { changeOfVariables } from 'mdml-engine/methods/probability'
import { mcmc } from 'mdml-engine/methods/sampling'
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
    id: 'probability/change-of-variables',
    title: 'Change of variables',
    summary:
      'Normal samples and a grid pushed through a bijection that deforms from the identity: the density follows the Jacobian.',
    method: changeOfVariables,
    steps: 60,
    layout: [
      { kind: 'probe', path: 'map/density', title: 'pushed density, with the pushed samples' },
      { kind: 'probe', path: 'map/grid', title: 'the grid under the map' },
      { kind: 'curves', paths: ['map/entropy', 'map/meanLogDet'], title: 'entropy and mean log |det J|' },
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
]
