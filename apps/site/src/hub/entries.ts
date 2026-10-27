import type { IterativeEstimator } from 'mdml-engine/learn'
import { kmeans } from 'mdml-engine/methods/cluster'
import { bayesianRegression, lassoPath, logisticRegression, ridge } from 'mdml-engine/methods/linear'
import { bgmm, gmm } from 'mdml-engine/methods/mixture'
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
]
