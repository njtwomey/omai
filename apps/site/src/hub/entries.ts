import type { IterativeEstimator } from 'mdml-engine/learn'
import { kmeans } from 'mdml-engine/methods/cluster'
import { lassoPath, logisticRegression, ridge } from 'mdml-engine/methods/linear'
import { gmm } from 'mdml-engine/methods/mixture'

/** A hub entry is data: an id, words, an estimator on the contract, and the dataset collections it runs on. */
export interface HubEntry {
  /** `area/name`, also the page path under /hub. */
  readonly id: string
  readonly title: string
  readonly summary: string
  // biome-ignore lint/suspicious/noExplicitAny: entries of every particular share one list
  readonly estimator: IterativeEstimator<any, any, any, any>
  /** Ids in `datasetCollections` the entry can be run on. */
  readonly datasets: readonly string[]
}

/** The entries of the hub. */
export const ENTRIES: readonly HubEntry[] = [
  {
    id: 'cluster/kmeans',
    title: 'k-means',
    summary: 'Lloyd’s algorithm: assign rows to the nearest centre, move centres to the means, repeat.',
    estimator: kmeans,
    datasets: ['clustering2d'],
  },
  {
    id: 'mixture/gmm',
    title: 'Gaussian mixture',
    summary:
      'Expectation-maximisation: responsibilities of each component for each row, then new weights, means and covariances.',
    estimator: gmm,
    datasets: ['clustering2d'],
  },
  {
    id: 'linear/logistic-regression',
    title: 'Logistic regression',
    summary:
      'A linear classifier fitted by iteratively reweighted least squares; the decision surface follows the fit.',
    estimator: logisticRegression,
    datasets: ['classification2d'],
  },
  {
    id: 'linear/ridge',
    title: 'Ridge regression',
    summary: 'Least squares with an L2 penalty, solved in closed form.',
    estimator: ridge,
    datasets: ['regression1d'],
  },
  {
    id: 'linear/lasso-path',
    title: 'Lasso path',
    summary: 'The lasso along a decreasing penalty grid, each penalty warm-started from the last: play the path.',
    estimator: lassoPath,
    datasets: ['regression1d'],
  },
]
