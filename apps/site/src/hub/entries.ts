import type { IterativeEstimator } from 'mdml-engine/learn'
import { kmeans } from 'mdml-engine/methods/cluster'

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
]
