/**
 * k-means as a particular of the iterative base: Lloyd's algorithm from seeded initial centres, with restarts, the
 * centres, labels and inertia of every iteration on the tape, and the iteration-by-iteration record read back from
 * that tape.
 */

import { initialCentres, type LloydState, lloydAlgorithm, nearestCentres } from '../../lib/cluster/index.ts'
import { clusteringSteps } from '../../lib/exposition/index.ts'
import { type IterativeEstimator, iterative } from '../../lib/learn/index.ts'
import { choice, dict, int } from '../../lib/space/index.ts'
import { type Tensor, type TensorLike, tensor } from '../../lib/tensor/index.ts'
import { type Contract, defineContract, derived, Log, metricSeries, tensorSeries } from '../../lib/trace/index.ts'

/** The hyperparameters of {@link kmeans}. */
export interface KMeansHyper {
  /** The number of clusters. */
  readonly nClusters: number
  /** How the initial centres are chosen. */
  readonly init: 'k-means++' | 'random'
  /** Starts from fresh initial centres; the lowest inertia wins. */
  readonly nInit: number
  readonly maxIter: number
  /** Tolerance on the squared centre shift, relative to the mean feature variance. */
  readonly tol: number
}

/** The fitted parameters of {@link kmeans}: plain JSON. */
export type KMeansParams = {
  readonly centres: number[][]
  readonly labels: number[]
  readonly inertia: number
}

/** What a fitted k-means offers. */
export interface KMeansVerbs {
  /** The nearest centre of each row. */
  readonly predict: (x: TensorLike) => Tensor
}

/**
 * The paths of k-means beyond the solver’s: the cluster centres, derived.
 *
 * @example
 * ```ts
 * import type { KMeansPath } from 'omai-engine/methods/cluster'
 * const path: KMeansPath = 'model/centres'
 * path // => 'model/centres'
 * ```
 */
export type KMeansPath = 'model/centres'

/**
 * k-means clustering by Lloyd's algorithm.
 *
 * @example
 * ```ts
 * import { kmeans } from 'omai-engine/methods/cluster'
 * kmeans.with({ nClusters: 2 }).fit({ x: [[0, 0], [0, 1], [9, 9], [9, 8]] }).params.centres.length // => 2
 * ```
 */
export const kmeans: IterativeEstimator<KMeansHyper, LloydState, KMeansParams, KMeansVerbs, KMeansPath> = iterative<
  KMeansHyper,
  LloydState,
  KMeansParams,
  KMeansVerbs,
  KMeansPath
>({
  name: 'cluster/kmeans',
  kind: 'clusterer',
  defaults: { nClusters: 3, init: 'k-means++', nInit: 1, maxIter: 300, tol: 1e-4 },
  hyperSpace: (h) =>
    dict({
      nClusters: int(2, 10, { default: h.nClusters, label: 'clusters' }),
      init: choice(['k-means++', 'random'] as const, { default: h.init }),
    }),
  algorithm: (data, h, key) => lloydAlgorithm(data.x, initialCentres(data.x, h.nClusters, h.init, key), { tol: h.tol }),
  maxSteps: (h) => h.maxIter,
  restarts: { count: (h) => h.nInit, better: (a, b) => a.inertia < b.inertia },
  params: (s) => ({ centres: s.centres, labels: s.labels, inertia: s.inertia }),
  verbs: (p) => ({ predict: (x) => tensor(nearestCentres(x, p.centres).labels) }),
  contract: (): Contract<LloydState, KMeansPath> =>
    defineContract({
      'model/centres': derived<LloydState>({
        description: 'the cluster centres at this step',
        compute: (s) => s.centres,
      }),
    }),
  logs: {
    'solver/inertia': Log(),
    'solver/centres': Log({ detail: 'full' }),
    'solver/labels': Log({ detail: 'full' }),
  },
  projections: {
    clusteringSteps: (run) => {
      const centres = tensorSeries(run, 'solver/centres')
      const labels = tensorSeries(run, 'solver/labels')
      const objective = metricSeries(run, 'solver/inertia').values
      if (labels.length === 0 || centres.length === 0) return undefined
      return clusteringSteps({
        method: 'kmeans',
        assignments: labels.map((e) => [...(e.values ?? [])]),
        centres: centres.map((e) => {
          const v = e.values ?? []
          const d = e.shape[1] ?? 1
          return Array.from({ length: e.shape[0] ?? 0 }, (_, j) => v.slice(j * d, (j + 1) * d))
        }),
        objective: objective.slice(0, labels.length),
      })
    },
  },
})
