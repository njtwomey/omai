/**
 * k-medoids by PAM (Kaufman and Rousseeuw, 1990): clustering around actual data rows rather than means, with a greedy
 * BUILD phase and SWAP steps that exchange a medoid for a non-medoid while the total dissimilarity falls, or CLARA on
 * samples for larger data. Mirrors scikit-learn-extra's `KMedoids`.
 */

import { WhiteboxError } from '../../lib/base/index.ts'
import { clusteringSteps } from '../../lib/exposition/index.ts'
import {
  defineExplainedEstimator,
  type Estimator,
  EstimatorClass,
  type ExplainedFitted,
  fitRunOptions,
  matrixView,
} from '../../lib/learn/index.ts'
import { geometry } from '../../lib/numerics/index.ts'
import { foldIn, type Key, keys, subsample } from '../../lib/random/index.ts'
import { choice, dict, int } from '../../lib/space/index.ts'
import { fromStorage, type Tensor, type TensorLike, toArray } from '../../lib/tensor/index.ts'
import { type Algorithm, runAlgorithm } from '../../lib/trace/index.ts'

/** The state of PAM: the medoid rows, labels, total cost and history. */
export interface PamState {
  readonly medoids: number[]
  readonly labels: number[]
  readonly cost: number
  readonly improved: boolean
  readonly history: { readonly medoids: number[][]; readonly labels: number[][]; readonly cost: number[] }
}

/** Nearest-medoid labels and the total dissimilarity. */
function assign(d: readonly (readonly number[])[], medoids: readonly number[]): { labels: number[]; cost: number } {
  let cost = 0
  const labels = d.map((row) => {
    let best = 0
    for (let c = 1; c < medoids.length; c++) if ((row[medoids[c] ?? 0] ?? 0) < (row[medoids[best] ?? 0] ?? 0)) best = c
    cost += row[medoids[best] ?? 0] ?? 0
    return best
  })
  return { labels, cost }
}

/**
 * PAM's BUILD phase (Kaufman and Rousseeuw 1990): the medoid minimising the total dissimilarity, then repeatedly the
 * row whose addition lowers the cost the most.
 *
 * @param d - Dissimilarities `[n][n]`.
 * @param k - Medoids.
 * @returns The medoid rows.
 */
export function pamBuild(d: readonly (readonly number[])[], k: number): number[] {
  const n = d.length
  const medoids: number[] = []
  for (let c = 0; c < k; c++) {
    let best = -1
    let bestCost = Number.POSITIVE_INFINITY
    for (let i = 0; i < n; i++) {
      if (medoids.includes(i)) continue
      const cost = assign(d, [...medoids, i]).cost
      if (cost < bestCost) {
        bestCost = cost
        best = i
      }
    }
    medoids.push(best)
  }
  return medoids
}

/**
 * PAM's SWAP phase as a traced algorithm: each step tries every (medoid, non-medoid) exchange and makes the one that
 * lowers the total dissimilarity the most, stopping when none does.
 *
 * @param d - Dissimilarities `[n][n]`.
 * @param start - The initial medoids.
 * @returns The algorithm.
 */
export function pamSwapAlgorithm(d: readonly (readonly number[])[], start: readonly number[]): Algorithm<PamState> {
  const n = d.length
  return {
    name: 'pam-swap',
    config: { k: start.length },
    init: () => {
      const medoids = [...start]
      const { labels, cost } = assign(d, medoids)
      return { medoids, labels, cost, improved: true, history: { medoids: [medoids], labels: [labels], cost: [cost] } }
    },
    step: (s) => {
      let best = s.medoids
      let bestCost = s.cost
      for (let c = 0; c < s.medoids.length; c++)
        for (let i = 0; i < n; i++) {
          if (s.medoids.includes(i)) continue
          const trial = s.medoids.map((m, t) => (t === c ? i : m))
          const cost = assign(d, trial).cost
          if (cost < bestCost - 1e-12) {
            bestCost = cost
            best = trial
          }
        }
      const improved = best !== s.medoids
      const { labels, cost } = assign(d, best)
      return {
        medoids: best,
        labels,
        cost,
        improved,
        history: improved
          ? {
              medoids: [...s.history.medoids, best],
              labels: [...s.history.labels, labels],
              cost: [...s.history.cost, cost],
            }
          : s.history,
      }
    },
    done: (s, t) => (t > 0 && !s.improved ? 'converged' : undefined),
    observe: (s) => ({ metrics: { cost: s.cost } }),
  }
}

/** Hyperparameters of k-medoids. */
export type KMedoidsHyper = {
  readonly nClusters?: number
  /** PAM on all rows, or CLARA: PAM on random samples, the best medoids on all rows (default PAM). */
  readonly method?: 'pam' | 'clara'
  /** CLARA's samples and their size (default 5 and 40 + 2k). */
  readonly samples?: number
  readonly sampleSize?: number
  readonly maxIter?: number
}

/** The parameters of a fitted k-medoids. */
export type KMedoidsParams = {
  readonly medoidIndices: number[]
  readonly centres: number[][]
  readonly labels: number[]
  readonly inertia: number
}

/** The verbs of a fitted k-medoids: nearest-medoid labels. */
export interface MedoidVerbs {
  readonly predict: (x: TensorLike) => Tensor
  readonly nFeaturesIn: number
}

/**
 * k-medoids (sklearn_extra.cluster.KMedoids with method 'pam'): BUILD then traced SWAP steps on Euclidean
 * dissimilarities, or CLARA on samples. Its exposition is a `clusteringSteps` record (assignments, medoid coordinates
 * and total dissimilarity per swap).
 *
 * @param hyper - Clusters, the method and CLARA's sampling.
 * @returns The estimator.
 */
export function kMedoids(
  hyper: KMedoidsHyper = {},
): Estimator<KMedoidsHyper, ExplainedFitted<KMedoidsParams, MedoidVerbs>> {
  return defineExplainedEstimator<KMedoidsHyper, KMedoidsParams, MedoidVerbs>(
    {
      name: 'k-medoids',
      kind: 'clusterer',
      hyperSpace: (h) =>
        dict({
          nClusters: int(1, 20, { default: h.nClusters ?? 3, label: 'clusters' }),
          method: choice(['pam', 'clara'], {
            label: 'algorithm',
            description: 'PAM (exact swaps over all rows) or CLARA (PAM on samples, for large data).',
            default: h.method ?? 'pam',
          }),
        }),
      fit: (data, h, options) => {
        const m = matrixView(data.x, 'kMedoids')
        const k = h.nClusters ?? 3
        if (!(Number.isInteger(k) && k >= 1 && k <= m.n))
          throw new WhiteboxError('value', `kMedoids: nClusters must be in 1..${m.n}`)
        const rows = Array.from({ length: m.n }, (_, i) => Array.from(m.data.subarray(i * m.d, (i + 1) * m.d)))
        const d = toArray(geometry.cdist(rows, rows)) as number[][]
        const run = fitRunOptions(options, 'k-medoids')
        let start: number[]
        if (h.method === 'clara') {
          const size = Math.min(m.n, h.sampleSize ?? 40 + 2 * k)
          let bestCost = Number.POSITIVE_INFINITY
          start = []
          for (const sk of keys(foldIn(run.key as Key, 0), h.samples ?? 5)) {
            const sample = Array.from(subsample(sk, m.n, size).data)
            const sub = sample.map((i) => sample.map((j) => d[i]?.[j] ?? 0))
            const local = runAlgorithm(pamSwapAlgorithm(sub, pamBuild(sub, k)), { maxSteps: h.maxIter ?? 100 }).state
            const medoids = local.medoids.map((i) => sample[i] ?? 0)
            const cost = assign(d, medoids).cost
            if (cost < bestCost) {
              bestCost = cost
              start = medoids
            }
          }
        } else start = pamBuild(d, k)
        const out = runAlgorithm(pamSwapAlgorithm(d, start), {
          ...run,
          maxSteps: h.method === 'clara' ? 0 : (h.maxIter ?? 100),
        })
        const s = out.state
        const coords = (ms: readonly number[]): number[][] => ms.map((i) => rows[i] ?? [])
        return {
          params: { medoidIndices: s.medoids, centres: coords(s.medoids), labels: s.labels, inertia: s.cost },
          exposition: [
            clusteringSteps({
              method: h.method === 'clara' ? 'clara' : 'pam',
              assignments: s.history.labels,
              centres: s.history.medoids.map(coords),
              objective: s.history.cost,
            }),
          ],
        }
      },
      methods: (p) => {
        const dim = p.centres[0]?.length ?? 0
        return {
          predict: (x) => {
            matrixView(x, 'kMedoids', { nFeatures: dim })
            const dist = toArray(geometry.cdist(x, p.centres)) as number[][]
            const labels = dist.map((r) => r.indexOf(Math.min(...r)))
            return fromStorage(Int32Array.from(labels), [labels.length], 'int32')
          },
          nFeaturesIn: dim,
        }
      },
    },
    hyper,
  )
}

/** k-medoids as a class. */
export class KMedoids extends EstimatorClass<KMedoidsHyper, ExplainedFitted<KMedoidsParams, MedoidVerbs>> {
  /**
   * Make an unfitted model.
   *
   * @param hyper - The hyperparameters of {@link kMedoids}.
   */
  constructor(hyper: KMedoidsHyper = {}) {
    super(kMedoids(hyper))
  }
}
