/**
 * k-medoids: clustering around actual data rows rather than means. PAM (Kaufman and Rousseeuw 1990) as one algorithm,
 * a greedy BUILD that adds a medoid per step and then SWAP steps that exchange a medoid for a non-medoid while the
 * total dissimilarity falls, and CLARA, which runs PAM on a random sample per step and keeps the medoids that serve
 * all rows best. Works on any distance of the metric slot, or a precomputed dissimilarity matrix.
 */

import { MdmlError } from '../base/index.ts'
import {
  type DistanceAlias,
  type DistanceName,
  type DistanceOptions,
  distanceMatrix,
  type MetricSlot,
  pairwiseDistances,
} from '../distances/index.ts'
import { type IterativeEstimator, iterative, matrixView } from '../learn/index.ts'
import { subsample } from '../random/index.ts'
import { choice, dict, int } from '../space/index.ts'
import { fromStorage, type Tensor, type TensorLike, tensor, toFlat } from '../tensor/index.ts'
import { type Algorithm, channels, stateFromJson, stateToJson } from '../trace/index.ts'
import { rowsAt } from './_support.ts'

/** The state of PAM and CLARA: the medoid rows, the medoid each row is nearest (an index into `medoids`) and the cost. */
export interface MedoidState {
  readonly medoids: readonly number[]
  readonly labels: readonly number[]
  /** The total dissimilarity of the rows to their medoids. */
  readonly cost: number
  /** Whether the last step lowered the cost (PAM: made a swap or a build; CLARA: found better medoids). */
  readonly improved: boolean
  /** PAM: still adding medoids; CLARA: drawing samples. */
  readonly phase: 'build' | 'swap' | 'sample'
  /** CLARA: the samples drawn. */
  readonly round: number
}

/** Nearest-medoid labels (an index into `medoids`) and the total dissimilarity. */
function assign(
  d: ArrayLike<number>,
  n: number,
  medoids: readonly number[],
): { readonly labels: number[]; readonly cost: number } {
  let cost = 0
  const labels = new Array<number>(n).fill(0)
  for (let i = 0; i < n; i++) {
    let best = 0
    for (let c = 1; c < medoids.length; c++)
      if ((d[i * n + (medoids[c] ?? 0)] ?? 0) < (d[i * n + (medoids[best] ?? 0)] ?? 0)) best = c
    cost += d[i * n + (medoids[best] ?? 0)] ?? 0
    labels[i] = best
  }
  return { labels, cost }
}

/** The row whose addition to the medoids lowers the total dissimilarity the most (the first on ties). */
function bestAddition(d: ArrayLike<number>, n: number, medoids: readonly number[]): number {
  let best = -1
  let bestCost = Number.POSITIVE_INFINITY
  for (let i = 0; i < n; i++) {
    if (medoids.includes(i)) continue
    const cost = assign(d, n, [...medoids, i]).cost
    if (cost < bestCost) {
      bestCost = cost
      best = i
    }
  }
  return best
}

/** The single best exchange of a medoid for a non-medoid, or the medoids unchanged when none lowers the cost. */
function bestSwap(d: ArrayLike<number>, n: number, medoids: readonly number[], cost: number): readonly number[] {
  let best = medoids
  let bestCost = cost
  for (let c = 0; c < medoids.length; c++)
    for (let i = 0; i < n; i++) {
      if (medoids.includes(i)) continue
      const trial = medoids.map((m, t) => (t === c ? i : m))
      const trialCost = assign(d, n, trial).cost
      if (trialCost < bestCost - 1e-12) {
        bestCost = trialCost
        best = trial
      }
    }
  return best
}

/**
 * PAM's BUILD phase (Kaufman and Rousseeuw 1990): the medoid minimising the total dissimilarity, then repeatedly the
 * row whose addition lowers the cost the most.
 *
 * @param d - Dissimilarities `[n, n]`, flat row-major.
 * @param n - The number of rows.
 * @param k - Medoids.
 * @returns The medoid rows.
 * @example
 * ```ts
 * import { pamBuild } from 'mdml-engine/cluster'
 * pamBuild([0, 1, 5, 1, 0, 4, 5, 4, 0], 3, 1) // => [1]
 * ```
 */
export function pamBuild(d: ArrayLike<number>, n: number, k: number): number[] {
  const medoids: number[] = []
  for (let c = 0; c < k; c++) medoids.push(bestAddition(d, n, medoids))
  return medoids
}

/**
 * PAM run to the end on a dissimilarity matrix: BUILD then SWAP until no exchange lowers the cost.
 *
 * @param d - Dissimilarities `[n, n]`, flat row-major.
 * @param n - The number of rows.
 * @param k - Medoids.
 * @param maxIter - The most swaps (default 100).
 * @returns The medoid rows, the labels (indices into the medoids) and the cost.
 * @example
 * ```ts
 * import { pamSolve } from 'mdml-engine/cluster'
 * pamSolve([0, 1, 9, 9, 1, 0, 9, 9, 9, 9, 0, 1, 9, 9, 1, 0], 4, 2).medoids // => [0, 2]
 * ```
 */
export function pamSolve(
  d: ArrayLike<number>,
  n: number,
  k: number,
  maxIter = 100,
): { readonly medoids: readonly number[]; readonly labels: number[]; readonly cost: number } {
  let medoids: readonly number[] = pamBuild(d, n, k)
  let { labels, cost } = assign(d, n, medoids)
  for (let t = 0; t < maxIter; t++) {
    const next = bestSwap(d, n, medoids, cost)
    if (next === medoids) break
    medoids = next
    ;({ labels, cost } = assign(d, n, medoids))
  }
  return { medoids, labels, cost }
}

/** Options of the medoid algorithms. */
export interface MedoidOptions {
  readonly nClusters: number
  readonly metric?: MetricSlot
  readonly distanceOptions?: DistanceOptions
}

const medoidSchema = channels({
  metrics: { cost: { description: 'the total dissimilarity of the rows to their medoids', direction: 'lower' } },
  tensors: {
    medoids: { role: 'state', description: 'the medoid rows (indices)', shape: ['k'] },
    labels: { role: 'state', description: 'the nearest medoid of each row', shape: ['n'] },
  },
})

const observeMedoids = (s: MedoidState, n: number) => ({
  metrics: { cost: s.cost },
  tensors: {
    medoids: { role: 'state' as const, value: s.medoids, shape: [s.medoids.length] },
    labels: { role: 'state' as const, value: s.labels, shape: [n] },
  },
})

/**
 * PAM as a traced {@link Algorithm}: the first steps BUILD the medoids one at a time, then each step makes the single
 * exchange of a medoid for a non-medoid that lowers the total dissimilarity the most, until none does.
 *
 * @param x - Rows `[n, d]`, or with `metric: 'precomputed'` the dissimilarities.
 * @param options - The number of medoids and the metric.
 * @returns The algorithm.
 * @example
 * ```ts
 * import { pamAlgorithm } from 'mdml-engine/cluster'
 * import { runAlgorithm } from 'mdml-engine/trace'
 * const x = [[0], [1], [10], [11]]
 * runAlgorithm(pamAlgorithm(x, { nClusters: 2 }), { maxSteps: 20 }).state.medoids // => [1, 2]
 * ```
 */
export function pamAlgorithm(x: TensorLike, options: MedoidOptions): Algorithm<MedoidState> {
  const { n, data } = distanceMatrix(x, options.metric ?? 'euclidean', options.distanceOptions)
  const k = options.nClusters
  if (!(Number.isInteger(k) && k >= 1 && k <= n))
    throw new MdmlError('value', `pamAlgorithm: nClusters must be in 1..${n}`)
  const state = (medoids: readonly number[], improved: boolean, phase: 'build' | 'swap'): MedoidState => ({
    medoids,
    ...assign(data, n, medoids),
    improved,
    phase,
    round: 0,
  })
  return {
    name: 'pam',
    config: { k, n },
    schema: medoidSchema,
    init: () => state([bestAddition(data, n, [])], true, k === 1 ? 'swap' : 'build'),
    step: (s) => {
      if (s.phase === 'build') {
        const medoids = [...s.medoids, bestAddition(data, n, s.medoids)]
        return state(medoids, true, medoids.length >= k ? 'swap' : 'build')
      }
      const next = bestSwap(data, n, s.medoids, s.cost)
      return state(next, next !== s.medoids, 'swap')
    },
    done: (s) => (s.phase === 'swap' && !s.improved ? 'converged' : undefined),
    observe: (s) => observeMedoids(s, n),
    serialise: (s) => stateToJson(s),
    deserialise: (j) => stateFromJson(j) as MedoidState,
  }
}

/**
 * CLARA (Kaufman and Rousseeuw 1990) as a traced {@link Algorithm}: each step draws a random sample of rows, runs PAM
 * on it, and keeps those medoids if they serve all the rows better than the best so far.
 *
 * @param x - Rows `[n, d]`, or with `metric: 'precomputed'` the dissimilarities.
 * @param options - The number of medoids, the metric, and the sample size (default 40 + 2k, at most n).
 * @returns The algorithm; a step per sample.
 * @example
 * ```ts
 * import { claraAlgorithm } from 'mdml-engine/cluster'
 * import { key } from 'mdml-engine/random'
 * import { runAlgorithm } from 'mdml-engine/trace'
 * const x = [[0], [1], [10], [11], [0.5], [10.5]]
 * runAlgorithm(claraAlgorithm(x, { nClusters: 2, sampleSize: 4 }), { maxSteps: 3, key: key(0) }).state.medoids.length // => 2
 * ```
 */
export function claraAlgorithm(
  x: TensorLike,
  options: MedoidOptions & { readonly sampleSize?: number; readonly maxIter?: number },
): Algorithm<MedoidState> {
  const { n, data } = distanceMatrix(x, options.metric ?? 'euclidean', options.distanceOptions)
  const k = options.nClusters
  if (!(Number.isInteger(k) && k >= 1 && k <= n))
    throw new MdmlError('value', `claraAlgorithm: nClusters must be in 1..${n}`)
  const size = Math.max(k, Math.min(n, options.sampleSize ?? 40 + 2 * k))
  return {
    name: 'clara',
    config: { k, n, sampleSize: size },
    schema: medoidSchema,
    init: () => ({
      medoids: Array.from({ length: k }, (_, i) => i),
      labels: new Array<number>(n).fill(0),
      cost: Number.MAX_VALUE,
      improved: false,
      phase: 'sample',
      round: 0,
    }),
    step: (s, ctx) => {
      const sample = Array.from(subsample(ctx.key, n, size).data)
      const sub = new Float64Array(size * size)
      for (const [a, i] of sample.entries())
        for (const [b, j] of sample.entries()) sub[a * size + b] = data[i * n + j] ?? 0
      const medoids = pamSolve(sub, size, k, options.maxIter ?? 100).medoids.map((i) => sample[i] ?? 0)
      const { labels, cost } = assign(data, n, medoids)
      return cost < s.cost
        ? { medoids, labels, cost, improved: true, phase: 'sample', round: s.round + 1 }
        : { ...s, improved: false, round: s.round + 1 }
    },
    observe: (s) => observeMedoids(s, n),
    serialise: (s) => stateToJson(s),
    deserialise: (j) => stateFromJson(j) as MedoidState,
  }
}

/** Hyperparameters of {@link kMedoids}. */
export interface KMedoidsHyper {
  readonly nClusters: number
  /** PAM on all rows, or CLARA: PAM on random samples, the best medoids on all rows. */
  readonly method: 'pam' | 'clara'
  /** CLARA's sample size (`null`: 40 + 2k). */
  readonly sampleSize: number | null
  /** CLARA's samples. */
  readonly samples: number
  readonly maxIter: number
  readonly metric: DistanceName | DistanceAlias
}

/** The fitted parameters of {@link kMedoids}. */
export type KMedoidsParams = {
  readonly medoidIndices: number[]
  readonly centres: number[][]
  readonly labels: number[]
  readonly inertia: number
  readonly metric: DistanceName | DistanceAlias
}

/** The verbs of a fitted k-medoids: nearest-medoid labels. */
export interface MedoidVerbs {
  readonly predict: (x: TensorLike) => Tensor
}

/**
 * k-medoids (sklearn_extra.cluster.KMedoids, PAM or CLARA) on the iterative base: BUILD then SWAP steps, or a sample
 * per step. New rows take the nearest medoid under the same metric.
 *
 * @example
 * ```ts
 * import { kMedoids } from 'mdml-engine/cluster'
 * kMedoids.with({ nClusters: 2 }).fit({ x: [[0], [1], [10], [11]] }).params.medoidIndices // => [1, 2]
 * ```
 */
export const kMedoids: IterativeEstimator<KMedoidsHyper, MedoidState, KMedoidsParams, MedoidVerbs> = iterative<
  KMedoidsHyper,
  MedoidState,
  KMedoidsParams,
  MedoidVerbs
>({
  name: 'cluster/k-medoids',
  kind: 'clusterer',
  defaults: { nClusters: 3, method: 'pam', sampleSize: null, samples: 5, maxIter: 100, metric: 'euclidean' },
  hyperSpace: (h) =>
    dict({
      nClusters: int(1, 20, { default: h.nClusters, label: 'clusters' }),
      method: choice(['pam', 'clara'] as const, { default: h.method }),
    }),
  algorithm: (data, h) =>
    h.method === 'clara'
      ? claraAlgorithm(data.x, {
          nClusters: h.nClusters,
          metric: h.metric,
          maxIter: h.maxIter,
          ...(h.sampleSize === null ? {} : { sampleSize: h.sampleSize }),
        })
      : pamAlgorithm(data.x, { nClusters: h.nClusters, metric: h.metric }),
  maxSteps: (h) => (h.method === 'clara' ? h.samples : h.maxIter + h.nClusters),
  params: (s, data, h) => {
    const m = matrixView(data.x, 'kMedoids')
    return {
      medoidIndices: [...s.medoids],
      centres: rowsAt(m.data, m.d, s.medoids),
      labels: [...s.labels],
      inertia: s.cost,
      metric: h.metric,
    }
  },
  verbs: (p) => ({
    predict: (x) => {
      const m = matrixView(x, 'kMedoids.predict', { nFeatures: p.centres[0]?.length ?? 0 })
      const dist = toFlat(pairwiseDistances(fromStorage(m.data, [m.n, m.d], 'float64'), p.centres, p.metric))
      const k = p.centres.length
      return tensor(
        Array.from({ length: m.n }, (_, i) => {
          let best = 0
          for (let c = 1; c < k; c++) if ((dist[i * k + c] ?? 0) < (dist[i * k + best] ?? 0)) best = c
          return best
        }),
      )
    },
  }),
})
