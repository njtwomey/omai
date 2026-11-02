/**
 * Multidimensional scaling (scikit-learn's `MDS`): placing points in a few dimensions so their distances match given
 * dissimilarities. Classical (Torgerson) MDS in closed form by an eigendecomposition, and SMACOF, which lowers the
 * stress by repeated Guttman transforms, one per step, in its metric form and its non-metric form (disparities by
 * isotonic regression of the embedded distances on the dissimilarities).
 */

import { MdmlError } from '../base/index.ts'
import { type DistanceAlias, type DistanceName, distanceMatrix, type MetricSlot } from '../distances/index.ts'
import { type IterativeEstimator, iterative, matrixView, poolAdjacentViolators } from '../learn/index.ts'
import { linalg } from '../numerics/index.ts'
import { uniform } from '../random/index.ts'
import { choice, dict, int } from '../space/index.ts'
import { toArray, toFlat } from '../tensor/index.ts'
import { type Algorithm, channels, stateFromJson, stateToJson } from '../trace/index.ts'
import { type Embedding, type EmbeddingParams, type EmbeddingVerbs, embeddingVerbs, trainingRows } from './_support.ts'

/**
 * Classical (Torgerson) MDS: the top eigenvectors of the double-centred squared dissimilarities -1/2 J D^2 J, scaled
 * by the square roots of their eigenvalues (negative ones clipped to zero).
 *
 * @param d - Dissimilarities `[n, n]`.
 * @param k - The embedding dimension.
 * @returns The embedding `[n, k]` and the eigenvalues used.
 * @example
 * ```ts
 * import { classicalMds } from 'mdml-engine/manifold'
 * classicalMds([[0, 2], [2, 0]], 1).eigenvalues // => [2]
 * ```
 */
export function classicalMds(
  d: readonly (readonly number[])[],
  k: number,
): { readonly embedding: Embedding; readonly eigenvalues: number[] } {
  const n = d.length
  if (!(Number.isInteger(k) && k >= 1 && k <= n)) throw new MdmlError('value', `classicalMds: k must be in 1..${n}`)
  const sq = d.map((r) => r.map((v) => v * v))
  const rowMean = sq.map((r) => r.reduce((a, v) => a + v, 0) / n)
  const all = rowMean.reduce((a, v) => a + v, 0) / n
  const b = sq.map((r, i) => r.map((v, j) => -0.5 * (v - (rowMean[i] ?? 0) - (rowMean[j] ?? 0) + all)))
  const { eigenvalues, eigenvectors } = linalg.eigh(b)
  const values = toFlat(eigenvalues)
  const vectors = toArray(eigenvectors) as number[][]
  const top = Array.from({ length: k }, (_, c) => n - 1 - c)
  const lambda = top.map((c) => Math.max(0, values[c] ?? 0))
  return {
    embedding: vectors.map((row) => top.map((c, j) => (row[c] ?? 0) * Math.sqrt(lambda[j] ?? 0))),
    eigenvalues: lambda,
  }
}

/** The state of SMACOF: the configuration, its stress, and the relative stress decrease of the last step. */
export interface SmacofState {
  readonly x: Embedding
  readonly stress: number
  /** (previous stress - stress) / (half the sum of squared embedded distances) of the last step (a large number before). */
  readonly change: number
}

/** Options of {@link smacofAlgorithm}. */
export interface SmacofOptions {
  /** The relative tolerance (default 1e-6). */
  readonly eps?: number
  /** Metric (default): the disparities are the dissimilarities. Non-metric: isotonic regression of the distances on them. */
  readonly metric?: boolean
}

const euclid = (x: Embedding): { n: number; dist: Float64Array } => {
  const n = x.length
  const dist = new Float64Array(n * n)
  for (let i = 0; i < n; i++)
    for (let j = i + 1; j < n; j++) {
      let s = 0
      const a = x[i] as number[]
      const b = x[j] as number[]
      for (let q = 0; q < a.length; q++) s += ((a[q] ?? 0) - (b[q] ?? 0)) ** 2
      dist[i * n + j] = dist[j * n + i] = Math.sqrt(s)
    }
  return { n, dist }
}

/**
 * sklearn's non-metric disparities: the isotonic regression of the embedded distances on the dissimilarities over the
 * pairs i < j with a non-zero dissimilarity (equal dissimilarities pooled), scaled so the squared disparities sum to
 * n (n - 1) / 2, returned as a symmetric flat matrix. The first iteration uses the dissimilarities themselves.
 */
function disparitiesOf(d: ArrayLike<number>, dist: ArrayLike<number>, n: number, first: boolean): Float64Array {
  const pairs: { i: number; j: number; x: number; y: number }[] = []
  for (let i = 0; i < n; i++)
    for (let j = i + 1; j < n; j++)
      if ((d[i * n + j] ?? 0) !== 0) pairs.push({ i, j, x: d[i * n + j] ?? 0, y: dist[i * n + j] ?? 0 })
  let fitted: number[]
  if (first) fitted = pairs.map((p) => p.x)
  else {
    const order = pairs
      .map((_, t) => t)
      .sort((a, b) => (pairs[a]?.x ?? 0) - (pairs[b]?.x ?? 0) || (pairs[a]?.y ?? 0) - (pairs[b]?.y ?? 0))
    // Pool equal dissimilarities into their mean distance, then the weighted pool-adjacent-violators.
    const groups: { x: number; sum: number; count: number }[] = []
    for (const t of order) {
      const p = pairs[t] as { x: number; y: number }
      const last = groups.at(-1)
      if (last !== undefined && last.x === p.x) {
        last.sum += p.y
        last.count++
      } else groups.push({ x: p.x, sum: p.y, count: 1 })
    }
    const iso = poolAdjacentViolators(
      groups.map((g) => g.sum / g.count),
      groups.map((g) => g.count),
    )
    const byX = new Map(groups.map((g, t) => [g.x, iso[t] ?? 0]))
    fitted = pairs.map((p) => byX.get(p.x) ?? 0)
  }
  const sq = fitted.reduce((a, v) => a + v * v, 0)
  const scale = Math.sqrt((n * (n - 1)) / 2 / (sq || 1))
  const out = new Float64Array(n * n)
  for (const [t, p] of pairs.entries()) out[p.i * n + p.j] = out[p.j * n + p.i] = (fitted[t] ?? 0) * scale
  return out
}

/**
 * SMACOF (de Leeuw 1977; sklearn's `_smacof_single`) as a traced {@link Algorithm}: each step applies the Guttman
 * transform X <- B(X) X / n and records the raw stress; it stops when the stress decrease relative to half the sum of
 * squared embedded distances is below eps. Non-metric MDS replaces the dissimilarities by disparities fitted by isotonic
 * regression to the current distances. It observes the stress and the `embedding`.
 *
 * @param d - Dissimilarities `[n, n]`, flat.
 * @param init - The initial configuration `[n, k]`.
 * @param options - The tolerance and whether the scaling is metric.
 * @returns The algorithm.
 * @example
 * ```ts
 * import { smacofAlgorithm } from 'mdml-engine/manifold'
 * import { runAlgorithm } from 'mdml-engine/trace'
 * const d = [0, 1, 2, 1, 0, 1, 2, 1, 0]
 * const out = runAlgorithm(smacofAlgorithm(d, [[0, 0.1], [1, -0.1], [2.5, 0]]), { maxSteps: 50 })
 * out.state.stress < 1e-3 // => true
 * ```
 */
export function smacofAlgorithm(
  d: ArrayLike<number>,
  init: readonly (readonly number[])[],
  options: SmacofOptions = {},
): Algorithm<SmacofState> {
  const n = init.length
  const eps = options.eps ?? 1e-6
  const metric = options.metric ?? true
  const k = init[0]?.length ?? 0
  const stressOf = (dist: ArrayLike<number>, disp: ArrayLike<number>): number => {
    let s = 0
    for (let p = 0; p < n * n; p++) s += ((dist[p] ?? 0) - (disp[p] ?? 0)) ** 2
    return s / 2
  }
  return {
    name: metric ? 'smacof' : 'smacof-nonmetric',
    config: { n, eps, metric },
    schema: channels({
      metrics: {
        stress: {
          description: 'raw stress, half the sum of squared differences of distances and disparities',
          direction: 'lower',
        },
      },
      tensors: { embedding: { role: 'state', description: 'the configuration', shape: ['n', 'k'] } },
    }),
    init: () => {
      const x = init.map((r) => [...r])
      return { x, stress: stressOf(euclid(x).dist, d), change: Number.MAX_VALUE }
    },
    step: (s, ctx) => {
      const { dist } = euclid(s.x)
      const disp = metric ? d : disparitiesOf(d, dist, n, ctx.t === 0)
      const b = new Float64Array(n * n)
      for (let i = 0; i < n; i++) {
        let row = 0
        for (let j = 0; j < n; j++) {
          if (i === j) continue
          const v = (disp[i * n + j] ?? 0) / ((dist[i * n + j] ?? 0) === 0 ? 1e-5 : (dist[i * n + j] as number))
          b[i * n + j] = -v
          row += v
        }
        b[i * n + i] = row
      }
      const x = Array.from({ length: n }, (_, i) =>
        Array.from({ length: k }, (_, c) => {
          let t = 0
          for (let j = 0; j < n; j++) t += (b[i * n + j] ?? 0) * (s.x[j]?.[c] ?? 0)
          return t / n
        }),
      )
      const after = euclid(x).dist
      const stress = stressOf(after, disp)
      let ss = 0
      for (const v of after) ss += v * v
      return { x, stress, change: (s.stress - stress) / (ss / 2) }
    },
    done: (s, t) => (t > 1 && s.change < eps ? 'converged' : undefined),
    observe: (s) => ({
      metrics: { stress: s.stress },
      tensors: { embedding: { role: 'state', value: s.x.flat(), shape: [n, k] } },
    }),
    serialise: (s) => stateToJson(s),
    deserialise: (j) => stateFromJson(j) as SmacofState,
  }
}

/**
 * Classical MDS as a one-step {@link Algorithm}, so that it runs and is probed like the others.
 *
 * @param d - Dissimilarities `[n, n]`, flat.
 * @param n - The number of rows.
 * @param k - The embedding dimension.
 * @returns The algorithm.
 * @example
 * ```ts
 * import { classicalMdsAlgorithm } from 'mdml-engine/manifold'
 * import { runAlgorithm } from 'mdml-engine/trace'
 * runAlgorithm(classicalMdsAlgorithm([0, 2, 2, 0], 2, 1), { maxSteps: 1 }).state.x.length // => 2
 * ```
 */
export function classicalMdsAlgorithm(d: ArrayLike<number>, n: number, k: number): Algorithm<SmacofState> {
  const rows = Array.from({ length: n }, (_, i) => Array.from({ length: n }, (_, j) => d[i * n + j] ?? 0))
  return {
    name: 'classical-mds',
    config: { n, k },
    schema: channels({
      metrics: { stress: { description: 'raw stress of the configuration' } },
      tensors: { embedding: { role: 'state', description: 'the configuration', shape: ['n', 'k'] } },
    }),
    init: () => ({
      x: rows.map(() => new Array<number>(k).fill(0)),
      stress: Number.MAX_VALUE,
      change: Number.MAX_VALUE,
    }),
    step: () => {
      const { embedding } = classicalMds(rows, k)
      const { dist } = euclid(embedding)
      let stress = 0
      for (let p = 0; p < n * n; p++) stress += ((dist[p] ?? 0) - (d[p] ?? 0)) ** 2
      return { x: embedding, stress: stress / 2, change: 0 }
    },
    done: (s) => (s.change === 0 ? 'completed' : undefined),
    observe: (s) => ({
      metrics: s.stress === Number.MAX_VALUE ? {} : { stress: s.stress },
      tensors: { embedding: { role: 'state', value: s.x.flat(), shape: [n, k] } },
    }),
    serialise: (s) => stateToJson(s),
    deserialise: (j) => stateFromJson(j) as SmacofState,
  }
}

/** Hyperparameters of {@link mds}. */
export interface MdsHyper {
  readonly nComponents: number
  /** Classical (closed form), metric SMACOF or non-metric SMACOF. */
  readonly variant: 'classical' | 'metric' | 'nonmetric'
  /** The dissimilarity among rows (`'precomputed'`: the data are the dissimilarity matrix). */
  readonly dissimilarity: DistanceName | DistanceAlias | 'precomputed'
  readonly maxIter: number
  readonly eps: number
  /** The starting configuration (`null`: uniform random). */
  readonly init: number[][] | null
}

/**
 * Multidimensional scaling (sklearn.manifold.MDS) on the iterative base: a Guttman transform per step, or classical MDS
 * in one step. New rows are placed by their nearest training rows (there is no closed-form extension).
 *
 * @index estimator Metric and non-metric multidimensional scaling (SMACOF).
 * @example
 * ```ts
 * import { mds } from 'mdml-engine/manifold'
 * mds.with({ variant: 'classical' }).fit({ x: [[0, 0], [3, 0], [0, 4]] }).params.embedding.length // => 3
 * ```
 */
export const mds: IterativeEstimator<MdsHyper, SmacofState, EmbeddingParams, EmbeddingVerbs> = iterative<
  MdsHyper,
  SmacofState,
  EmbeddingParams,
  EmbeddingVerbs
>({
  name: 'manifold/mds',
  kind: 'transformer',
  defaults: { nComponents: 2, variant: 'metric', dissimilarity: 'euclidean', maxIter: 300, eps: 1e-6, init: null },
  hyperSpace: (h) =>
    dict({
      nComponents: int(1, 3, { default: h.nComponents, label: 'dimensions' }),
      variant: choice(['classical', 'metric', 'nonmetric'] as const, { default: h.variant }),
      maxIter: int(1, 1000, { default: h.maxIter, label: 'iterations' }),
    }),
  algorithm: (data, h, key) => {
    const m = matrixView(data.x, 'mds', { minSamples: 2 })
    const { n, data: d } = distanceMatrix(data.x, h.dissimilarity as MetricSlot)
    if (h.variant === 'classical') return classicalMdsAlgorithm(d, n, h.nComponents)
    const init = h.init?.map((r) => [...r]) ?? (toArray(uniform(key, [m.n, h.nComponents])) as number[][])
    return smacofAlgorithm(d, init, { eps: h.eps, metric: h.variant === 'metric' })
  },
  maxSteps: (h) => (h.variant === 'classical' ? 1 : h.maxIter),
  params: (s, data) => ({
    embedding: s.x,
    data: trainingRows(data.x, 'mds'),
    objective: s.stress === Number.MAX_VALUE ? 0 : s.stress,
  }),
  verbs: (p) => embeddingVerbs(p, 'mds'),
})
