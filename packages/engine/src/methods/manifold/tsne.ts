import { WhiteboxError } from '../../lib/base/index.ts'
import { embeddingSteps } from '../../lib/exposition/index.ts'
import {
  defineExplainedEstimator,
  type Estimator,
  EstimatorClass,
  fitRunOptions,
  matrixView,
} from '../../lib/learn/index.ts'
import { geometry } from '../../lib/numerics/index.ts'
import { foldIn, type Key, normal } from '../../lib/random/index.ts'
import { choice, dict, int, real } from '../../lib/space/index.ts'
import { type TensorLike, toArray } from '../../lib/tensor/index.ts'
import type { Algorithm } from '../../lib/trace/index.ts'
import { pca } from '../decomposition/index.ts'
import { recordedRun } from './_record.ts'
import type { EmbeddingParams, FittedEmbedding } from './mds.ts'

const MACHINE_EPSILON = Number.EPSILON

/**
 * Conditional probabilities p(j | i) with each row's precision beta found by bisection so that the row's entropy is
 * log(perplexity) (sklearn's `_binary_search_perplexity`, natural logarithms).
 *
 * @param sqdist - Squared distances `[n][n]`.
 * @param perplexity - The target perplexity.
 * @returns The conditional probabilities and each row's beta (precision 1 / (2 sigma^2)).
 */
export function conditionalProbabilities(
  sqdist: readonly (readonly number[])[],
  perplexity: number,
): { readonly p: number[][]; readonly beta: number[] } {
  const n = sqdist.length
  const desired = Math.log(perplexity)
  const betas: number[] = []
  const p = sqdist.map((row, i) => {
    let beta = 1
    let lo = Number.NEGATIVE_INFINITY
    let hi = Number.POSITIVE_INFINITY
    let pi = new Array<number>(n).fill(0)
    for (let step = 0; step < 100; step++) {
      pi = row.map((d, j) => (j === i ? 0 : Math.exp(-d * beta)))
      let sum = pi.reduce((a, v) => a + v, 0)
      if (sum === 0) sum = 1e-8
      let sumDist = 0
      pi = pi.map((v, j) => {
        const q = v / sum
        sumDist += (row[j] ?? 0) * q
        return q
      })
      const diff = Math.log(sum) + beta * sumDist - desired
      if (Math.abs(diff) <= 1e-5) break
      if (diff > 0) {
        lo = beta
        beta = hi === Number.POSITIVE_INFINITY ? beta * 2 : (beta + hi) / 2
      } else {
        hi = beta
        beta = lo === Number.NEGATIVE_INFINITY ? beta / 2 : (beta + lo) / 2
      }
    }
    betas.push(beta)
    return pi
  })
  return { p, beta: betas }
}

/**
 * t-SNE's joint probabilities: (p(j|i) + p(i|j)) / (2n) floored at machine epsilon (sklearn's `_joint_probabilities`).
 *
 * @param x - Points `[n, d]`.
 * @param perplexity - The perplexity.
 * @returns The joint probabilities `[n][n]` (zero diagonal) and the calibrated bandwidths sigma_i.
 */
export function jointProbabilities(
  x: TensorLike,
  perplexity: number,
): { readonly p: number[][]; readonly sigmas: number[] } {
  const sq = toArray(geometry.cdist(x, x, 'sqeuclidean')) as number[][]
  const { p: cond, beta } = conditionalProbabilities(sq, perplexity)
  const sym = cond.map((r, i) => r.map((v, j) => v + (cond[j]?.[i] ?? 0)))
  const total = Math.max(
    sym.reduce((a, r) => a + r.reduce((s, v) => s + v, 0), 0),
    MACHINE_EPSILON,
  )
  return {
    p: sym.map((r, i) => r.map((v, j) => (i === j ? 0 : Math.max(v / total, MACHINE_EPSILON)))),
    sigmas: beta.map((b) => Math.sqrt(1 / (2 * b))),
  }
}

/**
 * The exact t-SNE objective and gradient (sklearn's `_kl_divergence`) for a Student-t kernel with `dof` degrees of
 * freedom.
 *
 * @param p - Joint probabilities `[n][n]`.
 * @param y - The embedding `[n][k]`.
 * @param dof - Degrees of freedom (default 1).
 * @returns The KL divergence and its gradient `[n][k]`.
 */
export function tsneGradient(
  p: readonly (readonly number[])[],
  y: readonly (readonly number[])[],
  dof = 1,
): { readonly kl: number; readonly grad: number[][] } {
  const n = y.length
  const k = y[0]?.length ?? 0
  const w = toArray(geometry.cdist(y as number[][], y as number[][], 'sqeuclidean')) as number[][]
  let total = 0
  const kern = w.map((r, i) =>
    r.map((v, j) => {
      if (i === j) return 0
      const out = (1 + v / dof) ** ((dof + 1) / -2)
      total += out
      return out
    }),
  )
  let kl = 0
  const grad = Array.from({ length: n }, () => new Array<number>(k).fill(0))
  const c = (2 * (dof + 1)) / dof
  for (let i = 0; i < n; i++)
    for (let j = 0; j < n; j++) {
      if (i === j) continue
      const q = Math.max((kern[i]?.[j] ?? 0) / total, MACHINE_EPSILON)
      const pij = p[i]?.[j] ?? 0
      kl += pij * Math.log(Math.max(pij, MACHINE_EPSILON) / q)
      const coef = (pij - q) * (kern[i]?.[j] ?? 0)
      const g = grad[i] as number[]
      for (let a = 0; a < k; a++) g[a] = (g[a] ?? 0) + c * coef * ((y[i]?.[a] ?? 0) - (y[j]?.[a] ?? 0))
    }
  return { kl, grad }
}

/** The state of t-SNE's gradient descent. */
export interface TsneState {
  readonly y: number[][]
  readonly update: number[][]
  readonly gains: number[][]
  readonly kl: number
}

/**
 * t-SNE's gradient descent with momentum and per-coordinate gains (sklearn's `_gradient_descent`) as a traced
 * algorithm: early exaggeration of P with momentum 0.5 for the first `exaggerationIter` steps, then momentum 0.8.
 *
 * @param p - Joint probabilities.
 * @param init - The initial embedding.
 * @param options - Learning rate, early exaggeration (12) and its length (250 steps).
 * @returns The algorithm.
 */
export function tsneAlgorithm(
  p: readonly (readonly number[])[],
  init: readonly (readonly number[])[],
  options: { readonly learningRate: number; readonly earlyExaggeration?: number; readonly exaggerationIter?: number },
): Algorithm<TsneState> {
  const ex = options.earlyExaggeration ?? 12
  const exIter = options.exaggerationIter ?? 250
  const pEx = p.map((r) => r.map((v) => v * ex))
  const lr = options.learningRate
  return {
    name: 'tsne',
    config: { learningRate: lr, earlyExaggeration: ex, exaggerationIter: exIter },
    init: () => ({
      y: init.map((r) => [...r]),
      update: init.map((r) => r.map(() => 0)),
      gains: init.map((r) => r.map(() => 1)),
      kl: tsneGradient(p, init).kl,
    }),
    step: (s, ctx) => {
      const exploring = ctx.t < exIter
      const momentum = exploring ? 0.5 : 0.8
      const { kl, grad } = tsneGradient(exploring ? pEx : p, s.y)
      const gains = s.gains.map((r, i) =>
        r.map((gv, a) => {
          const inc = (s.update[i]?.[a] ?? 0) * (grad[i]?.[a] ?? 0) < 0
          return Math.max(inc ? gv + 0.2 : gv * 0.8, 0.01)
        }),
      )
      const update = s.update.map((r, i) =>
        r.map((u, a) => momentum * u - lr * (gains[i]?.[a] ?? 1) * (grad[i]?.[a] ?? 0)),
      )
      const y = s.y.map((r, i) => r.map((v, a) => v + (update[i]?.[a] ?? 0)))
      return { y, update, gains, kl: exploring ? tsneGradient(p, y).kl : kl }
    },
    observe: (s) => ({ metrics: { kl: s.kl } }),
  }
}

/** Hyperparameters of t-SNE. */
export type TsneHyper = {
  readonly nComponents?: number
  readonly perplexity?: number
  readonly earlyExaggeration?: number
  /** The step size, or 'auto' (max(n / exaggeration / 4, 50), sklearn's default). */
  readonly learningRate?: number | 'auto'
  readonly maxIter?: number
  /** PCA (scaled to standard deviation 1e-4, default) or random Gaussian initialisation, or given positions. */
  readonly init?: 'pca' | 'random' | readonly (readonly number[])[]
  /** Record the embedding every this many iterations (default 25). */
  readonly recordEvery?: number
}

/**
 * t-SNE (van der Maaten and Hinton 2008; sklearn.manifold.TSNE with the exact gradient): perplexity-calibrated joint
 * probabilities, then traced gradient descent on the KL divergence to a Student-t similarity. Its exposition is an
 * `embeddingSteps` record: positions every few iterations, the KL divergence, each row's calibrated bandwidth and
 * the strongest affinities as a graph. TODO: the Barnes-Hut gradient.
 *
 * @param hyper - Dimension, perplexity, exaggeration, learning rate, iterations, initialisation and recording.
 * @returns The estimator.
 */
export function tsne(hyper: TsneHyper = {}): Estimator<TsneHyper, FittedEmbedding> {
  return defineExplainedEstimator<TsneHyper, EmbeddingParams, { readonly nFeaturesIn: number }>(
    {
      name: 'tsne',
      kind: 'transformer',
      hyperSpace: (h) =>
        dict({
          perplexity: real(2, 100, { default: h.perplexity ?? 30, label: 'perplexity' }),
          maxIter: int(250, 2000, { default: h.maxIter ?? 1000, label: 'iterations' }),
          earlyExaggeration: real(1, 50, { default: h.earlyExaggeration ?? 12 }),
          init: choice(['pca', 'random'], { default: typeof h.init === 'string' ? h.init : 'pca' }),
        }),
      fit: (data, h, options) => {
        const m = matrixView(data.x, 'tsne', { minSamples: 4 })
        const perplexity = h.perplexity ?? 30
        if (!(perplexity < m.n)) throw new WhiteboxError('value', `tsne: perplexity must be below n = ${m.n}`)
        const k = h.nComponents ?? 2
        const run = fitRunOptions(options, 'tsne')
        const key = run.key as Key
        const init = h.init ?? 'pca'
        let start: number[][]
        if (typeof init !== 'string') start = init.map((r) => [...r])
        else if (init === 'random')
          start = (toArray(normal(foldIn(key, 0), [m.n, k])) as number[][]).map((r) => r.map((v) => v * 1e-4))
        else {
          const z = toArray(pca({ nComponents: k }).fit(data).transform(data.x)) as number[][]
          const col = z.map((r) => r[0] ?? 0)
          const mu = col.reduce((a, v) => a + v, 0) / col.length
          const sd = Math.sqrt(col.reduce((a, v) => a + (v - mu) ** 2, 0) / col.length) || 1
          start = z.map((r) => r.map((v) => (v / sd) * 1e-4))
        }
        const { p, sigmas } = jointProbabilities(data.x, perplexity)
        const ex = h.earlyExaggeration ?? 12
        const lr =
          h.learningRate === undefined || h.learningRate === 'auto' ? Math.max(m.n / ex / 4, 50) : h.learningRate
        const out = recordedRun(
          tsneAlgorithm(p, start, { learningRate: lr, earlyExaggeration: ex, exaggerationIter: 250 }),
          { ...run, key: foldIn(key, 1), maxSteps: h.maxIter ?? 1000 },
          h.recordEvery ?? 25,
          (s) => s.y,
          (s) => s.kl,
        )
        const strongest = Math.max(1, Math.min(m.n - 1, Math.round(perplexity / 3)))
        const graph = p.flatMap((row, i) =>
          row
            .map((v, j) => [j, v] as const)
            .filter(([j]) => j !== i)
            .sort((a, b) => b[1] - a[1])
            .slice(0, strongest)
            .map(([j, v]): [number, number, number] => [i, j, v]),
        )
        return {
          params: { embedding: out.state.y, objective: out.state.kl, nIter: out.steps, nFeaturesIn: m.d },
          exposition: [
            embeddingSteps({
              method: 'tsne',
              positions: out.positions,
              iterations: out.iterations,
              objective: out.objective,
              graph,
              bandwidths: sigmas,
            }),
          ],
        }
      },
      methods: (p) => ({ nFeaturesIn: p.nFeaturesIn }),
    },
    hyper,
  )
}

/** t-SNE as a class. */
export class TSNE extends EstimatorClass<TsneHyper, FittedEmbedding> {
  /**
   * Make an unfitted model.
   *
   * @param hyper - The hyperparameters of {@link tsne}.
   */
  constructor(hyper: TsneHyper = {}) {
    super(tsne(hyper))
  }
}
