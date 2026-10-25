/**
 * t-SNE (van der Maaten and Hinton, 2008; scikit-learn's `TSNE` with the exact gradient): neighbour probabilities
 * calibrated to a perplexity in the data are matched by a heavy-tailed Student-t similarity in the embedding, by
 * gradient descent on their KL divergence with early exaggeration, momentum and gains, traced step by step.
 * `tsneSession` sets a run up for pages that step it live.
 */

import { WhiteboxError } from '../../lib/base/index.ts'
import { embeddingSteps } from '../../lib/exposition/index.ts'
import {
  defineExplainedEstimator,
  type Estimator,
  EstimatorClass,
  fitRunOptions,
  type LearnData,
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
 * The exact KL divergence and gradient on flat buffers: `p` `[n * n]` row-major, `y` `[n * k]`, the gradient written
 * into `grad` `[n * k]` (unless it is undefined: the divergence alone) and the Student-t kernel into `kern` `[n * n]`
 * (scratch, reused across iterations). The gradient uses `exaggeration` times p (early exaggeration) while the
 * divergence is of p itself. Sums run in the order of sklearn's `_kl_divergence`.
 */
function klAndGradient(
  p: Float64Array,
  y: Float64Array,
  n: number,
  k: number,
  dof: number,
  exaggeration: number,
  kern: Float64Array,
  grad: Float64Array | undefined,
): number {
  const power = (dof + 1) / -2
  let total = 0
  for (let i = 0; i < n; i++)
    for (let j = 0; j < n; j++) {
      if (i === j) {
        kern[i * n + j] = 0
        continue
      }
      let d2 = 0
      for (let a = 0; a < k; a++) {
        const diff = (y[i * k + a] ?? 0) - (y[j * k + a] ?? 0)
        d2 += diff * diff
      }
      const v = (1 + d2 / dof) ** power
      kern[i * n + j] = v
      total += v
    }
  let kl = 0
  grad?.fill(0)
  const c = (2 * (dof + 1)) / dof
  for (let i = 0; i < n; i++)
    for (let j = 0; j < n; j++) {
      if (i === j) continue
      const kij = kern[i * n + j] ?? 0
      const q = Math.max(kij / total, MACHINE_EPSILON)
      const pij = p[i * n + j] ?? 0
      kl += pij * Math.log(Math.max(pij, MACHINE_EPSILON) / q)
      if (grad === undefined) continue
      const coef = (exaggeration === 1 ? pij : pij * exaggeration) - q
      const ck = c * (coef * kij)
      for (let a = 0; a < k; a++)
        grad[i * k + a] = (grad[i * k + a] ?? 0) + ck * ((y[i * k + a] ?? 0) - (y[j * k + a] ?? 0))
    }
  return kl
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
  const grad = new Float64Array(n * k)
  const kl = klAndGradient(
    Float64Array.from(p.flat()),
    Float64Array.from(y.flat()),
    n,
    k,
    dof,
    1,
    new Float64Array(n * n),
    grad,
  )
  return { kl, grad: Array.from({ length: n }, (_, i) => Array.from(grad.subarray(i * k, (i + 1) * k))) }
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
 * algorithm: early exaggeration of P with momentum 0.5 for the first `exaggerationIter` steps, then momentum 0.8. The
 * reported KL divergence is always of the unexaggerated P. Each step is one pass over the pairs, into buffers reused
 * across steps.
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
  const lr = options.learningRate
  const n = init.length
  const k = init[0]?.length ?? 0
  const pFlat = Float64Array.from(p.flat())
  const kern = new Float64Array(n * n)
  const grad = new Float64Array(n * k)
  const y = new Float64Array(n * k)
  return {
    name: 'tsne',
    config: { learningRate: lr, earlyExaggeration: ex, exaggerationIter: exIter },
    init: () => {
      y.set(init.flat())
      return {
        y: init.map((r) => [...r]),
        update: init.map((r) => r.map(() => 0)),
        gains: init.map((r) => r.map(() => 1)),
        kl: klAndGradient(pFlat, y, n, k, 1, 1, kern, grad),
      }
    },
    step: (s, ctx) => {
      const exploring = ctx.t < exIter
      const momentum = exploring ? 0.5 : 0.8
      for (const [i, r] of s.y.entries()) for (const [a, v] of r.entries()) y[i * k + a] = v
      // The divergence of p at the step's starting positions, and the gradient (of exaggerated p while exploring).
      const klBefore = klAndGradient(pFlat, y, n, k, 1, exploring ? ex : 1, kern, grad)
      const gains = s.gains.map((r, i) =>
        r.map((gv, a) => {
          const inc = (s.update[i]?.[a] ?? 0) * (grad[i * k + a] ?? 0) < 0
          return Math.max(inc ? gv + 0.2 : gv * 0.8, 0.01)
        }),
      )
      const update = s.update.map((r, i) =>
        r.map((u, a) => momentum * u - lr * (gains[i]?.[a] ?? 1) * (grad[i * k + a] ?? 0)),
      )
      const next = s.y.map((r, i) => r.map((v, a) => v + (update[i]?.[a] ?? 0)))
      if (!exploring) return { y: next, update, gains, kl: klBefore }
      // While exploring, the divergence reported is at the new positions (a divergence-only pass).
      for (const [i, r] of next.entries()) for (const [a, v] of r.entries()) y[i * k + a] = v
      return { y: next, update, gains, kl: klAndGradient(pFlat, y, n, k, 1, 1, kern, undefined) }
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

/** A t-SNE run set up and ready to step: what {@link tsne} fits, for pages that run it live. */
export interface TsneSession {
  readonly algorithm: Algorithm<TsneState>
  /** The iteration budget (`maxIter`). */
  readonly maxSteps: number
  /** The joint probabilities and each row's calibrated bandwidth. */
  readonly p: readonly (readonly number[])[]
  readonly sigmas: readonly number[]
}

/**
 * Set up t-SNE without running it: the perplexity-calibrated joint probabilities, the initial embedding (PCA scaled
 * small, random or given) and the gradient-descent algorithm {@link tsne} runs. Step it with `runAlgorithm`, or live
 * frame by frame (render's `useLiveRun`), and read each state's positions (`state.y`) and divergence (`state.kl`).
 *
 * @param data - The rows to embed.
 * @param hyper - As {@link tsne}.
 * @param key - The key for random initialisation.
 * @returns The algorithm, the iteration budget, and the affinities and bandwidths.
 * @throws WhiteboxError `value` when the perplexity is not below the number of rows.
 */
export function tsneSession(data: LearnData, hyper: TsneHyper, key: Key): TsneSession {
  const m = matrixView(data.x, 'tsne', { minSamples: 4 })
  const perplexity = hyper.perplexity ?? 30
  if (!(perplexity < m.n)) throw new WhiteboxError('value', `tsne: perplexity must be below n = ${m.n}`)
  const k = hyper.nComponents ?? 2
  const init = hyper.init ?? 'pca'
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
  const ex = hyper.earlyExaggeration ?? 12
  const lr =
    hyper.learningRate === undefined || hyper.learningRate === 'auto' ? Math.max(m.n / ex / 4, 50) : hyper.learningRate
  return {
    algorithm: tsneAlgorithm(p, start, { learningRate: lr, earlyExaggeration: ex, exaggerationIter: 250 }),
    maxSteps: hyper.maxIter ?? 1000,
    p,
    sigmas,
  }
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
          perplexity: real(2, 100, {
            default: h.perplexity ?? 30,
            label: 'perplexity',
            description: 'The effective number of neighbours each point’s bandwidth is calibrated to.',
          }),
          maxIter: int(250, 2000, { default: h.maxIter ?? 1000, label: 'iterations' }),
          earlyExaggeration: real(1, 50, {
            default: h.earlyExaggeration ?? 12,
            label: 'early exaggeration',
            description: 'How much P is multiplied by for the first 250 iterations, pulling clusters apart early.',
          }),
          init: choice(['pca', 'random'], {
            default: typeof h.init === 'string' ? h.init : 'pca',
            label: 'initialisation',
            description: 'Start from the first principal components (scaled small) or from small Gaussian noise.',
          }),
        }),
      fit: (data, h, options) => {
        const run = fitRunOptions(options, 'tsne')
        const key = run.key as Key
        const session = tsneSession(data, h, key)
        const { p, sigmas } = session
        const m = matrixView(data.x, 'tsne', { minSamples: 4 })
        const perplexity = h.perplexity ?? 30
        const out = recordedRun(
          session.algorithm,
          { ...run, key: foldIn(key, 1), maxSteps: session.maxSteps },
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
