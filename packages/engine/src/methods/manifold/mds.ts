/**
 * Multidimensional scaling (as in scikit-learn's `MDS`): placing points in a few dimensions so their distances match
 * given dissimilarities. Classical (Torgerson) MDS in closed form by an eigendecomposition, and metric SMACOF, which
 * lowers the stress by repeated Guttman transforms, traced iteration by iteration. It also defines the embedding types
 * the other manifold methods share.
 */

import { MdmlError } from '../../lib/base/index.ts'
import { embeddingSteps } from '../../lib/exposition/index.ts'
import {
  defineExplainedEstimator,
  type Estimator,
  EstimatorClass,
  type ExplainedFitted,
  fitRunOptions,
  matrixView,
} from '../../lib/learn/index.ts'
import { geometry, linalg } from '../../lib/numerics/index.ts'
import { type Key, uniform } from '../../lib/random/index.ts'
import { bool, dict, int } from '../../lib/space/index.ts'
import { asTensor, type TensorLike, toArray, toFlat } from '../../lib/tensor/index.ts'
import type { Algorithm } from '../../lib/trace/index.ts'
import { recordedRun } from './_record.ts'

/** Rows of an embedding. */
export type Embedding = number[][]

/**
 * The square dissimilarity matrix of points (Euclidean), or the given matrix when precomputed.
 *
 * @param x - Points `[n, d]`, or dissimilarities `[n, n]`.
 * @param precomputed - Whether x is already a dissimilarity matrix.
 * @returns Dissimilarities `[n][n]`.
 */
export function dissimilarities(x: TensorLike, precomputed = false): number[][] {
  return toArray(precomputed ? asTensor(x) : geometry.cdist(x, x)) as number[][]
}

/**
 * Classical (Torgerson) MDS: the top eigenvectors of the double-centred squared dissimilarities -1/2 J D^2 J, scaled
 * by the square roots of their eigenvalues (negative ones clipped to zero).
 *
 * @param d - Dissimilarities `[n][n]`.
 * @param k - The embedding dimension.
 * @returns The embedding `[n][k]` and the eigenvalues used.
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

/** The state of SMACOF: the configuration, its stress, and the recorded steps. */
export interface SmacofState {
  readonly x: Embedding
  readonly stress: number
  readonly change: number
}

/**
 * Metric SMACOF (de Leeuw 1977; sklearn's `_smacof_single`) as a traced algorithm: each step applies the Guttman
 * transform X <- B(X) X / n and records the raw stress; it stops when the stress decrease relative to half the sum
 * of squared distances is below eps.
 *
 * @param d - Dissimilarities `[n][n]`.
 * @param init - The initial configuration `[n][k]`.
 * @param eps - The relative tolerance (default 1e-6).
 * @returns The algorithm.
 */
export function smacofAlgorithm(
  d: readonly (readonly number[])[],
  init: readonly (readonly number[])[],
  eps = 1e-6,
): Algorithm<SmacofState> {
  const n = d.length
  const distances = (x: Embedding): number[][] => toArray(geometry.cdist(x, x)) as number[][]
  const stressOf = (dist: number[][]): number =>
    dist.reduce((a, r, i) => a + r.reduce((s, v, j) => s + (v - (d[i]?.[j] ?? 0)) ** 2, 0), 0) / 2
  return {
    name: 'smacof',
    config: { n, eps },
    init: () => {
      const x = init.map((r) => [...r])
      return { x, stress: stressOf(distances(x)), change: Number.POSITIVE_INFINITY }
    },
    step: (s) => {
      const dist = distances(s.x)
      const ratio = dist.map((r, i) => r.map((v, j) => (d[i]?.[j] ?? 0) / (v === 0 ? 1e-5 : v)))
      const bm = ratio.map((r, i) => r.map((v, j) => (i === j ? r.reduce((a, w) => a + w, 0) - v : -v)))
      const k = s.x[0]?.length ?? 0
      const x = bm.map((r) =>
        Array.from({ length: k }, (_, c) => r.reduce((a, v, j) => a + v * (s.x[j]?.[c] ?? 0), 0) / n),
      )
      const after = distances(x)
      const stress = stressOf(after)
      const ss = after.reduce((a, r) => a + r.reduce((t, v) => t + v * v, 0), 0)
      return { x, stress, change: (s.stress - stress) / (ss / 2) }
    },
    done: (s, t) => (t > 1 && s.change < eps ? 'converged' : undefined),
    observe: (s) => ({ metrics: { stress: s.stress } }),
  }
}

/** Hyperparameters of MDS. */
export type MdsHyper = {
  readonly nComponents?: number
  /** Classical MDS in closed form instead of SMACOF from a random start (default false). */
  readonly classical?: boolean
  /** The data are a dissimilarity matrix (default false). */
  readonly precomputed?: boolean
  readonly maxIter?: number
  readonly eps?: number
  /** The starting configuration (default uniform random). */
  readonly init?: readonly (readonly number[])[]
  /** Record the configuration every this many iterations (default 5). */
  readonly recordEvery?: number
}

/** The parameters of a fitted embedding: the training embedding and its objective. */
export type EmbeddingParams = {
  readonly embedding: number[][]
  readonly objective: number
  readonly nIter: number
  readonly nFeaturesIn: number
}

/** A fitted embedding of the training rows (no out-of-sample transform). */
export type FittedEmbedding = ExplainedFitted<EmbeddingParams, { readonly nFeaturesIn: number }>

/**
 * Multidimensional scaling (sklearn.manifold.MDS): metric SMACOF traced iteration by iteration (stress per recorded
 * step), or classical MDS in closed form. Its exposition is an `embeddingSteps` record. TODO: non-metric MDS (needs
 * an isotonic regression in the core).
 *
 * @param hyper - Dimension, the variant, precomputed dissimilarities, iteration cap, tolerance and start.
 * @returns The estimator.
 */
export function mds(hyper: MdsHyper = {}): Estimator<MdsHyper, FittedEmbedding> {
  return defineExplainedEstimator<MdsHyper, EmbeddingParams, { readonly nFeaturesIn: number }>(
    {
      name: 'mds',
      kind: 'transformer',
      hyperSpace: (h) =>
        dict({
          nComponents: int(1, 3, { default: h.nComponents ?? 2, label: 'dimensions' }),
          classical: bool({
            label: 'classical (Torgerson)',
            description:
              'Classical MDS from the eigendecomposition of the double-centred distances, instead of SMACOF.',
            default: h.classical ?? false,
          }),
          maxIter: int(1, 1000, { default: h.maxIter ?? 300, label: 'iterations' }),
        }),
      fit: (data, h, options) => {
        const m = matrixView(data.x, 'mds', { minSamples: 2 })
        const d = dissimilarities(data.x, h.precomputed ?? false)
        const k = h.nComponents ?? 2
        if (h.classical === true) {
          const { embedding, eigenvalues } = classicalMds(d, k)
          return {
            params: { embedding, objective: eigenvalues.reduce((a, v) => a + v, 0), nIter: 0, nFeaturesIn: m.d },
            exposition: [embeddingSteps({ method: 'classicalMds', positions: [embedding], iterations: [0] })],
          }
        }
        const key = fitRunOptions(options, 'mds').key as Key
        const init = h.init?.map((r) => [...r]) ?? (toArray(uniform(key, [m.n, k])) as number[][])
        const run = recordedRun(
          smacofAlgorithm(d, init, h.eps ?? 1e-6),
          { ...fitRunOptions(options, 'mds'), maxSteps: h.maxIter ?? 300 },
          h.recordEvery ?? 5,
          (st) => st.x,
          (st) => st.stress,
        )
        const out = run
        return {
          params: { embedding: out.state.x, objective: out.state.stress, nIter: out.steps, nFeaturesIn: m.d },
          exposition: [
            embeddingSteps({
              method: 'smacof',
              positions: run.positions,
              iterations: run.iterations,
              objective: run.objective,
            }),
          ],
        }
      },
      methods: (p) => ({ nFeaturesIn: p.nFeaturesIn }),
    },
    hyper,
  )
}

/** MDS as a class. */
export class MDS extends EstimatorClass<MdsHyper, FittedEmbedding> {
  /**
   * Make an unfitted model.
   *
   * @param hyper - The hyperparameters of {@link mds}.
   */
  constructor(hyper: MdsHyper = {}) {
    super(mds(hyper))
  }
}
