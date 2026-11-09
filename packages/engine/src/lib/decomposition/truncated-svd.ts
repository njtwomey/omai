/**
 * Truncated singular value decomposition (scikit-learn's `TruncatedSVD`, latent semantic analysis on term matrices): the
 * leading singular triplets of the data as it is, without centring, so it is PCA of the uncentred matrix. The scores
 * are `X V^T` (which are `U Sigma` on the training rows), the inverse transform `Z V`. The solver is the exact thin SVD
 * or the randomised one of `numerics/linalg` (Halko, Martinsson and Tropp), with its oversampling and power iterations.
 */

import { OmaiError } from '../base/index.ts'
import { type IterativeEstimator, iterative, matrixView } from '../learn/index.ts'
import { linalg } from '../numerics/index.ts'
import { choice, dict, int } from '../space/index.ts'
import { fromStorage, type Tensor, type TensorLike, toFlat } from '../tensor/index.ts'
import { type Algorithm, channels, stateFromJson, stateToJson } from '../trace/index.ts'
import { flipSigns, project, reconstruct, thinSvd } from './_linear.ts'

/** The state of {@link truncatedSvdAlgorithm}: the axes and their spectrum; empty before the step. */
export type TruncatedSvdState = {
  /** components[c][f]: the right singular vectors, by decreasing singular value. */
  readonly components: number[][]
  readonly singularValues: number[]
  /** The variance of the scores of each component (scikit-learn's `explained_variance_`). */
  readonly explainedVariance: number[]
  readonly explainedVarianceRatio: number[]
  /** The total variance of the features of the data (not centred away: the column variances summed). */
  readonly totalVariance: number
  /** The number of features of the data. */
  readonly nFeatures: number
}

/** Options of {@link truncatedSvdAlgorithm}. */
export interface TruncatedSvdOptions {
  /** `full`: the exact thin SVD; `randomized` (default): randomised range finding. */
  readonly solver?: 'full' | 'randomized'
  /** Power iterations of the randomised solver (default 5, scikit-learn's). */
  readonly nIter?: number
  /** Extra sketch columns of the randomised solver (default 10, scikit-learn's). */
  readonly nOversamples?: number
}

const schema = channels({
  metrics: { explained: { description: 'the share of the feature variance the components explain' } },
  tensors: { components: { role: 'state', description: 'the right singular vectors', shape: ['k', 'd'] } },
})

const variance = (values: ArrayLike<number>): number => {
  let m = 0
  for (let i = 0; i < values.length; i++) m += (values[i] ?? 0) / values.length
  let v = 0
  for (let i = 0; i < values.length; i++) v += ((values[i] ?? 0) - m) ** 2 / values.length
  return v
}

/**
 * Truncated SVD as a traced algorithm: one step finds the leading `k` singular triplets of the uncentred data and the
 * variance their scores explain, as scikit-learn's `TruncatedSVD.fit`.
 *
 * @param x - Data `[n, d]`.
 * @param k - Components, in `1..min(n, d)`.
 * @param options - The solver and, for the randomised one, its power iterations and oversampling.
 * @returns The algorithm; its randomised solver draws from the key of the run.
 * @throws OmaiError with code `value` unless `k` is an integer in `1..min(n, d)`.
 * @index algorithm Truncated SVD of the uncentred data, exact or randomised.
 * @example
 * ```ts
 * import { truncatedSvdAlgorithm } from 'omai-engine/decomposition'
 * import { runAlgorithm } from 'omai-engine/trace'
 * const x = [[3, 0], [0, 1], [3, 0], [0, 1]]
 * runAlgorithm(truncatedSvdAlgorithm(x, 1, { solver: 'full' }), { maxSteps: 1 }).state.singularValues[0].toFixed(4) // => '4.2426'
 * ```
 */
export function truncatedSvdAlgorithm(
  x: TensorLike,
  k: number,
  options: TruncatedSvdOptions = {},
): Algorithm<TruncatedSvdState> {
  const m = matrixView(x, 'truncatedSvd', { minSamples: 1 })
  const full = Math.min(m.n, m.d)
  if (!(Number.isInteger(k) && k >= 1 && k <= full))
    throw new OmaiError('value', `truncatedSvd: nComponents must be in 1..${full}, got ${k}`)
  const solver = options.solver ?? 'randomized'
  let totalVariance = 0
  for (let j = 0; j < m.d; j++)
    totalVariance += variance(Array.from({ length: m.n }, (_, i) => m.data[i * m.d + j] ?? 0))
  return {
    name: 'truncated-svd',
    schema,
    config: { k, solver, nIter: options.nIter ?? 5, nOversamples: options.nOversamples ?? 10 },
    init: () => ({
      nFeatures: m.d,
      components: [],
      singularValues: [],
      explainedVariance: [],
      explainedVarianceRatio: [],
      totalVariance,
    }),
    step: (_s, ctx) => {
      let vt: number[][]
      let sv: number[]
      if (solver === 'randomized') {
        const r = linalg.randomizedSvd(fromStorage(m.data, [m.n, m.d], 'float64'), k, {
          key: ctx.key,
          powerIterations: options.nIter ?? 5,
          oversample: options.nOversamples ?? 10,
        })
        const flat = toFlat(r.vh)
        vt = Array.from({ length: k }, (_, c) => Array.from(flat.slice(c * m.d, (c + 1) * m.d)))
        sv = Array.from(toFlat(r.s)).slice(0, k)
      } else {
        const t = thinSvd(m.data, m.n, m.d)
        vt = t.vt.slice(0, k)
        sv = t.s.slice(0, k)
      }
      flipSigns(undefined, vt)
      const scores = project(x, new Array<number>(m.d).fill(0), vt)
      const flat = toFlat(scores)
      const explainedVariance = vt.map((_, c) => variance(Array.from({ length: m.n }, (_, i) => flat[i * k + c] ?? 0)))
      return {
        components: vt,
        singularValues: sv,
        explainedVariance,
        explainedVarianceRatio: explainedVariance.map((v) => (totalVariance === 0 ? 0 : v / totalVariance)),
        totalVariance,
        nFeatures: m.d,
      }
    },
    done: (s) => (s.components.length >= k ? 'completed' : undefined),
    observe: (s) => ({
      metrics: { explained: s.explainedVarianceRatio.reduce((a, v) => a + v, 0) },
      tensors: { components: { role: 'state', value: s.components.flat(), shape: [s.components.length, m.d] } },
    }),
    serialise: (s) => stateToJson(s),
    deserialise: (j) => stateFromJson(j) as TruncatedSvdState,
  }
}

/** Hyperparameters of {@link truncatedSvd}. */
export interface TruncatedSvdHyper {
  readonly nComponents: number
  readonly solver: 'full' | 'randomized'
  readonly nIter: number
  readonly nOversamples: number
}

/** The verbs of a fitted truncated SVD. */
export interface TruncatedSvdVerbs {
  /** The scores `X V^T` of rows `[m, k]`. */
  readonly transform: (x: TensorLike) => Tensor
  /** `Z V`: rows back in the feature space. */
  readonly inverseTransform: (z: TensorLike) => Tensor
  readonly nFeaturesIn: number
  readonly featureNamesOut: (input?: readonly string[]) => string[]
}

/**
 * Truncated SVD (`sklearn.decomposition.TruncatedSVD`) on the iterative base: PCA without centring, the transformer
 * latent semantic analysis uses on term-document matrices. `params` holds the components, the singular values and the
 * explained variance (of the scores) and its ratio to the total feature variance.
 *
 * @index estimator Truncated SVD of uncentred data (LSA); exact or randomised.
 * @example
 * ```ts
 * import { truncatedSvd } from 'omai-engine/decomposition'
 * import { toArray } from 'omai-engine/tensor'
 * const x = [[3, 0], [0, 1], [3, 0], [0, 1]]
 * const f = truncatedSvd.with({ nComponents: 1, solver: 'full' }).fit({ x })
 * toArray(f.transform([[3, 0]])) // => [[3]]
 * ```
 */
export const truncatedSvd: IterativeEstimator<
  TruncatedSvdHyper,
  TruncatedSvdState,
  TruncatedSvdState,
  TruncatedSvdVerbs
> = iterative<TruncatedSvdHyper, TruncatedSvdState, TruncatedSvdState, TruncatedSvdVerbs>({
  name: 'decomposition/truncated-svd',
  kind: 'transformer',
  defaults: { nComponents: 2, solver: 'randomized', nIter: 5, nOversamples: 10 },
  hyperSpace: (h) =>
    dict({
      nComponents: int(1, 50, { default: h.nComponents, label: 'components' }),
      solver: choice(['full', 'randomized'] as const, { default: h.solver, label: 'solver' }),
    }),
  algorithm: (data, h) =>
    truncatedSvdAlgorithm(data.x, h.nComponents, { solver: h.solver, nIter: h.nIter, nOversamples: h.nOversamples }),
  maxSteps: () => 1,
  params: (s) => s,
  verbs: (p) => {
    const d = p.nFeatures
    const zero = new Array<number>(d).fill(0)
    return {
      transform: (x) => project(x, zero, p.components),
      inverseTransform: (z) => reconstruct(z, zero, p.components),
      nFeaturesIn: d,
      featureNamesOut: () => p.components.map((_, c) => `truncatedsvd${c}`),
    }
  },
})
