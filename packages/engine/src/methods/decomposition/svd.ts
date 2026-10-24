import { WhiteboxError } from '../../lib/base/index.ts'
import { componentsRecord } from '../../lib/exposition/index.ts'
import {
  defineExplainedEstimator,
  type Estimator,
  EstimatorClass,
  type ExplainedFitted,
  featureNamesOf,
  fitRunOptions,
  matrixView,
} from '../../lib/learn/index.ts'
import { linalg } from '../../lib/numerics/index.ts'
import type { Key } from '../../lib/random/index.ts'
import { choice, dict, int } from '../../lib/space/index.ts'
import { type Tensor, type TensorLike, toArray, toFlat } from '../../lib/tensor/index.ts'
import { flipSigns, project, reconstruct, rowsOf, thinSvd } from './_linear.ts'

/** The parameters of a fitted truncated SVD. */
export type TruncatedSvdParams = {
  readonly components: number[][]
  readonly singularValues: number[]
  readonly explainedVariance: number[]
  readonly explainedVarianceRatio: number[]
}

/** The verbs of an uncentred linear projection. */
export interface UncentredVerbs {
  readonly transform: (x: TensorLike) => Tensor
  readonly inverseTransform: (z: TensorLike) => Tensor
  readonly nFeaturesIn: number
}

/** Hyperparameters of the truncated SVD. */
export type TruncatedSvdHyper = {
  readonly nComponents?: number
  /** The exact thin SVD, or the randomised range finder (default exact). */
  readonly algorithm?: 'exact' | 'randomized'
}

/** A fitted truncated SVD. */
export type FittedTruncatedSvd = ExplainedFitted<TruncatedSvdParams, UncentredVerbs>

const variance = (col: readonly number[]): number => {
  const mu = col.reduce((a, v) => a + v, 0) / col.length
  return col.reduce((a, v) => a + (v - mu) ** 2, 0) / col.length
}

/**
 * Truncated SVD without centring (sklearn.decomposition.TruncatedSVD, latent semantic analysis) by the SVD of
 * `numerics/linalg`. Its exposition is a `components` record (axes, the variance of each score and its share of the
 * total, the training scores).
 *
 * @param hyper - Components and the algorithm.
 * @returns The estimator.
 */
export function truncatedSvd(hyper: TruncatedSvdHyper = {}): Estimator<TruncatedSvdHyper, FittedTruncatedSvd> {
  return defineExplainedEstimator<TruncatedSvdHyper, TruncatedSvdParams, UncentredVerbs>(
    {
      name: 'truncated-svd',
      kind: 'transformer',
      hyperSpace: (h) =>
        dict({
          nComponents: int(1, 50, { default: h.nComponents ?? 2, label: 'components' }),
          algorithm: choice(['exact', 'randomized'], { default: h.algorithm ?? 'exact' }),
        }),
      fit: (data, h, options) => {
        const m = matrixView(data.x, 'truncatedSvd')
        const k = h.nComponents ?? 2
        if (!(Number.isInteger(k) && k >= 1 && k <= Math.min(m.n, m.d)))
          throw new WhiteboxError('value', `truncatedSvd: nComponents must be in 1..${Math.min(m.n, m.d)}`)
        let s: number[]
        let vt: number[][]
        if (h.algorithm === 'randomized') {
          const r = linalg.randomizedSvd(data.x, k, { key: fitRunOptions(options, 'truncated-svd').key as Key })
          s = toFlat(r.s)
          vt = toArray(r.vh) as number[][]
          flipSigns(undefined, vt)
        } else ({ s, vt } = thinSvd(m.data, m.n, m.d))
        const components = vt.slice(0, k)
        const zero = new Array<number>(m.d).fill(0)
        const scores = rowsOf(project(data.x, zero, components))
        const total = Array.from({ length: m.d }, (_, j) =>
          variance(Array.from({ length: m.n }, (_, i) => m.data[i * m.d + j] ?? 0)),
        ).reduce((a, v) => a + v, 0)
        const ev = components.map((_, c) => variance(scores.map((r) => r[c] ?? 0)))
        return {
          params: {
            components,
            singularValues: s.slice(0, k),
            explainedVariance: ev,
            explainedVarianceRatio: ev.map((v) => v / total),
          },
          exposition: [
            componentsRecord({
              method: 'truncatedSvd',
              features: featureNamesOf(data, m.d),
              components,
              explainedVariance: ev,
              explainedVarianceRatio: ev.map((v) => v / total),
              scores,
            }),
          ],
        }
      },
      methods: (p) => {
        const zero = new Array<number>(p.components[0]?.length ?? 0).fill(0)
        return {
          transform: (x) => project(x, zero, p.components),
          inverseTransform: (z) => reconstruct(z, zero, p.components),
          nFeaturesIn: zero.length,
        }
      },
    },
    hyper,
  )
}

/** The truncated SVD as a class. */
export class TruncatedSVD extends EstimatorClass<TruncatedSvdHyper, FittedTruncatedSvd> {
  /**
   * Make an unfitted model.
   *
   * @param hyper - The hyperparameters of {@link truncatedSvd}.
   */
  constructor(hyper: TruncatedSvdHyper = {}) {
    super(truncatedSvd(hyper))
  }
}
