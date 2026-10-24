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
import { ksvd, mod, onlineDictionaryLearning, sparseEncode } from '../../lib/optim/index.ts'
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
          algorithm: choice(['exact', 'randomized'], {
            label: 'algorithm',
            description: 'An exact SVD, or a randomised one (faster on large matrices, approximate).',
            default: h.algorithm ?? 'exact',
          }),
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

// ---------------------------------------------------------------------------------------------------------------
// Dictionary learning

/** Hyperparameters of dictionary learning. */
export type DictionaryLearningHyper = {
  readonly nComponents?: number
  /** K-SVD and MOD (with OMP coding at `sparsity` atoms), or online learning with lasso coding (default online). */
  readonly method?: 'ksvd' | 'mod' | 'online'
  readonly sparsity?: number
  /** The lasso weight of online learning and of `transform` (default 1). */
  readonly alpha?: number
  readonly maxIter?: number
}

/** The parameters of a fitted dictionary: atoms as rows and the reconstruction error after each iteration. */
export type DictionaryParams = {
  readonly components: number[][]
  readonly error: number
  readonly history: number[]
  readonly alpha: number
  readonly sparsity: number
  readonly method: 'ksvd' | 'mod' | 'online'
}

/** The verbs of a fitted dictionary: sparse codes of new rows and reconstructions. */
export interface DictionaryVerbs {
  readonly transform: (x: TensorLike) => Tensor
  readonly inverseTransform: (code: TensorLike) => Tensor
  readonly nFeaturesIn: number
}

/**
 * Dictionary learning and sparse coding (sklearn.decomposition.DictionaryLearning and MiniBatchDictionaryLearning)
 * over `optim`'s K-SVD, MOD and online learners and its `sparseEncode`. Its exposition is a `components` record with
 * the atoms and the training codes as scores.
 *
 * @param hyper - Atoms, the learner, OMP sparsity, lasso weight and iteration cap.
 * @returns The estimator.
 */
export function dictionaryLearning(
  hyper: DictionaryLearningHyper = {},
): Estimator<DictionaryLearningHyper, ExplainedFitted<DictionaryParams, DictionaryVerbs>> {
  const encode = (p: DictionaryParams, x: TensorLike): Tensor =>
    p.method === 'online'
      ? sparseEncode(x, p.components, { algorithm: 'lassoCd', alpha: p.alpha })
      : sparseEncode(x, p.components, { algorithm: 'omp', nNonzeroCoefs: p.sparsity })
  return defineExplainedEstimator<DictionaryLearningHyper, DictionaryParams, DictionaryVerbs>(
    {
      name: 'dictionary-learning',
      kind: 'transformer',
      hyperSpace: (h) =>
        dict({
          nComponents: int(1, 50, { default: h.nComponents ?? 4, label: 'atoms' }),
          method: choice(['online', 'ksvd', 'mod'], {
            label: 'algorithm',
            description: 'Online dictionary learning, K-SVD or the method of optimal directions.',
            default: h.method ?? 'online',
          }),
          sparsity: int(1, 20, { default: h.sparsity ?? 1, label: 'atoms per row' }),
        }),
      fit: (data, h, options) => {
        const m = matrixView(data.x, 'dictionaryLearning')
        const k = h.nComponents ?? 4
        const sparsity = Math.min(h.sparsity ?? 1, k)
        const method = h.method ?? 'online'
        const run = {
          key: fitRunOptions(options, 'dictionary-learning').key as Key,
          ...(h.maxIter === undefined ? {} : { maxIterations: h.maxIter }),
        }
        const r =
          method === 'ksvd'
            ? ksvd(data.x, k, sparsity, run)
            : method === 'mod'
              ? mod(data.x, k, sparsity, run)
              : onlineDictionaryLearning(data.x, k, { ...run, alpha: h.alpha ?? 1 })
        const components = toArray(r.dictionary) as number[][]
        return {
          params: { components, error: r.error, history: [...r.history], alpha: h.alpha ?? 1, sparsity, method },
          exposition: [
            componentsRecord({
              method: `dictionary-${method}`,
              features: featureNamesOf(data, m.d),
              components,
              scores: toArray(r.code) as number[][],
            }),
          ],
        }
      },
      methods: (p) => ({
        transform: (x) => {
          matrixView(x, 'dictionaryLearning', { nFeatures: p.components[0]?.length ?? 0 })
          return encode(p, x)
        },
        inverseTransform: (code) =>
          reconstruct(code, new Array<number>(p.components[0]?.length ?? 0).fill(0), p.components),
        nFeaturesIn: p.components[0]?.length ?? 0,
      }),
    },
    hyper,
  )
}
