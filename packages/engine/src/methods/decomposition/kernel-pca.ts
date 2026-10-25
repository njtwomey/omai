/**
 * Kernel PCA (as in scikit-learn's `KernelPCA`): principal components in a kernel's feature space, computed from the
 * centred Gram matrix, so nonlinear structure becomes linear directions. Wraps the core's kernel PCA, with pre-images
 * by kernel ridge regression for mapping back to the input space.
 */

import { MdmlError } from '../../lib/base/index.ts'
import { componentsRecord } from '../../lib/exposition/index.ts'
import { kernelPca as coreKernelPca, type PairwiseKernelMetric, pairwiseKernels } from '../../lib/kernels/index.ts'
import {
  defineExplainedEstimator,
  type Estimator,
  EstimatorClass,
  type ExplainedFitted,
  matrixView,
} from '../../lib/learn/index.ts'
import { linalg } from '../../lib/numerics/index.ts'
import { choice, dict, int, real } from '../../lib/space/index.ts'
import { fromStorage, type Tensor, type TensorLike, toArray, toFlat } from '../../lib/tensor/index.ts'
import { mm } from './_linear.ts'

/** Hyperparameters of kernel PCA. */
export type KernelPcaHyper = {
  readonly nComponents?: number
  /** The kernel, by scikit-learn's pairwise names (default rbf). */
  readonly kernel?: PairwiseKernelMetric
  /** gamma (default 1 / n_features), degree and coef0 of the kernel. */
  readonly gamma?: number
  readonly degree?: number
  readonly coef0?: number
  /** Learn pre-images by kernel ridge regression from the projections back to the data (default false). */
  readonly fitInverseTransform?: boolean
  /** The ridge of the pre-image map (default 1). */
  readonly alpha?: number
}

/** The parameters of a fitted kernel PCA. */
export type KernelPcaParams = {
  readonly data: number[][]
  readonly eigenvalues: number[]
  readonly projections: number[][]
  readonly gamma: number
  /** Kernel ridge coefficients of the pre-image map, when fitted. */
  readonly dualCoef: number[][] | null
}

/** The verbs of kernel PCA. */
export interface KernelPcaVerbs {
  readonly transform: (x: TensorLike) => Tensor
  readonly inverseTransform?: (z: TensorLike) => Tensor
  readonly nFeaturesIn: number
}

/** A fitted kernel PCA. */
export type FittedKernelPca = ExplainedFitted<KernelPcaParams, KernelPcaVerbs>

/**
 * Kernel PCA (sklearn.decomposition.KernelPCA) over the core's `kernelPca` on a Gram matrix from `kernels`'
 * pairwise kernels, with pre-images by kernel ridge regression. Its exposition is a `components` record whose
 * components are the training rows' coefficients (dual axes) and whose scores are the training projections.
 *
 * @param hyper - Components, kernel and its parameters, and the pre-image map.
 * @returns The estimator.
 */
export function kernelPca(hyper: KernelPcaHyper = {}): Estimator<KernelPcaHyper, FittedKernelPca> {
  const kernelOf = (h: KernelPcaHyper, gamma: number) => (a: TensorLike, b?: TensorLike) =>
    pairwiseKernels(a, b, h.kernel ?? 'rbf', {
      gamma,
      ...(h.degree === undefined ? {} : { degree: h.degree }),
      ...(h.coef0 === undefined ? {} : { coef0: h.coef0 }),
    })
  return defineExplainedEstimator<KernelPcaHyper, KernelPcaParams, KernelPcaVerbs>(
    {
      name: 'kernel-pca',
      kind: 'transformer',
      hyperSpace: (h) =>
        dict({
          nComponents: int(1, 50, { default: h.nComponents ?? 2, label: 'components' }),
          kernel: choice(['rbf', 'poly', 'sigmoid', 'cosine', 'linear'], {
            label: 'kernel',
            description: 'The kernel whose feature space the components live in.',
            default: h.kernel ?? 'rbf',
          }),
          gamma: real(1e-3, 100, {
            label: 'kernel γ',
            description: 'The kernel width: larger γ makes similarity more local.',
            log: true,
            default: h.gamma ?? 1,
          }),
        }),
      fit: (data, h) => {
        const m = matrixView(data.x, 'kernelPca', { minSamples: 2 })
        const k = h.nComponents ?? m.n
        if (!(Number.isInteger(k) && k >= 1 && k <= m.n))
          throw new MdmlError('value', `kernelPca: nComponents must be in 1..${m.n}`)
        const gamma = h.gamma ?? 1 / m.d
        const kernel = kernelOf(h, gamma)
        const rows = Array.from({ length: m.n }, (_, i) => Array.from(m.data.subarray(i * m.d, (i + 1) * m.d)))
        const fit = coreKernelPca(kernel(rows), { nComponents: k })
        const projections = toArray(fit.projections) as number[][]
        let dualCoef: number[][] | null = null
        if (h.fitInverseTransform === true) {
          const kz = toArray(kernel(projections)) as number[][]
          const ridge = kz.map((r, i) => r.map((v, j) => (i === j ? v + (h.alpha ?? 1) : v)))
          dualCoef = toArray(linalg.solve(ridge, rows)) as number[][]
        }
        const eigenvectors = toArray(fit.eigenvectors) as number[][]
        return {
          params: { data: rows, eigenvalues: toFlat(fit.eigenvalues), projections, gamma, dualCoef },
          exposition: [
            componentsRecord({
              method: 'kernelPca',
              features: rows.map((_, i) => `row${i}`),
              space: 'samples',
              components: Array.from({ length: k }, (_, c) => eigenvectors.map((r) => r[c] ?? 0)),
              explainedVariance: toFlat(fit.eigenvalues).map((v) => v / m.n),
              scores: projections,
            }),
          ],
        }
      },
      methods: (p, h) => {
        const kernel = kernelOf(h, p.gamma)
        const fit = coreKernelPca(kernel(p.data), { nComponents: p.eigenvalues.length })
        const d = p.data[0]?.length ?? 0
        const dual = p.dualCoef
        return {
          transform: (x) => {
            matrixView(x, 'transform', { nFeatures: d })
            return fit.transform(kernel(x, p.data)) as Tensor
          },
          ...(dual === null
            ? {}
            : {
                inverseTransform: (z: TensorLike) => {
                  const kz = toArray(kernel(z, p.projections)) as number[][]
                  return rowsToTensor(mm(kz, dual))
                },
              }),
          nFeaturesIn: d,
        }
      },
    },
    hyper,
  )
}

const rowsToTensor = (rows: number[][]): Tensor =>
  fromStorage(Float64Array.from(rows.flat()), [rows.length, rows[0]?.length ?? 0], 'float64')

/** Kernel PCA as a class. */
export class KernelPCA extends EstimatorClass<KernelPcaHyper, FittedKernelPca> {
  /**
   * Make an unfitted model.
   *
   * @param hyper - The hyperparameters of {@link kernelPca}.
   */
  constructor(hyper: KernelPcaHyper = {}) {
    super(kernelPca(hyper))
  }
}
