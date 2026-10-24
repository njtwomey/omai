import { WhiteboxError } from '../../lib/base/index.ts'
import { anomalyScores, type Exposition, marginRecord } from '../../lib/exposition/index.ts'
import {
  decisionSurfaceOf,
  defineExplainedEstimator,
  type ExplainedEstimator,
  ExplainedEstimatorClass,
  type FitOptions,
  type Fitted,
  featureNamesOf,
  type LearnData,
  matrixView,
} from '../../lib/learn/index.ts'
import { choice, dict, real, type Space } from '../../lib/space/index.ts'
import { fromStorage, type Tensor, type TensorLike } from '../../lib/tensor/index.ts'
import { type KernelHyper, type KernelParams, kernelMatrix, resolveKernel, runDual } from './_dual.ts'

/** Hyperparameters of {@link oneClassSvm}. */
export interface OneClassSvmHyper extends KernelHyper {
  /** An upper bound on the fraction of outliers and a lower bound on the fraction of support vectors (default 0.5). */
  readonly nu: number
  readonly tol: number
  readonly maxIter: number
}

/** Parameters of a fitted one-class model: support vectors, their coefficients and the offset. */
export type OneClassParams = {
  readonly kernel: KernelParams
  readonly vectors: number[][]
  readonly coef: number[]
  /** The decision function is sum_i coef_i K(x_i, x) + offset (SVDD: R^2 - |phi(x) - a|^2). */
  readonly offset: number
  /** SVDD only: the squared norm of the centre, alpha^T K alpha. */
  readonly centreNorm: number
  readonly method: 'one-class-svm' | 'svdd'
  readonly nFeatures: number
}

/** Verbs of a fitted one-class model. */
export interface OneClassVerbs {
  /** +1 for inliers, -1 for outliers (scikit-learn's convention). */
  readonly predict: (x: TensorLike) => Tensor
  /** Signed distance to the boundary: positive inside. */
  readonly decisionFunction: (x: TensorLike) => Tensor
  readonly nFeaturesIn: number
}

/** A fitted one-class model. */
export type FittedOneClass = Fitted<OneClassParams> & OneClassVerbs

function oneClassVerbs(p: OneClassParams): OneClassVerbs {
  const decision = (x: TensorLike): Float64Array => {
    const m = matrixView(x, p.method, { nFeatures: p.nFeatures })
    const k = kernelMatrix(p.kernel, x, p.vectors)
    const nv = p.vectors.length
    const self = p.method === 'svdd' ? kernelDiagonal(p.kernel, x, m.n) : undefined
    return Float64Array.from({ length: m.n }, (_, i) => {
      let s = 0
      for (let v = 0; v < nv; v++) s += (p.coef[v] ?? 0) * (k[i * nv + v] ?? 0)
      // SVDD: R^2 - (k(x, x) - 2 sum alpha_i k(x_i, x) + alpha^T K alpha).
      return p.method === 'svdd' ? p.offset - ((self?.[i] ?? 0) - 2 * s + p.centreNorm) : s + p.offset
    })
  }
  return {
    predict: (x) => {
      const d = decision(x)
      return fromStorage(
        Float64Array.from(d, (v) => (v >= 0 ? 1 : -1)),
        [d.length],
        'float64',
      )
    },
    decisionFunction: (x) => {
      const d = decision(x)
      return fromStorage(d, [d.length], 'float64')
    },
    nFeaturesIn: p.nFeatures,
  }
}

function kernelDiagonal(k: KernelParams, x: TensorLike, n: number): Float64Array {
  const full = kernelMatrix(k, x)
  return Float64Array.from({ length: n }, (_, i) => full[i * n + i] ?? 0)
}

function oneClassEstimator(
  method: 'one-class-svm' | 'svdd',
  hyper: OneClassSvmHyper,
): ExplainedEstimator<OneClassSvmHyper, FittedOneClass> {
  const est: ExplainedEstimator<OneClassSvmHyper, FittedOneClass> = defineExplainedEstimator(
    {
      name: method,
      kind: 'outlier',
      hyperSpace: (h: OneClassSvmHyper): Space =>
        dict({
          nu: real(0.01, 1, { default: h.nu, label: method === 'svdd' ? 'outlier fraction (1 / (n C))' : 'nu' }),
          kernel: choice(['rbf', 'linear', 'poly', 'sigmoid'], {
            label: 'kernel',
            description: 'The kernel measuring similarity between rows.',
            default: h.kernel,
          }),
        }),
      fit: (data: LearnData, h: OneClassSvmHyper, options: FitOptions) => {
        const m = matrixView(data.x, method)
        const n = m.n
        if (!(h.nu > 0 && h.nu <= 1)) throw new WhiteboxError('value', `${method}: nu must be in (0, 1]`)
        const kp = resolveKernel(h, m.data, m.d)
        const rows = Array.from({ length: n }, (_, i) => Array.from(m.data.subarray(i * m.d, (i + 1) * m.d)))
        const K = kernelMatrix(kp, rows)
        const y = new Float64Array(n).fill(1)
        let dual: ReturnType<typeof runDual>
        if (method === 'one-class-svm') {
          // libsvm's one-class dual: min 1/2 a^T K a, 0 <= a <= 1, sum a = nu n, started from the first nu n at 1.
          const total = h.nu * n
          const alpha0 = Float64Array.from({ length: n }, (_, i) =>
            i < Math.floor(total) ? 1 : i === Math.floor(total) ? total - Math.floor(total) : 0,
          )
          const Q = Array.from({ length: n }, (_, i) => Array.from({ length: n }, (_, j) => K[i * n + j] ?? 0))
          dual = runDual({ Q, p: new Float64Array(n), y, C: 1, alpha0 }, options, { tol: h.tol, maxIter: h.maxIter })
        } else {
          // SVDD (Tax and Duin): min a^T K a - sum a_i K_ii, 0 <= a <= C, sum a = 1, with C = 1 / (nu n).
          const C = 1 / (h.nu * n)
          const Q = Array.from({ length: n }, (_, i) => Array.from({ length: n }, (_, j) => 2 * (K[i * n + j] ?? 0)))
          const p = Float64Array.from({ length: n }, (_, i) => -(K[i * n + i] ?? 0))
          dual = runDual({ Q, p, y, C, alpha0: new Float64Array(n).fill(1 / n) }, options, {
            tol: h.tol,
            maxIter: h.maxIter,
          })
        }
        const alpha = dual.alpha
        const support = Array.from(alpha.keys()).filter((i) => (alpha[i] ?? 0) > 0)
        let offset: number
        let centreNorm = 0
        let coef: number[]
        if (method === 'one-class-svm') {
          offset = -dual.rho
          coef = support.map((i) => alpha[i] ?? 0)
        } else {
          for (const i of support)
            for (const j of support) centreNorm += (alpha[i] ?? 0) * (alpha[j] ?? 0) * (K[i * n + j] ?? 0)
          // R^2 from the free support vectors (0 < a < C), the distance of any of them to the centre.
          const C = 1 / (h.nu * n)
          const free = support.filter((i) => (alpha[i] ?? 0) < C - 1e-12)
          const ref = free.length > 0 ? free : support
          const dist = (i: number): number => {
            let s = 0
            for (const j of support) s += (alpha[j] ?? 0) * (K[i * n + j] ?? 0)
            return (K[i * n + i] ?? 0) - 2 * s + centreNorm
          }
          offset = ref.reduce((a, i) => a + dist(i), 0) / Math.max(ref.length, 1)
          coef = support.map((i) => alpha[i] ?? 0)
        }
        const params: OneClassParams = {
          kernel: kp,
          vectors: support.map((i) => rows[i] as number[]),
          coef,
          offset,
          centreNorm,
          method,
          nFeatures: m.d,
        }
        const fitted = est.revive(params)
        const decision = Array.from(fitted.decisionFunction(data.x).data)
        const exposition: Exposition[] = [
          marginRecord({
            support,
            dualCoefficients: coef,
            intercept: offset,
            margins: decision,
            dualObjective: dual.objective,
            workingSets: dual.workingSets,
          }),
          anomalyScores({ method, scores: decision.map((v) => -v), threshold: 0 }),
        ]
        if (m.d === 2) exposition.push(decisionSurfaceOf(fitted, data.x, { featureNames: featureNamesOf(data, m.d) }))
        return { params, exposition }
      },
      methods: (p: OneClassParams) => oneClassVerbs(p),
    },
    hyper,
  )
  return est
}

const defaults = (hyper: Partial<OneClassSvmHyper>): OneClassSvmHyper => ({
  nu: hyper.nu ?? 0.5,
  kernel: hyper.kernel ?? 'rbf',
  gamma: hyper.gamma ?? 'scale',
  degree: hyper.degree ?? 3,
  coef0: hyper.coef0 ?? 0,
  tol: hyper.tol ?? 1e-3,
  maxIter: hyper.maxIter ?? 100000,
})

/**
 * The one-class SVM (sklearn.svm.OneClassSVM, libsvm's nu formulation): a kernel boundary around the data with at
 * most a fraction nu of the rows outside, by the core's traced SMO. Exposition: the margin record (support vectors,
 * coefficients, offset, each row's decision value, dual objective and working pairs), anomaly scores (minus the
 * decision value, threshold 0) and, for two features, the decision surface.
 *
 * @param hyper - nu (default 0.5), kernel with gamma ('scale'), degree and coef0, tolerance and iterations.
 * @returns The estimator.
 */
export function oneClassSvm(
  hyper: Partial<OneClassSvmHyper> = {},
): ExplainedEstimator<OneClassSvmHyper, FittedOneClass> {
  return oneClassEstimator('one-class-svm', defaults(hyper))
}

/**
 * Support vector data description (Tax and Duin 2004): the smallest kernel-space ball holding the data, with C =
 * 1 / (nu n) letting a fraction nu of rows fall outside; the decision is R^2 minus the squared distance to the centre.
 * Exposition as {@link oneClassSvm}.
 *
 * @param hyper - As {@link oneClassSvm} (nu sets C = 1 / (nu n)).
 * @returns The estimator.
 */
export function svdd(hyper: Partial<OneClassSvmHyper> = {}): ExplainedEstimator<OneClassSvmHyper, FittedOneClass> {
  return oneClassEstimator('svdd', defaults({ nu: 0.1, ...hyper }))
}

/** {@link oneClassSvm} as a class. */
export class OneClassSVM extends ExplainedEstimatorClass<OneClassSvmHyper, FittedOneClass> {
  /** @param hyper - As {@link oneClassSvm}. */
  constructor(hyper: Partial<OneClassSvmHyper> = {}) {
    super(oneClassSvm(hyper))
  }
}
