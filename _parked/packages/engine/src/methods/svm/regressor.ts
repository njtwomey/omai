/**
 * Epsilon-support vector regression (as in scikit-learn's `SVR`, libsvm's epsilon-SVR): a kernel regression that
 * ignores errors smaller than epsilon and penalises larger ones linearly, so only rows on or outside the epsilon tube
 * become support vectors. The dual is solved by the core's traced SMO.
 */

import { type Distribution, normal } from '../../lib/dists/index.ts'
import { type Exposition, marginRecord } from '../../lib/exposition/index.ts'
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
  sampleWeightsOf,
  targetVector,
} from '../../lib/learn/index.ts'
import { choice, dict, real, type Space } from '../../lib/space/index.ts'
import { fromStorage, type Tensor, type TensorLike } from '../../lib/tensor/index.ts'
import { type KernelHyper, type KernelParams, kernelMatrix, resolveKernel, runDual } from './_dual.ts'

// TODO: nu-SVR.

/** Hyperparameters of {@link svr}. */
export interface SvrHyper extends KernelHyper {
  readonly C: number
  /** Half-width of the insensitive tube (default 0.1). */
  readonly epsilon: number
  readonly tol: number
  readonly maxIter: number
}

/** Parameters of a fitted SVR. */
export type SvrParams = {
  readonly kernel: KernelParams
  readonly vectors: number[][]
  /** alpha_i - alpha*_i of each support vector. */
  readonly coef: number[]
  readonly intercept: number
  readonly nFeatures: number
  /** The standard deviation of the training residuals (the scale of `predictDistribution`). */
  readonly residualScale: number
}

/** Verbs of a fitted SVR. */
export interface SvrVerbs {
  readonly predict: (x: TensorLike) => Tensor
  /** Normal with the prediction as mean and the training residuals' standard deviation. */
  readonly predictDistribution: (x: TensorLike) => Distribution
  readonly nFeaturesIn: number
}

/** A fitted SVR. */
export type FittedSvr = Fitted<SvrParams> & SvrVerbs

/**
 * Epsilon-support vector regression (sklearn.svm.SVR, libsvm's epsilon-SVR): the dual over (alpha, alpha*) with
 * Q = [K, -K; -K, K] solved by the core's traced SMO; prediction sum_i (alpha_i - alpha*_i) K(x_i, x) + b. Exposition:
 * the margin record (support vectors, their coefficients, the intercept, each row's residual y - f(x) as its margin,
 * the dual objective and working pairs per step) and, for two features, the prediction surface.
 *
 * @param hyper - C (default 1), epsilon (0.1), kernel with gamma ('scale'), degree and coef0, tolerance, iterations.
 * @returns The estimator.
 */
export function svr(hyper: Partial<SvrHyper> = {}): ExplainedEstimator<SvrHyper, FittedSvr> {
  const est: ExplainedEstimator<SvrHyper, FittedSvr> = defineExplainedEstimator(
    {
      name: 'svr',
      kind: 'regressor',
      hyperSpace: (h: SvrHyper): Space =>
        dict({
          C: real(1e-2, 1e3, {
            label: 'C (inverse penalty)',
            description: 'The inverse of the regularisation strength: larger C fits the training data more closely.',
            log: true,
            default: h.C,
          }),
          epsilon: real(0, 1, { default: h.epsilon, label: 'tube half-width' }),
          kernel: choice(['rbf', 'linear', 'poly', 'sigmoid'], {
            label: 'kernel',
            description: 'The kernel measuring similarity between rows.',
            default: h.kernel,
          }),
        }),
      fit: (data: LearnData, h: SvrHyper, options: FitOptions) => {
        const m = matrixView(data.x, 'svr')
        const t = targetVector(data.y ?? [], 'svr', m.n)
        const w = sampleWeightsOf(data.weights, m.n, 'svr')
        const kp = resolveKernel(h, m.data, m.d)
        const rows = Array.from({ length: m.n }, (_, i) => Array.from(m.data.subarray(i * m.d, (i + 1) * m.d)))
        const K = kernelMatrix(kp, rows)
        const n = m.n
        const y = Array.from({ length: 2 * n }, (_, i) => (i < n ? 1 : -1))
        const Q = Array.from({ length: 2 * n }, (_, i) =>
          Array.from({ length: 2 * n }, (_, j) => (y[i] ?? 1) * (y[j] ?? 1) * (K[(i % n) * n + (j % n)] ?? 0)),
        )
        const p = Array.from({ length: 2 * n }, (_, i) =>
          i < n ? h.epsilon - (t[i] ?? 0) : h.epsilon + (t[i - n] ?? 0),
        )
        const C = Array.from({ length: 2 * n }, (_, i) => h.C * (w[i % n] ?? 1))
        const dual = runDual({ Q, p, y, C }, options, { tol: h.tol, maxIter: h.maxIter })
        const beta = Array.from({ length: n }, (_, i) => (dual.alpha[i] ?? 0) - (dual.alpha[i + n] ?? 0))
        const support = beta.flatMap((b, i) => (b !== 0 ? [i] : []))
        const intercept = -dual.rho
        const predictRow = (r: number): number =>
          support.reduce((a, i) => a + (beta[i] ?? 0) * (K[r * n + i] ?? 0), intercept)
        const residuals = Array.from({ length: n }, (_, r) => (t[r] ?? 0) - predictRow(r))
        const params: SvrParams = {
          kernel: kp,
          vectors: support.map((i) => rows[i] as number[]),
          coef: support.map((i) => beta[i] ?? 0),
          intercept,
          nFeatures: m.d,
          residualScale: Math.sqrt(residuals.reduce((a, r) => a + r * r, 0) / n) || 1e-12,
        }
        const exposition: Exposition[] = [
          marginRecord({
            support,
            dualCoefficients: params.coef,
            intercept,
            margins: residuals,
            dualObjective: dual.objective,
            workingSets: dual.workingSets.map(([i, j]) => [i % n, j % n]),
          }),
        ]
        if (m.d === 2)
          exposition.push(decisionSurfaceOf(est.revive(params), data.x, { featureNames: featureNamesOf(data, m.d) }))
        return { params, exposition }
      },
      methods: (p: SvrParams): SvrVerbs => {
        const predict = (x: TensorLike): Tensor => {
          const k = kernelMatrix(p.kernel, x, p.vectors)
          const nv = p.vectors.length
          const rowsCount = nv === 0 ? matrixView(x, 'svr', { nFeatures: p.nFeatures }).n : k.length / nv
          return fromStorage(
            Float64Array.from({ length: rowsCount }, (_, i) =>
              p.coef.reduce((a, c, s) => a + c * (k[i * nv + s] ?? 0), p.intercept),
            ),
            [rowsCount],
            'float64',
          )
        }
        return {
          predict,
          predictDistribution: (x) => {
            const mean = predict(x)
            return normal(
              mean,
              fromStorage(new Float64Array(mean.data.length).fill(p.residualScale), [mean.data.length], 'float64'),
            )
          },
          nFeaturesIn: p.nFeatures,
        }
      },
    },
    {
      C: hyper.C ?? 1,
      epsilon: hyper.epsilon ?? 0.1,
      kernel: hyper.kernel ?? 'rbf',
      gamma: hyper.gamma ?? 'scale',
      degree: hyper.degree ?? 3,
      coef0: hyper.coef0 ?? 0,
      tol: hyper.tol ?? 1e-3,
      maxIter: hyper.maxIter ?? 100000,
    },
  )
  return est
}

/** {@link svr} as a class. */
export class SVR extends ExplainedEstimatorClass<SvrHyper, FittedSvr> {
  /** @param hyper - As {@link svr}. */
  constructor(hyper: Partial<SvrHyper> = {}) {
    super(svr(hyper))
  }
}
