import { argminOf, linspaceNumbers } from '../../lib/base/index.ts'
import { type Distribution, normal } from '../../lib/dists/index.ts'
import { crossValidationCurve, type Exposition, posteriorGrid } from '../../lib/exposition/index.ts'
import { type PairwiseKernelMetric, pairwiseKernels } from '../../lib/kernels/index.ts'
import {
  decisionSurfaceOf,
  defineExplainedEstimator,
  type ExplainedEstimator,
  ExplainedEstimatorClass,
  type Fitted,
  featureNamesOf,
  type LearnData,
  matrixView,
  targetVector,
} from '../../lib/learn/index.ts'
import { eigh } from '../../lib/numerics/linalg/index.ts'
import { choice, dict, real, type Space } from '../../lib/space/index.ts'
import { fromStorage, primalOf, type Tensor, type TensorLike, toFlat } from '../../lib/tensor/index.ts'

// TODO: kernel logistic regression and kernel Fisher discriminants (kernel-classifiers.ts), random features,
// kernel mean embeddings.

/** Hyperparameters of {@link kernelRidge}. */
export interface KernelRidgeHyper {
  /** The ridge penalty (scikit-learn's alpha, default 1). */
  readonly alpha: number
  /** A kernel by scikit-learn's pairwise names (default linear). */
  readonly kernel: Extract<PairwiseKernelMetric, 'linear' | 'rbf' | 'laplacian' | 'polynomial' | 'sigmoid' | 'cosine'>
  /** gamma (default 1 / d). */
  readonly gamma?: number
  readonly degree: number
  readonly coef0: number
  /** Points of the leave-one-out curve over alpha (default 25; 0 for none). */
  readonly curveLength: number
}

/** Parameters of a fitted kernel ridge regression. */
export type KernelRidgeParams = {
  readonly x: number[][]
  readonly dualCoef: number[]
  /** The root mean leave-one-out squared error (the scale of `predictDistribution`). */
  readonly scale: number
}

/** Verbs of a fitted kernel ridge regression. */
export interface KernelRidgeVerbs {
  readonly predict: (x: TensorLike) => Tensor
  /** Normal with the prediction as mean and the leave-one-out root mean squared error as scale. */
  readonly predictDistribution: (x: TensorLike) => Distribution
  readonly nFeaturesIn: number
}

/** A fitted kernel ridge regression. */
export type FittedKernelRidge = Fitted<KernelRidgeParams> & KernelRidgeVerbs

/**
 * Kernel ridge regression (sklearn.kernel_ridge.KernelRidge): dual coefficients (K + alpha I)^{-1} y, predictions
 * k(x, X) times them, no intercept. It equals the posterior mean of a GP with that kernel and noise variance alpha.
 * Exposition: the exact leave-one-out error over a log grid of alphas around the chosen one (from one eigendecomposition
 * of K, residuals e_i / (1 - H_ii)), and the fitted function on a grid (one feature) or the prediction surface (two).
 *
 * @param hyper - alpha (default 1), kernel (default linear) with gamma, degree and coef0, and the curve length.
 * @returns The estimator.
 */
export function kernelRidge(
  hyper: Partial<KernelRidgeHyper> = {},
): ExplainedEstimator<KernelRidgeHyper, FittedKernelRidge> {
  const options = (h: KernelRidgeHyper) => ({
    ...(h.gamma === undefined ? {} : { gamma: h.gamma }),
    degree: h.degree,
    coef0: h.coef0,
  })
  const est: ExplainedEstimator<KernelRidgeHyper, FittedKernelRidge> = defineExplainedEstimator(
    {
      name: 'kernel-ridge',
      kind: 'regressor',
      hyperSpace: (h: KernelRidgeHyper): Space =>
        dict({
          alpha: real(1e-4, 1e2, { log: true, default: h.alpha, label: 'penalty' }),
          kernel: choice(['linear', 'rbf', 'laplacian', 'polynomial', 'sigmoid'], {
            label: 'kernel',
            description: 'The kernel whose feature space the ridge fit lives in.',
            default: h.kernel,
          }),
          gamma: real(1e-3, 10, {
            label: 'kernel γ',
            description: 'The kernel width: larger γ makes similarity more local.',
            log: true,
            default: h.gamma ?? 1,
          }),
        }),
      fit: (data: LearnData, h: KernelRidgeHyper) => {
        const m = matrixView(data.x, 'kernelRidge')
        const y = targetVector(data.y ?? [], 'kernelRidge', m.n)
        const x = Array.from({ length: m.n }, (_, i) => Array.from(m.data.subarray(i * m.d, (i + 1) * m.d)))
        const K = toFlat(primalOf(pairwiseKernels(x, undefined, h.kernel, options(h))))
        const n = m.n
        // K = V diag(w) V^T once: dual coefficients and leave-one-out residuals for any alpha.
        const e = eigh(fromStorage(Float64Array.from(K), [n, n], 'float64'))
        const w = toFlat(primalOf(e.eigenvalues))
        const v = toFlat(primalOf(e.eigenvectors))
        const vty = Array.from({ length: n }, (_, a) => {
          let s = 0
          for (let i = 0; i < n; i++) s += (v[i * n + a] ?? 0) * (y[i] ?? 0)
          return s
        })
        const dualFor = (alpha: number): number[] =>
          Array.from({ length: n }, (_, i) => {
            let s = 0
            for (let a = 0; a < n; a++) s += ((v[i * n + a] ?? 0) * (vty[a] ?? 0)) / ((w[a] ?? 0) + alpha)
            return s
          })
        const looFor = (alpha: number): number[] => {
          const c = dualFor(alpha)
          // The residual of the ridge smoother is alpha c_i, and 1 - H_ii = alpha [(K + alpha I)^{-1}]_ii.
          return Array.from({ length: n }, (_, i) => {
            let g = 0
            for (let a = 0; a < n; a++) g += (v[i * n + a] ?? 0) ** 2 / ((w[a] ?? 0) + alpha)
            return ((c[i] ?? 0) / g) ** 2
          })
        }
        const loo = looFor(h.alpha)
        const params: KernelRidgeParams = {
          x,
          dualCoef: dualFor(h.alpha),
          scale: Math.sqrt(loo.reduce((a, b) => a + b, 0) / n) || 1e-12,
        }
        const fitted = est.revive(params)
        const exposition: Exposition[] = []
        if (h.curveLength > 0) {
          const alphas = linspaceNumbers(Math.log10(h.alpha) - 3, Math.log10(h.alpha) + 3, h.curveLength).map(
            (t) => 10 ** t,
          )
          const errors = alphas.map(looFor)
          const means = errors.map((es) => es.reduce((a, b) => a + b, 0) / n)
          const ses = errors.map((es, i) =>
            Math.sqrt(es.reduce((a, b) => a + (b - (means[i] ?? 0)) ** 2, 0) / (n - 1) / n),
          )
          exposition.push(
            crossValidationCurve({
              parameter: 'alpha',
              values: alphas,
              meanScores: means,
              standardErrors: ses,
              higherIsBetter: false,
              best: alphas[argminOf(means)] ?? h.alpha,
              scoreName: 'leave-one-out squared error',
              penalty: true,
            }),
          )
        }
        const features = featureNamesOf(data, m.d)
        if (m.d === 1) {
          const xs = x.map((r) => r[0] ?? 0)
          const lo = Math.min(...xs)
          const hi = Math.max(...xs)
          const pad = 0.1 * (hi - lo || 1)
          const grid = linspaceNumbers(lo - pad, hi + pad, 60).map((t) => [t])
          exposition.push(
            posteriorGrid({
              inputs: grid,
              mean: toFlat(fitted.predict(grid)),
              variance: grid.map(() => 0),
              observed: { inputs: x, targets: Array.from(y) },
            }),
          )
        }
        if (m.d === 2) exposition.push(decisionSurfaceOf(fitted, data.x, { featureNames: features }))
        return { params, exposition }
      },
      methods: (p: KernelRidgeParams, h: KernelRidgeHyper): KernelRidgeVerbs => {
        const predict = (q: TensorLike): Tensor => {
          const d = p.x[0]?.length ?? 0
          const rows = matrixView(q, 'kernelRidge', { nFeatures: d }).n
          const k = toFlat(primalOf(pairwiseKernels(q, p.x, h.kernel, options(h))))
          const n = p.x.length
          return fromStorage(
            Float64Array.from({ length: rows }, (_, i) =>
              p.dualCoef.reduce((a, c, j) => a + c * (k[i * n + j] ?? 0), 0),
            ),
            [rows],
            'float64',
          )
        }
        return {
          predict,
          predictDistribution: (q) => {
            const mean = predict(q)
            const m = mean.data.length
            return normal(mean, fromStorage(new Float64Array(m).fill(p.scale), [m], 'float64'))
          },
          nFeaturesIn: p.x[0]?.length ?? 0,
        }
      },
    },
    {
      alpha: hyper.alpha ?? 1,
      kernel: hyper.kernel ?? 'linear',
      ...(hyper.gamma === undefined ? {} : { gamma: hyper.gamma }),
      degree: hyper.degree ?? 3,
      coef0: hyper.coef0 ?? 1,
      curveLength: hyper.curveLength ?? 25,
    },
  )
  return est
}

/** {@link kernelRidge} as a class. */
export class KernelRidge extends ExplainedEstimatorClass<KernelRidgeHyper, FittedKernelRidge> {
  /** @param hyper - As {@link kernelRidge}. */
  constructor(hyper: Partial<KernelRidgeHyper> = {}) {
    super(kernelRidge(hyper))
  }
}
