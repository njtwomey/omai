/**
 * Kernel ridge regression (Tikhonov-regularised least squares in a reproducing kernel Hilbert space) as scikit-learn's
 * `KernelRidge`: the closed-form dual solve `(K + alpha I)^(-1) y` over a kernel chosen by name or any `Kernel` of the
 * library, the exact leave-one-out shortcuts that follow from the same inverse (`hatDiagonal`, `looPredictions`,
 * `looPress`) without refitting, and the `kernelRidge` estimator on the iterative base.
 */

import { MdmlError } from '../base/index.ts'
import { type Distribution, normal } from '../dists/index.ts'
import {
  type Kernel,
  type KernelChoice,
  kernelMatrix,
  kernelSpace,
  type PairwiseKernelMetric,
  type ResolvedKernel,
  resolveKernel,
} from '../kernels/index.ts'
import { type IterativeEstimator, iterative, type LearnData, matrixView } from '../learn/index.ts'
import { r2Score } from '../metrics/index.ts'
import { inv } from '../numerics/linalg/index.ts'
import { dict, real } from '../space/index.ts'
import { asTensor, fromStorage, primalOf, type Tensor, type TensorLike, toFlat } from '../tensor/index.ts'
import { type Algorithm, channels, stateFromJson, stateToJson } from '../trace/index.ts'

/** The hyperparameters of {@link kernelRidge}. */
export interface KernelRidgeHyper {
  /** Regularisation strength `alpha > 0` in `(K + alpha I)^(-1) y` (default 1). */
  readonly alpha: number
  /** A metric name of `pairwiseKernels` or a `Kernel` (default `'linear'`, scikit-learn's). */
  readonly kernel: PairwiseKernelMetric | Kernel
  /** The kernel scale: a number, `'scale'` or `'auto'` (1 / d; default). */
  readonly gamma: number | 'scale' | 'auto'
  /** Degree of the polynomial kernel (default 3). */
  readonly degree: number
  /** Offset of the polynomial and sigmoid kernels (default 1). */
  readonly coef0: number
}

/** The state of {@link kernelRidge}: the dual solve is one step, so the state is the solution once it has run. */
export interface KernelRidgeState {
  readonly solved: boolean
  /** Dual weights, row i holding the `k` outputs of training row i. */
  readonly dual: number[][]
  /** The diagonal of `(K + alpha I)^(-1)`. */
  readonly inverseDiagonal: number[]
  /** The training residual sum of squares. */
  readonly rss: number
}

/** The fitted parameters of {@link kernelRidge}: plain JSON. */
export type KernelRidgeParams = {
  /** The resolved kernel (`custom` when a `Kernel` object was given; the verbs then use the hyperparameter's). */
  readonly kernel: ResolvedKernel
  readonly alpha: number
  /** The training inputs `[n, d]`. */
  readonly xFit: number[][]
  /** The dual weights `[n, k]`, `k` the number of outputs. */
  readonly dualCoef: number[][]
  /** Whether the targets were a matrix (k outputs) rather than a vector. */
  readonly multiOutput: boolean
  /** The effective degrees of freedom `tr(K (K + alpha I)^(-1))`. */
  readonly effectiveDf: number
  /** Exact leave-one-out sum of squared prediction errors. */
  readonly looPress: number
  /** The residual standard deviation `sqrt(RSS / (n - edf))` of the training fit, used by `predictDistribution`. */
  readonly residualStd: number
}

/** What a fitted kernel ridge regression offers. */
export interface KernelRidgeVerbs {
  /** Predictions `[m]` (or `[m, k]` for several outputs). */
  readonly predict: (x: TensorLike) => Tensor
  /** A normal predictive distribution around the prediction, with the training residual standard deviation. */
  readonly predictDistribution: (x: TensorLike) => Distribution
  /** The coefficient of determination (uniform average over outputs). */
  readonly score: (data: LearnData) => number
  /** The diagonal `[n]` of the smoother `K (K + alpha I)^(-1)` on the training rows. */
  readonly hatDiagonal: () => Tensor
  /** The exact leave-one-out predictions on the training rows, shaped like the targets. */
  readonly looPredictions: () => Tensor
  /** The exact leave-one-out sum of squared errors. */
  readonly looPress: () => number
  /** The dual weights, `[n]` or `[n, k]`. */
  readonly dualCoef: Tensor
  /** The training inputs `[n, d]`. */
  readonly xFit: Tensor
  readonly nFeaturesIn: number
}

const WHO = 'kernelRidge'

function validateAlpha(alpha: number, who: string): number {
  if (!Number.isFinite(alpha) || alpha <= 0) {
    throw new MdmlError('value', `${who}: alpha must be a positive finite number, got ${alpha}`)
  }
  return alpha
}

interface GramSystem {
  readonly n: number
  /** `(K + alpha I)^(-1)`, row-major. */
  readonly inverse: Float64Array
}

function gramSystem(k: TensorLike, alpha: number, who: string): GramSystem {
  const t = primalOf(asTensor(k))
  const n = t.shape[0] ?? 0
  if (t.shape.length !== 2 || t.shape[1] !== n) {
    throw new MdmlError('shape', `${who}: expected a square Gram matrix [n, n], got [${t.shape.join(', ')}]`)
  }
  const a = Float64Array.from(toFlat(t))
  for (let i = 0; i < n; i++) a[i * n + i] = (a[i * n + i] ?? 0) + alpha
  return { n, inverse: Float64Array.from(toFlat(primalOf(inv(fromStorage(a, [n, n], 'float64'))))) }
}

/** Read targets as `n` rows of `k` outputs. */
function targetRows(
  y: TensorLike,
  who: string,
): { readonly rows: Float64Array; readonly n: number; readonly k: number; readonly multi: boolean } {
  const t = primalOf(asTensor(y))
  if (t.shape.length !== 1 && t.shape.length !== 2) {
    throw new MdmlError('shape', `${who}: expected y to be 1-D or 2-D, got [${t.shape.join(', ')}]`)
  }
  const rows = Float64Array.from(toFlat(t))
  if (rows.some((v) => !Number.isFinite(v))) throw new MdmlError('value', `${who}: y has non-finite entries`)
  return { rows, n: t.shape[0] ?? 0, k: t.shape[1] ?? 1, multi: t.shape.length === 2 }
}

/** Multiply the `n x n` matrix by the `n x k` rows. */
function matmulRows(a: Float64Array, b: Float64Array, n: number, k: number): Float64Array {
  const out = new Float64Array(n * k)
  for (let i = 0; i < n; i++) {
    for (let l = 0; l < n; l++) {
      const v = a[i * n + l] ?? 0
      if (v === 0) continue
      for (let j = 0; j < k; j++) out[i * k + j] = (out[i * k + j] ?? 0) + v * (b[l * k + j] ?? 0)
    }
  }
  return out
}

/**
 * Solve the dual kernel ridge system `(K + alpha I) c = y`. The dual weights are the posterior mean weights of
 * Gaussian-process regression with noise variance `alpha`.
 *
 * @param k - Symmetric positive semi-definite Gram matrix `[n, n]`.
 * @param y - Targets `[n]` or `[n, m]`.
 * @param alpha - Regularisation `alpha > 0` (default 1).
 * @returns The dual weights, shaped like `y`.
 * @throws MdmlError `shape` if `k` is not square or disagrees with `y`; `value` if `alpha` is not positive.
 * @concept kernel-ridge
 * @reference C. Saunders, A. Gammerman and V. Vovk, Ridge regression learning algorithm in dual variables (ICML 1998).
 * @example
 * ```ts
 * import { solveKernelRidge } from 'mdml-engine/kernel-methods'
 * import { toFlat } from 'mdml-engine/tensor'
 * toFlat(solveKernelRidge([[2, 0], [0, 2]], [3, 6], 1)) // => [1, 2]
 * ```
 */
export function solveKernelRidge(k: TensorLike, y: TensorLike, alpha = 1): Tensor {
  const sys = gramSystem(k, validateAlpha(alpha, 'solveKernelRidge'), 'solveKernelRidge')
  const t = targetRows(y, 'solveKernelRidge')
  if (t.n !== sys.n) {
    throw new MdmlError('shape', `solveKernelRidge: the Gram matrix has ${sys.n} rows but y has ${t.n}`)
  }
  const c = matmulRows(sys.inverse, t.rows, sys.n, t.k)
  return fromStorage(c, t.multi ? [t.n, t.k] : [t.n], 'float64')
}

/**
 * The diagonal of the kernel ridge smoother (hat) matrix `H = K (K + alpha I)^(-1) = I - alpha (K + alpha I)^(-1)`:
 * the leverage of each observation. Its sum is the effective degrees of freedom.
 *
 * @param k - Symmetric Gram matrix `[n, n]`.
 * @param alpha - Regularisation `alpha > 0` (default 1).
 * @returns The leverages `[n]`.
 * @concept kernel-ridge
 * @example
 * ```ts
 * import { hatDiagonal } from 'mdml-engine/kernel-methods'
 * import { toFlat } from 'mdml-engine/tensor'
 * toFlat(hatDiagonal([[3, 0], [0, 1]], 1)) // => [0.75, 0.5]
 * ```
 */
export function hatDiagonal(k: TensorLike, alpha = 1): Tensor {
  const a = validateAlpha(alpha, 'hatDiagonal')
  const { n, inverse } = gramSystem(k, a, 'hatDiagonal')
  return fromStorage(
    Float64Array.from({ length: n }, (_, i) => 1 - a * (inverse[i * n + i] ?? 0)),
    [n],
    'float64',
  )
}

/**
 * Exact leave-one-out predictions of kernel ridge regression without refitting `n` times. With
 * `c = (K + alpha I)^(-1) y`, removing observation `i` leaves the residual
 * `y_i - yhat_i^(-i) = c_i / [(K + alpha I)^(-1)]_ii = (y_i - yhat_i) / (1 - H_ii)`.
 *
 * @param k - Symmetric Gram matrix `[n, n]`.
 * @param y - Targets `[n]` or `[n, m]`.
 * @param alpha - Regularisation `alpha > 0` (default 1).
 * @returns The leave-one-out predictions, shaped like `y`.
 * @concept kernel-ridge
 * @reference S. An, W. Liu and S. Venkatesh, Fast cross-validation algorithms for least squares support vector machine
 *   and kernel ridge regression (Pattern Recognition 40, 2007).
 * @example
 * ```ts
 * import { looPredictions } from 'mdml-engine/kernel-methods'
 * import { toFlat } from 'mdml-engine/tensor'
 * toFlat(looPredictions([[2, 1], [1, 2]], [3, 3], 1)) // => [1, 1]
 * ```
 */
export function looPredictions(k: TensorLike, y: TensorLike, alpha = 1): Tensor {
  const sys = gramSystem(k, validateAlpha(alpha, 'looPredictions'), 'looPredictions')
  const t = targetRows(y, 'looPredictions')
  if (t.n !== sys.n) {
    throw new MdmlError('shape', `looPredictions: the Gram matrix has ${sys.n} rows but y has ${t.n}`)
  }
  const c = matmulRows(sys.inverse, t.rows, sys.n, t.k)
  const out = new Float64Array(t.n * t.k)
  for (let i = 0; i < t.n; i++) {
    const d = sys.inverse[i * t.n + i] ?? 1
    for (let j = 0; j < t.k; j++) out[i * t.k + j] = (t.rows[i * t.k + j] ?? 0) - (c[i * t.k + j] ?? 0) / d
  }
  return fromStorage(out, t.multi ? [t.n, t.k] : [t.n], 'float64')
}

/**
 * The exact leave-one-out predicted residual sum of squares (PRESS) of kernel ridge regression,
 * `sum_i ((y_i - yhat_i) / (1 - H_ii))^2`, from one inverse.
 *
 * @param k - Symmetric Gram matrix `[n, n]`.
 * @param y - Targets `[n]` or `[n, m]`.
 * @param alpha - Regularisation `alpha > 0` (default 1).
 * @returns The sum of squared leave-one-out errors over all entries.
 * @concept kernel-ridge
 * @example
 * ```ts
 * import { looPress } from 'mdml-engine/kernel-methods'
 * looPress([[2, 1], [1, 2]], [3, 3], 1) // => 8
 * ```
 */
export function looPress(k: TensorLike, y: TensorLike, alpha = 1): number {
  const pred = toFlat(looPredictions(k, y, alpha))
  const truth = toFlat(primalOf(asTensor(y)))
  let s = 0
  for (let i = 0; i < truth.length; i++) s += ((truth[i] ?? 0) - (pred[i] ?? 0)) ** 2
  return s
}

interface Prepared {
  readonly n: number
  readonly d: number
  readonly k: number
  readonly multi: boolean
  readonly kernel: ResolvedKernel
  readonly gram: Float64Array
  readonly y: Float64Array
}

function prepare(data: LearnData, h: KernelRidgeHyper): Prepared {
  const m = matrixView(data.x, WHO)
  const t = targetRows(data.y ?? [], WHO)
  if (t.n !== m.n) throw new MdmlError('shape', `${WHO}: x has ${m.n} rows but y has ${t.n}`)
  if (data.weights !== undefined) throw new MdmlError('value', `${WHO}: sample weights are not supported`)
  validateAlpha(h.alpha, WHO)
  const kernel = resolveKernel(h as KernelChoice, data.x)
  return { n: m.n, d: m.d, k: t.k, multi: t.multi, kernel, gram: kernelMatrix(h, kernel, data.x), y: t.rows }
}

function verbsOf(p: KernelRidgeParams, h: KernelRidgeHyper): KernelRidgeVerbs {
  const n = p.xFit.length
  const d = p.xFit[0]?.length ?? 0
  const k = p.dualCoef[0]?.length ?? 1
  const dual = Float64Array.from(p.dualCoef.flat())
  const asOutput = (rows: Float64Array, m: number): Tensor => fromStorage(rows, p.multiOutput ? [m, k] : [m], 'float64')
  const predict = (x: TensorLike): Tensor => {
    const m = matrixView(x, WHO, { nFeatures: d })
    const cross = kernelMatrix(h, p.kernel, x, p.xFit)
    const out = new Float64Array(m.n * k)
    for (let i = 0; i < m.n; i++) {
      for (let l = 0; l < n; l++) {
        const v = cross[i * n + l] ?? 0
        for (let j = 0; j < k; j++) out[i * k + j] = (out[i * k + j] ?? 0) + v * (dual[l * k + j] ?? 0)
      }
    }
    return asOutput(out, m.n)
  }
  const training = (): Tensor => fromStorage(Float64Array.from(p.xFit.flat()), [n, d], 'float64')
  const gram = (): Tensor => fromStorage(kernelMatrix(h, p.kernel, training()), [n, n], 'float64')
  return {
    predict,
    predictDistribution: (x) => {
      const mean = predict(x)
      const std = fromStorage(
        new Float64Array(mean.shape.reduce((a, b) => a * b, 1)).fill(Math.max(p.residualStd, 1e-12)),
        [...mean.shape],
        'float64',
      )
      return normal(mean, std)
    },
    score: (data) => r2Score(data.y ?? [], predict(data.x)).estimate.value,
    hatDiagonal: () => hatDiagonal(gram(), p.alpha),
    looPredictions: () => {
      // The targets are recovered from the dual weights: y = (K + alpha I) c.
      const g = Float64Array.from(toFlat(gram()))
      for (let i = 0; i < n; i++) g[i * n + i] = (g[i * n + i] ?? 0) + p.alpha
      return looPredictions(gram(), asOutput(matmulRows(g, dual, n, k), n), p.alpha)
    },
    looPress: () => p.looPress,
    dualCoef: asOutput(dual, n),
    xFit: training(),
    nFeaturesIn: d,
  }
}

/**
 * Kernel ridge regression (scikit-learn `KernelRidge`): minimises `||y - K c||^2 + alpha c' K c` in closed form by
 * `c = (K + alpha I)^(-1) y`, over a named kernel (`linear`, `rbf`, `poly`, `sigmoid`, `laplacian`, ...) or any `Kernel`
 * of the library. The fit is one step of the dual solve, logging the leave-one-out error and the effective degrees
 * of freedom; the fitted value predicts, gives a normal predictive distribution and the exact leave-one-out diagnostics
 * (`hatDiagonal`, `looPredictions`, `looPress`) without refitting. Targets may be a matrix (one output per column).
 *
 * @concept kernel-ridge
 * @reference K. P. Murphy, Machine Learning: A Probabilistic Perspective (2012), section 14.4.3.
 * @example
 * ```ts
 * import { kernelRidge } from 'mdml-engine/kernel-methods'
 * import { toFlat } from 'mdml-engine/tensor'
 * const fitted = kernelRidge.with({ kernel: 'linear', alpha: 1e-8 }).fit({ x: [[1, 0], [1, 1], [1, 2]], y: [1, 3, 5] })
 * toFlat(fitted.predict([[1, 3]])).map((v) => Math.round(v * 1000) / 1000) // => [7]
 * ```
 */
export const kernelRidge: IterativeEstimator<
  KernelRidgeHyper,
  KernelRidgeState,
  KernelRidgeParams,
  KernelRidgeVerbs,
  never
> = iterative<KernelRidgeHyper, KernelRidgeState, KernelRidgeParams, KernelRidgeVerbs, never>({
  name: 'kernel-methods/kernel-ridge',
  kind: 'regressor',
  defaults: { alpha: 1, kernel: 'linear', gamma: 'auto', degree: 3, coef0: 1 },
  hyperSpace: (h) =>
    dict({
      alpha: real(Math.min(1e-6, h.alpha), Math.max(1e3, h.alpha), { log: true, default: h.alpha, label: 'alpha' }),
      ...kernelSpace(h),
    }),
  algorithm: (data, h): Algorithm<KernelRidgeState, never> => {
    const p = prepare(data, h)
    const solve = (): KernelRidgeState => {
      const sys = gramSystem(fromStorage(p.gram, [p.n, p.n], 'float64'), h.alpha, WHO)
      const c = matmulRows(sys.inverse, p.y, p.n, p.k)
      const fitted = matmulRows(p.gram, c, p.n, p.k)
      let rss = 0
      for (let i = 0; i < p.y.length; i++) rss += ((p.y[i] ?? 0) - (fitted[i] ?? 0)) ** 2
      return {
        solved: true,
        dual: Array.from({ length: p.n }, (_, i) => Array.from(c.subarray(i * p.k, (i + 1) * p.k))),
        inverseDiagonal: Array.from({ length: p.n }, (_, i) => sys.inverse[i * p.n + i] ?? 0),
        rss,
      }
    }
    return {
      name: 'kernel-ridge',
      config: { alpha: h.alpha, kernel: p.kernel.kernel, gamma: p.kernel.gamma },
      schema: channels({
        metrics: {
          looMse: { description: 'the exact leave-one-out mean squared error', direction: 'lower' },
          effectiveDf: { description: 'the effective degrees of freedom tr(K (K + alpha I)^(-1))' },
        },
      }),
      init: () => ({ solved: false, dual: [], inverseDiagonal: [], rss: 0 }),
      step: () => solve(),
      done: (s) => (s.solved ? 'converged' : undefined),
      observe: (s) => {
        if (!s.solved) return {}
        let press = 0
        for (let i = 0; i < p.n; i++) {
          for (let j = 0; j < p.k; j++) {
            press += ((s.dual[i]?.[j] ?? 0) / (s.inverseDiagonal[i] ?? 1)) ** 2
          }
        }
        const edf = s.inverseDiagonal.reduce((a, v) => a + 1 - h.alpha * v, 0)
        return { metrics: { looMse: press / (p.n * p.k), effectiveDf: edf } }
      },
      serialise: (s) => stateToJson(s),
      deserialise: (j) => stateFromJson(j) as KernelRidgeState,
    }
  },
  maxSteps: () => 1,
  params: (s, data, h) => {
    const p = prepare(data, h)
    let press = 0
    for (let i = 0; i < p.n; i++) {
      for (let j = 0; j < p.k; j++) press += ((s.dual[i]?.[j] ?? 0) / (s.inverseDiagonal[i] ?? 1)) ** 2
    }
    const edf = s.inverseDiagonal.reduce((a, v) => a + 1 - h.alpha * v, 0)
    const m = matrixView(data.x, WHO)
    return {
      kernel: p.kernel,
      alpha: h.alpha,
      xFit: Array.from({ length: m.n }, (_, i) => Array.from(m.data.subarray(i * m.d, (i + 1) * m.d))),
      dualCoef: s.dual,
      multiOutput: p.multi,
      effectiveDf: edf,
      looPress: press,
      residualStd: Math.sqrt(s.rss / Math.max(p.n * p.k - edf * p.k, 1)),
    }
  },
  verbs: verbsOf,
})
