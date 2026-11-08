/**
 * Asymmetric kernel regression: kernel expectile regression (`kernelExpectileRegression`), the conditional expectile as a
 * function in a reproducing kernel Hilbert space, a smooth convex problem whose objective is `kernelExpectileProblem` for
 * the solver slot (iteratively reweighted ridge in the dual by default, any solver of `smoothSolvers` otherwise); and sparse
 * kernel quantile regression (`kernelQuantileRegression`), the conditional quantile as an expansion over the training
 * points with an L1 penalty on its coefficients, solved exactly as a linear programme by the quantile regression of
 * `linear` on the Gram matrix.
 */

import { OmaiError } from '../base/index.ts'
import { type KernelChoice, kernelMatrix, kernelSpace, type ResolvedKernel, resolveKernel } from '../kernels/index.ts'
import { type IterativeEstimator, iterative, matrixView, targetVector } from '../learn/index.ts'
import { quantileRegression } from '../linear/index.ts'
import { solve } from '../numerics/linalg/index.ts'
import { type SmoothProblem, smoothSolvers } from '../optim/index.ts'
import { dict, real } from '../space/index.ts'
import { asTensor, fromStorage, primalOf, type Tensor, type TensorLike, toFlat } from '../tensor/index.ts'
import { Log } from '../trace/index.ts'
import { KERNEL_DEFAULTS, type KernelOptions, queryView } from './_classifier.ts'
import { infNorm, problemOf, type SlotState, type Smooth, type SolverChoice, slotted } from './_slot.ts'
import { type ExpansionParams, type OnlineRegressorVerbs, regressorVerbs } from './online-learners.ts'

/** The asymmetric squared loss `2 |tau - 1[r < 0]| r^2` of the residuals `y - K c`, and the RKHS penalty `alpha c' K c`. */
function expectileSmooth(k: Float64Array, y: Float64Array, n: number, tau: number, alpha: number): Smooth {
  const residual = (c: ArrayLike<number>): Float64Array => {
    const r = new Float64Array(n)
    for (let i = 0; i < n; i++) {
      let s = 0
      for (let j = 0; j < n; j++) s += (k[i * n + j] ?? 0) * (c[j] ?? 0)
      r[i] = (y[i] ?? 0) - s
    }
    return r
  }
  const weights = (r: Float64Array): Float64Array => Float64Array.from(r, (v) => 2 * (v >= 0 ? tau : 1 - tau))
  return {
    value: (c) => {
      const r = residual(c)
      const w = weights(r)
      let v = 0
      for (let i = 0; i < n; i++) v += (w[i] ?? 0) * (r[i] ?? 0) ** 2
      for (let i = 0; i < n; i++)
        for (let j = 0; j < n; j++) v += alpha * (c[i] ?? 0) * (k[i * n + j] ?? 0) * (c[j] ?? 0)
      return v
    },
    gradient: (c) => {
      const r = residual(c)
      const w = weights(r)
      const g = new Float64Array(n)
      for (let i = 0; i < n; i++) {
        let s = 0
        for (let j = 0; j < n; j++) s += (k[i * n + j] ?? 0) * (alpha * (c[j] ?? 0) - (w[j] ?? 0) * (r[j] ?? 0))
        g[i] = 2 * s
      }
      return g
    },
    hessian: (c) => {
      const w = weights(residual(c))
      const h = new Float64Array(n * n)
      for (let i = 0; i < n; i++)
        for (let j = 0; j < n; j++) {
          let s = alpha * (k[i * n + j] ?? 0)
          for (let l = 0; l < n; l++) s += (k[i * n + l] ?? 0) * (w[l] ?? 0) * (k[l * n + j] ?? 0)
          h[i * n + j] = 2 * s
        }
      return h
    },
  }
}

/** One reweighted ridge step in the dual: `(W K + alpha I) c = W y` with the weights at the current residuals. */
function expectileIrls(
  k: Float64Array,
  y: Float64Array,
  n: number,
  tau: number,
  alpha: number,
  c: ArrayLike<number>,
): Float64Array {
  const w = new Float64Array(n)
  for (let i = 0; i < n; i++) {
    let s = 0
    for (let j = 0; j < n; j++) s += (k[i * n + j] ?? 0) * (c[j] ?? 0)
    w[i] = 2 * ((y[i] ?? 0) - s >= 0 ? tau : 1 - tau)
  }
  const a = new Float64Array(n * n)
  for (let i = 0; i < n; i++)
    for (let j = 0; j < n; j++) a[i * n + j] = (w[i] ?? 0) * (k[i * n + j] ?? 0) + (i === j ? alpha : 0)
  const rhs = Float64Array.from(w, (v, i) => v * (y[i] ?? 0))
  return Float64Array.from(
    toFlat(primalOf(solve(fromStorage(a, [n, n], 'float64'), fromStorage(rhs, [n, 1], 'float64')))),
  )
}

function checkTau(tau: number, who: string): void {
  if (!(tau > 0 && tau < 1)) throw new OmaiError('value', `${who}: the level must be in (0, 1), got ${tau}`)
}

/**
 * The kernel expectile objective as a smooth problem for the solver slot: over the dual weights `c` of `f = K c`, the
 * asymmetric squared loss `sum_i 2 |tau - 1[r_i < 0]| r_i^2` (`r = y - K c`, so `tau = 1/2` is the squared error) plus
 * the RKHS penalty `alpha c' K c`, with its gradient and generalised Hessian. At `tau = 1/2` its minimiser is kernel ridge
 * regression with the same `alpha`.
 *
 * @param k - Gram matrix `[n, n]`.
 * @param y - Targets `[n]`.
 * @param tau - The expectile level in (0, 1) (default 0.5).
 * @param alpha - The penalty `alpha > 0` (default 1).
 * @returns The problem, starting at zero.
 * @throws OmaiError `value` for a level outside (0, 1) or a non-positive penalty; `shape` for a mismatched Gram matrix.
 * @concept kernel-regression
 * @reference W. K. Newey and J. L. Powell, Asymmetric least squares estimation and testing (Econometrica 55, 1987);
 *   Y. Yang, T. Zhang and H. Zou, Flexible expectile regression in reproducing kernel Hilbert spaces (Technometrics 60, 2018).
 * @example
 * ```ts
 * import { kernelExpectileProblem } from 'omai-engine/kernel-methods'
 * kernelExpectileProblem([[1, 0], [0, 1]], [1, 2], 0.8, 0.5).x0 // => [0, 0]
 * ```
 */
export function kernelExpectileProblem(k: TensorLike, y: TensorLike, tau = 0.5, alpha = 1): SmoothProblem {
  checkTau(tau, 'kernelExpectileProblem')
  if (!(alpha > 0)) throw new OmaiError('value', `kernelExpectileProblem: alpha must be positive, got ${alpha}`)
  const g = primalOf(asTensor(k))
  const n = g.shape[0] ?? 0
  if (g.shape.length !== 2 || g.shape[1] !== n) throw new OmaiError('shape', 'kernelExpectileProblem: k must be square')
  const t = targetVector(y, 'kernelExpectileProblem', n)
  return problemOf(expectileSmooth(Float64Array.from(toFlat(g)), t, n, tau, alpha), new Array<number>(n).fill(0))
}

/** The hyperparameters of {@link kernelExpectileRegression}. */
export interface KernelExpectileHyper extends KernelOptions {
  /** The expectile level in (0, 1); 0.5 is the mean (default). */
  readonly tau: number
  /** The RKHS penalty (default 1). */
  readonly alpha: number
  /** `'irls'` (iteratively reweighted kernel ridge, default) or a value of `smoothSolvers.space`. */
  readonly solver: SolverChoice
  readonly tol: number
  readonly maxIter: number
}

/** The state of {@link kernelExpectileRegression}: the dual weights. */
export type KernelExpectileState = SlotState & {
  readonly theta: number[]
  readonly objective: number
  readonly gradient: number
}

const EXPECTILE_DEFAULTS: KernelExpectileHyper = {
  ...KERNEL_DEFAULTS,
  tau: 0.5,
  alpha: 1,
  solver: 'irls',
  tol: 1e-10,
  maxIter: 100,
}

/**
 * Kernel expectile regression: the conditional `tau`-expectile of the target as a function in a reproducing kernel
 * Hilbert space, the minimiser of the asymmetric squared loss plus an RKHS penalty. At `tau = 1/2` it is kernel ridge
 * regression; elsewhere it tracks the upper or lower part of the conditional distribution. The minimisation is the solver
 * slot: iteratively reweighted kernel ridge (`'irls'`, Newton's method for this loss) or any solver of `smoothSolvers`.
 * There is no intercept term, as in kernel ridge.
 *
 * @param hyper - The kernel options, `tau`, `alpha`, `solver`, `tol`, `maxIter`.
 * @returns The estimator.
 * @concept kernel-regression
 * @example
 * ```ts
 * import { kernelExpectileRegression } from 'omai-engine/kernel-methods'
 * import { toFlat } from 'omai-engine/tensor'
 * const x = [[0], [1], [2], [3], [4], [5]]
 * const y = [0.1, 1.3, 1.8, 3.4, 3.7, 5.2]
 * const hi = toFlat(kernelExpectileRegression({ kernel: 'linear', tau: 0.9, alpha: 0.1 }).fit({ x, y }).predict([[2.5]]))[0] ?? 0
 * const mid = toFlat(kernelExpectileRegression({ kernel: 'linear', tau: 0.5, alpha: 0.1 }).fit({ x, y }).predict([[2.5]]))[0] ?? 0
 * hi > mid // => true
 * ```
 */
export function kernelExpectileRegression(
  hyper: Partial<KernelExpectileHyper> = {},
): IterativeEstimator<KernelExpectileHyper, KernelExpectileState, ExpansionParams, OnlineRegressorVerbs> {
  const who = 'kernelExpectileRegression'
  const setup = (data: { readonly x: TensorLike; readonly y?: TensorLike }, h: KernelExpectileHyper) => {
    checkTau(h.tau, who)
    if (!(h.alpha > 0)) throw new OmaiError('value', `${who}: alpha must be positive, got ${h.alpha}`)
    const m = matrixView(data.x, who)
    const y = targetVector(data.y ?? [], who, m.n)
    const resolved = resolveKernel(h as KernelChoice, data.x)
    return { m, y, resolved, k: kernelMatrix(h, resolved, data.x) }
  }
  return iterative<KernelExpectileHyper, KernelExpectileState, ExpansionParams, OnlineRegressorVerbs>(
    {
      name: 'kernel-methods/kernel-expectile-regression',
      kind: 'regressor',
      defaults: EXPECTILE_DEFAULTS,
      hyperSpace: (h) =>
        dict({
          tau: real(0.01, 0.99, { default: h.tau, label: 'expectile level' }),
          alpha: real(Math.min(1e-6, h.alpha), Math.max(1e3, h.alpha), {
            log: true,
            default: h.alpha,
            label: 'penalty',
          }),
          ...kernelSpace(h),
          ...(h.solver === 'irls' ? {} : { solver: smoothSolvers.space }),
        }),
      algorithm: (data, h, key) => {
        const { m, y, resolved, k } = setup(data, h)
        const smooth = expectileSmooth(k, y, m.n, h.tau, h.alpha)
        return slotted<KernelExpectileState>({
          name: 'kernel-expectile-regression',
          config: { tau: h.tau, alpha: h.alpha, kernel: resolved.kernel },
          schema: {
            metrics: {
              objective: { description: 'the asymmetric squared loss plus the penalty', direction: 'lower' },
              gradient: { description: 'the largest gradient entry', direction: 'lower' },
            },
            tensors: {},
          },
          problem: problemOf(smooth, new Array<number>(m.n).fill(0)),
          smooth,
          solver: h.solver,
          direct: (c) => expectileIrls(k, y, m.n, h.tau, h.alpha, c),
          key,
          tol: h.tol,
          view: (r) => ({ theta: Array.from(r.x), objective: r.value, gradient: infNorm(r.grad) }),
          observe: (s) => ({ metrics: { objective: s.objective, gradient: s.gradient } }),
        })
      },
      maxSteps: (h) => h.maxIter,
      params: (s, data, h) => {
        const { m, resolved } = setup(data, h)
        return {
          kernel: resolved,
          classes: [],
          centres: Array.from({ length: m.n }, (_, i) => Array.from(m.data.subarray(i * m.d, (i + 1) * m.d))),
          coef: s.theta,
          nFeaturesIn: m.d,
          nObservations: m.n,
          mistakes: 0,
          meanLoss: s.objective / m.n,
        }
      },
      verbs: (p, h) => regressorVerbs(p, h, who),
      logs: { 'solver/objective': Log(), 'solver/gradient': Log() },
    },
    { ...EXPECTILE_DEFAULTS, ...hyper },
  )
}

/** Options of {@link kernelQuantileRegression}. */
export interface KernelQuantileOptions {
  /** The kernel (default RBF with `gamma = 1 / d`). */
  readonly kernel?: KernelOptions
  /** The quantile level in (0, 1) (default 0.5). */
  readonly quantile?: number
  /** The L1 penalty on the dual coefficients, on the mean pinball loss (default 0.01); larger is sparser. */
  readonly alpha?: number
}

/** A fitted sparse kernel quantile regression. */
export interface KernelQuantileFit {
  readonly quantile: number
  readonly kernel: ResolvedKernel
  /** The dual coefficients `[n]`; most are zero for a large penalty. */
  readonly coef: number[]
  readonly intercept: number
  /** The training rows with a non-zero coefficient. */
  readonly supportCount: number
  /** The conditional quantile at rows `[m, d]`. */
  predict(x: TensorLike): Tensor
}

/**
 * Sparse kernel quantile regression: the conditional `quantile` as `f(x) = sum_i c_i k(x_i, x) + b`, the minimiser of the
 * mean pinball loss plus `alpha |c|_1`, an exact linear programme (the quantile regression of `linear` with the Gram
 * matrix as the design). The L1 penalty keeps few support points; the RKHS-norm penalty of the quadratic-programme form is
 * not offered.
 *
 * @param x - Features `[n, d]`.
 * @param y - Targets `[n]`.
 * @param options - The kernel, the quantile and the penalty.
 * @returns The fit, with a `predict`.
 * @throws OmaiError `value` for a quantile outside (0, 1) or a failed programme.
 * @concept kernel-regression
 * @reference I. Takeuchi, Q. V. Le, T. D. Sears and A. J. Smola, Nonparametric quantile estimation (JMLR 7, 2006);
 *   Y. Li, Y. Liu and J. Zhu, Quantile regression in reproducing kernel Hilbert spaces (JASA 102, 2007).
 * @example
 * ```ts
 * import { kernelQuantileRegression } from 'omai-engine/kernel-methods'
 * import { toFlat } from 'omai-engine/tensor'
 * const x = [[0], [1], [2], [3], [4]]
 * const fit = kernelQuantileRegression(x, [0, 1, 2, 3, 50], { kernel: { kernel: 'linear', gamma: 1, degree: 1, coef0: 0 }, alpha: 0 })
 * toFlat(fit.predict([[2]]))[0]?.toFixed(2) // => '2.00'
 * ```
 */
export function kernelQuantileRegression(
  x: TensorLike,
  y: TensorLike,
  options: KernelQuantileOptions = {},
): KernelQuantileFit {
  const who = 'kernelQuantileRegression'
  const tau = options.quantile ?? 0.5
  checkTau(tau, who)
  const choice = options.kernel ?? KERNEL_DEFAULTS
  const m = matrixView(x, who)
  const targets = targetVector(y, who, m.n)
  const resolved = resolveKernel(choice as KernelChoice, x)
  const gram = fromStorage(kernelMatrix(choice, resolved, x), [m.n, m.n], 'float64')
  const fit = quantileRegression(
    { x: gram, y: targets },
    { quantile: tau, alpha: options.alpha ?? 0.01, fitIntercept: true },
  )
  const centres = Array.from({ length: m.n }, (_, i) => Array.from(m.data.subarray(i * m.d, (i + 1) * m.d)))
  return {
    quantile: tau,
    kernel: resolved,
    coef: fit.coef,
    intercept: fit.intercept,
    supportCount: fit.coef.filter((c) => Math.abs(c) > 1e-9).length,
    predict: (xq) => {
      const q = queryView(xq, m.d, who)
      const cross = kernelMatrix(choice, resolved, xq, centres)
      return fromStorage(
        Float64Array.from({ length: q.n }, (_, i) => {
          let s = fit.intercept
          for (let j = 0; j < m.n; j++) s += (cross[i * m.n + j] ?? 0) * (fit.coef[j] ?? 0)
          return s
        }),
        [q.n],
        'float64',
      )
    },
  }
}
