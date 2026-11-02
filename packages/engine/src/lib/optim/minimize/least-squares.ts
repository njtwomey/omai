/**
 * Nonlinear least squares and curve fitting, scipy.optimize style: robust losses, damped Gauss-Newton as a traced
 * algorithm, the `leastSquares` front end over Gauss-Newton, Levenberg-Marquardt, trust region reflective and dogbox,
 * and `curveFit` for fitting a model to data with parameter covariances (as `scipy.optimize.curve_fit`).
 */

import { MdmlError } from '../../base/index.ts'
import { cholesky, daxpy, ddot, dgesvd, dnrm2, dscal, solveTriangular } from '../../numerics/linalg/index.ts'
import {
  asTensor,
  div,
  fromStorage,
  sub,
  type Tensor,
  type TensorLike,
  toFlat,
  type Value,
} from '../../tensor/index.ts'
import type { Algorithm } from '../../trace/index.ts'
import { backtracking, type LineSearch, type LineSearchResult } from '../line-search.ts'
import { jtv, lstsqSvd, robustCost, runLeastSquares, scaleForLoss } from './_least-squares.ts'
import type { VectorState } from './_shared.ts'
import { configOf, LINE_SEARCH_CHANNEL, lineSearchMessage, observeVector, vectorDone, vectorSchema } from './_shared.ts'
import type { DogboxOptions } from './least-squares-bounded.ts'
import { trf } from './least-squares-trf.ts'
import { type LevenbergMarquardtOptions, levenbergMarquardt } from './levenberg-marquardt.ts'
import {
  type FlatResiduals,
  flatResiduals,
  infNorm,
  type MinimizeResult,
  type Parameters,
  type ResidualFunction,
  type VectorMethodOptions,
} from './objective.ts'

/** A robust loss rho(z) of z = (f / fScale)^2 (scipy's least_squares losses). */
export type RobustLoss = 'linear' | 'soft_l1' | 'huber' | 'cauchy' | 'arctan'

/** The state of a least-squares method: `value` is the cost and `grad` its gradient J'f. */
export interface LeastSquaresState extends VectorState {
  /** The residuals at x (unscaled). */
  readonly residuals: Float64Array
  /** The Jacobian at x, m x n row-major. */
  readonly jacobian: Float64Array
  readonly njev: number
  /** -1 at a lower bound, 1 at an upper bound, 0 free (bounded methods). */
  readonly activeMask?: Int8Array
}

/** Options shared by the nonlinear least-squares methods. */
export interface LeastSquaresOptions extends VectorMethodOptions {
  /** Stop when the cost changes by less than ftol times itself (default 1e-8). */
  readonly ftol?: number
  /** Stop when the step is below xtol (xtol + |x|) (default 1e-8). */
  readonly xtol?: number
  /** Residual evaluations in total (default 100 n). */
  readonly maxEvaluations?: number
}

/** Result of a least-squares method (scipy's least_squares result). */
export interface LeastSquaresResult extends MinimizeResult {
  /** Half the sum of squared residuals (of the robust loss, if one is used). */
  readonly cost: number
  /** The residuals at the solution. */
  readonly residuals: Tensor
  /** The Jacobian at the solution, m x n. */
  readonly jacobian: Tensor
  /** The infinity norm of the (projected) gradient. */
  readonly optimality: number
  /** Jacobian evaluations. */
  readonly njev: number
  /** -1 at a lower bound, 1 at an upper bound, 0 free. */
  readonly activeMask: readonly number[]
}

const SCHEMA_EXTRA = {
  metrics: { cost: { description: 'half the sum of squared residuals' } },
}

/** Options of {@link gaussNewton}. */
export interface GaussNewtonOptions extends LeastSquaresOptions {
  /** The line search on the cost along the Gauss-Newton direction (default safeguarded Armijo backtracking). */
  readonly lineSearch?: LineSearch
  /** A robust loss of the residuals (default linear) and its scale. */
  readonly loss?: RobustLoss
  readonly fScale?: number
}

/** The state of a {@link gaussNewton} run. */
export interface GaussNewtonState extends LeastSquaresState {
  readonly lineSearch?: LineSearchResult
}

/**
 * The damped Gauss-Newton method as a traced algorithm: each step solves the linearised least-squares problem
 * J p = -f (minimum norm, by the SVD) and searches along p on the cost 0.5 |f|^2 (or a robust cost). Records the cost,
 * gradient norm and line-search trials.
 *
 * @param r - The residual function.
 * @param x0 - The start.
 * @param options - Line search, robust loss and tolerances (gtol on the infinity norm of J'f, default 1e-8).
 * @returns The algorithm.
 * @concept least-squares
 * @example
 * ```ts
 * import { gaussNewtonAlgorithm } from 'mdml-engine/optim'
 * import { sub, type Value } from 'mdml-engine/tensor'
 * gaussNewtonAlgorithm((p: Value) => sub(p, 1), [0]).name // => 'gaussNewton'
 * ```
 */
export function gaussNewtonAlgorithm(
  r: ResidualFunction,
  x0: Parameters,
  options: GaussNewtonOptions = {},
): Algorithm<GaussNewtonState> {
  return gaussNewtonOn(flatResiduals(r, x0), options)
}

function gaussNewtonOn(res: FlatResiduals, options: GaussNewtonOptions): Algorithm<GaussNewtonState> {
  const { m, n } = res
  const gtol = options.gradTol ?? 1e-8
  const ftol = options.ftol ?? 1e-8
  const xtol = options.xtol ?? 1e-8
  const loss = options.loss ?? 'linear'
  const fScale = options.fScale ?? 1
  const search = options.lineSearch ?? backtracking({ safeguard: [0.1, 0.5] })
  const evaluate = (x: Float64Array) => {
    const f = res.residuals(x)
    const J = res.jacobian(x)
    const scaled = scaleForLoss(J, f, m, n, loss, fScale)
    return { f, J, cost: robustCost(f, loss, fScale), g: jtv(scaled.J, m, n, scaled.f), scaled }
  }
  return {
    name: 'gaussNewton',
    config: configOf(options),
    schema: vectorSchema({ ...SCHEMA_EXTRA, messages: LINE_SEARCH_CHANNEL }),
    init: () => {
      const e = evaluate(res.x0)
      return {
        x: res.x0,
        value: e.cost,
        grad: e.g,
        nfev: 1,
        ngev: 1,
        nhev: 0,
        njev: 1,
        step: 0,
        residuals: e.f,
        jacobian: e.J,
        ...(infNorm(e.g) < gtol ? { stop: { status: 'converged' as const, message: 'gradient norm below gtol' } } : {}),
      }
    },
    step: (s) => {
      const scaled = scaleForLoss(s.jacobian, s.residuals, m, n, loss, fScale)
      const p = lstsqSvd(scaled.J, m, n, dscal(-1, scaled.f))
      const slope = ddot(s.grad, p)
      const phi = {
        at: (alpha: number) => {
          const f = res.residuals(daxpy(alpha, p, s.x))
          // The slope along p from the Gauss-Newton model at the trial point is not needed by Armijo searches.
          return { value: robustCost(f, loss, fScale), slope: Number.NaN }
        },
        valueAt: (alpha: number) => robustCost(res.residuals(daxpy(alpha, p, s.x)), loss, fScale),
      }
      const ls = search(phi, { value: s.value, slope, alpha: 1 })
      if (!ls.success) {
        return {
          ...s,
          nfev: s.nfev + ls.evaluations,
          lineSearch: ls,
          stop: { status: 'failed', message: `line search failed: ${ls.message}` },
        }
      }
      const step = dscal(ls.alpha, p)
      const x = daxpy(1, step, s.x)
      const e = evaluate(x)
      const dF = s.value - e.cost
      const stop =
        infNorm(e.g) < gtol
          ? { status: 'converged' as const, message: 'gradient norm below gtol' }
          : dF < ftol * e.cost
            ? { status: 'converged' as const, message: 'cost reduction below ftol' }
            : dnrm2(step) < xtol * (xtol + dnrm2(x))
              ? { status: 'converged' as const, message: 'step below xtol' }
              : undefined
      return {
        x,
        value: e.cost,
        grad: e.g,
        nfev: s.nfev + ls.evaluations + 1,
        ngev: s.ngev + 1,
        nhev: 0,
        njev: s.njev + 1,
        step: dnrm2(step),
        residuals: e.f,
        jacobian: e.J,
        lineSearch: ls,
        ...(stop === undefined ? {} : { stop }),
      }
    },
    done: vectorDone,
    observe: (s) =>
      observeVector(s, {
        metrics: { cost: s.value },
        ...(s.lineSearch === undefined ? {} : { messages: [lineSearchMessage(s.lineSearch)] }),
      }),
  }
}

/**
 * Solve a nonlinear least-squares problem by damped Gauss-Newton.
 *
 * @param r - The residual function.
 * @param x0 - The start.
 * @param options - Line search, robust loss, tolerances, iteration cap (default 100 n) and run options.
 * @returns The result with the cost, residuals and Jacobian.
 * @concept least-squares
 * @example
 * ```ts
 * import { gaussNewton } from 'mdml-engine/optim'
 * import { mul, sub, tensor, toFlat, type Value } from 'mdml-engine/tensor'
 * // x^2 = [4, 9] elementwise from [1, 1].
 * const r = gaussNewton((x: Value) => sub(mul(x, x), tensor([4, 9])), [1, 1])
 * toFlat(r.x as Value).map((v) => Math.round(v * 1e8) / 1e8) // => [2, 3]
 * ```
 */
export function gaussNewton(r: ResidualFunction, x0: Parameters, options: GaussNewtonOptions = {}): LeastSquaresResult {
  const res = flatResiduals(r, x0)
  return runLeastSquares(gaussNewtonOn(res, options), res, options, 100 * res.n)
}

/** A least-squares method as a function value, for {@link leastSquares}. */
export type LeastSquaresMethod = (r: ResidualFunction, x0: Parameters, options?: object) => LeastSquaresResult

/** Options of {@link leastSquares}. */
export type LeastSquaresSolveOptions = LevenbergMarquardtOptions &
  GaussNewtonOptions &
  DogboxOptions & {
    /** The method (default the trust region reflective method, as scipy's). */
    readonly method?: LeastSquaresMethod
  }

/**
 * Solve a nonlinear least-squares problem, scipy.optimize.least_squares style: minimise 0.5 sum rho(f_i^2) subject to
 * bounds. The method is a function value: {@link trf} (default, as scipy's), {@link dogbox},
 * {@link levenbergMarquardt} (no bounds or losses) or {@link gaussNewton}.
 *
 * @param r - The residual function.
 * @param x0 - The start.
 * @param options - The method and its options (bounds, loss, scaling, tolerances).
 * @returns The result.
 * @concept least-squares
 * @index function Nonlinear least squares, scipy style, with bounds and robust losses.
 * @example
 * ```ts
 * import { leastSquares } from 'mdml-engine/optim'
 * import { sub, type Value } from 'mdml-engine/tensor'
 * const r = leastSquares((p: Value) => sub(p, 3), [0], { bounds: [[null, 1]] })
 * [r.converged, Math.round(r.cost * 1e8) / 1e8, r.activeMask] // => [true, 2, [1]]
 * ```
 */
export function leastSquares(
  r: ResidualFunction,
  x0: Parameters,
  options: LeastSquaresSolveOptions = {},
): LeastSquaresResult {
  const { method, ...rest } = options
  return (method ?? (trf as LeastSquaresMethod))(r, x0, rest)
}

/** Options of {@link curveFit}. */
export interface CurveFitOptions extends Omit<LeastSquaresSolveOptions, 'loss' | 'fScale'> {
  /** Uncertainties of ydata (a scalar or one per point), or their covariance matrix. */
  readonly sigma?: TensorLike
  /** Treat sigma as absolute (pcov not rescaled by the residual variance; default false). */
  readonly absoluteSigma?: boolean
  /** A robust loss (fitted by the trust region reflective method). */
  readonly loss?: RobustLoss
  readonly fScale?: number
}

/** Result of {@link curveFit}. */
export interface CurveFitResult {
  /** The fitted parameters, with the structure of p0. */
  readonly popt: Parameters
  /** Their covariance (Infinity when it cannot be estimated). */
  readonly pcov: Tensor
  /** Standard errors, sqrt(diag(pcov)). */
  readonly perr: Tensor
  /** The underlying least-squares result. */
  readonly result: LeastSquaresResult
}

/**
 * Fit a model to data by nonlinear least squares (scipy.optimize.curve_fit): minimise the sum of squared weighted
 * residuals (model(xdata, p) - ydata) / sigma (or whitened by the Cholesky factor of a covariance sigma), and estimate
 * the parameter covariance from the Jacobian at the solution by the SVD, scaled by the residual variance unless
 * `absoluteSigma`.
 *
 * @param model - The model, a function of the inputs and the parameters.
 * @param xdata - The inputs (passed to the model as a tensor).
 * @param ydata - The observations.
 * @param p0 - The initial parameters (any parameter tree).
 * @param options - sigma, absoluteSigma, bounds, loss and the least-squares options.
 * @returns The fitted parameters, covariance, standard errors and the least-squares result.
 * @throws MdmlError with code `convergence` when the fit fails.
 * @concept least-squares
 * @index function Fit a parametric curve to data by nonlinear least squares.
 * @example
 * ```ts
 * import { curveFit } from 'mdml-engine/optim'
 * import { add, mul, sliceAxis, toFlat, type Value } from 'mdml-engine/tensor'
 * const line = (x: Value, p: unknown) => add(mul(sliceAxis(p as Value, 0, 0, 1), x), sliceAxis(p as Value, 0, 1, 2))
 * const fit = curveFit(line, [0, 1, 2, 3], [1, 3, 5, 7], [0, 0])
 * toFlat(fit.popt as Value).map((v) => Math.round(v * 1e8) / 1e8) // => [2, 1]
 * ```
 */
export function curveFit(
  model: (xdata: Value, params: Parameters) => Value,
  xdata: TensorLike,
  ydata: TensorLike,
  p0: Parameters,
  options: CurveFitOptions = {},
): CurveFitResult {
  const x = asTensor(xdata)
  const y = asTensor(ydata)
  const ysize = toFlat(y).length
  const { sigma, absoluteSigma, ...rest } = options
  let whiten: (d: Value) => Value = (d) => d
  if (sigma !== undefined) {
    const sg = asTensor(sigma)
    if (sg.shape.length === 2) {
      const L = cholesky(sg)
      whiten = (d) => solveTriangular(L, d, { lower: true })
    } else whiten = (d) => div(d, sg)
  }
  const r = (p: unknown): Value => whiten(sub(model(x, p as Parameters), y))
  // scipy's curve_fit: Levenberg-Marquardt without bounds or a robust loss, the trust region reflective method with.
  const plain = rest.bounds === undefined && (rest.loss === undefined || rest.loss === 'linear')
  const result = leastSquares(r, p0, {
    ...rest,
    method: rest.method ?? (plain ? (levenbergMarquardt as LeastSquaresMethod) : (trf as LeastSquaresMethod)),
  })
  if (!result.converged && result.status !== 'completed')
    throw new MdmlError('convergence', `curveFit: optimal parameters not found: ${result.message}`)
  const J = Float64Array.from(toFlat(result.jacobian))
  const [m, n] = result.jacobian.shape as [number, number]
  const { s, v } = dgesvd(J, m, n)
  const k = s.length
  const threshold = Number.EPSILON * Math.max(m, n) * (s[0] ?? 0)
  const pcov = new Float64Array(n * n)
  for (let c = 0; c < k; c++) {
    const sc = s[c] ?? 0
    if (!(sc > threshold)) continue
    for (let i = 0; i < n; i++)
      for (let j = 0; j < n; j++)
        pcov[i * n + j] = (pcov[i * n + j] ?? 0) + ((v[i * k + c] ?? 0) * (v[j * k + c] ?? 0)) / (sc * sc)
  }
  const nparams = n
  if (absoluteSigma !== true) {
    if (ysize > nparams) {
      const sSq = (2 * result.cost) / (ysize - nparams)
      for (let i = 0; i < pcov.length; i++) pcov[i] = (pcov[i] ?? 0) * sSq
    } else pcov.fill(Infinity)
  }
  if (pcov.some((p) => Number.isNaN(p))) pcov.fill(Infinity)
  const perr = new Float64Array(n)
  for (let i = 0; i < n; i++) perr[i] = Math.sqrt(pcov[i * n + i] ?? Number.NaN)
  return {
    popt: result.x as Parameters,
    pcov: fromStorage(pcov, [n, n], 'float64'),
    perr: fromStorage(perr, [n], 'float64'),
    result,
  }
}
