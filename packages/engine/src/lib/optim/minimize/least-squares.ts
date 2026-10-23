import { WhiteboxError } from '../../base/index.ts'
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
import {
  jtv,
  jv,
  lstsqSvd,
  requireOverdetermined,
  robustCost,
  runLeastSquares,
  scaleForLoss,
} from './_least-squares.ts'
import type { VectorState } from './_shared.ts'
import { configOf, LINE_SEARCH_CHANNEL, lineSearchMessage, observeVector, vectorDone, vectorSchema } from './_shared.ts'
import { type DogboxOptions, dogbox } from './least-squares-bounded.ts'
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
 * import { gaussNewtonAlgorithm } from 'whitebox-engine/optim'
 * import { sub, type Value } from 'whitebox-engine/tensor'
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
 * import { gaussNewton } from 'whitebox-engine/optim'
 * import { mul, sub, tensor, toFlat, type Value } from 'whitebox-engine/tensor'
 * // x^2 = [4, 9] elementwise from [1, 1].
 * const r = gaussNewton((x: Value) => sub(mul(x, x), tensor([4, 9])), [1, 1])
 * toFlat(r.x as Value).map((v) => Math.round(v * 1e8) / 1e8) // => [2, 3]
 * ```
 */
export function gaussNewton(r: ResidualFunction, x0: Parameters, options: GaussNewtonOptions = {}): LeastSquaresResult {
  const res = flatResiduals(r, x0)
  return runLeastSquares(gaussNewtonOn(res, options), res, options, 100 * res.n)
}

/** Options of {@link levenbergMarquardt}. */
export interface LevenbergMarquardtOptions extends LeastSquaresOptions {
  /** Variable scales: 'jac' (default, MINPACK's automatic scaling by Jacobian column norms) or one scale per variable. */
  readonly xScale?: 'jac' | ArrayLike<number> | number
  /** The initial trust radius as a multiple of |D x0| (default 100). */
  readonly factor?: number
}

/** The state of a {@link levenbergMarquardt} run. */
export interface LevenbergMarquardtState extends LeastSquaresState {
  readonly diag: Float64Array
  /** The trust radius on |D p|. */
  readonly delta: number
  /** The Levenberg-Marquardt parameter. */
  readonly par: number
  /** The last ratio of actual to predicted reduction. */
  readonly ratio: number
  /** MINPACK's iteration counter (successful steps + 1). */
  readonly iter: number
}

const DWARF = 2.2250738585072014e-308

/** MINPACK's lmpar on the SVD: the parameter with |D p| within 10% of delta, and the step p. */
function lmpar(
  J: Float64Array,
  m: number,
  n: number,
  f: Float64Array,
  diag: Float64Array,
  delta: number,
  par0: number,
): { par: number; p: Float64Array } {
  const Js = new Float64Array(m * n)
  for (let i = 0; i < m; i++) for (let j = 0; j < n; j++) Js[i * n + j] = (J[i * n + j] ?? 0) / (diag[j] ?? 1)
  const { u, s, v } = dgesvd(Js, m, n)
  const k = s.length
  const utf = new Float64Array(k)
  for (let c = 0; c < k; c++) {
    let t = 0
    for (let i = 0; i < m; i++) t += (u[i * k + c] ?? 0) * (f[i] ?? 0)
    utf[c] = t
  }
  const tol = Number.EPSILON * Math.max(m, n) * (s[0] ?? 0)
  const nsing = s.reduce((c, si) => c + (si > tol ? 1 : 0), 0)
  // y(lambda) = -sum s_i (u_i'f) / (s_i^2 + lambda) v_i, in the scaled variables y = D p.
  const yOf = (lambda: number): Float64Array => {
    const y = new Float64Array(n)
    for (let c = 0; c < k; c++) {
      const sc = s[c] ?? 0
      if (sc <= tol && lambda === 0) continue
      const w = -(sc * (utf[c] ?? 0)) / (sc * sc + lambda)
      for (let j = 0; j < n; j++) y[j] = (y[j] ?? 0) + w * (v[j * k + c] ?? 0)
    }
    return y
  }
  const curvature = (y: Float64Array, ynorm: number, lambda: number): number => {
    let zz = 0
    for (let c = 0; c < k; c++) {
      let vy = 0
      for (let j = 0; j < n; j++) vy += (v[j * k + c] ?? 0) * (y[j] ?? 0)
      const sc = s[c] ?? 0
      zz += (vy / ynorm) ** 2 / (sc * sc + lambda)
    }
    return zz
  }
  const unscale = (y: Float64Array): Float64Array => y.map((yi, j) => yi / (diag[j] ?? 1))
  let y = yOf(0)
  let dxnorm = dnrm2(y)
  let fp = dxnorm - delta
  if (fp <= 0.1 * delta) return { par: 0, p: unscale(y) }
  let parl = 0
  if (nsing === n) parl = fp / (delta * curvature(y, dxnorm, 0))
  let gnorm = 0
  for (let c = 0; c < k; c++) gnorm += ((s[c] ?? 0) * (utf[c] ?? 0)) ** 2
  gnorm = Math.sqrt(gnorm)
  let paru = gnorm / delta
  if (paru === 0) paru = DWARF / Math.min(delta, 0.1)
  let par = Math.min(Math.max(par0, parl), paru)
  if (par === 0) par = gnorm / dxnorm
  for (let iter = 1; ; iter++) {
    if (par === 0) par = Math.max(DWARF, 0.001 * paru)
    y = yOf(par)
    dxnorm = dnrm2(y)
    const temp = fp
    fp = dxnorm - delta
    if (Math.abs(fp) <= 0.1 * delta || (parl === 0 && fp <= temp && temp < 0) || iter === 10) break
    const parc = fp / (delta * curvature(y, dxnorm, par))
    if (fp > 0) parl = Math.max(parl, par)
    if (fp < 0) paru = Math.min(paru, par)
    par = Math.max(parl, par + parc)
  }
  return { par, p: unscale(y) }
}

const columnNorms = (J: Float64Array, m: number, n: number): Float64Array => {
  const out = new Float64Array(n)
  for (let j = 0; j < n; j++) {
    let s = 0
    for (let i = 0; i < m; i++) s += (J[i * n + j] ?? 0) ** 2
    out[j] = Math.sqrt(s)
  }
  return out
}

/**
 * The Levenberg-Marquardt method as a traced algorithm, following MINPACK's lmder (scipy's least_squares with
 * method 'lm' and leastsq): a trust region on the scaled step |D p| whose Levenberg-Marquardt parameter is found by
 * lmpar's safeguarded Newton iteration, with MINPACK's radius updates and its ftol, xtol and gtol tests. Each step is
 * one Jacobian evaluation and the trial steps it takes.
 *
 * @param r - The residual function (at least as many residuals as parameters).
 * @param x0 - The start.
 * @param options - Variable scaling, factor, tolerances (ftol, xtol, gtol: default 1e-8) and the evaluation cap.
 * @returns The algorithm.
 * @throws WhiteboxError with code `shape` when there are fewer residuals than parameters.
 * @concept least-squares
 * @reference J. J. More, The Levenberg-Marquardt algorithm: implementation and theory, Numerical Analysis (1978).
 * @example
 * ```ts
 * import { levenbergMarquardtAlgorithm } from 'whitebox-engine/optim'
 * import { sub, type Value } from 'whitebox-engine/tensor'
 * levenbergMarquardtAlgorithm((p: Value) => sub(p, 1), [0]).name // => 'levenbergMarquardt'
 * ```
 */
export function levenbergMarquardtAlgorithm(
  r: ResidualFunction,
  x0: Parameters,
  options: LevenbergMarquardtOptions = {},
): Algorithm<LevenbergMarquardtState> {
  return lmOn(flatResiduals(r, x0), options)
}

function lmOn(res: FlatResiduals, options: LevenbergMarquardtOptions): Algorithm<LevenbergMarquardtState> {
  requireOverdetermined(res, 'levenbergMarquardt')
  const { m, n } = res
  const ftol = options.ftol ?? 1e-8
  const xtol = options.xtol ?? 1e-8
  const gtol = options.gradTol ?? 1e-8
  const factor = options.factor ?? 100
  const maxfev = options.maxEvaluations ?? 100 * n
  const xScale = options.xScale ?? 'jac'
  const auto = xScale === 'jac'
  const eps = Number.EPSILON
  const cost = (f: Float64Array): number => 0.5 * ddot(f, f)
  return {
    name: 'levenbergMarquardt',
    config: configOf(options),
    schema: vectorSchema({
      metrics: {
        ...SCHEMA_EXTRA.metrics,
        delta: { description: 'trust radius on the scaled step' },
        par: { description: 'Levenberg-Marquardt parameter' },
        ratio: { description: 'actual over predicted reduction of the last trial' },
      },
    }),
    init: () => {
      const x = Float64Array.from(res.x0)
      const f = res.residuals(x)
      const J = res.jacobian(x)
      const norms = columnNorms(J, m, n)
      const diag = auto
        ? norms.map((c) => (c === 0 ? 1 : c))
        : typeof xScale === 'number'
          ? new Float64Array(n).fill(1 / xScale)
          : Float64Array.from(xScale as ArrayLike<number>, (v) => 1 / v)
      const xnorm = dnrm2(x.map((xi, j) => xi * (diag[j] ?? 1)))
      const delta = xnorm === 0 ? factor : factor * xnorm
      return {
        x,
        value: cost(f),
        grad: jtv(J, m, n, f),
        nfev: 1,
        ngev: 1,
        nhev: 0,
        njev: 1,
        step: 0,
        residuals: f,
        jacobian: J,
        diag,
        delta,
        par: 0,
        ratio: Number.NaN,
        iter: 1,
      }
    },
    step: (s) => {
      const J = s.jacobian
      let f = s.residuals
      let fnorm = dnrm2(f)
      const norms = columnNorms(J, m, n)
      const g = jtv(J, m, n, f)
      let gnorm = 0
      if (fnorm !== 0)
        for (let j = 0; j < n; j++)
          if ((norms[j] ?? 0) !== 0) gnorm = Math.max(gnorm, Math.abs((g[j] ?? 0) / fnorm) / (norms[j] ?? 1))
      if (gnorm <= gtol)
        return { ...s, stop: { status: 'converged', message: 'the cosine of the gradient angle is below gtol' } }
      const diag = auto ? s.diag.map((d, j) => Math.max(d, norms[j] ?? 0)) : s.diag
      let { delta, par, iter, nfev } = s
      let x = s.x
      let xnorm = dnrm2(x.map((xi, j) => xi * (diag[j] ?? 1)))
      let ratio = 0
      let stepNorm = 0
      for (;;) {
        const lp = lmpar(J, m, n, f, diag, delta, par)
        par = lp.par
        const p = lp.p
        const xNew = daxpy(1, p, x)
        const pnorm = dnrm2(p.map((pi, j) => pi * (diag[j] ?? 1)))
        if (iter === 1) delta = Math.min(delta, pnorm)
        const fNew = res.residuals(xNew)
        nfev += 1
        const fnorm1 = dnrm2(fNew)
        const actred = 0.1 * fnorm1 < fnorm ? 1 - (fnorm1 / fnorm) ** 2 : -1
        const temp1 = dnrm2(jv(J, m, n, p)) / fnorm
        const temp2 = (Math.sqrt(par) * pnorm) / fnorm
        const prered = temp1 * temp1 + (temp2 * temp2) / 0.5
        const dirder = -(temp1 * temp1 + temp2 * temp2)
        ratio = prered === 0 ? 0 : actred / prered
        if (ratio <= 0.25) {
          let temp = actred >= 0 ? 0.5 : (0.5 * dirder) / (dirder + 0.5 * actred)
          if (0.1 * fnorm1 >= fnorm || temp < 0.1) temp = 0.1
          delta = temp * Math.min(delta, pnorm / 0.1)
          par /= temp
        } else if (par === 0 || ratio >= 0.75) {
          delta = pnorm / 0.5
          par *= 0.5
        }
        const success = ratio >= 1e-4
        if (success) {
          x = xNew
          f = fNew
          xnorm = dnrm2(x.map((xi, j) => xi * (diag[j] ?? 1)))
          fnorm = fnorm1
          iter += 1
          stepNorm = dnrm2(p)
        }
        const ftolMet = Math.abs(actred) <= ftol && prered <= ftol && 0.5 * ratio <= 1
        const xtolMet = delta <= xtol * xnorm
        let stop: LevenbergMarquardtState['stop']
        if (ftolMet || xtolMet)
          stop = {
            status: 'converged',
            message: ftolMet && xtolMet ? 'ftol and xtol satisfied' : ftolMet ? 'ftol satisfied' : 'xtol satisfied',
          }
        else if (nfev >= maxfev) stop = { status: 'failed', message: 'maximum number of residual evaluations reached' }
        else if ((Math.abs(actred) <= eps && prered <= eps && 0.5 * ratio <= 1) || delta <= eps * xnorm || gnorm <= eps)
          stop = { status: 'converged', message: 'tolerances too small: no further improvement possible' }
        if (stop !== undefined || success) {
          const Jn = success ? res.jacobian(x) : J
          return {
            x,
            value: cost(f),
            grad: jtv(Jn, m, n, f),
            nfev,
            ngev: s.ngev + (success ? 1 : 0),
            nhev: 0,
            njev: s.njev + (success ? 1 : 0),
            step: stepNorm,
            residuals: f,
            jacobian: Jn,
            diag,
            delta,
            par,
            ratio,
            iter,
            ...(stop === undefined ? {} : { stop }),
          }
        }
      }
    },
    done: vectorDone,
    observe: (s) => observeVector(s, { metrics: { cost: s.value, delta: s.delta, par: s.par, ratio: s.ratio } }),
  }
}

/**
 * Solve a nonlinear least-squares problem by Levenberg-Marquardt (MINPACK's lmder, scipy's method 'lm').
 *
 * @param r - The residual function.
 * @param x0 - The start.
 * @param options - Scaling, factor, tolerances, the evaluation cap (default 100 n), iteration cap and run options.
 * @returns The result with the cost, residuals and Jacobian.
 * @concept least-squares
 * @example
 * ```ts
 * import { levenbergMarquardt } from 'whitebox-engine/optim'
 * import { mul, sub, tensor, toFlat, type Value } from 'whitebox-engine/tensor'
 * // Fit y = a x through (1, 2), (2, 4), (3, 6).
 * const r = levenbergMarquardt((a: Value) => sub(mul(a, tensor([1, 2, 3])), tensor([2, 4, 6])), [0])
 * toFlat(r.x as Value).map((v) => Math.round(v * 1e8) / 1e8) // => [2]
 * ```
 */
export function levenbergMarquardt(
  r: ResidualFunction,
  x0: Parameters,
  options: LevenbergMarquardtOptions = {},
): LeastSquaresResult {
  const res = flatResiduals(r, x0)
  return runLeastSquares(lmOn(res, options), res, options, 100 * res.n)
}

/** A least-squares method as a function value, for {@link leastSquares}. */
export type LeastSquaresMethod = (r: ResidualFunction, x0: Parameters, options?: object) => LeastSquaresResult

/** Options of {@link leastSquares}. */
export type LeastSquaresSolveOptions = LevenbergMarquardtOptions &
  GaussNewtonOptions &
  DogboxOptions & {
    /** The method (default Levenberg-Marquardt, or the bounded dogbox method when bounds or a robust loss are given). */
    readonly method?: LeastSquaresMethod
  }

/**
 * Solve a nonlinear least-squares problem, scipy.optimize.least_squares style: minimise 0.5 sum rho(f_i^2) subject to
 * bounds. The method is a function value: {@link levenbergMarquardt} (default without bounds), {@link dogbox}
 * (default with bounds or a robust loss) or {@link gaussNewton}.
 *
 * @param r - The residual function.
 * @param x0 - The start.
 * @param options - The method and its options (bounds, loss, scaling, tolerances).
 * @returns The result.
 * @concept least-squares
 * @example
 * ```ts
 * import { leastSquares } from 'whitebox-engine/optim'
 * import { sub, type Value } from 'whitebox-engine/tensor'
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
  const m =
    method ??
    (options.bounds !== undefined || (options.loss !== undefined && options.loss !== 'linear')
      ? (dogbox as LeastSquaresMethod)
      : (levenbergMarquardt as LeastSquaresMethod))
  return m(r, x0, rest)
}

/** Options of {@link curveFit}. */
export interface CurveFitOptions extends Omit<LeastSquaresSolveOptions, 'loss' | 'fScale'> {
  /** Uncertainties of ydata (a scalar or one per point), or their covariance matrix. */
  readonly sigma?: TensorLike
  /** Treat sigma as absolute (pcov not rescaled by the residual variance; default false). */
  readonly absoluteSigma?: boolean
  /** A robust loss (uses the bounded dogbox method). */
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
 * @throws WhiteboxError with code `convergence` when the fit fails.
 * @concept least-squares
 * @example
 * ```ts
 * import { curveFit } from 'whitebox-engine/optim'
 * import { add, mul, sliceAxis, toFlat, type Value } from 'whitebox-engine/tensor'
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
  const result = leastSquares(r, p0, rest)
  if (!result.converged && result.status !== 'completed')
    throw new WhiteboxError('convergence', `curveFit: optimal parameters not found: ${result.message}`)
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
