import { daxpy, ddot, dgesvd, dnrm2 } from '../../numerics/linalg/index.ts'
import type { Algorithm } from '../../trace/index.ts'
import { jtv, jv, requireOverdetermined, runLeastSquares } from './_least-squares.ts'
import { configOf, observeVector, vectorDone, vectorSchema } from './_shared.ts'
import type { LeastSquaresOptions, LeastSquaresResult, LeastSquaresState } from './least-squares.ts'
import { type FlatResiduals, flatResiduals, type Parameters, type ResidualFunction } from './objective.ts'

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
        cost: { description: 'half the sum of squared residuals' },
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
