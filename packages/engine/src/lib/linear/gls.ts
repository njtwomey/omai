/**
 * Generalised least squares, for errors that are not independent with a common variance. `gls` is the closed form for a
 * known error covariance (statsmodels' `GLS`): the data whitened by the inverse square root of the covariance, then
 * ordinary least squares. `glsar` estimates the covariance of autoregressive errors by alternating a whitened fit with
 * Yule-Walker estimates of the AR coefficients from its residuals (statsmodels' `GLSAR.iterative_fit`, the
 * Cochrane-Orcutt family). `heteroscedasticRegressor` models the variance as `exp(z gamma)`: `fgls` iterates a weighted
 * fit with a regression of the log squared residuals on `z` (feasible GLS; one round is the textbook estimator, and
 * further rounds are not guaranteed to converge), and `ml` maximises the joint Gaussian
 * likelihood of the mean and variance coefficients over the solver slot (Harvey's multiplicative heteroscedasticity).
 */

import { type Json, MdmlError } from '../base/index.ts'
import type { CoefficientTable, Exposition } from '../exposition/index.ts'
import {
  defineExplainedEstimator,
  type ExplainedEstimator,
  type Fitted,
  type IterativeEstimator,
  iterative,
  type LearnData,
} from '../learn/index.ts'
import { dgemv, dpotrf } from '../numerics/linalg/index.ts'
import { analyticObjective, type SmoothProblem, smoothSolvers } from '../optim/index.ts'
import type { Key } from '../random/index.ts'
import { choice, dict, int, type SpaceValue } from '../space/index.ts'
import { type Algorithm, channels, Log, stateFromJson, stateToJson } from '../trace/index.ts'
import { inverseOf, nested } from './_cov.ts'
import { infNorm, newtonAt, type SlotState, slotAlgorithm } from './_slot.ts'
import { type AffineVerbs, affineVerbs } from './_verbs.ts'
import { type Design, designOf, withIntercept } from './design.ts'
import { waldTable } from './inference.ts'
import { leastSquaresSolve } from './solvers.ts'

// ---------------------------------------------------------------------------------------------------------------------
// Shared: a least-squares fit of a transformed problem

/** The result of a least-squares fit of already-transformed data. */
interface Transformed {
  readonly beta: Float64Array
  /** `(A^T A)^-1` of the transformed design. */
  readonly bread: Float64Array
  /** The sum of squared transformed residuals. */
  readonly ssr: number
  readonly rows: number
}

/** Ordinary least squares of `y` on the columns of `a` (an intercept, if any, is one of them). */
function leastSquares(a: Float64Array, y: Float64Array, n: number, p: number): Transformed {
  const design: Design = { x: { data: a, n, d: p }, y, w: new Float64Array(n).fill(1) }
  const fit = leastSquaresSolve(design, 0, false, 'qr')
  const gram = new Float64Array(p * p)
  for (let i = 0; i < n; i++)
    for (let j = 0; j < p; j++) {
      const aj = a[i * p + j] ?? 0
      for (let k = 0; k < p; k++) gram[j * p + k] = (gram[j * p + k] ?? 0) + aj * (a[i * p + k] ?? 0)
    }
  const fitted = dgemv(a, n, p, fit.coef)
  let ssr = 0
  for (let i = 0; i < n; i++) ssr += ((y[i] ?? 0) - (fitted[i] ?? 0)) ** 2
  return { beta: fit.coef, bread: inverseOf(gram, p), ssr, rows: n }
}

/** The Gaussian-regression summary every generalised fit reports. */
export type GlsParams = {
  readonly coef: number[]
  readonly intercept: number
  /** The error variance scale: the whitened sum of squares over `n - p`. */
  readonly scale: number
  /** The covariance of the slopes then the intercept, `[p][p]`. */
  readonly covariance: number[][]
  readonly logLikelihood: number
  /** The residual degrees of freedom: the rows used (all of them, but for an AR fit) minus the coefficients. */
  readonly dfResid: number
}

function summary(fit: Transformed, n: number, p: number, fitIntercept: boolean, logDet: number): GlsParams {
  // The residual degrees of freedom count the rows the fit used (an AR fit drops its first p rows).
  const dfResid = fit.rows - p
  const scale = fit.ssr / dfResid
  const cov = Float64Array.from(fit.bread, (v) => v * scale)
  const d = p - (fitIntercept ? 1 : 0)
  // statsmodels' concentrated Gaussian log-likelihood of the whitened data, less half the log determinant.
  const nobs2 = n / 2
  const llf = -Math.log(fit.ssr) * nobs2 - (1 + Math.log(Math.PI / nobs2)) * nobs2 - 0.5 * logDet
  return {
    coef: Array.from(fit.beta.slice(0, d)),
    intercept: fitIntercept ? (fit.beta[d] ?? 0) : 0,
    scale,
    covariance: nested(cov, p),
    logLikelihood: llf,
    dfResid,
  }
}

function tableOf(par: GlsParams, fitIntercept: boolean): CoefficientTable {
  const p = par.covariance.length
  const names = [...par.coef.map((_, j) => `x${j}`), ...(fitIntercept ? ['intercept'] : [])]
  const cov = Float64Array.from(
    { length: p * p },
    (_, idx) => par.covariance[Math.floor(idx / p)]?.[idx % p] ?? Number.NaN,
  )
  return waldTable(names, [...par.coef, ...(fitIntercept ? [par.intercept] : [])], cov, { df: par.dfResid })
}

// ---------------------------------------------------------------------------------------------------------------------
// GLS with a known covariance

/**
 * The covariance matrix of a stationary AR(1) process up to the innovation variance: `rho^|i - j|` (what statsmodels'
 * GLS documentation builds with `toeplitz(rho ** arange(n))`).
 *
 * @param n - The number of observations.
 * @param rho - The autocorrelation, in (-1, 1).
 * @returns The `[n][n]` matrix.
 * @example
 * ```ts
 * import { ar1Covariance } from 'mdml-engine/linear'
 * ar1Covariance(3, 0.5) // => [[1, 0.5, 0.25], [0.5, 1, 0.5], [0.25, 0.5, 1]]
 * ```
 */
export function ar1Covariance(n: number, rho: number): number[][] {
  return Array.from({ length: n }, (_, i) => Array.from({ length: n }, (_, j) => rho ** Math.abs(i - j)))
}

/** The hyperparameters of {@link gls}. */
export interface GlsHyper {
  /**
   * The error covariance up to a scale: a number (the same variance for every row), the per-row variances (a vector of
   * length `n`), or the full `[n][n]` matrix. Default 1: ordinary least squares.
   */
  readonly sigma: number | readonly number[] | readonly (readonly number[])[]
  readonly fitIntercept: boolean
}

/** What a fitted generalised least-squares model offers. */
export type GlsVerbs = AffineVerbs

/** The covariance as a full inverse and the log of its determinant. */
function precisionOf(
  sigma: GlsHyper['sigma'],
  n: number,
  who: string,
): { inverse: Float64Array | null; diag: Float64Array | null; logDet: number } {
  if (typeof sigma === 'number') {
    if (!(sigma > 0)) throw new MdmlError('value', `${who}: the variance must be positive, got ${sigma}`)
    return { inverse: null, diag: new Float64Array(n).fill(sigma), logDet: n * Math.log(sigma) }
  }
  if (sigma.length !== n) throw new MdmlError('shape', `${who}: sigma has ${sigma.length} rows for ${n} observations`)
  const first = sigma[0]
  if (typeof first === 'number') {
    const diag = Float64Array.from(sigma as readonly number[])
    if (diag.some((v) => !(v > 0))) throw new MdmlError('value', `${who}: the variances must be positive`)
    return { inverse: null, diag, logDet: diag.reduce((a, v) => a + Math.log(v), 0) }
  }
  const matrix = new Float64Array(n * n)
  for (let i = 0; i < n; i++) {
    const row = (sigma as readonly (readonly number[])[])[i]
    if (row === undefined || row.length !== n) throw new MdmlError('shape', `${who}: sigma must be ${n} x ${n}`)
    for (let j = 0; j < n; j++) matrix[i * n + j] = row[j] ?? 0
  }
  const chol = dpotrf(matrix, n)
  if (chol.info !== 0) throw new MdmlError('value', `${who}: sigma is not positive definite`)
  let logDet = 0
  for (let i = 0; i < n; i++) logDet += 2 * Math.log(chol.factor[i * n + i] ?? 1)
  return { inverse: inverseOf(matrix, n), diag: null, logDet }
}

/**
 * Generalised least squares for a known error covariance: minimise `(y - X b)' S^-1 (y - X b)`.
 *
 * @param data - Features and targets (the sample weights are ignored: the covariance is the weighting).
 * @param hyper - The covariance `sigma` and the intercept.
 * @returns The coefficients, scale, covariance of the estimates and Gaussian log-likelihood.
 * @throws MdmlError `shape` for a covariance of the wrong size, `value` for one that is not positive definite.
 * @example
 * ```ts
 * import { glsFit } from 'mdml-engine/linear'
 * const x = [[0], [1], [2], [3]]
 * const f = glsFit({ x, y: [1, 3, 5, 7] }, { sigma: [1, 1, 1, 1] })
 * f.coef // => [2]
 * ```
 */
export function glsFit(data: LearnData, hyper: Partial<GlsHyper> = {}): GlsParams {
  const fitIntercept = hyper.fitIntercept ?? true
  const design = designOf(data, 'gls')
  const { n } = design.x
  const { a, p } = withIntercept(design, fitIntercept)
  if (n <= p) throw new MdmlError('value', 'gls: needs more observations than coefficients')
  const prec = precisionOf(hyper.sigma ?? 1, n, 'gls')
  const aw = new Float64Array(n * p)
  const yw = new Float64Array(n)
  if (prec.diag !== null) {
    for (let i = 0; i < n; i++) {
      const r = 1 / Math.sqrt(prec.diag[i] ?? 1)
      yw[i] = r * (design.y[i] ?? 0)
      for (let j = 0; j < p; j++) aw[i * p + j] = r * (a[i * p + j] ?? 0)
    }
  } else if (prec.inverse !== null) {
    // Whiten with the Cholesky factor of the precision: W'W = S^-1.
    const chol = dpotrf(prec.inverse, n)
    if (chol.info !== 0) throw new MdmlError('value', 'gls: sigma is not positive definite')
    const u = chol.factor
    // W = U (upper): rows of W times the data.
    for (let i = 0; i < n; i++) {
      for (let k = i; k < n; k++) {
        const uik = u[i * n + k] ?? 0
        yw[i] = (yw[i] ?? 0) + uik * (design.y[k] ?? 0)
        for (let j = 0; j < p; j++) aw[i * p + j] = (aw[i * p + j] ?? 0) + uik * (a[k * p + j] ?? 0)
      }
    }
  }
  return summary(leastSquares(aw, yw, n, p), n, p, fitIntercept, prec.logDet)
}

/**
 * Generalised least squares with a known error covariance (statsmodels' `GLS`): the closed form, whitening the data by
 * the inverse square root of `sigma` and solving ordinary least squares. With `sigma` a number it is ordinary least
 * squares; with a vector, weighted least squares with weights `1 / sigma`; with an `[n][n]` matrix (an AR(1)
 * structure from {@link ar1Covariance}, say), the full GLS. The covariance describes the training rows, so `predict`
 * is the mean `x beta + b` only. The exposition is the coefficient table with t statistics on `n - p` degrees of
 * freedom.
 *
 * @param hyper - `sigma`, `fitIntercept`.
 * @returns The estimator.
 * @reference A. C. Aitken, On least squares and linear combination of observations, Proc. Roy. Soc. Edinburgh 55 (1935).
 * @example
 * ```ts
 * import { ar1Covariance, gls } from 'mdml-engine/linear'
 * const x = [[0], [1], [2], [3], [4], [5]]
 * const y = [0.1, 1.2, 1.8, 3.3, 3.9, 5.2]
 * const f = gls({ sigma: ar1Covariance(6, 0.5) }).fit({ x, y })
 * Math.round((f.params.coef[0] ?? 0) * 100) / 100 // => 1.01
 * ```
 */
export function gls(hyper: Partial<GlsHyper> = {}): ExplainedEstimator<GlsHyper, Fitted<GlsParams> & GlsVerbs> {
  return defineExplainedEstimator(
    {
      name: 'gls',
      kind: 'regressor',
      fit: (data, h) => {
        const params = glsFit(data, h)
        return { params, exposition: [tableOf(params, h.fitIntercept)] }
      },
      methods: (p) => affineVerbs(p.coef, p.intercept, 'gls'),
    },
    { sigma: 1, fitIntercept: true, ...hyper },
  )
}

// ---------------------------------------------------------------------------------------------------------------------
// GLSAR: AR(p) errors

/** Options of {@link glsarAlgorithm}. */
export interface GlsarOptions {
  /** The order p of the autoregression (default 1). */
  readonly order?: number
  /** The starting AR coefficients (default zeros: the first fit is ordinary least squares). */
  readonly rho?: readonly number[]
  /** statsmodels' `maxiter`: the loop alternates `maxIter - 1` times, then fits once more (default 3). */
  readonly maxIter?: number
  /** Stop when the largest relative change of the coefficients is below this (default 1e-4). */
  readonly rtol?: number
  readonly fitIntercept?: boolean
}

/** The state of {@link glsarAlgorithm}. */
export type GlsarState = {
  /** The current AR coefficients. */
  readonly rho: number[]
  /** The coefficients of the last whitened fit (the slopes, then the intercept). */
  readonly coef: number[]
  readonly iteration: number
  /** The largest relative change of the coefficients in the last round (infinite on the first). */
  readonly change: number
  readonly converged: boolean
}

/** The data whitened for AR coefficients `rho`: `x_t - sum_i rho_i x_{t-i}`, the first `p` rows dropped. */
function whitenAr(m: Float64Array, n: number, cols: number, rho: readonly number[]): Float64Array {
  const p = rho.length
  const out = new Float64Array((n - p) * cols)
  for (let t = p; t < n; t++)
    for (let j = 0; j < cols; j++) {
      let v = m[t * cols + j] ?? 0
      for (let i = 0; i < p; i++) v -= (rho[i] ?? 0) * (m[(t - i - 1) * cols + j] ?? 0)
      out[(t - p) * cols + j] = v
    }
  return out
}

/** statsmodels' `yule_walker` (adjusted, demeaned): the AR coefficients of a series. */
function yuleWalker(x: ArrayLike<number>, order: number): number[] {
  const n = x.length
  let mean = 0
  for (let i = 0; i < n; i++) mean += (x[i] ?? 0) / n
  const c = Array.from(x, (v) => v - mean)
  const r = new Array<number>(order + 1).fill(0)
  r[0] = c.reduce((a, v) => a + v * v, 0) / n
  for (let k = 1; k <= order; k++) {
    let s = 0
    for (let i = 0; i + k < n; i++) s += (c[i] ?? 0) * (c[i + k] ?? 0)
    r[k] = s / (n - k)
  }
  const toeplitz = new Float64Array(order * order)
  for (let i = 0; i < order; i++) for (let j = 0; j < order; j++) toeplitz[i * order + j] = r[Math.abs(i - j)] ?? 0
  const inv = inverseOf(toeplitz, order)
  return Array.from({ length: order }, (_, i) => {
    let s = 0
    for (let j = 0; j < order; j++) s += (inv[i * order + j] ?? 0) * (r[j + 1] ?? 0)
    return s
  })
}

/** The whitened fit at `rho` and the residuals of the original data under its coefficients. */
function glsarFitAt(
  a: Float64Array,
  y: Float64Array,
  n: number,
  p: number,
  rho: readonly number[],
): { fit: Transformed; residuals: Float64Array } {
  const aw = whitenAr(a, n, p, rho)
  const yw = whitenAr(y, n, 1, rho)
  const fit = leastSquares(aw, yw, n - rho.length, p)
  const fitted = dgemv(a, n, p, fit.beta)
  const residuals = Float64Array.from(y, (v, i) => v - (fitted[i] ?? 0))
  return { fit, residuals }
}

/**
 * Regression with AR(p) errors as a traced algorithm: each step is one round of statsmodels'
 * `GLSAR.iterative_fit` (a whitened fit at the current AR coefficients, then new coefficients from the Yule-Walker
 * equations of its residuals), ending when the coefficients stop changing or after `maxIter - 1` rounds. The
 * estimator then fits once more at the final AR coefficients.
 *
 * @param data - Features and targets, in time order.
 * @param options - The order, the starting coefficients, the iteration limits and the intercept.
 * @returns The algorithm.
 * @example
 * ```ts
 * import { glsarAlgorithm } from 'mdml-engine/linear'
 * import { runAlgorithm } from 'mdml-engine/trace'
 * const x = Array.from({ length: 12 }, (_, i) => [i])
 * const y = x.map((r, i) => 1 + 2 * (r[0] ?? 0) + Math.sin(i))
 * runAlgorithm(glsarAlgorithm({ x, y }), { maxSteps: 5 }).state.rho.length // => 1
 * ```
 */
export function glsarAlgorithm(data: LearnData, options: GlsarOptions = {}): Algorithm<GlsarState> {
  const order = options.order ?? 1
  const rtol = options.rtol ?? 1e-4
  const fitIntercept = options.fitIntercept ?? true
  const maxIter = options.maxIter ?? 3
  if (!Number.isInteger(order) || order < 1)
    throw new MdmlError('value', `glsar: the order must be a positive integer, got ${order}`)
  const design = designOf(data, 'glsar')
  const { n } = design.x
  const { a, p } = withIntercept(design, fitIntercept)
  if (n - order <= p) throw new MdmlError('value', 'glsar: needs more observations than coefficients and the order')
  const start = options.rho === undefined ? new Array<number>(order).fill(0) : [...options.rho]
  if (start.length !== order)
    throw new MdmlError('shape', `glsar: ${start.length} starting coefficients for order ${order}`)
  return {
    name: 'glsar',
    config: { order, rtol, maxIter, fitIntercept } as Json,
    schema: channels({
      metrics: { change: { description: 'the largest relative change of the coefficients', direction: 'lower' } },
      tensors: {
        rho: { role: 'param', description: 'the AR coefficients', shape: [order] },
        parameters: { role: 'param', description: 'the slopes then the intercept', shape: [p] },
      },
    }),
    init: () => ({ rho: start, coef: [], iteration: 0, change: Number.POSITIVE_INFINITY, converged: false }),
    step: (s) => {
      const { fit, residuals } = glsarFitAt(a, design.y, n, p, s.rho)
      const coef = Array.from(fit.beta)
      let change = Number.POSITIVE_INFINITY
      if (s.iteration > 0) {
        change = Math.max(...coef.map((v, j) => Math.abs((s.coef[j] ?? 0) - v) / Math.abs(s.coef[j] ?? 0)))
        if (change < rtol) return { ...s, coef, change, converged: true, iteration: s.iteration + 1 }
      }
      return { rho: yuleWalker(residuals, order), coef, iteration: s.iteration + 1, change, converged: false }
    },
    done: (s) => (s.converged || s.iteration >= Math.max(maxIter - 1, 0) ? 'converged' : undefined),
    observe: (s) => ({
      metrics: { change: Number.isFinite(s.change) ? s.change : 0 },
      tensors: {
        rho: { role: 'param', value: s.rho, shape: [order] },
        parameters: { role: 'param', value: s.coef.length === p ? s.coef : new Array<number>(p).fill(0), shape: [p] },
      },
    }),
    serialise: (s) => stateToJson(s) as Json,
    deserialise: (j) => stateFromJson(j) as GlsarState,
  }
}

/** The hyperparameters of {@link glsar}. */
export interface GlsarHyper {
  readonly order: number
  /** The starting AR coefficients (default zeros). */
  readonly rho: readonly number[] | null
  readonly maxIter: number
  readonly rtol: number
  readonly fitIntercept: boolean
}

/** The fitted parameters of {@link glsar}. */
export type GlsarParams = GlsParams & {
  /** The AR coefficients of the errors. */
  readonly rho: number[]
  readonly iterations: number
  readonly converged: boolean
}

const GLSAR_DEFAULTS: GlsarHyper = { order: 1, rho: null, maxIter: 3, rtol: 1e-4, fitIntercept: true }

/**
 * Regression with autoregressive errors (statsmodels' `GLSAR` with `iterative_fit`): the errors follow
 * `u_t = rho_1 u_{t-1} + ... + rho_p u_{t-p} + e_t`, and the coefficients and `rho` are estimated by alternating a
 * whitened least-squares fit (which drops the first `p` rows) with the Yule-Walker equations of its residuals. The
 * residual degrees of freedom are `n - p_coefficients`, as statsmodels reports them. Rows must be in time order.
 *
 * @param hyper - `order`, `rho` (the start), `maxIter` (statsmodels' `maxiter`, default 3), `rtol`, `fitIntercept`.
 * @returns The estimator.
 * @reference D. Cochrane and G. H. Orcutt, Application of least squares regression to relationships containing
 * auto-correlated error terms, JASA 44 (1949).
 * @example
 * ```ts
 * import { glsar } from 'mdml-engine/linear'
 * const x = Array.from({ length: 30 }, (_, i) => [i / 10])
 * const e: number[] = []
 * const y = x.map((r, i) => {
 *   e[i] = 0.7 * (e[i - 1] ?? 0) + Math.sin(i * 12.9898)
 *   return 1 + 2 * (r[0] ?? 0) + (e[i] ?? 0)
 * })
 * glsar({ order: 1, maxIter: 6 }).fit({ x, y }).params.rho.length // => 1
 * ```
 */
export function glsar(
  hyper: Partial<GlsarHyper> = {},
): IterativeEstimator<GlsarHyper, GlsarState, GlsarParams, GlsVerbs> {
  const options = (h: GlsarHyper): GlsarOptions => ({
    order: h.order,
    ...(h.rho === null ? {} : { rho: h.rho }),
    maxIter: h.maxIter,
    rtol: h.rtol,
    fitIntercept: h.fitIntercept,
  })
  const paramsOf = (s: GlsarState, data: LearnData, h: GlsarHyper): GlsarParams => {
    const design = designOf(data, 'glsar')
    const { n } = design.x
    const { a, p } = withIntercept(design, h.fitIntercept)
    const { fit } = glsarFitAt(a, design.y, n, p, s.rho)
    const base = summary(fit, n, p, h.fitIntercept, 0)
    return { ...base, rho: s.rho, iterations: s.iteration, converged: s.converged }
  }
  return iterative<GlsarHyper, GlsarState, GlsarParams, GlsVerbs>(
    {
      name: 'glsar',
      kind: 'regressor',
      defaults: GLSAR_DEFAULTS,
      hyperSpace: (h) => dict({ order: int(1, 6, { default: h.order }) }),
      algorithm: (data, h) => glsarAlgorithm(data, options(h)),
      maxSteps: (h) => Math.max(h.maxIter, 1),
      params: paramsOf,
      verbs: (par) => affineVerbs(par.coef, par.intercept, 'glsar'),
      records: (s, data, h): readonly Exposition[] => [tableOf(paramsOf(s, data, h), h.fitIntercept)],
      logs: {
        'solver/change': Log(),
        'solver/rho': Log({ detail: 'full' }),
        'solver/parameters': Log({ detail: 'full' }),
      },
    },
    { ...GLSAR_DEFAULTS, ...hyper },
  )
}

// ---------------------------------------------------------------------------------------------------------------------
// Heteroscedastic regression: variance exp(z gamma)

/** How the variance coefficients are found. */
export type HeteroscedasticMethod = 'fgls' | 'ml'

/** The hyperparameters of {@link heteroscedasticRegressor}. */
export interface HeteroscedasticHyper {
  /** Feasible GLS (iterated log-residual regression) or maximum likelihood over the solver slot. */
  readonly method: HeteroscedasticMethod
  /** The features whose log-variance is modelled (a constant is always included): `'all'` or column indices. */
  readonly varianceFeatures: 'all' | readonly number[]
  readonly fitIntercept: boolean
  /** The tolerance: the largest change of the coefficients (fgls) or the gradient (ml). */
  readonly tol: number
  readonly maxIter: number
  /** The solver of the `ml` method: a value of `smoothSolvers.space` (default Newton). */
  readonly solver: SpaceValue
}

/** The mean design, the variance design (a constant last) and the targets of a heteroscedastic fit. */
interface HeteroProblem {
  readonly n: number
  readonly p: number
  readonly q: number
  readonly a: Float64Array
  readonly z: Float64Array
  readonly y: Float64Array
  readonly fitIntercept: boolean
}

function heteroProblemOf(
  data: LearnData,
  h: Pick<HeteroscedasticHyper, 'varianceFeatures' | 'fitIntercept'>,
): HeteroProblem {
  const design = designOf(data, 'heteroscedasticRegressor')
  const { n, d } = design.x
  const { a, p } = withIntercept(design, h.fitIntercept)
  const cols = h.varianceFeatures === 'all' ? Array.from({ length: d }, (_, j) => j) : [...h.varianceFeatures]
  for (const j of cols)
    if (!Number.isInteger(j) || j < 0 || j >= d)
      throw new MdmlError('value', `heteroscedasticRegressor: variance feature ${j} is not a column`)
  const q = cols.length + 1
  const z = new Float64Array(n * q)
  for (let i = 0; i < n; i++) {
    cols.forEach((j, k) => {
      z[i * q + k] = design.x.data[i * d + j] ?? 0
    })
    z[i * q + q - 1] = 1
  }
  if (n <= p + q) throw new MdmlError('value', 'heteroscedasticRegressor: needs more observations than coefficients')
  return { n, p, q, a, z, y: design.y, fitIntercept: h.fitIntercept }
}

/** The variance-weighted fit with row variances `exp(z gamma)`. */
function weightedFit(
  problem: HeteroProblem,
  gamma: ArrayLike<number>,
): Transformed & { readonly weights: Float64Array } {
  const { n, p, q, a, z, y } = problem
  const logVar = dgemv(z, n, q, gamma)
  const weights = Float64Array.from(logVar, (v) => Math.exp(-v))
  const aw = new Float64Array(n * p)
  const yw = new Float64Array(n)
  for (let i = 0; i < n; i++) {
    const r = Math.sqrt(weights[i] ?? 1)
    yw[i] = r * (y[i] ?? 0)
    for (let j = 0; j < p; j++) aw[i * p + j] = r * (a[i * p + j] ?? 0)
  }
  return { ...leastSquares(aw, yw, n, p), weights }
}

/** The state of the feasible-GLS algorithm. */
export type FglsState = {
  /** The mean coefficients (slopes, then the intercept) of the last weighted fit. */
  readonly beta: number[]
  /** The variance coefficients (the features', then the constant). */
  readonly gamma: number[]
  readonly iteration: number
  readonly change: number
}

/**
 * Feasible GLS with multiplicative heteroscedasticity as a traced algorithm: each step fits the mean by weighted least
 * squares with weights `exp(-z gamma)` (ordinary least squares at the start), then re-estimates `gamma` by regressing the
 * log squared residuals on `z` (the regression of Wooldridge's textbook FGLS). It logs the largest change of the mean
 * coefficients and the tensors `parameters` and `gamma`.
 *
 * @param problem - The design.
 * @param tol - Stop when the mean coefficients change by less than this (default 1e-10).
 * @returns The algorithm.
 */
function fglsAlgorithm(problem: HeteroProblem, tol: number): Algorithm<FglsState> {
  const { n, p, q, z, y, a } = problem
  return {
    name: 'fgls',
    config: { tol } as Json,
    schema: channels({
      metrics: { change: { description: 'the largest change of the mean coefficients', direction: 'lower' } },
      tensors: {
        parameters: { role: 'param', description: 'the slopes then the intercept', shape: [p] },
        gamma: { role: 'param', description: 'the log-variance coefficients', shape: [q] },
      },
    }),
    init: () => ({
      beta: new Array<number>(p).fill(0),
      gamma: new Array<number>(q).fill(0),
      iteration: 0,
      change: Number.POSITIVE_INFINITY,
    }),
    step: (s) => {
      const fit = weightedFit(problem, s.gamma)
      const beta = Array.from(fit.beta)
      const fitted = dgemv(a, n, p, beta)
      const logSq = Float64Array.from(y, (v, i) => Math.log(Math.max((v - (fitted[i] ?? 0)) ** 2, 1e-300)))
      const reg = leastSquares(z, logSq, n, q)
      const change =
        s.iteration === 0 ? Number.POSITIVE_INFINITY : Math.max(...beta.map((v, j) => Math.abs(v - (s.beta[j] ?? 0))))
      return { beta, gamma: Array.from(reg.beta), iteration: s.iteration + 1, change }
    },
    done: (s) => (s.change <= tol ? 'converged' : undefined),
    observe: (s) => ({
      metrics: { change: Number.isFinite(s.change) ? s.change : 0 },
      tensors: {
        parameters: { role: 'param', value: s.beta, shape: [p] },
        gamma: { role: 'param', value: s.gamma, shape: [q] },
      },
    }),
    serialise: (s) => stateToJson(s) as Json,
    deserialise: (j) => stateFromJson(j) as FglsState,
  }
}

/**
 * The negative Gaussian log-likelihood per observation of a mean `a beta` with variance `exp(z gamma)` (the constants
 * dropped), its gradient and Hessian, in `theta = (beta, gamma)`.
 *
 * @param problem - The design.
 * @param theta - The `p + q` parameters.
 * @returns The value, gradient and row-major Hessian.
 */
function heteroObjective(
  problem: HeteroProblem,
  theta: ArrayLike<number>,
): { readonly value: number; readonly gradient: Float64Array; readonly hessian: Float64Array } {
  const { n, p, q, a, z, y } = problem
  const m = p + q
  const beta = Array.from({ length: p }, (_, j) => theta[j] ?? 0)
  const gamma = Array.from({ length: q }, (_, j) => theta[p + j] ?? 0)
  const mean = dgemv(a, n, p, beta)
  const logVar = dgemv(z, n, q, gamma)
  let value = 0
  const gradient = new Float64Array(m)
  const hessian = new Float64Array(m * m)
  for (let i = 0; i < n; i++) {
    const r = (y[i] ?? 0) - (mean[i] ?? 0)
    const inv = Math.exp(-(logVar[i] ?? 0))
    const r2inv = r * r * inv
    value += (0.5 * ((logVar[i] ?? 0) + r2inv)) / n
    for (let j = 0; j < p; j++) {
      const aj = a[i * p + j] ?? 0
      gradient[j] = (gradient[j] ?? 0) - (r * inv * aj) / n
      for (let k = 0; k < p; k++) hessian[j * m + k] = (hessian[j * m + k] ?? 0) + (inv * aj * (a[i * p + k] ?? 0)) / n
      for (let k = 0; k < q; k++)
        hessian[j * m + p + k] = (hessian[j * m + p + k] ?? 0) + (r * inv * aj * (z[i * q + k] ?? 0)) / n
    }
    for (let j = 0; j < q; j++) {
      const zj = z[i * q + j] ?? 0
      gradient[p + j] = (gradient[p + j] ?? 0) + (0.5 * (1 - r2inv) * zj) / n
      for (let k = 0; k < q; k++)
        hessian[(p + j) * m + p + k] =
          (hessian[(p + j) * m + p + k] ?? 0) + (0.5 * r2inv * zj * (z[i * q + k] ?? 0)) / n
    }
  }
  for (let j = 0; j < p; j++) for (let k = 0; k < q; k++) hessian[(p + k) * m + j] = hessian[j * m + p + k] ?? 0
  return { value, gradient, hessian }
}

/** The state of the maximum-likelihood heteroscedastic fit. */
export type HeteroMlState = SlotState & {
  readonly theta: number[]
  readonly objective: number
  readonly gradient: number
  readonly evaluations: number
  readonly stalled: boolean
}

/**
 * The heteroscedastic Gaussian likelihood as a smooth problem for the solver slot, started at the least-squares fit
 * with a constant variance.
 */
function heteroSmoothProblem(problem: HeteroProblem): SmoothProblem {
  const { n, p, q, a, y } = problem
  const ols = leastSquares(a, y, n, p)
  const fitted = dgemv(a, n, p, ols.beta)
  const s2 = Math.max(Array.from(y).reduce((acc, v, i) => acc + (v - (fitted[i] ?? 0)) ** 2, 0) / n, 1e-12)
  const x0 = new Array<number>(p + q).fill(0)
  for (let j = 0; j < p; j++) x0[j] = ols.beta[j] ?? 0
  x0[p + q - 1] = Math.log(s2)
  return {
    objective: analyticObjective({
      value: (t) => heteroObjective(problem, t).value,
      gradient: (t) => heteroObjective(problem, t).gradient,
      hessian: (t) => heteroObjective(problem, t).hessian,
    }),
    x0,
  }
}

function heteroMlAlgorithm(
  problem: HeteroProblem,
  tol: number,
  solver: SpaceValue,
  key: Key,
): Algorithm<HeteroMlState> {
  const m = problem.p + problem.q
  return slotAlgorithm<HeteroMlState>({
    name: 'heteroscedastic-ml',
    config: { tol, solver } as Json,
    schema: {
      metrics: {
        objective: {
          description: 'the negative log-likelihood per observation, constants dropped',
          direction: 'lower',
        },
        gradient: { description: 'the largest gradient entry', direction: 'lower' },
      },
      tensors: {
        parameters: { role: 'param', description: 'the mean then the log-variance coefficients', shape: [m] },
      },
    },
    problem: heteroSmoothProblem(problem),
    solver,
    key,
    tol,
    view: (r, evaluations, stalled) => ({
      theta: Array.from(r.x),
      objective: r.value,
      gradient: infNorm(r.grad),
      evaluations,
      stalled,
    }),
    observe: (s) => ({
      metrics: { objective: s.objective, gradient: s.gradient },
      tensors: { parameters: { role: 'param', value: s.theta, shape: [m] } },
    }),
  })
}

/** The state of {@link heteroscedasticRegressor}: that of the method chosen. */
export type HeteroscedasticState = FglsState | HeteroMlState

/** The fitted parameters of {@link heteroscedasticRegressor}. */
export type HeteroscedasticParams = GlsParams & {
  readonly method: HeteroscedasticMethod
  /** The log-variance coefficients: the modelled features', then the constant. */
  readonly gamma: number[]
  readonly varianceFeatures: number[]
  readonly iterations: number
  readonly converged: boolean
}

const HETERO_DEFAULTS: HeteroscedasticHyper = {
  method: 'fgls',
  varianceFeatures: 'all',
  fitIntercept: true,
  tol: 1e-10,
  maxIter: 500,
  solver: newtonAt(1e-10),
}

const isMl = (s: HeteroscedasticState): s is HeteroMlState => 'solver' in s

/**
 * Regression with a modelled error variance `Var(e_i) = exp(z_i gamma)` (Harvey's multiplicative heteroscedasticity),
 * where `z` is a constant and a choice of the features. `method: 'fgls'` is feasible GLS: iterate a weighted
 * least-squares fit with an OLS regression of the log squared residuals on `z`. `method: 'ml'` maximises the joint
 * Gaussian likelihood of the mean and variance coefficients; it is the solver slot (Newton by default, or any of
 * `smoothSolvers`). The coefficient table's errors come from the weighted fit at the final variances.
 *
 * @param hyper - `method`, `varianceFeatures`, `fitIntercept`, `tol`, `maxIter`, `solver`.
 * @returns The estimator.
 * @reference W. H. Greene, Econometric Analysis, ch. 9 (multiplicative heteroscedasticity); A. C. Harvey, Estimating
 * regression models with multiplicative heteroscedasticity, Econometrica 44 (1976).
 * @example
 * ```ts
 * import { heteroscedasticRegressor } from 'mdml-engine/linear'
 * const x = Array.from({ length: 40 }, (_, i) => [i / 8])
 * const y = x.map((r, i) => 1 + 2 * (r[0] ?? 0) + (0.2 + 0.4 * (r[0] ?? 0)) * Math.sin(i * 12.9898))
 * const f = heteroscedasticRegressor({ method: 'fgls' }).fit({ x, y })
 * const slope = f.params.gamma[0] ?? 0
 * slope > 0 // => true
 * ```
 */
export function heteroscedasticRegressor(
  hyper: Partial<HeteroscedasticHyper> = {},
): IterativeEstimator<HeteroscedasticHyper, HeteroscedasticState, HeteroscedasticParams, GlsVerbs> {
  const paramsOf = (s: HeteroscedasticState, data: LearnData, h: HeteroscedasticHyper): HeteroscedasticParams => {
    const problem = heteroProblemOf(data, h)
    const { n, p, q } = problem
    const gamma: number[] = isMl(s) ? s.theta.slice(p) : s.gamma
    const fit = weightedFit(problem, gamma)
    const beta: number[] = isMl(s) ? s.theta.slice(0, p) : Array.from(fit.beta)
    const base = summary({ ...fit, beta: Float64Array.from(beta) }, n, p, h.fitIntercept, 0)
    // The joint Gaussian log-likelihood at (beta, gamma).
    const mean = dgemv(problem.a, n, p, beta)
    const logVar = dgemv(problem.z, n, q, gamma)
    let llf = 0
    for (let i = 0; i < n; i++)
      llf -=
        0.5 *
        (Math.log(2 * Math.PI) +
          (logVar[i] ?? 0) +
          ((problem.y[i] ?? 0) - (mean[i] ?? 0)) ** 2 * Math.exp(-(logVar[i] ?? 0)))
    const cols =
      h.varianceFeatures === 'all'
        ? Array.from({ length: p - (h.fitIntercept ? 1 : 0) }, (_, j) => j)
        : [...h.varianceFeatures]
    return {
      ...base,
      logLikelihood: llf,
      coef: beta.slice(0, p - (h.fitIntercept ? 1 : 0)),
      intercept: h.fitIntercept ? (beta[p - 1] ?? 0) : 0,
      method: h.method,
      gamma: [...gamma],
      varianceFeatures: cols,
      iterations: isMl(s) ? s.evaluations : s.iteration,
      converged: isMl(s) ? s.gradient <= h.tol : s.change <= h.tol,
    }
  }
  return iterative<HeteroscedasticHyper, HeteroscedasticState, HeteroscedasticParams, GlsVerbs>(
    {
      name: 'heteroscedastic-regressor',
      kind: 'regressor',
      defaults: HETERO_DEFAULTS,
      hyperSpace: (h) => dict({ method: choice(['fgls', 'ml'], { default: h.method }), solver: smoothSolvers.space }),
      algorithm: (data, h, key): Algorithm<HeteroscedasticState> => {
        const problem = heteroProblemOf(data, h)
        return h.method === 'ml' ? heteroMlAlgorithm(problem, h.tol, h.solver, key) : fglsAlgorithm(problem, h.tol)
      },
      maxSteps: (h) => h.maxIter,
      params: paramsOf,
      verbs: (par) => affineVerbs(par.coef, par.intercept, 'heteroscedasticRegressor'),
      records: (s, data, h): readonly Exposition[] => [tableOf(paramsOf(s, data, h), h.fitIntercept)],
      logs: { 'solver/parameters': Log({ detail: 'full' }) },
    },
    { ...HETERO_DEFAULTS, ...hyper },
  )
}
