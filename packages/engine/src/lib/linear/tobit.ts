/**
 * Censored regression (the Tobit model of Tobin): a latent `y* = x beta + sigma eps` with Gaussian noise is observed
 * only between a lower and an upper limit, and an observation at a limit is censored, so the likelihood has a density
 * term for each uncensored row and a tail probability for each censored one. The likelihood is maximised in Olsen's
 * parametrisation `(beta / sigma, 1 / sigma)`, in which it is concave, so every solver of the slot has one optimum to
 * find; `tobitProblem` states it (value, gradient and Hessian), `tobitAlgorithm` runs any solver of `smoothSolvers` on
 * it (Newton by default), and `tobit` is the estimator, with Wald standard errors of `(beta, sigma)` by the delta
 * method from the observed information.
 */

import { type Json, OmaiError } from '../base/index.ts'
import type { CoefficientTable, Exposition } from '../exposition/index.ts'
import { type IterativeEstimator, iterative, type LearnData } from '../learn/index.ts'
import { logNdtr, ndtr, normalLogPdf, normalPdf } from '../numerics/special/index.ts'
import { analyticObjective, type SmoothProblem, smoothSolvers } from '../optim/index.ts'
import type { Key } from '../random/index.ts'
import { dict, real, type SpaceValue } from '../space/index.ts'
import { primalOf, type Tensor, type TensorLike, toFlat } from '../tensor/index.ts'
import { type Algorithm, Log } from '../trace/index.ts'
import { inverseOf, nested } from './_cov.ts'
import { infNorm, newtonAt, type SlotState, slotAlgorithm } from './_slot.ts'
import { type AffineVerbs, affineVerbs } from './_verbs.ts'
import { type Design, designOf, linearPredictor, vectorTensor, withIntercept } from './design.ts'
import { waldTable } from './inference.ts'
import { leastSquaresSolve } from './solvers.ts'

/** A censored regression problem. */
export interface TobitProblem {
  readonly n: number
  /** The columns of `a`, the intercept included (last). */
  readonly p: number
  /** The slopes, without the intercept. */
  readonly d: number
  readonly design: Design
  readonly a: Float64Array
  readonly y: Float64Array
  readonly w: Float64Array
  /** Per row: 0 observed, -1 censored at the lower limit, 1 censored at the upper limit. */
  readonly censoring: Int8Array
  readonly lower: number
  readonly upper: number
  readonly fitIntercept: boolean
}

/**
 * Read a censored regression problem: a row whose target is at or below `lower` is left-censored, one at or above
 * `upper` right-censored, the others observed.
 *
 * @param data - Features, targets and sample weights.
 * @param options - The limits (default none) and the intercept (default true).
 * @param who - The caller, for errors.
 * @returns The problem.
 * @throws OmaiError `value` when the lower limit is not below the upper one.
 * @example
 * ```ts
 * import { tobitProblemOf } from 'omai-engine/linear'
 * const problem = tobitProblemOf({ x: [[0], [1], [2]], y: [0, 0, 1.5] }, { lower: 0 }, 'demo')
 * Array.from(problem.censoring) // => [-1, -1, 0]
 * ```
 */
export function tobitProblemOf(
  data: LearnData,
  options: { readonly lower?: number | null; readonly upper?: number | null; readonly fitIntercept?: boolean },
  who: string,
): TobitProblem {
  const lower = options.lower ?? Number.NEGATIVE_INFINITY
  const upper = options.upper ?? Number.POSITIVE_INFINITY
  const fitIntercept = options.fitIntercept ?? true
  if (!(lower < upper))
    throw new OmaiError('value', `${who}: the lower limit ${lower} must be below the upper ${upper}`)
  const design = designOf(data, who)
  const { a, p } = withIntercept(design, fitIntercept)
  const censoring = new Int8Array(design.x.n)
  for (let i = 0; i < design.x.n; i++) {
    const yi = design.y[i] ?? 0
    censoring[i] = yi <= lower ? -1 : yi >= upper ? 1 : 0
  }
  return { n: design.x.n, p, d: design.x.d, design, a, y: design.y, w: design.w, censoring, lower, upper, fitIntercept }
}

/**
 * The mean negative log-likelihood of a censored regression in Olsen's parameters `theta = (delta, gamma)` with
 * `delta = beta / sigma` (the intercept last) and `gamma = 1 / sigma`, its gradient and its Hessian. It is `+Infinity`
 * where `gamma <= 0`.
 *
 * @param problem - The problem.
 * @param theta - The parameters, `p + 1` of them.
 * @returns The value per unit weight, the gradient and the row-major Hessian.
 */
export function tobitObjective(
  problem: TobitProblem,
  theta: ArrayLike<number>,
): { readonly value: number; readonly gradient: Float64Array; readonly hessian: Float64Array } {
  const { n, p, a, y, w, censoring, lower, upper } = problem
  const m = p + 1
  const gamma = theta[p] ?? 0
  const gradient = new Float64Array(m)
  const hessian = new Float64Array(m * m)
  if (!(gamma > 0)) return { value: Number.POSITIVE_INFINITY, gradient, hessian }
  const total = w.reduce((x, v) => x + v, 0)
  const eta = new Float64Array(n)
  const u = new Float64Array(n)
  for (let i = 0; i < n; i++) {
    let e = 0
    for (let j = 0; j < p; j++) e += (a[i * p + j] ?? 0) * (theta[j] ?? 0)
    eta[i] = e
    const c = censoring[i] ?? 0
    u[i] = c === 0 ? gamma * (y[i] ?? 0) - e : c < 0 ? gamma * lower - e : e - gamma * upper
  }
  const logTail = new Float64Array(n)
  const logPdf = new Float64Array(n)
  if (censoring.some((c) => c !== 0)) {
    logTail.set(toFlat(primalOf(logNdtr(vectorTensor(u)))))
    logPdf.set(toFlat(primalOf(normalLogPdf(vectorTensor(u)))))
  }
  let value = 0
  for (let i = 0; i < n; i++) {
    const wi = (w[i] ?? 0) / total
    const c = censoring[i] ?? 0
    const ui = u[i] ?? 0
    let gEta: number
    let gGamma: number
    let hEta: number
    let hCross: number
    let hGamma: number
    if (c === 0) {
      const yi = y[i] ?? 0
      value += wi * (-Math.log(gamma) + 0.5 * Math.log(2 * Math.PI) + 0.5 * ui * ui)
      gEta = ui
      gGamma = 1 / gamma - ui * yi
      hEta = -1
      hCross = yi
      hGamma = -1 / (gamma * gamma) - yi * yi
    } else {
      const lambda = Math.exp((logPdf[i] ?? 0) - (logTail[i] ?? 0))
      const curvature = -lambda * (ui + lambda)
      const s = c < 0 ? -1 : 1
      const cu = c < 0 ? lower : -upper
      value += -wi * (logTail[i] ?? 0)
      gEta = s * lambda
      gGamma = cu * lambda
      hEta = curvature
      hCross = s * cu * curvature
      hGamma = cu * cu * curvature
    }
    for (let j = 0; j < p; j++) {
      const aj = a[i * p + j] ?? 0
      gradient[j] = (gradient[j] ?? 0) - wi * gEta * aj
      hessian[j * m + p] = (hessian[j * m + p] ?? 0) - wi * hCross * aj
      for (let k = 0; k <= j; k++) hessian[j * m + k] = (hessian[j * m + k] ?? 0) - wi * hEta * aj * (a[i * p + k] ?? 0)
    }
    gradient[p] = (gradient[p] ?? 0) - wi * gGamma
    hessian[p * m + p] = (hessian[p * m + p] ?? 0) - wi * hGamma
  }
  for (let j = 0; j < p; j++) {
    hessian[p * m + j] = hessian[j * m + p] ?? 0
    for (let k = j + 1; k < p; k++) hessian[j * m + k] = hessian[k * m + j] ?? 0
  }
  return { value, gradient, hessian }
}

/**
 * The censored-regression likelihood as a smooth problem for the solver slot, started at the least-squares fit that
 * ignores the censoring.
 *
 * @param problem - The problem.
 * @returns The smooth problem in Olsen's parameters.
 * @example
 * ```ts
 * import { tobitProblemOf, tobitSmoothProblem } from 'omai-engine/linear'
 * const problem = tobitProblemOf({ x: [[0], [1], [2], [3]], y: [0, 0.4, 1.1, 2.2] }, { lower: 0 }, 'demo')
 * tobitSmoothProblem(problem).x0.length // => 3
 * ```
 */
export function tobitSmoothProblem(problem: TobitProblem): SmoothProblem {
  const { n, d, p } = problem
  const ols = leastSquaresSolve(problem.design, 0, problem.fitIntercept, 'cholesky')
  let rss = 0
  let sw = 0
  for (let i = 0; i < n; i++) {
    let e = problem.fitIntercept ? ols.intercept : 0
    for (let j = 0; j < d; j++) e += (problem.a[i * p + j] ?? 0) * (ols.coef[j] ?? 0)
    rss += (problem.w[i] ?? 0) * ((problem.y[i] ?? 0) - e) ** 2
    sw += problem.w[i] ?? 0
  }
  const gamma = 1 / Math.max(Math.sqrt(rss / sw), 1e-8)
  const x0 = new Array<number>(p + 1).fill(0)
  for (let j = 0; j < d; j++) x0[j] = (ols.coef[j] ?? 0) * gamma
  if (problem.fitIntercept) x0[d] = ols.intercept * gamma
  x0[p] = gamma
  return {
    objective: analyticObjective({
      value: (t) => tobitObjective(problem, t).value,
      gradient: (t) => tobitObjective(problem, t).gradient,
      hessian: (t) => tobitObjective(problem, t).hessian,
    }),
    x0,
  }
}

/** Options of {@link tobitAlgorithm}. */
export interface TobitOptions {
  /** Stop when the largest gradient entry falls below this (default 1e-8). */
  readonly tol?: number
  /** The solver, a value of `smoothSolvers.space` (default Newton at `tol`). */
  readonly solver?: SpaceValue
  /** The run's key (for the stochastic solver's batches). */
  readonly key?: Key
}

/** The state of {@link tobitAlgorithm}. */
export type TobitState = SlotState & {
  /** Olsen's parameters: `beta / sigma` (the intercept last), then `1 / sigma`. */
  readonly theta: number[]
  /** The mean negative log-likelihood. */
  readonly objective: number
  readonly gradient: number
  readonly evaluations: number
  readonly stalled: boolean
}

/**
 * Censored regression as a traced algorithm over the solver slot. It logs `objective`, `gradient` and the tensor
 * `parameters` (Olsen's parameters).
 *
 * @param problem - The data and the limits.
 * @param options - The tolerance, the solver and the key.
 * @returns The algorithm.
 * @example
 * ```ts
 * import { tobitAlgorithm, tobitProblemOf } from 'omai-engine/linear'
 * import { runAlgorithm } from 'omai-engine/trace'
 * const x = [[0], [1], [2], [3], [4], [5]]
 * const problem = tobitProblemOf({ x, y: [0, 0, 0.6, 1.9, 2.4, 3.8] }, { lower: 0 }, 'demo')
 * runAlgorithm(tobitAlgorithm(problem), { maxSteps: 100 }).status // => 'converged'
 * ```
 */
export function tobitAlgorithm(problem: TobitProblem, options: TobitOptions = {}): Algorithm<TobitState> {
  const tol = options.tol ?? 1e-8
  const solver = options.solver ?? newtonAt(tol)
  return slotAlgorithm<TobitState>({
    name: 'tobit',
    config: {
      lower: Number.isFinite(problem.lower) ? problem.lower : null,
      upper: Number.isFinite(problem.upper) ? problem.upper : null,
      tol,
      solver,
    } as Json,
    schema: {
      metrics: {
        objective: { description: 'the mean negative log-likelihood', direction: 'lower' },
        gradient: { description: 'the largest gradient entry', direction: 'lower' },
      },
      tensors: {
        parameters: {
          role: 'param',
          description: 'beta / sigma (intercept last), then 1 / sigma',
          shape: [problem.p + 1],
        },
      },
    },
    problem: tobitSmoothProblem(problem),
    solver,
    ...(options.key === undefined ? {} : { key: options.key }),
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
      tensors: { parameters: { role: 'param', value: s.theta, shape: [problem.p + 1] } },
    }),
  })
}

/** The hyperparameters of {@link tobit}. */
export interface TobitHyper {
  /** Observations at or below this are left-censored (default `-Infinity`: none). */
  readonly lower: number
  /** Observations at or above this are right-censored (default `Infinity`: none). */
  readonly upper: number
  readonly fitIntercept: boolean
  readonly tol: number
  readonly maxIter: number
  /** How the likelihood is maximised: a value of `smoothSolvers.space` (default Newton). */
  readonly solver: SpaceValue
}

/** The fitted parameters of {@link tobit}. */
export type TobitParams = {
  readonly coef: number[]
  readonly intercept: number
  /** The standard deviation of the latent noise. */
  readonly sigma: number
  /** The log-likelihood at the maximum (the weighted sum). */
  readonly logLikelihood: number
  readonly nLeftCensored: number
  readonly nRightCensored: number
  /** The limits, `null` where there is none. */
  readonly lower: number | null
  readonly upper: number | null
  /** The covariance of the slopes, the intercept and sigma (`[p + 1][p + 1]`), by the delta method. */
  readonly covariance: number[][]
  readonly iterations: number
  readonly converged: boolean
}

/** What a fitted censored regression offers. */
export interface TobitVerbs extends AffineVerbs {
  /** The expected observed target `E[clip(y*, lower, upper)]`, with the censoring applied. */
  readonly predictCensored: (x: TensorLike) => Tensor
}

const DEFAULTS: TobitHyper = {
  lower: Number.NEGATIVE_INFINITY,
  upper: Number.POSITIVE_INFINITY,
  fitIntercept: true,
  tol: 1e-8,
  maxIter: 200,
  solver: newtonAt(1e-8),
}

function paramsOf(problem: TobitProblem, h: TobitHyper, s: TobitState): TobitParams {
  const { p, d } = problem
  const m = p + 1
  const gamma = s.theta[p] ?? 1
  const total = problem.w.reduce((x, v) => x + v, 0)
  const { hessian } = tobitObjective(problem, s.theta)
  const info = Float64Array.from(hessian, (v) => v * total)
  const covOlsen = inverseOf(info, m)
  // The delta method from (delta, gamma) to (beta, sigma): beta = delta / gamma, sigma = 1 / gamma.
  const jac = new Float64Array(m * m)
  for (let j = 0; j < p; j++) {
    jac[j * m + j] = 1 / gamma
    jac[j * m + p] = -(s.theta[j] ?? 0) / (gamma * gamma)
  }
  jac[p * m + p] = -1 / (gamma * gamma)
  const cov = new Float64Array(m * m)
  const jc = new Float64Array(m * m)
  for (let i = 0; i < m; i++)
    for (let k = 0; k < m; k++) {
      const jik = jac[i * m + k] ?? 0
      if (jik !== 0) for (let j = 0; j < m; j++) jc[i * m + j] = (jc[i * m + j] ?? 0) + jik * (covOlsen[k * m + j] ?? 0)
    }
  for (let i = 0; i < m; i++)
    for (let j = 0; j < m; j++) {
      let v = 0
      for (let k = 0; k < m; k++) v += (jc[i * m + k] ?? 0) * (jac[j * m + k] ?? 0)
      cov[i * m + j] = v
    }
  let left = 0
  let right = 0
  for (const c of problem.censoring) {
    if (c < 0) left++
    else if (c > 0) right++
  }
  return {
    coef: s.theta.slice(0, d).map((v) => v / gamma),
    intercept: problem.fitIntercept ? (s.theta[d] ?? 0) / gamma : 0,
    sigma: 1 / gamma,
    logLikelihood: -s.objective * total,
    nLeftCensored: left,
    nRightCensored: right,
    lower: Number.isFinite(problem.lower) ? problem.lower : null,
    upper: Number.isFinite(problem.upper) ? problem.upper : null,
    covariance: nested(cov, m),
    iterations: s.evaluations,
    converged: s.gradient <= h.tol,
  }
}

function tableOf(par: TobitParams, fitIntercept: boolean): CoefficientTable {
  const names = [...par.coef.map((_, j) => `x${j}`), ...(fitIntercept ? ['intercept'] : []), 'sigma']
  const m = par.covariance.length
  // Without an intercept the covariance still has the (slopes, intercept slot, sigma) layout of the problem.
  const keep = names.map((_, k) => (fitIntercept || k < par.coef.length ? k : m - 1))
  const cov = Float64Array.from(
    { length: names.length * names.length },
    (_, idx) =>
      par.covariance[keep[Math.floor(idx / names.length)] ?? 0]?.[keep[idx % names.length] ?? 0] ?? Number.NaN,
  )
  return waldTable(names, [...par.coef, ...(fitIntercept ? [par.intercept] : []), par.sigma], cov)
}

/**
 * Censored (Tobit) regression: the latent `y* = x beta + b + sigma eps` with Gaussian noise, observed clipped to
 * `[lower, upper]`; rows at a limit are censored. `predict` is the latent mean `x beta + b` (the coefficients'
 * interpretation), `predictCensored` the expected observed value. The likelihood is maximised over the solver slot
 * (Newton by default; every solver of `smoothSolvers` reaches the same optimum, the problem being concave in Olsen's
 * parametrisation). Without censoring the fit is ordinary least squares with the maximum-likelihood `sigma`. The
 * exposition is the Wald table of the slopes, the intercept and sigma.
 *
 * @param hyper - `lower`, `upper`, `fitIntercept`, `tol`, `maxIter`, `solver`.
 * @returns The estimator.
 * @reference J. Tobin, Estimation of relationships for limited dependent variables, Econometrica 26 (1958);
 * R. Olsen, Note on the uniqueness of the maximum likelihood estimator for the Tobit model, Econometrica 46 (1978).
 * @example
 * ```ts
 * import { tobit } from 'omai-engine/linear'
 * const x = [[0], [1], [2], [3], [4], [5], [6], [7]]
 * const y = [0, 0, 0, 0.9, 1.7, 3.1, 3.9, 5.2]
 * const f = tobit({ lower: 0 }).fit({ x, y })
 * f.params.nLeftCensored // => 3
 * ```
 */
export function tobit(
  hyper: Partial<TobitHyper> = {},
): IterativeEstimator<TobitHyper, TobitState, TobitParams, TobitVerbs> {
  const who = 'tobit'
  return iterative<TobitHyper, TobitState, TobitParams, TobitVerbs>(
    {
      name: 'tobit',
      kind: 'regressor',
      defaults: DEFAULTS,
      hyperSpace: () =>
        dict({ tol: real(1e-12, 1e-3, { log: true, default: DEFAULTS.tol }), solver: smoothSolvers.space }),
      algorithm: (data, h, key) => tobitAlgorithm(tobitProblemOf(data, h, who), { tol: h.tol, solver: h.solver, key }),
      maxSteps: (h) => h.maxIter,
      params: (s, data, h) => paramsOf(tobitProblemOf(data, h, who), h, s),
      verbs: (par) => {
        const base = affineVerbs(par.coef, par.intercept, who)
        const lo = par.lower ?? Number.NEGATIVE_INFINITY
        const hi = par.upper ?? Number.POSITIVE_INFINITY
        return {
          ...base,
          predictCensored: (x: TensorLike): Tensor => {
            const eta = linearPredictor(x, par.coef, par.intercept, who)
            const lowZ = Float64Array.from(eta, (e) => (lo - e) / par.sigma)
            const highZ = Float64Array.from(eta, (e) => (hi - e) / par.sigma)
            const cdfLow = toFlat(primalOf(ndtr(vectorTensor(lowZ))))
            const cdfHigh = toFlat(primalOf(ndtr(vectorTensor(highZ))))
            const pdfLow = toFlat(primalOf(normalPdf(vectorTensor(lowZ))))
            const pdfHigh = toFlat(primalOf(normalPdf(vectorTensor(highZ))))
            // Limit times mass, zero where the limit is infinite (its mass is then zero too).
            const atLimit = (limit: number, mass: number): number => (Number.isFinite(limit) ? limit * mass : 0)
            return vectorTensor(
              Float64Array.from(
                eta,
                (e, i) =>
                  atLimit(lo, cdfLow[i] ?? 0) +
                  atLimit(hi, 1 - (cdfHigh[i] ?? 0)) +
                  e * ((cdfHigh[i] ?? 0) - (cdfLow[i] ?? 0)) +
                  par.sigma * ((pdfLow[i] ?? 0) - (pdfHigh[i] ?? 0)),
              ),
            )
          },
        }
      },
      records: (s, data, h): readonly Exposition[] => [
        tableOf(paramsOf(tobitProblemOf(data, h, who), h, s), h.fitIntercept),
      ],
      logs: { 'solver/objective': Log(), 'solver/gradient': Log(), 'solver/parameters': Log({ detail: 'full' }) },
    },
    { ...DEFAULTS, ...hyper },
  )
}
