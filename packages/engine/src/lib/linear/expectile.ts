/**
 * Expectile regression (Newey and Powell): the conditional expectile of the target as a linear function of the features.
 * The tau-expectile minimises the asymmetric squared loss `|tau - 1[r < 0]| r^2`, so tau = 1/2 is the mean and the
 * others weight over- and under-prediction differently; it is the least-squares counterpart of quantile regression and
 * a smooth objective. `expectileProblem` states that objective (value, gradient and the generalised Hessian) for the
 * solver slot, `expectileAlgorithm` runs any solver of `smoothSolvers` on it (Newton by default, which is iteratively
 * reweighted least squares), and `expectileRegressor` is the estimator, with the sandwich covariance of the estimates.
 */

import { type Json, OmaiError } from '../base/index.ts'
import type { CoefficientTable, Exposition } from '../exposition/index.ts'
import { type IterativeEstimator, iterative, type LearnData } from '../learn/index.ts'
import { analyticObjective, type SmoothProblem, smoothSolvers } from '../optim/index.ts'
import type { Key } from '../random/index.ts'
import { dict, real, type SpaceValue } from '../space/index.ts'
import { type Algorithm, Log } from '../trace/index.ts'
import { inverseOf, sandwich } from './_cov.ts'
import { infNorm, newtonAt, type SlotState, slotAlgorithm } from './_slot.ts'
import { type AffineVerbs, affineVerbs } from './_verbs.ts'
import { type Design, designOf, withIntercept } from './design.ts'
import { waldTable } from './inference.ts'

/** An expectile problem: the design (the intercept column last when fitted), targets, weights and the expectile level. */
export interface ExpectileProblem {
  readonly n: number
  /** The number of columns of `a`, the intercept included. */
  readonly p: number
  /** The slopes, without the intercept. */
  readonly d: number
  readonly a: Float64Array
  readonly y: Float64Array
  readonly w: Float64Array
  readonly tau: number
  /** The ridge penalty `alpha/2 |beta|^2` added to the mean asymmetric loss (the intercept is not penalised). */
  readonly alpha: number
  readonly fitIntercept: boolean
}

/**
 * Read an expectile problem from data.
 *
 * @param data - Features, targets and sample weights.
 * @param options - The expectile level (default 0.5), the ridge penalty (default 0) and the intercept (default true).
 * @param who - The caller, for errors.
 * @returns The problem.
 * @throws OmaiError `value` for a level outside (0, 1) or a negative penalty.
 * @example
 * ```ts
 * import { expectileProblemOf } from 'omai-engine/linear'
 * expectileProblemOf({ x: [[0], [1], [2]], y: [0, 1, 3] }, { tau: 0.8 }, 'demo').p // => 2
 * ```
 */
export function expectileProblemOf(
  data: LearnData,
  options: { readonly tau?: number; readonly alpha?: number; readonly fitIntercept?: boolean },
  who: string,
): ExpectileProblem {
  const tau = options.tau ?? 0.5
  const alpha = options.alpha ?? 0
  const fitIntercept = options.fitIntercept ?? true
  if (!(tau > 0 && tau < 1)) throw new OmaiError('value', `${who}: the expectile level must be in (0, 1), got ${tau}`)
  if (!(alpha >= 0)) throw new OmaiError('value', `${who}: the penalty must be non-negative, got ${alpha}`)
  const design: Design = designOf(data, who)
  const { a, p } = withIntercept(design, fitIntercept)
  return { n: design.x.n, p, d: design.x.d, a, y: design.y, w: design.w, tau, alpha, fitIntercept }
}

/** The residuals `y - A theta` and the asymmetry weight of each. */
function residualsOf(problem: ExpectileProblem, theta: ArrayLike<number>): { r: Float64Array; s: Float64Array } {
  const { n, p, a, y, tau } = problem
  const r = new Float64Array(n)
  const s = new Float64Array(n)
  for (let i = 0; i < n; i++) {
    let eta = 0
    for (let j = 0; j < p; j++) eta += (a[i * p + j] ?? 0) * (theta[j] ?? 0)
    const ri = (y[i] ?? 0) - eta
    r[i] = ri
    s[i] = ri >= 0 ? tau : 1 - tau
  }
  return { r, s }
}

/**
 * The mean asymmetric squared loss, its gradient and its generalised Hessian at `theta` (the slopes, then the
 * intercept when fitted).
 *
 * @param problem - The problem.
 * @param theta - The parameters.
 * @returns The objective per unit weight, the gradient and the row-major Hessian.
 */
export function expectileObjective(
  problem: ExpectileProblem,
  theta: ArrayLike<number>,
): { readonly value: number; readonly gradient: Float64Array; readonly hessian: Float64Array } {
  const { n, p, d, a, w, alpha } = problem
  const total = w.reduce((x, y) => x + y, 0)
  const { r, s } = residualsOf(problem, theta)
  let value = 0
  const gradient = new Float64Array(p)
  const hessian = new Float64Array(p * p)
  for (let i = 0; i < n; i++) {
    const wi = (w[i] ?? 0) / total
    const ri = r[i] ?? 0
    const si = s[i] ?? 0
    value += wi * si * ri * ri
    for (let j = 0; j < p; j++) {
      const aj = a[i * p + j] ?? 0
      gradient[j] = (gradient[j] ?? 0) - 2 * wi * si * ri * aj
      for (let k = 0; k <= j; k++)
        hessian[j * p + k] = (hessian[j * p + k] ?? 0) + 2 * wi * si * aj * (a[i * p + k] ?? 0)
    }
  }
  for (let j = 0; j < p; j++) for (let k = j + 1; k < p; k++) hessian[j * p + k] = hessian[k * p + j] ?? 0
  for (let j = 0; j < d; j++) {
    value += 0.5 * alpha * (theta[j] ?? 0) ** 2
    gradient[j] = (gradient[j] ?? 0) + alpha * (theta[j] ?? 0)
    hessian[j * p + j] = (hessian[j * p + j] ?? 0) + alpha
  }
  return { value, gradient, hessian }
}

/**
 * The expectile objective as a smooth problem for the solver slot: value, gradient and generalised Hessian, starting at
 * the weighted mean of the targets.
 *
 * @param problem - The data and the level.
 * @returns The smooth problem.
 * @example
 * ```ts
 * import { expectileProblemOf, expectileSmoothProblem } from 'omai-engine/linear'
 * const problem = expectileProblemOf({ x: [[0], [1], [2]], y: [0, 1, 3] }, { tau: 0.8 }, 'demo')
 * expectileSmoothProblem(problem).x0.length // => 2
 * ```
 */
export function expectileSmoothProblem(problem: ExpectileProblem): SmoothProblem {
  const total = problem.w.reduce((x, y) => x + y, 0)
  let mean = 0
  for (let i = 0; i < problem.n; i++) mean += ((problem.w[i] ?? 0) * (problem.y[i] ?? 0)) / total
  const x0 = new Array<number>(problem.p).fill(0)
  if (problem.fitIntercept) x0[problem.p - 1] = mean
  return {
    objective: analyticObjective({
      value: (t) => expectileObjective(problem, t).value,
      gradient: (t) => expectileObjective(problem, t).gradient,
      hessian: (t) => expectileObjective(problem, t).hessian,
    }),
    x0,
  }
}

/** Options of {@link expectileAlgorithm}. */
export interface ExpectileOptions {
  /** Stop when the largest gradient entry falls below this (default 1e-10). */
  readonly tol?: number
  /** The solver, a value of `smoothSolvers.space` (default Newton at `tol`). */
  readonly solver?: SpaceValue
  /** The run's key (for the stochastic solver's batches). */
  readonly key?: Key
}

/** The state of {@link expectileAlgorithm}. */
export type ExpectileState = SlotState & {
  /** The slopes, then the intercept when fitted. */
  readonly theta: number[]
  /** The mean asymmetric squared loss (with the penalty). */
  readonly objective: number
  readonly gradient: number
  readonly evaluations: number
  readonly stalled: boolean
}

/**
 * Expectile regression as a traced algorithm over the solver slot: the objective of {@link expectileSmoothProblem}
 * minimised by Newton (iteratively reweighted least squares) or any other solver of `smoothSolvers`. It logs
 * `objective`, `gradient` and the tensor `parameters`.
 *
 * @param problem - The data and the level.
 * @param options - The tolerance, the solver and the key.
 * @returns The algorithm.
 * @example
 * ```ts
 * import { expectileAlgorithm, expectileProblemOf } from 'omai-engine/linear'
 * import { runAlgorithm } from 'omai-engine/trace'
 * const problem = expectileProblemOf({ x: [[0], [1], [2], [3]], y: [0, 1.2, 1.9, 3.4] }, { tau: 0.9 }, 'demo')
 * runAlgorithm(expectileAlgorithm(problem), { maxSteps: 50 }).status // => 'converged'
 * ```
 */
export function expectileAlgorithm(
  problem: ExpectileProblem,
  options: ExpectileOptions = {},
): Algorithm<ExpectileState> {
  const tol = options.tol ?? 1e-10
  const solver = options.solver ?? newtonAt(tol)
  return slotAlgorithm<ExpectileState>({
    name: 'expectile-regression',
    config: { tau: problem.tau, alpha: problem.alpha, fitIntercept: problem.fitIntercept, tol, solver } as Json,
    schema: {
      metrics: {
        objective: { description: 'the mean asymmetric squared loss', direction: 'lower' },
        gradient: { description: 'the largest gradient entry', direction: 'lower' },
      },
      tensors: { parameters: { role: 'param', description: 'the slopes, then the intercept', shape: [problem.p] } },
    },
    problem: expectileSmoothProblem(problem),
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
      tensors: { parameters: { role: 'param', value: s.theta, shape: [problem.p] } },
    }),
  })
}

/** The hyperparameters of {@link expectileRegressor}. */
export interface ExpectileHyper {
  /** The expectile level in (0, 1); 0.5 is the mean (default). */
  readonly tau: number
  /** The ridge penalty `alpha/2 |beta|^2` on the mean asymmetric loss (default 0). */
  readonly alpha: number
  readonly fitIntercept: boolean
  readonly tol: number
  readonly maxIter: number
  /** How the loss is minimised: a value of `smoothSolvers.space` (default Newton, which is IRLS). */
  readonly solver: SpaceValue
}

/** The fitted parameters of {@link expectileRegressor}. */
export type ExpectileParams = {
  readonly coef: number[]
  readonly intercept: number
  readonly tau: number
  /** The mean asymmetric squared loss at the solution (the penalty included). */
  readonly objective: number
  /** The sandwich covariance of the slopes then the intercept, `[p][p]`. */
  readonly covariance: number[][]
  readonly iterations: number
  readonly converged: boolean
}

const DEFAULTS: ExpectileHyper = {
  tau: 0.5,
  alpha: 0,
  fitIntercept: true,
  tol: 1e-10,
  maxIter: 200,
  solver: newtonAt(1e-10),
}

function paramsOf(problem: ExpectileProblem, h: ExpectileHyper, s: ExpectileState): ExpectileParams {
  const { n, p, a, w } = problem
  const { r, s: asym } = residualsOf(problem, s.theta)
  // Bread: sum_i w_i s_i a_i a_i^T (+ n alpha / 2 on the slopes); meat: sum_i (w_i s_i r_i)^2 a_i a_i^T.
  const bread = new Float64Array(p * p)
  const meat = new Float64Array(p * p)
  for (let i = 0; i < n; i++) {
    const wi = w[i] ?? 0
    const bi = wi * (asym[i] ?? 0)
    const mi = (bi * (r[i] ?? 0)) ** 2
    for (let j = 0; j < p; j++)
      for (let k = 0; k < p; k++) {
        const prod = (a[i * p + j] ?? 0) * (a[i * p + k] ?? 0)
        bread[j * p + k] = (bread[j * p + k] ?? 0) + bi * prod
        meat[j * p + k] = (meat[j * p + k] ?? 0) + mi * prod
      }
  }
  const total = w.reduce((x, y) => x + y, 0)
  for (let j = 0; j < problem.d; j++) bread[j * p + j] = (bread[j * p + j] ?? 0) + 0.5 * h.alpha * total
  const inverse = inverseOf(bread, p)
  const cov = sandwich(inverse, meat, p)
  const covariance = Array.from({ length: p }, (_, i) =>
    Array.from({ length: p }, (_, j) => cov[i * p + j] ?? Number.NaN),
  )
  return {
    coef: s.theta.slice(0, problem.d),
    intercept: problem.fitIntercept ? (s.theta[problem.d] ?? 0) : 0,
    tau: problem.tau,
    objective: s.objective,
    covariance,
    iterations: s.evaluations,
    converged: s.gradient <= h.tol,
  }
}

function tableOf(par: ExpectileParams, fitIntercept: boolean): CoefficientTable {
  const p = par.covariance.length
  const names = [...par.coef.map((_, j) => `x${j}`), ...(fitIntercept ? ['intercept'] : [])]
  const cov = Float64Array.from(
    { length: p * p },
    (_, idx) => par.covariance[Math.floor(idx / p)]?.[idx % p] ?? Number.NaN,
  )
  return waldTable(names, [...par.coef, ...(fitIntercept ? [par.intercept] : [])], cov)
}

/**
 * Expectile regression: the tau-expectile of the target given the features as a linear function, the minimiser of the
 * mean asymmetric squared loss `|tau - 1[r < 0]| r^2` (plus an optional ridge penalty). At tau = 0.5 it is ordinary
 * least squares; as tau moves away from 0.5 it tracks the upper or lower part of the conditional distribution the way
 * a quantile does, but with a smooth, convex objective. The minimisation is the solver slot: Newton (IRLS) by default,
 * or any of `smoothSolvers`. The exposition is the coefficient table with the sandwich standard errors of Newey and
 * Powell.
 *
 * @param hyper - `tau`, `alpha`, `fitIntercept`, `tol`, `maxIter`, `solver`.
 * @returns The estimator.
 * @reference W. K. Newey and J. L. Powell, Asymmetric least squares estimation and testing, Econometrica 55 (1987).
 * @example
 * ```ts
 * import { expectileRegressor } from 'omai-engine/linear'
 * import { toFlat } from 'omai-engine/tensor'
 * const x = [[0], [1], [2], [3], [4], [5]]
 * const y = [0.1, 1.3, 1.8, 3.4, 3.7, 5.2]
 * const upper = toFlat(expectileRegressor({ tau: 0.9 }).fit({ x, y }).predict([[2.5]]))[0] ?? 0
 * const mean = toFlat(expectileRegressor({ tau: 0.5 }).fit({ x, y }).predict([[2.5]]))[0] ?? 0
 * upper > mean // => true
 * ```
 */
export function expectileRegressor(
  hyper: Partial<ExpectileHyper> = {},
): IterativeEstimator<ExpectileHyper, ExpectileState, ExpectileParams, AffineVerbs> {
  return iterative<ExpectileHyper, ExpectileState, ExpectileParams, AffineVerbs>(
    {
      name: 'expectile-regressor',
      kind: 'regressor',
      defaults: DEFAULTS,
      hyperSpace: (h) =>
        dict({
          tau: real(0.01, 0.99, { default: h.tau, label: 'expectile level' }),
          alpha: real(0, Math.max(10, h.alpha), { default: h.alpha, label: 'penalty strength' }),
          solver: smoothSolvers.space,
        }),
      algorithm: (data, h, key) =>
        expectileAlgorithm(expectileProblemOf(data, h, 'expectileRegressor'), { tol: h.tol, solver: h.solver, key }),
      maxSteps: (h) => h.maxIter,
      params: (s, data, h) => paramsOf(expectileProblemOf(data, h, 'expectileRegressor'), h, s),
      verbs: (par) => affineVerbs(par.coef, par.intercept, 'expectileRegressor'),
      records: (s, data, h): readonly Exposition[] => [
        tableOf(paramsOf(expectileProblemOf(data, h, 'expectileRegressor'), h, s), h.fitIntercept),
      ],
      logs: { 'solver/objective': Log(), 'solver/gradient': Log(), 'solver/parameters': Log({ detail: 'full' }) },
    },
    { ...DEFAULTS, ...hyper },
  )
}
