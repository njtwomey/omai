/**
 * Cox's proportional hazards regression: the hazard of a subject is a baseline hazard times `exp(x beta)`, and `beta`
 * is fitted by maximising the partial likelihood, which ignores the baseline. The partial likelihood is built as a
 * smooth objective of `beta` (`coxProblem`: Breslow's or Efron's treatment of tied event times, optional strata and a
 * ridge penalty) and handed to the solver slot, so the fit is Newton by default and any other smooth solver
 * (L-BFGS, BFGS, conjugate gradient, the stochastic gradient mode) reaches the same optimum. `coxPh` adds the
 * covariance from the observed information, the Wald, likelihood-ratio and score tests, Breslow's baseline cumulative
 * hazard and predictions of survival.
 */

import { hessian, valueAndGrad } from '../autodiff/index.ts'
import { MdmlError, type TestResult } from '../base/index.ts'
import { loggedSolver, type SmoothProblem, smoothSolvers } from '../optim/index.ts'
import type { Key } from '../random/index.ts'
import { key as makeKey } from '../random/index.ts'
import type { SpaceValue } from '../space/index.ts'
import {
  add,
  asTensor,
  concat,
  cumsum,
  exp,
  fromStorage,
  log,
  matmul,
  max,
  mul,
  neg,
  square,
  stopGradient,
  sub,
  sum,
  type Tensor,
  type TensorLike,
  take,
  tensor,
  toFlat,
  type Value,
} from '../tensor/index.ts'
import { type RunOptions, runAlgorithm } from '../trace/index.ts'
import { chi2Sf, inverseSpd, normPpf, normSf, rowsOfMatrix } from './_util.ts'
import { concordanceIndex } from './concordance.ts'
import { type SurvivalSample, survivalSample } from './data.ts'

/** How tied event times are handled: Efron's approximation (default) or Breslow's. */
export type CoxTies = 'efron' | 'breslow'

/** Covariates with survival times: what the Cox model is fitted to. */
export interface CoxData {
  readonly sample: SurvivalSample
  /** The covariates, row-major `n x p`. */
  readonly x: Float64Array
  readonly p: number
  /** The stratum of every subject (all 0 when there are none). */
  readonly strata: Float64Array
  readonly names: readonly string[]
}

/**
 * Read the data of a Cox model.
 *
 * @param time - Observed times.
 * @param event - Event indicators (omitted: all events).
 * @param x - Covariates `[n, p]` (a vector is one covariate).
 * @param options - `strata` (a baseline hazard for each value) and the covariate `names`.
 * @returns The data.
 * @throws MdmlError with code `shape` when the lengths disagree.
 * @example
 * ```ts
 * import { coxData } from 'mdml-engine/survival'
 * coxData([1, 2, 3], [1, 0, 1], [[0.5], [1], [2]]).p // => 1
 * ```
 */
export function coxData(
  time: TensorLike,
  event: TensorLike | undefined,
  x: TensorLike,
  options: { readonly strata?: TensorLike; readonly names?: readonly string[] } = {},
): CoxData {
  const sample = survivalSample(time, event, 'coxData')
  const t = asTensor(x)
  const n = t.shape[0] ?? 0
  const p = t.shape.length === 1 ? 1 : (t.shape[1] ?? 0)
  if (n !== sample.n || p < 1) throw new MdmlError('shape', `coxData: ${n} covariate rows for ${sample.n} times`)
  const strata =
    options.strata === undefined ? new Float64Array(n) : Float64Array.from(toFlat(asTensor(options.strata)))
  if (strata.length !== n) throw new MdmlError('shape', `coxData: ${strata.length} strata for ${n} subjects`)
  const names = options.names ?? Array.from({ length: p }, (_, j) => `x${j}`)
  if (names.length !== p) throw new MdmlError('shape', `coxData: ${names.length} names for ${p} covariates`)
  return { sample, x: Float64Array.from(toFlat(t)), p, strata, names }
}

/** The subjects in (stratum, time) order and the index tables the partial likelihood is computed from. */
interface CoxPlan {
  readonly n: number
  readonly p: number
  /** Subject at each sorted position. */
  readonly order: Int32Array
  readonly x: Float64Array
  readonly event: Float64Array
  /** Per tied event-time group: first sorted position, one past the last, and one past the last of its stratum. */
  readonly groupStart: Int32Array
  readonly groupEnd: Int32Array
  readonly stratumEnd: Int32Array
  readonly groupEvents: Float64Array
  readonly groupTime: Float64Array
  readonly groupStratum: Float64Array
  /** Per event slot (a group's k-th event): its group and Efron's fraction `k / d`. */
  readonly slotGroup: Int32Array
  readonly slotFraction: Float64Array
}

function planOf(data: CoxData, ties: CoxTies): CoxPlan {
  const { sample, strata, p } = data
  const n = sample.n
  const order = Int32Array.from(
    Array.from({ length: n }, (_, i) => i).sort(
      (a, b) => (strata[a] ?? 0) - (strata[b] ?? 0) || (sample.time[a] ?? 0) - (sample.time[b] ?? 0),
    ),
  )
  const x = new Float64Array(n * p)
  const event = new Float64Array(n)
  for (let k = 0; k < n; k++) {
    const i = order[k] ?? 0
    event[k] = sample.event[i] ?? 0
    for (let j = 0; j < p; j++) x[k * p + j] = data.x[i * p + j] ?? 0
  }
  const stratumEndAt = new Int32Array(n)
  for (let k = n - 1; k >= 0; k--) {
    const next = k + 1 < n && (strata[order[k + 1] ?? 0] ?? 0) === (strata[order[k] ?? 0] ?? 0)
    stratumEndAt[k] = next ? (stratumEndAt[k + 1] ?? n) : k + 1
  }
  const gs: number[] = []
  const ge: number[] = []
  const se: number[] = []
  const gd: number[] = []
  const gt: number[] = []
  const gst: number[] = []
  const slotGroup: number[] = []
  const slotFraction: number[] = []
  let k = 0
  while (k < n) {
    let end = k
    let d = 0
    const t = sample.time[order[k] ?? 0] ?? 0
    const st = strata[order[k] ?? 0] ?? 0
    while (end < n && (sample.time[order[end] ?? 0] ?? 0) === t && (strata[order[end] ?? 0] ?? 0) === st) {
      d += event[end] ?? 0
      end++
    }
    if (d > 0) {
      const g = gs.length
      gs.push(k)
      ge.push(end)
      se.push(stratumEndAt[k] ?? n)
      gd.push(d)
      gt.push(t)
      gst.push(st)
      for (let s = 0; s < d; s++) {
        slotGroup.push(g)
        slotFraction.push(ties === 'efron' ? s / d : 0)
      }
    }
    k = end
  }
  return {
    n,
    p,
    order,
    x,
    event,
    groupStart: Int32Array.from(gs),
    groupEnd: Int32Array.from(ge),
    stratumEnd: Int32Array.from(se),
    groupEvents: Float64Array.from(gd),
    groupTime: Float64Array.from(gt),
    groupStratum: Float64Array.from(gst),
    slotGroup: Int32Array.from(slotGroup),
    slotFraction: Float64Array.from(slotFraction),
  }
}

/** The Cox partial likelihood as a smooth problem, with the views the fit and its inference need. */
export interface CoxProblem extends SmoothProblem {
  /** The number of coefficients. */
  readonly p: number
  /** The partial log-likelihood (without the ridge penalty) at `beta`. */
  logLikelihood(beta: ArrayLike<number>): number
  /** The score: the gradient of the partial log-likelihood at `beta`. */
  score(beta: ArrayLike<number>): number[]
  /** The observed information (minus the Hessian of the penalised partial log-likelihood), row-major `p x p`. */
  information(beta: ArrayLike<number>): Float64Array
}

/** Options of {@link coxProblem}. */
export interface CoxProblemOptions {
  /** Tied event times: `'efron'` (default) or `'breslow'`. */
  readonly ties?: CoxTies
  /** A ridge penalty: `ridge / 2 |beta|^2` is added to the objective, which is the mean negative partial log-likelihood. */
  readonly ridge?: number
}

/**
 * The Cox partial likelihood as a {@link SmoothProblem}: the objective is the negative partial log-likelihood per
 * subject plus `ridge / 2 |beta|^2`, written in differentiable tensor operations (a reverse cumulative sum gives every
 * risk set's total in one pass), so its gradient and Hessian come from autodiff and any smooth solver can minimise it.
 * With the log-likelihood `sum_i delta_i x_i beta - sum_slots log(R_t - (k / d_t) D_t)`, where `R_t` is the sum of
 * `exp(x beta)` over the risk set at `t`, `D_t` the sum over the `d_t` events tied at `t` and `k = 0 .. d_t - 1`
 * (Efron; Breslow takes `k = 0` throughout).
 *
 * @param data - The data.
 * @param options - `ties` and `ridge`.
 * @returns The problem, starting at `beta = 0`.
 * @example
 * ```ts
 * import { coxData, coxProblem } from 'mdml-engine/survival'
 * const prob = coxProblem(coxData([1, 2, 3, 4], [1, 1, 1, 0], [0, 1, 2, 3]), { ties: 'breslow' })
 * Number(prob.logLikelihood([0]).toFixed(4)) // => -3.1781
 * ```
 */
export function coxProblem(data: CoxData, options: CoxProblemOptions = {}): CoxProblem {
  const ties = options.ties ?? 'efron'
  const ridge = options.ridge ?? 0
  if (!(ridge >= 0)) throw new MdmlError('value', `coxProblem: ridge must be >= 0, got ${ridge}`)
  const plan = planOf(data, ties)
  const { n, p } = plan
  if (plan.slotGroup.length === 0) throw new MdmlError('value', 'coxProblem: needs at least one event')
  const X = fromStorage(plan.x, [n, p], 'float64')
  const delta = tensor(Array.from(plan.event))
  const zero = tensor([0])
  const riskStart = Array.from(plan.groupStart)
  const stratumEnd = Array.from(plan.stratumEnd)
  const groupStart = Array.from(plan.groupStart)
  const groupEnd = Array.from(plan.groupEnd)
  const slotGroup = Array.from(plan.slotGroup)
  const fraction = tensor(Array.from(plan.slotFraction))
  const logLik = (b: Value): Value => {
    const eta = matmul(X, b)
    const shift = stopGradient(max(eta))
    const e = exp(sub(eta, shift))
    const reverse = concat([cumsum(e, 0, true), zero], 0)
    const risk = sub(take(reverse, riskStart), take(reverse, stratumEnd))
    const forward = concat([zero, cumsum(mul(e, delta), 0)], 0)
    const tied = sub(take(forward, groupEnd), take(forward, groupStart))
    const denominator = sub(take(risk, slotGroup), mul(take(tied, slotGroup), fraction))
    return sub(sum(mul(delta, sub(eta, shift))), sum(log(denominator)))
  }
  const objective = (b: Value): Value => {
    const nll = mul(neg(logLik(b)), 1 / n)
    return ridge === 0 ? nll : add(nll, mul(sum(square(b)), ridge / 2))
  }
  const asVector = (beta: ArrayLike<number>): Tensor => tensor(Array.from(beta))
  return {
    objective,
    x0: new Array<number>(p).fill(0),
    p,
    logLikelihood: (beta) => toFlat(logLik(asVector(beta)))[0] ?? Number.NaN,
    score: (beta) => {
      const g = valueAndGrad((b: Value) => logLik(b))(asVector(beta)).grad as Value
      return toFlat(g)
    },
    information: (beta) => {
      const h = toFlat(hessian(objective)(asVector(beta)))
      return Float64Array.from(h, (v) => v * n)
    },
  }
}

/** The hyperparameters of {@link coxPh}. */
export interface CoxHyper {
  /** Tied event times: `'efron'` (default) or `'breslow'`. */
  readonly ties: CoxTies
  /** A ridge penalty on the coefficients (default 0). */
  readonly ridge: number
  /**
   * The solver, a value of `smoothSolvers.space` (default Newton to a gradient tolerance of 1e-10, which reproduces
   * the reference fits; any other smooth solver reaches the same optimum).
   */
  readonly solver: SpaceValue
  /** The step budget of the solver (default 100). */
  readonly maxIter: number
  /** The confidence level of the intervals (default 0.95). */
  readonly level: number
}

/** The default hyperparameters of {@link coxPh}. */
export const coxDefaults: CoxHyper = {
  ties: 'efron',
  ridge: 0,
  solver: { kind: 'newton', gradTol: 1e-10 },
  maxIter: 100,
  level: 0.95,
}

/**
 * The traced solver run that fits a Cox model: the solver of the slot minimising {@link coxProblem}, logging
 * `solver/objective`, `solver/gradNorm` and `solver/step` per step.
 *
 * @param data - The data.
 * @param hyper - The hyperparameters (defaults filled in).
 * @param key - The run's key.
 * @returns The algorithm.
 * @example
 * ```ts
 * import { coxAlgorithm, coxData } from 'mdml-engine/survival'
 * import { key } from 'mdml-engine/random'
 * coxAlgorithm(coxData([1, 2, 3, 4], [1, 1, 1, 0], [0, 1, 2, 3]), {}, key(0)).name // => 'newton'
 * ```
 */
export function coxAlgorithm(
  data: CoxData,
  hyper: Partial<CoxHyper> = {},
  key: Key = makeKey(0),
): ReturnType<typeof loggedSolver> {
  const h = { ...coxDefaults, ...hyper }
  return loggedSolver(coxProblem(data, { ties: h.ties, ridge: h.ridge }), h.solver, key)
}

/** One coefficient of a fitted Cox model. */
export interface CoxCoefficient {
  readonly name: string
  readonly coefficient: number
  readonly standardError: number
  readonly z: number
  readonly pValue: number
  /** `exp(coefficient)`, the hazard ratio of a unit increase. */
  readonly hazardRatio: number
  /** The interval of the coefficient at the fit's level. */
  readonly lower: number
  readonly upper: number
  /** The interval of the hazard ratio. */
  readonly hazardRatioLower: number
  readonly hazardRatioUpper: number
}

/** The baseline cumulative hazard of one stratum (Breslow's estimator at covariates zero). */
export interface BaselineHazard {
  readonly stratum: number
  readonly time: readonly number[]
  readonly cumulativeHazard: readonly number[]
}

/** A fitted Cox model. */
export interface CoxFit {
  readonly coefficients: readonly CoxCoefficient[]
  readonly beta: readonly number[]
  /** The covariance of the coefficients: the inverse observed information. */
  readonly covariance: readonly (readonly number[])[]
  readonly logLikelihood: number
  readonly nullLogLikelihood: number
  /** Likelihood-ratio, Wald and score tests of all coefficients being zero. */
  readonly likelihoodRatio: TestResult
  readonly wald: TestResult
  readonly score: TestResult
  /** Harrell's C of the linear predictor. */
  readonly concordance: number
  readonly baseline: readonly BaselineHazard[]
  readonly ties: CoxTies
  readonly n: number
  readonly nEvents: number
  readonly iterations: number
  readonly converged: boolean
  readonly names: readonly string[]
}

function quadraticForm(a: ArrayLike<number>, inv: ArrayLike<number>, p: number): number {
  let s = 0
  for (let i = 0; i < p; i++) for (let j = 0; j < p; j++) s += (a[i] ?? 0) * (inv[i * p + j] ?? 0) * (a[j] ?? 0)
  return s
}

function chiSquareTest(name: string, statistic: number, df: number, method: string): TestResult {
  return { test: name, statistic, pValue: chi2Sf(df, statistic), alternative: 'greater', df, method }
}

/** Breslow's estimator of the baseline cumulative hazard `sum_{t_j <= t} d_j / sum_{R(t_j)} exp(x beta)` per stratum. */
function breslowBaseline(data: CoxData, beta: readonly number[], ties: CoxTies): BaselineHazard[] {
  const plan = planOf(data, ties)
  const { n, p } = plan
  const e = new Float64Array(n)
  for (let k = 0; k < n; k++) {
    let eta = 0
    for (let j = 0; j < p; j++) eta += (plan.x[k * p + j] ?? 0) * (beta[j] ?? 0)
    e[k] = Math.exp(eta)
  }
  const reverse = new Float64Array(n + 1)
  for (let k = n - 1; k >= 0; k--) reverse[k] = (reverse[k + 1] ?? 0) + (e[k] ?? 0)
  const out: BaselineHazard[] = []
  let current: { stratum: number; time: number[]; cum: number[] } | undefined
  let cumulative = 0
  for (let g = 0; g < plan.groupStart.length; g++) {
    const stratum = plan.groupStratum[g] ?? 0
    if (current === undefined || current.stratum !== stratum) {
      if (current !== undefined)
        out.push({ stratum: current.stratum, time: current.time, cumulativeHazard: current.cum })
      current = { stratum, time: [], cum: [] }
      cumulative = 0
    }
    const risk = (reverse[plan.groupStart[g] ?? 0] ?? 0) - (reverse[plan.stratumEnd[g] ?? 0] ?? 0)
    cumulative += (plan.groupEvents[g] ?? 0) / risk
    current.time.push(plan.groupTime[g] ?? 0)
    current.cum.push(cumulative)
  }
  if (current !== undefined) out.push({ stratum: current.stratum, time: current.time, cumulativeHazard: current.cum })
  return out
}

/**
 * Fit a Cox proportional hazards model (R's `coxph`, lifelines' `CoxPHFitter`, statsmodels' `PHReg`): the partial
 * likelihood is minimised by the solver of the slot (Newton by default), and the covariance, tests, concordance and
 * Breslow's baseline hazard are computed at the optimum.
 *
 * @param data - The data from {@link coxData}.
 * @param hyper - `ties`, `ridge`, `solver`, `maxIter` and `level`.
 * @param run - Run options (`key`, `tracer`, `policy`, `parent`) to record the solver's steps.
 * @returns The fit.
 * @throws MdmlError with code `value` when there is no event.
 * @example
 * ```ts
 * import { coxData, coxPh } from 'mdml-engine/survival'
 * const fit = coxPh(coxData([1, 2, 3, 4, 5, 6], [1, 1, 0, 1, 1, 1], [0.1, 0.8, 0.3, 1.5, 1.1, 2.0]))
 * fit.converged // => true
 * fit.coefficients[0]?.hazardRatio !== undefined // => true
 * ```
 */
export function coxPh(
  data: CoxData,
  hyper: Partial<CoxHyper> = {},
  run: Partial<Pick<RunOptions, 'key' | 'tracer' | 'policy' | 'parent'>> = {},
): CoxFit {
  const h = { ...coxDefaults, ...hyper }
  const problem = coxProblem(data, { ties: h.ties, ridge: h.ridge })
  const key = run.key ?? makeKey(0)
  const algorithm = loggedSolver(problem, h.solver, key)
  const result = runAlgorithm(algorithm, { ...run, key, maxSteps: h.maxIter })
  const beta = smoothSolvers.params(h.solver, result.state).x
  const p = problem.p
  const n = data.sample.n
  const info = problem.information(beta)
  const inv = inverseSpd(info, p)
  const covariance = inv === undefined ? new Float64Array(p * p).fill(Number.NaN) : inv
  const z = normPpf(0.5 + h.level / 2)
  const coefficients: CoxCoefficient[] = beta.map((b, j) => {
    const se = Math.sqrt(covariance[j * p + j] ?? Number.NaN)
    const stat = b / se
    return {
      name: data.names[j] ?? `x${j}`,
      coefficient: b,
      standardError: se,
      z: stat,
      pValue: 2 * normSf(Math.abs(stat)),
      hazardRatio: Math.exp(b),
      lower: b - z * se,
      upper: b + z * se,
      hazardRatioLower: Math.exp(b - z * se),
      hazardRatioUpper: Math.exp(b + z * se),
    }
  })
  const logLikelihood = problem.logLikelihood(beta)
  const zeros = new Array<number>(p).fill(0)
  const nullLogLikelihood = problem.logLikelihood(zeros)
  const infoNull = problem.information(zeros)
  const invNull = inverseSpd(infoNull, p)
  const scoreNull = problem.score(zeros)
  const eta = new Float64Array(n)
  for (let i = 0; i < n; i++) {
    let s = 0
    for (let j = 0; j < p; j++) s += (data.x[i * p + j] ?? 0) * (beta[j] ?? 0)
    eta[i] = s
  }
  let concordance = Number.NaN
  try {
    concordance = concordanceIndex(data.sample.time, data.sample.event, eta).concordance
  } catch {
    concordance = Number.NaN
  }
  let nEvents = 0
  for (let i = 0; i < n; i++) nEvents += data.sample.event[i] ?? 0
  const steps = (result as { steps?: number }).steps ?? 0
  return {
    coefficients,
    beta,
    covariance: rowsOfMatrix(covariance, p),
    logLikelihood,
    nullLogLikelihood,
    likelihoodRatio: chiSquareTest(
      'Likelihood-ratio test',
      2 * (logLikelihood - nullLogLikelihood),
      p,
      'partial likelihood',
    ),
    wald: chiSquareTest('Wald test', quadraticForm(beta, info, p), p, 'observed information'),
    score: chiSquareTest(
      'Score (log-rank) test',
      invNull === undefined ? Number.NaN : quadraticForm(scoreNull, invNull, p),
      p,
      'score at zero',
    ),
    concordance,
    baseline: breslowBaseline(data, beta, h.ties),
    ties: h.ties,
    n,
    nEvents,
    iterations: steps,
    converged: result.status === 'converged',
    names: data.names,
  }
}

/**
 * The baseline cumulative hazard of a stratum at a time (a step function, 0 before the first event).
 *
 * @param fit - The fit.
 * @param t - The time.
 * @param stratum - The stratum (default 0).
 * @returns `H0(t)`.
 * @example
 * ```ts
 * import { baselineHazardAt, coxData, coxPh } from 'mdml-engine/survival'
 * const fit = coxPh(coxData([1, 2, 3, 4], [1, 1, 1, 0], [0.2, 0.9, 0.4, 1.3]))
 * baselineHazardAt(fit, 0.5) // => 0
 * ```
 */
export function baselineHazardAt(fit: Pick<CoxFit, 'baseline'>, t: number, stratum = 0): number {
  const b = fit.baseline.find((s) => s.stratum === stratum)
  if (b === undefined) return Number.NaN
  let value = 0
  for (let k = 0; k < b.time.length && (b.time[k] ?? Number.POSITIVE_INFINITY) <= t; k++)
    value = b.cumulativeHazard[k] ?? value
  return value
}

/**
 * The survival function of a subject with covariates `x` at the given times, `exp(-H0(t) exp(x beta))`.
 *
 * @param fit - The fit.
 * @param x - The subject's covariates (length `p`).
 * @param times - The times.
 * @param stratum - The stratum (default 0).
 * @returns `S(t | x)` for each time.
 * @example
 * ```ts
 * import { coxData, coxPh, coxPredictSurvival } from 'mdml-engine/survival'
 * const fit = coxPh(coxData([1, 2, 3, 4], [1, 1, 1, 0], [0.2, 0.9, 0.4, 1.3]))
 * coxPredictSurvival(fit, [0.5], [0, 10])[0] // => 1
 * ```
 */
export function coxPredictSurvival(
  fit: Pick<CoxFit, 'baseline' | 'beta'>,
  x: ArrayLike<number>,
  times: ArrayLike<number>,
  stratum = 0,
): number[] {
  let eta = 0
  for (let j = 0; j < fit.beta.length; j++) eta += (x[j] ?? 0) * (fit.beta[j] ?? 0)
  const risk = Math.exp(eta)
  return Array.from(times, (t) => Math.exp(-baselineHazardAt(fit, t, stratum) * risk))
}

/**
 * Schoenfeld residuals of a fitted Cox model: at each event, the covariates of the subject who failed minus their
 * average over the risk set weighted by `exp(x beta)` (Breslow's form, also what statsmodels' `PHReg` reports). They
 * average to zero under proportional hazards, and a trend of a covariate's residuals against time is evidence that its
 * effect changes with time.
 *
 * @param data - The data the model was fitted to.
 * @param beta - The coefficients.
 * @returns An `n x p` tensor in the order of the data, with `NaN` rows for censored subjects.
 * @example
 * ```ts
 * import { coxData, coxPh, schoenfeldResiduals } from 'mdml-engine/survival'
 * const data = coxData([1, 2, 3, 4], [1, 1, 1, 0], [0.2, 0.9, 0.4, 1.3])
 * const r = schoenfeldResiduals(data, coxPh(data).beta)
 * r.shape // => [4, 1]
 * ```
 */
export function schoenfeldResiduals(data: CoxData, beta: readonly number[]): Tensor {
  const plan = planOf(data, 'breslow')
  const { n, p } = plan
  const e = new Float64Array(n)
  for (let k = 0; k < n; k++) {
    let eta = 0
    for (let j = 0; j < p; j++) eta += (plan.x[k * p + j] ?? 0) * (beta[j] ?? 0)
    e[k] = Math.exp(eta)
  }
  const reverse = new Float64Array(n + 1)
  const reverseX = new Float64Array((n + 1) * p)
  for (let k = n - 1; k >= 0; k--) {
    reverse[k] = (reverse[k + 1] ?? 0) + (e[k] ?? 0)
    for (let j = 0; j < p; j++)
      reverseX[k * p + j] = (reverseX[(k + 1) * p + j] ?? 0) + (e[k] ?? 0) * (plan.x[k * p + j] ?? 0)
  }
  const out = new Float64Array(n * p).fill(Number.NaN)
  for (let g = 0; g < plan.groupStart.length; g++) {
    const start = plan.groupStart[g] ?? 0
    const stop = plan.stratumEnd[g] ?? n
    const risk = (reverse[start] ?? 0) - (reverse[stop] ?? 0)
    for (let k = start; k < (plan.groupEnd[g] ?? start); k++) {
      if (plan.event[k] !== 1) continue
      const i = plan.order[k] ?? 0
      for (let j = 0; j < p; j++) {
        const mean = ((reverseX[start * p + j] ?? 0) - (reverseX[stop * p + j] ?? 0)) / risk
        out[i * p + j] = (plan.x[k * p + j] ?? 0) - mean
      }
    }
  }
  return fromStorage(out, [n, p], 'float64')
}

/**
 * Martingale residuals of a fitted Cox model, `delta_i - H0(t_i) exp(x_i beta)`: the observed minus the expected number
 * of events of each subject (they sum to zero and lie in `(-Infinity, 1]`), the residuals to plot against a covariate
 * to see its functional form.
 *
 * @param data - The data the model was fitted to.
 * @param fit - The fit.
 * @returns One residual per subject, in the order of the data.
 * @example
 * ```ts
 * import { coxData, coxPh, martingaleResiduals } from 'mdml-engine/survival'
 * const data = coxData([1, 2, 3, 4], [1, 1, 1, 0], [0.2, 0.9, 0.4, 1.3])
 * martingaleResiduals(data, coxPh(data, { ties: 'breslow' })).length // => 4
 * ```
 */
export function martingaleResiduals(data: CoxData, fit: Pick<CoxFit, 'beta' | 'baseline'>): number[] {
  const { sample, strata, p } = data
  return Array.from({ length: sample.n }, (_, i) => {
    let eta = 0
    for (let j = 0; j < p; j++) eta += (data.x[i * p + j] ?? 0) * (fit.beta[j] ?? 0)
    return (sample.event[i] ?? 0) - baselineHazardAt(fit, sample.time[i] ?? 0, strata[i] ?? 0) * Math.exp(eta)
  })
}
