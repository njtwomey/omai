/**
 * Accelerated failure time models: `log T = x beta + sigma W`, where the error `W` has a fixed distribution, so a
 * covariate multiplies survival time by `exp(beta)` (the time ratio). The Weibull (extreme-value errors, which with
 * `sigma = 1` is the exponential), log-normal and log-logistic models are fitted to right-censored data by maximum
 * likelihood. The negative log-likelihood is a smooth objective of `(beta, log sigma)` (`aftProblem`), so the fit takes
 * its solver from the slot (Newton by default, any other smooth solver reaches the same optimum); `aft` adds the
 * covariance from the observed information, the coefficient table and predictions of survival and quantiles.
 */

import { hessian } from '../autodiff/index.ts'
import { LOG_SQRT_2PI, MdmlError } from '../base/index.ts'
import { logNdtr } from '../numerics/special/index.ts'
import { loggedSolver, type SmoothProblem, smoothSolvers } from '../optim/index.ts'
import { type Key, key as makeKey } from '../random/index.ts'
import type { SpaceValue } from '../space/index.ts'
import {
  add,
  asTensor,
  exp,
  fromStorage,
  matmul,
  mul,
  neg,
  softplus,
  square,
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
import { inverseSpd, normPpf, normSf, rowsOfMatrix } from './_util.ts'
import { concordanceIndex } from './concordance.ts'
import { type SurvivalSample, survivalSample } from './data.ts'

/** The error distribution of the log time: `weibull`, `lognormal`, `loglogistic`, or `exponential` (Weibull with `sigma = 1`). */
export type AftDistribution = 'weibull' | 'lognormal' | 'loglogistic' | 'exponential'

/** Covariates with survival times, for an accelerated failure time model. */
export interface AftData {
  readonly sample: SurvivalSample
  /** Covariates, row-major `n x p` (`p` may be 0: a model of the survival distribution alone). */
  readonly x: Float64Array
  readonly p: number
  readonly names: readonly string[]
}

/**
 * Read the data of an accelerated failure time model.
 *
 * @param time - Observed times, positive.
 * @param event - Event indicators (omitted: all events).
 * @param x - Covariates `[n, p]` (a vector is one covariate); omitted for no covariates.
 * @param names - Covariate names.
 * @returns The data.
 * @throws MdmlError with code `value` for a time that is not positive, `shape` when the lengths disagree.
 * @example
 * ```ts
 * import { aftData } from 'mdml-engine/survival'
 * aftData([1, 2, 3], [1, 0, 1], [[0.5], [1], [2]]).p // => 1
 * ```
 */
export function aftData(time: TensorLike, event?: TensorLike, x?: TensorLike, names?: readonly string[]): AftData {
  const sample = survivalSample(time, event, 'aftData')
  for (let i = 0; i < sample.n; i++)
    if (!((sample.time[i] ?? 0) > 0)) throw new MdmlError('value', `aftData: time ${i} must be > 0`)
  if (x === undefined) return { sample, x: new Float64Array(0), p: 0, names: [] }
  const t = asTensor(x)
  const n = t.shape[0] ?? 0
  const p = t.shape.length === 1 ? 1 : (t.shape[1] ?? 0)
  if (n !== sample.n) throw new MdmlError('shape', `aftData: ${n} covariate rows for ${sample.n} times`)
  const nm = names ?? Array.from({ length: p }, (_, j) => `x${j}`)
  if (nm.length !== p) throw new MdmlError('shape', `aftData: ${nm.length} names for ${p} covariates`)
  return { sample, x: Float64Array.from(toFlat(t)), p, names: nm }
}

/** Options of {@link aftProblem}. */
export interface AftProblemOptions {
  /** The error distribution (default `'weibull'`). */
  readonly distribution?: AftDistribution
  /** Whether the model has an intercept (default true). */
  readonly fitIntercept?: boolean
}

/** An accelerated failure time model's negative log-likelihood as a smooth problem. */
export interface AftProblem extends SmoothProblem {
  readonly distribution: AftDistribution
  /** The parameters: the intercept (if any), the `p` slopes, and `log sigma` (absent for the exponential). */
  readonly dimension: number
  readonly names: readonly string[]
  /** The log-likelihood of the data at the parameters. */
  logLikelihood(theta: ArrayLike<number>): number
  /** The observed information (minus the Hessian of the log-likelihood), row-major. */
  information(theta: ArrayLike<number>): Float64Array
}

/** Least squares of log time on the covariates (all treated as events): the starting point of the fit. */
function startingPoint(data: AftData, intercept: boolean, scale: boolean): number[] {
  const { sample, p } = data
  const n = sample.n
  const k = p + (intercept ? 1 : 0)
  const logT = Array.from(sample.time, Math.log)
  const design = (i: number, j: number): number =>
    intercept ? (j === 0 ? 1 : (data.x[i * p + j - 1] ?? 0)) : (data.x[i * p + j] ?? 0)
  const theta = new Array<number>(k).fill(0)
  if (k > 0) {
    const a = new Float64Array(k * k)
    const b = new Float64Array(k)
    for (let i = 0; i < n; i++)
      for (let r = 0; r < k; r++) {
        b[r] = (b[r] ?? 0) + design(i, r) * (logT[i] ?? 0)
        for (let c = 0; c < k; c++) a[r * k + c] = (a[r * k + c] ?? 0) + design(i, r) * design(i, c)
      }
    for (let r = 0; r < k; r++) a[r * k + r] = (a[r * k + r] ?? 0) + 1e-8
    const inv = inverseSpd(a, k)
    if (inv !== undefined)
      for (let r = 0; r < k; r++)
        for (let c = 0; c < k; c++) theta[r] = (theta[r] ?? 0) + (inv[r * k + c] ?? 0) * (b[c] ?? 0)
  }
  if (!scale) return theta
  let rss = 0
  for (let i = 0; i < n; i++) {
    let fit = 0
    for (let r = 0; r < k; r++) fit += design(i, r) * (theta[r] ?? 0)
    rss += ((logT[i] ?? 0) - fit) ** 2
  }
  return [...theta, 0.5 * Math.log(Math.max(rss / n, 1e-6))]
}

/**
 * The accelerated failure time model's negative log-likelihood per subject as a {@link SmoothProblem}, with
 * `z = (log t - x beta) / sigma`: an event contributes `log f0(z) - log sigma - log t`, a censored time `log S0(z)`,
 * where `(f0, S0)` is the density and survival of the error (the Gumbel minimum, the standard normal or the standard
 * logistic). Gradient and Hessian come from autodiff; the start is least squares of log time on the covariates.
 *
 * @param data - The data.
 * @param options - `distribution` and `fitIntercept`.
 * @returns The problem.
 * @example
 * ```ts
 * import { aftData, aftProblem } from 'mdml-engine/survival'
 * const prob = aftProblem(aftData([1, 2, 3, 4], [1, 1, 0, 1]), { distribution: 'exponential' })
 * prob.dimension // => 1
 * ```
 */
export function aftProblem(data: AftData, options: AftProblemOptions = {}): AftProblem {
  const distribution = options.distribution ?? 'weibull'
  const intercept = options.fitIntercept ?? true
  const { sample, p } = data
  const n = sample.n
  const k = p + (intercept ? 1 : 0)
  const scaled = distribution !== 'exponential'
  const dimension = k + (scaled ? 1 : 0)
  if (dimension === 0) throw new MdmlError('value', 'aftProblem: needs an intercept, a covariate or a free scale')
  let events = 0
  for (let i = 0; i < n; i++) events += sample.event[i] ?? 0
  if (events === 0) throw new MdmlError('value', 'aftProblem: needs at least one event')
  const design = new Float64Array(n * k)
  for (let i = 0; i < n; i++)
    for (let j = 0; j < k; j++)
      design[i * k + j] = intercept ? (j === 0 ? 1 : (data.x[i * p + j - 1] ?? 0)) : (data.x[i * p + j] ?? 0)
  const X = fromStorage(design, [n, Math.max(k, 1)], 'float64')
  const logT = tensor(Array.from(sample.time, Math.log))
  const delta = tensor(Array.from(sample.event, Number))
  const censored = tensor(Array.from(sample.event, (e) => 1 - e))
  const names = [...(intercept ? ['(intercept)'] : []), ...data.names, ...(scaled ? ['log(scale)'] : [])]
  const logLik = (theta: Value): Value => {
    const thetaBeta = k === 0 ? undefined : sliceVector(theta, 0, k)
    const logScale = scaled ? sliceVector(theta, k, k + 1) : undefined
    const eta = thetaBeta === undefined ? undefined : matmul(X, thetaBeta)
    const s = logScale === undefined ? tensor(0) : sum(logScale)
    const location = eta === undefined ? mul(logT, 0) : eta
    const z = mul(sub(logT, location), exp(neg(s)))
    const eventTerm = neg(add(s, logT))
    switch (distribution) {
      case 'lognormal': {
        const density = sub(eventTerm, add(mul(square(z), 0.5), LOG_SQRT_2PI))
        const survival = logNdtr(neg(z))
        return add(sum(mul(delta, density)), sum(mul(censored, survival)))
      }
      case 'loglogistic': {
        const density = add(eventTerm, sub(z, mul(softplus(z), 2)))
        return add(sum(mul(delta, density)), neg(sum(mul(censored, softplus(z)))))
      }
      default: {
        const density = add(eventTerm, z)
        return sub(sum(mul(delta, density)), sum(exp(z)))
      }
    }
  }
  const objective = (theta: Value): Value => mul(neg(logLik(theta)), 1 / n)
  const asVector = (theta: ArrayLike<number>): Tensor => tensor(Array.from(theta))
  return {
    objective,
    x0: startingPoint(data, intercept, scaled),
    distribution,
    dimension,
    names,
    logLikelihood: (theta) => toFlat(logLik(asVector(theta)))[0] ?? Number.NaN,
    information: (theta) => Float64Array.from(toFlat(hessian(objective)(asVector(theta))), (v) => v * n),
  }
}

/** A slice of a vector value (`[start, stop)`), differentiable. */
function sliceVector(v: Value, start: number, stop: number): Value {
  return take(
    v,
    Array.from({ length: stop - start }, (_, i) => start + i),
  )
}

/** The hyperparameters of {@link aft}. */
export interface AftHyper {
  /** The error distribution (default `'weibull'`). */
  readonly distribution: AftDistribution
  /** Whether the model has an intercept (default true). */
  readonly fitIntercept: boolean
  /** The solver, a value of `smoothSolvers.space` (default Newton to a gradient tolerance of 1e-10). */
  readonly solver: SpaceValue
  /** The step budget of the solver (default 200). */
  readonly maxIter: number
  /** The confidence level of the intervals (default 0.95). */
  readonly level: number
}

/** The default hyperparameters of {@link aft}. */
export const aftDefaults: AftHyper = {
  distribution: 'weibull',
  fitIntercept: true,
  solver: { kind: 'newton', gradTol: 1e-10 },
  maxIter: 200,
  level: 0.95,
}

/**
 * The traced solver run that fits an accelerated failure time model.
 *
 * @param data - The data.
 * @param hyper - The hyperparameters (defaults filled in).
 * @param key - The run's key.
 * @returns The algorithm.
 * @example
 * ```ts
 * import { aftAlgorithm, aftData } from 'mdml-engine/survival'
 * aftAlgorithm(aftData([1, 2, 3, 4], [1, 1, 0, 1]), { distribution: 'lognormal' }).name.length > 0 // => true
 * ```
 */
export function aftAlgorithm(
  data: AftData,
  hyper: Partial<AftHyper> = {},
  key: Key = makeKey(0),
): ReturnType<typeof loggedSolver> {
  const h = { ...aftDefaults, ...hyper }
  return loggedSolver(aftProblem(data, { distribution: h.distribution, fitIntercept: h.fitIntercept }), h.solver, key)
}

/** One parameter of a fitted accelerated failure time model. */
export interface AftCoefficient {
  readonly name: string
  readonly coefficient: number
  readonly standardError: number
  readonly z: number
  readonly pValue: number
  /** `exp(coefficient)`: the factor a unit increase multiplies survival time by (for the slopes). */
  readonly timeRatio: number
  readonly lower: number
  readonly upper: number
}

/** A fitted accelerated failure time model. */
export interface AftFit {
  readonly distribution: AftDistribution
  readonly coefficients: readonly AftCoefficient[]
  /** The fitted parameters: intercept (if any), slopes, `log sigma` (if free). */
  readonly theta: readonly number[]
  readonly intercept: number
  /** The slopes. */
  readonly beta: readonly number[]
  /** The scale `sigma` of the error (1 for the exponential). */
  readonly scale: number
  /** The standard error of `sigma` by the delta method (0 when fixed). */
  readonly scaleStandardError: number
  readonly covariance: readonly (readonly number[])[]
  readonly logLikelihood: number
  readonly aic: number
  readonly concordance: number
  readonly n: number
  readonly nEvents: number
  readonly iterations: number
  readonly converged: boolean
  readonly names: readonly string[]
}

/**
 * Fit an accelerated failure time model by maximum likelihood, the solver coming from the slot (lifelines'
 * `WeibullAFTFitter`, `LogNormalAFTFitter` and `LogLogisticAFTFitter`; R's `survreg`).
 *
 * @param data - The data from {@link aftData}.
 * @param hyper - `distribution`, `fitIntercept`, `solver`, `maxIter` and `level`.
 * @param run - Run options (`key`, `tracer`, `policy`, `parent`) to record the solver's steps.
 * @returns The fit.
 * @example
 * ```ts
 * import { aft, aftData } from 'mdml-engine/survival'
 * const fit = aft(aftData([2, 3, 4, 6, 7, 9, 12, 15], [1, 1, 0, 1, 1, 0, 1, 1]), { distribution: 'weibull' })
 * fit.scale > 0 // => true
 * fit.converged // => true
 * ```
 */
export function aft(
  data: AftData,
  hyper: Partial<AftHyper> = {},
  run: Partial<Pick<RunOptions, 'key' | 'tracer' | 'policy' | 'parent'>> = {},
): AftFit {
  const h = { ...aftDefaults, ...hyper }
  const problem = aftProblem(data, { distribution: h.distribution, fitIntercept: h.fitIntercept })
  const key = run.key ?? makeKey(0)
  const result = runAlgorithm(loggedSolver(problem, h.solver, key), { ...run, key, maxSteps: h.maxIter })
  const theta = smoothSolvers.params(h.solver, result.state).x
  const m = problem.dimension
  const info = problem.information(theta)
  const inv = inverseSpd(info, m)
  const cov = inv === undefined ? new Float64Array(m * m).fill(Number.NaN) : inv
  const z = normPpf(0.5 + h.level / 2)
  const coefficients: AftCoefficient[] = theta.map((b, j) => {
    const se = Math.sqrt(cov[j * m + j] ?? Number.NaN)
    return {
      name: problem.names[j] ?? `theta${j}`,
      coefficient: b,
      standardError: se,
      z: b / se,
      pValue: 2 * normSf(Math.abs(b / se)),
      timeRatio: Math.exp(b),
      lower: b - z * se,
      upper: b + z * se,
    }
  })
  const scaled = h.distribution !== 'exponential'
  const logScale = scaled ? (theta[m - 1] ?? 0) : 0
  const scale = Math.exp(logScale)
  const k = m - (scaled ? 1 : 0)
  const intercept = h.fitIntercept ? (theta[0] ?? 0) : 0
  const beta = theta.slice(h.fitIntercept ? 1 : 0, k)
  const n = data.sample.n
  const eta = Array.from({ length: n }, (_, i) => linearPredictor(data, intercept, beta, i))
  let concordance = Number.NaN
  try {
    concordance = concordanceIndex(data.sample.time, data.sample.event, eta, { score: 'time' }).concordance
  } catch {
    concordance = Number.NaN
  }
  const logLikelihood = problem.logLikelihood(theta)
  let nEvents = 0
  for (let i = 0; i < n; i++) nEvents += data.sample.event[i] ?? 0
  return {
    distribution: h.distribution,
    coefficients,
    theta,
    intercept,
    beta,
    scale,
    scaleStandardError: scaled ? scale * Math.sqrt(cov[(m - 1) * m + (m - 1)] ?? Number.NaN) : 0,
    covariance: rowsOfMatrix(cov, m),
    logLikelihood,
    aic: 2 * m - 2 * logLikelihood,
    concordance,
    n,
    nEvents,
    iterations: result.steps,
    converged: result.status === 'converged',
    names: problem.names,
  }
}

function linearPredictor(data: AftData, intercept: number, beta: readonly number[], i: number): number {
  let eta = intercept
  for (let j = 0; j < data.p; j++) eta += (data.x[i * data.p + j] ?? 0) * (beta[j] ?? 0)
  return eta
}

/** The standard error distribution: the survival `S0(z)` and the quantile function `z = F0^-1(p)`. */
function baseSurvival(distribution: AftDistribution, z: number): number {
  switch (distribution) {
    case 'lognormal':
      return normSf(z)
    case 'loglogistic':
      return 1 / (1 + Math.exp(z))
    default:
      return Math.exp(-Math.exp(z))
  }
}

function baseQuantile(distribution: AftDistribution, p: number): number {
  switch (distribution) {
    case 'lognormal':
      return normPpf(p)
    case 'loglogistic':
      return Math.log(p / (1 - p))
    default:
      return Math.log(-Math.log(1 - p))
  }
}

/**
 * The survival function of a subject with covariates `x` at the given times, `S0((log t - x beta) / sigma)`.
 *
 * @param fit - The fit.
 * @param x - The covariates (length `p`; empty for none).
 * @param times - The times (positive).
 * @returns `S(t | x)` for each time.
 * @example
 * ```ts
 * import { aft, aftData, aftPredictSurvival } from 'mdml-engine/survival'
 * const fit = aft(aftData([2, 3, 4, 6, 7, 9, 12, 15], [1, 1, 0, 1, 1, 0, 1, 1]), { distribution: 'lognormal' })
 * aftPredictSurvival(fit, [], [1e-6])[0] // => 1
 * ```
 */
export function aftPredictSurvival(
  fit: Pick<AftFit, 'distribution' | 'intercept' | 'beta' | 'scale'>,
  x: ArrayLike<number>,
  times: ArrayLike<number>,
): number[] {
  let eta = fit.intercept
  for (let j = 0; j < fit.beta.length; j++) eta += (x[j] ?? 0) * (fit.beta[j] ?? 0)
  return Array.from(times, (t) => {
    const s = baseSurvival(fit.distribution, (Math.log(t) - eta) / fit.scale)
    return Math.min(1, Math.max(0, Number.isNaN(s) ? 0 : s))
  })
}

/**
 * The p-quantile of the survival time of a subject with covariates `x`: `exp(x beta + sigma F0^-1(p))`; the median is
 * `p = 0.5`.
 *
 * @param fit - The fit.
 * @param x - The covariates (length `p`; empty for none).
 * @param p - The probability in (0, 1) (default 0.5).
 * @returns The time by which a share `p` of such subjects has failed.
 * @example
 * ```ts
 * import { aft, aftData, aftPredictQuantile } from 'mdml-engine/survival'
 * const fit = aft(aftData([2, 3, 4, 6, 7, 9, 12, 15]), { distribution: 'lognormal' })
 * aftPredictQuantile(fit, [], 0.5) === Math.exp(fit.intercept) // => true
 * ```
 */
export function aftPredictQuantile(
  fit: Pick<AftFit, 'distribution' | 'intercept' | 'beta' | 'scale'>,
  x: ArrayLike<number>,
  p = 0.5,
): number {
  if (!(p > 0 && p < 1)) throw new MdmlError('value', `aftPredictQuantile: p must be in (0, 1), got ${p}`)
  let eta = fit.intercept
  for (let j = 0; j < fit.beta.length; j++) eta += (x[j] ?? 0) * (fit.beta[j] ?? 0)
  return Math.exp(eta + fit.scale * baseQuantile(fit.distribution, p))
}
