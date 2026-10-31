/**
 * Exponential smoothing as ETS state-space models (Hyndman et al., statsmodels' `ETSModel`): the error is additive or
 * multiplicative, the trend none, additive or multiplicative and optionally damped, the seasonal none, additive or
 * multiplicative. Simple smoothing, Holt's linear and damped trend, and Holt-Winters are presets. The model runs one
 * observation per step as an `Algorithm` that records the fitted value, the residual and the running likelihood;
 * the smoothing parameters and initial states are fitted by maximum likelihood with a traced BFGS algorithm, and
 * forecasts carry analytic prediction variances for additive models.
 */

import { MdmlError } from '../base/index.ts'
import { ndtri } from '../numerics/special/index.ts'
import type { BfgsState } from '../optim/index.ts'
import { type Key, normal } from '../random/index.ts'
import { item, toFlat } from '../tensor/index.ts'
import { type Algorithm, channels, runAlgorithm, stateFromJson, stateToJson } from '../trace/index.ts'
import { likelihoodAlgorithm } from './_ml.ts'
import { asSeries } from './_series.ts'

/** The structure of an ETS model: the error, trend and seasonal types. */
export interface EtsSpec {
  readonly error: 'add' | 'mul'
  readonly trend: 'none' | 'add' | 'mul'
  /** Damp the trend with the factor phi (default false). */
  readonly damped?: boolean
  readonly seasonal: 'none' | 'add' | 'mul'
  /** The seasonal period m (required with a seasonal component). */
  readonly period?: number
}

/**
 * The smoothing parameters and initial states of an ETS model, in statsmodels' convention: `beta` is the weight alpha
 * beta* of the trend and `gamma` the weight (1 - alpha) gamma* of the seasonal component.
 */
export interface EtsParameters {
  readonly alpha: number
  readonly beta?: number
  readonly gamma?: number
  /** The damping factor (1 when absent). */
  readonly phi?: number
  readonly level0: number
  readonly trend0?: number
  /** The initial seasonal values s_(-m), ..., s_(-1), oldest first. */
  readonly seasonal0?: readonly number[]
}

/**
 * The presets of the classical smoothing methods as ETS specifications.
 *
 * @example
 * ```ts
 * import { smoothingModels } from 'mdml-engine/timeseries'
 * Object.keys(smoothingModels) // => ['simple', 'holt', 'damped']
 * ```
 */
export const smoothingModels: Readonly<Record<string, EtsSpec>> = {
  /** Simple exponential smoothing: ETS(A, N, N). */
  simple: { error: 'add', trend: 'none', seasonal: 'none' },
  /** Holt's linear trend: ETS(A, A, N). */
  holt: { error: 'add', trend: 'add', seasonal: 'none' },
  /** Holt's damped trend: ETS(A, Ad, N). */
  damped: { error: 'add', trend: 'add', damped: true, seasonal: 'none' },
}

/**
 * Holt-Winters with additive seasonality of the given period: ETS(A, A, A).
 *
 * @param period - The seasonal period.
 * @returns The specification.
 * @example
 * ```ts
 * import { holtWintersAdditive } from 'mdml-engine/timeseries'
 * holtWintersAdditive(12).seasonal // => 'add'
 * ```
 */
export function holtWintersAdditive(period: number): EtsSpec {
  return { error: 'add', trend: 'add', seasonal: 'add', period }
}

/**
 * Holt-Winters with multiplicative seasonality of the given period: ETS(M, A, M).
 *
 * @param period - The seasonal period.
 * @returns The specification.
 * @example
 * ```ts
 * import { holtWintersMultiplicative } from 'mdml-engine/timeseries'
 * holtWintersMultiplicative(4).error // => 'mul'
 * ```
 */
export function holtWintersMultiplicative(period: number): EtsSpec {
  return { error: 'mul', trend: 'add', seasonal: 'mul', period }
}

interface Resolved {
  readonly spec: EtsSpec
  readonly m: number
  readonly damped: boolean
}

function resolve(spec: EtsSpec): Resolved {
  const m = spec.seasonal === 'none' ? 0 : (spec.period ?? 0)
  if (spec.seasonal !== 'none' && !(Number.isInteger(m) && m >= 2))
    throw new MdmlError('value', 'ETS: a seasonal component needs a period of at least 2')
  return { spec, m, damped: spec.trend !== 'none' && spec.damped === true }
}

/** The mutable run state of the filter: level, trend and the circular buffer of seasonal values. */
interface Run {
  level: number
  trend: number
  readonly ring: Float64Array
  /** Index of s_(t-m) in the ring (the oldest value). */
  pos: number
}

function startRun(r: Resolved, p: EtsParameters): Run {
  const ring = new Float64Array(r.m)
  for (let j = 0; j < r.m; j++) ring[j] = p.seasonal0?.[j] ?? (r.spec.seasonal === 'mul' ? 1 : 0)
  return { level: p.level0, trend: p.trend0 ?? (r.spec.trend === 'mul' ? 1 : 0), ring, pos: 0 }
}

/** Advance the run over one observation; returns the one-step forecast. */
function advance(r: Resolved, p: EtsParameters, s: Run, y: number): number {
  const { spec } = r
  const alpha = p.alpha
  const phi = r.damped ? (p.phi ?? 1) : 1
  const addTrend = spec.trend === 'add'
  const mulTrend = spec.trend === 'mul'
  const trendTerm = addTrend ? phi * s.trend : mulTrend ? s.trend ** phi : 0
  const base = addTrend ? s.level + trendTerm : mulTrend ? s.level * trendTerm : s.level
  const season = r.m === 0 ? (spec.seasonal === 'mul' ? 1 : 0) : (s.ring[s.pos] ?? 0)
  const mulSeason = spec.seasonal === 'mul'
  const yhat = mulSeason ? base * season : base + season
  const level = alpha * (mulSeason ? y / season : y - season) + (1 - alpha) * base
  if (addTrend) {
    const betaStar = (p.beta ?? 0) / alpha
    s.trend = betaStar * (level - s.level) + (1 - betaStar) * trendTerm
  } else if (mulTrend) {
    const betaStar = (p.beta ?? 0) / alpha
    s.trend = betaStar * (level / s.level) + (1 - betaStar) * trendTerm
  }
  if (r.m > 0) {
    const gammaStar = (p.gamma ?? 0) / (1 - alpha)
    s.ring[s.pos] = gammaStar * (mulSeason ? y / level : y - level) + (1 - gammaStar) * season
    s.pos = (s.pos + 1) % r.m
  }
  s.level = level
  return yhat
}

const residualOf = (r: Resolved, y: number, yhat: number): number =>
  r.spec.error === 'mul' ? (y - yhat) / yhat : y - yhat

/** The concentrated Gaussian log-likelihood from the sum of squared residuals over n observations. */
function concentratedLogLikelihood(n: number, sse: number, sumLogYhat: number): number {
  return -0.5 * n * (Math.log((2 * Math.PI * sse) / n) + 1) - sumLogYhat
}

/** The result of running an ETS filter over a series. */
export interface EtsFilterResult {
  /** The one-step forecasts. */
  readonly fitted: Float64Array
  /** The residuals (relative for multiplicative errors). */
  readonly residuals: Float64Array
  /** The level after each observation. */
  readonly level: Float64Array
  /** The trend after each observation (zeros without a trend). */
  readonly trend: Float64Array
  /** The log-likelihood with the variance concentrated out (statsmodels' `llf`). */
  readonly logLikelihood: number
  /** The mean squared residual, the maximum-likelihood variance of the errors. */
  readonly sigma2: number
  /** The final state, ready for forecasting. */
  readonly final: { readonly level: number; readonly trend: number; readonly seasonal: readonly number[] }
}

/**
 * Run an ETS model with given parameters over a series.
 *
 * @param y - The series (positive for multiplicative components).
 * @param spec - The structure.
 * @param params - Smoothing parameters and initial states.
 * @returns Fitted values, residuals, states, the log-likelihood and the final state.
 * @example
 * ```ts
 * import { etsFilter } from 'mdml-engine/timeseries'
 * // Simple smoothing with alpha = 0.5 from level 10: the level moves halfway to each observation.
 * const f = etsFilter([12, 14], { error: 'add', trend: 'none', seasonal: 'none' }, { alpha: 0.5, level0: 10 })
 * Array.from(f.level) // => [11, 12.5]
 * ```
 */
export function etsFilter(y: ArrayLike<number>, spec: EtsSpec, params: EtsParameters): EtsFilterResult {
  const x = asSeries(Float64Array.from(y), 'etsFilter')
  const r = resolve(spec)
  const s = startRun(r, params)
  const n = x.length
  const fitted = new Float64Array(n)
  const residuals = new Float64Array(n)
  const level = new Float64Array(n)
  const trend = new Float64Array(n)
  let sse = 0
  let sumLog = 0
  for (let t = 0; t < n; t++) {
    const yhat = advance(r, params, s, x[t] ?? 0)
    const e = residualOf(r, x[t] ?? 0, yhat)
    fitted[t] = yhat
    residuals[t] = e
    level[t] = s.level
    trend[t] = s.trend
    sse += e * e
    if (spec.error === 'mul') sumLog += Math.log(Math.abs(yhat) || 1e-32)
  }
  return {
    fitted,
    residuals,
    level,
    trend,
    logLikelihood: concentratedLogLikelihood(n, sse, sumLog),
    sigma2: sse / n,
    final: { level: s.level, trend: s.trend, seasonal: ordered(s) },
  }
}

function ordered(s: Run): number[] {
  const m = s.ring.length
  return Array.from({ length: m }, (_, j) => s.ring[(s.pos + j) % m] ?? 0)
}

/** The state of {@link etsAlgorithm}: the filter's state after `t` observations and the running sums of the likelihood. */
export interface EtsState {
  /** Observations consumed. */
  readonly t: number
  readonly level: number
  readonly trend: number
  /** The circular buffer of seasonal values and the index of the oldest. */
  readonly ring: number[]
  readonly pos: number
  readonly sse: number
  readonly sumLogYhat: number
  /** The latest one-step forecast and residual. */
  readonly fitted: number
  readonly residual: number
}

/**
 * An ETS model as an algorithm, one observation per step: each step forecasts the next observation, scores it against
 * the truth and updates level, trend and seasonal values. It records the forecast, the residual, the states and the
 * running log-likelihood, and its state serialises.
 *
 * @param y - The series.
 * @param spec - The structure.
 * @param params - Smoothing parameters and initial states.
 * @returns The algorithm; it is done after the last observation.
 * @example
 * ```ts
 * import { etsAlgorithm } from 'mdml-engine/timeseries'
 * import { runAlgorithm } from 'mdml-engine/trace'
 * const spec = { error: 'add', trend: 'none', seasonal: 'none' } as const
 * runAlgorithm(etsAlgorithm([12, 14, 13], spec, { alpha: 0.5, level0: 10 }), { maxSteps: 10 }).state.level // => 12.75
 * ```
 */
export function etsAlgorithm(y: ArrayLike<number>, spec: EtsSpec, params: EtsParameters): Algorithm<EtsState> {
  const x = asSeries(Float64Array.from(y), 'etsAlgorithm')
  const r = resolve(spec)
  const n = x.length
  return {
    name: 'ets',
    config: { error: spec.error, trend: spec.trend, damped: r.damped, seasonal: spec.seasonal, period: r.m },
    schema: channels({
      metrics: {
        logLikelihood: {
          description: 'the concentrated log-likelihood of the observations so far',
          direction: 'higher',
        },
        fitted: { description: 'the one-step forecast of this observation' },
        residual: { description: 'the one-step residual' },
        level: { description: 'the level after this observation' },
        trend: { description: 'the trend after this observation' },
      },
      tensors: { seasonal: { role: 'state', description: 'the seasonal values, oldest first' } },
    }),
    init: () => {
      const s = startRun(r, params)
      return {
        t: 0,
        level: s.level,
        trend: s.trend,
        ring: Array.from(s.ring),
        pos: 0,
        sse: 0,
        sumLogYhat: 0,
        fitted: Number.NaN,
        residual: Number.NaN,
      }
    },
    step: (st) => {
      const s: Run = { level: st.level, trend: st.trend, ring: Float64Array.from(st.ring), pos: st.pos }
      const yt = x[st.t] ?? Number.NaN
      const yhat = advance(r, params, s, yt)
      const e = residualOf(r, yt, yhat)
      return {
        t: st.t + 1,
        level: s.level,
        trend: s.trend,
        ring: Array.from(s.ring),
        pos: s.pos,
        sse: st.sse + e * e,
        sumLogYhat: st.sumLogYhat + (spec.error === 'mul' ? Math.log(Math.abs(yhat) || 1e-32) : 0),
        fitted: yhat,
        residual: e,
      }
    },
    done: (st) => (st.t >= n ? 'converged' : undefined),
    observe: (st) => ({
      metrics: {
        logLikelihood: st.t === 0 ? 0 : concentratedLogLikelihood(st.t, st.sse, st.sumLogYhat),
        fitted: st.fitted,
        residual: st.residual,
        level: st.level,
        trend: st.trend,
      },
      tensors: { seasonal: { role: 'state', value: seasonalOf(st), shape: [r.m] } },
    }),
    serialise: (st) => stateToJson(st),
    deserialise: (j) => stateFromJson(j) as EtsState,
  }
}

function seasonalOf(st: EtsState): number[] {
  const m = st.ring.length
  return Array.from({ length: m }, (_, j) => st.ring[(st.pos + j) % m] ?? 0)
}

/** A fitted ETS model: plain data. */
export interface EtsFit {
  readonly spec: EtsSpec
  readonly params: EtsParameters
  readonly logLikelihood: number
  /** The mean squared residual. */
  readonly sigma2: number
  /** The number of fitted parameters: the smoothing parameters, damping, and the initial states. */
  readonly nparams: number
  readonly aic: number
  readonly bic: number
  readonly nobs: number
  /** The state after the last observation. */
  readonly final: { readonly level: number; readonly trend: number; readonly seasonal: readonly number[] }
  readonly converged: boolean
  readonly iterations: number
}

const logistic = (u: number): number => 1 / (1 + Math.exp(-u))
const logit = (p: number): number => Math.log(p / (1 - p))

/** The layout of the unconstrained parameter vector of an ETS fit and its map to natural parameters. */
function layoutOf(r: Resolved): { readonly size: number; readonly toParams: (u: ArrayLike<number>) => EtsParameters } {
  const hasTrend = r.spec.trend !== 'none'
  const mulSeason = r.spec.seasonal === 'mul'
  const size = 1 + (hasTrend ? 1 : 0) + (r.m > 0 ? 1 : 0) + (r.damped ? 1 : 0) + 1 + (hasTrend ? 1 : 0) + r.m
  return {
    size,
    toParams: (u) => {
      let k = 0
      const alpha = 1e-4 + (1 - 2e-4) * logistic(u[k++] ?? 0)
      const beta = hasTrend ? alpha * logistic(u[k++] ?? 0) : undefined
      const gamma = r.m > 0 ? (1 - alpha) * logistic(u[k++] ?? 0) : undefined
      const phi = r.damped ? 0.8 + 0.18 * logistic(u[k++] ?? 0) : undefined
      const level0 = u[k++] ?? 0
      const trend0 = hasTrend ? (u[k++] ?? 0) : undefined
      let seasonal0: number[] | undefined
      if (r.m > 0) {
        const raw = Array.from({ length: r.m }, (_, j) => u[k + j] ?? 0)
        if (mulSeason) {
          const e = raw.map((v) => Math.exp(v))
          const total = e.reduce((a, b) => a + b, 0)
          seasonal0 = e.map((v) => (r.m * v) / total)
        } else {
          const mean = raw.reduce((a, b) => a + b, 0) / r.m
          seasonal0 = raw.map((v) => v - mean)
        }
      }
      return {
        alpha,
        ...(beta === undefined ? {} : { beta }),
        ...(gamma === undefined ? {} : { gamma }),
        ...(phi === undefined ? {} : { phi }),
        level0,
        ...(trend0 === undefined ? {} : { trend0: r.spec.trend === 'mul' ? Math.exp(trend0) : trend0 }),
        ...(seasonal0 === undefined ? {} : { seasonal0 }),
      }
    },
  }
}

function startVector(x: Float64Array, r: Resolved): number[] {
  const m = r.m
  const hasTrend = r.spec.trend !== 'none'
  const out: number[] = [logit(0.3)]
  if (hasTrend) out.push(logit(0.1))
  if (m > 0) out.push(logit(0.1))
  if (r.damped) out.push(logit(0.5))
  const first = m > 0 ? x.subarray(0, m) : x.subarray(0, Math.min(x.length, 4))
  const mean = first.reduce((a, b) => a + b, 0) / Math.max(first.length, 1)
  out.push(mean)
  if (hasTrend) {
    const slope =
      m > 0 && x.length >= 2 * m
        ? (x.subarray(m, 2 * m).reduce((a, b) => a + b, 0) - first.reduce((a, b) => a + b, 0)) / (m * m)
        : ((x[Math.min(x.length - 1, 3)] ?? 0) - (x[0] ?? 0)) / Math.max(Math.min(x.length - 1, 3), 1)
    out.push(r.spec.trend === 'mul' ? 0 : slope)
  }
  if (m > 0) {
    for (let j = 0; j < m; j++)
      out.push(r.spec.seasonal === 'mul' ? Math.log(Math.max((x[j] ?? 1) / (mean || 1), 1e-3)) : (x[j] ?? 0) - mean)
  }
  return out
}

/** Options of {@link etsFit}. */
export interface EtsFitOptions {
  /** The BFGS iteration cap (default 300). */
  readonly maxIterations?: number
  /** The gradient tolerance of the per-observation negative log-likelihood (default 1e-6). */
  readonly gradTol?: number
}

/** An ETS fit as an algorithm plus the map from its final state to the fitted model. */
export interface EtsProblem {
  readonly algorithm: Algorithm<BfgsState>
  readonly fit: (state: BfgsState, iterations?: number, converged?: boolean) => EtsFit
}

/**
 * The maximum-likelihood fit of an ETS model as a traced algorithm: BFGS on the smoothing parameters (kept inside their
 * admissible ranges by logistic maps: 0 < alpha < 1, 0 < beta < alpha, 0 < gamma < 1 - alpha, 0.8 <= phi <= 0.98) and
 * the initial states (seasonal values centred or normalised), one iteration per step with the log-likelihood recorded.
 *
 * @param y - The series.
 * @param spec - The structure.
 * @param options - The iteration cap and gradient tolerance.
 * @returns The algorithm and the map from its state to the fit.
 * @example
 * ```ts
 * import { etsFitAlgorithm } from 'mdml-engine/timeseries'
 * etsFitAlgorithm([3, 5, 4, 6, 5, 7, 6, 8], { error: 'add', trend: 'add', seasonal: 'none' }).algorithm.name // => 'ets-fit'
 * ```
 */
export function etsFitAlgorithm(y: ArrayLike<number>, spec: EtsSpec, options: EtsFitOptions = {}): EtsProblem {
  const x = asSeries(Float64Array.from(y), 'etsFit')
  const r = resolve(spec)
  const layout = layoutOf(r)
  if (x.length < layout.size + 1) throw new MdmlError('value', 'etsFit: the series is too short for the model')
  if ((spec.error === 'mul' || spec.trend === 'mul' || spec.seasonal === 'mul') && x.some((v) => !(v > 0)))
    throw new MdmlError('value', 'etsFit: multiplicative components need a positive series')
  const objective = (u: Float64Array): number => {
    const f = etsFilterQuick(x, r, layout.toParams(u))
    return Number.isFinite(f) ? -f / x.length : 1e10
  }
  const algorithm = likelihoodAlgorithm(
    'ets-fit',
    { ...spec, damped: r.damped },
    objective,
    startVector(x, r),
    x.length,
    options,
  )
  return {
    algorithm,
    fit: (state, iterations = 0, converged = true) => {
      const params = layout.toParams(state.x)
      const f = etsFilter(x, spec, params)
      const k = layout.size
      return {
        spec,
        params,
        logLikelihood: f.logLikelihood,
        sigma2: f.sigma2,
        nparams: k,
        aic: -2 * f.logLikelihood + 2 * (k + 1),
        bic: -2 * f.logLikelihood + (k + 1) * Math.log(x.length),
        nobs: x.length,
        final: f.final,
        converged,
        iterations,
      }
    },
  }
}

/** The log-likelihood only, without allocating the series of states (the inner loop of the fit). */
function etsFilterQuick(x: Float64Array, r: Resolved, params: EtsParameters): number {
  const s = startRun(r, params)
  let sse = 0
  let sumLog = 0
  for (let t = 0; t < x.length; t++) {
    const yhat = advance(r, params, s, x[t] ?? 0)
    const e = residualOf(r, x[t] ?? 0, yhat)
    sse += e * e
    if (r.spec.error === 'mul') sumLog += Math.log(Math.abs(yhat) || 1e-32)
  }
  return concentratedLogLikelihood(x.length, sse, sumLog)
}

/**
 * Fit an ETS model by maximum likelihood.
 *
 * @param y - The series.
 * @param spec - The structure (see {@link smoothingModels} for the classical presets).
 * @param options - The iteration cap and gradient tolerance.
 * @returns The fitted model.
 * @example
 * ```ts
 * import { etsFit, smoothingModels } from 'mdml-engine/timeseries'
 * const f = etsFit([3, 5, 4, 6, 5, 7, 6, 8, 7, 9, 8, 10], smoothingModels['holt'] as never)
 * f.params.alpha > 0 // => true
 * ```
 */
export function etsFit(y: ArrayLike<number>, spec: EtsSpec, options: EtsFitOptions = {}): EtsFit {
  const { algorithm, fit } = etsFitAlgorithm(y, spec, options)
  const run = runAlgorithm(algorithm, { maxSteps: options.maxIterations ?? 300 })
  return fit(run.state, run.steps, run.status === 'converged')
}

/** Forecasts of an ETS model. */
export interface EtsForecast {
  readonly mean: readonly number[]
  /** Standard errors for additive models (undefined for multiplicative ones, which have no closed form). */
  readonly se?: readonly number[]
  /** Central intervals by level, with the standard errors. */
  readonly intervals?: readonly {
    readonly level: number
    readonly lower: readonly number[]
    readonly upper: readonly number[]
  }[]
}

/**
 * Forecast `steps` ahead: the damped or undamped trend extrapolated from the final level and trend, times or plus the
 * seasonal value of the matching phase. For models with additive error, trend and seasonal the prediction variance is
 * sigma^2 (1 + sum_(j<h) c_j^2) with c_j the weights of the innovations in the state-space form (Hyndman et al.,
 * class 1), and intervals follow.
 *
 * @param fit - The fitted model.
 * @param steps - How many steps ahead.
 * @param levels - Interval levels (default `[0.95]`).
 * @returns The forecast.
 * @example
 * ```ts
 * import { etsForecast } from 'mdml-engine/timeseries'
 * const fit = { spec: { error: 'add', trend: 'add', seasonal: 'none' }, params: { alpha: 0.5, beta: 0.2, level0: 0, trend0: 0 },
 *   logLikelihood: 0, sigma2: 1, nparams: 4, aic: 0, bic: 0, nobs: 10, final: { level: 10, trend: 2, seasonal: [] },
 *   converged: true, iterations: 0 } as const
 * etsForecast(fit, 3).mean // => [12, 14, 16]
 * ```
 */
export function etsForecast(fit: EtsFit, steps: number, levels: readonly number[] = [0.95]): EtsForecast {
  const r = resolve(fit.spec)
  const { level, trend, seasonal } = fit.final
  const phi = r.damped ? (fit.params.phi ?? 1) : 1
  const mean: number[] = []
  let cumulative = 0
  let power = 1
  for (let h = 1; h <= steps; h++) {
    power *= phi
    cumulative += power
    const base =
      fit.spec.trend === 'add'
        ? level + cumulative * trend
        : fit.spec.trend === 'mul'
          ? level * trend ** cumulative
          : level
    const season = r.m === 0 ? 0 : (seasonal[(h - 1) % r.m] ?? 0)
    mean.push(fit.spec.seasonal === 'mul' ? base * season : base + season)
  }
  const additive = fit.spec.error === 'add' && fit.spec.trend !== 'mul' && fit.spec.seasonal !== 'mul'
  if (!additive) return { mean }
  // The weight of the innovation j steps back in the form y_(t+h) = ... + sum_(j<h) c_j e_(t+h-j) + e_(t+h):
  // c_j = alpha + beta (phi + ... + phi^j) + gamma [m divides j].
  const alpha = fit.params.alpha
  const beta = fit.params.beta ?? 0
  const gamma = fit.params.gamma ?? 0
  const hasTrend = fit.spec.trend === 'add'
  let cumPhi = 0
  let ph = 1
  let sumSq = 0
  const variance: number[] = [fit.sigma2]
  for (let h = 2; h <= steps; h++) {
    const j = h - 1
    ph *= phi
    cumPhi += ph
    let c = alpha + (hasTrend ? beta * cumPhi : 0)
    if (r.m > 0 && j % r.m === 0) c += gamma
    sumSq += c * c
    variance.push(fit.sigma2 * (1 + sumSq))
  }
  const se = variance.map(Math.sqrt)
  return {
    mean,
    se,
    intervals: levels.map((level2) => {
      const q = item(ndtri((1 + level2) / 2))
      return {
        level: level2,
        lower: mean.map((v, k) => v - q * (se[k] ?? 0)),
        upper: mean.map((v, k) => v + q * (se[k] ?? 0)),
      }
    }),
  }
}

/**
 * Sample paths of an ETS model from its final state (Hyndman et al.'s simulation of future values): each step draws
 * the innovation e ~ N(0, sigma2), forms y = yhat + e for additive error or y = yhat (1 + e) for multiplicative error,
 * and advances the states by the model's own recursion. It works for every combination of error, trend and seasonal
 * type, so it gives prediction intervals where no closed form exists.
 *
 * @param fit - The fitted model (spec, parameters, sigma2 and final state).
 * @param steps - The horizon.
 * @param paths - The number of paths.
 * @param key - The random key.
 * @returns The paths, `[path][step]`.
 * @example
 * ```ts
 * import { etsSimulate } from 'mdml-engine/timeseries'
 * import { key } from 'mdml-engine/random'
 * const fit = { spec: { error: 'mul', trend: 'none', seasonal: 'none' }, params: { alpha: 0.3, level0: 10 }, sigma2: 0.01,
 *   final: { level: 10, trend: 0, seasonal: [] } } as never
 * etsSimulate(fit, 3, 4, key(0)).length // => 4
 * ```
 */
export function etsSimulate(
  fit: Pick<EtsFit, 'spec' | 'params' | 'sigma2' | 'final'>,
  steps: number,
  paths: number,
  key: Key,
): number[][] {
  const r = resolve(fit.spec)
  const z = toFlat(normal(key, [paths * steps]))
  const sd = Math.sqrt(fit.sigma2)
  const out: number[][] = []
  for (let p = 0; p < paths; p++) {
    const run: Run = {
      level: fit.final.level,
      trend: fit.final.trend,
      ring: Float64Array.from({ length: r.m }, (_, j) => fit.final.seasonal[j] ?? 0),
      pos: 0,
    }
    const path: number[] = []
    for (let h = 0; h < steps; h++) {
      const probe: Run = { level: run.level, trend: run.trend, ring: Float64Array.from(run.ring), pos: run.pos }
      const yhat = advance(r, fit.params, probe, 0)
      const e = sd * (z[p * steps + h] ?? 0)
      const y = fit.spec.error === 'mul' ? yhat * (1 + e) : yhat + e
      advance(r, fit.params, run, y)
      path.push(y)
    }
    out.push(path)
  }
  return out
}
