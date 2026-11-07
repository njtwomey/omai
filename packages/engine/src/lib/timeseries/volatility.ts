/**
 * Conditional volatility models in the form of the `arch` package: GARCH(p, o, q) with a constant or zero mean, where
 * the variance follows sigma2_t = omega + sum alpha_i e2_(t-i) + sum gamma_k e2_(t-k) [e_(t-k) < 0] + sum beta_j
 * sigma2_(t-j) (ARCH is q = 0, GJR-GARCH is o > 0), with Gaussian or standardised Student-t innovations. The variance
 * recursion runs one observation per step as an `Algorithm`; the maximum-likelihood fit is a traced BFGS algorithm
 * on parameters mapped into the stationary region; variance forecasts and simulation complete it.
 */

import { MdmlError, meanOf, varianceOf } from '../base/index.ts'
import type { BfgsState } from '../optim/index.ts'
import { type Key, normal, studentT } from '../random/index.ts'
import type { SpaceValue } from '../space/index.ts'
import { toFlat } from '../tensor/index.ts'
import { type Algorithm, channels, runAlgorithm, stateFromJson, stateToJson } from '../trace/index.ts'
import {
  inverseDiagonalRoots,
  type LikelihoodState,
  likelihoodAlgorithm,
  likelihoodSlot,
  numericalHessian,
} from './_ml.ts'
import { asSeries } from './_series.ts'
import { backcastOf, studentConstant } from './_volatility.ts'

/** The structure of a GARCH model; the names are the `arch` package's. */
export interface GarchSpec {
  /** Lags of the squared residual (default 1). */
  readonly p?: number
  /** Lags of the asymmetric term e2 [e < 0] (default 0; above 0 gives GJR-GARCH). */
  readonly o?: number
  /** Lags of the conditional variance (default 1; 0 gives ARCH(p)). */
  readonly q?: number
  /** A constant mean or zero mean (default constant). */
  readonly mean?: 'constant' | 'zero'
  /** Gaussian or standardised Student-t innovations (default Gaussian). */
  readonly distribution?: 'normal' | 't'
}

/** The parameters of a GARCH model. */
export interface GarchParameters {
  readonly mu: number
  readonly omega: number
  readonly alpha: readonly number[]
  readonly gamma: readonly number[]
  readonly beta: readonly number[]
  /** Degrees of freedom of the Student-t distribution (above 2). */
  readonly nu?: number
}

interface Resolved {
  readonly p: number
  readonly o: number
  readonly q: number
  readonly constantMean: boolean
  readonly student: boolean
}

function resolve(spec: GarchSpec): Resolved {
  const p = spec.p ?? 1
  const o = spec.o ?? 0
  const q = spec.q ?? 1
  for (const v of [p, o, q])
    if (!(Number.isInteger(v) && v >= 0)) throw new MdmlError('value', 'garch: lags must be non-negative integers')
  if (p + o === 0) throw new MdmlError('value', 'garch: need at least one lag of the residual (p + o > 0)')
  return { p, o, q, constantMean: spec.mean !== 'zero', student: spec.distribution === 't' }
}

/** One step of the variance recursion given the lags (most recent first). */
function nextVariance(
  r: Resolved,
  par: GarchParameters,
  sq: ArrayLike<number>,
  asym: ArrayLike<number>,
  sigma: ArrayLike<number>,
): number {
  let v = par.omega
  for (let i = 0; i < r.p; i++) v += (par.alpha[i] ?? 0) * (sq[i] ?? 0)
  for (let k = 0; k < r.o; k++) v += (par.gamma[k] ?? 0) * (asym[k] ?? 0)
  for (let j = 0; j < r.q; j++) v += (par.beta[j] ?? 0) * (sigma[j] ?? 0)
  return v
}

/**
 * The conditional variances of residuals under given parameters, with the pre-sample squared residuals and variances
 * set to the `arch` backcast.
 *
 * @param eps - The residuals (the series less its mean).
 * @param spec - The structure.
 * @param params - The parameters.
 * @returns sigma2_t for every residual.
 * @example
 * ```ts
 * import { garchVariance } from 'mdml-engine/timeseries'
 * // ARCH(1): sigma2_t = 1 + 0.5 e_(t-1)^2, the first from the backcast of the sample.
 * Array.from(garchVariance([2, 0], { p: 1, q: 0 }, { mu: 0, omega: 1, alpha: [0.5], gamma: [], beta: [] })).slice(1) // => [3]
 * ```
 */
export function garchVariance(eps: ArrayLike<number>, spec: GarchSpec, params: GarchParameters): Float64Array {
  const r = resolve(spec)
  const n = eps.length
  const back = backcastOf(eps)
  const out = new Float64Array(n)
  const lagSq = Math.max(r.p, r.o)
  const sq = new Float64Array(lagSq).fill(back)
  const asym = new Float64Array(r.o).fill(0.5 * back)
  const sigma = new Float64Array(r.q).fill(back)
  for (let t = 0; t < n; t++) {
    const v = nextVariance(r, params, sq, asym, sigma)
    out[t] = v
    const e = eps[t] ?? 0
    for (let i = lagSq - 1; i > 0; i--) sq[i] = sq[i - 1] ?? 0
    for (let k = r.o - 1; k > 0; k--) asym[k] = asym[k - 1] ?? 0
    for (let j = r.q - 1; j > 0; j--) sigma[j] = sigma[j - 1] ?? 0
    if (lagSq > 0) sq[0] = e * e
    if (r.o > 0) asym[0] = e < 0 ? e * e : 0
    if (r.q > 0) sigma[0] = v
  }
  return out
}

/** The log-density of one residual e with conditional variance v. */
function logDensity(student: boolean, nu: number, lgammaTerm: number, e: number, v: number): number {
  if (!student) return -0.5 * (Math.log(2 * Math.PI) + Math.log(v) + (e * e) / v)
  return lgammaTerm - 0.5 * Math.log(v) - ((nu + 1) / 2) * Math.log(1 + (e * e) / (v * (nu - 2)))
}

/**
 * The log-likelihood of a series under a GARCH model.
 *
 * @param y - The series.
 * @param spec - The structure.
 * @param params - The parameters.
 * @returns The Gaussian or Student-t log-likelihood.
 * @example
 * ```ts
 * import { garchLogLikelihood } from 'mdml-engine/timeseries'
 * // ARCH(1) with alpha = 0 is i.i.d. N(mu, omega): one observation at its mean.
 * garchLogLikelihood([0], { q: 0, mean: 'zero' }, { mu: 0, omega: 1, alpha: [0], gamma: [], beta: [] }).toFixed(6) // => '-0.918939'
 * ```
 */
export function garchLogLikelihood(y: ArrayLike<number>, spec: GarchSpec, params: GarchParameters): number {
  const r = resolve(spec)
  const x = asSeries(Float64Array.from(y), 'garch')
  const eps = Float64Array.from(x, (v) => v - (r.constantMean ? params.mu : 0))
  const sigma2 = garchVariance(eps, spec, params)
  const nu = params.nu ?? Number.NaN
  const c = r.student ? studentConstant(nu) : 0
  let ll = 0
  for (let t = 0; t < x.length; t++) ll += logDensity(r.student, nu, c, eps[t] ?? 0, sigma2[t] ?? Number.NaN)
  return ll
}

/** The state of {@link garchFilterAlgorithm}. */
export interface GarchFilterState {
  /** Observations consumed. */
  readonly t: number
  /** Lags of e2, of e2 [e < 0] and of sigma2, most recent first. */
  readonly sq: readonly number[]
  readonly asym: readonly number[]
  readonly sigma: readonly number[]
  /** The variance used for the latest observation, and its residual. */
  readonly variance: number
  readonly residual: number
  /** The log-likelihood so far. */
  readonly logLikelihood: number
}

/**
 * The GARCH variance recursion as an algorithm, one observation per step: it predicts the variance of the observation
 * from the lagged squared residuals and variances, scores the observation under the Gaussian or Student-t density, and
 * updates the lags. It records the variance, the standardised residual and the running log-likelihood.
 *
 * @param y - The series.
 * @param spec - The structure.
 * @param params - The parameters.
 * @returns The algorithm; it is done after the last observation.
 * @example
 * ```ts
 * import { garchFilterAlgorithm } from 'mdml-engine/timeseries'
 * import { runAlgorithm } from 'mdml-engine/trace'
 * const params = { mu: 0, omega: 0.1, alpha: [0.1], gamma: [], beta: [0.8] }
 * runAlgorithm(garchFilterAlgorithm([0.5, -1, 0.2, 2], { mean: 'zero' }, params), { maxSteps: 10 }).steps // => 4
 * ```
 */
export function garchFilterAlgorithm(
  y: ArrayLike<number>,
  spec: GarchSpec,
  params: GarchParameters,
): Algorithm<GarchFilterState> {
  const r = resolve(spec)
  const x = asSeries(Float64Array.from(y), 'garch')
  const eps = Float64Array.from(x, (v) => v - (r.constantMean ? params.mu : 0))
  const back = backcastOf(eps)
  const lagSq = Math.max(r.p, r.o)
  const nu = params.nu ?? Number.NaN
  const c = r.student ? studentConstant(nu) : 0
  return {
    name: 'garch-filter',
    config: {
      p: r.p,
      o: r.o,
      q: r.q,
      mean: r.constantMean ? 'constant' : 'zero',
      distribution: r.student ? 't' : 'normal',
    },
    schema: channels({
      metrics: {
        variance: { description: 'the conditional variance of this observation' },
        standardisedResidual: { description: 'the residual over the conditional standard deviation' },
        logLikelihood: { description: 'the log-likelihood of the observations so far', direction: 'higher' },
      },
    }),
    init: () => ({
      t: 0,
      sq: new Array<number>(lagSq).fill(back),
      asym: new Array<number>(r.o).fill(0.5 * back),
      sigma: new Array<number>(r.q).fill(back),
      variance: Number.NaN,
      residual: Number.NaN,
      logLikelihood: 0,
    }),
    step: (s) => {
      const v = nextVariance(r, params, s.sq, s.asym, s.sigma)
      const e = eps[s.t] ?? Number.NaN
      return {
        t: s.t + 1,
        sq: [e * e, ...s.sq].slice(0, lagSq),
        asym: [e < 0 ? e * e : 0, ...s.asym].slice(0, r.o),
        sigma: [v, ...s.sigma].slice(0, r.q),
        variance: v,
        residual: e,
        logLikelihood: s.logLikelihood + logDensity(r.student, nu, c, e, v),
      }
    },
    done: (s) => (s.t >= x.length ? 'converged' : undefined),
    observe: (s) => ({
      metrics: {
        variance: s.variance,
        standardisedResidual: s.residual / Math.sqrt(s.variance),
        logLikelihood: s.logLikelihood,
      },
    }),
    serialise: (s) => stateToJson(s),
    deserialise: (j) => stateFromJson(j) as GarchFilterState,
  }
}

/** Layout and transforms of the unconstrained parameter vector of a fit (on data scaled to unit spread). */
function layoutOf(r: Resolved): {
  readonly size: number
  readonly toParams: (u: ArrayLike<number>) => GarchParameters
  readonly start: (variance: number, mean: number) => number[]
} {
  const m = r.p + r.o + r.q
  const size = (r.constantMean ? 1 : 0) + 1 + m + (r.student ? 1 : 0)
  return {
    size,
    toParams: (u) => {
      let k = 0
      const mu = r.constantMean ? (u[k++] ?? 0) : 0
      const omega = Math.exp(u[k++] ?? 0)
      // Shares of the persistence budget: a softmax over the m coefficients and one slack entry, so the weights sum below 1.
      const v = Array.from({ length: m }, (_, i) => u[k + i] ?? 0)
      k += m
      const top = Math.max(0, ...v)
      const e = [...v.map((x) => Math.exp(x - top)), Math.exp(-top)]
      const total = e.reduce((a, b) => a + b, 0)
      const w = e.slice(0, m).map((x) => x / total)
      const nu = r.student ? 2.01 + Math.exp(u[k] ?? 0) : undefined
      return {
        mu,
        omega,
        alpha: w.slice(0, r.p),
        gamma: w.slice(r.p, r.p + r.o).map((x) => 2 * x),
        beta: w.slice(r.p + r.o),
        ...(nu === undefined ? {} : { nu }),
      }
    },
    start: (variance, mean) => {
      const out: number[] = []
      if (r.constantMean) out.push(mean)
      out.push(Math.log(variance * 0.1))
      // Start near persistence 0.9: 0.05 on the ARCH terms, 0.025 (gamma / 2) on the asymmetric ones and the rest on the
      // GARCH lags (or all of it on the residual terms for a pure ARCH).
      const resid = r.p + r.o
      const raw = Array.from({ length: resid + r.q }, (_, i) => {
        if (r.q === 0) return 0.9 / resid
        if (i < r.p) return 0.05 / r.p
        if (i < resid) return 0.025 / r.o
        return 0.85 / r.q
      })
      const slack = 1 - raw.reduce((a, b) => a + b, 0)
      for (const x of raw) out.push(Math.log(x / slack))
      if (r.student) out.push(Math.log(6))
      return out
    },
  }
}

/** A fitted GARCH model: plain data. */
export interface GarchFit {
  readonly spec: GarchSpec
  readonly params: GarchParameters
  readonly logLikelihood: number
  readonly aic: number
  readonly bic: number
  readonly nobs: number
  /** The conditional variance of every observation. */
  readonly conditionalVariance: readonly number[]
  /** The residuals over the conditional standard deviation. */
  readonly standardisedResiduals: readonly number[]
  /** The persistence sum alpha + gamma / 2 + beta. */
  readonly persistence: number
  /** The unconditional variance omega / (1 - persistence). */
  readonly unconditionalVariance: number
  /** Standard errors from the numerical Hessian, in the order mu, omega, alpha, gamma, beta, nu. */
  readonly stdErrors: readonly number[]
  readonly converged: boolean
  readonly iterations: number
}

/** Options of {@link garchFitAlgorithm}. */
export interface GarchFitOptions {
  /** The solver, a value of `smoothSolvers.space` (BFGS by default with the traced `garchFitAlgorithm`; see {@link garchFitSlot}). */
  readonly solver?: SpaceValue
  /** The run's key (for the stochastic solver's batches). */
  readonly key?: Key
  /** The BFGS iteration cap (default 300). */
  readonly maxIterations?: number
  /** The gradient tolerance (default 1e-6). */
  readonly gradTol?: number
}

/** A GARCH fit as an algorithm plus the map from its final state to the fitted model. */
export interface GarchProblem {
  readonly algorithm: Algorithm<BfgsState>
  readonly fit: (state: BfgsState, iterations?: number, converged?: boolean) => GarchFit
}

const flatten = (p: GarchParameters, r: Resolved): number[] => [
  ...(r.constantMean ? [p.mu] : []),
  p.omega,
  ...p.alpha,
  ...p.gamma,
  ...p.beta,
  ...(p.nu === undefined ? [] : [p.nu]),
]

/** The objective, start and map from the parameter vector to the fit, shared by every way of minimising it. */
function prepareGarchFit(y: ArrayLike<number>, spec: GarchSpec) {
  const r = resolve(spec)
  const x = asSeries(Float64Array.from(y), 'garchFit')
  const n = x.length
  if (n < 10 * (r.p + r.o + r.q + 2)) throw new MdmlError('value', 'garchFit: the series is too short for the model')
  const centre = meanOf(x)
  const scale = Math.sqrt(varianceOf(x, 0)) || 1
  const z = Float64Array.from(x, (v) => v / scale)
  const layout = layoutOf(r)
  const objective = (u: Float64Array): number => {
    // The likelihood of the original series: the scaled one less n log(scale), so the traced metric is comparable.
    const ll = garchLogLikelihood(z, spec, layout.toParams(u))
    return Number.isFinite(ll) ? -ll / n + Math.log(scale) : 1e10
  }
  const start = layout.start(varianceOf(z, 0), centre / scale)
  return {
    r,
    n,
    objective,
    start,
    fit: (u: ArrayLike<number>, iterations = 0, converged = true): GarchFit => {
      const scaled = layout.toParams(u)
      const params: GarchParameters = {
        mu: scaled.mu * scale,
        omega: scaled.omega * scale * scale,
        alpha: scaled.alpha,
        gamma: scaled.gamma,
        beta: scaled.beta,
        ...(scaled.nu === undefined ? {} : { nu: scaled.nu }),
      }
      const eps = Float64Array.from(x, (v) => v - (r.constantMean ? params.mu : 0))
      const variance = garchVariance(eps, spec, params)
      const ll = garchLogLikelihood(x, spec, params)
      const persistence =
        params.alpha.reduce((a, b) => a + b, 0) +
        params.gamma.reduce((a, b) => a + b, 0) / 2 +
        params.beta.reduce((a, b) => a + b, 0)
      const k = layout.size
      const natural = flatten(params, r)
      const toNatural = (v: Float64Array): GarchParameters => {
        let at = 0
        const mu = r.constantMean ? (v[at++] ?? 0) : 0
        const omega = v[at++] ?? Number.NaN
        const alpha = Array.from(v.subarray(at, at + r.p))
        at += r.p
        const gamma = Array.from(v.subarray(at, at + r.o))
        at += r.o
        const beta = Array.from(v.subarray(at, at + r.q))
        at += r.q
        return { mu, omega, alpha, gamma, beta, ...(r.student ? { nu: v[at] ?? Number.NaN } : {}) }
      }
      const stdErrors = inverseDiagonalRoots(
        numericalHessian((v) => -garchLogLikelihood(x, spec, toNatural(v)), natural, 1e-4),
        k,
      )
      return {
        spec,
        params,
        logLikelihood: ll,
        aic: -2 * ll + 2 * k,
        bic: -2 * ll + k * Math.log(n),
        nobs: n,
        conditionalVariance: Array.from(variance),
        standardisedResiduals: Array.from(eps, (e, t) => e / Math.sqrt(variance[t] ?? Number.NaN)),
        persistence,
        unconditionalVariance: params.omega / (1 - persistence),
        stdErrors,
        converged,
        iterations,
      }
    },
  }
}

/**
 * The maximum-likelihood fit of a GARCH model as a traced algorithm: BFGS on the unconstrained parameters (the
 * constant term on a log scale, the coefficients as shares of a persistence budget below one, the degrees of freedom
 * above two), on the series scaled to unit spread for conditioning. One iteration per step, with the log-likelihood
 * of the original series recorded.
 *
 * @param y - The series (returns, typically).
 * @param spec - The structure.
 * @param options - The iteration cap and gradient tolerance.
 * @returns The algorithm and the map from its state to the fit.
 * @example
 * ```ts
 * import { garchFitAlgorithm, garchSimulate } from 'mdml-engine/timeseries'
 * import { key } from 'mdml-engine/random'
 * const y = garchSimulate(key(2), 200, {}, { mu: 0, omega: 0.2, alpha: [0.1], gamma: [], beta: [0.8] })
 * garchFitAlgorithm(y, {}).algorithm.name // => 'garch-fit'
 * ```
 */
export function garchFitAlgorithm(y: ArrayLike<number>, spec: GarchSpec, options: GarchFitOptions = {}): GarchProblem {
  const prepared = prepareGarchFit(y, spec)
  const algorithm = likelihoodAlgorithm(
    'garch-fit',
    { p: prepared.r.p, o: prepared.r.o, q: prepared.r.q, spec },
    prepared.objective,
    prepared.start,
    prepared.n,
    options,
  )
  return { algorithm, fit: (state, iterations, converged) => prepared.fit(state.x, iterations, converged) }
}

/**
 * The GARCH fit over the solver slot: the likelihood is a smooth objective of the unconstrained parameters and any
 * solver of `smoothSolvers` minimises it (BFGS by default).
 *
 * @param y - The series.
 * @param spec - The structure.
 * @param options - The solver, gradient tolerance, iteration cap and key.
 * @returns The algorithm and the fit run from it.
 * @example
 * ```ts
 * import { garchFitSlot, garchSimulate } from 'mdml-engine/timeseries'
 * import { key } from 'mdml-engine/random'
 * const y = garchSimulate(key(2), 300, {}, { mu: 0, omega: 0.2, alpha: [0.1], gamma: [], beta: [0.8] })
 * garchFitSlot(y, { mean: 'zero' }, { solver: { kind: 'lbfgs', gradTol: 1e-8 } }).fit.persistence < 1 // => true
 * ```
 */
export function garchFitSlot(
  y: ArrayLike<number>,
  spec: GarchSpec = {},
  options: GarchFitOptions = {},
): { readonly algorithm: Algorithm<LikelihoodState>; readonly fit: GarchFit } {
  const prepared = prepareGarchFit(y, spec)
  const algorithm = likelihoodSlot(
    'garch-fit',
    { p: prepared.r.p, o: prepared.r.o, q: prepared.r.q, spec },
    prepared.objective,
    prepared.start,
    prepared.n,
    {
      ...(options.solver === undefined ? {} : { solver: options.solver }),
      ...(options.gradTol === undefined ? {} : { tol: options.gradTol }),
      ...(options.key === undefined ? {} : { key: options.key }),
    },
  )
  const run = runAlgorithm(algorithm, { maxSteps: options.maxIterations ?? 300 })
  return { algorithm, fit: prepared.fit(run.state.u, run.steps, run.status === 'converged') }
}

/**
 * Fit a GARCH model by maximum likelihood.
 *
 * @param y - The series (returns, typically).
 * @param spec - The structure.
 * @param options - The iteration cap and gradient tolerance.
 * @returns The fitted model.
 * @example
 * ```ts
 * import { garchSimulate, garchFit } from 'mdml-engine/timeseries'
 * import { key } from 'mdml-engine/random'
 * const y = garchSimulate(key(1), 600, {}, { mu: 0, omega: 0.2, alpha: [0.1], gamma: [], beta: [0.8] })
 * garchFit(y, { mean: 'zero' }).persistence < 1 // => true
 * ```
 */
export function garchFit(y: ArrayLike<number>, spec: GarchSpec = {}, options: GarchFitOptions = {}): GarchFit {
  if (options.solver !== undefined) return garchFitSlot(y, spec, options).fit
  const { algorithm, fit } = garchFitAlgorithm(y, spec, options)
  const run = runAlgorithm(algorithm, { maxSteps: options.maxIterations ?? 300 })
  return fit(run.state, run.steps, run.status === 'converged')
}

/**
 * Forecast the conditional variance `steps` ahead: the first from the fitted recursion, then with the expected squared
 * residual equal to the forecast variance (and half of it for the asymmetric term).
 *
 * @param fit - The fitted model.
 * @param y - The series it was fitted to.
 * @param steps - How many steps ahead.
 * @returns The variances and volatilities; they tend to the unconditional variance when the model is stationary.
 * @example
 * ```ts
 * import { garchForecast } from 'mdml-engine/timeseries'
 * const fit = { spec: { mean: 'zero' }, params: { mu: 0, omega: 0.1, alpha: [0.1], gamma: [], beta: [0.8] } } as never
 * garchForecast(fit, [0.5, -1, 0.2, 2], 3).variance.length // => 3
 * ```
 */
export function garchForecast(
  fit: Pick<GarchFit, 'spec' | 'params'>,
  y: ArrayLike<number>,
  steps: number,
): { readonly variance: readonly number[]; readonly volatility: readonly number[] } {
  const r = resolve(fit.spec)
  const x = asSeries(Float64Array.from(y), 'garchForecast')
  const eps = Float64Array.from(x, (v) => v - (r.constantMean ? fit.params.mu : 0))
  const sigma2 = garchVariance(eps, fit.spec, fit.params)
  const n = x.length
  const forecastVariance: number[] = []
  const back = backcastOf(eps)
  const sqAt = (t: number): number => (t < 0 ? back : t < n ? (eps[t] ?? 0) ** 2 : (forecastVariance[t - n] ?? 0))
  const asymAt = (t: number): number =>
    t < 0 ? 0.5 * back : t < n ? ((eps[t] ?? 0) < 0 ? (eps[t] ?? 0) ** 2 : 0) : 0.5 * (forecastVariance[t - n] ?? 0)
  const sigmaAt = (t: number): number => (t < 0 ? back : t < n ? (sigma2[t] ?? 0) : (forecastVariance[t - n] ?? 0))
  for (let h = 0; h < steps; h++) {
    const t = n + h
    let v = fit.params.omega
    for (let i = 0; i < r.p; i++) v += (fit.params.alpha[i] ?? 0) * sqAt(t - 1 - i)
    for (let k = 0; k < r.o; k++) v += (fit.params.gamma[k] ?? 0) * asymAt(t - 1 - k)
    for (let j = 0; j < r.q; j++) v += (fit.params.beta[j] ?? 0) * sigmaAt(t - 1 - j)
    forecastVariance.push(v)
  }
  return { variance: forecastVariance, volatility: forecastVariance.map(Math.sqrt) }
}

/**
 * Simulate a GARCH process with a burn-in.
 *
 * @param key - The random key.
 * @param n - The number of observations returned.
 * @param spec - The structure.
 * @param params - The parameters.
 * @param burnIn - Observations simulated and discarded first (default 200).
 * @returns The series.
 * @example
 * ```ts
 * import { garchSimulate } from 'mdml-engine/timeseries'
 * import { key } from 'mdml-engine/random'
 * garchSimulate(key(0), 5, { mean: 'zero' }, { mu: 0, omega: 0.1, alpha: [0.1], gamma: [], beta: [0.8] }).length // => 5
 * ```
 */
export function garchSimulate(key: Key, n: number, spec: GarchSpec, params: GarchParameters, burnIn = 200): number[] {
  const r = resolve(spec)
  const total = n + burnIn
  const z = r.student
    ? Array.from(
        toFlat(studentT(key, [total], params.nu ?? 8)),
        (v) => v / Math.sqrt((params.nu ?? 8) / ((params.nu ?? 8) - 2)),
      )
    : toFlat(normal(key, [total]))
  const lagSq = Math.max(r.p, r.o)
  const stationary =
    params.alpha.reduce((a, b) => a + b, 0) +
    params.gamma.reduce((a, b) => a + b, 0) / 2 +
    params.beta.reduce((a, b) => a + b, 0)
  const start = params.omega / Math.max(1 - stationary, 1e-3)
  let sq = new Array<number>(lagSq).fill(start)
  let asym = new Array<number>(r.o).fill(start / 2)
  let sigma = new Array<number>(r.q).fill(start)
  const out: number[] = []
  for (let t = 0; t < total; t++) {
    const v = nextVariance(r, params, sq, asym, sigma)
    const e = Math.sqrt(v) * (z[t] ?? 0)
    sq = [e * e, ...sq].slice(0, lagSq)
    asym = [e < 0 ? e * e : 0, ...asym].slice(0, r.o)
    sigma = [v, ...sigma].slice(0, r.q)
    if (t >= burnIn) out.push(e + (r.constantMean ? params.mu : 0))
  }
  return out
}
