/**
 * Asymmetric volatility models beside GARCH and GJR: EGARCH(p, o, q), where the log variance follows
 * ln s2_t = omega + sum alpha_i (|z_(t-i)| - E|z|) + sum gamma_k z_(t-k) + sum beta_j ln s2_(t-j) with z the standardised
 * residual (so no positivity constraints are needed and a negative shock raises volatility more when gamma < 0), and
 * APARCH(p, q) of Ding, Granger and Engle, s_t^delta = omega + sum alpha_i (|e| - gamma_i e)^delta + sum beta_j
 * s_(t-j)^delta with the power delta estimated. Both are fitted by maximum likelihood over the solver slot (BFGS by
 * default), forecast by simulation beyond one step, and summarised by the news-impact curve of Engle and Ng: the next
 * variance as a function of today's shock, with the variance held at its unconditional level, which makes the
 * asymmetry visible (a symmetric parabola for GARCH, tilted for GJR, EGARCH and APARCH).
 */

import { MdmlError, meanOf, varianceOf } from '../base/index.ts'
import { lgamma } from '../numerics/special/index.ts'
import { type Key, normal, studentT } from '../random/index.ts'
import type { SpaceValue } from '../space/index.ts'
import { item, toFlat } from '../tensor/index.ts'
import { type Algorithm, runAlgorithm } from '../trace/index.ts'
import { type LikelihoodState, likelihoodSlot } from './_ml.ts'
import { asSeries } from './_series.ts'
import { backcastOf, studentConstant } from './_volatility.ts'
import type { GarchFit, GarchParameters, GarchSpec } from './volatility.ts'

/** Which asymmetric model. */
export type AsymmetricKind = 'egarch' | 'aparch'

/** The structure of an EGARCH or APARCH model (the lags are those of the `arch` package). */
export interface AsymmetricSpec {
  readonly kind: AsymmetricKind
  /** Lags of |z| (EGARCH) or of the power term (APARCH); default 1. */
  readonly p?: number
  /** Lags of the signed term z (EGARCH only; default 1); APARCH has one gamma per alpha. */
  readonly o?: number
  /** Lags of the log variance (EGARCH) or of s^delta (APARCH); default 1. */
  readonly q?: number
  readonly mean?: 'constant' | 'zero'
  readonly distribution?: 'normal' | 't'
}

/** The parameters of an asymmetric model. */
export interface AsymmetricParameters {
  readonly mu: number
  readonly omega: number
  readonly alpha: readonly number[]
  /** The asymmetry coefficients (signed z terms for EGARCH; |gamma| < 1 shifts for APARCH). */
  readonly gamma: readonly number[]
  readonly beta: readonly number[]
  /** The power of the APARCH model. */
  readonly delta?: number
  readonly nu?: number
}

interface VolatilitySpec {
  readonly egarch: boolean
  readonly p: number
  readonly o: number
  readonly q: number
  readonly constantMean: boolean
  readonly student: boolean
}

function shapeOf(spec: AsymmetricSpec): VolatilitySpec {
  const egarch = spec.kind === 'egarch'
  const p = spec.p ?? 1
  const q = spec.q ?? 1
  const o = egarch ? (spec.o ?? 1) : p
  for (const v of [p, o, q])
    if (!(Number.isInteger(v) && v >= 0)) throw new MdmlError('value', 'volatility: lags must be non-negative integers')
  if (p + (egarch ? o : 0) === 0) throw new MdmlError('value', 'volatility: need at least one lag of the shock')
  return { egarch, p, o, q, constantMean: spec.mean !== 'zero', student: spec.distribution === 't' }
}

/** E|z| of a standardised innovation: sqrt(2 / pi) for the normal, the Student-t value with unit variance otherwise. */
function expectedAbs(student: boolean, nu: number): number {
  if (!student) return Math.sqrt(2 / Math.PI)
  return (
    (2 * Math.sqrt(nu - 2) * Math.exp(item(lgamma((nu + 1) / 2)) - item(lgamma(nu / 2)))) /
    ((nu - 1) * Math.sqrt(Math.PI))
  )
}

/**
 * The conditional variances of residuals under an EGARCH or APARCH model. The pre-sample log variance (EGARCH) or
 * variance power (APARCH) is the `arch` backcast, and the pre-sample shocks are at their expectation.
 *
 * @param eps - The residuals.
 * @param spec - The structure.
 * @param params - The parameters.
 * @returns s2_t for every residual.
 * @example
 * ```ts
 * import { asymmetricVariance } from 'mdml-engine/timeseries'
 * // EGARCH(1,1,1) with all coefficients zero except omega: constant variance exp(omega).
 * Array.from(asymmetricVariance([1, -1, 2], { kind: 'egarch' }, { mu: 0, omega: 0.5, alpha: [0], gamma: [0], beta: [0] }))[2]?.toFixed(6) // => '1.648721'
 * ```
 */
export function asymmetricVariance(
  eps: ArrayLike<number>,
  spec: AsymmetricSpec,
  params: AsymmetricParameters,
): Float64Array {
  const r = shapeOf(spec)
  const n = eps.length
  const out = new Float64Array(n)
  const back = backcastOf(eps)
  if (r.egarch) {
    const kappa = expectedAbs(r.student, params.nu ?? 8)
    const lnv = new Float64Array(n)
    const z = new Float64Array(n)
    const lnBack = Math.log(back)
    for (let t = 0; t < n; t++) {
      let v = params.omega
      for (let i = 0; i < r.p; i++)
        v += (params.alpha[i] ?? 0) * (t - 1 - i < 0 ? 0 : Math.abs(z[t - 1 - i] ?? 0) - kappa)
      for (let k = 0; k < r.o; k++) v += (params.gamma[k] ?? 0) * (t - 1 - k < 0 ? 0 : (z[t - 1 - k] ?? 0))
      for (let j = 0; j < r.q; j++) v += (params.beta[j] ?? 0) * (t - 1 - j < 0 ? lnBack : (lnv[t - 1 - j] ?? 0))
      lnv[t] = Math.min(Math.max(v, -700), 700)
      out[t] = Math.exp(lnv[t] ?? 0)
      z[t] = (eps[t] ?? 0) / Math.sqrt(out[t] ?? 1)
    }
    return out
  }
  const delta = params.delta ?? 2
  const sd = new Float64Array(n)
  const backPower = back ** (delta / 2)
  for (let t = 0; t < n; t++) {
    let v = params.omega
    for (let i = 0; i < r.p; i++) {
      const e = t - 1 - i < 0 ? null : (eps[t - 1 - i] ?? 0)
      v +=
        (params.alpha[i] ?? 0) *
        (e === null ? backPower : Math.max(Math.abs(e) - (params.gamma[i] ?? 0) * e, 0) ** delta)
    }
    for (let j = 0; j < r.q; j++) v += (params.beta[j] ?? 0) * (t - 1 - j < 0 ? backPower : (sd[t - 1 - j] ?? 0))
    sd[t] = Math.max(v, 1e-300)
    out[t] = (sd[t] ?? 0) ** (2 / delta)
  }
  return out
}

/**
 * The log-likelihood of a series under an EGARCH or APARCH model.
 *
 * @param y - The series.
 * @param spec - The structure.
 * @param params - The parameters.
 * @returns The log-likelihood (Gaussian or standardised Student-t innovations).
 * @example
 * ```ts
 * import { asymmetricLogLikelihood } from 'mdml-engine/timeseries'
 * asymmetricLogLikelihood([1], { kind: 'egarch', mean: 'zero' }, { mu: 0, omega: 0, alpha: [0], gamma: [0], beta: [0] }).toFixed(6) // => '-1.418939'
 * ```
 */
export function asymmetricLogLikelihood(
  y: ArrayLike<number>,
  spec: AsymmetricSpec,
  params: AsymmetricParameters,
): number {
  const r = shapeOf(spec)
  const x = asSeries(Float64Array.from(y), 'asymmetric')
  const eps = Float64Array.from(x, (v) => v - (r.constantMean ? params.mu : 0))
  const s2 = asymmetricVariance(eps, spec, params)
  const nu = params.nu ?? Number.NaN
  const c = r.student ? studentConstant(nu) : 0
  let ll = 0
  for (let t = 0; t < x.length; t++) {
    const e = eps[t] ?? 0
    const v = s2[t] ?? Number.NaN
    ll += r.student
      ? c - 0.5 * Math.log(v) - ((nu + 1) / 2) * Math.log(1 + (e * e) / (v * (nu - 2)))
      : -0.5 * (Math.log(2 * Math.PI) + Math.log(v) + (e * e) / v)
  }
  return ll
}

/** The unconstrained parameterisation: omega free (EGARCH) or positive (APARCH), betas in (-1/q, 1/q) or shares of one. */
function layoutOf(r: VolatilitySpec): {
  readonly size: number
  readonly toParams: (u: ArrayLike<number>) => AsymmetricParameters
  readonly start: (variance: number, mean: number) => number[]
} {
  const nAlpha = r.p
  const nGamma = r.egarch ? r.o : r.p
  const size = (r.constantMean ? 1 : 0) + 1 + nAlpha + nGamma + r.q + (r.egarch ? 0 : 1) + (r.student ? 1 : 0)
  return {
    size,
    toParams: (u) => {
      let k = 0
      const mu = r.constantMean ? (u[k++] ?? 0) : 0
      const omega = r.egarch ? (u[k++] ?? 0) : Math.exp(u[k++] ?? 0)
      const alpha = Array.from({ length: nAlpha }, () => (r.egarch ? (u[k++] ?? 0) : Math.exp(u[k++] ?? 0)))
      const gamma = Array.from({ length: nGamma }, () => (r.egarch ? (u[k++] ?? 0) : Math.tanh(u[k++] ?? 0)))
      const beta = Array.from({ length: r.q }, () => Math.tanh(u[k++] ?? 0) / Math.max(r.q, 1))
      const delta = r.egarch ? undefined : 0.5 + Math.exp(u[k++] ?? 0)
      const nu = r.student ? 2.01 + Math.exp(u[k] ?? 0) : undefined
      return {
        mu,
        omega,
        alpha,
        gamma,
        beta,
        ...(delta === undefined ? {} : { delta }),
        ...(nu === undefined ? {} : { nu }),
      }
    },
    start: (variance, mean) => {
      const out: number[] = []
      if (r.constantMean) out.push(mean)
      if (r.egarch) {
        out.push(Math.log(variance) * 0.1)
        for (let i = 0; i < nAlpha; i++) out.push(0.1)
        for (let i = 0; i < nGamma; i++) out.push(-0.02)
        for (let i = 0; i < r.q; i++) out.push(Math.atanh(0.9))
      } else {
        out.push(Math.log(variance * 0.05))
        for (let i = 0; i < nAlpha; i++) out.push(Math.log(0.08))
        for (let i = 0; i < nGamma; i++) out.push(0)
        for (let i = 0; i < r.q; i++) out.push(Math.atanh(0.85))
        out.push(Math.log(1.5))
      }
      if (r.student) out.push(Math.log(6))
      return out
    },
  }
}

/** A fitted asymmetric volatility model. */
export interface AsymmetricFit {
  readonly spec: AsymmetricSpec
  readonly params: AsymmetricParameters
  readonly logLikelihood: number
  readonly aic: number
  readonly bic: number
  readonly nobs: number
  readonly conditionalVariance: readonly number[]
  readonly standardisedResiduals: readonly number[]
  /** The persistence: the sum of the betas (EGARCH, the decay of a log-variance shock) or alpha + beta (APARCH, approximate). */
  readonly persistence: number
  readonly converged: boolean
  readonly iterations: number
}

/** Options of the asymmetric fits. */
export interface AsymmetricFitOptions {
  /** A value of `smoothSolvers.space` (default BFGS). */
  readonly solver?: SpaceValue
  readonly maxIterations?: number
  /** The gradient tolerance of the per-observation negative log-likelihood (default 1e-6). */
  readonly gradTol?: number
  readonly key?: Key
}

/**
 * The EGARCH or APARCH fit as a traced algorithm over the solver slot, and the map from a state to the fitted model.
 * The series is fitted on its unit-spread scaling and the parameters mapped back (omega shifts by 2 ln(scale)(1 - sum
 * beta) for EGARCH and scales by scale^delta for APARCH).
 *
 * @param y - The series.
 * @param spec - The structure.
 * @param options - The solver and tolerances.
 * @returns The algorithm and the map from its unconstrained vector to the fit.
 * @example
 * ```ts
 * import { asymmetricFitAlgorithm } from 'mdml-engine/timeseries'
 * const y = Array.from({ length: 60 }, (_, i) => Math.sin(i * 1.7) * (1 + 0.5 * Math.cos(i * 0.3)))
 * asymmetricFitAlgorithm(y, { kind: 'egarch' }).algorithm.name // => 'egarch-fit'
 * ```
 */
export function asymmetricFitAlgorithm(
  y: ArrayLike<number>,
  spec: AsymmetricSpec,
  options: AsymmetricFitOptions = {},
): {
  readonly algorithm: Algorithm<LikelihoodState>
  readonly fit: (u: ArrayLike<number>, iterations?: number, converged?: boolean) => AsymmetricFit
} {
  const r = shapeOf(spec)
  const x = asSeries(Float64Array.from(y), 'asymmetricFit')
  const n = x.length
  if (n < 10 * (r.p + r.o + r.q + 2))
    throw new MdmlError('value', 'asymmetricFit: the series is too short for the model')
  const centre = meanOf(x)
  const scale = Math.sqrt(varianceOf(x, 0)) || 1
  const z = Float64Array.from(x, (v) => v / scale)
  const layout = layoutOf(r)
  const objective = (u: Float64Array): number => {
    const ll = asymmetricLogLikelihood(z, spec, layout.toParams(u))
    return Number.isFinite(ll) ? -ll / n + Math.log(scale) : 1e10
  }
  const algorithm = likelihoodSlot(
    `${spec.kind}-fit`,
    { spec },
    objective,
    layout.start(varianceOf(z, 0), centre / scale),
    n,
    {
      ...(options.solver === undefined ? {} : { solver: options.solver }),
      ...(options.gradTol === undefined ? {} : { tol: options.gradTol }),
      ...(options.key === undefined ? {} : { key: options.key }),
    },
  )
  return {
    algorithm,
    fit: (u, iterations = 0, converged = true) => {
      const s = layout.toParams(u)
      const sumBeta = s.beta.reduce((a, b) => a + b, 0)
      const params: AsymmetricParameters = {
        ...s,
        mu: s.mu * scale,
        omega: r.egarch ? s.omega + 2 * Math.log(scale) * (1 - sumBeta) : s.omega * scale ** (s.delta ?? 2),
      }
      const eps = Float64Array.from(x, (v) => v - (r.constantMean ? params.mu : 0))
      const variance = asymmetricVariance(eps, spec, params)
      const ll = asymmetricLogLikelihood(x, spec, params)
      const k = layout.size
      return {
        spec,
        params,
        logLikelihood: ll,
        aic: -2 * ll + 2 * k,
        bic: -2 * ll + k * Math.log(n),
        nobs: n,
        conditionalVariance: Array.from(variance),
        standardisedResiduals: Array.from(eps, (e, t) => e / Math.sqrt(variance[t] ?? Number.NaN)),
        persistence: r.egarch ? sumBeta : s.alpha.reduce((a, b) => a + b, 0) + sumBeta,
        converged,
        iterations,
      }
    },
  }
}

/**
 * Fit an EGARCH or APARCH model by maximum likelihood.
 *
 * @param y - The series (returns, typically).
 * @param spec - The structure.
 * @param options - The solver and tolerances.
 * @returns The fitted model.
 * @example
 * ```ts
 * import { asymmetricFit, asymmetricSimulate } from 'mdml-engine/timeseries'
 * import { key } from 'mdml-engine/random'
 * const spec = { kind: 'egarch', mean: 'zero' } as const
 * const y = asymmetricSimulate(key(3), 800, spec, { mu: 0, omega: -0.1, alpha: [0.2], gamma: [-0.1], beta: [0.9] })
 * asymmetricFit(y, spec).params.beta[0] > 0.5 // => true
 * ```
 */
export function asymmetricFit(
  y: ArrayLike<number>,
  spec: AsymmetricSpec,
  options: AsymmetricFitOptions = {},
): AsymmetricFit {
  const { algorithm, fit } = asymmetricFitAlgorithm(y, spec, options)
  const run = runAlgorithm(algorithm, { maxSteps: options.maxIterations ?? 300 })
  return fit(run.state.u, run.steps, run.status === 'converged')
}

/**
 * Simulate an EGARCH or APARCH process with a burn-in.
 *
 * @param key - The random key.
 * @param n - The number of observations returned.
 * @param spec - The structure.
 * @param params - The parameters.
 * @param burnIn - Observations simulated and discarded first (default 200).
 * @returns The series.
 * @example
 * ```ts
 * import { asymmetricSimulate } from 'mdml-engine/timeseries'
 * import { key } from 'mdml-engine/random'
 * asymmetricSimulate(key(0), 5, { kind: 'egarch', mean: 'zero' }, { mu: 0, omega: 0, alpha: [0.1], gamma: [0], beta: [0.5] }).length // => 5
 * ```
 */
export function asymmetricSimulate(
  key: Key,
  n: number,
  spec: AsymmetricSpec,
  params: AsymmetricParameters,
  burnIn = 200,
): number[] {
  const r = shapeOf(spec)
  const total = n + burnIn
  const nu = params.nu ?? 8
  const z = r.student
    ? Array.from(toFlat(studentT(key, [total], nu)), (v) => v / Math.sqrt(nu / (nu - 2)))
    : toFlat(normal(key, [total]))
  const kappa = expectedAbs(r.student, nu)
  const delta = params.delta ?? 2
  const zs: number[] = []
  const lnv: number[] = []
  const es: number[] = []
  const sd: number[] = []
  const out: number[] = []
  for (let t = 0; t < total; t++) {
    let v: number
    let s2: number
    if (r.egarch) {
      v = params.omega
      for (let i = 0; i < r.p; i++)
        v += (params.alpha[i] ?? 0) * (t - 1 - i < 0 ? 0 : Math.abs(zs[t - 1 - i] ?? 0) - kappa)
      for (let k = 0; k < r.o; k++) v += (params.gamma[k] ?? 0) * (t - 1 - k < 0 ? 0 : (zs[t - 1 - k] ?? 0))
      for (let j = 0; j < r.q; j++)
        v +=
          (params.beta[j] ?? 0) *
          (t - 1 - j < 0 ? params.omega / Math.max(1 - (params.beta[0] ?? 0), 0.05) : (lnv[t - 1 - j] ?? 0))
      v = Math.min(Math.max(v, -50), 50)
      lnv.push(v)
      s2 = Math.exp(v)
    } else {
      const start = (params.omega / Math.max(1 - (params.beta[0] ?? 0) - (params.alpha[0] ?? 0), 0.05)) ** 1
      v = params.omega
      for (let i = 0; i < r.p; i++)
        v +=
          (params.alpha[i] ?? 0) *
          (t - 1 - i < 0
            ? start
            : Math.max(Math.abs(es[t - 1 - i] ?? 0) - (params.gamma[i] ?? 0) * (es[t - 1 - i] ?? 0), 0) ** delta)
      for (let j = 0; j < r.q; j++) v += (params.beta[j] ?? 0) * (t - 1 - j < 0 ? start : (sd[t - 1 - j] ?? 0))
      v = Math.max(v, 1e-300)
      sd.push(v)
      s2 = v ** (2 / delta)
    }
    const e = Math.sqrt(s2) * (z[t] ?? 0)
    zs.push(z[t] ?? 0)
    es.push(e)
    if (t >= burnIn) out.push(e + (r.constantMean ? params.mu : 0))
  }
  return out
}

/**
 * The one-step variance forecast and, beyond it, the mean variance over simulated paths (the models are nonlinear in
 * the shock, so multi-step forecasts have no closed form).
 *
 * @param fit - The fitted model.
 * @param y - The series it was fitted to.
 * @param steps - How many steps ahead.
 * @param key - The random key for the simulated paths.
 * @param paths - The number of simulated paths (default 2000).
 * @returns The variance forecasts.
 * @example
 * ```ts
 * import { asymmetricForecast } from 'mdml-engine/timeseries'
 * import { key } from 'mdml-engine/random'
 * const fit = { spec: { kind: 'egarch', mean: 'zero' }, params: { mu: 0, omega: 0, alpha: [0], gamma: [0], beta: [0.5] } } as never
 * asymmetricForecast(fit, [0.5, -1, 0.2, 1], 3, key(0), 50).variance.length // => 3
 * ```
 */
export function asymmetricForecast(
  fit: Pick<AsymmetricFit, 'spec' | 'params'>,
  y: ArrayLike<number>,
  steps: number,
  key: Key,
  paths = 2000,
): { readonly variance: readonly number[] } {
  const r = shapeOf(fit.spec)
  const x = Float64Array.from(y)
  const eps = Float64Array.from(x, (v) => v - (r.constantMean ? fit.params.mu : 0))
  const acc = new Float64Array(steps)
  const z = toFlat(normal(key, [paths * steps]))
  for (let pth = 0; pth < paths; pth++) {
    const path = new Float64Array(eps.length + steps)
    path.set(eps)
    for (let h = 0; h < steps; h++) {
      // The variance at time eps.length + h comes from the recursion over the history and the paths so far.
      const ext = asymmetricVariance(path.subarray(0, eps.length + h + 1), fit.spec, fit.params)
      const next = ext[eps.length + h] ?? Number.NaN
      acc[h] = (acc[h] ?? 0) + next
      path[eps.length + h] = Math.sqrt(next) * (z[pth * steps + h] ?? 0)
    }
  }
  return { variance: Array.from(acc, (v) => v / paths) }
}

/** The news-impact curve of a volatility model: the next variance for each shock, the variance held at its stationary level. */
export interface NewsImpactCurve {
  readonly shocks: readonly number[]
  readonly variance: readonly number[]
  /** The stationary variance the curve is taken around. */
  readonly stationaryVariance: number
}

/**
 * The news-impact curve of Engle and Ng for a fitted GARCH/GJR, EGARCH or APARCH model: the variance after a shock e
 * (on the first lag of the shock; the other lags and the lagged variances sit at the stationary level).
 *
 * @param model - A `GarchFit` (including GJR) or an `AsymmetricFit`.
 * @param shocks - The shocks to evaluate.
 * @returns The curve.
 * @example
 * ```ts
 * import { newsImpactCurve } from 'mdml-engine/timeseries'
 * const fit = { spec: { p: 1, o: 1, q: 1 }, params: { mu: 0, omega: 0.1, alpha: [0.05], gamma: [0.1], beta: [0.8] } } as never
 * const c = newsImpactCurve(fit, [-2, 2])
 * c.variance[0] > c.variance[1] // => true
 * ```
 */
export function newsImpactCurve(
  model: Pick<GarchFit, 'spec' | 'params'> | Pick<AsymmetricFit, 'spec' | 'params'>,
  shocks: readonly number[],
): NewsImpactCurve {
  const spec = model.spec as GarchSpec | AsymmetricSpec
  if ('kind' in spec) {
    const r = shapeOf(spec)
    const par = model.params as AsymmetricParameters
    if (r.egarch) {
      const lnStationary = par.omega / (1 - par.beta.reduce((a, b) => a + b, 0))
      const kappa = expectedAbs(r.student, par.nu ?? 8)
      const sigma = Math.exp(lnStationary / 2)
      const variance = shocks.map((e) => {
        const z = e / sigma
        const v =
          par.omega +
          (par.alpha[0] ?? 0) * (Math.abs(z) - kappa) +
          (par.gamma[0] ?? 0) * z +
          par.beta.reduce((a, b) => a + b, 0) * lnStationary
        return Math.exp(v)
      })
      return { shocks, variance, stationaryVariance: Math.exp(lnStationary) }
    }
    const delta = par.delta ?? 2
    const lambda = (g: number): number => 0.5 * (Math.abs(1 - g) ** delta + Math.abs(1 + g) ** delta)
    const persistence =
      par.alpha.reduce((a, g, i) => a + g * lambda(par.gamma[i] ?? 0), 0) + par.beta.reduce((a, b) => a + b, 0)
    const sdelta = par.omega / (1 - persistence)
    const variance = shocks.map((e) => {
      const v =
        par.omega +
        (par.alpha[0] ?? 0) * Math.max(Math.abs(e) - (par.gamma[0] ?? 0) * e, 0) ** delta +
        par.beta.reduce((a, b) => a + b, 0) * sdelta
      return v ** (2 / delta)
    })
    return { shocks, variance, stationaryVariance: sdelta ** (2 / delta) }
  }
  const par = model.params as GarchParameters
  const persistence =
    par.alpha.reduce((a, b) => a + b, 0) +
    par.gamma.reduce((a, b) => a + b, 0) / 2 +
    par.beta.reduce((a, b) => a + b, 0)
  const stationary = par.omega / (1 - persistence)
  const rest = par.beta.reduce((a, b) => a + b, 0) * stationary
  const restAlpha =
    par.alpha.slice(1).reduce((a, b) => a + b, 0) * stationary +
    (par.gamma.slice(1).reduce((a, b) => a + b, 0) * stationary) / 2
  return {
    shocks,
    variance: shocks.map(
      (e) => par.omega + (par.alpha[0] ?? 0) * e * e + (par.gamma[0] ?? 0) * (e < 0 ? e * e : 0) + rest + restAlpha,
    ),
    stationaryVariance: stationary,
  }
}
