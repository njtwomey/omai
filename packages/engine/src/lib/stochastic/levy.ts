/**
 * Levy processes, which have independent stationary increments and may jump: compound Poisson processes, alpha-stable
 * laws (Chambers-Mallows-Stuck sampling, characteristic function, density and CDF by Fourier inversion, as
 * scipy.stats.levy_stable) and the stable process, and the variance gamma process (Brownian motion run on a gamma
 * clock) with its closed-form density and moments.
 */

import { WhiteboxError } from '../base/index.ts'
import { type Distribution, gamma } from '../dists/index.ts'
import { quad } from '../numerics/quadrature/index.ts'
import { kv, lgamma } from '../numerics/special/index.ts'
import { exponential, foldIn, type Key, normal, poisson, uniform } from '../random/index.ts'
import {
  abs,
  add,
  arctan,
  asFloat,
  asTensor,
  cos,
  div,
  exp,
  fromStorage,
  greater,
  item,
  log,
  mul,
  neg,
  pow,
  promoteFloat,
  sign,
  sin,
  sqrt,
  square,
  sub,
  type TensorLike,
  type Value,
  where,
  zeros,
} from '../tensor/index.ts'
import { asTimes, positiveInteger, primalNumbers, type TimesLike, traceOf } from './_shared.ts'
import { type Paths, simulatePaths, type TraceOptions } from './paths.ts'

/** Options of {@link compoundPoisson}. */
export interface CompoundPoissonOptions extends TraceOptions {
  /** The jump intensity lambda. */
  readonly rate: number
  /** The law of the jump sizes. */
  readonly jumps: Distribution
  /** A deterministic drift per unit time (default 0). */
  readonly drift?: number
  /** The number of paths (default 1). */
  readonly paths?: number
}

/**
 * A compound Poisson process X_t = drift t + sum_{i <= N_t} J_i, simulated exactly on a grid: in each interval a
 * Poisson(lambda dt) number of independent jumps is added. E X_t = (drift + lambda E J) t and Var X_t = lambda t E J^2.
 *
 * @param key - The key.
 * @param times - The grid.
 * @param options - Rate, jump law, drift, paths and tracing.
 * @returns Paths `[paths, times.length]` starting at 0.
 * @concept compound-poisson-process
 * @example
 * ```ts
 * import { normal } from 'whitebox-engine/dists'
 * import { key } from 'whitebox-engine/random'
 * import { compoundPoisson, timeGrid } from 'whitebox-engine/stochastic'
 * compoundPoisson(key(0), timeGrid(1, 10), { rate: 3, jumps: normal(0, 1), paths: 2 }).values.shape // => [2, 11]
 * ```
 */
export function compoundPoisson(key: Key, times: TimesLike, options: CompoundPoissonOptions): Paths {
  const t = asTimes(times, 'compoundPoisson')
  if (!(options.rate >= 0)) throw new WhiteboxError('value', 'compoundPoisson: the rate must be >= 0')
  const paths = positiveInteger(options.paths ?? 1, 'paths', 'compoundPoisson')
  const drift = options.drift ?? 0
  return simulatePaths(
    key,
    {
      name: 'compound-poisson',
      config: { rate: options.rate, paths },
      times: t,
      x0: zeros([paths]),
      step: (x, c) => {
        const counts = poisson(foldIn(c.key, 0), options.rate * c.dt, [paths]).data
        const total = counts.reduce((a, b) => a + b, 0)
        const draws = total > 0 ? primalNumbers(options.jumps.sample(foldIn(c.key, 1), [total])) : []
        const inc = new Float64Array(paths)
        let at = 0
        for (let j = 0; j < paths; j++) {
          inc[j] = drift * c.dt
          for (let k = 0; k < (counts[j] ?? 0); k++) inc[j] = (inc[j] ?? 0) + (draws[at++] ?? 0)
        }
        return add(x, fromStorage(inc, [paths], 'float64'))
      },
    },
    traceOf(options),
  )
}

/**
 * The mean and variance of a compound Poisson process at time t: (drift + lambda E J) t and lambda t E J^2.
 *
 * @param t - The time.
 * @param rate - The jump intensity.
 * @param jumps - The jump law.
 * @param drift - The drift (default 0).
 * @returns The mean and variance.
 * @concept compound-poisson-process
 * @example
 * ```ts
 * import { normal } from 'whitebox-engine/dists'
 * import { compoundPoissonMoments } from 'whitebox-engine/stochastic'
 * compoundPoissonMoments(2, 3, normal(1, 1)) // => { mean: 6, variance: 12 }
 * ```
 */
export function compoundPoissonMoments(
  t: number,
  rate: number,
  jumps: Distribution,
  drift = 0,
): { readonly mean: number; readonly variance: number } {
  const m = item(jumps.mean())
  const second = item(jumps.variance()) + m * m
  return { mean: (drift + rate * m) * t, variance: rate * t * second }
}

/** Parameters of an alpha-stable law in the S1 parameterisation (scipy.stats.levy_stable's default). */
export interface StableParams {
  /** The stability index alpha in (0, 2]. */
  readonly alpha: TensorLike
  /** The skewness beta in [-1, 1] (default 0). */
  readonly beta?: TensorLike
  /** The scale (default 1). */
  readonly scale?: TensorLike
  /** The location (default 0). */
  readonly loc?: TensorLike
}

function stableParts(p: StableParams): { alpha: Value; beta: Value; scale: Value; loc: Value; a: number; b: number } {
  const [alpha, beta, scale, loc] = promoteFloat(p.alpha, p.beta ?? 0, p.scale ?? 1, p.loc ?? 0) as [
    Value,
    Value,
    Value,
    Value,
  ]
  const a = primalNumbers(alpha)[0] ?? Number.NaN
  const b = primalNumbers(beta)[0] ?? Number.NaN
  if (!(a > 0 && a <= 2)) throw new WhiteboxError('value', 'stable: alpha must be in (0, 2]')
  if (!(b >= -1 && b <= 1)) throw new WhiteboxError('value', 'stable: beta must be in [-1, 1]')
  if (!((primalNumbers(scale)[0] ?? 0) > 0)) throw new WhiteboxError('value', 'stable: the scale must be positive')
  return { alpha, beta, scale, loc, a, b }
}

/**
 * Alpha-stable samples by the Chambers-Mallows-Stuck method (in Weron's form for the S1 parameterisation): from
 * V ~ U(-pi/2, pi/2) and W ~ Exp(1), X = S sin(alpha (V + B)) / cos(V)^{1/alpha} (cos(V - alpha (V + B)) / W)^{(1 - alpha)
 * / alpha} for alpha != 1, with the separate formula at alpha = 1. alpha = 2 gives N(loc, 2 scale^2), alpha = 1 with
 * beta = 0 the Cauchy law. Reparameterised: differentiable in every parameter.
 *
 * @param key - The key.
 * @param shape - The output shape.
 * @param params - alpha, beta, scale and location.
 * @returns Samples.
 * @concept stable-distributions
 * @reference J. M. Chambers, C. L. Mallows and B. W. Stuck, A method for simulating stable random variables, JASA 71
 *   (1976); R. Weron, On the Chambers-Mallows-Stuck method, Statistics and Probability Letters 28 (1996).
 * @example
 * ```ts
 * import { key } from 'whitebox-engine/random'
 * import { stableSample } from 'whitebox-engine/stochastic'
 * stableSample(key(0), [5], { alpha: 1.5, beta: 0.5 }).shape // => [5]
 * ```
 */
export function stableSample(key: Key, shape: readonly number[], params: StableParams): Value {
  const { alpha, beta, scale, loc, a } = stableParts(params)
  const v = add(mul(uniform(foldIn(key, 0), shape), Math.PI), -Math.PI / 2)
  const w = exponential(foldIn(key, 1), shape)
  if (Math.abs(a - 1) < 1e-12) {
    const half = add(mul(beta, v), Math.PI / 2)
    const x = mul(
      sub(mul(half, div(sin(v), cos(v))), mul(beta, log(div(mul(mul(w, cos(v)), Math.PI / 2), half)))),
      2 / Math.PI,
    )
    return add(add(mul(scale, x), mul(mul(mul(beta, scale), log(scale)), 2 / Math.PI)), loc)
  }
  const t = div(sin(mul(alpha, Math.PI / 2)), cos(mul(alpha, Math.PI / 2)))
  const bt = mul(beta, t)
  const b = div(arctan(bt), alpha)
  const s = pow(add(square(bt), 1), div(0.5, alpha))
  const vb = mul(alpha, add(v, b))
  const x = mul(
    mul(s, div(sin(vb), pow(cos(v), div(1, alpha)))),
    pow(div(cos(sub(v, vb)), w), div(sub(1, alpha), alpha)),
  )
  return add(mul(scale, x), loc)
}

/**
 * The characteristic function E e^{iuX} of an alpha-stable law (S1): exp(i loc u - |scale u|^alpha (1 - i beta sign(u)
 * tan(pi alpha / 2))) for alpha != 1 and exp(i loc u - scale |u| (1 + i beta (2/pi) sign(u) log |u|)) for alpha = 1.
 *
 * @param u - Frequencies.
 * @param params - alpha, beta, scale and location.
 * @returns The real and imaginary parts.
 * @concept stable-distributions
 * @example
 * ```ts
 * import { stableCharacteristicFunction } from 'whitebox-engine/stochastic'
 * import { toArray } from 'whitebox-engine/tensor'
 * toArray(stableCharacteristicFunction([0, 1], { alpha: 2 }).re) // => [1, Math.exp(-1)]
 * ```
 */
export function stableCharacteristicFunction(
  u: TensorLike,
  params: StableParams,
): { readonly re: Value; readonly im: Value } {
  const { alpha, beta, scale, loc, a } = stableParts(params)
  const uu = asFloat(asTensor(u))
  const sgn = sign(uu)
  let modulus: Value
  let phase: Value
  if (Math.abs(a - 1) < 1e-12) {
    const au = abs(uu)
    modulus = neg(mul(scale, au))
    phase = sub(mul(loc, uu), mul(mul(mul(mul(scale, au), beta), sgn), mul(logSafe(au), 2 / Math.PI)))
  } else {
    const p = pow(abs(mul(scale, uu)), alpha)
    const tan = div(sin(mul(alpha, Math.PI / 2)), cos(mul(alpha, Math.PI / 2)))
    modulus = neg(p)
    phase = add(mul(loc, uu), mul(mul(mul(p, beta), sgn), tan))
  }
  const r = exp(modulus)
  return { re: mul(r, cos(phase)), im: mul(r, sin(phase)) }
}

/** log x with log 0 read as 0 (the terms it multiplies vanish there). */
function logSafe(x: Value): Value {
  return log(where(greater(x, 0), x, 1))
}

function inversion(
  params: StableParams,
  x: TensorLike,
  kernel: (u: Value, phase: Value, decay: Value) => Value,
): Value {
  const { a, b } = stableParts(params)
  const scale = primalNumbers(promoteFloat(params.scale ?? 1)[0] as Value)[0] ?? 1
  const loc = primalNumbers(promoteFloat(params.loc ?? 0)[0] as Value)[0] ?? 0
  const xs = primalNumbers(x)
  const tan = Math.tan((Math.PI * a) / 2)
  const out = xs.map((xi) => {
    const f = (u: Value): Value => {
      const su = mul(u, scale)
      const decay = exp(neg(pow(su, a)))
      const phase =
        Math.abs(a - 1) < 1e-12
          ? sub(mul(u, loc - xi), mul(mul(su, logSafe(u)), (2 / Math.PI) * b))
          : add(mul(u, loc - xi), mul(pow(su, a), b * tan))
      return kernel(u, phase, decay)
    }
    return item(quad(f, 0, Number.POSITIVE_INFINITY, { epsabs: 1e-12, epsrel: 1e-10, limit: 200 }).value)
  })
  return fromStorage(Float64Array.from(out), [...asTensor(x).shape], 'float64')
}

/**
 * The alpha-stable density by Fourier inversion, f(x) = (1 / pi) int_0^inf e^{-(scale u)^alpha} cos(theta(u)) du with
 * the phase theta of the characteristic function (adaptive quadrature over the half line).
 *
 * @param x - Points.
 * @param params - alpha, beta, scale and location (numbers).
 * @returns The densities.
 * @concept stable-distributions
 * @example
 * ```ts
 * import { stablePdf } from 'whitebox-engine/stochastic'
 * import { toArray } from 'whitebox-engine/tensor'
 * // alpha = 1, beta = 0 is the Cauchy law: f(0) = 1 / pi
 * Math.abs(toArray(stablePdf([0], { alpha: 1 }))[0] - 1 / Math.PI) < 1e-9 // => true
 * ```
 */
export function stablePdf(x: TensorLike, params: StableParams): Value {
  return mul(
    inversion(params, x, (_u, phase, decay) => mul(decay, cos(phase))),
    1 / Math.PI,
  )
}

/**
 * The alpha-stable CDF by the Gil-Pelaez inversion formula, F(x) = 1/2 - (1 / pi) int_0^inf Im(e^{-iux} phi(u)) / u du.
 *
 * @param x - Points.
 * @param params - alpha, beta, scale and location (numbers).
 * @returns The probabilities.
 * @concept stable-distributions
 * @example
 * ```ts
 * import { stableCdf } from 'whitebox-engine/stochastic'
 * import { toArray } from 'whitebox-engine/tensor'
 * Math.abs(toArray(stableCdf([0], { alpha: 1.3 }))[0] - 0.5) < 1e-9 // => true
 * ```
 */
export function stableCdf(x: TensorLike, params: StableParams): Value {
  const integral = inversion(params, x, (u, phase, decay) => div(mul(decay, sin(phase)), u))
  return sub(0.5, mul(integral, 1 / Math.PI))
}

/**
 * An alpha-stable Levy process on a grid: independent stable increments, X_{t+dt} - X_t ~ S1(alpha, beta, scale
 * dt^{1/alpha}) (with the drift correction of the S1 form at alpha = 1), simulated exactly. Self-similar with index
 * 1 / alpha; for alpha < 2 its paths jump.
 *
 * @param key - The key.
 * @param times - The grid.
 * @param params - alpha, beta and scale (per unit time), paths and tracing.
 * @returns Paths `[paths, times.length]` from 0.
 * @concept stable-distributions
 * @example
 * ```ts
 * import { key } from 'whitebox-engine/random'
 * import { stableProcess, timeGrid } from 'whitebox-engine/stochastic'
 * stableProcess(key(0), timeGrid(1, 20), { alpha: 1.7, paths: 3 }).values.shape // => [3, 21]
 * ```
 */
export function stableProcess(
  key: Key,
  times: TimesLike,
  params: StableParams & { readonly paths?: number } & TraceOptions,
): Paths {
  const t = asTimes(times, 'stableProcess')
  const { alpha, beta, scale, a } = stableParts(params)
  const paths = positiveInteger(params.paths ?? 1, 'paths', 'stableProcess')
  return simulatePaths(
    key,
    {
      name: 'stable-process',
      config: { alpha: a, paths },
      times: t,
      x0: zeros([paths]),
      step: (x, c) => {
        const s = mul(scale, c.dt ** (1 / a))
        // S1 at alpha = 1: X_t ~ S(scale t, beta, (2/pi) beta scale t log t)
        const loc = Math.abs(a - 1) < 1e-12 ? mul(mul(mul(beta, scale), c.dt * Math.log(c.dt)), 2 / Math.PI) : 0
        return add(x, stableSample(c.key, [paths], { alpha, beta, scale: s, loc }))
      },
    },
    traceOf(params),
  )
}

/** Parameters of the variance gamma process. */
export interface VarianceGammaParams {
  /** The drift of the subordinated Brownian motion. */
  readonly theta: TensorLike
  /** Its volatility (positive). */
  readonly sigma: TensorLike
  /** The variance rate of the gamma clock (positive). */
  readonly nu: TensorLike
}

/**
 * The variance gamma process (Madan, Carr and Chang 1998): Brownian motion with drift theta and volatility sigma run on
 * a gamma clock G with E G_t = t and Var G_t = nu t, X_t = theta G_t + sigma W(G_t), simulated exactly on a grid
 * (gamma increments by implicit reparameterisation, so paths differentiate in all three parameters). Mean theta t,
 * variance (sigma^2 + nu theta^2) t.
 *
 * @param key - The key.
 * @param times - The grid.
 * @param params - theta, sigma, nu, paths and tracing.
 * @returns Paths `[paths, times.length]` from 0.
 * @concept variance-gamma
 * @example
 * ```ts
 * import { key } from 'whitebox-engine/random'
 * import { timeGrid, varianceGammaProcess } from 'whitebox-engine/stochastic'
 * varianceGammaProcess(key(0), timeGrid(1, 10), { theta: -0.1, sigma: 0.2, nu: 0.3, paths: 2 }).values.shape // => [2, 11]
 * ```
 */
export function varianceGammaProcess(
  key: Key,
  times: TimesLike,
  params: VarianceGammaParams & { readonly paths?: number } & TraceOptions,
): Paths {
  const t = asTimes(times, 'varianceGammaProcess')
  const [theta, sigma, nu] = promoteFloat(params.theta, params.sigma, params.nu) as [Value, Value, Value]
  const paths = positiveInteger(params.paths ?? 1, 'paths', 'varianceGammaProcess')
  return simulatePaths(
    key,
    {
      name: 'variance-gamma',
      config: { paths },
      times: t,
      x0: zeros([paths]),
      step: (x, c) => {
        const g = gamma(div(c.dt, nu), div(1, nu)).rsample(foldIn(c.key, 0), [paths])
        const z = normal(foldIn(c.key, 1), [paths])
        return add(x, add(mul(theta, g), mul(mul(sigma, sqrt(g)), z)))
      },
    },
    traceOf(params),
  )
}

/**
 * The variance gamma density at time t in closed form:
 * f(x) = 2 e^{theta x / sigma^2} / (nu^{t/nu} sqrt(2 pi) sigma Gamma(t/nu)) (x^2 / (2 sigma^2 / nu + theta^2))^{t/(2nu) -
 * 1/4} K_{t/nu - 1/2}(sqrt(x^2 (2 sigma^2 / nu + theta^2)) / sigma^2), with K the modified Bessel function of the
 * second kind. Differentiable in x and the parameters.
 *
 * @param x - Points (non-zero when t / nu <= 1/2, where the density has a pole at 0).
 * @param t - The time.
 * @param params - theta, sigma and nu.
 * @returns The densities.
 * @concept variance-gamma
 * @example
 * ```ts
 * import { varianceGammaDensity } from 'whitebox-engine/stochastic'
 * import { item } from 'whitebox-engine/tensor'
 * item(varianceGammaDensity(0.1, 1, { theta: 0, sigma: 0.2, nu: 0.1 })) > 0 // => true
 * ```
 */
export function varianceGammaDensity(x: TensorLike, t: TensorLike, params: VarianceGammaParams): Value {
  const [xx, tt, theta, sigma, nu] = promoteFloat(x, t, params.theta, params.sigma, params.nu) as [
    Value,
    Value,
    Value,
    Value,
    Value,
  ]
  const s2 = square(sigma)
  const shape = div(tt, nu)
  const c = add(div(mul(s2, 2), nu), square(theta))
  const logFront = sub(
    sub(add(Math.LN2, div(mul(theta, xx), s2)), add(mul(shape, log(nu)), lgamma(shape))),
    add(log(sigma), 0.5 * Math.log(2 * Math.PI)),
  )
  const ax = abs(xx)
  const logPower = mul(sub(div(shape, 2), 0.25), sub(log(square(ax)), log(c)))
  const bessel = kv(sub(shape, 0.5), div(mul(ax, sqrt(c)), s2))
  return mul(exp(add(logFront, logPower)), bessel)
}

/**
 * The characteristic function of the variance gamma law at time t, (1 - i u theta nu + sigma^2 nu u^2 / 2)^{-t / nu}.
 *
 * @param u - Frequencies.
 * @param t - The time.
 * @param params - theta, sigma and nu.
 * @returns The real and imaginary parts.
 * @concept variance-gamma
 * @example
 * ```ts
 * import { varianceGammaCharacteristicFunction } from 'whitebox-engine/stochastic'
 * import { toArray } from 'whitebox-engine/tensor'
 * toArray(varianceGammaCharacteristicFunction([0], 1, { theta: 0.1, sigma: 0.2, nu: 0.5 }).re) // => [1]
 * ```
 */
export function varianceGammaCharacteristicFunction(
  u: TensorLike,
  t: TensorLike,
  params: VarianceGammaParams,
): { readonly re: Value; readonly im: Value } {
  const [uu, tt, theta, sigma, nu] = promoteFloat(u, t, params.theta, params.sigma, params.nu) as [
    Value,
    Value,
    Value,
    Value,
    Value,
  ]
  // z = 1 + sigma^2 nu u^2 / 2 - i u theta nu, raised to -t / nu in polar form
  const re = add(1, mul(mul(mul(square(sigma), nu), square(uu)), 0.5))
  const im = neg(mul(mul(uu, theta), nu))
  const power = neg(div(tt, nu))
  const modulus = pow(add(square(re), square(im)), mul(power, 0.5))
  const angle = mul(power, arg(re, im))
  return { re: mul(modulus, cos(angle)), im: mul(modulus, sin(angle)) }
}

/** The argument of re + i im for re > 0 (always the case here). */
function arg(re: Value, im: Value): Value {
  return arctan(div(im, re))
}

/**
 * The mean and variance of the variance gamma law at time t: theta t and (sigma^2 + nu theta^2) t.
 *
 * @param t - The time.
 * @param params - theta, sigma and nu.
 * @returns The mean and variance.
 * @concept variance-gamma
 * @example
 * ```ts
 * import { varianceGammaMoments } from 'whitebox-engine/stochastic'
 * varianceGammaMoments(2, { theta: 0.5, sigma: 1, nu: 2 }) // => { mean: 1, variance: 3 }
 * ```
 */
export function varianceGammaMoments(
  t: number,
  params: { readonly theta: number; readonly sigma: number; readonly nu: number },
): { readonly mean: number; readonly variance: number } {
  return { mean: params.theta * t, variance: (params.sigma ** 2 + params.nu * params.theta ** 2) * t }
}
