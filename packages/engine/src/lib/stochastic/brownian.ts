/**
 * Brownian motion, the continuous random walk at the heart of stochastic calculus: exact simulation of Brownian
 * increments, motion with drift and Brownian bridges (pinned at both ends), the Levy midpoint construction, realised
 * quadratic variation and covariation, and the reflection principle with the closed-form laws of the running maximum
 * and of first-passage times.
 */

import { OmaiError } from '../base/index.ts'
import { type Distribution, inverseGamma, inverseGaussian } from '../dists/index.ts'
import { cholesky } from '../numerics/linalg/index.ts'
import { logNdtr, ndtr } from '../numerics/special/index.ts'
import { foldIn, type Key, normal } from '../random/index.ts'
import {
  add,
  asTensor,
  div,
  exp,
  expandDims,
  fromStorage,
  greaterEqual,
  lessEqual,
  log,
  matmul,
  maximum,
  mul,
  neg,
  promoteFloat,
  sliceAxis,
  sqrt,
  square,
  squeeze,
  stack,
  sub,
  sum,
  type Tensor,
  type TensorLike,
  transpose,
  type Value,
  where,
  zeros,
} from '../tensor/index.ts'
import { asTimes, positiveInteger, primalNumbers, stepsOf, type TimesLike, traceOf, valuesOf } from './_shared.ts'
import { type Paths, pathIncrements, simulatePaths, type TraceOptions } from './paths.ts'

/** Options of {@link brownianIncrements}. */
export interface IncrementOptions {
  /** The number of paths (default 1). */
  readonly paths?: number
  /** The shape of the noise at one time (default `[]`, scalar noise). */
  readonly shape?: readonly number[]
}

/**
 * Standard Brownian increments on a grid: independent N(0, dt_i) draws, the noise every simulator in `stochastic`
 * accepts so that several schemes can be driven by the same Brownian path.
 *
 * @param key - The key.
 * @param times - The grid (increasing).
 * @param options - The number of paths and the noise shape.
 * @returns `[paths, times.length - 1, ...shape]`.
 * @concept brownian-motion
 * @example
 * ```ts
 * import { key } from 'omai-engine/random'
 * import { brownianIncrements, timeGrid } from 'omai-engine/stochastic'
 * brownianIncrements(key(0), timeGrid(1, 10), { paths: 3, shape: [2] }).shape // => [3, 10, 2]
 * ```
 */
export function brownianIncrements(key: Key, times: TimesLike, options: IncrementOptions = {}): Tensor<'float64'> {
  const t = asTimes(times, 'brownianIncrements')
  const paths = positiveInteger(options.paths ?? 1, 'paths', 'brownianIncrements')
  const shape = options.shape ?? []
  const dts = stepsOf(t)
  const size = shape.reduce((a, b) => a * b, 1)
  const z = normal(key, [paths, dts.length, ...shape]).data
  const out = new Float64Array(z.length)
  for (let i = 0; i < z.length; i++) out[i] = (z[i] ?? 0) * Math.sqrt(dts[Math.floor(i / size) % dts.length] ?? 0)
  return fromStorage(out, [paths, dts.length, ...shape], 'float64')
}

/** Options of {@link brownianMotion}. */
export interface BrownianOptions extends TraceOptions {
  /** The number of paths (default 1). */
  readonly paths?: number
  /** The dimension; omitted, the process is scalar (inferred from `covariance` when given). */
  readonly dim?: number
  /** The starting point (default 0), broadcast to `[paths, ...state]`. */
  readonly x0?: TensorLike
  /** The drift mu per unit time (default 0): a scalar or `[dim]`. */
  readonly drift?: TensorLike
  /** The volatility sigma (default 1): a scalar or `[dim]` (independent coordinates). */
  readonly sigma?: TensorLike
  /** A covariance per unit time `[dim, dim]` instead of `sigma` (correlated coordinates, by its Cholesky factor). */
  readonly covariance?: TensorLike
  /** Standard Brownian increments `[paths, steps, ...state]` to use instead of drawing them (see {@link brownianIncrements}). */
  readonly increments?: TensorLike
}

function slice1(x: Value, i: number): Value {
  return squeeze(sliceAxis(x, 1, i, i + 1), 1)
}

/**
 * Brownian motion with drift, X_t = x0 + mu t + sigma W_t (or + L W_t with L L^T the covariance), simulated exactly on
 * any grid as a traced path simulation. Differentiable in the start, drift, volatility and covariance (pathwise).
 *
 * @param key - The key.
 * @param times - The grid (increasing).
 * @param options - Paths, dimension, start, drift, volatility or covariance, increments to reuse, and tracing.
 * @returns The paths `[paths, times.length, ...state]`.
 * @concept brownian-motion
 * @index function Brownian motion sample paths on a time grid.
 * @example
 * ```ts
 * import { key } from 'omai-engine/random'
 * import { brownianMotion, timeGrid } from 'omai-engine/stochastic'
 * import { mean, item, sliceAxis } from 'omai-engine/tensor'
 * const w = brownianMotion(key(0), timeGrid(1, 50), { paths: 2000, drift: 0.5 })
 * w.values.shape // => [2000, 51]
 * Math.abs(item(mean(sliceAxis(w.values, 1, 50, 51))) - 0.5) < 0.1 // => true
 * ```
 */
export function brownianMotion(key: Key, times: TimesLike, options: BrownianOptions = {}): Paths {
  const t = asTimes(times, 'brownianMotion')
  const paths = positiveInteger(options.paths ?? 1, 'paths', 'brownianMotion')
  const cov = options.covariance === undefined ? undefined : (promoteFloat(options.covariance)[0] as Value)
  const dim = cov === undefined ? options.dim : (cov.shape[0] ?? 1)
  if (dim !== undefined) positiveInteger(dim, 'dim', 'brownianMotion')
  const state = dim === undefined ? [] : [dim]
  const scale = cov === undefined ? undefined : transpose(cholesky(cov))
  const [drift, sigma] = promoteFloat(options.drift ?? 0, options.sigma ?? 1) as [Value, Value]
  const x0 = add(zeros([paths, ...state]), options.x0 ?? 0)
  const given = options.increments === undefined ? undefined : asTensor(options.increments)
  if (given !== undefined && (given.shape[0] !== paths || given.shape[1] !== t.length - 1)) {
    throw new OmaiError('shape', `brownianMotion: increments must be [${paths}, ${t.length - 1}, ...state]`)
  }
  return simulatePaths(
    key,
    {
      name: 'brownian-motion',
      config: { paths, steps: t.length - 1, dim: dim ?? null },
      times: t,
      x0,
      step: (x, c) => {
        const dw = given === undefined ? mul(normal(c.key, [paths, ...state]), Math.sqrt(c.dt)) : slice1(given, c.i)
        const noise = scale === undefined ? mul(sigma, dw) : matmul(dw, scale)
        return add(add(x, mul(drift, c.dt)), noise)
      },
    },
    traceOf(options),
  )
}

/** Options of {@link brownianBridge}. */
export interface BridgeOptions extends TraceOptions {
  readonly paths?: number
  /** The dimension (omitted: scalar). */
  readonly dim?: number
  /** The start value at the first grid time (default 0). */
  readonly x0?: TensorLike
  /** The end value at the last grid time (default 0). */
  readonly x1?: TensorLike
  /** The volatility (default 1). */
  readonly sigma?: TensorLike
}

/**
 * A Brownian bridge: Brownian motion pinned to `x0` at the first grid time and `x1` at the last, simulated exactly by
 * its Markov transition X_{i+1} | X_i ~ N(X_i + (x1 - X_i) dt / (T - t_i), sigma^2 dt (T - t_{i+1}) / (T - t_i)).
 * Differentiable in the end points and the volatility.
 *
 * @param key - The key.
 * @param times - The grid; its ends are the pinned times.
 * @param options - Paths, dimension, end points, volatility and tracing.
 * @returns The paths `[paths, times.length, ...state]`, equal to `x1` at the end.
 * @concept brownian-bridge
 * @example
 * ```ts
 * import { key } from 'omai-engine/random'
 * import { brownianBridge, timeGrid, valuesAt } from 'omai-engine/stochastic'
 * import { toArray } from 'omai-engine/tensor'
 * toArray(valuesAt(brownianBridge(key(0), timeGrid(1, 8), { paths: 2, x1: 3 }), -1)) // => [3, 3]
 * ```
 */
export function brownianBridge(key: Key, times: TimesLike, options: BridgeOptions = {}): Paths {
  const t = asTimes(times, 'brownianBridge')
  if (t.length < 2) throw new OmaiError('value', 'brownianBridge: need at least two times')
  const paths = positiveInteger(options.paths ?? 1, 'paths', 'brownianBridge')
  const state = options.dim === undefined ? [] : [positiveInteger(options.dim, 'dim', 'brownianBridge')]
  const [x1, sigma] = promoteFloat(options.x1 ?? 0, options.sigma ?? 1) as [Value, Value]
  const end = t[t.length - 1] ?? 1
  return simulatePaths(
    key,
    {
      name: 'brownian-bridge',
      config: { paths, steps: t.length - 1 },
      times: t,
      x0: add(zeros([paths, ...state]), options.x0 ?? 0),
      step: (x, c) => {
        const left = end - c.t
        const right = end - (c.t + c.dt)
        const m = add(x, mul(sub(x1, x), c.dt / left))
        if (c.i === t.length - 2) return add(zeros([paths, ...state]), x1)
        return add(m, mul(sigma, mul(normal(c.key, [paths, ...state]), Math.sqrt((c.dt * right) / left))))
      },
    },
    traceOf(options),
  )
}

/** Options of {@link bridgeMoments}. */
export interface BridgeMomentOptions {
  readonly t0?: number
  readonly t1?: number
  readonly x0?: TensorLike
  readonly x1?: TensorLike
  readonly sigma?: TensorLike
}

/**
 * The mean and variance of a Brownian bridge from (t0, x0) to (t1, x1) at time t: x0 + (x1 - x0)(t - t0)/(t1 - t0) and
 * sigma^2 (t - t0)(t1 - t)/(t1 - t0). Differentiable in every argument.
 *
 * @param t - Times in [t0, t1].
 * @param options - The pinned times (default 0 and 1), values (default 0) and volatility (default 1).
 * @returns `{ mean, variance }` with the shape of `t`.
 * @concept brownian-bridge
 * @example
 * ```ts
 * import { bridgeMoments } from 'omai-engine/stochastic'
 * import { item } from 'omai-engine/tensor'
 * item(bridgeMoments(0.5).variance) // => 0.25
 * item(bridgeMoments(0.25, { x1: 4 }).mean) // => 1
 * ```
 */
export function bridgeMoments(
  t: TensorLike,
  options: BridgeMomentOptions = {},
): { readonly mean: Value; readonly variance: Value } {
  const t0 = options.t0 ?? 0
  const t1 = options.t1 ?? 1
  if (!(t1 > t0)) throw new OmaiError('value', 'bridgeMoments: need t1 > t0')
  const [tt, x0, x1, sigma] = promoteFloat(t, options.x0 ?? 0, options.x1 ?? 0, options.sigma ?? 1) as [
    Value,
    Value,
    Value,
    Value,
  ]
  const u = div(sub(tt, t0), t1 - t0)
  return {
    mean: add(x0, mul(sub(x1, x0), u)),
    variance: mul(square(sigma), div(mul(sub(tt, t0), sub(t1, tt)), t1 - t0)),
  }
}

/**
 * One level of the Levy (bridge) construction: insert the midpoint of every grid interval, drawn from the Brownian
 * bridge between its neighbours, N((a + b) / 2, sigma^2 dt / 4). The refined paths are exact Brownian paths on the finer
 * grid with the old points unchanged.
 *
 * @param key - The key.
 * @param paths - Brownian paths (scalar or vector state).
 * @param sigma - The volatility (default 1).
 * @returns Paths on the grid with every midpoint added (`2 n - 1` times).
 * @concept brownian-bridge
 * @example
 * ```ts
 * import { key } from 'omai-engine/random'
 * import { bridgeRefine, brownianMotion, timeGrid } from 'omai-engine/stochastic'
 * Array.from(bridgeRefine(key(1), brownianMotion(key(0), timeGrid(1, 2))).times) // => [0, 0.25, 0.5, 0.75, 1]
 * ```
 */
export function bridgeRefine(key: Key, paths: Paths, sigma: TensorLike = 1): Paths {
  const x = asTensor(paths.values)
  const n = paths.times.length
  const dts = stepsOf(paths.times)
  const parts: Value[] = []
  const times: number[] = []
  const rest = x.shape.slice(2)
  for (let i = 0; i < n; i++) {
    const a = slice1(x, i)
    parts.push(a)
    times.push(paths.times[i] ?? 0)
    if (i === n - 1) break
    const b = slice1(x, i + 1)
    const z = normal(foldIn(key, i), [x.shape[0] ?? 1, ...rest])
    parts.push(add(mul(add(a, b), 0.5), mul(mul(sigma, z), Math.sqrt((dts[i] ?? 0) / 4))))
    times.push((paths.times[i] ?? 0) + (dts[i] ?? 0) / 2)
  }
  return { times: Float64Array.from(times), values: stack(parts, 1) }
}

/** Options of {@link quadraticVariation}. */
export interface VariationOptions {
  /** Return the running sum at every grid time (starting at 0) instead of the total. */
  readonly cumulative?: boolean
}

function cumulate(terms: Value, cumulative: boolean): Value {
  if (!cumulative) return sum(terms, 1)
  const n = terms.shape[1] ?? 0
  const out: Value[] = [zeros([terms.shape[0] ?? 1, ...terms.shape.slice(2)])]
  for (let i = 0; i < n; i++) out.push(add(out[i] as Value, slice1(terms, i)))
  return stack(out, 1)
}

/**
 * The realised quadratic variation sum_i (X_{t_{i+1}} - X_{t_i})^2, which converges to sigma^2 T for Brownian motion as
 * the grid is refined (and to the integral of the squared diffusion for an Ito process). Differentiable.
 *
 * @param x - Paths, or values `[paths, n, ...state]` (a single path `[n]` is allowed).
 * @param options - `cumulative` for the running variation.
 * @returns `[paths, ...state]`, or `[paths, n, ...state]` when cumulative.
 * @concept quadratic-variation
 * @example
 * ```ts
 * import { quadraticVariation } from 'omai-engine/stochastic'
 * import { toArray } from 'omai-engine/tensor'
 * toArray(quadraticVariation([0, 1, -1, 0])) // => [6]
 * ```
 */
export function quadraticVariation(x: Paths | TensorLike, options: VariationOptions = {}): Value {
  return cumulate(square(pathIncrements(valuesOf(x))), options.cumulative ?? false)
}

/**
 * The realised covariation sum_i dX_i dY_i of two processes on the same grid ([W, W] = t, [W, B] = 0 for independent
 * Brownian motions, rho t for correlated ones). Differentiable.
 *
 * @param x - Paths or values.
 * @param y - Paths or values of the same shape.
 * @param options - `cumulative` for the running covariation.
 * @returns `[paths, ...state]`, or the running sums.
 * @concept quadratic-variation
 * @example
 * ```ts
 * import { covariation } from 'omai-engine/stochastic'
 * import { toArray } from 'omai-engine/tensor'
 * toArray(covariation([0, 1, 2], [0, -1, 1])) // => [1]
 * ```
 */
export function covariation(x: Paths | TensorLike, y: Paths | TensorLike, options: VariationOptions = {}): Value {
  return cumulate(mul(pathIncrements(valuesOf(x)), pathIncrements(valuesOf(y))), options.cumulative ?? false)
}

/**
 * The running maximum max_{s <= t_i} X_s of each path along time. Differentiable (almost everywhere).
 *
 * @param x - Paths or values.
 * @returns `[paths, n, ...state]`.
 * @concept reflection-principle
 * @example
 * ```ts
 * import { runningMaximum } from 'omai-engine/stochastic'
 * import { toArray } from 'omai-engine/tensor'
 * toArray(runningMaximum([0, 2, 1, 3])) // => [[0, 2, 2, 3]]
 * ```
 */
export function runningMaximum(x: Paths | TensorLike): Value {
  const v = valuesOf(x)
  const out: Value[] = [slice1(v, 0)]
  for (let i = 1; i < (v.shape[1] ?? 0); i++) out.push(maximum(out[i - 1] as Value, slice1(v, i)))
  return stack(out, 1)
}

/**
 * The reflection principle as a path transformation: after a path first reaches `level` it is reflected about it,
 * X~_t = 2 level - X_t. For Brownian motion the reflected path is again a Brownian motion, which gives
 * P(max W >= a) = 2 P(W_t >= a). Paths starting below the level are reflected at their first crossing upwards, those
 * starting above at their first crossing downwards.
 *
 * @param x - Paths or values (scalar state).
 * @param level - The reflecting level.
 * @returns The reflected values `[paths, n]`.
 * @concept reflection-principle
 * @example
 * ```ts
 * import { reflectPath } from 'omai-engine/stochastic'
 * import { toArray } from 'omai-engine/tensor'
 * toArray(reflectPath([0, 1, 2, 1.5], 1)) // => [[0, 1, 0, 0.5]]
 * ```
 */
export function reflectPath(x: Paths | TensorLike, level: TensorLike): Value {
  const v = valuesOf(x)
  const start = slice1(v, 0)
  const above = runningMaximum(v)
  const below = neg(runningMaximum(neg(v)))
  const startsBelow = expandDims(lessEqual(start, level), 1)
  const reached = where(startsBelow, greaterEqual(above, level), lessEqual(below, level))
  return where(reached, sub(mul(level, 2), v), v)
}

/** Drift and volatility of a scalar Brownian motion with drift. */
export interface DriftOptions {
  /** The drift (default 0). */
  readonly drift?: TensorLike
  /** The volatility (default 1). */
  readonly sigma?: TensorLike
}

/**
 * The distribution of the running maximum by the reflection principle: P(max_{s <= t} X_s <= m) for X = mu s + sigma W_s
 * started at 0, Phi((m - mu t)/(sigma sqrt t)) - exp(2 mu m / sigma^2) Phi((-m - mu t)/(sigma sqrt t)) for m >= 0 (and
 * 0 for m < 0). Differentiable in every argument.
 *
 * @param m - Levels (non-negative).
 * @param t - Horizons (positive).
 * @param options - Drift and volatility.
 * @returns The probabilities.
 * @concept reflection-principle
 * @example
 * ```ts
 * import { maximumCdf } from 'omai-engine/stochastic'
 * import { item } from 'omai-engine/tensor'
 * Math.round(item(maximumCdf(0, 1)) * 1e12) / 1e12 // => 0
 * Math.abs(item(maximumCdf(1, 1)) - 0.682689492137) < 1e-11 // => true
 * ```
 */
export function maximumCdf(m: TensorLike, t: TensorLike, options: DriftOptions = {}): Value {
  const [mm, tt, mu, sigma] = promoteFloat(m, t, options.drift ?? 0, options.sigma ?? 1) as [Value, Value, Value, Value]
  const safe = maximum(mm, 0)
  const s = mul(sigma, sqrt(tt))
  const a = ndtr(div(sub(safe, mul(mu, tt)), s))
  const logB = add(div(mul(mul(mu, safe), 2), square(sigma)), logNdtr(div(sub(neg(safe), mul(mu, tt)), s)))
  return where(greaterEqual(mm, 0), maximum(sub(a, exp(logB)), 0), 0)
}

function towardLevel(level: Value, mu: Value): { a: Value; mu: Value } {
  const up = greaterEqual(level, 0)
  return { a: where(up, level, neg(level)), mu: where(up, mu, neg(mu)) }
}

/**
 * The first-passage CDF P(tau_a <= t) of a Brownian motion with drift started at 0 to the level a (either side), by the
 * reflection principle: 1 - P(max <= |a|) with the drift's sign turned towards the level. With a drift away from the
 * level the distribution is defective: P(tau < infinity) = exp(-2 |mu a| / sigma^2). Differentiable.
 *
 * @param t - Times (positive).
 * @param level - The level a (non-zero).
 * @param options - Drift and volatility.
 * @returns The probabilities.
 * @concept first-passage-time
 * @example
 * ```ts
 * import { firstPassageCdf } from 'omai-engine/stochastic'
 * import { item } from 'omai-engine/tensor'
 * Math.abs(item(firstPassageCdf(1, -1)) - 0.317310507863) < 1e-11 // => true
 * ```
 */
export function firstPassageCdf(t: TensorLike, level: TensorLike, options: DriftOptions = {}): Value {
  const [tt, l, mu0, sigma] = promoteFloat(t, level, options.drift ?? 0, options.sigma ?? 1) as [
    Value,
    Value,
    Value,
    Value,
  ]
  const { a, mu } = towardLevel(l, mu0)
  return sub(1, maximumCdf(a, tt, { drift: mu, sigma }))
}

/**
 * The first-passage density a / (sigma sqrt(2 pi t^3)) exp(-(a - mu t)^2 / (2 sigma^2 t)) of a Brownian motion with drift
 * started at 0 to the level a (the inverse Gaussian density for a drift towards the level, Levy's for none).
 * Differentiable.
 *
 * @param t - Times (positive).
 * @param level - The level (non-zero).
 * @param options - Drift and volatility.
 * @returns The densities.
 * @concept first-passage-time
 * @example
 * ```ts
 * import { firstPassageDensity } from 'omai-engine/stochastic'
 * import { item } from 'omai-engine/tensor'
 * Math.abs(item(firstPassageDensity(1, 1)) - Math.exp(-0.5) / Math.sqrt(2 * Math.PI)) < 1e-15 // => true
 * ```
 */
export function firstPassageDensity(t: TensorLike, level: TensorLike, options: DriftOptions = {}): Value {
  const [tt, l, mu0, sigma] = promoteFloat(t, level, options.drift ?? 0, options.sigma ?? 1) as [
    Value,
    Value,
    Value,
    Value,
  ]
  const { a, mu } = towardLevel(l, mu0)
  const logDensity = sub(
    sub(log(a), add(log(sigma), mul(add(log(tt), Math.log(2 * Math.PI) / 3), 1.5))),
    div(square(sub(a, mul(mu, tt))), mul(mul(square(sigma), tt), 2)),
  )
  return exp(logDensity)
}

/**
 * The first-passage time to a level as a distribution: inverse Gaussian with mean |a| / mu and shape a^2 / sigma^2 for
 * a drift towards the level, and Levy (an inverse gamma with shape 1/2 and scale a^2 / (2 sigma^2)) for none.
 *
 * @param level - The level (non-zero number).
 * @param options - Drift and volatility (numbers).
 * @returns The distribution of tau.
 * @throws OmaiError with code `value` for a drift away from the level (the passage time is infinite with positive
 *   probability) or a zero level.
 * @concept first-passage-time
 * @example
 * ```ts
 * import { firstPassageDistribution } from 'omai-engine/stochastic'
 * import { item } from 'omai-engine/tensor'
 * item(firstPassageDistribution(2, { drift: 0.5 }).mean()) // => 4
 * firstPassageDistribution(1).name // => 'inverseGamma'
 * ```
 */
export function firstPassageDistribution(
  level: number,
  options: { drift?: number; sigma?: number } = {},
): Distribution {
  const mu = options.drift ?? 0
  const sigma = options.sigma ?? 1
  if (level === 0 || !Number.isFinite(level)) throw new OmaiError('value', 'firstPassageDistribution: level != 0')
  const a = Math.abs(level)
  const towards = Math.sign(level) * mu
  if (towards < 0) {
    throw new OmaiError('value', 'firstPassageDistribution: the drift points away from the level (defective law)')
  }
  if (towards === 0) return inverseGamma(0.5, (a * a) / (2 * sigma * sigma))
  return inverseGaussian(a / towards, (a * a) / (sigma * sigma))
}

/**
 * The first time each path reaches a level, interpolating linearly between grid points (Infinity for paths that never
 * reach it). The level is approached from the side of each path's start.
 *
 * @param paths - Scalar paths.
 * @param level - The level.
 * @returns One time per path.
 * @concept first-passage-time
 * @example
 * ```ts
 * import { firstPassageTimes } from 'omai-engine/stochastic'
 * Array.from(firstPassageTimes({ times: Float64Array.from([0, 1, 2]), values: [[0, 0.5, 1.5], [0, -1, -2]] }, 1)) // => [1.5, Infinity]
 * ```
 */
export function firstPassageTimes(paths: Paths, level: number): Float64Array {
  const v = valuesOf(paths)
  const [p = 1, n = 0] = v.shape
  if (v.shape.length !== 2) throw new OmaiError('shape', 'firstPassageTimes: needs scalar paths [paths, n]')
  const data = primalNumbers(v)
  const out = new Float64Array(p).fill(Number.POSITIVE_INFINITY)
  for (let j = 0; j < p; j++) {
    const x0 = data[j * n] ?? 0
    const up = x0 <= level
    if (x0 === level) {
      out[j] = paths.times[0] ?? 0
      continue
    }
    for (let i = 1; i < n; i++) {
      const a = data[j * n + i - 1] ?? 0
      const b = data[j * n + i] ?? 0
      if (up ? b >= level : b <= level) {
        const ta = paths.times[i - 1] ?? 0
        const tb = paths.times[i] ?? 0
        out[j] = ta + ((level - a) / (b - a)) * (tb - ta)
        break
      }
    }
  }
  return out
}
