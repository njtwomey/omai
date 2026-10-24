import { grad, hessian } from '../autodiff/index.ts'
import { WhiteboxError } from '../base/index.ts'
import type { Distribution } from '../dists/index.ts'
import { fsolve } from '../numerics/roots/index.ts'
import { exponential, foldIn, type Key, normal, poisson, uniform } from '../random/index.ts'
import {
  add,
  asFloat,
  asTensor,
  div,
  exp,
  expandDims,
  fromStorage,
  greater,
  log,
  mul,
  neg,
  promoteFloat,
  sliceAxis,
  sub,
  sum,
  type Tensor,
  type TensorLike,
  type Value,
  where,
  zeros,
} from '../tensor/index.ts'
import { type Algorithm, channels, runAlgorithm } from '../trace/index.ts'
import { positiveInteger, primalNumbers, traceOf } from './_shared.ts'
import { type CovarianceFunction, sampleGaussianProcess } from './gaussian-processes.ts'
import { type History, historyArray, type TraceOptions } from './paths.ts'

/** The state of an event simulation: the clock, the accepted events and whether the horizon was passed. */
export interface EventState {
  readonly t: number
  readonly events: History<number> | undefined
  readonly count: number
  /** Process-specific memory (the excitation of a Hawkes process). */
  readonly memory: number
  readonly finished: boolean
}

/** One proposal of an event simulation: the candidate time, whether it is accepted, and the memory after it. */
interface Proposal {
  readonly t: number
  readonly accept: boolean
  readonly memory: number
}

function eventAlgorithm(
  name: string,
  config: Record<string, number>,
  horizon: number,
  propose: (s: EventState, key: Key) => Proposal,
): Algorithm<EventState> {
  if (!(horizon > 0 && Number.isFinite(horizon))) throw new WhiteboxError('value', `${name}: the horizon must be > 0`)
  return {
    name,
    config: { ...config, horizon },
    schema: channels({
      metrics: { t: { description: 'the clock' }, count: { description: 'events so far' } },
    }),
    init: () => ({ t: 0, events: undefined, count: 0, memory: 0, finished: false }),
    step: (s, ctx) => {
      const p = propose(s, ctx.key)
      if (!(p.t > s.t) || p.t > horizon) return { ...s, finished: true }
      return p.accept
        ? { t: p.t, events: { item: p.t, previous: s.events }, count: s.count + 1, memory: p.memory, finished: false }
        : { ...s, t: p.t, memory: p.memory }
    },
    done: (s) => (s.finished ? 'completed' : undefined),
    observe: (s) => ({ metrics: { t: s.t, count: s.count } }),
  }
}

/** Options of the event simulations: the proposal cap and tracing. */
export interface EventOptions extends TraceOptions {
  /** The largest number of proposals (default 1e6). */
  readonly maxEvents?: number
}

function runEvents(key: Key, alg: Algorithm<EventState>, options: EventOptions): Float64Array {
  const cap = options.maxEvents ?? 1e6
  const r = runAlgorithm(alg, { ...traceOf(options), key, maxSteps: cap + 1 })
  if (!r.state.finished) throw new WhiteboxError('convergence', `${alg.name}: more than ${cap} proposals`)
  return Float64Array.from(historyArray(r.state.events))
}

function draw(key: Key, rate: number): number {
  return exponential(key, [1], rate).data[0] ?? Number.POSITIVE_INFINITY
}

/**
 * A homogeneous Poisson process on [0, T] as a traced event simulation: exponential gaps with the given rate.
 *
 * @param key - The key.
 * @param rate - The intensity (>= 0).
 * @param horizon - T.
 * @param options - The event cap and tracing.
 * @returns The event times, increasing.
 * @concept poisson-process
 * @example
 * ```ts
 * import { key } from 'whitebox-engine/random'
 * import { poissonProcess } from 'whitebox-engine/stochastic'
 * poissonProcess(key(0), 0, 10).length // => 0
 * poissonProcess(key(0), 5, 2).every((t) => t > 0 && t <= 2) // => true
 * ```
 */
export function poissonProcess(key: Key, rate: number, horizon: number, options: EventOptions = {}): Float64Array {
  if (!(rate >= 0)) throw new WhiteboxError('value', 'poissonProcess: the rate must be >= 0')
  if (rate === 0) return new Float64Array(0)
  const alg = eventAlgorithm('poisson-process', { rate }, horizon, (s, k) => ({
    t: s.t + draw(k, rate),
    accept: true,
    memory: 0,
  }))
  return runEvents(key, alg, options)
}

/**
 * An inhomogeneous Poisson process on [0, T] with intensity lambda(t) <= lambdaMax, by Lewis-Shedler thinning: propose
 * the points of a rate-lambdaMax process and keep each with probability lambda(t) / lambdaMax.
 *
 * @param key - The key.
 * @param intensity - lambda(t).
 * @param horizon - T.
 * @param maxIntensity - A bound lambdaMax on lambda over [0, T].
 * @param options - The proposal cap and tracing.
 * @returns The event times.
 * @throws WhiteboxError with code `value` when the intensity exceeds the bound.
 * @concept poisson-process
 * @example
 * ```ts
 * import { key } from 'whitebox-engine/random'
 * import { inhomogeneousPoisson } from 'whitebox-engine/stochastic'
 * inhomogeneousPoisson(key(0), (t) => (t < 1 ? 0 : 3), 2, 3).every((t) => t >= 1) // => true
 * ```
 */
export function inhomogeneousPoisson(
  key: Key,
  intensity: (t: number) => number,
  horizon: number,
  maxIntensity: number,
  options: EventOptions = {},
): Float64Array {
  if (!(maxIntensity > 0)) throw new WhiteboxError('value', 'inhomogeneousPoisson: the bound must be positive')
  const alg = eventAlgorithm('thinning', { maxIntensity }, horizon, (s, k) => {
    const t = s.t + draw(foldIn(k, 0), maxIntensity)
    const lambda = t <= horizon ? intensity(t) : 0
    if (!(lambda >= 0) || lambda > maxIntensity * (1 + 1e-12)) {
      throw new WhiteboxError('value', `inhomogeneousPoisson: intensity ${lambda} at ${t} is outside [0, bound]`)
    }
    return { t, accept: (uniform(foldIn(k, 1), [1]).data[0] ?? 1) * maxIntensity < lambda, memory: 0 }
  })
  return runEvents(key, alg, options)
}

/**
 * The counting process N(t) = #{events <= t} of a point pattern at given times.
 *
 * @param events - Increasing event times.
 * @param times - Query times.
 * @returns The counts.
 * @concept counting-process
 * @example
 * ```ts
 * import { countingProcess } from 'whitebox-engine/stochastic'
 * Array.from(countingProcess([0.5, 1.2, 3], [0, 1, 2, 5])) // => [0, 1, 2, 3]
 * ```
 */
export function countingProcess(events: ArrayLike<number>, times: ArrayLike<number>): Int32Array {
  const out = new Int32Array(times.length)
  for (let k = 0; k < times.length; k++) {
    let lo = 0
    let hi = events.length
    const t = times[k] ?? 0
    while (lo < hi) {
      const mid = (lo + hi) >> 1
      if ((events[mid] ?? 0) <= t) lo = mid + 1
      else hi = mid
    }
    out[k] = lo
  }
  return out
}

/**
 * A renewal process on [0, T]: independent identically distributed gaps from the interarrival law (a Poisson process for
 * exponential gaps), as a traced event simulation.
 *
 * @param key - The key.
 * @param interarrival - The law of the gaps (non-negative).
 * @param horizon - T.
 * @param options - The event cap and tracing.
 * @returns The renewal times.
 * @concept renewal-process
 * @example
 * ```ts
 * import { gamma } from 'whitebox-engine/dists'
 * import { key } from 'whitebox-engine/random'
 * import { renewalProcess } from 'whitebox-engine/stochastic'
 * renewalProcess(key(0), gamma(2, 2), 10).every((t) => t <= 10) // => true
 * ```
 */
export function renewalProcess(
  key: Key,
  interarrival: Distribution,
  horizon: number,
  options: EventOptions = {},
): Float64Array {
  const alg = eventAlgorithm('renewal-process', {}, horizon, (s, k) => {
    const gap = primalNumbers(interarrival.sample(k, [1]))[0] ?? Number.NaN
    if (!(gap >= 0)) throw new WhiteboxError('value', 'renewalProcess: gaps must be non-negative')
    // a zero gap would not advance the clock: nudge by the smallest step
    return { t: s.t + Math.max(gap, Number.MIN_VALUE), accept: true, memory: 0 }
  })
  return runEvents(key, alg, options)
}

/**
 * The renewal function m(t) = E N(t) on a grid, by solving the renewal equation m(t) = F(t) + int_0^t m(t - s) dF(s)
 * with the Stieltjes increments of the gap CDF (exact as the grid is refined; m(t) = lambda t for Poisson processes and
 * m(t) / t tends to 1 / E[gap] by the elementary renewal theorem).
 *
 * @param interarrival - The law of the gaps (with a CDF).
 * @param horizon - The end of the grid.
 * @param steps - The number of grid steps (default 1000).
 * @returns `{ times, values }` with `steps + 1` points.
 * @concept renewal-process
 * @example
 * ```ts
 * import { exponential } from 'whitebox-engine/dists'
 * import { renewalFunction } from 'whitebox-engine/stochastic'
 * Math.abs(renewalFunction(exponential(2), 3, 3000).values.at(-1) - 6) < 0.01 // => true
 * ```
 */
export function renewalFunction(
  interarrival: Distribution,
  horizon: number,
  steps = 1000,
): { readonly times: Float64Array; readonly values: Float64Array } {
  positiveInteger(steps, 'steps', 'renewalFunction')
  if (!(horizon > 0)) throw new WhiteboxError('value', 'renewalFunction: the horizon must be positive')
  const h = horizon / steps
  const times = Float64Array.from({ length: steps + 1 }, (_, i) => i * h)
  const f = primalNumbers(interarrival.cdf(fromStorage(times, [steps + 1], 'float64')))
  const m = new Float64Array(steps + 1)
  // midpoint rule on the Stieltjes increments dF_j = F(j h) - F((j - 1) h), with m between grid points averaged
  const df0 = f[1] ?? 0
  for (let i = 1; i <= steps; i++) {
    let s = 0
    for (let j = 2; j <= i; j++) {
      const dF = (f[j] ?? 0) - (f[j - 1] ?? 0)
      s += dF * (((m[i - j + 1] ?? 0) + (m[i - j] ?? 0)) / 2)
    }
    // the j = 1 term involves m(t_i) itself: m_i = F_i + s + dF_1 (m_i + m_{i-1}) / 2
    m[i] = ((f[i] ?? 0) + s + (df0 * (m[i - 1] ?? 0)) / 2) / (1 - df0 / 2)
  }
  return { times, values: m }
}

/** Parameters of a Hawkes process with an exponential kernel. */
export interface HawkesParams {
  /** The baseline intensity mu > 0. */
  readonly mu: TensorLike
  /** The jump of the intensity at each event, alpha >= 0. */
  readonly alpha: TensorLike
  /** The decay rate beta > 0 (the branching ratio is alpha / beta). */
  readonly beta: TensorLike
}

function hawkesNumbers(p: HawkesParams): { mu: number; alpha: number; beta: number } {
  const [mu, alpha, beta] = [p.mu, p.alpha, p.beta].map((x) => primalNumbers(x)[0] ?? Number.NaN) as [
    number,
    number,
    number,
  ]
  if (!(mu > 0 && alpha >= 0 && beta > 0)) throw new WhiteboxError('value', 'hawkes: need mu > 0, alpha >= 0, beta > 0')
  return { mu, alpha, beta }
}

/**
 * A self-exciting Hawkes process with intensity lambda(t) = mu + sum_{t_i < t} alpha e^{-beta (t - t_i)}, simulated by
 * Ogata's thinning: between events the intensity decreases, so its current value bounds it until the next proposal;
 * a proposal at t' is kept with probability lambda(t') / bound. Stationary when alpha < beta, with mean intensity
 * mu / (1 - alpha / beta).
 *
 * @param key - The key.
 * @param params - mu, alpha and beta.
 * @param horizon - T.
 * @param options - The proposal cap and tracing.
 * @returns The event times.
 * @concept hawkes-process
 * @reference Y. Ogata, On Lewis' simulation method for point processes, IEEE Trans. Inf. Theory 27 (1981).
 * @example
 * ```ts
 * import { key } from 'whitebox-engine/random'
 * import { hawkesProcess } from 'whitebox-engine/stochastic'
 * hawkesProcess(key(0), { mu: 1, alpha: 0.5, beta: 1 }, 10).every((t) => t <= 10) // => true
 * ```
 */
export function hawkesProcess(
  key: Key,
  params: HawkesParams,
  horizon: number,
  options: EventOptions = {},
): Float64Array {
  const { mu, alpha, beta } = hawkesNumbers(params)
  const alg = eventAlgorithm('hawkes-ogata', { mu, alpha, beta }, horizon, (s, k) => {
    // memory holds the excitation sum at the clock (right limit)
    const bound = mu + s.memory
    const t = s.t + draw(foldIn(k, 0), bound)
    const decayed = s.memory * Math.exp(-beta * (t - s.t))
    const accept = (uniform(foldIn(k, 1), [1]).data[0] ?? 1) * bound <= mu + decayed
    return { t, accept, memory: accept ? decayed + alpha : decayed }
  })
  return runEvents(key, alg, options)
}

/**
 * The conditional intensity of an exponential Hawkes process at given times given the events before them.
 *
 * @param params - mu, alpha and beta.
 * @param events - The event times (increasing).
 * @param times - Query times.
 * @returns lambda(t) (left limits at event times).
 * @concept hawkes-process
 * @example
 * ```ts
 * import { hawkesIntensity } from 'whitebox-engine/stochastic'
 * import { toArray } from 'whitebox-engine/tensor'
 * toArray(hawkesIntensity({ mu: 1, alpha: 2, beta: 1 }, [0], [0, 1])).map((v) => Math.round(v * 1e12) / 1e12) // => [1, 1.735758882343]
 * ```
 */
export function hawkesIntensity(params: HawkesParams, events: TensorLike, times: TensorLike): Value {
  const [mu, alpha, beta] = promoteFloat(params.mu, params.alpha, params.beta) as [Value, Value, Value]
  const ev = expandDims(asFloat(asTensor(events)), 0)
  const t = expandDims(asFloat(asTensor(times)), 1)
  const lag = sub(t, ev)
  const before = greater(lag, 0)
  const kernel = where(before, exp(neg(mul(beta, where(before, lag, 0)))), 0)
  return add(mu, mul(alpha, sum(kernel, 1)))
}

/**
 * The log-likelihood of event times on [0, T] under an exponential Hawkes process,
 * sum_i log lambda(t_i) - mu T - (alpha / beta) sum_i (1 - e^{-beta (T - t_i)}), computed for all events at once.
 * Differentiable in the parameters.
 *
 * @param params - mu, alpha and beta (values, for gradients).
 * @param events - The event times in [0, T].
 * @param horizon - T.
 * @returns The log-likelihood.
 * @concept hawkes-process
 * @example
 * ```ts
 * import { hawkesLogLikelihood } from 'whitebox-engine/stochastic'
 * import { item } from 'whitebox-engine/tensor'
 * // no excitation: a Poisson process, n log mu - mu T
 * Math.abs(item(hawkesLogLikelihood({ mu: 2, alpha: 0, beta: 1 }, [0.5, 1.5], 3)) - (2 * Math.log(2) - 6)) < 1e-12 // => true
 * ```
 */
export function hawkesLogLikelihood(params: HawkesParams, events: TensorLike, horizon: number): Value {
  const [mu, alpha, beta] = promoteFloat(params.mu, params.alpha, params.beta) as [Value, Value, Value]
  const ev = asFloat(asTensor(events))
  const n = ev.shape[0] ?? 0
  if (n === 0) return neg(mul(mu, horizon))
  const lambda = hawkesIntensity({ mu, alpha, beta }, ev, ev)
  const tail = sum(sub(1, exp(neg(mul(beta, sub(horizon, ev))))))
  return sub(sub(sum(log(lambda)), mul(mu, horizon)), mul(div(alpha, beta), tail))
}

/** The fitted Hawkes model. */
export interface HawkesFit {
  readonly mu: number
  readonly alpha: number
  readonly beta: number
  readonly logLikelihood: number
  readonly iterations: number
  readonly converged: boolean
}

/**
 * Maximum-likelihood estimation of an exponential Hawkes process: Newton's method (numerics `fsolve` with the Hessian
 * from autodiff) on the score equations in log-parameters, so the estimates stay positive.
 *
 * @param events - The event times in [0, T].
 * @param horizon - T.
 * @param options - The starting point (default: mu = n / (2T), alpha = beta / 2, beta = n / T) and the iteration cap.
 * @returns The estimates and the maximised log-likelihood.
 * @concept hawkes-process
 * @example
 * ```ts
 * import { key } from 'whitebox-engine/random'
 * import { fitHawkes, hawkesProcess } from 'whitebox-engine/stochastic'
 * const events = hawkesProcess(key(0), { mu: 1, alpha: 0.6, beta: 1.5 }, 60)
 * fitHawkes(events, 60).converged // => true
 * ```
 */
export function fitHawkes(
  events: TensorLike,
  horizon: number,
  options: { readonly init?: { mu: number; alpha: number; beta: number }; readonly maxIter?: number } = {},
): HawkesFit {
  const ev = asFloat(asTensor(events))
  const n = ev.shape[0] ?? 0
  if (n < 3) throw new WhiteboxError('value', 'fitHawkes: needs at least three events')
  const rate = n / horizon
  const init = options.init ?? { mu: rate / 2, alpha: rate / 2, beta: rate }
  const nll = (logp: Value): Value => {
    const p = exp(logp)
    return neg(
      hawkesLogLikelihood(
        { mu: sliceAxis(p, 0, 0, 1), alpha: sliceAxis(p, 0, 1, 2), beta: sliceAxis(p, 0, 2, 3) },
        ev,
        horizon,
      ),
    )
  }
  const score = (logp: Value): Value => grad(nll)(logp)
  const res = fsolve(score, [Math.log(init.mu), Math.log(init.alpha), Math.log(init.beta)], {
    jacobian: (x) => hessian(nll)(x),
    maxIterations: options.maxIter ?? 100,
    ftol: 1e-10,
  })
  const [mu = 0, alpha = 0, beta = 0] = primalNumbers(res.x).map(Math.exp)
  const ll = primalNumbers(hawkesLogLikelihood({ mu, alpha, beta }, ev, horizon))[0] ?? Number.NaN
  return { mu, alpha, beta, logLikelihood: ll, iterations: res.iterations, converged: res.converged }
}

/**
 * A Cox (doubly stochastic Poisson) process: draw a random intensity function, then a Poisson process with that
 * intensity by thinning.
 *
 * @param key - The key.
 * @param randomIntensity - Draws an intensity function and a bound on it from a key.
 * @param horizon - T.
 * @param options - The proposal cap and tracing.
 * @returns The event times and the intensity drawn.
 * @concept cox-process
 * @example
 * ```ts
 * import { key, uniform } from 'whitebox-engine/random'
 * import { coxProcess } from 'whitebox-engine/stochastic'
 * // a mixed Poisson process: the rate is uniform on [1, 5]
 * const c = coxProcess(key(0), (k) => { const r = 1 + 4 * uniform(k, [1]).data[0]; return { intensity: () => r, max: r } }, 4)
 * c.events.every((t) => t <= 4) // => true
 * ```
 */
export function coxProcess(
  key: Key,
  randomIntensity: (key: Key) => { readonly intensity: (t: number) => number; readonly max: number },
  horizon: number,
  options: EventOptions = {},
): { readonly events: Float64Array; readonly intensity: (t: number) => number } {
  const drawn = randomIntensity(foldIn(key, 0))
  return {
    events: inhomogeneousPoisson(foldIn(key, 1), drawn.intensity, horizon, drawn.max, options),
    intensity: drawn.intensity,
  }
}

/**
 * A log-Gaussian Cox process on a grid of cells: log-intensities from a Gaussian process at the cell centres (with the
 * given mean), and independent Poisson counts with means exp(log-intensity) times the cell size.
 *
 * @param key - The key.
 * @param centres - The cell centres `[cells]` or `[cells, d]`.
 * @param cellSize - The length (area, volume) of each cell.
 * @param options - The GP covariance and mean (default 0) of the log-intensity.
 * @returns The intensities and counts per cell.
 * @concept cox-process
 * @example
 * ```ts
 * import { key } from 'whitebox-engine/random'
 * import { logGaussianCoxProcess, ouCovariance } from 'whitebox-engine/stochastic'
 * logGaussianCoxProcess(key(0), [0.5, 1.5, 2.5], 1, { covariance: ouCovariance({ theta: 1, sigma: 0.5 }), mean: 1 }).counts.length // => 3
 * ```
 */
export function logGaussianCoxProcess(
  key: Key,
  centres: TensorLike,
  cellSize: number,
  options: { readonly covariance: CovarianceFunction; readonly mean?: TensorLike },
): { readonly intensity: Float64Array; readonly counts: Int32Array } {
  if (!(cellSize > 0)) throw new WhiteboxError('value', 'logGaussianCoxProcess: the cell size must be positive')
  const c = asFloat(asTensor(centres))
  const cells = c.shape[0] ?? 0
  const meanValue = options.mean ?? 0
  const field = sampleGaussianProcess(foldIn(key, 0), options.covariance, c, {
    mean: add(zeros([cells]), meanValue),
  })
  const intensity = Float64Array.from(primalNumbers(field), Math.exp)
  const counts = Int32Array.from(intensity, (lam, i) => poisson(foldIn(key, 1 + i), lam * cellSize, [1]).data[0] ?? 0)
  return { intensity, counts }
}

/**
 * A marked Poisson process: event times of a homogeneous Poisson process with independent marks from a law.
 *
 * @param key - The key.
 * @param rate - The intensity.
 * @param horizon - T.
 * @param marks - The mark law.
 * @param options - The event cap and tracing.
 * @returns The times and their marks.
 * @concept marked-point-process
 * @example
 * ```ts
 * import { exponential } from 'whitebox-engine/dists'
 * import { key } from 'whitebox-engine/random'
 * import { markedPoissonProcess } from 'whitebox-engine/stochastic'
 * const m = markedPoissonProcess(key(0), 3, 5, exponential(1))
 * m.marks.length === m.times.length // => true
 * ```
 */
export function markedPoissonProcess(
  key: Key,
  rate: number,
  horizon: number,
  marks: Distribution,
  options: EventOptions = {},
): { readonly times: Float64Array; readonly marks: Float64Array } {
  const times = poissonProcess(foldIn(key, 0), rate, horizon, options)
  const m = times.length === 0 ? [] : primalNumbers(marks.sample(foldIn(key, 1), [times.length]))
  return { times, marks: Float64Array.from(m) }
}

/** A rectangular observation window. */
export interface Window {
  readonly xmin: number
  readonly xmax: number
  readonly ymin: number
  readonly ymax: number
}

const UNIT: Window = { xmin: 0, xmax: 1, ymin: 0, ymax: 1 }

function areaOf(w: Window): number {
  const a = (w.xmax - w.xmin) * (w.ymax - w.ymin)
  if (!(a > 0)) throw new WhiteboxError('value', 'window: needs xmax > xmin and ymax > ymin')
  return a
}

function uniformPoints(key: Key, n: number, w: Window): number[] {
  const u = uniform(key, [n, 2]).data
  const out: number[] = []
  for (let i = 0; i < n; i++) {
    out.push(w.xmin + (u[2 * i] ?? 0) * (w.xmax - w.xmin), w.ymin + (u[2 * i + 1] ?? 0) * (w.ymax - w.ymin))
  }
  return out
}

function asPoints(flat: readonly number[]): Tensor<'float64'> {
  return fromStorage(Float64Array.from(flat), [flat.length / 2, 2], 'float64')
}

/**
 * A spatial Poisson process on a rectangle: a Poisson(lambda |W|) number of uniform points, thinned by
 * lambda(x, y) / bound for an inhomogeneous intensity.
 *
 * @param key - The key.
 * @param intensity - A constant intensity, or a function lambda(x, y).
 * @param window - The rectangle (default the unit square).
 * @param maxIntensity - A bound on an intensity function (required for one).
 * @returns The points `[n, 2]`.
 * @throws WhiteboxError with code `value` when a function exceeds its bound.
 * @concept spatial-point-process
 * @example
 * ```ts
 * import { key } from 'whitebox-engine/random'
 * import { spatialPoisson } from 'whitebox-engine/stochastic'
 * spatialPoisson(key(0), 50).shape[1] // => 2
 * ```
 */
export function spatialPoisson(
  key: Key,
  intensity: number | ((x: number, y: number) => number),
  window: Window = UNIT,
  maxIntensity?: number,
): Tensor<'float64'> {
  const area = areaOf(window)
  const bound = typeof intensity === 'number' ? intensity : maxIntensity
  if (bound === undefined || !(bound >= 0)) {
    throw new WhiteboxError('value', 'spatialPoisson: a non-negative intensity (or bound for a function) is needed')
  }
  const n = poisson(foldIn(key, 0), bound * area, [1]).data[0] ?? 0
  const pts = uniformPoints(foldIn(key, 1), n, window)
  if (typeof intensity === 'number') return asPoints(pts)
  const u = uniform(foldIn(key, 2), [n]).data
  const kept: number[] = []
  for (let i = 0; i < n; i++) {
    const x = pts[2 * i] ?? 0
    const y = pts[2 * i + 1] ?? 0
    const lam = intensity(x, y)
    if (!(lam >= 0) || lam > bound * (1 + 1e-12)) throw new WhiteboxError('value', 'spatialPoisson: bound exceeded')
    if ((u[i] ?? 1) * bound < lam) kept.push(x, y)
  }
  return asPoints(kept)
}

/** Parameters of the Neyman-Scott cluster processes. */
export interface ClusterParams {
  /** The intensity kappa of the parent points. */
  readonly parentRate: number
  /** The mean number of children per parent. */
  readonly meanChildren: number
}

function clusterProcess(
  key: Key,
  params: ClusterParams,
  window: Window,
  margin: number,
  offset: (key: Key, n: number) => number[],
): Tensor<'float64'> {
  areaOf(window)
  if (!(params.parentRate >= 0 && params.meanChildren >= 0)) throw new WhiteboxError('value', 'cluster: bad rates')
  // parents in a window enlarged by the margin, so clusters straddling the edge are not lost
  const big = {
    xmin: window.xmin - margin,
    xmax: window.xmax + margin,
    ymin: window.ymin - margin,
    ymax: window.ymax + margin,
  }
  const parents = primalNumbers(spatialPoisson(foldIn(key, 0), params.parentRate, big))
  const kept: number[] = []
  for (let p = 0; p < parents.length / 2; p++) {
    const kids = poisson(foldIn(key, 1 + 2 * p), params.meanChildren, [1]).data[0] ?? 0
    const d = offset(foldIn(key, 2 + 2 * p), kids)
    for (let c = 0; c < kids; c++) {
      const x = (parents[2 * p] ?? 0) + (d[2 * c] ?? 0)
      const y = (parents[2 * p + 1] ?? 0) + (d[2 * c + 1] ?? 0)
      if (x >= window.xmin && x <= window.xmax && y >= window.ymin && y <= window.ymax) kept.push(x, y)
    }
  }
  return asPoints(kept)
}

/**
 * The Thomas cluster process: Poisson parents with intensity kappa, each with a Poisson(mu) number of children displaced
 * by N(0, sigma^2 I). Its K function is pi r^2 + (1 - exp(-r^2 / (4 sigma^2))) / kappa ({@link thomasK}).
 *
 * @param key - The key.
 * @param params - Parent intensity, mean children and the displacement scale sigma.
 * @param window - The rectangle (default the unit square).
 * @returns The points `[n, 2]`.
 * @concept cluster-process
 * @example
 * ```ts
 * import { key } from 'whitebox-engine/random'
 * import { thomasProcess } from 'whitebox-engine/stochastic'
 * thomasProcess(key(0), { parentRate: 10, meanChildren: 5, sigma: 0.02 }).shape[1] // => 2
 * ```
 */
export function thomasProcess(
  key: Key,
  params: ClusterParams & { readonly sigma: number },
  window: Window = UNIT,
): Tensor<'float64'> {
  if (!(params.sigma > 0)) throw new WhiteboxError('value', 'thomasProcess: sigma must be positive')
  return clusterProcess(key, params, window, 4 * params.sigma, (k, n) =>
    Array.from(normal(k, [n, 2]).data, (z) => z * params.sigma),
  )
}

/**
 * The Matern cluster process: Poisson parents with intensity kappa, each with a Poisson(mu) number of children uniform
 * in the disc of radius R around it.
 *
 * @param key - The key.
 * @param params - Parent intensity, mean children and the radius R.
 * @param window - The rectangle (default the unit square).
 * @returns The points `[n, 2]`.
 * @concept cluster-process
 * @example
 * ```ts
 * import { key } from 'whitebox-engine/random'
 * import { maternClusterProcess } from 'whitebox-engine/stochastic'
 * maternClusterProcess(key(0), { parentRate: 10, meanChildren: 5, radius: 0.05 }).shape[1] // => 2
 * ```
 */
export function maternClusterProcess(
  key: Key,
  params: ClusterParams & { readonly radius: number },
  window: Window = UNIT,
): Tensor<'float64'> {
  if (!(params.radius > 0)) throw new WhiteboxError('value', 'maternClusterProcess: the radius must be positive')
  return clusterProcess(key, params, window, params.radius, (k, n) => {
    const u = uniform(k, [n, 2]).data
    const out: number[] = []
    for (let i = 0; i < n; i++) {
      const r = params.radius * Math.sqrt(u[2 * i] ?? 0)
      const a = 2 * Math.PI * (u[2 * i + 1] ?? 0)
      out.push(r * Math.cos(a), r * Math.sin(a))
    }
    return out
  })
}

/**
 * The K function of the Thomas process, pi r^2 + (1 - exp(-r^2 / (4 sigma^2))) / kappa (Poisson's pi r^2 plus the
 * clustering excess).
 *
 * @param r - Distances.
 * @param params - Parent intensity kappa and displacement scale sigma.
 * @returns K(r).
 * @concept ripleys-k
 * @example
 * ```ts
 * import { thomasK } from 'whitebox-engine/stochastic'
 * import { toArray } from 'whitebox-engine/tensor'
 * toArray(thomasK([0], { parentRate: 10, sigma: 0.1 })) // => [0]
 * ```
 */
export function thomasK(r: TensorLike, params: { readonly parentRate: number; readonly sigma: number }): Value {
  const [rr] = promoteFloat(r) as [Value]
  const r2 = mul(rr, rr)
  return add(mul(r2, Math.PI), mul(sub(1, exp(mul(r2, -1 / (4 * params.sigma ** 2)))), 1 / params.parentRate))
}

/** The edge corrections of {@link ripleysK} (astropy.stats.RipleysKEstimator's modes). */
export type RipleyCorrection = 'none' | 'translation' | 'ohser' | 'var-width' | 'ripley'

/**
 * Ripley's K function of a planar point pattern in a rectangle, K(r) = |W| / (n (n - 1)) sum_{i != j} w_ij 1(d_ij < r),
 * with the edge corrections of astropy.stats.RipleysKEstimator: none, translation (Stoyan's 1 / |W cap (W + d_ij)|
 * weights), Ohser's isotropised set covariance, variable width (only points farther than r from the boundary) and
 * Ripley's isotropic correction. Under complete spatial randomness K(r) = pi r^2; clustering raises it.
 *
 * @param points - The points `[n, 2]` inside the window.
 * @param radii - The distances r.
 * @param options - The window (default the unit square) and the correction (default `none`).
 * @returns K at each radius.
 * @concept ripleys-k
 * @reference B. D. Ripley, The second-order analysis of stationary point processes, J. Appl. Prob. 13 (1976).
 * @example
 * ```ts
 * import { ripleysK } from 'whitebox-engine/stochastic'
 * Array.from(ripleysK([[0.1, 0.1], [0.2, 0.1], [0.9, 0.9]], [0.05, 0.2])) // => [0, 0.3333333333333333]
 * ```
 */
export function ripleysK(
  points: TensorLike,
  radii: ArrayLike<number>,
  options: { readonly window?: Window; readonly correction?: RipleyCorrection } = {},
): Float64Array {
  const w = options.window ?? UNIT
  const area = areaOf(w)
  const p = primalNumbers(points)
  const n = p.length / 2
  if (n < 2) throw new WhiteboxError('value', 'ripleysK: needs at least two points')
  const width = w.xmax - w.xmin
  const height = w.ymax - w.ymin
  const out = new Float64Array(radii.length)
  const mode = options.correction ?? 'none'
  if (mode === 'var-width') {
    for (let r = 0; r < radii.length; r++) {
      const rad = radii[r] ?? 0
      let total = 0
      let interior = 0
      for (let i = 0; i < n; i++) {
        const xi = p[2 * i] ?? 0
        const yi = p[2 * i + 1] ?? 0
        const edge = Math.min(Math.min(w.xmax - xi, w.ymax - yi), Math.min(xi - w.xmin, yi - w.ymin))
        if (!(edge > rad)) continue
        interior += 1
        for (let j = 0; j < n; j++)
          if (j !== i && Math.hypot(xi - (p[2 * j] ?? 0), yi - (p[2 * j + 1] ?? 0)) < rad) total += 1
      }
      out[r] = (area * (interior === 0 ? total : total / interior)) / n
    }
    return out
  }
  const ratio = Math.max(height / width, width / height)
  for (let i = 0; i < n - 1; i++) {
    const xi = p[2 * i] ?? 0
    const yi = p[2 * i + 1] ?? 0
    const hor = Math.min(w.xmax - xi, xi - w.xmin)
    const ver = Math.min(w.ymax - yi, yi - w.ymin)
    for (let j = i + 1; j < n; j++) {
      const dx = Math.abs(xi - (p[2 * j] ?? 0))
      const dy = Math.abs(yi - (p[2 * j + 1] ?? 0))
      const d = Math.hypot(dx, dy)
      let weight = 1
      if (mode === 'translation') weight = 1 / ((width - dx) * (height - dy))
      else if (mode === 'ohser') {
        const x = d / Math.sqrt(area / ratio)
        const u = x > 1 ? Math.sqrt(x * x - 1) : 0
        const v = x > ratio && x < Math.sqrt(ratio * ratio + 1) ? Math.sqrt(x * x - ratio * ratio) : 0
        let c = 0
        if (x >= 0 && x <= 1) c = Math.PI - 2 * x * (1 + 1 / ratio) + (x * x) / ratio
        else if (x > 1 && x <= ratio) c = 2 * Math.asin(1 / x) - 1 / ratio - 2 * (x - u)
        else if (x > ratio && x < Math.sqrt(ratio * ratio + 1)) {
          c = 2 * Math.asin((ratio - u * v) / (x * x)) + 2 * u + (2 * v) / ratio - ratio - (1 + x * x) / ratio
        }
        weight = 1 / ((area / Math.PI) * c)
      } else if (mode === 'ripley') {
        const inside = d <= Math.hypot(hor, ver)
        const w1 = 1 - (Math.acos(Math.min(ver, d) / d) + Math.acos(Math.min(hor, d) / d)) / Math.PI
        const w2 = 0.75 - (0.5 * (Math.acos(ver / d) + Math.acos(hor / d))) / Math.PI
        weight = 1 / (inside ? w1 : w2)
      }
      for (let r = 0; r < radii.length; r++) if (d < (radii[r] ?? 0)) out[r] = (out[r] ?? 0) + weight
    }
  }
  const scale = mode === 'none' || mode === 'ripley' ? (2 * area) / (n * (n - 1)) : (2 * area * area) / (n * (n - 1))
  for (let r = 0; r < radii.length; r++) out[r] = (out[r] ?? 0) * scale
  return out
}

/**
 * Besag's L function L(r) = sqrt(K(r) / pi), which is r under complete spatial randomness.
 *
 * @param points - The points `[n, 2]`.
 * @param radii - The distances.
 * @param options - Window and edge correction, as for {@link ripleysK}.
 * @returns L at each radius.
 * @concept ripleys-k
 * @example
 * ```ts
 * import { ripleysL } from 'whitebox-engine/stochastic'
 * Array.from(ripleysL([[0.1, 0.1], [0.2, 0.1], [0.9, 0.9]], [0.05])) // => [0]
 * ```
 */
export function ripleysL(
  points: TensorLike,
  radii: ArrayLike<number>,
  options: { readonly window?: Window; readonly correction?: RipleyCorrection } = {},
): Float64Array {
  return Float64Array.from(ripleysK(points, radii, options), (k) => Math.sqrt(k / Math.PI))
}
