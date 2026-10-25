/**
 * Temporal point processes, random sequences of event times: homogeneous and inhomogeneous Poisson processes (by
 * thinning), renewal processes and the renewal function, the self-exciting Hawkes process with its intensity,
 * log-likelihood and fitting by EM, Cox and log-Gaussian Cox processes with random intensities, and marked Poisson
 * processes.
 */

import { grad, hessian } from '../autodiff/index.ts'
import { WhiteboxError } from '../base/index.ts'
import type { Distribution } from '../dists/index.ts'
import { brentq, fsolve } from '../numerics/roots/index.ts'
import { exponential, foldIn, type Key, poisson, uniform } from '../random/index.ts'
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
  /** EM iterations, then Newton iterations. */
  readonly iterations: number
  readonly converged: boolean
}

/** The state of {@link hawkesEmAlgorithm}: the parameters and the log-likelihood they reach. */
export interface HawkesEmState {
  readonly mu: number
  readonly alpha: number
  readonly beta: number
  readonly logLikelihood: number
  /** The largest relative change of a parameter in the last step. */
  readonly change: number
}

/**
 * The EM algorithm for an exponential Hawkes process (Veen and Schoenberg 2008) as a traced {@link Algorithm}: the
 * E-step assigns each event to the background or to an earlier event in proportion to their contributions to the
 * intensity, the M-step re-estimates mu = (expected immigrants) / T, the branching ratio alpha / beta = (expected
 * offspring) / sum_j (1 - e^{-beta (T - t_j)}), and beta by maximising the expected complete-data log-likelihood
 * (a one-dimensional root by Brent's method), so the likelihood never decreases.
 *
 * @param events - The event times in [0, T].
 * @param horizon - T.
 * @param init - The starting parameters.
 * @param options - The tolerance on the largest relative change of a parameter (default 1e-10).
 * @returns The algorithm.
 * @concept hawkes-process
 * @reference A. Veen and F. P. Schoenberg, Estimation of space-time branching process models in seismology using an
 *   EM-type algorithm, JASA 103 (2008).
 * @example
 * ```ts
 * import { hawkesEmAlgorithm } from 'whitebox-engine/stochastic'
 * hawkesEmAlgorithm([1, 2, 2.5], 3, { mu: 1, alpha: 0.5, beta: 1 }).name // => 'hawkes-em'
 * ```
 */
export function hawkesEmAlgorithm(
  events: TensorLike,
  horizon: number,
  init: { readonly mu: number; readonly alpha: number; readonly beta: number },
  options: { readonly tol?: number } = {},
): Algorithm<HawkesEmState> {
  const t = primalNumbers(events)
  const n = t.length
  hawkesNumbers(init)
  const tol = options.tol ?? 1e-10
  // The log-likelihood in O(n) by the recursion A_i = e^{-beta (t_i - t_{i-1})} (A_{i-1} + 1).
  const ll = (mu: number, alpha: number, beta: number): number => {
    let a = 0
    let total = 0
    for (let i = 0; i < n; i++) {
      if (i > 0) a = Math.exp(-beta * ((t[i] ?? 0) - (t[i - 1] ?? 0))) * (a + 1)
      total += Math.log(mu + alpha * a) - (alpha / beta) * (1 - Math.exp(-beta * (horizon - (t[i] ?? 0))))
    }
    return total - mu * horizon
  }
  return {
    name: 'hawkes-em',
    config: { events: n, horizon },
    schema: channels({
      metrics: { logLikelihood: { description: 'log-likelihood', direction: 'higher' } },
    }),
    init: () => ({ ...init, logLikelihood: ll(init.mu, init.alpha, init.beta), change: Number.POSITIVE_INFINITY }),
    step: (s) => {
      // E-step: the expected numbers of immigrants and offspring, and the offspring's expected total delay
      // (sums over earlier events by the recursions A_i = e^{-beta d} (A_{i-1} + 1), B_i = e^{-beta d} (B_{i-1} + d
      // (A_{i-1} + 1)) for sum_j e^{-beta lag} and sum_j lag e^{-beta lag}, with d = t_i - t_{i-1})
      let immigrants = 0
      let offspring = 0
      let delays = 0
      let a = 0
      let b = 0
      for (let i = 0; i < n; i++) {
        if (i > 0) {
          const d = (t[i] ?? 0) - (t[i - 1] ?? 0)
          const decay = Math.exp(-s.beta * d)
          b = decay * (b + d * (a + 1))
          a = decay * (a + 1)
        }
        const lambda = s.mu + s.alpha * a
        immigrants += s.mu / lambda
        offspring += (s.alpha * a) / lambda
        delays += (s.alpha * b) / lambda
      }
      // M-step: with alpha = r beta, Q(r, beta) = offspring (log r + log beta) - beta delays - r sum_j (1 - e^{-beta
      // (T - t_j)}); profile r out and solve dQ/dbeta = 0 for beta by Brent's method.
      const exposure = (b: number): number => t.reduce((acc, tj) => acc + 1 - Math.exp(-b * (horizon - tj)), 0)
      const slope = (b: number): number => {
        const r = offspring / Math.max(exposure(b), 1e-300)
        const tail = t.reduce((acc, tj) => acc + (horizon - tj) * Math.exp(-b * (horizon - tj)), 0)
        return offspring / b - delays - r * tail
      }
      let beta = s.beta
      if (offspring > 1e-12 && delays > 0) {
        let hi = Math.max(1, (2 * offspring) / delays)
        for (let k = 0; k < 60 && slope(hi) > 0; k++) hi *= 2
        let lo = Math.min(1e-8, hi / 2)
        for (let k = 0; k < 60 && slope(lo) < 0; k++) lo /= 2
        if (slope(lo) > 0 && slope(hi) < 0) beta = brentq(slope, lo, hi, { xtol: 1e-14 }).root
      }
      const mu = Math.max(immigrants / horizon, 1e-300)
      const alpha = Math.max((beta * offspring) / Math.max(exposure(beta), 1e-300), 1e-300)
      const value = ll(mu, alpha, beta)
      const change = Math.max(
        Math.abs(mu - s.mu) / s.mu,
        Math.abs(alpha - s.alpha) / s.alpha,
        Math.abs(beta - s.beta) / s.beta,
      )
      return { mu, alpha, beta, logLikelihood: value, change }
    },
    done: (s) => (s.change <= tol ? 'converged' : undefined),
    observe: (s) => ({ metrics: { logLikelihood: s.logLikelihood } }),
  }
}

/**
 * Maximum-likelihood estimation of an exponential Hawkes process: EM ({@link hawkesEmAlgorithm}, O(n) per iteration)
 * to the maximum, then, for up to `polishBelow` events, Newton's method on the score equations in log-parameters
 * (numerics `fsolve`, with the Hessian from autodiff of the dense O(n^2) likelihood) to polish it.
 *
 * @param events - The event times in [0, T].
 * @param horizon - T.
 * @param options - The starting point (default: mu = n / (2T), alpha = beta / 2, beta = n / T), the EM and Newton
 *   iteration caps (default 5000 and 50), the event count below which Newton polishes (default 500) and tracing of the
 *   EM run.
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
  options: {
    readonly init?: { mu: number; alpha: number; beta: number }
    readonly maxIter?: number
    readonly newtonIter?: number
    readonly polishBelow?: number
  } & TraceOptions = {},
): HawkesFit {
  const ev = asFloat(asTensor(events))
  const n = ev.shape[0] ?? 0
  if (n < 3) throw new WhiteboxError('value', 'fitHawkes: needs at least three events')
  const rate = n / horizon
  const init = options.init ?? { mu: rate / 2, alpha: rate / 2, beta: rate }
  const em = runAlgorithm(hawkesEmAlgorithm(ev, horizon, init), {
    ...traceOf(options),
    maxSteps: options.maxIter ?? 5000,
  })
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
  const emOnly = { mu: em.state.mu, alpha: em.state.alpha, beta: em.state.beta }
  if (n > (options.polishBelow ?? 500)) {
    return {
      ...emOnly,
      logLikelihood: em.state.logLikelihood,
      iterations: em.steps,
      converged: em.status === 'converged',
    }
  }
  const res = fsolve(
    (logp: Value) => grad(nll)(logp),
    [Math.log(em.state.mu), Math.log(em.state.alpha), Math.log(em.state.beta)],
    {
      jacobian: (x) => hessian(nll)(x),
      maxIterations: options.newtonIter ?? 50,
      ftol: 1e-10,
    },
  )
  const polished = primalNumbers(res.x).map(Math.exp)
  const [pm = 0, pa = 0, pb = 0] = polished
  const llNewton = primalNumbers(hawkesLogLikelihood({ mu: pm, alpha: pa, beta: pb }, ev, horizon))[0] ?? Number.NaN
  // keep the Newton point only when it does not lose likelihood (it can wander to a saddle from a poor start)
  const useNewton = Number.isFinite(llNewton) && llNewton >= em.state.logLikelihood - 1e-9
  const [mu, alpha, beta] = useNewton ? [pm, pa, pb] : [emOnly.mu, emOnly.alpha, emOnly.beta]
  return {
    mu,
    alpha,
    beta,
    logLikelihood: useNewton ? llNewton : em.state.logLikelihood,
    iterations: em.steps + res.iterations,
    converged: useNewton ? res.converged : em.status === 'converged',
  }
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
