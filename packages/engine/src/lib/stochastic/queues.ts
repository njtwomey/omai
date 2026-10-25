/**
 * Queueing theory: steady-state measures of the M/M/1, M/M/c and M/G/1 queues (the last by the Pollaczek-Khinchine
 * formula), the Erlang B and C formulas for blocking and waiting, and simulation of a first-in first-out G/G/c queue as
 * a traced algorithm, with Little's law L = lambda W checked on the result.
 */

import { heap, WhiteboxError } from '../base/index.ts'
import { type Distribution, exponential } from '../dists/index.ts'
import { foldIn, type Key } from '../random/index.ts'
import { item } from '../tensor/index.ts'
import { type Algorithm, channels, runAlgorithm } from '../trace/index.ts'
import { positiveInteger, primalNumbers, traceOf } from './_shared.ts'
import { type History, historyArray, type TraceOptions } from './paths.ts'

/** Steady-state measures of a queue (Kendall's notation; times in the units of the rates). */
export interface QueueMetrics {
  /** The utilisation rho = lambda / (c mu) per server. */
  readonly utilisation: number
  /** The mean number in the system. */
  readonly L: number
  /** The mean number waiting. */
  readonly Lq: number
  /** The mean time in the system. */
  readonly W: number
  /** The mean waiting time before service. */
  readonly Wq: number
  /** The probability that the system is empty. */
  readonly p0: number
  /** The probability that an arrival has to wait. */
  readonly waitProbability: number
}

function stable(lambda: number, capacity: number, where: string): void {
  if (!(lambda > 0 && capacity > 0)) throw new WhiteboxError('value', `${where}: rates must be positive`)
  if (!(lambda < capacity)) throw new WhiteboxError('value', `${where}: unstable (lambda >= service capacity)`)
}

/**
 * The M/M/1 queue: Poisson arrivals at rate lambda, exponential service at rate mu, one server. With rho = lambda / mu:
 * P(N = n) = (1 - rho) rho^n, L = rho / (1 - rho), W = 1 / (mu - lambda), and the sojourn time is exponential with rate
 * mu - lambda.
 *
 * @param lambda - The arrival rate.
 * @param mu - The service rate (greater than lambda).
 * @returns The steady-state measures, the state probabilities and the sojourn-time CDF.
 * @throws WhiteboxError with code `value` when the queue is unstable.
 * @concept queueing-theory
 * @example
 * ```ts
 * import { mm1 } from 'whitebox-engine/stochastic'
 * const q = mm1(2, 4)
 * [q.utilisation, q.L, q.W, q.Lq] // => [0.5, 1, 0.5, 0.5]
 * ```
 */
export function mm1(
  lambda: number,
  mu: number,
): QueueMetrics & { readonly stateProbability: (n: number) => number; readonly sojournCdf: (t: number) => number } {
  stable(lambda, mu, 'mm1')
  const rho = lambda / mu
  return {
    utilisation: rho,
    L: rho / (1 - rho),
    Lq: (rho * rho) / (1 - rho),
    W: 1 / (mu - lambda),
    Wq: rho / (mu - lambda),
    p0: 1 - rho,
    waitProbability: rho,
    stateProbability: (n) => (Number.isInteger(n) && n >= 0 ? (1 - rho) * rho ** n : 0),
    sojournCdf: (t) => (t <= 0 ? 0 : -Math.expm1(-(mu - lambda) * t)),
  }
}

/**
 * The Erlang B formula: the blocking probability of an M/M/c/c loss system with offered load a = lambda / mu, by the
 * stable recursion B(k) = a B(k - 1) / (k + a B(k - 1)), B(0) = 1.
 *
 * @param servers - The number of servers c.
 * @param load - The offered load a (Erlangs).
 * @returns B(c, a).
 * @concept queueing-theory
 * @example
 * ```ts
 * import { erlangB } from 'whitebox-engine/stochastic'
 * erlangB(1, 1) // => 0.5
 * ```
 */
export function erlangB(servers: number, load: number): number {
  positiveInteger(servers, 'servers', 'erlangB')
  if (!(load >= 0)) throw new WhiteboxError('value', 'erlangB: the load must be >= 0')
  let b = 1
  for (let k = 1; k <= servers; k++) b = (load * b) / (k + load * b)
  return b
}

/**
 * The Erlang C formula: the probability that an arrival to an M/M/c queue must wait,
 * C = c B / (c - a (1 - B)) with B the Erlang B value (a < c).
 *
 * @param servers - The number of servers c.
 * @param load - The offered load a = lambda / mu (less than c).
 * @returns C(c, a).
 * @throws WhiteboxError with code `value` when a >= c (unstable).
 * @concept queueing-theory
 * @example
 * ```ts
 * import { erlangC } from 'whitebox-engine/stochastic'
 * Math.round(erlangC(1, 0.5) * 1e12) / 1e12 // => 0.5
 * ```
 */
export function erlangC(servers: number, load: number): number {
  const b = erlangB(servers, load)
  if (!(load < servers)) throw new WhiteboxError('value', 'erlangC: unstable (load >= servers)')
  return (servers * b) / (servers - load * (1 - b))
}

/**
 * The M/M/c queue: Poisson arrivals, exponential service at rate mu, c servers. With a = lambda / mu and Erlang C value
 * C: Lq = C rho / (1 - rho), Wq = C / (c mu - lambda), W = Wq + 1 / mu, L = lambda W, and P(Wq > t) = C e^{-(c mu -
 * lambda) t}.
 *
 * @param lambda - The arrival rate.
 * @param mu - The service rate per server.
 * @param servers - The number of servers.
 * @returns The steady-state measures, state probabilities and the waiting-time CDF.
 * @throws WhiteboxError with code `value` when the queue is unstable.
 * @concept queueing-theory
 * @example
 * ```ts
 * import { mm1, mmc } from 'whitebox-engine/stochastic'
 * Math.abs(mmc(2, 4, 1).L - mm1(2, 4).L) < 1e-12 // => true
 * ```
 */
export function mmc(
  lambda: number,
  mu: number,
  servers: number,
): QueueMetrics & { readonly stateProbability: (n: number) => number; readonly waitCdf: (t: number) => number } {
  positiveInteger(servers, 'servers', 'mmc')
  stable(lambda, servers * mu, 'mmc')
  const a = lambda / mu
  const c = servers
  const rho = a / c
  const wait = erlangC(c, a)
  const Wq = wait / (c * mu - lambda)
  const W = Wq + 1 / mu
  // p0 from the Erlang B value: p_c = p0 a^c / c! and B = p_c / sum_{k<=c} p0 a^k / k! (finite part)
  let logTerm = 0
  let finite = 1
  for (let k = 1; k < c; k++) {
    logTerm += Math.log(a / k)
    finite += Math.exp(logTerm)
  }
  const top = Math.exp(logTerm + Math.log(a / c))
  const p0 = 1 / (finite + top / (1 - rho))
  const stateProbability = (n: number): number => {
    if (!(Number.isInteger(n) && n >= 0)) return 0
    let log = 0
    for (let k = 1; k <= Math.min(n, c); k++) log += Math.log(a / k)
    if (n > c) log += (n - c) * Math.log(rho)
    return p0 * Math.exp(log)
  }
  return {
    utilisation: rho,
    L: lambda * W,
    Lq: lambda * Wq,
    W,
    Wq,
    p0,
    waitProbability: wait,
    stateProbability,
    waitCdf: (t) => (t < 0 ? 0 : 1 - wait * Math.exp(-(c * mu - lambda) * t)),
  }
}

/** The first two moments of a service time. */
export interface ServiceMoments {
  readonly mean: number
  readonly variance: number
}

/**
 * The M/G/1 queue by the Pollaczek-Khinchine formula: with rho = lambda E[S] < 1, Lq = lambda^2 E[S^2] / (2 (1 - rho)),
 * Wq = Lq / lambda, W = Wq + E[S], L = lambda W. Exponential service recovers M/M/1; deterministic service (M/D/1)
 * halves the queue.
 *
 * @param lambda - The arrival rate.
 * @param service - The service-time law, or its mean and variance.
 * @returns The steady-state measures.
 * @throws WhiteboxError with code `value` when rho >= 1.
 * @concept queueing-theory
 * @example
 * ```ts
 * import { mg1 } from 'whitebox-engine/stochastic'
 * mg1(0.5, { mean: 1, variance: 0 }).Lq // => 0.25
 * ```
 */
export function mg1(lambda: number, service: Distribution | ServiceMoments): QueueMetrics {
  const moments =
    'logProb' in service
      ? { mean: item(service.mean()), variance: item(service.variance()) }
      : (service as ServiceMoments)
  if (!(moments.mean > 0 && moments.variance >= 0)) throw new WhiteboxError('value', 'mg1: bad service moments')
  stable(lambda, 1 / moments.mean, 'mg1')
  const rho = lambda * moments.mean
  const second = moments.variance + moments.mean ** 2
  const Lq = (lambda * lambda * second) / (2 * (1 - rho))
  const Wq = Lq / lambda
  const W = Wq + moments.mean
  return { utilisation: rho, L: lambda * W, Lq, W, Wq, p0: 1 - rho, waitProbability: rho }
}

/** The specification of a simulated FIFO queue G/G/c. */
export interface QueueSpec {
  /** The law of the times between arrivals (a rate means Poisson arrivals). */
  readonly interarrival: Distribution | number
  /** The law of the service times (a rate means exponential service). */
  readonly service: Distribution | number
  /** The number of servers (default 1). */
  readonly servers?: number
}

/** One customer of a simulated queue. */
export interface Customer {
  readonly arrival: number
  readonly start: number
  readonly departure: number
}

/** The state of {@link queueAlgorithm}: the customers so far and when each server becomes free. */
export interface QueueState {
  readonly clock: number
  /** The customers so far, newest first. */
  readonly customers: History<Customer> | undefined
  /** When each server next becomes free. */
  readonly free: readonly number[]
}

function lawOf(x: Distribution | number, where: string): Distribution {
  if (typeof x !== 'number') return x
  if (!(x > 0)) throw new WhiteboxError('value', `${where}: rates must be positive`)
  return exponential(x)
}

/**
 * A first-in first-out G/G/c queue as a traced {@link Algorithm}, one step per customer: the customer arrives an
 * interarrival time after the previous one and is served by the server that frees up first (Kiefer-Wolfowitz
 * recursion; Lindley's for one server).
 *
 * @param spec - Interarrival and service laws (or rates) and the number of servers.
 * @returns The algorithm.
 * @concept queueing-theory
 * @example
 * ```ts
 * import { queueAlgorithm } from 'whitebox-engine/stochastic'
 * queueAlgorithm({ interarrival: 1, service: 2, servers: 2 }).name // => 'queue'
 * ```
 */
export function queueAlgorithm(spec: QueueSpec): Algorithm<QueueState> {
  const arrivalLaw = lawOf(spec.interarrival, 'queue')
  const serviceLaw = lawOf(spec.service, 'queue')
  const c = positiveInteger(spec.servers ?? 1, 'servers', 'queue')
  return {
    name: 'queue',
    config: { servers: c },
    schema: channels({
      metrics: {
        wait: { description: 'the waiting time of the latest customer', direction: 'lower' },
        sojourn: { description: 'the time in system of the latest customer', direction: 'lower' },
      },
    }),
    init: () => ({ clock: 0, customers: undefined, free: new Array<number>(c).fill(0) }),
    step: (s, ctx) => {
      const gap = primalNumbers(arrivalLaw.sample(foldIn(ctx.key, 0), [1]))[0] ?? 0
      const work = primalNumbers(serviceLaw.sample(foldIn(ctx.key, 1), [1]))[0] ?? 0
      if (!(gap >= 0 && work >= 0)) throw new WhiteboxError('value', 'queue: sampled times must be >= 0')
      const arrival = s.clock + gap
      const servers = heap((a: number, b: number) => a - b, s.free)
      const earliest = servers.pop() ?? 0
      const start = Math.max(arrival, earliest)
      const departure = start + work
      servers.push(departure)
      return {
        clock: arrival,
        customers: { item: { arrival, start, departure }, previous: s.customers },
        free: servers.toArray(),
      }
    },
    observe: (s) => {
      const last = s.customers?.item
      if (last === undefined) return { metrics: {} }
      return { metrics: { wait: last.start - last.arrival, sojourn: last.departure - last.arrival } }
    },
  }
}

/** A simulated queue: every customer's times and the empirical measures. */
export interface QueueSimulation {
  readonly arrivals: Float64Array
  readonly starts: Float64Array
  readonly departures: Float64Array
  /** Waiting times before service. */
  readonly waits: Float64Array
  /** Times in the system. */
  readonly sojourns: Float64Array
  /** The time-average number in the system over [0, last departure]. */
  readonly L: number
  /** The time-average number waiting. */
  readonly Lq: number
  /** The mean sojourn time. */
  readonly W: number
  /** The mean waiting time. */
  readonly Wq: number
  /** The arrival rate: customers per unit time over the observation window. */
  readonly arrivalRate: number
  /** The length of the observation window. */
  readonly horizon: number
}

function timeAverage(ups: Float64Array, downs: Float64Array, horizon: number): number {
  const events: [number, number][] = []
  for (const t of ups) events.push([t, 1])
  for (const t of downs) events.push([t, -1])
  events.sort((a, b) => a[0] - b[0] || a[1] - b[1])
  let area = 0
  let count = 0
  let last = 0
  for (const [t, d] of events) {
    area += count * (t - last)
    count += d
    last = t
  }
  return horizon > 0 ? area / horizon : 0
}

/**
 * Simulate a FIFO G/G/c queue ({@link queueAlgorithm}) for a number of customers and measure it.
 *
 * @param key - The key.
 * @param spec - Interarrival and service laws (or rates) and servers.
 * @param customers - The number of customers.
 * @param options - Tracing.
 * @returns The customers' times and the empirical L, Lq, W, Wq and arrival rate.
 * @concept queueing-theory
 * @example
 * ```ts
 * import { key } from 'whitebox-engine/random'
 * import { simulateQueue } from 'whitebox-engine/stochastic'
 * simulateQueue(key(0), { interarrival: 1, service: 2 }, 100).waits.length // => 100
 * ```
 */
export function simulateQueue(
  key: Key,
  spec: QueueSpec,
  customers: number,
  options: TraceOptions = {},
): QueueSimulation {
  positiveInteger(customers, 'customers', 'simulateQueue')
  const r = runAlgorithm(queueAlgorithm(spec), { ...traceOf(options), key, maxSteps: customers })
  const all = historyArray(r.state.customers)
  const arrivals = Float64Array.from(all, (x) => x.arrival)
  const starts = Float64Array.from(all, (x) => x.start)
  const departures = Float64Array.from(all, (x) => x.departure)
  const waits = Float64Array.from(starts, (s, i) => s - (arrivals[i] ?? 0))
  const sojourns = Float64Array.from(departures, (d, i) => d - (arrivals[i] ?? 0))
  let horizon = 0
  for (const d of departures) horizon = Math.max(horizon, d)
  const mean = (xs: Float64Array): number => xs.reduce((a, b) => a + b, 0) / Math.max(1, xs.length)
  return {
    arrivals,
    starts,
    departures,
    waits,
    sojourns,
    L: timeAverage(arrivals, departures, horizon),
    Lq: timeAverage(arrivals, starts, horizon),
    W: mean(sojourns),
    Wq: mean(waits),
    arrivalRate: customers / horizon,
    horizon,
  }
}

/**
 * Little's law L = lambda W checked on a simulated queue: over a window in which the system starts and ends empty, the
 * time-average number in the system equals the arrival rate times the mean time in the system (exactly, as a sample-path
 * identity), and likewise for the queue alone (Lq = lambda Wq).
 *
 * @param sim - A queue simulation.
 * @returns Both sides for the system and for the queue.
 * @concept littles-law
 * @example
 * ```ts
 * import { key } from 'whitebox-engine/random'
 * import { littlesLaw, simulateQueue } from 'whitebox-engine/stochastic'
 * const l = littlesLaw(simulateQueue(key(0), { interarrival: 1, service: 1.5 }, 200))
 * Math.abs(l.L - l.lambdaW) < 1e-9 // => true
 * ```
 */
export function littlesLaw(sim: QueueSimulation): {
  readonly L: number
  readonly lambdaW: number
  readonly Lq: number
  readonly lambdaWq: number
} {
  return { L: sim.L, lambdaW: sim.arrivalRate * sim.W, Lq: sim.Lq, lambdaWq: sim.arrivalRate * sim.Wq }
}
