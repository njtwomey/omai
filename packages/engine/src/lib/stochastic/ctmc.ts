/**
 * Continuous-time Markov chains, which jump between states after exponential holding times set by a generator matrix Q:
 * transition matrices exp(tQ), distributions at time t, the stationary law, the embedded jump chain, uniformisation and
 * expected hitting times, and simulation by Gillespie's algorithm (a traced algorithm) or exactly on a grid.
 */

import { MdmlError } from '../base/index.ts'
import { expm, solve } from '../numerics/linalg/index.ts'
import { categorical, exponential, foldIn, type Key } from '../random/index.ts'
import {
  add,
  asFloat,
  asTensor,
  div,
  eye,
  matmul,
  mul,
  neg,
  ones,
  sliceAxis,
  type Tensor,
  type TensorLike,
  type Value,
} from '../tensor/index.ts'
import { type Algorithm, channels, runAlgorithm } from '../trace/index.ts'
import {
  assemble,
  block,
  classify,
  generatorRows,
  maybeGeneratorRows,
  stationarySolve,
  targetMask,
  tensorOf,
} from './_chains.ts'
import { positiveInteger, primalNumbers, traceOf, vector } from './_shared.ts'
import { hittingProbabilities, nextStates } from './markov-chains.ts'
import { type History, historyArray, type Paths, simulatePaths, type TimesLike, type TraceOptions } from './paths.ts'

function jumpRows(q: TensorLike, where: string): { n: number; rows: number[][]; rates: number[] } {
  const { n, rows } = generatorRows(q, where)
  const rates = rows.map((r, i) => -(r[i] ?? 0))
  const jump = rows.map((r, i) => {
    const rate = rates[i] ?? 0
    return r.map((v, j) => (rate <= 0 ? (i === j ? 1 : 0) : i === j ? 0 : v / rate))
  })
  return { n, rows: jump, rates }
}

/**
 * The transition matrix P(t) = exp(t Q) of a continuous-time chain with generator Q, solving the Kolmogorov backward
 * and forward equations P' = Q P = P Q. Differentiable in Q and t.
 *
 * @param q - The generator `[n, n]` (off-diagonal rates >= 0, rows summing to zero).
 * @param t - The time.
 * @returns P(t).
 * @concept continuous-time-markov-chains
 * @example
 * ```ts
 * import { ctmcTransition } from 'mdml-engine/stochastic'
 * import { toArray } from 'mdml-engine/tensor'
 * toArray(ctmcTransition([[-1, 1], [1, -1]], 0)) // => [[1, 0], [0, 1]]
 * ```
 */
export function ctmcTransition(q: TensorLike, t: TensorLike): Value {
  maybeGeneratorRows(q, 'ctmcTransition')
  return expm(mul(asFloat(asTensor(q)), t))
}

/**
 * The distribution of a continuous-time chain at time t, p0 exp(t Q). Differentiable.
 *
 * @param q - The generator.
 * @param p0 - The initial distribution.
 * @param t - The time.
 * @returns The distribution at t.
 * @concept continuous-time-markov-chains
 * @example
 * ```ts
 * import { ctmcDistribution } from 'mdml-engine/stochastic'
 * import { toArray } from 'mdml-engine/tensor'
 * toArray(ctmcDistribution([[-1, 1], [1, -1]], [1, 0], 50)).map((v) => Math.round(v * 1e9) / 1e9) // => [0.5, 0.5]
 * ```
 */
export function ctmcDistribution(q: TensorLike, p0: TensorLike, t: TensorLike): Value {
  return matmul(asFloat(asTensor(p0)), ctmcTransition(q, t))
}

/**
 * The stationary distribution pi Q = 0 of a continuous-time chain with a single closed class. Differentiable in Q.
 *
 * @param q - The generator.
 * @returns pi.
 * @throws MdmlError with code `value` when there are several closed classes.
 * @concept stationary-distribution
 * @example
 * ```ts
 * import { ctmcStationary } from 'mdml-engine/stochastic'
 * import { toArray } from 'mdml-engine/tensor'
 * toArray(ctmcStationary([[-2, 2], [1, -1]])).map((v) => Math.round(v * 1e12) / 1e12) // => [0.333333333333, 0.666666666667]
 * ```
 */
export function ctmcStationary(q: TensorLike): Value {
  const { n, rows } = maybeGeneratorRows(q, 'ctmcStationary')
  if (rows !== undefined && classify(n, jumpRows(q, 'ctmcStationary').rows).closed.filter((x) => x).length !== 1) {
    throw new MdmlError('value', 'ctmcStationary: several closed classes, no unique stationary law')
  }
  return stationarySolve(asFloat(asTensor(q)), n)
}

/**
 * The embedded jump chain of a continuous-time chain, P_ij = q_ij / q_i for i != j (absorbing states, q_i = 0, stay
 * put), with the holding rates q_i = -q_ii.
 *
 * @param q - The generator.
 * @returns The jump chain's transition matrix and the holding rates.
 * @concept continuous-time-markov-chains
 * @example
 * ```ts
 * import { embeddedChain } from 'mdml-engine/stochastic'
 * import { toArray } from 'mdml-engine/tensor'
 * const e = embeddedChain([[-4, 1, 3], [0, 0, 0], [1, 1, -2]])
 * toArray(e.transition) // => [[0, 0.25, 0.75], [0, 1, 0], [0.5, 0.5, 0]]
 * Array.from(e.rates) // => [4, 0, 2]
 * ```
 */
export function embeddedChain(q: TensorLike): { readonly transition: Tensor<'float64'>; readonly rates: Float64Array } {
  const { rows, rates } = jumpRows(q, 'embeddedChain')
  return { transition: tensorOf(rows), rates: Float64Array.from(rates, (r) => r + 0) }
}

/**
 * Uniformisation: P = I + Q / lambda for a rate lambda >= max_i q_i, so that the continuous-time chain is the discrete
 * chain P run at the jumps of a Poisson(lambda) process, exp(t Q) = sum_k e^{-lambda t} (lambda t)^k / k! P^k.
 * Differentiable in Q.
 *
 * @param q - The generator.
 * @param rate - The uniformisation rate (default max_i q_i).
 * @returns The uniformised chain and its rate.
 * @throws MdmlError with code `value` when the rate is below the largest holding rate.
 * @concept continuous-time-markov-chains
 * @example
 * ```ts
 * import { uniformise } from 'mdml-engine/stochastic'
 * import { toArray } from 'mdml-engine/tensor'
 * toArray(uniformise([[-2, 2], [1, -1]]).transition) // => [[0, 1], [0.5, 0.5]]
 * ```
 */
export function uniformise(q: TensorLike, rate?: number): { readonly transition: Value; readonly rate: number } {
  const { rows } = generatorRows(q, 'uniformise')
  const biggest = rows.reduce((m, r, i) => Math.max(m, -(r[i] ?? 0)), 0)
  const lambda = rate ?? biggest
  if (!(lambda >= biggest && lambda > 0)) {
    throw new MdmlError('value', 'uniformise: the rate must be positive and at least the largest holding rate')
  }
  return { transition: add(eye(rows.length), div(asFloat(asTensor(q)), lambda)), rate: lambda }
}

/**
 * The expected time for a continuous-time chain to reach a set of states: k = 0 on the target and -Q k = 1 elsewhere
 * (Infinity from states that miss it with positive probability). Differentiable in Q.
 *
 * @param q - The generator.
 * @param target - The target states.
 * @returns `[n]`.
 * @concept hitting-times
 * @example
 * ```ts
 * import { ctmcHittingTimes } from 'mdml-engine/stochastic'
 * import { toArray } from 'mdml-engine/tensor'
 * toArray(ctmcHittingTimes([[-2, 2], [0, 0]], [1])) // => [0.5, 0]
 * ```
 */
export function ctmcHittingTimes(q: TensorLike, target: readonly number[]): Value {
  const { n, rows } = jumpRows(q, 'ctmcHittingTimes')
  const mask = targetMask(n, target, 'ctmcHittingTimes')
  const h = primalNumbers(hittingProbabilities(tensorOf(rows), target))
  const sure = Array.from({ length: n }, (_, i) => i).filter((i) => !mask[i] && Math.abs((h[i] ?? 0) - 1) < 1e-9)
  const parts = new Array<Value | number>(n).fill(Number.POSITIVE_INFINITY)
  for (const s of target) parts[s] = 0
  if (sure.length > 0) {
    const k = solve(neg(block(asFloat(asTensor(q)), sure, sure)), ones([sure.length]))
    for (const [i, s] of sure.entries()) parts[s] = sliceAxis(k, 0, i, i + 1)
  }
  return assemble(parts)
}

/** A continuous-time trajectory: the jump times and the states entered at them. */
export interface Trajectory {
  /** The jump times, starting with the start time 0. */
  readonly times: Float64Array
  /** The state held from each time until the next. */
  readonly states: Int32Array
}

/** The state of {@link ctmcAlgorithm}: the time, state and trajectory so far. */
export interface GillespieState {
  readonly t: number
  readonly x: number
  /** The jumps so far (time and state entered), newest first. */
  readonly jumps: History<{ readonly t: number; readonly x: number }>
  /** Whether the horizon or an absorbing state has been reached. */
  readonly finished: boolean
}

/**
 * Gillespie's algorithm for a continuous-time chain as a traced {@link Algorithm}: each step waits an exponential
 * holding time with rate q_x and jumps by the embedded chain, until the horizon or an absorbing state.
 *
 * @param q - The generator.
 * @param x0 - The initial state.
 * @param horizon - The end time.
 * @returns The algorithm; its state holds the trajectory so far.
 * @concept gillespie-algorithm
 * @example
 * ```ts
 * import { ctmcAlgorithm } from 'mdml-engine/stochastic'
 * ctmcAlgorithm([[-1, 1], [1, -1]], 0, 5).name // => 'gillespie'
 * ```
 */
export function ctmcAlgorithm(q: TensorLike, x0: number, horizon: number): Algorithm<GillespieState> {
  const { rows, rates } = jumpRows(q, 'ctmcAlgorithm')
  if (!(Number.isInteger(x0) && x0 >= 0 && x0 < rows.length)) throw new MdmlError('value', 'ctmc: bad x0')
  if (!(horizon > 0)) throw new MdmlError('value', 'ctmc: the horizon must be positive')
  return {
    name: 'gillespie',
    config: { states: rows.length, horizon },
    schema: channels({
      metrics: { t: { description: 'the time of the last jump' }, state: { description: 'the current state' } },
    }),
    init: () => ({ t: 0, x: x0, jumps: { item: { t: 0, x: x0 }, previous: undefined }, finished: false }),
    step: (s, ctx) => {
      const rate = rates[s.x] ?? 0
      if (rate <= 0) return { ...s, finished: true }
      const t = s.t + (exponential(foldIn(ctx.key, 0), [1], rate).data[0] ?? 0)
      if (t >= horizon) return { ...s, finished: true }
      const next = categorical(foldIn(ctx.key, 1), rows[s.x] ?? [], [1]).data[0] ?? 0
      return { t, x: next, jumps: { item: { t, x: next }, previous: s.jumps }, finished: false }
    },
    done: (s) => (s.finished ? 'completed' : undefined),
    observe: (s) => ({ metrics: { t: s.t, state: s.x } }),
  }
}

/**
 * Simulate a continuous-time chain by Gillespie's algorithm ({@link ctmcAlgorithm}).
 *
 * @param key - The key.
 * @param q - The generator.
 * @param x0 - The initial state.
 * @param horizon - The end time.
 * @param options - The jump cap (default 100000) and tracing.
 * @returns The trajectory.
 * @throws MdmlError with code `convergence` when the jump cap is reached before the horizon.
 * @concept gillespie-algorithm
 * @example
 * ```ts
 * import { key } from 'mdml-engine/random'
 * import { simulateCtmc } from 'mdml-engine/stochastic'
 * simulateCtmc(key(0), [[-1, 1], [0, 0]], 0, 100).states.at(-1) // => 1
 * ```
 */
export function simulateCtmc(
  key: Key,
  q: TensorLike,
  x0: number,
  horizon: number,
  options: { readonly maxJumps?: number } & TraceOptions = {},
): Trajectory {
  const cap = options.maxJumps ?? 100000
  const r = runAlgorithm(ctmcAlgorithm(q, x0, horizon), { ...traceOf(options), key, maxSteps: cap + 1 })
  if (!r.state.finished) throw new MdmlError('convergence', `simulateCtmc: more than ${cap} jumps before the horizon`)
  const jumps = historyArray(r.state.jumps)
  return { times: Float64Array.from(jumps, (j) => j.t), states: Int32Array.from(jumps, (j) => j.x) }
}

/**
 * The states of a trajectory at given times (the state held at each time).
 *
 * @param trajectory - A trajectory.
 * @param times - Query times (at or after the start).
 * @returns The states.
 * @concept continuous-time-markov-chains
 * @example
 * ```ts
 * import { trajectoryAt } from 'mdml-engine/stochastic'
 * Array.from(trajectoryAt({ times: Float64Array.from([0, 1, 2.5]), states: Int32Array.from([0, 2, 1]) }, [0.5, 1, 3])) // => [0, 2, 1]
 * ```
 */
export function trajectoryAt(trajectory: Trajectory, times: ArrayLike<number>): Int32Array {
  const out = new Int32Array(times.length)
  for (let k = 0; k < times.length; k++) {
    const t = times[k] ?? 0
    let lo = 0
    let hi = trajectory.times.length - 1
    while (lo < hi) {
      const mid = (lo + hi + 1) >> 1
      if ((trajectory.times[mid] ?? 0) <= t) lo = mid
      else hi = mid - 1
    }
    out[k] = trajectory.states[lo] ?? 0
  }
  return out
}

/**
 * A continuous-time chain sampled exactly on a grid: a discrete chain with transition matrices exp(dt_i Q) between grid
 * times, as a traced path simulation.
 *
 * @param key - The key.
 * @param q - The generator.
 * @param x0 - The initial state.
 * @param times - The grid.
 * @param options - The number of paths (default 1) and tracing.
 * @returns Paths `[paths, times.length]` of states.
 * @concept continuous-time-markov-chains
 * @example
 * ```ts
 * import { key } from 'mdml-engine/random'
 * import { ctmcPaths } from 'mdml-engine/stochastic'
 * ctmcPaths(key(0), [[-1, 1], [1, -1]], 0, [0, 0.5, 1], { paths: 4 }).values.shape // => [4, 3]
 * ```
 */
export function ctmcPaths(
  key: Key,
  q: TensorLike,
  x0: number,
  times: TimesLike,
  options: { readonly paths?: number } & TraceOptions = {},
): Paths {
  const { n } = generatorRows(q, 'ctmcPaths')
  const paths = positiveInteger(options.paths ?? 1, 'paths', 'ctmcPaths')
  if (!(Number.isInteger(x0) && x0 >= 0 && x0 < n)) throw new MdmlError('value', 'ctmcPaths: bad x0')
  const cache = new Map<number, number[][]>()
  const rowsAt = (dt: number): number[][] => {
    const hit = cache.get(dt)
    if (hit !== undefined) return hit
    const m = primalNumbers(ctmcTransition(q, dt))
    const rows = Array.from({ length: n }, (_, i) => m.slice(i * n, (i + 1) * n).map((v) => Math.max(v, 0)))
    cache.set(dt, rows)
    return rows
  }
  return simulatePaths(
    key,
    {
      name: 'ctmc-grid',
      config: { states: n, paths },
      times,
      x0: vector(new Array<number>(paths).fill(x0)),
      step: (x, c) => nextStates(x, rowsAt(c.dt), c.key),
    },
    traceOf(options),
  )
}
