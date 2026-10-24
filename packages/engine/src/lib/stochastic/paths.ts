import { type Json, WhiteboxError } from '../base/index.ts'
import type { Key } from '../random/index.ts'
import {
  asTensor,
  isTraced,
  item,
  mean,
  primalOf,
  reshape,
  sliceAxis,
  squeeze,
  stack,
  sub,
  sum,
  type TensorLike,
  type Value,
} from '../tensor/index.ts'
import { type Algorithm, channels, type RunOptions, runAlgorithm } from '../trace/index.ts'
import { asTimes, observable, positiveInteger, stepsOf, type TimesLike } from './_shared.ts'

export type { TimesLike } from './_shared.ts'

/**
 * Sample paths of a process on a time grid: `values` has shape `[paths, times.length, ...state]`, with an empty state
 * shape for a scalar process and `[d]` for a d-dimensional one. Every simulator in `stochastic` returns this shape, and
 * the calculus functions read time from axis 1.
 */
export interface Paths {
  readonly times: Float64Array
  readonly values: Value
}

/** How a simulation is traced: the tracer and policy (or a parent run) and the run id. */
export type TraceOptions = Partial<Pick<RunOptions, 'tracer' | 'policy' | 'parent' | 'runId'>>

/** What one step of a path simulation sees. */
export interface PathStepContext {
  /** The index of the step: it moves the state from `times[i]` to `times[i + 1]`. */
  readonly i: number
  /** The time at the start of the step. */
  readonly t: number
  /** The step size. */
  readonly dt: number
  /** The key of this step (the same at step `i` however the run is paused or replayed). */
  readonly key: Key
}

/** A Markov path simulation: an initial state and a transition from one grid time to the next. */
export interface PathSpec {
  readonly name: string
  readonly config?: Json
  readonly times: TimesLike
  /** The initial state `[paths, ...state]`. */
  readonly x0: TensorLike
  /** Draw the state at `times[i + 1]` given the state at `times[i]`. */
  readonly step: (x: Value, ctx: PathStepContext) => Value
}

/** A persistent list, newest first: a simulation step adds one entry in O(1) without copying the past. */
export interface History<T> {
  readonly item: T
  readonly previous: History<T> | undefined
}

/** The states visited by a path simulation so far. */
export type PathHistory = History<Value>

/** The state of a path simulation. */
export interface PathState {
  readonly i: number
  readonly x: Value
  readonly history: PathHistory
}

/**
 * Generate a time grid of `steps` equal steps from `t0` to `t1`.
 *
 * @param t1 - The end time.
 * @param steps - The number of steps (a positive integer).
 * @param t0 - The start time (default 0).
 * @returns `steps + 1` increasing times.
 * @throws WhiteboxError with code `value` unless `t1 > t0` and `steps` is a positive integer.
 * @example
 * ```ts
 * import { timeGrid } from 'whitebox-engine/stochastic'
 * Array.from(timeGrid(1, 4)) // => [0, 0.25, 0.5, 0.75, 1]
 * ```
 */
export function timeGrid(t1: number, steps: number, t0 = 0): Float64Array {
  positiveInteger(steps, 'steps', 'timeGrid')
  if (!(t1 > t0)) throw new WhiteboxError('value', 'timeGrid: need t1 > t0')
  const h = (t1 - t0) / steps
  return Float64Array.from({ length: steps + 1 }, (_, i) => (i === steps ? t1 : t0 + i * h))
}

/**
 * A path simulation as a traced {@link Algorithm}: one step per grid interval, each drawing its noise from the step
 * key. Observations record the time, the mean of the state over paths and the state itself.
 *
 * @param spec - The name, grid, initial state and transition.
 * @returns The algorithm (run it with `runAlgorithm`, or use {@link simulatePaths}).
 * @concept stochastic-processes
 * @example
 * ```ts
 * import { pathAlgorithm } from 'whitebox-engine/stochastic'
 * import { add } from 'whitebox-engine/tensor'
 * pathAlgorithm({ name: 'drift', times: [0, 1, 2], x0: [0], step: (x, c) => add(x, c.dt) }).name // => 'drift'
 * ```
 */
export function pathAlgorithm(spec: PathSpec): Algorithm<PathState> {
  const times = asTimes(spec.times, spec.name)
  const dts = stepsOf(times)
  const x0 = asTensor(spec.x0)
  return {
    name: spec.name,
    config: spec.config ?? { steps: dts.length },
    schema: channels({
      metrics: {
        t: { description: 'the time reached' },
        mean: { description: 'the mean of the state over paths and coordinates' },
      },
      tensors: { state: { role: 'state', description: 'the state of every path' } },
    }),
    init: () => ({ i: 0, x: x0, history: { item: x0, previous: undefined } }),
    step: (s, ctx) => {
      const x = spec.step(s.x, { i: s.i, t: times[s.i] ?? 0, dt: dts[s.i] ?? 0, key: ctx.key })
      return { i: s.i + 1, x, history: { item: x, previous: s.history } }
    },
    done: (s) => (s.i >= dts.length ? 'completed' : undefined),
    observe: (s) => {
      const value = isTraced(s.x) ? primalOf(s.x) : s.x
      const shown = observable(s.x)
      return {
        metrics: { t: times[s.i] ?? 0, mean: item(mean(value)) },
        ...(shown === undefined ? {} : { tensors: { state: { role: 'state' as const, value: shown } } }),
      }
    },
  }
}

/**
 * Simulate paths by running {@link pathAlgorithm}: the states at every grid time stacked on axis 1. The simulation is
 * built from tensor operations, so paths differentiate in any parameters the transition uses (pathwise derivatives).
 *
 * @param key - The key; step `i` draws from a key derived from it and `i`.
 * @param spec - The name, grid, initial state `[paths, ...state]` and transition.
 * @param options - Tracer, policy, parent run and run id.
 * @returns The paths `[paths, times.length, ...state]`.
 * @concept stochastic-processes
 * @example
 * ```ts
 * import { key } from 'whitebox-engine/random'
 * import { simulatePaths } from 'whitebox-engine/stochastic'
 * import { add, toArray } from 'whitebox-engine/tensor'
 * toArray(simulatePaths(key(0), { name: 'clock', times: [0, 1, 3], x0: [0], step: (x, c) => add(x, c.dt) }).values) // => [[0, 1, 3]]
 * ```
 */
export function simulatePaths(key: Key, spec: PathSpec, options: TraceOptions = {}): Paths {
  const times = asTimes(spec.times, spec.name)
  const result = runAlgorithm(pathAlgorithm(spec), { ...options, key, maxSteps: times.length - 1 })
  return { times, values: stack(historyArray(result.state.history), 1) }
}

/**
 * The entries of a history in chronological order (oldest first).
 *
 * @param history - A history (or undefined for none).
 * @returns The items, oldest first.
 * @example
 * ```ts
 * import { historyArray } from 'whitebox-engine/stochastic'
 * historyArray({ item: 2, previous: { item: 1, previous: undefined } }) // => [1, 2]
 * ```
 */
export function historyArray<T>(history: History<T> | undefined): T[] {
  const out: T[] = []
  for (let h = history; h !== undefined; h = h.previous) out.push(h.item)
  return out.reverse()
}

/**
 * The increments of paths along time: `values[:, i + 1] - values[:, i]`.
 *
 * @param values - Paths `[paths, n, ...state]`.
 * @returns `[paths, n - 1, ...state]`.
 * @example
 * ```ts
 * import { pathIncrements } from 'whitebox-engine/stochastic'
 * import { toArray } from 'whitebox-engine/tensor'
 * toArray(pathIncrements([[0, 1, 3]])) // => [[1, 2]]
 * ```
 */
export function pathIncrements(values: TensorLike): Value {
  const x = asTensor(values)
  if (x.shape.length < 2) throw new WhiteboxError('shape', 'pathIncrements: values must be [paths, n, ...state]')
  const n = x.shape[1] ?? 0
  return sub(sliceAxis(x, 1, 1, n), sliceAxis(x, 1, 0, n - 1))
}

/**
 * Sum consecutive increments in groups of `factor`: the increments of the same paths on a grid `factor` times
 * coarser, as used by strong convergence studies.
 *
 * @param increments - Increments `[paths, n, ...]` with `n` divisible by `factor`.
 * @param factor - The coarsening factor (a positive integer).
 * @returns `[paths, n / factor, ...]`.
 * @throws WhiteboxError with code `shape` when `n` is not a multiple of `factor`.
 * @example
 * ```ts
 * import { coarsenIncrements } from 'whitebox-engine/stochastic'
 * import { toArray } from 'whitebox-engine/tensor'
 * toArray(coarsenIncrements([[1, 2, 3, 4]], 2)) // => [[3, 7]]
 * ```
 */
export function coarsenIncrements(increments: TensorLike, factor: number): Value {
  positiveInteger(factor, 'factor', 'coarsenIncrements')
  const x = asTensor(increments)
  const [p = 1, n = 0, ...rest] = x.shape
  if (n % factor !== 0)
    throw new WhiteboxError('shape', `coarsenIncrements: ${n} steps are not a multiple of ${factor}`)
  return sum(reshape(x, [p, n / factor, factor, ...rest]), 2)
}

/**
 * The states of every path at one grid index (negative indices count from the end).
 *
 * @param paths - Paths, or their values `[paths, n, ...state]`.
 * @param index - The time index.
 * @returns `[paths, ...state]`.
 * @example
 * ```ts
 * import { valuesAt } from 'whitebox-engine/stochastic'
 * import { toArray } from 'whitebox-engine/tensor'
 * toArray(valuesAt([[0, 1, 3], [0, 2, 5]], -1)) // => [3, 5]
 * ```
 */
export function valuesAt(paths: Paths | TensorLike, index: number): Value {
  const x = asTensor(isPaths(paths) ? paths.values : paths)
  const n = x.shape[1] ?? 0
  const i = index < 0 ? n + index : index
  if (!(Number.isInteger(i) && i >= 0 && i < n))
    throw new WhiteboxError('value', `valuesAt: index ${index} out of range`)
  return squeeze(sliceAxis(x, 1, i, i + 1), 1)
}

/**
 * Whether a value is a {@link Paths} object.
 *
 * @param x - Anything.
 * @returns `true` for `{ times, values }`.
 * @example
 * ```ts
 * import { isPaths, timeGrid } from 'whitebox-engine/stochastic'
 * isPaths({ times: timeGrid(1, 2), values: [[0, 0, 0]] }) // => true
 * ```
 */
export function isPaths(x: unknown): x is Paths {
  return typeof x === 'object' && x !== null && 'times' in x && 'values' in x
}
