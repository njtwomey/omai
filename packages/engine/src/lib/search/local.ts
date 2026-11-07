/**
 * Local search, which improves a single complete state by moving between neighbours instead of building paths: hill
 * climbing (with sideways moves and random restarts), simulated annealing with exponential, geometric and logarithmic
 * cooling schedules, tabu search, local beam search and late acceptance hill climbing, as traced algorithms. Also
 * neighbourhoods for permutations and bit strings, and local search over any configuration space.
 */

import { argmaxOf, type Json, MdmlError } from '../base/index.ts'
import { Generator, type Key } from '../random/index.ts'
import { type Space, type SpaceValue, sample, neighbours as spaceNeighbours } from '../space/index.ts'
import { type Algorithm, channels, runAlgorithm, type TraceSchema } from '../trace/index.ts'
import { type SearchRunOptions, type StateKey, stateKeyOf } from './problem.ts'

/**
 * A local search problem (AIMA 4.1): complete states, a neighbourhood, and a value to maximise. Searches start from
 * `initial` or a `random` state, and stop early at a state that passes `isGoal`. `randomNeighbour` (default: uniform
 * over `neighbours`) serves the searches that sample moves.
 *
 * @concept local-search
 */
export interface LocalSearchProblem<S> {
  readonly initial?: S
  random?(key: Key): S
  neighbours(state: S): readonly S[]
  randomNeighbour?(state: S, key: Key): S | undefined
  value(state: S): number
  isGoal?(state: S): boolean
  key?(state: S): StateKey
}

/** How a local search ended. */
export type LocalStatus = 'goal' | 'local-optimum' | 'frozen' | 'budget'

/** The outcome of a local search: the best state found, its value, and the effort. */
export interface LocalSearchResult<S> {
  readonly state: S
  readonly value: number
  readonly status: LocalStatus
  readonly iterations: number
  /** Calls of `value`. */
  readonly evaluations: number
  /** The current value after each iteration (the start first). */
  readonly history: readonly number[]
}

/** The state of a traced local search, as seen by observers. */
export interface LocalSearchState<S> {
  readonly status: 'running' | LocalStatus
  readonly current: S
  readonly value: number
  readonly best: S
  readonly bestValue: number
  readonly iterations: number
  readonly evaluations: number
  /** The temperature of simulated annealing (NaN elsewhere). */
  readonly temperature: number
  /** The last move: the candidate's value and whether it was accepted. */
  readonly move: { readonly value: number; readonly accepted: boolean } | null
  readonly history: readonly number[]
}

interface Mutable<S> {
  status: 'running' | LocalStatus
  current: S
  value: number
  best: S
  bestValue: number
  iterations: number
  evaluations: number
  temperature: number
  move: { value: number; accepted: boolean } | null
  history: number[]
}

/** Options shared by the local searches: an iteration budget and tracing. */
export interface LocalSearchOptions extends SearchRunOptions {
  /** Stop after this many iterations (default 10000). */
  readonly maxIterations?: number
}

const LOCAL_SCHEMA: TraceSchema = channels({
  metrics: {
    value: { description: 'value of the current state', direction: 'higher' },
    best: { description: 'best value so far', direction: 'higher' },
    temperature: { description: 'annealing temperature' },
    evaluations: { description: 'value evaluations so far' },
  },
  messages: {
    move: { description: 'a candidate move', payload: { value: 'candidate value', accepted: 'whether it was taken' } },
  },
})

function startState<S>(problem: LocalSearchProblem<S>, key: Key, name: string): S {
  if (problem.initial !== undefined) return problem.initial
  if (problem.random !== undefined) return problem.random(key)
  throw new MdmlError('value', `${name}: the problem needs an initial state or a random one`)
}

function begin<S>(problem: LocalSearchProblem<S>, state: S): Mutable<S> {
  const value = problem.value(state)
  return {
    status: problem.isGoal?.(state) === true ? 'goal' : 'running',
    current: state,
    value,
    best: state,
    bestValue: value,
    iterations: 0,
    evaluations: 1,
    temperature: Number.NaN,
    move: null,
    history: [value],
  }
}

/** Move to a state (already evaluated), updating the best and the goal test. */
function moveTo<S>(problem: LocalSearchProblem<S>, s: Mutable<S>, next: S, value: number): void {
  s.current = next
  s.value = value
  if (value > s.bestValue) {
    s.best = next
    s.bestValue = value
  }
  if (problem.isGoal?.(next) === true) s.status = 'goal'
}

function evaluate<S>(problem: LocalSearchProblem<S>, s: Mutable<S>, state: S): number {
  s.evaluations += 1
  return problem.value(state)
}

function pick<T>(rng: Generator, items: readonly T[]): T | undefined {
  if (items.length === 0) return undefined
  return items[Math.min(items.length - 1, Math.floor((rng.uniform([1]).data[0] ?? 0) * items.length))]
}

function randomNeighbour<S>(problem: LocalSearchProblem<S>, state: S, rng: Generator): S | undefined {
  if (problem.randomNeighbour !== undefined) return problem.randomNeighbour(state, rng.nextKey())
  return pick(rng, problem.neighbours(state))
}

function observeLocal<S>(s: LocalSearchState<S>): {
  metrics: Record<string, number>
  messages: { kind: string; payload: Json }[]
} {
  return {
    metrics: { value: s.value, best: s.bestValue, temperature: s.temperature, evaluations: s.evaluations },
    messages: s.move === null ? [] : [{ kind: 'move', payload: { value: s.move.value, accepted: s.move.accepted } }],
  }
}

/** A local search's state with the memory of its variant (sideways count, tabu list, acceptance history). */
interface WithMemory<S, X> extends Mutable<S> {
  readonly memory: X
}

function localAlgorithm<S, X>(
  name: string,
  config: Json,
  problem: LocalSearchProblem<S>,
  memory: (s: Mutable<S>) => X,
  stepFn: (s: Mutable<S>, memory: X, rng: Generator) => void,
): Algorithm<LocalSearchState<S>> {
  const algorithm: Algorithm<WithMemory<S, X>> = {
    name,
    config,
    schema: LOCAL_SCHEMA,
    init: (key) => {
      const s = begin(problem, startState(problem, key, name))
      return { ...s, memory: memory(s) }
    },
    step: (s, ctx) => {
      s.move = null
      stepFn(s, s.memory, new Generator(ctx.key))
      s.iterations += 1
      s.history.push(s.value)
      return s
    },
    done: (s) => (s.status === 'running' ? undefined : s.status === 'goal' ? 'converged' : 'stopped'),
    observe: observeLocal,
  }
  return algorithm
}

function runLocal<S>(algorithm: Algorithm<LocalSearchState<S>>, options: LocalSearchOptions): LocalSearchResult<S> {
  const { maxIterations = 10_000, ...run } = options
  const { state } = runAlgorithm(algorithm, { ...run, maxSteps: maxIterations })
  return {
    state: state.best,
    value: state.bestValue,
    status: state.status === 'running' ? 'budget' : state.status,
    iterations: state.iterations,
    evaluations: state.evaluations,
    history: state.history,
  }
}

/** Options of {@link hillClimbing}. */
export interface HillClimbingOptions extends LocalSearchOptions {
  /**
   * Steepest ascent (the best neighbour, ties to the first; default), first choice (the first sampled neighbour that
   * improves, from at most `tries` samples) or stochastic (a uniformly random improving neighbour).
   */
  readonly variant?: 'steepest' | 'first-choice' | 'stochastic'
  /** Sideways moves allowed in a row on plateaus (default 0; AIMA uses 100 for 8-queens). */
  readonly sideways?: number
  /** Samples per first-choice step (default 100). */
  readonly tries?: number
}

function hillConfig(options: HillClimbingOptions): {
  variant: 'steepest' | 'first-choice' | 'stochastic'
  sideways: number
  tries: number
} {
  const variant = options.variant ?? 'steepest'
  const sideways = options.sideways ?? 0
  const tries = options.tries ?? 100
  if (!(Number.isInteger(sideways) && sideways >= 0)) {
    throw new MdmlError('value', `hillClimbing: sideways ${sideways} must be a non-negative integer`)
  }
  return { variant, sideways, tries }
}

/**
 * Hill climbing as a traced algorithm (AIMA figure 4.2, aima-python's `hill_climbing`): move to a better neighbour
 * until none is better, optionally taking a limited number of sideways moves across plateaus. Each step makes one move.
 *
 * @param problem - The problem.
 * @param options - The variant, sideways moves and first-choice samples.
 * @returns The algorithm; it ends `local-optimum` when no move improves, or `goal`.
 * @concept local-search
 * @example
 * ```ts
 * import { hillClimbingAlgorithm } from 'mdml-engine/search'
 * const p = { initial: 0, neighbours: (x: number) => [x - 1, x + 1], value: (x: number) => -((x - 3) ** 2) }
 * hillClimbingAlgorithm(p).name // => 'hill-climbing'
 * ```
 */
export function hillClimbingAlgorithm<S>(
  problem: LocalSearchProblem<S>,
  options: HillClimbingOptions = {},
): Algorithm<LocalSearchState<S>> {
  const { variant, sideways, tries } = hillConfig(options)
  return localAlgorithm(
    'hill-climbing',
    { variant, sideways, tries },
    problem,
    () => ({ flat: 0 }),
    (s, memory, rng) => {
      let next: S | undefined
      let nextValue = Number.NEGATIVE_INFINITY
      if (variant === 'first-choice') {
        for (let i = 0; i < tries; i++) {
          const n = randomNeighbour(problem, s.current, rng)
          if (n === undefined) break
          const v = evaluate(problem, s, n)
          if (v > s.value) {
            next = n
            nextValue = v
            break
          }
        }
      } else {
        const scored = problem.neighbours(s.current).map((n) => ({ n, v: evaluate(problem, s, n) }))
        if (variant === 'stochastic') {
          const up = scored.filter((x) => x.v > s.value)
          const chosen = pick(rng, up.length > 0 ? up : scored.filter((x) => x.v === s.value && memory.flat < sideways))
          if (chosen !== undefined) {
            next = chosen.n
            nextValue = chosen.v
          }
        } else {
          for (const x of scored) {
            if (x.v > nextValue) {
              next = x.n
              nextValue = x.v
            }
          }
        }
      }
      const improves = next !== undefined && nextValue > s.value
      const sideStep = next !== undefined && nextValue === s.value && memory.flat < sideways
      s.move = next === undefined ? null : { value: nextValue, accepted: improves || sideStep }
      if (next !== undefined && (improves || sideStep)) {
        memory.flat = improves ? 0 : memory.flat + 1
        moveTo(problem, s, next, nextValue)
      } else s.status = 'local-optimum'
    },
  )
}

/**
 * Hill climbing.
 *
 * @param problem - The problem.
 * @param options - The variant, sideways moves, first-choice samples, the iteration budget and tracing.
 * @returns The best state reached and its value.
 * @concept local-search
 * @example
 * ```ts
 * import { hillClimbing } from 'mdml-engine/search'
 * const p = { initial: 0, neighbours: (x: number) => [x - 1, x + 1], value: (x: number) => -((x - 3) ** 2) }
 * const r = hillClimbing(p)
 * [r.state, r.status, r.iterations] // => [3, 'local-optimum', 4]
 * ```
 */
export function hillClimbing<S>(
  problem: LocalSearchProblem<S>,
  options: HillClimbingOptions = {},
): LocalSearchResult<S> {
  return runLocal(hillClimbingAlgorithm(problem, options), options)
}

/** The state of random-restart hill climbing: the restarts made and the best climb so far. */
export interface RestartState<S> {
  readonly restarts: number
  readonly best: LocalSearchResult<S> | null
  readonly evaluations: number
}

/**
 * Random-restart hill climbing as a traced algorithm (AIMA 4.1.1): each step runs a whole hill climb from a fresh
 * random state as a child run, keeping the best, until a climb reaches a goal.
 *
 * @param problem - The problem (needs `random`).
 * @param options - The climbing options for each restart.
 * @returns The algorithm.
 * @throws MdmlError with code `value` when the problem has no `random` state.
 * @concept local-search
 * @example
 * ```ts
 * import { randomRestartHillClimbingAlgorithm } from 'mdml-engine/search'
 * const p = { random: () => 0, neighbours: (x: number) => [x - 1, x + 1], value: (x: number) => -Math.abs(x) }
 * randomRestartHillClimbingAlgorithm(p).name // => 'random-restart-hill-climbing'
 * ```
 */
export function randomRestartHillClimbingAlgorithm<S>(
  problem: LocalSearchProblem<S>,
  options: HillClimbingOptions = {},
): Algorithm<RestartState<S>> {
  const random = problem.random
  if (random === undefined) throw new MdmlError('value', 'randomRestartHillClimbing: the problem needs random')
  const {
    maxIterations = 10_000,
    tracer: _t,
    policy: _p,
    parent: _pa,
    runId: _r,
    seed: _s,
    key: _k,
    ...climb
  } = options
  return {
    name: 'random-restart-hill-climbing',
    config: { ...hillConfig(options), maxIterations },
    schema: channels({
      metrics: {
        best: { description: 'best value over restarts', direction: 'higher' },
        restarts: { description: 'restarts so far' },
      },
    }),
    init: () => ({ restarts: 0, best: null, evaluations: 0 }),
    step: (s, ctx) => {
      const fresh = { ...problem, initial: random(ctx.key) }
      const algorithm = hillClimbingAlgorithm(fresh, climb)
      const { state } = runAlgorithm(algorithm, { maxSteps: maxIterations, parent: ctx.run, key: ctx.key })
      const r: LocalSearchResult<S> = {
        state: state.best,
        value: state.bestValue,
        status: state.status === 'running' ? 'budget' : state.status,
        iterations: state.iterations,
        evaluations: state.evaluations,
        history: state.history,
      }
      return {
        restarts: s.restarts + 1,
        best: s.best === null || r.value > s.best.value ? r : s.best,
        evaluations: s.evaluations + r.evaluations,
      }
    },
    done: (s) => (s.best?.status === 'goal' ? 'converged' : undefined),
    observe: (s) => ({ metrics: { best: s.best?.value ?? Number.NaN, restarts: s.restarts } }),
  }
}

/**
 * Random-restart hill climbing: complete with probability approaching 1 when goals exist.
 *
 * @param problem - The problem (needs `random`).
 * @param options - The number of restarts (default 100), the climbing options for each, and tracing.
 * @returns The best climb, with its status (`goal` when a restart reached one) and the evaluations of all restarts.
 * @throws MdmlError with code `value` when the problem has no `random` state.
 * @concept local-search
 * @example
 * ```ts
 * import { randint } from 'mdml-engine/random'
 * import { randomRestartHillClimbing } from 'mdml-engine/search'
 * // Two peaks: x = -4 (value 2) and x = 4 (value 5); restarts from random points in [-9, 9] find the higher one.
 * const p = { random: (k: Parameters<typeof randint>[0]) => randint(k, [], -9, 10).data[0] ?? 0, neighbours: (x: number) => [x - 1, x + 1], value: (x: number) => Math.max(2 - Math.abs(x + 4), 5 - Math.abs(x - 4)), isGoal: (x: number) => x === 4 }
 * randomRestartHillClimbing(p, { restarts: 20, seed: 0 }).state // => 4
 * ```
 */
export function randomRestartHillClimbing<S>(
  problem: LocalSearchProblem<S>,
  options: HillClimbingOptions & { readonly restarts?: number } = {},
): LocalSearchResult<S> {
  const { restarts = 100, tracer, policy, parent, runId, seed, key, ...climb } = options
  const run = {
    ...(tracer === undefined ? {} : { tracer }),
    ...(policy === undefined ? {} : { policy }),
    ...(parent === undefined ? {} : { parent }),
    ...(runId === undefined ? {} : { runId }),
    ...(seed === undefined ? {} : { seed }),
    ...(key === undefined ? {} : { key }),
  }
  const { state } = runAlgorithm(randomRestartHillClimbingAlgorithm(problem, climb), { ...run, maxSteps: restarts })
  const best = state.best
  if (best === null) throw new MdmlError('value', 'randomRestartHillClimbing: restarts must be at least 1')
  return { ...best, evaluations: state.evaluations }
}

/** A cooling schedule: the temperature at iteration t (0 stops the search). */
export type CoolingSchedule = (t: number) => number

/**
 * aima-python's `exp_schedule`: T(t) = k exp(-lambda t) for t below the limit, then 0.
 *
 * @param options - k (default 20), lambda (default 0.005) and the limit (default 100).
 * @returns The schedule.
 * @example
 * ```ts
 * import { exponentialSchedule } from 'mdml-engine/search'
 * const T = exponentialSchedule()
 * [T(0), T(100)] // => [20, 0]
 * ```
 */
export function exponentialSchedule(
  options: { readonly k?: number; readonly lambda?: number; readonly limit?: number } = {},
): CoolingSchedule {
  const { k = 20, lambda = 0.005, limit = 100 } = options
  return (t) => (t < limit ? k * Math.exp(-lambda * t) : 0)
}

/**
 * Geometric cooling T(t) = t0 alpha^t, stopped (0) once below `min`.
 *
 * @param t0 - The starting temperature.
 * @param alpha - The cooling factor in (0, 1).
 * @param min - The temperature at which to stop (default 1e-3).
 * @returns The schedule.
 * @throws MdmlError with code `value` when alpha is not in (0, 1) or t0 is not positive.
 * @example
 * ```ts
 * import { geometricSchedule } from 'mdml-engine/search'
 * geometricSchedule(10, 0.5)(2) // => 2.5
 * ```
 */
export function geometricSchedule(t0: number, alpha: number, min = 1e-3): CoolingSchedule {
  if (!(alpha > 0 && alpha < 1) || !(t0 > 0)) {
    throw new MdmlError('value', 'geometricSchedule: need t0 > 0 and 0 < alpha < 1')
  }
  return (t) => {
    const T = t0 * alpha ** t
    return T < min ? 0 : T
  }
}

/**
 * Logarithmic cooling T(t) = c / log(t + 2) (Geman and Geman 1984; the schedule under which annealing converges in
 * probability to a global optimum), stopped after `limit` iterations.
 *
 * @param c - The constant (at least the depth of the deepest local minimum for the guarantee).
 * @param limit - The iteration at which to stop (default 10000).
 * @returns The schedule.
 * @example
 * ```ts
 * import { logarithmicSchedule } from 'mdml-engine/search'
 * logarithmicSchedule(2)(0).toFixed(4) // => '2.8854'
 * ```
 */
export function logarithmicSchedule(c: number, limit = 10_000): CoolingSchedule {
  return (t) => (t < limit ? c / Math.log(t + 2) : 0)
}

/**
 * Simulated annealing as a traced algorithm (AIMA figure 4.5, aima-python's `simulated_annealing`): at temperature
 * T(t) pick a random neighbour, take it if it is better, else with probability exp(delta / T); stop when T reaches 0.
 * The best state seen is kept.
 *
 * @param problem - The problem.
 * @param options - The schedule (default aima-python's `exp_schedule`).
 * @returns The algorithm; it ends `frozen` at T = 0.
 * @concept local-search
 * @example
 * ```ts
 * import { simulatedAnnealingAlgorithm } from 'mdml-engine/search'
 * const p = { initial: 0, neighbours: (x: number) => [x - 1, x + 1], value: (x: number) => -Math.abs(x - 3) }
 * simulatedAnnealingAlgorithm(p).name // => 'simulated-annealing'
 * ```
 */
export function simulatedAnnealingAlgorithm<S>(
  problem: LocalSearchProblem<S>,
  options: { readonly schedule?: CoolingSchedule } = {},
): Algorithm<LocalSearchState<S>> {
  const schedule = options.schedule ?? exponentialSchedule()
  return localAlgorithm(
    'simulated-annealing',
    {},
    problem,
    () => null,
    (s, _memory, rng) => {
      const T = schedule(s.iterations)
      s.temperature = T
      if (!(T > 0)) {
        s.status = 'frozen'
        return
      }
      const next = randomNeighbour(problem, s.current, rng)
      if (next === undefined) {
        s.status = 'local-optimum'
        return
      }
      const v = evaluate(problem, s, next)
      const delta = v - s.value
      const accepted = delta > 0 || (rng.uniform([1]).data[0] ?? 1) < Math.exp(delta / T)
      s.move = { value: v, accepted }
      if (accepted) moveTo(problem, s, next, v)
    },
  )
}

/**
 * Simulated annealing.
 *
 * @param problem - The problem.
 * @param options - The schedule, the iteration budget, a seed or key, and tracing.
 * @returns The best state seen and its value.
 * @concept local-search
 * @index function Simulated annealing local search.
 * @example
 * ```ts
 * import { geometricSchedule, simulatedAnnealing } from 'mdml-engine/search'
 * const p = { initial: 0, neighbours: (x: number) => [x - 1, x + 1], value: (x: number) => -Math.abs(x - 7), isGoal: (x: number) => x === 7 }
 * simulatedAnnealing(p, { schedule: geometricSchedule(2, 0.99), seed: 1 }).state // => 7
 * ```
 */
export function simulatedAnnealing<S>(
  problem: LocalSearchProblem<S>,
  options: LocalSearchOptions & { readonly schedule?: CoolingSchedule } = {},
): LocalSearchResult<S> {
  const { schedule, ...rest } = options
  return runLocal(simulatedAnnealingAlgorithm(problem, schedule === undefined ? {} : { schedule }), rest)
}

/**
 * Tabu search as a traced algorithm (Glover 1986): move to the best neighbour that is not tabu, even if it is worse;
 * the last `tenure` states visited are tabu, unless a move to one beats the best so far (aspiration). Each step makes
 * one move.
 *
 * @param problem - The problem.
 * @param options - The tabu tenure (default 10) and whether aspiration is used (default true).
 * @returns The algorithm.
 * @concept local-search
 * @example
 * ```ts
 * import { tabuSearchAlgorithm } from 'mdml-engine/search'
 * const p = { initial: 0, neighbours: (x: number) => [x - 1, x + 1], value: (x: number) => -Math.abs(x - 3) }
 * tabuSearchAlgorithm(p).name // => 'tabu'
 * ```
 */
export function tabuSearchAlgorithm<S>(
  problem: LocalSearchProblem<S>,
  options: { readonly tenure?: number; readonly aspiration?: boolean } = {},
): Algorithm<LocalSearchState<S>> {
  const tenure = options.tenure ?? 10
  const aspiration = options.aspiration ?? true
  const key = stateKeyOf(problem)
  return localAlgorithm(
    'tabu',
    { tenure, aspiration },
    problem,
    () => ({ tabu: [] as StateKey[] }),
    (s, memory) => {
      let next: S | undefined
      let nextValue = Number.NEGATIVE_INFINITY
      for (const n of problem.neighbours(s.current)) {
        const v = evaluate(problem, s, n)
        const allowed = !memory.tabu.includes(key(n)) || (aspiration && v > s.bestValue)
        if (allowed && v > nextValue) {
          next = n
          nextValue = v
        }
      }
      if (next === undefined) {
        s.status = 'local-optimum'
        return
      }
      s.move = { value: nextValue, accepted: true }
      memory.tabu.push(key(s.current))
      if (memory.tabu.length > tenure) memory.tabu.shift()
      moveTo(problem, s, next, nextValue)
    },
  )
}

/**
 * Tabu search: escapes local optima by forbidding recent states; stops at a goal or on the iteration budget.
 *
 * @param problem - The problem.
 * @param options - The tenure, aspiration, the iteration budget (default 1000) and tracing.
 * @returns The best state seen and its value.
 * @concept local-search
 * @example
 * ```ts
 * import { tabuSearch } from 'mdml-engine/search'
 * // A local peak at 2 (value 2) before the global one at 6 (value 6).
 * const f = (x: number) => [0, 1, 2, 1, 0, 3, 6, 2][x] ?? -10
 * const p = { initial: 0, neighbours: (x: number) => [x - 1, x + 1].filter((y) => y >= 0 && y < 8), value: f }
 * tabuSearch(p, { tenure: 3, maxIterations: 20 }).state // => 6
 * ```
 */
export function tabuSearch<S>(
  problem: LocalSearchProblem<S>,
  options: LocalSearchOptions & { readonly tenure?: number; readonly aspiration?: boolean } = {},
): LocalSearchResult<S> {
  const { tenure, aspiration, maxIterations = 1000, ...rest } = options
  const algorithm = tabuSearchAlgorithm(problem, {
    ...(tenure === undefined ? {} : { tenure }),
    ...(aspiration === undefined ? {} : { aspiration }),
  })
  return runLocal(algorithm, { ...rest, maxIterations })
}

/** The state of a local beam search: the k states held. */
export interface BeamState<S> {
  readonly states: readonly S[]
  readonly values: readonly number[]
  readonly best: S
  readonly bestValue: number
  readonly evaluations: number
  readonly status: 'running' | LocalStatus
}

/**
 * Local beam search as a traced algorithm (AIMA 4.1.3): hold k states; each step pools all their neighbours and keeps
 * the k best (or, stochastically, k drawn with probability proportional to exp(value / temperature)). Stops at a goal
 * or when no neighbour beats the beam's best.
 *
 * @param problem - The problem (needs `random` unless `initial` is given, which then seeds every slot).
 * @param options - The beam width k (default 4), stochastic selection and its temperature (default 1).
 * @returns The algorithm.
 * @throws MdmlError with code `value` when k is not a positive integer.
 * @concept local-search
 * @example
 * ```ts
 * import { localBeamSearchAlgorithm } from 'mdml-engine/search'
 * const p = { initial: 0, neighbours: (x: number) => [x - 1, x + 1], value: (x: number) => -Math.abs(x - 3) }
 * localBeamSearchAlgorithm(p).name // => 'local-beam'
 * ```
 */
export function localBeamSearchAlgorithm<S>(
  problem: LocalSearchProblem<S>,
  options: { readonly k?: number; readonly stochastic?: boolean; readonly temperature?: number } = {},
): Algorithm<BeamState<S>> {
  const k = options.k ?? 4
  if (!(Number.isInteger(k) && k >= 1))
    throw new MdmlError('value', `localBeamSearch: k ${k} must be a positive integer`)
  const temperature = options.temperature ?? 1
  const key = stateKeyOf(problem)
  type Mut = { -readonly [P in keyof BeamState<S>]: BeamState<S>[P] }
  return {
    name: 'local-beam',
    config: { k, stochastic: options.stochastic ?? false },
    schema: channels({
      metrics: {
        best: { description: 'best value so far', direction: 'higher' },
        evaluations: { description: 'evaluations' },
      },
    }),
    init: (rootKey) => {
      const rng = new Generator(rootKey)
      const states = Array.from({ length: k }, () => startState(problem, rng.nextKey(), 'localBeamSearch'))
      const values = states.map((x) => problem.value(x))
      const b = argmaxOf(values)
      const best = states[b] as S
      const s: Mut = {
        states,
        values,
        best,
        bestValue: values[b] ?? Number.NEGATIVE_INFINITY,
        evaluations: k,
        status: states.some((x) => problem.isGoal?.(x) === true) ? 'goal' : 'running',
      }
      if (s.status === 'goal') s.best = states.find((x) => problem.isGoal?.(x) === true) as S
      return s
    },
    step: (s0, ctx) => {
      const s = s0 as Mut
      const rng = new Generator(ctx.key)
      const seen = new Set<StateKey>()
      const pool: { x: S; v: number }[] = []
      for (const x of s.states) {
        for (const n of problem.neighbours(x)) {
          const kn = key(n)
          if (seen.has(kn)) continue
          seen.add(kn)
          const v = problem.value(n)
          s.evaluations += 1
          pool.push({ x: n, v })
          if (problem.isGoal?.(n) === true) {
            s.best = n
            s.bestValue = v
            s.status = 'goal'
            return s
          }
        }
      }
      if (pool.length === 0 || pool.every((p) => p.v <= s.bestValue)) {
        s.status = 'local-optimum'
        return s
      }
      let chosen: { x: S; v: number }[]
      if (options.stochastic === true) {
        let hi = Number.NEGATIVE_INFINITY
        for (const p of pool) if (p.v > hi) hi = p.v
        const w = pool.map((p) => Math.exp((p.v - hi) / temperature))
        const total = w.reduce((a, b) => a + b, 0)
        chosen = Array.from({ length: k }, () => {
          let u = (rng.uniform([1]).data[0] ?? 0) * total
          for (let i = 0; i < pool.length; i++) {
            u -= w[i] ?? 0
            if (u <= 0) return pool[i] as { x: S; v: number }
          }
          return pool[pool.length - 1] as { x: S; v: number }
        })
      } else {
        chosen = pool
          .map((p, i) => ({ p, i }))
          .sort((a, b) => b.p.v - a.p.v || a.i - b.i)
          .slice(0, k)
          .map((q) => q.p)
      }
      s.states = chosen.map((c) => c.x)
      s.values = chosen.map((c) => c.v)
      for (const c of chosen) {
        if (c.v > s.bestValue) {
          s.best = c.x
          s.bestValue = c.v
        }
      }
      return s
    },
    done: (s) => (s.status === 'running' ? undefined : s.status === 'goal' ? 'converged' : 'stopped'),
    observe: (s) => ({ metrics: { best: s.bestValue, evaluations: s.evaluations } }),
  }
}

/**
 * Local beam search.
 *
 * @param problem - The problem.
 * @param options - The width, stochastic selection, the iteration budget (default 1000), a seed or key, and tracing.
 * @returns The best state found.
 * @throws MdmlError with code `value` when k is not a positive integer.
 * @concept local-search
 * @example
 * ```ts
 * import { localBeamSearch } from 'mdml-engine/search'
 * const p = { random: () => 0, neighbours: (x: number) => [x - 1, x + 1], value: (x: number) => -Math.abs(x - 5), isGoal: (x: number) => x === 5 }
 * localBeamSearch(p, { k: 2 }).state // => 5
 * ```
 */
export function localBeamSearch<S>(
  problem: LocalSearchProblem<S>,
  options: LocalSearchOptions & {
    readonly k?: number
    readonly stochastic?: boolean
    readonly temperature?: number
  } = {},
): LocalSearchResult<S> {
  const { k, stochastic, temperature, maxIterations = 1000, ...run } = options
  const algorithm = localBeamSearchAlgorithm(problem, {
    ...(k === undefined ? {} : { k }),
    ...(stochastic === undefined ? {} : { stochastic }),
    ...(temperature === undefined ? {} : { temperature }),
  })
  const { state, steps } = runAlgorithm(algorithm, { ...run, maxSteps: maxIterations })
  return {
    state: state.best,
    value: state.bestValue,
    status: state.status === 'running' ? 'budget' : state.status,
    iterations: steps,
    evaluations: state.evaluations,
    history: [],
  }
}

/**
 * Late acceptance hill climbing as a traced algorithm (Burke and Bykov 2017): a random neighbour is accepted when it
 * is no worse than the current state or than the state `length` iterations ago, a one-parameter, schedule-free
 * alternative to annealing.
 *
 * @param problem - The problem.
 * @param options - The history length (default 50).
 * @returns The algorithm.
 * @throws MdmlError with code `value` when the length is not a positive integer.
 * @concept local-search
 * @example
 * ```ts
 * import { lateAcceptanceAlgorithm } from 'mdml-engine/search'
 * const p = { initial: 0, neighbours: (x: number) => [x - 1, x + 1], value: (x: number) => -Math.abs(x - 3) }
 * lateAcceptanceAlgorithm(p).name // => 'late-acceptance'
 * ```
 */
export function lateAcceptanceAlgorithm<S>(
  problem: LocalSearchProblem<S>,
  options: { readonly length?: number } = {},
): Algorithm<LocalSearchState<S>> {
  const length = options.length ?? 50
  if (!(Number.isInteger(length) && length >= 1)) {
    throw new MdmlError('value', `lateAcceptance: length ${length} must be a positive integer`)
  }
  return localAlgorithm(
    'late-acceptance',
    { length },
    problem,
    (s) => new Float64Array(length).fill(s.value),
    (s, past, rng) => {
      const next = randomNeighbour(problem, s.current, rng)
      if (next === undefined) {
        s.status = 'local-optimum'
        return
      }
      const v = evaluate(problem, s, next)
      const slot = s.iterations % length
      const accepted = v >= (past[slot] ?? Number.NEGATIVE_INFINITY) || v >= s.value
      s.move = { value: v, accepted }
      if (accepted) moveTo(problem, s, next, v)
      past[slot] = s.value
    },
  )
}

/**
 * Late acceptance hill climbing.
 *
 * @param problem - The problem.
 * @param options - The history length, the iteration budget, a seed or key, and tracing.
 * @returns The best state seen and its value.
 * @concept local-search
 * @example
 * ```ts
 * import { lateAcceptance } from 'mdml-engine/search'
 * const p = { initial: 0, neighbours: (x: number) => [x - 1, x + 1], value: (x: number) => -Math.abs(x - 9), isGoal: (x: number) => x === 9 }
 * lateAcceptance(p, { length: 5, seed: 2 }).state // => 9
 * ```
 */
export function lateAcceptance<S>(
  problem: LocalSearchProblem<S>,
  options: LocalSearchOptions & { readonly length?: number } = {},
): LocalSearchResult<S> {
  const { length, ...rest } = options
  return runLocal(lateAcceptanceAlgorithm(problem, length === undefined ? {} : { length }), rest)
}

/**
 * A local search problem over a {@link Space} (from `mdml-engine/space`): random states are samples, neighbours
 * are the space's one-move neighbourhood, and the value is the given function, so any configuration space can be
 * searched by hill climbing, annealing or tabu search.
 *
 * @param space - The space.
 * @param value - The value of a configuration (maximised).
 * @param options - Grid points per real dimension for neighbours (default 11) and a starting configuration.
 * @returns The problem.
 * @concept local-search
 * @example
 * ```ts
 * import { dict, int } from 'mdml-engine/space'
 * import { hillClimbing, spaceProblem } from 'mdml-engine/search'
 * const p = spaceProblem(dict({ a: int(0, 9), b: int(0, 9) }), (v) => -(((v as { a: number }).a - 2) ** 2) - (((v as { b: number }).b - 7) ** 2), { initial: { a: 0, b: 0 } })
 * hillClimbing(p).state // => { a: 2, b: 7 }
 * ```
 */
export function spaceProblem(
  space: Space,
  value: (v: SpaceValue) => number,
  options: { readonly points?: number; readonly initial?: SpaceValue } = {},
): LocalSearchProblem<SpaceValue> {
  const points = options.points ?? 11
  return {
    ...(options.initial === undefined ? {} : { initial: options.initial }),
    random: (k) => sample(space, k),
    neighbours: (v) => spaceNeighbours(space, v, { points }),
    value,
  }
}

/**
 * The neighbours of a permutation under a move: swapping two positions, reversing a segment (2-opt, the classic
 * travelling-salesman move), or moving one element elsewhere (insertion).
 *
 * @param perm - The permutation.
 * @param move - The move (default `'swap'`).
 * @returns Every neighbour, in a fixed order.
 * @example
 * ```ts
 * import { permutationNeighbours } from 'mdml-engine/search'
 * permutationNeighbours([0, 1, 2]) // => [[1, 0, 2], [2, 1, 0], [0, 2, 1]]
 * permutationNeighbours([0, 1, 2, 3], 'two-opt').length // => 6
 * ```
 */
export function permutationNeighbours(
  perm: readonly number[],
  move: 'swap' | 'two-opt' | 'insert' = 'swap',
): number[][] {
  const n = perm.length
  const out: number[][] = []
  for (let i = 0; i < n; i++) {
    for (let j = i + 1; j < n; j++) {
      if (move === 'swap') {
        const p = [...perm]
        p[i] = perm[j] as number
        p[j] = perm[i] as number
        out.push(p)
      } else if (move === 'two-opt') {
        out.push([...perm.slice(0, i), ...perm.slice(i, j + 1).reverse(), ...perm.slice(j + 1)])
      } else {
        // Move the element at i to position j, and the element at j to position i.
        const a = [...perm]
        const [x] = a.splice(i, 1)
        a.splice(j, 0, x as number)
        out.push(a)
        if (j > i + 1) {
          const b = [...perm]
          const [y] = b.splice(j, 1)
          b.splice(i, 0, y as number)
          out.push(b)
        }
      }
    }
  }
  return out
}

/**
 * The neighbours of a bit string: every single-bit flip.
 *
 * @param bits - The bits (0 or 1).
 * @returns One neighbour per position.
 * @example
 * ```ts
 * import { bitFlipNeighbours } from 'mdml-engine/search'
 * bitFlipNeighbours([0, 1]) // => [[1, 1], [0, 0]]
 * ```
 */
export function bitFlipNeighbours(bits: readonly number[]): number[][] {
  return bits.map((_, i) => bits.map((b, j) => (i === j ? 1 - b : b)))
}
