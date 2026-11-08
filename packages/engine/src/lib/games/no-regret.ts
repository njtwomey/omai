/**
 * No-regret learning in normal-form games, whose average play converges to coarse correlated equilibria (and to Nash
 * equilibria in two-player zero-sum games). Regret matching (Hart and Mas-Colell), as in OpenSpiel, and multiplicative
 * weights (Hedge) in self-play, as traced algorithms.
 */

import { maxOf, OmaiError } from '../base/index.ts'
import { type Algorithm, channels, runAlgorithm, type TensorChannel, type TraceSchema } from '../trace/index.ts'
import {
  actionValues,
  correlatedGains,
  type GameRunOptions,
  jointDistribution,
  type NormalFormGame,
  nashConv,
} from './normal-form/index.ts'

/** The state of a no-regret learning run in a normal-form game. */
export interface NoRegretState {
  readonly t: number
  /** Every player's current mixed strategy. */
  readonly strategies: readonly (readonly number[])[]
  /** Every player's cumulative regrets (regret matching) or cumulative action values (Hedge). */
  readonly regrets: readonly (readonly number[])[]
  /** The (weighted) sum of every player's strategies, and the total weight. */
  readonly sums: readonly (readonly number[])[]
  readonly weight: number
  /** The (weighted) sum of the joint profile distributions played. */
  readonly joint: readonly number[]
}

/** The outcome of {@link regretMatching} and {@link multiplicativeWeights}. */
export interface NoRegretResult {
  readonly iterations: number
  /** The average strategies, which converge to a Nash equilibrium in two-player zero-sum games. */
  readonly average: readonly (readonly number[])[]
  /** The last strategies. */
  readonly current: readonly (readonly number[])[]
  /** The average joint distribution, which converges to the coarse correlated equilibria. */
  readonly joint: readonly number[]
  /** NashConv of the average strategies. */
  readonly nashConv: number
  /** The coarse-correlated gap of the average joint distribution, per player. */
  readonly externalRegret: readonly number[]
}

function schemaFor(game: NormalFormGame): TraceSchema {
  const tensors: Record<string, TensorChannel> = {}
  for (const [p, k] of game.shape.entries()) {
    tensors[`average${p}`] = { role: 'state', description: `the average strategy of player ${p}`, shape: [k] }
    tensors[`strategy${p}`] = { role: 'state', description: `the current strategy of player ${p}`, shape: [k] }
    tensors[`regret${p}`] = { role: 'state', description: `the cumulative regrets of player ${p}`, shape: [k] }
  }
  return channels({
    metrics: {
      nashConv: { description: 'NashConv of the average strategies', direction: 'lower' },
      externalRegret: { description: 'largest average external regret of the joint play', direction: 'lower' },
    },
    tensors,
  })
}

function averages(s: NoRegretState): number[][] {
  return s.sums.map((x) => x.map((v) => v / (s.weight > 0 ? s.weight : 1)))
}

function observeState(
  game: NormalFormGame,
  s: NoRegretState,
): ReturnType<NonNullable<Algorithm<NoRegretState>['observe']>> {
  const avg = averages(s)
  const tensors: Record<string, { role: 'state'; value: readonly number[] }> = {}
  for (const [p, x] of avg.entries()) {
    tensors[`average${p}`] = { role: 'state', value: x }
    tensors[`strategy${p}`] = { role: 'state', value: s.strategies[p] ?? [] }
    tensors[`regret${p}`] = { role: 'state', value: s.regrets[p] ?? [] }
  }
  const joint = s.joint.map((v) => v / (s.weight > 0 ? s.weight : 1))
  return {
    metrics: { nashConv: nashConv(game, avg), externalRegret: maxOf(correlatedGains(game, joint, { coarse: true })) },
    tensors,
  }
}

function accumulate(
  game: NormalFormGame,
  s: Pick<NoRegretState, 'sums' | 'joint' | 'weight'>,
  strategies: readonly (readonly number[])[],
  w: number,
): Pick<NoRegretState, 'sums' | 'joint' | 'weight'> {
  const j = jointDistribution(game, strategies)
  return {
    sums: s.sums.map((x, p) => x.map((v, a) => v + w * (strategies[p]?.[a] ?? 0))),
    joint: s.joint.map((v, k) => v + w * (j[k] ?? 0)),
    weight: s.weight + w,
  }
}

function initialState(game: NormalFormGame, strategies: readonly (readonly number[])[], regret: number): NoRegretState {
  const zero = {
    sums: game.shape.map((k) => new Array<number>(k).fill(0)),
    joint: jointDistribution(game, strategies).map(() => 0),
    weight: 0,
  }
  return {
    t: 0,
    strategies,
    regrets: game.shape.map((k) => new Array<number>(k).fill(regret)),
    ...accumulate(game, zero, strategies, 1),
  }
}

function startStrategies(game: NormalFormGame, initial: readonly (readonly number[])[] | undefined): number[][] {
  if (
    initial !== undefined &&
    (initial.length !== game.shape.length || initial.some((x, p) => x.length !== game.shape[p]))
  )
    throw new OmaiError('shape', `${game.name}: one initial strategy per player, one probability per action`)
  return game.shape.map((k, p) =>
    initial?.[p] === undefined ? new Array<number>(k).fill(1 / k) : [...(initial[p] ?? [])],
  )
}

function result(game: NormalFormGame, s: NoRegretState): NoRegretResult {
  const average = averages(s)
  const joint = s.joint.map((v) => v / s.weight)
  return {
    iterations: s.t,
    average,
    current: s.strategies,
    joint,
    nashConv: nashConv(game, average),
    externalRegret: correlatedGains(game, joint, { coarse: true }),
  }
}

/** Options for {@link regretMatching}. */
export interface RegretMatchingOptions extends GameRunOptions {
  /** Iterations (default 1000). */
  readonly iterations?: number
  /** Starting strategies (default uniform). */
  readonly initialStrategies?: readonly (readonly number[])[]
  /** The starting cumulative regret of every action (default 0; OpenSpiel uses 1e-6). */
  readonly initialRegret?: number
  /** Mixing weight of the uniform strategy into every update (default 0; OpenSpiel uses 1e-6). */
  readonly gamma?: number
  /** Regret matching+ (Tammelin 2014): clip cumulative regrets at zero after every update (default false). */
  readonly plus?: boolean
  /** Weight iteration t by t in the averages (linear averaging, as CFR+; default false). */
  readonly linearAveraging?: boolean
}

/**
 * Regret matching (Hart and Mas-Colell 2000) in a normal-form game with simultaneous expected updates, as OpenSpiel's
 * `regret_matching`: every iteration each player adds to its cumulative regrets the advantage of each action over its
 * current strategy, and plays proportionally to the positive regrets (uniform when none is positive), mixed with
 * `gamma` of the uniform strategy. The average strategies, starting with the initial ones, converge to the coarse
 * correlated equilibria, and to Nash equilibria in two-player zero-sum games.
 *
 * @param game - The game.
 * @param options - Initial strategies and regrets, exploration, the plus variant and averaging.
 * @returns The algorithm; each step is one iteration.
 * @concept regret-matching
 * @example
 * ```ts
 * import { regretMatchingAlgorithm, rockPaperScissors } from 'omai-engine/games'
 * import { runAlgorithm } from 'omai-engine/trace'
 * runAlgorithm(regretMatchingAlgorithm(rockPaperScissors()), { maxSteps: 5 }).state.t // => 5
 * ```
 */
export function regretMatchingAlgorithm(
  game: NormalFormGame,
  options: Omit<RegretMatchingOptions, 'iterations'> = {},
): Algorithm<NoRegretState> {
  const gamma = options.gamma ?? 0
  const plus = options.plus ?? false
  const linear = options.linearAveraging ?? false
  const start = startStrategies(game, options.initialStrategies)
  const initialRegret = options.initialRegret ?? 0
  return {
    name: 'regret-matching',
    config: { gamma, plus, linearAveraging: linear, initialRegret },
    schema: schemaFor(game),
    init: () => initialState(game, start, initialRegret),
    step: (s) => {
      const regrets: number[][] = []
      const strategies: number[][] = []
      for (const [p, x] of s.strategies.entries()) {
        const v = actionValues(game, p, s.strategies)
        let mean = 0
        for (const [a, q] of x.entries()) mean += q * (v[a] ?? 0)
        const r = (s.regrets[p] ?? []).map((g, a) => {
          const next = g + (v[a] ?? 0) - mean
          return plus ? Math.max(0, next) : next
        })
        regrets.push(r)
        let total = 0
        for (const g of r) if (g > 0) total += g
        const k = r.length
        strategies.push(
          total > 0 ? r.map((g) => gamma / k + (1 - gamma) * (Math.max(0, g) / total)) : r.map(() => 1 / k),
        )
      }
      const t = s.t + 1
      return { t, strategies, regrets, ...accumulate(game, s, strategies, linear ? t : 1) }
    },
    observe: (s) => observeState(game, s),
  }
}

/**
 * Run regret matching (see {@link regretMatchingAlgorithm}).
 *
 * @param game - The game.
 * @param options - Iterations, initial strategies and regrets, exploration, the plus variant, averaging and tracing.
 * @returns The average and last strategies, the average joint play, its NashConv and external regrets.
 * @concept regret-matching
 * @example
 * ```ts
 * import { matchingPennies, regretMatching } from 'omai-engine/games'
 * regretMatching(matchingPennies(), { iterations: 2000 }).nashConv < 0.01 // => true
 * ```
 */
export function regretMatching(game: NormalFormGame, options: RegretMatchingOptions = {}): NoRegretResult {
  const { iterations = 1000, ...rest } = options
  return result(game, runAlgorithm(regretMatchingAlgorithm(game, rest), { ...rest, maxSteps: iterations }).state)
}

/** Options for {@link multiplicativeWeights}. */
export interface MultiplicativeWeightsOptions extends GameRunOptions {
  /** Iterations (default 1000). */
  readonly iterations?: number
  /** The step size eta (default `sqrt(8 ln k / iterations)` per player, the Hedge rate for payoffs in [0, 1]). */
  readonly learningRate?: number
  /** Starting strategies (default uniform). */
  readonly initialStrategies?: readonly (readonly number[])[]
}

/**
 * Multiplicative weights (Hedge, Freund and Schapire 1997) in self-play: every iteration each player observes the
 * expected value of each of its actions against the others' current strategies and plays proportionally to
 * `exp(eta * cumulative value)`. Each player's external regret grows as `O(sqrt(T log k))`, so the average joint
 * play converges to the coarse correlated equilibria, and the averages to a Nash equilibrium in zero-sum games.
 *
 * @param game - The game.
 * @param options - The learning rate and initial strategies (the default rate needs `iterations`).
 * @returns The algorithm; each step is one iteration.
 * @concept multiplicative-weights
 * @example
 * ```ts
 * import { multiplicativeWeightsAlgorithm, rockPaperScissors } from 'omai-engine/games'
 * import { runAlgorithm } from 'omai-engine/trace'
 * runAlgorithm(multiplicativeWeightsAlgorithm(rockPaperScissors()), { maxSteps: 3 }).state.t // => 3
 * ```
 */
export function multiplicativeWeightsAlgorithm(
  game: NormalFormGame,
  options: MultiplicativeWeightsOptions = {},
): Algorithm<NoRegretState> {
  const horizon = options.iterations ?? 1000
  const rates = game.shape.map((k) => options.learningRate ?? Math.sqrt((8 * Math.log(Math.max(2, k))) / horizon))
  const start = startStrategies(game, options.initialStrategies)
  // Cumulative values start at log(x0) / eta, so that the initial strategies are the softmax of them.
  const logs = start.map((x, p) => x.map((q) => Math.log(Math.max(q, 1e-300)) / (rates[p] ?? 1)))
  return {
    name: 'multiplicative-weights',
    config: { learningRate: options.learningRate ?? null, horizon },
    schema: schemaFor(game),
    init: () => ({ ...initialState(game, start, 0), regrets: logs }),
    step: (s) => {
      const regrets = s.strategies.map((_, p) => {
        const v = actionValues(game, p, s.strategies)
        return (s.regrets[p] ?? []).map((g, a) => g + (v[a] ?? 0))
      })
      const strategies = regrets.map((g, p) => {
        const z = g.map((v) => v * (rates[p] ?? 1))
        const top = maxOf(z)
        const w = z.map((v) => Math.exp(v - top))
        let total = 0
        for (const v of w) total += v
        return w.map((v) => v / total)
      })
      return { t: s.t + 1, strategies, regrets, ...accumulate(game, s, strategies, 1) }
    },
    observe: (s) => observeState(game, s),
  }
}

/**
 * Run multiplicative weights in self-play (see {@link multiplicativeWeightsAlgorithm}).
 *
 * @param game - The game.
 * @param options - Iterations, learning rate, initial strategies and tracing.
 * @returns The average and last strategies, the average joint play, its NashConv and external regrets.
 * @concept multiplicative-weights
 * @example
 * ```ts
 * import { multiplicativeWeights, rockPaperScissors } from 'omai-engine/games'
 * multiplicativeWeights(rockPaperScissors(), { iterations: 2000 }).nashConv < 0.05 // => true
 * ```
 */
export function multiplicativeWeights(
  game: NormalFormGame,
  options: MultiplicativeWeightsOptions = {},
): NoRegretResult {
  const iterations = options.iterations ?? 1000
  return result(
    game,
    runAlgorithm(multiplicativeWeightsAlgorithm(game, { ...options, iterations }), { ...options, maxSteps: iterations })
      .state,
  )
}
