/**
 * Learning in normal-form games, where players repeatedly adapt to each other's past play rather than computing an
 * equilibrium directly. Fictitious play (best responding to the others' empirical frequencies), its stochastic variant
 * with perturbed utilities as in nashpy, and best-response dynamics, each as a traced algorithm.
 */

import { maxOf, OmaiError } from '../base/index.ts'
import { uniformStream } from '../random/index.ts'
import { type Algorithm, channels, runAlgorithm, type TensorChannel, type TraceSchema } from '../trace/index.ts'
import { actionValues, type GameRunOptions, type NormalFormGame, nashConv } from './normal-form/index.ts'

/** Options shared by the fictitious-play variants. */
export interface FictitiousPlayOptions extends GameRunOptions {
  /** Rounds of play (default 1000). */
  readonly iterations?: number
  /** Starting play counts of every player (default zeros, as nashpy). */
  readonly initialCounts?: readonly (readonly number[])[]
  /** Brown's alternating version: players respond in turn, each to the others' updated counts (default false). */
  readonly alternating?: boolean
  /** Ties between best responses: the first action (default) or a uniformly random one (nashpy's rule). */
  readonly tieBreak?: 'first' | 'random'
}

/** The state of fictitious play. */
export interface FictitiousPlayState {
  readonly t: number
  /** Every player's play counts. */
  readonly counts: readonly (readonly number[])[]
  /** The actions played in each round so far. */
  readonly plays: readonly (readonly number[])[]
}

/** The outcome of fictitious play. */
export interface FictitiousPlayResult extends FictitiousPlayState {
  /** The empirical mixed strategies (counts normalised). */
  readonly strategies: readonly (readonly number[])[]
  /** The NashConv of the empirical strategies. */
  readonly nashConv: number
}

/** Counts normalised to a mixed strategy (uniform when all are zero). */
function normalised(counts: readonly number[]): number[] {
  let s = 0
  for (const c of counts) s += c
  return s > 0 ? counts.map((c) => c / s) : counts.map(() => 1 / counts.length)
}

function playerTensors(game: NormalFormGame, prefix: string, description: string): Record<string, TensorChannel> {
  const out: Record<string, TensorChannel> = {}
  for (const [p, k] of game.shape.entries())
    out[`${prefix}${p}`] = { role: 'state', description: `${description} of player ${p}`, shape: [k] }
  return out
}

function pick(values: readonly number[], tieBreak: 'first' | 'random', next: () => number): number {
  const best = maxOf(values)
  const ties: number[] = []
  for (let a = 0; a < values.length; a++) if ((values[a] ?? -Infinity) === best) ties.push(a)
  if (tieBreak === 'first' || ties.length === 1) return ties[0] ?? 0
  return ties[Math.min(ties.length - 1, Math.floor(next() * ties.length))] ?? 0
}

function startCounts(game: NormalFormGame, initial: FictitiousPlayOptions['initialCounts']): number[][] {
  if (
    initial !== undefined &&
    (initial.length !== game.shape.length || initial.some((c, p) => c.length !== game.shape[p]))
  )
    throw new OmaiError('shape', `${game.name}: one initial count per action of every player`)
  return game.shape.map((k, p) => (initial?.[p] === undefined ? new Array<number>(k).fill(0) : [...(initial[p] ?? [])]))
}

/**
 * Fictitious play (Brown 1951) as a traced algorithm: every round each player best responds to the empirical play
 * counts of the others (simultaneously, as nashpy's `fictitious_play`, or in turn with `alternating`). The empirical
 * strategies converge to equilibrium in two-player zero-sum games, 2 x n games and potential games, but can cycle
 * (Shapley's game).
 *
 * @param game - The game.
 * @param options - Initial counts, alternation and tie-breaking.
 * @returns The algorithm; each step is one round.
 * @concept fictitious-play
 * @example
 * ```ts
 * import { fictitiousPlayAlgorithm, matchingPennies } from 'omai-engine/games'
 * import { runAlgorithm } from 'omai-engine/trace'
 * runAlgorithm(fictitiousPlayAlgorithm(matchingPennies()), { maxSteps: 4 }).state.plays // => [[0, 0], [0, 1], [0, 1], [1, 1]]
 * ```
 */
export function fictitiousPlayAlgorithm(
  game: NormalFormGame,
  options: Omit<FictitiousPlayOptions, 'iterations'> = {},
): Algorithm<FictitiousPlayState> {
  const tieBreak = options.tieBreak ?? 'first'
  const alternating = options.alternating ?? false
  const counts0 = startCounts(game, options.initialCounts)
  const schema: TraceSchema = channels({
    metrics: { nashConv: { description: 'NashConv of the empirical strategies', direction: 'lower' } },
    tensors: playerTensors(game, 'strategy', 'the empirical strategy'),
    messages: { plays: { description: 'the actions played this round' } },
  })
  return {
    name: 'fictitious-play',
    config: { alternating, tieBreak },
    schema,
    init: () => ({ t: 0, counts: counts0, plays: [] }),
    step: (s, ctx) => {
      const next = uniformStream(ctx.key)
      const counts = s.counts.map((c) => [...c])
      const play = new Array<number>(game.shape.length).fill(0)
      for (let p = 0; p < game.shape.length; p++) {
        const a = pick(actionValues(game, p, alternating ? counts : s.counts), tieBreak, next)
        play[p] = a
        if (alternating) counts[p] = (counts[p] ?? []).map((c, b) => c + (b === a ? 1 : 0))
      }
      if (!alternating)
        for (const [p, a] of play.entries()) counts[p] = (counts[p] ?? []).map((c, b) => c + (b === a ? 1 : 0))
      return { t: s.t + 1, counts, plays: [...s.plays, play] }
    },
    observe: (s) => {
      const strategies = s.counts.map(normalised)
      const tensors: Record<string, { role: 'state'; value: number[] }> = {}
      for (const [p, x] of strategies.entries()) tensors[`strategy${p}`] = { role: 'state', value: x }
      const last = s.plays[s.plays.length - 1]
      return {
        metrics: { nashConv: nashConv(game, strategies) },
        tensors,
        ...(last === undefined ? {} : { messages: [{ kind: 'plays', payload: [...last] }] }),
      }
    },
  }
}

function finish(game: NormalFormGame, state: FictitiousPlayState): FictitiousPlayResult {
  const strategies = state.counts.map(normalised)
  return { ...state, strategies, nashConv: nashConv(game, strategies) }
}

/**
 * Run fictitious play for a number of rounds (see {@link fictitiousPlayAlgorithm}).
 *
 * @param game - The game.
 * @param options - Rounds, initial counts, alternation, tie-breaking and tracing.
 * @returns The counts, the plays of every round, the empirical strategies and their NashConv.
 * @concept fictitious-play
 * @example
 * ```ts
 * import { fictitiousPlay, rockPaperScissors } from 'omai-engine/games'
 * const r = fictitiousPlay(rockPaperScissors(), { iterations: 3000 })
 * r.nashConv < 0.05 // => true
 * ```
 */
export function fictitiousPlay(game: NormalFormGame, options: FictitiousPlayOptions = {}): FictitiousPlayResult {
  const { iterations = 1000, ...rest } = options
  const { state } = runAlgorithm(fictitiousPlayAlgorithm(game, rest), { ...rest, maxSteps: iterations })
  return finish(game, state)
}

/** Options for {@link stochasticFictitiousPlay}. */
export interface StochasticFictitiousPlayOptions extends Omit<FictitiousPlayOptions, 'alternating' | 'tieBreak'> {
  /** The logit temperature (default 0.1, as nashpy). */
  readonly eta?: number
  /** The scale of the uniform noise added to the utilities (default 0.01, as nashpy). */
  readonly epsilonBar?: number
}

/**
 * Stochastic fictitious play (nashpy's `stochastic_fictitious_play`, Fudenberg and Levine 1998): each round every
 * player perturbs its utilities against the others' empirical strategies with uniform noise in `[0, epsilonBar)`,
 * then samples its action from the logit choice `softmax(utilities / eta)`. Randomness comes from the run's key.
 *
 * @param game - The game.
 * @param options - Rounds, temperature, noise, initial counts and tracing.
 * @returns The algorithm; each step is one round.
 * @concept fictitious-play
 * @example
 * ```ts
 * import { matchingPennies, stochasticFictitiousPlayAlgorithm } from 'omai-engine/games'
 * import { runAlgorithm } from 'omai-engine/trace'
 * runAlgorithm(stochasticFictitiousPlayAlgorithm(matchingPennies()), { maxSteps: 50, seed: 1 }).state.t // => 50
 * ```
 */
export function stochasticFictitiousPlayAlgorithm(
  game: NormalFormGame,
  options: Omit<StochasticFictitiousPlayOptions, 'iterations'> = {},
): Algorithm<FictitiousPlayState> {
  const eta = options.eta ?? 0.1
  const epsilonBar = options.epsilonBar ?? 0.01
  if (!(eta > 0)) throw new OmaiError('value', 'stochasticFictitiousPlay: eta must be positive')
  const base = fictitiousPlayAlgorithm(
    game,
    options.initialCounts === undefined ? {} : { initialCounts: options.initialCounts },
  )
  return {
    ...base,
    name: 'stochastic-fictitious-play',
    config: { eta, epsilonBar },
    step: (s, ctx) => {
      const next = uniformStream(ctx.key)
      const strategies = s.counts.map(normalised)
      const play = game.shape.map((_, p) => {
        const noisy = actionValues(game, p, strategies).map((u) => (u + next() * epsilonBar) / eta)
        const top = maxOf(noisy)
        const w = noisy.map((v) => Math.exp(v - top))
        let total = 0
        for (const v of w) total += v
        let u = next() * total
        for (let a = 0; a < w.length; a++) {
          u -= w[a] ?? 0
          if (u < 0) return a
        }
        return w.length - 1
      })
      const counts = s.counts.map((c, p) => c.map((v, b) => v + (b === play[p] ? 1 : 0)))
      return { t: s.t + 1, counts, plays: [...s.plays, play] }
    },
  }
}

/**
 * Run stochastic fictitious play (see {@link stochasticFictitiousPlayAlgorithm}).
 *
 * @param game - The game.
 * @param options - Rounds, temperature, noise, initial counts, seed or key, and tracing.
 * @returns The counts, plays, empirical strategies and their NashConv.
 * @concept fictitious-play
 * @example
 * ```ts
 * import { matchingPennies, stochasticFictitiousPlay } from 'omai-engine/games'
 * stochasticFictitiousPlay(matchingPennies(), { iterations: 2000, seed: 3 }).nashConv < 0.1 // => true
 * ```
 */
export function stochasticFictitiousPlay(
  game: NormalFormGame,
  options: StochasticFictitiousPlayOptions = {},
): FictitiousPlayResult {
  const { iterations = 1000, ...rest } = options
  const { state } = runAlgorithm(stochasticFictitiousPlayAlgorithm(game, rest), { ...rest, maxSteps: iterations })
  return finish(game, state)
}

/** Options for {@link bestResponseDynamics}. */
export interface BestResponseDynamicsOptions extends GameRunOptions {
  /** The starting pure profile (default every player's action 0). */
  readonly start?: readonly number[]
  /** Stop after this many switches (default 10000). */
  readonly maxIterations?: number
}

/** The state of best-response dynamics. */
export interface BestResponseDynamicsState {
  readonly profile: readonly number[]
  /** The profiles visited, the start first. */
  readonly path: readonly (readonly number[])[]
  /** `'equilibrium'` at a pure Nash equilibrium, `'cycle'` when a profile repeats. */
  readonly status: 'running' | 'equilibrium' | 'cycle'
}

/**
 * Best-response dynamics as a traced algorithm: in each step the first player (in order) that can improve switches to
 * its first best response. It reaches a pure Nash equilibrium in every finite potential game (congestion games, pure
 * coordination); elsewhere it can cycle, which is detected when a profile repeats.
 *
 * @param game - The game.
 * @param options - The starting profile.
 * @returns The algorithm.
 * @concept best-response
 * @example
 * ```ts
 * import { bestResponseDynamicsAlgorithm, prisonersDilemma } from 'omai-engine/games'
 * import { runAlgorithm } from 'omai-engine/trace'
 * runAlgorithm(bestResponseDynamicsAlgorithm(prisonersDilemma()), { maxSteps: 10 }).state.path // => [[0, 0], [1, 0], [1, 1]]
 * ```
 */
export function bestResponseDynamicsAlgorithm(
  game: NormalFormGame,
  options: Pick<BestResponseDynamicsOptions, 'start'> = {},
): Algorithm<BestResponseDynamicsState> {
  const start = [...(options.start ?? game.shape.map(() => 0))]
  if (start.length !== game.shape.length || start.some((a, p) => a < 0 || a >= (game.shape[p] ?? 0)))
    throw new OmaiError('value', `${game.name}: the start must be one valid action per player`)
  const pure = (profile: readonly number[]): number[][] =>
    profile.map((a, p) => Array.from({ length: game.shape[p] ?? 0 }, (_, b) => (b === a ? 1 : 0)))
  const improve = (profile: readonly number[]): number[] | undefined => {
    for (let p = 0; p < profile.length; p++) {
      const v = actionValues(game, p, pure(profile))
      const best = maxOf(v)
      if ((v[profile[p] ?? 0] ?? 0) < best) {
        const next = [...profile]
        next[p] = v.indexOf(best)
        return next
      }
    }
    return undefined
  }
  return {
    name: 'best-response-dynamics',
    config: { start },
    schema: channels({ messages: { switch: { description: 'the profile after a unilateral improvement' } } }),
    init: () => ({ profile: start, path: [start], status: improve(start) === undefined ? 'equilibrium' : 'running' }),
    step: (s) => {
      const next = improve(s.profile)
      if (next === undefined) return { ...s, status: 'equilibrium' }
      const seen = s.path.some((q) => q.every((a, p) => a === next[p]))
      const path = [...s.path, next]
      const status = improve(next) === undefined ? 'equilibrium' : seen ? 'cycle' : 'running'
      return { profile: next, path, status }
    },
    done: (s) => (s.status === 'equilibrium' ? 'converged' : s.status === 'cycle' ? 'stopped' : undefined),
    observe: (s) => ({ messages: [{ kind: 'switch', payload: [...s.profile] }] }),
  }
}

/**
 * Run best-response dynamics (see {@link bestResponseDynamicsAlgorithm}).
 *
 * @param game - The game.
 * @param options - The starting profile, a cap on switches and tracing.
 * @returns The final state: the profile, the path and whether it stopped at an equilibrium or a cycle.
 * @concept best-response
 * @example
 * ```ts
 * import { bestResponseDynamics, matchingPennies } from 'omai-engine/games'
 * bestResponseDynamics(matchingPennies()).status // => 'cycle'
 * ```
 */
export function bestResponseDynamics(
  game: NormalFormGame,
  options: BestResponseDynamicsOptions = {},
): BestResponseDynamicsState {
  const { maxIterations = 10_000, start, ...rest } = options
  return runAlgorithm(bestResponseDynamicsAlgorithm(game, start === undefined ? {} : { start }), {
    ...rest,
    maxSteps: maxIterations,
  }).state
}
