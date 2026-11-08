/**
 * Goal-conditioned reinforcement learning with hindsight experience replay (Andrychowicz et al. 2017). In a task with a
 * sparse reward for reaching a goal, almost every episode fails and teaches nothing; hindsight relabels a failed episode
 * with a goal it did reach, so that the same transitions become successes for that goal. This module holds the
 * relabelling (the final, future and episode strategies) and a goal-conditioned Q-learning agent over linear features
 * that replays every episode's transitions with their relabelled goals.
 */

import { type Json, maxOf, OmaiError } from '../base/index.ts'
import { foldIn, type Key } from '../random/index.ts'
import type { Agent } from '../rl/index.ts'
import type { Space } from '../space/index.ts'
import { argmaxRandom, type Rate, rateAt, unit } from './_tabular.ts'
import { type ActionFeatureMap, featureValue, type SparseFeatures } from './approximation.ts'

/** How a failed episode's goal is replaced. */
export type HerStrategy = 'final' | 'future' | 'episode' | 'none'

/** One transition of a goal-conditioned task: the observation carries the goal. */
export interface GoalStep {
  readonly observation: readonly number[]
  readonly action: number
  readonly reward: number
  readonly nextObservation: readonly number[]
  readonly terminated: boolean
}

/** How goals sit in an observation and what reaching one pays. */
export interface GoalSpec {
  /** The goal the observation has achieved (default: the first half of the observation). */
  readonly achieved?: (observation: readonly number[]) => readonly number[]
  /** The goal being pursued (default: the second half of the observation). */
  readonly desired?: (observation: readonly number[]) => readonly number[]
  /** The observation with its goal replaced (default: the second half replaced). */
  readonly withGoal?: (observation: readonly number[], goal: readonly number[]) => readonly number[]
  /** The reward for having achieved a goal when the pursued one was another (default 0 within `tolerance`, else -1). */
  readonly reward?: (achieved: readonly number[], desired: readonly number[]) => number
  /** The distance that counts as reaching a goal, for the default reward (default 0.075). */
  readonly tolerance?: number
}

const resolved = (spec: GoalSpec): Required<GoalSpec> => {
  const half = (o: readonly number[]): number => Math.floor(o.length / 2)
  const tolerance = spec.tolerance ?? 0.075
  return {
    achieved: spec.achieved ?? ((o) => o.slice(0, half(o))),
    desired: spec.desired ?? ((o) => o.slice(half(o))),
    withGoal: spec.withGoal ?? ((o, g) => [...o.slice(0, half(o)), ...g]),
    reward: spec.reward ?? ((a, d) => (Math.hypot(...a.map((x, i) => x - (d[i] ?? 0))) <= tolerance ? 0 : -1)),
    tolerance,
  }
}

/** Options of {@link relabelEpisode}. */
export interface RelabelOptions extends GoalSpec {
  /** `final` relabels with the episode's last achieved goal, `future` with goals achieved at or after the step, `episode` with any. */
  readonly strategy: Exclude<HerStrategy, 'none'>
  /** The relabelled goals per step for `future` and `episode` (default 4); `final` gives one. */
  readonly k?: number
}

/**
 * Hindsight relabelling of an episode: for every step, the same transition with another goal in its observations and the
 * reward that goal would have paid, ending the episode (terminated) where the goal is reached. The goals are the
 * achieved goals of the episode: the last (`final`), `k` of those from the step on (`future`, the best strategy of the
 * paper), or `k` of any step's (`episode`).
 *
 * @param steps - The episode's transitions, in order.
 * @param options - The strategy, `k`, and how goals sit in observations.
 * @param key - The key that draws the goals (unused by `final`).
 * @returns The relabelled transitions, step by step.
 * @throws OmaiError with code `value` for an empty episode.
 * @concept hindsight-experience-replay
 * @example
 * ```ts
 * import { relabelEpisode } from 'omai-engine/agents'
 * import { key } from 'omai-engine/random'
 * const steps = [
 *   { observation: [0, 0, 1, 1], action: 0, reward: -1, nextObservation: [0.1, 0, 1, 1], terminated: false },
 *   { observation: [0.1, 0, 1, 1], action: 0, reward: -1, nextObservation: [0.2, 0, 1, 1], terminated: false },
 * ]
 * relabelEpisode(steps, { strategy: 'final' }, key(0)).map((s) => [s.observation[2], s.reward]) // => [[0.2, -1], [0.2, 0]]
 * ```
 */
export function relabelEpisode(steps: readonly GoalStep[], options: RelabelOptions, key: Key): GoalStep[] {
  if (steps.length === 0) throw new OmaiError('value', 'relabelEpisode: the episode is empty')
  const g = resolved(options)
  const k = options.strategy === 'final' ? 1 : (options.k ?? 4)
  const reached = steps.map((s) => g.achieved(s.nextObservation))
  const out: GoalStep[] = []
  for (const [t, s] of steps.entries()) {
    for (let j = 0; j < k; j++) {
      let goal: readonly number[]
      if (options.strategy === 'final') goal = reached[reached.length - 1] ?? []
      else {
        const lo = options.strategy === 'future' ? t : 0
        const u = unit(foldIn(foldIn(key, t), j))
        goal = reached[lo + Math.min(steps.length - lo - 1, Math.floor(u * (steps.length - lo)))] ?? []
      }
      const reward = g.reward(reached[t] ?? [], goal)
      out.push({
        observation: g.withGoal(s.observation, goal),
        action: s.action,
        reward,
        nextObservation: g.withGoal(s.nextObservation, goal),
        terminated: reward >= 0,
      })
    }
  }
  return out
}

/** Options of {@link goalConditionedQ}. */
export interface GoalConditionedQOptions extends GoalSpec {
  /** The features of an observation (with its goal) and an action. */
  readonly features: ActionFeatureMap<readonly number[]>
  /** The number of actions (default from a discrete action space). */
  readonly nActions?: number
  /** The step size (default 0.05; with tile coding divide by the number of tilings). */
  readonly alpha?: number
  /** The discount (default 0.95). */
  readonly gamma?: number
  /** The exploration rate (default 0.2). */
  readonly epsilon?: Rate
  /** Hindsight relabelling at the end of each episode: the strategy (default `future`) and `k` (default 4). */
  readonly her?: { readonly strategy?: HerStrategy; readonly k?: number }
  /** Passes of Q-learning over the relabelled transitions of an episode (default 1). */
  readonly replayPasses?: number
}

/** The state of {@link goalConditionedQ}. */
export interface GoalConditionedState {
  readonly w: Float64Array
  /** The transitions of the episode in progress. */
  readonly episode: readonly GoalStep[]
  readonly episodes: number
  readonly steps: number
  /** Episodes that reached their goal, and a moving average of success (rate 0.02). */
  readonly successes: number
  readonly successRate: number
  /** The relabelled transitions learnt from so far. */
  readonly relabelled: number
  readonly tdError: number
}

/**
 * Goal-conditioned Q-learning over linear features, with hindsight experience replay: a universal value function
 * `q(s, g, a) = w . x(s, g, a)` that takes the goal as part of its observation, an epsilon-greedy policy, one Q-learning
 * update per real transition and, when an episode ends, the same updates over its relabelled transitions
 * ({@link relabelEpisode}), so that the sparse reward is met in every episode. With `her.strategy` `none` it is plain
 * goal-conditioned Q-learning, which on a task where the goal is rarely reached learns almost nothing.
 *
 * @param env - The environment (its action space).
 * @param options - The features, step size, discount, exploration, relabelling and goal layout.
 * @returns The agent; its `stats` report the success rate and the relabelled transitions learnt from.
 * @concept hindsight-experience-replay
 * @example
 * ```ts
 * import { actionTileFeatures, goalConditionedQ, tileCoder } from 'omai-engine/agents'
 * import { reachGoal } from 'omai-engine/environments'
 * import { key } from 'omai-engine/random'
 * const coder = tileCoder({ low: [-1, -1], high: [1, 1], tilings: 4, tiles: 10 })
 * const relative = (o: readonly number[]): number[] => [(o[2] ?? 0) - (o[0] ?? 0), (o[3] ?? 0) - (o[1] ?? 0)]
 * const agent = goalConditionedQ(reachGoal.create(), { features: actionTileFeatures(coder, 4, relative) })
 * agent.init(key(0)).w.length // => 1936
 * ```
 */
export function goalConditionedQ(
  env: { readonly actionSpace: Space },
  options: GoalConditionedQOptions,
): Agent<GoalConditionedState, readonly number[], number> {
  const n = options.nActions ?? (env.actionSpace.kind === 'discrete' ? env.actionSpace.n : undefined)
  if (n === undefined) throw new OmaiError('value', 'goalConditionedQ: needs a discrete action space or nActions')
  const alpha = options.alpha ?? 0.05
  const gamma = options.gamma ?? 0.95
  const epsilon = options.epsilon ?? 0.2
  const strategy = options.her?.strategy ?? 'future'
  const k = options.her?.k ?? 4
  const passes = options.replayPasses ?? 1
  const goal = resolved(options)
  const qs = (w: ArrayLike<number>, o: readonly number[]): Float64Array =>
    Float64Array.from({ length: n }, (_, a) => featureValue(w, options.features.features(o, a)))
  // One Q-learning update on a transition, in place on a copy of the weights; returns the TD error.
  const update = (w: Float64Array, s: GoalStep): number => {
    const phi: SparseFeatures = options.features.features(s.observation, s.action)
    const next = s.terminated ? 0 : maxOf(qs(w, s.nextObservation))
    const delta = s.reward + gamma * next - featureValue(w, phi)
    for (const [j, i] of phi.indices.entries()) w[i] = (w[i] ?? 0) + alpha * delta * (phi.values?.[j] ?? 1)
    return delta
  }
  return {
    name: 'goalConditionedQ',
    config: { alpha, gamma, strategy, k, passes, features: options.features.size, nActions: n } as Json,
    metrics: {
      success: { description: 'moving average of episodes that reached their goal', direction: 'higher' },
      relabelled: { description: 'relabelled transitions learnt from' },
      tdError: { description: 'TD error of the last update' },
    },
    init: () => ({
      w: new Float64Array(options.features.size),
      episode: [],
      episodes: 0,
      steps: 0,
      successes: 0,
      successRate: 0,
      relabelled: 0,
      tdError: 0,
    }),
    act: (s, o, key) => {
      const eps = rateAt(epsilon, s.steps)
      if (eps > 0 && unit(foldIn(key, 0)) < eps) return Math.min(n - 1, Math.floor(unit(foldIn(key, 1)) * n))
      return argmaxRandom(qs(s.w, o), 0, n, foldIn(key, 2))
    },
    greedy: (s, o) => {
      const q = qs(s.w, o)
      let best = 0
      for (let a = 1; a < n; a++) if ((q[a] ?? 0) > (q[best] ?? 0)) best = a
      return best
    },
    learn: (s, t, key) => {
      const step: GoalStep = {
        observation: t.observation,
        action: t.action,
        reward: t.reward,
        nextObservation: t.nextObservation,
        terminated: t.terminated,
      }
      const w = s.w.slice()
      let delta = update(w, step)
      const episode = [...s.episode, step]
      if (!(t.terminated || t.truncated)) return { ...s, w, episode, steps: s.steps + 1, tdError: delta }
      let relabelled = s.relabelled
      if (strategy !== 'none') {
        const fresh = relabelEpisode(episode, { ...options, strategy, k }, foldIn(key, 3))
        for (let p = 0; p < passes; p++) for (const r of fresh) delta = update(w, r)
        relabelled += fresh.length * passes
      }
      const reached =
        t.terminated && goal.reward(goal.achieved(t.nextObservation), goal.desired(t.nextObservation)) >= 0
      return {
        ...s,
        w,
        episode: [],
        episodes: s.episodes + 1,
        steps: s.steps + 1,
        successes: s.successes + (reached ? 1 : 0),
        successRate: s.successRate + 0.02 * ((reached ? 1 : 0) - s.successRate),
        relabelled,
        tdError: delta,
      }
    },
    stats: (s) => ({ success: s.successRate, relabelled: s.relabelled, tdError: s.tdError }),
  }
}
