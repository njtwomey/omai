import type { Json } from '../base/index.ts'
import { foldIn, type Key, key as makeKey } from '../random/index.ts'
import { type Algorithm, channels, type RunOptions, runAlgorithm } from '../trace/index.ts'
import type { Env } from './env.ts'

/** A policy: an action for an observation (and a key, for stochastic policies). */
export type Policy<O, A> = (observation: O, key: Key) => A

/** The state of a rollout. */
export interface RolloutState<S, O> {
  readonly envState: S
  readonly observation: O
  readonly episode: number
  readonly t: number
  readonly episodeReturn: number
  readonly returns: readonly number[]
  readonly lengths: readonly number[]
}

/** Options of {@link rolloutAlgorithm}. */
export interface RolloutOptions {
  /** Number of episodes (default 1). */
  readonly episodes?: number
}

/**
 * Running a policy in an environment as an {@link Algorithm}: one environment step per algorithm step, resetting
 * between episodes. Records the reward each step, an `episode` event (return, length, terminated) at the end of each
 * episode, and, by policy, a rendered frame per step for animations and replays.
 *
 * @param env - The environment.
 * @param policy - The policy.
 * @param options - The number of episodes.
 * @returns The algorithm.
 * @concept rollouts
 * @example
 * ```ts
 * import { make, rolloutAlgorithm } from 'whitebox-engine/rl'
 * import { cartPole } from 'whitebox-engine/methods/rl'
 * rolloutAlgorithm(make(cartPole), () => 0).name // => 'rollout'
 * ```
 */
export function rolloutAlgorithm<S, O, A>(
  env: Env<S, O, A>,
  policy: Policy<O, A>,
  options: RolloutOptions = {},
): Algorithm<RolloutState<S, O>> {
  const episodes = options.episodes ?? 1
  let resetKey: Key = makeKey(0)
  return {
    name: 'rollout',
    config: { env: env.spec.id, episodes } as Json,
    schema: channels({
      metrics: { reward: { description: 'reward of the step' }, return: { description: 'return so far' } },
    }),
    init: (k) => {
      resetKey = foldIn(k, 1)
      const r = env.reset(foldIn(resetKey, 0))
      return {
        envState: r.state,
        observation: r.observation,
        episode: 0,
        t: 0,
        episodeReturn: 0,
        returns: [],
        lengths: [],
      }
    },
    step: (s, ctx) => {
      const action = policy(s.observation, foldIn(ctx.key, 0))
      const r = env.step(s.envState, action, foldIn(ctx.key, 1))
      const ret = s.episodeReturn + r.reward
      if (ctx.run.wants('metric', 'reward')) ctx.run.metric(ctx.t + 1, 'reward', r.reward)
      if (!(r.terminated || r.truncated)) {
        return { ...s, envState: r.state, observation: r.observation, t: s.t + 1, episodeReturn: ret }
      }
      ctx.run.episode(s.episode, ret, s.t + 1, r.terminated)
      const next = env.reset(foldIn(resetKey, s.episode + 1))
      return {
        envState: next.state,
        observation: next.observation,
        episode: s.episode + 1,
        t: 0,
        episodeReturn: 0,
        returns: [...s.returns, ret],
        lengths: [...s.lengths, s.t + 1],
      }
    },
    done: (s) => (s.episode >= episodes ? 'completed' : undefined),
    observe: (s) => ({ metrics: { return: s.episodeReturn }, frame: env.render(s.envState) as unknown as Json }),
  }
}

/** Result of {@link evaluatePolicy}. */
export interface EvaluationResult {
  readonly returns: readonly number[]
  readonly lengths: readonly number[]
  readonly meanReturn: number
  readonly stdReturn: number
}

/**
 * Run a policy for a number of episodes and summarise the returns (stable-baselines3's evaluate_policy).
 *
 * @param env - The environment (with its time limit, as from `make`).
 * @param policy - The policy.
 * @param options - Episodes (default 10), a step budget, and the run options (key, seed, tracer, policy).
 * @returns Returns, lengths, and their mean and standard deviation.
 * @concept rollouts
 * @example
 * ```ts
 * import { make, evaluatePolicy } from 'whitebox-engine/rl'
 * import { cartPole } from 'whitebox-engine/methods/rl'
 * evaluatePolicy(make(cartPole), () => 0, { episodes: 3 }).returns.length // => 3
 * ```
 */
export function evaluatePolicy<S, O, A>(
  env: Env<S, O, A>,
  policy: Policy<O, A>,
  options: { readonly episodes?: number; readonly maxSteps?: number } & Partial<Omit<RunOptions, 'maxSteps'>> = {},
): EvaluationResult {
  const { episodes = 10, maxSteps, ...run } = options
  const result = runAlgorithm(rolloutAlgorithm(env, policy, { episodes }), { ...run, maxSteps: maxSteps ?? 1_000_000 })
  const returns = result.state.returns
  const mean = returns.reduce((a, b) => a + b, 0) / Math.max(1, returns.length)
  const sd = Math.sqrt(returns.reduce((a, b) => a + (b - mean) ** 2, 0) / Math.max(1, returns.length))
  return { returns, lengths: result.state.lengths, meanReturn: mean, stdReturn: sd }
}
