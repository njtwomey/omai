/**
 * Running policies in environments: a rollout as a traced algorithm stepping one environment step at a time, policy
 * evaluation over episodes as stable-baselines3's `evaluate_policy`, single episodes as pure functions of a key, and
 * paired comparison of policies under common random numbers, which reduces the variance of their difference.
 */

import { type Json, meanOf, varianceOf, WhiteboxError } from '../base/index.ts'
import { foldIn, type Key, key as makeKey } from '../random/index.ts'
import type { Scene } from '../scene/index.ts'
import { type Algorithm, channels, type RunOptions, runAlgorithm } from '../trace/index.ts'
import type { Policy } from './agent.ts'
import { type Env, resetAfter } from './env.ts'

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
 * episode, and, by policy, a rendered frame per step for animations and replays. Every key is derived from the step
 * key, so a rollout resumed from any state replays exactly.
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
  return {
    name: 'rollout',
    config: { env: env.spec.id, episodes } as Json,
    schema: channels({
      metrics: { reward: { description: 'reward of the step' }, return: { description: 'return so far' } },
      episodes: true,
      frames: true,
    }),
    init: (k) => {
      const r = env.reset(foldIn(k, 0))
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
      const next = resetAfter(env, r.state, foldIn(ctx.key, 2))
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

function meanStd(xs: readonly number[]): { readonly mean: number; readonly std: number } {
  return xs.length === 0 ? { mean: 0, std: 0 } : { mean: meanOf(xs), std: Math.sqrt(varianceOf(xs)) }
}

/**
 * Run a policy for a number of episodes and summarise the returns (stable-baselines3's evaluate_policy).
 *
 * @param env - The environment (with its time limit, as from `make`).
 * @param policy - The policy.
 * @param options - Episodes (default 10), a step budget, and the run options (key, seed, tracer, policy).
 * @returns Returns, lengths, and their mean and (population) standard deviation.
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
  const { mean, std } = meanStd(returns)
  return { returns, lengths: result.state.lengths, meanReturn: mean, stdReturn: std }
}

/** One episode, step by step. */
export interface Episode<S, O, A> {
  /** Observations from the reset to the last step (`length + 1` of them). */
  readonly observations: readonly O[]
  readonly states: readonly S[]
  readonly actions: readonly A[]
  readonly rewards: readonly number[]
  readonly infos: readonly Readonly<Record<string, Json>>[]
  readonly return: number
  readonly length: number
  readonly terminated: boolean
  readonly truncated: boolean
  /** Rendered frames, one per state, when requested. */
  readonly frames?: readonly Scene[]
}

/**
 * Run one episode of a policy as a pure function of a key (no tracing): the reset uses `foldIn(key, 0)`, step `t` draws
 * the policy's key from `foldIn(foldIn(key, 1), t)` and the environment's from `foldIn(foldIn(key, 2), t)`. The same
 * key reproduces the same episode, which is what common random numbers and replay rely on.
 *
 * @param env - The environment (with a time limit, or pass `maxSteps`).
 * @param policy - The policy.
 * @param key - The key.
 * @param options - A step cap (default 100000) and whether to render frames.
 * @returns The episode.
 * @throws WhiteboxError with code `value` when the cap is reached before the episode ends and `strict` is set.
 * @concept rollouts
 * @example
 * ```ts
 * import { make, runEpisode } from 'whitebox-engine/rl'
 * import { cartPole } from 'whitebox-engine/methods/rl'
 * import { key } from 'whitebox-engine/random'
 * const ep = runEpisode(make(cartPole), () => 1, key(0))
 * [ep.length === ep.rewards.length, ep.terminated] // => [true, true]
 * ```
 */
export function runEpisode<S, O, A>(
  env: Env<S, O, A>,
  policy: Policy<O, A>,
  key: Key,
  options: {
    readonly maxSteps?: number
    readonly frames?: boolean
    readonly strict?: boolean
    readonly start?: S
  } = {},
): Episode<S, O, A> {
  const cap = options.maxSteps ?? 100_000
  const r0 = env.reset(foldIn(key, 0))
  let state = options.start ?? r0.state
  let obs = options.start !== undefined && env.observe !== undefined ? env.observe(options.start) : r0.observation
  const observations: O[] = [obs]
  const states: S[] = [state]
  const actions: A[] = []
  const rewards: number[] = []
  const infos: Readonly<Record<string, Json>>[] = []
  const frames: Scene[] = options.frames === true ? [env.render(state)] : []
  const pk = foldIn(key, 1)
  const ek = foldIn(key, 2)
  let terminated = false
  let truncated = false
  let ret = 0
  for (let t = 0; t < cap; t++) {
    const a = policy(obs, foldIn(pk, t))
    const r = env.step(state, a, foldIn(ek, t))
    actions.push(a)
    rewards.push(r.reward)
    infos.push(r.info)
    ret += r.reward
    state = r.state
    obs = r.observation
    observations.push(obs)
    states.push(state)
    if (options.frames === true) frames.push(env.render(state))
    if (r.terminated || r.truncated) {
      terminated = r.terminated
      truncated = r.truncated
      break
    }
  }
  if (!terminated && !truncated && options.strict === true) {
    throw new WhiteboxError('value', `runEpisode: no end of episode within ${cap} steps`)
  }
  return {
    observations,
    states,
    actions,
    rewards,
    infos,
    return: ret,
    length: actions.length,
    terminated,
    truncated,
    ...(options.frames === true ? { frames } : {}),
  }
}

/** A paired comparison of two policies under common random numbers. */
export interface PairedComparison {
  readonly a: string
  readonly b: string
  /** Mean of the per-episode differences `return(a) - return(b)`. */
  readonly meanDifference: number
  /** Standard error of that mean. */
  readonly standardError: number
  /** Fraction of episodes in which `a` did strictly better. */
  readonly winRate: number
}

/** Result of {@link compareAgents}. */
export interface ComparisonResult {
  readonly names: readonly string[]
  readonly returns: Readonly<Record<string, readonly number[]>>
  readonly meanReturn: Readonly<Record<string, number>>
  readonly stdReturn: Readonly<Record<string, number>>
  /** Every ordered pair (a before b in `names`). */
  readonly paired: readonly PairedComparison[]
  /** The name with the highest mean return. */
  readonly best: string
}

/**
 * Compare policies with common random numbers: episode `e` of every policy uses the same key (the same start states,
 * environment noise and policy noise streams), so the per-episode differences have far less variance than independent
 * runs and paired standard errors are valid.
 *
 * @param env - The environment (with a time limit).
 * @param policies - Named policies (use `greedyPolicyOf(agent, state)` for trained agents).
 * @param options - Episodes (default 20), a key or seed, and a per-episode step cap.
 * @returns Returns per policy, summaries and paired comparisons.
 * @concept rollouts
 * @example
 * ```ts
 * import { compareAgents, make } from 'whitebox-engine/rl'
 * import { cartPole } from 'whitebox-engine/methods/rl'
 * const balance = (o: readonly number[]) => ((o[2] ?? 0) + 0.5 * (o[3] ?? 0) > 0 ? 1 : 0)
 * compareAgents(make(cartPole), { balance, push: () => 1 }, { episodes: 3 }).best // => 'balance'
 * ```
 */
export function compareAgents<S, O, A>(
  env: Env<S, O, A>,
  policies: Readonly<Record<string, Policy<O, A>>>,
  options: { readonly episodes?: number; readonly key?: Key; readonly seed?: number; readonly maxSteps?: number } = {},
): ComparisonResult {
  const names = Object.keys(policies)
  if (names.length === 0) throw new WhiteboxError('value', 'compareAgents: no policies given')
  const episodes = options.episodes ?? 20
  const root = options.key ?? makeKey(options.seed ?? 0)
  const returns: Record<string, number[]> = {}
  const meanReturn: Record<string, number> = {}
  const stdReturn: Record<string, number> = {}
  for (const name of names) {
    const policy = policies[name] as Policy<O, A>
    const rs: number[] = []
    for (let e = 0; e < episodes; e++) {
      rs.push(
        runEpisode(env, policy, foldIn(root, e), options.maxSteps === undefined ? {} : { maxSteps: options.maxSteps })
          .return,
      )
    }
    returns[name] = rs
    const { mean, std } = meanStd(rs)
    meanReturn[name] = mean
    stdReturn[name] = std
  }
  const paired: PairedComparison[] = []
  for (let i = 0; i < names.length; i++) {
    for (let j = i + 1; j < names.length; j++) {
      const a = names[i] as string
      const b = names[j] as string
      const d = (returns[a] ?? []).map((x, e) => x - ((returns[b] ?? [])[e] ?? 0))
      const { mean } = meanStd(d)
      const sd = d.length > 1 ? Math.sqrt(varianceOf(d, 1)) : 0
      paired.push({
        a,
        b,
        meanDifference: mean,
        standardError: d.length > 0 ? sd / Math.sqrt(d.length) : 0,
        winRate: d.length > 0 ? d.filter((x) => x > 0).length / d.length : 0,
      })
    }
  }
  let best = names[0] as string
  for (const n of names) if ((meanReturn[n] ?? -Infinity) > (meanReturn[best] ?? -Infinity)) best = n
  return { names, returns, meanReturn, stdReturn, paired, best }
}
