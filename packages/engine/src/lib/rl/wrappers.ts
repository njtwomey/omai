/**
 * Environment wrappers that change episodes, rewards and actions, as in gymnasium: time limits, episode statistics,
 * running mean and variance, reward normalisation, transformation and clipping, action transformation, clipping and
 * rescaling, action repeat (frame skip), sticky actions, and observation transformation.
 */

import { WhiteboxError } from '../base/index.ts'
import { foldIn, type Key, uniform } from '../random/index.ts'
import { type BoxSpace, box, type Space } from '../space/index.ts'
import { type Env, type EnvModel, type EnvOracle, type Reset, resetAfter, type Step } from './env.ts'

// Stateless wrappers keep the inner state; they rewrite the spaces, observations, rewards or actions.
function mapped<S, O, A, O2, A2>(
  env: Env<S, O, A>,
  parts: {
    readonly observationSpace?: Space
    readonly actionSpace?: Space
    readonly observation?: (o: O) => O2
    readonly action?: (a: A2) => A
    readonly reward?: (r: number) => number
  },
): Env<S, O2, A2> {
  const obs = parts.observation ?? ((o: O) => o as unknown as O2)
  const act = parts.action ?? ((a: A2) => a as unknown as A)
  const rew = parts.reward ?? ((r: number) => r)
  const lift = (r: Reset<S, O>): Reset<S, O2> => ({ ...r, observation: obs(r.observation) })
  const inner = env.observe
  const resetFrom = env.resetFrom
  const keepModel = parts.action === undefined && parts.reward === undefined
  return {
    spec: env.spec,
    observationSpace: parts.observationSpace ?? env.observationSpace,
    actionSpace: parts.actionSpace ?? env.actionSpace,
    reset: (k) => lift(env.reset(k)),
    ...(resetFrom === undefined ? {} : { resetFrom: (p: S, k: Key) => lift(resetFrom(p, k)) }),
    step: (s, a, k) => {
      const r = env.step(s, act(a), k)
      return { ...r, observation: obs(r.observation), reward: rew(r.reward) }
    },
    render: (s) => env.render(s),
    ...(inner === undefined ? {} : { observe: (s: S) => obs(inner(s)) }),
    // Without action or reward maps the model and oracle still describe the wrapped environment exactly.
    ...(keepModel && env.model !== undefined ? { model: env.model as unknown as EnvModel<S, A2> } : {}),
    ...(keepModel && env.oracle !== undefined ? { oracle: env.oracle as unknown as EnvOracle<S, A2> } : {}),
    wrapped: env as Env<unknown, unknown, unknown>,
  }
}

/** The state of {@link timeLimit}: the inner state and the steps taken this episode. */
export interface TimeLimitState<S> {
  readonly inner: S
  readonly t: number
}

/**
 * Truncate episodes after `maxSteps` steps (gymnasium's TimeLimit). The step count lives in the state.
 *
 * @param env - The environment.
 * @param maxSteps - The time limit.
 * @returns The limited environment.
 * @concept environments
 * @example
 * ```ts
 * import { timeLimit } from 'whitebox-engine/rl'
 * import { cartPole } from 'whitebox-engine/methods/rl'
 * timeLimit(cartPole.create(), 10).spec.maxEpisodeSteps // => 10
 * ```
 */
export function timeLimit<S, O, A>(env: Env<S, O, A>, maxSteps: number): Env<TimeLimitState<S>, O, A> {
  if (!(maxSteps >= 1)) throw new WhiteboxError('value', `timeLimit: maxSteps must be at least 1, got ${maxSteps}`)
  const resetFrom = env.resetFrom
  const inner = env.observe
  const start = (r: Reset<S, O>): Reset<TimeLimitState<S>, O> => ({ ...r, state: { inner: r.state, t: 0 } })
  return {
    spec: { ...env.spec, maxEpisodeSteps: maxSteps },
    observationSpace: env.observationSpace,
    actionSpace: env.actionSpace,
    reset: (k) => start(env.reset(k)),
    ...(resetFrom === undefined ? {} : { resetFrom: (p: TimeLimitState<S>, k: Key) => start(resetFrom(p.inner, k)) }),
    step: (s, action, k) => {
      const r = env.step(s.inner, action, k)
      const t = s.t + 1
      return { ...r, state: { inner: r.state, t }, truncated: r.truncated || (t >= maxSteps && !r.terminated) }
    },
    render: (s) => env.render(s.inner),
    ...(inner === undefined ? {} : { observe: (s: TimeLimitState<S>) => inner(s.inner) }),
    wrapped: env as Env<unknown, unknown, unknown>,
  }
}

/** The state of {@link episodeStatistics}. */
export interface EpisodeStatisticsState<S> {
  readonly inner: S
  readonly episodeReturn: number
  readonly episodeLength: number
  readonly episodes: number
  /** Returns of the most recent finished episodes (oldest first). */
  readonly returns: readonly number[]
  readonly lengths: readonly number[]
}

/**
 * Record episode returns and lengths (gymnasium's RecordEpisodeStatistics): the last step of every episode carries
 * `info.episode = { r, l }`, and the state keeps the most recent `bufferLength` returns and lengths across resets
 * (through `resetFrom`).
 *
 * @param env - The environment.
 * @param options - How many finished episodes to keep (default 100).
 * @returns The recording environment.
 * @concept environments
 * @example
 * ```ts
 * import { episodeStatistics } from 'whitebox-engine/rl'
 * import { cartPole } from 'whitebox-engine/methods/rl'
 * import { key } from 'whitebox-engine/random'
 * const env = episodeStatistics(cartPole.create())
 * env.step(env.reset(key(0)).state, 1, key(1)).state.episodeLength // => 1
 * ```
 */
export function episodeStatistics<S, O, A>(
  env: Env<S, O, A>,
  options: { readonly bufferLength?: number } = {},
): Env<EpisodeStatisticsState<S>, O, A> {
  const keep = options.bufferLength ?? 100
  const inner = env.observe
  const fresh = (r: Reset<S, O>, prev?: EpisodeStatisticsState<S>): Reset<EpisodeStatisticsState<S>, O> => ({
    ...r,
    state: {
      inner: r.state,
      episodeReturn: 0,
      episodeLength: 0,
      episodes: prev?.episodes ?? 0,
      returns: prev?.returns ?? [],
      lengths: prev?.lengths ?? [],
    },
  })
  return {
    spec: env.spec,
    observationSpace: env.observationSpace,
    actionSpace: env.actionSpace,
    reset: (k) => fresh(env.reset(k)),
    resetFrom: (p, k) => fresh(resetAfter(env, p.inner, k), p),
    step: (s, action, k) => {
      const r = env.step(s.inner, action, k)
      const ret = s.episodeReturn + r.reward
      const len = s.episodeLength + 1
      if (!(r.terminated || r.truncated)) {
        return { ...r, state: { ...s, inner: r.state, episodeReturn: ret, episodeLength: len } }
      }
      return {
        ...r,
        info: { ...r.info, episode: { r: ret, l: len } },
        state: {
          inner: r.state,
          episodeReturn: ret,
          episodeLength: len,
          episodes: s.episodes + 1,
          returns: [...s.returns, ret].slice(-keep),
          lengths: [...s.lengths, len].slice(-keep),
        },
      }
    },
    render: (s) => env.render(s.inner),
    ...(inner === undefined ? {} : { observe: (s: EpisodeStatisticsState<S>) => inner(s.inner) }),
    wrapped: env as Env<unknown, unknown, unknown>,
  }
}

/** Running mean and variance with a count (gymnasium's RunningMeanStd, Chan et al.'s parallel update). */
export interface RunningMeanStd {
  readonly mean: readonly number[]
  readonly var: readonly number[]
  readonly count: number
}

/**
 * Fresh running statistics: mean 0, variance 1 and a small prior count (gymnasium's defaults).
 *
 * @param size - The number of coordinates.
 * @param epsilon - The prior count (default 1e-4).
 * @returns The statistics.
 * @example
 * ```ts
 * import { runningMeanStd } from 'whitebox-engine/rl'
 * runningMeanStd(2).var // => [1, 1]
 * ```
 */
export function runningMeanStd(size: number, epsilon = 1e-4): RunningMeanStd {
  return { mean: new Array<number>(size).fill(0), var: new Array<number>(size).fill(1), count: epsilon }
}

/**
 * Fold a batch of rows into running statistics (gymnasium's `update_mean_var_count_from_moments`).
 *
 * @param stats - The statistics.
 * @param batch - Rows of the same length as the statistics.
 * @returns The updated statistics.
 * @example
 * ```ts
 * import { runningMeanStd, updateRunningMeanStd } from 'whitebox-engine/rl'
 * updateRunningMeanStd(runningMeanStd(1, 0), [[1], [3]]).mean // => [2]
 * ```
 */
export function updateRunningMeanStd(stats: RunningMeanStd, batch: readonly (readonly number[])[]): RunningMeanStd {
  const n = batch.length
  if (n === 0) return stats
  const d = stats.mean.length
  const mean: number[] = []
  const variance: number[] = []
  const total = stats.count + n
  for (let j = 0; j < d; j++) {
    let bm = 0
    for (const row of batch) bm += row[j] ?? 0
    bm /= n
    let bv = 0
    for (const row of batch) bv += ((row[j] ?? 0) - bm) ** 2
    bv /= n
    const m0 = stats.mean[j] ?? 0
    const delta = bm - m0
    mean.push(m0 + (delta * n) / total)
    const m2 = (stats.var[j] ?? 0) * stats.count + bv * n + (delta * delta * stats.count * n) / total
    variance.push(m2 / total)
  }
  return { mean, var: variance, count: total }
}

/** The state of {@link normalizeReward}. */
export interface NormalizeRewardState<S> {
  readonly inner: S
  readonly discounted: number
  readonly stats: RunningMeanStd
}

/**
 * Scale rewards so that their discounted sum has roughly unit variance (gymnasium's NormalizeReward): the running
 * variance of the discounted return is updated every step and carried across episodes by `resetFrom`.
 *
 * @param env - The environment.
 * @param options - The discount (default 0.99), the stability constant (default 1e-8) and whether to keep updating.
 * @returns The normalising environment.
 * @concept environments
 * @example
 * ```ts
 * import { normalizeReward } from 'whitebox-engine/rl'
 * import { cartPole } from 'whitebox-engine/methods/rl'
 * import { key } from 'whitebox-engine/random'
 * const env = normalizeReward(cartPole.create())
 * env.step(env.reset(key(0)).state, 1, key(1)).reward > 1 // => true
 * ```
 */
export function normalizeReward<S, O, A>(
  env: Env<S, O, A>,
  options: { readonly gamma?: number; readonly epsilon?: number; readonly update?: boolean } = {},
): Env<NormalizeRewardState<S>, O, A> {
  const gamma = options.gamma ?? 0.99
  const eps = options.epsilon ?? 1e-8
  const update = options.update ?? true
  if (!(gamma >= 0 && gamma <= 1)) throw new WhiteboxError('value', 'normalizeReward: gamma must lie in [0, 1]')
  if (!(eps > 0)) throw new WhiteboxError('value', 'normalizeReward: epsilon must be positive')
  const inner = env.observe
  return {
    spec: env.spec,
    observationSpace: env.observationSpace,
    actionSpace: env.actionSpace,
    reset: (k) => {
      const r = env.reset(k)
      return { ...r, state: { inner: r.state, discounted: 0, stats: runningMeanStd(1) } }
    },
    resetFrom: (p, k) => {
      const r = resetAfter(env, p.inner, k)
      return { ...r, state: { ...p, inner: r.state } }
    },
    step: (s, action, k) => {
      const r = env.step(s.inner, action, k)
      const discounted = s.discounted * gamma * (r.terminated ? 0 : 1) + r.reward
      const stats = update ? updateRunningMeanStd(s.stats, [[discounted]]) : s.stats
      return {
        ...r,
        reward: r.reward / Math.sqrt((stats.var[0] ?? 1) + eps),
        state: { inner: r.state, discounted, stats },
      }
    },
    render: (s) => env.render(s.inner),
    ...(inner === undefined ? {} : { observe: (s: NormalizeRewardState<S>) => inner(s.inner) }),
    wrapped: env as Env<unknown, unknown, unknown>,
  }
}

/**
 * Transform every reward (gymnasium's TransformReward).
 *
 * @param env - The environment.
 * @param f - The reward map.
 * @returns The transformed environment.
 * @concept environments
 * @example
 * ```ts
 * import { transformReward } from 'whitebox-engine/rl'
 * import { cartPole } from 'whitebox-engine/methods/rl'
 * import { key } from 'whitebox-engine/random'
 * const env = transformReward(cartPole.create(), (r) => 2 * r)
 * env.step(env.reset(key(0)).state, 1, key(1)).reward // => 2
 * ```
 */
export function transformReward<S, O, A>(env: Env<S, O, A>, f: (reward: number) => number): Env<S, O, A> {
  return mapped<S, O, A, O, A>(env, { reward: f })
}

/**
 * Clip rewards to `[min, max]` (gymnasium's ClipReward; Atari agents clip to [-1, 1]).
 *
 * @param env - The environment.
 * @param min - Lower bound (default -1).
 * @param max - Upper bound (default 1).
 * @returns The clipped environment.
 * @concept environments
 * @example
 * ```ts
 * import { clipReward } from 'whitebox-engine/rl'
 * import { cartPole } from 'whitebox-engine/methods/rl'
 * import { key } from 'whitebox-engine/random'
 * const env = clipReward(cartPole.create(), 0, 0.5)
 * env.step(env.reset(key(0)).state, 1, key(1)).reward // => 0.5
 * ```
 */
export function clipReward<S, O, A>(env: Env<S, O, A>, min = -1, max = 1): Env<S, O, A> {
  if (!(max >= min)) throw new WhiteboxError('value', 'clipReward: need max >= min')
  return transformReward(env, (r) => Math.min(max, Math.max(min, r)))
}

/**
 * Transform every action before it reaches the environment (gymnasium's TransformAction).
 *
 * @param env - The environment.
 * @param f - The map from the new actions to the environment's.
 * @param actionSpace - The new action space (default unchanged).
 * @returns The transformed environment.
 * @concept environments
 * @example
 * ```ts
 * import { transformAction } from 'whitebox-engine/rl'
 * import { cartPole } from 'whitebox-engine/methods/rl'
 * import { key } from 'whitebox-engine/random'
 * const env = transformAction(cartPole.create(), (push: boolean) => (push ? 1 : 0))
 * env.step(env.reset(key(0)).state, true, key(1)).reward // => 1
 * ```
 */
export function transformAction<S, O, A, A2>(
  env: Env<S, O, A>,
  f: (action: A2) => A,
  actionSpace?: Space,
): Env<S, O, A2> {
  return mapped<S, O, A, O, A2>(env, { action: f, ...(actionSpace ? { actionSpace } : {}) })
}

function boxOf(
  space: Space,
  name: string,
): { readonly low: number[]; readonly high: number[]; readonly box: BoxSpace } {
  if (space.kind !== 'box') throw new WhiteboxError('value', `${name}: needs a box action space`)
  const n = space.shape.reduce((a, b) => a * b, 1)
  const at = (b: number | readonly number[], i: number): number => (typeof b === 'number' ? b : (b[i] ?? 0))
  return {
    low: Array.from({ length: n }, (_, i) => at(space.low, i)),
    high: Array.from({ length: n }, (_, i) => at(space.high, i)),
    box: space,
  }
}

/**
 * Clip continuous actions into the action box (gymnasium's ClipAction); the new action space is unbounded.
 *
 * @param env - An environment with a box action space.
 * @returns The clipping environment.
 * @throws WhiteboxError with code `value` when the action space is not a box.
 * @concept environments
 * @example
 * ```ts
 * import { clipAction } from 'whitebox-engine/rl'
 * import { pendulum } from 'whitebox-engine/methods/rl'
 * import { key } from 'whitebox-engine/random'
 * const env = clipAction(pendulum.create())
 * const s = env.reset(key(0)).state
 * env.step(s, [100], key(1)).state[1] === env.step(s, [2], key(1)).state[1] // => true
 * ```
 */
export function clipAction<S, O>(env: Env<S, O, readonly number[]>): Env<S, O, readonly number[]> {
  const { low, high, box: b } = boxOf(env.actionSpace, 'clipAction')
  return mapped<S, O, readonly number[], O, readonly number[]>(env, {
    actionSpace: box(-Number.MAX_VALUE, Number.MAX_VALUE, b.shape),
    action: (a) => a.map((v, i) => Math.min(high[i] ?? 0, Math.max(low[i] ?? 0, v))),
  })
}

/**
 * Rescale continuous actions affinely from `[min, max]` to the action box (gymnasium's RescaleAction), so an agent can
 * act in `[-1, 1]` whatever the environment's bounds.
 *
 * @param env - An environment with a bounded box action space.
 * @param min - The new lower bound (default -1).
 * @param max - The new upper bound (default 1).
 * @returns The rescaling environment.
 * @throws WhiteboxError with code `value` when the action space is not a box or `min >= max`.
 * @concept environments
 * @example
 * ```ts
 * import { rescaleAction } from 'whitebox-engine/rl'
 * import { pendulum } from 'whitebox-engine/methods/rl'
 * import { key } from 'whitebox-engine/random'
 * const env = rescaleAction(pendulum.create(), -1, 1)
 * const s = env.reset(key(0)).state
 * env.step(s, [0.5], key(1)).state[1] === pendulum.create().step(s, [1], key(1)).state[1] // => true
 * ```
 */
export function rescaleAction<S, O>(
  env: Env<S, O, readonly number[]>,
  min = -1,
  max = 1,
): Env<S, O, readonly number[]> {
  if (!(max > min)) throw new WhiteboxError('value', 'rescaleAction: need min < max')
  const { low, high, box: b } = boxOf(env.actionSpace, 'rescaleAction')
  return mapped<S, O, readonly number[], O, readonly number[]>(env, {
    actionSpace: box(min, max, b.shape),
    action: (a) => a.map((v, i) => (low[i] ?? 0) + ((high[i] ?? 0) - (low[i] ?? 0)) * ((v - min) / (max - min))),
  })
}

/**
 * Repeat each action `k` times, summing the rewards and stopping early at the end of an episode (gymnasium's
 * RepeatAction, the frame skip of Atari agents).
 *
 * @param env - The environment.
 * @param k - The number of repeats.
 * @returns The repeating environment.
 * @concept environments
 * @example
 * ```ts
 * import { actionRepeat } from 'whitebox-engine/rl'
 * import { cartPole } from 'whitebox-engine/methods/rl'
 * import { key } from 'whitebox-engine/random'
 * const env = actionRepeat(cartPole.create(), 4)
 * env.step(env.reset(key(0)).state, 1, key(1)).reward // => 4
 * ```
 */
export function actionRepeat<S, O, A>(env: Env<S, O, A>, k: number): Env<S, O, A> {
  if (!(Number.isInteger(k) && k >= 1)) throw new WhiteboxError('value', 'actionRepeat: k must be a positive integer')
  const resetFrom = env.resetFrom
  const inner = env.observe
  return {
    spec: env.spec,
    observationSpace: env.observationSpace,
    actionSpace: env.actionSpace,
    reset: (key) => env.reset(key),
    ...(resetFrom === undefined ? {} : { resetFrom: (p: S, key: Key) => resetFrom(p, key) }),
    step: (s, action, key) => {
      let total = 0
      let r: Step<S, O> | undefined
      let state = s
      for (let i = 0; i < k; i++) {
        r = env.step(state, action, foldIn(key, i))
        total += r.reward
        state = r.state
        if (r.terminated || r.truncated) break
      }
      return { ...(r as Step<S, O>), reward: total }
    },
    render: (s) => env.render(s),
    ...(inner === undefined ? {} : { observe: inner }),
    wrapped: env as Env<unknown, unknown, unknown>,
  }
}

/** The state of {@link stickyAction}: the inner state and the action last executed. */
export interface StickyActionState<S, A> {
  readonly inner: S
  readonly last?: A
}

/**
 * Sticky actions (Machado et al. 2018; gymnasium's StickyAction): with probability `p` the environment repeats the
 * previously executed action instead of the chosen one, which makes deterministic environments stochastic.
 *
 * @param env - The environment.
 * @param p - The repeat probability (default 0.25).
 * @returns The sticky environment.
 * @concept environments
 * @example
 * ```ts
 * import { stickyAction } from 'whitebox-engine/rl'
 * import { cartPole } from 'whitebox-engine/methods/rl'
 * import { key } from 'whitebox-engine/random'
 * const env = stickyAction(cartPole.create(), 1)
 * const s1 = env.step(env.reset(key(0)).state, 1, key(1)).state
 * env.step(s1, 0, key(2)).state.last // => 1
 * ```
 */
export function stickyAction<S, O, A>(env: Env<S, O, A>, p = 0.25): Env<StickyActionState<S, A>, O, A> {
  if (!(p >= 0 && p <= 1)) throw new WhiteboxError('value', 'stickyAction: p must lie in [0, 1]')
  const resetFrom = env.resetFrom
  const inner = env.observe
  const start = (r: Reset<S, O>): Reset<StickyActionState<S, A>, O> => ({ ...r, state: { inner: r.state } })
  return {
    spec: { ...env.spec, deterministic: false },
    observationSpace: env.observationSpace,
    actionSpace: env.actionSpace,
    reset: (k) => start(env.reset(k)),
    ...(resetFrom === undefined
      ? {}
      : { resetFrom: (prev: StickyActionState<S, A>, k: Key) => start(resetFrom(prev.inner, k)) }),
    step: (s, action, k) => {
      const u = uniform(foldIn(k, 0x5354), []).data[0] ?? 1
      const executed = s.last !== undefined && u < p ? s.last : action
      const r = env.step(s.inner, executed, foldIn(k, 0))
      return { ...r, state: { inner: r.state, last: executed } }
    },
    render: (s) => env.render(s.inner),
    ...(inner === undefined ? {} : { observe: (s: StickyActionState<S, A>) => inner(s.inner) }),
    wrapped: env as Env<unknown, unknown, unknown>,
  }
}

/**
 * Transform every observation (gymnasium's TransformObservation).
 *
 * @param env - The environment.
 * @param f - The observation map.
 * @param observationSpace - The new observation space (default unchanged).
 * @returns The transformed environment.
 * @concept environments
 * @example
 * ```ts
 * import { transformObservation } from 'whitebox-engine/rl'
 * import { cartPole } from 'whitebox-engine/methods/rl'
 * import { key } from 'whitebox-engine/random'
 * const env = transformObservation(cartPole.create(), (o) => o.length)
 * env.reset(key(0)).observation // => 4
 * ```
 */
export function transformObservation<S, O, A, O2>(
  env: Env<S, O, A>,
  f: (observation: O) => O2,
  observationSpace?: Space,
): Env<S, O2, A> {
  return mapped<S, O, A, O2, A>(env, { observation: f, ...(observationSpace ? { observationSpace } : {}) })
}
