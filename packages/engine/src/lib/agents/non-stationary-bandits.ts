/**
 * Bandit agents for rewards whose means change over time (Garivier and Moulines 2011): sliding-window UCB, which
 * computes its upper confidence bounds from the last `tau` pulls only, and discounted UCB, which weights each pull by
 * `gamma` to the power of its age. Both forget at a rate set by one knob, so after an abrupt change in the arms'
 * means they recover in about `tau` (or `1 / (1 - gamma)`) pulls where plain UCB, whose averages hold the whole history,
 * keeps pulling the arm that used to be best.
 */

import { argmaxOf, OmaiError } from '../base/index.ts'
import type { Agent } from '../rl/index.ts'
import type { BanditSpaces } from './bandits.ts'

/** The state of {@link slidingWindowUcb}: the last `tau` pulls in a ring buffer, and the counts and sums over them. */
export interface SlidingWindowState {
  /** Pulls and reward sums per arm within the window. */
  readonly counts: Float64Array
  readonly sums: Float64Array
  /** The arms and rewards of the window, in a ring buffer of `tau` slots (`-1` for an empty slot). */
  readonly arms: Int32Array
  readonly rewards: Float64Array
  /** The pulls made so far. */
  readonly t: number
}

/** Options of {@link slidingWindowUcb}. */
export interface SlidingWindowUcbOptions {
  /** The window: the number of most recent pulls the indices are computed from (default 200). */
  readonly window?: number
  /** The exploration constant `xi` of the bonus `B sqrt(xi log(min(t, tau)) / n)` (default 0.6, as in the paper). */
  readonly xi?: number
  /** The range `B` of the rewards (default 1). */
  readonly range?: number
}

const armCount = (env: BanditSpaces, name: string): number => {
  if (env.actionSpace.kind !== 'discrete') throw new OmaiError('value', `${name}: needs discrete arms`)
  return env.actionSpace.n
}

/**
 * Sliding-window UCB (Garivier and Moulines 2011): pull each arm once, then the arm with the largest
 * `mean_tau + B sqrt(xi log(min(t, tau)) / n_tau)`, where the mean and the count `n_tau` use only the last `tau` pulls. An
 * arm that has left the window is unseen (an infinite index), so every arm is revisited at least every `tau` pulls, and
 * after an abrupt change the estimates are replaced within a window. Against `Upsilon` change points over `T` pulls its
 * regret is `O(sqrt(Upsilon T log T))` with the window `tau ~ sqrt(T log T / Upsilon)`.
 *
 * @param env - The bandit.
 * @param options - The window, exploration constant and reward range.
 * @returns The agent; its greedy action is the arm with the best mean in the window.
 * @throws OmaiError with code `value` for a window below one.
 * @concept upper-confidence-bounds
 * @example
 * ```ts
 * import { slidingWindowUcb } from 'omai-engine/agents'
 * import { changePointBandit } from 'omai-engine/environments'
 * import { key } from 'omai-engine/random'
 * const agent = slidingWindowUcb(changePointBandit.create(), { window: 100 })
 * agent.act(agent.init(key(0)), 0, key(1)) // => 0
 * ```
 */
export function slidingWindowUcb(
  env: BanditSpaces,
  options: SlidingWindowUcbOptions = {},
): Agent<SlidingWindowState, unknown, number> {
  const k = armCount(env, 'slidingWindowUcb')
  const tau = options.window ?? 200
  const xi = options.xi ?? 0.6
  const range = options.range ?? 1
  if (!(Number.isInteger(tau) && tau >= 1))
    throw new OmaiError('value', `slidingWindowUcb: window ${tau} must be an integer >= 1`)
  const indices = (s: SlidingWindowState): number[] =>
    Array.from({ length: k }, (_, a) => {
      const n = s.counts[a] ?? 0
      if (n === 0) return Number.POSITIVE_INFINITY
      return (s.sums[a] ?? 0) / n + range * Math.sqrt((xi * Math.log(Math.min(Math.max(s.t, 1), tau))) / n)
    })
  return {
    name: 'slidingWindowUcb',
    config: { arms: k, window: tau, xi, range },
    metrics: { window: { description: 'pulls in the window' } },
    init: () => ({
      counts: new Float64Array(k),
      sums: new Float64Array(k),
      arms: new Int32Array(tau).fill(-1),
      rewards: new Float64Array(tau),
      t: 0,
    }),
    act: (s) => argmaxOf(indices(s)),
    greedy: (s) =>
      argmaxOf(
        Array.from({ length: k }, (_, a) =>
          (s.counts[a] ?? 0) === 0 ? Number.NEGATIVE_INFINITY : (s.sums[a] ?? 0) / (s.counts[a] ?? 1),
        ),
      ),
    learn: (s, tr) => {
      const counts = s.counts.slice()
      const sums = s.sums.slice()
      const arms = s.arms.slice()
      const rewards = s.rewards.slice()
      const slot = s.t % tau
      const old = arms[slot] ?? -1
      if (old >= 0) {
        counts[old] = (counts[old] ?? 0) - 1
        sums[old] = (sums[old] ?? 0) - (rewards[slot] ?? 0)
      }
      arms[slot] = tr.action
      rewards[slot] = tr.reward
      counts[tr.action] = (counts[tr.action] ?? 0) + 1
      sums[tr.action] = (sums[tr.action] ?? 0) + tr.reward
      return { counts, sums, arms, rewards, t: s.t + 1 }
    },
    stats: (s) => ({ window: Math.min(s.t, tau) }),
  }
}

/** The state of {@link discountedUcb}: discounted pull counts and reward sums. */
export interface DiscountedState {
  readonly counts: Float64Array
  readonly sums: Float64Array
  readonly t: number
}

/** Options of {@link discountedUcb}. */
export interface DiscountedUcbOptions {
  /** The discount `gamma` applied to every past pull at each step (default 0.99, a memory of about 100 pulls). */
  readonly discount?: number
  /** The exploration constant `xi` of the bonus `2 B sqrt(xi log(n) / N)` (default 0.6). */
  readonly xi?: number
  /** The range `B` of the rewards (default 1). */
  readonly range?: number
}

/**
 * Discounted UCB (Kocsis and Szepesvari 2006; Garivier and Moulines 2011): every pull's weight is multiplied by
 * `gamma` at each step, so the discounted count `N_i` and mean of an arm have an effective memory of `1 / (1 - gamma)`
 * pulls, and the index is `mean_i + 2 B sqrt(xi log(n) / N_i)` with `n = sum_i N_i` the total discounted count.
 *
 * @param env - The bandit.
 * @param options - The discount, exploration constant and reward range.
 * @returns The agent.
 * @throws OmaiError with code `value` for a discount outside (0, 1).
 * @concept upper-confidence-bounds
 * @example
 * ```ts
 * import { discountedUcb } from 'omai-engine/agents'
 * import { changePointBandit } from 'omai-engine/environments'
 * import { key } from 'omai-engine/random'
 * const agent = discountedUcb(changePointBandit.create(), { discount: 0.9 })
 * const t = { observation: 0, action: 0, reward: 1, nextObservation: 0, terminated: false, truncated: false }
 * agent.learn(agent.learn(agent.init(key(0)), t, key(1)), t, key(2)).counts[0] // => 1.9
 * ```
 */
export function discountedUcb(
  env: BanditSpaces,
  options: DiscountedUcbOptions = {},
): Agent<DiscountedState, unknown, number> {
  const k = armCount(env, 'discountedUcb')
  const gamma = options.discount ?? 0.99
  const xi = options.xi ?? 0.6
  const range = options.range ?? 1
  if (!(gamma > 0 && gamma < 1)) throw new OmaiError('value', `discountedUcb: discount ${gamma} must be in (0, 1)`)
  return {
    name: 'discountedUcb',
    config: { arms: k, discount: gamma, xi, range },
    metrics: { memory: { description: 'total discounted pulls' } },
    init: () => ({ counts: new Float64Array(k), sums: new Float64Array(k), t: 0 }),
    act: (s) => {
      const total = s.counts.reduce((a, b) => a + b, 0)
      return argmaxOf(
        Array.from({ length: k }, (_, a) => {
          const n = s.counts[a] ?? 0
          if (n <= 1e-12) return Number.POSITIVE_INFINITY
          return (s.sums[a] ?? 0) / n + 2 * range * Math.sqrt((xi * Math.log(Math.max(total, 1))) / n)
        }),
      )
    },
    greedy: (s) =>
      argmaxOf(
        Array.from({ length: k }, (_, a) =>
          (s.counts[a] ?? 0) <= 1e-12 ? Number.NEGATIVE_INFINITY : (s.sums[a] ?? 0) / (s.counts[a] ?? 1),
        ),
      ),
    learn: (s, tr) => {
      const counts = s.counts.map((c) => c * gamma)
      const sums = s.sums.map((x) => x * gamma)
      counts[tr.action] = (counts[tr.action] ?? 0) + 1
      sums[tr.action] = (sums[tr.action] ?? 0) + tr.reward
      return { counts, sums, t: s.t + 1 }
    },
    stats: (s) => ({ memory: s.counts.reduce((a, b) => a + b, 0) }),
  }
}
