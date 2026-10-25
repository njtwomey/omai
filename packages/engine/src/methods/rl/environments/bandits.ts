/**
 * Multi-armed bandit environments, where each pull of an arm pays a random reward: Bernoulli and Gaussian arms, a
 * contextual linear bandit that shows a feature vector per arm every round, an oblivious adversarial bandit whose best
 * arm keeps changing, and Sutton and Barto's non-stationary bandit whose means drift as random walks. Each carries an
 * oracle of the best expected reward, so an agent's regret can be measured.
 */

import { argmaxOf, MdmlError, maxOf, minOf } from '../../../lib/base/index.ts'
import { foldIn, type Key, key as makeKey, normal, uniform } from '../../../lib/random/index.ts'
import type { BanditEnv, EnvDefinition } from '../../../lib/rl/index.ts'
import { rect, type Scene, scene, text } from '../../../lib/scene/index.ts'
import { box, dict, discrete, int, real, tuple } from '../../../lib/space/index.ts'

const u1 = (k: Key): number => uniform(k, []).data[0] ?? 0
const n1 = (k: Key): number => normal(k, []).data[0] ?? 0

function renderArms(values: readonly number[], last: number | undefined, title: string): Scene {
  const w = 60
  const h = 200
  const min = Math.min(0, minOf(values))
  const span = Math.max(Math.max(0, maxOf(values)) - min, 1e-9)
  return scene(
    values.length * w,
    h + 30,
    [
      ...values.map((v, i) =>
        rect(i * w + 10, h - ((v - min) / span) * h, w - 20, ((v - min) / span) * h, {
          fill: i === last ? 'tab1' : 'tab0',
          id: `arm${i}`,
          title: `arm ${i}: expected reward ${v.toFixed(3)}`,
        }),
      ),
      ...values.map((_, i) => text(i * w + w / 2, h + 20, String(i), { anchor: 'middle', size: 14 })),
    ],
    { title },
  )
}

/** The state of a context-free bandit: the pull count and the last arm pulled. */
export interface BanditState {
  readonly t: number
  readonly last?: number
}

const list = (
  options: Readonly<Record<string, unknown>> | undefined,
  name: string,
  fallback: readonly number[],
): number[] => {
  const v = options?.[name]
  return Array.isArray(v) && v.every((x) => typeof x === 'number') ? (v as number[]) : [...fallback]
}

const horizonOf = (options: Readonly<Record<string, unknown>> | undefined, fallback = 1000): number => {
  const v = options?.['horizon']
  return typeof v === 'number' ? v : fallback
}

function stationary(
  id: string,
  description: string,
  means: readonly number[],
  horizon: number,
  draw: (arm: number, k: Key) => number,
): BanditEnv<BanditState, number> {
  if (means.length < 1) throw new MdmlError('value', `${id}: need at least one arm`)
  const best = maxOf(means)
  const bestArm = argmaxOf(means)
  return {
    spec: { id, description, maxEpisodeSteps: horizon },
    observationSpace: discrete(1, { label: 'no context' }),
    actionSpace: discrete(means.length, { label: 'arm' }),
    reset: () => ({ state: { t: 0 }, observation: 0, info: {} }),
    step: (s, arm, k) => ({
      state: { t: s.t + 1, last: arm },
      observation: 0,
      reward: draw(arm, k),
      terminated: false,
      truncated: false,
      info: {},
    }),
    render: (s) => renderArms(means, s.last, id),
    observe: () => 0,
    oracle: {
      expectedReward: (_, arm) => means[arm] ?? 0,
      optimalExpectedReward: () => best,
      optimalAction: () => bestArm,
    },
  }
}

/**
 * BernoulliBandit-v0: arms pay 1 with their probabilities `probs` (default 0.1 to 0.9 over five arms) and 0 otherwise,
 * over a horizon of pulls (default 1000).
 *
 * @concept bandits
 * @example
 * ```ts
 * import { bernoulliBandit } from 'mdml-engine/methods/rl'
 * bernoulliBandit.create({ probs: [0.2, 0.8] }).oracle?.optimalExpectedReward({ t: 0 }) // => 0.8
 * ```
 */
export const bernoulliBandit: EnvDefinition<BanditState, number, number> = {
  id: 'BernoulliBandit-v0',
  description: 'Pull arms that pay 1 with unknown probabilities.',
  options: dict({ horizon: int(1, 1_000_000, { default: 1000 }) }),
  create: (options) => {
    const probs = list(options, 'probs', [0.1, 0.3, 0.5, 0.7, 0.9])
    return stationary(
      'BernoulliBandit-v0',
      'Pull arms that pay 1 with unknown probabilities.',
      probs,
      horizonOf(options),
      (a, k) => (u1(k) < (probs[a] ?? 0) ? 1 : 0),
    )
  },
}

/**
 * GaussianBandit-v0: arm `a` pays a normal reward with mean `means[a]` and standard deviation `sigma` (default 1);
 * without `means`, the means are drawn from N(0, 1) with the `seed` option (the 10-armed testbed of Sutton and Barto,
 * section 2.3).
 *
 * @concept bandits
 * @example
 * ```ts
 * import { gaussianBandit } from 'mdml-engine/methods/rl'
 * gaussianBandit.create().actionSpace // => { kind: 'discrete', n: 10, label: 'arm' }
 * ```
 */
export const gaussianBandit: EnvDefinition<BanditState, number, number> = {
  id: 'GaussianBandit-v0',
  description: 'Pull arms with normally distributed rewards of unknown means (the 10-armed testbed).',
  options: dict({
    arms: int(1, 1000, { default: 10 }),
    sigma: real(0, 10, { default: 1 }),
    seed: int(0, 2 ** 31, { default: 0 }),
    horizon: int(1, 1_000_000, { default: 1000 }),
  }),
  create: (options) => {
    const arms = typeof options?.['arms'] === 'number' ? options['arms'] : 10
    const seed = typeof options?.['seed'] === 'number' ? options['seed'] : 0
    const sigma = typeof options?.['sigma'] === 'number' ? options['sigma'] : 1
    const means = list(options, 'means', Array.from(normal(makeKey(seed), [arms]).data))
    return stationary(
      'GaussianBandit-v0',
      'Pull arms with normally distributed rewards of unknown means.',
      means,
      horizonOf(options),
      (a, k) => (means[a] ?? 0) + sigma * n1(k),
    )
  },
}

/** The state of a contextual linear bandit: the pull count, the current arm features and the last arm. */
export interface ContextualBanditState {
  readonly t: number
  readonly features: readonly (readonly number[])[]
  readonly last?: number
}

/**
 * ContextualLinearBandit-v0: each round shows a feature vector per arm (`nArms` vectors in `dimension` dimensions,
 * drawn uniformly on the unit sphere) and arm `a` pays `theta . x_a` plus N(0, noise^2) noise, for a hidden unit
 * vector `theta` drawn from the `seed` option (the setting of LinUCB and linear Thompson sampling).
 *
 * @concept bandits
 * @example
 * ```ts
 * import { contextualLinearBandit } from 'mdml-engine/methods/rl'
 * import { key } from 'mdml-engine/random'
 * contextualLinearBandit.create({ arms: 3, dimension: 4 }).reset(key(0)).observation.length // => 3
 * ```
 */
export const contextualLinearBandit: EnvDefinition<ContextualBanditState, readonly (readonly number[])[], number> = {
  id: 'ContextualLinearBandit-v0',
  description: 'Choose among arms described by feature vectors whose rewards are linear in the features.',
  options: dict({
    arms: int(2, 100, { default: 5 }),
    dimension: int(1, 100, { default: 5 }),
    noise: real(0, 10, { default: 0.1 }),
    seed: int(0, 2 ** 31, { default: 0 }),
    horizon: int(1, 1_000_000, { default: 1000 }),
  }),
  create: (options): BanditEnv<ContextualBanditState, readonly (readonly number[])[]> => {
    const nArms = typeof options?.['arms'] === 'number' ? options['arms'] : 5
    const d = typeof options?.['dimension'] === 'number' ? options['dimension'] : 5
    const noise = typeof options?.['noise'] === 'number' ? options['noise'] : 0.1
    const seed = typeof options?.['seed'] === 'number' ? options['seed'] : 0
    const unit = (k: Key): number[] => {
      const v = Array.from(normal(k, [d]).data)
      let n = 0
      for (const x of v) n += x * x
      return v.map((x) => x / Math.sqrt(n))
    }
    const theta = list(options, 'theta', unit(makeKey(seed)))
    const features = (k: Key): number[][] => Array.from({ length: nArms }, (_, a) => unit(foldIn(k, a)))
    const value = (x: readonly number[]): number => x.reduce((acc, v, i) => acc + v * (theta[i] ?? 0), 0)
    return {
      spec: {
        id: 'ContextualLinearBandit-v0',
        description: 'Choose among arms whose rewards are linear in their features.',
        maxEpisodeSteps: horizonOf(options),
      },
      observationSpace: tuple(
        Array.from({ length: nArms }, () => box(-1, 1, [d])),
        { label: 'one feature vector per arm' },
      ),
      actionSpace: discrete(nArms, { label: 'arm' }),
      reset: (k) => {
        const f = features(k)
        return { state: { t: 0, features: f }, observation: f, info: {} }
      },
      step: (s, arm, k) => {
        const f = features(foldIn(k, 1))
        const reward = value(s.features[arm] ?? []) + noise * n1(foldIn(k, 0))
        return {
          state: { t: s.t + 1, features: f, last: arm },
          observation: f,
          reward,
          terminated: false,
          truncated: false,
          info: {},
        }
      },
      render: (s) => renderArms(s.features.map(value), s.last, 'ContextualLinearBandit'),
      observe: (s) => s.features,
      oracle: {
        expectedReward: (s, arm) => value(s.features[arm] ?? []),
        optimalExpectedReward: (s) => maxOf(s.features.map(value)),
        optimalAction: (s) => argmaxOf(s.features.map(value)),
      },
    }
  },
}

/**
 * AdversarialBandit-v0: an oblivious adversary fixes every reward in advance (from the `seed` option): rewards in
 * [0, 1] whose best arm changes every `period` pulls, so no arm stays best. The oracle reports the best fixed arm in
 * hindsight, so cumulative regret is the weak regret that EXP3 bounds.
 *
 * @concept bandits
 * @example
 * ```ts
 * import { adversarialBandit } from 'mdml-engine/methods/rl'
 * adversarialBandit.create({ arms: 3, horizon: 10 }).spec.maxEpisodeSteps // => 10
 * ```
 */
export const adversarialBandit: EnvDefinition<BanditState, number, number> = {
  id: 'AdversarialBandit-v0',
  description: 'Pull arms whose rewards are fixed in advance by an adversary that keeps changing the best arm.',
  options: dict({
    arms: int(2, 100, { default: 3 }),
    period: int(1, 100_000, { default: 100 }),
    seed: int(0, 2 ** 31, { default: 0 }),
    horizon: int(1, 100_000, { default: 1000 }),
  }),
  create: (options): BanditEnv<BanditState, number> => {
    const nArms = typeof options?.['arms'] === 'number' ? options['arms'] : 3
    const period = typeof options?.['period'] === 'number' ? options['period'] : 100
    const seed = typeof options?.['seed'] === 'number' ? options['seed'] : 0
    const horizon = horizonOf(options)
    const noise = uniform(makeKey(seed), [horizon * nArms]).data
    // The favoured arm cycles; it pays around 0.8 and the others around 0.4.
    const table = (t: number, a: number): number => {
      const favoured = Math.floor(t / period) % nArms
      return Math.min(1, Math.max(0, (a === favoured ? 0.8 : 0.4) + 0.2 * ((noise[t * nArms + a] ?? 0.5) - 0.5)))
    }
    const totals = Array.from({ length: nArms }, (_, a) => {
      let s = 0
      for (let t = 0; t < horizon; t++) s += table(t, a)
      return s
    })
    const bestFixed = argmaxOf(totals)
    return {
      spec: {
        id: 'AdversarialBandit-v0',
        description: 'Rewards fixed in advance by an adversary.',
        maxEpisodeSteps: horizon,
      },
      observationSpace: discrete(1, { label: 'no context' }),
      actionSpace: discrete(nArms, { label: 'arm' }),
      reset: () => ({ state: { t: 0 }, observation: 0, info: {} }),
      step: (s, arm) => ({
        state: { t: s.t + 1, last: arm },
        observation: 0,
        reward: table(s.t % horizon, arm),
        terminated: false,
        truncated: false,
        info: {},
      }),
      render: (s) =>
        renderArms(
          Array.from({ length: nArms }, (_, a) => table(s.t % horizon, a)),
          s.last,
          'AdversarialBandit',
        ),
      observe: () => 0,
      oracle: {
        expectedReward: (s, arm) => table(s.t % horizon, arm),
        optimalExpectedReward: (s) => table(s.t % horizon, bestFixed),
        optimalAction: () => bestFixed,
      },
    }
  },
}

/** The state of a non-stationary bandit: the pull count, the current means and the last arm. */
export interface DriftingBanditState {
  readonly t: number
  readonly means: readonly number[]
  readonly last?: number
}

/**
 * NonStationaryBandit-v0 (Sutton and Barto, Exercise 2.5): all arms start with mean 0 and every mean takes an
 * independent N(0, drift^2) random-walk step after each pull (drift 0.01); rewards are N(mean, 1). Sample averages
 * fall behind; constant step sizes track the drift.
 *
 * @concept bandits
 * @example
 * ```ts
 * import { nonStationaryBandit } from 'mdml-engine/methods/rl'
 * import { key } from 'mdml-engine/random'
 * nonStationaryBandit.create().reset(key(0)).state.means.length // => 10
 * ```
 */
export const nonStationaryBandit: EnvDefinition<DriftingBanditState, number, number> = {
  id: 'NonStationaryBandit-v0',
  description: 'Pull arms whose mean rewards drift by a random walk.',
  options: dict({
    arms: int(1, 1000, { default: 10 }),
    drift: real(0, 1, { default: 0.01 }),
    horizon: int(1, 1_000_000, { default: 10_000 }),
  }),
  create: (options): BanditEnv<DriftingBanditState, number> => {
    const nArms = typeof options?.['arms'] === 'number' ? options['arms'] : 10
    const drift = typeof options?.['drift'] === 'number' ? options['drift'] : 0.01
    return {
      spec: {
        id: 'NonStationaryBandit-v0',
        description: 'Pull arms whose mean rewards drift by a random walk.',
        maxEpisodeSteps: horizonOf(options, 10_000),
      },
      observationSpace: discrete(1, { label: 'no context' }),
      actionSpace: discrete(nArms, { label: 'arm' }),
      reset: () => ({ state: { t: 0, means: new Array<number>(nArms).fill(0) }, observation: 0, info: {} }),
      step: (s, arm, k) => {
        const reward = (s.means[arm] ?? 0) + n1(foldIn(k, 0))
        const steps = normal(foldIn(k, 1), [nArms]).data
        const means = s.means.map((m, i) => m + drift * (steps[i] ?? 0))
        return {
          state: { t: s.t + 1, means, last: arm },
          observation: 0,
          reward,
          terminated: false,
          truncated: false,
          info: {},
        }
      },
      render: (s) => renderArms(s.means, s.last, 'NonStationaryBandit'),
      observe: () => 0,
      oracle: {
        expectedReward: (s, arm) => s.means[arm] ?? 0,
        optimalExpectedReward: (s) => maxOf(s.means),
        optimalAction: (s) => argmaxOf(s.means),
      },
    }
  },
}
