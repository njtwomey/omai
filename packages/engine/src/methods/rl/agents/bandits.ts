/**
 * Multi-armed bandit agents: learners that repeatedly pick one of several arms with unknown reward distributions and
 * must balance trying arms against exploiting the best so far. Epsilon-greedy, UCB1, KL-UCB, Thompson sampling for
 * Bernoulli and Gaussian arms, EXP3 for adversarial rewards and the gradient bandit, the contextual linear bandits
 * `linUcb` and `linearThompson`, and `runBandit`, which runs an agent and measures its cumulative regret against the
 * environment's oracle.
 */

import { argmaxOf, type Json, MdmlError, maxOf } from '../../../lib/base/index.ts'
import { cholesky, choSolve, ddot, dnrm2, solveTriangular } from '../../../lib/numerics/linalg/index.ts'
import { categorical, foldIn, gammaVariates, type Key, key as makeKey, normal } from '../../../lib/random/index.ts'
import {
  type Agent,
  type BanditFamily,
  boltzmannProbabilities,
  cumulativeRegret,
  type Env,
  isBandit,
  klUcbIndex,
  ucb1Index,
} from '../../../lib/rl/index.ts'
import { flatDim, type Space } from '../../../lib/space/index.ts'
import { fromStorage, toFlat } from '../../../lib/tensor/index.ts'
import { argmaxRandom, type Rate, randomIndex, rateAt, unit } from './_tabular.ts'

// ------------------------------------------------------------------------------------------------ shared machinery

/** What a bandit agent needs from an environment: its (discrete) action space, the arms. */
export interface BanditSpaces {
  readonly observationSpace?: Space
  readonly actionSpace: Space
}

/** The state every context-free bandit agent keeps: pulls, sample means and spreads of each arm. */
export interface BanditState {
  /** Pulls of each arm. */
  readonly counts: Float64Array
  /** Sample mean reward of each arm. */
  readonly means: Float64Array
  /** Sum of squared deviations from the mean of each arm (Welford), for variance estimates. */
  readonly m2: Float64Array
  /** Pulls so far. */
  readonly t: number
  readonly totalReward: number
}

const arms = (env: BanditSpaces, name: string): number => {
  if (env.actionSpace.kind !== 'discrete') throw new MdmlError('value', `${name}: needs discrete arms`)
  return env.actionSpace.n
}

function emptyBandit(k: number): BanditState {
  return { counts: new Float64Array(k), means: new Float64Array(k), m2: new Float64Array(k), t: 0, totalReward: 0 }
}

function recordPull(s: BanditState, arm: number, reward: number): BanditState {
  const counts = s.counts.slice()
  const means = s.means.slice()
  const m2 = s.m2.slice()
  const n = (counts[arm] ?? 0) + 1
  const old = means[arm] ?? 0
  const mean = old + (reward - old) / n
  counts[arm] = n
  means[arm] = mean
  m2[arm] = (m2[arm] ?? 0) + (reward - old) * (reward - mean)
  return { counts, means, m2, t: s.t + 1, totalReward: s.totalReward + reward }
}

const BANDIT_METRICS = {
  meanReward: { description: 'average reward per pull so far', direction: 'higher' },
  topArmShare: { description: 'share of pulls on the most-pulled arm' },
} as const

function banditStats(s: BanditState): Record<string, number> {
  return { meanReward: s.t === 0 ? 0 : s.totalReward / s.t, topArmShare: s.t === 0 ? 0 : maxOf(s.counts) / s.t }
}

interface BanditSpec<G extends BanditState, O> {
  readonly name: string
  readonly config: Json
  readonly metrics?: Agent<G, O, number>['metrics']
  init(base: BanditState): G
  choose(state: G, observation: O, key: Key): number
  greedy(state: G, observation: O): number
  /** Fold a pull into the agent-specific part of the state (the shared counts are already updated). */
  update?(state: G, arm: number, reward: number, observation: O, key: Key): G
  stats?(state: G): Readonly<Record<string, number>>
}

function banditAgent<G extends BanditState, O>(k: number, spec: BanditSpec<G, O>): Agent<G, O, number> {
  return {
    name: spec.name,
    config: spec.config,
    metrics: { ...BANDIT_METRICS, ...(spec.metrics ?? {}) },
    init: () => spec.init(emptyBandit(k)),
    act: (s, o, key) => spec.choose(s, o, key),
    greedy: (s, o) => spec.greedy(s, o),
    learn: (s, t, key) => {
      const next = { ...s, ...recordPull(s, t.action, t.reward) }
      return spec.update === undefined ? next : spec.update(next, t.action, t.reward, t.observation, key)
    },
    stats: (s) => ({ ...banditStats(s), ...(spec.stats === undefined ? {} : spec.stats(s)) }),
  }
}

const entropy = (p: ArrayLike<number>): number => {
  let h = 0
  for (let i = 0; i < p.length; i++) {
    const x = p[i] ?? 0
    if (x > 0) h -= x * Math.log(x)
  }
  return h
}

// ------------------------------------------------------------------------------------------------- epsilon-greedy

/** Options of {@link epsilonGreedyBandit}. */
export interface EpsilonGreedyBanditOptions {
  /** Exploration rate: a constant (default 0.1) or a schedule over pulls. */
  readonly epsilon?: Rate
  /** A constant step size for the estimates (tracks drifting arms); default sample averages. */
  readonly alpha?: number
  /** Initial estimates (default 0; optimistic values drive early exploration). */
  readonly initialValue?: number
}

/** The state of {@link epsilonGreedyBandit}: the shared statistics and the action-value estimates. */
export interface EpsilonGreedyBanditState extends BanditState {
  readonly estimates: Float64Array
}

/**
 * The epsilon-greedy bandit agent of Sutton and Barto (chapter 2): estimates by sample averages (section 2.4) or a
 * constant step size for non-stationary arms (section 2.5), optimistic initial values (section 2.6), and the greedy
 * arm with probability `1 - epsilon` (ties broken at random), a uniformly random arm otherwise.
 *
 * @param env - The bandit (its arms).
 * @param options - Exploration rate or schedule, step size and initial values.
 * @returns The agent.
 * @concept bandits
 * @example
 * ```ts
 * import { epsilonGreedyBandit } from 'mdml-engine/methods/rl/agents'
 * import { bernoulliBandit } from 'mdml-engine/methods/rl'
 * import { key } from 'mdml-engine/random'
 * const agent = epsilonGreedyBandit(bernoulliBandit.create(), { alpha: 0.5, initialValue: 2 })
 * const t = { observation: 0, action: 1, reward: 1, nextObservation: 0, terminated: false, truncated: false }
 * agent.learn(agent.init(key(0)), t, key(1)).estimates[1] // => 1.5
 * ```
 */
export function epsilonGreedyBandit(
  env: BanditSpaces,
  options: EpsilonGreedyBanditOptions = {},
): Agent<EpsilonGreedyBanditState, unknown, number> {
  const k = arms(env, 'epsilonGreedyBandit')
  const epsilon = options.epsilon ?? 0.1
  const init = options.initialValue ?? 0
  return banditAgent<EpsilonGreedyBanditState, unknown>(k, {
    name: 'epsilonGreedyBandit',
    config: {
      arms: k,
      epsilon: typeof epsilon === 'number' ? epsilon : 'schedule',
      alpha: options.alpha ?? 'sample-average',
      initialValue: init,
    },
    metrics: { epsilon: { description: 'exploration rate' } },
    init: (b) => ({ ...b, estimates: new Float64Array(k).fill(init) }),
    choose: (s, _, key) =>
      unit(foldIn(key, 0)) < rateAt(epsilon, s.t)
        ? randomIndex(foldIn(key, 1), k)
        : argmaxRandom(s.estimates, 0, k, foldIn(key, 2)),
    greedy: (s) => argmaxOf(s.estimates),
    update: (s, arm, reward) => {
      const estimates = s.estimates.slice()
      const step = options.alpha ?? 1 / (s.counts[arm] ?? 1)
      estimates[arm] = (estimates[arm] ?? 0) + step * (reward - (estimates[arm] ?? 0))
      return { ...s, estimates }
    },
    stats: (s) => ({ epsilon: rateAt(epsilon, s.t) }),
  })
}

// ---------------------------------------------------------------------------------------------------- UCB family

/** Options of {@link ucb1}. */
export interface Ucb1Options {
  /** The exploration constant `c` of `mean + sqrt(c log t / n)` (default 2, Auer et al.'s UCB1). */
  readonly c?: number
}

/**
 * UCB1 (Auer, Cesa-Bianchi and Fischer 2002): pull every arm once, then the arm with the largest upper confidence
 * bound `mean + sqrt(c log t / n)` (`ucb1Index` of `lib/rl`; the first such arm on ties). Its regret grows
 * logarithmically.
 *
 * @param env - The bandit.
 * @param options - The exploration constant.
 * @returns The agent.
 * @concept upper-confidence-bounds
 * @example
 * ```ts
 * import { ucb1 } from 'mdml-engine/methods/rl/agents'
 * import { bernoulliBandit } from 'mdml-engine/methods/rl'
 * import { key } from 'mdml-engine/random'
 * const agent = ucb1(bernoulliBandit.create())
 * agent.act(agent.init(key(0)), 0, key(1)) // => 0
 * ```
 */
export function ucb1(env: BanditSpaces, options: Ucb1Options = {}): Agent<BanditState, unknown, number> {
  const k = arms(env, 'ucb1')
  const c = options.c ?? 2
  const indices = (s: BanditState): number[] =>
    Array.from({ length: k }, (_, a) => ucb1Index(s.means[a] ?? 0, s.counts[a] ?? 0, s.t, c))
  return banditAgent<BanditState, unknown>(k, {
    name: 'ucb1',
    config: { arms: k, c },
    init: (b) => b,
    choose: (s) => argmaxOf(indices(s)),
    greedy: (s) => argmaxOf(s.means),
  })
}

/** Options of {@link klUcb}. */
export interface KlUcbOptions {
  /** The reward family whose KL divergence defines the index (default Bernoulli). */
  readonly family?: BanditFamily
  /** The `c` of the `log t + c log log t` exploration budget (default 0). */
  readonly c?: number
  /** The reward variance, for the Gaussian family (default 1). */
  readonly variance?: number
}

/**
 * KL-UCB (Garivier and Cappe 2011): the arm whose index, the largest mean `q` with `n KL(mean, q) <= log t +
 * c log log t` for the chosen family (`klUcbIndex` of `lib/rl`), is highest. For Bernoulli arms it is asymptotically
 * optimal (it meets the Lai-Robbins bound) and beats UCB1.
 *
 * @param env - The bandit.
 * @param options - The family, `c` and the Gaussian variance.
 * @returns The agent.
 * @concept upper-confidence-bounds
 * @example
 * ```ts
 * import { klUcb } from 'mdml-engine/methods/rl/agents'
 * import { gaussianBandit } from 'mdml-engine/methods/rl'
 * klUcb(gaussianBandit.create(), { family: 'gaussian' }).name // => 'klUcb'
 * ```
 */
export function klUcb(env: BanditSpaces, options: KlUcbOptions = {}): Agent<BanditState, unknown, number> {
  const k = arms(env, 'klUcb')
  const family = options.family ?? 'bernoulli'
  const opts = { family, c: options.c ?? 0, variance: options.variance ?? 1 }
  return banditAgent<BanditState, unknown>(k, {
    name: 'klUcb',
    config: { arms: k, ...opts },
    init: (b) => b,
    choose: (s) =>
      argmaxOf(Array.from({ length: k }, (_, a) => klUcbIndex(s.means[a] ?? 0, s.counts[a] ?? 0, s.t, opts))),
    greedy: (s) => argmaxOf(s.means),
  })
}

// ---------------------------------------------------------------------------------------------- Thompson sampling

/** Options of {@link thompsonBernoulli}. */
export interface ThompsonBernoulliOptions {
  /** The Beta prior `[a, b]` of every arm (default `[1, 1]`, uniform). */
  readonly prior?: readonly [number, number]
}

/** The state of {@link thompsonBernoulli}: the shared statistics and each arm's Beta posterior. */
export interface ThompsonBernoulliState extends BanditState {
  readonly alpha: Float64Array
  readonly beta: Float64Array
}

/**
 * Thompson sampling for Bernoulli arms (Thompson 1933; Agrawal and Goyal 2012): each arm has a Beta posterior over its
 * success probability; each pull draws one sample per arm (as a ratio of gamma variates) and pulls the largest.
 * Rewards in [0, 1] that are not 0 or 1 are first turned into Bernoulli trials with that success probability (the
 * key of `learn` draws them), Agrawal and Goyal's reduction.
 *
 * @param env - The bandit.
 * @param options - The Beta prior.
 * @returns The agent.
 * @concept thompson-sampling
 * @example
 * ```ts
 * import { thompsonBernoulli } from 'mdml-engine/methods/rl/agents'
 * import { bernoulliBandit } from 'mdml-engine/methods/rl'
 * import { key } from 'mdml-engine/random'
 * const agent = thompsonBernoulli(bernoulliBandit.create())
 * const t = { observation: 0, action: 2, reward: 1, nextObservation: 0, terminated: false, truncated: false }
 * Array.from(agent.learn(agent.init(key(0)), t, key(1)).alpha) // => [1, 1, 2, 1, 1]
 * ```
 */
export function thompsonBernoulli(
  env: BanditSpaces,
  options: ThompsonBernoulliOptions = {},
): Agent<ThompsonBernoulliState, unknown, number> {
  const k = arms(env, 'thompsonBernoulli')
  const [a0, b0] = options.prior ?? [1, 1]
  return banditAgent<ThompsonBernoulliState, unknown>(k, {
    name: 'thompsonBernoulli',
    config: { arms: k, prior: [a0, b0] },
    init: (b) => ({ ...b, alpha: new Float64Array(k).fill(a0), beta: new Float64Array(k).fill(b0) }),
    choose: (s, _, key) => {
      const g = gammaVariates(key, [...s.alpha, ...s.beta])
      return argmaxOf(Array.from({ length: k }, (_, a) => (g[a] ?? 0) / ((g[a] ?? 0) + (g[k + a] ?? 0))))
    },
    greedy: (s) =>
      argmaxOf(Array.from({ length: k }, (_, a) => (s.alpha[a] ?? 0) / ((s.alpha[a] ?? 0) + (s.beta[a] ?? 0)))),
    update: (s, arm, reward, _, key) => {
      const r = Math.min(Math.max(reward, 0), 1)
      const success = r === 0 || r === 1 ? r === 1 : unit(key) < r
      const alpha = s.alpha.slice()
      const beta = s.beta.slice()
      if (success) alpha[arm] = (alpha[arm] ?? 0) + 1
      else beta[arm] = (beta[arm] ?? 0) + 1
      return { ...s, alpha, beta }
    },
  })
}

/** Options of {@link thompsonGaussian}. */
export interface ThompsonGaussianOptions {
  /** The known reward variance; when omitted the variance is unknown and estimated (a normal-gamma posterior). */
  readonly variance?: number
  /** Prior mean of every arm (default 0). */
  readonly priorMean?: number
  /** Prior variance of the means when the reward variance is known (default 1). */
  readonly priorVariance?: number
  /** Normal-gamma prior pseudo-count of the mean (default 1). */
  readonly kappa0?: number
  /** Normal-gamma prior shape (default 1). */
  readonly alpha0?: number
  /** Normal-gamma prior rate (default 1). */
  readonly beta0?: number
}

/** A Gaussian arm posterior: `N(mean, 1 / precision)` (known variance), or normal-gamma `(mean, kappa, shape, rate)`. */
export interface GaussianPosterior {
  readonly mean: number[]
  /** Precision of the mean (known variance), or the normal-gamma `kappa`. */
  readonly precision: number[]
  readonly shape?: number[]
  readonly rate?: number[]
}

/**
 * The posterior of every arm of a Gaussian Thompson-sampling state: conjugate normal updates with a known variance,
 * or the normal-gamma posterior over the mean and precision when the variance is estimated.
 *
 * @param state - The agent state (any bandit state: counts, means and spreads).
 * @param options - The options the agent was built with.
 * @returns Posterior parameters per arm.
 * @concept thompson-sampling
 * @example
 * ```ts
 * import { gaussianPosterior, thompsonGaussian } from 'mdml-engine/methods/rl/agents'
 * import { gaussianBandit } from 'mdml-engine/methods/rl'
 * import { key } from 'mdml-engine/random'
 * const agent = thompsonGaussian(gaussianBandit.create({ means: [0, 1] }), { variance: 1 })
 * const t = { observation: 0, action: 1, reward: 2, nextObservation: 0, terminated: false, truncated: false }
 * gaussianPosterior(agent.learn(agent.init(key(0)), t, key(1)), { variance: 1 }) // => { mean: [0, 1], precision: [1, 2] }
 * ```
 */
export function gaussianPosterior(state: BanditState, options: ThompsonGaussianOptions = {}): GaussianPosterior {
  const k = state.counts.length
  const mu0 = options.priorMean ?? 0
  const n = Array.from(state.counts)
  const xbar = Array.from(state.means)
  if (options.variance !== undefined) {
    const tau0 = 1 / (options.priorVariance ?? 1)
    const tau = 1 / options.variance
    const precision = n.map((ni) => tau0 + ni * tau)
    const mean = precision.map((p, a) => (tau0 * mu0 + tau * (n[a] ?? 0) * (xbar[a] ?? 0)) / p)
    return { mean, precision }
  }
  const kappa0 = options.kappa0 ?? 1
  const alpha0 = options.alpha0 ?? 1
  const beta0 = options.beta0 ?? 1
  const kappa = n.map((ni) => kappa0 + ni)
  const mean = kappa.map((kp, a) => (kappa0 * mu0 + (n[a] ?? 0) * (xbar[a] ?? 0)) / kp)
  const shape = n.map((ni) => alpha0 + ni / 2)
  const rate = Array.from({ length: k }, (_, a) => {
    const ni = n[a] ?? 0
    const d = (xbar[a] ?? 0) - mu0
    return beta0 + (state.m2[a] ?? 0) / 2 + (kappa0 * ni * d * d) / (2 * (kappa[a] ?? 1))
  })
  return { mean, precision: kappa, shape, rate }
}

/**
 * Thompson sampling for Gaussian arms: with a known reward variance each arm's mean has a normal posterior; with the
 * variance unknown (the default) a normal-gamma posterior, sampled as a precision `tau ~ Gamma(shape, rate)` and then
 * a mean `N(mean, 1 / (kappa tau))`. Each pull samples every arm's mean and pulls the largest.
 *
 * @param env - The bandit.
 * @param options - The known variance or the normal-gamma prior.
 * @returns The agent.
 * @concept thompson-sampling
 * @example
 * ```ts
 * import { thompsonGaussian } from 'mdml-engine/methods/rl/agents'
 * import { gaussianBandit } from 'mdml-engine/methods/rl'
 * thompsonGaussian(gaussianBandit.create(), { variance: 1 }).name // => 'thompsonGaussian'
 * ```
 */
export function thompsonGaussian(
  env: BanditSpaces,
  options: ThompsonGaussianOptions = {},
): Agent<BanditState, unknown, number> {
  const k = arms(env, 'thompsonGaussian')
  return banditAgent<BanditState, unknown>(k, {
    name: 'thompsonGaussian',
    config: {
      arms: k,
      variance: options.variance ?? 'estimated',
      priorMean: options.priorMean ?? 0,
      ...(options.variance === undefined
        ? { kappa0: options.kappa0 ?? 1, alpha0: options.alpha0 ?? 1, beta0: options.beta0 ?? 1 }
        : { priorVariance: options.priorVariance ?? 1 }),
    },
    init: (b) => b,
    choose: (s, _, key) => {
      const p = gaussianPosterior(s, options)
      const z = normal(foldIn(key, 0), [k]).data
      if (p.shape === undefined || p.rate === undefined) {
        return argmaxOf(p.mean.map((m, a) => m + (z[a] ?? 0) / Math.sqrt(p.precision[a] ?? 1)))
      }
      const g = gammaVariates(foldIn(key, 1), p.shape)
      const rate = p.rate
      return argmaxOf(
        p.mean.map((m, a) => {
          const tau = (g[a] ?? 1) / (rate[a] ?? 1)
          return m + (z[a] ?? 0) / Math.sqrt((p.precision[a] ?? 1) * tau)
        }),
      )
    },
    greedy: (s) => argmaxOf(gaussianPosterior(s, options).mean),
  })
}

// --------------------------------------------------------------------------------------------------------- EXP3

/** Options of {@link exp3}. */
export interface Exp3Options {
  /** The exploration mixture `gamma` in (0, 1] (default 0.1). */
  readonly gamma?: number
  /** The range of the rewards, rescaled to [0, 1] (default `[0, 1]`). */
  readonly rewardRange?: readonly [number, number]
}

/** The state of {@link exp3}: the shared statistics, log-weights and the current arm probabilities. */
export interface Exp3State extends BanditState {
  readonly logWeights: Float64Array
  readonly probabilities: Float64Array
}

/**
 * EXP3 (Auer, Cesa-Bianchi, Freund and Schapire 2002), for adversarial rewards: arm probabilities
 * `p_i = (1 - gamma) w_i / sum w + gamma / K`; after pulling arm `a` with reward `x` in [0, 1] the importance-weighted
 * estimate `x / p_a` multiplies its weight by `exp(gamma x / (p_a K))`. Weights are kept as logarithms. Its weak
 * regret against the best fixed arm is `O(sqrt(T K log K))`.
 *
 * @param env - The bandit.
 * @param options - `gamma` and the reward range.
 * @returns The agent.
 * @concept adversarial-bandits
 * @example
 * ```ts
 * import { exp3 } from 'mdml-engine/methods/rl/agents'
 * import { adversarialBandit } from 'mdml-engine/methods/rl'
 * import { key } from 'mdml-engine/random'
 * const agent = exp3(adversarialBandit.create({ arms: 2 }), { gamma: 0.5 })
 * Array.from(agent.init(key(0)).probabilities) // => [0.5, 0.5]
 * ```
 */
export function exp3(env: BanditSpaces, options: Exp3Options = {}): Agent<Exp3State, unknown, number> {
  const k = arms(env, 'exp3')
  const gamma = options.gamma ?? 0.1
  if (!(gamma > 0 && gamma <= 1)) throw new MdmlError('value', 'exp3: gamma must be in (0, 1]')
  const [lo, hi] = options.rewardRange ?? [0, 1]
  const probabilitiesOf = (lw: Float64Array): Float64Array => {
    const m = maxOf(lw)
    let total = 0
    const w = Array.from(lw, (x) => Math.exp(x - m))
    for (const x of w) total += x
    return Float64Array.from(w, (x) => ((1 - gamma) * x) / total + gamma / k)
  }
  return banditAgent<Exp3State, unknown>(k, {
    name: 'exp3',
    config: { arms: k, gamma, rewardRange: [lo, hi] },
    metrics: { entropy: { description: 'entropy of the arm probabilities' } },
    init: (b) => {
      const logWeights = new Float64Array(k)
      return { ...b, logWeights, probabilities: probabilitiesOf(logWeights) }
    },
    choose: (s, _, key) => categorical(key, Array.from(s.probabilities), []).data[0] ?? 0,
    greedy: (s) => argmaxOf(s.probabilities),
    update: (s, arm, reward) => {
      const x = Math.min(Math.max((reward - lo) / (hi - lo), 0), 1)
      const logWeights = s.logWeights.slice()
      logWeights[arm] = (logWeights[arm] ?? 0) + (gamma * x) / ((s.probabilities[arm] ?? 1) * k)
      return { ...s, logWeights, probabilities: probabilitiesOf(logWeights) }
    },
    stats: (s) => ({ entropy: entropy(s.probabilities) }),
  })
}

// ---------------------------------------------------------------------------------------------- gradient bandits

/** Options of {@link gradientBandit}. */
export interface GradientBanditOptions {
  /** The step size of the preferences (default 0.1). */
  readonly alpha?: number
  /** Subtract the average reward as a baseline (default true). */
  readonly baseline?: boolean
  /** A constant step size for the baseline (default the sample average of all rewards). */
  readonly baselineAlpha?: number
}

/** The state of {@link gradientBandit}: the shared statistics, preferences, the baseline and the policy. */
export interface GradientBanditState extends BanditState {
  readonly preferences: Float64Array
  readonly baseline: number
  readonly probabilities: Float64Array
}

/**
 * The gradient bandit algorithm (Sutton and Barto, section 2.8): a softmax policy over preferences `H`, updated by
 * stochastic gradient ascent on the expected reward, `H_a += alpha (R - baseline) (1 - pi_a)` for the pulled arm and
 * `H_b -= alpha (R - baseline) pi_b` for the others, with the average reward as the baseline.
 *
 * @param env - The bandit.
 * @param options - Step size and baseline.
 * @returns The agent.
 * @concept policy-gradient
 * @example
 * ```ts
 * import { gradientBandit } from 'mdml-engine/methods/rl/agents'
 * import { bernoulliBandit } from 'mdml-engine/methods/rl'
 * import { key } from 'mdml-engine/random'
 * const agent = gradientBandit(bernoulliBandit.create({ probs: [0.5, 0.5] }), { alpha: 1, baseline: false })
 * const t = { observation: 0, action: 0, reward: 1, nextObservation: 0, terminated: false, truncated: false }
 * Array.from(agent.learn(agent.init(key(0)), t, key(1)).preferences) // => [0.5, -0.5]
 * ```
 */
export function gradientBandit(
  env: BanditSpaces,
  options: GradientBanditOptions = {},
): Agent<GradientBanditState, unknown, number> {
  const k = arms(env, 'gradientBandit')
  const alpha = options.alpha ?? 0.1
  const useBaseline = options.baseline ?? true
  const policy = (h: Float64Array): Float64Array => Float64Array.from(boltzmannProbabilities(h, 1))
  return banditAgent<GradientBanditState, unknown>(k, {
    name: 'gradientBandit',
    config: { arms: k, alpha, baseline: useBaseline, baselineAlpha: options.baselineAlpha ?? 'sample-average' },
    metrics: { entropy: { description: 'entropy of the softmax policy' } },
    init: (b) => {
      const preferences = new Float64Array(k)
      return { ...b, preferences, baseline: 0, probabilities: policy(preferences) }
    },
    choose: (s, _, key) => categorical(key, Array.from(s.probabilities), []).data[0] ?? 0,
    greedy: (s) => argmaxOf(s.preferences),
    update: (s, arm, reward) => {
      // The baseline is the average of the rewards before this one, and the first reward itself on the first pull
      // (Sutton and Barto, equation 2.12).
      const advantage = useBaseline ? reward - (s.t === 1 ? reward : s.baseline) : reward
      const preferences = Float64Array.from(
        s.preferences,
        (h, b) => h + alpha * advantage * ((b === arm ? 1 : 0) - (s.probabilities[b] ?? 0)),
      )
      const step = options.baselineAlpha ?? 1 / s.t
      return {
        ...s,
        preferences,
        probabilities: policy(preferences),
        baseline: s.baseline + step * (reward - s.baseline),
      }
    },
    stats: (s) => ({ entropy: entropy(s.probabilities) }),
  })
}

// ------------------------------------------------------------------------------------------------ linear bandits

/** Per-arm feature vectors, the observation of a contextual linear bandit. */
export type ArmFeatures = readonly (readonly number[])[]

/** Options shared by {@link linUcb} and {@link linearThompson}. */
export interface LinearBanditOptions {
  /** The ridge penalty `lambda` (the prior precision; default 1). */
  readonly lambda?: number
  /** One model shared by all arms (default, as in contextualLinearBandit) or a disjoint model per arm (Li et al.). */
  readonly model?: 'shared' | 'disjoint'
  /** The feature dimension (default from the observation space). */
  readonly dimension?: number
}

/** The state of a linear bandit agent: ridge statistics, estimates and Cholesky factors of each model. */
export interface LinearBanditState extends BanditState {
  readonly dimension: number
  /** `A = lambda I + sum x x^T` of each model (row-major `d * d`). */
  readonly A: readonly Float64Array[]
  /** `b = sum r x` of each model. */
  readonly b: readonly Float64Array[]
  /** The ridge estimate `theta = A^{-1} b` of each model. */
  readonly theta: readonly Float64Array[]
  /** The lower Cholesky factor of each `A`. */
  readonly L: readonly Float64Array[]
}

function dimensionOf(env: BanditSpaces, options: LinearBanditOptions, name: string): number {
  if (options.dimension !== undefined) return options.dimension
  const space = env.observationSpace
  if (space?.kind === 'tuple' && space.spaces[0] !== undefined) return flatDim(space.spaces[0])
  throw new MdmlError('value', `${name}: give the feature dimension`)
}

function linearInit(base: BanditState, models: number, d: number, lambda: number): LinearBanditState {
  const eye = new Float64Array(d * d)
  for (let i = 0; i < d; i++) eye[i * d + i] = lambda
  const root = new Float64Array(d * d)
  for (let i = 0; i < d; i++) root[i * d + i] = Math.sqrt(lambda)
  return {
    ...base,
    dimension: d,
    A: Array.from({ length: models }, () => eye.slice()),
    b: Array.from({ length: models }, () => new Float64Array(d)),
    theta: Array.from({ length: models }, () => new Float64Array(d)),
    L: Array.from({ length: models }, () => root.slice()),
  }
}

function linearUpdate(s: LinearBanditState, m: number, x: readonly number[], reward: number): LinearBanditState {
  const d = s.dimension
  const A = (s.A[m] as Float64Array).slice()
  const b = (s.b[m] as Float64Array).slice()
  for (let i = 0; i < d; i++) {
    b[i] = (b[i] ?? 0) + reward * (x[i] ?? 0)
    for (let j = 0; j < d; j++) A[i * d + j] = (A[i * d + j] ?? 0) + (x[i] ?? 0) * (x[j] ?? 0)
  }
  const L = cholesky(fromStorage(A, [d, d], 'float64'))
  const theta = Float64Array.from(toFlat(choSolve(L, fromStorage(b, [d], 'float64'))))
  const replace = <T>(xs: readonly T[], v: T): T[] => xs.map((old, i) => (i === m ? v : old))
  return {
    ...s,
    A: replace(s.A, A),
    b: replace(s.b, b),
    theta: replace(s.theta, theta),
    L: replace(s.L, Float64Array.from(toFlat(L))),
  }
}

/** `sqrt(x^T A^{-1} x) = |L^{-1} x|` for the Cholesky factor `L` of `A`. */
function confidenceWidth(L: Float64Array, x: readonly number[], d: number): number {
  return dnrm2(toFlat(solveTriangular(fromStorage(L, [d, d], 'float64'), [...x], { lower: true })))
}

function linearAgent(
  env: BanditSpaces,
  options: LinearBanditOptions,
  name: string,
  extra: Readonly<Record<string, Json>>,
  score: (s: LinearBanditState, features: ArmFeatures, key: Key | undefined) => number[],
): Agent<LinearBanditState, ArmFeatures, number> {
  const k = arms(env, name)
  const d = dimensionOf(env, options, name)
  const shared = (options.model ?? 'shared') === 'shared'
  const lambda = options.lambda ?? 1
  return banditAgent<LinearBanditState, ArmFeatures>(k, {
    name,
    config: { arms: k, dimension: d, lambda, model: shared ? 'shared' : 'disjoint', ...extra },
    init: (b) => linearInit(b, shared ? 1 : k, d, lambda),
    choose: (s, features, key) => argmaxOf(score(s, features, key)),
    greedy: (s, features) => argmaxOf(features.map((x, a) => ddot(s.theta[shared ? 0 : a] as Float64Array, x))),
    update: (s, arm, reward, features) => linearUpdate(s, shared ? 0 : arm, features[arm] ?? [], reward),
  })
}

/** Options of {@link linUcb}. */
export interface LinUcbOptions extends LinearBanditOptions {
  /** The width multiplier `alpha` of the confidence bound (default 1). */
  readonly alpha?: number
}

/**
 * LinUCB (Li, Chu, Langford and Schapire 2010) for arms described by feature vectors with linear expected rewards:
 * ridge regression `theta = A^{-1} b` with `A = lambda I + sum x x^T` (kept as a Cholesky factor, `lib/numerics/linalg`)
 * and the arm maximising `theta . x + alpha sqrt(x^T A^{-1} x)`. One model is shared by the arms (the setting of
 * contextualLinearBandit), or each arm has its own (Li et al.'s disjoint model).
 *
 * @param env - The contextual bandit (observations are one feature vector per arm).
 * @param options - `alpha`, the ridge penalty, shared or disjoint models and the dimension.
 * @returns The agent.
 * @concept contextual-bandits
 * @example
 * ```ts
 * import { linUcb } from 'mdml-engine/methods/rl/agents'
 * import { contextualLinearBandit } from 'mdml-engine/methods/rl'
 * import { key } from 'mdml-engine/random'
 * const agent = linUcb(contextualLinearBandit.create({ arms: 2, dimension: 2 }), { lambda: 3 })
 * const t = { observation: [[1, 0], [0, 1]], action: 0, reward: 2, nextObservation: [[1, 0], [0, 1]], terminated: false, truncated: false }
 * Array.from(agent.learn(agent.init(key(0)), t, key(1)).theta[0] ?? []) // => [0.5, 0]
 * ```
 */
export function linUcb(env: BanditSpaces, options: LinUcbOptions = {}): Agent<LinearBanditState, ArmFeatures, number> {
  const alpha = options.alpha ?? 1
  const shared = (options.model ?? 'shared') === 'shared'
  return linearAgent(env, options, 'linUcb', { alpha }, (s, features) =>
    features.map((x, a) => {
      const m = shared ? 0 : a
      return ddot(s.theta[m] as Float64Array, x) + alpha * confidenceWidth(s.L[m] as Float64Array, x, s.dimension)
    }),
  )
}

/** Options of {@link linearThompson}. */
export interface LinearThompsonOptions extends LinearBanditOptions {
  /** The scale `v` of the posterior `N(theta, v^2 A^{-1})` (default 1; the reward noise's standard deviation). */
  readonly v?: number
}

/**
 * Linear Thompson sampling (Agrawal and Goyal 2013): sample `theta~ ~ N(theta, v^2 A^{-1})` from the ridge posterior
 * (`theta + v L^{-T} z` with the Cholesky factor `L` of `A`) and pull the arm maximising `theta~ . x`; one sample is
 * shared by all arms of a shared model.
 *
 * @param env - The contextual bandit.
 * @param options - The posterior scale `v`, the ridge penalty, shared or disjoint models and the dimension.
 * @returns The agent.
 * @concept thompson-sampling
 * @example
 * ```ts
 * import { linearThompson } from 'mdml-engine/methods/rl/agents'
 * import { contextualLinearBandit } from 'mdml-engine/methods/rl'
 * linearThompson(contextualLinearBandit.create(), { v: 0.5 }).name // => 'linearThompson'
 * ```
 */
export function linearThompson(
  env: BanditSpaces,
  options: LinearThompsonOptions = {},
): Agent<LinearBanditState, ArmFeatures, number> {
  const v = options.v ?? 1
  const shared = (options.model ?? 'shared') === 'shared'
  return linearAgent(env, options, 'linearThompson', { v }, (s, features, key) => {
    const d = s.dimension
    const draws = s.theta.map((theta, m) => {
      const z = normal(foldIn(key ?? makeKey(0), m), [d])
      const L = fromStorage(s.L[m] as Float64Array, [d, d], 'float64')
      const noise = toFlat(solveTriangular(L, z, { lower: true, transpose: true }))
      return Float64Array.from(theta, (x, i) => x + v * (noise[i] ?? 0))
    })
    return features.map((x, a) => ddot(draws[shared ? 0 : a] as Float64Array, x))
  })
}

// ------------------------------------------------------------------------------------------------------ running

/** Result of {@link runBandit}. */
export interface BanditRun<G> {
  readonly agent: G
  readonly actions: number[]
  readonly rewards: number[]
  /** Cumulative pseudo-regret after each pull (from the environment's oracle). */
  readonly regret: number[]
}

/**
 * Run a learning agent on a bandit for a number of pulls and measure its cumulative pseudo-regret with the
 * environment's oracle (`cumulativeRegret` of `lib/rl`), which `trainAgent` cannot see because it needs the
 * environment state before every pull. Keys derive from the run key and the pull number.
 *
 * @param env - The bandit (with an oracle).
 * @param agent - The agent.
 * @param options - Pulls (default the horizon, else 1000) and the key or seed.
 * @returns The final agent state, the actions, rewards and cumulative regret.
 * @throws MdmlError with code `value` when the environment has no oracle.
 * @concept regret
 * @example
 * ```ts
 * import { runBandit, ucb1 } from 'mdml-engine/methods/rl/agents'
 * import { bernoulliBandit } from 'mdml-engine/methods/rl'
 * const env = bernoulliBandit.create({ probs: [0.25, 0.75] })
 * runBandit(env, ucb1(env), { pulls: 2 }).regret // => [0.5, 0.5]
 * ```
 */
export function runBandit<G, S, O>(
  env: Env<S, O, number>,
  agent: Agent<G, O, number>,
  options: { readonly pulls?: number; readonly seed?: number; readonly key?: Key } = {},
): BanditRun<G> {
  if (!isBandit(env)) throw new MdmlError('value', `runBandit: ${env.spec.id} has no oracle or discrete arms`)
  const oracle = env.oracle
  const pulls = options.pulls ?? env.spec.maxEpisodeSteps ?? 1000
  const root = options.key ?? makeKey(options.seed ?? 0)
  let g = agent.init(foldIn(root, 0))
  let r = env.reset(foldIn(root, 1))
  let state = r.state
  let observation = r.observation
  const states: S[] = []
  const actions: number[] = []
  const rewards: number[] = []
  for (let t = 0; t < pulls; t++) {
    const k = foldIn(foldIn(root, 2), t)
    const action = agent.act(g, observation, foldIn(k, 0))
    const step = env.step(state, action, foldIn(k, 1))
    states.push(state)
    actions.push(action)
    rewards.push(step.reward)
    g = agent.learn(
      g,
      {
        observation,
        action,
        reward: step.reward,
        nextObservation: step.observation,
        terminated: step.terminated,
        truncated: step.truncated,
      },
      foldIn(k, 2),
    )
    if (step.terminated || step.truncated) {
      r = env.reset(foldIn(k, 3))
      state = r.state
      observation = r.observation
    } else {
      state = step.state
      observation = step.observation
    }
  }
  return { agent: g, actions, rewards, regret: cumulativeRegret(oracle, states, actions) }
}
