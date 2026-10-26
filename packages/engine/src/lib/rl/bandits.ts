/**
 * Multi-armed bandits, the simplest reinforcement learning problem: repeatedly choose among arms with unknown reward
 * distributions, balancing exploration against exploitation. Bandits as environments with an oracle for cumulative
 * pseudo-regret, KL divergences of the usual reward families, the UCB1 and KL-UCB indices, and the Lai-Robbins lower
 * bound on regret.
 */

import { MdmlError } from '../base/index.ts'
import { brentq } from '../numerics/roots/index.ts'
import type { Env, EnvOracle } from './env.ts'

/**
 * A multi-armed bandit as an environment: a continuing episode of `horizon` pulls (the time limit) whose actions are
 * arms, with an oracle giving each arm's expected reward in the current state (so regret can be computed). The
 * observation is the context (a constant for context-free bandits).
 *
 * @concept bandits
 */
export type BanditEnv<S, O> = Env<S, O, number> & { readonly oracle: EnvOracle<S, number> }

/**
 * Whether an environment is a bandit: a discrete action space and an oracle.
 *
 * @param env - The environment.
 * @returns Whether regret can be computed for it.
 * @concept bandits
 * @example
 * ```ts
 * import { isBandit } from 'mdml-engine/rl'
 * import { bernoulliBandit, cartPole } from 'mdml-engine/environments'
 * [isBandit(bernoulliBandit.create()), isBandit(cartPole.create())] // => [true, false]
 * ```
 */
export function isBandit<S, O>(env: Env<S, O, number>): env is BanditEnv<S, O> {
  return env.oracle !== undefined && env.actionSpace.kind === 'discrete'
}

/**
 * Cumulative pseudo-regret along a trajectory: `sum_t [max_a mu_t(a) - mu_t(a_t)]`, from the oracle's expected
 * rewards in the state before each pull (so noise in the rewards does not enter).
 *
 * @param oracle - The environment's oracle.
 * @param states - The state before each pull.
 * @param actions - The arms pulled.
 * @returns The cumulative regret after each pull.
 * @concept regret
 * @example
 * ```ts
 * import { cumulativeRegret } from 'mdml-engine/rl'
 * const oracle = { expectedReward: (_: null, a: number) => [0.25, 0.75][a] ?? 0, optimalExpectedReward: () => 0.75 }
 * cumulativeRegret(oracle, [null, null, null], [0, 1, 0]) // => [0.5, 0.5, 1]
 * ```
 */
export function cumulativeRegret<S, A>(oracle: EnvOracle<S, A>, states: readonly S[], actions: readonly A[]): number[] {
  if (states.length < actions.length) {
    throw new MdmlError('shape', 'cumulativeRegret: need the state before every action')
  }
  const out: number[] = []
  let total = 0
  for (const [t, a] of actions.entries()) {
    const s = states[t] as S
    total += oracle.optimalExpectedReward(s) - oracle.expectedReward(s, a)
    out.push(total)
  }
  return out
}

/** A one-parameter exponential family for the KL helpers. */
export type BanditFamily = 'bernoulli' | 'gaussian' | 'poisson' | 'exponential'

const clampUnit = (p: number): number => Math.min(Math.max(p, 1e-15), 1 - 1e-15)

/**
 * Kullback-Leibler divergence between Bernoulli distributions, `KL(Ber(p) || Ber(q))` (probabilities clamped away
 * from 0 and 1).
 *
 * @param p - First mean.
 * @param q - Second mean.
 * @returns The divergence in nats.
 * @concept kl-divergence
 * @example
 * ```ts
 * import { klBernoulli } from 'mdml-engine/rl'
 * klBernoulli(0.5, 0.5) // => 0
 * ```
 */
export function klBernoulli(p: number, q: number): number {
  const a = clampUnit(p)
  const b = clampUnit(q)
  return a * Math.log(a / b) + (1 - a) * Math.log((1 - a) / (1 - b))
}

/**
 * Kullback-Leibler divergence between Gaussians with a common variance: `(p - q)^2 / (2 variance)`.
 *
 * @param p - First mean.
 * @param q - Second mean.
 * @param variance - The common variance (default 1).
 * @returns The divergence in nats.
 * @concept kl-divergence
 * @example
 * ```ts
 * import { klGaussian } from 'mdml-engine/rl'
 * klGaussian(0, 2) // => 2
 * ```
 */
export function klGaussian(p: number, q: number, variance = 1): number {
  return (p - q) ** 2 / (2 * variance)
}

/**
 * Kullback-Leibler divergence between Poisson distributions with means `p` and `q`: `q - p + p log(p / q)`.
 *
 * @param p - First mean.
 * @param q - Second mean.
 * @returns The divergence in nats.
 * @concept kl-divergence
 * @example
 * ```ts
 * import { klPoisson } from 'mdml-engine/rl'
 * klPoisson(2, 2) // => 0
 * ```
 */
export function klPoisson(p: number, q: number): number {
  const a = Math.max(p, 1e-15)
  const b = Math.max(q, 1e-15)
  return b - a + a * Math.log(a / b)
}

/**
 * Kullback-Leibler divergence between exponential distributions with means `p` and `q`: `p/q - 1 - log(p/q)`.
 *
 * @param p - First mean.
 * @param q - Second mean.
 * @returns The divergence in nats.
 * @concept kl-divergence
 * @example
 * ```ts
 * import { klExponential } from 'mdml-engine/rl'
 * klExponential(1, 1) // => 0
 * ```
 */
export function klExponential(p: number, q: number): number {
  const r = p / q
  return r - 1 - Math.log(r)
}

/**
 * The KL divergence of a family.
 *
 * @param family - The family.
 * @param p - First mean.
 * @param q - Second mean.
 * @param variance - The variance, for the Gaussian family (default 1).
 * @returns The divergence.
 * @concept kl-divergence
 * @example
 * ```ts
 * import { klDivergence } from 'mdml-engine/rl'
 * klDivergence('gaussian', 1, 0, 0.5) // => 1
 * ```
 */
export function klDivergence(family: BanditFamily, p: number, q: number, variance = 1): number {
  switch (family) {
    case 'bernoulli':
      return klBernoulli(p, q)
    case 'gaussian':
      return klGaussian(p, q, variance)
    case 'poisson':
      return klPoisson(p, q)
    case 'exponential':
      return klExponential(p, q)
  }
}

/**
 * The UCB1 index (Auer, Cesa-Bianchi and Fischer 2002): `mean + sqrt(c log t / n)`; unpulled arms get `Infinity`.
 *
 * @param mean - The arm's empirical mean.
 * @param pulls - Times the arm was pulled.
 * @param t - Total pulls so far.
 * @param c - The exploration constant (default 2).
 * @returns The index.
 * @concept bandits
 * @example
 * ```ts
 * import { ucb1Index } from 'mdml-engine/rl'
 * ucb1Index(0.5, 2, Math.E, 2) // => 1.5
 * ```
 */
export function ucb1Index(mean: number, pulls: number, t: number, c = 2): number {
  if (pulls === 0) return Infinity
  return mean + Math.sqrt((c * Math.log(Math.max(t, 1))) / pulls)
}

/**
 * The KL-UCB index (Garivier and Cappe 2011): the largest mean `q >= mean` with
 * `pulls KL(mean, q) <= log t + c log log t`, found by Brent's method; unpulled arms get `Infinity`.
 *
 * @param mean - The arm's empirical mean.
 * @param pulls - Times the arm was pulled.
 * @param t - Total pulls so far.
 * @param options - The family (default Bernoulli), `c` (default 0), the Gaussian variance and an upper search bound.
 * @returns The index.
 * @concept bandits
 * @example
 * ```ts
 * import { klUcbIndex } from 'mdml-engine/rl'
 * Math.abs(klUcbIndex(0, 1, 100, { family: 'gaussian' }) - Math.sqrt(2 * Math.log(100))) < 1e-9 // => true
 * ```
 */
export function klUcbIndex(
  mean: number,
  pulls: number,
  t: number,
  options: {
    readonly family?: BanditFamily
    readonly c?: number
    readonly variance?: number
    readonly upper?: number
  } = {},
): number {
  if (pulls === 0) return Infinity
  const family = options.family ?? 'bernoulli'
  const c = options.c ?? 0
  const lt = Math.log(Math.max(t, 1))
  const bound = (lt + c * Math.log(Math.max(lt, 1e-300))) / pulls
  if (!(bound > 0)) return mean
  const variance = options.variance ?? 1
  if (family === 'gaussian') return mean + Math.sqrt(2 * variance * bound)
  const hi =
    options.upper ??
    (family === 'bernoulli'
      ? 1
      : Math.max(1, mean) * (family === 'poisson' ? 1 + 2 * bound + 10 : Math.exp(bound + 2) + 10))
  const f = (q: number): number => klDivergence(family, mean, q, variance) - bound
  if (f(hi) <= 0) return hi
  const lo = family === 'bernoulli' ? Math.min(Math.max(mean, 0), 1) : Math.max(mean, 1e-12)
  if (f(lo) >= 0) return lo
  return brentq(f, lo, hi, { xtol: 1e-12 }).root
}

/**
 * The Lai-Robbins asymptotic lower bound on the regret of any consistent policy (Lai and Robbins 1985):
 * `sum over suboptimal arms of (mu* - mu_a) / KL(mu_a, mu*) times log T`.
 *
 * @param means - The arms' means.
 * @param horizon - The number of pulls `T`.
 * @param options - The family (default Bernoulli) and the Gaussian variance.
 * @returns The bound.
 * @concept regret
 * @example
 * ```ts
 * import { laiRobbinsBound } from 'mdml-engine/rl'
 * laiRobbinsBound([0, 1], Math.E, { family: 'gaussian' }) // => 2
 * ```
 */
export function laiRobbinsBound(
  means: readonly number[],
  horizon: number,
  options: { readonly family?: BanditFamily; readonly variance?: number } = {},
): number {
  let best = -Infinity
  for (const m of means) best = Math.max(best, m)
  let constant = 0
  for (const m of means) {
    if (m < best) constant += (best - m) / klDivergence(options.family ?? 'bernoulli', m, best, options.variance ?? 1)
  }
  return constant * Math.log(horizon)
}
