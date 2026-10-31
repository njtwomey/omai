/**
 * Off-policy Monte Carlo (Sutton and Barto, sections 5.5 to 5.7): learning about a target policy from episodes generated
 * by another, by importance sampling. Prediction estimates the state values of a target policy from behaviour episodes
 * with ordinary importance sampling (the mean of the ratio-weighted returns: unbiased, with a variance that can be
 * infinite) or weighted importance sampling (the ratio-weighted average: biased, consistent, with bounded variance), and
 * control learns the greedy policy from any soft behaviour policy by incremental weighted (or ordinary) importance
 * sampling, learning from the tail of each episode where the behaviour agreed with the greedy target.
 */

import { argmaxOf, type Json, MdmlError } from '../base/index.ts'
import type { Key } from '../random/index.ts'
import type { Agent } from '../rl/index.ts'
import { type TabularCodingOptions, type TabularEnvLike, tabularCoding, unit } from './_tabular.ts'

/** How the returns of an episode are weighted by the importance ratio of the rest of the episode. */
export type ImportanceWeighting = 'weighted' | 'ordinary'

/** One step of a stored episode: the state and action, the reward that followed and the two policies' probabilities of the action. */
interface McStep {
  readonly s: number
  readonly a: number
  readonly r: number
  /** The behaviour's probability of the action. */
  readonly b: number
  /** The target's probability of the action (for control, whether the action is the greedy one, 0 or 1). */
  readonly p: number
}

const sampleFrom = (probs: readonly number[], k: Key): number => {
  const u = unit(k)
  let acc = 0
  for (const [a, p] of probs.entries()) {
    acc += p
    if (u < acc) return a
  }
  return probs.length - 1
}

/** Options of {@link offPolicyMonteCarloPrediction}. */
export interface OffPolicyPredictionOptions<O> extends TabularCodingOptions<O> {
  /** The probabilities of the policy being evaluated, per observation. */
  readonly target: (observation: O) => readonly number[]
  /** The probabilities of the policy that generates the episodes (default uniform); the agent acts by it. */
  readonly behaviour?: (observation: O) => readonly number[]
  /** `weighted` (default) or `ordinary` importance sampling. */
  readonly weighting?: ImportanceWeighting
  /** The discount (default 1). */
  readonly gamma?: number
  /** Count only the first visit of a state in an episode (default true) or every visit. */
  readonly firstVisit?: boolean
}

/** The state of {@link offPolicyMonteCarloPrediction}. */
export interface OffPolicyPredictionState {
  readonly nStates: number
  /** The sums of ratio-weighted returns per state. */
  readonly numerator: Float64Array
  /** The sums of ratios (weighted) or the counts of visits (ordinary) per state. */
  readonly denominator: Float64Array
  /** The estimate of every state's value. */
  readonly V: Float64Array
  /** The importance-weighted return of the last update, and the largest ratio seen. */
  readonly lastSample: number
  readonly maxRatio: number
  readonly episode: readonly McStep[]
  readonly episodes: number
  readonly steps: number
}

/**
 * Off-policy Monte Carlo prediction by importance sampling: the agent acts by the behaviour policy, and at the end of
 * each episode every (first) visit of a state `s` at time `t` contributes the sample `rho_{t:T-1} G_t`, where
 * `rho_{t:T-1}` is the product of `target / behaviour` probabilities of the actions from `t` to the end. The estimate
 * is the sum of samples over the number of visits (ordinary) or over the sum of ratios (weighted). Ordinary importance
 * sampling is unbiased but its variance can be unbounded when ratios compound, as in `infiniteVariance`; weighted
 * importance sampling is biased at first and has finite variance. An episode that ends by truncation is treated as ended.
 *
 * @param env - The environment.
 * @param options - The target and behaviour policies, weighting, discount and observation coding.
 * @returns The agent; its estimates are `state.V`.
 * @throws MdmlError with code `value` for a behaviour that gives an action probability 0 that the target takes.
 * @concept off-policy-learning
 * @example
 * ```ts
 * import { offPolicyMonteCarloPrediction } from 'mdml-engine/agents'
 * import { infiniteVariance } from 'mdml-engine/environments'
 * import { key } from 'mdml-engine/random'
 * const agent = offPolicyMonteCarloPrediction(infiniteVariance.create(), { target: () => [1, 0] })
 * const t = { observation: 0, action: 0, reward: 1, nextObservation: 0, terminated: true, truncated: false }
 * agent.learn(agent.init(key(0)), t, key(1)).V[0] // => 1
 * ```
 */
export function offPolicyMonteCarloPrediction<O>(
  env: TabularEnvLike<O>,
  options: OffPolicyPredictionOptions<O>,
): Agent<OffPolicyPredictionState, O, number> {
  const coding = tabularCoding(env, options, 'offPolicyMonteCarloPrediction')
  const { nStates, nActions } = coding
  const gamma = options.gamma ?? 1
  const weighting = options.weighting ?? 'weighted'
  const firstVisit = options.firstVisit ?? true
  const uniformProbs = Array.from({ length: nActions }, () => 1 / nActions)
  const behaviour = options.behaviour ?? (() => uniformProbs)
  return {
    name: 'offPolicyMonteCarloPrediction',
    config: { nStates, nActions, gamma, weighting, firstVisit } as Json,
    metrics: {
      sample: { description: 'the importance-weighted return of the last update' },
      maxRatio: { description: 'the largest importance ratio of an episode so far' },
    },
    init: () => ({
      nStates,
      numerator: new Float64Array(nStates),
      denominator: new Float64Array(nStates),
      V: new Float64Array(nStates),
      lastSample: 0,
      maxRatio: 0,
      episode: [],
      episodes: 0,
      steps: 0,
    }),
    act: (_, o, k) => sampleFrom(behaviour(o), k),
    greedy: (_, o) => argmaxOf(options.target(o)),
    learn: (s, t) => {
      const b = behaviour(t.observation)[t.action] ?? 0
      if (!(b > 0))
        throw new MdmlError('value', 'offPolicyMonteCarloPrediction: the behaviour gave the action probability 0')
      const p = options.target(t.observation)[t.action] ?? 0
      const episode = [...s.episode, { s: coding.encode(t.observation), a: t.action, r: t.reward, b, p }]
      if (!(t.terminated || t.truncated)) return { ...s, episode, steps: s.steps + 1 }
      const numerator = s.numerator.slice()
      const denominator = s.denominator.slice()
      const first = new Map<number, number>()
      for (const [j, e] of episode.entries()) if (!first.has(e.s)) first.set(e.s, j)
      let g = 0
      let rho = 1
      let last = s.lastSample
      let maxRatio = s.maxRatio
      for (let j = episode.length - 1; j >= 0; j--) {
        const e = episode[j] as McStep
        g = e.r + gamma * g
        rho *= e.p / e.b
        maxRatio = Math.max(maxRatio, rho)
        if (firstVisit && first.get(e.s) !== j) continue
        numerator[e.s] = (numerator[e.s] ?? 0) + rho * g
        denominator[e.s] = (denominator[e.s] ?? 0) + (weighting === 'ordinary' ? 1 : rho)
        last = rho * g
      }
      const V = Float64Array.from(numerator, (x, i) => ((denominator[i] ?? 0) > 0 ? x / (denominator[i] ?? 1) : 0))
      return {
        ...s,
        numerator,
        denominator,
        V,
        lastSample: last,
        maxRatio,
        episode: [],
        episodes: s.episodes + 1,
        steps: s.steps + 1,
      }
    },
    stats: (s) => ({ sample: s.lastSample, maxRatio: s.maxRatio }),
  }
}

// ------------------------------------------------------------------------------------------------------ control

/** Options of {@link offPolicyMonteCarloControl}. */
export interface OffPolicyControlOptions<O> extends TabularCodingOptions<O> {
  /** The soft policy that generates the episodes, per observation (default uniform over the actions). */
  readonly behaviour?: (observation: O) => readonly number[]
  /** `weighted` (default) or `ordinary` importance sampling. */
  readonly weighting?: ImportanceWeighting
  /** The discount (default 1). */
  readonly gamma?: number
  /** The initial action values (default 0). */
  readonly initialValue?: number
}

/** The state of {@link offPolicyMonteCarloControl}. */
export interface OffPolicyControlState {
  readonly nStates: number
  readonly nActions: number
  /** The action values, and the cumulative importance weights (weighted) or visit counts (ordinary) behind them. */
  readonly Q: Float64Array
  readonly C: Float64Array
  readonly episode: readonly McStep[]
  readonly episodes: number
  readonly steps: number
  /** The importance weight reached at the earliest step the last update used (the length of the agreeing tail shows in it). */
  readonly lastWeight: number
  readonly tdError: number
}

/**
 * Off-policy Monte Carlo control (Sutton and Barto, section 5.7): the target is the greedy policy of the action values,
 * learnt from episodes of a soft behaviour policy. At the end of an episode the steps are replayed from the last: the
 * return `G` accumulates, the weight `W` starts at 1, the action value moves by `W / C (G - Q)` where `C` sums the
 * weights seen (weighted sampling; ordinary sampling averages `W G` over the visits instead), and the replay stops at
 * the first step whose action the greedy policy would not have taken, since `W` is zero from there back. Learning is
 * therefore from the agreeing tail of each episode, slow at first and sharpest near the goal. Ordinary sampling is
 * offered for the comparison: its values are unnormalised products of ratios, and it does not control well.
 *
 * @param env - The environment (a discrete action space).
 * @param options - The behaviour policy, weighting, discount, initial values and observation coding.
 * @returns The agent; it acts by the behaviour policy and its `greedy` action is the target's.
 * @concept off-policy-learning
 * @example
 * ```ts
 * import { offPolicyMonteCarloControl } from 'mdml-engine/agents'
 * import { randomWalk } from 'mdml-engine/environments'
 * import { key } from 'mdml-engine/random'
 * const agent = offPolicyMonteCarloControl(randomWalk.create({ n: 3 }))
 * const t = { observation: 2, action: 1, reward: 1, nextObservation: 2, terminated: true, truncated: false }
 * agent.learn(agent.init(key(0)), t, key(1)).Q[5] // => 1
 * ```
 */
export function offPolicyMonteCarloControl<O>(
  env: TabularEnvLike<O>,
  options: OffPolicyControlOptions<O> = {},
): Agent<OffPolicyControlState, O, number> {
  const coding = tabularCoding(env, options, 'offPolicyMonteCarloControl')
  const { nStates, nActions } = coding
  const gamma = options.gamma ?? 1
  const weighting = options.weighting ?? 'weighted'
  const uniformProbs = Array.from({ length: nActions }, () => 1 / nActions)
  const behaviour = options.behaviour ?? (() => uniformProbs)
  const greedyAt = (Q: Float64Array, s: number): number => {
    let best = 0
    for (let a = 1; a < nActions; a++) if ((Q[s * nActions + a] ?? 0) > (Q[s * nActions + best] ?? 0)) best = a
    return best
  }
  return {
    name: 'offPolicyMonteCarloControl',
    config: { nStates, nActions, gamma, weighting } as Json,
    metrics: { lastWeight: { description: 'the importance weight at the earliest step of the last update' } },
    init: () => ({
      nStates,
      nActions,
      Q: new Float64Array(nStates * nActions).fill(options.initialValue ?? 0),
      C: new Float64Array(nStates * nActions),
      episode: [],
      episodes: 0,
      steps: 0,
      lastWeight: 0,
      tdError: 0,
    }),
    act: (_, o, k) => sampleFrom(behaviour(o), k),
    greedy: (s, o) => greedyAt(s.Q, coding.encode(o)),
    learn: (s, t) => {
      const b = behaviour(t.observation)[t.action] ?? 0
      if (!(b > 0))
        throw new MdmlError('value', 'offPolicyMonteCarloControl: the behaviour gave the action probability 0')
      const state = coding.encode(t.observation)
      const episode = [...s.episode, { s: state, a: t.action, r: t.reward, b, p: 0 }]
      if (!(t.terminated || t.truncated)) return { ...s, episode, steps: s.steps + 1 }
      const Q = s.Q.slice()
      const C = s.C.slice()
      let g = 0
      let w = 1
      let lastWeight = s.lastWeight
      let delta = s.tdError
      for (let j = episode.length - 1; j >= 0; j--) {
        const e = episode[j] as McStep
        g = e.r + gamma * g
        const i = e.s * nActions + e.a
        C[i] = (C[i] ?? 0) + (weighting === 'ordinary' ? 1 : w)
        delta = (weighting === 'ordinary' ? w * g : g) - (Q[i] ?? 0)
        Q[i] = (Q[i] ?? 0) + ((weighting === 'ordinary' ? 1 : w) / (C[i] ?? 1)) * delta
        lastWeight = w
        if (greedyAt(Q, e.s) !== e.a) break
        w /= e.b
      }
      return { ...s, Q, C, episode: [], episodes: s.episodes + 1, steps: s.steps + 1, lastWeight, tdError: delta }
    },
    stats: (s) => ({ lastWeight: s.lastWeight }),
  }
}
