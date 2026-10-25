/**
 * Imitation learning and inverse reinforcement learning: learning a policy from an expert's behaviour, or the reward
 * that behaviour implies. DAgger (repeatedly labelling the learner's own states with the expert's actions), GAIL (a
 * policy trained against a discriminator that tells its behaviour from the expert's), and maximum-entropy IRL for
 * tabular MDPs with its building blocks: soft value iteration, expected state visitation frequencies and feature
 * expectations of demonstrations.
 */

import { type Json, meanOf, WhiteboxError } from '../../../lib/base/index.ts'
import { binaryCrossEntropyWithLogits } from '../../../lib/losses/index.ts'
import type { ActivationName, Params } from '../../../lib/nn/index.ts'
import { adam, type Tree } from '../../../lib/optim/index.ts'
import { foldIn, type Key, key as makeKey, randint, uniform } from '../../../lib/random/index.ts'
import {
  type Agent,
  type Env,
  occupancyMeasure,
  type Policy,
  resetAfter,
  type TabularMdp,
  type Transition,
} from '../../../lib/rl/index.ts'
import {
  concat,
  fromStorage,
  logsumexp,
  oneHot,
  softplus,
  sum,
  type Tensor,
  toFlat,
  type Value,
} from '../../../lib/tensor/index.ts'
import { type Algorithm, channels, type RunOptions, runAlgorithm } from '../../../lib/trace/index.ts'
import { gradientStep, observationSize, rowsTensor, type Spaces, vectorTensor } from './_common.ts'
import { type Demonstration, learner } from './_learner.ts'
import { mlpNet } from './_networks.ts'
import type { PolicyAction } from './_on-policy.ts'
import { type PolicyGradientState, type PpoOptions, ppo } from './policy-gradient.ts'

export type { Demonstration } from './_learner.ts'

type Vec = readonly number[]

// --------------------------------------------------------------------------------------------------------- DAgger

/** Options of {@link daggerAlgorithm} and {@link dagger}. */
export interface DaggerOptions {
  /** Episodes collected per iteration (default 2). */
  readonly episodesPerIteration?: number
  /** The probability of following the expert at iteration i (default `0.5^i`: the expert alone first). */
  readonly beta?: (iteration: number) => number
  /** Hidden widths of the learner (default `[32]`). */
  readonly hidden?: readonly number[]
  readonly activation?: ActivationName
  readonly learningRate?: number
  /** Passes over the aggregated dataset per iteration (default 20). */
  readonly epochs?: number
  readonly batchSize?: number
  /** A cap on episode length (default 1000; use a time-limited environment). */
  readonly maxEpisodeSteps?: number
  /** The learner's initialisation key (default key(0)). */
  readonly key?: Key
}

/** The state of {@link daggerAlgorithm}. */
export interface DaggerState<A> {
  readonly params: Params
  readonly optState: unknown
  /** The aggregated dataset of visited observations labelled by the expert. */
  readonly dataset: readonly Demonstration<A>[]
  readonly iteration: number
  /** The probability of following the expert in the next iteration. */
  readonly beta: number
  readonly loss: number
  /** Agreement of the learner with the expert on the dataset (discrete) or minus the mean squared error. */
  readonly agreement: number
  /** Mean return of the episodes collected in the last iteration. */
  readonly episodeReturn: number
}

/**
 * DAgger (Ross, Gordon and Bagnell 2011) as an {@link Algorithm}: each iteration rolls out the mixture policy that
 * follows the expert with probability `beta_i` and the learner otherwise (per step), labels every visited observation
 * with the expert's action, aggregates the labels into one dataset, and retrains the learner (an MLP classifier for
 * discrete actions, trained with cross-entropy, or a regressor with squared error for box actions) by Adam on
 * minibatches. Records the loss, the learner's agreement with the expert, beta, the dataset size and the return of the
 * collected episodes.
 *
 * @param env - The environment (vector observations; time-limited).
 * @param expert - The expert policy.
 * @param options - Iteration schedule, learner and optimiser options.
 * @returns The algorithm.
 * @concept imitation-learning
 * @reference Ross, Gordon and Bagnell (2011), A reduction of imitation learning and structured prediction to no-regret
 *   online learning.
 * @example
 * ```ts
 * import { daggerAlgorithm } from 'whitebox-engine/methods/rl/agents'
 * import { cartPole } from 'whitebox-engine/methods/rl'
 * daggerAlgorithm(cartPole.create(), (o) => ((o[2] ?? 0) > 0 ? 1 : 0)).name // => 'dagger'
 * ```
 */
export function daggerAlgorithm<S, A extends PolicyAction>(
  env: Env<S, Vec, A>,
  expert: Policy<Vec, A>,
  options: DaggerOptions = {},
): Algorithm<DaggerState<A>> {
  const net = learner(env, options)
  const opt = adam({ learningRate: options.learningRate ?? 1e-2 })
  const beta = options.beta ?? ((i: number) => 0.5 ** i)
  const episodes = options.episodesPerIteration ?? 2
  const epochs = options.epochs ?? 20
  const batchSize = options.batchSize ?? 64
  const cap = options.maxEpisodeSteps ?? 1000
  return {
    name: 'dagger',
    config: { episodesPerIteration: episodes, epochs, batchSize, hidden: [...(options.hidden ?? [32])] } as Json,
    schema: channels({
      metrics: {
        loss: { description: 'training loss on the aggregated dataset', direction: 'lower' },
        agreement: { description: 'agreement of the learner with the expert on the dataset', direction: 'higher' },
        beta: { description: 'probability of following the expert in the next iteration' },
        datasetSize: { description: 'labelled observations aggregated so far' },
        episodeReturn: { description: 'mean return of the episodes collected in the last iteration' },
      },
    }),
    init: () => ({
      params: net.params,
      optState: opt.init(net.params),
      dataset: [],
      iteration: 0,
      beta: beta(0),
      loss: 0,
      agreement: 0,
      episodeReturn: 0,
    }),
    step: (s, ctx) => {
      const b = s.beta
      const collected: Demonstration<A>[] = []
      const returns: number[] = []
      let previous: S | undefined
      for (let e = 0; e < episodes; e++) {
        const ek = foldIn(ctx.key, e)
        const r = resetAfter(env, previous, foldIn(ek, 0))
        let state = r.state
        let obs = r.observation
        let ret = 0
        for (let t = 0; t < cap; t++) {
          const tk = foldIn(foldIn(ek, 1), t)
          const label = expert(obs, foldIn(tk, 0))
          collected.push({ observation: obs, action: label })
          const useExpert = (uniform(foldIn(tk, 1), []).data[0] ?? 0) < b
          const action = useExpert ? label : (net.predict(s.params, obs) as A)
          const step = env.step(state, action, foldIn(tk, 2))
          ret += step.reward
          state = step.state
          obs = step.observation
          if (step.terminated || step.truncated) break
        }
        previous = state
        returns.push(ret)
      }
      const dataset = [...s.dataset, ...collected]
      let params = s.params
      let optState = s.optState
      let loss = 0
      for (let epoch = 0; epoch < epochs; epoch++) {
        const n = Math.min(batchSize, dataset.length)
        const idx = Array.from(randint(foldIn(foldIn(ctx.key, 100), epoch), [n], 0, dataset.length).data)
        const batch = idx.map((i) => dataset[i] as Demonstration<A>)
        const step = gradientStep((p: Tree) => net.loss(p as Params, batch), params, opt, optState)
        params = step.params as Params
        optState = step.optState
        loss = step.loss
      }
      return {
        params,
        optState,
        dataset,
        iteration: s.iteration + 1,
        beta: beta(s.iteration + 1),
        loss,
        agreement: net.agreement(params, dataset),
        episodeReturn: meanOf(returns),
      }
    },
    observe: (s) => ({
      metrics: {
        loss: s.loss,
        agreement: s.agreement,
        beta: s.beta,
        datasetSize: s.dataset.length,
        episodeReturn: s.episodeReturn,
      },
    }),
  }
}

/** The result of {@link dagger}. */
export interface DaggerResult<A> {
  readonly params: Params
  /** The learner's (deterministic) policy. */
  readonly policy: Policy<Vec, A>
  readonly dataset: readonly Demonstration<A>[]
  readonly iterations: number
  readonly agreement: number
}

/**
 * Run DAgger for a number of iterations and return the learner (see {@link daggerAlgorithm}).
 *
 * @param env - The environment (vector observations; time-limited).
 * @param expert - The expert policy.
 * @param options - DAgger's options, the number of iterations (default 5) and the run options.
 * @returns The learner's parameters and policy, the aggregated dataset and the final agreement.
 * @concept imitation-learning
 * @example
 * ```ts
 * import { dagger } from 'whitebox-engine/methods/rl/agents'
 * import { cartPole } from 'whitebox-engine/methods/rl'
 * import { timeLimit } from 'whitebox-engine/rl'
 * dagger(timeLimit(cartPole.create(), 20), (o) => ((o[2] ?? 0) > 0 ? 1 : 0), { iterations: 1 }).dataset.length // => 40
 * ```
 */
export function dagger<S, A extends PolicyAction>(
  env: Env<S, Vec, A>,
  expert: Policy<Vec, A>,
  options: DaggerOptions & { readonly iterations?: number } & Partial<Omit<RunOptions, 'maxSteps'>> = {},
): DaggerResult<A> {
  const { iterations, episodesPerIteration, beta, hidden, activation, learningRate, epochs, batchSize, ...rest } =
    options
  const { maxEpisodeSteps, key, ...run } = rest
  const algorithmOptions: DaggerOptions = {
    ...(episodesPerIteration === undefined ? {} : { episodesPerIteration }),
    ...(beta === undefined ? {} : { beta }),
    ...(hidden === undefined ? {} : { hidden }),
    ...(activation === undefined ? {} : { activation }),
    ...(learningRate === undefined ? {} : { learningRate }),
    ...(epochs === undefined ? {} : { epochs }),
    ...(batchSize === undefined ? {} : { batchSize }),
    ...(maxEpisodeSteps === undefined ? {} : { maxEpisodeSteps }),
    ...(key === undefined ? {} : { key }),
  }
  const net = learner(env, algorithmOptions)
  const result = runAlgorithm(daggerAlgorithm(env, expert, algorithmOptions), { ...run, maxSteps: iterations ?? 5 })
  const params = result.state.params
  return {
    params,
    policy: (o) => net.predict(params, o) as A,
    dataset: result.state.dataset,
    iterations: result.state.iteration,
    agreement: result.state.agreement,
  }
}

// ----------------------------------------------------------------------------------------------------------- GAIL

/** Options of {@link gail}. */
export interface GailOptions extends PpoOptions {
  /** Hidden widths of the discriminator (default `[32]`). */
  readonly discriminatorHidden?: readonly number[]
  readonly discriminatorLearningRate?: number
  /** Discriminator steps per rollout (default 5). */
  readonly discriminatorSteps?: number
  /** `'gail'`: `-log(1 - D(s, a))` (default); `'airl'`: `log D - log(1 - D)`, the discriminator logit. */
  readonly rewardType?: 'gail' | 'airl'
}

/** The state of {@link gail}. */
export interface GailState {
  /** The PPO learner's state. */
  readonly policy: PolicyGradientState
  readonly discriminator: Params
  readonly discriminatorOptState: unknown
  readonly pending: readonly Transition<Vec, PolicyAction>[]
  readonly steps: number
  readonly stats: Readonly<Record<string, number>>
}

/**
 * Generative adversarial imitation learning (Ho and Ermon 2016): a discriminator `D(s, a)` (an MLP on the observation
 * and the one-hot or continuous action) is trained by logistic regression to tell expert pairs (label 1) from the
 * policy's (label 0), and the policy is trained by {@link ppo} on the discriminator's reward `-log(1 - D(s, a))`
 * instead of the environment's. Every `nSteps` steps the discriminator takes its steps on the new rollout and minibatches
 * of demonstrations, the rollout is relabelled with the updated discriminator, and PPO updates on it. The environment's
 * reward is ignored (kept in `info.envReward` of the relabelled transitions).
 *
 * @param env - The environment's spaces.
 * @param demonstrations - Expert (observation, action) pairs.
 * @param options - PPO's options and the discriminator's.
 * @returns The agent.
 * @concept imitation-learning
 * @reference Ho and Ermon (2016), Generative adversarial imitation learning.
 * @example
 * ```ts
 * import { gail } from 'whitebox-engine/methods/rl/agents'
 * import { cartPole } from 'whitebox-engine/methods/rl'
 * gail(cartPole.create(), [{ observation: [0, 0, 0, 0], action: 1 }]).name // => 'gail'
 * ```
 */
export function gail<A extends PolicyAction = PolicyAction>(
  env: Spaces,
  demonstrations: readonly Demonstration<A>[],
  options: GailOptions = {},
): Agent<GailState, Vec, A> {
  if (demonstrations.length === 0) throw new WhiteboxError('value', 'gail: needs demonstrations')
  const inner = ppo<A>(env, options)
  const nSteps = options.nSteps ?? 256
  const d = observationSize(env.observationSpace)
  const discrete = env.actionSpace.kind === 'discrete'
  const k = discrete ? (env.actionSpace.kind === 'discrete' ? env.actionSpace.n : 0) : observationSize(env.actionSpace)
  const disc = mlpNet(
    [d + k, ...(options.discriminatorHidden ?? [32]), 1],
    'tanh',
    foldIn(options.key ?? makeKey(0), 77),
  )
  const opt = adam({ learningRate: options.discriminatorLearningRate ?? 1e-3 })
  const steps = options.discriminatorSteps ?? 5
  const rewardType = options.rewardType ?? 'gail'
  const actionRows = (actions: readonly PolicyAction[]): Tensor =>
    discrete ? (oneHot(vectorTensor(actions as readonly number[]), k) as Tensor) : rowsTensor(actions as Vec[], k)
  const inputs = (pairs: readonly { observation: Vec; action: PolicyAction }[]): Value =>
    concat(
      [
        rowsTensor(
          pairs.map((p) => p.observation),
          d,
        ),
        actionRows(pairs.map((p) => p.action)),
      ],
      1,
    )
  const logits = (p: Params, pairs: readonly { observation: Vec; action: PolicyAction }[]): Value =>
    sum(disc.apply(p, inputs(pairs)), -1)

  const relabel = (s: GailState, k2: Key): GailState => {
    const policyPairs = s.pending
    let params = s.discriminator
    let optState = s.discriminatorOptState
    let loss = 0
    for (let i = 0; i < steps; i++) {
      const idx = Array.from(randint(foldIn(k2, i), [policyPairs.length], 0, demonstrations.length).data)
      const expertPairs = idx.map((j) => demonstrations[j] as Demonstration<A>)
      const labels = vectorTensor([...expertPairs.map(() => 1), ...policyPairs.map(() => 0)])
      const both = [...expertPairs, ...policyPairs]
      const step = gradientStep(
        (p: Tree) => binaryCrossEntropyWithLogits(logits(p as Params, both), labels),
        params,
        opt,
        optState,
      )
      params = step.params as Params
      optState = step.optState
      loss = step.loss
    }
    const z = logits(params, policyPairs)
    const rewards = toFlat(rewardType === 'gail' ? softplus(z) : z)
    const expertAccuracy = meanOf(toFlat(logits(params, demonstrations.slice(0, 256))).map((x) => (x > 0 ? 1 : 0)))
    const policyAccuracy = meanOf(toFlat(z).map((x) => (x < 0 ? 1 : 0)))
    let ppoState = s.policy
    for (const [i, t] of policyPairs.entries()) {
      const relabelled: Transition<Vec, A> = {
        ...(t as Transition<Vec, A>),
        reward: rewards[i] ?? 0,
        info: { ...(t.info ?? {}), envReward: t.reward },
      }
      ppoState = inner.learn(ppoState, relabelled, foldIn(k2, 1000 + i))
    }
    return {
      ...s,
      policy: ppoState,
      discriminator: params,
      discriminatorOptState: optState,
      pending: [],
      stats: {
        ...ppoState.stats,
        discriminatorLoss: loss,
        expertAccuracy,
        policyAccuracy,
        imitationReward: meanOf(rewards),
      },
    }
  }

  return {
    name: 'gail',
    config: { nSteps, discriminatorSteps: steps, rewardType, demonstrations: demonstrations.length } as Json,
    metrics: {
      discriminatorLoss: { description: 'logistic loss of the discriminator', direction: 'lower' },
      expertAccuracy: { description: 'fraction of expert pairs the discriminator labels as expert' },
      policyAccuracy: { description: 'fraction of policy pairs the discriminator labels as the policy' },
      imitationReward: { description: 'mean discriminator reward of the last rollout', direction: 'higher' },
      ...(inner.metrics ?? {}),
    },
    init: (k2) => ({
      policy: inner.init(k2),
      discriminator: disc.params,
      discriminatorOptState: opt.init(disc.params),
      pending: [],
      steps: 0,
      stats: {},
    }),
    act: (s, o, k2) => inner.act(s.policy, o, k2),
    greedy: (s, o) => inner.greedy(s.policy, o),
    learn: (s, t, k2) => {
      const next: GailState = { ...s, pending: [...s.pending, t as Transition<Vec, PolicyAction>], steps: s.steps + 1 }
      return next.pending.length >= nSteps ? relabel(next, k2) : next
    },
    stats: (s) => s.stats,
  }
}

// ------------------------------------------------------------------------------------- maximum-entropy IRL (tabular)

/** Options of {@link softValueIteration}. */
export interface SoftValueIterationOptions {
  /** The discount (default the MDP's). */
  readonly gamma?: number
  /** Stop when the largest value change falls below this (default 1e-12). */
  readonly tol?: number
  readonly maxIter?: number
}

/** The result of {@link softValueIteration}. */
export interface SoftValues {
  readonly V: number[]
  readonly Q: number[][]
  /** The maximum-entropy policy `pi(a | s) = exp(Q(s, a) - V(s))`. */
  readonly policy: number[][]
  readonly iterations: number
}

function rewardTable(mdp: TabularMdp, reward: readonly number[] | readonly (readonly number[])[]): Float64Array {
  const { nStates: nS, nActions: nA } = mdp
  const R = new Float64Array(nS * nA)
  for (let s = 0; s < nS; s++) {
    const r = reward[s]
    for (let a = 0; a < nA; a++) R[s * nA + a] = typeof r === 'number' ? r : (r?.[a] ?? 0)
  }
  return R
}

/**
 * Soft value iteration (Ziebart et al. 2008; Ziebart 2010): iterate `Q(s, a) = r(s, a) + gamma sum_s' P(s' | s, a)
 * V(s')` and the soft maximum `V(s) = log sum_a exp Q(s, a)` to a fixed point; the maximum-entropy policy is
 * `exp(Q - V)`. Outcomes that end the episode contribute no future value.
 *
 * @param mdp - The MDP (its transitions; its own rewards are not used).
 * @param reward - Rewards per state `r(s)` or per state-action `r(s, a)`.
 * @param options - Discount, tolerance and iteration cap.
 * @returns Soft state and action values and the policy.
 * @concept inverse-reinforcement-learning
 * @example
 * ```ts
 * import { softValueIteration } from 'whitebox-engine/methods/rl/agents'
 * import { tabularMdp } from 'whitebox-engine/rl'
 * const mdp = tabularMdp({ transitions: [[[0, 1], [0, 1]], [[0, 1], [0, 1]]], rewards: [[0, 0], [0, 0]], terminalStates: [1] })
 * softValueIteration(mdp, [0, 0]).policy[0] // => [0.5, 0.5]
 * ```
 */
export function softValueIteration(
  mdp: TabularMdp,
  reward: readonly number[] | readonly (readonly number[])[],
  options: SoftValueIterationOptions = {},
): SoftValues {
  const { nStates: nS, nActions: nA, transitions: P } = mdp
  const gamma = options.gamma ?? mdp.gamma
  const tol = options.tol ?? 1e-12
  const maxIter = options.maxIter ?? 100_000
  const R = rewardTable(mdp, reward)
  let V = new Float64Array(nS)
  const Q = new Float64Array(nS * nA)
  let iterations = 0
  for (; iterations < maxIter; iterations++) {
    for (let j = 0; j < nS * nA; j++) {
      let future = 0
      const base = j * nS
      for (let s2 = 0; s2 < nS; s2++) future += (P[base + s2] ?? 0) * (V[s2] ?? 0)
      Q[j] = (R[j] ?? 0) + gamma * future
    }
    const next = Float64Array.from(toFlat(logsumexp(fromStorage(Float64Array.from(Q), [nS, nA], 'float64'), 1)))
    let delta = 0
    for (let s = 0; s < nS; s++) delta = Math.max(delta, Math.abs((next[s] ?? 0) - (V[s] ?? 0)))
    V = next
    if (delta < tol) {
      iterations += 1
      break
    }
  }
  // One more backup so that Q is consistent with the final V, and V its soft maximum.
  for (let j = 0; j < nS * nA; j++) {
    let future = 0
    const base = j * nS
    for (let s2 = 0; s2 < nS; s2++) future += (P[base + s2] ?? 0) * (V[s2] ?? 0)
    Q[j] = (R[j] ?? 0) + gamma * future
  }
  const Vs = toFlat(logsumexp(fromStorage(Float64Array.from(Q), [nS, nA], 'float64'), 1))
  const Qs = Array.from({ length: nS }, (_, s) => Array.from(Q.subarray(s * nA, (s + 1) * nA)))
  return {
    V: Vs,
    Q: Qs,
    policy: Qs.map((q, s) => q.map((x) => Math.exp(x - (Vs[s] ?? 0)))),
    iterations,
  }
}

/**
 * Expected state visitation frequencies of a stochastic policy from the MDP's initial distribution: over a finite
 * horizon, `D = sum_{t < T} D_t` with `D_0 = mu_0` and `D_{t+1}(s') = sum_{s, a} D_t(s) pi(a | s) P(s' | s, a)`
 * (Ziebart et al. 2008, algorithm 1, with a stationary policy; mass that ends the episode leaves), or without a horizon
 * the discounted occupancy measure (lib/rl `occupancyMeasure`).
 *
 * @param mdp - The MDP.
 * @param policy - Action probabilities per state.
 * @param options - The horizon `T`, or the discount for the infinite-horizon form (default the MDP's).
 * @returns The expected visits per state.
 * @concept inverse-reinforcement-learning
 * @example
 * ```ts
 * import { stateVisitationFrequencies } from 'whitebox-engine/methods/rl/agents'
 * import { tabularMdp } from 'whitebox-engine/rl'
 * const mdp = tabularMdp({ transitions: [[[0, 1]], [[0, 1]]], rewards: [[0], [0]], terminalStates: [1], initial: [1, 0] })
 * stateVisitationFrequencies(mdp, [[1], [1]], { horizon: 3 }) // => [1, 0]
 * ```
 */
export function stateVisitationFrequencies(
  mdp: TabularMdp,
  policy: readonly (readonly number[])[],
  options: { readonly horizon?: number; readonly gamma?: number } = {},
): number[] {
  if (options.horizon === undefined) return occupancyMeasure(mdp, policy, { gamma: options.gamma ?? mdp.gamma }).state
  const { nStates: nS, nActions: nA, transitions: P } = mdp
  let D = Array.from(mdp.initial)
  const total = new Array<number>(nS).fill(0)
  for (let t = 0; t < options.horizon; t++) {
    const next = new Array<number>(nS).fill(0)
    for (let s = 0; s < nS; s++) {
      const ds = D[s] ?? 0
      total[s] = (total[s] ?? 0) + ds
      if (ds === 0) continue
      for (let a = 0; a < nA; a++) {
        const w = ds * (policy[s]?.[a] ?? 0)
        if (w === 0) continue
        const base = (s * nA + a) * nS
        for (let s2 = 0; s2 < nS; s2++) next[s2] = (next[s2] ?? 0) + w * (P[base + s2] ?? 0)
      }
    }
    D = next
  }
  return total
}

/**
 * Empirical feature expectations of demonstrations: the mean over trajectories of `sum_t gamma^t phi(s_t)` (with
 * `gamma = 1` for the finite-horizon form), over the first `horizon` states of each.
 *
 * @param trajectories - State-index sequences (the states in which actions were taken).
 * @param features - Feature vectors per state `phi[s]`.
 * @param options - Discount (default 1) and horizon (default all states).
 * @returns The feature expectations.
 * @concept inverse-reinforcement-learning
 * @example
 * ```ts
 * import { featureExpectations } from 'whitebox-engine/methods/rl/agents'
 * featureExpectations([[0, 1], [0]], [[1, 0], [0, 1]]) // => [1, 0.5]
 * ```
 */
export function featureExpectations(
  trajectories: readonly (readonly number[])[],
  features: readonly (readonly number[])[],
  options: { readonly gamma?: number; readonly horizon?: number } = {},
): number[] {
  const m = features[0]?.length ?? 0
  const out = new Array<number>(m).fill(0)
  const gamma = options.gamma ?? 1
  for (const traj of trajectories) {
    let w = 1
    for (const [t, s] of traj.entries()) {
      if (options.horizon !== undefined && t >= options.horizon) break
      const phi = features[s] ?? []
      for (let j = 0; j < m; j++) out[j] = (out[j] ?? 0) + (w * (phi[j] ?? 0)) / trajectories.length
      w *= gamma
    }
  }
  return out
}

/** Options of {@link maxEntIrlAlgorithm} and {@link maxEntIrl}. */
export interface MaxEntIrlOptions {
  /** Feature vectors per state (default one-hot states). */
  readonly features?: readonly (readonly number[])[]
  /** Gradient-ascent step size on the reward weights (default 0.1). */
  readonly learningRate?: number
  /**
   * `'discounted'` (default): discounted occupancies and discounted expert features, consistent with the stationary
   * soft-optimal policy; a number: Ziebart's finite-horizon visitation frequencies over that many steps.
   */
  readonly horizon?: number | 'discounted'
  /** The discount of the soft value iteration and of discounted occupancies (default the MDP's). */
  readonly gamma?: number
  /** Stop when the gradient norm falls below this (default 1e-6). */
  readonly tol?: number
}

/** The state of {@link maxEntIrlAlgorithm}. */
export interface MaxEntIrlState {
  readonly theta: readonly number[]
  readonly reward: readonly number[]
  readonly policy: readonly (readonly number[])[]
  readonly svf: readonly number[]
  readonly gradientNorm: number
}

/**
 * Maximum-entropy inverse reinforcement learning (Ziebart, Maas, Bagnell and Dey 2008) for tabular MDPs with a model,
 * as an {@link Algorithm}: the reward is linear in state features, `r(s) = phi(s) . theta`; each step solves the
 * maximum-entropy policy by {@link softValueIteration}, computes its expected state visitation frequencies
 * ({@link stateVisitationFrequencies}) and ascends the log-likelihood of the demonstrations, whose gradient is the
 * difference of feature expectations `f_expert - sum_s D(s) phi(s)`.
 *
 * @param mdp - The MDP (its transitions and initial distribution).
 * @param demonstrations - State-index trajectories of the expert.
 * @param options - Features, step size, horizon, discount and tolerance.
 * @returns The algorithm.
 * @concept inverse-reinforcement-learning
 * @reference Ziebart, Maas, Bagnell and Dey (2008), Maximum entropy inverse reinforcement learning.
 * @example
 * ```ts
 * import { maxEntIrlAlgorithm } from 'whitebox-engine/methods/rl/agents'
 * import { tabularMdp } from 'whitebox-engine/rl'
 * const mdp = tabularMdp({ transitions: [[[0, 1], [0, 1]], [[0, 1], [0, 1]]], rewards: [[0, 0], [0, 0]], terminalStates: [1] })
 * maxEntIrlAlgorithm(mdp, [[0]]).name // => 'maxEntIrl'
 * ```
 */
export function maxEntIrlAlgorithm(
  mdp: TabularMdp,
  demonstrations: readonly (readonly number[])[],
  options: MaxEntIrlOptions = {},
): Algorithm<MaxEntIrlState> {
  const nS = mdp.nStates
  const features =
    options.features ?? Array.from({ length: nS }, (_, s) => Array.from({ length: nS }, (_, j) => (s === j ? 1 : 0)))
  const m = features[0]?.length ?? 0
  const lr = options.learningRate ?? 0.1
  const gamma = options.gamma ?? mdp.gamma
  const discounted = typeof options.horizon !== 'number'
  const horizon = typeof options.horizon === 'number' ? options.horizon : 0
  const expert = featureExpectations(demonstrations, features, discounted ? { gamma } : { horizon })
  const tol = options.tol ?? 1e-6
  const evaluate = (theta: readonly number[]): Omit<MaxEntIrlState, 'theta'> => {
    const reward = features.map((phi) => {
      let r = 0
      for (let j = 0; j < m; j++) r += (phi[j] ?? 0) * (theta[j] ?? 0)
      return r
    })
    const { policy } = softValueIteration(mdp, reward, { gamma })
    const svf = stateVisitationFrequencies(mdp, policy, discounted ? { gamma } : { horizon })
    let norm = 0
    for (let j = 0; j < m; j++) {
      let learner = 0
      for (let s = 0; s < nS; s++) learner += (svf[s] ?? 0) * (features[s]?.[j] ?? 0)
      norm += ((expert[j] ?? 0) - learner) ** 2
    }
    return { reward, policy, svf, gradientNorm: Math.sqrt(norm) }
  }
  return {
    name: 'maxEntIrl',
    config: { learningRate: lr, gamma, horizon: discounted ? 'discounted' : horizon, features: m } as Json,
    schema: channels({
      metrics: {
        gradientNorm: { description: 'norm of the feature-expectation gap', direction: 'lower' },
      },
      tensors: {
        reward: { role: 'state', description: 'the recovered reward per state' },
        svf: { role: 'state', description: 'expected state visitation frequencies' },
      },
    }),
    init: () => {
      const theta = new Array<number>(m).fill(0)
      return { theta, ...evaluate(theta) }
    },
    step: (s) => {
      const theta = s.theta.map((t, j) => {
        let learner = 0
        for (let i = 0; i < nS; i++) learner += (s.svf[i] ?? 0) * (features[i]?.[j] ?? 0)
        return t + lr * ((expert[j] ?? 0) - learner)
      })
      return { theta, ...evaluate(theta) }
    },
    done: (s) => (s.gradientNorm < tol ? 'converged' : undefined),
    observe: (s) => ({
      metrics: { gradientNorm: s.gradientNorm },
      tensors: { reward: { role: 'state', value: s.reward }, svf: { role: 'state', value: s.svf } },
    }),
  }
}

/** The result of {@link maxEntIrl}. */
export interface MaxEntIrlResult extends MaxEntIrlState {
  readonly iterations: number
  readonly converged: boolean
}

/**
 * Recover a reward from demonstrations by maximum-entropy IRL (see {@link maxEntIrlAlgorithm}).
 *
 * @param mdp - The MDP.
 * @param demonstrations - State-index trajectories of the expert.
 * @param options - The algorithm's options, an iteration cap (default 200) and the run options.
 * @returns The reward weights, the reward per state, the maximum-entropy policy and its visitation frequencies.
 * @concept inverse-reinforcement-learning
 * @example
 * ```ts
 * import { maxEntIrl } from 'whitebox-engine/methods/rl/agents'
 * import { tabularMdp } from 'whitebox-engine/rl'
 * const mdp = tabularMdp({ transitions: [[[0, 1], [0, 1]], [[0, 1], [0, 1]]], rewards: [[0, 0], [0, 0]], terminalStates: [1], initial: [1, 0] })
 * maxEntIrl(mdp, [[0]]).converged // => true
 * ```
 */
export function maxEntIrl(
  mdp: TabularMdp,
  demonstrations: readonly (readonly number[])[],
  options: MaxEntIrlOptions & { readonly maxIter?: number } & Partial<Omit<RunOptions, 'maxSteps'>> = {},
): MaxEntIrlResult {
  const { features, learningRate, horizon, gamma, tol, maxIter, ...run } = options
  const algorithmOptions: MaxEntIrlOptions = {
    ...(features === undefined ? {} : { features }),
    ...(learningRate === undefined ? {} : { learningRate }),
    ...(horizon === undefined ? {} : { horizon }),
    ...(gamma === undefined ? {} : { gamma }),
    ...(tol === undefined ? {} : { tol }),
  }
  const result = runAlgorithm(maxEntIrlAlgorithm(mdp, demonstrations, algorithmOptions), {
    ...run,
    maxSteps: maxIter ?? 200,
  })
  return { ...result.state, iterations: result.steps, converged: result.status === 'converged' }
}
