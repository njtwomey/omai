/**
 * Exploration by intrinsic rewards: `intrinsicReward` wraps any agent so that it learns from the environment's reward
 * plus a novelty bonus, and bonus modules supply that bonus. Visit counts (MBIE-EB, with SimHash codes for vector
 * observations), random network distillation (the error of predicting a fixed random network's output) and the
 * intrinsic curiosity module (the error of a learnt forward model); each module keeps what it learns in its own state
 * beside the inner agent's.
 */

import { type Json, MdmlError } from '../base/index.ts'
import { crossEntropy, mse } from '../losses/index.ts'
import { mlp } from '../nn/index.ts'
import type { LearningRateSchedule, ParamTree } from '../optim/index.ts'
import { foldIn, type Key, key as makeKey } from '../random/index.ts'
import {
  type Agent,
  bufferAdd,
  bufferSample,
  type CountTable,
  type CuriosityModule,
  countBonus,
  countTable,
  countVisit,
  curiosityBonus,
  type RandomNetworkDistillation,
  type ReplayBuffer,
  type RunningMeanStd,
  replayBuffer,
  rndBonus,
  rndLoss,
  runningMeanStd,
  simHash,
  type Transition,
  updateRunningMeanStd,
  visits,
} from '../rl/index.ts'
import { add, concat, mul, stack, type Value } from '../tensor/index.ts'
import {
  gradientStep,
  gradientStepWithTerms,
  optimiserOf,
  rowsTensor,
  type Spaces,
  spaceFeatures,
  vectorTensor,
} from './_common.ts'

// ------------------------------------------------------------------------------------------------ the wrapper

/**
 * A source of intrinsic reward: pure functions over its own state. `update` scores a transition and learns from it
 * (counting it, training a predictor), returning the bonus it was given; `bonus` scores a transition without learning.
 *
 * @concept exploration
 */
export interface BonusModule<B, O, A> {
  readonly name: string
  readonly config: Json
  init(key: Key): B
  bonus(state: B, transition: Transition<O, A>): number
  update(state: B, transition: Transition<O, A>, key: Key): { readonly state: B; readonly bonus: number }
  stats?(state: B): Readonly<Record<string, number>>
}

/** Options of {@link intrinsicReward}. */
export interface IntrinsicRewardOptions {
  /** The weight of the bonus: a number or a schedule over environment steps (default 1). */
  readonly scale?: number | LearningRateSchedule
  /** Multiply the scale by `decay^t` after `t` steps (default 1: no decay). */
  readonly decay?: number
  /** The weight of the extrinsic reward (default 1; 0 trains on curiosity alone, as in Pathak et al.'s no-reward runs). */
  readonly extrinsic?: number
}

/** The state of an agent wrapped by {@link intrinsicReward}. */
export interface IntrinsicState<G, B> {
  readonly agent: G
  readonly bonus: B
  readonly steps: number
  /** The bonus of the last transition, before scaling. */
  readonly lastBonus: number
  /** The bonus scale at the last step. */
  readonly scale: number
}

/**
 * Wrap any agent so that it learns from `extrinsic r + scale_t b(s, a, s')`, with `b` from a bonus module: count-based
 * bonuses (MBIE-EB, #Exploration), random network distillation or the intrinsic curiosity module. Acting and greedy
 * evaluation are the inner agent's; the original and intrinsic rewards are kept in the transition's `info`.
 * Reports the inner agent's statistics plus `intrinsic` (the last bonus) and `intrinsicScale`, and the module's.
 *
 * @param agent - The agent to wrap.
 * @param module - The bonus module.
 * @param options - The bonus scale (or schedule), its decay and the extrinsic weight.
 * @returns The wrapped agent.
 * @concept exploration
 * @example
 * ```ts
 * import { countBonusModule, dqn, intrinsicReward } from 'mdml-engine/agents'
 * import { riverSwim } from 'mdml-engine/environments'
 * const env = riverSwim.create()
 * intrinsicReward(dqn(env), countBonusModule(env), { scale: 0.1 }).name // => 'dqn+count'
 * ```
 */
export function intrinsicReward<G, B, O, A>(
  agent: Agent<G, O, A>,
  module: BonusModule<B, O, A>,
  options: IntrinsicRewardOptions = {},
): Agent<IntrinsicState<G, B>, O, A> {
  const scale = options.scale ?? 1
  const decay = options.decay ?? 1
  const extrinsic = options.extrinsic ?? 1
  const scaleAt = (t: number): number => (typeof scale === 'number' ? scale : scale(t)) * decay ** t
  return {
    name: `${agent.name}+${module.name}`,
    config: {
      agent: agent.config,
      bonus: module.config,
      scale: typeof scale === 'number' ? scale : 'schedule',
      decay,
      extrinsic,
    } as Json,
    metrics: {
      ...agent.metrics,
      intrinsic: { description: 'intrinsic reward of the last transition (before scaling)' },
      intrinsicScale: { description: 'weight of the intrinsic reward' },
    },
    init: (k) => ({
      agent: agent.init(foldIn(k, 0)),
      bonus: module.init(foldIn(k, 1)),
      steps: 0,
      lastBonus: 0,
      scale: scaleAt(0),
    }),
    act: (s, obs, k) => agent.act(s.agent, obs, k),
    greedy: (s, obs) => agent.greedy(s.agent, obs),
    learn: (s, t, k) => {
      const { state: bonusState, bonus } = module.update(s.bonus, t, foldIn(k, 0))
      const w = scaleAt(s.steps)
      const shaped: Transition<O, A> = {
        ...t,
        reward: extrinsic * t.reward + w * bonus,
        info: { ...t.info, extrinsicReward: t.reward, intrinsicReward: bonus },
      }
      return {
        agent: agent.learn(s.agent, shaped, foldIn(k, 1)),
        bonus: bonusState,
        steps: s.steps + 1,
        lastBonus: bonus,
        scale: w,
      }
    },
    stats: (s) => ({
      ...agent.stats?.(s.agent),
      ...module.stats?.(s.bonus),
      intrinsic: s.lastBonus,
      intrinsicScale: s.scale,
    }),
  }
}

// --------------------------------------------------------------------------------------------- shared plumbing

/** Options shared by the learned bonus modules. */
export interface LearnedBonusOptions {
  /** Hidden widths of the networks (default `[64]`). */
  readonly hidden?: readonly number[]
  /** The feature size (default 32). */
  readonly features?: number
  readonly learningRate?: number
  /** Experience kept for training the module (default 1000 transitions). */
  readonly bufferSize?: number
  readonly batchSize?: number
  /** Train every this many steps (default 1). */
  readonly trainFrequency?: number
  readonly maxGradNorm?: number
  /** The key of the networks' initialisation (default key(0)). */
  readonly key?: Key
}

// ------------------------------------------------------------------------------------------------ count bonuses

/** Options of {@link countBonusModule}. */
export interface CountBonusOptions {
  /** The bonus scale `beta` of `beta / sqrt(n)` (default 1; the wrapper's scale multiplies it). */
  readonly beta?: number
  /** Count states, state-action pairs (MBIE-EB, the default) or next states. */
  readonly on?: 'state' | 'state-action' | 'next-state'
  /**
   * How observations become codes: `'exact'` (the observation itself; the default for discrete observations), or
   * SimHash with `bits` hyperplanes (Tang et al. 2017; the default for vector observations, with 32 bits).
   */
  readonly hash?: 'exact' | { readonly bits?: number; readonly key?: Key }
}

/**
 * Count-based exploration: visit counts `n` of states, state-action pairs or next states, and the bonus
 * `beta / sqrt(n)` (MBIE-EB, Strehl and Littman 2008; with SimHash codes of vector observations, #Exploration, Tang et
 * al. 2017). The count includes the transition being scored, so a first visit earns `beta`.
 *
 * @param env - The environment's spaces.
 * @param options - `beta`, what to count and how to code observations.
 * @returns The module; its state is a {@link CountTable}.
 * @concept exploration
 * @example
 * ```ts
 * import { countBonusModule } from 'mdml-engine/agents'
 * import { riverSwim } from 'mdml-engine/environments'
 * import { key } from 'mdml-engine/random'
 * const m = countBonusModule(riverSwim.create(), { beta: 0.5 })
 * const t = { observation: 0, action: 1, reward: 0, nextObservation: 1, terminated: false, truncated: false }
 * let s = m.init(key(0))
 * const bonuses: number[] = []
 * for (let i = 0; i < 4; i++) {
 *   const u = m.update(s, t, key(i))
 *   s = u.state
 *   bonuses.push(u.bonus)
 * }
 * [bonuses[0], bonuses[3]] // => [0.5, 0.25]
 * ```
 */
export function countBonusModule<O, A>(env: Spaces, options: CountBonusOptions = {}): BonusModule<CountTable, O, A> {
  const beta = options.beta ?? 1
  const on = options.on ?? 'state-action'
  const discrete = env.observationSpace.kind === 'discrete'
  const hash = options.hash ?? (discrete ? 'exact' : {})
  const features = spaceFeatures(env.observationSpace)
  const coder =
    hash === 'exact'
      ? (o: O): string => (discrete ? String(o) : JSON.stringify(o))
      : (() => {
          const h = simHash(hash.key ?? makeKey(0x5348), features.size, hash.bits ?? 32)
          return (o: O): string => h(features.of(o))
        })()
  const codeOf = (t: Transition<O, A>): string =>
    on === 'state'
      ? coder(t.observation)
      : on === 'next-state'
        ? coder(t.nextObservation)
        : `${coder(t.observation)}|${JSON.stringify(t.action)}`
  return {
    name: 'count',
    config: { beta, on, hash: hash === 'exact' ? 'exact' : { bits: hash.bits ?? 32 } } as Json,
    init: () => countTable(),
    bonus: (s, t) => countBonus(visits(s, codeOf(t)) + 1, beta),
    update: (s, t) => {
      const code = codeOf(t)
      const state = countVisit(s, code)
      return { state, bonus: countBonus(visits(state, code), beta) }
    },
    stats: (s) => ({ distinct: Object.keys(s.counts).length }),
  }
}

// --------------------------------------------------------------------------------- random network distillation

/** Options of {@link rndModule}. */
export interface RndOptions extends LearnedBonusOptions {
  /** Whiten observations with running statistics and clip them to `[-clip, clip]` (default true, clip 5). */
  readonly normalizeObservations?: boolean
  readonly clip?: number
  /** Divide bonuses by the running standard deviation of their discounted sum (default true). */
  readonly normalizeRewards?: boolean
  /** The discount of the intrinsic return used for that normalisation (default 0.99). */
  readonly gamma?: number
}

/** The state of {@link rndModule}. */
export interface RndState {
  readonly params: ParamTree
  readonly optState: unknown
  readonly observations: RunningMeanStd
  readonly returns: RunningMeanStd
  /** The running (non-episodic) discounted intrinsic return. */
  readonly intrinsicReturn: number
  readonly buffer: ReplayBuffer<readonly number[]>
  readonly steps: number
  readonly loss: number
  /** The last raw (unnormalised) prediction error. */
  readonly error: number
}

/**
 * Random network distillation (Burda et al. 2019): a fixed random target network `f` and a predictor `f_hat` (one
 * hidden layer deeper) trained to match it on visited observations; the bonus of a transition is the prediction error
 * `||f(s') - f_hat(s')||^2` at the next observation, which shrinks where the agent has been. As in the paper,
 * observations are whitened by running statistics and clipped, and bonuses are divided by a running estimate of the
 * standard deviation of the (non-episodic) intrinsic return. The predictor trains with Adam on minibatches of recent
 * observations.
 *
 * @param env - The environment's spaces.
 * @param options - Network sizes, optimiser, replay and the normalisations.
 * @returns The module.
 * @concept exploration
 * @example
 * ```ts
 * import { rndModule } from 'mdml-engine/agents'
 * import { riverSwim } from 'mdml-engine/environments'
 * rndModule(riverSwim.create(), { features: 8 }).name // => 'rnd'
 * ```
 */
export function rndModule<O, A>(env: Spaces, options: RndOptions = {}): BonusModule<RndState, O, A> {
  const obs = spaceFeatures(env.observationSpace)
  const d = obs.size
  const hidden = options.hidden ?? [64]
  const k = options.features ?? 32
  const key = options.key ?? makeKey(0)
  const targetNet = mlp([d, ...hidden, k], { key: foldIn(key, 0) })
  const predictorNet = mlp([d, ...hidden, ...hidden.slice(-1), k], { key: foldIn(key, 1) })
  const rnd: RandomNetworkDistillation<ParamTree> = {
    target: (x) => targetNet.apply(targetNet.params, x),
    predictor: (p, x) => predictorNet.apply(p as never, x),
  }
  const opt = optimiserOf(options, 1e-3)
  const clip = options.clip ?? 5
  const normObs = options.normalizeObservations ?? true
  const normRew = options.normalizeRewards ?? true
  const gamma = options.gamma ?? 0.99
  const batchSize = options.batchSize ?? 32
  const trainFrequency = options.trainFrequency ?? 1
  const whiten = (s: RndState, x: readonly number[]): number[] =>
    normObs
      ? x.map((v, j) => {
          const z = (v - (s.observations.mean[j] ?? 0)) / Math.sqrt((s.observations.var[j] ?? 1) + 1e-8)
          return Math.min(clip, Math.max(-clip, z))
        })
      : [...x]
  const rawError = (s: RndState, x: readonly number[]): number => rndBonus(rnd, s.params, [whiten(s, x)])[0] ?? 0
  const scaled = (s: RndState, e: number): number => (normRew ? e / Math.sqrt((s.returns.var[0] ?? 1) + 1e-8) : e)
  return {
    name: 'rnd',
    config: { hidden: [...hidden], features: k, normalizeObservations: normObs, normalizeRewards: normRew, gamma },
    init: () => ({
      params: predictorNet.params as ParamTree,
      optState: opt.init(predictorNet.params as ParamTree),
      observations: runningMeanStd(d),
      returns: runningMeanStd(1),
      intrinsicReturn: 0,
      buffer: replayBuffer(options.bufferSize ?? 1000),
      steps: 0,
      loss: 0,
      error: 0,
    }),
    bonus: (s, t) => scaled(s, rawError(s, obs.of(t.nextObservation))),
    update: (s0, t, key) => {
      const x = obs.of(t.nextObservation)
      let s: RndState = normObs ? { ...s0, observations: updateRunningMeanStd(s0.observations, [x]) } : s0
      const error = rawError(s, x)
      const intrinsicReturn = gamma * s.intrinsicReturn + error
      s = { ...s, intrinsicReturn, error, returns: updateRunningMeanStd(s.returns, [[intrinsicReturn]]) }
      const bonus = scaled(s, error)
      s = { ...s, buffer: bufferAdd(s.buffer, x), steps: s.steps + 1 }
      if (s.steps % trainFrequency === 0 && s.buffer.size >= Math.min(batchSize, 1 + s.buffer.capacity / 2)) {
        const rows = bufferSample(s.buffer, key, batchSize).items.map((r) => whiten(s, r))
        const batch = rowsTensor(rows, d)
        const step = gradientStep((p) => rndLoss(rnd, p, batch), s.params, opt, s.optState)
        s = { ...s, params: step.params, optState: step.optState, loss: step.loss }
      }
      return { state: s, bonus }
    },
    stats: (s) => ({ rndLoss: s.loss, rndError: s.error }),
  }
}

// --------------------------------------------------------------------------------------------- curiosity (ICM)

/** Options of {@link icmModule}. */
export interface IcmOptions extends LearnedBonusOptions {
  /** The weight of the forward loss against the inverse loss, `beta` (default 0.2, as in the paper). */
  readonly beta?: number
  /** The bonus scale `eta` of `eta / 2 ||phi_hat(s') - phi(s')||^2` (default 1). */
  readonly eta?: number
}

/** A transition as {@link icmModule} stores it: features, the action's encoding and its class index. */
export interface IcmItem {
  readonly observation: readonly number[]
  readonly action: readonly number[]
  readonly next: readonly number[]
  readonly label: number
}

/** The state of {@link icmModule}. */
export interface IcmState {
  readonly params: ParamTree
  readonly optState: unknown
  readonly buffer: ReplayBuffer<IcmItem>
  readonly steps: number
  readonly forwardLoss: number
  readonly inverseLoss: number
}

/**
 * The intrinsic curiosity module (Pathak et al. 2017): an encoder `phi`, an inverse model predicting the action from
 * `phi(s)` and `phi(s')` (cross-entropy for discrete actions, squared error for continuous ones), which shapes the
 * features to what the agent can affect, and a forward model predicting `phi(s')` from `phi(s)` and the action. The
 * bonus is the forward model's error `eta / 2 ||phi_hat(s') - phi(s')||^2`; all three networks train together on
 * `(1 - beta) L_inverse + beta L_forward` with Adam on minibatches of recent transitions.
 *
 * @param env - The environment's spaces (discrete or box actions).
 * @param options - Network sizes, optimiser, replay, `beta` and `eta`.
 * @returns The module.
 * @concept exploration
 * @example
 * ```ts
 * import { icmModule } from 'mdml-engine/agents'
 * import { riverSwim } from 'mdml-engine/environments'
 * icmModule(riverSwim.create()).config['beta'] // => 0.2
 * ```
 */
export function icmModule<O, A>(env: Spaces, options: IcmOptions = {}): BonusModule<IcmState, O, A> {
  const obs = spaceFeatures(env.observationSpace)
  const act = spaceFeatures(env.actionSpace)
  const discreteActions = env.actionSpace.kind === 'discrete'
  if (!(discreteActions || env.actionSpace.kind === 'box'))
    throw new MdmlError('value', 'icmModule: needs discrete or box actions')
  const hidden = options.hidden ?? [64]
  const k = options.features ?? 32
  const key = options.key ?? makeKey(0)
  const encoder = mlp([obs.size, ...hidden, k], { key: foldIn(key, 0) })
  const forwardNet = mlp([k + act.size, ...hidden, k], { key: foldIn(key, 1) })
  const inverseNet = mlp([2 * k, ...hidden, act.size], { key: foldIn(key, 2) })
  type P = { readonly encoder: never; readonly forward: never; readonly inverse: never }
  const icm: CuriosityModule<ParamTree> = {
    encode: (p, x) => encoder.apply((p as P).encoder, x),
    forward: (p, f, a) => forwardNet.apply((p as P).forward, concat([f, a], -1)),
    inverse: (p, f, g) => inverseNet.apply((p as P).inverse, concat([f, g], -1)),
  }
  const params: ParamTree = {
    encoder: encoder.params,
    forward: forwardNet.params,
    inverse: inverseNet.params,
  } as ParamTree
  const opt = optimiserOf(options, 1e-3)
  const beta = options.beta ?? 0.2
  const eta = options.eta ?? 1
  const batchSize = options.batchSize ?? 32
  const trainFrequency = options.trainFrequency ?? 1
  const start = env.actionSpace.kind === 'discrete' ? (env.actionSpace.start ?? 0) : 0
  const bonusOf = (s: IcmState, t: Transition<O, A>): number =>
    curiosityBonus(icm, s.params, [obs.of(t.observation)], [act.of(t.action)], [obs.of(t.nextObservation)], eta)[0] ?? 0
  // The weighted loss, with the forward and inverse losses as terms for the statistics.
  const loss = (p: ParamTree, batch: readonly IcmItem[]): readonly [Value, Value] => {
    const phi = icm.encode(
      p,
      rowsTensor(
        batch.map((b) => b.observation),
        obs.size,
      ),
    )
    const phiNext = icm.encode(
      p,
      rowsTensor(
        batch.map((b) => b.next),
        obs.size,
      ),
    )
    const a = rowsTensor(
      batch.map((b) => b.action),
      act.size,
    )
    // L_F = 1/2 ||phi_hat(s') - phi(s')||^2 per transition (mse also averages over the k features).
    const forward = mul(mse(icm.forward(p, phi, a), phiNext), k / 2)
    const logits = icm.inverse(p, phi, phiNext)
    const inverse = discreteActions ? crossEntropy(logits, vectorTensor(batch.map((b) => b.label))) : mse(logits, a)
    return [add(mul(inverse, 1 - beta), mul(forward, beta)), stack([forward, inverse])]
  }
  return {
    name: 'icm',
    config: { hidden: [...hidden], features: k, beta, eta },
    init: () => ({
      params,
      optState: opt.init(params),
      buffer: replayBuffer(options.bufferSize ?? 1000),
      steps: 0,
      forwardLoss: 0,
      inverseLoss: 0,
    }),
    bonus: bonusOf,
    update: (s0, t, key) => {
      const bonus = bonusOf(s0, t)
      let s: IcmState = {
        ...s0,
        buffer: bufferAdd(s0.buffer, {
          observation: obs.of(t.observation),
          action: act.of(t.action),
          next: obs.of(t.nextObservation),
          label: discreteActions ? (t.action as number) - start : 0,
        }),
        steps: s0.steps + 1,
      }
      if (s.steps % trainFrequency === 0 && s.buffer.size >= Math.min(batchSize, 1 + s.buffer.capacity / 2)) {
        const batch = bufferSample(s.buffer, key, batchSize).items
        const step = gradientStepWithTerms((p) => loss(p, batch), s.params, opt, s.optState)
        const [forwardLoss = 0, inverseLoss = 0] = step.terms
        s = { ...s, params: step.params, optState: step.optState, forwardLoss, inverseLoss }
      }
      return { state: s, bonus }
    },
    stats: (s) => ({ forwardLoss: s.forwardLoss, inverseLoss: s.inverseLoss }),
  }
}
