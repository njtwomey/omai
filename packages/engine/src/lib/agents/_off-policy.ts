/**
 * Plumbing shared by the off-policy actor-critic agents (actor-critic.ts: DDPG, TD3, SAC), private to the agents: their
 * options and state, replay batches, the agent that stores transitions, advances the exploration noise and updates, and
 * networks over box action spaces.
 */

import type { Json } from '../base/index.ts'
import type { ActivationName, Params } from '../nn/index.ts'
import { adam, type GradientTransformation } from '../optim/index.ts'
import { foldIn, type Key, key as makeKey, uniform } from '../random/index.ts'
import {
  type Agent,
  bufferAdd,
  bufferSample,
  type NoiseProcess,
  type ReplayBuffer,
  type Transition,
} from '../rl/index.ts'
import { concat, item, mean, primalOf, square, sub, sum, type Tensor, type Value } from '../tensor/index.ts'
import type { MetricChannel } from '../trace/index.ts'
import { actionBox, observationSize, rowsTensor, type Spaces, vectorTensor } from './_common.ts'
import { type Mlp, mlpNet } from './_networks.ts'

/** Options shared by the off-policy actor-critic agents (stable-baselines3's defaults, scaled down). */
export interface OffPolicyOptions {
  /** Hidden widths of the actor and critic networks (default `[64, 64]`). */
  readonly hidden?: readonly number[]
  /** The hidden activation (default `'relu'`). */
  readonly activation?: ActivationName
  /** Learning rate of the actor, critics and temperature (default 1e-3; 3e-4 for SAC). */
  readonly learningRate?: number
  readonly actorLearningRate?: number
  readonly criticLearningRate?: number
  readonly gamma?: number
  /** Polyak averaging weight of the target networks (default 0.005). */
  readonly tau?: number
  readonly bufferSize?: number
  readonly batchSize?: number
  /** Uniformly random actions for this many steps before learning starts (default 100). */
  readonly learningStarts?: number
  readonly trainFrequency?: number
  readonly gradientSteps?: number
  /** The initialisation key (default key(0)). */
  readonly key?: Key
}

/** The state of the off-policy actor-critic agents. */
export interface ActorCriticAgentState {
  readonly actor: Params
  /** The critic networks (`q1`, and `q2` for twin critics). */
  readonly critic: Params
  readonly actorTarget: Params
  readonly criticTarget: Params
  readonly actorOptState: unknown
  readonly criticOptState: unknown
  /** The log entropy temperature (SAC). */
  readonly logAlpha: number
  readonly alphaOptState: unknown
  readonly buffer: ReplayBuffer<Transition<readonly number[], number | readonly number[]>>
  /** The current exploration noise (DDPG and TD3), in normalised action units. */
  readonly noise: readonly number[]
  readonly steps: number
  readonly updates: number
  readonly stats: Readonly<Record<string, number>>
}

export type Stored = Transition<readonly number[], number | readonly number[]>

export const METRICS: Readonly<Record<string, MetricChannel>> = {
  actorLoss: { description: 'actor loss of the last update', direction: 'lower' },
  criticLoss: { description: 'critic (TD) loss of the last update', direction: 'lower' },
  qValue: { description: 'mean Q value of the last batch' },
}

export const SAC_METRICS: Readonly<Record<string, MetricChannel>> = {
  ...METRICS,
  alpha: { description: 'the entropy temperature' },
  entropy: { description: 'policy entropy estimate on the last batch' },
}

export interface Batch {
  readonly obs: Tensor
  readonly next: Tensor
  readonly rewards: Tensor
  readonly notDone: Tensor
  readonly items: readonly Stored[]
}

export function sampleBatch(s: ActorCriticAgentState, d: number, batchSize: number, k: Key): Batch {
  const items = bufferSample(s.buffer, k, batchSize).items
  return {
    obs: rowsTensor(
      items.map((t) => t.observation),
      d,
    ),
    next: rowsTensor(
      items.map((t) => t.nextObservation),
      d,
    ),
    rewards: vectorTensor(items.map((t) => t.reward)),
    // Only terminal transitions stop the bootstrap; truncated ones bootstrap from their final observation.
    notDone: vectorTensor(items.map((t) => (t.terminated ? 0 : 1))),
    items,
  }
}

export interface Machinery {
  readonly name: string
  readonly config: Json
  readonly metrics: Readonly<Record<string, MetricChannel>>
  readonly init: (k: Key) => ActorCriticAgentState
  /** The policy's action once learning has started (exploration included). */
  readonly act: (s: ActorCriticAgentState, obs: readonly number[], k: Key) => number | readonly number[]
  readonly random: (k: Key) => number | readonly number[]
  readonly greedy: (s: ActorCriticAgentState, obs: readonly number[]) => number | readonly number[]
  readonly update: (s: ActorCriticAgentState, k: Key) => ActorCriticAgentState
  readonly noise?: NoiseProcess
}

export function offPolicyAgent<A>(
  m: Machinery,
  options: OffPolicyOptions,
): Agent<ActorCriticAgentState, readonly number[], A> {
  const learningStarts = options.learningStarts ?? 100
  const trainFrequency = options.trainFrequency ?? 1
  const gradientSteps = options.gradientSteps ?? 1
  const batchSize = options.batchSize ?? 64
  return {
    name: m.name,
    config: m.config,
    metrics: m.metrics,
    init: m.init,
    act: (s, obs, k) => (s.steps < learningStarts ? m.random(k) : m.act(s, obs, k)) as A,
    greedy: (s, obs) => m.greedy(s, obs) as A,
    learn: (s, t, k) => {
      const done = t.terminated || t.truncated
      const noise = m.noise === undefined ? s.noise : done ? m.noise.init() : m.noise.step(s.noise, foldIn(k, 99))
      let next: ActorCriticAgentState = {
        ...s,
        buffer: bufferAdd(s.buffer, t as Stored),
        noise,
        steps: s.steps + 1,
      }
      if (next.steps >= learningStarts && next.steps % trainFrequency === 0 && next.buffer.size >= batchSize) {
        for (let g = 0; g < gradientSteps; g++) next = { ...m.update(next, foldIn(k, g)), updates: next.updates + 1 }
      }
      return next
    },
    stats: (s) => s.stats,
  }
}

/** Networks over a box action space: actions are normalised to [-1, 1] inside the critics, as stable-baselines3 does. */
export interface BoxNetworks {
  readonly d: number
  readonly k: number
  readonly box: { readonly size: number; readonly low: number[]; readonly high: number[] }
  readonly actor: Mlp
  readonly q1: Mlp
  readonly q2: Mlp
  readonly scale: number[]
  readonly centre: number[]
  /** Normalised actions in [-1, 1] (clipped) to environment actions. */
  toEnv(u: readonly number[]): number[]
  normalise(a: readonly number[]): number[]
  /** A critic's values `[B]` of observations and normalised actions. */
  q(net: 'q1' | 'q2', critic: Params, obs: Value, u: Value): Value
  /** The normalised actions of stored transitions. */
  actions(items: readonly Stored[]): Tensor
  random(key: Key): number[]
}

export function boxNetworks(
  env: Spaces,
  options: OffPolicyOptions,
  name: string,
  actorOutputs: (k: number) => number,
): BoxNetworks {
  const d = observationSize(env.observationSpace)
  const box = actionBox(env.actionSpace, name)
  const k = box.size
  const hidden = options.hidden ?? [64, 64]
  const activation = options.activation ?? 'relu'
  const key = options.key ?? makeKey(0)
  const actor = mlpNet([d, ...hidden, actorOutputs(k)], activation, foldIn(key, 0))
  const q1 = mlpNet([d + k, ...hidden, 1], activation, foldIn(key, 1))
  const q2 = mlpNet([d + k, ...hidden, 1], activation, foldIn(key, 2))
  const scale = box.low.map((l, i) => ((box.high[i] ?? 1) - l) / 2)
  const centre = box.low.map((l, i) => ((box.high[i] ?? 1) + l) / 2)
  const toEnv = (u: readonly number[]): number[] =>
    u.map((x, i) => (centre[i] ?? 0) + (scale[i] ?? 1) * Math.min(1, Math.max(-1, x)))
  const normalise = (a: readonly number[]): number[] => a.map((x, i) => (x - (centre[i] ?? 0)) / (scale[i] ?? 1))
  const q = (net: 'q1' | 'q2', critic: Params, obs: Value, u: Value): Value =>
    sum((net === 'q1' ? q1 : q2).apply(critic[net] as Params, concat([obs, u], 1)), -1)
  const actions = (items: readonly Stored[]): Tensor =>
    rowsTensor(
      items.map((t) => normalise(t.action as readonly number[])),
      k,
    )
  const random = (key2: Key): number[] => toEnv(Array.from(uniform(key2, [k], { low: -1, high: 1 }).data))
  return { d, k, box, actor, q1, q2, scale, centre, toEnv, normalise, q, actions, random }
}

export const mse = (a: Value, b: Value): Value => mean(square(sub(a, b)))
export const scalar = (v: Value): number => item(primalOf(v))

/** Discount, Polyak weight, batch and buffer sizes and the actor and critic optimisers of an agent. */
export interface Common {
  readonly gamma: number
  readonly tau: number
  readonly batchSize: number
  readonly actorOpt: GradientTransformation
  readonly criticOpt: GradientTransformation
  readonly bufferSize: number
}

export function common(options: OffPolicyOptions, lr: number): Common {
  return {
    gamma: options.gamma ?? 0.99,
    tau: options.tau ?? 0.005,
    batchSize: options.batchSize ?? 64,
    actorOpt: adam({ learningRate: options.actorLearningRate ?? options.learningRate ?? lr }),
    criticOpt: adam({ learningRate: options.criticLearningRate ?? options.learningRate ?? lr }),
    bufferSize: options.bufferSize ?? 10_000,
  }
}

export function baseConfig(options: OffPolicyOptions, c: Common): Record<string, Json> {
  return {
    hidden: [...(options.hidden ?? [64, 64])],
    gamma: c.gamma,
    tau: c.tau,
    batchSize: c.batchSize,
    bufferSize: c.bufferSize,
    learningStarts: options.learningStarts ?? 100,
  }
}
