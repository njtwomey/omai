/**
 * Cooperative multi-agent deep RL with centralised training and decentralised execution. Value decomposition builds the
 * team's action value from per-agent values so each agent can act greedily on its own: `vdn` sums them and `qmix` mixes
 * them through a monotonic network conditioned on the global state. `maddpg` trains per-agent deterministic actors
 * against centralised critics that see every agent's observation and action.
 */

import { argmaxOf, type Json, maxOf, WhiteboxError } from '../../../lib/base/index.ts'
import { mse } from '../../../lib/losses/index.ts'
import { elu, mlp } from '../../../lib/nn/index.ts'
import { adam, incrementalUpdate, linearSchedule, type Tree } from '../../../lib/optim/index.ts'
import { foldIn, type Key, key as makeKey, normal, uniform } from '../../../lib/random/index.ts'
import {
  type Agent,
  bufferAdd,
  bufferSample,
  type PerAgent,
  type ReplayBuffer,
  replayBuffer,
} from '../../../lib/rl/index.ts'
import type { Space } from '../../../lib/space/index.ts'
import {
  abs,
  add,
  concat,
  fromStorage,
  mean,
  mul,
  neg,
  oneHot,
  reshape,
  square,
  sum,
  tanh,
  toFlat,
  type Value,
} from '../../../lib/tensor/index.ts'
import { actionBox, actionCount, gradientStep, rowsTensor } from './_common.ts'
import { encodedSize, encodeObservation, randomIndex } from './_multi-agent.ts'
import type { MultiAgentSpaces } from './multi-agent.ts'

// ------------------------------------------------------------------------------------------ value decomposition

/** Options of {@link vdn} and {@link qmix}. */
export interface ValueDecompositionOptions {
  /** Hidden widths of the shared agent network (default `[64]`). */
  readonly hidden?: readonly number[]
  /** QMIX: the mixing embedding size (default 32). */
  readonly embed?: number
  readonly learningRate?: number
  readonly gamma?: number
  readonly bufferSize?: number
  readonly batchSize?: number
  /** Steps before learning (default 100), learning frequency (default 1) and hard target updates (default 200). */
  readonly learningStarts?: number
  readonly trainFrequency?: number
  readonly targetUpdateInterval?: number
  /** Epsilon-greedy schedule: linear from `epsilonStart` (1) to `epsilonEnd` (0.05) over `explorationSteps` (1000). */
  readonly epsilonStart?: number
  readonly epsilonEnd?: number
  readonly explorationSteps?: number
  /** The global state for the QMIX mixer (default every agent's observation, concatenated). */
  readonly state?: (observations: readonly (readonly number[])[]) => readonly number[]
  readonly key?: Key
}

interface JointSample {
  readonly observations: readonly (readonly number[])[]
  readonly actions: readonly number[]
  readonly reward: number
  readonly next: readonly (readonly number[])[]
  readonly terminated: boolean
}

/** The state of {@link vdn} and {@link qmix}. */
export interface ValueDecompositionState {
  readonly params: Tree
  readonly target: Tree
  readonly optState: unknown
  readonly buffer: ReplayBuffer<JointSample>
  readonly steps: number
  readonly updates: number
  readonly loss: number
}

function valueDecomposition(
  kind: 'vdn' | 'qmix',
  env: MultiAgentSpaces,
  options: ValueDecompositionOptions,
): Agent<ValueDecompositionState, PerAgent<unknown>, PerAgent<number>> {
  const agents = env.possibleAgents
  const n = agents.length
  const nA = actionCount(env.actionSpaces[agents[0] ?? ''] as Space, kind)
  for (const a of agents)
    if (actionCount(env.actionSpaces[a] as Space, kind) !== nA)
      throw new WhiteboxError('value', `${kind}: every agent needs the same number of actions`)
  const d = maxOf(agents.map((a) => encodedSize(env.observationSpaces[a])))
  const inputs = d + n
  const k0 = options.key ?? makeKey(0)
  const agentNet = mlp([inputs, ...(options.hidden ?? [64]), nA], { key: foldIn(k0, 0) })
  const embed = options.embed ?? 32
  const stateOf =
    options.state ?? ((obs: readonly (readonly number[])[]): readonly number[] => obs.flatMap((o) => [...o]))
  const sd = kind === 'qmix' ? stateOf(agents.map(() => new Array<number>(d).fill(0))).length : 0
  const hyper = {
    w1: mlp([sd, n * embed], { key: foldIn(k0, 1) }),
    b1: mlp([sd, embed], { key: foldIn(k0, 2) }),
    w2: mlp([sd, embed], { key: foldIn(k0, 3) }),
    v: mlp([sd, embed, 1], { key: foldIn(k0, 4) }),
  }
  const initial: Tree =
    kind === 'qmix'
      ? {
          agent: agentNet.params,
          mixer: { w1: hyper.w1.params, b1: hyper.b1.params, w2: hyper.w2.params, v: hyper.v.params },
        }
      : { agent: agentNet.params }
  const gamma = options.gamma ?? 0.99
  const batchSize = options.batchSize ?? 32
  const learningStarts = options.learningStarts ?? 100
  const trainFrequency = options.trainFrequency ?? 1
  const targetInterval = options.targetUpdateInterval ?? 200
  const opt = adam({ learningRate: options.learningRate ?? 5e-4 })
  const epsilon = linearSchedule(
    options.epsilonStart ?? 1,
    options.epsilonEnd ?? 0.05,
    options.explorationSteps ?? 1000,
  )
  const pad = (o: readonly number[], i: number): number[] => {
    const row = new Array<number>(inputs).fill(0)
    for (let j = 0; j < o.length && j < d; j++) row[j] = o[j] ?? 0
    row[d + i] = 1
    return row
  }
  const encodeAll = (obs: PerAgent<unknown>): number[][] =>
    agents.map((a) => encodeObservation(env.observationSpaces[a], obs[a]))
  type P = { readonly agent: Tree; readonly mixer?: Readonly<Record<string, Tree>> }
  // Per-agent action values [B, n, nA] of a batch of joint observations.
  const agentQ = (p: P, batch: readonly (readonly (readonly number[])[])[]): Value =>
    reshape(
      agentNet.apply(
        p.agent as typeof agentNet.params,
        rowsTensor(
          batch.flatMap((obs) => obs.map((o, i) => pad(o, i))),
          inputs,
        ),
      ),
      [batch.length, n, nA],
    )
  // The joint value of per-agent values [B, n] in the states of a batch.
  const mix = (p: P, chosen: Value, batch: readonly (readonly (readonly number[])[])[]): Value => {
    const B = batch.length
    if (kind === 'vdn') return sum(chosen, -1)
    const m = p.mixer ?? {}
    const st = rowsTensor(
      batch.map((obs) => stateOf(obs)),
      sd,
    )
    const w1 = abs(reshape(hyper.w1.apply(m['w1'] as typeof hyper.w1.params, st), [B, n, embed]))
    const b1 = hyper.b1.apply(m['b1'] as typeof hyper.b1.params, st)
    const hidden = elu(add(sum(mul(reshape(chosen, [B, n, 1]), w1), 1), b1))
    const w2 = abs(hyper.w2.apply(m['w2'] as typeof hyper.w2.params, st))
    const v = reshape(hyper.v.apply(m['v'] as typeof hyper.v.params, st), [B])
    return add(sum(mul(hidden, w2), -1), v)
  }
  const greedyActions = (p: Tree, obs: PerAgent<unknown>): number[] => {
    const q = toFlat(agentQ(p as P, [encodeAll(obs)]))
    return agents.map((_, i) => argmaxOf(q.slice(i * nA, (i + 1) * nA)))
  }
  const update = (s: ValueDecompositionState, k: Key): ValueDecompositionState => {
    const batch = bufferSample(s.buffer, k, batchSize).items
    const obs = batch.map((b) => b.observations)
    const next = batch.map((b) => b.next)
    const mask = oneHot(fromStorage(Int32Array.from(batch.flatMap((b) => b.actions)), [batch.length, n], 'int32'), nA)
    // Targets r + gamma (1 - done) Q_tot^-(s', argmax) from the target networks, outside the gradient.
    const nextQ = agentQ(s.target as P, next)
    const nextFlat = toFlat(nextQ)
    const best = Array.from({ length: batch.length * n }, (_, j) => maxOf(nextFlat.slice(j * nA, (j + 1) * nA)))
    const nextTot = toFlat(mix(s.target as P, fromStorage(Float64Array.from(best), [batch.length, n], 'float64'), next))
    const y = fromStorage(
      Float64Array.from(batch.map((b, i) => b.reward + (b.terminated ? 0 : gamma * (nextTot[i] ?? 0)))),
      [batch.length],
      'float64',
    )
    const step = gradientStep(
      (p) => {
        const chosen = sum(mul(agentQ(p as P, obs), mask), -1)
        return mse(mix(p as P, chosen, obs), y)
      },
      s.params,
      opt,
      s.optState,
    )
    return { ...s, params: step.params, optState: step.optState, updates: s.updates + 1, loss: step.loss }
  }
  return {
    name: kind,
    config: { agents: [...agents], hidden: [...(options.hidden ?? [64])], gamma, batchSize, embed } as Json,
    metrics: {
      epsilon: { description: 'exploration rate' },
      loss: { description: 'TD loss of the joint value', direction: 'lower' },
    },
    init: () => ({
      params: initial,
      target: initial,
      optState: opt.init(initial),
      buffer: replayBuffer(options.bufferSize ?? 5000),
      steps: 0,
      updates: 0,
      loss: 0,
    }),
    act: (s, obs, k) => {
      const greedy = greedyActions(s.params, obs)
      const eps = epsilon(s.steps)
      return Object.fromEntries(
        agents.map((a, i) => {
          const ki = foldIn(k, i)
          const explore = (uniform(foldIn(ki, 0), []).data[0] ?? 1) < eps
          return [a, explore ? randomIndex(foldIn(ki, 1), nA) : (greedy[i] ?? 0)]
        }),
      )
    },
    greedy: (s, obs) => {
      const g = greedyActions(s.params, obs)
      return Object.fromEntries(agents.map((a, i) => [a, g[i] ?? 0]))
    },
    learn: (s, t, k) => {
      const sample: JointSample = {
        observations: encodeAll(t.observation),
        actions: agents.map((a) => t.action[a] ?? 0),
        reward: t.reward,
        next: encodeAll(t.nextObservation),
        terminated: t.terminated,
      }
      let next: ValueDecompositionState = { ...s, buffer: bufferAdd(s.buffer, sample), steps: s.steps + 1 }
      if (next.steps >= learningStarts && next.steps % trainFrequency === 0 && next.buffer.size >= batchSize)
        next = update(next, k)
      if (next.steps % targetInterval === 0) next = { ...next, target: next.params }
      return next
    },
    stats: (s) => ({ epsilon: epsilon(s.steps), loss: s.loss }),
  }
}

/**
 * Value-decomposition networks (VDN, Sunehag et al. 2018) for cooperative tasks: one agent network (shared, with the
 * agent's one-hot id appended to its observation) gives per-agent action values, the team value is their sum
 * `Q_tot = sum_i Q_i(o_i, a_i)`, and it is trained end to end on the team reward with replay and a target network;
 * each agent acts epsilon-greedily on its own values (decentralised execution). Runs on `jointEnv` (whose scalar
 * reward is the team's).
 *
 * @param env - The cooperative parallel environment (discrete actions, the same number for every agent).
 * @param options - Network, replay, exploration and target-update options.
 * @returns The joint agent.
 * @concept value-decomposition
 * @example
 * ```ts
 * import { vdn } from 'whitebox-engine/methods/rl/agents'
 * import { cooperativeNavigation } from 'whitebox-engine/methods/rl'
 * vdn(cooperativeNavigation.create({ agents: 2 }), { hidden: [16] }).name // => 'vdn'
 * ```
 */
export function vdn(
  env: MultiAgentSpaces,
  options: ValueDecompositionOptions = {},
): Agent<ValueDecompositionState, PerAgent<unknown>, PerAgent<number>> {
  return valueDecomposition('vdn', env, options)
}

/**
 * QMIX (Rashid et al. 2018): as {@link vdn}, but the team value is a monotonic mixing network of the agents' values
 * conditioned on the global state, `Q_tot = w2(s)^T elu(W1(s)^T Q + b1(s)) + V(s)`, whose weights come from
 * hypernetworks of the state passed through absolute values (so `dQ_tot / dQ_i >= 0` and each agent's greedy action
 * is consistent with the joint greedy action); `V(s)` is a two-layer network. The state defaults to every agent's
 * observation concatenated.
 *
 * @param env - The cooperative parallel environment.
 * @param options - Network, mixer (`embed`, `state`), replay, exploration and target-update options.
 * @returns The joint agent.
 * @concept value-decomposition
 * @example
 * ```ts
 * import { qmix } from 'whitebox-engine/methods/rl/agents'
 * import { cooperativeNavigation } from 'whitebox-engine/methods/rl'
 * qmix(cooperativeNavigation.create({ agents: 2 }), { hidden: [16], embed: 8 }).name // => 'qmix'
 * ```
 */
export function qmix(
  env: MultiAgentSpaces,
  options: ValueDecompositionOptions = {},
): Agent<ValueDecompositionState, PerAgent<unknown>, PerAgent<number>> {
  return valueDecomposition('qmix', env, options)
}

// --------------------------------------------------------------------------------------------------------- MADDPG

/** Options of {@link maddpg}. */
export interface MaddpgOptions {
  /** Hidden widths of the actors and critics (default `[64, 64]`). */
  readonly hidden?: readonly number[]
  readonly actorLearningRate?: number
  readonly criticLearningRate?: number
  readonly gamma?: number
  /** Polyak averaging of the target networks (default 0.01). */
  readonly tau?: number
  readonly bufferSize?: number
  readonly batchSize?: number
  /** Steps before learning (default 200) and learning frequency (default 1). */
  readonly learningStarts?: number
  readonly trainFrequency?: number
  /** Standard deviation of the Gaussian exploration noise, in action units (default 0.1). */
  readonly noise?: number
  /** Weight of the squared-action regulariser of the actor loss (default 1e-3, as the reference code). */
  readonly actionPenalty?: number
  readonly key?: Key
}

interface MaddpgSample {
  readonly observations: readonly (readonly number[])[]
  readonly actions: readonly (readonly number[])[]
  readonly rewards: readonly number[]
  readonly next: readonly (readonly number[])[]
  readonly terminated: boolean
}

/** The state of {@link maddpg}: per-agent actors and critics with their targets and optimisers, and the replay. */
export interface MaddpgState {
  readonly actors: readonly Tree[]
  readonly critics: readonly Tree[]
  readonly targetActors: readonly Tree[]
  readonly targetCritics: readonly Tree[]
  readonly actorOpt: readonly unknown[]
  readonly criticOpt: readonly unknown[]
  readonly buffer: ReplayBuffer<MaddpgSample>
  readonly steps: number
  readonly criticLoss: number
  readonly actorLoss: number
}

/**
 * MADDPG (Lowe et al. 2017): multi-agent deep deterministic policy gradients with centralised training and
 * decentralised execution. Each agent has a deterministic actor on its own observation (tanh, scaled to its action
 * box) and a centralised critic `Q_i(o_1..o_n, a_1..a_n)` on every agent's observation and action; critics regress
 * on `r_i + gamma Q_i'(o', mu_1'(o_1')..mu_n'(o_n'))` with target actors and critics, actors ascend
 * `Q_i(o, a_1..mu_i(o_i)..a_n)` with the other actions from the replay (plus a small action penalty), and targets
 * follow by Polyak averaging. Exploration adds Gaussian noise. Runs on `jointEnv` of a parallel environment with box
 * actions (each agent's own reward comes from `info.rewards`).
 *
 * @param env - The parallel environment (vector observations, box actions).
 * @param options - Networks, learning rates, discount, Polyak rate, replay and noise.
 * @returns The joint agent.
 * @concept actor-critic
 * @example
 * ```ts
 * import { maddpg } from 'whitebox-engine/methods/rl/agents'
 * import { cooperativeNavigation } from 'whitebox-engine/methods/rl'
 * import { key } from 'whitebox-engine/random'
 * const agent = maddpg(cooperativeNavigation.create({ agents: 2, continuous: true }), { hidden: [16] })
 * agent.init(key(0)).actors.length // => 2
 * ```
 */
export function maddpg(
  env: MultiAgentSpaces,
  options: MaddpgOptions = {},
): Agent<MaddpgState, PerAgent<readonly number[]>, PerAgent<readonly number[]>> {
  const agents = env.possibleAgents
  const n = agents.length
  const sizes = agents.map((a) => encodedSize(env.observationSpaces[a]))
  const boxes = agents.map((a) => actionBox(env.actionSpaces[a] as Space, 'maddpg'))
  const totalObs = sizes.reduce((x, y) => x + y, 0)
  const totalAct = boxes.reduce((x, b) => x + b.size, 0)
  const hidden = options.hidden ?? [64, 64]
  const k0 = options.key ?? makeKey(0)
  const actorNets = agents.map((_, i) =>
    mlp([sizes[i] ?? 0, ...hidden, boxes[i]?.size ?? 0], { key: foldIn(k0, 2 * i) }),
  )
  const criticNets = agents.map((_, i) => mlp([totalObs + totalAct, ...hidden, 1], { key: foldIn(k0, 2 * i + 1) }))
  const gamma = options.gamma ?? 0.95
  const tau = options.tau ?? 0.01
  const batchSize = options.batchSize ?? 64
  const learningStarts = options.learningStarts ?? 200
  const trainFrequency = options.trainFrequency ?? 1
  const sigma = options.noise ?? 0.1
  const penalty = options.actionPenalty ?? 1e-3
  const actorOpt = adam({ learningRate: options.actorLearningRate ?? 1e-3 })
  const criticOpt = adam({ learningRate: options.criticLearningRate ?? 1e-3 })
  // Actor outputs in [-1, 1], mapped to the action box.
  const scale = (i: number, u: Value): Value => {
    const b = boxes[i] ?? { size: 0, low: [], high: [] }
    const half = fromStorage(Float64Array.from(b.high.map((h, j) => (h - (b.low[j] ?? 0)) / 2)), [b.size], 'float64')
    const mid = fromStorage(Float64Array.from(b.high.map((h, j) => (h + (b.low[j] ?? 0)) / 2)), [b.size], 'float64')
    return add(mul(u, half), mid)
  }
  const actor = (i: number, p: Tree, x: Value): Value =>
    scale(i, tanh((actorNets[i] as (typeof actorNets)[0]).apply(p as never, x)))
  const critic = (i: number, p: Tree, x: Value): Value =>
    reshape((criticNets[i] as (typeof criticNets)[0]).apply(p as never, x), [x.shape[0] ?? 1])
  const obsTensor = (rows: readonly (readonly (readonly number[])[])[], i: number): Value =>
    rowsTensor(
      rows.map((r) => r[i] ?? []),
      sizes[i] ?? 0,
    )
  const encodeAll = (obs: PerAgent<readonly number[]>): number[][] =>
    agents.map((a) => encodeObservation(env.observationSpaces[a], obs[a]))
  const actions = (s: MaddpgState, obs: PerAgent<readonly number[]>): number[][] => {
    const enc = encodeAll(obs)
    return agents.map((_, i) => toFlat(actor(i, s.actors[i], rowsTensor([enc[i] ?? []], sizes[i] ?? 0))))
  }
  const update = (s: MaddpgState, k: Key): MaddpgState => {
    const batch = bufferSample(s.buffer, k, batchSize).items
    const B = batch.length
    const obsAll = rowsTensor(
      batch.map((b) => b.observations.flat()),
      totalObs,
    )
    const nextAll = rowsTensor(
      batch.map((b) => b.next.flat()),
      totalObs,
    )
    const stored = agents.map((_, i) =>
      rowsTensor(
        batch.map((b) => b.actions[i] ?? []),
        boxes[i]?.size ?? 0,
      ),
    )
    const nextActions = agents.map((_, i) =>
      actor(
        i,
        s.targetActors[i],
        obsTensor(
          batch.map((b) => b.next),
          i,
        ),
      ),
    )
    const nextInput = concat([nextAll, ...nextActions], -1)
    const input = concat([obsAll, ...stored], -1)
    const critics = [...s.critics]
    const actorsNext = [...s.actors]
    const cOpt = [...s.criticOpt]
    const aOpt = [...s.actorOpt]
    let criticLoss = 0
    let actorLoss = 0
    for (let i = 0; i < n; i++) {
      const qNext = toFlat(critic(i, s.targetCritics[i], nextInput))
      const y = fromStorage(
        Float64Array.from(batch.map((b, j) => (b.rewards[i] ?? 0) + (b.terminated ? 0 : gamma * (qNext[j] ?? 0)))),
        [B],
        'float64',
      )
      const c = gradientStep((p) => mse(critic(i, p, input), y), critics[i], criticOpt, cOpt[i])
      critics[i] = c.params
      cOpt[i] = c.optState
      criticLoss += c.loss / n
      const own = obsTensor(
        batch.map((b) => b.observations),
        i,
      )
      const a = gradientStep(
        (p) => {
          const mine = actor(i, p, own)
          const joint = concat([obsAll, ...stored.map((x, j) => (j === i ? mine : x))], -1)
          return add(neg(mean(critic(i, critics[i], joint))), mul(mean(square(mine)), penalty))
        },
        actorsNext[i],
        actorOpt,
        aOpt[i],
      )
      actorsNext[i] = a.params
      aOpt[i] = a.optState
      actorLoss += a.loss / n
    }
    return {
      ...s,
      actors: actorsNext,
      critics,
      actorOpt: aOpt,
      criticOpt: cOpt,
      targetActors: actorsNext.map((p, i) => incrementalUpdate(p, s.targetActors[i], tau)),
      targetCritics: critics.map((p, i) => incrementalUpdate(p, s.targetCritics[i], tau)),
      criticLoss,
      actorLoss,
    }
  }
  return {
    name: 'maddpg',
    config: { agents: [...agents], hidden: [...hidden], gamma, tau, batchSize, noise: sigma } as Json,
    metrics: {
      criticLoss: { description: 'mean squared TD error of the critics', direction: 'lower' },
      actorLoss: { description: 'negated critic value of the actors', direction: 'lower' },
    },
    init: () => {
      const actors = actorNets.map((m) => m.params as Tree)
      const critics = criticNets.map((m) => m.params as Tree)
      return {
        actors,
        critics,
        targetActors: actors,
        targetCritics: critics,
        actorOpt: actors.map((p) => actorOpt.init(p)),
        criticOpt: critics.map((p) => criticOpt.init(p)),
        buffer: replayBuffer(options.bufferSize ?? 10_000),
        steps: 0,
        criticLoss: 0,
        actorLoss: 0,
      }
    },
    act: (s, obs, k) => {
      const greedy = actions(s, obs)
      return Object.fromEntries(
        agents.map((a, i) => {
          const b = boxes[i] ?? { size: 0, low: [], high: [] }
          const z = normal(foldIn(k, i), [b.size]).data
          const noisy = (greedy[i] ?? []).map((v, j) => {
            const width = (b.high[j] ?? 1) - (b.low[j] ?? -1)
            return Math.min(b.high[j] ?? 1, Math.max(b.low[j] ?? -1, v + (sigma * width * (z[j] ?? 0)) / 2))
          })
          return [a, noisy]
        }),
      )
    },
    greedy: (s, obs) => {
      const g = actions(s, obs)
      return Object.fromEntries(agents.map((a, i) => [a, g[i] ?? []]))
    },
    learn: (s, t, k) => {
      const rewards = t.info?.['rewards'] as Readonly<Record<string, number>> | undefined
      const sample: MaddpgSample = {
        observations: encodeAll(t.observation),
        actions: agents.map((a) => [...(t.action[a] ?? [])]),
        rewards: agents.map((a) => rewards?.[a] ?? t.reward),
        next: encodeAll(t.nextObservation),
        terminated: t.terminated,
      }
      const next: MaddpgState = { ...s, buffer: bufferAdd(s.buffer, sample), steps: s.steps + 1 }
      if (next.steps >= learningStarts && next.steps % trainFrequency === 0 && next.buffer.size >= batchSize)
        return update(next, k)
      return next
    },
    stats: (s) => ({ criticLoss: s.criticLoss, actorLoss: s.actorLoss }),
  }
}
