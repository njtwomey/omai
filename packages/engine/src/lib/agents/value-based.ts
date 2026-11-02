/**
 * Deep Q-networks (DQN, Mnih et al. 2015): a neural network that estimates each action's value, trained from a replay
 * buffer against a periodically synced target network. The Q-network builder (an MLP trunk with optional duelling heads
 * and noisy layers, initialised as PyTorch initialises linear layers), the `dqn` agent with double Q-learning,
 * prioritised replay, n-step returns and noisy exploration as options, and a reader of its action values.
 */

import { argmaxOf, type Json, MdmlError } from '../base/index.ts'
import { huber } from '../losses/index.ts'
import { linear } from '../nn/index.ts'
import { type GradientTransformation, incrementalUpdate, linearSchedule, type Tree } from '../optim/index.ts'
import { foldIn, type Key, key as makeKey, normal, uniform } from '../random/index.ts'
import {
  type Agent,
  bufferAdd,
  bufferSample,
  type NStepBuffer,
  type NStepTransition,
  nStepAdd,
  nStepBuffer,
  type PrioritizedReplayBuffer,
  prioritizedAdd,
  prioritizedReplay,
  prioritizedSample,
  prioritizedUpdate,
  type ReplayBuffer,
  replayBuffer,
  type Transition,
} from '../rl/index.ts'
import {
  abs,
  add,
  argmax,
  fromStorage,
  mean,
  mul,
  oneHot,
  relu,
  reshape,
  sign,
  sqrt,
  sub,
  sum,
  type Tensor,
  toFlat,
  type Value,
} from '../tensor/index.ts'
import {
  actionCount,
  gradientStep,
  observationSize,
  optimiserOf,
  rowsTensor,
  type Spaces,
  vectorTensor,
} from './_common.ts'

// ----------------------------------------------------------------------------------------------- Q networks

/** Parameters of a Q network: a tree of tensors. */
export type QParams = Readonly<Record<string, Readonly<Record<string, Tensor | Value>>>>

/** Options of {@link qNetwork}. */
export interface QNetworkOptions {
  /** Hidden layer widths (default `[64, 64]`). */
  readonly hidden?: readonly number[]
  /** Separate value and advantage streams combined as `V + A - mean(A)` (Wang et al. 2016). */
  readonly dueling?: boolean
  /** Factorised Gaussian noisy layers (Fortunato et al. 2018) instead of linear ones. */
  readonly noisy?: boolean
  /** Initial noise scale of noisy layers (default 0.5, the paper's sigma_0). */
  readonly sigma0?: number
  /** Outputs per action (atoms of a return distribution; default 1). */
  readonly atoms?: number
}

/**
 * A Q network as a pure function of its parameters: `apply(params, x, noise?)` maps a batch of observations `[B, d]`
 * to action values `[B, nActions]` (or `[B, nActions, atoms]` for distributional agents). With noisy layers a key
 * draws fresh factorised noise; without one the layers use their mean weights (greedy evaluation).
 *
 * @concept deep-q-learning
 */
export interface QNetwork {
  readonly params: QParams
  readonly nActions: number
  readonly atoms: number
  apply(params: QParams, x: Value, noise?: Key): Value
}

function uniformTensor(k: Key, shape: number[], bound: number): Tensor {
  return uniform(k, shape, { low: -bound, high: bound })
}

// f(x) = sign(x) sqrt(|x|), the noise transform of factorised noisy layers.
const scaleNoise = (x: Tensor): Value => mul(sign(x), sqrt(abs(x)))

/** The parameters of one Q-network layer: `weight` and `bias`, or the means and scales of a noisy layer. */
export type QLayerParams = Readonly<Record<string, Tensor | Value>>

/**
 * Initialise one layer of a Q network as PyTorch initialises linear layers (uniform in `+-1 / sqrt(inputs)`), or as a
 * factorised noisy layer (Fortunato et al. 2018) with means drawn the same way and scales `sigma0 / sqrt(inputs)`.
 *
 * @param k - The key.
 * @param inputs - The input width.
 * @param outputs - The output width.
 * @param noisy - Whether the layer is noisy.
 * @param sigma0 - The initial noise scale of a noisy layer.
 * @returns The layer's parameters.
 * @concept deep-q-learning
 * @example
 * ```ts
 * import { qLayerInit } from 'mdml-engine/agents'
 * import { key } from 'mdml-engine/random'
 * Object.keys(qLayerInit(key(0), 3, 2, true, 0.5)) // => ['weightMu', 'biasMu', 'weightSigma', 'biasSigma']
 * ```
 */
export function qLayerInit(
  k: Key,
  inputs: number,
  outputs: number,
  noisy: boolean,
  sigma0: number,
): Record<string, Tensor> {
  const bound = 1 / Math.sqrt(inputs)
  if (!noisy)
    return {
      weight: uniformTensor(foldIn(k, 0), [outputs, inputs], bound),
      bias: uniformTensor(foldIn(k, 1), [outputs], bound),
    }
  const s = sigma0 / Math.sqrt(inputs)
  return {
    weightMu: uniformTensor(foldIn(k, 0), [outputs, inputs], bound),
    biasMu: uniformTensor(foldIn(k, 1), [outputs], bound),
    weightSigma: fromStorage(new Float64Array(outputs * inputs).fill(s), [outputs, inputs], 'float64'),
    biasSigma: fromStorage(new Float64Array(outputs).fill(s), [outputs], 'float64'),
  }
}

/**
 * Apply one layer of a Q network: `x W^T + b`, or for a noisy layer `x (W_mu + W_sigma * eps_W)^T + b_mu + b_sigma *
 * eps_b` with factorised noise `eps_W = f(eps_out) f(eps_in)^T`, `f(x) = sign(x) sqrt(|x|)`, drawn from `noise`.
 * Without a key a noisy layer uses its mean weights.
 *
 * @param p - The layer's parameters (from {@link qLayerInit}).
 * @param x - The input `[..., inputs]`.
 * @param noise - The noise key (noisy layers only).
 * @returns The output `[..., outputs]`, differentiable in the parameters.
 * @concept deep-q-learning
 * @example
 * ```ts
 * import { qLayer, qLayerInit } from 'mdml-engine/agents'
 * import { key } from 'mdml-engine/random'
 * import { tensor } from 'mdml-engine/tensor'
 * qLayer(qLayerInit(key(0), 3, 2, false, 0.5), tensor([[1, 2, 3]])).shape // => [1, 2]
 * ```
 */
export function qLayer(p: QLayerParams, x: Value, noise?: Key): Value {
  const w = p['weight']
  if (w !== undefined) return linear(x, w, p['bias'])
  const wMu = p['weightMu'] as Value
  const bMu = p['biasMu'] as Value
  if (noise === undefined) return linear(x, wMu, bMu)
  const [outputs = 0, inputs = 0] = wMu.shape
  const eIn = scaleNoise(normal(foldIn(noise, 0), [inputs]))
  const eOut = scaleNoise(normal(foldIn(noise, 1), [outputs]))
  const epsW = mul(reshape(eOut, [outputs, 1]), reshape(eIn, [1, inputs]))
  return linear(x, add(wMu, mul(p['weightSigma'] as Value, epsW)), add(bMu, mul(p['biasSigma'] as Value, eOut)))
}

/**
 * Build a Q network: an MLP trunk with ReLUs, then one head (or value and advantage heads when duelling), with
 * linear or noisy layers, initialised as PyTorch initialises linear layers.
 *
 * @param inputSize - The observation size.
 * @param nActions - The number of actions.
 * @param options - Hidden widths, duelling, noisy layers and atoms.
 * @param key - The initialisation key.
 * @returns The network.
 * @concept deep-q-learning
 * @example
 * ```ts
 * import { qNetwork } from 'mdml-engine/agents'
 * import { key } from 'mdml-engine/random'
 * import { tensor } from 'mdml-engine/tensor'
 * const net = qNetwork(4, 2, { hidden: [8], dueling: true }, key(0))
 * net.apply(net.params, tensor([[0, 0, 0, 0]])).shape // => [1, 2]
 * ```
 */
export function qNetwork(inputSize: number, nActions: number, options: QNetworkOptions, key: Key): QNetwork {
  const hidden = options.hidden ?? [64, 64]
  const noisy = options.noisy ?? false
  const sigma0 = options.sigma0 ?? 0.5
  const atoms = options.atoms ?? 1
  const dueling = options.dueling ?? false
  const params: Record<string, Record<string, Tensor>> = {}
  let width = inputSize
  for (const [i, h] of hidden.entries()) {
    // With noisy layers every layer is noisy, as in Fortunato et al. (2018).
    params[`h${i}`] = qLayerInit(foldIn(key, i), width, h, noisy, sigma0)
    width = h
  }
  if (dueling) {
    params['value'] = qLayerInit(foldIn(key, 100), width, atoms, noisy, sigma0)
    params['advantage'] = qLayerInit(foldIn(key, 101), width, nActions * atoms, noisy, sigma0)
  } else params['out'] = qLayerInit(foldIn(key, 100), width, nActions * atoms, noisy, sigma0)
  const apply = (p: QParams, x: Value, noise?: Key): Value => {
    let h = x
    for (let i = 0; i < hidden.length; i++)
      h = relu(qLayer(p[`h${i}`] ?? {}, h, noise === undefined ? undefined : foldIn(noise, i)))
    const batch = x.shape[0] ?? 1
    const head = (name: string, _outputs: number, j: number): Value =>
      qLayer(p[name] ?? {}, h, noise === undefined ? undefined : foldIn(noise, 100 + j))
    let q: Value
    if (dueling) {
      const v = reshape(head('value', atoms, 0), [batch, 1, atoms])
      const a = reshape(head('advantage', nActions * atoms, 1), [batch, nActions, atoms])
      q = add(v, sub(a, mean(a, 1, true)))
    } else q = reshape(head('out', nActions * atoms, 0), [batch, nActions, atoms])
    return atoms === 1 ? reshape(q, [batch, nActions]) : q
  }
  return { params, nActions, atoms, apply }
}

// ----------------------------------------------------------------------------------------------------- DQN

/** Options of {@link dqn}. */
export interface DqnOptions extends QNetworkOptions {
  readonly learningRate?: number
  /** An optimiser replacing Adam with `learningRate`. */
  readonly optimizer?: GradientTransformation
  readonly gamma?: number
  readonly bufferSize?: number
  readonly batchSize?: number
  /** Environment steps before learning starts (default 100). */
  readonly learningStarts?: number
  /** Learn every this many steps (default 1). */
  readonly trainFrequency?: number
  /** Gradient steps per learning step (default 1). */
  readonly gradientSteps?: number
  /** Copy the online network to the target every this many steps (default 250); ignored when `tau < 1`. */
  readonly targetUpdateInterval?: number
  /** Soft target updates `target <- tau online + (1 - tau) target` every step (default 1: hard copies). */
  readonly tau?: number
  /** Epsilon schedule: linear from `epsilonStart` (1) to `epsilonEnd` (0.05) over `explorationSteps` (1000). */
  readonly epsilonStart?: number
  readonly epsilonEnd?: number
  readonly explorationSteps?: number
  /** Double DQN targets (van Hasselt et al. 2016): the online network chooses, the target network evaluates. */
  readonly double?: boolean
  /** Prioritised replay (Schaul et al. 2016) with exponents alpha (default 0.6) and beta annealed to 1. */
  readonly prioritized?: boolean | { readonly alpha?: number; readonly beta?: number; readonly betaSteps?: number }
  /** n-step returns (default 1). */
  readonly nStep?: number
  /** Huber loss threshold (default 1; stable-baselines3's smooth L1). */
  readonly huberDelta?: number
  /** Clip gradients to this global norm (default none). */
  readonly maxGradNorm?: number
  /** The initialisation key (default key(0)). */
  readonly key?: Key
}

/** The state of a DQN agent. */
export interface DqnState {
  readonly params: QParams
  readonly target: QParams
  readonly optState: unknown
  readonly buffer: ReplayBuffer<NStepTransition<readonly number[], number>>
  readonly prioritized?: PrioritizedReplayBuffer<NStepTransition<readonly number[], number>>
  readonly nstep: NStepBuffer<readonly number[], number>
  readonly steps: number
  readonly updates: number
  readonly loss: number
  readonly tdError: number
  readonly qValue: number
}

const stack = (rows: readonly (readonly number[])[], d: number): Tensor => rowsTensor(rows, d)

/**
 * Deep Q-networks (Mnih et al. 2015) with the standard improvements as options: double Q-learning, duelling heads,
 * prioritised replay, n-step returns and noisy networks (which replace epsilon-greedy exploration). The agent keeps
 * its replay buffer, networks and optimiser state in its state, learns every `trainFrequency` steps from minibatches
 * with a Huber loss, and syncs a target network by hard or soft updates. Reports epsilon, the loss, the mean absolute
 * TD error and the mean Q value.
 *
 * @param env - The environment's spaces (vector observations, discrete actions).
 * @param options - Network, optimiser, replay, exploration and the variants.
 * @returns The agent.
 * @concept deep-q-learning
 * @index agent Deep Q-network agent.
 * @example
 * ```ts
 * import { dqn } from 'mdml-engine/agents'
 * import { cartPole } from 'mdml-engine/environments'
 * dqn(cartPole.create(), { double: true, dueling: true }).name // => 'dqn'
 * ```
 */
export function dqn(env: Spaces, options: DqnOptions = {}): Agent<DqnState, readonly number[], number> {
  const d = observationSize(env.observationSpace)
  const nA = actionCount(env.actionSpace, 'dqn')
  const net = qNetwork(d, nA, options, options.key ?? makeKey(0))
  const gamma = options.gamma ?? 0.99
  const batchSize = options.batchSize ?? 32
  const learningStarts = options.learningStarts ?? 100
  const trainFrequency = options.trainFrequency ?? 1
  const gradientSteps = options.gradientSteps ?? 1
  const targetInterval = options.targetUpdateInterval ?? 250
  const tau = options.tau ?? 1
  const opt = optimiserOf(options)
  const epsilon = linearSchedule(
    options.epsilonStart ?? 1,
    options.epsilonEnd ?? 0.05,
    options.explorationSteps ?? 1000,
  )
  const noisy = options.noisy ?? false
  const per =
    options.prioritized === undefined || options.prioritized === false
      ? undefined
      : options.prioritized === true
        ? {}
        : options.prioritized
  const beta0 = per?.beta ?? 0.4
  const betaSteps = per?.betaSteps ?? 10_000
  const delta = options.huberDelta ?? 1
  const n = options.nStep ?? 1
  const qOf = (p: QParams, obs: readonly number[], noise?: Key): number[] =>
    toFlat(net.apply(p, stack([obs], d), noise))

  const update = (s: DqnState, k: Key): DqnState => {
    const beta = Math.min(1, beta0 + (1 - beta0) * (s.steps / betaSteps))
    const sample =
      s.prioritized !== undefined
        ? prioritizedSample(s.prioritized, foldIn(k, 0), batchSize, beta)
        : { ...bufferSample(s.buffer, foldIn(k, 0), batchSize), weights: undefined }
    const batch = sample.items
    const obs = stack(
      batch.map((t) => t.observation),
      d,
    )
    const next = stack(
      batch.map((t) => t.nextObservation),
      d,
    )
    const actions = vectorTensor(batch.map((t) => t.action))
    // Targets r + gamma^k (1 - done) Q_target(s', argmax), computed outside the gradient.
    const qNextTarget = net.apply(s.target, next, noisy ? foldIn(k, 1) : undefined)
    const nextBest =
      options.double === true
        ? argmax(net.apply(s.params, next, noisy ? foldIn(k, 2) : undefined), -1)
        : argmax(qNextTarget, -1)
    const nextQ = toFlat(sum(mul(qNextTarget, oneHot(nextBest, nA)), -1))
    const targets = batch.map((t, i) => t.reward + (t.terminated ? 0 : t.discount * (nextQ[i] ?? 0)))
    const y = vectorTensor(targets)
    const mask = oneHot(actions, nA)
    const weights = sample.weights === undefined ? undefined : vectorTensor(sample.weights)
    const noiseKey = noisy ? foldIn(k, 3) : undefined
    const lossFn = (p: Tree): Value => {
      const q = sum(mul(net.apply(p as QParams, obs, noiseKey), mask), -1)
      return huber(q, y, { delta, ...(weights === undefined ? {} : { weight: weights }) })
    }
    const step = gradientStep(lossFn, s.params, opt, s.optState)
    const q = toFlat(sum(mul(net.apply(s.params, obs, noiseKey), mask), -1))
    const errors = targets.map((t, i) => t - (q[i] ?? 0))
    let meanAbs = 0
    let meanQ = 0
    for (const [i, e] of errors.entries()) {
      meanAbs += Math.abs(e) / errors.length
      meanQ += (q[i] ?? 0) / errors.length
    }
    return {
      ...s,
      params: step.params as QParams,
      optState: step.optState,
      ...(s.prioritized === undefined ? {} : { prioritized: prioritizedUpdate(s.prioritized, sample.slots, errors) }),
      updates: s.updates + 1,
      loss: step.loss,
      tdError: meanAbs,
      qValue: meanQ,
    }
  }

  return {
    name: 'dqn',
    config: {
      hidden: [...(options.hidden ?? [64, 64])],
      gamma,
      batchSize,
      learningStarts,
      targetUpdateInterval: targetInterval,
      tau,
      double: options.double ?? false,
      dueling: options.dueling ?? false,
      prioritized: per !== undefined,
      nStep: n,
      noisy,
    } as Json,
    metrics: {
      epsilon: { description: 'exploration rate' },
      loss: { description: 'Huber TD loss of the last update', direction: 'lower' },
      tdError: { description: 'mean absolute TD error of the last batch', direction: 'lower' },
      qValue: { description: 'mean Q value of the last batch' },
    },
    init: () => ({
      params: net.params,
      target: net.params,
      optState: opt.init(net.params),
      buffer: replayBuffer(options.bufferSize ?? 10_000),
      ...(per === undefined
        ? {}
        : {
            prioritized: prioritizedReplay(
              options.bufferSize ?? 10_000,
              per.alpha === undefined ? {} : { alpha: per.alpha },
            ),
          }),
      nstep: nStepBuffer(n, gamma),
      steps: 0,
      updates: 0,
      loss: 0,
      tdError: 0,
      qValue: 0,
    }),
    act: (s, obs, k) => {
      if (noisy) return argmaxOf(qOf(s.params, obs, foldIn(k, 1)))
      const eps = epsilon(s.steps)
      // Explore without a forward pass when the coin says so.
      if ((uniform(foldIn(k, 0), []).data[0] ?? 1) < eps)
        return Math.min(nA - 1, Math.floor((uniform(foldIn(k, 2), []).data[0] ?? 0) * nA))
      return argmaxOf(qOf(s.params, obs))
    },
    greedy: (s, obs) => argmaxOf(qOf(s.params, obs)),
    learn: (s, t: Transition<readonly number[], number>, k) => {
      const acc = nStepAdd(s.nstep, t)
      let buffer = s.buffer
      let prioritized = s.prioritized
      for (const ready of acc.ready) {
        if (prioritized !== undefined) prioritized = prioritizedAdd(prioritized, ready)
        else buffer = bufferAdd(buffer, ready)
      }
      let next: DqnState = {
        ...s,
        nstep: acc.buffer,
        buffer,
        ...(prioritized === undefined ? {} : { prioritized }),
        steps: s.steps + 1,
      }
      const size = prioritized?.buffer.size ?? buffer.size
      if (next.steps >= learningStarts && next.steps % trainFrequency === 0 && size >= batchSize) {
        for (let g = 0; g < gradientSteps; g++) next = update(next, foldIn(k, g))
      }
      if (tau < 1) next = { ...next, target: incrementalUpdate(next.params, next.target, tau) as QParams }
      else if (next.steps % targetInterval === 0) next = { ...next, target: next.params }
      return next
    },
    stats: (s) => ({ epsilon: noisy ? 0 : epsilon(s.steps), loss: s.loss, tdError: s.tdError, qValue: s.qValue }),
  }
}

/**
 * The greedy action values of a DQN-family agent state for an observation (for inspection and figures).
 *
 * @param env - The environment's spaces.
 * @param state - The agent state.
 * @param observation - The observation.
 * @param options - The network options the agent was built with.
 * @returns The action values.
 * @concept deep-q-learning
 * @example
 * ```ts
 * import { dqn, qValues } from 'mdml-engine/agents'
 * import { cartPole } from 'mdml-engine/environments'
 * import { key } from 'mdml-engine/random'
 * const env = cartPole.create()
 * qValues(env, dqn(env, { hidden: [8] }).init(key(0)), [0, 0, 0, 0], { hidden: [8] }).length // => 2
 * ```
 */
export function qValues(
  env: Spaces,
  state: DqnState,
  observation: readonly number[],
  options: QNetworkOptions = {},
): number[] {
  const d = observationSize(env.observationSpace)
  const nA = actionCount(env.actionSpace, 'qValues')
  const net = qNetwork(d, nA, options, makeKey(0))
  if (net.atoms !== 1) throw new MdmlError('value', 'qValues: distributional networks need their own readout')
  return toFlat(net.apply(state.params, rowsTensor([observation], d)))
}
