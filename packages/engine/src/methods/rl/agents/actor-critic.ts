import { argmaxOf, type Json } from '../../../lib/base/index.ts'
import type { ActivationName, Params } from '../../../lib/nn/index.ts'
import { adam, incrementalUpdate, type Tree } from '../../../lib/optim/index.ts'
import { foldIn, type Key, key as makeKey, normal, uniform } from '../../../lib/random/index.ts'
import {
  type Agent,
  bufferAdd,
  bufferSample,
  categoricalPolicy,
  gaussianNoise,
  gaussianPolicy,
  type NoiseProcess,
  ornsteinUhlenbeckNoise,
  type ReplayBuffer,
  replayBuffer,
  type Transition,
} from '../../../lib/rl/index.ts'
import {
  add,
  clip,
  concat,
  exp,
  fromStorage,
  item,
  logSoftmax,
  mean,
  minimum,
  mul,
  oneHot,
  primalOf,
  sliceAxis,
  softmax,
  square,
  sub,
  sum,
  type Tensor,
  tanh,
  toFlat,
  type Value,
} from '../../../lib/tensor/index.ts'
import type { MetricChannel } from '../../../lib/trace/index.ts'
import {
  actionBox,
  actionCount,
  gradientStep,
  observationSize,
  rowsTensor,
  type Spaces,
  vectorTensor,
} from './_common.ts'
import { mlpNet } from './_networks.ts'

// ------------------------------------------------------------------------------------------------ shared plumbing

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

type Stored = Transition<readonly number[], number | readonly number[]>

const METRICS: Readonly<Record<string, MetricChannel>> = {
  actorLoss: { description: 'actor loss of the last update', direction: 'lower' },
  criticLoss: { description: 'critic (TD) loss of the last update', direction: 'lower' },
  qValue: { description: 'mean Q value of the last batch' },
}

const SAC_METRICS: Readonly<Record<string, MetricChannel>> = {
  ...METRICS,
  alpha: { description: 'the entropy temperature' },
  entropy: { description: 'policy entropy estimate on the last batch' },
}

interface Batch {
  readonly obs: Tensor
  readonly next: Tensor
  readonly rewards: Tensor
  readonly notDone: Tensor
  readonly items: readonly Stored[]
}

function sampleBatch(s: ActorCriticAgentState, d: number, batchSize: number, k: Key): Batch {
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

interface Machinery {
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

function offPolicyAgent<A>(
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
function boxNetworks(env: Spaces, options: OffPolicyOptions, name: string, actorOutputs: (k: number) => number) {
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

const mse = (a: Value, b: Value): Value => mean(square(sub(a, b)))
const scalar = (v: Value): number => item(primalOf(v))

function common(options: OffPolicyOptions, lr: number) {
  return {
    gamma: options.gamma ?? 0.99,
    tau: options.tau ?? 0.005,
    batchSize: options.batchSize ?? 64,
    actorOpt: adam({ learningRate: options.actorLearningRate ?? options.learningRate ?? lr }),
    criticOpt: adam({ learningRate: options.criticLearningRate ?? options.learningRate ?? lr }),
    bufferSize: options.bufferSize ?? 10_000,
  }
}

function baseConfig(options: OffPolicyOptions, c: ReturnType<typeof common>): Record<string, Json> {
  return {
    hidden: [...(options.hidden ?? [64, 64])],
    gamma: c.gamma,
    tau: c.tau,
    batchSize: c.batchSize,
    bufferSize: c.bufferSize,
    learningStarts: options.learningStarts ?? 100,
  }
}

// ------------------------------------------------------------------------------------------------- DDPG and TD3

/** Options of {@link ddpg}. */
export interface DdpgOptions extends OffPolicyOptions {
  /** Exploration noise: Gaussian (default), Ornstein-Uhlenbeck, or any process (in normalised action units). */
  readonly noise?: 'gaussian' | 'ornstein-uhlenbeck' | NoiseProcess
  /** The noise scale (default 0.1). */
  readonly noiseSigma?: number
}

/** Options of {@link td3}. */
export interface Td3Options extends DdpgOptions {
  /** Update the actor and the targets every this many critic updates (default 2). */
  readonly policyDelay?: number
  /** Standard deviation of the target policy smoothing noise (default 0.2). */
  readonly targetPolicyNoise?: number
  /** Clip of the target policy smoothing noise (default 0.5). */
  readonly targetNoiseClip?: number
}

function deterministic(env: Spaces, options: Td3Options, twin: boolean, name: string) {
  const net = boxNetworks(env, options, name, (k) => k)
  const c = common(options, 1e-3)
  const sigma = options.noiseSigma ?? 0.1
  const noise: NoiseProcess =
    typeof options.noise === 'object'
      ? options.noise
      : options.noise === 'ornstein-uhlenbeck'
        ? ornsteinUhlenbeckNoise(net.k, { sigma })
        : gaussianNoise(net.k, { sigma })
  const delay = twin ? (options.policyDelay ?? 2) : 1
  const smoothing = twin ? (options.targetPolicyNoise ?? 0.2) : 0
  const noiseClip = options.targetNoiseClip ?? 0.5
  const mu = (actor: Params, obs: Value): Value => tanh(net.actor.apply(actor, obs))
  const muOf = (actor: Params, obs: readonly number[]): number[] => toFlat(mu(actor, rowsTensor([obs], net.d)))
  const critic: Params = twin ? { q1: net.q1.params, q2: net.q2.params } : { q1: net.q1.params }

  const update = (s: ActorCriticAgentState, k: Key): ActorCriticAgentState => {
    const b = sampleBatch(s, net.d, c.batchSize, foldIn(k, 0))
    const u = net.actions(b.items)
    // Targets r + gamma (1 - terminated) Q'(s', mu'(s') [+ clipped noise]), with twin critics the smaller of the two.
    let nextU = mu(s.actorTarget, b.next)
    if (smoothing > 0) {
      const eps = clip(mul(normal(foldIn(k, 1), [...nextU.shape]), smoothing), -noiseClip, noiseClip)
      nextU = clip(add(nextU, eps), -1, 1)
    }
    const qNext = twin
      ? minimum(net.q('q1', s.criticTarget, b.next, nextU), net.q('q2', s.criticTarget, b.next, nextU))
      : net.q('q1', s.criticTarget, b.next, nextU)
    const y = primalOf(add(b.rewards, mul(mul(b.notDone, c.gamma), qNext)))
    let qMean = 0
    const criticLoss = (p: Tree): Value => {
      const q1 = net.q('q1', p as Params, b.obs, u)
      qMean = scalar(mean(q1))
      const l1 = mse(q1, y)
      return twin ? add(l1, mse(net.q('q2', p as Params, b.obs, u), y)) : l1
    }
    const cs = gradientStep(criticLoss, s.critic, c.criticOpt, s.criticOptState)
    let next: ActorCriticAgentState = {
      ...s,
      critic: cs.params as Params,
      criticOptState: cs.optState,
      stats: { ...s.stats, criticLoss: cs.loss, qValue: qMean },
    }
    if ((s.updates + 1) % delay !== 0) return next
    const criticNow = next.critic
    const actorLoss = (p: Tree): Value => mul(-1, mean(net.q('q1', criticNow, b.obs, mu(p as Params, b.obs))))
    const as = gradientStep(actorLoss, s.actor, c.actorOpt, s.actorOptState)
    next = {
      ...next,
      actor: as.params as Params,
      actorOptState: as.optState,
      actorTarget: incrementalUpdate(as.params, s.actorTarget, c.tau) as Params,
      criticTarget: incrementalUpdate(next.critic, s.criticTarget, c.tau) as Params,
      stats: { ...next.stats, actorLoss: as.loss },
    }
    return next
  }

  const machinery: Machinery = {
    name,
    config: {
      ...baseConfig(options, c),
      noise: typeof options.noise === 'object' ? 'custom' : (options.noise ?? 'gaussian'),
      noiseSigma: sigma,
      ...(twin ? { policyDelay: delay, targetPolicyNoise: smoothing, targetNoiseClip: noiseClip } : {}),
    },
    metrics: METRICS,
    init: () => ({
      actor: net.actor.params,
      critic,
      actorTarget: net.actor.params,
      criticTarget: critic,
      actorOptState: c.actorOpt.init(net.actor.params),
      criticOptState: c.criticOpt.init(critic),
      logAlpha: 0,
      alphaOptState: null,
      buffer: replayBuffer(c.bufferSize),
      noise: noise.init(),
      steps: 0,
      updates: 0,
      stats: { actorLoss: 0, criticLoss: 0, qValue: 0 },
    }),
    act: (s, obs) => net.toEnv(muOf(s.actor, obs).map((x, i) => x + (s.noise[i] ?? 0))),
    random: net.random,
    greedy: (s, obs) => net.toEnv(muOf(s.actor, obs)),
    update,
    noise,
  }
  return machinery
}

/**
 * Deep deterministic policy gradient (Lillicrap et al. 2016): a deterministic tanh-bounded actor `mu(s)` and a Q
 * critic, both with Polyak-averaged target networks (`incrementalUpdate`, tau 0.005), learning from uniform replay;
 * critic targets `r + gamma (1 - terminated) Q'(s', mu'(s'))`, the actor ascends `Q(s, mu(s))`. Exploration adds
 * Gaussian or Ornstein-Uhlenbeck noise (lib/rl/policies.ts) to the actor's action; the noise process advances in
 * `learn` and restarts with every episode. Reports the actor and critic losses and the mean Q value.
 *
 * @param env - The environment's spaces (vector observations, box actions).
 * @param options - Networks, optimisers, replay and exploration noise.
 * @returns The agent.
 * @concept actor-critic
 * @reference Lillicrap et al. (2016), Continuous control with deep reinforcement learning.
 * @example
 * ```ts
 * import { ddpg } from 'whitebox-engine/methods/rl/agents'
 * import { pendulum } from 'whitebox-engine/methods/rl'
 * ddpg(pendulum.create(), { noise: 'ornstein-uhlenbeck' }).name // => 'ddpg'
 * ```
 */
export function ddpg<A extends readonly number[] = readonly number[]>(
  env: Spaces,
  options: DdpgOptions = {},
): Agent<ActorCriticAgentState, readonly number[], A> {
  return offPolicyAgent<A>(deterministic(env, options, false, 'ddpg'), options)
}

/**
 * Twin delayed DDPG (TD3, Fujimoto et al. 2018): DDPG with twin critics and clipped double-Q targets
 * `r + gamma (1 - terminated) min(Q1', Q2')(s', a')`, target policy smoothing `a' = clip(mu'(s') + clip(N(0, 0.2),
 * -0.5, 0.5), -1, 1)`, and delayed policy updates (the actor and every target network update once per `policyDelay`
 * critic updates).
 *
 * @param env - The environment's spaces (vector observations, box actions).
 * @param options - DDPG's options plus the delay and the smoothing noise.
 * @returns The agent.
 * @concept actor-critic
 * @reference Fujimoto, van Hoof and Meger (2018), Addressing function approximation error in actor-critic methods.
 * @example
 * ```ts
 * import { td3 } from 'whitebox-engine/methods/rl/agents'
 * import { pendulum } from 'whitebox-engine/methods/rl'
 * td3(pendulum.create(), { policyDelay: 3 }).name // => 'td3'
 * ```
 */
export function td3<A extends readonly number[] = readonly number[]>(
  env: Spaces,
  options: Td3Options = {},
): Agent<ActorCriticAgentState, readonly number[], A> {
  return offPolicyAgent<A>(deterministic(env, options, true, 'td3'), options)
}

// ------------------------------------------------------------------------------------------------------------ SAC

/** Options of {@link sac} and {@link sacDiscrete}. */
export interface SacOptions extends OffPolicyOptions {
  /** A fixed entropy temperature; by default it is tuned automatically to the target entropy. */
  readonly alpha?: number
  /** The initial temperature when tuning (default 1). */
  readonly alphaInit?: number
  /** The learning rate of the temperature (default `learningRate`, else 3e-4). */
  readonly alphaLearningRate?: number
  /** The target entropy (default `-dim A` for continuous actions, `-0.98 log(1 / |A|)` for discrete ones). */
  readonly targetEntropy?: number
  /** For discrete SAC, the fraction of the maximum entropy `log |A|` to target (default 0.98). */
  readonly targetEntropyScale?: number
}

function temperature(options: SacOptions) {
  const tune = options.alpha === undefined
  const alphaOpt = adam({ learningRate: options.alphaLearningRate ?? options.learningRate ?? 3e-4 })
  const logAlpha0 = Math.log(options.alpha ?? options.alphaInit ?? 1)
  // The temperature loss -log(alpha) g with g = E[log pi] + target (stable-baselines3's ent_coef_loss), minimised over
  // log(alpha): the temperature rises while the entropy -E[log pi] is below the target.
  const step = (s: ActorCriticAgentState, g: number): { logAlpha: number; alphaOptState: unknown } => {
    if (!tune) return { logAlpha: s.logAlpha, alphaOptState: s.alphaOptState }
    const t = gradientStep(
      (p: Tree) => mul(-g, (p as { logAlpha: Value }).logAlpha),
      { logAlpha: fromStorage(Float64Array.of(s.logAlpha), [], 'float64') },
      alphaOpt,
      s.alphaOptState,
    )
    return { logAlpha: item((t.params as { logAlpha: Value }).logAlpha), alphaOptState: t.optState }
  }
  const init = (): unknown => alphaOpt.init({ logAlpha: fromStorage(Float64Array.of(logAlpha0), [], 'float64') })
  return { tune, logAlpha0, step, init }
}

const LOG_STD_MIN = -20
const LOG_STD_MAX = 2

/**
 * Soft actor-critic (Haarnoja et al. 2018) for continuous actions: a squashed Gaussian actor (`gaussianPolicy` with
 * tanh squashing to the action bounds and state-dependent log std), twin soft Q critics with Polyak-averaged targets
 * `r + gamma (1 - terminated) (min Q'(s', a') - alpha log pi(a' | s'))`, the reparameterised actor loss
 * `mean(alpha log pi(a~ | s) - min Q(s, a~))`, and automatic tuning of the temperature to a target entropy (`-dim A`
 * in normalised action units, as stable-baselines3) by gradient descent on `-log(alpha) (log pi + target)`. Reports
 * the losses, Q values, alpha and the entropy estimate `-mean log pi`.
 *
 * @param env - The environment's spaces (vector observations, box actions).
 * @param options - Networks, optimisers, replay and the temperature.
 * @returns The agent.
 * @concept actor-critic
 * @reference Haarnoja, Zhou, Abbeel and Levine (2018), Soft actor-critic; Haarnoja et al. (2018), Soft actor-critic
 *   algorithms and applications.
 * @example
 * ```ts
 * import { sac } from 'whitebox-engine/methods/rl/agents'
 * import { pendulum } from 'whitebox-engine/methods/rl'
 * sac(pendulum.create(), { alpha: 0.2 }).name // => 'sac'
 * ```
 */
export function sac<A extends readonly number[] = readonly number[]>(
  env: Spaces,
  options: SacOptions = {},
): Agent<ActorCriticAgentState, readonly number[], A> {
  const net = boxNetworks(env, options, 'sac', (k) => 2 * k)
  const c = common(options, 3e-4)
  const temp = temperature(options)
  // The log density of bounded actions is that of [-1, 1] actions minus sum log(scale): shift the target to match.
  let logScale = 0
  for (const s of net.scale) logScale += Math.log(s)
  const targetEntropy = options.targetEntropy ?? -net.k
  const target = targetEntropy + logScale
  const critic: Params = { q1: net.q1.params, q2: net.q2.params }
  const bounds = { squash: true, low: [...net.box.low], high: [...net.box.high] }
  const head = (actor: Params, obs: Value) => {
    const out = net.actor.apply(actor, obs)
    const ls = clip(sliceAxis(out, 1, net.k, 2 * net.k), LOG_STD_MIN, LOG_STD_MAX)
    return gaussianPolicy(sliceAxis(out, 1, 0, net.k), ls, bounds)
  }
  const normaliseT = (a: Value): Value =>
    mul(
      sub(a, net.centre),
      net.scale.map((x) => 1 / x),
    )
  const minQ = (p: Params, obs: Value, u: Value): Value => minimum(net.q('q1', p, obs, u), net.q('q2', p, obs, u))

  const update = (s: ActorCriticAgentState, k: Key): ActorCriticAgentState => {
    const b = sampleBatch(s, net.d, c.batchSize, foldIn(k, 0))
    const u = net.actions(b.items)
    const alpha = Math.exp(s.logAlpha)
    // Temperature first, from a fresh sample of the current actor (stable-baselines3's order).
    const fresh = head(s.actor, b.obs).sampleWithLogProb(foldIn(k, 1), false)
    const meanLogPi = item(mean(fresh.logProb))
    const t = temp.step(s, meanLogPi + target)
    const alphaNow = temp.tune ? Math.exp(t.logAlpha) : alpha
    const nextSample = head(s.actor, b.next).sampleWithLogProb(foldIn(k, 2), false)
    const qNext = sub(minQ(s.criticTarget, b.next, normaliseT(nextSample.action)), mul(alphaNow, nextSample.logProb))
    const y = primalOf(add(b.rewards, mul(mul(b.notDone, c.gamma), qNext)))
    let qMean = 0
    const criticLoss = (p: Tree): Value => {
      const q1 = net.q('q1', p as Params, b.obs, u)
      qMean = scalar(mean(q1))
      return add(mse(q1, y), mse(net.q('q2', p as Params, b.obs, u), y))
    }
    const cs = gradientStep(criticLoss, s.critic, c.criticOpt, s.criticOptState)
    const criticNow = cs.params as Params
    const actorKey = foldIn(k, 3)
    let entropy = 0
    const actorLoss = (p: Tree): Value => {
      const { action, logProb } = head(p as Params, b.obs).sampleWithLogProb(actorKey, true)
      entropy = -scalar(mean(logProb))
      return mean(sub(mul(alphaNow, logProb), minQ(criticNow, b.obs, normaliseT(action))))
    }
    const as = gradientStep(actorLoss, s.actor, c.actorOpt, s.actorOptState)
    return {
      ...s,
      actor: as.params as Params,
      critic: criticNow,
      criticTarget: incrementalUpdate(criticNow, s.criticTarget, c.tau) as Params,
      actorOptState: as.optState,
      criticOptState: cs.optState,
      logAlpha: t.logAlpha,
      alphaOptState: t.alphaOptState,
      stats: { actorLoss: as.loss, criticLoss: cs.loss, qValue: qMean, alpha: alphaNow, entropy: entropy - logScale },
    }
  }
  const one = (obs: readonly number[]): Tensor => rowsTensor([obs], net.d)
  const machinery: Machinery = {
    name: 'sac',
    config: { ...baseConfig(options, c), targetEntropy, autoAlpha: temp.tune },
    metrics: SAC_METRICS,
    init: () => ({
      actor: net.actor.params,
      critic,
      actorTarget: net.actor.params,
      criticTarget: critic,
      actorOptState: c.actorOpt.init(net.actor.params),
      criticOptState: c.criticOpt.init(critic),
      logAlpha: temp.logAlpha0,
      alphaOptState: temp.init(),
      buffer: replayBuffer(c.bufferSize),
      noise: [],
      steps: 0,
      updates: 0,
      stats: { actorLoss: 0, criticLoss: 0, qValue: 0, alpha: Math.exp(temp.logAlpha0), entropy: 0 },
    }),
    act: (s, obs, k) => toFlat(head(s.actor, one(obs)).sample(k)),
    random: net.random,
    greedy: (s, obs) => toFlat(head(s.actor, one(obs)).mode()),
    update,
  }
  return offPolicyAgent<A>(machinery, options)
}

/**
 * Soft actor-critic for discrete actions (Christodoulou 2019): a categorical actor and twin critics that output a Q
 * value per action, so expectations over actions replace sampling: targets `r + gamma (1 - terminated)
 * sum_a' pi(a' | s') (min Q'(s', a') - alpha log pi(a' | s'))`, the policy loss
 * `mean sum_a pi(a | s) (alpha log pi(a | s) - min Q(s, a))`, and temperature tuning towards the target entropy
 * `-0.98 log(1 / |A|)` (or `targetEntropy`). Critic targets are Polyak-averaged.
 *
 * @param env - The environment's spaces (vector observations, discrete actions).
 * @param options - Networks, optimisers, replay and the temperature.
 * @returns The agent.
 * @concept actor-critic
 * @reference Christodoulou (2019), Soft actor-critic for discrete action settings.
 * @example
 * ```ts
 * import { sacDiscrete } from 'whitebox-engine/methods/rl/agents'
 * import { cartPole } from 'whitebox-engine/methods/rl'
 * sacDiscrete(cartPole.create(), { targetEntropyScale: 0.5 }).name // => 'sacDiscrete'
 * ```
 */
export function sacDiscrete(
  env: Spaces,
  options: SacOptions = {},
): Agent<ActorCriticAgentState, readonly number[], number> {
  const d = observationSize(env.observationSpace)
  const nA = actionCount(env.actionSpace, 'sacDiscrete')
  const hidden = options.hidden ?? [64, 64]
  const activation = options.activation ?? 'relu'
  const key = options.key ?? makeKey(0)
  const actor = mlpNet([d, ...hidden, nA], activation, foldIn(key, 0))
  const q1 = mlpNet([d, ...hidden, nA], activation, foldIn(key, 1))
  const q2 = mlpNet([d, ...hidden, nA], activation, foldIn(key, 2))
  const c = common(options, 3e-4)
  const temp = temperature(options)
  const targetEntropy = options.targetEntropy ?? -(options.targetEntropyScale ?? 0.98) * Math.log(1 / nA)
  const critic: Params = { q1: q1.params, q2: q2.params }
  const minQ = (p: Params, obs: Value): Value =>
    minimum(q1.apply(p['q1'] as Params, obs), q2.apply(p['q2'] as Params, obs))

  const update = (s: ActorCriticAgentState, k: Key): ActorCriticAgentState => {
    const b = sampleBatch(s, d, c.batchSize, foldIn(k, 0))
    const mask = oneHot(vectorTensor(b.items.map((t) => t.action as number)), nA)
    const logits = actor.apply(s.actor, b.obs)
    const logPi = logSoftmax(logits, -1)
    // Temperature, with the exact expectation E_pi[log pi] = -H over the actions.
    const negEntropy = item(mean(sum(mul(softmax(logits, -1), logPi), -1)))
    const t = temp.step(s, negEntropy + targetEntropy)
    const alpha = temp.tune ? Math.exp(t.logAlpha) : Math.exp(s.logAlpha)
    const nextLogits = actor.apply(s.actor, b.next)
    const nextLogPi = logSoftmax(nextLogits, -1)
    const vNext = sum(mul(exp(nextLogPi), sub(minQ(s.criticTarget, b.next), mul(alpha, nextLogPi))), -1)
    const y = primalOf(add(b.rewards, mul(mul(b.notDone, c.gamma), vNext)))
    let qMean = 0
    const criticLoss = (p: Tree): Value => {
      const q = p as Params
      const qa1 = sum(mul(q1.apply(q['q1'] as Params, b.obs), mask), -1)
      const qa2 = sum(mul(q2.apply(q['q2'] as Params, b.obs), mask), -1)
      qMean = scalar(mean(qa1))
      return add(mse(qa1, y), mse(qa2, y))
    }
    const cs = gradientStep(criticLoss, s.critic, c.criticOpt, s.criticOptState)
    const qNow = primalOf(minQ(cs.params as Params, b.obs))
    let entropy = 0
    const actorLoss = (p: Tree): Value => {
      const lp = logSoftmax(actor.apply(p as Params, b.obs), -1)
      const pi = exp(lp)
      entropy = -scalar(mean(sum(mul(pi, lp), -1)))
      return mean(sum(mul(pi, sub(mul(alpha, lp), qNow)), -1))
    }
    const as = gradientStep(actorLoss, s.actor, c.actorOpt, s.actorOptState)
    return {
      ...s,
      actor: as.params as Params,
      critic: cs.params as Params,
      criticTarget: incrementalUpdate(cs.params, s.criticTarget, c.tau) as Params,
      actorOptState: as.optState,
      criticOptState: cs.optState,
      logAlpha: t.logAlpha,
      alphaOptState: t.alphaOptState,
      stats: { actorLoss: as.loss, criticLoss: cs.loss, qValue: qMean, alpha, entropy },
    }
  }
  const one = (obs: readonly number[]): Tensor => rowsTensor([obs], d)
  const machinery: Machinery = {
    name: 'sacDiscrete',
    config: { ...baseConfig(options, c), targetEntropy, autoAlpha: temp.tune },
    metrics: SAC_METRICS,
    init: () => ({
      actor: actor.params,
      critic,
      actorTarget: actor.params,
      criticTarget: critic,
      actorOptState: c.actorOpt.init(actor.params),
      criticOptState: c.criticOpt.init(critic),
      logAlpha: temp.logAlpha0,
      alphaOptState: temp.init(),
      buffer: replayBuffer(c.bufferSize),
      noise: [],
      steps: 0,
      updates: 0,
      stats: { actorLoss: 0, criticLoss: 0, qValue: 0, alpha: Math.exp(temp.logAlpha0), entropy: 0 },
    }),
    act: (s, obs, k) => item(categoricalPolicy(actor.apply(s.actor, one(obs))).sample(k)),
    random: (k) => Math.min(nA - 1, Math.floor((uniform(k, []).data[0] ?? 0) * nA)),
    greedy: (s, obs) => argmaxOf(toFlat(actor.apply(s.actor, one(obs)))),
    update,
  }
  return offPolicyAgent<number>(machinery, options)
}
