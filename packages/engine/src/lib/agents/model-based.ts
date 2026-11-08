/**
 * Model-based deep RL agents, which learn a model of the environment and plan or train inside it: PETS (an ensemble of
 * probabilistic networks with cross-entropy-method planning over sampled trajectories), MBPO (short rollouts of a
 * learnt ensemble feeding any off-policy learner), a small Dreamer (a latent world model whose actor and critic learn
 * purely in imagination) and MuZero-style tree search over a learnt hidden-state model.
 */

import { argmaxOf, type Json, meanOf, minOf } from '../base/index.ts'
import { adam, type ParamTree } from '../optim/index.ts'
import { foldIn, type Key, key as makeKey, uniform } from '../random/index.ts'
import {
  type Agent,
  bufferAdd,
  bufferItems,
  bufferSample,
  cem,
  type Ensemble,
  type EpisodeStore,
  ensembleSample,
  episodeAdd,
  episodeStore,
  modelEnv,
  modelTargets,
  type ReplayBuffer,
  replayBuffer,
  sampleSequences,
  type Transition,
} from '../rl/index.ts'
import type { Space } from '../space/index.ts'
import { add, argmax, concat, mean, mul, square, squeeze, sub, sum, toFlat, type Value } from '../tensor/index.ts'
import {
  actionBox,
  actionCount,
  gradientStep,
  observationSize,
  rowsTensor,
  type Spaces,
  vectorTensor,
} from './_common.ts'
import {
  actorHead,
  type DreamerParams,
  type DreamerSizes,
  dreamerNets,
  dreamerParams,
  imagine,
  posterior,
  type SequenceBatch,
  transition,
  worldPass,
} from './_dreamer.ts'
import { type Member, type ProbabilisticNet, particleReturns, probabilisticNet, trainEnsemble } from './_ensemble.ts'
import {
  type MuZeroEpisode,
  type MuZeroParams,
  type MuZeroStep,
  muZeroNets,
  muZeroParams,
  predict,
  represent,
  type SearchResult,
  searchFrom,
  type UnrollTargets,
  unrollLoss,
  unrollTargets,
} from './_muzero.ts'

// ------------------------------------------------------------------------------------------- learned ensembles

/** Options of the probabilistic ensemble that PETS and MBPO learn. */
export interface EnsembleOptions {
  /** Hidden widths of each member (default `[64, 64]`, SiLU activations). */
  readonly hidden?: readonly number[]
  /** Ensemble size (default 5). */
  readonly members?: number
  readonly learningRate?: number
  /** Minibatch size and Adam steps per member per training round (default 64 and 100). */
  readonly batchSize?: number
  readonly gradientSteps?: number
  /** Retrain every this many environment steps (default 200). */
  readonly trainEvery?: number
  readonly bufferSize?: number
  /** The initialisation key (default key(0)). */
  readonly key?: Key
}

/** The model-learning part of a model-based agent's state. */
export interface ModelBasedState {
  readonly members: readonly Member[]
  readonly optStates: readonly unknown[]
  readonly buffer: ReplayBuffer<Transition<readonly number[], readonly number[] | number>>
  readonly steps: number
  /** Training rounds so far (the model is used once there is at least one). */
  readonly trainings: number
  readonly modelLoss: number
}

/** How actions enter the model: box actions as they are, discrete ones one-hot. */
function actionEncoding(space: Space): { readonly size: number; encode(a: readonly number[] | number): number[] } {
  if (space.kind === 'discrete')
    return {
      size: space.n,
      encode: (a) => {
        const i = typeof a === 'number' ? a : (a[0] ?? 0)
        return Array.from({ length: space.n }, (_, j) => (j === i - (space.start ?? 0) ? 1 : 0))
      },
    }
  const size = actionBox(space, 'model').size
  return { size, encode: (a) => (typeof a === 'number' ? [a] : [...a]) }
}

interface Learner {
  readonly net: ProbabilisticNet
  readonly encode: (a: readonly number[] | number) => number[]
  init(): Pick<ModelBasedState, 'members' | 'optStates' | 'buffer' | 'steps' | 'trainings' | 'modelLoss'>
  record(s: ModelBasedState, t: Transition<readonly number[], readonly number[] | number>, k: Key): ModelBasedState
  ensemble(s: ModelBasedState): Ensemble<Member>
}

function ensembleLearner(env: Spaces, options: EnsembleOptions, warmup: number): Learner {
  const d = observationSize(env.observationSpace)
  const { size, encode } = actionEncoding(env.actionSpace)
  const net = probabilisticNet(d, size, options.hidden ?? [64, 64], options.key ?? makeKey(0))
  const n = options.members ?? 5
  const opt = adam({ learningRate: options.learningRate ?? 1e-3 })
  const trainEvery = options.trainEvery ?? 200
  const k0 = options.key ?? makeKey(0)
  return {
    net,
    encode,
    init: () => {
      const members = Array.from({ length: n }, (_, i) => net.init(foldIn(k0, 1000 + i)))
      return {
        members,
        optStates: members.map((m) => opt.init(m.params)),
        buffer: replayBuffer(options.bufferSize ?? 100_000),
        steps: 0,
        trainings: 0,
        modelLoss: 0,
      }
    },
    record: (s, t, k) => {
      const next = { ...s, buffer: bufferAdd(s.buffer, t), steps: s.steps + 1 }
      if (next.steps < warmup || next.steps % trainEvery !== 0) return next
      const items = bufferItems(next.buffer)
      const data = modelTargets(items)
      const inputs = data.observations.map((o, i) => [...o, ...encode(items[i]?.action ?? 0)])
      const r = trainEnsemble(
        net,
        s.members,
        s.optStates,
        { inputs, targets: data.targets },
        {
          optimizer: opt,
          gradientSteps: options.gradientSteps ?? 100,
          batchSize: options.batchSize ?? 64,
        },
        k,
      )
      return { ...next, members: r.members, optStates: r.optStates, trainings: s.trainings + 1, modelLoss: r.loss }
    },
    ensemble: (s) => ({ model: net.model, members: s.members }),
  }
}

const sameObservation = (a: readonly number[] | undefined, b: readonly number[]): boolean =>
  a !== undefined && a.length === b.length && a.every((v, i) => v === b[i])

// --------------------------------------------------------------------------------------------------------- PETS

/** Options of {@link pets}. */
export interface PetsOptions extends EnsembleOptions {
  /** Random-action steps before the first model is trained (default 200). */
  readonly warmup?: number
  /** CEM over action sequences: horizon, candidates, elites, iterations and initial spread. */
  readonly horizon?: number
  readonly population?: number
  readonly elites?: number
  readonly iterations?: number
  readonly sigma?: number
  /** TS-1 particles per candidate sequence (default 4). */
  readonly particles?: number
  readonly gamma?: number
  /** A known reward function, replacing the learned reward (as in the PETS experiments). */
  readonly reward?: (observation: readonly number[], action: readonly number[], next: readonly number[]) => number
  /** A known termination function. */
  readonly terminated?: (observation: readonly number[]) => boolean
}

/** The state of a PETS agent: the ensemble, its data, and the current plan. */
export interface PetsState extends ModelBasedState {
  readonly plan?: readonly (readonly number[])[]
  readonly plannedFor?: readonly number[]
  readonly planValue: number
}

/**
 * Probabilistic ensembles with trajectory sampling (PETS; Chua et al. 2018): an ensemble of probabilistic MLPs
 * (Gaussian outputs with learned log-variance bounds, input and target normalisation) is fitted by Gaussian negative
 * log-likelihood, each member on its own bootstrap resample of the data, and actions are planned by the cross-entropy
 * method (lib/rl `cem`) over action sequences scored with TS-1 particles (each particle steps with a member drawn
 * afresh at every step, lib/rl `ensembleSample`), every candidate and particle propagated together in one batch. The
 * first action of the plan is applied and the shifted plan warm-starts the next decision; plans are made in `learn`
 * for the next observation (run the agent with `trainAgent`). Before the first training it acts at random.
 *
 * @param env - The environment's spaces (vector observations, box actions).
 * @param options - Ensemble, training schedule, planner and known reward or termination.
 * @returns The agent.
 * @concept model-based-rl
 * @example
 * ```ts
 * import { pets } from 'omai-engine/agents'
 * import { pendulum } from 'omai-engine/environments'
 * import { key } from 'omai-engine/random'
 * const agent = pets(pendulum.create(), { hidden: [8], members: 2 })
 * agent.act(agent.init(key(0)), [1, 0, 0], key(1)).length // => 1
 * ```
 */
export function pets(env: Spaces, options: PetsOptions = {}): Agent<PetsState, readonly number[], readonly number[]> {
  const warmup = options.warmup ?? 200
  const learner = ensembleLearner(env, options, warmup)
  const { low, high, size } = actionBox(env.actionSpace, 'pets')
  const horizon = options.horizon ?? 15
  const particles = options.particles ?? 4
  const shell = modelEnv(
    { model: learner.net.model, members: [] },
    { observationSpace: env.observationSpace, actionSpace: env.actionSpace, initial: () => [] },
  )
  const random = (k: Key): number[] => {
    const u = uniform(k, [size]).data
    return low.map((l, i) => l + ((high[i] ?? l) - l) * (u[i] ?? 0))
  }
  const plan = (s: PetsState, obs: readonly number[], k: Key): { plan: number[][]; value: number } => {
    const ensemble = learner.ensemble(s)
    const warm =
      s.plan === undefined ? undefined : [...s.plan.slice(1).map((u) => [...u]), [...(s.plan[s.plan.length - 1] ?? [])]]
    const r = cem(
      shell,
      { observation: obs, t: 0 },
      {
        horizon,
        population: options.population ?? 64,
        iterations: options.iterations ?? 3,
        key: k,
        score: (plans, sk) =>
          particleReturns(ensemble, obs, plans, particles, sk, {
            encode: learner.encode,
            ...(options.reward === undefined ? {} : { reward: options.reward }),
            ...(options.terminated === undefined ? {} : { terminated: options.terminated }),
            ...(options.gamma === undefined ? {} : { gamma: options.gamma }),
          }),
        ...(options.elites === undefined ? {} : { elites: options.elites }),
        ...(options.sigma === undefined ? {} : { sigma: options.sigma }),
        ...(warm === undefined ? {} : { initial: warm }),
      },
    )
    return { plan: r.plan, value: r.value }
  }
  const choose = (s: PetsState, obs: readonly number[], k: Key): number[] => {
    if (s.trainings === 0) return random(k)
    if (s.plan !== undefined && sameObservation(s.plannedFor, obs)) return [...(s.plan[0] ?? [])]
    return plan(s, obs, k).plan[0] ?? random(k)
  }
  return {
    name: 'pets',
    config: {
      hidden: [...(options.hidden ?? [64, 64])],
      members: options.members ?? 5,
      horizon,
      particles,
      population: options.population ?? 64,
      warmup,
    } as Json,
    metrics: {
      modelLoss: { description: 'Gaussian NLL of the last training round', direction: 'lower' },
      planValue: { description: 'predicted return of the current plan', direction: 'higher' },
    },
    init: () => ({ ...learner.init(), planValue: 0 }),
    act: (s, obs, k) => choose(s, obs, k),
    greedy: (s, obs) => choose(s, obs, makeKey(0)),
    learn: (s, t, k) => {
      const base = learner.record(s, t, foldIn(k, 0))
      const next: PetsState = { ...s, ...base }
      if (next.trainings === 0 || t.terminated || t.truncated) {
        const { plan: _p, plannedFor: _f, ...rest } = next
        return rest
      }
      const r = plan(next, t.nextObservation, foldIn(k, 1))
      return { ...next, plan: r.plan, plannedFor: [...t.nextObservation], planValue: r.value }
    },
    stats: (s) => ({ modelLoss: s.modelLoss, planValue: s.planValue }),
  }
}

// --------------------------------------------------------------------------------------------------------- MBPO

/** Options of {@link mbpo}. */
export interface MbpoOptions extends EnsembleOptions {
  /** Real steps before the first model is trained (default 200). */
  readonly warmup?: number
  /** Steps of each branched model rollout (default 1). */
  readonly rolloutLength?: number
  /** Model rollouts branched from replayed real states per real step (default 4). */
  readonly rollouts?: number
  /** A known termination function (learned models rarely predict termination well; MBPO uses known ones). */
  readonly terminated?: (observation: readonly number[]) => boolean
  /** A known reward function, replacing the learned reward. */
  readonly reward?: (observation: readonly number[], action: readonly number[], next: readonly number[]) => number
  /** Map a predicted observation onto the observation space (for example to the nearest one-hot vector). */
  readonly project?: (observation: readonly number[]) => readonly number[]
  /** Step the members by their means, without the aleatoric noise (default false: TS-1 samples). */
  readonly deterministic?: boolean
}

/** The state of an MBPO agent: the wrapped learner's state, the ensemble and the real data. */
export interface MbpoState<G> extends ModelBasedState {
  readonly learner: G
  /** Model transitions fed to the learner so far. */
  readonly modelTransitions: number
}

/**
 * Model-based policy optimisation in the style of Janner et al. (2019), wrapping any off-policy learner: real
 * transitions train both the learner and a probabilistic ensemble (as in {@link pets}); once the ensemble is trained,
 * every real step also branches `rollouts` short model rollouts of `rolloutLength` steps from replayed real states,
 * acting with the learner's current policy and sampling the ensemble by TS-1, and feeds those model transitions to
 * the learner as well, so it takes several updates per real step. The learner is any agent with vector observations
 * (`dqn` for discrete tasks, an actor-critic for continuous ones); acting is the learner's.
 *
 * @param env - The environment's spaces.
 * @param learner - The off-policy learner that consumes real and model transitions.
 * @param options - Ensemble, warm-up, rollout length and count, and known termination, reward or projection.
 * @returns The agent.
 * @concept model-based-rl
 * @example
 * ```ts
 * import { dqn, mbpo } from 'omai-engine/agents'
 * import { cartPole } from 'omai-engine/environments'
 * const env = cartPole.create()
 * mbpo(env, dqn(env, { hidden: [8] }), { hidden: [8] }).name // => 'mbpo-dqn'
 * ```
 */
export function mbpo<G, A extends number | readonly number[]>(
  env: Spaces,
  learner: Agent<G, readonly number[], A>,
  options: MbpoOptions = {},
): Agent<MbpoState<G>, readonly number[], A> {
  const warmup = options.warmup ?? 200
  const models = ensembleLearner(env, options, warmup)
  const length = options.rolloutLength ?? 1
  const count = options.rollouts ?? 4
  const project = options.project ?? ((o: readonly number[]) => o)
  const branch = (s: MbpoState<G>, k: Key): MbpoState<G> => {
    const ensemble = models.ensemble(s)
    let g = s.learner
    let fed = 0
    let obs = bufferSample(s.buffer, foldIn(k, 0), count).items.map((t) => [...t.observation])
    for (let h = 0; h < length && obs.length > 0; h++) {
      const kh = foldIn(k, h + 1)
      const actions = obs.map((o, i) => learner.act(g, o, foldIn(kh, i)))
      const samples = ensembleSample(
        ensemble,
        obs,
        actions.map((a) => models.encode(a)),
        foldIn(kh, 1_000_000),
        options.deterministic === true ? { deterministic: true } : {},
      )
      const alive: number[][] = []
      for (const [i, p] of samples.entries()) {
        const o = obs[i] ?? []
        const a = actions[i] as A
        const next = [...project(p.observation)]
        const reward =
          options.reward?.(o, typeof a === 'number' ? [a] : [...(a as readonly number[])], next) ?? p.reward
        const terminated = options.terminated?.(next) ?? false
        g = learner.learn(
          g,
          { observation: o, action: a, reward, nextObservation: next, terminated, truncated: false },
          foldIn(kh, 1000 + i),
        )
        fed += 1
        if (!terminated) alive.push(next)
      }
      obs = alive
    }
    return { ...s, learner: g, modelTransitions: s.modelTransitions + fed }
  }
  return {
    name: `mbpo-${learner.name}`,
    config: { learner: learner.config, rolloutLength: length, rollouts: count, warmup } as Json,
    metrics: {
      ...learner.metrics,
      modelLoss: { description: 'Gaussian NLL of the last ensemble training round', direction: 'lower' },
    },
    init: (k) => ({ ...models.init(), learner: learner.init(k), modelTransitions: 0 }),
    act: (s, obs, k) => learner.act(s.learner, obs, k),
    greedy: (s, obs) => learner.greedy(s.learner, obs),
    learn: (s, t, k) => {
      const g = learner.learn(s.learner, t, foldIn(k, 0))
      const base = models.record(s, t as Transition<readonly number[], readonly number[] | number>, foldIn(k, 1))
      const next: MbpoState<G> = { ...s, ...base, learner: g }
      return next.trainings === 0 ? next : branch(next, foldIn(k, 2))
    },
    stats: (s) => ({ ...(learner.stats?.(s.learner) ?? {}), modelLoss: s.modelLoss }),
  }
}

// ------------------------------------------------------------------------------------------------------ Dreamer

/** Options of {@link dreamer}. */
export interface DreamerOptions extends Partial<DreamerSizes> {
  /** Training sequences: batch size and length (default 8 and 8). */
  readonly batchSize?: number
  readonly sequenceLength?: number
  /** Imagination horizon (default 8), discount and lambda of the returns (default 0.99 and 0.95). */
  readonly horizon?: number
  readonly gamma?: number
  readonly lambda?: number
  /** KL balancing weight on the prior (default 0.8), free nats (default 1) and KL scale (default 1). */
  readonly klBalance?: number
  readonly freeNats?: number
  readonly klScale?: number
  /** Learning rates of the world model, actor and critic (default 3e-3, 1e-3, 1e-3). */
  readonly modelLearningRate?: number
  readonly actorLearningRate?: number
  readonly criticLearningRate?: number
  /** Actor entropy bonus (default 1e-3). */
  readonly entropy?: number
  /** Random-action steps before training (default 100), and an update every this many steps (default 5). */
  readonly warmup?: number
  readonly trainEvery?: number
  /** Finished episodes kept (default 100). */
  readonly capacity?: number
  readonly key?: Key
}

/** One stored step: an observation, the action taken there (encoded), its reward and whether the episode went on. */
interface DreamerItem {
  readonly observation: readonly number[]
  readonly action: readonly number[]
  readonly reward: number
  readonly continuation: number
}

/** The state of a Dreamer agent. */
export interface DreamerState {
  readonly params: DreamerParams
  readonly optStates: { readonly world: unknown; readonly actor: unknown; readonly critic: unknown }
  readonly store: EpisodeStore<DreamerItem>
  /** The filtered latent state for the current observation (what the actor acts on). */
  readonly belief?: { readonly h: readonly number[]; readonly z: readonly number[]; readonly for: readonly number[] }
  readonly steps: number
  readonly updates: number
  readonly worldLoss: number
  readonly kl: number
  readonly actorLoss: number
  readonly criticLoss: number
  /** Mean lambda-return of the last imagined batch. */
  readonly imaginedReturn: number
}

/**
 * A small latent world model in the style of Dreamer (Hafner et al. 2020), with DreamerV2's KL balancing: an encoder,
 * a latent dynamics model with a deterministic state (an MLP with tanh in place of the GRU, `h_t = f(h_{t-1},
 * z_{t-1}, a_{t-1})`) and a stochastic state (Gaussian prior `p(z | h)` and posterior `q(z | h, o)`), and decoder,
 * reward and continuation heads, trained on sampled sequences by the ELBO (reconstruction, reward and continuation
 * terms and `alpha KL(sg(q) || p) + (1 - alpha) KL(q || sg(p))` with free nats). An actor and a critic learn purely
 * in imagination: from the posterior states of the batch the world model is rolled forward with the actor, and
 * lambda-returns of the predicted rewards and discounts train the critic (regression) and the actor (backpropagation
 * through the model's dynamics for continuous actions, REINFORCE with the critic as baseline for discrete ones, plus
 * an entropy bonus). The agent filters its latent state along the episode and acts from it. Kept small: MLPs of one
 * hidden layer, short sequences and horizons.
 *
 * @param env - The environment's spaces (vector observations; box or discrete actions).
 * @param options - Sizes, sequence batch, imagination, KL balancing, learning rates and schedule.
 * @returns The agent.
 * @concept model-based-rl
 * @index agent Dreamer world-model agent.
 * @example
 * ```ts
 * import { dreamer } from 'omai-engine/agents'
 * import { pendulum } from 'omai-engine/environments'
 * import { key } from 'omai-engine/random'
 * const agent = dreamer(pendulum.create(), { hidden: 8 })
 * agent.act(agent.init(key(0)), [1, 0, 0], key(1)).length // => 1
 * ```
 */
export function dreamer<A extends number | readonly number[] = readonly number[]>(
  env: Spaces,
  options: DreamerOptions = {},
): Agent<DreamerState, readonly number[], A> {
  const d = observationSize(env.observationSpace)
  const isDiscrete = env.actionSpace.kind === 'discrete'
  const enc = actionEncoding(env.actionSpace)
  const bounds = isDiscrete ? { low: [], high: [] } : actionBox(env.actionSpace, 'dreamer')
  const sizes: DreamerSizes = {
    embed: options.embed ?? 16,
    deter: options.deter ?? 16,
    stoch: options.stoch ?? 4,
    hidden: options.hidden ?? 32,
  }
  const k0 = options.key ?? makeKey(0)
  const nets = dreamerNets(d, { size: enc.size, discrete: isDiscrete, low: bounds.low, high: bounds.high }, sizes, k0)
  const worldOpt = adam({ learningRate: options.modelLearningRate ?? 3e-3 })
  const actorOpt = adam({ learningRate: options.actorLearningRate ?? 1e-3 })
  const criticOpt = adam({ learningRate: options.criticLearningRate ?? 1e-3 })
  const B = options.batchSize ?? 8
  const L = options.sequenceLength ?? 8
  const H = options.horizon ?? 8
  const gamma = options.gamma ?? 0.99
  const lambda = options.lambda ?? 0.95
  const eta = options.entropy ?? 1e-3
  const warmup = options.warmup ?? 100
  const trainEvery = options.trainEvery ?? 5
  const kl = { klBalance: options.klBalance ?? 0.8, freeNats: options.freeNats ?? 1, klScale: options.klScale ?? 1 }

  const beliefOf = (
    params: DreamerParams,
    previous: { h: readonly number[]; z: readonly number[] } | undefined,
    action: readonly number[] | undefined,
    obs: readonly number[],
  ): { h: number[]; z: number[] } => {
    const h0 = rowsTensor([previous?.h ?? new Array<number>(sizes.deter).fill(0)], sizes.deter)
    const z0 = rowsTensor([previous?.z ?? new Array<number>(sizes.stoch).fill(0)], sizes.stoch)
    const a0 = rowsTensor([action ?? new Array<number>(enc.size).fill(0)], enc.size)
    const h = transition(nets, params.world, h0, z0, a0)
    const q = posterior(nets, params.world, h, rowsTensor([obs], d))
    return { h: toFlat(h), z: toFlat(q.mean) }
  }
  const choose = (s: DreamerState, obs: readonly number[], k: Key | undefined): A => {
    const b =
      s.belief !== undefined && sameObservation(s.belief.for, obs)
        ? s.belief
        : beliefOf(s.params, undefined, undefined, obs)
    if (s.steps < warmup && k !== undefined) {
      if (isDiscrete) return (Math.floor((uniform(k, []).data[0] ?? 0) * enc.size) % enc.size) as A
      const u = uniform(k, [enc.size]).data
      return bounds.low.map((l, i) => l + ((bounds.high[i] ?? l) - l) * (u[i] ?? 0)) as unknown as A
    }
    const head = actorHead(nets, s.params.actor, rowsTensor([[...b.h, ...b.z]], sizes.deter + sizes.stoch))
    const a = k === undefined ? head.mode() : head.sample(k)
    return (isDiscrete ? (toFlat(a)[0] ?? 0) : toFlat(a)) as unknown as A
  }

  const update = (s: DreamerState, k: Key): DreamerState => {
    const raw = sampleSequences(s.store, foldIn(k, 0), B, L)
    const len = minOf(raw.map((q) => q.length))
    if (len < 2) return s
    const seqs = raw.map((q) => q.slice(0, len))
    const at = (t: number, f: (it: DreamerItem) => readonly number[]): Value =>
      rowsTensor(
        seqs.map((q) => f(q[t] as DreamerItem)),
        f(seqs[0]?.[t] as DreamerItem).length,
      )
    const batch: SequenceBatch = {
      observations: Array.from({ length: len }, (_, t) => at(t, (it) => it.observation)),
      actions: Array.from({ length: len }, (_, t) => at(t, (it) => it.action)),
      rewards: Array.from({ length: len }, (_, t) => vectorTensor(seqs.map((q) => q[t]?.reward ?? 0))),
      continuations: Array.from({ length: len }, (_, t) => vectorTensor(seqs.map((q) => q[t]?.continuation ?? 0))),
    }
    const kw = foldIn(k, 1)
    const w = gradientStep(
      (p) => worldPass(nets, p as DreamerParams['world'], batch, kw, kl).loss,
      s.params.world,
      worldOpt,
      s.optStates.world,
    )
    const world = w.params as DreamerParams['world']
    // The posterior states of the batch (recomputed with the same noise) start the imagined rollouts.
    const filtered = worldPass(nets, world, batch, kw, kl)
    const h0 = concat(filtered.h as Value[], 0)
    const z0 = concat(filtered.z as Value[], 0)
    const ki = foldIn(k, 2)
    const opts = { gamma, lambda }
    let actor: ParamTree
    let actorLoss: number
    let actorState: unknown
    if (!isDiscrete) {
      const step = gradientStep(
        (ap) => {
          const im = imagine(nets, { world, actor: ap, critic: s.params.critic }, h0, z0, H, ki, opts)
          let ret: Value = mean(im.returns[0] as Value)
          for (let j = 1; j < H; j++) ret = add(ret, mean(im.returns[j] as Value))
          return sub(mul(ret, -1 / H), mul(im.entropy, eta))
        },
        s.params.actor,
        actorOpt,
        s.optStates.actor,
      )
      ;[actor, actorLoss, actorState] = [step.params, step.loss, step.optState]
    } else {
      const im = imagine(nets, s.params, h0, z0, H, ki, opts)
      const advantages = im.returns.map((r, j) => vectorTensor(toFlat(sub(r, im.values[j] as Value))))
      const step = gradientStep(
        (ap) => {
          let obj: Value = vectorTensor([0])
          let ent: Value = vectorTensor([0])
          for (let j = 0; j < H; j++) {
            const head = actorHead(nets, ap, im.features[j] as Value)
            const idx = argmax(im.actions[j] as Value, -1)
            obj = add(obj, mean(mul(head.logProb(idx), advantages[j] as Value)))
            ent = add(ent, mean(head.entropy()))
          }
          return sum(sub(mul(obj, -1 / H), mul(ent, eta / H)))
        },
        s.params.actor,
        actorOpt,
        s.optStates.actor,
      )
      ;[actor, actorLoss, actorState] = [step.params, step.loss, step.optState]
    }
    // The critic regresses on the lambda-returns of an imagined rollout of the updated actor.
    const im = imagine(nets, { world, actor, critic: s.params.critic }, h0, z0, H, ki, opts)
    const feats = im.features.slice(0, H)
    const targets = im.returns.map((r) => vectorTensor(toFlat(r)))
    const c = gradientStep(
      (cp) => {
        let loss: Value = vectorTensor([0])
        for (let j = 0; j < H; j++) {
          const v = squeeze(nets.critic.apply(cp as never, feats[j] as Value), -1)
          loss = add(loss, mean(mul(square(sub(v, targets[j] as Value)), 0.5)))
        }
        return sum(mul(loss, 1 / H))
      },
      s.params.critic,
      criticOpt,
      s.optStates.critic,
    )
    let imagined = 0
    for (const r of targets) imagined += meanOf(toFlat(r)) / H
    return {
      ...s,
      params: { world, actor, critic: c.params },
      optStates: { world: w.optState, actor: actorState, critic: c.optState },
      updates: s.updates + 1,
      worldLoss: w.loss,
      kl: toFlat(filtered.kl)[0] ?? 0,
      actorLoss,
      criticLoss: c.loss,
      imaginedReturn: imagined,
    }
  }

  return {
    name: 'dreamer',
    config: { ...sizes, batchSize: B, sequenceLength: L, horizon: H, gamma, lambda, ...kl } as Json,
    metrics: {
      worldLoss: { description: 'world-model ELBO loss of the last update', direction: 'lower' },
      kl: { description: 'balanced KL of the last update' },
      actorLoss: { description: 'actor loss (minus imagined lambda-returns)', direction: 'lower' },
      criticLoss: { description: 'critic regression loss', direction: 'lower' },
      imaginedReturn: { description: 'mean imagined lambda-return', direction: 'higher' },
    },
    init: () => {
      const params = dreamerParams(nets)
      return {
        params,
        optStates: {
          world: worldOpt.init(params.world),
          actor: actorOpt.init(params.actor),
          critic: criticOpt.init(params.critic),
        },
        store: episodeStore(options.capacity ?? 100),
        steps: 0,
        updates: 0,
        worldLoss: 0,
        kl: 0,
        actorLoss: 0,
        criticLoss: 0,
        imaginedReturn: 0,
      }
    },
    act: (s, obs, k) => choose(s, obs, k),
    greedy: (s, obs) => choose(s, obs, undefined),
    learn: (s, t, k) => {
      const a = enc.encode(t.action as number | readonly number[])
      const done = t.terminated || t.truncated
      let store = episodeAdd(
        s.store,
        { observation: [...t.observation], action: a, reward: t.reward, continuation: t.terminated ? 0 : 1 },
        false,
      )
      if (done)
        store = episodeAdd(
          store,
          {
            observation: [...t.nextObservation],
            action: new Array<number>(enc.size).fill(0),
            reward: 0,
            continuation: 0,
          },
          true,
        )
      let next: DreamerState = { ...s, store, steps: s.steps + 1 }
      if (next.steps >= warmup && next.steps % trainEvery === 0 && store.episodes.length > 0)
        next = update(next, foldIn(k, 0))
      if (done) {
        const { belief: _b, ...rest } = next
        return rest
      }
      const previous =
        s.belief !== undefined && sameObservation(s.belief.for, t.observation)
          ? s.belief
          : beliefOf(next.params, undefined, undefined, t.observation)
      const b = beliefOf(next.params, previous, a, t.nextObservation)
      return { ...next, belief: { ...b, for: [...t.nextObservation] } }
    },
    stats: (s) => ({
      worldLoss: s.worldLoss,
      kl: s.kl,
      actorLoss: s.actorLoss,
      criticLoss: s.criticLoss,
      imaginedReturn: s.imaginedReturn,
    }),
  }
}

// ------------------------------------------------------------------------------------------------------- MuZero

/** Options of {@link muZero}. */
export interface MuZeroOptions {
  /** Hidden-state size and MLP width (default 16 and 32). */
  readonly latent?: number
  readonly hidden?: number
  /** Unrolled steps K and the n of the n-step value targets (default 3 and 5). */
  readonly unroll?: number
  readonly nStep?: number
  readonly gamma?: number
  /** Search: simulations per move, PUCT constant and depth (default 16, 1.25 and 5). */
  readonly simulations?: number
  readonly c?: number
  readonly depth?: number
  /** Sampling temperature over visit counts when acting (default 1; greedy actions take the most visited). */
  readonly temperature?: number
  readonly batchSize?: number
  readonly learningRate?: number
  /** Steps before training and an update every this many steps (default 32 and 1). */
  readonly warmup?: number
  readonly trainEvery?: number
  /** Finished episodes kept (default 200). */
  readonly capacity?: number
  readonly key?: Key
}

/** The state of a MuZero agent. */
export interface MuZeroState {
  readonly params: MuZeroParams
  readonly optState: unknown
  readonly episodes: readonly MuZeroEpisode[]
  readonly current: readonly MuZeroStep[]
  /** The search made in `learn` for the next observation. */
  readonly search?: SearchResult & { readonly for: readonly number[] }
  readonly steps: number
  readonly updates: number
  readonly loss: number
  readonly rootValue: number
}

/**
 * MuZero-style planning with a learned model (Schrittwieser et al. 2020): a representation network maps the
 * observation to a hidden state (scaled to [0, 1]), a dynamics network maps a hidden state and an action to the next
 * hidden state and the reward, and a prediction network gives a policy and a value. Every move searches the learned
 * model: it is wrapped as an environment over hidden states for lib/rl `mcts`, with the predicted policy as the PUCT
 * prior and the predicted value at the leaves. The visit distribution is the policy target and the root value the
 * bootstrap of n-step value targets; training unrolls the model K steps along stored actions from sampled positions,
 * matching values, rewards and policies (dynamics gradients scaled by one half). Actions are sampled from the visit
 * counts (with a temperature); `greedy` takes the most visited. Searches are made in `learn` for the next
 * observation and reused by `act`.
 *
 * @param env - The environment's spaces (vector observations, discrete actions).
 * @param options - Network sizes, unroll and n-step, search, sampling temperature, optimiser and schedule.
 * @returns The agent.
 * @concept model-based-rl
 * @example
 * ```ts
 * import { muZero } from 'omai-engine/agents'
 * import { cartPole } from 'omai-engine/environments'
 * import { key } from 'omai-engine/random'
 * const agent = muZero(cartPole.create(), { hidden: 8, simulations: 4 })
 * [0, 1].includes(agent.greedy(agent.init(key(0)), [0, 0, 0, 0])) // => true
 * ```
 */
export function muZero(env: Spaces, options: MuZeroOptions = {}): Agent<MuZeroState, readonly number[], number> {
  const d = observationSize(env.observationSpace)
  const nA = actionCount(env.actionSpace, 'muZero')
  const nets = muZeroNets(d, nA, options.latent ?? 16, options.hidden ?? 32, options.key ?? makeKey(0))
  const opt = adam({ learningRate: options.learningRate ?? 3e-3 })
  const K = options.unroll ?? 3
  const n = options.nStep ?? 5
  const gamma = options.gamma ?? 0.97
  const search = {
    simulations: options.simulations ?? 16,
    c: options.c ?? 1.25,
    gamma,
    depth: options.depth ?? 5,
  }
  const temperature = options.temperature ?? 1
  const B = options.batchSize ?? 16
  const warmup = options.warmup ?? 32
  const trainEvery = options.trainEvery ?? 1
  const capacity = options.capacity ?? 200
  const searchOf = (s: MuZeroState, obs: readonly number[], k: Key): SearchResult =>
    s.search !== undefined && sameObservation(s.search.for, obs)
      ? s.search
      : searchFrom(nets, s.params, obs, { ...search, key: k })
  const train = (s: MuZeroState, k: Key): MuZeroState => {
    const pool = s.episodes
    if (pool.length === 0) return s
    const lengths = pool.map((e) => e.steps.length)
    let total = 0
    for (const l of lengths) total += l
    const u = uniform(foldIn(k, 0), [B]).data
    const obs: (readonly number[])[] = []
    const targets: UnrollTargets[] = []
    for (let b = 0; b < B; b++) {
      let r = Math.floor((u[b] ?? 0) * total)
      let e = 0
      while (r >= (lengths[e] ?? 0)) {
        r -= lengths[e] ?? 0
        e += 1
      }
      const ep = pool[e] as MuZeroEpisode
      obs.push(ep.steps[r]?.observation ?? [])
      targets.push(unrollTargets(ep, r, K, n, gamma, nA))
    }
    const step = gradientStep((p) => unrollLoss(nets, p as MuZeroParams, obs, targets, K), s.params, opt, s.optState)
    return {
      ...s,
      params: step.params as MuZeroParams,
      optState: step.optState,
      updates: s.updates + 1,
      loss: step.loss,
    }
  }
  const pick = (r: SearchResult, k: Key | undefined): number => {
    if (k === undefined || temperature === 0) return argmaxOf(r.visits)
    const w = r.visits.map((v) => v ** (1 / temperature))
    let total = 0
    for (const x of w) total += x
    let u = (uniform(k, []).data[0] ?? 0) * total
    for (const [i, x] of w.entries()) {
      u -= x
      if (u < 0) return i
    }
    return w.length - 1
  }
  return {
    name: 'muzero',
    config: { latent: nets.latent, unroll: K, nStep: n, ...search, batchSize: B } as Json,
    metrics: {
      loss: { description: 'unrolled MuZero loss of the last update', direction: 'lower' },
      rootValue: { description: 'value of the last search root', direction: 'higher' },
    },
    init: () => {
      const params = muZeroParams(nets)
      return {
        params,
        optState: opt.init(params),
        episodes: [],
        current: [],
        steps: 0,
        updates: 0,
        loss: 0,
        rootValue: 0,
      }
    },
    act: (s, obs, k) => pick(searchOf(s, obs, foldIn(k, 0)), foldIn(k, 1)),
    greedy: (s, obs) => pick(searchOf(s, obs, makeKey(0)), undefined),
    learn: (s, t, k) => {
      const root = searchOf(s, t.observation, foldIn(k, 0))
      let visits = 0
      for (const v of root.visits) visits += v
      const step: MuZeroStep = {
        observation: [...t.observation],
        action: t.action,
        reward: t.reward,
        policy: root.visits.map((v) => v / Math.max(visits, 1)),
        rootValue: root.rootValue,
      }
      const done = t.terminated || t.truncated
      let next: MuZeroState = { ...s, current: [...s.current, step], steps: s.steps + 1, rootValue: root.rootValue }
      if (done) {
        const final = t.terminated
          ? 0
          : (toFlat(predict(nets, s.params, represent(nets, s.params, rowsTensor([t.nextObservation], d))).value)[0] ??
            0)
        const episode: MuZeroEpisode = { steps: next.current, final, terminated: t.terminated }
        next = { ...next, episodes: [...next.episodes, episode].slice(-capacity), current: [] }
      }
      if (next.steps >= warmup && next.steps % trainEvery === 0) next = train(next, foldIn(k, 1))
      if (done) {
        const { search: _s, ...rest } = next
        return rest
      }
      const ahead = searchFrom(nets, next.params, t.nextObservation, { ...search, key: foldIn(k, 2) })
      return { ...next, search: { ...ahead, for: [...t.nextObservation] } }
    },
    stats: (s) => ({ loss: s.loss, rootValue: s.rootValue }),
  }
}
