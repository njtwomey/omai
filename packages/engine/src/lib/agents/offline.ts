/**
 * Offline (batch) reinforcement learning: learning from a fixed dataset of transitions collected by some behaviour
 * policy, without interacting with the environment. Collecting a dataset and splitting it into episodes, the
 * `OfflineLearner` contract (pure functions over a state, updated on minibatches), `offlineAlgorithm` and
 * `trainOffline` to run a learner as a traced algorithm, `offlineAgent` to evaluate the result like any other agent,
 * and behaviour cloning as the simplest learner.
 */

import { argmaxOf, type Json, OmaiError } from '../base/index.ts'
import { crossEntropy, gaussianNll, mse } from '../losses/index.ts'
import type { ParamTree } from '../optim/index.ts'
import { foldIn, type Key, key as makeKey, normal, randint } from '../random/index.ts'
import { type Agent, categoricalPolicy, type Env, type Policy, runEpisode, type Transition } from '../rl/index.ts'
import { exp, mul, toFlat, type Value } from '../tensor/index.ts'
import { type Algorithm, channels, type MetricChannel, type RunOptions, runAlgorithm } from '../trace/index.ts'
import { gradientStepWithTerms, optimiserOf, rowsTensor, type Spaces, vectorTensor } from './_common.ts'
import { clipToBox, network, type OfflineSpaces, offlineBatch, offlineSpaces, squashToBox } from './_offline.ts'

// ------------------------------------------------------------------------------------------------- datasets

/** A fixed dataset of transitions (in collection order, so episodes can be recovered). */
export type TransitionDataset<O, A> = readonly Transition<O, A>[]

/** Options of {@link collectDataset}. */
export interface CollectOptions {
  /** Collect this many whole episodes (default 100 unless `steps` is given). */
  readonly episodes?: number
  /** Collect whole episodes until at least this many transitions. */
  readonly steps?: number
  /** The key (default from `seed`, default 0). */
  readonly key?: Key
  readonly seed?: number
}

/**
 * Collect a dataset of whole episodes with a behaviour policy (an expert, a noisy expert, a random policy or a
 * trained agent's policy): episode `i` runs with `foldIn(key, i)`, as {@link runEpisode} does.
 *
 * @param env - The environment (with its time limit, as from `make`).
 * @param policy - The behaviour policy.
 * @param options - How many episodes or steps, and the key or seed.
 * @returns The transitions, episode after episode.
 * @concept offline-rl
 * @example
 * ```ts
 * import { collectDataset } from 'omai-engine/agents'
 * import { cartPole } from 'omai-engine/environments'
 * import { make } from 'omai-engine/rl'
 * const data = collectDataset(make(cartPole), () => 1, { episodes: 2 })
 * data.filter((t) => t.terminated).length // => 2
 * ```
 */
export function collectDataset<S, O, A>(
  env: Env<S, O, A>,
  policy: Policy<O, A>,
  options: CollectOptions = {},
): Transition<O, A>[] {
  const key = options.key ?? makeKey(options.seed ?? 0)
  const episodes = options.episodes ?? (options.steps === undefined ? 100 : Infinity)
  const steps = options.steps ?? Infinity
  const out: Transition<O, A>[] = []
  for (let i = 0; i < episodes && out.length < steps; i++) {
    const ep = runEpisode(env, policy, foldIn(key, i))
    for (let t = 0; t < ep.length; t++) {
      const last = t === ep.length - 1
      out.push({
        observation: ep.observations[t] as O,
        action: ep.actions[t] as A,
        reward: ep.rewards[t] ?? 0,
        nextObservation: ep.observations[t + 1] as O,
        terminated: last && ep.terminated,
        truncated: last && ep.truncated,
        info: ep.infos[t] ?? {},
      })
    }
  }
  return out
}

/**
 * Split a dataset into its episodes (at terminated or truncated transitions; a trailing unfinished episode is kept).
 *
 * @param dataset - The dataset, in collection order.
 * @returns The episodes.
 * @concept offline-rl
 * @example
 * ```ts
 * import { datasetEpisodes } from 'omai-engine/agents'
 * const t = { observation: 0, action: 0, reward: 1, nextObservation: 0, terminated: false, truncated: false }
 * datasetEpisodes([t, { ...t, terminated: true }, t]).map((e) => e.length) // => [2, 1]
 * ```
 */
export function datasetEpisodes<O, A>(dataset: TransitionDataset<O, A>): Transition<O, A>[][] {
  const out: Transition<O, A>[][] = []
  let current: Transition<O, A>[] = []
  for (const t of dataset) {
    current.push(t)
    if (t.terminated || t.truncated) {
      out.push(current)
      current = []
    }
  }
  if (current.length > 0) out.push(current)
  return out
}

// ----------------------------------------------------------------------------------------- the offline runner

/**
 * An offline method as pure functions over its state: `init` sees the dataset (for normalisation statistics),
 * `update` takes one gradient step (or several coupled ones) on a minibatch, `greedy` is the learned policy.
 *
 * @concept offline-rl
 */
export interface OfflineLearner<G, O, A> {
  readonly name: string
  readonly config: Json
  readonly metrics: Readonly<Record<string, MetricChannel>>
  init(key: Key, dataset: TransitionDataset<O, A>): G
  update(state: G, batch: readonly Transition<O, A>[], key: Key): G
  stats(state: G): Readonly<Record<string, number>>
  greedy(state: G, observation: O): A
  /** A stochastic action (default the greedy one). */
  act?(state: G, observation: O, key: Key): A
}

/** Options of {@link offlineAlgorithm} and {@link trainOffline}. */
export interface OfflineOptions {
  /** Gradient steps (default 1000). */
  readonly steps?: number
  /** Minibatch size, sampled uniformly with replacement (default 64). */
  readonly batchSize?: number
}

/** The state of {@link offlineAlgorithm}. */
export interface OfflineState<G> {
  readonly learner: G
  readonly step: number
}

/**
 * An offline learner as an {@link Algorithm}: each step samples a minibatch of the dataset and updates the learner,
 * recording the learner's statistics.
 *
 * @param learner - The learner.
 * @param dataset - The dataset.
 * @param options - Steps and batch size.
 * @returns The algorithm.
 * @throws OmaiError with code `value` for an empty dataset.
 * @concept offline-rl
 * @example
 * ```ts
 * import { behaviourCloningLearner, offlineAlgorithm } from 'omai-engine/agents'
 * import { cartPole } from 'omai-engine/environments'
 * import { key } from 'omai-engine/random'
 * const t = { observation: [0, 0, 0, 0], action: 1, reward: 1, nextObservation: [0, 0, 0, 0], terminated: false, truncated: false }
 * offlineAlgorithm(behaviourCloningLearner(cartPole.create()), [t]).name // => 'behaviourCloning'
 * ```
 */
export function offlineAlgorithm<G, O, A>(
  learner: OfflineLearner<G, O, A>,
  dataset: TransitionDataset<O, A>,
  options: OfflineOptions = {},
): Algorithm<OfflineState<G>> {
  if (dataset.length === 0) throw new OmaiError('value', `${learner.name}: the dataset is empty`)
  const steps = options.steps ?? 1000
  const batchSize = options.batchSize ?? 64
  return {
    name: learner.name,
    config: { ...(learner.config as Record<string, Json>), steps, batchSize, transitions: dataset.length } as Json,
    schema: channels({ metrics: learner.metrics }),
    init: (k) => ({ learner: learner.init(k, dataset), step: 0 }),
    step: (s, ctx) => {
      const idx = randint(foldIn(ctx.key, 0), [batchSize], 0, dataset.length).data
      const batch = Array.from(idx, (i) => dataset[i] as Transition<O, A>)
      return { learner: learner.update(s.learner, batch, foldIn(ctx.key, 1)), step: s.step + 1 }
    },
    done: (s) => (s.step >= steps ? 'completed' : undefined),
    observe: (s) => ({ metrics: learner.stats(s.learner) }),
  }
}

/**
 * An offline learner's trained state seen as an {@link Agent}: `init` returns the trained state, `act` and `greedy`
 * are the learned policy, and `learn` ignores online transitions, so the result works with `evaluatePolicy`,
 * `greedyPolicyOf`, `compareAgents` and `trainAgent`.
 *
 * @param learner - The learner.
 * @param trained - Its trained state.
 * @returns The agent.
 * @concept offline-rl
 * @example
 * ```ts
 * import { behaviourCloningLearner, offlineAgent } from 'omai-engine/agents'
 * import { cartPole } from 'omai-engine/environments'
 * import { key } from 'omai-engine/random'
 * const learner = behaviourCloningLearner(cartPole.create(), { hidden: [4] })
 * offlineAgent(learner, learner.init(key(0), [])).name // => 'behaviourCloning'
 * ```
 */
export function offlineAgent<G, O, A>(learner: OfflineLearner<G, O, A>, trained: G): Agent<G, O, A> {
  return {
    name: learner.name,
    config: learner.config,
    metrics: learner.metrics,
    init: () => trained,
    act: (s, o, k) => (learner.act === undefined ? learner.greedy(s, o) : learner.act(s, o, k)),
    learn: (s) => s,
    greedy: (s, o) => learner.greedy(s, o),
    stats: (s) => learner.stats(s),
  }
}

/** The outcome of {@link trainOffline}. */
export interface OfflineResult<G, O, A> {
  readonly state: G
  /** The trained state as an agent. */
  readonly agent: Agent<G, O, A>
  /** The greedy policy. */
  readonly policy: Policy<O, A>
  readonly steps: number
}

/**
 * Train an offline learner on a dataset and return its state, agent view and greedy policy.
 *
 * @param learner - The learner.
 * @param dataset - The dataset.
 * @param options - Steps, batch size and the run options (seed or key, tracer, trace policy).
 * @returns The result.
 * @concept offline-rl
 * @example
 * ```ts
 * import { behaviourCloningLearner, trainOffline } from 'omai-engine/agents'
 * import { cartPole } from 'omai-engine/environments'
 * import { key } from 'omai-engine/random'
 * const t = { observation: [0, 0, 0, 0], action: 1, reward: 1, nextObservation: [0, 0, 0, 0], terminated: false, truncated: false }
 * trainOffline(behaviourCloningLearner(cartPole.create(), { hidden: [4] }), [t], { steps: 20 }).policy([0, 0, 0, 0], key(0)) // => 1
 * ```
 */
export function trainOffline<G, O, A>(
  learner: OfflineLearner<G, O, A>,
  dataset: TransitionDataset<O, A>,
  options: OfflineOptions & Partial<Omit<RunOptions, 'maxSteps'>> = {},
): OfflineResult<G, O, A> {
  const { steps, batchSize, ...run } = options
  const algo = offlineAlgorithm(learner, dataset, {
    ...(steps === undefined ? {} : { steps }),
    ...(batchSize === undefined ? {} : { batchSize }),
  })
  const result = runAlgorithm(algo, { ...run, maxSteps: (steps ?? 1000) + 1 })
  const state = result.state.learner
  return {
    state,
    agent: offlineAgent(learner, state),
    policy: (o) => learner.greedy(state, o),
    steps: result.state.step,
  }
}

// ------------------------------------------------------------------------------------- behaviour cloning

/** Options shared by the offline learners. */
export interface OfflineNetOptions {
  /** Hidden widths (default `[64, 64]`). */
  readonly hidden?: readonly number[]
  readonly learningRate?: number
  readonly maxGradNorm?: number
  /** The initialisation key (default key(0)). */
  readonly key?: Key
}

/** Options of {@link behaviourCloning}. */
export interface BehaviourCloningOptions extends OfflineNetOptions {
  /**
   * The likelihood for box actions: `'mse'` (a deterministic policy, the default) or `'gaussian'` (a Gaussian with a
   * learned state-independent standard deviation, fitted by maximum likelihood). Discrete actions use cross-entropy.
   */
  readonly loss?: 'mse' | 'gaussian'
}

/** The state of a behaviour-cloning learner. */
export interface BehaviourCloningState {
  readonly params: ParamTree
  readonly optState: unknown
  readonly loss: number
  readonly updates: number
}

function policyNet(sp: OfflineSpaces, hidden: readonly number[], key: Key) {
  return network([sp.obsSize, ...hidden, sp.actionSize], key)
}

/**
 * Behaviour cloning: supervised learning of the behaviour policy from the dataset's (observation, action) pairs,
 * by cross-entropy for discrete actions and by squared error (a deterministic policy squashed into the action box) or
 * Gaussian maximum likelihood for continuous ones. Discrete agents act by sampling their softmax policy (greedy:
 * the most likely action); Gaussian agents sample (greedy: the mean).
 *
 * @param env - The environment's spaces.
 * @param options - Network, optimiser and the continuous likelihood.
 * @returns The learner.
 * @concept offline-rl
 * @example
 * ```ts
 * import { behaviourCloningLearner } from 'omai-engine/agents'
 * import { pendulum } from 'omai-engine/environments'
 * behaviourCloningLearner(pendulum.create(), { loss: 'gaussian' }).config // => { hidden: [64, 64], loss: 'gaussian' }
 * ```
 */
export function behaviourCloningLearner<O, A>(
  env: Spaces,
  options: BehaviourCloningOptions = {},
): OfflineLearner<BehaviourCloningState, O, A> {
  const sp = offlineSpaces(env)
  const hidden = options.hidden ?? [64, 64]
  const kind = sp.discrete ? 'crossEntropy' : (options.loss ?? 'mse')
  const net = policyNet(sp, hidden, options.key ?? makeKey(0))
  const opt = optimiserOf(options, 1e-3)
  const init: ParamTree =
    kind === 'gaussian'
      ? { net: net.params, logStd: vectorTensor(new Array<number>(sp.actionSize).fill(0)) }
      : { net: net.params }
  const head = (p: ParamTree, x: Value): Value => {
    const u = net.apply((p as { net: ParamTree }).net, x)
    return sp.discrete ? u : squashToBox(sp, u)
  }
  const out = (s: BehaviourCloningState, o: O): Value => head(s.params, rowsTensor([sp.features(o)], sp.obsSize))
  const toAction = (v: Value): A => (sp.discrete ? (argmaxOf(toFlat(v)) as A) : (clipToBox(sp, toFlat(v)) as A))
  return {
    name: 'behaviourCloning',
    config: { hidden: [...hidden], ...(sp.discrete ? {} : { loss: kind }) },
    metrics: { loss: { description: 'imitation loss of the last minibatch', direction: 'lower' } },
    init: () => ({ params: init, optState: opt.init(init), loss: 0, updates: 0 }),
    update: (s, batch) => {
      const b = offlineBatch(sp, batch)
      const loss = (p: ParamTree): readonly [Value, Value] => {
        const y = head(p, b.obs)
        let l: Value
        if (kind === 'crossEntropy') l = crossEntropy(y, b.labels)
        else if (kind === 'mse') l = mse(y, b.actions)
        else {
          const logStd = (p as { logStd: Value }).logStd
          l = mul(gaussianNll(y, b.actions, exp(mul(logStd, 2)), { full: true }), sp.actionSize)
        }
        return [l, l]
      }
      const step = gradientStepWithTerms(loss, s.params, opt, s.optState)
      return { params: step.params, optState: step.optState, loss: step.loss, updates: s.updates + 1 }
    },
    stats: (s) => ({ loss: s.loss }),
    greedy: (s, o) => toAction(out(s, o)),
    act: (s, o, k) => {
      const y = out(s, o)
      if (sp.discrete) return toFlat(categoricalPolicy(y).sample(k))[0] as A
      if (kind !== 'gaussian') return toAction(y)
      const std = toFlat(exp((s.params as { logStd: Value }).logStd))
      const noise = toFlat(normal(k, [sp.actionSize]))
      return clipToBox(
        sp,
        toFlat(y).map((m, i) => m + (std[i] ?? 0) * (noise[i] ?? 0)),
      ) as A
    },
  }
}

/**
 * Behaviour cloning trained on a dataset ({@link behaviourCloningLearner} run by {@link trainOffline}).
 *
 * @param env - The environment's spaces.
 * @param dataset - The dataset.
 * @param options - The learner's options, steps, batch size and run options.
 * @returns The trained state, agent and policy.
 * @concept offline-rl
 * @example
 * ```ts
 * import { behaviourCloning } from 'omai-engine/agents'
 * import { cartPole } from 'omai-engine/environments'
 * import { key } from 'omai-engine/random'
 * const t = { observation: [0, 0, 0, 0], action: 0, reward: 1, nextObservation: [0, 0, 0, 0], terminated: false, truncated: false }
 * behaviourCloning(cartPole.create(), [t], { hidden: [4], steps: 50, learningRate: 0.05 }).policy([0, 0, 0, 0], key(0)) // => 0
 * ```
 */
export function behaviourCloning<O, A>(
  env: Spaces,
  dataset: TransitionDataset<O, A>,
  options: BehaviourCloningOptions & OfflineOptions & Partial<Omit<RunOptions, 'maxSteps'>> = {},
): OfflineResult<BehaviourCloningState, O, A> {
  return trainOffline(behaviourCloningLearner<O, A>(env, options), dataset, options)
}
