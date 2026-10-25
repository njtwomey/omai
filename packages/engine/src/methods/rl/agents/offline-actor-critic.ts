/**
 * Offline actor-critic methods for learning from a fixed dataset: implicit Q-learning (IQL, for discrete and continuous
 * actions), which fits its values with an expectile loss and so never queries the critic at actions outside the
 * dataset, and TD3+BC, TD3 with a behaviour-cloning term in the actor's objective that keeps the policy near the data.
 * Each is an offline learner with a function that trains it on a dataset.
 */

import { argmaxOf, type Json, MdmlError } from '../../../lib/base/index.ts'
import { mse } from '../../../lib/losses/index.ts'
import { incrementalUpdate, type Tree } from '../../../lib/optim/index.ts'
import { foldIn, type Key, key as makeKey, normal } from '../../../lib/random/index.ts'
import { categoricalPolicy, gaussianPolicy, type Transition } from '../../../lib/rl/index.ts'
import {
  add,
  clip,
  concat,
  less,
  logSoftmax,
  mean,
  minimum,
  mul,
  neg,
  reshape,
  square,
  stack,
  sub,
  sum,
  toFlat,
  type Value,
  where,
} from '../../../lib/tensor/index.ts'
import type { RunOptions } from '../../../lib/trace/index.ts'
import { gradientStepWithTerms, optimiserOf, rowsTensor, type Spaces, vectorTensor } from './_common.ts'
import {
  type Net,
  network,
  type OfflineBatch,
  type OfflineSpaces,
  offlineBatch,
  offlineSpaces,
  squashToBox,
} from './_offline.ts'
import {
  type Dataset,
  type OfflineLearner,
  type OfflineNetOptions,
  type OfflineOptions,
  type OfflineResult,
  trainOffline,
} from './offline.ts'

// A critic Q(s, a) for either action kind: discrete critics output one value per action and pick the data action;
// continuous critics take the action as an input.
function critic(
  sp: OfflineSpaces,
  hidden: readonly number[],
  key: Key,
): { net: Net; q: (p: Tree, x: Value, a: Value) => Value } {
  if (sp.discrete) {
    const net = network([sp.obsSize, ...hidden, sp.actionSize], key)
    return { net, q: (p, x, a) => sum(mul(net.apply(p, x), a), -1) }
  }
  const net = network([sp.obsSize + sp.actionSize, ...hidden, 1], key)
  return { net, q: (p, x, a) => reshape(net.apply(p, concat([x, a], -1)), [x.shape[0] ?? 1]) }
}

const column = (x: Value): Value => reshape(x, [x.shape[0] ?? 1])

// --------------------------------------------------------------------------------------------------------- IQL

/** Options of {@link iql}. */
export interface IqlOptions extends OfflineNetOptions {
  readonly gamma?: number
  /** The expectile `tau` of the value regression (default 0.7). */
  readonly expectile?: number
  /** The inverse temperature `beta` of advantage-weighted extraction (default 3). */
  readonly temperature?: number
  /** The largest advantage weight (default 100). */
  readonly maxWeight?: number
  /** Polyak weight of the target critics (default 0.005). */
  readonly targetTau?: number
}

/** The state of an IQL learner. */
export interface IqlState {
  readonly params: Tree
  readonly target: Tree
  readonly optState: unknown
  readonly updates: number
  readonly valueLoss: number
  readonly criticLoss: number
  readonly actorLoss: number
  readonly advantage: number
}

/**
 * The expectile loss `|tau - 1{u < 0}| u^2` averaged over a batch (IQL's value objective; `tau = 0.5` is half the
 * squared error, larger `tau` approaches the maximum).
 *
 * @param u - The residuals `Q(s, a) - V(s)`.
 * @param tau - The expectile.
 * @returns The mean loss, differentiable.
 * @concept offline-rl
 * @example
 * ```ts
 * import { expectileLoss } from 'mdml-engine/methods/rl/agents'
 * import { item, tensor } from 'mdml-engine/tensor'
 * item(expectileLoss(tensor([2, -2]), 0.75)) // => 2
 * ```
 */
export function expectileLoss(u: Value, tau: number): Value {
  return mean(mul(where(less(u, 0), 1 - tau, tau), square(u)))
}

/**
 * Implicit Q-learning (Kostrikov, Nair and Levine 2021): an in-sample form of Q-learning that never evaluates the
 * critic at actions outside the dataset. `V(s)` regresses the target critics' `min(Q1, Q2)(s, a)` by an upper
 * expectile; the twin critics regress `r + gamma (1 - done) V(s')`; the policy is extracted by advantage-weighted
 * regression, maximising `exp(beta (Q(s, a) - V(s))) log pi(a|s)` (weights capped). Discrete policies are softmax
 * classifiers; continuous ones are Gaussians with tanh-bounded means and a learned state-independent log standard
 * deviation (greedy: the mean).
 *
 * @param env - The environment's spaces (discrete or box actions).
 * @param options - Networks, optimiser, discount, expectile, temperature and target averaging.
 * @returns The learner.
 * @concept offline-rl
 * @example
 * ```ts
 * import { iqlLearner } from 'mdml-engine/methods/rl/agents'
 * import { pendulum } from 'mdml-engine/methods/rl'
 * iqlLearner(pendulum.create()).config['expectile'] // => 0.7
 * ```
 */
export function iqlLearner<O, A>(env: Spaces, options: IqlOptions = {}): OfflineLearner<IqlState, O, A> {
  const sp = offlineSpaces(env)
  const hidden = options.hidden ?? [64, 64]
  const key = options.key ?? makeKey(0)
  const gamma = options.gamma ?? 0.99
  const tauE = options.expectile ?? 0.7
  const beta = options.temperature ?? 3
  const maxWeight = options.maxWeight ?? 100
  const polyak = options.targetTau ?? 0.005
  const opt = optimiserOf(options, 3e-4)
  const c1 = critic(sp, hidden, foldIn(key, 0))
  const c2 = critic(sp, hidden, foldIn(key, 1))
  const vNet = network([sp.obsSize, ...hidden, 1], foldIn(key, 2))
  const piNet = network([sp.obsSize, ...hidden, sp.actionSize], foldIn(key, 3))
  type P = { readonly q1: Tree; readonly q2: Tree; readonly v: Tree; readonly pi: Tree; readonly logStd?: Value }
  const params: Tree = {
    q1: c1.net.params,
    q2: c2.net.params,
    v: vNet.params,
    pi: piNet.params,
    ...(sp.discrete ? {} : { logStd: vectorTensor(new Array<number>(sp.actionSize).fill(0)) }),
  }
  const targetOf = (p: Tree): Tree => ({ q1: (p as P).q1, q2: (p as P).q2 })
  const qMin = (p: Tree, x: Value, a: Value): Value => minimum(c1.q((p as P).q1, x, a), c2.q((p as P).q2, x, a))
  const v = (p: Tree, x: Value): Value => column(vNet.apply((p as P).v, x))
  const policyOut = (p: Tree, x: Value): Value => {
    const u = piNet.apply((p as P).pi, x)
    return sp.discrete ? u : squashToBox(sp, u)
  }
  const logProb = (p: Tree, x: Value, b: OfflineBatch): Value => {
    const out = policyOut(p, x)
    if (sp.discrete) return sum(mul(logSoftmax(out, -1), b.actions), -1)
    return gaussianPolicy(out, (p as P).logStd as Value).logProb(b.actions)
  }
  const forward = (s: IqlState, o: O): Value => policyOut(s.params, rowsTensor([sp.features(o)], sp.obsSize))
  return {
    name: 'iql',
    config: { hidden: [...hidden], gamma, expectile: tauE, temperature: beta, maxWeight, targetTau: polyak } as Json,
    metrics: {
      valueLoss: { description: 'expectile loss of V', direction: 'lower' },
      criticLoss: { description: 'TD loss of the twin critics', direction: 'lower' },
      actorLoss: { description: 'advantage-weighted negative log-likelihood' },
      advantage: { description: 'mean advantage Q(s, a) - V(s) on the data' },
    },
    init: () => ({
      params,
      target: targetOf(params),
      optState: opt.init(params),
      updates: 0,
      valueLoss: 0,
      criticLoss: 0,
      actorLoss: 0,
      advantage: 0,
    }),
    update: (s, batch) => {
      const b = offlineBatch(sp, batch)
      // Everything the three objectives regress onto, computed outside the gradient.
      const qT = qMin(s.target, b.obs, b.actions)
      const vNow = toFlat(v(s.params, b.obs))
      const vNext = toFlat(v(s.params, b.next))
      const r = toFlat(b.rewards)
      const nd = toFlat(b.notDone)
      const y = vectorTensor(r.map((ri, i) => ri + gamma * (nd[i] ?? 0) * (vNext[i] ?? 0)))
      const adv = toFlat(qT).map((q, i) => q - (vNow[i] ?? 0))
      const w = vectorTensor(adv.map((a) => Math.min(maxWeight, Math.exp(beta * a))))
      const loss = (p: Tree): readonly [Value, Value] => {
        const valueLoss = expectileLoss(sub(qT, v(p, b.obs)), tauE)
        const criticLoss = add(mse(c1.q((p as P).q1, b.obs, b.actions), y), mse(c2.q((p as P).q2, b.obs, b.actions), y))
        const actorLoss = neg(mean(mul(w, logProb(p, b.obs, b))))
        return [add(add(valueLoss, criticLoss), actorLoss), stack([valueLoss, criticLoss, actorLoss])]
      }
      const step = gradientStepWithTerms(loss, s.params, opt, s.optState)
      const [valueLoss = 0, criticLoss = 0, actorLoss = 0] = step.terms
      let advantage = 0
      for (const a of adv) advantage += a / adv.length
      return {
        params: step.params,
        target: incrementalUpdate(targetOf(step.params), s.target, polyak),
        optState: step.optState,
        updates: s.updates + 1,
        valueLoss,
        criticLoss,
        actorLoss,
        advantage,
      }
    },
    stats: (s) => ({
      valueLoss: s.valueLoss,
      criticLoss: s.criticLoss,
      actorLoss: s.actorLoss,
      advantage: s.advantage,
    }),
    greedy: (s, o) => {
      const out = toFlat(forward(s, o))
      return (sp.discrete ? argmaxOf(out) : clipBox(sp, out)) as A
    },
    act: (s, o, k) => {
      const out = forward(s, o)
      if (sp.discrete) return toFlat(categoricalPolicy(out).sample(k))[0] as A
      const head = gaussianPolicy(out, (s.params as P).logStd as Value)
      return clipBox(sp, toFlat(head.sample(k))) as A
    },
  }
}

function clipBox(sp: OfflineSpaces, a: readonly number[]): number[] {
  return a.map((x, i) => Math.min(sp.high[i] ?? x, Math.max(sp.low[i] ?? x, x)))
}

/**
 * IQL trained on a dataset ({@link iqlLearner} run by {@link trainOffline}).
 *
 * @param env - The environment's spaces.
 * @param dataset - The dataset.
 * @param options - The learner's options, steps, batch size and run options.
 * @returns The trained state, agent and policy.
 * @concept offline-rl
 * @example
 * ```ts
 * import { iql } from 'mdml-engine/methods/rl/agents'
 * import { cartPole } from 'mdml-engine/methods/rl'
 * import { key } from 'mdml-engine/random'
 * const t = { observation: [0, 0, 0, 0], action: 1, reward: 1, nextObservation: [0, 0, 0, 0], terminated: true, truncated: false }
 * iql(cartPole.create(), [t], { hidden: [4], steps: 30, learningRate: 1e-2 }).policy([0, 0, 0, 0], key(0)) // => 1
 * ```
 */
export function iql<O, A>(
  env: Spaces,
  dataset: Dataset<O, A>,
  options: IqlOptions & OfflineOptions & Partial<Omit<RunOptions, 'maxSteps'>> = {},
): OfflineResult<IqlState, O, A> {
  return trainOffline(iqlLearner<O, A>(env, options), dataset, options)
}

// ------------------------------------------------------------------------------------------------------ TD3+BC

/** Options of {@link td3bc}. */
export interface Td3BcOptions extends OfflineNetOptions {
  readonly gamma?: number
  /** Polyak weight of the target networks (default 0.005). */
  readonly tau?: number
  /** Target policy smoothing: noise scale and clip, as fractions of the action range's half-width (0.2, 0.5). */
  readonly policyNoise?: number
  readonly noiseClip?: number
  /** Update the actor and the targets every this many critic updates (default 2). */
  readonly policyFrequency?: number
  /** The weight `alpha` of the Q term against behaviour cloning (default 2.5). */
  readonly alpha?: number
  /** Normalise observations by the dataset's mean and standard deviation (default true). */
  readonly normalize?: boolean
}

/** The state of a TD3+BC learner. */
export interface Td3BcState {
  readonly actor: Tree
  readonly critic: Tree
  readonly actorTarget: Tree
  readonly criticTarget: Tree
  readonly actorOpt: unknown
  readonly criticOpt: unknown
  readonly updates: number
  readonly criticLoss: number
  readonly actorLoss: number
  /** The normalised weight `lambda = alpha / mean |Q|` of the last actor update. */
  readonly lambda: number
  readonly qValue: number
  /** Observation normalisation (mean and standard deviation of the dataset's observations). */
  readonly obsMean: readonly number[]
  readonly obsStd: readonly number[]
}

/**
 * TD3+BC (Fujimoto and Gu 2021): TD3 (twin critics regressing `r + gamma (1 - done) min(Q1', Q2')(s', a')` with
 * target policy smoothing `a' = clip(pi'(s') + clip(noise))`, delayed actor and target updates, Polyak targets) whose
 * actor maximises `lambda Q1(s, pi(s)) - (pi(s) - a)^2` with `lambda = alpha / mean |Q1(s, pi(s))|`, so the policy
 * stays close to the data's actions; observations are normalised by the dataset's statistics. Box actions only.
 *
 * @param env - The environment's spaces (box actions).
 * @param options - Networks, optimiser, TD3's noise and delays, `alpha` and normalisation.
 * @returns The learner.
 * @concept offline-rl
 * @example
 * ```ts
 * import { td3bcLearner } from 'mdml-engine/methods/rl/agents'
 * import { pendulum } from 'mdml-engine/methods/rl'
 * td3bcLearner(pendulum.create()).config['alpha'] // => 2.5
 * ```
 */
export function td3bcLearner<O>(
  env: Spaces,
  options: Td3BcOptions = {},
): OfflineLearner<Td3BcState, O, readonly number[]> {
  const sp = offlineSpaces(env)
  if (sp.discrete) throw new MdmlError('value', 'td3bc: needs a box action space')
  const hidden = options.hidden ?? [64, 64]
  const key = options.key ?? makeKey(0)
  const gamma = options.gamma ?? 0.99
  const tau = options.tau ?? 0.005
  const policyNoise = options.policyNoise ?? 0.2
  const noiseClip = options.noiseClip ?? 0.5
  const frequency = options.policyFrequency ?? 2
  const alpha = options.alpha ?? 2.5
  const normalize = options.normalize ?? true
  const actorOpt = optimiserOf(options, 3e-4)
  const criticOpt = optimiserOf(options, 3e-4)
  const actorNet = network([sp.obsSize, ...hidden, sp.actionSize], foldIn(key, 0))
  const c1 = critic(sp, hidden, foldIn(key, 1))
  const c2 = critic(sp, hidden, foldIn(key, 2))
  type C = { readonly q1: Tree; readonly q2: Tree }
  const criticParams: Tree = { q1: c1.net.params, q2: c2.net.params }
  const pi = (p: Tree, x: Value): Value => squashToBox(sp, actorNet.apply(p, x))
  const m = sp.actionSize
  const scale = vectorTensor(sp.scale)
  const low = vectorTensor(sp.low)
  const high = vectorTensor(sp.high)
  const norm =
    (s: { readonly obsMean: readonly number[]; readonly obsStd: readonly number[] }) =>
    (x: number[]): number[] =>
      x.map((v, j) => (v - (s.obsMean[j] ?? 0)) / (s.obsStd[j] ?? 1))
  return {
    name: 'td3bc',
    config: { hidden: [...hidden], gamma, tau, policyNoise, noiseClip, policyFrequency: frequency, alpha, normalize },
    metrics: {
      criticLoss: { description: 'TD loss of the twin critics', direction: 'lower' },
      actorLoss: { description: '-lambda Q + behaviour-cloning loss of the last actor update', direction: 'lower' },
      lambda: { description: 'normalised Q weight alpha / mean |Q|' },
      qValue: { description: 'mean Q1 of the data actions' },
    },
    init: (_k, dataset: Dataset<O, readonly number[]>) => {
      const d = sp.obsSize
      const obsMean = new Array<number>(d).fill(0)
      const obsStd = new Array<number>(d).fill(1)
      if (normalize && dataset.length > 0) {
        const rows = dataset.map((t) => sp.features(t.observation))
        for (let j = 0; j < d; j++) {
          let s1 = 0
          for (const r of rows) s1 += r[j] ?? 0
          const mu = s1 / rows.length
          let s2 = 0
          for (const r of rows) s2 += ((r[j] ?? 0) - mu) ** 2
          obsMean[j] = mu
          obsStd[j] = Math.sqrt(s2 / rows.length) + 1e-3
        }
      }
      return {
        actor: actorNet.params,
        critic: criticParams,
        actorTarget: actorNet.params,
        criticTarget: criticParams,
        actorOpt: actorOpt.init(actorNet.params),
        criticOpt: criticOpt.init(criticParams),
        updates: 0,
        criticLoss: 0,
        actorLoss: 0,
        lambda: 0,
        qValue: 0,
        obsMean,
        obsStd,
      }
    },
    update: (s, batch: readonly Transition<O, readonly number[]>[], k) => {
      const b = offlineBatch(sp, batch, norm(s))
      // Target policy smoothing: clipped Gaussian noise on the target actor's action, clipped to the box.
      const noise = clip(
        mul(normal(foldIn(k, 0), [b.size, m]), mul(scale, policyNoise)),
        mul(scale, -noiseClip),
        mul(scale, noiseClip),
      )
      const aNext = clip(add(pi(s.actorTarget, b.next), noise), low, high)
      const ct = s.criticTarget as C
      const qNext = toFlat(minimum(c1.q(ct.q1, b.next, aNext), c2.q(ct.q2, b.next, aNext)))
      const r = toFlat(b.rewards)
      const nd = toFlat(b.notDone)
      const y = vectorTensor(r.map((ri, i) => ri + gamma * (nd[i] ?? 0) * (qNext[i] ?? 0)))
      const criticStep = gradientStepWithTerms(
        (p) => {
          const q1 = c1.q((p as C).q1, b.obs, b.actions)
          const l = add(mse(q1, y), mse(c2.q((p as C).q2, b.obs, b.actions), y))
          return [l, reshape(mean(q1), [1])]
        },
        s.critic,
        criticOpt,
        s.criticOpt,
      )
      let next: Td3BcState = {
        ...s,
        critic: criticStep.params,
        criticOpt: criticStep.optState,
        updates: s.updates + 1,
        criticLoss: criticStep.loss,
        qValue: criticStep.terms[0] ?? 0,
      }
      if (next.updates % frequency !== 0) return next
      const critic1 = (next.critic as C).q1
      const qNow = toFlat(c1.q(critic1, b.obs, pi(next.actor, b.obs)))
      let meanAbs = 0
      for (const q of qNow) meanAbs += Math.abs(q) / qNow.length
      const lambda = alpha / Math.max(meanAbs, 1e-8)
      const actorStep = gradientStepWithTerms(
        (p) => {
          const a = pi(p, b.obs)
          const l = add(mul(mean(c1.q(critic1, b.obs, a)), -lambda), mse(a, b.actions))
          return [l, reshape(l, [1])]
        },
        next.actor,
        actorOpt,
        next.actorOpt,
      )
      next = {
        ...next,
        actor: actorStep.params,
        actorOpt: actorStep.optState,
        actorLoss: actorStep.loss,
        lambda,
        actorTarget: incrementalUpdate(actorStep.params, next.actorTarget, tau),
        criticTarget: incrementalUpdate(next.critic, next.criticTarget, tau),
      }
      return next
    },
    stats: (s) => ({ criticLoss: s.criticLoss, actorLoss: s.actorLoss, lambda: s.lambda, qValue: s.qValue }),
    greedy: (s, o) => clipBox(sp, toFlat(pi(s.actor, rowsTensor([norm(s)(sp.features(o))], sp.obsSize)))),
  }
}

/**
 * TD3+BC trained on a dataset ({@link td3bcLearner} run by {@link trainOffline}).
 *
 * @param env - The environment's spaces.
 * @param dataset - The dataset.
 * @param options - The learner's options, steps, batch size and run options.
 * @returns The trained state, agent and policy.
 * @concept offline-rl
 * @example
 * ```ts
 * import { td3bc } from 'mdml-engine/methods/rl/agents'
 * import { pendulum } from 'mdml-engine/methods/rl'
 * const t = { observation: [1, 0, 0], action: [0.5], reward: 0, nextObservation: [1, 0, 0], terminated: false, truncated: false }
 * td3bc(pendulum.create(), [t], { hidden: [4], steps: 4 }).agent.name // => 'td3bc'
 * ```
 */
export function td3bc<O>(
  env: Spaces,
  dataset: Dataset<O, readonly number[]>,
  options: Td3BcOptions & OfflineOptions & Partial<Omit<RunOptions, 'maxSteps'>> = {},
): OfflineResult<Td3BcState, O, readonly number[]> {
  return trainOffline(td3bcLearner<O>(env, options), dataset, options)
}
