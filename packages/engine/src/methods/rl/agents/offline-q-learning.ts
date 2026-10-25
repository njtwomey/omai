/**
 * Discrete value-based offline RL, each method an `OfflineLearner` over a fixed dataset of transitions: offline DQN,
 * the uncorrected baseline that overestimates actions the data never took, conservative Q-learning (CQL, which pushes
 * down the values of unsupported actions) and discrete batch-constrained Q-learning (BCQ, which only considers actions
 * a learnt model of the behaviour policy finds likely).
 */

import { argmaxOf, MdmlError } from '../../../lib/base/index.ts'
import { crossEntropy, huber } from '../../../lib/losses/index.ts'
import { incrementalUpdate, type Tree } from '../../../lib/optim/index.ts'
import { foldIn, key as makeKey } from '../../../lib/random/index.ts'
import {
  add,
  argmax,
  greater,
  logsumexp,
  max,
  mean,
  mul,
  oneHot,
  softmax,
  square,
  stack,
  sub,
  sum,
  type Tensor,
  toFlat,
  type Value,
  where,
} from '../../../lib/tensor/index.ts'
import type { RunOptions } from '../../../lib/trace/index.ts'
import { gradientStepWithTerms, optimiserOf, rowsTensor, type Spaces, vectorTensor } from './_common.ts'
import { network, type OfflineBatch, offlineBatch, offlineSpaces } from './_offline.ts'

import {
  type Dataset,
  type OfflineLearner,
  type OfflineNetOptions,
  type OfflineOptions,
  type OfflineResult,
  trainOffline,
} from './offline.ts'

// ------------------------------------------------------------------------- discrete value-based offline RL

/** Options of the discrete offline Q-learners ({@link cql}, {@link offlineDqn}, {@link bcq}). */
export interface OfflineQOptions extends OfflineNetOptions {
  readonly gamma?: number
  /** Copy the online network to the target every this many updates (default 100); ignored when `tau < 1`. */
  readonly targetUpdateInterval?: number
  /** Polyak averaging weight of soft target updates (default 1: hard copies). */
  readonly tau?: number
  /** Huber threshold of the TD loss (default 1). */
  readonly huberDelta?: number
  /** Double Q-learning targets (default false). */
  readonly double?: boolean
}

/** Options of {@link cql}. */
export interface CqlOptions extends OfflineQOptions {
  /** The weight `alpha` of the conservative regulariser (default 1; 0 is offline DQN). */
  readonly alpha?: number
}

/** The state of an offline Q-learner. */
export interface OfflineQState {
  readonly params: Tree
  readonly target: Tree
  readonly optState: unknown
  readonly updates: number
  readonly loss: number
  readonly tdLoss: number
  /** CQL's regulariser `E[logsumexp Q(s, .) - Q(s, a)]`, or BCQ's imitation loss. */
  readonly regulariser: number
  readonly qValue: number
}

function qLearnerParts(env: Spaces, options: OfflineQOptions, name: string) {
  const sp = offlineSpaces(env)
  if (!sp.discrete) throw new MdmlError('value', `${name}: needs a discrete action space`)
  const hidden = options.hidden ?? [64, 64]
  const opt = optimiserOf(options, 1e-3)
  const gamma = options.gamma ?? 0.99
  const interval = options.targetUpdateInterval ?? 100
  const tau = options.tau ?? 1
  const syncTarget = (s: OfflineQState): OfflineQState =>
    tau < 1
      ? { ...s, target: incrementalUpdate(s.params, s.target, tau) }
      : s.updates % interval === 0
        ? { ...s, target: s.params }
        : s
  return { sp, hidden, opt, gamma, interval, tau, syncTarget }
}

const chosen = (q: Value, mask: Value): Value => sum(mul(q, mask), -1)

/**
 * Conservative Q-learning for discrete actions (Kumar et al. 2020, CQL(H)) on a DQN-style critic: the TD loss to
 * `r + gamma (1 - done) max_a Q_target(s', a)` plus `alpha E_D[logsumexp_a Q(s, a) - Q(s, a_D)]`, which pushes down
 * the values of actions the dataset does not support and up those it does, so the greedy policy stays within the
 * data. With `alpha = 0` it is plain offline DQN.
 *
 * @param env - The environment's spaces (discrete actions).
 * @param options - Network, optimiser, discount, target updates and `alpha`.
 * @returns The learner.
 * @concept offline-rl
 * @example
 * ```ts
 * import { cqlLearner } from 'mdml-engine/methods/rl/agents'
 * import { cartPole } from 'mdml-engine/methods/rl'
 * cqlLearner(cartPole.create(), { alpha: 4 }).config['alpha'] // => 4
 * ```
 */
export function cqlLearner<O>(env: Spaces, options: CqlOptions = {}): OfflineLearner<OfflineQState, O, number> {
  return offlineQLearner(env, options, options.alpha ?? 1, 'cql')
}

function offlineQLearner<O>(
  env: Spaces,
  options: OfflineQOptions,
  alpha: number,
  name: string,
): OfflineLearner<OfflineQState, O, number> {
  const { sp, hidden, opt, gamma, interval, tau, syncTarget } = qLearnerParts(env, options, name)
  const net = network([sp.obsSize, ...hidden, sp.actionSize], options.key ?? makeKey(0))
  const delta = options.huberDelta ?? 1
  const q = (p: Tree, o: O): number[] => toFlat(net.apply(p, rowsTensor([sp.features(o)], sp.obsSize)))
  return {
    name,
    config: { hidden: [...hidden], gamma, targetUpdateInterval: interval, tau, ...(name === 'cql' ? { alpha } : {}) },
    metrics: {
      loss: { description: 'total loss of the last minibatch', direction: 'lower' },
      tdLoss: { description: 'Huber TD loss', direction: 'lower' },
      ...(alpha > 0 ? { regulariser: { description: 'logsumexp Q(s, .) - Q(s, a) on the data' } } : {}),
      qValue: { description: 'mean Q value of the data actions' },
    },
    init: () => ({
      params: net.params,
      target: net.params,
      optState: opt.init(net.params),
      updates: 0,
      loss: 0,
      tdLoss: 0,
      regulariser: 0,
      qValue: 0,
    }),
    update: (s, batch) => {
      const b = offlineBatch(sp, batch)
      const y = qTargets(
        b,
        net.apply(s.target, b.next),
        options.double === true ? net.apply(s.params, b.next) : undefined,
        gamma,
        sp.actionSize,
      )
      const loss = (p: Tree): readonly [Value, Value] => {
        const all = net.apply(p, b.obs)
        const qa = chosen(all, b.actions)
        const td = huber(qa, y, { delta })
        const reg = alpha > 0 ? mean(sub(logsumexp(all, -1), qa)) : mul(td, 0)
        return [add(td, mul(reg, alpha)), stack([td, reg, mean(qa)])]
      }
      const step = gradientStepWithTerms(loss, s.params, opt, s.optState)
      const [tdLoss = 0, regulariser = 0, qValue = 0] = step.terms
      return syncTarget({
        ...s,
        params: step.params,
        optState: step.optState,
        updates: s.updates + 1,
        loss: step.loss,
        tdLoss,
        regulariser,
        qValue,
      })
    },
    stats: (s) => ({
      loss: s.loss,
      tdLoss: s.tdLoss,
      ...(alpha > 0 ? { regulariser: s.regulariser } : {}),
      qValue: s.qValue,
    }),
    greedy: (s, o) => argmaxOf(q(s.params, o)),
  }
}

// r + gamma (1 - done) Q_target(s', a*), with a* from the target (or, for double Q-learning, the online) values.
function qTargets(
  b: OfflineBatch,
  qTargetNext: Value,
  qOnlineNext: Value | undefined,
  gamma: number,
  nA: number,
): Tensor {
  const best = oneHot(argmax(qOnlineNext ?? qTargetNext, -1), nA)
  const next = toFlat(chosen(qTargetNext, best))
  const r = toFlat(b.rewards)
  const nd = toFlat(b.notDone)
  return vectorTensor(r.map((ri, i) => ri + gamma * (nd[i] ?? 0) * (next[i] ?? 0)))
}

/**
 * CQL trained on a dataset ({@link cqlLearner} run by {@link trainOffline}).
 *
 * @param env - The environment's spaces.
 * @param dataset - The dataset.
 * @param options - The learner's options, steps, batch size and run options.
 * @returns The trained state, agent and policy.
 * @concept offline-rl
 * @example
 * ```ts
 * import { cql } from 'mdml-engine/methods/rl/agents'
 * import { cartPole } from 'mdml-engine/methods/rl'
 * import { key } from 'mdml-engine/random'
 * const t = { observation: [0, 0, 0, 0], action: 1, reward: 1, nextObservation: [0, 0, 0, 0], terminated: true, truncated: false }
 * cql(cartPole.create(), [t], { hidden: [4], steps: 50 }).policy([0, 0, 0, 0], key(0)) // => 1
 * ```
 */
export function cql<O>(
  env: Spaces,
  dataset: Dataset<O, number>,
  options: CqlOptions & OfflineOptions & Partial<Omit<RunOptions, 'maxSteps'>> = {},
): OfflineResult<OfflineQState, O, number> {
  return trainOffline(cqlLearner<O>(env, options), dataset, options)
}

/**
 * Offline DQN: Q-learning on a fixed dataset with no correction for actions the data does not cover (the baseline
 * whose extrapolation error BCQ and CQL address). The learner is CQL with `alpha = 0`.
 *
 * @param env - The environment's spaces.
 * @param dataset - The dataset.
 * @param options - Network, optimiser, discount, target updates, steps, batch size and run options.
 * @returns The trained state, agent and policy.
 * @concept offline-rl
 * @example
 * ```ts
 * import { offlineDqn } from 'mdml-engine/methods/rl/agents'
 * import { cartPole } from 'mdml-engine/methods/rl'
 * import { key } from 'mdml-engine/random'
 * const t = { observation: [0, 0, 0, 0], action: 1, reward: 1, nextObservation: [0, 0, 0, 0], terminated: true, truncated: false }
 * offlineDqn(cartPole.create(), [t], { hidden: [4], steps: 5 }).agent.name // => 'offlineDqn'
 * ```
 */
export function offlineDqn<O>(
  env: Spaces,
  dataset: Dataset<O, number>,
  options: OfflineQOptions & OfflineOptions & Partial<Omit<RunOptions, 'maxSteps'>> = {},
): OfflineResult<OfflineQState, O, number> {
  return trainOffline(offlineQLearner<O>(env, options, 0, 'offlineDqn'), dataset, options)
}

// --------------------------------------------------------------------------------------------- discrete BCQ

/** Options of {@link bcq}. */
export interface BcqOptions extends OfflineQOptions {
  /** Actions whose relative behaviour probability `G(a|s) / max_a G(a|s)` is below this are excluded (default 0.3). */
  readonly threshold?: number
}

/**
 * Discrete batch-constrained Q-learning (Fujimoto, Conti, Ghavamzadeh and Pineau 2019): a Q network and a generative
 * model `G(a|s)` of the behaviour policy (a classifier, trained by cross-entropy with a small penalty on its logits)
 * learned together; both the target `r + gamma Q_target(s', a')` and the policy maximise Q only over actions with
 * `G(a|s) / max_b G(b|s) > threshold`, so the agent never relies on actions the data does not support. The threshold
 * interpolates between Q-learning (0) and behaviour cloning (1).
 *
 * @param env - The environment's spaces (discrete actions).
 * @param options - Networks, optimiser, discount, target updates and the threshold.
 * @returns The learner.
 * @concept offline-rl
 * @example
 * ```ts
 * import { bcqLearner } from 'mdml-engine/methods/rl/agents'
 * import { cartPole } from 'mdml-engine/methods/rl'
 * bcqLearner(cartPole.create()).config['threshold'] // => 0.3
 * ```
 */
export function bcqLearner<O>(env: Spaces, options: BcqOptions = {}): OfflineLearner<OfflineQState, O, number> {
  const { sp, hidden, opt, gamma, interval, tau, syncTarget } = qLearnerParts(env, options, 'bcq')
  const threshold = options.threshold ?? 0.3
  const key = options.key ?? makeKey(0)
  const qNet = network([sp.obsSize, ...hidden, sp.actionSize], foldIn(key, 0))
  const gNet = network([sp.obsSize, ...hidden, sp.actionSize], foldIn(key, 1))
  const delta = options.huberDelta ?? 1
  const params: Tree = { q: qNet.params, g: gNet.params }
  type P = { readonly q: Tree; readonly g: Tree }
  // Q with unsupported actions pushed to -1e8 (the paper's mask), for the target's and the policy's argmax.
  const masked = (p: Tree, x: Value): Value => {
    const probs = softmax(gNet.apply((p as P).g, x), -1)
    const ok = greater(probs, mul(max(probs, -1, true), threshold))
    return where(ok, qNet.apply((p as P).q, x), -1e8)
  }
  return {
    name: 'bcq',
    config: { hidden: [...hidden], gamma, targetUpdateInterval: interval, tau, threshold },
    metrics: {
      loss: { description: 'total loss of the last minibatch', direction: 'lower' },
      tdLoss: { description: 'Huber TD loss', direction: 'lower' },
      regulariser: { description: 'imitation (cross-entropy) loss of the generative model', direction: 'lower' },
      qValue: { description: 'mean Q value of the data actions' },
    },
    init: () => ({
      params,
      target: params,
      optState: opt.init(params),
      updates: 0,
      loss: 0,
      tdLoss: 0,
      regulariser: 0,
      qValue: 0,
    }),
    update: (s, batch) => {
      const b = offlineBatch(sp, batch)
      const best = oneHot(argmax(masked(s.params, b.next), -1), sp.actionSize)
      const next = toFlat(chosen(qNet.apply((s.target as P).q, b.next), best))
      const r = toFlat(b.rewards)
      const nd = toFlat(b.notDone)
      const y = vectorTensor(r.map((ri, i) => ri + gamma * (nd[i] ?? 0) * (next[i] ?? 0)))
      const loss = (p: Tree): readonly [Value, Value] => {
        const qa = chosen(qNet.apply((p as P).q, b.obs), b.actions)
        const logits = gNet.apply((p as P).g, b.obs)
        const td = huber(qa, y, { delta })
        const imitation = crossEntropy(logits, b.labels)
        const total = add(add(td, imitation), mul(mean(square(logits)), 1e-2))
        return [total, stack([td, imitation, mean(qa)])]
      }
      const step = gradientStepWithTerms(loss, s.params, opt, s.optState)
      const [tdLoss = 0, regulariser = 0, qValue = 0] = step.terms
      return syncTarget({
        ...s,
        params: step.params,
        optState: step.optState,
        updates: s.updates + 1,
        loss: step.loss,
        tdLoss,
        regulariser,
        qValue,
      })
    },
    stats: (s) => ({ loss: s.loss, tdLoss: s.tdLoss, regulariser: s.regulariser, qValue: s.qValue }),
    greedy: (s, o) => argmaxOf(toFlat(masked(s.params, rowsTensor([sp.features(o)], sp.obsSize)))),
  }
}

/**
 * Discrete BCQ trained on a dataset ({@link bcqLearner} run by {@link trainOffline}).
 *
 * @param env - The environment's spaces.
 * @param dataset - The dataset.
 * @param options - The learner's options, steps, batch size and run options.
 * @returns The trained state, agent and policy.
 * @concept offline-rl
 * @example
 * ```ts
 * import { bcq } from 'mdml-engine/methods/rl/agents'
 * import { cartPole } from 'mdml-engine/methods/rl'
 * import { key } from 'mdml-engine/random'
 * const t = { observation: [0, 0, 0, 0], action: 1, reward: 0, nextObservation: [0, 0, 0, 0], terminated: true, truncated: false }
 * bcq(cartPole.create(), [t], { hidden: [4], steps: 50, learningRate: 0.05 }).policy([0, 0, 0, 0], key(0)) // => 1
 * ```
 */
export function bcq<O>(
  env: Spaces,
  dataset: Dataset<O, number>,
  options: BcqOptions & OfflineOptions & Partial<Omit<RunOptions, 'maxSteps'>> = {},
): OfflineResult<OfflineQState, O, number> {
  return trainOffline(bcqLearner<O>(env, options), dataset, options)
}
