/**
 * Machinery shared by the on-policy agents (policy-gradient.ts, safe.ts, imitation.ts), private to the agents:
 * actor-critic networks over discrete or box action spaces, KL divergences between their policies, rollout batches with
 * generalised advantage estimation, and the epochs of minibatch updates that PPO and its constrained variant share.
 */

import { valueAndGrad } from '../autodiff/index.ts'
import { meanOf, OmaiError, varianceOf } from '../base/index.ts'
import { explainedVarianceScore } from '../metrics/index.ts'
import type { ActivationName, Params } from '../nn/index.ts'
import type { GradientTransformation, ParamTree } from '../optim/index.ts'
import { foldIn, type Key, key as makeKey } from '../random/index.ts'
import {
  categoricalPolicy,
  gaussianPolicy,
  minibatchIndices,
  type PolicyHead,
  type RolloutStep,
  rolloutAdvantages,
  type Transition,
} from '../rl/index.ts'
import {
  add,
  broadcastTo,
  clip,
  exp,
  fromStorage,
  item,
  logSoftmax,
  mean,
  mul,
  primalOf,
  sliceAxis,
  square,
  sub,
  sum,
  type Tensor,
  take,
  toFlat,
  type Value,
} from '../tensor/index.ts'
import {
  actionBox,
  actionCount,
  applyGradient,
  observationSize,
  rowsTensor,
  type Spaces,
  vectorTensor,
} from './_common.ts'
import { mlpNet } from './_networks.ts'

/** An action of a policy-gradient agent: an index for discrete spaces, a vector for box spaces. */
export type PolicyAction = number | readonly number[]

/** Network options shared by the policy-gradient agents. */
export interface ActorCriticOptions {
  /** Hidden widths of the policy network (default `[64, 64]`, stable-baselines3's MlpPolicy). */
  readonly hidden?: readonly number[]
  /** Hidden widths of the value network (default `hidden`). */
  readonly valueHidden?: readonly number[]
  /** The hidden activation (default `'tanh'`, as stable-baselines3's on-policy agents). */
  readonly activation?: ActivationName
  /** Initial log standard deviation of Gaussian policies (default 0). */
  readonly logStdInit?: number
  /** The policy network outputs the log standard deviation too (default false: one learned vector). */
  readonly stateDependentStd?: boolean
  /** Squash Gaussian actions through tanh into the action bounds (default false: actions are unbounded samples). */
  readonly squash?: boolean
  /** The initialisation key (default key(0)). */
  readonly key?: Key
}

/** The parameters of an actor-critic: the policy (network and log std), the value network and an optional cost one. */
export interface ActorCriticParams {
  readonly policy: Params
  readonly value: Params
  readonly cost?: Params
}

/** A batch of policy distributions by their parameters. */
export type PolicyDist =
  | { readonly kind: 'categorical'; readonly logits: Value }
  | { readonly kind: 'gaussian'; readonly mean: Value; readonly logStd: Value }

/** An actor-critic network pair over an environment's spaces. */
export interface ActorCritic {
  readonly params: ActorCriticParams
  readonly discrete: boolean
  readonly observationSize: number
  /** The number of actions (discrete) or the action dimension (box). */
  readonly actionSize: number
  readonly low: readonly number[]
  readonly high: readonly number[]
  dist(policy: Params, observations: Value): PolicyDist
  head(dist: PolicyDist): PolicyHead
  /** State values `[B]` of a value network. */
  value(net: Params, observations: Value): Value
  /** Actions as a tensor (`[B]` indices or `[B, d]` vectors). */
  actions(actions: readonly PolicyAction[]): Tensor
  sample(policy: Params, observation: readonly number[], key: Key): PolicyAction
  greedy(policy: Params, observation: readonly number[]): PolicyAction
}

const LOG_STD_MIN = -20
const LOG_STD_MAX = 2

/** Build the actor-critic of an environment (`cost` adds a second value network for constraint costs). */
export function actorCritic(
  env: Spaces,
  options: ActorCriticOptions,
  name: string,
  extra: { readonly cost?: boolean } = {},
): ActorCritic {
  const d = observationSize(env.observationSpace)
  const discrete = env.actionSpace.kind === 'discrete'
  const box = discrete ? undefined : actionBox(env.actionSpace, name)
  const k = discrete ? actionCount(env.actionSpace, name) : (box?.size ?? 1)
  const hidden = options.hidden ?? [64, 64]
  const valueHidden = options.valueHidden ?? hidden
  const act = options.activation ?? 'tanh'
  const key = options.key ?? makeKey(0)
  const stateStd = !discrete && options.stateDependentStd === true
  const squash = !discrete && options.squash === true
  const low = box?.low ?? []
  const high = box?.high ?? []
  const policyNet = mlpNet([d, ...hidden, stateStd ? 2 * k : k], act, foldIn(key, 0), 0.01)
  const valueNet = mlpNet([d, ...valueHidden, 1], act, foldIn(key, 1))
  const costNet = mlpNet([d, ...valueHidden, 1], act, foldIn(key, 2))
  const logStd0 = fromStorage(new Float64Array(k).fill(options.logStdInit ?? 0), [k], 'float64')
  const policyParams: Params =
    discrete || stateStd ? { net: policyNet.params } : { net: policyNet.params, logStd: logStd0 }
  const dist = (p: Params, x: Value): PolicyDist => {
    const out = policyNet.apply(p['net'] as Params, x)
    if (discrete) return { kind: 'categorical', logits: out }
    if (stateStd) {
      const ls = clip(sliceAxis(out, 1, k, 2 * k), LOG_STD_MIN, LOG_STD_MAX)
      return { kind: 'gaussian', mean: sliceAxis(out, 1, 0, k), logStd: ls }
    }
    return { kind: 'gaussian', mean: out, logStd: broadcastTo(p['logStd'] as Value, out.shape) }
  }
  const head = (q: PolicyDist): PolicyHead =>
    q.kind === 'categorical'
      ? categoricalPolicy(q.logits)
      : gaussianPolicy(q.mean, q.logStd, squash ? { squash: true, low: [...low], high: [...high] } : {})
  const one = (o: readonly number[]): Tensor => rowsTensor([o], d)
  const toAction = (v: Value): PolicyAction => {
    const flat = toFlat(v)
    return discrete ? (flat[0] ?? 0) : flat
  }
  return {
    params: {
      policy: policyParams,
      value: valueNet.params,
      ...(extra.cost === true ? { cost: costNet.params } : {}),
    },
    discrete,
    observationSize: d,
    actionSize: k,
    low,
    high,
    dist,
    head,
    value: (net, x) => sum(valueNet.apply(net, x), -1),
    actions: (actions) =>
      discrete ? vectorTensor(actions as readonly number[]) : rowsTensor(actions as readonly (readonly number[])[], k),
    sample: (p, o, key2) => toAction(head(dist(p, one(o))).sample(key2)),
    greedy: (p, o) => {
      const a = toAction(head(dist(p, one(o))).mode())
      if (typeof a === 'number') return a
      return a.map((x, i) => Math.min(high[i] ?? Infinity, Math.max(low[i] ?? -Infinity, x)))
    },
  }
}

/**
 * KL(old || new) per batch row between two policy distributions of the same kind, in closed form (categorical, or
 * diagonal Gaussian summed over action dimensions; tanh squashing does not change the KL).
 */
export function policyKl(old: PolicyDist, next: PolicyDist): Value {
  if (old.kind === 'categorical' && next.kind === 'categorical') {
    const lp = logSoftmax(old.logits, -1)
    const lq = logSoftmax(next.logits, -1)
    return sum(mul(exp(lp), sub(lp, lq)), -1)
  }
  if (old.kind === 'gaussian' && next.kind === 'gaussian') {
    // log(s_new / s_old) + (s_old^2 + (m_old - m_new)^2) / (2 s_new^2) - 1/2, per dimension.
    const varOld = exp(mul(old.logStd, 2))
    const quad = mul(add(varOld, square(sub(old.mean, next.mean))), mul(0.5, exp(mul(next.logStd, -2))))
    return sum(sub(add(sub(next.logStd, old.logStd), quad), 0.5), -1)
  }
  throw new OmaiError('value', 'policyKl: distributions of different kinds')
}

/** The rows of a batch of distributions. */
export function distRows(q: PolicyDist, rows: Tensor): PolicyDist {
  return q.kind === 'categorical'
    ? { kind: 'categorical', logits: take(q.logits, rows, 0) }
    : { kind: 'gaussian', mean: take(q.mean, rows, 0), logStd: take(q.logStd, rows, 0) }
}

/** The concrete (untraced) version of a distribution. */
function concrete(q: PolicyDist): PolicyDist {
  return q.kind === 'categorical'
    ? { kind: 'categorical', logits: primalOf(q.logits) }
    : { kind: 'gaussian', mean: primalOf(q.mean), logStd: primalOf(q.logStd) }
}

/** Values, advantages and returns of one signal (reward or cost) over a rollout. */
export interface SignalEstimates {
  readonly values: readonly number[]
  readonly advantages: readonly number[]
  readonly returns: readonly number[]
}

/** A rollout ready for updates: inputs, the behaviour policy's log-probabilities and distributions, and estimates. */
export interface RolloutBatch {
  readonly observations: readonly (readonly number[])[]
  readonly actions: readonly PolicyAction[]
  readonly logProbs: readonly number[]
  readonly oldDist: PolicyDist
  readonly reward: SignalEstimates
  readonly cost?: SignalEstimates
}

function estimates(
  pending: readonly Transition<readonly number[], PolicyAction>[],
  signal: readonly number[],
  allValues: readonly number[],
  boot: ReadonlyMap<number, number>,
  logProbs: readonly number[],
  gamma: number,
  lambda: number,
): SignalEstimates {
  const n = pending.length
  const values = allValues.slice(0, n)
  const steps: RolloutStep<readonly number[], PolicyAction>[] = pending.map((t, i) => {
    const b = boot.get(i)
    return {
      observation: t.observation,
      action: t.action,
      reward: signal[i] ?? 0,
      value: values[i] ?? 0,
      logProb: logProbs[i] ?? 0,
      terminated: t.terminated,
      truncated: t.truncated,
      ...(b === undefined ? {} : { bootstrapValue: allValues[b] ?? 0 }),
    }
  })
  const lastIndex = boot.get(n)
  const last = lastIndex === undefined ? 0 : (allValues[lastIndex] ?? 0)
  const { advantages, returns } = rolloutAdvantages({ steps }, last, gamma, lambda)
  return { values, advantages, returns }
}

/**
 * Turn the pending transitions of a rollout into a batch in one forward pass of each network: values of every
 * observation, of the final observations of truncated steps (their bootstrap) and of the observation after the last
 * step, the behaviour policy's log-probabilities, and GAE advantages and returns (with `cost`, for the cost too).
 */
export function prepareRollout(
  net: ActorCritic,
  params: ActorCriticParams,
  pending: readonly Transition<readonly number[], PolicyAction>[],
  gamma: number,
  lambda: number,
  cost?: (t: Transition<readonly number[], PolicyAction>) => number,
): RolloutBatch {
  const n = pending.length
  const d = net.observationSize
  const observations = pending.map((t) => t.observation)
  const rows: (readonly number[])[] = [...observations]
  // boot maps a step index (or n, for the step after the rollout) to the row holding the observation to bootstrap from.
  const boot = new Map<number, number>()
  for (const [i, t] of pending.entries()) {
    if (t.truncated && !t.terminated) {
      boot.set(i, rows.length)
      rows.push(t.nextObservation)
    }
  }
  const last = pending[n - 1]
  if (last !== undefined && !(last.terminated || last.truncated)) {
    boot.set(n, rows.length)
    rows.push(last.nextObservation)
  }
  const all = rowsTensor(rows, d)
  const obs = rowsTensor(observations, d)
  const actions = pending.map((t) => t.action)
  const oldDist = concrete(net.dist(params.policy, obs))
  const logProbs = toFlat(net.head(oldDist).logProb(net.actions(actions)))
  const values = toFlat(net.value(params.value, all))
  const reward = estimates(
    pending,
    pending.map((t) => t.reward),
    values,
    boot,
    logProbs,
    gamma,
    lambda,
  )
  if (cost === undefined || params.cost === undefined) return { observations, actions, logProbs, oldDist, reward }
  const costValues = toFlat(net.value(params.cost, all))
  return {
    observations,
    actions,
    logProbs,
    oldDist,
    reward,
    cost: estimates(pending, pending.map(cost), costValues, boot, logProbs, gamma, lambda),
  }
}

/** Standardise to mean 0 and (sample) standard deviation 1, as stable-baselines3 normalises advantages. */
export function standardize(xs: readonly number[]): number[] {
  if (xs.length < 2) return [...xs]
  const m = meanOf(xs)
  const s = Math.sqrt(varianceOf(xs, 1))
  return xs.map((x) => (x - m) / (s + 1e-8))
}

/** The explained variance of value predictions for their targets (stable-baselines3's diagnostic). */
export function explainedVariance(values: readonly number[], returns: readonly number[]): number {
  return explainedVarianceScore(returns, values).estimate.value
}

/** The inputs of a policy objective on one minibatch. */
export interface ObjectiveInputs {
  readonly logProb: Value
  readonly oldLogProb: Tensor
  readonly advantages: Tensor
  /** KL(old || new) per row (computed on demand). */
  kl(): Value
}

/** A value head trained alongside the policy: its parameter slot, targets and the values at collection time. */
export interface ValueHead {
  readonly slot: 'value' | 'cost'
  readonly returns: readonly number[]
  readonly oldValues: readonly number[]
}

/** The configuration of {@link ppoEpochs}. */
export interface PpoEpochConfig {
  readonly epochs: number
  readonly batchSize: number
  /** The surrogate objective to maximise on a minibatch. */
  readonly objective: (inputs: ObjectiveInputs) => Value
  readonly clipRange: number
  readonly clipRangeVf?: number
  readonly entCoef: number
  readonly vfCoef: number
  readonly normalizeAdvantage: boolean
  readonly targetKl?: number
  /** Learning-rate factor of this update (annealing). */
  readonly scale: number
}

/** What a run of PPO epochs did. */
export interface PpoEpochResult {
  readonly params: ActorCriticParams
  readonly optState: unknown
  readonly stats: Readonly<Record<string, number>>
}

const pick = <T>(xs: readonly T[], idx: readonly number[]): T[] => idx.map((i) => xs[i] as T)

/**
 * Epochs of minibatch updates on a rollout (stable-baselines3's PPO.train): per minibatch, the surrogate objective,
 * the (optionally clipped) value losses of every value head, the entropy bonus and one optimiser step; with a target
 * KL, training stops as soon as a minibatch's approximate KL exceeds 1.5 times the target (before that step).
 */
export function ppoEpochs(
  net: ActorCritic,
  params: ActorCriticParams,
  optState: unknown,
  opt: GradientTransformation,
  batch: RolloutBatch,
  advantages: readonly number[],
  heads: readonly ValueHead[],
  config: PpoEpochConfig,
  key: Key,
): PpoEpochResult {
  const n = batch.observations.length
  const d = net.observationSize
  let p = params
  let state = optState
  const sums: Record<string, number> = {}
  let count = 0
  let stopped = false
  for (let epoch = 0; epoch < config.epochs && !stopped; epoch++) {
    for (const idx of minibatchIndices(n, config.batchSize, foldIn(key, epoch))) {
      const obs = rowsTensor(pick(batch.observations, idx), d)
      const actions = net.actions(pick(batch.actions, idx))
      const oldLogProb = vectorTensor(pick(batch.logProbs, idx))
      const adv = pick(advantages, idx)
      const advT = vectorTensor(config.normalizeAdvantage ? standardize(adv) : adv)
      const oldDist = distRows(batch.oldDist, vectorTensor(idx))
      const targets = heads.map((h) => ({
        slot: h.slot,
        returns: vectorTensor(pick(h.returns, idx)),
        old: vectorTensor(pick(h.oldValues, idx)),
      }))
      const aux: Record<string, number> = {}
      const loss = (tree: ParamTree): Value => {
        const q = tree as ActorCriticParams
        const dist = net.dist(q.policy, obs)
        const head = net.head(dist)
        const logProb = head.logProb(actions)
        const objective = config.objective({
          logProb,
          oldLogProb,
          advantages: advT,
          kl: () => policyKl(oldDist, dist),
        })
        const entropy = mean(head.entropy())
        let total = sub(mul(config.entCoef, neg1(entropy)), objective)
        for (const t of targets) {
          const net2 = q[t.slot]
          if (net2 === undefined) continue
          let v = net.value(net2, obs)
          if (config.clipRangeVf !== undefined)
            v = add(t.old, clip(sub(v, t.old), -config.clipRangeVf, config.clipRangeVf))
          const vl = mean(square(sub(t.returns, v)))
          aux[t.slot === 'value' ? 'valueLoss' : 'costValueLoss'] = item(primalOf(vl))
          total = add(total, mul(config.vfCoef, vl))
        }
        const logRatio = toFlat(primalOf(sub(logProb, oldLogProb)))
        let kl = 0
        let clipped = 0
        for (const r of logRatio) {
          kl += (Math.exp(r) - 1 - r) / logRatio.length
          if (Math.abs(Math.exp(r) - 1) > config.clipRange) clipped += 1 / logRatio.length
        }
        aux['policyLoss'] = -item(primalOf(objective))
        aux['entropy'] = item(primalOf(entropy))
        aux['approxKl'] = kl
        aux['clipFraction'] = clipped
        return total
      }
      const { grad } = valueAndGrad(loss)(p)
      if (config.targetKl !== undefined && (aux['approxKl'] ?? 0) > 1.5 * config.targetKl) {
        stopped = true
        break
      }
      const step = applyGradient(grad, p, opt, state, config.scale)
      p = step.params as ActorCriticParams
      state = step.optState
      for (const [k, v] of Object.entries(aux)) sums[k] = (sums[k] ?? 0) + v
      count += 1
    }
  }
  const stats: Record<string, number> = {}
  for (const [k, v] of Object.entries(sums)) stats[k] = v / Math.max(1, count)
  stats['explainedVariance'] = explainedVariance(batch.reward.values, batch.reward.returns)
  stats['earlyStop'] = stopped ? 1 : 0
  return { params: p, optState: state, stats }
}

const neg1 = (x: Value): Value => mul(x, -1)
