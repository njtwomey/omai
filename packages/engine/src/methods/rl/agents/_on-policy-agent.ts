// Private plumbing of the policy-gradient agents (policy-gradient.ts): their state, metrics, optimiser clipping and
// the agent that collects transitions and updates when a rollout or episode is complete.
import type { Json } from '../../../lib/base/index.ts'
import type { Params } from '../../../lib/nn/index.ts'
import { chain, clipByGlobalNorm, type GradientTransformation, type Tree } from '../../../lib/optim/index.ts'
import type { Key } from '../../../lib/random/index.ts'
import type { Agent, Transition } from '../../../lib/rl/index.ts'
import { mean, square, sub, type Value } from '../../../lib/tensor/index.ts'
import type { MetricChannel } from '../../../lib/trace/index.ts'
import type { ActorCritic, ActorCriticOptions, ActorCriticParams, PolicyAction } from './_on-policy.ts'

/** The state of the policy-gradient agents. */
export interface PolicyGradientState {
  readonly params: ActorCriticParams
  /** Optimiser state of the policy (or of the whole actor-critic, for A2C and PPO). */
  readonly optState: unknown
  /** Optimiser state of the separate value network (REINFORCE, TRPO). */
  readonly valueOptState: unknown
  /** Transitions since the last update. */
  readonly pending: readonly Transition<readonly number[], PolicyAction>[]
  readonly steps: number
  readonly updates: number
  readonly episodes: number
  /** REINFORCE's running-average baseline. */
  readonly baseline: number
  /** The penalty coefficient of KL-penalised PPO. */
  readonly klCoef: number
  readonly stats: Readonly<Record<string, number>>
}

export const METRICS: Readonly<Record<string, MetricChannel>> = {
  policyLoss: { description: 'policy loss (negative surrogate objective) of the last update', direction: 'lower' },
  valueLoss: { description: 'value regression loss of the last update', direction: 'lower' },
  entropy: { description: 'mean policy entropy on the last rollout' },
  approxKl: { description: 'KL divergence between the behaviour and the updated policy' },
  explainedVariance: { description: 'explained variance of the value predictions', direction: 'higher' },
}

/** Options shared by the policy-gradient agents. */
export interface PolicyGradientOptions extends ActorCriticOptions {
  readonly learningRate?: number
  /** An optimiser replacing the default one built from `learningRate`. */
  readonly optimizer?: GradientTransformation
  readonly gamma?: number
  /** The entropy bonus coefficient (default 0). */
  readonly entCoef?: number
  /** Clip gradients to this global norm (A2C and PPO default 0.5). */
  readonly maxGradNorm?: number
}

export function initialState(
  net: ActorCritic,
  optState: unknown,
  valueOptState: unknown,
  klCoef = 0,
): PolicyGradientState {
  return {
    params: net.params,
    optState,
    valueOptState,
    pending: [],
    steps: 0,
    updates: 0,
    episodes: 0,
    baseline: 0,
    klCoef,
    stats: {},
  }
}

export function withClip(opt: GradientTransformation, maxGradNorm: number | undefined): GradientTransformation {
  return maxGradNorm === undefined ? opt : chain(clipByGlobalNorm(maxGradNorm), opt)
}

/** An agent that collects transitions and calls `update` when `ready` says so (after appending the transition). */
export function onPolicyAgent<A>(
  name: string,
  config: Json,
  metrics: Readonly<Record<string, MetricChannel>>,
  net: ActorCritic,
  init: () => PolicyGradientState,
  ready: (s: PolicyGradientState, t: Transition<readonly number[], PolicyAction>) => boolean,
  update: (s: PolicyGradientState, key: Key) => PolicyGradientState,
): Agent<PolicyGradientState, readonly number[], A> {
  return {
    name,
    config,
    metrics,
    init,
    act: (s, obs, k) => net.sample(s.params.policy, obs, k) as A,
    greedy: (s, obs) => net.greedy(s.params.policy, obs) as A,
    learn: (s, t, k) => {
      const tr = t as Transition<readonly number[], PolicyAction>
      const next: PolicyGradientState = {
        ...s,
        pending: [...s.pending, tr],
        steps: s.steps + 1,
        episodes: s.episodes + (t.terminated || t.truncated ? 1 : 0),
      }
      if (!ready(next, tr)) return next
      return { ...update(next, k), pending: [], updates: next.updates + 1 }
    },
    stats: (s) => s.stats,
  }
}

export const valueLossOf =
  (net: ActorCritic, obs: Value, targets: Value) =>
  (p: Tree): Value =>
    mean(square(sub(targets, net.value(p as Params, obs))))
