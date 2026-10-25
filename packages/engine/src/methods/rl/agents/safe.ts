/**
 * Safe reinforcement learning: constrained MDPs, where an agent maximises reward while keeping an expected constraint
 * cost under a limit, and shielding, which keeps an agent from taking unsafe actions. A wrapper that adds a cost to
 * every step of an environment (the Safety Gym convention), Lagrangian PPO, which learns a multiplier that prices the
 * cost, and shields built from a safety predicate or a model lookahead, applied before the agent chooses or after it
 * proposes.
 */

import { argminOf, type Json, meanOf, minOf, WhiteboxError } from '../../../lib/base/index.ts'
import { adam, chain, clipByGlobalNorm } from '../../../lib/optim/index.ts'
import type { Key } from '../../../lib/random/index.ts'
import type { Agent, Env, EnvModel, Step, Transition } from '../../../lib/rl/index.ts'
import type { MetricChannel } from '../../../lib/trace/index.ts'
import type { Spaces } from './_common.ts'
import {
  actorCritic,
  type ObjectiveInputs,
  type PolicyAction,
  ppoEpochs,
  prepareRollout,
  standardize,
} from './_on-policy.ts'
import { type PolicyGradientState, type PpoOptions, ppoClipObjective } from './policy-gradient.ts'

// ------------------------------------------------------------------------------------------------ constrained MDPs

/** A cost of one environment step: the state it left, the action and the step's result. */
export type CostFunction<S, O, A> = (state: S, action: A, step: Step<S, O>) => number

/**
 * Add a constraint cost to every step of an environment, as `info.cost` (the Safety Gym convention), turning it into a
 * constrained MDP; any other `info` fields are kept.
 *
 * @param env - The environment.
 * @param cost - The cost of a step.
 * @returns The environment with `info.cost`.
 * @concept safe-reinforcement-learning
 * @example
 * ```ts
 * import { costWrapper } from 'whitebox-engine/methods/rl/agents'
 * import { cartPole } from 'whitebox-engine/methods/rl'
 * import { key } from 'whitebox-engine/random'
 * const env = costWrapper(cartPole.create(), (_s, a) => (a === 1 ? 0.5 : 0))
 * env.step(env.reset(key(0)).state, 1, key(1)).info.cost // => 0.5
 * ```
 */
export function costWrapper<S, O, A>(env: Env<S, O, A>, cost: CostFunction<S, O, A>): Env<S, O, A> {
  return {
    ...env,
    step: (s, a, k) => {
      const r = env.step(s, a, k)
      return { ...r, info: { ...r.info, cost: cost(s, a, r) } }
    },
    wrapped: env as Env<unknown, unknown, unknown>,
  }
}

/**
 * The constraint cost of a transition: its `info.cost` (0 when absent).
 *
 * @param transition - The transition.
 * @returns The cost.
 * @concept safe-reinforcement-learning
 * @example
 * ```ts
 * import { transitionCost } from 'whitebox-engine/methods/rl/agents'
 * const t = { observation: 0, action: 0, reward: 1, nextObservation: 1, terminated: false, truncated: false }
 * [transitionCost({ ...t, info: { cost: 2 } }), transitionCost(t)] // => [2, 0]
 * ```
 */
export function transitionCost<O, A>(transition: Transition<O, A>): number {
  const c = transition.info?.['cost']
  return typeof c === 'number' ? c : 0
}

// --------------------------------------------------------------------------------------------- Lagrangian PPO

/** Options of {@link lagrangianPpo}. */
export interface LagrangianPpoOptions extends PpoOptions {
  /** The limit `d` on the expected cost per episode (default 0). */
  readonly costLimit?: number
  /** The step size of gradient ascent on the multiplier (default 0.05). */
  readonly lagrangianLearningRate?: number
  /** The initial multiplier (default 0). */
  readonly lagrangeInit?: number
  /** PID control of the multiplier (Stooke et al. 2020) instead of gradient ascent. */
  readonly pid?: { readonly kp?: number; readonly ki?: number; readonly kd?: number }
  /** The cost of a transition (default its `info.cost`). */
  readonly cost?: (transition: Transition<readonly number[], PolicyAction>) => number
}

/** The state of {@link lagrangianPpo}. */
export interface LagrangianPpoState extends PolicyGradientState {
  /** The Lagrange multiplier. */
  readonly lagrange: number
  /** The integral term of PID control. */
  readonly integral: number
  /** The episode cost of the previous update (PID's derivative term). */
  readonly previousCost: number
  /** The cost of the current episode so far. */
  readonly episodeCost: number
  /** Costs of the episodes finished since the last update. */
  readonly episodeCosts: readonly number[]
}

/**
 * Lagrangian PPO for constrained MDPs (Ray, Achiam and Amodei 2019; PPO-Lagrangian of Safety Gym): PPO with a second
 * value head for the cost, GAE cost advantages, and the penalised objective `(A_r - lambda A_c) / (1 + lambda)` in the
 * clipped surrogate (reward advantages standardised, cost advantages centred); before each policy update the
 * multiplier takes a projected gradient-ascent step `lambda <- max(0, lambda + lr (J_c - d))` on the average cost of
 * the episodes finished since the last update, or follows a PID controller of that error (Stooke, Achiam and Abbeel
 * 2020). Costs come from `info.cost` (see {@link costWrapper}) or a cost function. Reports the PPO statistics, the
 * multiplier, the average episode cost and the cost value loss.
 *
 * @param env - The environment's spaces.
 * @param options - PPO's options, the cost limit and the multiplier's dynamics.
 * @returns The agent.
 * @concept safe-reinforcement-learning
 * @reference Ray, Achiam and Amodei (2019), Benchmarking safe exploration in deep reinforcement learning.
 * @example
 * ```ts
 * import { lagrangianPpo } from 'whitebox-engine/methods/rl/agents'
 * import { cartPole } from 'whitebox-engine/methods/rl'
 * lagrangianPpo(cartPole.create(), { costLimit: 5 }).config // => { costLimit: 5, nSteps: 256, gamma: 0.99, pid: false }
 * ```
 */
export function lagrangianPpo<A extends PolicyAction = PolicyAction>(
  env: Spaces,
  options: LagrangianPpoOptions = {},
): Agent<LagrangianPpoState, readonly number[], A> {
  const net = actorCritic(env, options, 'lagrangianPpo', { cost: true })
  const gamma = options.gamma ?? 0.99
  const lambda = options.gaeLambda ?? 0.95
  const nSteps = options.nSteps ?? 256
  const clipRange = options.clipRange ?? 0.2
  const limit = options.costLimit ?? 0
  const lr = options.lagrangianLearningRate ?? 0.05
  const pid = options.pid
  const costOf = options.cost ?? transitionCost
  const opt = chain(
    clipByGlobalNorm(options.maxGradNorm ?? 0.5),
    options.optimizer ?? adam({ learningRate: options.learningRate ?? 3e-4, eps: 1e-5 }),
  )
  const config = {
    epochs: options.nEpochs ?? 10,
    batchSize: options.batchSize ?? 64,
    clipRange,
    ...(options.clipRangeVf === undefined ? {} : { clipRangeVf: options.clipRangeVf }),
    entCoef: options.entCoef ?? 0,
    vfCoef: options.vfCoef ?? 0.5,
    normalizeAdvantage: false,
    ...(options.targetKl === undefined ? {} : { targetKl: options.targetKl }),
    scale: 1,
    objective: (i: ObjectiveInputs) => ppoClipObjective(i.logProb, i.oldLogProb, i.advantages, clipRange),
  }

  const multiplier = (s: LagrangianPpoState, jc: number) => {
    const error = jc - limit
    if (pid === undefined) return { lagrange: Math.max(0, s.lagrange + lr * error), integral: s.integral }
    const integral = Math.max(0, s.integral + error)
    const derivative = Math.max(0, jc - s.previousCost)
    return {
      lagrange: Math.max(0, (pid.kp ?? 0.1) * error + (pid.ki ?? 0.01) * integral + (pid.kd ?? 0) * derivative),
      integral,
    }
  }

  const update = (s: LagrangianPpoState, k: Key): LagrangianPpoState => {
    const batch = prepareRollout(net, s.params, s.pending, gamma, lambda, costOf)
    const costs = batch.cost
    if (costs === undefined) throw new WhiteboxError('value', 'lagrangianPpo: the cost head is missing')
    // With no finished episode in this rollout, keep the previous estimate of the episode cost.
    const jc = s.episodeCosts.length > 0 ? meanOf(s.episodeCosts) : s.previousCost
    const { lagrange, integral } = multiplier(s, jc)
    const ar = standardize(batch.reward.advantages)
    const meanCost = meanOf(costs.advantages)
    const advantages = ar.map((a, i) => (a - lagrange * ((costs.advantages[i] ?? 0) - meanCost)) / (1 + lagrange))
    const result = ppoEpochs(
      net,
      s.params,
      s.optState,
      opt,
      batch,
      advantages,
      [
        { slot: 'value', returns: batch.reward.returns, oldValues: batch.reward.values },
        { slot: 'cost', returns: costs.returns, oldValues: costs.values },
      ],
      config,
      k,
    )
    return {
      ...s,
      params: result.params,
      optState: result.optState,
      lagrange,
      integral,
      previousCost: jc,
      episodeCosts: [],
      stats: { ...result.stats, lagrangeMultiplier: lagrange, episodeCost: jc },
    }
  }

  const metrics: Readonly<Record<string, MetricChannel>> = {
    policyLoss: { description: 'policy loss of the last update', direction: 'lower' },
    valueLoss: { description: 'reward value loss of the last update', direction: 'lower' },
    costValueLoss: { description: 'cost value loss of the last update', direction: 'lower' },
    lagrangeMultiplier: { description: 'the Lagrange multiplier of the cost constraint' },
    episodeCost: { description: 'average cost per episode since the previous update', direction: 'lower' },
    approxKl: { description: 'approximate KL of the last update' },
  }
  return {
    name: 'lagrangianPpo',
    config: { costLimit: limit, nSteps, gamma, pid: pid !== undefined } as Json,
    metrics,
    init: () => ({
      params: net.params,
      optState: opt.init(net.params),
      valueOptState: null,
      pending: [],
      steps: 0,
      updates: 0,
      episodes: 0,
      baseline: 0,
      klCoef: 0,
      stats: {},
      lagrange: options.lagrangeInit ?? 0,
      integral: 0,
      previousCost: 0,
      episodeCost: 0,
      episodeCosts: [],
    }),
    act: (s, obs, k) => net.sample(s.params.policy, obs, k) as A,
    greedy: (s, obs) => net.greedy(s.params.policy, obs) as A,
    learn: (s, t, k) => {
      const tr = t as Transition<readonly number[], PolicyAction>
      const done = t.terminated || t.truncated
      const cost = s.episodeCost + costOf(tr)
      const next: LagrangianPpoState = {
        ...s,
        pending: [...s.pending, tr],
        steps: s.steps + 1,
        episodes: s.episodes + (done ? 1 : 0),
        episodeCost: done ? 0 : cost,
        episodeCosts: done ? [...s.episodeCosts, cost] : s.episodeCosts,
      }
      if (next.pending.length < nSteps) return next
      return { ...update(next, k), pending: [], updates: next.updates + 1 }
    },
    stats: (s) => s.stats,
  }
}

// ------------------------------------------------------------------------------------------------------ shielding

/**
 * A shield (Alshiekh et al. 2018) over a finite action set: the risk of each action in a context (a state or an
 * observation), and the threshold up to which actions are allowed.
 *
 * @concept safe-reinforcement-learning
 */
export interface Shield<C, A> {
  readonly actions: readonly A[]
  /** The risk of an action (0 is safe; a probability of reaching an unsafe state for model shields). */
  risk(context: C, action: A): number
  /** Actions with risk at most this are allowed (default 0 for the shields here). */
  readonly threshold: number
}

/**
 * A shield from a safety predicate: an action is allowed when the predicate holds (risk 0), otherwise its risk is 1.
 *
 * @param isSafe - Whether an action is safe in a context.
 * @param actions - The candidate actions.
 * @returns The shield.
 * @concept safe-reinforcement-learning
 * @example
 * ```ts
 * import { predicateShield, shieldAction } from 'whitebox-engine/methods/rl/agents'
 * const s = predicateShield((x: number, a: number) => x + a <= 3, [0, 1, 2])
 * shieldAction(s, 2, 2) // => { action: 0, replaced: true }
 * ```
 */
export function predicateShield<C, A>(isSafe: (context: C, action: A) => boolean, actions: readonly A[]): Shield<C, A> {
  return { actions, risk: (c, a) => (isSafe(c, a) ? 0 : 1), threshold: 0 }
}

/**
 * A shield from a model lookahead: the risk of an action in a state is the probability of entering an unsafe state
 * within `horizon` steps when that action is taken and the safest action is taken afterwards (a finite-horizon
 * minimum-probability recursion over the environment's transition model). Terminal outcomes that are not unsafe end
 * the risk.
 *
 * @param model - The environment's model (states and actions enumerated, outcomes with probabilities).
 * @param unsafe - Whether a state is unsafe.
 * @param options - The lookahead horizon (default 1) and the allowed risk (default 0).
 * @returns The shield, over the model's states.
 * @concept safe-reinforcement-learning
 * @example
 * ```ts
 * import { modelShield } from 'whitebox-engine/methods/rl/agents'
 * import { frozenLake } from 'whitebox-engine/methods/rl'
 * // On the dry lake, moving down from cell 1 falls into the hole at cell 5.
 * const model = frozenLake.create({ isSlippery: false }).model
 * const shield = model === undefined ? undefined : modelShield(model, (s) => s === 5)
 * [shield?.risk(1, 1), shield?.risk(1, 2)] // => [1, 0]
 * ```
 */
export function modelShield<S, A>(
  model: EnvModel<S, A>,
  unsafe: (state: S) => boolean,
  options: { readonly horizon?: number; readonly threshold?: number } = {},
): Shield<S, A> {
  const horizon = Math.max(1, options.horizon ?? 1)
  const states = model.states
  const bad = states.map((s) => unsafe(s))
  // Q_h(s, a) = sum_o p_o (1 if o is unsafe, 0 if o ends the episode, else V_{h-1}(o)).
  const actionRisk = (s: S, a: A, next: readonly number[]): number => {
    let r = 0
    for (const o of model.transitions(s, a)) {
      const j = model.stateIndex(o.state)
      r += o.probability * (bad[j] === true ? 1 : o.terminated ? 0 : (next[j] ?? 0))
    }
    return r
  }
  // V_0(s) is 1 on unsafe states; V_h(s) = min_a Q_h(s, a) elsewhere. An action's risk is Q_horizon, from V_{horizon-1}.
  let values: number[] = states.map((_, i) => (bad[i] === true ? 1 : 0))
  for (let h = 1; h < horizon; h++) {
    const previous = values
    values = states.map((s, i) => (bad[i] === true ? 1 : minOf(model.actions.map((a) => actionRisk(s, a, previous)))))
  }
  const lookahead = values
  return {
    actions: model.actions,
    risk: (s, a) => actionRisk(s, a, lookahead),
    threshold: options.threshold ?? 0,
  }
}

/**
 * Apply a shield to a proposed action: keep it when its risk is within the threshold, else replace it with the safest
 * allowed action (the lowest risk, the first such action on ties), or the safest action of all when none is allowed.
 *
 * @param shield - The shield.
 * @param context - The state or observation the shield reads.
 * @param proposed - The proposed action.
 * @returns The action to execute and whether it was replaced.
 * @concept safe-reinforcement-learning
 * @example
 * ```ts
 * import { predicateShield, shieldAction } from 'whitebox-engine/methods/rl/agents'
 * shieldAction(predicateShield((_: null, a: number) => a !== 1, [0, 1]), null, 0) // => { action: 0, replaced: false }
 * ```
 */
export function shieldAction<C, A>(
  shield: Shield<C, A>,
  context: C,
  proposed: A,
): { readonly action: A; readonly replaced: boolean } {
  if (shield.risk(context, proposed) <= shield.threshold) return { action: proposed, replaced: false }
  const risks = shield.actions.map((a) => shield.risk(context, a))
  const i = argminOf(risks)
  const action = shield.actions[i]
  if (action === undefined) throw new WhiteboxError('value', 'shieldAction: the shield has no actions')
  return { action, replaced: true }
}

/**
 * Preemptive shielding (Alshiekh et al. 2018) as a wrapper agent: the wrapped agent proposes, the shield replaces
 * unsafe proposals with the safest allowed action, and the agent learns from the actions actually executed. Works
 * around any agent; the shield reads a context computed from the observation (for one-hot observations of a tabular
 * environment, the state index `argmaxOf`).
 *
 * @param agent - The agent to shield.
 * @param shield - The shield.
 * @param contextOf - The shield's context of an observation (default the observation itself).
 * @returns The shielded agent; its greedy actions are shielded too.
 * @concept safe-reinforcement-learning
 * @reference Alshiekh, Bloem, Ehlers, Koenighofer, Niekum and Topcu (2018), Safe reinforcement learning via shielding.
 * @example
 * ```ts
 * import { predicateShield, shield } from 'whitebox-engine/methods/rl/agents'
 * import { policyAgent } from 'whitebox-engine/rl'
 * import { key } from 'whitebox-engine/random'
 * const safe = shield(policyAgent(() => 1), predicateShield((_o: number, a: number) => a === 0, [0, 1]))
 * safe.act(safe.init(key(0)), 0, key(1)) // => 0
 * ```
 */
export function shield<G, O, A, C = O>(
  agent: Agent<G, O, A>,
  shield: Shield<C, A>,
  contextOf: (observation: O) => C = (o) => o as unknown as C,
): Agent<G, O, A> {
  return {
    ...agent,
    name: `shielded:${agent.name}`,
    act: (s, o, k) => shieldAction(shield, contextOf(o), agent.act(s, o, k)).action,
    greedy: (s, o) => shieldAction(shield, contextOf(o), agent.greedy(s, o)).action,
  }
}

/**
 * Post-posed shielding (Alshiekh et al. 2018) as an environment wrapper: the shield checks every action against the
 * environment's state before the step and replaces unsafe ones, reporting `info.shielded` (1 when replaced) and
 * `info.proposedAction`, and optionally subtracting a penalty from the reward of replaced actions so that the learner
 * is told its proposal was unsafe.
 *
 * @param env - The environment.
 * @param shield - A shield over the environment's states.
 * @param options - The penalty for replaced actions (default 0).
 * @returns The shielded environment.
 * @concept safe-reinforcement-learning
 * @example
 * ```ts
 * import { predicateShield, shieldedEnv } from 'whitebox-engine/methods/rl/agents'
 * import { cliffWalking } from 'whitebox-engine/methods/rl'
 * import { key } from 'whitebox-engine/random'
 * const env = shieldedEnv(cliffWalking.create(), predicateShield((s: number, a: number) => !(s === 36 && a === 1), [0, 1, 2, 3]))
 * env.step(36, 1, key(0)).info.shielded // => 1
 * ```
 */
export function shieldedEnv<S, O, A>(
  env: Env<S, O, A>,
  shield: Shield<S, A>,
  options: { readonly penalty?: number } = {},
): Env<S, O, A> {
  const penalty = options.penalty ?? 0
  const { model: _model, ...rest } = env
  return {
    ...rest,
    step: (s, a, k) => {
      const { action, replaced } = shieldAction(shield, s, a)
      const r = env.step(s, action, k)
      return {
        ...r,
        reward: replaced ? r.reward - penalty : r.reward,
        info: { ...r.info, shielded: replaced ? 1 : 0, proposedAction: a as unknown as Json },
      }
    },
    wrapped: env as Env<unknown, unknown, unknown>,
  }
}
