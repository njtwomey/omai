/**
 * On-policy policy-gradient agents, which move a parametrised policy along the gradient of expected return: REINFORCE
 * (Monte Carlo returns with a baseline), A2C (advantage actor-critic on short rollouts), TRPO (natural-gradient steps
 * inside a KL trust region) and PPO (clipped or KL-penalised surrogate objectives, as stable-baselines3 implements it).
 * The surrogate objectives, the adaptive KL rule and the TRPO step are exported on their own as well.
 */

import { grad, jvp, valueAndGrad } from '../autodiff/index.ts'
import { type Json, MdmlError, meanOf } from '../base/index.ts'
import type { Params } from '../nn/index.ts'
import { dnrm2 } from '../numerics/linalg/index.ts'
import { cg } from '../numerics/sparse/index.ts'
import { adam, type ParamTree, rmsprop } from '../optim/index.ts'
import { foldIn, type Key } from '../random/index.ts'
import { type Agent, discountedReturns, minibatchIndices } from '../rl/index.ts'
import {
  add,
  clip,
  exp,
  item,
  mean,
  minimum,
  mul,
  primalOf,
  ravel,
  square,
  sub,
  type TensorLike,
  toFlat,
  type Value,
} from '../tensor/index.ts'
import { gradientStep, rowsTensor, type Spaces, vectorTensor } from './_common.ts'
import {
  type ActorCriticParams,
  actorCritic,
  explainedVariance,
  type PolicyAction,
  policyKl,
  ppoEpochs,
  prepareRollout,
  standardize,
} from './_on-policy.ts'

import {
  initialState,
  METRICS,
  onPolicyAgent,
  type PolicyGradientOptions,
  type PolicyGradientState,
  valueLossOf,
  withClip,
} from './_on-policy-agent.ts'

export type { ActorCriticOptions, ActorCriticParams, PolicyAction } from './_on-policy.ts'
export type { PolicyGradientOptions, PolicyGradientState } from './_on-policy-agent.ts'

// ------------------------------------------------------------------------------------------------- objectives

/**
 * The clipped surrogate objective of PPO (Schulman et al. 2017), to be maximised:
 * `mean(min(r A, clip(r, 1 - eps, 1 + eps) A))` with probability ratios `r = exp(logProb - oldLogProb)` (the negative
 * of stable-baselines3's `policy_loss`).
 *
 * @param logProb - Log-probabilities of the actions under the current policy `[B]`.
 * @param oldLogProb - Log-probabilities under the behaviour policy `[B]`.
 * @param advantages - Advantage estimates `[B]`.
 * @param clipRange - The clipping parameter epsilon (default 0.2).
 * @returns The objective (a scalar value, differentiable in `logProb`).
 * @concept policy-gradient
 * @reference Schulman, Wolski, Dhariwal, Radford and Klimov (2017), Proximal policy optimization algorithms.
 * @example
 * ```ts
 * import { ppoClipObjective } from 'mdml-engine/agents'
 * import { item } from 'mdml-engine/tensor'
 * item(ppoClipObjective([0, 0], [0, 0], [1, 3])) // => 2
 * item(ppoClipObjective([Math.log(2)], [0], [1], 0.2)) // => 1.2
 * ```
 */
export function ppoClipObjective(
  logProb: TensorLike,
  oldLogProb: TensorLike,
  advantages: TensorLike,
  clipRange = 0.2,
): Value {
  const ratio = exp(sub(logProb, oldLogProb))
  return mean(minimum(mul(ratio, advantages), mul(clip(ratio, 1 - clipRange, 1 + clipRange), advantages)))
}

/**
 * The KL-penalised surrogate objective of PPO (Schulman et al. 2017, section 4), to be maximised:
 * `mean(r A) - beta mean(KL(old || new))`.
 *
 * @param logProb - Log-probabilities under the current policy `[B]`.
 * @param oldLogProb - Log-probabilities under the behaviour policy `[B]`.
 * @param advantages - Advantage estimates `[B]`.
 * @param kl - KL divergences from the behaviour policy to the current one per state `[B]`.
 * @param beta - The penalty coefficient.
 * @returns The objective.
 * @concept policy-gradient
 * @reference Schulman et al. (2017), Proximal policy optimization algorithms, section 4.
 * @example
 * ```ts
 * import { ppoKlObjective } from 'mdml-engine/agents'
 * import { item } from 'mdml-engine/tensor'
 * item(ppoKlObjective([0, 0], [0, 0], [1, 3], [0.5, 0.5], 2)) // => 1
 * ```
 */
export function ppoKlObjective(
  logProb: TensorLike,
  oldLogProb: TensorLike,
  advantages: TensorLike,
  kl: TensorLike,
  beta: number,
): Value {
  return sub(mean(mul(exp(sub(logProb, oldLogProb)), advantages)), mul(beta, mean(kl)))
}

/**
 * The adaptive penalty rule of KL-penalised PPO: halve `beta` when the measured KL is below `target / 1.5`, double it
 * above `1.5 target`, and keep it otherwise.
 *
 * @param beta - The current coefficient.
 * @param kl - The mean KL divergence measured after the update.
 * @param target - The target KL.
 * @returns The next coefficient.
 * @concept policy-gradient
 * @example
 * ```ts
 * import { adaptKlCoefficient } from 'mdml-engine/agents'
 * [adaptKlCoefficient(1, 0.1, 0.01), adaptKlCoefficient(1, 0.001, 0.01), adaptKlCoefficient(1, 0.01, 0.01)] // => [2, 0.5, 1]
 * ```
 */
export function adaptKlCoefficient(beta: number, kl: number, target: number): number {
  if (kl < target / 1.5) return beta / 2
  if (kl > target * 1.5) return beta * 2
  return beta
}

/** Options of {@link trpoStep}. */
export interface TrpoStepOptions {
  /** The trust-region radius delta on the mean KL (default 0.01). */
  readonly maxKl?: number
  /** Damping added to the Fisher matrix, `F + damping I` (default 0.1, sb3-contrib's `cg_damping`). */
  readonly damping?: number
  /** Conjugate-gradient iterations (default 15, sb3-contrib's `cg_max_steps`). */
  readonly cgIterations?: number
}

/** The natural-gradient step of {@link trpoStep}. */
export interface TrpoStepResult {
  /** The search direction `x ~ (F + damping I)^{-1} g` from conjugate gradients. */
  readonly direction: number[]
  /** `x^T (F + damping I) x`. */
  readonly curvature: number
  /** The largest step along `x` within the quadratic trust region, `sqrt(2 delta / x^T F x)`. */
  readonly stepSize: number
  /** `stepSize * x`, the full step before the line search. */
  readonly step: number[]
  readonly iterations: number
  readonly residualNorm: number
}

/**
 * TRPO's natural-gradient step (Schulman et al. 2015, appendix C): solve `(F + damping I) x = g` by conjugate gradients
 * (numerics/sparse `cg`) using only Fisher-vector products, then scale `x` to the edge of the quadratic KL trust region
 * `1/2 s^T F s = delta`.
 *
 * @param fvp - The Fisher-vector product `v -> F v` (without damping).
 * @param gradient - The gradient `g` of the surrogate objective.
 * @param options - Trust-region radius, damping and iterations.
 * @returns The direction, its curvature, the step size and the full step.
 * @concept policy-gradient
 * @reference Schulman, Levine, Moritz, Jordan and Abbeel (2015), Trust region policy optimization.
 * @example
 * ```ts
 * import { trpoStep } from 'mdml-engine/agents'
 * trpoStep((v) => v.map((x) => 4 * x), [2, 0], { maxKl: 0.5, damping: 0 }).step // => [0.5, 0]
 * ```
 */
export function trpoStep(
  fvp: (v: readonly number[]) => number[],
  gradient: readonly number[],
  options: TrpoStepOptions = {},
): TrpoStepResult {
  const n = gradient.length
  const damping = options.damping ?? 0.1
  const maxKl = options.maxKl ?? 0.01
  const product = (v: readonly number[]): number[] => {
    const fv = fvp(v)
    return fv.map((x, i) => x + damping * (v[i] ?? 0))
  }
  const solve = cg(
    {
      shape: [n, n],
      matvec: (x: Value) => vectorTensor(product(toFlat(x))),
    },
    vectorTensor(gradient),
    { rtol: 1e-12, atol: 1e-14, maxIterations: options.cgIterations ?? 15 },
  )
  const direction = toFlat(solve.x)
  const ax = product(direction)
  let curvature = 0
  for (const [i, x] of direction.entries()) curvature += x * (ax[i] ?? 0)
  const stepSize = curvature > 0 ? Math.sqrt((2 * maxKl) / curvature) : 0
  return {
    direction,
    curvature,
    stepSize,
    step: direction.map((x) => x * stepSize),
    iterations: solve.iterations,
    residualNorm: solve.residualNorm,
  }
}

// ------------------------------------------------------------------------------------------------------ REINFORCE

/** Options of {@link reinforce}. */
export interface ReinforceOptions extends PolicyGradientOptions {
  /** The baseline subtracted from returns (default `'value'`: a learned state-value function). */
  readonly baseline?: 'none' | 'average' | 'value'
  /** Step size of the running-average baseline (default 0.1). */
  readonly baselineRate?: number
  /** Weight each action by the return from its step on (default true) rather than the whole episode's return. */
  readonly rewardToGo?: boolean
  /** Standardise the weights (returns minus baseline) of each update (default false). */
  readonly normalizeReturns?: boolean
  /** Episodes per update (default 1). */
  readonly episodesPerUpdate?: number
  /** Learning rate of the value baseline (default 1e-2). */
  readonly valueLearningRate?: number
  /** Gradient steps on the value baseline per update (default 5). */
  readonly valueSteps?: number
}

/**
 * REINFORCE (Williams 1992): after every `episodesPerUpdate` episodes, one gradient step on
 * `-mean(log pi(a_t | s_t) (G_t - b_t))` with reward-to-go returns `G_t` (or the whole episode's return) and a baseline:
 * none, a running average of the returns, or a learned value function `V(s_t)` regressed on the returns (which also
 * bootstraps the return of a truncated episode from its final observation). Works with categorical policies for
 * discrete actions and Gaussian ones (optionally tanh-squashed) for box actions. Reports the policy and value losses,
 * the entropy and the explained variance.
 *
 * @param env - The environment's spaces (vector observations).
 * @param options - Network, optimiser, baseline and return options.
 * @returns The agent.
 * @concept policy-gradient
 * @reference Williams (1992), Simple statistical gradient-following algorithms for connectionist reinforcement learning.
 * @example
 * ```ts
 * import { reinforce } from 'mdml-engine/agents'
 * import { cartPole } from 'mdml-engine/environments'
 * reinforce(cartPole.create(), { hidden: [16], baseline: 'average' }).name // => 'reinforce'
 * ```
 */
export function reinforce<A extends PolicyAction = PolicyAction>(
  env: Spaces,
  options: ReinforceOptions = {},
): Agent<PolicyGradientState, readonly number[], A> {
  const net = actorCritic(env, options, 'reinforce')
  const gamma = options.gamma ?? 0.99
  const baseline = options.baseline ?? 'value'
  const rate = options.baselineRate ?? 0.1
  const rewardToGo = options.rewardToGo ?? true
  const perUpdate = options.episodesPerUpdate ?? 1
  const entCoef = options.entCoef ?? 0
  const opt = withClip(options.optimizer ?? adam({ learningRate: options.learningRate ?? 5e-3 }), options.maxGradNorm)
  const valueOpt = adam({ learningRate: options.valueLearningRate ?? 1e-2 })
  const valueSteps = options.valueSteps ?? 5
  const d = net.observationSize

  const update = (s: PolicyGradientState): PolicyGradientState => {
    const steps = s.pending
    const obs = rowsTensor(
      steps.map((t) => t.observation),
      d,
    )
    const values = baseline === 'value' ? toFlat(net.value(s.params.value, obs)) : []
    // Returns per episode, bootstrapping truncated episodes from the value of their final observation.
    const returns: number[] = []
    let start = 0
    for (const [i, t] of steps.entries()) {
      if (!(t.terminated || t.truncated) && i < steps.length - 1) continue
      const episode = steps.slice(start, i + 1)
      const bootstrap =
        baseline === 'value' && t.truncated && !t.terminated
          ? item(net.value(s.params.value, rowsTensor([t.nextObservation], d)))
          : 0
      const g = discountedReturns(
        episode.map((e) => e.reward),
        gamma,
        { bootstrap },
      )
      returns.push(...(rewardToGo ? g : g.map(() => g[0] ?? 0)))
      start = i + 1
    }
    const b =
      baseline === 'value'
        ? values
        : baseline === 'average'
          ? returns.map(() => (s.updates === 0 ? meanOf(returns) : s.baseline))
          : returns.map(() => 0)
    const raw = returns.map((g, i) => g - (b[i] ?? 0))
    const weights = vectorTensor(options.normalizeReturns === true ? standardize(raw) : raw)
    const actions = net.actions(steps.map((t) => t.action))
    let entropy = 0
    const policyLoss = (p: ParamTree): Value => {
      const head = net.head(net.dist(p as Params, obs))
      const ent = mean(head.entropy())
      entropy = item(primalOf(ent))
      return sub(mul(-1, mean(mul(head.logProb(actions), weights))), mul(entCoef, ent))
    }
    const step = gradientStep(policyLoss, s.params.policy, opt, s.optState)
    let value = s.params.value
    let valueOptState = s.valueOptState
    let valueLoss = 0
    if (baseline === 'value') {
      const targets = vectorTensor(returns)
      for (let i = 0; i < valueSteps; i++) {
        const v = gradientStep(valueLossOf(net, obs, targets), value, valueOpt, valueOptState)
        value = v.params as Params
        valueOptState = v.optState
        valueLoss = v.loss
      }
    }
    return {
      ...s,
      params: { ...s.params, policy: step.params as Params, value },
      optState: step.optState,
      valueOptState,
      baseline:
        baseline === 'average'
          ? s.updates === 0
            ? meanOf(returns)
            : s.baseline + rate * (meanOf(returns) - s.baseline)
          : 0,
      stats: {
        policyLoss: step.loss,
        valueLoss,
        entropy,
        explainedVariance: baseline === 'value' ? explainedVariance(values, returns) : 0,
      },
    }
  }

  return onPolicyAgent<A>(
    'reinforce',
    {
      hidden: [...(options.hidden ?? [64, 64])],
      gamma,
      baseline,
      rewardToGo,
      normalizeReturns: options.normalizeReturns ?? false,
      episodesPerUpdate: perUpdate,
    } as Json,
    METRICS,
    net,
    () => initialState(net, opt.init(net.params.policy), valueOpt.init(net.params.value)),
    (s, t) => (t.terminated || t.truncated) && s.episodes % perUpdate === 0,
    update,
  )
}

// ------------------------------------------------------------------------------------------------------------ A2C

/** Options of {@link a2c}. */
export interface A2cOptions extends PolicyGradientOptions {
  /** Steps per rollout (default 5). */
  readonly nSteps?: number
  /** `'nstep'` bootstrapped returns over the rollout (default, stable-baselines3's `gae_lambda = 1`) or `'gae'`. */
  readonly advantages?: 'nstep' | 'gae'
  /** The GAE lambda when `advantages` is `'gae'` (default 0.95). */
  readonly gaeLambda?: number
  /** The value loss coefficient (default 0.5). */
  readonly vfCoef?: number
  readonly normalizeAdvantage?: boolean
}

/**
 * Advantage actor-critic (A2C, the synchronous form of Mnih et al. 2016's A3C, as in stable-baselines3): every
 * `nSteps` steps one gradient step on the rollout with `policy loss + vfCoef value loss - entCoef entropy`, advantages
 * from n-step returns (or GAE) bootstrapped by the value network (including truncated steps, from their final
 * observations), RMSprop (alpha 0.99, eps 1e-5) and gradient clipping at norm 0.5 by default.
 *
 * @param env - The environment's spaces.
 * @param options - Network, optimiser, rollout length and loss coefficients.
 * @returns The agent.
 * @concept actor-critic
 * @reference Mnih et al. (2016), Asynchronous methods for deep reinforcement learning.
 * @example
 * ```ts
 * import { a2c } from 'mdml-engine/agents'
 * import { cartPole } from 'mdml-engine/environments'
 * a2c(cartPole.create(), { nSteps: 8 }).config // => { hidden: [64, 64], gamma: 0.99, nSteps: 8, gaeLambda: 1, entCoef: 0, vfCoef: 0.5 }
 * ```
 */
export function a2c<A extends PolicyAction = PolicyAction>(
  env: Spaces,
  options: A2cOptions = {},
): Agent<PolicyGradientState, readonly number[], A> {
  const net = actorCritic(env, options, 'a2c')
  const gamma = options.gamma ?? 0.99
  const nSteps = options.nSteps ?? 5
  const lambda = options.advantages === 'gae' ? (options.gaeLambda ?? 0.95) : 1
  const entCoef = options.entCoef ?? 0
  const vfCoef = options.vfCoef ?? 0.5
  const opt = withClip(
    options.optimizer ?? rmsprop({ learningRate: options.learningRate ?? 7e-4, decay: 0.99, eps: 1e-5 }),
    options.maxGradNorm ?? 0.5,
  )
  const d = net.observationSize
  const update = (s: PolicyGradientState): PolicyGradientState => {
    const batch = prepareRollout(net, s.params, s.pending, gamma, lambda)
    const obs = rowsTensor(batch.observations, d)
    const actions = net.actions(batch.actions)
    const adv = batch.reward.advantages
    const advT = vectorTensor(options.normalizeAdvantage === true ? standardize(adv) : adv)
    const returns = vectorTensor(batch.reward.returns)
    const aux = { policyLoss: 0, valueLoss: 0, entropy: 0 }
    const loss = (p: ParamTree): Value => {
      const q = p as ActorCriticParams
      const head = net.head(net.dist(q.policy, obs))
      const pl = mul(-1, mean(mul(head.logProb(actions), advT)))
      const vl = mean(square(sub(returns, net.value(q.value, obs))))
      const ent = mean(head.entropy())
      aux.policyLoss = item(primalOf(pl))
      aux.valueLoss = item(primalOf(vl))
      aux.entropy = item(primalOf(ent))
      return sub(add(pl, mul(vfCoef, vl)), mul(entCoef, ent))
    }
    const step = gradientStep(loss, s.params, opt, s.optState)
    return {
      ...s,
      params: step.params as ActorCriticParams,
      optState: step.optState,
      stats: { ...aux, explainedVariance: explainedVariance(batch.reward.values, batch.reward.returns) },
    }
  }
  return onPolicyAgent<A>(
    'a2c',
    { hidden: [...(options.hidden ?? [64, 64])], gamma, nSteps, gaeLambda: lambda, entCoef, vfCoef } as Json,
    METRICS,
    net,
    () => initialState(net, opt.init(net.params), null),
    (s) => s.pending.length >= nSteps,
    update,
  )
}

// ----------------------------------------------------------------------------------------------------------- TRPO

/** Options of {@link trpo}. */
export interface TrpoOptions extends PolicyGradientOptions, TrpoStepOptions {
  /** Steps per rollout (default 512; sb3-contrib uses 2048). */
  readonly nSteps?: number
  readonly gaeLambda?: number
  /** Minibatch size of the value regression (default 128). */
  readonly batchSize?: number
  /** Epochs of value regression per update (default 10, sb3-contrib's `n_critic_updates`). */
  readonly criticUpdates?: number
  /** Backtracking factor of the line search (default 0.8). */
  readonly lineSearchShrink?: number
  /** Line-search trials (default 10). */
  readonly lineSearchSteps?: number
  readonly normalizeAdvantage?: boolean
}

/**
 * Trust region policy optimisation (Schulman et al. 2015, as sb3-contrib's TRPO): every `nSteps` steps, the natural
 * gradient of the importance-weighted surrogate `mean(r A)` by conjugate gradients on Fisher-vector products (the
 * Hessian of the mean KL, by forward-over-reverse differentiation), scaled to the trust region `KL <= maxKl`, then a
 * backtracking line search that accepts the first step with `KL < maxKl` and an improved surrogate (else the policy is
 * kept); the value network is regressed separately with Adam (`learningRate`, default 1e-3) for `criticUpdates`
 * epochs. Reports the line-search outcome besides the usual losses.
 *
 * @param env - The environment's spaces.
 * @param options - Network, trust region, conjugate gradients, line search and critic options.
 * @returns The agent.
 * @concept policy-gradient
 * @reference Schulman, Levine, Moritz, Jordan and Abbeel (2015), Trust region policy optimization.
 * @example
 * ```ts
 * import { trpo } from 'mdml-engine/agents'
 * import { cartPole } from 'mdml-engine/environments'
 * trpo(cartPole.create(), { maxKl: 0.02 }).name // => 'trpo'
 * ```
 */
export function trpo<A extends PolicyAction = PolicyAction>(
  env: Spaces,
  options: TrpoOptions = {},
): Agent<PolicyGradientState, readonly number[], A> {
  const net = actorCritic(env, options, 'trpo')
  const gamma = options.gamma ?? 0.99
  const lambda = options.gaeLambda ?? 0.95
  const nSteps = options.nSteps ?? 512
  const maxKl = options.maxKl ?? 0.01
  const shrink = options.lineSearchShrink ?? 0.8
  const trials = options.lineSearchSteps ?? 10
  const batchSize = options.batchSize ?? 128
  const criticUpdates = options.criticUpdates ?? 10
  const valueOpt = withClip(
    options.optimizer ?? adam({ learningRate: options.learningRate ?? 1e-3 }),
    options.maxGradNorm,
  )
  const d = net.observationSize
  const update = (s: PolicyGradientState, k: Key): PolicyGradientState => {
    const batch = prepareRollout(net, s.params, s.pending, gamma, lambda)
    const obs = rowsTensor(batch.observations, d)
    const actions = net.actions(batch.actions)
    const oldLogProb = vectorTensor(batch.logProbs)
    const adv = batch.reward.advantages
    const advT = vectorTensor((options.normalizeAdvantage ?? true) ? standardize(adv) : adv)
    const surrogate = (p: Params): Value =>
      mean(mul(exp(sub(net.head(net.dist(p, obs)).logProb(actions), oldLogProb)), advT))
    const meanKl = (p: Params): Value => mean(policyKl(batch.oldDist, net.dist(p, obs)))
    const policy = s.params.policy
    const { value: l0, grad: g } = valueAndGrad(surrogate)(policy)
    const { flat: theta, unravel } = ravel(policy)
    const gradKl = grad(meanKl)
    const fvp = (v: readonly number[]): number[] => Array.from(ravel(jvp(gradKl, policy, unravel(v)).tangent).flat)
    const gFlat = ravel(g).flat
    const base = item(l0)
    let next = policy
    let accepted = 0
    let kl = 0
    let objective = base
    if (dnrm2(gFlat) > 0) {
      const { step } = trpoStep(fvp, Array.from(gFlat), options)
      for (let i = 0; i < trials; i++) {
        const alpha = shrink ** i
        const candidate = unravel(Array.from(theta, (x, j) => x + alpha * (step[j] ?? 0)))
        const klC = item(meanKl(candidate))
        const objC = item(surrogate(candidate))
        if (klC < maxKl && objC > base) {
          next = candidate
          accepted = 1
          kl = klC
          objective = objC
          break
        }
      }
    }
    // Value regression on the lambda-returns, in minibatch epochs.
    let value = s.params.value
    let valueOptState = s.valueOptState
    let valueLoss = 0
    for (let epoch = 0; epoch < criticUpdates; epoch++) {
      for (const idx of minibatchIndices(batch.observations.length, batchSize, foldIn(k, epoch))) {
        const o = rowsTensor(
          idx.map((i) => batch.observations[i] ?? []),
          d,
        )
        const y = vectorTensor(idx.map((i) => batch.reward.returns[i] ?? 0))
        const v = gradientStep(valueLossOf(net, o, y), value, valueOpt, valueOptState)
        value = v.params as Params
        valueOptState = v.optState
        valueLoss = v.loss
      }
    }
    return {
      ...s,
      params: { ...s.params, policy: next, value },
      valueOptState,
      stats: {
        policyLoss: -objective,
        valueLoss,
        entropy: item(mean(net.head(net.dist(next, obs)).entropy())),
        approxKl: kl,
        lineSearchSuccess: accepted,
        explainedVariance: explainedVariance(batch.reward.values, batch.reward.returns),
      },
    }
  }
  return onPolicyAgent<A>(
    'trpo',
    { hidden: [...(options.hidden ?? [64, 64])], gamma, gaeLambda: lambda, nSteps, maxKl } as Json,
    { ...METRICS, lineSearchSuccess: { description: 'whether the line search accepted a step (1) or not (0)' } },
    net,
    () => initialState(net, null, valueOpt.init(net.params.value)),
    (s) => s.pending.length >= nSteps,
    update,
  )
}

// ------------------------------------------------------------------------------------------------------------ PPO

/** Options of {@link ppo}. */
export interface PpoOptions extends PolicyGradientOptions {
  /** Steps per rollout (default 256; stable-baselines3 uses 2048). */
  readonly nSteps?: number
  /** Minibatch size (default 64). */
  readonly batchSize?: number
  /** Epochs over each rollout (default 10). */
  readonly nEpochs?: number
  readonly gaeLambda?: number
  /** `'clip'` (the clipped surrogate, default) or `'kl'` (the adaptive KL penalty). */
  readonly objective?: 'clip' | 'kl'
  readonly clipRange?: number
  /** Clip value predictions to within this distance of their old values (default none). */
  readonly clipRangeVf?: number
  readonly vfCoef?: number
  /** Standardise advantages per minibatch (default true). */
  readonly normalizeAdvantage?: boolean
  /** Stop the epochs of an update once the approximate KL exceeds 1.5 times this (default none). */
  readonly targetKl?: number
  /** The KL target of the penalty variant (default 0.01). */
  readonly klTarget?: number
  /** The initial penalty coefficient of the penalty variant (default 1). */
  readonly klCoef?: number
  /** Anneal the learning rate linearly to zero over `totalSteps` environment steps (default false). */
  readonly annealLearningRate?: boolean
  readonly totalSteps?: number
}

/**
 * Proximal policy optimisation (Schulman et al. 2017) as stable-baselines3 implements it, with its defaults scaled
 * down for small problems: every `nSteps` steps, `nEpochs` epochs over shuffled minibatches of the rollout, maximising
 * the clipped surrogate ({@link ppoClipObjective}) or the KL-penalised one ({@link ppoKlObjective}, with
 * {@link adaptKlCoefficient} after each update), with GAE advantages standardised per minibatch, an optionally
 * clipped value loss, an entropy bonus, gradient clipping at norm 0.5, optional target-KL early stopping and linear
 * learning-rate annealing. Truncated steps bootstrap from the value of their final observation. Reports the policy
 * and value losses, entropy, approximate KL, clip fraction and explained variance.
 *
 * @param env - The environment's spaces.
 * @param options - Network, optimiser, rollout, objective and regularisation options.
 * @returns The agent.
 * @concept policy-gradient
 * @reference Schulman et al. (2017), Proximal policy optimization algorithms.
 * @index agent Proximal policy optimisation; REINFORCE, A2C and TRPO beside it.
 * @example
 * ```ts
 * import { ppo } from 'mdml-engine/agents'
 * import { cartPole } from 'mdml-engine/environments'
 * ppo(cartPole.create(), { objective: 'kl' }).name // => 'ppo'
 * ```
 */
export function ppo<A extends PolicyAction = PolicyAction>(
  env: Spaces,
  options: PpoOptions = {},
): Agent<PolicyGradientState, readonly number[], A> {
  const net = actorCritic(env, options, 'ppo')
  const gamma = options.gamma ?? 0.99
  const lambda = options.gaeLambda ?? 0.95
  const nSteps = options.nSteps ?? 256
  const clipRange = options.clipRange ?? 0.2
  const variant = options.objective ?? 'clip'
  const klTarget = options.klTarget ?? 0.01
  const anneal = options.annealLearningRate ?? false
  const totalSteps = options.totalSteps
  if (anneal && totalSteps === undefined) throw new MdmlError('value', 'ppo: annealing needs totalSteps')
  const opt = withClip(
    options.optimizer ?? adam({ learningRate: options.learningRate ?? 3e-4, eps: 1e-5 }),
    options.maxGradNorm ?? 0.5,
  )
  const base = {
    epochs: options.nEpochs ?? 10,
    batchSize: options.batchSize ?? 64,
    clipRange,
    ...(options.clipRangeVf === undefined ? {} : { clipRangeVf: options.clipRangeVf }),
    entCoef: options.entCoef ?? 0,
    vfCoef: options.vfCoef ?? 0.5,
    normalizeAdvantage: options.normalizeAdvantage ?? true,
    ...(options.targetKl === undefined ? {} : { targetKl: options.targetKl }),
  }
  const update = (s: PolicyGradientState, k: Key): PolicyGradientState => {
    const batch = prepareRollout(net, s.params, s.pending, gamma, lambda)
    const scale = anneal ? Math.max(0, 1 - s.steps / (totalSteps ?? 1)) : 1
    const beta = s.klCoef
    const result = ppoEpochs(
      net,
      s.params,
      s.optState,
      opt,
      batch,
      batch.reward.advantages,
      [{ slot: 'value', returns: batch.reward.returns, oldValues: batch.reward.values }],
      {
        ...base,
        scale,
        objective:
          variant === 'clip'
            ? (i) => ppoClipObjective(i.logProb, i.oldLogProb, i.advantages, clipRange)
            : (i) => ppoKlObjective(i.logProb, i.oldLogProb, i.advantages, i.kl(), beta),
      },
      k,
    )
    let klCoef = beta
    const stats: Record<string, number> = { ...result.stats }
    if (variant === 'kl') {
      const obs = rowsTensor(batch.observations, net.observationSize)
      const kl = item(mean(policyKl(batch.oldDist, net.dist(result.params.policy, obs))))
      klCoef = adaptKlCoefficient(beta, kl, klTarget)
      stats['kl'] = kl
      stats['klCoef'] = klCoef
    }
    return { ...s, params: result.params, optState: result.optState, klCoef, stats }
  }
  return onPolicyAgent<A>(
    'ppo',
    {
      hidden: [...(options.hidden ?? [64, 64])],
      gamma,
      gaeLambda: lambda,
      nSteps,
      batchSize: base.batchSize,
      nEpochs: base.epochs,
      objective: variant,
      clipRange,
    } as Json,
    {
      ...METRICS,
      clipFraction: { description: 'fraction of probability ratios outside the clip range' },
      ...(variant === 'kl' ? { klCoef: { description: 'the adaptive KL penalty coefficient' } } : {}),
    },
    net,
    () => initialState(net, opt.init(net.params), null, options.klCoef ?? 1),
    (s) => s.pending.length >= nSteps,
    update,
  )
}
