/**
 * Off-policy evaluation: estimating how well a new policy would do from data logged under a different one. For
 * contextual bandits, inverse propensity scoring, its self-normalised and clipped forms, the direct method and the
 * doubly robust estimator; for sequential decisions, fitted Q evaluation, exactly in the tabular case or with any
 * regressor, as traced algorithms.
 */

import { type Estimate, estimate, type Json, meanOf, OmaiError, varianceOf } from '../base/index.ts'
import { type Algorithm, channels, type RunOptions, runAlgorithm } from '../trace/index.ts'
import type { Transition } from './agent.ts'
import type { TabularPolicy } from './mdp.ts'

// ----------------------------------------------------------------------------------------- contextual bandit data

/**
 * Logged contextual-bandit data: for each round the context shown, the action the behaviour policy took, the reward
 * it received and the behaviour policy's probability of that action (its propensity).
 *
 * @concept off-policy-evaluation
 */
export interface LoggedBandit<C> {
  readonly contexts: readonly C[]
  readonly actions: readonly number[]
  readonly rewards: readonly number[]
  readonly propensities: readonly number[]
}

/** A target policy: the probability of every action in a context. */
export type TargetPolicy<C> = (context: C) => readonly number[]

/** A reward model: the predicted expected reward of an action in a context (for the direct method). */
export type RewardModel<C> = (context: C, action: number) => number

function checkLogged<C>(data: LoggedBandit<C>, name: string): number {
  const n = data.actions.length
  if (n === 0) throw new OmaiError('value', `${name}: no logged rounds`)
  if (data.contexts.length !== n || data.rewards.length !== n || data.propensities.length !== n) {
    throw new OmaiError('shape', `${name}: contexts, actions, rewards and propensities differ in length`)
  }
  for (const p of data.propensities) {
    if (!(p > 0 && p <= 1)) throw new OmaiError('value', `${name}: propensities must be in (0, 1], got ${p}`)
  }
  return n
}

/** The mean of per-round terms with its standard error `sd / sqrt(n)` (sample standard deviation). */
function meanEstimate(terms: readonly number[]): Estimate {
  const n = terms.length
  return estimate(meanOf(terms), n > 1 ? Math.sqrt(varianceOf(terms, 1) / n) : 0)
}

/**
 * The importance weights `pi(a_i | x_i) / mu(a_i | x_i)` of logged rounds under a target policy.
 *
 * @param data - The logged rounds.
 * @param target - The target policy.
 * @returns One weight per round.
 * @throws OmaiError with code `value` for empty data or propensities outside (0, 1].
 * @concept importance-sampling
 * @example
 * ```ts
 * import { importanceWeights } from 'omai-engine/rl'
 * const data = { contexts: [0, 0], actions: [0, 1], rewards: [1, 0], propensities: [0.5, 0.5] }
 * importanceWeights(data, () => [1, 0]) // => [2, 0]
 * ```
 */
export function importanceWeights<C>(data: LoggedBandit<C>, target: TargetPolicy<C>): number[] {
  checkLogged(data, 'importanceWeights')
  return data.actions.map((a, i) => (target(data.contexts[i] as C)[a] ?? 0) / (data.propensities[i] as number))
}

/**
 * Inverse propensity scoring (Horvitz and Thompson 1952): the unbiased estimate `mean_i w_i r_i` of the target
 * policy's value, with the standard error of the mean of the terms.
 *
 * @param data - The logged rounds.
 * @param target - The target policy.
 * @returns The estimate with its standard error.
 * @concept off-policy-evaluation
 * @example
 * ```ts
 * import { ips } from 'omai-engine/rl'
 * const data = { contexts: [0, 0], actions: [0, 1], rewards: [1, 0], propensities: [0.5, 0.5] }
 * ips(data, () => [1, 0]).value // => 1
 * ```
 */
export function ips<C>(data: LoggedBandit<C>, target: TargetPolicy<C>): Estimate {
  const w = importanceWeights(data, target)
  return meanEstimate(w.map((x, i) => x * (data.rewards[i] ?? 0)))
}

/**
 * Self-normalised inverse propensity scoring (Swaminathan and Joachims 2015): `sum w_i r_i / sum w_i`, slightly biased
 * but with far lower variance when weights are large; the standard error is the delta-method
 * `sqrt(sum w_i^2 (r_i - V)^2) / sum w_i`.
 *
 * @param data - The logged rounds.
 * @param target - The target policy.
 * @returns The estimate with its standard error.
 * @throws OmaiError with code `value` when every weight is zero.
 * @concept off-policy-evaluation
 * @example
 * ```ts
 * import { snips } from 'omai-engine/rl'
 * const data = { contexts: [0, 0, 0], actions: [0, 1, 0], rewards: [1, 0, 0], propensities: [0.5, 0.5, 0.5] }
 * snips(data, () => [1, 0]).value // => 0.5
 * ```
 */
export function snips<C>(data: LoggedBandit<C>, target: TargetPolicy<C>): Estimate {
  const w = importanceWeights(data, target)
  let sw = 0
  let swr = 0
  for (const [i, x] of w.entries()) {
    sw += x
    swr += x * (data.rewards[i] ?? 0)
  }
  if (!(sw > 0)) throw new OmaiError('value', 'snips: the target policy gives every logged action probability 0')
  const value = swr / sw
  let ss = 0
  for (const [i, x] of w.entries()) ss += x * x * ((data.rewards[i] ?? 0) - value) ** 2
  return estimate(value, Math.sqrt(ss) / sw)
}

/**
 * Clipped (truncated) inverse propensity scoring (Ionides 2008; Bottou et al. 2013): IPS with every weight capped at
 * `maxWeight`, trading a downward bias for bounded variance.
 *
 * @param data - The logged rounds.
 * @param target - The target policy.
 * @param options - The cap `maxWeight` (default 10).
 * @returns The estimate with its standard error.
 * @concept off-policy-evaluation
 * @example
 * ```ts
 * import { clippedIps } from 'omai-engine/rl'
 * const data = { contexts: [0, 0], actions: [0, 1], rewards: [1, 0], propensities: [0.25, 0.75] }
 * clippedIps(data, () => [1, 0], { maxWeight: 2 }).value // => 1
 * ```
 */
export function clippedIps<C>(
  data: LoggedBandit<C>,
  target: TargetPolicy<C>,
  options: { readonly maxWeight?: number } = {},
): Estimate {
  const cap = options.maxWeight ?? 10
  const w = importanceWeights(data, target)
  return meanEstimate(w.map((x, i) => Math.min(x, cap) * (data.rewards[i] ?? 0)))
}

/** The model's expected reward under the target policy in context `x`: `sum_a pi(a | x) r^(x, a)`. */
function modelValue<C>(x: C, target: TargetPolicy<C>, model: RewardModel<C>): number {
  let v = 0
  for (const [a, p] of target(x).entries()) if (p !== 0) v += p * model(x, a)
  return v
}

/**
 * The direct method: the reward model's expected reward under the target policy, averaged over the logged contexts,
 * `mean_i sum_a pi(a | x_i) r^(x_i, a)`. Unbiased only when the model is right; low variance.
 *
 * @param data - The logged rounds (only the contexts are used).
 * @param target - The target policy.
 * @param model - The reward model.
 * @returns The estimate with its standard error.
 * @concept off-policy-evaluation
 * @example
 * ```ts
 * import { directMethod } from 'omai-engine/rl'
 * const data = { contexts: [0, 1], actions: [0, 0], rewards: [0, 0], propensities: [1, 1] }
 * directMethod(data, () => [0.5, 0.5], (x, a) => x + a).value // => 1
 * ```
 */
export function directMethod<C>(data: LoggedBandit<C>, target: TargetPolicy<C>, model: RewardModel<C>): Estimate {
  checkLogged(data, 'directMethod')
  return meanEstimate(data.contexts.map((x) => modelValue(x, target, model)))
}

/**
 * The doubly robust estimator (Dudik, Langford and Li 2011): the direct method corrected by importance-weighted
 * residuals, `mean_i [sum_a pi(a | x_i) r^(x_i, a) + w_i (r_i - r^(x_i, a_i))]`. Unbiased when either the
 * propensities or the reward model are right, with lower variance than IPS when the model is good. `maxWeight` clips
 * the weights.
 *
 * @param data - The logged rounds.
 * @param target - The target policy.
 * @param model - The reward model.
 * @param options - An optional weight cap.
 * @returns The estimate with its standard error.
 * @concept off-policy-evaluation
 * @example
 * ```ts
 * import { doublyRobust } from 'omai-engine/rl'
 * const data = { contexts: [0, 0], actions: [0, 1], rewards: [1, 0], propensities: [0.5, 0.5] }
 * doublyRobust(data, () => [1, 0], () => 0.5).value // => 1
 * ```
 */
export function doublyRobust<C>(
  data: LoggedBandit<C>,
  target: TargetPolicy<C>,
  model: RewardModel<C>,
  options: { readonly maxWeight?: number } = {},
): Estimate {
  const cap = options.maxWeight ?? Infinity
  const w = importanceWeights(data, target)
  return meanEstimate(
    data.contexts.map((x, i) => {
      const a = data.actions[i] as number
      return modelValue(x, target, model) + Math.min(w[i] ?? 0, cap) * ((data.rewards[i] ?? 0) - model(x, a))
    }),
  )
}

// ------------------------------------------------------------------------------------------ fitted Q evaluation

/** Options of the tabular fitted Q evaluation. */
export interface TabularFqeOptions {
  readonly nStates: number
  readonly nActions: number
  /** Discount (default 0.99). */
  readonly gamma?: number
  /** Stop when the largest change of `Q` falls below this (default 1e-10). */
  readonly tol?: number
  /** The initial-state distribution, for the policy's value (default none). */
  readonly initial?: readonly number[]
}

/** The state of {@link tabularFqeAlgorithm}. */
export interface TabularFqeState {
  readonly Q: Float64Array
  readonly delta: number
  readonly iteration: number
}

/** Empirical model of logged tabular transitions: visit counts, mean rewards and continuation probabilities. */
interface EmpiricalModel {
  readonly counts: Float64Array
  readonly rewards: Float64Array
  /** Sparse continuation probabilities: for each pair, `[next state, probability]` of non-terminal outcomes. */
  readonly next: readonly (readonly (readonly [number, number])[])[]
}

function empiricalModel(transitions: readonly Transition<number, number>[], nS: number, nA: number): EmpiricalModel {
  const counts = new Float64Array(nS * nA)
  const sums = new Float64Array(nS * nA)
  const nexts: Map<number, number>[] = Array.from({ length: nS * nA }, () => new Map<number, number>())
  for (const t of transitions) {
    const i = t.observation * nA + t.action
    if (!(i >= 0 && i < nS * nA && Number.isInteger(i))) {
      throw new OmaiError('value', `tabularFqe: pair (${t.observation}, ${t.action}) is outside the table`)
    }
    counts[i] = (counts[i] ?? 0) + 1
    sums[i] = (sums[i] ?? 0) + t.reward
    if (!t.terminated) {
      const m = nexts[i] as Map<number, number>
      m.set(t.nextObservation, (m.get(t.nextObservation) ?? 0) + 1)
    }
  }
  const rewards = Float64Array.from(sums, (s, i) => ((counts[i] ?? 0) > 0 ? s / (counts[i] ?? 1) : 0))
  const next = nexts.map((m, i) => [...m.entries()].map(([s, c]): [number, number] => [s, c / (counts[i] ?? 1)]))
  return { counts, rewards, next }
}

function policyRows(policy: TabularPolicy, nS: number, nA: number): Float64Array {
  const pi = new Float64Array(nS * nA)
  for (let s = 0; s < nS; s++) {
    const p = policy[s]
    if (typeof p === 'number') pi[s * nA + p] = 1
    else for (let a = 0; a < nA; a++) pi[s * nA + a] = p?.[a] ?? 0
  }
  return pi
}

/**
 * Tabular fitted Q evaluation as an {@link Algorithm}: from logged transitions it forms the empirical model (counts,
 * mean rewards and continuation frequencies; terminal outcomes end the return) and iterates the evaluation operator
 * `Q(s, a) <- r^(s, a) + gamma sum_s' P^(s' | s, a) sum_a' pi(a' | s') Q(s', a')` exactly. Unvisited pairs keep the
 * value 0. Records the largest change and, with an initial distribution, the policy's value.
 *
 * @param transitions - Logged transitions with integer states and actions (any behaviour policy).
 * @param policy - The target policy: one action per state or action probabilities per state.
 * @param options - Table sizes, discount, tolerance and the initial distribution.
 * @returns The algorithm.
 * @concept off-policy-evaluation
 * @example
 * ```ts
 * import { tabularFqeAlgorithm } from 'omai-engine/rl'
 * const t = { observation: 0, action: 0, reward: 1, nextObservation: 0, terminated: true, truncated: false }
 * tabularFqeAlgorithm([t], [0], { nStates: 1, nActions: 1 }).name // => 'tabularFqe'
 * ```
 */
export function tabularFqeAlgorithm(
  transitions: readonly Transition<number, number>[],
  policy: TabularPolicy,
  options: TabularFqeOptions,
): Algorithm<TabularFqeState> {
  const { nStates: nS, nActions: nA } = options
  const gamma = options.gamma ?? 0.99
  const tol = options.tol ?? 1e-10
  const model = empiricalModel(transitions, nS, nA)
  const pi = policyRows(policy, nS, nA)
  const valuesOf = (Q: Float64Array): Float64Array => {
    const V = new Float64Array(nS)
    for (let s = 0; s < nS; s++) {
      let v = 0
      for (let a = 0; a < nA; a++) v += (pi[s * nA + a] ?? 0) * (Q[s * nA + a] ?? 0)
      V[s] = v
    }
    return V
  }
  const startValue = (Q: Float64Array): number => {
    const V = valuesOf(Q)
    let v = 0
    for (const [s, p] of (options.initial ?? []).entries()) v += p * (V[s] ?? 0)
    return v
  }
  return {
    name: 'tabularFqe',
    config: { nStates: nS, nActions: nA, gamma, tol, transitions: transitions.length } as Json,
    schema: channels({
      metrics: {
        delta: { description: 'largest change of the action values', direction: 'lower' },
        startValue: { description: 'value of the initial distribution under the target policy' },
      },
      tensors: { Q: { role: 'state', description: 'action values of the target policy' } },
    }),
    init: () => ({ Q: new Float64Array(nS * nA), delta: Infinity, iteration: 0 }),
    step: (s) => {
      const V = valuesOf(s.Q)
      const Q = new Float64Array(nS * nA)
      let delta = 0
      for (let i = 0; i < nS * nA; i++) {
        if ((model.counts[i] ?? 0) === 0) continue
        let future = 0
        for (const [s2, p] of model.next[i] ?? []) future += p * (V[s2] ?? 0)
        Q[i] = (model.rewards[i] ?? 0) + gamma * future
        delta = Math.max(delta, Math.abs((Q[i] ?? 0) - (s.Q[i] ?? 0)))
      }
      return { Q, delta, iteration: s.iteration + 1 }
    },
    done: (s) => (s.delta < tol ? 'converged' : undefined),
    observe: (s) => ({
      metrics: { delta: Number.isFinite(s.delta) ? s.delta : Number.MAX_VALUE, startValue: startValue(s.Q) },
      tensors: { Q: { role: 'state', value: s.Q, shape: [nS, nA] } },
    }),
  }
}

/** The result of {@link tabularFqe}. */
export interface TabularFqeResult {
  readonly Q: number[][]
  readonly V: number[]
  /** The value of the initial distribution, when one was given. */
  readonly value?: number
  readonly iterations: number
  readonly converged: boolean
}

/**
 * Evaluate a target policy from logged tabular transitions by fitted Q evaluation with exact tabular fits (Le, Voloshin
 * and Yue 2019, tabular case): the fixed point is the policy's value in the empirical MDP of the data.
 *
 * @param transitions - Logged transitions.
 * @param policy - The target policy.
 * @param options - Table sizes, discount, tolerance, initial distribution, iteration cap and the run options.
 * @returns `Q`, `V`, the initial value and convergence.
 * @concept off-policy-evaluation
 * @example
 * ```ts
 * import { tabularFqe } from 'omai-engine/rl'
 * const t = { observation: 0, action: 0, reward: 1, nextObservation: 0, terminated: false, truncated: false }
 * tabularFqe([t], [0], { nStates: 1, nActions: 1, gamma: 0.5 }).V.map((v) => Number(v.toFixed(6))) // => [2]
 * ```
 */
export function tabularFqe(
  transitions: readonly Transition<number, number>[],
  policy: TabularPolicy,
  options: TabularFqeOptions & { readonly maxIter?: number } & Partial<Omit<RunOptions, 'maxSteps'>>,
): TabularFqeResult {
  const { maxIter, nStates, nActions, gamma, tol, initial, ...run } = options
  const solver: TabularFqeOptions = {
    nStates,
    nActions,
    ...(gamma === undefined ? {} : { gamma }),
    ...(tol === undefined ? {} : { tol }),
    ...(initial === undefined ? {} : { initial }),
  }
  const result = runAlgorithm(tabularFqeAlgorithm(transitions, policy, solver), { ...run, maxSteps: maxIter ?? 10_000 })
  const Q = Array.from({ length: nStates }, (_, s) =>
    Array.from(result.state.Q.subarray(s * nActions, (s + 1) * nActions)),
  )
  const pi = policyRows(policy, nStates, nActions)
  const V = Q.map((row, s) => row.reduce((acc, q, a) => acc + (pi[s * nActions + a] ?? 0) * q, 0))
  return {
    Q,
    V,
    ...(initial === undefined ? {} : { value: initial.reduce((acc, p, s) => acc + p * (V[s] ?? 0), 0) }),
    iterations: result.state.iteration,
    converged: result.status === 'converged',
  }
}

/** A fitted regressor: predictions for rows of features. */
export type Predictor = (inputs: readonly (readonly number[])[]) => number[]

/** Options of {@link fittedQEvaluationAlgorithm}. */
export interface FqeOptions<O> {
  readonly nActions: number
  /** Features of a state-action pair, the regression inputs. */
  features(observation: O, action: number): readonly number[]
  /** Fit a regressor to inputs and targets (any learner: least squares, trees, networks). */
  fit(inputs: readonly (readonly number[])[], targets: readonly number[]): Predictor
  /** Discount (default 0.99). */
  readonly gamma?: number
  /** Stop when predictions on the data change by less than this (default 1e-8). */
  readonly tol?: number
  /** Observations whose values average to the policy's value (default the logged first observations). */
  readonly initialObservations?: readonly O[]
}

/** The state of {@link fittedQEvaluationAlgorithm}. */
export interface FqeState {
  /** The current Q function (undefined before the first fit, when Q = 0). */
  readonly predict?: Predictor
  /** Predictions at the logged pairs. */
  readonly current: readonly number[]
  readonly delta: number
  readonly iteration: number
}

/**
 * Fitted Q evaluation (Ernst, Geurts and Wehenkel 2005; Le, Voloshin and Yue 2019) as an {@link Algorithm}: each step
 * regresses the targets `r + gamma (1 - terminated) sum_a' pi(a' | s') Q_k(s', a')` on the features of the logged
 * pairs with the given learner, giving `Q_{k+1}`. The learner is a callback, so any regressor can be plugged in.
 * Records the largest change of the predictions on the data and the policy's value at the initial observations.
 *
 * @param transitions - Logged transitions (any behaviour policy).
 * @param target - The target policy.
 * @param options - Actions, features, the regression callback, discount, tolerance and initial observations.
 * @returns The algorithm.
 * @concept off-policy-evaluation
 * @example
 * ```ts
 * import { fittedQEvaluationAlgorithm } from 'omai-engine/rl'
 * const fit = () => () => [0]
 * fittedQEvaluationAlgorithm([], () => [1], { nActions: 1, features: () => [1], fit }).name // => 'fittedQEvaluation'
 * ```
 */
export function fittedQEvaluationAlgorithm<O>(
  transitions: readonly Transition<O, number>[],
  target: TargetPolicy<O>,
  options: FqeOptions<O>,
): Algorithm<FqeState> {
  const nA = options.nActions
  const gamma = options.gamma ?? 0.99
  const tol = options.tol ?? 1e-8
  const inputs = transitions.map((t) => options.features(t.observation, t.action))
  // Features of every action at every next observation, and the target policy there, computed once.
  const nextInputs = transitions.flatMap((t) =>
    Array.from({ length: nA }, (_, a) => options.features(t.nextObservation, a)),
  )
  const nextProbs = transitions.map((t) => target(t.nextObservation))
  const starts = options.initialObservations ?? (transitions[0] === undefined ? [] : [transitions[0].observation])
  const startInputs = starts.flatMap((o) => Array.from({ length: nA }, (_, a) => options.features(o, a)))
  const startProbs = starts.map((o) => target(o))
  const policyValues = (
    predict: Predictor | undefined,
    xs: readonly (readonly number[])[],
    probs: readonly (readonly number[])[],
  ): number[] => {
    const q = predict === undefined || xs.length === 0 ? new Array<number>(xs.length).fill(0) : predict(xs)
    return probs.map((p, i) => {
      let v = 0
      for (let a = 0; a < nA; a++) v += (p[a] ?? 0) * (q[i * nA + a] ?? 0)
      return v
    })
  }
  return {
    name: 'fittedQEvaluation',
    config: { nActions: nA, gamma, tol, transitions: transitions.length } as Json,
    schema: channels({
      metrics: {
        delta: { description: 'largest change of the predicted values on the data', direction: 'lower' },
        startValue: { description: 'mean value of the initial observations under the target policy' },
      },
    }),
    init: () => ({ current: new Array<number>(transitions.length).fill(0), delta: Infinity, iteration: 0 }),
    step: (s) => {
      const next = policyValues(s.predict, nextInputs, nextProbs)
      const targets = transitions.map((t, i) => t.reward + (t.terminated ? 0 : gamma * (next[i] ?? 0)))
      const predict = options.fit(inputs, targets)
      const current = inputs.length === 0 ? [] : predict(inputs)
      let delta = 0
      for (const [i, q] of current.entries()) delta = Math.max(delta, Math.abs(q - (s.current[i] ?? 0)))
      return { predict, current, delta, iteration: s.iteration + 1 }
    },
    done: (s) => (s.delta < tol ? 'converged' : undefined),
    observe: (s) => {
      const v = policyValues(s.predict, startInputs, startProbs)
      return {
        metrics: {
          delta: Number.isFinite(s.delta) ? s.delta : Number.MAX_VALUE,
          startValue: v.length === 0 ? 0 : meanOf(v),
        },
      }
    },
  }
}

/** The result of {@link fittedQEvaluation}. */
export interface FqeResult<O> {
  /** The fitted action value of a pair. */
  q(observation: O, action: number): number
  /** The target policy's value averaged over the initial observations, with its standard error across them. */
  readonly value: Estimate
  readonly iterations: number
  readonly converged: boolean
}

/**
 * Evaluate a target policy from logged transitions by fitted Q evaluation with any regressor: iterate the regression
 * of bootstrapped targets to convergence (or `maxIter`), then average the policy's value over the initial
 * observations.
 *
 * @param transitions - Logged transitions.
 * @param target - The target policy.
 * @param options - Actions, features, the regression callback, discount, tolerance, initial observations, the
 *   iteration cap (default 200) and the run options.
 * @returns The Q function, the value estimate and convergence.
 * @concept off-policy-evaluation
 * @example
 * ```ts
 * import { fittedQEvaluation } from 'omai-engine/rl'
 * const t = { observation: 0, action: 0, reward: 1, nextObservation: 0, terminated: false, truncated: false }
 * // A constant regressor: the mean target.
 * const fit = (_: unknown, y: readonly number[]) => (xs: readonly unknown[]) => xs.map(() => y.reduce((a, b) => a + b, 0) / y.length)
 * const result = fittedQEvaluation([t], () => [1], { nActions: 1, features: () => [1], fit, gamma: 0.5, tol: 1e-12 })
 * Number(result.value.value.toFixed(6)) // => 2
 * ```
 */
export function fittedQEvaluation<O>(
  transitions: readonly Transition<O, number>[],
  target: TargetPolicy<O>,
  options: FqeOptions<O> & { readonly maxIter?: number } & Partial<Omit<RunOptions, 'maxSteps'>>,
): FqeResult<O> {
  const { maxIter, nActions, features, fit, gamma, tol, initialObservations, ...run } = options
  const fqe: FqeOptions<O> = {
    nActions,
    features,
    fit,
    ...(gamma === undefined ? {} : { gamma }),
    ...(tol === undefined ? {} : { tol }),
    ...(initialObservations === undefined ? {} : { initialObservations }),
  }
  const result = runAlgorithm(fittedQEvaluationAlgorithm(transitions, target, fqe), {
    ...run,
    maxSteps: maxIter ?? 200,
  })
  const predict = result.state.predict
  const q = (o: O, a: number): number => (predict === undefined ? 0 : (predict([features(o, a)])[0] ?? 0))
  const starts = initialObservations ?? (transitions[0] === undefined ? [] : [transitions[0].observation])
  const values = starts.map((o) => target(o).reduce((acc, p, a) => acc + p * q(o, a), 0))
  return {
    q,
    value: values.length === 0 ? estimate(0) : meanEstimate(values),
    iterations: result.state.iteration,
    converged: result.status === 'converged',
  }
}
