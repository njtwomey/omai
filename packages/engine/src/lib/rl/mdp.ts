/**
 * Tabular Markov decision processes and their exact solutions: MDPs in arrays, built directly or from an environment's
 * model, one-step Bellman backups and greedy policies, value iteration and policy iteration as traced algorithms, exact
 * policy evaluation by a linear solve, discounted occupancy measures, and discrete environments built from a transition
 * table as gymnasium's toy-text ones are.
 */

import { argmaxOf, type Json, maxOf, OmaiError, toRows } from '../base/index.ts'
import { solve } from '../numerics/linalg/index.ts'
import { uniform } from '../random/index.ts'
import type { Scene } from '../scene/index.ts'
import { dict, discrete, type OptionTable, optionTable } from '../space/index.ts'
import { fromStorage, toFlat } from '../tensor/index.ts'
import {
  type Algorithm,
  type Contract,
  channels,
  defineContract,
  type RunOptions,
  runAlgorithm,
  scalar,
  tensor,
} from '../trace/index.ts'
import type { Env, EnvModel, Outcome } from './env.ts'

/**
 * A finite Markov decision process in arrays: `transitions[(s * nActions + a) * nStates + s']` is the probability of
 * moving to `s'` and continuing, `terminal[s * nActions + a]` the probability that the action ends the episode, and
 * `rewards[s * nActions + a]` the expected immediate reward. Terminal outcomes contribute their reward but no future
 * value, which matches gymnasium's `P[s][a]` tables (`terminated` flags on outcomes) exactly.
 *
 * @concept markov-decision-processes
 */
export interface TabularMdp {
  readonly nStates: number
  readonly nActions: number
  readonly transitions: Float64Array
  readonly terminal: Float64Array
  readonly rewards: Float64Array
  readonly initial: Float64Array
  readonly gamma: number
}

/** A policy over a tabular MDP: one action per state, or action probabilities `[nStates][nActions]`. */
export type TabularPolicy = readonly number[] | readonly (readonly number[])[]

/**
 * Build a tabular MDP from nested arrays.
 *
 * @param spec - `transitions[s][a][s']` probabilities, `rewards[s][a]` expected rewards (or `rewards[s][a][s']`),
 *   optional absorbing `terminalStates`, an initial distribution (default uniform) and the discount (default 1).
 * @returns The MDP.
 * @throws OmaiError with code `value` when a row of probabilities does not sum to one.
 * @concept markov-decision-processes
 * @index structure A tabular MDP from transitions and rewards.
 * @example
 * ```ts
 * import { tabularMdp } from 'omai-engine/rl'
 * const mdp = tabularMdp({ transitions: [[[1, 0], [0, 1]], [[0, 1], [0, 1]]], rewards: [[0, 1], [0, 0]], terminalStates: [1], gamma: 0.9 })
 * mdp.nStates // => 2
 * ```
 */
export function tabularMdp(spec: {
  readonly transitions: readonly (readonly (readonly number[])[])[]
  readonly rewards: readonly (readonly (number | readonly number[])[])[]
  readonly terminalStates?: readonly number[]
  readonly initial?: readonly number[]
  readonly gamma?: number
}): TabularMdp {
  const nS = spec.transitions.length
  const nA = spec.transitions[0]?.length ?? 0
  const P = new Float64Array(nS * nA * nS)
  const T = new Float64Array(nS * nA)
  const R = new Float64Array(nS * nA)
  const absorbing = new Set(spec.terminalStates ?? [])
  for (let s = 0; s < nS; s++) {
    for (let a = 0; a < nA; a++) {
      const row = spec.transitions[s]?.[a] ?? []
      let total = 0
      for (let s2 = 0; s2 < nS; s2++) total += row[s2] ?? 0
      if (Math.abs(total - 1) > 1e-9) {
        throw new OmaiError('value', `tabularMdp: transitions[${s}][${a}] sums to ${total}, not 1`)
      }
      const r = spec.rewards[s]?.[a] ?? 0
      let er = 0
      for (let s2 = 0; s2 < nS; s2++) {
        const p = row[s2] ?? 0
        er += p * (typeof r === 'number' ? r : (r[s2] ?? 0))
        if (absorbing.has(s2) || absorbing.has(s)) T[s * nA + a] = (T[s * nA + a] ?? 0) + p
        else P[(s * nA + a) * nS + s2] = p
      }
      R[s * nA + a] = absorbing.has(s) ? 0 : er
    }
  }
  const initial = Float64Array.from(spec.initial ?? new Array<number>(nS).fill(1 / nS))
  return { nStates: nS, nActions: nA, transitions: P, terminal: T, rewards: R, initial, gamma: spec.gamma ?? 1 }
}

/**
 * The tabular MDP of an environment model (gymnasium's `P` table), with every outcome's reward and `terminated` flag.
 *
 * @param model - The model (states, actions, initial distribution, outcomes).
 * @param gamma - The discount (default 1).
 * @returns The MDP.
 * @concept markov-decision-processes
 * @example
 * ```ts
 * import { mdpFromModel } from 'omai-engine/rl'
 * import { frozenLake } from 'omai-engine/environments'
 * const model = frozenLake.create().model
 * model === undefined ? 0 : mdpFromModel(model, 0.99).nStates // => 16
 * ```
 */
export function mdpFromModel<S, A>(model: EnvModel<S, A>, gamma = 1): TabularMdp {
  const nS = model.states.length
  const nA = model.actions.length
  const P = new Float64Array(nS * nA * nS)
  const T = new Float64Array(nS * nA)
  const R = new Float64Array(nS * nA)
  for (const [si, s] of model.states.entries()) {
    for (const [ai, a] of model.actions.entries()) {
      for (const o of model.transitions(s, a)) {
        const j = si * nA + ai
        R[j] = (R[j] ?? 0) + o.probability * o.reward
        if (o.terminated) T[j] = (T[j] ?? 0) + o.probability
        else {
          const k = j * nS + model.stateIndex(o.state)
          P[k] = (P[k] ?? 0) + o.probability
        }
      }
    }
  }
  const initial = new Float64Array(nS)
  for (const { probability, state } of model.initial) {
    const i = model.stateIndex(state)
    initial[i] = (initial[i] ?? 0) + probability
  }
  return { nStates: nS, nActions: nA, transitions: P, terminal: T, rewards: R, initial, gamma }
}

/**
 * The tabular MDP of an environment that exposes a model.
 *
 * @param env - The environment.
 * @param gamma - The discount (default 1).
 * @returns The MDP.
 * @throws OmaiError with code `value` when the environment has no model.
 * @concept markov-decision-processes
 * @example
 * ```ts
 * import { mdpFromEnv } from 'omai-engine/rl'
 * import { cliffWalking } from 'omai-engine/environments'
 * mdpFromEnv(cliffWalking.create()).nActions // => 4
 * ```
 */
export function mdpFromEnv<S, O, A>(env: Env<S, O, A>, gamma = 1): TabularMdp {
  if (env.model === undefined) throw new OmaiError('value', `mdpFromEnv: ${env.spec.id} has no model`)
  return mdpFromModel(env.model, gamma)
}

function probabilities(mdp: TabularMdp, policy: TabularPolicy): Float64Array {
  const { nStates: nS, nActions: nA } = mdp
  const pi = new Float64Array(nS * nA)
  for (let s = 0; s < nS; s++) {
    const p = policy[s]
    if (typeof p === 'number') pi[s * nA + p] = 1
    else for (let a = 0; a < nA; a++) pi[s * nA + a] = p?.[a] ?? 0
  }
  return pi
}

function backup(mdp: TabularMdp, V: ArrayLike<number>, gamma: number): Float64Array {
  const { nStates: nS, nActions: nA, transitions: P, rewards: R } = mdp
  const Q = new Float64Array(nS * nA)
  for (let j = 0; j < nS * nA; j++) {
    let q = R[j] ?? 0
    let future = 0
    const base = j * nS
    for (let s2 = 0; s2 < nS; s2++) future += (P[base + s2] ?? 0) * (V[s2] ?? 0)
    q += gamma * future
    Q[j] = q
  }
  return Q
}

/**
 * Action values from state values by one Bellman backup: `Q(s, a) = R(s, a) + gamma sum_s' P(s' | s, a) V(s')`.
 *
 * @param mdp - The MDP.
 * @param V - State values.
 * @param gamma - The discount (default the MDP's).
 * @returns `Q[s][a]`.
 * @concept markov-decision-processes
 * @example
 * ```ts
 * import { qFromV, tabularMdp } from 'omai-engine/rl'
 * const mdp = tabularMdp({ transitions: [[[1, 0], [0, 1]], [[0, 1], [0, 1]]], rewards: [[0, 1], [0, 0]], terminalStates: [1] })
 * qFromV(mdp, [0, 0]) // => [[0, 1], [0, 0]]
 * ```
 */
export function qFromV(mdp: TabularMdp, V: readonly number[], gamma: number = mdp.gamma): number[][] {
  return toRows(backup(mdp, V, gamma), mdp.nStates, mdp.nActions)
}

/**
 * The greedy policy of action values: the first maximising action in every state.
 *
 * @param Q - `Q[s][a]`.
 * @returns One action per state.
 * @concept markov-decision-processes
 * @example
 * ```ts
 * import { greedyPolicy } from 'omai-engine/rl'
 * greedyPolicy([[0, 2, 1], [3, 3, 0]]) // => [1, 0]
 * ```
 */
export function greedyPolicy(Q: readonly (readonly number[])[]): number[] {
  return Q.map((q) => argmaxOf(q))
}

/** Options of the iterative MDP solvers. */
export interface MdpSolveOptions {
  /** Stop when the largest value change falls below this (default 1e-12). */
  readonly tol?: number
  /** Iteration cap (default 10000). */
  readonly maxIter?: number
  /** The discount (default the MDP's). */
  readonly gamma?: number
}

/** The state of {@link valueIterationAlgorithm}. */
export interface ValueIterationState {
  readonly V: Float64Array
  readonly delta: number
  readonly iteration: number
}

/**
 * Value iteration as an {@link Algorithm}: each step applies the Bellman optimality operator
 * `V(s) <- max_a Q(s, a)` and records the largest change (the sup-norm residual) and the value of the start states.
 *
 * @param mdp - The MDP.
 * @param options - Tolerance and discount.
 * @returns The algorithm.
 * @concept value-iteration
 * @index algorithm Value iteration, one sweep per step.
 * @example
 * ```ts
 * import { tabularMdp, valueIterationAlgorithm } from 'omai-engine/rl'
 * const mdp = tabularMdp({ transitions: [[[1, 0], [0, 1]], [[0, 1], [0, 1]]], rewards: [[0, 1], [0, 0]], terminalStates: [1] })
 * valueIterationAlgorithm(mdp).name // => 'valueIteration'
 * ```
 */
export function valueIterationAlgorithm(
  mdp: TabularMdp,
  options: MdpSolveOptions = {},
): Algorithm<ValueIterationState> {
  const tol = options.tol ?? 1e-12
  const gamma = options.gamma ?? mdp.gamma
  const { nStates: nS, nActions: nA } = mdp
  return {
    name: 'valueIteration',
    config: { nStates: nS, nActions: nA, gamma, tol } as Json,
    schema: channels({
      metrics: {
        delta: { description: 'largest change in the state values', direction: 'lower' },
        startValue: { description: 'expected value of the initial state distribution', direction: 'higher' },
      },
      tensors: { V: { role: 'state', description: 'state values' } },
    }),
    init: () => ({ V: new Float64Array(nS), delta: Infinity, iteration: 0 }),
    step: (s) => {
      const Q = backup(mdp, s.V, gamma)
      const V = new Float64Array(nS)
      let delta = 0
      for (let i = 0; i < nS; i++) {
        V[i] = maxOf(Q.subarray(i * nA, (i + 1) * nA))
        delta = Math.max(delta, Math.abs((V[i] ?? 0) - (s.V[i] ?? 0)))
      }
      return { V, delta, iteration: s.iteration + 1 }
    },
    done: (s) => (s.delta < tol ? 'converged' : undefined),
    observe: (s) => {
      let start = 0
      for (let i = 0; i < nS; i++) start += (mdp.initial[i] ?? 0) * (s.V[i] ?? 0)
      return {
        metrics: { delta: Number.isFinite(s.delta) ? s.delta : Number.MAX_VALUE, startValue: start },
        tensors: { V: { role: 'state', value: s.V } },
      }
    },
  }
}

/** The solution of a tabular MDP. */
export interface MdpSolution {
  readonly V: number[]
  readonly Q: number[][]
  readonly policy: number[]
  readonly iterations: number
  readonly converged: boolean
}

/**
 * Solve an MDP by value iteration (Bellman 1957): iterate the optimality operator to a fixed point, then act greedily.
 *
 * @param mdp - The MDP.
 * @param options - Tolerance, iteration cap, discount and the run options (tracer, policy).
 * @returns Optimal values, action values and a greedy policy.
 * @concept value-iteration
 * @index function Exact value iteration on a tabular MDP.
 * @example
 * ```ts
 * import { tabularMdp, valueIteration } from 'omai-engine/rl'
 * const mdp = tabularMdp({ transitions: [[[1, 0], [0, 1]], [[0, 1], [0, 1]]], rewards: [[0, 1], [0, 0]], terminalStates: [1] })
 * valueIteration(mdp).V // => [1, 0]
 * ```
 */
export function valueIteration(
  mdp: TabularMdp,
  options: MdpSolveOptions & Partial<Omit<RunOptions, 'maxSteps'>> = {},
): MdpSolution {
  const { tol, maxIter, gamma, ...run } = options
  const solver: MdpSolveOptions = {
    ...(tol === undefined ? {} : { tol }),
    ...(gamma === undefined ? {} : { gamma }),
  }
  const result = runAlgorithm(valueIterationAlgorithm(mdp, solver), { ...run, maxSteps: maxIter ?? 10_000 })
  const V = Array.from(result.state.V)
  const Q = qFromV(mdp, V, gamma ?? mdp.gamma)
  return { V, Q, policy: greedyPolicy(Q), iterations: result.state.iteration, converged: result.status === 'converged' }
}

/** The reward vector and transition matrix of the chain a policy induces: `V = Rpi + gamma Ppi V`. */
function policyChain(mdp: TabularMdp, policy: TabularPolicy): { Rpi: Float64Array; Ppi: Float64Array } {
  const { nStates: nS, nActions: nA, transitions: P, rewards: R } = mdp
  const pi = probabilities(mdp, policy)
  const Rpi = new Float64Array(nS)
  const Ppi = new Float64Array(nS * nS)
  for (let s = 0; s < nS; s++) {
    for (let a = 0; a < nA; a++) {
      const w = pi[s * nA + a] ?? 0
      if (w === 0) continue
      Rpi[s] = (Rpi[s] ?? 0) + w * (R[s * nA + a] ?? 0)
      const base = (s * nA + a) * nS
      for (let s2 = 0; s2 < nS; s2++) Ppi[s * nS + s2] = (Ppi[s * nS + s2] ?? 0) + w * (P[base + s2] ?? 0)
    }
  }
  return { Rpi, Ppi }
}

/** The paths policy evaluation logs, as a type. */
export type PolicyEvaluationPath = 'eval/sweep' | 'eval/delta' | 'eval/V'

/**
 * What iterative policy evaluation logs, by path: the sweep just made, the largest change in a state value in it (the
 * sup-norm residual) and the state values after it.
 */
export const policyEvaluationContract: Contract<unknown, PolicyEvaluationPath> = defineContract({
  'eval/sweep': scalar({ description: 'the Bellman sweep just made, counting from 1' }),
  'eval/delta': scalar({ description: 'the largest change in a state value in the sweep' }),
  'eval/V': tensor({ role: 'state', shape: ['states'], description: 'the state values after the sweep' }),
})

/** The state of {@link policyEvaluationAlgorithm}. */
export interface PolicyEvaluationState {
  readonly V: Float64Array
  readonly delta: number
  readonly iteration: number
}

/** How a sweep updates the values: from the values of the last sweep, or in place as it goes. */
export type SweepOrder = 'synchronous' | 'in-place'

/** What a sweep order can do, for validity rules and selectors. */
export interface SweepProperties {
  /** Whether a sweep reads only the values of the previous sweep (Jacobi), not the ones it has just written. */
  readonly usesPreviousSweep: boolean
}

/**
 * The sweep orders of iterative policy evaluation as one slot: `synchronous` (every state backed up from the previous
 * sweep's values, as a matrix-vector product) and `in-place` (Gauss-Seidel: states in index order, each using the
 * newest values, which usually converges in fewer sweeps).
 *
 * @concept policy-evaluation
 * @example
 * ```ts
 * import { sweepOrders } from 'omai-engine/rl'
 * sweepOrders.where((p) => p.usesPreviousSweep) // => ['synchronous']
 * ```
 */
export const sweepOrders: OptionTable<SweepOrder, SweepProperties> = optionTable(
  {
    synchronous: {
      description: 'every state is backed up from the values of the previous sweep',
      properties: { usesPreviousSweep: true },
      space: dict({}),
      make: () => 'synchronous',
    },
    'in-place': {
      description: 'states in index order, each backed up from the newest values (Gauss-Seidel)',
      properties: { usesPreviousSweep: false },
      space: dict({}),
      make: () => 'in-place',
    },
  },
  {
    default: 'synchronous',
    label: 'sweep order',
    description: 'Whether a sweep of policy evaluation reads the old values or the ones it has just written.',
  },
)

/**
 * Iterative policy evaluation as an {@link Algorithm}: each step is one sweep of Bellman expectation backups
 * `V(s) <- R_pi(s) + gamma sum_s' P_pi(s, s') V(s')` over every state, from `V = 0`. It is done when the largest
 * change in a sweep falls below the tolerance.
 *
 * @param mdp - The MDP.
 * @param policy - One action per state, or action probabilities per state.
 * @param options - `tol` (default 1e-12), `gamma` (default the MDP's) and the `sweep` order (default synchronous).
 * @returns The algorithm.
 * @concept policy-evaluation
 * @example
 * ```ts
 * import { policyEvaluationAlgorithm, tabularMdp } from 'omai-engine/rl'
 * import { runAlgorithm } from 'omai-engine/trace'
 * const mdp = tabularMdp({ transitions: [[[1, 0], [0, 1]], [[0, 1], [0, 1]]], rewards: [[0, 1], [0, 0]], terminalStates: [1], gamma: 0.5 })
 * Array.from(runAlgorithm(policyEvaluationAlgorithm(mdp, [1, 0]), { maxSteps: 50 }).state.V) // => [1, 0]
 * ```
 */
export function policyEvaluationAlgorithm(
  mdp: TabularMdp,
  policy: TabularPolicy,
  options: MdpSolveOptions & { readonly sweep?: SweepOrder } = {},
): Algorithm<PolicyEvaluationState, PolicyEvaluationPath> {
  const gamma = options.gamma ?? mdp.gamma
  const tol = options.tol ?? 1e-12
  const order = sweepOrders.make({ kind: options.sweep ?? 'synchronous' })
  const nS = mdp.nStates
  const { Rpi, Ppi } = policyChain(mdp, policy)
  return {
    name: 'policyEvaluation',
    config: { nStates: nS, nActions: mdp.nActions, gamma, tol, sweep: order } as Json,
    init: () => ({ V: new Float64Array(nS), delta: Number.POSITIVE_INFINITY, iteration: 0 }),
    step: (s, ctx) => {
      const V = order === 'in-place' ? Float64Array.from(s.V) : new Float64Array(nS)
      const read = order === 'in-place' ? V : s.V
      let delta = 0
      for (let i = 0; i < nS; i++) {
        let v = Rpi[i] ?? 0
        for (let j = 0; j < nS; j++) v += gamma * (Ppi[i * nS + j] ?? 0) * (read[j] ?? 0)
        delta = Math.max(delta, Math.abs(v - (s.V[i] ?? 0)))
        V[i] = v
      }
      const iteration = s.iteration + 1
      if (ctx.wants('eval/sweep')) ctx.log('eval/sweep', iteration)
      if (ctx.wants('eval/delta')) ctx.log('eval/delta', delta)
      if (ctx.wants('eval/V')) ctx.log('eval/V', Float64Array.from(V))
      return { V, delta, iteration }
    },
    done: (s) => (s.delta < tol ? 'converged' : undefined),
    serialise: (s) => ({
      V: Array.from(s.V),
      delta: Number.isFinite(s.delta) ? s.delta : 'Infinity',
      iteration: s.iteration,
    }),
    deserialise: (j) => {
      const o = j as { readonly V: readonly number[]; readonly delta: number | string; readonly iteration: number }
      return { V: Float64Array.from(o.V), delta: Number(o.delta), iteration: o.iteration }
    },
  }
}

/**
 * The values of a fixed policy: exactly, by solving `(I - gamma P_pi) V = R_pi` (the default), or iteratively by
 * repeated Bellman expectation backups ({@link policyEvaluationAlgorithm} run to the end).
 *
 * @param mdp - The MDP.
 * @param policy - One action per state, or action probabilities per state.
 * @param options - `method` (`'exact'` or `'iterative'`), tolerance, iteration cap, discount and the `sweep` order.
 * @returns The state values.
 * @throws OmaiError with code `value` when the exact system is singular (an improper policy with `gamma = 1`).
 * @concept policy-evaluation
 * @example
 * ```ts
 * import { policyEvaluation, tabularMdp } from 'omai-engine/rl'
 * const mdp = tabularMdp({ transitions: [[[1, 0], [0, 1]], [[0, 1], [0, 1]]], rewards: [[0, 1], [0, 0]], terminalStates: [1], gamma: 0.5 })
 * policyEvaluation(mdp, [0, 0]) // => [0, 0]
 * ```
 */
export function policyEvaluation(
  mdp: TabularMdp,
  policy: TabularPolicy,
  options: MdpSolveOptions & { readonly method?: 'exact' | 'iterative'; readonly sweep?: SweepOrder } = {},
): number[] {
  const gamma = options.gamma ?? mdp.gamma
  const nS = mdp.nStates
  if ((options.method ?? 'exact') === 'exact') {
    const { Rpi, Ppi } = policyChain(mdp, policy)
    const A = new Float64Array(nS * nS)
    for (let i = 0; i < nS; i++)
      for (let j = 0; j < nS; j++) A[i * nS + j] = (i === j ? 1 : 0) - gamma * (Ppi[i * nS + j] ?? 0)
    try {
      return toFlat(solve(fromStorage(A, [nS, nS], 'float64'), fromStorage(Rpi, [nS], 'float64')))
    } catch {
      throw new OmaiError('value', 'policyEvaluation: I - gamma P_pi is singular (the policy never terminates)')
    }
  }
  const run = runAlgorithm(policyEvaluationAlgorithm(mdp, policy, options), { maxSteps: options.maxIter ?? 100_000 })
  return Array.from(run.state.V)
}

/** The state of {@link policyIterationAlgorithm}. */
export interface PolicyIterationState {
  readonly policy: readonly number[]
  readonly V: readonly number[]
  readonly changed: number
  readonly iteration: number
}

/**
 * Policy iteration (Howard 1960) as an {@link Algorithm}: each step evaluates the current policy exactly and improves
 * it greedily, keeping the current action on ties so the iteration terminates; it converges when no action changes.
 *
 * @param mdp - The MDP.
 * @param options - The starting policy (default action 0 everywhere; it must terminate when `gamma = 1`) and discount.
 * @returns The algorithm.
 * @concept policy-iteration
 * @example
 * ```ts
 * import { policyIterationAlgorithm, tabularMdp } from 'omai-engine/rl'
 * const mdp = tabularMdp({ transitions: [[[1, 0], [0, 1]], [[0, 1], [0, 1]]], rewards: [[0, 1], [0, 0]], terminalStates: [1], gamma: 0.9 })
 * policyIterationAlgorithm(mdp).name // => 'policyIteration'
 * ```
 */
export function policyIterationAlgorithm(
  mdp: TabularMdp,
  options: { readonly policy?: readonly number[]; readonly gamma?: number } = {},
): Algorithm<PolicyIterationState> {
  const gamma = options.gamma ?? mdp.gamma
  const { nStates: nS, nActions: nA } = mdp
  const start = options.policy ?? new Array<number>(nS).fill(0)
  return {
    name: 'policyIteration',
    config: { nStates: nS, nActions: nA, gamma } as Json,
    schema: channels({
      metrics: {
        changed: { description: 'actions changed by the improvement step', direction: 'lower' },
        startValue: { description: 'expected value of the current policy from the start', direction: 'higher' },
      },
      tensors: { V: { role: 'state', description: 'values of the current policy' } },
    }),
    init: () => ({ policy: start, V: policyEvaluation(mdp, start, { gamma }), changed: nS, iteration: 0 }),
    step: (s) => {
      const Q = backup(mdp, s.V, gamma)
      let changed = 0
      const policy = s.policy.map((current, i) => {
        const q = Q.subarray(i * nA, (i + 1) * nA)
        const best = maxOf(q)
        // Keep the current action unless another is strictly (beyond rounding) better.
        if ((q[current] ?? -Infinity) >= best - 1e-12 * Math.max(1, Math.abs(best))) return current
        changed += 1
        return argmaxOf(q)
      })
      return { policy, V: policyEvaluation(mdp, policy, { gamma }), changed, iteration: s.iteration + 1 }
    },
    done: (s) => (s.iteration > 0 && s.changed === 0 ? 'converged' : undefined),
    observe: (s) => {
      let v = 0
      for (let i = 0; i < nS; i++) v += (mdp.initial[i] ?? 0) * (s.V[i] ?? 0)
      return { metrics: { changed: s.changed, startValue: v }, tensors: { V: { role: 'state', value: s.V } } }
    },
  }
}

/**
 * Solve an MDP by policy iteration.
 *
 * @param mdp - The MDP.
 * @param options - Starting policy, iteration cap, discount and run options.
 * @returns Optimal values, action values and policy.
 * @concept policy-iteration
 * @example
 * ```ts
 * import { policyIteration, tabularMdp } from 'omai-engine/rl'
 * const mdp = tabularMdp({ transitions: [[[1, 0], [0, 1]], [[0, 1], [0, 1]]], rewards: [[0, 1], [0, 0]], terminalStates: [1], gamma: 0.9 })
 * policyIteration(mdp).policy // => [1, 0]
 * ```
 */
export function policyIteration(
  mdp: TabularMdp,
  options: { readonly policy?: readonly number[]; readonly gamma?: number; readonly maxIter?: number } & Partial<
    Omit<RunOptions, 'maxSteps'>
  > = {},
): MdpSolution {
  const { policy, gamma, maxIter, ...run } = options
  const alg = policyIterationAlgorithm(mdp, {
    ...(policy === undefined ? {} : { policy }),
    ...(gamma === undefined ? {} : { gamma }),
  })
  const result = runAlgorithm(alg, { ...run, maxSteps: maxIter ?? 1000 })
  const V = [...result.state.V]
  return {
    V,
    Q: qFromV(mdp, V, gamma ?? mdp.gamma),
    policy: [...result.state.policy],
    iterations: result.state.iteration,
    converged: result.status === 'converged',
  }
}

/**
 * The discounted occupancy measure of a policy: the expected discounted number of visits to each state,
 * `d = mu0^T (I - gamma P_pi)^{-1}`, and to each state-action pair, `d(s) pi(a | s)`. With `normalize` the measures
 * are scaled to sum to one (the discounted state distribution of policy-gradient theorems).
 *
 * @param mdp - The MDP.
 * @param policy - The policy.
 * @param options - Discount and normalisation.
 * @returns State and state-action occupancies.
 * @concept markov-decision-processes
 * @example
 * ```ts
 * import { occupancyMeasure, tabularMdp } from 'omai-engine/rl'
 * const mdp = tabularMdp({ transitions: [[[1, 0], [0, 1]], [[0, 1], [0, 1]]], rewards: [[0, 1], [0, 0]], terminalStates: [1], initial: [1, 0], gamma: 0.5 })
 * occupancyMeasure(mdp, [0, 0]).state // => [2, 0]
 * ```
 */
export function occupancyMeasure(
  mdp: TabularMdp,
  policy: TabularPolicy,
  options: { readonly gamma?: number; readonly normalize?: boolean } = {},
): { readonly state: number[]; readonly stateAction: number[][] } {
  const gamma = options.gamma ?? mdp.gamma
  const { nStates: nS, nActions: nA, transitions: P } = mdp
  const pi = probabilities(mdp, policy)
  // Solve (I - gamma P_pi)^T d = mu0.
  const A = new Float64Array(nS * nS)
  for (let s = 0; s < nS; s++) {
    for (let a = 0; a < nA; a++) {
      const w = pi[s * nA + a] ?? 0
      if (w === 0) continue
      const base = (s * nA + a) * nS
      for (let s2 = 0; s2 < nS; s2++) A[s2 * nS + s] = (A[s2 * nS + s] ?? 0) - gamma * w * (P[base + s2] ?? 0)
    }
  }
  for (let i = 0; i < nS; i++) A[i * nS + i] = (A[i * nS + i] ?? 0) + 1
  const d = toFlat(solve(fromStorage(A, [nS, nS], 'float64'), fromStorage(mdp.initial, [nS], 'float64')))
  let total = 0
  for (const x of d) total += x
  const scale = options.normalize === true && total > 0 ? 1 / total : 1
  const state = d.map((x) => x * scale)
  const stateAction = state.map((x, s) => Array.from({ length: nA }, (_, a) => x * (pi[s * nA + a] ?? 0)))
  return { state, stateAction }
}

// TODO: linear-programming solutions of MDPs (the primal LP over values and the dual LP over occupancy measures) wait
// for `linprog` in optim/linear-programming.ts (section 4.18), which is not built yet.

/** What {@link tabularEnv} needs to build an environment. */
export interface TabularEnvSpec {
  readonly id: string
  readonly description: string
  readonly nStates: number
  readonly nActions: number
  /** The outcomes of an action (gymnasium's `P[s][a]`). */
  transitions(state: number, action: number): readonly Outcome<number>[]
  /** Initial-state probabilities. */
  readonly initial: readonly number[]
  render(state: number): Scene
  /** The observation of a state (default the state itself). */
  observe?(state: number): number
  readonly maxEpisodeSteps?: number
  readonly rewardThreshold?: number
  readonly stateLabel?: string
  readonly actionLabel?: string
  /** Extra information per step (gymnasium puts the outcome probability in `info.prob`). */
  info?(state: number): Readonly<Record<string, Json>>
}

/** gymnasium's `categorical_sample`: the first index whose cumulative probability exceeds a uniform draw. */
function categoricalIndex(probs: readonly number[], u: number): number {
  let c = 0
  for (const [i, p] of probs.entries()) {
    c += p
    if (c > u) return i
  }
  return probs.length - 1
}

/**
 * A discrete environment from its transition table (gymnasium's toy-text environments): states and actions are
 * integers, `step` samples an outcome of `transitions(s, a)` (by inverse CDF, as gymnasium's `categorical_sample`),
 * and the table is exposed as the environment's `model`, so value iteration and the other planners can solve it.
 *
 * @param spec - The table, initial distribution, renderer and spec fields.
 * @returns The environment.
 * @concept environments
 * @example
 * ```ts
 * import { tabularEnv } from 'omai-engine/rl'
 * import { scene } from 'omai-engine/scene'
 * import { key } from 'omai-engine/random'
 * const coin = tabularEnv({
 *   id: 'Coin-v0', description: 'one flip', nStates: 2, nActions: 1, initial: [1, 0], render: () => scene(1, 1, []),
 *   transitions: () => [{ probability: 1, state: 1, reward: 1, terminated: true }],
 * })
 * coin.step(0, 0, key(0)).reward // => 1
 * ```
 */
export function tabularEnv(spec: TabularEnvSpec): Env<number, number, number> {
  const states = Array.from({ length: spec.nStates }, (_, i) => i)
  const actions = Array.from({ length: spec.nActions }, (_, i) => i)
  const observe = spec.observe ?? ((s: number) => s)
  const info = spec.info
  return {
    spec: {
      id: spec.id,
      description: spec.description,
      ...(spec.maxEpisodeSteps === undefined ? {} : { maxEpisodeSteps: spec.maxEpisodeSteps }),
      ...(spec.rewardThreshold === undefined ? {} : { rewardThreshold: spec.rewardThreshold }),
    },
    observationSpace: discrete(spec.nStates, spec.stateLabel === undefined ? {} : { label: spec.stateLabel }),
    actionSpace: discrete(spec.nActions, spec.actionLabel === undefined ? {} : { label: spec.actionLabel }),
    reset: (k) => {
      const s = categoricalIndex(spec.initial, uniform(k, []).data[0] ?? 0)
      return { state: s, observation: observe(s), info: { prob: 1, ...(info === undefined ? {} : info(s)) } }
    },
    step: (state, action, k) => {
      const outcomes = spec.transitions(state, action)
      const i = categoricalIndex(
        outcomes.map((o) => o.probability),
        uniform(k, []).data[0] ?? 0,
      )
      const o = outcomes[i] as Outcome<number>
      return {
        state: o.state,
        observation: observe(o.state),
        reward: o.reward,
        terminated: o.terminated,
        truncated: false,
        info: { prob: o.probability, ...(info === undefined ? {} : info(o.state)) },
      }
    },
    render: spec.render,
    observe,
    model: {
      states,
      actions,
      stateIndex: (s) => s,
      actionIndex: (a) => a,
      initial: spec.initial.flatMap((p, s) => (p > 0 ? [{ probability: p, state: s }] : [])),
      transitions: (s, a) => spec.transitions(s, a),
    },
  }
}
