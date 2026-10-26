/**
 * Classical controllers as agents for continuous control, planning with a model of the environment rather than learning
 * from reward. The linear-quadratic regulator (`lqr`, the optimal linear feedback for linear dynamics and quadratic
 * cost, from the discrete algebraic Riccati equation) and `lqrAgent`, which linearises a differentiable model by
 * autodiff; an energy-based pendulum swing-up that hands over to LQR near upright; model predictive control re-planning
 * over a receding horizon; and a cross-entropy-method planner.
 */

import { argminOf, clamp, type Json, MdmlError, maxOf } from '../base/index.ts'
import { dlqr, type IlqrOptions, ilqr } from '../dynamics/control/index.ts'
import { type ControlModel, controlModelEnv, linearize, pendulumModel, sequenceCosts } from '../environments/models.ts'
import { type Key, key as makeKey } from '../random/index.ts'
import { type Agent, cem, type Env, mppi, type Transition } from '../rl/index.ts'
import { eye, type TensorLike, toArray, toFlat, type Value } from '../tensor/index.ts'
import { rowsTensor } from './_common.ts'

type Matrix = readonly (readonly number[])[]
const asRows = (x: Value): number[][] => toArray(x) as number[][]

// ------------------------------------------------------------------------------------------------------------ LQR

/** The solution of an infinite-horizon discrete-time LQR problem. */
export interface LqrSolution {
  /** The gain: the optimal control is `u = -K x`. */
  readonly K: Value
  /** The cost-to-go matrix: the optimal cost from `x` is `x^T P x`. */
  readonly P: Value
}

/**
 * The infinite-horizon discrete-time linear-quadratic regulator: minimise `sum x^T Q x + u^T R u` subject to
 * `x' = A x + B u`; `K = (R + B^T P B)^{-1} B^T P A` with `P` from the discrete algebraic Riccati equation. A thin
 * wrapper of the canonical `dlqr` of lib/dynamics/control (which also takes a cross weight and reports the
 * closed-loop eigenvalues).
 *
 * @param a - `A`, `[n, n]`.
 * @param b - `B`, `[n, m]`.
 * @param q - `Q`, `[n, n]` (positive semi-definite).
 * @param r - `R`, `[m, m]` (positive definite).
 * @returns The gain and the cost-to-go matrix.
 * @concept optimal-control
 * @example
 * ```ts
 * import { lqr } from 'mdml-engine/agents'
 * import { item } from 'mdml-engine/tensor'
 * Number(item(lqr([[1]], [[1]], [[1]], [[1]]).K).toFixed(12)) // => 0.618033988750
 * ```
 */
export function lqr(a: TensorLike, b: TensorLike, q: TensorLike, r: TensorLike): LqrSolution {
  const { K, S } = dlqr(a, b, q, r)
  return { K, P: S }
}

// The environment action for a control: the nearest of a discrete environment's control values, else the control
// clipped to the model's bounds.
function toEnvAction<A>(model: ControlModel, u: readonly number[]): A {
  const choices = model.discreteActions
  if (choices !== undefined) {
    const d = choices.map((c) => c.reduce((s, v, i) => s + (v - (u[i] ?? 0)) ** 2, 0))
    return argminOf(d) as A
  }
  return u.map((v, i) => clamp(v, model.actionLow?.[i] ?? -Infinity, model.actionHigh?.[i] ?? Infinity)) as A
}

/** Options of {@link lqrAgent}. */
export interface LqrAgentOptions {
  /** State cost (default the identity). */
  readonly Q?: Matrix
  /** Control cost (default the identity). */
  readonly R?: Matrix
  /** The operating point to linearise at and regulate to (default the model's equilibrium). */
  readonly state?: readonly number[]
  readonly action?: readonly number[]
  /** The model state of an observation (default the observation itself). */
  readonly toState?: (observation: readonly number[]) => readonly number[]
}

/** The state of an LQR agent: its gain and operating point (fixed: the agent does not learn). */
export interface LqrAgentState {
  readonly K: readonly (readonly number[])[]
  readonly state: readonly number[]
  readonly action: readonly number[]
}

/**
 * A linear-quadratic regulator as an agent: linearise the model at its operating point (by autodiff), solve the
 * Riccati equation, and act with `u = u0 - K (x - x0)`, clipped to the action bounds or, for a discrete environment
 * such as CartPole, mapped to the nearest of its actions (bang-bang LQR). The agent's type parameter is the
 * environment's action type (`number` for discrete environments).
 *
 * @param model - The model (for example `cartPoleModel()` or `doubleIntegratorModel()`).
 * @param options - Cost matrices, operating point and the state of an observation.
 * @returns The agent.
 * @throws MdmlError with code `value` when no operating point is given and the model has no equilibrium.
 * @concept optimal-control
 * @example
 * ```ts
 * import { lqrAgent } from 'mdml-engine/agents'
 * import { cartPoleModel } from 'mdml-engine/environments'
 * import { key } from 'mdml-engine/random'
 * const agent = lqrAgent<number>(cartPoleModel())
 * agent.greedy(agent.init(key(0)), [0, 0, 0.05, 0]) // => 1
 * ```
 */
export function lqrAgent<A = readonly number[]>(
  model: ControlModel,
  options: LqrAgentOptions = {},
): Agent<LqrAgentState, readonly number[], A> {
  const x0 = options.state ?? model.equilibrium?.state
  const u0 = options.action ?? model.equilibrium?.action
  if (x0 === undefined || u0 === undefined) throw new MdmlError('value', 'lqrAgent: give an operating point')
  const toState = options.toState ?? ((o: readonly number[]) => o)
  const Q = options.Q ?? asRows(eye(model.stateSize))
  const R = options.R ?? asRows(eye(model.actionSize))
  const policy = (s: LqrAgentState, obs: readonly number[]): A => {
    const x = toState(obs)
    const dx = s.state.map((v, i) => (x[i] ?? 0) - v)
    const u = s.action.map((v, i) => v - (s.K[i] ?? []).reduce((acc, k, j) => acc + k * (dx[j] ?? 0), 0))
    return toEnvAction<A>(model, u)
  }
  return {
    name: 'lqr',
    config: { model: model.id, Q: Q.map((r) => [...r]), R: R.map((r) => [...r]) } as Json,
    init: () => {
      const { A, B } = linearize(model, x0, u0)
      return {
        K: asRows(lqr(A, B, rowsTensor(Q, model.stateSize), rowsTensor(R, model.actionSize)).K),
        state: x0,
        action: u0,
      }
    },
    act: (s, obs) => policy(s, obs),
    greedy: (s, obs) => policy(s, obs),
    learn: (s) => s,
  }
}

// ------------------------------------------------------------------------------------------------------- swing-up

/** Options of {@link swingUpAgent}. */
export interface SwingUpOptions {
  /** Gravity of the pendulum (default 10, gymnasium's). */
  readonly g?: number
  /** Gain of the energy controller (default 1; with the torque limit the pump is nearly bang-bang). */
  readonly energyGain?: number
  /** Hand over to LQR within this angle of upright (default 0.35 rad). */
  readonly switchAngle?: number
  /** LQR costs at upright (default `diag(1, 0.1)` and 0.1). */
  readonly Q?: Matrix
  readonly R?: Matrix
  /** The torque limit (default 2). */
  readonly maxTorque?: number
}

/** The state of a swing-up agent: its LQR gain at upright. */
export interface SwingUpState {
  readonly K: readonly number[]
}

/**
 * Swing up and balance gymnasium's Pendulum (Astrom and Furuta 2000): far from upright, an energy controller pumps the
 * pendulum's energy `E = thetad^2 / 2 + (3 g / 2)(cos theta - 1)` (zero at upright, where `dE/dt = 3 u thetad`)
 * towards zero with `u = sat(k (0 - E) sign(thetad))`; within `switchAngle` of upright an LQR controller from the
 * linearised {@link pendulumModel} catches and holds it. Observations are gymnasium's `[cos, sin, thetad]`. (A cart
 * double pendulum is not covered: energy shaping does not extend to it simply, and swinging it up needs a staged
 * controller or trajectory optimisation from a good initial guess.)
 *
 * @param options - Gravity, energy gain, switching angle, LQR costs and the torque limit.
 * @returns The agent.
 * @concept optimal-control
 * @example
 * ```ts
 * import { swingUpAgent } from 'mdml-engine/agents'
 * import { key } from 'mdml-engine/random'
 * const agent = swingUpAgent()
 * agent.greedy(agent.init(key(0)), [-1, 0, 1]) // => [2]
 * ```
 */
export function swingUpAgent(options: SwingUpOptions = {}): Agent<SwingUpState, readonly number[], readonly number[]> {
  const g = options.g ?? 10
  const k = options.energyGain ?? 1
  const switchAngle = options.switchAngle ?? 0.35
  const umax = options.maxTorque ?? 2
  const model = pendulumModel({ g })
  const Q = options.Q ?? [
    [1, 0],
    [0, 0.1],
  ]
  const R = options.R ?? [[0.1]]
  const policy = (s: SwingUpState, obs: readonly number[]): number[] => {
    const theta = Math.atan2(obs[1] ?? 0, obs[0] ?? 1)
    const w = obs[2] ?? 0
    if (Math.abs(theta) < switchAngle) {
      const u = -((s.K[0] ?? 0) * theta + (s.K[1] ?? 0) * w)
      return [clamp(u, -umax, umax)]
    }
    const energy = 0.5 * w * w + ((3 * g) / 2) * (Math.cos(theta) - 1)
    // Push along the motion to add energy (and against it to remove any excess); at rest, kick in either direction.
    const direction = w >= 0 ? 1 : -1
    return [clamp(k * -energy * direction, -umax, umax)]
  }
  return {
    name: 'swing-up',
    config: { g, energyGain: k, switchAngle, maxTorque: umax } as Json,
    init: () => {
      const { A, B } = linearize(model, [0, 0], [0])
      return { K: toFlat(lqr(A, B, rowsTensor(Q, 2), rowsTensor(R, 1)).K) }
    },
    act: (s, obs) => policy(s, obs),
    greedy: (s, obs) => policy(s, obs),
    learn: (s) => s,
  }
}

// ----------------------------------------------------------------------------------------------------------- iLQR

// iLQR and DDP live in lib/dynamics/control (their canonical home); the rl names are aliases of them. A ControlModel is
// a dynamics DiscreteModel (the same batched dynamics and cost), so the agents below plan with them directly.
export {
  type IlqrOptions,
  type IlqrResult,
  type IlqrState,
  ilqr,
  ilqrAlgorithm,
} from '../dynamics/control/index.ts'

// ------------------------------------------------------------------------------------------------- receding horizon

/** The state of a receding-horizon agent: the current plan, the observation it was made for and its value. */
export interface PlanState {
  readonly plan?: readonly (readonly number[])[]
  /** The observation the plan starts from (as a key), so `act` can reuse a plan made in `learn`. */
  readonly plannedFor?: string
  /** The planner's estimate of the plan's return. */
  readonly value: number
  readonly replans: number
}

const shift = (plan: readonly (readonly number[])[] | undefined): number[][] | undefined =>
  plan === undefined || plan.length === 0
    ? undefined
    : [...plan.slice(1).map((u) => [...u]), [...(plan[plan.length - 1] ?? [])]]

// The agent protocol around a planner: plans are made in `learn` for the next observation, warm-started by shifting
// the previous plan one step, and reused by `act`; a fresh episode plans in `act` from the stale plan.
function recedingHorizon<O, A>(
  name: string,
  config: Json,
  plan: (obs: O, warm: readonly (readonly number[])[] | undefined, key: Key) => { plan: number[][]; value: number },
  toAction: (u: readonly number[]) => A,
): Agent<PlanState, O, A> {
  const id = (o: O): string => JSON.stringify(o)
  const first = (s: PlanState, obs: O, k: Key): A => {
    if (s.plan !== undefined && s.plannedFor === id(obs)) return toAction(s.plan[0] ?? [])
    return toAction(plan(obs, s.plan, k).plan[0] ?? [])
  }
  return {
    name,
    config,
    metrics: { planValue: { description: "the planner's estimate of the current plan's return", direction: 'higher' } },
    init: () => ({ value: 0, replans: 0 }),
    act: first,
    greedy: (s, obs) => first(s, obs, makeKey(0)),
    learn: (s, t: Transition<O, A>, k) => {
      if (t.terminated || t.truncated) return { value: s.value, replans: s.replans }
      const warm = s.plannedFor === id(t.observation) ? shift(s.plan) : s.plan
      const r = plan(t.nextObservation, warm, k)
      return { plan: r.plan, plannedFor: id(t.nextObservation), value: r.value, replans: s.replans + 1 }
    },
    stats: (s) => ({ planValue: s.value }),
  }
}

/** Options of {@link mpcAgent}. */
export interface MpcAgentOptions {
  /** The planner: iLQR on the model (default), or the cross-entropy method or MPPI with batched model rollouts. */
  readonly planner?: 'ilqr' | 'cem' | 'mppi'
  readonly horizon?: number
  /** Planner iterations per decision (default 3 for iLQR and CEM, 1 for MPPI). */
  readonly iterations?: number
  /** Sampling planners: candidates, elites (CEM), sampling standard deviation and temperature (MPPI). */
  readonly population?: number
  readonly elites?: number
  readonly sigma?: number
  readonly lambda?: number
  /** Further iLQR options. */
  readonly ilqr?: Omit<IlqrOptions, 'horizon' | 'initial'>
  /** The model state of an observation (default the observation itself). */
  readonly toState?: (observation: readonly number[]) => readonly number[]
}

/**
 * Model predictive control: at every step, plan over a receding horizon in the differentiable model, apply the first
 * action, and warm-start the next plan with the rest of this one shifted by a step. The planner is iLQR, or CEM or
 * MPPI scoring whole populations of action sequences in one batched model rollout. Plans live in the agent state:
 * `learn` plans for the next observation (so run the agent with `trainAgent`), `act` reuses that plan. The type
 * parameter is the environment's action type (discrete environments get the nearest of their actions).
 *
 * @param model - The differentiable model.
 * @param options - Planner, horizon, iterations, sampling options and the state of an observation.
 * @returns The agent.
 * @concept model-predictive-control
 * @example
 * ```ts
 * import { mpcAgent } from 'mdml-engine/agents'
 * import { pendulumModel } from 'mdml-engine/environments'
 * import { key } from 'mdml-engine/random'
 * const toState = (o: readonly number[]) => [Math.atan2(o[1] ?? 0, o[0] ?? 1), o[2] ?? 0]
 * const agent = mpcAgent(pendulumModel(), { horizon: 10, toState })
 * Math.abs(agent.greedy(agent.init(key(0)), [1, 0.1, 0])[0] ?? 0) <= 2 // => true
 * ```
 */
export function mpcAgent<A = readonly number[]>(
  model: ControlModel,
  options: MpcAgentOptions = {},
): Agent<PlanState, readonly number[], A> {
  const planner = options.planner ?? 'ilqr'
  const horizon = options.horizon ?? 20
  const iterations = options.iterations ?? (planner === 'mppi' ? 1 : 3)
  const toState = options.toState ?? ((o: readonly number[]) => o)
  const env = controlModelEnv(model)
  // Default sampling spread: a quarter of the widest action range (1 for unbounded models).
  const ranges = model.actionHigh?.map((h, i) => (h - (model.actionLow?.[i] ?? -h)) / 4) ?? [1]
  const sigma = options.sigma ?? maxOf(ranges)
  const plan = (obs: readonly number[], warm: readonly (readonly number[])[] | undefined, k: Key) => {
    const x = toState(obs)
    if (planner === 'ilqr') {
      const r = ilqr(model, x, {
        ...options.ilqr,
        horizon,
        ...(warm === undefined ? {} : { initial: warm }),
        maxIterations: iterations,
      })
      return { plan: r.controls, value: -r.cost }
    }
    const score = (plans: readonly (readonly (readonly number[])[])[]): number[] =>
      sequenceCosts(model, x, plans).map((c) => -c)
    const common = {
      horizon,
      population: options.population ?? 64,
      sigma,
      score,
      ...(warm === undefined ? {} : { initial: warm }),
    }
    const r =
      planner === 'cem'
        ? cem(env, x, {
            ...common,
            iterations,
            key: k,
            ...(options.elites === undefined ? {} : { elites: options.elites }),
          })
        : mppi(env, x, { ...common, iterations, key: k, lambda: options.lambda ?? 1 })
    return { plan: r.plan, value: r.value }
  }
  return recedingHorizon<readonly number[], A>(
    `mpc-${planner}`,
    { model: model.id, planner, horizon, iterations } as Json,
    plan,
    (u) => toEnvAction<A>(model, u),
  )
}

/** Options of {@link cemAgent}. */
export interface CemAgentOptions<S, O> {
  /** The environment state of an observation (default the observation itself). */
  readonly toState?: (observation: O) => S
  readonly horizon?: number
  readonly population?: number
  readonly elites?: number
  /** CEM iterations per decision (default 3). */
  readonly iterations?: number
  readonly sigma?: number
  readonly alpha?: number
  readonly gamma?: number
}

/**
 * A cross-entropy-method planning agent that uses the true environment as its model (lib/rl `cem`): at every step it
 * recovers the environment state from the observation, refits a Gaussian over action sequences by simulating
 * candidates in the environment, and applies the first action; the shifted plan warm-starts the next decision. Plans
 * live in the agent state, made in `learn` for the next observation (run it with `trainAgent`).
 *
 * @param env - The environment to plan in (unwrapped, with a box action space).
 * @param options - The state of an observation, and the CEM options.
 * @returns The agent.
 * @concept model-predictive-control
 * @example
 * ```ts
 * import { cemAgent } from 'mdml-engine/agents'
 * import { pendulum } from 'mdml-engine/environments'
 * import { key } from 'mdml-engine/random'
 * const agent = cemAgent(pendulum.create(), { toState: (o) => [Math.atan2(o[1] ?? 0, o[0] ?? 1), o[2] ?? 0] as const, horizon: 5, population: 16 })
 * agent.act(agent.init(key(0)), [1, 0, 0], key(1)).length // => 1
 * ```
 */
export function cemAgent<S, O>(
  env: Env<S, O, readonly number[]>,
  options: CemAgentOptions<S, O> = {},
): Agent<PlanState, O, readonly number[]> {
  const toState = options.toState ?? ((o: O) => o as unknown as S)
  const horizon = options.horizon ?? 15
  const iterations = options.iterations ?? 3
  const plan = (obs: O, warm: readonly (readonly number[])[] | undefined, k: Key) => {
    const r = cem(env, toState(obs), {
      horizon,
      iterations,
      key: k,
      ...(options.population === undefined ? {} : { population: options.population }),
      ...(options.elites === undefined ? {} : { elites: options.elites }),
      ...(options.sigma === undefined ? {} : { sigma: options.sigma }),
      ...(options.alpha === undefined ? {} : { alpha: options.alpha }),
      ...(options.gamma === undefined ? {} : { gamma: options.gamma }),
      ...(warm === undefined ? {} : { initial: warm }),
    })
    return { plan: r.plan, value: r.value }
  }
  return recedingHorizon<O, readonly number[]>(
    'cem',
    { env: env.spec.id, horizon, iterations, population: options.population ?? 64 } as Json,
    plan,
    (u) => [...u],
  )
}
