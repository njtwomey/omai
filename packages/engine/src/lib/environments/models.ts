/**
 * Differentiable models of the control environments, for planners and model-based controllers: each environment's
 * dynamics and per-step cost written with tensor operations so that autodiff can differentiate them (CartPole,
 * Pendulum, MountainCarContinuous, Acrobot and the continuous-control tasks). Linearisation at an operating point,
 * trajectory derivatives for iLQR, batched rollouts and sequence costs for sampling planners such as CEM and MPPI, a
 * model seen as an environment, and a check that a model agrees with its environment.
 *
 * @index structure Differentiable models of control environments for planners.
 */

import { grad, jacfwd, jvp } from '../autodiff/index.ts'
import { OmaiError } from '../base/index.ts'
import type { Key } from '../random/index.ts'
import type { Env } from '../rl/index.ts'
import { scene } from '../scene/index.ts'
import { box } from '../space/index.ts'
import {
  add,
  fromStorage,
  mod,
  sliceAxis,
  squeeze,
  stack,
  sub,
  sum,
  type Tensor,
  type TensorLike,
  toFlat,
  type Value,
  zeros,
} from '../tensor/index.ts'
import {
  armDerivatives,
  cartDoublePendulumDerivativesOf,
  cartDoublePendulumRewardOf,
  dampedMassCoefficients,
  dampedMassStep,
  valueOps as o,
  pointMassRewardOf,
  quadrotorDerivatives,
  quadrotorRewardOf,
  reacherRewardOf,
  rk4,
} from './_mechanics.ts'
import {
  type CartDoublePendulumOptions,
  cartDoublePendulumParameters,
  type DoubleIntegratorOptions,
  doubleIntegratorParameters,
  type PlanarQuadrotorOptions,
  type PointMassOptions,
  planarQuadrotorParameters,
  pointMassParameters,
  type Reacher2DOptions,
  reacher2dParameters,
} from './continuous-control.ts'

/**
 * A differentiable model of a control system: `dynamics(x, u)` is one step of the environment's state and `cost(x, u)`
 * its per-step cost, both written with lib/tensor operations so lib/autodiff differentiates them (LQR, iLQR, MPC,
 * model-based agents). States `[..., stateSize]` and actions `[..., actionSize]` may carry leading batch axes; costs
 * are `[...]`. Where the environment's reward is a function of state and action, the cost is its negative.
 *
 * @concept optimal-control
 */
export interface ControlModel {
  /** The id of the environment the model describes. */
  readonly id: string
  readonly stateSize: number
  readonly actionSize: number
  /** The time step of one `dynamics` call. */
  readonly dt: number
  dynamics(state: Value, action: Value): Value
  cost(state: Value, action: Value): Value
  /** A cost on the final state of a horizon (default none). */
  terminalCost?(state: Value): Value
  /** An equilibrium: `dynamics(state, action) = state`. */
  readonly equilibrium?: { readonly state: readonly number[]; readonly action: readonly number[] }
  /** The bounds of the environment's actions, for planners (the model itself never clips). */
  readonly actionLow?: readonly number[]
  readonly actionHigh?: readonly number[]
  /** For a discrete environment, the control value of each of its actions (CartPole's forces of -10 and 10). */
  readonly discreteActions?: readonly (readonly number[])[]
}

// The components of a state or action on its last axis, and back.
const parts = (x: Value, n: number): Value[] =>
  Array.from({ length: n }, (_, i) => squeeze(sliceAxis(x, -1, i, i + 1), -1))
const join = (xs: readonly Value[]): Value => stack(xs, -1)
const f64 = (xs: readonly number[], shape: number[]): Tensor => fromStorage(Float64Array.from(xs), shape, 'float64')

interface Spec {
  readonly id: string
  readonly n: number
  readonly m: number
  readonly dt: number
  readonly step: (s: Value[], u: Value[]) => Value[]
  readonly cost: (s: Value[], u: Value[]) => Value
  readonly equilibrium?: { readonly state: readonly number[]; readonly action: readonly number[] }
  readonly low?: readonly number[]
  readonly high?: readonly number[]
  readonly discreteActions?: readonly (readonly number[])[]
}

function model(spec: Spec): ControlModel {
  return {
    id: spec.id,
    stateSize: spec.n,
    actionSize: spec.m,
    dt: spec.dt,
    dynamics: (x, u) => join(spec.step(parts(x, spec.n), parts(u, spec.m))),
    cost: (x, u) => spec.cost(parts(x, spec.n), parts(u, spec.m)),
    ...(spec.equilibrium === undefined ? {} : { equilibrium: spec.equilibrium }),
    ...(spec.low === undefined ? {} : { actionLow: spec.low }),
    ...(spec.high === undefined ? {} : { actionHigh: spec.high }),
    ...(spec.discreteActions === undefined ? {} : { discreteActions: spec.discreteActions }),
  }
}

const square = (x: Value | number): Value => o.mul(x, x)
const weighted = (terms: readonly (readonly [Value | number, number])[]): Value => {
  let s: Value | number = 0
  for (const [x, w] of terms) s = o.add(s, o.mul(square(x), w))
  return o.add(s, 0)
}

// ------------------------------------------------------------------------------------------- classic control models

// gymnasium's CartPoleEnv constants (as in classic-control.ts).
const CP = { g: 9.8, massPole: 0.1, total: 1.1, halfLength: 0.5, poleMassLength: 0.05, tau: 0.02 } as const

/**
 * The CartPole dynamics (gymnasium's Euler step) as a differentiable model with a continuous force `[F]`; the
 * environment's actions are the forces -10 and 10 (`discreteActions`), for which the model's step is the
 * environment's. The cost is a quadratic regulation cost `x^2 + 0.1 xd^2 + 10 theta^2 + 0.1 thetad^2 + 0.001 F^2`
 * (CartPole's reward of 1 per step is not a function to optimise); the upright state with no force is an equilibrium.
 *
 * @returns The model.
 * @concept optimal-control
 * @example
 * ```ts
 * import { cartPoleModel } from 'omai-engine/environments'
 * import { tensor, toArray } from 'omai-engine/tensor'
 * toArray(cartPoleModel().dynamics(tensor([0, 1, 0, 0]), tensor([0]))) // => [0.02, 1, 0, 0]
 * ```
 */
export function cartPoleModel(): ControlModel {
  return model({
    id: 'CartPole-v1',
    n: 4,
    m: 1,
    dt: CP.tau,
    step: ([x, xd, th, thd], [f]) => {
      const c = o.cos(th as Value)
      const s = o.sin(th as Value)
      const temp = o.div(o.add(f as Value, o.mul(o.mul(square(thd as Value), CP.poleMassLength), s)), CP.total)
      const den = o.mul(o.sub(4 / 3, o.div(o.mul(square(c), CP.massPole), CP.total)), CP.halfLength)
      const thAcc = o.div(o.sub(o.mul(s, CP.g), o.mul(c, temp)), den)
      const xAcc = o.sub(temp, o.div(o.mul(o.mul(thAcc, CP.poleMassLength), c), CP.total))
      return [
        o.add(x as Value, o.mul(xd as Value, CP.tau)),
        o.add(xd as Value, o.mul(xAcc, CP.tau)),
        o.add(th as Value, o.mul(thd as Value, CP.tau)),
        o.add(thd as Value, o.mul(thAcc, CP.tau)),
      ]
    },
    cost: ([x, xd, th, thd], [f]) =>
      weighted([
        [x as Value, 1],
        [xd as Value, 0.1],
        [th as Value, 10],
        [thd as Value, 0.1],
        [f as Value, 0.001],
      ]),
    equilibrium: { state: [0, 0, 0, 0], action: [0] },
    low: [-10],
    high: [10],
    discreteActions: [[-10], [10]],
  })
}

/**
 * An angle wrapped into [-pi, pi) as a differentiable operation (gymnasium's `angle_normalize`).
 *
 * @param x - Angles.
 * @returns The wrapped angles.
 * @example
 * ```ts
 * import { angleNormalizeValue } from 'omai-engine/environments'
 * import { tensor, toArray } from 'omai-engine/tensor'
 * toArray(angleNormalizeValue(tensor([0, -1]))) // => [0, -1]
 * ```
 */
export function angleNormalizeValue(x: TensorLike): Value {
  return sub(mod(add(x, Math.PI), 2 * Math.PI), Math.PI)
}

/**
 * The Pendulum dynamics (gymnasium's PendulumEnv.step) as a differentiable model of the angle and angular velocity
 * under a torque `[u]`, without the environment's clipping of the torque to [-2, 2] and the speed to [-8, 8] (inside
 * those bounds it is the environment's step). The cost is gymnasium's, `angle^2 + 0.1 thetad^2 + 0.001 u^2` with the
 * angle wrapped; upright at rest is an equilibrium.
 *
 * @param options - Gravity `g` (default 10).
 * @returns The model.
 * @concept optimal-control
 * @example
 * ```ts
 * import { pendulumModel } from 'omai-engine/environments'
 * import { item, tensor } from 'omai-engine/tensor'
 * item(pendulumModel().cost(tensor([0, 1]), tensor([0]))) // => 0.1
 * ```
 */
export function pendulumModel(options: { readonly g?: number } = {}): ControlModel {
  const g = options.g ?? 10
  const dt = 0.05
  return model({
    id: 'Pendulum-v1',
    n: 2,
    m: 1,
    dt,
    step: ([th, thd], [u]) => {
      const acc = o.add(o.mul(o.sin(th as Value), (3 * g) / 2), o.mul(u as Value, 3))
      const w = o.add(thd as Value, o.mul(acc, dt))
      return [o.add(th as Value, o.mul(w, dt)), w]
    },
    cost: ([th, thd], [u]) =>
      weighted([
        [angleNormalizeValue(th as Value), 1],
        [thd as Value, 0.1],
        [u as Value, 0.001],
      ]),
    equilibrium: { state: [0, 0], action: [0] },
    low: [-2],
    high: [2],
  })
}

/**
 * A smooth MountainCarContinuous model: `v' = v + 0.0015 u - 0.0025 cos(3 p)`, `p' = p + v'`, without the
 * environment's clipping of the force, speed and position, and in float64 (gymnasium stores the state in float32, so
 * the environment agrees with the model to float32 precision inside the bounds). The cost is the environment's effort
 * cost `0.1 u^2` plus a shaping term `(p - 0.45)^2` towards the flag (the +100 goal bonus is not smooth).
 *
 * @returns The model.
 * @concept optimal-control
 * @example
 * ```ts
 * import { mountainCarContinuousModel } from 'omai-engine/environments'
 * import { item, tensor } from 'omai-engine/tensor'
 * item(mountainCarContinuousModel().cost(tensor([0.45, 0]), tensor([1]))) // => 0.1
 * ```
 */
export function mountainCarContinuousModel(): ControlModel {
  return model({
    id: 'MountainCarContinuous-v0',
    n: 2,
    m: 1,
    dt: 1,
    step: ([p, v], [u]) => {
      const w = o.sub(o.add(v as Value, o.mul(u as Value, 0.0015)), o.mul(o.cos(o.mul(p as Value, 3)), 0.0025))
      return [o.add(p as Value, w), w]
    },
    cost: ([p], [u]) =>
      weighted([
        [u as Value, 0.1],
        [o.sub(p as Value, 0.45), 1],
      ]),
    low: [-1],
    high: [1],
  })
}

// gymnasium's AcrobotEnv._dsdt ("book" dynamics), as in classic-control.ts, with a continuous torque.
function acrobotDerivatives(s: readonly Value[], a: Value): Value[] {
  const [theta1, theta2, dtheta1, dtheta2] = s as [Value, Value, Value, Value]
  const m1 = 1
  const m2 = 1
  const l1 = 1
  const lc1 = 0.5
  const lc2 = 0.5
  const i2 = 1
  const g = 9.8
  const c2 = o.cos(theta2)
  const d1 = o.add(o.add(o.add(m1 * lc1 ** 2, o.mul(o.add(l1 ** 2 + lc2 ** 2, o.mul(c2, 2 * l1 * lc2)), m2)), 1), 1)
  const d2 = o.add(o.mul(o.add(lc2 ** 2, o.mul(c2, l1 * lc2)), m2), i2)
  const phi2 = o.mul(o.cos(o.sub(o.add(theta1, theta2), Math.PI / 2)), m2 * lc2 * g)
  const s2 = o.sin(theta2)
  const phi1 = o.add(
    o.add(
      o.sub(
        o.mul(o.mul(o.mul(dtheta2, dtheta2), s2), -m2 * l1 * lc2),
        o.mul(o.mul(o.mul(dtheta2, dtheta1), s2), 2 * m2 * l1 * lc2),
      ),
      o.mul(o.cos(o.sub(theta1, Math.PI / 2)), (m1 * lc1 + m2 * l1) * g),
    ),
    phi2,
  )
  const num = o.sub(
    o.sub(o.add(a, o.mul(o.div(d2, d1), phi1)), o.mul(o.mul(o.mul(dtheta1, dtheta1), s2), m2 * l1 * lc2)),
    phi2,
  )
  const ddtheta2 = o.div(num, o.sub(m2 * lc2 ** 2 + i2, o.div(o.mul(d2, d2), d1)))
  const ddtheta1 = o.neg(o.div(o.add(o.mul(d2, ddtheta2), phi1), d1))
  return [o.add(dtheta1, 0), o.add(dtheta2, 0), ddtheta1, ddtheta2]
}

/**
 * The Acrobot dynamics (gymnasium's AcrobotEnv: the "book" equations, one RK4 step of 0.2 s) as a differentiable
 * model with a continuous torque `[u]` on the second joint, without the environment's angle wrapping and velocity
 * bounds; the environment's actions are the torques -1, 0 and 1. The cost is the height of the tip below the bar's
 * pivot, `cos theta1 + cos(theta1 + theta2)`, plus `0.01 u^2` (a shaping of Acrobot's reach-the-bar task).
 *
 * @returns The model.
 * @concept optimal-control
 * @example
 * ```ts
 * import { acrobotModel } from 'omai-engine/environments'
 * import { tensor, toFlat } from 'omai-engine/tensor'
 * const next = toFlat(acrobotModel().dynamics(tensor([0, 0, 0, 0]), tensor([0])))
 * next.map((v) => Math.abs(v) < 1e-15) // => [true, true, true, true]
 * ```
 */
export function acrobotModel(): ControlModel {
  return model({
    id: 'Acrobot-v1',
    n: 4,
    m: 1,
    dt: 0.2,
    step: (s, [u]) => rk4(o, (y) => acrobotDerivatives(y, u as Value), s, 0.2),
    cost: ([t1, t2], [u]) =>
      o.add(o.add(o.cos(t1 as Value), o.cos(o.add(t1 as Value, t2 as Value))), o.mul(square(u as Value), 0.01)),
    equilibrium: { state: [0, 0, 0, 0], action: [0] },
    low: [-1],
    high: [1],
    discreteActions: [[-1], [0], [1]],
  })
}

// --------------------------------------------------------------------------------------- continuous-control models

/**
 * The point mass (PointMass-v0) as a differentiable model of `[x, y, vx, vy]` for a fixed goal: the exact step without
 * the walls (inside the arena it is the environment's step). The cost is minus the environment's reward for the step.
 *
 * @param options - The environment's options and the goal (default the origin).
 * @returns The model.
 * @concept optimal-control
 * @example
 * ```ts
 * import { pointMassModel } from 'omai-engine/environments'
 * import { item, tensor } from 'omai-engine/tensor'
 * item(pointMassModel().cost(tensor([0, 0, 0, 0]), tensor([0, 0]))) // => -1
 * ```
 */
export function pointMassModel(options: PointMassOptions & { readonly goal?: readonly number[] } = {}): ControlModel {
  const p = pointMassParameters(options)
  const goal = options.goal ?? [0, 0]
  const k = dampedMassCoefficients(p.mass, p.damping, p.dt)
  const step = (s: Value[], u: Value[]): Value[] =>
    dampedMassStep(
      o,
      k,
      [s[0] as Value, s[1] as Value],
      [s[2] as Value, s[3] as Value],
      [o.mul(u[0] as Value, p.gain), o.mul(u[1] as Value, p.gain)],
    )
  return model({
    id: 'PointMass-v0',
    n: 4,
    m: 2,
    dt: p.dt,
    step,
    cost: (s, u) => o.neg(pointMassRewardOf(o, step(s, u), u, goal, p.targetSize, p.margin)),
    equilibrium: { state: [goal[0] ?? 0, goal[1] ?? 0, 0, 0], action: [0, 0] },
    low: [-1, -1],
    high: [1, 1],
  })
}

/**
 * The 2-D reacher (Reacher2D-v0) as a differentiable model of `[q1, q2, qd1, qd2]` for a fixed target: the same RK4
 * step of the arm's manipulator equations. The cost is minus the environment's reward (fingertip distance plus the
 * weighted squared action).
 *
 * @param options - The environment's options and the target (default `[0.1, 0.1]`).
 * @returns The model.
 * @concept optimal-control
 * @example
 * ```ts
 * import { reacher2dModel } from 'omai-engine/environments'
 * import { item, tensor } from 'omai-engine/tensor'
 * item(reacher2dModel({ target: [0.1, 0.11] }).cost(tensor([0, Math.PI / 2, 0, 0]), tensor([0, 0]))) < 1e-15 // => true
 * ```
 */
export function reacher2dModel(options: Reacher2DOptions & { readonly target?: readonly number[] } = {}): ControlModel {
  const p = reacher2dParameters(options)
  const target = options.target ?? [0.1, 0.1]
  return model({
    id: 'Reacher2D-v0',
    n: 4,
    m: 2,
    dt: p.dt,
    step: (s, u) => rk4(o, (y) => armDerivatives(o, y, u, p), s, p.dt, p.substeps),
    cost: (s, u) => o.neg(reacherRewardOf(o, s, u, target, p, p.controlWeight)),
    low: [-1, -1],
    high: [1, 1],
  })
}

/**
 * The double integrator (DoubleIntegrator-v0) as a differentiable model: the exact linear step without the force
 * limit, and the quadratic cost `q_p p^2 + q_v v^2 + r u^2` (minus the reward).
 *
 * @param options - The environment's options.
 * @returns The model.
 * @concept optimal-control
 * @example
 * ```ts
 * import { doubleIntegratorModel } from 'omai-engine/environments'
 * import { tensor, toArray } from 'omai-engine/tensor'
 * toArray(doubleIntegratorModel({ dt: 0.5 }).dynamics(tensor([0, 1]), tensor([1]))) // => [0.625, 1.5]
 * ```
 */
export function doubleIntegratorModel(options: DoubleIntegratorOptions = {}): ControlModel {
  const p = doubleIntegratorParameters(options)
  const k = dampedMassCoefficients(p.mass, 0, p.dt)
  return model({
    id: 'DoubleIntegrator-v0',
    n: 2,
    m: 1,
    dt: p.dt,
    step: ([x, v], [u]) => dampedMassStep(o, k, [x as Value], [v as Value], [u as Value]),
    cost: ([x, v], [u]) =>
      weighted([
        [x as Value, p.positionWeight],
        [v as Value, p.velocityWeight],
        [u as Value, p.controlWeight],
      ]),
    equilibrium: { state: [0, 0], action: [0] },
    low: [-p.maxForce],
    high: [p.maxForce],
  })
}

/**
 * The planar quadrotor (PlanarQuadrotor-v0) as a differentiable model of `[x, z, theta, xd, zd, thetad]` for a fixed
 * goal: the same RK4 step without the thrust limits. The cost is minus the environment's reward; hovering at the goal
 * with `m g / 2` per rotor is an equilibrium.
 *
 * @param options - The environment's options and the goal (default the origin).
 * @returns The model.
 * @concept optimal-control
 * @example
 * ```ts
 * import { planarQuadrotorModel } from 'omai-engine/environments'
 * planarQuadrotorModel({ g: 10 }).equilibrium?.action // => [5, 5]
 * ```
 */
export function planarQuadrotorModel(
  options: PlanarQuadrotorOptions & { readonly goal?: readonly number[] } = {},
): ControlModel {
  const p = planarQuadrotorParameters(options)
  const goal = options.goal ?? [0, 0]
  const hover = (p.mass * p.g) / 2
  return model({
    id: 'PlanarQuadrotor-v0',
    n: 6,
    m: 2,
    dt: p.dt,
    step: (s, u) => rk4(o, (y) => quadrotorDerivatives(o, y, u, p), s, p.dt, p.substeps),
    cost: (s, u) => o.neg(quadrotorRewardOf(o, s, u, goal, p, p.controlWeight)),
    equilibrium: { state: [goal[0] ?? 0, goal[1] ?? 0, 0, 0, 0, 0], action: [hover, hover] },
    low: [0, 0],
    high: [p.maxThrust, p.maxThrust],
  })
}

/**
 * The cart double pendulum (CartDoublePendulum-v0) as a differentiable model: the same RK4 integration of the
 * Lagrangian equations without the force limit. The cost is minus the environment's reward for the step (evaluated on
 * the next state, as the environment does); upright at rest is an equilibrium.
 *
 * @param options - The environment's options.
 * @returns The model.
 * @concept optimal-control
 * @example
 * ```ts
 * import { cartDoublePendulumModel } from 'omai-engine/environments'
 * import { item, tensor } from 'omai-engine/tensor'
 * item(cartDoublePendulumModel().cost(tensor([0, 0, 0, 0, 0, 0]), tensor([0]))) // => -10
 * ```
 */
export function cartDoublePendulumModel(options: CartDoublePendulumOptions = {}): ControlModel {
  const p = cartDoublePendulumParameters(options)
  const step = (s: Value[], u: Value[]): Value[] =>
    rk4(o, (y) => cartDoublePendulumDerivativesOf(o, y, u[0] as Value, p), s, p.dt, p.substeps)
  return model({
    id: 'CartDoublePendulum-v0',
    n: 6,
    m: 1,
    dt: p.dt,
    step,
    cost: (s, u) => o.neg(cartDoublePendulumRewardOf(o, step(s, u), p)),
    equilibrium: { state: [0, 0, 0, 0, 0, 0], action: [0] },
    low: [-p.maxForce],
    high: [p.maxForce],
  })
}

// --------------------------------------------------------------------------------------------- using the models

/**
 * Linearise a model's dynamics at a state and action by forward-mode autodiff: `x' ~ A (x - x0) + B (u - u0) + f(x0,
 * u0)`, with `A = df/dx` and `B = df/du`.
 *
 * @param model - The model.
 * @param state - The state `[stateSize]`.
 * @param action - The action `[actionSize]`.
 * @returns `A` (`[n, n]`) and `B` (`[n, m]`).
 * @concept optimal-control
 * @example
 * ```ts
 * import { doubleIntegratorModel, linearize } from 'omai-engine/environments'
 * import { toArray } from 'omai-engine/tensor'
 * toArray(linearize(doubleIntegratorModel({ dt: 0.5 }), [0, 0], [0]).B) // => [[0.125], [0.5]]
 * ```
 */
export function linearize(
  model: ControlModel,
  state: readonly number[],
  action: readonly number[],
): { readonly A: Value; readonly B: Value } {
  const x = f64(state, [model.stateSize])
  const u = f64(action, [model.actionSize])
  return {
    A: jacfwd((s: Value) => model.dynamics(s, u))(x),
    B: jacfwd((a: Value) => model.dynamics(x, a))(u),
  }
}

/** Per-step derivatives of the dynamics and the cost along a trajectory (what iLQR's backward pass consumes). */
export interface TrajectoryDerivatives {
  /** `df/dx` and `df/du` at each step, `[T][n][n]` and `[T][n][m]`. */
  readonly fx: number[][][]
  readonly fu: number[][][]
  /** Cost gradients `[T][n]`, `[T][m]` and Hessians `[T][n][n]`, `[T][m][m]`, `[T][m][n]`. */
  readonly lx: number[][]
  readonly lu: number[][]
  readonly lxx: number[][][]
  readonly luu: number[][][]
  readonly lux: number[][][]
}

// Basis tangents: column `i` set to 1 on every row of a [T, d] batch.
const basisRows = (T: number, d: number, i: number): Tensor => {
  const data = new Float64Array(T * d)
  for (let t = 0; t < T; t++) data[t * d + i] = 1
  return fromStorage(data, [T, d], 'float64')
}

// Column j of a [T, d] tensor, written into out[t][r][j].
function scatterColumn(out: number[][][], flat: readonly number[], d: number, j: number): void {
  for (const [t, rows] of out.entries()) for (let r = 0; r < d; r++) (rows[r] as number[])[j] = flat[t * d + r] ?? 0
}

const blank = (T: number, r: number, c: number): number[][][] =>
  Array.from({ length: T }, () => Array.from({ length: r }, () => new Array<number>(c).fill(0)))

/**
 * The derivatives of a model along a trajectory, computed for all steps at once: the model is applied to the whole
 * batch of states and actions, so one forward-mode pass per state or action coordinate gives that column of every
 * step's Jacobian, and forward over reverse mode gives the cost Hessians.
 *
 * @param model - The model.
 * @param states - States `[T][n]`.
 * @param actions - Actions `[T][m]`.
 * @returns The Jacobians of the dynamics and the gradients and Hessians of the cost at each step.
 * @concept optimal-control
 * @example
 * ```ts
 * import { doubleIntegratorModel, trajectoryDerivatives } from 'omai-engine/environments'
 * const d = trajectoryDerivatives(doubleIntegratorModel(), [[1, 0], [0, 0]], [[0], [2]])
 * [d.fx[0], d.lx[0], d.luu[1]] // => [[[1, 0.1], [0, 1]], [2, 0], [[2]]]
 * ```
 */
export function trajectoryDerivatives(
  model: ControlModel,
  states: readonly (readonly number[])[],
  actions: readonly (readonly number[])[],
): TrajectoryDerivatives {
  const T = states.length
  const n = model.stateSize
  const m = model.actionSize
  const X = f64(states.flat(), [T, n])
  const U = f64(actions.flat(), [T, m])
  const fx = blank(T, n, n)
  const fu = blank(T, n, m)
  for (let i = 0; i < n; i++) {
    const t = jvp((x: Value) => model.dynamics(x, U), X, basisRows(T, n, i)).tangent
    scatterColumn(fx, toFlat(t), n, i)
  }
  for (let j = 0; j < m; j++) {
    const t = jvp((u: Value) => model.dynamics(X, u), U, basisRows(T, m, j)).tangent
    scatterColumn(fu, toFlat(t), n, j)
  }
  type Z = { readonly x: Value; readonly u: Value }
  const total = (z: Z): Value => sum(model.cost(z.x, z.u))
  const g = grad(total) as (z: Z) => Z
  const gz = g({ x: X, u: U })
  const gx = toFlat(gz.x)
  const gu = toFlat(gz.u)
  const lx = Array.from({ length: T }, (_, t) => gx.slice(t * n, (t + 1) * n))
  const lu = Array.from({ length: T }, (_, t) => gu.slice(t * m, (t + 1) * m))
  const lxx = blank(T, n, n)
  const luu = blank(T, m, m)
  const lux = blank(T, m, n)
  const zx = zeros([T, n])
  const zu = zeros([T, m])
  for (let i = 0; i < n; i++) {
    const t = jvp(g, { x: X, u: U }, { x: basisRows(T, n, i), u: zu }).tangent
    scatterColumn(lxx, toFlat(t.x), n, i)
    scatterColumn(lux, toFlat(t.u), m, i)
  }
  for (let j = 0; j < m; j++) {
    const t = jvp(g, { x: X, u: U }, { x: zx, u: basisRows(T, m, j) }).tangent
    scatterColumn(luu, toFlat(t.u), m, j)
  }
  return { fx, fu, lx, lu, lxx, luu, lux }
}

/**
 * Roll a model out from a state under a sequence of actions.
 *
 * @param model - The model.
 * @param state - The start state.
 * @param actions - One action per step.
 * @returns The states visited (`actions.length + 1` of them, the start first) and the total cost (with the terminal
 *   cost when the model has one).
 * @concept optimal-control
 * @example
 * ```ts
 * import { doubleIntegratorModel, rolloutModel } from 'omai-engine/environments'
 * rolloutModel(doubleIntegratorModel({ dt: 0.5 }), [0, 1], [[0]]) // => { states: [[0, 1], [0.5, 1]], cost: 1 }
 * ```
 */
export function rolloutModel(
  model: ControlModel,
  state: readonly number[],
  actions: readonly (readonly number[])[],
): { readonly states: number[][]; readonly cost: number } {
  let x: Value = f64(state, [model.stateSize])
  const states = [[...state]]
  let cost = 0
  for (const a of actions) {
    const u = f64(a, [model.actionSize])
    cost += toFlat(model.cost(x, u))[0] ?? 0
    x = model.dynamics(x, u)
    states.push(toFlat(x))
  }
  if (model.terminalCost !== undefined) cost += toFlat(model.terminalCost(x))[0] ?? 0
  return { states, cost }
}

/**
 * The total costs of many action sequences from one state, rolled out as one batch (one model call per step for the
 * whole population): what sampling planners (CEM, MPPI) score candidate plans with.
 *
 * @param model - The model.
 * @param state - The start state.
 * @param plans - Candidate plans `[P][H][m]`.
 * @returns The total cost of each plan (with the terminal cost when the model has one).
 * @concept model-predictive-control
 * @example
 * ```ts
 * import { doubleIntegratorModel, sequenceCosts } from 'omai-engine/environments'
 * sequenceCosts(doubleIntegratorModel(), [1, 0], [[[0]], [[1]]]) // => [1, 2]
 * ```
 */
export function sequenceCosts(
  model: ControlModel,
  state: readonly number[],
  plans: readonly (readonly (readonly number[])[])[],
): number[] {
  const P = plans.length
  const n = model.stateSize
  const m = model.actionSize
  if (P === 0) return []
  const H = plans[0]?.length ?? 0
  let x: Value = f64(Array.from({ length: P }, () => state).flat(), [P, n])
  let total: Value = zeros([P])
  for (let t = 0; t < H; t++) {
    const u = f64(
      plans.flatMap((plan) => Array.from({ length: m }, (_, j) => plan[t]?.[j] ?? 0)),
      [P, m],
    )
    total = add(total, model.cost(x, u))
    x = model.dynamics(x, u)
  }
  if (model.terminalCost !== undefined) total = add(total, model.terminalCost(x))
  return toFlat(total)
}

/**
 * A model as an environment over its own states (the reward is minus the cost), for planners and agents that act in a
 * model: `reset` draws from `initial`, observations are the states, and nothing terminates.
 *
 * @param model - The model.
 * @param options - The start-state sampler (default the equilibrium or zeros) and the horizon.
 * @returns The environment.
 * @concept model-predictive-control
 * @example
 * ```ts
 * import { controlModelEnv, doubleIntegratorModel } from 'omai-engine/environments'
 * import { key } from 'omai-engine/random'
 * controlModelEnv(doubleIntegratorModel()).step([1, 0], [0], key(0)).reward // => -1
 * ```
 */
export function controlModelEnv(
  model: ControlModel,
  options: { readonly initial?: (key: Key) => readonly number[]; readonly maxEpisodeSteps?: number } = {},
): Env<readonly number[], readonly number[], readonly number[]> {
  const n = model.stateSize
  const m = model.actionSize
  const initial = options.initial ?? (() => model.equilibrium?.state ?? new Array<number>(n).fill(0))
  const big = Number.MAX_VALUE
  return {
    spec: {
      id: `${model.id}-model`,
      description: `a differentiable model of ${model.id}`,
      deterministic: true,
      ...(options.maxEpisodeSteps === undefined ? {} : { maxEpisodeSteps: options.maxEpisodeSteps }),
    },
    observationSpace: box(-big, big, [n]),
    actionSpace: box(
      model.actionLow ?? new Array<number>(m).fill(-big),
      model.actionHigh ?? new Array<number>(m).fill(big),
      [m],
    ),
    reset: (k) => {
      const s = initial(k)
      return { state: s, observation: s, info: {} }
    },
    step: (s, a) => {
      const x = f64(s, [n])
      const u = f64(a, [m])
      const next = toFlat(model.dynamics(x, u))
      return {
        state: next,
        observation: next,
        reward: -(toFlat(model.cost(x, u))[0] ?? 0),
        terminated: false,
        truncated: false,
        info: {},
      }
    },
    render: (s) => scene(240, 40, [], { title: `${model.id} model at [${s.map((v) => v.toFixed(2)).join(', ')}]` }),
    observe: (s) => s,
  }
}

/**
 * Check that a model and its environment agree on a step (for tests and diagnostics): the largest absolute difference
 * between `model.dynamics` and the environment's next state.
 *
 * @param model - The model.
 * @param envStep - The environment's step as a function of a model state and a model action.
 * @param state - The state.
 * @param action - The action.
 * @returns The largest absolute difference.
 * @throws OmaiError with code `shape` when the sizes differ.
 * @concept optimal-control
 * @example
 * ```ts
 * import { doubleIntegratorDynamics, doubleIntegratorModel, doubleIntegratorParameters, modelStepError } from 'omai-engine/environments'
 * const p = doubleIntegratorParameters()
 * modelStepError(doubleIntegratorModel(), (s, a) => doubleIntegratorDynamics(s as never, a, p), [1, 2], [3]) // => 0
 * ```
 */
export function modelStepError(
  model: ControlModel,
  envStep: (state: readonly number[], action: readonly number[]) => readonly number[],
  state: readonly number[],
  action: readonly number[],
): number {
  const ours = toFlat(model.dynamics(f64(state, [model.stateSize]), f64(action, [model.actionSize])))
  const theirs = envStep(state, action)
  if (theirs.length < ours.length) throw new OmaiError('shape', 'modelStepError: the environment state is smaller')
  let worst = 0
  for (const [i, v] of ours.entries()) worst = Math.max(worst, Math.abs(v - (theirs[i] ?? 0)))
  return worst
}
