/**
 * Continuous-control environments with analytic dynamics, in the spirit of the DeepMind control suite and gymnasium's
 * MuJoCo tasks: a point mass reaching a target, a two-link planar reacher, the double integrator (the one-dimensional
 * LQR benchmark), a planar quadrotor flying to a goal and a cart double pendulum. Each exposes its parameters,
 * equations of motion, integration step and, where useful, its energy, so models and controllers can be checked against
 * it.
 */

import { clamp } from '../../../lib/base/index.ts'
import { type Key, uniform } from '../../../lib/random/index.ts'
import type { Env, EnvDefinition } from '../../../lib/rl/index.ts'
import { circle, line, polygon, rect, type Scene, scene } from '../../../lib/scene/index.ts'
import { bool, box, dict, int, real } from '../../../lib/space/index.ts'
import { c2dMatrices } from '../../../lib/systems/index.ts'
import { toArray } from '../../../lib/tensor/index.ts'
import {
  type ArmParameters,
  armDerivatives,
  armEnergy,
  armTip,
  type CartDoublePendulumParameters,
  cartDoublePendulumDerivativesOf,
  cartDoublePendulumEnergyOf,
  cartDoublePendulumRewardOf,
  cartDoublePendulumTipOf,
  dampedMassCoefficients,
  dampedMassStep,
  numberOps as o,
  pointMassRewardOf,
  type QuadrotorParameters,
  quadrotorDerivatives,
  quadrotorRewardOf,
  reacherRewardOf,
  rk4,
} from './_mechanics.ts'

export type { ArmParameters, CartDoublePendulumParameters, QuadrotorParameters } from './_mechanics.ts'

const draws = (k: Key, n: number, low: number, high: number): number[] =>
  Array.from(uniform(k, [n], { low, high }).data)
const BIG = Number.MAX_VALUE
const clipTo = (a: readonly number[], low: number, high: number, n: number): number[] =>
  Array.from({ length: n }, (_, i) => clamp(a[i] ?? 0, low, high))
const spec = (id: string, description: string, maxEpisodeSteps: number) => ({
  id,
  description,
  maxEpisodeSteps,
  deterministic: true,
})

// -------------------------------------------------------------------------------------------------------- point mass

/** Options of the point mass (DeepMind control suite's point_mass in spirit). */
export interface PointMassOptions {
  readonly mass?: number
  readonly damping?: number
  /** Force per unit action (default 0.1, the suite's actuator gear). */
  readonly gain?: number
  readonly dt?: number
  /** Half width of the square arena (default 0.3). */
  readonly arena?: number
  /** Goals are drawn uniformly in `[-goalRange, goalRange]^2` (default 0: the suite's fixed target at the origin). */
  readonly goalRange?: number
  readonly targetSize?: number
  /** Distance beyond the target over which the reward falls to 0.1 (default the target size, as in the suite). */
  readonly margin?: number
  readonly maxEpisodeSteps?: number
}

/** The resolved parameters of the point mass. */
export interface PointMassParameters {
  readonly mass: number
  readonly damping: number
  readonly gain: number
  readonly dt: number
  readonly arena: number
  readonly goalRange: number
  readonly targetSize: number
  readonly margin: number
}

/**
 * The point-mass parameters with defaults filled in: mass 0.3, damping 1, gain 0.1, dt 0.02, a 0.3 arena, a fixed
 * target of size 0.015 (the suite's `easy` task).
 *
 * @param options - Any subset of the options.
 * @returns The parameters.
 * @example
 * ```ts
 * import { pointMassParameters } from 'whitebox-engine/methods/rl'
 * pointMassParameters({ goalRange: 0.2 }).margin // => 0.015
 * ```
 */
export function pointMassParameters(options: PointMassOptions = {}): PointMassParameters {
  const targetSize = options.targetSize ?? 0.015
  return {
    mass: options.mass ?? 0.3,
    damping: options.damping ?? 1,
    gain: options.gain ?? 0.1,
    dt: options.dt ?? 0.02,
    arena: options.arena ?? 0.3,
    goalRange: options.goalRange ?? 0,
    targetSize,
    margin: options.margin ?? targetSize,
  }
}

/** The state of the point mass: position, velocity and goal, `[x, y, vx, vy, gx, gy]`. */
export type PointMassState = readonly number[]

/**
 * One step of the point mass: on each axis `m x'' = gain u - b x'`, integrated exactly over `dt` (zero-order hold, so
 * the step is the matrix exponential of the linear system), with actions clipped to [-1, 1]. Walls at `+-arena` stop
 * the mass (position clamped, outward velocity zeroed).
 *
 * @param state - The state.
 * @param action - The two actions.
 * @param params - The parameters.
 * @returns The next state.
 * @concept environments
 * @example
 * ```ts
 * import { pointMassDynamics, pointMassParameters } from 'whitebox-engine/methods/rl'
 * pointMassDynamics([0, 0, 0, 0, 0, 0], [0, 0], pointMassParameters()) // => [0, 0, 0, 0, 0, 0]
 * ```
 */
export function pointMassDynamics(
  state: PointMassState,
  action: readonly number[],
  params: PointMassParameters,
): PointMassState {
  const k = dampedMassCoefficients(params.mass, params.damping, params.dt)
  const u = clipTo(action, -1, 1, 2)
  const [x, y, vx, vy] = dampedMassStep(
    o,
    k,
    [state[0] ?? 0, state[1] ?? 0],
    [state[2] ?? 0, state[3] ?? 0],
    [(u[0] ?? 0) * params.gain, (u[1] ?? 0) * params.gain],
  ) as [number, number, number, number]
  const wall = (p: number, vel: number): [number, number] => {
    if (p > params.arena) return [params.arena, Math.min(vel, 0)]
    if (p < -params.arena) return [-params.arena, Math.max(vel, 0)]
    return [p, vel]
  }
  const [px, wx] = wall(x, vx)
  const [py, wy] = wall(y, vy)
  return [px, py, wx, wy, state[4] ?? 0, state[5] ?? 0]
}

function renderPointMass(s: PointMassState, p: PointMassParameters): Scene {
  const size = 400
  const scale = size / (2.2 * p.arena)
  const at = (x: number, y: number): [number, number] => [size / 2 + x * scale, size / 2 - y * scale]
  const [gx, gy] = at(s[4] ?? 0, s[5] ?? 0)
  const [mx, my] = at(s[0] ?? 0, s[1] ?? 0)
  const [ax, ay] = at(-p.arena, p.arena)
  return scene(
    size,
    size,
    [
      rect(ax, ay, 2 * p.arena * scale, 2 * p.arena * scale, { stroke: 'fg', strokeWidth: 2, id: 'arena' }),
      circle(gx, gy, Math.max(3, p.targetSize * scale), { fill: 'tab2', opacity: 0.6, id: 'target' }),
      circle(mx, my, Math.max(4, 0.01 * scale), {
        fill: 'tab0',
        id: 'mass',
        title: `mass at (${(s[0] ?? 0).toFixed(3)}, ${(s[1] ?? 0).toFixed(3)})`,
      }),
    ],
    { title: 'PointMass' },
  )
}

const POINT_MASS_DESCRIPTION = 'Push a point mass in a walled square arena to a target and keep it there.'

/**
 * Create the point mass environment (see {@link pointMass}).
 *
 * @param options - Physical parameters, goal range and horizon.
 * @returns The environment.
 * @concept environments
 * @example
 * ```ts
 * import { makePointMass } from 'whitebox-engine/methods/rl'
 * import { key } from 'whitebox-engine/random'
 * makePointMass().reset(key(0)).observation.length // => 6
 * ```
 */
export function makePointMass(
  options: PointMassOptions = {},
): Env<PointMassState, readonly number[], readonly number[]> {
  const p = pointMassParameters(options)
  return {
    spec: spec('PointMass-v0', POINT_MASS_DESCRIPTION, options.maxEpisodeSteps ?? 1000),
    observationSpace: box(
      [-p.arena, -p.arena, -BIG, -BIG, -p.goalRange, -p.goalRange],
      [p.arena, p.arena, BIG, BIG, p.goalRange, p.goalRange],
      [6],
    ),
    actionSpace: box(-1, 1, [2], { label: 'force on each axis' }),
    reset: (k) => {
      const [x = 0, y = 0, gx = 0, gy = 0] = draws(k, 4, -1, 1)
      const r = p.arena - 0.01
      const s = [r * x, r * y, 0, 0, p.goalRange * gx, p.goalRange * gy]
      return { state: s, observation: s, info: {} }
    },
    step: (s, a) => {
      const next = pointMassDynamics(s, a, p)
      const reward = pointMassRewardOf(o, next, clipTo(a, -1, 1, 2), [s[4] ?? 0, s[5] ?? 0], p.targetSize, p.margin)
      return { state: next, observation: next, reward, terminated: false, truncated: false, info: {} }
    },
    render: (s) => renderPointMass(s, p),
    observe: (s) => s,
  }
}

/**
 * PointMass-v0 (after the DeepMind control suite's point_mass): a 0.3 kg mass with viscous damping, pushed on each axis
 * by actions in [-1, 1], must reach a target and stay there. The reward is the suite's: a Gaussian tolerance of the
 * distance to the target (1 within `targetSize`, 0.1 at `targetSize + margin`) times `(4 + c) / 5`, where `c` is the
 * mean quadratic tolerance of the actions. Dynamics are the exact zero-order-hold solution of the linear system
 * (the suite integrates MuJoCo's). Observations are `[x, y, vx, vy, gx, gy]`; `make` truncates at 1000 steps.
 *
 * @concept environments
 * @example
 * ```ts
 * import { pointMass } from 'whitebox-engine/methods/rl'
 * import { key } from 'whitebox-engine/random'
 * const env = pointMass.create()
 * env.step([0, 0, 0, 0, 0, 0], [0, 0], key(0)).reward // => 1
 * ```
 */
export const pointMass: EnvDefinition<PointMassState, readonly number[], readonly number[]> = {
  id: 'PointMass-v0',
  description: POINT_MASS_DESCRIPTION,
  options: dict({
    mass: real(0.05, 5, { default: 0.3, unit: 'kg' }),
    damping: real(0, 10, { default: 1, description: 'viscous damping' }),
    gain: real(0.01, 1, { default: 0.1, description: 'force per unit action' }),
    dt: real(0.001, 0.1, { default: 0.02, unit: 's' }),
    goalRange: real(0, 0.3, { default: 0, description: 'goals uniform in [-goalRange, goalRange]^2' }),
    targetSize: real(0.005, 0.1, { default: 0.015 }),
    maxEpisodeSteps: int(10, 10_000, { default: 1000 }),
  }),
  create: (options) => makePointMass((options ?? {}) as PointMassOptions),
}

// ------------------------------------------------------------------------------------------------------ 2-D reacher

/** Options of the 2-D reacher. */
export interface Reacher2DOptions {
  readonly lengths?: readonly [number, number]
  readonly masses?: readonly [number, number]
  readonly armature?: number
  readonly damping?: number
  readonly gain?: number
  readonly dt?: number
  /** RK4 substeps per step (default 1). */
  readonly substeps?: number
  /** Targets are drawn uniformly on the disc of this radius (default 0.2, gymnasium's). */
  readonly goalRadius?: number
  /** Weight of the squared action in the reward (default 0.1, gymnasium Reacher-v5's). */
  readonly controlWeight?: number
  readonly maxEpisodeSteps?: number
}

/** The resolved parameters of the 2-D reacher. */
export interface Reacher2DParameters extends ArmParameters {
  readonly dt: number
  readonly substeps: number
  readonly goalRadius: number
  readonly controlWeight: number
}

/**
 * The reacher parameters with defaults filled in: gymnasium's link lengths 0.1 and 0.11, unit masses, armature 0.01,
 * damping 0.05, torque gain 0.2, dt 0.02.
 *
 * @param options - Any subset of the options.
 * @returns The parameters.
 * @example
 * ```ts
 * import { reacher2dParameters } from 'whitebox-engine/methods/rl'
 * reacher2dParameters().lengths // => [0.1, 0.11]
 * ```
 */
export function reacher2dParameters(options: Reacher2DOptions = {}): Reacher2DParameters {
  return {
    lengths: options.lengths ?? [0.1, 0.11],
    masses: options.masses ?? [1, 1],
    armature: options.armature ?? 0.01,
    damping: options.damping ?? 0.05,
    gain: options.gain ?? 0.2,
    dt: options.dt ?? 0.02,
    substeps: options.substeps ?? 1,
    goalRadius: options.goalRadius ?? 0.2,
    controlWeight: options.controlWeight ?? 0.1,
  }
}

/** The state of the reacher: joint angles, joint velocities and target, `[q1, q2, qd1, qd2, tx, ty]`. */
export type Reacher2DState = readonly number[]

/**
 * The continuous-time equations of motion of the reacher's arm: `d/dt [q1, q2, qd1, qd2]` under joint actions.
 *
 * @param state - The joint angles and velocities (a target after them is ignored).
 * @param action - The two joint actions (unclipped).
 * @param params - The parameters.
 * @returns The time derivative of the arm's state.
 * @concept environments
 * @example
 * ```ts
 * import { reacher2dDerivatives, reacher2dParameters } from 'whitebox-engine/methods/rl'
 * reacher2dDerivatives([0, 0, 1, 0], [0, 0], reacher2dParameters({ damping: 0 }))[0] // => 1
 * ```
 */
export function reacher2dDerivatives(
  state: readonly number[],
  action: readonly number[],
  params: ArmParameters,
): number[] {
  return armDerivatives(o, state, action, params)
}

/**
 * One step of the two-link planar arm (horizontal plane, uniform links, rotor armature and viscous damping): the
 * manipulator equations `M(q) qdd + c(q, qd) + b qd = gain u` integrated by RK4 over `dt`, actions clipped to [-1, 1].
 *
 * @param state - The state.
 * @param action - The two joint actions.
 * @param params - The parameters.
 * @returns The next state.
 * @concept environments
 * @example
 * ```ts
 * import { reacher2dDynamics, reacher2dParameters } from 'whitebox-engine/methods/rl'
 * reacher2dDynamics([0, 0, 0, 0, 0.1, 0], [0, 0], reacher2dParameters()) // => [0, 0, 0, 0, 0.1, 0]
 * ```
 */
export function reacher2dDynamics(
  state: Reacher2DState,
  action: readonly number[],
  params: Reacher2DParameters,
): Reacher2DState {
  const u = clipTo(action, -1, 1, 2)
  const next = rk4(o, (s) => armDerivatives(o, s, u, params), state.slice(0, 4), params.dt, params.substeps)
  return [...next, state[4] ?? 0, state[5] ?? 0]
}

/**
 * The kinetic energy of the reacher's arm (conserved without actuation or damping).
 *
 * @param state - The state.
 * @param params - The parameters.
 * @returns `qd^T M(q) qd / 2`.
 * @example
 * ```ts
 * import { reacher2dEnergy, reacher2dParameters } from 'whitebox-engine/methods/rl'
 * reacher2dEnergy([0, 0, 0, 0, 0, 0], reacher2dParameters()) // => 0
 * ```
 */
export function reacher2dEnergy(state: Reacher2DState, params: Reacher2DParameters): number {
  return armEnergy(o, state, params)
}

function reacherObservation(s: Reacher2DState, p: Reacher2DParameters): number[] {
  const [q1 = 0, q2 = 0, qd1 = 0, qd2 = 0, tx = 0, ty = 0] = s
  const [fx, fy] = armTip(o, q1, q2, p)
  return [Math.cos(q1), Math.cos(q2), Math.sin(q1), Math.sin(q2), tx, ty, qd1, qd2, fx - tx, fy - ty]
}

function renderReacher(s: Reacher2DState, p: Reacher2DParameters): Scene {
  const size = 400
  const reach = p.lengths[0] + p.lengths[1]
  const scale = size / (2.4 * reach)
  const at = (x: number, y: number): [number, number] => [size / 2 + x * scale, size / 2 - y * scale]
  const q1 = s[0] ?? 0
  const [ex, ey] = at(p.lengths[0] * Math.cos(q1), p.lengths[0] * Math.sin(q1))
  const tip = armTip(o, q1, s[1] ?? 0, p)
  const [fx, fy] = at(tip[0], tip[1])
  const [tx, ty] = at(s[4] ?? 0, s[5] ?? 0)
  const [cx, cy] = at(0, 0)
  return scene(
    size,
    size,
    [
      circle(cx, cy, reach * scale, { stroke: 'muted', strokeWidth: 1, id: 'workspace' }),
      circle(tx, ty, 6, { fill: 'tab3', id: 'target' }),
      line(cx, cy, ex, ey, { stroke: 'tab0', strokeWidth: 8, id: 'link1' }),
      line(ex, ey, fx, fy, { stroke: 'tab1', strokeWidth: 8, id: 'link2' }),
      circle(cx, cy, 6, { fill: 'fg' }),
      circle(fx, fy, 4, { fill: 'fg', id: 'fingertip' }),
    ],
    { title: 'Reacher2D' },
  )
}

const REACHER_DESCRIPTION = 'Move the tip of a two-link planar arm to a random target.'

/**
 * Create the 2-D reacher (see {@link reacher2d}).
 *
 * @param options - Physical parameters, target radius, control weight and horizon.
 * @returns The environment.
 * @concept environments
 * @example
 * ```ts
 * import { makeReacher2d } from 'whitebox-engine/methods/rl'
 * import { key } from 'whitebox-engine/random'
 * makeReacher2d().reset(key(0)).observation.length // => 10
 * ```
 */
export function makeReacher2d(
  options: Reacher2DOptions = {},
): Env<Reacher2DState, readonly number[], readonly number[]> {
  const p = reacher2dParameters(options)
  return {
    spec: spec('Reacher2D-v0', REACHER_DESCRIPTION, options.maxEpisodeSteps ?? 50),
    observationSpace: box(-BIG, BIG, [10]),
    actionSpace: box(-1, 1, [2], { label: 'joint torques' }),
    reset: (k) => {
      const [a = 0, b = 0, c = 0, d = 0, r = 0, phi = 0] = draws(k, 6, -1, 1)
      const radius = p.goalRadius * Math.sqrt((r + 1) / 2)
      const angle = Math.PI * phi
      const s = [0.1 * a, 0.1 * b, 0.005 * c, 0.005 * d, radius * Math.cos(angle), radius * Math.sin(angle)]
      return { state: s, observation: reacherObservation(s, p), info: {} }
    },
    step: (s, a) => {
      const next = reacher2dDynamics(s, a, p)
      const target = [s[4] ?? 0, s[5] ?? 0]
      // gymnasium scores the action and the fingertip distance before the step.
      const reward = reacherRewardOf(o, s, clipTo(a, -1, 1, 2), target, p, p.controlWeight)
      return {
        state: next,
        observation: reacherObservation(next, p),
        reward,
        terminated: false,
        truncated: false,
        info: {},
      }
    },
    render: (s) => renderReacher(s, p),
    observe: (s) => reacherObservation(s, p),
  }
}

/**
 * Reacher2D-v0 (gymnasium's Reacher with analytic dynamics in place of MuJoCo): a two-link arm in a horizontal plane,
 * torqued at both joints by actions in [-1, 1], must bring its fingertip to a target drawn uniformly on a disc of
 * radius 0.2. Reward: minus the fingertip distance minus `0.1 |u|^2` (Reacher-v5's weights, scored on the state before
 * the step as gymnasium does). Observations are gymnasium's ten: `cos q`, `sin q`, the target, `qd` and the fingertip
 * minus the target. No joint limits. `make` truncates at 50 steps.
 *
 * @concept environments
 * @example
 * ```ts
 * import { reacher2d } from 'whitebox-engine/methods/rl'
 * import { key } from 'whitebox-engine/random'
 * const r = reacher2d.create().step([0, 0, 0, 0, 0.21, 0], [1, 0], key(0)).reward
 * Number(r.toFixed(12)) // => -0.1
 * ```
 */
export const reacher2d: EnvDefinition<Reacher2DState, readonly number[], readonly number[]> = {
  id: 'Reacher2D-v0',
  description: REACHER_DESCRIPTION,
  options: dict({
    armature: real(0, 1, { default: 0.01, description: 'rotor inertia at each joint' }),
    damping: real(0, 1, { default: 0.05, description: 'joint damping' }),
    gain: real(0.01, 5, { default: 0.2, description: 'torque per unit action' }),
    dt: real(0.001, 0.1, { default: 0.02, unit: 's' }),
    goalRadius: real(0, 0.21, { default: 0.2 }),
    controlWeight: real(0, 1, { default: 0.1 }),
    maxEpisodeSteps: int(10, 10_000, { default: 50 }),
  }),
  create: (options) => makeReacher2d((options ?? {}) as Reacher2DOptions),
}

// ------------------------------------------------------------------------------------------------ double integrator

/** Options of the double integrator. */
export interface DoubleIntegratorOptions {
  readonly mass?: number
  readonly dt?: number
  readonly maxForce?: number
  /** Start positions and velocities are uniform in `[-initialRange, initialRange]` (default 1). */
  readonly initialRange?: number
  /** Cost weights `q_p p^2 + q_v v^2 + r u^2` (default 1, 1, 1: Q = I, R = 1). */
  readonly positionWeight?: number
  readonly velocityWeight?: number
  readonly controlWeight?: number
  readonly maxEpisodeSteps?: number
}

/** The resolved parameters of the double integrator. */
export interface DoubleIntegratorParameters {
  readonly mass: number
  readonly dt: number
  readonly maxForce: number
  readonly initialRange: number
  readonly positionWeight: number
  readonly velocityWeight: number
  readonly controlWeight: number
}

/**
 * The double-integrator parameters with defaults filled in (unit mass, dt 0.1, forces in [-10, 10], Q = I, R = 1).
 *
 * @param options - Any subset of the options.
 * @returns The parameters.
 * @example
 * ```ts
 * import { doubleIntegratorParameters } from 'whitebox-engine/methods/rl'
 * doubleIntegratorParameters().dt // => 0.1
 * ```
 */
export function doubleIntegratorParameters(options: DoubleIntegratorOptions = {}): DoubleIntegratorParameters {
  return {
    mass: options.mass ?? 1,
    dt: options.dt ?? 0.1,
    maxForce: options.maxForce ?? 10,
    initialRange: options.initialRange ?? 1,
    positionWeight: options.positionWeight ?? 1,
    velocityWeight: options.velocityWeight ?? 1,
    controlWeight: options.controlWeight ?? 1,
  }
}

/**
 * The exact zero-order-hold discretisation of `m p'' = u` (systems `c2dMatrices`, as scipy.signal.cont2discrete with
 * method 'zoh'):
 * `A = [[1, dt], [0, 1]]`, `B = [[dt^2 / 2m], [dt / m]]`.
 *
 * @param params - The parameters (mass and dt).
 * @returns `A` (2 x 2) and `B` (2 x 1).
 * @concept environments
 * @example
 * ```ts
 * import { doubleIntegratorMatrices } from 'whitebox-engine/methods/rl'
 * doubleIntegratorMatrices({ mass: 1, dt: 0.5 }) // => { A: [[1, 0.5], [0, 1]], B: [[0.125], [0.5]] }
 * ```
 */
export function doubleIntegratorMatrices(params: { readonly mass: number; readonly dt: number }): {
  readonly A: number[][]
  readonly B: number[][]
} {
  const d = c2dMatrices(
    [
      [0, 1],
      [0, 0],
    ],
    [[0], [1 / params.mass]],
    [[1, 0]],
    [[0]],
    params.dt,
    { method: 'zoh' },
  )
  return { A: toArray(d.a) as number[][], B: toArray(d.b) as number[][] }
}

/** The state of the double integrator: position and velocity. */
export type DoubleIntegratorState = readonly [number, number]

/**
 * One exact step of the double integrator under a force clipped to `[-maxForce, maxForce]`.
 *
 * @param state - Position and velocity.
 * @param action - The force.
 * @param params - The parameters.
 * @returns The next state.
 * @concept environments
 * @example
 * ```ts
 * import { doubleIntegratorDynamics, doubleIntegratorParameters } from 'whitebox-engine/methods/rl'
 * doubleIntegratorDynamics([0, 1], [0], doubleIntegratorParameters({ dt: 0.5 })) // => [0.5, 1]
 * ```
 */
export function doubleIntegratorDynamics(
  state: DoubleIntegratorState,
  action: readonly number[],
  params: DoubleIntegratorParameters,
): DoubleIntegratorState {
  const k = dampedMassCoefficients(params.mass, 0, params.dt)
  const u = clamp(action[0] ?? 0, -params.maxForce, params.maxForce)
  const [p, v] = dampedMassStep(o, k, [state[0]], [state[1]], [u]) as [number, number]
  return [p, v]
}

function renderDoubleIntegrator(s: DoubleIntegratorState): Scene {
  const width = 600
  const height = 120
  const scale = width / 8
  const x = clamp(width / 2 + s[0] * scale, 0, width)
  return scene(
    width,
    height,
    [
      line(0, 80, width, 80, { stroke: 'fg', id: 'track' }),
      line(width / 2, 70, width / 2, 90, { stroke: 'muted', id: 'origin' }),
      rect(x - 15, 50, 30, 30, { fill: 'tab0', id: 'mass', title: `p = ${s[0].toFixed(3)}, v = ${s[1].toFixed(3)}` }),
      line(x, 40, x + clamp(s[1] * scale * 0.5, -width, width), 40, { stroke: 'tab1', strokeWidth: 3, id: 'velocity' }),
    ],
    { title: 'DoubleIntegrator' },
  )
}

const DI_DESCRIPTION = 'Drive a unit mass on a line to the origin at rest, paying a quadratic cost (the LQR benchmark).'

/**
 * Create the double integrator (see {@link doubleIntegrator}).
 *
 * @param options - Mass, dt, force limit, start range, cost weights and horizon.
 * @returns The environment.
 * @concept environments
 * @example
 * ```ts
 * import { makeDoubleIntegrator } from 'whitebox-engine/methods/rl'
 * import { key } from 'whitebox-engine/random'
 * makeDoubleIntegrator().step([1, 0], [0], key(0)).reward // => -1
 * ```
 */
export function makeDoubleIntegrator(
  options: DoubleIntegratorOptions = {},
): Env<DoubleIntegratorState, readonly number[], readonly number[]> {
  const p = doubleIntegratorParameters(options)
  return {
    spec: spec('DoubleIntegrator-v0', DI_DESCRIPTION, options.maxEpisodeSteps ?? 200),
    observationSpace: box(-BIG, BIG, [2]),
    actionSpace: box(-p.maxForce, p.maxForce, [1], { label: 'force' }),
    reset: (k) => {
      const [a = 0, b = 0] = draws(k, 2, -p.initialRange, p.initialRange)
      const s: DoubleIntegratorState = [a, b]
      return { state: s, observation: s, info: {} }
    },
    step: (s, a) => {
      const u = clamp(a[0] ?? 0, -p.maxForce, p.maxForce)
      const reward = -(p.positionWeight * s[0] * s[0] + p.velocityWeight * s[1] * s[1] + p.controlWeight * u * u)
      const next = doubleIntegratorDynamics(s, a, p)
      return { state: next, observation: next, reward, terminated: false, truncated: false, info: {} }
    },
    render: renderDoubleIntegrator,
    observe: (s) => s,
  }
}

/**
 * DoubleIntegrator-v0: the one-dimensional LQR benchmark. A unit mass on a line, `p'' = u / m`, discretised exactly
 * (zero-order hold) at dt 0.1; the reward is minus `p^2 + v^2 + u^2` of the state before the step (Q = I, R = 1 by
 * default); no terminal state; `make` truncates at 200 steps. LQR on {@link doubleIntegratorMatrices} is optimal.
 *
 * @concept environments
 * @example
 * ```ts
 * import { doubleIntegrator } from 'whitebox-engine/methods/rl'
 * import { key } from 'whitebox-engine/random'
 * doubleIntegrator.create().step([0, 0], [1], key(0)).reward // => -1
 * ```
 */
export const doubleIntegrator: EnvDefinition<DoubleIntegratorState, readonly number[], readonly number[]> = {
  id: 'DoubleIntegrator-v0',
  description: DI_DESCRIPTION,
  options: dict({
    mass: real(0.1, 10, { default: 1, unit: 'kg' }),
    dt: real(0.001, 1, { default: 0.1, unit: 's' }),
    maxForce: real(0.1, 100, { default: 10 }),
    initialRange: real(0, 10, { default: 1 }),
    positionWeight: real(0, 100, { default: 1 }),
    velocityWeight: real(0, 100, { default: 1 }),
    controlWeight: real(0, 100, { default: 1 }),
    maxEpisodeSteps: int(10, 10_000, { default: 200 }),
  }),
  create: (options) => makeDoubleIntegrator((options ?? {}) as DoubleIntegratorOptions),
}

// ------------------------------------------------------------------------------------------------- planar quadrotor

/** Options of the planar quadrotor. */
export interface PlanarQuadrotorOptions {
  readonly mass?: number
  readonly inertia?: number
  readonly arm?: number
  readonly g?: number
  readonly dt?: number
  readonly substeps?: number
  /** Each rotor's maximum thrust (default `mass g`: either rotor alone can carry the weight). */
  readonly maxThrust?: number
  /** Goals are uniform in `[-goalRange, goalRange]^2` (default 1). */
  readonly goalRange?: number
  /** Weight of the squared thrust deviation from hover (default 0.01). */
  readonly controlWeight?: number
  readonly maxEpisodeSteps?: number
}

/** The resolved parameters of the planar quadrotor. */
export interface PlanarQuadrotorParameters extends QuadrotorParameters {
  readonly dt: number
  readonly substeps: number
  readonly maxThrust: number
  readonly goalRange: number
  readonly controlWeight: number
}

/**
 * The quadrotor parameters with defaults filled in: 1 kg, inertia 0.01, arm 0.2 m, g 9.81, dt 0.02.
 *
 * @param options - Any subset of the options.
 * @returns The parameters.
 * @example
 * ```ts
 * import { planarQuadrotorParameters } from 'whitebox-engine/methods/rl'
 * planarQuadrotorParameters().maxThrust // => 9.81
 * ```
 */
export function planarQuadrotorParameters(options: PlanarQuadrotorOptions = {}): PlanarQuadrotorParameters {
  const mass = options.mass ?? 1
  const g = options.g ?? 9.81
  return {
    mass,
    inertia: options.inertia ?? 0.01,
    arm: options.arm ?? 0.2,
    g,
    dt: options.dt ?? 0.02,
    substeps: options.substeps ?? 1,
    maxThrust: options.maxThrust ?? mass * g,
    goalRange: options.goalRange ?? 1,
    controlWeight: options.controlWeight ?? 0.01,
  }
}

/** The state of the quadrotor: `[x, z, theta, xd, zd, thetad, gx, gz]` (pose, velocities and goal). */
export type PlanarQuadrotorState = readonly number[]

/**
 * The continuous-time equations of motion of the planar quadrotor: `d/dt [x, z, theta, xd, zd, thetad]` under the
 * left and right thrusts.
 *
 * @param state - The pose and velocities (a goal after them is ignored).
 * @param action - The thrusts (unclipped).
 * @param params - The parameters.
 * @returns The time derivative of the pose and velocities.
 * @concept environments
 * @example
 * ```ts
 * import { planarQuadrotorDerivatives, planarQuadrotorParameters } from 'whitebox-engine/methods/rl'
 * planarQuadrotorDerivatives([0, 0, 0, 0, 0, 0], [0, 0], planarQuadrotorParameters({ g: 10 }))[4] // => -10
 * ```
 */
export function planarQuadrotorDerivatives(
  state: readonly number[],
  action: readonly number[],
  params: QuadrotorParameters,
): number[] {
  return quadrotorDerivatives(o, state, action, params)
}

/**
 * One step of the planar quadrotor: `m x'' = -(u1 + u2) sin theta`, `m z'' = (u1 + u2) cos theta - m g`,
 * `I theta'' = arm (u2 - u1)` (left and right rotors), integrated by RK4 with thrusts clipped to `[0, maxThrust]`.
 *
 * @param state - The state.
 * @param action - Left and right thrusts.
 * @param params - The parameters.
 * @returns The next state.
 * @concept environments
 * @example
 * ```ts
 * import { planarQuadrotorDynamics, planarQuadrotorParameters } from 'whitebox-engine/methods/rl'
 * const p = planarQuadrotorParameters({ g: 10 })
 * planarQuadrotorDynamics([0, 0, 0, 0, 0, 0, 0, 0], [5, 5], p) // => [0, 0, 0, 0, 0, 0, 0, 0]
 * ```
 */
export function planarQuadrotorDynamics(
  state: PlanarQuadrotorState,
  action: readonly number[],
  params: PlanarQuadrotorParameters,
): PlanarQuadrotorState {
  const u = clipTo(action, 0, params.maxThrust, 2)
  const next = rk4(o, (s) => quadrotorDerivatives(o, s, u, params), state.slice(0, 6), params.dt, params.substeps)
  return [...next, state[6] ?? 0, state[7] ?? 0]
}

function renderQuadrotor(s: PlanarQuadrotorState, p: PlanarQuadrotorParameters): Scene {
  const size = 500
  const half = Math.max(2, p.goalRange + 1)
  const scale = size / (2 * half)
  const at = (x: number, z: number): [number, number] => [size / 2 + x * scale, size / 2 - z * scale]
  const [x = 0, z = 0, th = 0] = s
  const c = Math.cos(th)
  const sn = Math.sin(th)
  const w = p.arm * 2
  const [lx, ly] = at(x - w * c, z - w * sn)
  const [rx, ry] = at(x + w * c, z + w * sn)
  const [gx, gy] = at(s[6] ?? 0, s[7] ?? 0)
  const rotor = (px: number, py: number, id: string) =>
    polygon(
      [
        [px - 12 * c, py + 12 * sn],
        [px + 12 * c, py - 12 * sn],
        [px + 12 * c - 4 * sn, py - 12 * sn - 4 * c],
        [px - 12 * c - 4 * sn, py + 12 * sn - 4 * c],
      ],
      { fill: 'tab1', id },
    )
  return scene(
    size,
    size,
    [
      circle(gx, gy, 6, { stroke: 'tab2', strokeWidth: 2, id: 'goal' }),
      line(lx, ly, rx, ry, {
        stroke: 'tab0',
        strokeWidth: 6,
        id: 'body',
        title: `pose (${x.toFixed(2)}, ${z.toFixed(2)})`,
      }),
      rotor(lx, ly, 'left-rotor'),
      rotor(rx, ry, 'right-rotor'),
    ],
    { title: 'PlanarQuadrotor' },
  )
}

const QUAD_DESCRIPTION = 'Fly a planar quadrotor with two rotor thrusts to a goal and hover there.'

/**
 * Create the planar quadrotor (see {@link planarQuadrotor}).
 *
 * @param options - Physical parameters, thrust limit, goal range, control weight and horizon.
 * @returns The environment.
 * @concept environments
 * @example
 * ```ts
 * import { makePlanarQuadrotor } from 'whitebox-engine/methods/rl'
 * import { key } from 'whitebox-engine/random'
 * makePlanarQuadrotor().reset(key(0)).observation.length // => 8
 * ```
 */
export function makePlanarQuadrotor(
  options: PlanarQuadrotorOptions = {},
): Env<PlanarQuadrotorState, readonly number[], readonly number[]> {
  const p = planarQuadrotorParameters(options)
  return {
    spec: spec('PlanarQuadrotor-v0', QUAD_DESCRIPTION, options.maxEpisodeSteps ?? 300),
    observationSpace: box(-BIG, BIG, [8]),
    actionSpace: box(0, p.maxThrust, [2], { label: 'left and right thrusts' }),
    reset: (k) => {
      const n = draws(k, 8, -1, 1)
      const s = [...n.slice(0, 6).map((x) => 0.1 * x), p.goalRange * (n[6] ?? 0), p.goalRange * (n[7] ?? 0)]
      return { state: s, observation: s, info: {} }
    },
    step: (s, a) => {
      const u = clipTo(a, 0, p.maxThrust, 2)
      const reward = quadrotorRewardOf(o, s, u, [s[6] ?? 0, s[7] ?? 0], p, p.controlWeight)
      const next = planarQuadrotorDynamics(s, a, p)
      return { state: next, observation: next, reward, terminated: false, truncated: false, info: {} }
    },
    render: (s) => renderQuadrotor(s, p),
    observe: (s) => s,
  }
}

/**
 * PlanarQuadrotor-v0: a 2-D quadrotor (six states: position, tilt and their rates) driven by its left and right rotor
 * thrusts in `[0, m g]`, flying to a goal drawn in `[-1, 1]^2`. Reward: minus `|p - goal|^2 + 0.1 theta^2 +
 * 0.01 |velocities|^2 + 0.01 |u - hover|^2` of the state before the step. RK4 at dt 0.02; no terminal state; `make`
 * truncates at 300 steps. Hovering at `u = m g / 2` per rotor is an equilibrium.
 *
 * @concept environments
 * @example
 * ```ts
 * import { planarQuadrotor } from 'whitebox-engine/methods/rl'
 * import { key } from 'whitebox-engine/random'
 * planarQuadrotor.create({ g: 10 }).step([0, 0, 0, 0, 0, 0, 1, 0], [5, 5], key(0)).reward // => -1
 * ```
 */
export const planarQuadrotor: EnvDefinition<PlanarQuadrotorState, readonly number[], readonly number[]> = {
  id: 'PlanarQuadrotor-v0',
  description: QUAD_DESCRIPTION,
  options: dict({
    mass: real(0.1, 10, { default: 1, unit: 'kg' }),
    inertia: real(0.001, 1, { default: 0.01 }),
    arm: real(0.05, 1, { default: 0.2, unit: 'm' }),
    g: real(0, 20, { default: 9.81 }),
    dt: real(0.001, 0.1, { default: 0.02, unit: 's' }),
    goalRange: real(0, 5, { default: 1 }),
    controlWeight: real(0, 1, { default: 0.01 }),
    maxEpisodeSteps: int(10, 10_000, { default: 300 }),
  }),
  create: (options) => makePlanarQuadrotor((options ?? {}) as PlanarQuadrotorOptions),
}

// --------------------------------------------------------------------------------------------- cart double pendulum

/** Options of the cart double pendulum. */
export interface CartDoublePendulumOptions {
  readonly cartMass?: number
  readonly masses?: readonly [number, number]
  readonly lengths?: readonly [number, number]
  readonly g?: number
  readonly cartDamping?: number
  readonly jointDamping?: number
  readonly dt?: number
  /** RK4 substeps per step (default 2). */
  readonly substeps?: number
  readonly maxForce?: number
  /** Start hanging down (no height termination) instead of near upright. */
  readonly swingUp?: boolean
  /** Episodes end when the cart leaves `[-xLimit, xLimit]` (default 2.4). */
  readonly xLimit?: number
  /** When balancing, episodes end when the tip falls to this height (default 5/6 of the total length). */
  readonly terminateHeight?: number
  readonly maxEpisodeSteps?: number
}

/** The resolved parameters of the cart double pendulum. */
export interface CartDoublePendulumEnvParameters extends CartDoublePendulumParameters {
  readonly dt: number
  readonly substeps: number
  readonly maxForce: number
  readonly swingUp: boolean
  readonly xLimit: number
  readonly terminateHeight: number
}

/**
 * The cart double pendulum parameters with defaults filled in: a 1 kg cart, two 0.1 kg uniform poles of 0.6 m
 * (gymnasium's lengths), g 9.81, no friction, dt 0.02 in two RK4 substeps, forces in [-20, 20].
 *
 * @param options - Any subset of the options.
 * @returns The parameters.
 * @example
 * ```ts
 * import { cartDoublePendulumParameters } from 'whitebox-engine/methods/rl'
 * cartDoublePendulumParameters().terminateHeight // => 1
 * ```
 */
export function cartDoublePendulumParameters(options: CartDoublePendulumOptions = {}): CartDoublePendulumEnvParameters {
  const lengths = options.lengths ?? [0.6, 0.6]
  return {
    cartMass: options.cartMass ?? 1,
    masses: options.masses ?? [0.1, 0.1],
    lengths,
    g: options.g ?? 9.81,
    cartDamping: options.cartDamping ?? 0,
    jointDamping: options.jointDamping ?? 0,
    dt: options.dt ?? 0.02,
    substeps: options.substeps ?? 2,
    maxForce: options.maxForce ?? 20,
    swingUp: options.swingUp ?? false,
    xLimit: options.xLimit ?? 2.4,
    terminateHeight: options.terminateHeight ?? Number((((lengths[0] + lengths[1]) * 5) / 6).toFixed(12)),
  }
}

/** The state of the cart double pendulum: `[x, theta1, theta2, xd, thetad1, thetad2]` (angles from upright). */
export type CartDoublePendulumState = readonly number[]

/**
 * The continuous-time equations of motion of the cart double pendulum: `d/dt state` under a horizontal force.
 *
 * @param state - The state.
 * @param force - The force on the cart.
 * @param params - The parameters.
 * @returns The time derivative of the state.
 * @concept environments
 * @example
 * ```ts
 * import { cartDoublePendulumDerivatives, cartDoublePendulumParameters } from 'whitebox-engine/methods/rl'
 * cartDoublePendulumDerivatives([0, 0, 0, 0, 0, 0], 0, cartDoublePendulumParameters()) // => [0, 0, 0, 0, 0, 0]
 * ```
 */
export function cartDoublePendulumDerivatives(
  state: CartDoublePendulumState,
  force: number,
  params: CartDoublePendulumParameters,
): number[] {
  return cartDoublePendulumDerivativesOf(o, state, force, params)
}

/**
 * One step of the cart double pendulum: the Lagrangian equations of motion ({@link cartDoublePendulumDerivatives})
 * integrated by RK4 in `substeps` substeps of `dt`, the force clipped to `[-maxForce, maxForce]`.
 *
 * @param state - The state.
 * @param action - The force.
 * @param params - The parameters.
 * @returns The next state.
 * @concept environments
 * @example
 * ```ts
 * import { cartDoublePendulumDynamics, cartDoublePendulumParameters } from 'whitebox-engine/methods/rl'
 * cartDoublePendulumDynamics([0, 0, 0, 0, 0, 0], [0], cartDoublePendulumParameters()) // => [0, 0, 0, 0, 0, 0]
 * ```
 */
export function cartDoublePendulumDynamics(
  state: CartDoublePendulumState,
  action: readonly number[],
  params: CartDoublePendulumEnvParameters,
): CartDoublePendulumState {
  const f = clamp(action[0] ?? 0, -params.maxForce, params.maxForce)
  return rk4(o, (s) => cartDoublePendulumDerivativesOf(o, s, f, params), state, params.dt, params.substeps)
}

/**
 * The mechanical energy of the cart double pendulum (kinetic plus potential, zero potential at the cart's height),
 * conserved without force or friction.
 *
 * @param state - The state.
 * @param params - The parameters.
 * @returns The energy.
 * @example
 * ```ts
 * import { cartDoublePendulumEnergy, cartDoublePendulumParameters } from 'whitebox-engine/methods/rl'
 * const p = cartDoublePendulumParameters({ g: 10, masses: [1, 1], lengths: [1, 1] })
 * cartDoublePendulumEnergy([0, 0, 0, 0, 0, 0], p) // => 20
 * ```
 */
export function cartDoublePendulumEnergy(state: CartDoublePendulumState, params: CartDoublePendulumParameters): number {
  return cartDoublePendulumEnergyOf(o, state, params)
}

function renderCartDoublePendulum(s: CartDoublePendulumState, p: CartDoublePendulumEnvParameters): Scene {
  const width = 600
  const height = 400
  const scale = width / (2 * (p.xLimit + 0.4))
  const cartY = height * 0.6
  const at = (x: number, y: number): [number, number] => [width / 2 + x * scale, cartY - y * scale]
  const [x = 0, th1 = 0, _th2 = 0] = s
  const [cx, cy] = at(x, 0)
  const [ex, ey] = at(x + p.lengths[0] * Math.sin(th1), p.lengths[0] * Math.cos(th1))
  const tip = cartDoublePendulumTipOf(o, s, p)
  const [tx, ty] = at(tip[0], tip[1])
  return scene(
    width,
    height,
    [
      line(0, cartY, width, cartY, { stroke: 'fg', id: 'track' }),
      rect(cx - 25, cy - 15, 50, 30, { fill: 'fg', id: 'cart', title: `cart at x = ${x.toFixed(3)}` }),
      line(cx, cy, ex, ey, { stroke: 'tab0', strokeWidth: 6, id: 'pole1' }),
      line(ex, ey, tx, ty, { stroke: 'tab1', strokeWidth: 6, id: 'pole2' }),
      circle(cx, cy, 4, { fill: 'tab2' }),
      circle(ex, ey, 4, { fill: 'tab2' }),
    ],
    { title: 'CartDoublePendulum' },
  )
}

const CDP_DESCRIPTION = 'Balance (or swing up) two linked poles on a cart pushed by a horizontal force.'

/**
 * Create the cart double pendulum (see {@link cartDoublePendulum}).
 *
 * @param options - Physical parameters, force limit, task (balance or swing-up), limits and horizon.
 * @returns The environment.
 * @concept environments
 * @example
 * ```ts
 * import { makeCartDoublePendulum } from 'whitebox-engine/methods/rl'
 * import { key } from 'whitebox-engine/random'
 * makeCartDoublePendulum({ swingUp: true }).reset(key(0)).state[1] > 3 // => true
 * ```
 */
export function makeCartDoublePendulum(
  options: CartDoublePendulumOptions = {},
): Env<CartDoublePendulumState, readonly number[], readonly number[]> {
  const p = cartDoublePendulumParameters(options)
  const id = 'CartDoublePendulum-v0'
  return {
    spec: spec(id, CDP_DESCRIPTION, options.maxEpisodeSteps ?? 1000),
    observationSpace: box(-BIG, BIG, [6]),
    actionSpace: box(-p.maxForce, p.maxForce, [1], { label: 'force on the cart' }),
    reset: (k) => {
      const n = draws(k, 6, -0.1, 0.1)
      const down = p.swingUp ? Math.PI : 0
      const s = [n[0] ?? 0, down + (n[1] ?? 0), down + (n[2] ?? 0), n[3] ?? 0, n[4] ?? 0, n[5] ?? 0]
      return { state: s, observation: s, info: {} }
    },
    step: (s, a) => {
      const next = cartDoublePendulumDynamics(s, a, p)
      const [, tipY] = cartDoublePendulumTipOf(o, next, p)
      const fell = !p.swingUp && tipY <= p.terminateHeight
      const terminated = fell || Math.abs(next[0] ?? 0) > p.xLimit
      const reward = cartDoublePendulumRewardOf(o, next, p)
      return { state: next, observation: next, reward, terminated, truncated: false, info: {} }
    },
    render: (s) => renderCartDoublePendulum(s, p),
    observe: (s) => s,
  }
}

/**
 * CartDoublePendulum-v0: a cart on a track carrying two linked uniform poles (the double inverted pendulum), pushed by
 * a force in [-20, 20] N. The equations of motion come from the Lagrangian and are integrated by RK4 (two substeps of
 * 0.01 s per 0.02 s step). The reward is gymnasium InvertedDoublePendulum's: 10 minus `0.01 x_tip^2 + (y_tip - L)^2 +
 * 0.001 w1^2 + 0.005 w2^2` after the step. Balancing episodes start within 0.1 of upright and end when the tip falls
 * to 5/6 of the total length `L` or the cart leaves [-2.4, 2.4]; with `swingUp` they start hanging down and only the
 * track ends them. `make` truncates at 1000 steps.
 *
 * @concept environments
 * @example
 * ```ts
 * import { cartDoublePendulum } from 'whitebox-engine/methods/rl'
 * import { key } from 'whitebox-engine/random'
 * cartDoublePendulum.create().step([0, 0, 0, 0, 0, 0], [0], key(0)).reward // => 10
 * ```
 */
export const cartDoublePendulum: EnvDefinition<CartDoublePendulumState, readonly number[], readonly number[]> = {
  id: 'CartDoublePendulum-v0',
  description: CDP_DESCRIPTION,
  options: dict({
    cartMass: real(0.1, 10, { default: 1, unit: 'kg' }),
    g: real(0, 20, { default: 9.81 }),
    cartDamping: real(0, 5, { default: 0 }),
    jointDamping: real(0, 1, { default: 0 }),
    dt: real(0.001, 0.05, { default: 0.02, unit: 's' }),
    maxForce: real(1, 100, { default: 20, unit: 'N' }),
    swingUp: bool({ default: false, description: 'start hanging down' }),
    xLimit: real(0.5, 10, { default: 2.4 }),
    maxEpisodeSteps: int(10, 10_000, { default: 1000 }),
  }),
  create: (options) => makeCartDoublePendulum((options ?? {}) as CartDoublePendulumOptions),
}
