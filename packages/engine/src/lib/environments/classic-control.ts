/**
 * The classic control environments of gymnasium, reproduced step for step: CartPole (balance a pole on a cart),
 * MountainCar and its continuous variant (an underpowered car must rock its way out of a valley), Pendulum (swing up
 * and balance with limited torque) and Acrobot (a two-link pendulum that must swing its tip above a bar). Each exposes
 * its dynamics as a plain function, as gymnasium computes it, and draws its state as a scene.
 */

import { numberOr } from '../base/index.ts'
import { uniform } from '../random/index.ts'
import type { Env, EnvDefinition } from '../rl/index.ts'
import { circle, line, path, polygon, rect, type Scene, scene } from '../scene/index.ts'
import { bool, box, dict, discrete, real } from '../space/index.ts'

const draws = (k: Parameters<typeof uniform>[0], n: number, low: number, high: number): number[] =>
  Array.from(uniform(k, [n], { low, high }).data)

// Rotate a point (x, y) by an angle (radians, counter-clockwise in a y-up frame).
const rotate = (x: number, y: number, a: number): [number, number] => [
  x * Math.cos(a) - y * Math.sin(a),
  x * Math.sin(a) + y * Math.cos(a),
]

// ---------------------------------------------------------------------------------------------------------- CartPole

/** The state of CartPole: cart position and velocity, pole angle (radians from upright) and angular velocity. */
export type CartPoleState = readonly [position: number, velocity: number, angle: number, angularVelocity: number]

// gymnasium's CartPoleEnv constants (Barto, Sutton and Anderson, 1983).
const GRAVITY = 9.8
const MASS_CART = 1
const MASS_POLE = 0.1
const TOTAL_MASS = MASS_CART + MASS_POLE
const HALF_LENGTH = 0.5
const POLE_MASS_LENGTH = MASS_POLE * HALF_LENGTH
const FORCE = 10
const TAU = 0.02
const THETA_LIMIT = (12 * 2 * Math.PI) / 360
const X_LIMIT = 2.4

/**
 * One Euler step of the cart-pole dynamics under a push left (0) or right (1), exactly as gymnasium computes it.
 *
 * @param state - The state.
 * @param action - 0 pushes left, 1 pushes right.
 * @returns The next state.
 * @concept environments
 * @example
 * ```ts
 * import { cartPoleDynamics } from 'mdml-engine/environments'
 * cartPoleDynamics([0, 0, 0, 0], 1)[1] > 0 // => true
 * ```
 */
export function cartPoleDynamics(state: CartPoleState, action: number): CartPoleState {
  const [x, xDot, theta, thetaDot] = state
  const force = action === 1 ? FORCE : -FORCE
  const cos = Math.cos(theta)
  const sin = Math.sin(theta)
  const temp = (force + POLE_MASS_LENGTH * thetaDot ** 2 * sin) / TOTAL_MASS
  const thetaAcc = (GRAVITY * sin - cos * temp) / (HALF_LENGTH * (4 / 3 - (MASS_POLE * cos ** 2) / TOTAL_MASS))
  const xAcc = temp - (POLE_MASS_LENGTH * thetaAcc * cos) / TOTAL_MASS
  return [x + TAU * xDot, xDot + TAU * xAcc, theta + TAU * thetaDot, thetaDot + TAU * thetaAcc]
}

/**
 * Whether a CartPole state is terminal (the cart left the track or the pole fell past 12 degrees).
 *
 * @param state - The state.
 * @returns Whether the episode ends.
 * @example
 * ```ts
 * import { cartPoleTerminated } from 'mdml-engine/environments'
 * cartPoleTerminated([3, 0, 0, 0]) // => true
 * ```
 */
export function cartPoleTerminated(state: CartPoleState): boolean {
  const [x, , theta] = state
  return x < -X_LIMIT || x > X_LIMIT || theta < -THETA_LIMIT || theta > THETA_LIMIT
}

/**
 * Draw a CartPole state: track, cart, pole and axle (gymnasium's 600 x 400 layout).
 *
 * @param state - The state.
 * @returns The scene.
 * @example
 * ```ts
 * import { renderCartPole } from 'mdml-engine/environments'
 * renderCartPole([0, 0, 0, 0]).width // => 600
 * ```
 */
export function renderCartPole(state: CartPoleState): Scene {
  const width = 600
  const height = 400
  const scale = width / (2 * X_LIMIT)
  const [x, , theta] = state
  const cartY = height - 100
  const cartX = x * scale + width / 2
  const poleLength = scale * 2 * HALF_LENGTH
  const pw = 5
  // The pole as a rotated rectangle from the axle (y grows downwards in scenes).
  const corners: [number, number][] = [
    [-pw, 0],
    [pw, 0],
    [pw, poleLength],
    [-pw, poleLength],
  ].map(([px, py]) => [
    cartX + (px ?? 0) * Math.cos(theta) + (py ?? 0) * Math.sin(theta),
    cartY - 15 - (py ?? 0) * Math.cos(theta) + (px ?? 0) * Math.sin(theta),
  ])
  return scene(
    width,
    height,
    [
      line(0, cartY, width, cartY, { stroke: 'fg', strokeWidth: 1 }),
      rect(cartX - 25, cartY - 30, 50, 30, { fill: 'fg', id: 'cart', title: `cart at x = ${x.toFixed(3)}` }),
      polygon(corners, { fill: 'tab1', id: 'pole', title: `pole at ${((theta * 180) / Math.PI).toFixed(1)} degrees` }),
      circle(cartX, cartY - 15, 5, { fill: 'tab0', id: 'axle' }),
    ],
    { title: 'CartPole' },
  )
}

const CARTPOLE_DESCRIPTION = 'Balance a pole on a cart by pushing the cart left or right.'

/**
 * CartPole-v1 (gymnasium): keep a pole balanced on a cart by pushing it left or right. Reward 1 per step, including
 * the terminating step (with `suttonBartoReward`, 0 per step and -1 on failure); episodes end when the pole passes 12
 * degrees or the cart leaves [-2.4, 2.4], and `make` truncates them at 500 steps. Observations are the float64 state
 * (gymnasium returns float32 copies).
 *
 * @concept environments
 * @index structure CartPole-v1 environment definition.
 * @example
 * ```ts
 * import { cartPole } from 'mdml-engine/environments'
 * import { key } from 'mdml-engine/random'
 * const env = cartPole.create()
 * env.step(env.reset(key(0)).state, 1, key(1)).reward // => 1
 * ```
 */
export const cartPole: EnvDefinition<CartPoleState, readonly number[], number> = {
  id: 'CartPole-v1',
  description: CARTPOLE_DESCRIPTION,
  options: dict({ suttonBartoReward: bool({ default: false, description: 'reward 0 per step and -1 on failure' }) }),
  create: (options): Env<CartPoleState, readonly number[], number> => {
    const sb = options?.['suttonBartoReward'] === true
    return {
      spec: {
        id: 'CartPole-v1',
        description: CARTPOLE_DESCRIPTION,
        maxEpisodeSteps: 500,
        rewardThreshold: 475,
        deterministic: true,
      },
      observationSpace: box(
        [-X_LIMIT * 2, -Number.MAX_VALUE, -THETA_LIMIT * 2, -Number.MAX_VALUE],
        [X_LIMIT * 2, Number.MAX_VALUE, THETA_LIMIT * 2, Number.MAX_VALUE],
        [4],
      ),
      actionSpace: discrete(2, { label: 'push left (0) or right (1)' }),
      reset: (k) => {
        const s = draws(k, 4, -0.05, 0.05) as unknown as CartPoleState
        return { state: s, observation: s, info: {} }
      },
      step: (state, action) => {
        const next = cartPoleDynamics(state, action)
        const terminated = cartPoleTerminated(next)
        const reward = sb ? (terminated ? -1 : 0) : 1
        return { state: next, observation: next, reward, terminated, truncated: false, info: {} }
      },
      render: renderCartPole,
      observe: (s) => s,
    }
  },
}

// ------------------------------------------------------------------------------------------------------- MountainCar

/** The state of MountainCar: position and velocity. */
export type MountainCarState = readonly [position: number, velocity: number]

const MC_MIN_POSITION = -1.2
const MC_MAX_POSITION = 0.6
const MC_MAX_SPEED = 0.07
const MC_FORCE = 0.001
const MC_GRAVITY = 0.0025

/**
 * One step of the discrete MountainCar dynamics (gymnasium's MountainCarEnv.step): action 0 pushes left, 1 coasts and
 * 2 pushes right.
 *
 * @param state - Position and velocity.
 * @param action - The push.
 * @returns The next state.
 * @example
 * ```ts
 * import { mountainCarDynamics } from 'mdml-engine/environments'
 * mountainCarDynamics([-1.2, -0.01], 0)[1] // => 0
 * ```
 */
export function mountainCarDynamics(state: MountainCarState, action: number): MountainCarState {
  let [position, velocity] = state
  velocity += (action - 1) * MC_FORCE + Math.cos(3 * position) * -MC_GRAVITY
  velocity = Math.min(MC_MAX_SPEED, Math.max(-MC_MAX_SPEED, velocity))
  position += velocity
  position = Math.min(MC_MAX_POSITION, Math.max(MC_MIN_POSITION, position))
  if (position === MC_MIN_POSITION && velocity < 0) velocity = 0
  return [position, velocity]
}

function renderMountainCar(state: MountainCarState, goal: number, title: string): Scene {
  const width = 600
  const height = 400
  const scale = width / (MC_MAX_POSITION - MC_MIN_POSITION)
  const hill = (x: number): number => Math.sin(3 * x) * 0.45 + 0.55
  const toScreen = (x: number, y: number): [number, number] => [x, height - y]
  const track: [number, number][] = []
  for (let i = 0; i < 100; i++) {
    const x = MC_MIN_POSITION + ((MC_MAX_POSITION - MC_MIN_POSITION) * i) / 99
    track.push(toScreen((x - MC_MIN_POSITION) * scale, hill(x) * scale))
  }
  const pos = state[0]
  const angle = Math.cos(3 * pos)
  const clearance = 10
  const base = (pos - MC_MIN_POSITION) * scale
  const baseY = clearance + hill(pos) * scale
  const car = (
    [
      [-20, 0],
      [-20, 20],
      [20, 20],
      [20, 0],
    ] as const
  ).map(([cx, cy]) => {
    const [rx, ry] = rotate(cx, cy, angle)
    return toScreen(rx + base, ry + baseY)
  })
  const wheels = [10, -10].map((wx) => {
    const [rx, ry] = rotate(wx, 0, angle)
    const [sx, sy] = toScreen(rx + base, ry + baseY)
    return circle(sx, sy, 8, { fill: 'muted' })
  })
  const flagX = (goal - MC_MIN_POSITION) * scale
  const flagY1 = hill(goal) * scale
  const flagY2 = flagY1 + 50
  const [fx, fy1] = toScreen(flagX, flagY1)
  const [, fy2] = toScreen(flagX, flagY2)
  return scene(
    width,
    height,
    [
      path(track, { stroke: 'fg', strokeWidth: 1, id: 'track' }),
      line(fx, fy1, fx, fy2, { stroke: 'fg', id: 'flagpole' }),
      polygon(
        [
          [fx, fy2],
          [fx, fy2 + 10],
          [fx + 25, fy2 + 5],
        ],
        { fill: '#cccc00', id: 'flag' },
      ),
      polygon(car, { fill: 'fg', id: 'car', title: `car at x = ${pos.toFixed(3)}` }),
      ...wheels,
    ],
    { title },
  )
}

const MC_DESCRIPTION = 'Drive an underpowered car up a hill by building momentum.'

/**
 * MountainCar-v0 (gymnasium; Moore 1990): an underpowered car in a valley must rock back and forth to reach the flag
 * at position 0.5. Actions push left, coast or push right; reward -1 per step; `make` truncates at 200 steps. Option
 * `goalVelocity` (default 0) also requires a minimum speed at the goal.
 *
 * @concept environments
 * @example
 * ```ts
 * import { mountainCar } from 'mdml-engine/environments'
 * import { key } from 'mdml-engine/random'
 * const env = mountainCar.create()
 * env.step(env.reset(key(0)).state, 2, key(1)).reward // => -1
 * ```
 */
export const mountainCar: EnvDefinition<MountainCarState, readonly number[], number> = {
  id: 'MountainCar-v0',
  description: MC_DESCRIPTION,
  options: dict({ goalVelocity: real(0, MC_MAX_SPEED, { default: 0, description: 'minimum speed at the goal' }) }),
  create: (options): Env<MountainCarState, readonly number[], number> => {
    const goalVelocity = numberOr(options, 'goalVelocity', 0)
    return {
      spec: {
        id: 'MountainCar-v0',
        description: MC_DESCRIPTION,
        maxEpisodeSteps: 200,
        rewardThreshold: -110,
        deterministic: true,
      },
      observationSpace: box([MC_MIN_POSITION, -MC_MAX_SPEED], [MC_MAX_POSITION, MC_MAX_SPEED], [2]),
      actionSpace: discrete(3, { label: 'push left (0), coast (1) or push right (2)' }),
      reset: (k) => {
        const s: MountainCarState = [draws(k, 1, -0.6, -0.4)[0] ?? -0.5, 0]
        return { state: s, observation: s, info: {} }
      },
      step: (state, action) => {
        const next = mountainCarDynamics(state, action)
        const terminated = next[0] >= 0.5 && next[1] >= goalVelocity
        return { state: next, observation: next, reward: -1, terminated, truncated: false, info: {} }
      },
      render: (s) => renderMountainCar(s, 0.5, 'MountainCar'),
      observe: (s) => s,
    }
  },
}

const MCC_POWER = 0.0015
const MCC_GOAL = 0.45

/**
 * One step of the continuous MountainCar dynamics (gymnasium's Continuous_MountainCarEnv.step). The state is stored
 * in float32 after every step, as gymnasium does, so trajectories agree with it to the bit.
 *
 * @param state - Position and velocity.
 * @param force - The applied force, clipped to [-1, 1].
 * @returns The next state.
 * @example
 * ```ts
 * import { continuousMountainCarDynamics } from 'mdml-engine/environments'
 * continuousMountainCarDynamics([-0.5, 0], 1)[1] > 0 // => true
 * ```
 */
export function continuousMountainCarDynamics(state: MountainCarState, force: number): MountainCarState {
  let [position, velocity] = state
  const f = Math.min(Math.max(force, -1), 1)
  velocity += f * MCC_POWER - 0.0025 * Math.cos(Math.fround(3 * position))
  // gymnasium's arithmetic mixes float32 state with Python floats: a clipped velocity is a Python float, so the
  // position update then happens in float32 (numpy's promotion rules); an unclipped one is float64.
  let clipped = false
  if (velocity > MC_MAX_SPEED) {
    velocity = MC_MAX_SPEED
    clipped = true
  }
  if (velocity < -MC_MAX_SPEED) {
    velocity = -MC_MAX_SPEED
    clipped = true
  }
  position = clipped ? Math.fround(position + Math.fround(velocity)) : position + velocity
  if (position > MC_MAX_POSITION) position = MC_MAX_POSITION
  if (position < MC_MIN_POSITION) position = MC_MIN_POSITION
  if (position === MC_MIN_POSITION && velocity < 0) velocity = 0
  return [Math.fround(position), Math.fround(velocity)]
}

const MCC_DESCRIPTION = 'Drive an underpowered car up a hill with a continuous force, paying for effort.'

/**
 * MountainCarContinuous-v0 (gymnasium): the mountain car with a continuous force in [-1, 1]. Reward +100 on reaching
 * the flag at 0.45 minus 0.1 times the squared force each step; `make` truncates at 999 steps.
 *
 * @concept environments
 * @example
 * ```ts
 * import { mountainCarContinuous } from 'mdml-engine/environments'
 * import { key } from 'mdml-engine/random'
 * const env = mountainCarContinuous.create()
 * env.step(env.reset(key(0)).state, [1], key(1)).reward // => -0.1
 * ```
 */
export const mountainCarContinuous: EnvDefinition<MountainCarState, readonly number[], readonly number[]> = {
  id: 'MountainCarContinuous-v0',
  description: MCC_DESCRIPTION,
  options: dict({ goalVelocity: real(0, MC_MAX_SPEED, { default: 0, description: 'minimum speed at the goal' }) }),
  create: (options): Env<MountainCarState, readonly number[], readonly number[]> => {
    const goalVelocity = numberOr(options, 'goalVelocity', 0)
    return {
      spec: {
        id: 'MountainCarContinuous-v0',
        description: MCC_DESCRIPTION,
        maxEpisodeSteps: 999,
        rewardThreshold: 90,
        deterministic: true,
      },
      observationSpace: box([MC_MIN_POSITION, -MC_MAX_SPEED], [MC_MAX_POSITION, MC_MAX_SPEED], [2]),
      actionSpace: box(-1, 1, [1], { label: 'force' }),
      reset: (k) => {
        const s: MountainCarState = [Math.fround(draws(k, 1, -0.6, -0.4)[0] ?? -0.5), 0]
        return { state: s, observation: s, info: {} }
      },
      step: (state, action) => {
        const a = action[0] ?? 0
        const next = continuousMountainCarDynamics(state, a)
        const terminated = next[0] >= MCC_GOAL && next[1] >= goalVelocity
        const reward = (terminated ? 100 : 0) - a ** 2 * 0.1
        return { state: next, observation: next, reward, terminated, truncated: false, info: {} }
      },
      render: (s) => renderMountainCar(s, MCC_GOAL, 'MountainCarContinuous'),
      observe: (s) => s,
    }
  },
}

// ---------------------------------------------------------------------------------------------------------- Pendulum

/** The state of Pendulum: angle (radians, 0 upright) and angular velocity. */
export type PendulumState = readonly [angle: number, angularVelocity: number]

const PENDULUM_MAX_SPEED = 8
const PENDULUM_MAX_TORQUE = 2
const PENDULUM_DT = 0.05

/**
 * An angle wrapped into [-pi, pi) (gymnasium's `angle_normalize`, with Python's floored modulo).
 *
 * @param x - The angle.
 * @returns The wrapped angle.
 * @example
 * ```ts
 * import { angleNormalize } from 'mdml-engine/environments'
 * angleNormalize(3 * Math.PI) // => -3.141592653589793
 * ```
 */
export function angleNormalize(x: number): number {
  const p = 2 * Math.PI
  let m = (x + Math.PI) % p
  if (m !== 0 && m < 0) m += p
  return m - Math.PI
}

/**
 * One step of the pendulum dynamics and its cost (gymnasium's PendulumEnv.step).
 *
 * @param state - Angle and angular velocity.
 * @param torque - The torque, clipped to [-2, 2].
 * @param g - Gravity (default 10).
 * @returns The next state and the cost of the step (the reward is its negative).
 * @example
 * ```ts
 * import { pendulumDynamics } from 'mdml-engine/environments'
 * pendulumDynamics([0, 0], 0).cost // => 0
 * ```
 */
export function pendulumDynamics(
  state: PendulumState,
  torque: number,
  g = 10,
): { readonly state: PendulumState; readonly cost: number } {
  const [th, thdot] = state
  const m = 1
  const l = 1
  const u = Math.min(PENDULUM_MAX_TORQUE, Math.max(-PENDULUM_MAX_TORQUE, torque))
  const a = angleNormalize(th)
  const cost = a * a + 0.1 * (thdot * thdot) + 0.001 * (u * u)
  let newthdot = thdot + (((3 * g) / (2 * l)) * Math.sin(th) + (3.0 / (m * (l * l))) * u) * PENDULUM_DT
  newthdot = Math.min(PENDULUM_MAX_SPEED, Math.max(-PENDULUM_MAX_SPEED, newthdot))
  const newth = th + newthdot * PENDULUM_DT
  return { state: [newth, newthdot], cost }
}

function renderPendulum(state: PendulumState): Scene {
  const size = 500
  const scale = size / 4.4
  const c = size / 2
  const rodLength = scale
  const width = 0.2 * scale
  const angle = state[0] + Math.PI / 2
  // gymnasium draws in a y-up frame; flip y for the scene.
  const corners = (
    [
      [0, -width / 2],
      [0, width / 2],
      [rodLength, width / 2],
      [rodLength, -width / 2],
    ] as const
  ).map(([x, y]): [number, number] => {
    const [rx, ry] = rotate(x, y, angle)
    return [c + rx, c - ry]
  })
  const [ex, ey] = rotate(rodLength, 0, angle)
  return scene(
    size,
    size,
    [
      polygon(corners, { fill: '#cc4c4c', id: 'rod', title: `angle ${state[0].toFixed(3)} rad` }),
      circle(c, c, width / 2, { fill: '#cc4c4c' }),
      circle(c + ex, c - ey, width / 2, { fill: '#cc4c4c' }),
      circle(c, c, 0.05 * scale, { fill: 'fg', id: 'axle' }),
    ],
    { title: 'Pendulum' },
  )
}

const PENDULUM_DESCRIPTION = 'Swing a pendulum up and keep it upright with a limited torque.'

/**
 * Pendulum-v1 (gymnasium): swing up and balance an inverted pendulum with torque in [-2, 2]. Observations are
 * `[cos(theta), sin(theta), thetaDot]`; the reward is minus `theta^2 + 0.1 thetaDot^2 + 0.001 u^2` (angle wrapped into
 * [-pi, pi)); there is no terminal state and `make` truncates at 200 steps. Option `g` sets gravity (default 10).
 *
 * @concept environments
 * @example
 * ```ts
 * import { pendulum } from 'mdml-engine/environments'
 * import { key } from 'mdml-engine/random'
 * pendulum.create().reset(key(0)).observation.length // => 3
 * ```
 */
export const pendulum: EnvDefinition<PendulumState, readonly number[], readonly number[]> = {
  id: 'Pendulum-v1',
  description: PENDULUM_DESCRIPTION,
  options: dict({ g: real(0, 20, { default: 10, description: 'gravitational acceleration' }) }),
  create: (options): Env<PendulumState, readonly number[], readonly number[]> => {
    const g = numberOr(options, 'g', 10)
    const observe = (s: PendulumState): number[] => [Math.cos(s[0]), Math.sin(s[0]), s[1]]
    return {
      spec: { id: 'Pendulum-v1', description: PENDULUM_DESCRIPTION, maxEpisodeSteps: 200, deterministic: true },
      observationSpace: box([-1, -1, -PENDULUM_MAX_SPEED], [1, 1, PENDULUM_MAX_SPEED], [3]),
      actionSpace: box(-PENDULUM_MAX_TORQUE, PENDULUM_MAX_TORQUE, [1], { label: 'torque' }),
      reset: (k) => {
        const u = draws(k, 2, -1, 1)
        const s: PendulumState = [Math.PI * (u[0] ?? 0), u[1] ?? 0]
        return { state: s, observation: observe(s), info: {} }
      },
      step: (state, action) => {
        const r = pendulumDynamics(state, action[0] ?? 0, g)
        return {
          state: r.state,
          observation: observe(r.state),
          reward: -r.cost,
          terminated: false,
          truncated: false,
          info: {},
        }
      },
      render: renderPendulum,
      observe,
    }
  },
}

// ----------------------------------------------------------------------------------------------------------- Acrobot

/** The state of Acrobot: the two joint angles and their angular velocities. */
export type AcrobotState = readonly [angle1: number, angle2: number, angularVelocity1: number, angularVelocity2: number]

const ACRO_DT = 0.2
const MAX_VEL_1 = 4 * Math.PI
const MAX_VEL_2 = 9 * Math.PI
const AVAIL_TORQUE = [-1.0, 0.0, 1]

// gymnasium's AcrobotEnv._dsdt ("book" dynamics), in the same order of operations.
function acrobotDerivatives(s: readonly number[]): number[] {
  const m1 = 1.0
  const m2 = 1.0
  const l1 = 1.0
  const lc1 = 0.5
  const lc2 = 0.5
  const I1 = 1.0
  const I2 = 1.0
  const g = 9.8
  const a = s[4] ?? 0
  const theta1 = s[0] ?? 0
  const theta2 = s[1] ?? 0
  const dtheta1 = s[2] ?? 0
  const dtheta2 = s[3] ?? 0
  const d1 = m1 * lc1 ** 2 + m2 * (l1 ** 2 + lc2 ** 2 + 2 * l1 * lc2 * Math.cos(theta2)) + I1 + I2
  const d2 = m2 * (lc2 ** 2 + l1 * lc2 * Math.cos(theta2)) + I2
  const phi2 = m2 * lc2 * g * Math.cos(theta1 + theta2 - Math.PI / 2.0)
  const phi1 =
    -m2 * l1 * lc2 * (dtheta2 * dtheta2) * Math.sin(theta2) -
    2 * m2 * l1 * lc2 * dtheta2 * dtheta1 * Math.sin(theta2) +
    (m1 * lc1 + m2 * l1) * g * Math.cos(theta1 - Math.PI / 2) +
    phi2
  const ddtheta2 =
    (a + (d2 / d1) * phi1 - m2 * l1 * lc2 * (dtheta1 * dtheta1) * Math.sin(theta2) - phi2) /
    (m2 * lc2 ** 2 + I2 - (d2 * d2) / d1)
  const ddtheta1 = -(d2 * ddtheta2 + phi1) / d1
  return [dtheta1, dtheta2, ddtheta1, ddtheta2, 0.0]
}

const wrapAngle = (x: number, m: number, M: number): number => {
  const diff = M - m
  let y = x
  while (y > M) y -= diff
  while (y < m) y += diff
  return y
}

/**
 * One step of the Acrobot dynamics (gymnasium's AcrobotEnv.step): a classical fourth-order Runge-Kutta step of 0.2 s
 * under torque -1, 0 or +1 on the second joint, then angle wrapping and velocity bounds.
 *
 * @param state - The joint angles and velocities.
 * @param action - 0, 1 or 2 for torque -1, 0, +1.
 * @returns The next state.
 * @example
 * ```ts
 * import { acrobotDynamics } from 'mdml-engine/environments'
 * acrobotDynamics([0, 0, 0, 0], 1).map((x) => Math.abs(x) < 1e-15) // => [true, true, true, true]
 * ```
 */
export function acrobotDynamics(state: AcrobotState, action: number): AcrobotState {
  const y0 = [...state, AVAIL_TORQUE[action] ?? 0]
  const dt = ACRO_DT - 0
  const dt2 = dt / 2.0
  const k1 = acrobotDerivatives(y0)
  const k2 = acrobotDerivatives(y0.map((v, i) => v + dt2 * (k1[i] ?? 0)))
  const k3 = acrobotDerivatives(y0.map((v, i) => v + dt2 * (k2[i] ?? 0)))
  const k4 = acrobotDerivatives(y0.map((v, i) => v + dt * (k3[i] ?? 0)))
  const y = y0.map((v, i) => v + (dt / 6.0) * ((k1[i] ?? 0) + 2 * (k2[i] ?? 0) + 2 * (k3[i] ?? 0) + (k4[i] ?? 0)))
  return [
    wrapAngle(y[0] ?? 0, -Math.PI, Math.PI),
    wrapAngle(y[1] ?? 0, -Math.PI, Math.PI),
    Math.min(Math.max(y[2] ?? 0, -MAX_VEL_1), MAX_VEL_1),
    Math.min(Math.max(y[3] ?? 0, -MAX_VEL_2), MAX_VEL_2),
  ]
}

/**
 * Whether the Acrobot's free end has swung above the bar (`-cos(theta1) - cos(theta1 + theta2) > 1`).
 *
 * @param state - The state.
 * @returns Whether the episode ends.
 * @example
 * ```ts
 * import { acrobotTerminated } from 'mdml-engine/environments'
 * acrobotTerminated([Math.PI, 0, 0, 0]) // => true
 * ```
 */
export function acrobotTerminated(state: AcrobotState): boolean {
  return -Math.cos(state[0]) - Math.cos(state[1] + state[0]) > 1.0
}

function renderAcrobot(state: AcrobotState): Scene {
  const size = 500
  const bound = 2.2
  const scale = size / (bound * 2)
  const c = size / 2
  const p1: [number, number] = [-Math.cos(state[0]), Math.sin(state[0])]
  const p2: [number, number] = [p1[0] - Math.cos(state[0] + state[1]), p1[1] + Math.sin(state[0] + state[1])]
  // gymnasium's frame has y up with the pivot at the centre; positions are (y, x) pairs of the link ends.
  const toScreen = (p: readonly [number, number]): [number, number] => [c + p[1] * scale, c - p[0] * scale]
  const pivot = toScreen([0, 0])
  const a = toScreen(p1)
  const b = toScreen(p2)
  return scene(
    size,
    size,
    [
      line(c - 2.2 * scale, c - 1 * scale, c + 2.2 * scale, c - 1 * scale, { stroke: 'fg', id: 'goal' }),
      line(pivot[0], pivot[1], a[0], a[1], { stroke: '#00cccc', strokeWidth: 0.2 * scale, id: 'link1' }),
      line(a[0], a[1], b[0], b[1], { stroke: '#00cccc', strokeWidth: 0.2 * scale, id: 'link2' }),
      circle(pivot[0], pivot[1], 0.1 * scale, { fill: '#cccc00' }),
      circle(a[0], a[1], 0.1 * scale, { fill: '#cccc00' }),
    ],
    { title: 'Acrobot' },
  )
}

const ACROBOT_DESCRIPTION = 'Swing the tip of a two-link pendulum above a bar by torquing the middle joint.'

/**
 * Acrobot-v1 (gymnasium; Sutton 1996): a two-link pendulum actuated at the middle joint must swing its tip above the
 * bar. Observations are `[cos t1, sin t1, cos t2, sin t2, dt1, dt2]`; reward -1 per step and 0 on success; `make`
 * truncates at 500 steps.
 *
 * @concept environments
 * @example
 * ```ts
 * import { acrobot } from 'mdml-engine/environments'
 * import { key } from 'mdml-engine/random'
 * acrobot.create().reset(key(0)).observation.length // => 6
 * ```
 */
export const acrobot: EnvDefinition<AcrobotState, readonly number[], number> = {
  id: 'Acrobot-v1',
  description: ACROBOT_DESCRIPTION,
  create: (): Env<AcrobotState, readonly number[], number> => {
    const observe = (s: AcrobotState): number[] => [
      Math.cos(s[0]),
      Math.sin(s[0]),
      Math.cos(s[1]),
      Math.sin(s[1]),
      s[2],
      s[3],
    ]
    return {
      spec: {
        id: 'Acrobot-v1',
        description: ACROBOT_DESCRIPTION,
        maxEpisodeSteps: 500,
        rewardThreshold: -100,
        deterministic: true,
      },
      observationSpace: box([-1, -1, -1, -1, -MAX_VEL_1, -MAX_VEL_2], [1, 1, 1, 1, MAX_VEL_1, MAX_VEL_2], [6]),
      actionSpace: discrete(3, { label: 'torque -1 (0), 0 (1) or +1 (2)' }),
      reset: (k) => {
        // gymnasium draws the start state in float32.
        const s = draws(k, 4, -0.1, 0.1).map(Math.fround) as unknown as AcrobotState
        return { state: s, observation: observe(s), info: {} }
      },
      step: (state, action) => {
        const next = acrobotDynamics(state, action)
        const terminated = acrobotTerminated(next)
        return {
          state: next,
          observation: observe(next),
          reward: terminated ? 0 : -1,
          terminated,
          truncated: false,
          info: {},
        }
      },
      render: renderAcrobot,
      observe,
    }
  },
}
