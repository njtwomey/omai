import { uniform } from '../../../lib/random/index.ts'
import type { Env, EnvDefinition } from '../../../lib/rl/index.ts'
import { circle, line, polygon, rect, type Scene, scene } from '../../../lib/scene/index.ts'
import { box, discrete } from '../../../lib/space/index.ts'

/** The state of CartPole: cart position and velocity, pole angle (radians from upright) and angular velocity. */
export type CartPoleState = readonly [number, number, number, number]

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

/** One Euler step of the cart-pole dynamics under a push left (0) or right (1), exactly as gymnasium computes it. */
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

/** Whether a CartPole state is terminal (the cart left the track or the pole fell past 12 degrees). */
export function cartPoleTerminated(state: CartPoleState): boolean {
  const [x, , theta] = state
  return x < -X_LIMIT || x > X_LIMIT || theta < -THETA_LIMIT || theta > THETA_LIMIT
}

/** Draw a CartPole state: track, cart, pole and axle (gymnasium's 600 x 400 layout). */
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

/**
 * CartPole-v1 (gymnasium): keep a pole balanced on a cart by pushing it left or right. Reward 1 per step, including
 * the terminating step; episodes end when the pole passes 12 degrees or the cart leaves [-2.4, 2.4], and `make`
 * truncates them at 500 steps. Observations are the float64 state (gymnasium returns float32 copies).
 *
 * @concept environments
 * @example
 * ```ts
 * import { cartPole } from 'whitebox-engine/methods/rl'
 * import { key } from 'whitebox-engine/random'
 * const env = cartPole.create()
 * env.step(env.reset(key(0)).state, 1, key(1)).reward // => 1
 * ```
 */
export const cartPole: EnvDefinition<CartPoleState, readonly number[], number> = {
  id: 'CartPole-v1',
  description: 'Balance a pole on a cart by pushing the cart left or right.',
  create: (): Env<CartPoleState, readonly number[], number> => ({
    spec: {
      id: 'CartPole-v1',
      description: 'Balance a pole on a cart by pushing the cart left or right.',
      maxEpisodeSteps: 500,
      rewardThreshold: 475,
    },
    observationSpace: box(
      [-X_LIMIT * 2, -Number.MAX_VALUE, -THETA_LIMIT * 2, -Number.MAX_VALUE],
      [X_LIMIT * 2, Number.MAX_VALUE, THETA_LIMIT * 2, Number.MAX_VALUE],
      [4],
    ),
    actionSpace: discrete(2, { label: 'push left (0) or right (1)' }),
    reset: (k) => {
      const s = Array.from(uniform(k, [4], { low: -0.05, high: 0.05 }).data) as unknown as CartPoleState
      return { state: s, observation: s, info: {} }
    },
    step: (state, action) => {
      const next = cartPoleDynamics(state, action)
      const terminated = cartPoleTerminated(next)
      return { state: next, observation: next, reward: 1, terminated, truncated: false, info: {} }
    },
    render: renderCartPole,
  }),
}
