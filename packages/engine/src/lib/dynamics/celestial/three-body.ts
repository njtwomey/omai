/**
 * The circular restricted three-body problem: a massless body moving in the field of two masses (1 - mu and mu) that
 * circle their barycentre, written in the frame that rotates with them. The field with its Coriolis terms, the
 * pseudo-potential and the Jacobi constant that the motion conserves, the five Lagrange points (the collinear three
 * found by root-finding, the triangular two exact) and their stability (Routh's critical mass ratio), the zero-velocity
 * curves that bound the region a given Jacobi constant allows, and a stepped integration that logs the constant.
 */

import { MdmlError } from '../../base/index.ts'
import { contourLines } from '../../numerics/geometry/index.ts'
import { brentq } from '../../numerics/roots/index.ts'
import { add, mul, pow, sliceAxis, stack, sub, sum, type Value } from '../../tensor/index.ts'
import { type Algorithm, type Contract, defineContract, scalar, tensor } from '../../trace/index.ts'
import { classifyFixedPoint, type FixedPointType, fieldJacobian } from '../fields.ts'
import type { ButcherTableau } from '../ode/explicit.ts'
import { type FlowState, flowAlgorithm } from '../ode/flow.ts'
import type { VectorField } from '../ode/problem.ts'

/** The mass ratio beyond which the triangular Lagrange points L4 and L5 are unstable, (1 - sqrt(23 / 27)) / 2. */
export const ROUTH_CRITICAL_MASS_RATIO: number = (1 - Math.sqrt(23 / 27)) / 2

const at = (y: Value, i: number): Value => sum(sliceAxis(y, 0, i, i + 1))

function checkMu(mu: number): void {
  if (!(mu > 0 && mu <= 0.5)) throw new MdmlError('value', `the mass ratio mu must be in (0, 1/2], got ${mu}`)
}

/**
 * The vector field of the circular restricted three-body problem in the rotating frame, with the masses 1 - mu at
 * (-mu, 0, 0) and mu at (1 - mu, 0, 0), distances in the primaries' separation, time such that they turn once in 2 pi.
 * The state is `[x, y, vx, vy]` (planar) or `[x, y, z, vx, vy, vz]` (spatial):
 * x'' = 2 y' + dOmega/dx, y'' = -2 x' + dOmega/dy, z'' = dOmega/dz.
 *
 * @param mu - The mass ratio of the smaller primary, in (0, 1/2].
 * @param dim - 2 (planar, default) or 3.
 * @returns The field.
 * @throws MdmlError with code `value` for mu outside (0, 1/2].
 * @concept restricted-three-body-problem
 * @example
 * ```ts
 * import { restrictedThreeBodyField } from 'mdml-engine/dynamics/celestial'
 * import { tensor, toFlat } from 'mdml-engine/tensor'
 * // Far away at rest the acceleration is the centrifugal x plus the pull of the total mass, about 1000 - 1e-6.
 * Math.round(toFlat(restrictedThreeBodyField(0.01)(0, tensor([1000, 0, 0, 0])))[2] ?? 0) // => 1000
 * ```
 */
export function restrictedThreeBodyField(mu: number, dim: 2 | 3 = 2): VectorField {
  checkMu(mu)
  return (_t, s) => {
    const x = at(s, 0)
    const y = at(s, 1)
    const z = dim === 3 ? at(s, 2) : undefined
    const vx = at(s, dim)
    const vy = at(s, dim + 1)
    const dx1 = add(x, mu)
    const dx2 = sub(x, 1 - mu)
    const out = (a: Value): Value => (z === undefined ? add(mul(y, y), a) : add(add(mul(y, y), mul(z, z)), a))
    // (1 - mu) / r1^3 and mu / r2^3.
    const k1 = mul(pow(out(mul(dx1, dx1)), -1.5), 1 - mu)
    const k2 = mul(pow(out(mul(dx2, dx2)), -1.5), mu)
    const ax = sub(sub(add(mul(vy, 2), x), mul(dx1, k1)), mul(dx2, k2))
    const ay = sub(sub(sub(y, mul(vx, 2)), mul(y, k1)), mul(y, k2))
    if (z === undefined) return stack([vx, vy, ax, ay])
    const az = mul(z, mul(add(k1, k2), -1))
    return stack([vx, vy, at(s, 5), ax, ay, az])
  }
}

/**
 * The pseudo-potential Omega = (x^2 + y^2) / 2 + (1 - mu) / r1 + mu / r2: half the centrifugal energy plus the
 * gravitational pull of both primaries. The accelerations are its gradient and the Jacobi constant is 2 Omega - v^2.
 *
 * @param mu - The mass ratio.
 * @param x - The x coordinate in the rotating frame.
 * @param y - The y coordinate.
 * @param z - The z coordinate (default 0).
 * @returns Omega at the point.
 * @concept restricted-three-body-problem
 * @example
 * ```ts
 * import { pseudoPotential } from 'mdml-engine/dynamics/celestial'
 * // At L4 of the equal-mass pair, (0, sqrt(3) / 2), both primaries are a unit away: Omega = 3 / 8 + 1.
 * Math.round(pseudoPotential(0.5, 0, Math.sqrt(3) / 2) * 1e12) / 1e12 // => 1.375
 * ```
 */
export function pseudoPotential(mu: number, x: number, y: number, z = 0): number {
  const r1 = Math.hypot(x + mu, y, z)
  const r2 = Math.hypot(x - 1 + mu, y, z)
  return 0.5 * (x * x + y * y) + (1 - mu) / r1 + mu / r2
}

/**
 * The effective potential -Omega: the potential energy of a unit mass at rest in the rotating frame, centrifugal term
 * included. Its maxima are L4 and L5 (stable only because of the Coriolis force), its saddles L1, L2 and L3, and the
 * zero-velocity curve of a Jacobi constant C is its level set -C / 2.
 *
 * @param mu - The mass ratio.
 * @param x - The x coordinate.
 * @param y - The y coordinate.
 * @param z - The z coordinate (default 0).
 * @returns -Omega at the point.
 * @concept restricted-three-body-problem
 * @example
 * ```ts
 * import { effectivePotential } from 'mdml-engine/dynamics/celestial'
 * effectivePotential(0.01, 0, 1) < 0 // => true
 * ```
 */
export function effectivePotential(mu: number, x: number, y: number, z = 0): number {
  return -pseudoPotential(mu, x, y, z)
}

/**
 * The Jacobi constant C = 2 Omega - |v|^2 of a state: the one integral of the restricted problem. A trajectory cannot
 * leave the region where 2 Omega >= C.
 *
 * @param mu - The mass ratio.
 * @param state - `[x, y, vx, vy]` or `[x, y, z, vx, vy, vz]`.
 * @returns C.
 * @throws MdmlError with code `shape` for a state of another length.
 * @concept restricted-three-body-problem
 * @example
 * ```ts
 * import { jacobiConstant } from 'mdml-engine/dynamics/celestial'
 * // At rest at L4 of the equal-mass pair Omega = 3/8 + 1, so C = 2 Omega = 2.75 (in general 3 - mu (1 - mu)).
 * Math.round(jacobiConstant(0.5, [0, Math.sqrt(3) / 2, 0, 0]) * 1e12) / 1e12 // => 2.75
 * ```
 */
export function jacobiConstant(mu: number, state: ArrayLike<number>): number {
  if (state.length !== 4 && state.length !== 6)
    throw new MdmlError('shape', 'jacobiConstant: the state has 4 or 6 entries')
  const d = state.length / 2
  const x = state[0] ?? 0
  const y = state[1] ?? 0
  const z = d === 3 ? (state[2] ?? 0) : 0
  let v2 = 0
  for (let k = d; k < 2 * d; k++) v2 += (state[k] ?? 0) ** 2
  return 2 * pseudoPotential(mu, x, y, z) - v2
}

/** The five equilibria of the rotating frame, in the plane. */
export interface LagrangePoints {
  readonly L1: readonly [number, number]
  readonly L2: readonly [number, number]
  readonly L3: readonly [number, number]
  readonly L4: readonly [number, number]
  readonly L5: readonly [number, number]
}

/**
 * The Lagrange points of the restricted problem: the collinear L1 (between the primaries), L2 (beyond the smaller) and
 * L3 (beyond the larger) as roots of dOmega/dx on the axis, found by Brent's method, and the triangular L4 and L5 at
 * (1/2 - mu, +-sqrt(3)/2), which make equilateral triangles with the primaries and are exact.
 *
 * @param mu - The mass ratio, in (0, 1/2].
 * @returns The five points.
 * @throws MdmlError with code `value` for mu outside (0, 1/2].
 * @concept lagrange-points
 * @example
 * ```ts
 * import { lagrangePoints } from 'mdml-engine/dynamics/celestial'
 * // Sun-Earth-like ratio: L1 sits about 1% of the separation inside the Earth.
 * Math.round(lagrangePoints(3.0035e-6).L1[0] * 1e4) / 1e4 // => 0.99
 * ```
 */
export function lagrangePoints(mu: number): LagrangePoints {
  checkMu(mu)
  const dOmega = (x: number): number =>
    x - ((1 - mu) * (x + mu)) / Math.abs(x + mu) ** 3 - (mu * (x - 1 + mu)) / Math.abs(x - 1 + mu) ** 3
  const root = (a: number, b: number): number => brentq(dOmega, a, b, { xtol: 1e-15, rtol: 4 * Number.EPSILON }).root
  const tiny = 1e-9
  return {
    L1: [root(-mu + tiny, 1 - mu - tiny), 0],
    L2: [root(1 - mu + tiny, 2), 0],
    L3: [root(-2, -mu - tiny), 0],
    L4: [0.5 - mu, Math.sqrt(3) / 2],
    L5: [0.5 - mu, -Math.sqrt(3) / 2],
  }
}

/** A Lagrange point with its position, linear stability and the eigenvalues of the linearisation. */
export interface LagrangeClassification {
  readonly name: 'L1' | 'L2' | 'L3' | 'L4' | 'L5'
  readonly point: readonly [number, number]
  readonly type: FixedPointType
  /** Whether the linearisation has no growing mode (a centre: only L4 and L5, below Routh's mass ratio). */
  readonly stable: boolean
  readonly eigenvalues: { readonly re: readonly number[]; readonly im: readonly number[] }
}

/**
 * The Lagrange points classified by the eigenvalues of the field's Jacobian there: L1, L2 and L3 are saddles (with a
 * centre direction) for every mass ratio; L4 and L5 are centres (linearly stable, the Coriolis force holding them)
 * when mu is below {@link ROUTH_CRITICAL_MASS_RATIO}, and unstable spirals beyond.
 *
 * @param mu - The mass ratio.
 * @returns The five points with their types.
 * @concept lagrange-points
 * @example
 * ```ts
 * import { classifyLagrangePoints } from 'mdml-engine/dynamics/celestial'
 * classifyLagrangePoints(0.01).map((p) => p.type) // => ['saddle', 'saddle', 'saddle', 'center', 'center']
 * ```
 */
export function classifyLagrangePoints(mu: number): LagrangeClassification[] {
  const field = restrictedThreeBodyField(mu)
  const points = lagrangePoints(mu)
  return (['L1', 'L2', 'L3', 'L4', 'L5'] as const).map((name) => {
    const [x, y] = points[name]
    const jac = fieldJacobian(field, [x, y, 0, 0])
    const c = classifyFixedPoint(jac, 1e-7)
    return {
      name,
      point: points[name],
      type: c.type,
      stable: c.stable || c.type === 'center',
      eigenvalues: c.eigenvalues,
    }
  })
}

/** Options of {@link zeroVelocityCurves}. */
export interface ZeroVelocityOptions {
  /** The window `[xmin, xmax, ymin, ymax]` (default `[-1.6, 1.6, -1.6, 1.6]`). */
  readonly window?: readonly [number, number, number, number]
  /** Grid points per side (default 400). */
  readonly n?: number
}

/**
 * The zero-velocity curves of a Jacobi constant: the level set 2 Omega(x, y) = C in the plane, the boundary of the
 * region where the body can be (it has speed zero there). Where 2 Omega < C the body can never go: the forbidden
 * region, which opens at the Lagrange points as C falls below their values.
 *
 * @param mu - The mass ratio.
 * @param C - The Jacobi constant.
 * @param options - The window and the grid.
 * @returns The curves, each a path of `[x, y]` points (closed ones repeat their first point).
 * @concept restricted-three-body-problem
 * @example
 * ```ts
 * import { zeroVelocityCurves } from 'mdml-engine/dynamics/celestial'
 * // A large constant keeps the body near one primary or far out: several separate curves.
 * zeroVelocityCurves(0.1, 4, { n: 200 }).length > 1 // => true
 * ```
 */
export function zeroVelocityCurves(mu: number, C: number, options: ZeroVelocityOptions = {}): [number, number][][] {
  checkMu(mu)
  const [x0, x1, y0, y1] = options.window ?? [-1.6, 1.6, -1.6, 1.6]
  const n = options.n ?? 400
  const xs = Array.from({ length: n }, (_, i) => x0 + ((x1 - x0) * i) / (n - 1))
  const ys = Array.from({ length: n }, (_, i) => y0 + ((y1 - y0) * i) / (n - 1))
  const grid = ys.map((y) => xs.map((x) => 2 * pseudoPotential(mu, x, y) - C))
  return contourLines(grid, 0, { x: xs, y: ys })
}

/** The paths of {@link restrictedThreeBodyAlgorithm}. */
export type ThreeBodyPath = 'mass/t' | 'mass/position' | 'mass/velocity' | 'mass/jacobi' | 'mass/jacobiDrift'

/** What a restricted three-body run logs. */
export const threeBodyContract: Contract<FlowState, ThreeBodyPath> = defineContract({
  'mass/t': scalar({ description: 'time' }),
  'mass/position': tensor({ role: 'state', shape: ['dim'], description: 'the test mass in the rotating frame' }),
  'mass/velocity': tensor({ role: 'state', shape: ['dim'], description: 'its velocity in the rotating frame' }),
  'mass/jacobi': scalar({ description: 'the Jacobi constant 2 Omega - v^2, conserved by the exact flow' }),
  'mass/jacobiDrift': scalar({ description: 'the Jacobi constant minus its starting value', direction: 'lower' }),
})

/** Options of {@link restrictedThreeBodyAlgorithm}. */
export interface ThreeBodyRunOptions {
  /** The step. */
  readonly dt: number
  /** The steps to take. */
  readonly steps: number
  /** Planar (default) or spatial. */
  readonly dim?: 2 | 3
  /** The Runge-Kutta method (default RK4). */
  readonly method?: ButcherTableau
}

/**
 * A test mass in the restricted three-body problem as a stepped algorithm (fixed-step Runge-Kutta, so a state that
 * serialises), logging time, position, velocity and the Jacobi constant with its drift from the start: the constant is
 * what a good integration conserves, and its drift measures the integrator.
 *
 * @param mu - The mass ratio.
 * @param state0 - `[x, y, vx, vy]` (or six entries when `dim` is 3).
 * @param options - Step, step count, dimension and method.
 * @returns The algorithm.
 * @concept restricted-three-body-problem
 * @example
 * ```ts
 * import { lagrangePoints, restrictedThreeBodyAlgorithm } from 'mdml-engine/dynamics/celestial'
 * import { runAlgorithm } from 'mdml-engine/trace'
 * // At rest at L4 the body stays there.
 * const [x, y] = lagrangePoints(0.01).L4
 * const alg = restrictedThreeBodyAlgorithm(0.01, [x, y, 0, 0], { dt: 0.01, steps: 100 })
 * Math.abs((runAlgorithm(alg, { maxSteps: 100 }).state.y[0] ?? 0) - x) < 1e-9 // => true
 * ```
 */
export function restrictedThreeBodyAlgorithm(
  mu: number,
  state0: readonly number[],
  options: ThreeBodyRunOptions,
): Algorithm<FlowState, ThreeBodyPath> {
  const dim = options.dim ?? 2
  const d = dim
  const field = restrictedThreeBodyField(mu, dim)
  const c0 = jacobiConstant(mu, state0)
  return flowAlgorithm<ThreeBodyPath>(field, state0, {
    dt: options.dt,
    steps: options.steps,
    name: 'restrictedThreeBody',
    config: { mu, dim, state0: [...state0] },
    ...(options.method === undefined ? {} : { method: options.method }),
    observe: (t, y, ctx) => {
      const c = jacobiConstant(mu, y)
      ctx.log('mass/t', t)
      ctx.log('mass/position', y.slice(0, d))
      ctx.log('mass/velocity', y.slice(d, 2 * d))
      ctx.log('mass/jacobi', c)
      ctx.log('mass/jacobiDrift', c - c0)
    },
  })
}
