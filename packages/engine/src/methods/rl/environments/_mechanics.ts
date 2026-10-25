/**
 * Equations of motion and rewards of the continuous-control systems, written once over a scalar algebra so the
 * environments (plain numbers, continuous-control.ts) and their differentiable models (tensor values, models.ts)
 * evaluate the same operations in the same order: the number and tensor algebras, a Runge-Kutta step and exact
 * damped-mass discretisations. Private to methods/rl/environments.
 */

import {
  add,
  cos,
  div,
  exp,
  less,
  mul,
  neg,
  sin,
  sqrt,
  sub,
  type TensorLike,
  type Value,
  where,
} from '../../../lib/tensor/index.ts'

/** The scalar operations the equations use: plain numbers, or tensor values (elementwise, so batches broadcast). */
export interface ScalarOps<T> {
  add(a: T | number, b: T | number): T
  sub(a: T | number, b: T | number): T
  mul(a: T | number, b: T | number): T
  div(a: T | number, b: T | number): T
  neg(a: T | number): T
  sin(a: T | number): T
  cos(a: T | number): T
  sqrt(a: T | number): T
  exp(a: T | number): T
  /** `a < b ? x : y`. */
  ifLess(a: T | number, b: T | number, x: T | number, y: T | number): T
}

/** Plain-number arithmetic. */
export const numberOps: ScalarOps<number> = {
  add: (a, b) => a + b,
  sub: (a, b) => a - b,
  mul: (a, b) => a * b,
  div: (a, b) => a / b,
  neg: (a) => -a,
  sin: Math.sin,
  cos: Math.cos,
  sqrt: Math.sqrt,
  exp: Math.exp,
  ifLess: (a, b, x, y) => (a < b ? x : y),
}

const v = (x: TensorLike): Value => x as Value

/** Differentiable tensor arithmetic (lib/tensor ops; elementwise with broadcasting). */
export const valueOps: ScalarOps<Value> = {
  add: (a, b) => add(a, b),
  sub: (a, b) => sub(a, b),
  mul: (a, b) => mul(a, b),
  div: (a, b) => div(a, b),
  neg: (a) => neg(a),
  sin: (a) => sin(a),
  cos: (a) => cos(a),
  sqrt: (a) => sqrt(a),
  exp: (a) => exp(a),
  ifLess: (a, b, x, y) => where(less(a, b), v(x), v(y)),
}

/** A sum of terms, left to right. */
function total<T>(o: ScalarOps<T>, terms: readonly (T | number)[]): T {
  let s: T | number = terms[0] ?? 0
  for (let i = 1; i < terms.length; i++) s = o.add(s, terms[i] ?? 0)
  return o.add(s, 0)
}

/**
 * A classical fourth-order Runge-Kutta step of `ds/dt = f(s)` (the input held constant over the step), repeated over
 * `substeps` equal substeps.
 */
export function rk4<T>(o: ScalarOps<T>, f: (s: readonly T[]) => T[], s: readonly T[], dt: number, substeps = 1): T[] {
  const h = dt / substeps
  let y: T[] = [...s]
  for (let n = 0; n < substeps; n++) {
    const k1 = f(y)
    const y2 = y.map((x, i) => o.add(x, o.mul(k1[i] as T, h / 2)))
    const k2 = f(y2)
    const y3 = y.map((x, i) => o.add(x, o.mul(k2[i] as T, h / 2)))
    const k3 = f(y3)
    const y4 = y.map((x, i) => o.add(x, o.mul(k3[i] as T, h)))
    const k4 = f(y4)
    y = y.map((x, i) => {
      const slope = total(o, [k1[i] as T, o.mul(k2[i] as T, 2), o.mul(k3[i] as T, 2), k4[i] as T])
      return o.add(x, o.mul(slope, h / 6))
    })
  }
  return y
}

// ------------------------------------------------------------------------------------------------- damped point mass

/** Exact zero-order-hold coefficients of `m x'' = F - b x'` over `dt`: `v' = a v + beta F`, `x' = x + c v + gamma F`. */
export interface DampedMassCoefficients {
  readonly a: number
  readonly c: number
  readonly beta: number
  readonly gamma: number
}

/** The exact discretisation of a damped mass (the undamped limit is the double integrator). */
export function dampedMassCoefficients(mass: number, damping: number, dt: number): DampedMassCoefficients {
  if (damping === 0) return { a: 1, c: dt, beta: dt / mass, gamma: (dt * dt) / (2 * mass) }
  const a = Math.exp((-damping * dt) / mass)
  const c = (mass / damping) * (1 - a)
  return { a, c, beta: (1 - a) / damping, gamma: (dt - c) / damping }
}

/** One exact step of independent damped masses, one per axis: positions then velocities, forces per axis. */
export function dampedMassStep<T>(
  o: ScalarOps<T>,
  k: DampedMassCoefficients,
  positions: readonly (T | number)[],
  velocities: readonly (T | number)[],
  forces: readonly (T | number)[],
): T[] {
  const x = positions.map((p, i) => o.add(o.add(p, o.mul(velocities[i] ?? 0, k.c)), o.mul(forces[i] ?? 0, k.gamma)))
  const vel = velocities.map((q, i) => o.add(o.mul(q, k.a), o.mul(forces[i] ?? 0, k.beta)))
  return [...x, ...vel]
}

/**
 * DeepMind control suite's `rewards.tolerance` with the Gaussian and quadratic sigmoids: 1 inside `[lower, upper]`,
 * falling to `valueAtMargin` at `margin` outside it.
 */
export function tolerance<T>(
  o: ScalarOps<T>,
  x: T | number,
  lower: number,
  upper: number,
  margin: number,
  sigmoid: 'gaussian' | 'quadratic',
  valueAtMargin: number,
): T {
  if (margin === 0) return o.ifLess(x, lower, 0, o.ifLess(upper, x, 0, 1))
  const d = o.ifLess(x, lower, o.div(o.sub(lower, x), margin), o.ifLess(upper, x, o.div(o.sub(x, upper), margin), 0))
  if (sigmoid === 'gaussian') {
    const scale = Math.sqrt(-2 * Math.log(valueAtMargin))
    const z = o.mul(d, scale)
    return o.exp(o.mul(o.mul(z, z), -0.5))
  }
  const z = o.mul(d, Math.sqrt(1 - valueAtMargin))
  return o.ifLess(z, 1, o.sub(1, o.mul(z, z)), 0)
}

/** DeepMind control suite's point_mass reward: near the target, times a small bonus for small actions. */
export function pointMassRewardOf<T>(
  o: ScalarOps<T>,
  s: readonly (T | number)[],
  u: readonly (T | number)[],
  goal: readonly number[],
  targetSize: number,
  margin: number,
): T {
  const dx = o.sub(s[0] ?? 0, goal[0] ?? 0)
  const dy = o.sub(s[1] ?? 0, goal[1] ?? 0)
  const dist = o.sqrt(o.add(o.mul(dx, dx), o.mul(dy, dy)))
  const near = tolerance(o, dist, 0, targetSize, margin, 'gaussian', 0.1)
  const c0 = tolerance(o, u[0] ?? 0, 0, 0, 1, 'quadratic', 0)
  const c1 = tolerance(o, u[1] ?? 0, 0, 0, 1, 'quadratic', 0)
  const small = o.div(o.add(o.div(o.add(c0, c1), 2), 4), 5)
  return o.mul(near, small)
}

// -------------------------------------------------------------------------------------------------- 2-D reacher

/** Physical parameters of the two-link planar arm. */
export interface ArmParameters {
  readonly lengths: readonly [number, number]
  readonly masses: readonly [number, number]
  /** Rotor inertia added at each joint. */
  readonly armature: number
  /** Viscous joint damping. */
  readonly damping: number
  /** Torque per unit action. */
  readonly gain: number
}

function armTerms<T>(o: ScalarOps<T>, q2: T | number, p: ArmParameters): { m11: T; m12: T; m22: number; h: T } {
  const [l1, l2] = p.lengths
  const [m1, m2] = p.masses
  const lc1 = l1 / 2
  const lc2 = l2 / 2
  const i1 = (m1 * l1 * l1) / 12
  const i2 = (m2 * l2 * l2) / 12
  const c2 = o.cos(q2)
  const m11 = o.add(m1 * lc1 * lc1 + i1 + m2 * (l1 * l1 + lc2 * lc2) + i2 + p.armature, o.mul(c2, 2 * m2 * l1 * lc2))
  const m12 = o.add(m2 * lc2 * lc2 + i2, o.mul(c2, m2 * l1 * lc2))
  const m22 = m2 * lc2 * lc2 + i2 + p.armature
  return { m11, m12, m22, h: o.mul(o.sin(q2), m2 * l1 * lc2) }
}

/** `d/dt [q1, q2, qd1, qd2]` of the arm in a horizontal plane: `M(q) qdd + c(q, qd) + b qd = gain u`. */
export function armDerivatives<T>(
  o: ScalarOps<T>,
  s: readonly (T | number)[],
  u: readonly (T | number)[],
  p: ArmParameters,
): T[] {
  const [, q2 = 0, qd1 = 0, qd2 = 0] = s
  const { m11, m12, m22, h } = armTerms(o, q2, p)
  const tau1 = total(o, [
    o.mul(u[0] ?? 0, p.gain),
    o.mul(qd1, -p.damping),
    o.mul(h, o.add(o.mul(o.mul(qd1, qd2), 2), o.mul(qd2, qd2))),
  ])
  const tau2 = total(o, [o.mul(u[1] ?? 0, p.gain), o.mul(qd2, -p.damping), o.neg(o.mul(h, o.mul(qd1, qd1)))])
  const det = o.sub(o.mul(m11, m22), o.mul(m12, m12))
  const qdd1 = o.div(o.sub(o.mul(tau1, m22), o.mul(m12, tau2)), det)
  const qdd2 = o.div(o.sub(o.mul(m11, tau2), o.mul(m12, tau1)), det)
  return [o.add(qd1, 0), o.add(qd2, 0), qdd1, qdd2]
}

/** The kinetic energy `qd^T M(q) qd / 2` of the arm. */
export function armEnergy<T>(o: ScalarOps<T>, s: readonly (T | number)[], p: ArmParameters): T {
  const [, q2 = 0, qd1 = 0, qd2 = 0] = s
  const { m11, m12, m22 } = armTerms(o, q2, p)
  const quad = total(o, [
    o.mul(m11, o.mul(qd1, qd1)),
    o.mul(o.mul(m12, 2), o.mul(qd1, qd2)),
    o.mul(o.mul(qd2, qd2), m22),
  ])
  return o.mul(quad, 0.5)
}

/** The fingertip of the arm: `l1 (cos q1, sin q1) + l2 (cos(q1 + q2), sin(q1 + q2))`. */
export function armTip<T>(o: ScalarOps<T>, q1: T | number, q2: T | number, p: ArmParameters): [T, T] {
  const [l1, l2] = p.lengths
  const q12 = o.add(q1, q2)
  return [o.add(o.mul(o.cos(q1), l1), o.mul(o.cos(q12), l2)), o.add(o.mul(o.sin(q1), l1), o.mul(o.sin(q12), l2))]
}

/** gymnasium Reacher's reward: minus the fingertip distance to the target, minus a weighted squared action. */
export function reacherRewardOf<T>(
  o: ScalarOps<T>,
  s: readonly (T | number)[],
  u: readonly (T | number)[],
  target: readonly number[],
  p: ArmParameters,
  controlWeight: number,
): T {
  const [tx, ty] = armTip(o, s[0] ?? 0, s[1] ?? 0, p)
  const dx = o.sub(tx, target[0] ?? 0)
  const dy = o.sub(ty, target[1] ?? 0)
  const dist = o.sqrt(o.add(o.mul(dx, dx), o.mul(dy, dy)))
  const u0 = u[0] ?? 0
  const u1 = u[1] ?? 0
  return o.neg(o.add(dist, o.mul(o.add(o.mul(u0, u0), o.mul(u1, u1)), controlWeight)))
}

// ----------------------------------------------------------------------------------------------- planar quadrotor

/** Physical parameters of the planar quadrotor. */
export interface QuadrotorParameters {
  readonly mass: number
  /** Moment of inertia about the axis out of the plane. */
  readonly inertia: number
  /** Distance from the centre to each rotor. */
  readonly arm: number
  readonly g: number
}

/** `d/dt [x, z, theta, xd, zd, thetad]` under left and right rotor thrusts `u`. */
export function quadrotorDerivatives<T>(
  o: ScalarOps<T>,
  s: readonly (T | number)[],
  u: readonly (T | number)[],
  p: QuadrotorParameters,
): T[] {
  const [, , th = 0, xd = 0, zd = 0, thd = 0] = s
  const thrust = o.add(u[0] ?? 0, u[1] ?? 0)
  return [
    o.add(xd, 0),
    o.add(zd, 0),
    o.add(thd, 0),
    o.div(o.neg(o.mul(thrust, o.sin(th))), p.mass),
    o.sub(o.div(o.mul(thrust, o.cos(th)), p.mass), p.g),
    o.div(o.mul(o.sub(u[1] ?? 0, u[0] ?? 0), p.arm), p.inertia),
  ]
}

/** The quadrotor's reward: minus a quadratic cost of the distance to the goal, tilt, speeds and thrust off hover. */
export function quadrotorRewardOf<T>(
  o: ScalarOps<T>,
  s: readonly (T | number)[],
  u: readonly (T | number)[],
  goal: readonly number[],
  p: QuadrotorParameters,
  controlWeight: number,
): T {
  const sq = (x: T | number): T => o.mul(x, x)
  const hover = (p.mass * p.g) / 2
  const [x = 0, z = 0, th = 0, xd = 0, zd = 0, thd = 0] = s
  const cost = total(o, [
    sq(o.sub(x, goal[0] ?? 0)),
    sq(o.sub(z, goal[1] ?? 0)),
    o.mul(sq(th), 0.1),
    o.mul(total(o, [sq(xd), sq(zd), sq(thd)]), 0.01),
    o.mul(o.add(sq(o.sub(u[0] ?? 0, hover)), sq(o.sub(u[1] ?? 0, hover))), controlWeight),
  ])
  return o.neg(cost)
}

// ------------------------------------------------------------------------------------------- cart double pendulum

/** Physical parameters of the cart with two linked uniform poles. */
export interface CartDoublePendulumParameters {
  readonly cartMass: number
  readonly masses: readonly [number, number]
  readonly lengths: readonly [number, number]
  readonly g: number
  /** Viscous friction of the cart. */
  readonly cartDamping: number
  /** Viscous friction of both joints (on their relative angular velocities). */
  readonly jointDamping: number
}

interface PendulumTerms<T> {
  readonly d11: number
  readonly d12: T
  readonly d13: T
  readonly d22: number
  readonly d23: T
  readonly d33: number
}

function pendulumTerms<T>(o: ScalarOps<T>, th1: T | number, th2: T | number, p: CartDoublePendulumParameters) {
  const [m1, m2] = p.masses
  const [l1, l2] = p.lengths
  const lc1 = l1 / 2
  const lc2 = l2 / 2
  const a1 = m1 * lc1 + m2 * l1
  const terms: PendulumTerms<T> = {
    d11: p.cartMass + m1 + m2,
    d12: o.mul(o.cos(th1), a1),
    d13: o.mul(o.cos(th2), m2 * lc2),
    d22: m1 * lc1 * lc1 + m2 * l1 * l1 + (m1 * l1 * l1) / 12,
    d23: o.mul(o.cos(o.sub(th1, th2)), m2 * l1 * lc2),
    d33: m2 * lc2 * lc2 + (m2 * l2 * l2) / 12,
  }
  return { terms, a1, b: m2 * l1 * lc2, c: m2 * lc2 }
}

/**
 * `d/dt [x, theta1, theta2, xd, thetad1, thetad2]` of a cart with two linked uniform poles (absolute angles from
 * upright) under a horizontal force: the Lagrangian equations `D(q) qdd + h(q, qd) + G(q) = tau`, solved by Cramer's
 * rule.
 */
export function cartDoublePendulumDerivativesOf<T>(
  o: ScalarOps<T>,
  s: readonly (T | number)[],
  force: T | number,
  p: CartDoublePendulumParameters,
): T[] {
  const [, th1 = 0, th2 = 0, xd = 0, w1 = 0, w2 = 0] = s
  const { terms: d, a1, b, c } = pendulumTerms(o, th1, th2, p)
  const s1 = o.sin(th1)
  const s2 = o.sin(th2)
  const s12 = o.sin(o.sub(th1, th2))
  const rel = o.sub(w2, w1)
  // Right-hand sides tau - h - G.
  const r1 = total(o, [
    force,
    o.mul(xd, -p.cartDamping),
    o.mul(o.mul(s1, o.mul(w1, w1)), a1),
    o.mul(o.mul(s2, o.mul(w2, w2)), c),
  ])
  const r2 = total(o, [
    o.mul(w1, -p.jointDamping),
    o.mul(rel, p.jointDamping),
    o.neg(o.mul(o.mul(s12, o.mul(w2, w2)), b)),
    o.mul(s1, a1 * p.g),
  ])
  const r3 = total(o, [o.mul(rel, -p.jointDamping), o.mul(o.mul(s12, o.mul(w1, w1)), b), o.mul(s2, c * p.g)])
  // Cramer's rule for the symmetric system [d11 d12 d13; d12 d22 d23; d13 d23 d33] qdd = r.
  const det3 = (
    a11: T | number,
    a12: T | number,
    a13: T | number,
    a21: T | number,
    a22: T | number,
    a23: T | number,
    a31: T | number,
    a32: T | number,
    a33: T | number,
  ): T =>
    total(o, [
      o.mul(a11, o.sub(o.mul(a22, a33), o.mul(a23, a32))),
      o.neg(o.mul(a12, o.sub(o.mul(a21, a33), o.mul(a23, a31)))),
      o.mul(a13, o.sub(o.mul(a21, a32), o.mul(a22, a31))),
    ])
  const det = det3(d.d11, d.d12, d.d13, d.d12, d.d22, d.d23, d.d13, d.d23, d.d33)
  const xdd = o.div(det3(r1, d.d12, d.d13, r2, d.d22, d.d23, r3, d.d23, d.d33), det)
  const thdd1 = o.div(det3(d.d11, r1, d.d13, d.d12, r2, d.d23, d.d13, r3, d.d33), det)
  const thdd2 = o.div(det3(d.d11, d.d12, r1, d.d12, d.d22, r2, d.d13, d.d23, r3), det)
  return [o.add(xd, 0), o.add(w1, 0), o.add(w2, 0), xdd, thdd1, thdd2]
}

/** The total mechanical energy `qd^T D qd / 2 + V` of the cart double pendulum. */
export function cartDoublePendulumEnergyOf<T>(
  o: ScalarOps<T>,
  s: readonly (T | number)[],
  p: CartDoublePendulumParameters,
): T {
  const [, th1 = 0, th2 = 0, xd = 0, w1 = 0, w2 = 0] = s
  const { terms: d, a1, c } = pendulumTerms(o, th1, th2, p)
  const kinetic = total(o, [
    o.mul(o.mul(xd, xd), d.d11),
    o.mul(o.mul(d.d12, 2), o.mul(xd, w1)),
    o.mul(o.mul(d.d13, 2), o.mul(xd, w2)),
    o.mul(o.mul(w1, w1), d.d22),
    o.mul(o.mul(d.d23, 2), o.mul(w1, w2)),
    o.mul(o.mul(w2, w2), d.d33),
  ])
  const potential = o.mul(o.add(o.mul(o.cos(th1), a1), o.mul(o.cos(th2), c)), p.g)
  return o.add(o.mul(kinetic, 0.5), potential)
}

/** The tip of the second pole: `(x + l1 sin theta1 + l2 sin theta2, l1 cos theta1 + l2 cos theta2)`. */
export function cartDoublePendulumTipOf<T>(
  o: ScalarOps<T>,
  s: readonly (T | number)[],
  p: CartDoublePendulumParameters,
): [T, T] {
  const [x = 0, th1 = 0, th2 = 0] = s
  const [l1, l2] = p.lengths
  return [
    total(o, [x, o.mul(o.sin(th1), l1), o.mul(o.sin(th2), l2)]),
    o.add(o.mul(o.cos(th1), l1), o.mul(o.cos(th2), l2)),
  ]
}

/** gymnasium InvertedDoublePendulum's reward: an alive bonus minus tip distance and angular velocity penalties. */
export function cartDoublePendulumRewardOf<T>(
  o: ScalarOps<T>,
  s: readonly (T | number)[],
  p: CartDoublePendulumParameters,
): T {
  const [tx, ty] = cartDoublePendulumTipOf(o, s, p)
  const top = o.sub(ty, p.lengths[0] + p.lengths[1])
  const w1 = s[4] ?? 0
  const w2 = s[5] ?? 0
  return total(o, [
    10,
    o.mul(o.mul(tx, tx), -0.01),
    o.neg(o.mul(top, top)),
    o.mul(o.mul(w1, w1), -1e-3),
    o.mul(o.mul(w2, w2), -5e-3),
  ])
}
