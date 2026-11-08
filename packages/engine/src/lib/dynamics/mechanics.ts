/**
 * Classical mechanics from the Lagrangian: the Euler-Lagrange equations derived by automatic differentiation from a
 * function L(q, qdot) (so the equations of motion are computed, not typed in), the energy function, the Legendre
 * transform to the Hamiltonian and Hamilton's equations as a vector field. The double pendulum is the worked example:
 * its Lagrangian, the closed-form accelerations the derivation must agree with, its Hamiltonian and energy, and the
 * time to the first flip that colours the flip map.
 */

import { grad, hessian } from '../autodiff/index.ts'
import { OmaiError } from '../base/index.ts'
import { solve } from '../numerics/linalg/index.ts'
import { add, concat, cos, div, item, mul, sin, sliceAxis, sub, sum, tensor, type Value } from '../tensor/index.ts'
import type { OdeVectorField } from './ode/problem.ts'

/** A Lagrangian L(q, qdot): the kinetic minus the potential energy, a scalar of the coordinates and their velocities. */
export type Lagrangian = (q: Value, qdot: Value) => Value

const part = (x: Value, from: number, to: number): Value => sliceAxis(x, 0, from, to)
const block = (m: Value, r0: number, r1: number, c0: number, c1: number): Value =>
  sliceAxis(sliceAxis(m, 0, r0, r1), 1, c0, c1)

/**
 * The Euler-Lagrange equations of a Lagrangian as a first-order field on `[q, qdot]`. With the Lagrangian a function of
 * q and qdot, d/dt (dL/dqdot) = dL/dq expands to (d2L/dqdot2) qddot = dL/dq - (d2L/dqdot dq) qdot, and both second
 * derivatives come from automatic differentiation: the mass matrix is solved for the accelerations, whatever the
 * Lagrangian.
 *
 * @param L - The Lagrangian.
 * @param dof - The number of coordinates.
 * @returns The field `(t, [q, qdot]) => [qdot, qddot]`.
 * @concept lagrangian-mechanics
 * @example
 * ```ts
 * import { lagrangianField } from 'omai-engine/dynamics'
 * import { mul, sub, sum, tensor, toFlat } from 'omai-engine/tensor'
 * // L = qdot^2 / 2 - q^2 / 2: the oscillator qddot = -q.
 * const f = lagrangianField((q, qd) => mul(sum(sub(mul(qd, qd), mul(q, q))), 0.5), 1)
 * toFlat(f(0, tensor([2, 3]))) // => [3, -2]
 * ```
 */
export function lagrangianField(L: Lagrangian, dof: number): OdeVectorField {
  const joint = (x: Value): Value => L(part(x, 0, dof), part(x, dof, 2 * dof))
  const g = grad(joint)
  const h = hessian(joint)
  return (_t, y) => {
    const qdot = part(y, dof, 2 * dof)
    const dLdq = part(g(y), 0, dof)
    const hx = h(y)
    const mass = block(hx, dof, 2 * dof, dof, 2 * dof)
    const coupling = block(hx, dof, 2 * dof, 0, dof)
    const rhs = sub(dLdq, sum(mul(coupling, qdot), 1))
    return concat([qdot, solve(mass, rhs)])
  }
}

/**
 * The energy function of a Lagrangian, E = qdot . dL/dqdot - L: the quantity the flow conserves when L does not depend
 * on time (the kinetic plus the potential energy for the usual Lagrangians).
 *
 * @param L - The Lagrangian.
 * @param dof - The number of coordinates.
 * @returns The function of the state `[q, qdot]` giving E.
 * @concept lagrangian-mechanics
 * @example
 * ```ts
 * import { lagrangianEnergy } from 'omai-engine/dynamics'
 * import { mul, sub, sum } from 'omai-engine/tensor'
 * const E = lagrangianEnergy((q, qd) => mul(sum(sub(mul(qd, qd), mul(q, q))), 0.5), 1)
 * E([2, 3]) // => 6.5
 * ```
 */
export function lagrangianEnergy(L: Lagrangian, dof: number): (state: ArrayLike<number>) => number {
  const joint = (x: Value): Value => L(part(x, 0, dof), part(x, dof, 2 * dof))
  const g = grad(joint)
  return (state) => {
    const y = tensor(Array.from(state))
    return item(sub(sum(mul(part(g(y), dof, 2 * dof), part(y, dof, 2 * dof))), joint(y)))
  }
}

/**
 * The Hamiltonian of a Lagrangian that is quadratic in the velocities, L = qdot.M(q) qdot / 2 + b(q).qdot - V(q), by
 * the Legendre transform: p = M qdot + b, so H(q, p) = (p - b).M^{-1}(p - b) / 2 + V. M, b and V are read from L by
 * automatic differentiation (M is the Hessian in the velocities, b the gradient there at zero velocity, V = -L at zero
 * velocity).
 *
 * @param L - The Lagrangian (quadratic in qdot).
 * @returns H(q, p) as a differentiable function.
 * @concept legendre-transform
 * @example
 * ```ts
 * import { hamiltonianOfLagrangian } from 'omai-engine/dynamics'
 * import { item, mul, sub, sum, tensor } from 'omai-engine/tensor'
 * const H = hamiltonianOfLagrangian((q, qd) => mul(sum(sub(mul(qd, qd), mul(q, q))), 0.5))
 * item(H(tensor([2]), tensor([3]))) // => 6.5
 * ```
 */
export function hamiltonianOfLagrangian(L: Lagrangian): (q: Value, p: Value) => Value {
  return (q, p) => {
    const atVelocity = (qd: Value): Value => L(q, qd)
    const zero = mul(p, 0)
    const mass = hessian(atVelocity)(zero)
    const b = grad(atVelocity)(zero)
    const v = mul(atVelocity(zero), -1)
    const rel = sub(p, b)
    return add(mul(sum(mul(rel, solve(mass, rel))), 0.5), v)
  }
}

/**
 * Hamilton's equations of a Hamiltonian H(q, p) as a field on `[q, p]`: dq/dt = dH/dp, dp/dt = -dH/dq. It is the form
 * for a non-separable Hamiltonian (the double pendulum), where no splitting method applies and a symplectic integrator
 * such as the implicit midpoint rule is the one that keeps the energy.
 *
 * @param H - The Hamiltonian, a scalar of (q, p).
 * @param dof - The number of coordinates.
 * @returns The field.
 * @concept hamiltonian-mechanics
 * @example
 * ```ts
 * import { hamiltonianField } from 'omai-engine/dynamics'
 * import { add, mul, sum, tensor, toFlat } from 'omai-engine/tensor'
 * const f = hamiltonianField((q, p) => mul(sum(add(mul(q, q), mul(p, p))), 0.5), 1)
 * toFlat(f(0, tensor([2, 3]))) // => [3, -2]
 * ```
 */
export function hamiltonianField(H: (q: Value, p: Value) => Value, dof: number): OdeVectorField {
  const joint = (x: Value): Value => H(part(x, 0, dof), part(x, dof, 2 * dof))
  const g = grad(joint)
  return (_t, y) => {
    const gx = g(y)
    return concat([part(gx, dof, 2 * dof), mul(part(gx, 0, dof), -1)])
  }
}

/** The parameters of the double pendulum: the two masses, the two rod lengths and gravity. */
export interface DoublePendulumParameters {
  /** The inner bob's mass (default 1). */
  readonly m1?: number
  /** The outer bob's mass (default 1). */
  readonly m2?: number
  /** The inner rod (default 1). */
  readonly l1?: number
  /** The outer rod (default 1). */
  readonly l2?: number
  /** Gravity (default 9.81). */
  readonly g?: number
}

function resolved(p: DoublePendulumParameters): Required<DoublePendulumParameters> {
  const { m1 = 1, m2 = 1, l1 = 1, l2 = 1, g = 9.81 } = p
  if (!(m1 > 0 && m2 > 0 && l1 > 0 && l2 > 0))
    throw new OmaiError('value', 'doublePendulum: masses and lengths must be positive')
  return { m1, m2, l1, l2, g }
}

/**
 * The Lagrangian of the double pendulum in the angles from the downward vertical, L = T - V with
 * T = (m1 + m2) l1^2 w1^2 / 2 + m2 l2^2 w2^2 / 2 + m2 l1 l2 w1 w2 cos(t1 - t2) and
 * V = -(m1 + m2) g l1 cos t1 - m2 g l2 cos t2. Feed it to {@link lagrangianField} to derive the equations of motion.
 *
 * @param parameters - Masses, lengths and gravity.
 * @returns The Lagrangian on `q = [theta1, theta2]`.
 * @concept lagrangian-mechanics
 * @example
 * ```ts
 * import { doublePendulumLagrangian } from 'omai-engine/dynamics'
 * import { item, tensor } from 'omai-engine/tensor'
 * // Hanging at rest the Lagrangian is minus the (lowest) potential energy: g (m1 l1 + ... ) at the bottom.
 * item(doublePendulumLagrangian()(tensor([0, 0]), tensor([0, 0]))) // => 29.43
 * ```
 */
export function doublePendulumLagrangian(parameters: DoublePendulumParameters = {}): Lagrangian {
  const { m1, m2, l1, l2, g } = resolved(parameters)
  return (q, qd) => {
    const t1 = part(q, 0, 1)
    const t2 = part(q, 1, 2)
    const w1 = part(qd, 0, 1)
    const w2 = part(qd, 1, 2)
    const kinetic = add(
      add(mul(mul(w1, w1), 0.5 * (m1 + m2) * l1 * l1), mul(mul(w2, w2), 0.5 * m2 * l2 * l2)),
      mul(mul(mul(w1, w2), cos(sub(t1, t2))), m2 * l1 * l2),
    )
    const potential = mul(add(mul(cos(t1), (m1 + m2) * g * l1), mul(cos(t2), m2 * g * l2)), -1)
    return sum(sub(kinetic, potential))
  }
}

/**
 * The double pendulum's equations of motion on `(theta1, theta2, omega1, omega2)` in closed form, the result of the
 * derivation that {@link lagrangianField} performs on {@link doublePendulumLagrangian}. A chaotic system for large
 * swings: nearby starts part exponentially fast.
 *
 * @param parameters - Masses, lengths and gravity.
 * @returns The vector field.
 * @concept double-pendulum
 * @example
 * ```ts
 * import { doublePendulumField } from 'omai-engine/dynamics'
 * import { tensor, toFlat } from 'omai-engine/tensor'
 * // Hanging at rest is an equilibrium.
 * toFlat(doublePendulumField()(0, tensor([0, 0, 0, 0]))).map((v) => v + 0) // => [0, 0, 0, 0]
 * ```
 */
export function doublePendulumField(parameters: DoublePendulumParameters = {}): OdeVectorField {
  const { m1, m2, l1, l2, g } = resolved(parameters)
  return (_t, y) => {
    const t1 = part(y, 0, 1)
    const t2 = part(y, 1, 2)
    const w1 = part(y, 2, 3)
    const w2 = part(y, 3, 4)
    const delta = sub(t1, t2)
    const c = cos(delta)
    const s = sin(delta)
    const common = sub(2 * m1 + m2, mul(cos(mul(delta, 2)), m2))
    const a1 = div(
      sub(
        sub(mul(sin(t1), -g * (2 * m1 + m2)), mul(sin(sub(t1, mul(t2, 2))), m2 * g)),
        mul(mul(s, add(mul(mul(w2, w2), l2), mul(mul(mul(w1, w1), l1), c))), 2 * m2),
      ),
      mul(common, l1),
    )
    const a2 = div(
      mul(
        mul(
          s,
          add(
            add(mul(mul(w1, w1), l1 * (m1 + m2)), mul(cos(t1), g * (m1 + m2))),
            mul(mul(mul(w2, w2), l2), mul(c, m2)),
          ),
        ),
        2,
      ),
      mul(common, l2),
    )
    return concat([w1, w2, a1, a2])
  }
}

/**
 * The energy of the double pendulum, T + V, from `(theta1, theta2, omega1, omega2)`: conserved by the flow.
 *
 * @param state - The state.
 * @param parameters - Masses, lengths and gravity.
 * @returns The total energy.
 * @concept double-pendulum
 * @example
 * ```ts
 * import { doublePendulumEnergy } from 'omai-engine/dynamics'
 * // At rest hanging down the energy is the lowest potential energy, -(2 m1 + m2) g l... for unit masses and lengths -3 g.
 * doublePendulumEnergy([0, 0, 0, 0]) // => -29.43
 * ```
 */
export function doublePendulumEnergy(state: ArrayLike<number>, parameters: DoublePendulumParameters = {}): number {
  const { m1, m2, l1, l2, g } = resolved(parameters)
  const [t1, t2, w1, w2] = [state[0] ?? 0, state[1] ?? 0, state[2] ?? 0, state[3] ?? 0]
  const kinetic =
    0.5 * (m1 + m2) * l1 * l1 * w1 * w1 + 0.5 * m2 * l2 * l2 * w2 * w2 + m2 * l1 * l2 * w1 * w2 * Math.cos(t1 - t2)
  return kinetic - (m1 + m2) * g * l1 * Math.cos(t1) - m2 * g * l2 * Math.cos(t2)
}

/**
 * The double pendulum as a Hamiltonian system on `[theta1, theta2, p1, p2]` with its conjugate momenta
 * p = dL/dqdot: the field of Hamilton's equations for the Hamiltonian that the Legendre transform of the Lagrangian
 * gives. Not separable, so the symplectic splittings do not apply; the implicit midpoint rule is symplectic for it.
 *
 * @param parameters - Masses, lengths and gravity.
 * @returns The Hamiltonian `H(q, p)` and its field.
 * @concept double-pendulum
 * @example
 * ```ts
 * import { doublePendulumHamiltonian } from 'omai-engine/dynamics'
 * import { item, tensor } from 'omai-engine/tensor'
 * // The Hamiltonian is the energy: at rest hanging down, -3 g for unit masses and lengths.
 * item(doublePendulumHamiltonian().H(tensor([0, 0]), tensor([0, 0]))) // => -29.43
 * ```
 */
export function doublePendulumHamiltonian(parameters: DoublePendulumParameters = {}): {
  readonly H: (q: Value, p: Value) => Value
  readonly field: OdeVectorField
} {
  const H = hamiltonianOfLagrangian(doublePendulumLagrangian(parameters))
  return { H, field: hamiltonianField(H, 2) }
}

/** Options of {@link doublePendulumFlipTime}. */
export interface FlipOptions {
  /** Give up after this time (default 10). */
  readonly tMax?: number
  /** The Runge-Kutta step (default 1/200). */
  readonly dt?: number
}

/**
 * The time until the double pendulum first flips: until either rod passes over the top (its angle reaches pi in
 * magnitude), starting at rest from the given angles. The flip time over a grid of initial angles is the flip map, whose
 * boundary between quick flips and long waits is fractal. It integrates the closed-form equations with classical
 * fourth-order Runge-Kutta on plain numbers, linear interpolation locating the crossing within a step.
 *
 * @param theta1 - Inner angle from the downward vertical.
 * @param theta2 - Outer angle.
 * @param parameters - Masses, lengths and gravity.
 * @param options - The time limit and the step.
 * @returns The flip time, or `Infinity` when it does not flip before `tMax`.
 * @concept double-pendulum
 * @example
 * ```ts
 * import { doublePendulumFlipTime } from 'omai-engine/dynamics'
 * // Released at rest from small angles the pendulum never flips.
 * doublePendulumFlipTime(0.1, 0.1, {}, { tMax: 5 }) // => Infinity
 * ```
 */
export function doublePendulumFlipTime(
  theta1: number,
  theta2: number,
  parameters: DoublePendulumParameters = {},
  options: FlipOptions = {},
): number {
  const { m1, m2, l1, l2, g } = resolved(parameters)
  const tMax = options.tMax ?? 10
  const dt = options.dt ?? 1 / 200
  const f = (y: readonly number[]): number[] => {
    const [t1, t2, w1, w2] = y as [number, number, number, number]
    const d = t1 - t2
    const common = 2 * m1 + m2 - m2 * Math.cos(2 * d)
    const a1 =
      (-g * (2 * m1 + m2) * Math.sin(t1) -
        m2 * g * Math.sin(t1 - 2 * t2) -
        2 * Math.sin(d) * m2 * (w2 * w2 * l2 + w1 * w1 * l1 * Math.cos(d))) /
      (l1 * common)
    const a2 =
      (2 * Math.sin(d) * (w1 * w1 * l1 * (m1 + m2) + g * (m1 + m2) * Math.cos(t1) + w2 * w2 * l2 * m2 * Math.cos(d))) /
      (l2 * common)
    return [w1, w2, a1, a2]
  }
  const axpy = (y: readonly number[], k: readonly number[], h: number): number[] => y.map((v, i) => v + h * (k[i] ?? 0))
  let y = [theta1, theta2, 0, 0]
  const flipped = (v: readonly number[]): boolean => Math.abs(v[0] ?? 0) >= Math.PI || Math.abs(v[1] ?? 0) >= Math.PI
  if (flipped(y)) return 0
  for (let t = 0; t < tMax; t += dt) {
    const k1 = f(y)
    const k2 = f(axpy(y, k1, dt / 2))
    const k3 = f(axpy(y, k2, dt / 2))
    const k4 = f(axpy(y, k3, dt))
    const next = y.map((v, i) => v + (dt / 6) * ((k1[i] ?? 0) + 2 * (k2[i] ?? 0) + 2 * (k3[i] ?? 0) + (k4[i] ?? 0)))
    if (flipped(next)) {
      // The angle that crossed pi first, and where in the step.
      const share = [0, 1]
        .map((i) => {
          const a = Math.abs(y[i] ?? 0)
          const b = Math.abs(next[i] ?? 0)
          return b >= Math.PI && b > a ? (Math.PI - a) / (b - a) : 1
        })
        .reduce((m, s) => Math.min(m, s), 1)
      return t + share * dt
    }
    y = next
  }
  return Number.POSITIVE_INFINITY
}
