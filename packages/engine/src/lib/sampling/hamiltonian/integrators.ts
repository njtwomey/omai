/**
 * The numerical integrators and geometry of Hamiltonian Monte Carlo: Gaussian Euclidean metrics (diagonal or dense mass
 * matrices) with the generalised U-turn criterion, leapfrog and other palindromic splitting integrators (as BlackJAX),
 * the Hamiltonian energy of a phase-space state, and whole trajectories for figures and teaching.
 */

import { MdmlError } from '../../base/index.ts'
import { ddot, dgemv } from '../../numerics/linalg/index.ts'
import { type Key, normal } from '../../random/index.ts'
import { asTensor, reshape, type Tensor, type TensorLike } from '../../tensor/index.ts'
import { inverse, lowerCholesky, squareMatrix, vectorOf } from '../_linear.ts'
import type { LogDensity } from '../target.ts'

/**
 * A Gaussian Euclidean metric (BlackJAX `gaussian_euclidean`): momenta p ~ N(0, M), kinetic energy p^T M^-1 p / 2,
 * velocity M^-1 p, and the generalised U-turn criterion. The inverse mass matrix is diagonal (a vector) or dense.
 *
 * @concept hamiltonian-monte-carlo
 */
export interface EuclideanMetric {
  readonly dim: number
  readonly dense: boolean
  /** The diagonal of M^-1 (length dim) or M^-1 row-major (dim * dim). */
  readonly inverseMassMatrix: Float64Array
  kineticEnergy(momentum: ArrayLike<number>): number
  velocity(momentum: ArrayLike<number>): Float64Array
  sampleMomentum(key: Key): Float64Array
  /**
   * The generalised U-turn criterion (Betancourt 2013, NumPyro's form): with rho = sum - (p_left + p_right) / 2, the
   * trajectory turns when v_left . rho <= 0 or v_right . rho <= 0.
   */
  isTurning(left: ArrayLike<number>, right: ArrayLike<number>, sum: ArrayLike<number>): boolean
}

/**
 * The Gaussian Euclidean metric with a given inverse mass matrix: a scalar or vector for a diagonal M^-1, a matrix for
 * a dense one. Momenta are drawn as p = L z with L L^T = M (dense) or p_i = z_i / sqrt(M^-1_ii) (diagonal).
 *
 * @param inverseMassMatrix - M^-1: a number, a vector of length dim, a `[dim, dim]` matrix, or that matrix flattened
 *   row-major (length dim^2, as adaptation stores it); default 1, the identity.
 * @param dim - The dimension.
 * @returns The metric.
 * @throws MdmlError with code `value` for non-positive diagonals or a matrix that is not positive definite.
 * @concept hamiltonian-monte-carlo
 * @example
 * ```ts
 * import { euclideanMetric } from 'mdml-engine/sampling'
 * const m = euclideanMetric([2, 0.5], 2)
 * m.kineticEnergy([1, 2]) // => 2
 * Array.from(m.velocity([1, 2])) // => [2, 1]
 * ```
 */
export function euclideanMetric(inverseMassMatrix: TensorLike | number = 1, dim: number): EuclideanMetric {
  const t = typeof inverseMassMatrix === 'number' ? undefined : (asTensor(inverseMassMatrix) as Tensor)
  const flatDense = t !== undefined && t.shape.length === 1 && dim > 1 && t.shape[0] === dim * dim
  if (t !== undefined && (t.shape.length === 2 || flatDense)) {
    const inv = squareMatrix(flatDense ? reshape(t, [dim, dim]) : t, dim, 'euclideanMetric')
    const massSqrt = lowerCholesky(inverse(inv, dim), dim, 'euclideanMetric')
    const velocity = (p: ArrayLike<number>): Float64Array => dgemv(inv, dim, dim, p)
    return {
      dim,
      dense: true,
      inverseMassMatrix: inv,
      velocity,
      kineticEnergy: (p) => 0.5 * ddot(velocity(p), p),
      sampleMomentum: (key) => dgemv(massSqrt, dim, dim, normal(key, [dim]).data),
      isTurning: (l, r, s) => turning(velocity(l), velocity(r), l, r, s),
    }
  }
  const diag = vectorOf(inverseMassMatrix as TensorLike | number, dim, 'euclideanMetric')
  if (!diag.every((v) => v > 0 && Number.isFinite(v))) {
    throw new MdmlError('value', 'euclideanMetric: the inverse mass matrix diagonal must be positive and finite')
  }
  const scale = diag.map((v) => 1 / Math.sqrt(v))
  const velocity = (p: ArrayLike<number>): Float64Array =>
    Float64Array.from({ length: dim }, (_, i) => (diag[i] ?? 1) * (p[i] ?? 0))
  return {
    dim,
    dense: false,
    inverseMassMatrix: diag,
    velocity,
    kineticEnergy: (p) => 0.5 * ddot(velocity(p), p),
    sampleMomentum: (key) => normal(key, [dim]).data.map((z, i) => z * (scale[i] ?? 1)),
    isTurning: (l, r, s) => turning(velocity(l), velocity(r), l, r, s),
  }
}

function turning(
  vLeft: Float64Array,
  vRight: Float64Array,
  left: ArrayLike<number>,
  right: ArrayLike<number>,
  sum: ArrayLike<number>,
): boolean {
  const rho = Float64Array.from(
    { length: vLeft.length },
    (_, i) => (sum[i] ?? 0) - ((right[i] ?? 0) + (left[i] ?? 0)) / 2,
  )
  return !(ddot(vLeft, rho) > 0) || !(ddot(vRight, rho) > 0)
}

/** A point of phase space with the log density and its gradient at the position. */
export interface IntegratorState {
  readonly position: Float64Array
  readonly momentum: Float64Array
  readonly logDensity: number
  readonly grad: Float64Array
}

/** One step of a symplectic integrator. */
export type Integrator = (state: IntegratorState, stepSize: number) => IntegratorState

/**
 * Coefficients of the palindromic splitting integrators (BlackJAX): alternating momentum and position updates,
 * starting and ending with momentum. Velocity Verlet (leapfrog) is second order with one gradient per step; McLachlan's
 * minimal-error two-stage scheme, Yoshida's and Omelyan's higher-order schemes use two, three and five.
 *
 * @concept symplectic-integrators
 * @example
 * ```ts
 * import { INTEGRATOR_COEFFICIENTS } from 'mdml-engine/sampling'
 * INTEGRATOR_COEFFICIENTS.velocityVerlet // => [0.5, 1, 0.5]
 * // The position coefficients (odd places) sum to one step.
 * INTEGRATOR_COEFFICIENTS.yoshida.filter((_, i) => i % 2 === 1).reduce((a, b) => a + b, 0) // => 1
 * ```
 */
export const INTEGRATOR_COEFFICIENTS: {
  readonly velocityVerlet: readonly number[]
  readonly mcLachlan: readonly number[]
  readonly yoshida: readonly number[]
  readonly omelyan: readonly number[]
} = (() => {
  const mb1 = 0.1931833275037836
  const yb1 = 0.11888010966548
  const ya1 = 0.29619504261126
  const ob1 = 0.08398315262876693
  const oa1 = 0.2539785108410595
  const ob2 = 0.6822365335719091
  const oa2 = -0.03230286765269967
  const ob3 = 0.5 - ob1 - ob2
  const oa3 = 1 - 2 * (oa1 + oa2)
  return {
    velocityVerlet: [0.5, 1, 0.5],
    mcLachlan: [mb1, 0.5, 1 - 2 * mb1, 0.5, mb1],
    yoshida: [yb1, ya1, 0.5 - yb1, 1 - 2 * ya1, 0.5 - yb1, ya1, yb1],
    omelyan: [ob1, oa1, ob2, oa2, ob3, oa3, ob3, oa2, ob2, oa1, ob1],
  }
})()

/** A named splitting scheme or explicit palindromic coefficients. */
export type IntegratorScheme = keyof typeof INTEGRATOR_COEFFICIENTS | readonly number[]

/**
 * A palindromic splitting integrator for H(x, p) = -log pi(x) + K(p) (BlackJAX's `generalized_two_stage_integrator`):
 * for coefficients c_0, c_1, ..., momentum updates p += eps c_i grad log pi(x) alternate with position updates
 * x += eps c_i M^-1 p, starting and ending with momentum. Each position update costs one gradient.
 *
 * @param target - The target (with a gradient).
 * @param metric - The metric.
 * @param scheme - `velocityVerlet` (default, the leapfrog), `mcLachlan`, `yoshida`, `omelyan` or coefficients of odd
 *   length.
 * @returns The integrator step.
 * @throws MdmlError with code `value` for a target without a gradient or coefficients of even length.
 * @concept symplectic-integrators
 * @reference R. I. McLachlan, On the numerical integration of ordinary differential equations by symmetric composition
 *   methods (SIAM J. Sci. Comput. 1995).
 * @example
 * ```ts
 * import { euclideanMetric, integrator, integratorState, numericTarget } from 'mdml-engine/sampling'
 * const t = numericTarget((x) => -0.5 * (x[0] ?? 0) ** 2, 1, { grad: (x) => [-(x[0] ?? 0)] })
 * const step = integrator(t, euclideanMetric(1, 1))
 * const s = step(integratorState(t, [1], [0]), 0.1)
 * Math.round((s.position[0] ?? 0) * 1e6) / 1e6 // => 0.995
 * ```
 */
export function integrator(
  target: LogDensity,
  metric: EuclideanMetric,
  scheme: IntegratorScheme = 'velocityVerlet',
): Integrator {
  if (!target.hasGradient) throw new MdmlError('value', 'integrator: the target has no gradient')
  const coefficients = typeof scheme === 'string' ? INTEGRATOR_COEFFICIENTS[scheme] : scheme
  if (coefficients.length % 2 !== 1) throw new MdmlError('value', 'integrator: need an odd number of coefficients')
  const dim = target.dim
  return (state, eps) => {
    let x = state.position
    let p = state.momentum
    let logDensity = state.logDensity
    let grad = state.grad
    const last = coefficients.length - 1
    for (let i = 0; i < last; i++) {
      const c = coefficients[i] ?? 0
      if (i % 2 === 0) {
        const g = grad
        p = p.map((v, j) => v + eps * c * (g[j] ?? 0))
      } else {
        const v = metric.velocity(p)
        x = Float64Array.from({ length: dim }, (_, j) => (x[j] ?? 0) + eps * c * (v[j] ?? 0))
        const r = target.valueAndGrad(x)
        logDensity = r.value
        grad = r.grad
      }
    }
    const c = coefficients[last] ?? 0
    const g = grad
    p = p.map((v, j) => v + eps * c * (g[j] ?? 0))
    return { position: x, momentum: p, logDensity, grad }
  }
}

/**
 * The leapfrog (velocity Verlet) integrator: a half momentum step, a full position step and a half momentum step. It
 * is symplectic and time-reversible, so Hamiltonian Monte Carlo with it leaves the target invariant.
 *
 * @param target - The target (with a gradient).
 * @param metric - The metric (default the identity).
 * @returns The integrator step.
 * @concept leapfrog-integrator
 * @example
 * ```ts
 * import { integratorState, leapfrog, numericTarget } from 'mdml-engine/sampling'
 * const t = numericTarget((x) => -0.5 * (x[0] ?? 0) ** 2, 1, { grad: (x) => [-(x[0] ?? 0)] })
 * const s = leapfrog(t)(integratorState(t, [1], [0]), 0.1)
 * Math.round((s.momentum[0] ?? 0) * 1e6) / 1e6 // => -0.09975
 * ```
 */
export function leapfrog(target: LogDensity, metric?: EuclideanMetric): Integrator {
  return integrator(target, metric ?? euclideanMetric(1, target.dim), 'velocityVerlet')
}

/**
 * A phase-space state at a position and momentum, with the log density and gradient evaluated there.
 *
 * @param target - The target (with a gradient).
 * @param position - The position.
 * @param momentum - The momentum.
 * @returns The state.
 * @throws MdmlError with code `shape` when the lengths differ from the target's dimension.
 * @example
 * ```ts
 * import { integratorState, numericTarget } from 'mdml-engine/sampling'
 * const t = numericTarget((x) => -((x[0] ?? 0) ** 2), 1, { grad: (x) => [-2 * (x[0] ?? 0)] })
 * Array.from(integratorState(t, [3], [0]).grad) // => [-6]
 * ```
 */
export function integratorState(
  target: LogDensity,
  position: ArrayLike<number>,
  momentum: ArrayLike<number>,
): IntegratorState {
  if (position.length !== target.dim || momentum.length !== target.dim) {
    throw new MdmlError('shape', `integratorState: position and momentum need ${target.dim} values`)
  }
  const x = Float64Array.from(position)
  const r = target.valueAndGrad(x)
  return { position: x, momentum: Float64Array.from(momentum), logDensity: r.value, grad: r.grad }
}

/**
 * The Hamiltonian H(x, p) = -log pi(x) + p^T M^-1 p / 2 of a phase-space state (NaN reads as +Infinity, so a broken
 * state always counts as divergent).
 *
 * @param state - The state.
 * @param metric - The metric.
 * @returns The energy.
 * @concept hamiltonian-monte-carlo
 * @example
 * ```ts
 * import { euclideanMetric, hamiltonian } from 'mdml-engine/sampling'
 * const s = { position: new Float64Array([0]), momentum: new Float64Array([2]), logDensity: -1, grad: new Float64Array([0]) }
 * hamiltonian(s, euclideanMetric(1, 1)) // => 3
 * ```
 */
export function hamiltonian(state: IntegratorState, metric: EuclideanMetric): number {
  const h = -state.logDensity + metric.kineticEnergy(state.momentum)
  return Number.isNaN(h) ? Number.POSITIVE_INFINITY : h
}

/**
 * Integrate `numSteps` steps and return every state, the trajectory an HMC proposal follows (for figures and for
 * teaching). The integration stops early, after the first state whose energy is not finite.
 *
 * @param step - The integrator.
 * @param start - The starting state.
 * @param stepSize - The step size (negative integrates backwards in time).
 * @param numSteps - The number of steps (at most 100000).
 * @returns The states, starting with `start`.
 * @throws MdmlError with code `value` for a negative, fractional or excessive number of steps.
 * @concept leapfrog-integrator
 * @example
 * ```ts
 * import { hamiltonianTrajectory, integratorState, leapfrog, numericTarget } from 'mdml-engine/sampling'
 * const t = numericTarget((x) => -0.5 * (x[0] ?? 0) ** 2, 1, { grad: (x) => [-(x[0] ?? 0)] })
 * hamiltonianTrajectory(leapfrog(t), integratorState(t, [1], [0]), 0.1, 20).length // => 21
 * ```
 */
export function hamiltonianTrajectory(
  step: Integrator,
  start: IntegratorState,
  stepSize: number,
  numSteps: number,
): IntegratorState[] {
  if (!(Number.isInteger(numSteps) && numSteps >= 0 && numSteps <= 100000)) {
    throw new MdmlError('value', 'hamiltonianTrajectory: numSteps must be an integer in [0, 100000]')
  }
  const out = [start]
  let s = start
  for (let i = 0; i < numSteps; i++) {
    s = step(s, stepSize)
    out.push(s)
    if (!Number.isFinite(s.logDensity)) break
  }
  return out
}
