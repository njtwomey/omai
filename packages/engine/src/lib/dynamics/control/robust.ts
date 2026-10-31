/**
 * H-infinity synthesis by the two Riccati equations (Doyle, Glover, Khargonekar and Francis): for a generalised plant with
 * disturbance input w, control u, performance output z and measurement y, the central controller that makes the closed-loop
 * gain from w to z less than gamma, when one exists. The state-feedback case (one Riccati equation), the output-feedback
 * controller built from both, the three conditions that say whether gamma is achievable, the closed loop assembled and its
 * H-infinity norm checked, and the search for the smallest achievable gamma as a stepped bisection.
 */

import { type Json, MdmlError } from '../../base/index.ts'
import { eigvals, solve, solveCare } from '../../numerics/linalg/index.ts'
import { hinfNorm, type StateSpace, ss } from '../../systems/index.ts'
import {
  add,
  asTensor,
  concat,
  eye,
  item,
  matmul,
  mul,
  neg,
  sub,
  type TensorLike,
  toFlat,
  transpose,
  type Value,
  zeros,
} from '../../tensor/index.ts'
import { type Algorithm, type Contract, defineContract, scalar, stateFromJson, stateToJson } from '../../trace/index.ts'

/** The part of a generalised plant that state feedback uses: the dynamics and the performance output. */
export interface StateFeedbackPlant {
  readonly a: TensorLike
  /** The disturbance input `[n, nw]`. */
  readonly b1: TensorLike
  /** The control input `[n, nu]`. */
  readonly b2: TensorLike
  /** The performance output `[nz, n]`. */
  readonly c1: TensorLike
  /** The control's share of the performance output `[nz, nu]` (default: zero on the state rows, identity below). */
  readonly d12?: TensorLike
}

/**
 * A generalised plant x' = A x + B1 w + B2 u, z = C1 x + D12 u, y = C2 x + D21 w, in the normalised form the central
 * solution assumes: D12'D12 = I and D12'C1 = 0 (the performance output weights the state and the control separately, the
 * control with unit weight), D21 D21' = I and B1 D21' = 0 (the measurement noise is separate from the disturbance and has
 * unit weight), no direct feedthrough from w to z or u to y. (A, B2) must be stabilisable and (C2, A) detectable.
 */
export interface GeneralisedPlant extends StateFeedbackPlant {
  /** The measurement `[ny, n]`. */
  readonly c2: TensorLike
  /** The noise's share of the measurement `[ny, nw]` (default: zero on the disturbance columns, identity after). */
  readonly d21?: TensorLike
}

const T = (x: Value): Value => transpose(x)
const sym = (x: Value): Value => mul(add(x, T(x)), 0.5)

interface Parts {
  readonly A: Value
  readonly B1: Value
  readonly B2: Value
  readonly C1: Value
  readonly C2: Value
  readonly D12: Value
  readonly D21: Value
  readonly n: number
  readonly nw: number
  readonly nu: number
  readonly nz: number
  readonly ny: number
}

function partsOf(plant: StateFeedbackPlant | GeneralisedPlant, measured = true): Parts {
  const A = asTensor(plant.a)
  const B1 = asTensor(plant.b1)
  const B2 = asTensor(plant.b2)
  const C1 = asTensor(plant.c1)
  const C2 = measured ? asTensor((plant as GeneralisedPlant).c2) : zeros([1, A.shape[0] ?? 0])
  const n = A.shape[0] ?? 0
  const nw = B1.shape[1] ?? 0
  const nu = B2.shape[1] ?? 0
  const nz = C1.shape[0] ?? 0
  const ny = C2.shape[0] ?? 0
  const D12 = plant.d12 === undefined ? concat([zeros([nz - nu, nu]), eye(nu)], 0) : asTensor(plant.d12)
  const d21 = measured ? (plant as GeneralisedPlant).d21 : undefined
  const D21 = d21 === undefined ? concat([zeros([ny, nw - ny]), eye(ny)], 1) : asTensor(d21)
  const near = (m: Value, target: Value, what: string): void => {
    const d = toFlat(sub(m, target))
    if (d.some((v) => Math.abs(v) > 1e-9)) throw new MdmlError('value', `hinfSynthesis: ${what}`)
  }
  near(matmul(T(D12), D12), eye(nu), "D12 must satisfy D12' D12 = I")
  near(matmul(T(D12), C1), zeros([nu, n]), "D12' C1 must be zero")
  if (measured) {
    near(matmul(D21, T(D21)), eye(ny), "D21 must satisfy D21 D21' = I")
    near(matmul(B1, T(D21)), zeros([n, ny]), "B1 D21' must be zero")
  }
  return { A, B1, B2, C1, C2, D12, D21, n, nw, nu, nz, ny }
}

/** The stabilising solution of the H-infinity control Riccati equation, X = Riccati(A, B2, B1, C1'C1, gamma). */
function riccati(a: Value, bu: Value, bw: Value, q: Value, gamma: number): Value {
  const nu = bu.shape[1] ?? 0
  const nw = bw.shape[1] ?? 0
  const b = concat([bu, bw], 1)
  const r = concat(
    [concat([eye(nu), zeros([nu, nw])], 1), concat([zeros([nw, nu]), mul(eye(nw), -gamma * gamma)], 1)],
    0,
  )
  return sym(solveCare(a, b, q, r))
}

/** The result of H-infinity state feedback. */
export interface HinfStateFeedback {
  /** The gain, u = K x. */
  readonly K: Value
  /** The Riccati solution X. */
  readonly X: Value
  readonly gamma: number
}

/**
 * H-infinity state feedback at a level gamma: the stabilising solution X of A'X + XA + X(B1B1'/gamma^2 - B2B2')X + C1'C1 = 0
 * gives u = -B2'X x, under which the gain from w to z is below gamma (when X exists and is positive semidefinite; as gamma
 * grows the equation tends to the LQR one and K to the LQR gain).
 *
 * @param plant - The generalised plant (only A, B1, B2, C1 and D12 are used).
 * @param gamma - The level.
 * @returns The gain and X.
 * @throws MdmlError with code `value` when the plant violates the normalisation, or `convergence` when no stabilising solution exists at this gamma.
 * @concept h-infinity-control
 * @example
 * ```ts
 * import { hinfStateFeedback } from 'mdml-engine/dynamics/control'
 * import { item } from 'mdml-engine/tensor'
 * // x' = x + w + u, z = (x, u): at gamma = 100 the gain is close to the LQR one, -(1 + sqrt 2).
 * const plant = { a: [[1]], b1: [[1]], b2: [[1]], c1: [[1], [0]], d12: [[0], [1]] }
 * Math.abs(item(hinfStateFeedback(plant, 100).K) + 1 + Math.SQRT2) < 1e-3 // => true
 * ```
 */
export function hinfStateFeedback(plant: StateFeedbackPlant, gamma: number): HinfStateFeedback {
  const p = partsOf(plant, false)
  const X = riccati(p.A, p.B2, p.B1, matmul(T(p.C1), p.C1), gamma)
  return { K: neg(matmul(T(p.B2), X)), X, gamma }
}

/** Why a level gamma is or is not achievable, and the controller if it is. */
export interface HinfSolution {
  readonly gamma: number
  readonly feasible: boolean
  /** The three conditions: both Riccati solutions exist and are positive semidefinite, and rho(XY) < gamma^2. */
  readonly conditions: {
    readonly xExists: boolean
    readonly yExists: boolean
    readonly xPositive: boolean
    readonly yPositive: boolean
    /** The spectral radius of XY over gamma^2 (feasible when below 1). */
    readonly coupling: number
  }
  /** The central controller K(s) from y to u (when feasible). */
  readonly controller?: StateSpace
  readonly X?: Value
  readonly Y?: Value
}

/**
 * H-infinity output-feedback synthesis at a level gamma: the central controller of the two-Riccati solution. X solves the
 * state-feedback equation, Y its dual (A Y + Y A' + Y(C1'C1/gamma^2 - C2'C2)Y + B1B1' = 0); gamma is achievable exactly when
 * both exist and are positive semidefinite and the spectral radius of XY is below gamma^2. The controller is
 * x_c' = (A + B1B1'X/gamma^2 + B2F + Z L C2) x_c - Z L y, u = F x_c, with F = -B2'X, L = -Y C2' and Z = (I - YX/gamma^2)^-1.
 * When gamma is not achievable the result says which condition failed and has no controller.
 *
 * @param plant - The generalised plant.
 * @param gamma - The level.
 * @returns The conditions, and the controller when they hold.
 * @throws MdmlError with code `value` when the plant violates the normalisation.
 * @concept h-infinity-control
 * @example
 * ```ts
 * import { hinfSynthesis } from 'mdml-engine/dynamics/control'
 * const plant = { a: [[1]], b1: [[1, 0]], b2: [[1]], c1: [[1], [0]], c2: [[1]], d12: [[0], [1]], d21: [[0, 1]] }
 * [hinfSynthesis(plant, 50).feasible, hinfSynthesis(plant, 0.5).feasible] // => [true, false]
 * ```
 */
export function hinfSynthesis(plant: GeneralisedPlant, gamma: number): HinfSolution {
  const p = partsOf(plant)
  const failed = (conditions: Partial<HinfSolution['conditions']>): HinfSolution => ({
    gamma,
    feasible: false,
    conditions: {
      xExists: false,
      yExists: false,
      xPositive: false,
      yPositive: false,
      coupling: Number.POSITIVE_INFINITY,
      ...conditions,
    },
  })
  let X: Value
  try {
    X = riccati(p.A, p.B2, p.B1, matmul(T(p.C1), p.C1), gamma)
  } catch (e) {
    if (e instanceof MdmlError && (e.code === 'convergence' || e.code === 'value')) return failed({})
    throw e
  }
  let Y: Value
  try {
    Y = riccati(T(p.A), T(p.C2), T(p.C1), matmul(p.B1, T(p.B1)), gamma)
  } catch (e) {
    if (e instanceof MdmlError && (e.code === 'convergence' || e.code === 'value')) return failed({ xExists: true })
    throw e
  }
  const psd = (m: Value): boolean => {
    const ev = eigvals(m).re
    return toFlat(ev).every((v) => v >= -1e-9 * Math.max(1, ...toFlat(ev).map(Math.abs)))
  }
  const xPositive = psd(X)
  const yPositive = psd(Y)
  const coupling = Math.max(...toFlat(eigvals(matmul(X, Y)).re)) / (gamma * gamma)
  const conditions = { xExists: true, yExists: true, xPositive, yPositive, coupling }
  if (!(xPositive && yPositive && coupling < 1)) return { gamma, feasible: false, conditions, X, Y }
  const g2 = 1 / (gamma * gamma)
  const F = neg(matmul(T(p.B2), X))
  const L = neg(matmul(Y, T(p.C2)))
  const Z = solve(sub(eye(p.n), mul(matmul(Y, X), g2)), eye(p.n))
  const ZL = matmul(Z, L)
  const Ac = sub(
    sub(sub(p.A, mul(matmul(matmul(p.B1, T(p.B1)), X), -g2)), mul(matmul(p.B2, F), -1)),
    mul(matmul(ZL, p.C2), -1),
  )
  const controller = ss(Ac, neg(ZL), F, zeros([p.nu, p.ny]))
  return { gamma, feasible: true, conditions, controller, X, Y }
}

/**
 * The closed loop from the disturbance w to the performance output z when a controller (from y to u) is wired to the plant.
 *
 * @param plant - The generalised plant.
 * @param controller - A strictly proper controller K(s) from y to u.
 * @returns The closed-loop state space, states `[x; x_c]`.
 * @concept h-infinity-control
 * @example
 * ```ts
 * import { closedLoopFromDisturbance, hinfSynthesis } from 'mdml-engine/dynamics/control'
 * import { hinfNorm } from 'mdml-engine/systems'
 * import { item } from 'mdml-engine/tensor'
 * const plant = { a: [[1]], b1: [[1, 0]], b2: [[1]], c1: [[1], [0]], c2: [[1]], d12: [[0], [1]], d21: [[0, 1]] }
 * const k = hinfSynthesis(plant, 5).controller
 * k === undefined ? false : item(hinfNorm(closedLoopFromDisturbance(plant, k)).norm) < 5 // => true
 * ```
 */
export function closedLoopFromDisturbance(plant: GeneralisedPlant, controller: StateSpace): StateSpace {
  const p = partsOf(plant)
  const Bc = controller.b
  const Cc = controller.c
  const Ac = controller.a
  const a = concat([concat([p.A, matmul(p.B2, Cc)], 1), concat([matmul(Bc, p.C2), Ac], 1)], 0)
  const b = concat([p.B1, matmul(Bc, p.D21)], 0)
  const c = concat([p.C1, matmul(p.D12, Cc)], 1)
  return ss(a, b, c, zeros([p.nz, p.nw]))
}

/** The state of {@link hinfBisectionAlgorithm}. */
export interface HinfSearchState {
  /** The smallest level known to be achievable, and the largest known not to be. */
  readonly feasible: number
  readonly infeasible: number
  /** The level tried last and whether it was achievable. */
  readonly tried: number
  readonly succeeded: boolean
  /** rho(XY) / gamma^2 at the level tried (infinite when a Riccati solution failed). */
  readonly coupling: number
  /** The H-infinity norm of the closed loop under the controller at the best feasible level so far. */
  readonly achieved: number
  readonly iteration: number
}

/** The paths of {@link hinfBisectionAlgorithm}. */
export type HinfSearchPath = 'hinf/gamma' | 'hinf/feasible' | 'hinf/coupling' | 'hinf/bracket' | 'hinf/achieved'

/** What a gamma search logs. */
export const hinfSearchContract: Contract<HinfSearchState, HinfSearchPath> = defineContract({
  'hinf/gamma': scalar({ description: 'the level tried' }),
  'hinf/feasible': scalar({
    description: '1 if the level is achievable (both Riccati solutions, rho(XY) < gamma^2), else 0',
  }),
  'hinf/coupling': scalar({ description: 'rho(XY) / gamma^2: below 1 for an achievable level' }),
  'hinf/bracket': scalar({ description: 'the width of the bracket on the optimal level', direction: 'lower' }),
  'hinf/achieved': scalar({
    description: 'the closed-loop H-infinity norm of the best controller so far',
    direction: 'lower',
  }),
})

/** Options of {@link hinfBisectionAlgorithm}. */
export interface HinfSearchOptions {
  /** A level known to be achievable (default 1000: a large level is the LQG-like problem). */
  readonly upper?: number
  /** A level known not to be achievable (default 1e-3). */
  readonly lower?: number
}

/**
 * The search for the smallest achievable gamma as a stepped bisection (on the logarithm of gamma): each step tries the
 * geometric mean of the bracket, keeps the controller's closed-loop norm when it succeeds, and shrinks the bracket either
 * way. The central controller at a level just above the optimum has a closed-loop norm below that level.
 *
 * @param plant - The generalised plant.
 * @param options - The starting bracket.
 * @returns The algorithm; it converges when the bracket is tight to a relative 1e-6.
 * @throws MdmlError with code `value` when the upper level of the bracket is not achievable.
 * @concept h-infinity-control
 * @example
 * ```ts
 * import { hinfBisectionAlgorithm } from 'mdml-engine/dynamics/control'
 * import { runAlgorithm } from 'mdml-engine/trace'
 * const plant = { a: [[1]], b1: [[1, 0]], b2: [[1]], c1: [[1], [0]], c2: [[1]], d12: [[0], [1]], d21: [[0, 1]] }
 * runAlgorithm(hinfBisectionAlgorithm(plant), { maxSteps: 80 }).state.feasible > 1 // => true
 * ```
 */
export function hinfBisectionAlgorithm(
  plant: GeneralisedPlant,
  options: HinfSearchOptions = {},
): Algorithm<HinfSearchState, HinfSearchPath> {
  const upper = options.upper ?? 1000
  const lower = options.lower ?? 1e-3
  const achievedAt = (solution: HinfSolution): number => {
    if (solution.controller === undefined) return Number.POSITIVE_INFINITY
    return item(hinfNorm(closedLoopFromDisturbance(plant, solution.controller)).norm)
  }
  return {
    name: 'hinfBisection',
    config: { upper, lower } as Json,
    init: () => {
      const top = hinfSynthesis(plant, upper)
      if (!top.feasible) throw new MdmlError('value', 'hinfBisectionAlgorithm: the upper level is not achievable')
      return {
        feasible: upper,
        infeasible: lower,
        tried: upper,
        succeeded: true,
        coupling: top.conditions.coupling,
        achieved: achievedAt(top),
        iteration: 0,
      }
    },
    step: (s, ctx) => {
      const gamma = Math.sqrt(s.feasible * s.infeasible)
      const solution = hinfSynthesis(plant, gamma)
      const achieved = solution.feasible ? Math.min(s.achieved, achievedAt(solution)) : s.achieved
      const next: HinfSearchState = {
        feasible: solution.feasible ? gamma : s.feasible,
        infeasible: solution.feasible ? s.infeasible : gamma,
        tried: gamma,
        succeeded: solution.feasible,
        coupling: solution.conditions.coupling,
        achieved,
        iteration: s.iteration + 1,
      }
      ctx.log('hinf/gamma', gamma)
      ctx.log('hinf/feasible', solution.feasible ? 1 : 0)
      ctx.log('hinf/coupling', solution.conditions.coupling)
      ctx.log('hinf/bracket', next.feasible - next.infeasible)
      ctx.log('hinf/achieved', achieved)
      return next
    },
    done: (s) => (s.feasible - s.infeasible <= 1e-6 * s.feasible ? 'converged' : undefined),
    serialise: (s) => stateToJson(s),
    deserialise: (j: Json) => stateFromJson(j) as HinfSearchState,
  }
}
