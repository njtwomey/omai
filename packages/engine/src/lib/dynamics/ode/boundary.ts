/**
 * Boundary value problems for ODEs, where conditions are given at both ends of an interval rather than at the start.
 * `solveBvp` ports scipy's `solve_bvp` (fourth-order Lobatto IIIA collocation with damped Newton and mesh refinement),
 * alongside single and multiple shooting with Jacobians taken by differentiating through the integrator.
 */

import { MdmlError, maxOf } from '../../base/index.ts'
import { pinv } from '../../numerics/linalg/index.ts'
import { fsolveAlgorithm, type SystemPath, type SystemState, systemContract } from '../../numerics/roots/index.ts'
import { concat, reshape, sliceAxis, sub, type Tensor, type TensorLike, type Value } from '../../tensor/index.ts'
import {
  type Algorithm,
  type Contract,
  defineContract,
  runAlgorithm,
  scalar,
  stateFromJson,
  stateToJson,
  tensor,
} from '../../trace/index.ts'
import {
  collocation,
  column,
  createSpline,
  modifyMesh,
  type NumBc,
  type NumField,
  rmsResiduals,
  solveNewton,
} from './_bvp.ts'
import { floatState, fromNumbers, numbersOf } from './_vec.ts'
import type { Stepper, VectorField } from './problem.ts'
import { type OdeMethodName, solveIvp } from './solve-ivp.ts'

/** A vectorised boundary-value field: x `[m]`, y `[n, m]`, parameters `[k]` to dy/dx `[n, m]` (scipy's `fun`). */
export type BvpField = (x: Tensor, y: Tensor, p: Tensor) => Value

/** Boundary conditions: ya `[n]`, yb `[n]`, parameters `[k]` to residuals `[n + k]` (scipy's `bc`). */
export type BvpConditions = (ya: Tensor, yb: Tensor, p: Tensor) => Value

/** Options of {@link solveBvp}. */
export interface SolveBvpOptions {
  /** Initial guesses of unknown parameters. */
  readonly p?: TensorLike
  /** The singular term S of y' = S y / (x - a) + f(x, y, p). */
  readonly S?: TensorLike
  /** Relative residual tolerance (default 1e-3). */
  readonly tol?: number
  /** The mesh-node budget (default 1000). */
  readonly maxNodes?: number
  /** Boundary-condition tolerance (default tol). */
  readonly bcTol?: number
}

/** The result of {@link solveBvp} (scipy's `BVPResult`). */
export interface BvpSolution {
  /** The final mesh. */
  readonly x: readonly number[]
  /** The solution at the mesh `[n, m]`. */
  readonly y: Tensor
  /** Its derivative at the mesh `[n, m]`. */
  readonly yp: Tensor
  /** The parameters found, when there were any. */
  readonly p: Tensor | undefined
  /** The C1 cubic spline solution: values `[n]`, or derivatives with `nu = 1`. */
  readonly sol: (x: number, nu?: 0 | 1) => Tensor
  readonly rmsResiduals: readonly number[]
  readonly niter: number
  /** 0 converged, 1 too many nodes, 2 singular Jacobian, 3 boundary conditions not met. */
  readonly status: 0 | 1 | 2 | 3
  readonly message: string
  readonly success: boolean
}

const MESSAGES = [
  'The algorithm converged to the desired accuracy.',
  'The maximum number of mesh nodes is exceeded.',
  'A singular Jacobian encountered when solving the collocation system.',
  'The solver was unable to satisfy boundary conditions tolerance on iteration 10.',
] as const

/** The state of {@link solveBvpAlgorithm}: one collocation solve and mesh refinement per iteration. */
export interface BvpState {
  /** Mesh iterations done. */
  readonly iteration: number
  /** Why it stopped (the status of {@link BvpSolution}), `null` while it runs. */
  readonly status: 0 | 1 | 2 | 3 | null
  /** The mesh (refined for the next iteration while running) and the solution values on it, row-major `[n, m]`. */
  readonly x: readonly number[]
  readonly y: readonly number[]
  /** The unknown parameters. */
  readonly p: readonly number[]
  /** The derivative at the mesh of the last solve and its relative residuals per interval. */
  readonly f: readonly number[]
  readonly rms: readonly number[]
  /** The largest boundary-condition residual of the last solve. */
  readonly maxBc: number
}

/** The paths {@link solveBvpAlgorithm} logs, as a type. */
export type BvpPath = 'bvp/residual' | 'bvp/mesh-size' | 'bvp/bc-residual' | 'bvp/mesh' | 'bvp/solution'

/**
 * What {@link solveBvpAlgorithm} logs, by path: the largest relative collocation residual of the iteration, the mesh
 * size and the mesh itself, the largest boundary residual, and the solution on the mesh.
 */
export const bvpContract: Contract<BvpState, BvpPath> = defineContract({
  'bvp/residual': scalar({ description: 'the largest relative collocation residual', direction: 'lower' }),
  'bvp/mesh-size': scalar({ description: 'the number of mesh nodes' }),
  'bvp/bc-residual': scalar({ description: 'the largest boundary-condition residual', direction: 'lower' }),
  'bvp/mesh': tensor({ role: 'state', shape: ['node'], description: 'the mesh' }),
  'bvp/solution': tensor({ role: 'state', shape: ['component', 'node'], description: 'the solution on the mesh' }),
})

/**
 * Solve a boundary value problem y' = f(x, y, p), bc(y(a), y(b), p) = 0 by collocation as a steppable algorithm, one
 * mesh iteration per step: a port of scipy's `solve_bvp` (Kierzenka and Shampine's fourth-order Lobatto IIIA collocation with a damped Newton method on the
 * global sparse system, residuals estimated by Lobatto quadrature and the mesh refined where they exceed `tol`).
 * Unknown parameters and a singular term S/(x - a) are supported. Never throws on hard problems: the status says why
 * it stopped.
 *
 * @param fun - The vectorised field (x `[m]`, y `[n, m]`, p `[k]`).
 * @param bc - The boundary residuals (`[n + k]`).
 * @param x - The initial mesh (increasing).
 * @param y - The initial guess `[n, m]`.
 * @param options - Parameters, singular term, tolerances and node budget.
 * @returns The algorithm; it is done when the residuals meet the tolerance (`converged`), the node budget is exceeded
 * (`failed`), the Jacobian is singular (`diverged`) or the boundary conditions are not met after ten iterations
 * (`failed`). Read the solution off the final state with {@link bvpSolution}.
 * @throws MdmlError with code `shape` or `value` for bad inputs.
 * @concept boundary-value-problem
 * @example
 * ```ts
 * import { solveBvpAlgorithm } from 'mdml-engine/dynamics/ode'
 * import { concat, neg, sliceAxis, sub, tensor } from 'mdml-engine/tensor'
 * import { runAlgorithm } from 'mdml-engine/trace'
 * const fun = (_x, y) => concat([sliceAxis(y, 0, 1, 2), neg(sliceAxis(y, 0, 0, 1))], 0)
 * const bc = (ya, yb) => concat([sliceAxis(ya, 0, 0, 1), sub(sliceAxis(yb, 0, 0, 1), 1)], 0)
 * const alg = solveBvpAlgorithm(fun, bc, [0, 0.5, 1, Math.PI / 2], tensor([[0, 0, 0, 0], [1, 1, 1, 1]]), { tol: 1e-6 })
 * runAlgorithm(alg, { maxSteps: 20 }).state.status // => 0
 * ```
 */
export function solveBvpAlgorithm(
  fun: BvpField,
  bc: BvpConditions,
  x: TensorLike,
  y: TensorLike,
  options: SolveBvpOptions = {},
): Algorithm<BvpState, BvpPath> {
  const xs0: Float64Array = numbersOf(floatState(x))
  const yv = floatState(y)
  if (yv.shape.length !== 2) throw new MdmlError('shape', 'solveBvp: y must be [n, m]')
  const n = yv.shape[0] ?? 0
  const m0 = yv.shape[1] ?? 0
  if (m0 !== xs0.length) throw new MdmlError('shape', `solveBvp: y needs ${xs0.length} columns`)
  const h0: Float64Array = Float64Array.from({ length: xs0.length - 1 }, (_, j) => (xs0[j + 1] ?? 0) - (xs0[j] ?? 0))
  if (h0.some((v) => !(v > 0))) throw new MdmlError('value', 'solveBvp: x must be strictly increasing')
  const yN0: Float64Array = Float64Array.from(numbersOf(yv))
  const p0: Float64Array =
    options.p === undefined ? new Float64Array(0) : Float64Array.from(numbersOf(floatState(options.p)))
  const k = p0.length
  const tol = Math.max(options.tol ?? 1e-3, 100 * Number.EPSILON)
  const bcTol = options.bcTol ?? tol
  const maxNodes = options.maxNodes ?? 1000
  const a = xs0[0] ?? 0
  let B: Float64Array | undefined
  let D: Float64Array | undefined
  let S: Float64Array | undefined
  if (options.S !== undefined) {
    const sv = floatState(options.S)
    S = Float64Array.from(numbersOf(sv))
    if (S.length !== n * n) throw new MdmlError('shape', `solveBvp: S must be ${n} x ${n}`)
    const ps = numbersOf(pinv(sv))
    B = new Float64Array(n * n)
    for (let i = 0; i < n; i++) {
      for (let j = 0; j < n; j++) {
        let s = 0
        for (let c = 0; c < n; c++) s += (ps[i * n + c] ?? 0) * (S[c * n + j] ?? 0)
        B[i * n + j] = (i === j ? 1 : 0) - s
      }
    }
    const ims = Float64Array.from(S, (v, idx) => (Math.floor(idx / n) === idx % n ? 1 : 0) - v)
    D = Float64Array.from(numbersOf(pinv(fromNumbers(ims, [n, n]))))
    const y0 = column(yN0, n, m0, 0)
    for (let i = 0; i < n; i++) {
      let s = 0
      for (let c = 0; c < n; c++) s += (B[i * n + c] ?? 0) * (y0[c] ?? 0)
      yN0[i * m0] = s
    }
  }
  const raw: NumField = (xv, yy, pp) => {
    const out = numbersOf(
      fun(fromNumbers(xv, [xv.length]), fromNumbers(yy, [n, xv.length]), fromNumbers(pp, [pp.length])),
    )
    if (out.length !== n * xv.length) throw new MdmlError('shape', `solveBvp: fun must return [${n}, ${xv.length}]`)
    return Float64Array.from(out)
  }
  const field: NumField =
    S === undefined
      ? raw
      : (xv, yy, pp) => {
          const mm = xv.length
          const f = raw(xv, yy, pp)
          const Sv = S as Float64Array
          for (let j = 0; j < mm; j++) {
            const xj = xv[j] ?? 0
            if (j === 0 && xj === a) {
              const f0 = column(f, n, mm, 0)
              for (let i = 0; i < n; i++) {
                let s = 0
                for (let c = 0; c < n; c++) s += (D?.[i * n + c] ?? 0) * (f0[c] ?? 0)
                f[i * mm] = s
              }
              continue
            }
            for (let i = 0; i < n; i++) {
              let s = 0
              for (let c = 0; c < n; c++) s += (Sv[i * n + c] ?? 0) * (yy[c * mm + j] ?? 0)
              f[i * mm + j] = (f[i * mm + j] ?? 0) + s / (xj - a)
            }
          }
          return f
        }
  const bcN: NumBc = (ya, yb, pp) => {
    const out = numbersOf(bc(fromNumbers(ya, [n]), fromNumbers(yb, [n]), fromNumbers(pp, [pp.length])))
    if (out.length !== n + k) throw new MdmlError('shape', `solveBvp: bc must return ${n + k} residuals`)
    return Float64Array.from(out)
  }
  const stateOf = (
    iteration: number,
    status: 0 | 1 | 2 | 3 | null,
    xs: Float64Array,
    yN: Float64Array,
    p: Float64Array,
    f: Float64Array,
    rms: Float64Array,
    maxBc: number,
  ): BvpState => ({
    iteration,
    status,
    x: Array.from(xs),
    y: Array.from(yN),
    p: Array.from(p),
    f: Array.from(f),
    rms: Array.from(rms),
    maxBc,
  })
  const bvpStep = (s: BvpState): BvpState => {
    let xs = Float64Array.from(s.x)
    const h = Float64Array.from({ length: xs.length - 1 }, (_, j) => (xs[j + 1] ?? 0) - (xs[j] ?? 0))
    const m = xs.length
    const r = solveNewton(field, bcN, xs, h, Float64Array.from(s.y), Float64Array.from(s.p), B, tol, bcTol, n)
    let yN = r.y
    const p = r.p
    const iteration = s.iteration + 1
    const col = collocation(field, yN, p, xs, h, n)
    const f = col.f
    const bcRes = bcN(column(yN, n, m, 0), column(yN, n, m, m - 1), p)
    const maxBc = maxOf(Array.from(bcRes, Math.abs))
    const rMid = Float64Array.from(col.colRes, (v, idx) => (1.5 * v) / (h[idx % (m - 1)] ?? 1))
    const sol = createSpline(yN, f, xs, h, n)
    const rms = rmsResiduals(field, sol, xs, h, p, rMid, col.fMid, n)
    if (r.singular) return stateOf(iteration, 2, xs, yN, p, f, rms, maxBc)
    const one: number[] = []
    const two: number[] = []
    rms.forEach((v, j) => {
      if (v > tol && v < 100 * tol) one.push(j)
      else if (v >= 100 * tol) two.push(j)
    })
    const added = one.length + 2 * two.length
    if (m + added > maxNodes) return stateOf(iteration, 1, xs, yN, p, f, rms, maxBc)
    if (added > 0) {
      const solved = stateOf(iteration, null, xs, yN, p, f, rms, maxBc)
      xs = Float64Array.from(modifyMesh(xs, one, two))
      const next = new Float64Array(n * xs.length)
      xs.forEach((xv, j) => {
        const v = sol.at(xv)
        for (let i = 0; i < n; i++) next[i * xs.length + j] = v[i] ?? 0
      })
      yN = next
      // The mesh is refined for the next iteration; the residuals shown are those of the mesh just solved.
      return { ...solved, x: Array.from(xs), y: Array.from(yN) }
    }
    if (maxBc <= bcTol) return stateOf(iteration, 0, xs, yN, p, f, rms, maxBc)
    if (iteration >= 10) return stateOf(iteration, 3, xs, yN, p, f, rms, maxBc)
    return stateOf(iteration, null, xs, yN, p, f, rms, maxBc)
  }
  return {
    name: 'solve-bvp',
    config: { tol, bcTol, maxNodes, n, parameters: k },
    init: () => ({
      iteration: 0,
      status: null,
      x: Array.from(xs0),
      y: Array.from(yN0),
      p: Array.from(p0),
      f: [],
      rms: [],
      maxBc: Number.NaN,
    }),
    step: (s, ctx) => {
      const next = bvpStep(s)
      if (ctx.wants('bvp/residual')) ctx.log('bvp/residual', maxOf(next.rms))
      if (ctx.wants('bvp/mesh-size')) ctx.log('bvp/mesh-size', next.x.length)
      if (ctx.wants('bvp/bc-residual')) ctx.log('bvp/bc-residual', next.maxBc)
      if (ctx.wants('bvp/mesh')) ctx.log('bvp/mesh', next.x)
      if (ctx.wants('bvp/solution')) ctx.log('bvp/solution', fromNumbers(next.y, [n, next.x.length]))
      return next
    },
    done: (s) =>
      s.status === null ? undefined : s.status === 0 ? 'converged' : s.status === 2 ? 'diverged' : 'failed',
    serialise: (s) => stateToJson(s),
    deserialise: (j) => stateFromJson(j) as BvpState,
  }
}

/**
 * The solution of a finished {@link solveBvpAlgorithm} run: the mesh, the values and derivatives on it, the C1 cubic
 * spline, the residuals and the status, as {@link solveBvp} returns them.
 *
 * @param state - The state of a run to its end.
 * @param n - The number of components of y.
 * @returns The solution.
 * @example
 * ```ts
 * import { bvpSolution } from 'mdml-engine/dynamics/ode'
 * bvpSolution({ iteration: 1, status: 0, x: [0, 1], y: [0, 1], p: [], f: [1, 1], rms: [0], maxBc: 0 }, 1).success // => true
 * ```
 */
export function bvpSolution(state: BvpState, n: number): BvpSolution {
  const xs = Float64Array.from(state.x)
  const h = Float64Array.from({ length: xs.length - 1 }, (_, j) => (xs[j + 1] ?? 0) - (xs[j] ?? 0))
  const sol = createSpline(Float64Array.from(state.y), Float64Array.from(state.f), xs, h, n)
  const status = (state.status ?? 3) as 0 | 1 | 2 | 3
  const k = state.p.length
  return {
    x: [...state.x],
    y: fromNumbers(state.y, [n, xs.length]),
    yp: fromNumbers(state.f, [n, xs.length]),
    p: k === 0 ? undefined : fromNumbers(state.p, [k]),
    sol: (xv, nu = 0) => fromNumbers(sol.at(xv, nu), [n]),
    rmsResiduals: [...state.rms],
    niter: state.iteration,
    status,
    message: MESSAGES[status],
    success: status === 0,
  }
}

/**
 * Solve a boundary value problem y' = f(x, y, p), bc(y(a), y(b), p) = 0 by collocation ({@link solveBvpAlgorithm} run to
 * the end): a port of scipy's `solve_bvp` (Kierzenka and Shampine's fourth-order Lobatto IIIA collocation with a damped Newton method on the
 * global sparse system, residuals estimated by Lobatto quadrature and the mesh refined where they exceed `tol`).
 * Unknown parameters and a singular term S/(x - a) are supported. Never throws on hard problems: the status says why
 * it stopped.
 *
 * @param fun - The vectorised field (x `[m]`, y `[n, m]`, p `[k]`).
 * @param bc - The boundary residuals (`[n + k]`).
 * @param x - The initial mesh (increasing).
 * @param y - The initial guess `[n, m]`.
 * @param options - Parameters, singular term, tolerances and node budget.
 * @returns The solution with its mesh, spline, residuals and status.
 * @throws MdmlError with code `shape` or `value` for bad inputs.
 * @concept boundary-value-problem
 * @index function Boundary value problems by collocation.
 * @example
 * ```ts
 * import { solveBvp } from 'mdml-engine/dynamics/ode'
 * import { concat, neg, sliceAxis, sub, tensor } from 'mdml-engine/tensor'
 * // y'' = -y on [0, pi/2], y(0) = 0, y(pi/2) = 1: y = sin x.
 * const fun = (_x, y) => concat([sliceAxis(y, 0, 1, 2), neg(sliceAxis(y, 0, 0, 1))], 0)
 * const bc = (ya, yb) => concat([sliceAxis(ya, 0, 0, 1), sub(sliceAxis(yb, 0, 0, 1), 1)], 0)
 * const r = solveBvp(fun, bc, [0, 0.5, 1, Math.PI / 2], tensor([[0, 0, 0, 0], [1, 1, 1, 1]]), { tol: 1e-6 })
 * Math.abs((r.sol(Math.PI / 6).data[0] ?? 0) - 0.5) < 1e-5 // => true
 * ```
 */
export function solveBvp(
  fun: BvpField,
  bc: BvpConditions,
  x: TensorLike,
  y: TensorLike,
  options: SolveBvpOptions = {},
): BvpSolution {
  const yv = floatState(y)
  const algorithm = solveBvpAlgorithm(fun, bc, x, y, options)
  // Each iteration either settles, refines the mesh or repeats on the same mesh (at most ten of those).
  const state = runAlgorithm(algorithm, { maxSteps: 1_000_000 }).state
  return bvpSolution(state, yv.shape[0] ?? 0)
}

/** Options of the shooting methods. */
export interface ShootingOptions {
  /** The integrator (default DOP853). */
  readonly method?: OdeMethodName | Stepper
  readonly rtol?: number
  readonly atol?: number
  /** Newton iterations (default 50). */
  readonly maxIterations?: number
  /** Residual tolerance (default 1e-10). */
  readonly ftol?: number
}

/** The result of a shooting method. */
export interface ShootingResult {
  /** The initial state found. */
  readonly y0: Tensor
  /** The states at the shooting nodes `[nodes, n]` (the initial state alone for single shooting). */
  readonly nodes: Tensor
  readonly residual: number
  readonly iterations: number
  readonly converged: boolean
  readonly message: string
}

function integrate(f: VectorField, t0: number, t1: number, y0: Value, options: ShootingOptions): Value {
  const sol = solveIvp(f, [t0, t1], y0, {
    method: options.method ?? 'DOP853',
    rtol: options.rtol ?? 1e-10,
    atol: options.atol ?? 1e-12,
    tEval: [t1],
  })
  if (!sol.success) throw new MdmlError('convergence', `shooting: the integration failed: ${sol.message}`)
  return sol.ys[sol.ys.length - 1] as Value
}

/** The paths the shooting algorithms log, as a type: the Newton iteration's own, plus the miss and the unknown. */
export type ShootPath = SystemPath | 'shoot/miss' | 'shoot/slope'

/**
 * What the shooting methods log per Newton iteration: the norm of the miss (the boundary residual at the end of the
 * integration), the unknown states, and the paths of the Newton iteration itself (the step length, backtracks).
 */
export const shootContract: Contract<unknown, ShootPath> = defineContract({
  ...(systemContract as unknown as Contract<unknown, SystemPath>),
  'shoot/miss': scalar({ description: 'the norm of the boundary residual at the iterate', direction: 'lower' }),
  'shoot/slope': tensor({
    role: 'state',
    shape: ['unknowns'],
    description: 'the unknown initial state (all node states for multiple shooting)',
  }),
})

/** Wrap Newton's method on a shooting residual as an algorithm that also logs the miss and the unknowns. */
function shootingAlgorithm(
  name: string,
  residual: (z: Value) => Value,
  z0: Value,
  options: ShootingOptions,
): Algorithm<SystemState, ShootPath> {
  const maxIterations = options.maxIterations ?? 50
  const inner = fsolveAlgorithm(residual, z0, { maxIterations, ftol: options.ftol ?? 1e-10 })
  return {
    ...inner,
    name,
    step: (s, ctx) => {
      const next = inner.step(s, ctx)
      ctx.log('shoot/miss', next.history[next.history.length - 1] ?? Number.NaN)
      ctx.log('shoot/slope', Float64Array.from(next.x))
      return next
    },
  }
}

/** The result of a finished shooting run from its Newton state. */
function shootingResult(s: SystemState, maxIterations: number, n: number, segments: number): ShootingResult {
  return {
    y0: fromNumbers(s.x.slice(0, n), [n]),
    nodes: fromNumbers(s.x, [segments, n]),
    residual: s.history[s.history.length - 1] ?? Number.NaN,
    iterations: s.converged ? s.iteration : maxIterations,
    converged: s.converged,
    message: s.converged ? 'converged' : 'maximum iterations',
  }
}

/**
 * Single shooting as a stepped algorithm, one Newton iteration per step: find the initial state s with bc(s, y(b; s)) = 0,
 * integrating from a to b for the miss and differentiating through the integrator for the Jacobian. It logs the miss
 * (`shoot/miss`), the unknown initial state (`shoot/slope`) and the Newton iteration's step length and backtracks, and is
 * done (`converged`) when the miss or the step is small and `stopped` at the iteration cap. The state is a
 * {@link SystemState} and serialises; read the result with {@link shootingOf}.
 *
 * @param f - The vector field dy/dt = f(t, y).
 * @param bc - Boundary residuals bc(ya, yb) of length n.
 * @param span - [a, b].
 * @param guess - The initial-state guess `[n]`.
 * @param options - Integrator, tolerances and iteration cap.
 * @returns The algorithm.
 * @concept shooting-method
 * @example
 * ```ts
 * import { shootingOf, singleShootingAlgorithm } from 'mdml-engine/dynamics/ode'
 * import { runAlgorithm } from 'mdml-engine/trace'
 * import { concat, neg, sliceAxis, sub } from 'mdml-engine/tensor'
 * const f = (_t, y) => concat([sliceAxis(y, 0, 1, 2), neg(sliceAxis(y, 0, 0, 1))], 0)
 * const bc = (ya, yb) => concat([sliceAxis(ya, 0, 0, 1), sub(sliceAxis(yb, 0, 0, 1), 1)], 0)
 * const run = runAlgorithm(singleShootingAlgorithm(f, bc, [0, Math.PI / 2], [0, 0.5]), { maxSteps: 50 })
 * Math.abs((shootingOf(run.state, 2).y0.data[1] ?? 0) - 1) < 1e-8 // => true
 * ```
 */
export function singleShootingAlgorithm(
  f: VectorField,
  bc: (ya: Value, yb: Value) => Value,
  span: readonly [number, number],
  guess: TensorLike,
  options: ShootingOptions = {},
): Algorithm<SystemState, ShootPath> {
  const s0 = floatState(guess)
  const residual = (s: Value): Value => bc(s, integrate(f, span[0], span[1], s, options))
  return shootingAlgorithm('ode/shooting', residual, s0, options)
}

/**
 * The result of a shooting state: the initial state, the states at the nodes, the final miss and the iterations.
 *
 * @param state - A state of {@link singleShootingAlgorithm} or {@link multipleShootingAlgorithm}.
 * @param n - The dimension of the state.
 * @param options - The options the run used (for the iteration cap).
 * @returns The result.
 * @concept shooting-method
 * @example
 * ```ts
 * import { shootingOf } from 'mdml-engine/dynamics/ode'
 * shootingOf({ x: [0, 1], residual: [0], iteration: 3, functionCalls: 5, history: [1, 0], jacobian: null, stepLength: 1, backtracks: 0, jacobianAge: 0, converged: true }, 2).iterations // => 3
 * ```
 */
export function shootingOf(state: SystemState, n: number, options: ShootingOptions = {}): ShootingResult {
  return shootingResult(state, options.maxIterations ?? 50, n, Math.max(1, Math.floor(state.x.length / n)))
}

/**
 * Single shooting ({@link singleShootingAlgorithm} run to the end): find the initial state s with bc(s, y(b; s)) = 0 by Newton's method, the Jacobian of the end
 * state with respect to s taken by differentiating through the integrator.
 *
 * @param f - The vector field dy/dt = f(t, y).
 * @param bc - Boundary residuals bc(ya, yb) of length n.
 * @param span - [a, b].
 * @param guess - The initial-state guess `[n]`.
 * @param options - Integrator, tolerances and iteration cap.
 * @returns The initial state and convergence information.
 * @concept shooting-method
 * @example
 * ```ts
 * import { shooting } from 'mdml-engine/dynamics/ode'
 * import { concat, neg, sliceAxis, sub } from 'mdml-engine/tensor'
 * const f = (_t, y) => concat([sliceAxis(y, 0, 1, 2), neg(sliceAxis(y, 0, 0, 1))], 0)
 * const bc = (ya, yb) => concat([sliceAxis(ya, 0, 0, 1), sub(sliceAxis(yb, 0, 0, 1), 1)], 0)
 * const r = shooting(f, bc, [0, Math.PI / 2], [0, 0.5])
 * Math.abs((r.y0.data[1] ?? 0) - 1) < 1e-8 // => true
 * ```
 */
export function shooting(
  f: VectorField,
  bc: (ya: Value, yb: Value) => Value,
  span: readonly [number, number],
  guess: TensorLike,
  options: ShootingOptions = {},
): ShootingResult {
  const n = floatState(guess).shape[0] ?? 0
  const run = runAlgorithm(singleShootingAlgorithm(f, bc, span, guess, options), {
    maxSteps: options.maxIterations ?? 50,
  })
  const r = shootingResult(run.state, options.maxIterations ?? 50, n, 1)
  return { ...r, nodes: reshape(fromNumbers(run.state.x, [run.state.x.length]), [1, -1]) as Tensor }
}

/**
 * Multiple shooting as a stepped algorithm, one Newton iteration per step: unknown states at every node, matched by
 * continuity and the boundary conditions, all solved together. It logs what {@link singleShootingAlgorithm} logs, with the
 * unknowns the states at all the nodes but the last.
 *
 * @param f - The vector field.
 * @param bc - Boundary residuals bc(ya, yb) of length n.
 * @param nodes - The shooting nodes, starting at a, ending at b.
 * @param guesses - Guesses `[nodes - 1, n]` for the state at each node but the last.
 * @param options - Integrator, tolerances and iteration cap.
 * @returns The algorithm.
 * @throws MdmlError with code `value` for fewer than two nodes or mismatched guesses.
 * @concept shooting-method
 * @example
 * ```ts
 * import { multipleShootingAlgorithm } from 'mdml-engine/dynamics/ode'
 * import { concat, neg, sliceAxis, sub } from 'mdml-engine/tensor'
 * const f = (_t, y) => concat([sliceAxis(y, 0, 1, 2), neg(sliceAxis(y, 0, 0, 1))], 0)
 * const bc = (ya, yb) => concat([sliceAxis(ya, 0, 0, 1), sub(sliceAxis(yb, 0, 0, 1), 1)], 0)
 * multipleShootingAlgorithm(f, bc, [0, 0.8, Math.PI / 2], [[0, 0.5], [0.5, 0.5]]).name // => 'ode/multiple-shooting'
 * ```
 */
export function multipleShootingAlgorithm(
  f: VectorField,
  bc: (ya: Value, yb: Value) => Value,
  nodes: readonly number[],
  guesses: TensorLike,
  options: ShootingOptions = {},
): Algorithm<SystemState, ShootPath> {
  if (nodes.length < 2) throw new MdmlError('value', 'multipleShooting: give at least two nodes')
  const g = floatState(guesses)
  const segments = nodes.length - 1
  if (g.shape.length !== 2 || g.shape[0] !== segments) {
    throw new MdmlError('value', `multipleShooting: give ${segments} guesses as [${segments}, n]`)
  }
  const n = g.shape[1] ?? 0
  const residual = (z: Value): Value => {
    const s = (j: number): Value => sliceAxis(z, 0, j * n, (j + 1) * n)
    const parts: Value[] = []
    for (let j = 0; j < segments - 1; j++) {
      parts.push(sub(integrate(f, nodes[j] as number, nodes[j + 1] as number, s(j), options), s(j + 1)))
    }
    const end = integrate(f, nodes[segments - 1] as number, nodes[segments] as number, s(segments - 1), options)
    parts.push(bc(s(0), end))
    return concat(parts, 0)
  }
  return shootingAlgorithm('ode/multiple-shooting', residual, reshape(g, [-1]), options)
}

/**
 * Multiple shooting ({@link multipleShootingAlgorithm} run to the end): unknown states at every node t_0 < ... < t_M-1 (with t_0 = a and the last interval ending at b),
 * matched by continuity y(t_{j+1}; s_j) = s_{j+1} and the boundary conditions, all solved together by Newton's
 * method. Better conditioned than single shooting for unstable or long intervals.
 *
 * @param f - The vector field.
 * @param bc - Boundary residuals bc(ya, yb) of length n.
 * @param nodes - The shooting nodes, starting at a, ending at b.
 * @param guesses - Guesses `[nodes - 1, n]` for the state at each node but the last.
 * @param options - Integrator, tolerances and iteration cap.
 * @returns The states at the nodes and convergence information.
 * @throws MdmlError with code `value` for fewer than two nodes or mismatched guesses.
 * @concept shooting-method
 * @example
 * ```ts
 * import { multipleShooting } from 'mdml-engine/dynamics/ode'
 * import { concat, neg, sliceAxis, sub } from 'mdml-engine/tensor'
 * const f = (_t, y) => concat([sliceAxis(y, 0, 1, 2), neg(sliceAxis(y, 0, 0, 1))], 0)
 * const bc = (ya, yb) => concat([sliceAxis(ya, 0, 0, 1), sub(sliceAxis(yb, 0, 0, 1), 1)], 0)
 * const r = multipleShooting(f, bc, [0, 0.8, Math.PI / 2], [[0, 0.5], [0.5, 0.5]])
 * Math.abs((r.y0.data[1] ?? 0) - 1) < 1e-8 // => true
 * ```
 */
export function multipleShooting(
  f: VectorField,
  bc: (ya: Value, yb: Value) => Value,
  nodes: readonly number[],
  guesses: TensorLike,
  options: ShootingOptions = {},
): ShootingResult {
  const alg = multipleShootingAlgorithm(f, bc, nodes, guesses, options)
  const g = floatState(guesses)
  const run = runAlgorithm(alg, { maxSteps: options.maxIterations ?? 50 })
  return shootingResult(run.state, options.maxIterations ?? 50, g.shape[1] ?? 0, nodes.length - 1)
}
