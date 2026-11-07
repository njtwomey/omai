/**
 * Solving systems of nonlinear equations F(x) = 0 with Jacobians from autodiff: damped Newton with a backtracking line
 * search (`fsolve`), Broyden's quasi-Newton method, fixed-point iteration with Steffensen or Anderson acceleration, a
 * `root` front end as in scipy.optimize, and `rootOf`, a solution differentiable in its parameters by the implicit
 * function theorem. Each solver is also a steppable algorithm over a serialisable state (`fsolveAlgorithm`,
 * `broydenAlgorithm`, `fixedPointAlgorithm`; one step is one iteration).
 */

import { customVjp, jvp as forward, jacobian, vjp } from '../../autodiff/index.ts'
import { MdmlError, maxOf } from '../../base/index.ts'
import {
  add,
  concreteOrThrow,
  fromStorage,
  matmul,
  mul,
  neg,
  primalOf,
  promoteFloat,
  sub,
  swapaxes,
  type Tensor,
  type TensorLike,
  toFlat,
  type Value,
} from '../../tensor/index.ts'
import {
  type Algorithm,
  type Contract,
  defineContract,
  type RunStatus,
  runAlgorithm,
  type StepContext,
  scalar,
  stateFromJson,
  stateToJson,
  tensor,
} from '../../trace/index.ts'
import { lstsq, dnrm2 as norm, solve } from '../linalg/index.ts'

/** A vector function F: R^n -> R^n, written with operations (so its Jacobian comes from autodiff). */
export type VectorFunction = (x: Value) => Value

/** Result of a system solver. */
export interface SystemResult {
  readonly x: Tensor
  /** F at the solution. */
  readonly residual: Tensor
  readonly iterations: number
  readonly functionCalls: number
  readonly converged: boolean
  readonly message: string
  /** The residual norm after each iteration. */
  readonly history: readonly number[]
}

/** Options of the system solvers. */
export interface SystemOptions {
  /** Stop when the step is below xtol (|x| + xtol) or the residual norm below ftol. */
  readonly xtol?: number
  readonly ftol?: number
  readonly maxIterations?: number
  /** The Jacobian; by default reverse- or forward-mode autodiff of F. */
  readonly jacobian?: (x: Value) => Value
}

const vec = (x: TensorLike): Tensor => concreteOrThrow(promoteFloat(x)[0] as Tensor, 'vector')

/** The state of {@link fsolveAlgorithm} and {@link broydenAlgorithm}: the iterate, its residual and the Jacobian in use. */
export interface SystemState {
  readonly x: readonly number[]
  /** F at the iterate. */
  readonly residual: readonly number[]
  readonly iteration: number
  readonly functionCalls: number
  /** The residual norm at the start and after each iteration. */
  readonly history: readonly number[]
  /** The Jacobian estimate, row-major (Broyden; `null` for Newton, which recomputes it). */
  readonly jacobian: readonly number[] | null
  /** The line-search step length taken in the last iteration (1 when no backtracking was needed). */
  readonly stepLength: number
  /** How many times the last line search halved the step. */
  readonly backtracks: number
  /** Iterations since the Jacobian was exact (0 for Newton, which recomputes it every time). */
  readonly jacobianAge: number
  readonly converged: boolean
}

/** The paths the system solvers log, as a type. */
export type SystemPath =
  | 'system/x'
  | 'system/residual'
  | 'system/step-length'
  | 'system/backtracks'
  | 'system/jacobian-age'

/**
 * What the system solvers log, by path: the iterate, the residual norm, the line-search step length and the number of
 * halvings, and (Broyden) the iterations since the Jacobian was exact.
 */
export const systemContract: Contract<SystemState, SystemPath> = defineContract({
  'system/x': tensor({ role: 'state', shape: ['n'], description: 'the iterate' }),
  'system/residual': scalar({ description: 'the norm of F at the iterate', direction: 'lower' }),
  'system/step-length': scalar({ description: 'the line-search step length (1 = the full Newton step)' }),
  'system/backtracks': scalar({ description: 'halvings of the step by the line search' }),
  'system/jacobian-age': scalar({ description: 'iterations since the Jacobian was computed exactly' }),
})

const asVector = (v: readonly number[]): Tensor => fromStorage(Float64Array.from(v), [v.length], 'float64')

function logSystem(ctx: StepContext<SystemPath>, s: SystemState): void {
  if (ctx.wants('system/x')) ctx.log('system/x', Float64Array.from(s.x))
  if (ctx.wants('system/residual')) ctx.log('system/residual', s.history[s.history.length - 1] ?? 0)
  if (ctx.wants('system/step-length')) ctx.log('system/step-length', s.stepLength)
  if (ctx.wants('system/backtracks')) ctx.log('system/backtracks', s.backtracks)
  if (ctx.wants('system/jacobian-age')) ctx.log('system/jacobian-age', s.jacobianAge)
}

const statusOfSystem = (s: SystemState, max: number): RunStatus | undefined =>
  s.converged ? 'converged' : s.iteration >= max ? 'stopped' : undefined

function systemResult(s: SystemState, max: number): SystemResult {
  return {
    x: asVector(s.x),
    residual: asVector(s.residual),
    iterations: s.converged ? s.iteration : max,
    functionCalls: s.functionCalls,
    converged: s.converged,
    message: s.converged ? 'converged' : 'maximum iterations',
    history: s.history,
  }
}

/** The backtracking (Armijo) line search shared by the Newton and Broyden iterations. */
function lineSearch(
  f: VectorFunction,
  x: Tensor,
  step: Tensor,
  f0: number,
): {
  readonly next: Tensor
  readonly fn: Tensor
  readonly t: number
  readonly backtracks: number
  readonly calls: number
} {
  let t = 1
  let next = vec(add(x, mul(step, t)))
  let fn = vec(f(next))
  let calls = 1
  let backtracks = 0
  while (norm(toFlat(fn)) > (1 - 1e-4 * t) * f0 && t > 1e-10) {
    t /= 2
    next = vec(add(x, mul(step, t)))
    fn = vec(f(next))
    calls += 1
    backtracks += 1
  }
  return { next, fn, t, backtracks, calls }
}

/**
 * Newton's method with a backtracking (Armijo) line search on ||F||^2, as a steppable algorithm: one step solves with the
 * Jacobian from autodiff, searches along the Newton step and moves. The state holds the iterate, its residual, the
 * residual norms so far and what the line search did.
 *
 * @param f - The system, built from operations.
 * @param x0 - Starting point `[n]`.
 * @param options - Tolerances, iteration cap and an optional Jacobian.
 * @returns The algorithm; its state is a {@link SystemState}.
 * @concept root-finding
 * @example
 * ```ts
 * import { fsolveAlgorithm } from 'mdml-engine/numerics/roots'
 * import { runAlgorithm } from 'mdml-engine/trace'
 * import { sub, tensor } from 'mdml-engine/tensor'
 * runAlgorithm(fsolveAlgorithm((v) => sub(v, tensor([3, 4])), [0, 0]), { maxSteps: 20 }).state.x // => [3, 4]
 * ```
 */
export function fsolveAlgorithm(
  f: VectorFunction,
  x0: TensorLike,
  options: SystemOptions = {},
): Algorithm<SystemState, SystemPath> {
  const xtol = options.xtol ?? 1.49012e-8
  const ftol = options.ftol ?? 1e-14
  const max = options.maxIterations ?? 100
  const jac = options.jacobian ?? ((x: Value) => jacobian(f)(x))
  return {
    name: 'fsolve',
    config: { xtol, ftol, maxIterations: max },
    init: () => {
      const x = vec(x0)
      const fx = vec(f(x))
      return {
        x: toFlat(x),
        residual: toFlat(fx),
        iteration: 0,
        functionCalls: 1,
        history: [norm(toFlat(fx))],
        jacobian: null,
        stepLength: 1,
        backtracks: 0,
        jacobianAge: 0,
        converged: false,
      }
    },
    step: (s, ctx) => {
      const x = asVector(s.x)
      const step = vec(solve(jac(x), neg(asVector(s.residual))))
      const search = lineSearch(f, x, step, s.history[s.history.length - 1] ?? 0)
      const moved = norm(toFlat(sub(search.next, x)))
      const fnorm = norm(toFlat(search.fn))
      const nx = toFlat(search.next)
      const next: SystemState = {
        x: nx,
        residual: toFlat(search.fn),
        iteration: s.iteration + 1,
        functionCalls: s.functionCalls + search.calls,
        history: [...s.history, fnorm],
        jacobian: null,
        stepLength: search.t,
        backtracks: search.backtracks,
        jacobianAge: 0,
        converged: fnorm <= ftol || moved <= xtol * (norm(nx) + xtol),
      }
      logSystem(ctx, next)
      return next
    },
    done: (s) => statusOfSystem(s, max),
    serialise: (s) => stateToJson(s),
    deserialise: (j) => stateFromJson(j) as SystemState,
  }
}

/**
 * Solve F(x) = 0 by Newton's method with a backtracking (Armijo) line search on ||F||^2, using the Jacobian from
 * autodiff (scipy.optimize.fsolve's role; scipy uses MINPACK's hybrid method). {@link fsolveAlgorithm} run to the end.
 *
 * @param f - The system, built from operations.
 * @param x0 - Starting point `[n]`.
 * @param options - Tolerances, iteration cap and an optional Jacobian.
 * @returns The solution with its residual history.
 * @concept root-finding
 * @example
 * ```ts
 * import { fsolve } from 'mdml-engine/numerics/roots'
 * import { sub, mul, tensor, toArray, sliceAxis, concat, square } from 'mdml-engine/tensor'
 * // x^2 = 4 and x y = 2 near (1, 1): (2, 1).
 * const F = (v) => concat([sub(square(sliceAxis(v, 0, 0, 1)), 4), sub(mul(sliceAxis(v, 0, 0, 1), sliceAxis(v, 0, 1, 2)), 2)])
 * toArray(fsolve(F, [1, 1]).x).map((v) => Math.round(v * 1e10) / 1e10) // => [2, 1]
 * ```
 */
export function fsolve(f: VectorFunction, x0: TensorLike, options: SystemOptions = {}): SystemResult {
  const max = options.maxIterations ?? 100
  return systemResult(runAlgorithm(fsolveAlgorithm(f, x0, options), { maxSteps: max }).state, max)
}

/**
 * Broyden's ("good") quasi-Newton method as a steppable algorithm: the Jacobian at the start (from autodiff) is updated by
 * rank-one secant corrections instead of being recomputed, with a backtracking line search. The state keeps the Jacobian
 * estimate and logs how many iterations old it is.
 *
 * @param f - The system.
 * @param x0 - Starting point.
 * @param options - Tolerances, iteration cap and an optional initial Jacobian.
 * @returns The algorithm; its state is a {@link SystemState}.
 * @example
 * ```ts
 * import { broydenAlgorithm } from 'mdml-engine/numerics/roots'
 * import { runAlgorithm } from 'mdml-engine/trace'
 * import { mul, sub, tensor } from 'mdml-engine/tensor'
 * runAlgorithm(broydenAlgorithm((v) => sub(mul(v, 2), tensor([2, 4])), [0, 0]), { maxSteps: 20 }).state.x // => [1, 2]
 * ```
 */
export function broydenAlgorithm(
  f: VectorFunction,
  x0: TensorLike,
  options: SystemOptions = {},
): Algorithm<SystemState, SystemPath> {
  const xtol = options.xtol ?? 1.49012e-8
  const ftol = options.ftol ?? 1e-14
  const max = options.maxIterations ?? 200
  return {
    name: 'broyden',
    config: { xtol, ftol, maxIterations: max },
    init: () => {
      const x = vec(x0)
      const fx = vec(f(x))
      const b = vec((options.jacobian ?? ((v: Value) => jacobian(f)(v)))(x))
      return {
        x: toFlat(x),
        residual: toFlat(fx),
        iteration: 0,
        functionCalls: 1,
        history: [norm(toFlat(fx))],
        jacobian: toFlat(b),
        stepLength: 1,
        backtracks: 0,
        jacobianAge: 0,
        converged: false,
      }
    },
    step: (s, ctx) => {
      const n = s.x.length
      const x = asVector(s.x)
      const fx = asVector(s.residual)
      let b: Tensor = fromStorage(Float64Array.from(s.jacobian ?? []), [n, n], 'float64')
      const step = vec(solve(b, neg(fx)))
      const search = lineSearch(f, x, step, s.history[s.history.length - 1] ?? 0)
      const dx = toFlat(sub(search.next, x))
      const df = toFlat(sub(search.fn, fx))
      // B <- B + (df - B dx) dx^T / (dx^T dx).
      const bdx = toFlat(matmul(b, fromStorage(Float64Array.from(dx), [dx.length], 'float64')))
      const denom = dx.reduce((sum, v) => sum + v * v, 0)
      if (denom > 0) {
        const update = new Float64Array(n * n)
        for (let r = 0; r < n; r++)
          for (let c = 0; c < n; c++) update[r * n + c] = (((df[r] ?? 0) - (bdx[r] ?? 0)) * (dx[c] ?? 0)) / denom
        b = vec(add(b, fromStorage(update, [n, n], 'float64')))
      }
      const fnorm = norm(toFlat(search.fn))
      const nx = toFlat(search.next)
      const next: SystemState = {
        x: nx,
        residual: toFlat(search.fn),
        iteration: s.iteration + 1,
        functionCalls: s.functionCalls + search.calls,
        history: [...s.history, fnorm],
        jacobian: toFlat(b),
        stepLength: search.t,
        backtracks: search.backtracks,
        jacobianAge: s.jacobianAge + 1,
        converged: fnorm <= ftol || Math.sqrt(denom) <= xtol * (norm(nx) + xtol),
      }
      logSystem(ctx, next)
      return next
    },
    done: (s) => statusOfSystem(s, max),
    serialise: (s) => stateToJson(s),
    deserialise: (j) => stateFromJson(j) as SystemState,
  }
}

/**
 * Broyden's ("good") quasi-Newton method: the Jacobian at the start (from autodiff) is updated by rank-one secant
 * corrections instead of being recomputed, with a backtracking line search.
 *
 * @param f - The system.
 * @param x0 - Starting point.
 * @param options - Tolerances, iteration cap and an optional initial Jacobian.
 * @returns The solution with its residual history.
 * @example
 * ```ts
 * import { broyden } from 'mdml-engine/numerics/roots'
 * import { sub, mul, tensor, toArray } from 'mdml-engine/tensor'
 * toArray(broyden((v) => sub(mul(v, 2), tensor([2, 4])), [0, 0]).x) // => [1, 2]
 * ```
 */
export function broyden(f: VectorFunction, x0: TensorLike, options: SystemOptions = {}): SystemResult {
  const max = options.maxIterations ?? 200
  return systemResult(runAlgorithm(broydenAlgorithm(f, x0, options), { maxSteps: max }).state, max)
}

/** Options of {@link fixedPoint}. */
export interface FixedPointOptions {
  /**
   * `del2` (default, as scipy: Steffensen's acceleration with Aitken's delta-squared), `iteration` (plain x <- g(x)),
   * or `anderson` (Anderson mixing of the last `memory` iterates).
   */
  readonly method?: 'del2' | 'iteration' | 'anderson'
  readonly xtol?: number
  readonly maxIterations?: number
  /** Anderson memory (default 5). */
  readonly memory?: number
}

/** The state of {@link fixedPointAlgorithm}: the iterate, the residual x - g(x) there and the memory of Anderson mixing. */
export interface FixedPointState {
  readonly x: readonly number[]
  /** x - g(x) at the iterate. */
  readonly residual: readonly number[]
  readonly iteration: number
  readonly functionCalls: number
  /** The norm of the residual after each iteration. */
  readonly history: readonly number[]
  /** Anderson mixing: the residuals g(x_k) - x_k and the images g(x_k) of the iterates so far. */
  readonly residuals: readonly (readonly number[])[]
  readonly images: readonly (readonly number[])[]
  readonly converged: boolean
}

/** The paths {@link fixedPointAlgorithm} logs, as a type. */
export type FixedPointPath = 'fixed-point/x' | 'fixed-point/residual' | 'fixed-point/change'

/** What {@link fixedPointAlgorithm} logs, by path: the iterate, the norm of x - g(x) and the largest relative change. */
export const fixedPointContract: Contract<FixedPointState, FixedPointPath> = defineContract({
  'fixed-point/x': tensor({ role: 'state', shape: ['n'], description: 'the iterate' }),
  'fixed-point/residual': scalar({ description: 'the norm of x - g(x)', direction: 'lower' }),
  'fixed-point/change': scalar({
    description: 'the largest relative change of a coordinate in this step',
    direction: 'lower',
  }),
})

/**
 * Solve x = g(x) (scipy.optimize.fixed_point) as a steppable algorithm, plainly or with Steffensen's (`del2`) or
 * Anderson's acceleration: one step takes one accelerated iterate.
 *
 * @param g - The map.
 * @param x0 - Starting point `[n]`.
 * @param options - Method, tolerance, iteration cap and Anderson memory.
 * @returns The algorithm; its state is a {@link FixedPointState}.
 * @concept fixed-points
 * @example
 * ```ts
 * import { fixedPointAlgorithm } from 'mdml-engine/numerics/roots'
 * import { runAlgorithm } from 'mdml-engine/trace'
 * import { cos } from 'mdml-engine/tensor'
 * Math.abs((runAlgorithm(fixedPointAlgorithm((x) => cos(x), [1]), { maxSteps: 100 }).state.x[0] ?? 0) - 0.7390851332151607) < 1e-6 // => true
 * ```
 */
export function fixedPointAlgorithm(
  g: VectorFunction,
  x0: TensorLike,
  options: FixedPointOptions = {},
): Algorithm<FixedPointState, FixedPointPath> {
  const xtol = options.xtol ?? 1e-8
  const max = options.maxIterations ?? 500
  const method = options.method ?? 'del2'
  const start = toFlat(vec(x0))
  const n = start.length
  const apply = (v: readonly number[]): number[] => toFlat(vec(g(fromStorage(Float64Array.from(v), [n], 'float64'))))
  const close = (next: readonly number[], prev: readonly number[]): boolean =>
    next.every((v, i) => {
      const p = prev[i] ?? 0
      return Math.abs(p === 0 ? v : (v - p) / p) < xtol
    })
  return {
    name: 'fixed-point',
    config: { method, xtol, maxIterations: max, memory: options.memory ?? 5 },
    init: () => ({
      x: start,
      residual: [],
      iteration: 0,
      functionCalls: 0,
      history: [],
      residuals: [],
      images: [],
      converged: false,
    }),
    step: (s, ctx) => {
      const x = s.x
      let calls = s.functionCalls
      const residuals = s.residuals.map((r) => [...r])
      const iterates = s.images.map((r) => [...r])
      let next: number[]
      if (method === 'del2') {
        const p1 = apply(x)
        const p2 = apply(p1)
        calls += 2
        next = x.map((p0, k) => {
          const d = (p2[k] ?? 0) - 2 * (p1[k] ?? 0) + p0
          return d === 0 ? (p2[k] ?? 0) : p0 - ((p1[k] ?? 0) - p0) ** 2 / d
        })
      } else if (method === 'iteration') {
        next = apply(x)
        calls += 1
      } else {
        const gx = apply(x)
        calls += 1
        const r = gx.map((v, k) => v - (x[k] ?? 0))
        residuals.push(r)
        iterates.push(gx)
        const m = Math.min(options.memory ?? 5, residuals.length - 1)
        if (m === 0) next = gx
        else {
          // Minimise |r_k - sum_j gamma_j (r_k - r_{k-j})| and mix the images accordingly.
          const rk = residuals[residuals.length - 1] as number[]
          const dr = Array.from({ length: m }, (_, j) =>
            (residuals[residuals.length - 2 - j] as number[]).map((v, k) => (rk[k] ?? 0) - v),
          )
          const a = fromStorage(
            Float64Array.from({ length: n * m }, (_, idx) => dr[idx % m]?.[Math.floor(idx / m)] ?? 0),
            [n, m],
            'float64',
          )
          const gamma = toFlat(lstsq(a, fromStorage(Float64Array.from(rk), [n], 'float64')))
          const gk = iterates[iterates.length - 1] as number[]
          next = gk.map(
            (v, k) =>
              v - gamma.reduce((sum, gj, j) => sum + gj * (v - (iterates[iterates.length - 2 - j]?.[k] ?? 0)), 0),
          )
        }
      }
      const image = apply(next)
      calls += 1
      const residual = next.map((v, k) => v - (image[k] ?? 0))
      if (ctx.wants('fixed-point/x')) ctx.log('fixed-point/x', Float64Array.from(next))
      if (ctx.wants('fixed-point/residual')) ctx.log('fixed-point/residual', norm(residual))
      if (ctx.wants('fixed-point/change'))
        ctx.log(
          'fixed-point/change',
          maxOf(next.map((v, i) => Math.abs((x[i] ?? 0) === 0 ? v : (v - (x[i] ?? 0)) / (x[i] ?? 1)))),
        )
      return {
        x: next,
        residual,
        iteration: s.iteration + 1,
        functionCalls: calls,
        history: [...s.history, norm(residual)],
        residuals,
        images: iterates,
        converged: close(next, x),
      }
    },
    done: (s): RunStatus | undefined => (s.converged ? 'converged' : s.iteration >= max ? 'stopped' : undefined),
    serialise: (s) => stateToJson(s),
    deserialise: (j) => stateFromJson(j) as FixedPointState,
  }
}

/**
 * Solve x = g(x) (scipy.optimize.fixed_point), plainly or with Steffensen's or Anderson's acceleration.
 * {@link fixedPointAlgorithm} run to convergence.
 *
 * @param g - The map.
 * @param x0 - Starting point `[n]`.
 * @param options - Method, tolerance, iteration cap and Anderson memory.
 * @returns The fixed point; the residual is x - g(x), and the history holds its norms.
 * @concept fixed-points
 * @example
 * ```ts
 * import { fixedPoint } from 'mdml-engine/numerics/roots'
 * import { cos, toArray } from 'mdml-engine/tensor'
 * Math.abs(toArray(fixedPoint((x) => cos(x), [1]).x)[0] - 0.7390851332151607) < 1e-10 // => true
 * ```
 */
export function fixedPoint(g: VectorFunction, x0: TensorLike, options: FixedPointOptions = {}): SystemResult {
  const max = options.maxIterations ?? 500
  const s = runAlgorithm(fixedPointAlgorithm(g, x0, options), { maxSteps: max }).state
  if (!s.converged) throw new MdmlError('convergence', `fixedPoint: no convergence after ${max} iterations`)
  return {
    x: asVector(s.x),
    residual: asVector(s.residual),
    iterations: s.iteration,
    functionCalls: s.functionCalls,
    converged: true,
    message: 'converged',
    history: s.history,
  }
}

/**
 * Solve F(x) = 0 with a chosen method (scipy.optimize.root's role): `newton` (damped Newton, default), `broyden`, or
 * `anderson` (Anderson-accelerated iteration on x <- x - F(x)).
 *
 * @param f - The system.
 * @param x0 - Starting point.
 * @param options - The method and its options.
 * @returns The solution.
 * @index function Solve F(x) = 0 with newton, broyden or other methods.
 * @example
 * ```ts
 * import { root } from 'mdml-engine/numerics/roots'
 * import { sub, tensor, toArray } from 'mdml-engine/tensor'
 * toArray(root((v) => sub(v, tensor([3, 4])), [0, 0]).x) // => [3, 4]
 * ```
 */
export function root(
  f: VectorFunction,
  x0: TensorLike,
  options: SystemOptions & { readonly method?: 'newton' | 'broyden' | 'anderson' } = {},
): SystemResult {
  const { method = 'newton', ...rest } = options
  if (method === 'broyden') return broyden(f, x0, rest)
  if (method === 'anderson') {
    const r = fixedPoint((x) => sub(x, f(x)), x0, {
      method: 'anderson',
      ...(rest.maxIterations === undefined ? {} : { maxIterations: rest.maxIterations }),
    })
    return { ...r, residual: vec(f(r.x)) }
  }
  return fsolve(f, x0, rest)
}

/**
 * A solution of F(x, theta) = 0 that is differentiable in theta by the implicit function theorem: solve numerically on
 * the primal values, then dx/dtheta = -F_x^{-1} F_theta (in reverse mode, theta_bar = -F_theta^T F_x^{-T} x_bar).
 *
 * @param f - The system, differentiable in both arguments.
 * @param theta - The parameters (possibly traced).
 * @param x0 - Starting point for the solver.
 * @param options - Options of the Newton solve.
 * @returns The solution as a value, differentiable in theta.
 * @concept implicit-differentiation
 * @example
 * ```ts
 * import { rootOf } from 'mdml-engine/numerics/roots'
 * import { jacobian } from 'mdml-engine/autodiff'
 * import { mul, sub, tensor, toArray } from 'mdml-engine/tensor'
 * // 2x = theta, so dx/dtheta = I / 2.
 * toArray(jacobian((t) => rootOf((x, th) => sub(mul(x, 2), th), t, [0, 0]))(tensor([1, 2]))) // => [[0.5, 0], [0, 0.5]]
 * ```
 */
export function rootOf(
  f: (x: Value, theta: Value) => Value,
  theta: Value,
  x0: TensorLike,
  options: SystemOptions = {},
): Value {
  const fixed = primalOf(theta)
  const solution = fsolve((x) => f(x, fixed), x0, options).x
  const op = customVjp({
    name: 'rootOf',
    forward: () => solution,
    vjp: (g, [th], out) => {
      const jx = jacobian((x: Value) => f(x, th as Value))(out)
      const lambda = solve(swapaxes(jx, 0, 1), g)
      const { pullback } = vjp((t: Value) => f(out, t), th as Value)
      return [neg(pullback(lambda) as Value)]
    },
    // dx = -F_x^{-1} (F_theta dtheta).
    jvp: ([th], [dth], out) => {
      if (dth === null || dth === undefined) return null
      const jx = jacobian((x: Value) => f(x, th as Value))(out)
      return neg(solve(jx, forward((t: Value) => f(out, t), th as Value, dth).tangent))
    },
  })
  return op(theta)
}
