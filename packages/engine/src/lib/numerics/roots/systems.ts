/**
 * Solving systems of nonlinear equations F(x) = 0 with Jacobians from autodiff: damped Newton with a backtracking line
 * search (`fsolve`), Broyden's quasi-Newton method, fixed-point iteration with Steffensen or Anderson acceleration, a
 * `root` front end as in scipy.optimize, and `rootOf`, a solution differentiable in its parameters by the implicit
 * function theorem.
 */

import { customVjp, jvp as forward, jacobian, vjp } from '../../autodiff/index.ts'
import { MdmlError } from '../../base/index.ts'
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

/**
 * Solve F(x) = 0 by Newton's method with a backtracking (Armijo) line search on ||F||^2, using the Jacobian from
 * autodiff (scipy.optimize.fsolve's role; scipy uses MINPACK's hybrid method).
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
  const xtol = options.xtol ?? 1.49012e-8
  const ftol = options.ftol ?? 1e-14
  const jac = options.jacobian ?? ((x: Value) => jacobian(f)(x))
  let x = vec(x0)
  let fx = vec(f(x))
  let calls = 1
  const history: number[] = [norm(toFlat(fx))]
  for (let i = 1; i <= (options.maxIterations ?? 100); i++) {
    const step = vec(solve(jac(x), neg(fx)))
    const f0 = history[history.length - 1] ?? 0
    let t = 1
    let next = vec(add(x, mul(step, t)))
    let fn = vec(f(next))
    calls += 1
    while (norm(toFlat(fn)) > (1 - 1e-4 * t) * f0 && t > 1e-10) {
      t /= 2
      next = vec(add(x, mul(step, t)))
      fn = vec(f(next))
      calls += 1
    }
    const moved = norm(toFlat(sub(next, x)))
    x = next
    fx = fn
    history.push(norm(toFlat(fx)))
    if ((history[history.length - 1] ?? 0) <= ftol || moved <= xtol * (norm(toFlat(x)) + xtol)) {
      return { x, residual: fx, iterations: i, functionCalls: calls, converged: true, message: 'converged', history }
    }
  }
  return {
    x,
    residual: fx,
    iterations: options.maxIterations ?? 100,
    functionCalls: calls,
    converged: false,
    message: 'maximum iterations',
    history,
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
  const xtol = options.xtol ?? 1.49012e-8
  const ftol = options.ftol ?? 1e-14
  let x = vec(x0)
  let fx = vec(f(x))
  let calls = 1
  let b = vec((options.jacobian ?? ((v: Value) => jacobian(f)(v)))(x))
  const history: number[] = [norm(toFlat(fx))]
  for (let i = 1; i <= (options.maxIterations ?? 200); i++) {
    const step = vec(solve(b, neg(fx)))
    const f0 = history[history.length - 1] ?? 0
    let t = 1
    let next = vec(add(x, mul(step, t)))
    let fn = vec(f(next))
    calls += 1
    while (norm(toFlat(fn)) > (1 - 1e-4 * t) * f0 && t > 1e-10) {
      t /= 2
      next = vec(add(x, mul(step, t)))
      fn = vec(f(next))
      calls += 1
    }
    const dx = toFlat(sub(next, x))
    const df = toFlat(sub(fn, fx))
    // B <- B + (df - B dx) dx^T / (dx^T dx).
    const bdx = toFlat(matmul(b, fromStorage(Float64Array.from(dx), [dx.length], 'float64')))
    const denom = dx.reduce((s, v) => s + v * v, 0)
    if (denom > 0) {
      const n = dx.length
      const update = new Float64Array(n * n)
      for (let r = 0; r < n; r++)
        for (let c = 0; c < n; c++) update[r * n + c] = (((df[r] ?? 0) - (bdx[r] ?? 0)) * (dx[c] ?? 0)) / denom
      b = vec(add(b, fromStorage(update, [n, n], 'float64')))
    }
    x = next
    fx = fn
    history.push(norm(toFlat(fx)))
    if ((history[history.length - 1] ?? 0) <= ftol || Math.sqrt(denom) <= xtol * (norm(toFlat(x)) + xtol)) {
      return { x, residual: fx, iterations: i, functionCalls: calls, converged: true, message: 'converged', history }
    }
  }
  return {
    x,
    residual: fx,
    iterations: options.maxIterations ?? 200,
    functionCalls: calls,
    converged: false,
    message: 'maximum iterations',
    history,
  }
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

/**
 * Solve x = g(x) (scipy.optimize.fixed_point), plainly or with Steffensen's or Anderson's acceleration.
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
  const xtol = options.xtol ?? 1e-8
  const maxIterations = options.maxIterations ?? 500
  const method = options.method ?? 'del2'
  let x = toFlat(vec(x0))
  const n = x.length
  const apply = (v: readonly number[]): number[] => toFlat(vec(g(fromStorage(Float64Array.from(v), [n], 'float64'))))
  let calls = 0
  const history: number[] = []
  const done = (next: number[], prev: readonly number[]): boolean =>
    next.every((v, i) => {
      const p = prev[i] ?? 0
      return Math.abs(p === 0 ? v : (v - p) / p) < xtol
    })
  const residuals: number[][] = []
  const iterates: number[][] = []
  for (let i = 1; i <= maxIterations; i++) {
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
          (v, k) => v - gamma.reduce((s, gj, j) => s + gj * (v - (iterates[iterates.length - 2 - j]?.[k] ?? 0)), 0),
        )
      }
    }
    const residual = apply(next)
    calls += 1
    history.push(norm(next.map((v, k) => v - (residual[k] ?? 0))))
    if (done(next, x)) {
      const xt = fromStorage(Float64Array.from(next), [n], 'float64')
      return {
        x: xt,
        residual: vec(sub(xt, fromStorage(Float64Array.from(residual), [n], 'float64'))),
        iterations: i,
        functionCalls: calls,
        converged: true,
        message: 'converged',
        history,
      }
    }
    x = next
  }
  throw new MdmlError('convergence', `fixedPoint: no convergence after ${maxIterations} iterations`)
}

/**
 * Solve F(x) = 0 with a chosen method (scipy.optimize.root's role): `newton` (damped Newton, default), `broyden`, or
 * `anderson` (Anderson-accelerated iteration on x <- x - F(x)).
 *
 * @param f - The system.
 * @param x0 - Starting point.
 * @param options - The method and its options.
 * @returns The solution.
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
