/**
 * Continuation methods for nonlinear equations: the Newton homotopy, which deforms an easy problem into F(x) = 0 and
 * follows the solution with Newton corrections (often reaching roots plain Newton misses), and Keller's
 * pseudo-arclength continuation, which traces a solution curve of F(x, lambda) = 0 through turning points.
 */

import { jacobian } from '../../autodiff/index.ts'
import {
  concat,
  fromStorage,
  mul,
  promoteFloat,
  reshape,
  sliceAxis,
  sub,
  sum,
  type Tensor,
  type TensorLike,
  toFlat,
  type Value,
} from '../../tensor/index.ts'
import { slogdet, solve } from '../linalg/index.ts'
import { fsolve, type SystemResult, type VectorFunction } from './systems.ts'

/** Result of {@link homotopy}. */
export interface HomotopyResult extends SystemResult {
  /** The solution at each value of t from 0 to 1. */
  readonly path: readonly { readonly t: number; readonly x: readonly number[] }[]
}

/**
 * Solve F(x) = 0 by following the Newton (global) homotopy H(x, t) = F(x) - (1 - t) F(x0) from t = 0, where x0 solves
 * it, to t = 1, correcting with Newton's method at each step: often reaches roots that plain Newton from x0 misses.
 *
 * @param f - The system.
 * @param x0 - Starting point.
 * @param options - Number of t steps (default 20).
 * @returns The solution and the path. When the corrector fails at some t the path stops there: `converged` is false,
 *   `status` is `failed`, the message names t, and `x` is the last point that solved the deformed system (the
 *   function does not throw).
 * @concept continuation
 * @example
 * ```ts
 * import { homotopy } from 'omai-engine/numerics/roots'
 * import { pow, sub, tensor, toArray } from 'omai-engine/tensor'
 * Math.abs(toArray(homotopy((x) => sub(pow(x, 3), tensor([8])), [10]).x)[0] - 2) < 1e-10 // => true
 * ```
 */
export function homotopy(f: VectorFunction, x0: TensorLike, options: { readonly steps?: number } = {}): HomotopyResult {
  const start = promoteFloat(x0)[0] as Value
  const f0 = f(start)
  const steps = options.steps ?? 20
  let x: TensorLike = start
  const path: { t: number; x: readonly number[] }[] = [{ t: 0, x: toFlat(start as Tensor) }]
  let last: SystemResult | undefined
  for (let k = 1; k <= steps; k++) {
    const t = k / steps
    const corrected = fsolve((v) => sub(f(v), mul(f0, 1 - t)), x)
    if (!corrected.converged) {
      // The corrector failed: return the path followed so far, from the last point that was a solution of H(., t).
      const at = promoteFloat(x)[0] as Tensor
      return {
        ...corrected,
        x: at,
        residual: f(at) as Tensor,
        converged: false,
        status: 'failed',
        message: `homotopy: the corrector failed at t = ${t}`,
        path,
      }
    }
    last = corrected
    x = last.x
    path.push({ t, x: toFlat(last.x) })
  }
  const final = last as SystemResult
  return { ...final, residual: f(final.x) as Tensor, path }
}

/** A point of a continuation path. */
export interface ContinuationPoint {
  readonly x: readonly number[]
  readonly lambda: number
}

/** Result of {@link arcLengthContinuation}. */
export interface ContinuationResult {
  readonly points: readonly ContinuationPoint[]
  /** Indices of points just past a fold (the curve turned back in lambda). */
  readonly folds: readonly number[]
  /** Indices of points just past a branch point (the augmented Jacobian's determinant changed sign without a fold). */
  readonly branchPoints: readonly number[]
}

/** Options of {@link arcLengthContinuation}. */
export interface ContinuationOptions {
  /** Arclength step (default 0.05). */
  readonly ds?: number
  readonly steps?: number
  /** Initial direction of travel in lambda: +1 (default) or -1. */
  readonly direction?: 1 | -1
}

/**
 * Pseudo-arclength continuation of the solution curve of F(x, lambda) = 0 (Keller): predict along the tangent, then
 * correct with Newton's method on F = 0 together with the arclength condition, so the path passes around folds where
 * lambda turns back. Folds and branch points are detected along the way.
 *
 * @param f - The system F(x, lambda) (lambda a scalar value), built from operations.
 * @param x0 - A solution at lambda0.
 * @param lambda0 - The starting parameter.
 * @param options - Step, number of steps and direction.
 * @returns The path, with fold and branch-point indices.
 * @concept continuation
 * @example
 * ```ts
 * import { arcLengthContinuation } from 'omai-engine/numerics/roots'
 * import { add, square } from 'omai-engine/tensor'
 * // x^2 + lambda = 0 folds at lambda = 0.
 * arcLengthContinuation((x, l) => add(square(x), l), [-1], -1, { ds: 0.1, steps: 30 }).folds.length // => 1
 * ```
 */
export function arcLengthContinuation(
  f: (x: Value, lambda: Value) => Value,
  x0: TensorLike,
  lambda0: number,
  options: ContinuationOptions = {},
): ContinuationResult {
  const ds = options.ds ?? 0.05
  const start = toFlat(promoteFloat(x0)[0] as Tensor)
  const n = start.length
  // y = (x, lambda) in R^{n+1}; G(y) = F(x, lambda).
  const g = (y: Value): Value => {
    const parts = toFlatParts(y, n)
    return f(parts.x, parts.lambda)
  }
  const jac = (y: readonly number[]): Tensor =>
    jacobian(g)(fromStorage(Float64Array.from(y), [n + 1], 'float64')) as Tensor
  const tangentAt = (y: readonly number[], previous: readonly number[]): number[] => {
    // Solve [J; t_prev^T] t = e_{n+1}, then normalise.
    const j = toFlat(jac(y))
    const aug = new Float64Array((n + 1) * (n + 1))
    for (let r = 0; r < n; r++) for (let c = 0; c <= n; c++) aug[r * (n + 1) + c] = j[r * (n + 1) + c] ?? 0
    for (let c = 0; c <= n; c++) aug[n * (n + 1) + c] = previous[c] ?? 0
    const rhs = new Float64Array(n + 1)
    rhs[n] = 1
    const t = toFlat(solve(fromStorage(aug, [n + 1, n + 1], 'float64'), fromStorage(rhs, [n + 1], 'float64')))
    const nt = Math.sqrt(t.reduce((s, v) => s + v * v, 0))
    return t.map((v) => v / nt)
  }
  const augmentedSign = (y: readonly number[], t: readonly number[]): number => {
    const j = toFlat(jac(y))
    const aug = new Float64Array((n + 1) * (n + 1))
    for (let r = 0; r < n; r++) for (let c = 0; c <= n; c++) aug[r * (n + 1) + c] = j[r * (n + 1) + c] ?? 0
    for (let c = 0; c <= n; c++) aug[n * (n + 1) + c] = t[c] ?? 0
    return toFlat(slogdet(fromStorage(aug, [n + 1, n + 1], 'float64')).sign)[0] ?? 0
  }
  let y = [...start, lambda0]
  let t = tangentAt(y, [...new Array<number>(n).fill(0), options.direction ?? 1])
  const points: ContinuationPoint[] = [{ x: start, lambda: lambda0 }]
  const folds: number[] = []
  const branchPoints: number[] = []
  let sign = augmentedSign(y, t)
  for (let k = 1; k <= (options.steps ?? 100); k++) {
    const predicted = y.map((v, i) => v + ds * (t[i] ?? 0))
    const tp = fromStorage(Float64Array.from(t), [n + 1], 'float64')
    const yp = fromStorage(Float64Array.from(predicted), [n + 1], 'float64')
    // Corrector: F(y) = 0 and t . (y - y_pred) = 0.
    const corrected = fsolve((v) => concat([g(v), reshapeScalar(sumProduct(tp, sub(v, yp)))]), yp)
    if (!corrected.converged) break
    const next = toFlat(corrected.x)
    const nextT = tangentAt(next, t)
    if (Math.sign(nextT[n] ?? 0) !== Math.sign(t[n] ?? 0) && (t[n] ?? 0) !== 0) folds.push(k)
    const nextSign = augmentedSign(next, nextT)
    if (nextSign !== sign && !folds.includes(k)) branchPoints.push(k)
    sign = nextSign
    y = next
    t = nextT
    points.push({ x: next.slice(0, n), lambda: next[n] ?? 0 })
  }
  return { points, folds, branchPoints }
}

function toFlatParts(y: Value, n: number): { x: Value; lambda: Value } {
  return { x: sliceAxis(y, 0, 0, n), lambda: reshape(sliceAxis(y, 0, n, n + 1), []) }
}

function sumProduct(a: Value, b: Value): Value {
  return sum(mul(a, b))
}

function reshapeScalar(v: Value): Value {
  return reshape(v, [1])
}
