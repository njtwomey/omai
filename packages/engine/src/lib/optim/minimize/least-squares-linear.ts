/**
 * Linear least squares with constraints on the variables: non-negative least squares by the Lawson-Hanson active-set
 * method (`scipy.optimize.nnls`) and bounded-variable least squares by Stark and Parker's BVLS
 * (`scipy.optimize.lsq_linear`), both as traced algorithms.
 */

import { MdmlError } from '../../base/index.ts'
import { daxpy, ddot, dnrm2 } from '../../numerics/linalg/index.ts'
import { asTensor, fromStorage, type Tensor, type TensorLike, toFlat } from '../../tensor/index.ts'
import { type Algorithm, channels, type RunOptions, runAlgorithm } from '../../trace/index.ts'
import { jtv, jv, lstsqSvd } from './_least-squares.ts'
import { type Bounds, boxOf } from './objective.ts'

/** Result of {@link nnls} and {@link lsqLinear}. */
export interface LinearLeastSquaresResult {
  readonly x: Tensor
  /** Half the sum of squared residuals. */
  readonly cost: number
  /** The residual norm |A x - b|. */
  readonly residualNorm: number
  /** A x - b. */
  readonly residuals: Tensor
  /** The first-order optimality measure (KKT violation of the bounds). */
  readonly optimality: number
  /** -1 at a lower bound, 1 at an upper bound, 0 free. */
  readonly activeMask: readonly number[]
  readonly iterations: number
  readonly converged: boolean
  readonly message: string
}

/** Options of the linear least-squares solvers. */
export interface LinearLeastSquaresOptions extends Partial<Omit<RunOptions, 'maxSteps' | 'start'>> {
  readonly maxIterations?: number
  /** Optimality tolerance (default 1e-10 for lsqLinear; for nnls, 10 eps max(m, n) |A|_1). */
  readonly tol?: number
}

/** The state of an active-set linear least-squares run. */
export interface ActiveSetState {
  readonly x: Float64Array
  /** -1 / 1 at a bound, 0 free (nnls: -1 for variables held at zero). */
  readonly onBound: Int8Array
  readonly cost: number
  readonly optimality: number
  readonly done: boolean
  readonly moves: number
}

function matrix(a: TensorLike, b: TensorLike): { A: Float64Array; bv: Float64Array; m: number; n: number } {
  const At = asTensor(a)
  if (At.shape.length !== 2) throw new MdmlError('shape', 'A must be a matrix')
  const [m, n] = At.shape as [number, number]
  const bv = Float64Array.from(toFlat(asTensor(b)))
  if (bv.length !== m) throw new MdmlError('shape', `b has ${bv.length} entries for ${m} rows`)
  return { A: Float64Array.from(toFlat(At)), bv, m, n }
}

/** The least-squares solution over the columns in `cols`, with the other variables fixed at x. */
function subsolve(
  A: Float64Array,
  b: Float64Array,
  m: number,
  n: number,
  x: Float64Array,
  cols: readonly number[],
): Float64Array {
  const k = cols.length
  const Af = new Float64Array(m * k)
  const rhs = Float64Array.from(b)
  const inCols = new Uint8Array(n)
  for (const j of cols) inCols[j] = 1
  for (let i = 0; i < m; i++) {
    for (const [c, j] of cols.entries()) Af[i * k + c] = A[i * n + j] ?? 0
    for (let j = 0; j < n; j++) if (inCols[j] === 0) rhs[i] = (rhs[i] ?? 0) - (A[i * n + j] ?? 0) * (x[j] ?? 0)
  }
  return lstsqSvd(Af, m, k, rhs)
}

const SCHEMA = channels({
  metrics: {
    cost: { description: 'half the sum of squared residuals', direction: 'lower' },
    optimality: { description: 'KKT violation' },
    free: { description: 'free (passive) variables' },
  },
  tensors: { x: { role: 'param', description: 'current solution' } },
})

const observe = (s: ActiveSetState) => ({
  metrics: { cost: s.cost, optimality: s.optimality, free: s.onBound.reduce((c, v) => c + (v === 0 ? 1 : 0), 0) },
  tensors: { x: { role: 'param' as const, value: s.x } },
})

/**
 * The Lawson-Hanson active-set method for non-negative least squares as a traced algorithm: each step frees the
 * variable with the largest positive dual (A'(b - Ax)), solves the least-squares problem on the free set, and backs
 * off along the segment to the old point whenever that would make a free variable negative.
 *
 * @param a - The matrix A (m x n).
 * @param b - The right-hand side.
 * @param options - Tolerance.
 * @returns The algorithm.
 * @concept least-squares
 * @reference C. L. Lawson and R. J. Hanson, Solving Least Squares Problems, Prentice-Hall (1974), chapter 23.
 * @example
 * ```ts
 * import { nnlsAlgorithm } from 'mdml-engine/optim'
 * nnlsAlgorithm([[1, 0], [0, 1]], [1, -1]).name // => 'nnls'
 * ```
 */
export function nnlsAlgorithm(
  a: TensorLike,
  b: TensorLike,
  options: LinearLeastSquaresOptions = {},
): Algorithm<ActiveSetState> {
  const { A, bv, m, n } = matrix(a, b)
  let anorm = 0
  for (let j = 0; j < n; j++) {
    let s = 0
    for (let i = 0; i < m; i++) s += Math.abs(A[i * n + j] ?? 0)
    anorm = Math.max(anorm, s)
  }
  const tol = options.tol ?? 10 * Number.EPSILON * Math.max(m, n) * anorm
  const dual = (x: Float64Array): Float64Array => {
    const r = daxpy(-1, jv(A, m, n, x), bv)
    return jtv(A, m, n, r)
  }
  const costOf = (x: Float64Array): number => {
    const r = daxpy(-1, bv, jv(A, m, n, x))
    return 0.5 * ddot(r, r)
  }
  const optimality = (w: Float64Array, onBound: Int8Array): number => {
    let mx = 0
    for (let j = 0; j < n; j++) mx = Math.max(mx, onBound[j] === 0 ? 0 : (w[j] ?? 0))
    return mx
  }
  return {
    name: 'nnls',
    config: { tol },
    schema: SCHEMA,
    init: () => {
      const x = new Float64Array(n)
      const onBound = new Int8Array(n).fill(-1)
      const w = dual(x)
      const opt = optimality(w, onBound)
      return { x, onBound, cost: costOf(x), optimality: opt, done: opt <= tol || n === 0, moves: 0 }
    },
    step: (s) => {
      const x = Float64Array.from(s.x)
      const onBound = Int8Array.from(s.onBound)
      const w = dual(x)
      let t = -1
      let best = tol
      for (let j = 0; j < n; j++)
        if (onBound[j] !== 0 && (w[j] ?? 0) > best) {
          best = w[j] ?? 0
          t = j
        }
      if (t < 0) return { ...s, done: true }
      onBound[t] = 0
      let moves = s.moves
      for (let inner = 0; inner < 3 * n + 3; inner++) {
        const P = Array.from({ length: n }, (_, j) => j).filter((j) => onBound[j] === 0)
        // Variables held at zero contribute nothing, so solve over the passive set alone.
        const z = subsolve(A, bv, m, n, new Float64Array(n), P)
        const sv = new Float64Array(n)
        for (const [c, j] of P.entries()) sv[j] = z[c] ?? 0
        if (P.every((j) => (sv[j] ?? 0) > 0)) {
          x.set(sv)
          break
        }
        let alpha = Infinity
        for (const j of P) {
          const sj = sv[j] ?? 0
          if (sj <= 0) {
            const xj = x[j] ?? 0
            alpha = Math.min(alpha, xj / (xj - sj))
          }
        }
        for (const j of P) x[j] = (x[j] ?? 0) + alpha * ((sv[j] ?? 0) - (x[j] ?? 0))
        for (const j of P) {
          if ((x[j] ?? 0) <= tol * 1e-3 || ((sv[j] ?? 0) <= 0 && Math.abs(x[j] ?? 0) <= Number.EPSILON * 16)) {
            x[j] = 0
            onBound[j] = -1
            moves += 1
          }
        }
      }
      const opt = optimality(dual(x), onBound)
      return {
        x,
        onBound,
        cost: costOf(x),
        optimality: opt,
        done: opt <= tol || onBound.every((v) => v === 0),
        moves: moves + 1,
      }
    },
    done: (s) => (s.done ? 'converged' : undefined),
    observe,
  }
}

function finish(
  alg: Algorithm<ActiveSetState>,
  options: LinearLeastSquaresOptions,
  steps: number,
  A: Float64Array,
  bv: Float64Array,
  m: number,
  n: number,
): LinearLeastSquaresResult {
  const { maxIterations, tol: _t, ...run } = options
  const r = runAlgorithm(alg, { ...run, maxSteps: maxIterations ?? steps })
  const s = r.state
  const res = daxpy(-1, bv, jv(A, m, n, s.x))
  return {
    x: fromStorage(Float64Array.from(s.x), [n], 'float64'),
    cost: s.cost,
    residualNorm: dnrm2(res),
    residuals: fromStorage(res, [m], 'float64'),
    optimality: s.optimality,
    activeMask: Array.from(s.onBound),
    iterations: r.steps,
    converged: r.status === 'converged',
    message:
      r.status === 'converged' ? 'the KKT conditions hold to the tolerance' : 'maximum number of iterations reached',
  }
}

/**
 * Solve min |A x - b| subject to x >= 0 (scipy.optimize.nnls) by the Lawson-Hanson active-set method.
 *
 * @param a - The matrix A (m x n).
 * @param b - The right-hand side.
 * @param options - Tolerance, iteration cap (default 3 n) and run options.
 * @returns The solution, residual norm and active mask.
 * @concept least-squares
 * @example
 * ```ts
 * import { nnls } from 'mdml-engine/optim'
 * import { toArray } from 'mdml-engine/tensor'
 * const r = nnls([[1, 0], [1, 0], [0, 1]], [2, 1, -1])
 * [toArray(r.x), r.residualNorm] // => [[1.5, 0], 1.224744871391589]
 * ```
 */
export function nnls(a: TensorLike, b: TensorLike, options: LinearLeastSquaresOptions = {}): LinearLeastSquaresResult {
  const { A, bv, m, n } = matrix(a, b)
  return finish(nnlsAlgorithm(a, b, options), options, 3 * n + 1, A, bv, m, n)
}

/**
 * The bounded-variable least-squares method of Stark and Parker (BVLS) as a traced algorithm, a port of scipy's
 * lsq_linear with method 'bvls': start from the unbounded solution clipped to the box, then repeatedly free the
 * variable that most violates the KKT conditions and solve on the free set, stepping back to the first bound crossed.
 *
 * @param a - The matrix A (m x n).
 * @param b - The right-hand side.
 * @param bounds - Bounds on x.
 * @param options - Tolerance (default 1e-10) on optimality and on the relative cost change.
 * @returns The algorithm.
 * @concept least-squares
 * @reference P. B. Stark and R. L. Parker, Bounded-variable least-squares: an algorithm and applications, Computational
 * Statistics 10 (1995).
 * @example
 * ```ts
 * import { bvlsAlgorithm } from 'mdml-engine/optim'
 * bvlsAlgorithm([[1, 0], [0, 1]], [3, -3], { lower: -1, upper: 1 }).name // => 'bvls'
 * ```
 */
export function bvlsAlgorithm(
  a: TensorLike,
  b: TensorLike,
  bounds: Bounds,
  options: LinearLeastSquaresOptions = {},
): Algorithm<ActiveSetState> {
  const { A, bv, m, n } = matrix(a, b)
  const box = boxOf(bounds, n)
  const tol = options.tol ?? 1e-10
  const lb = box.lower
  const ub = box.upper
  const residual = (x: Float64Array): Float64Array => daxpy(-1, bv, jv(A, m, n, x))
  const kkt = (g: Float64Array, onBound: Int8Array): number => {
    let mx = -Infinity
    for (let j = 0; j < n; j++)
      mx = Math.max(mx, onBound[j] === 0 ? Math.abs(g[j] ?? 0) : (g[j] ?? 0) * (onBound[j] ?? 0))
    return n === 0 ? 0 : mx
  }
  const solveFree = (x: Float64Array, onBound: Int8Array): { z: Float64Array; free: number[] } => {
    const free = Array.from({ length: n }, (_, j) => j).filter((j) => onBound[j] === 0)
    return { z: subsolve(A, bv, m, n, x, free), free }
  }
  return {
    name: 'bvls',
    config: { tol },
    schema: SCHEMA,
    init: () => {
      // The unbounded solution, clipped, then the initial loop that fixes violated variables.
      const x = lstsqSvd(A, m, n, bv)
      const onBound = new Int8Array(n)
      for (let j = 0; j < n; j++) {
        if ((x[j] ?? 0) <= (lb[j] ?? -Infinity)) {
          x[j] = lb[j] ?? 0
          onBound[j] = -1
        } else if ((x[j] ?? 0) >= (ub[j] ?? Infinity)) {
          x[j] = ub[j] ?? 0
          onBound[j] = 1
        }
      }
      let free = Array.from({ length: n }, (_, j) => j).filter((j) => onBound[j] === 0)
      while (free.length > 0) {
        const z = subsolve(A, bv, m, n, x, free)
        const violated: boolean[] = []
        for (const [c, j] of free.entries()) {
          const zc = z[c] ?? 0
          if (zc < (lb[j] ?? -Infinity)) {
            x[j] = lb[j] ?? 0
            onBound[j] = -1
            violated.push(true)
          } else if (zc > (ub[j] ?? Infinity)) {
            x[j] = ub[j] ?? 0
            onBound[j] = 1
            violated.push(true)
          } else {
            x[j] = zc
            violated.push(false)
          }
        }
        if (!violated.some(Boolean)) break
        free = free.filter((_, c) => violated[c] !== true)
      }
      const r = residual(x)
      const g = jtv(A, m, n, r)
      const opt = kkt(g, onBound)
      return { x, onBound, cost: 0.5 * ddot(r, r), optimality: opt, done: opt < tol, moves: 0 }
    },
    step: (s) => {
      const x = Float64Array.from(s.x)
      const onBound = Int8Array.from(s.onBound)
      const g0 = jtv(A, m, n, residual(x))
      // Free the variable with the largest g * onBound (loop A).
      let move = 0
      let best = -Infinity
      for (let j = 0; j < n; j++) {
        const v = (g0[j] ?? 0) * (onBound[j] ?? 0)
        if (v > best) {
          best = v
          move = j
        }
      }
      onBound[move] = 0
      // Loop B.
      for (let guard = 0; guard < 10 * n + 10; guard++) {
        const { z, free } = solveFree(x, onBound)
        const xf = Float64Array.from(free, (j) => x[j] ?? 0)
        const lbv: number[] = []
        const ubv: number[] = []
        for (const [c, j] of free.entries()) {
          if ((z[c] ?? 0) < (lb[j] ?? -Infinity)) lbv.push(c)
          if ((z[c] ?? 0) > (ub[j] ?? Infinity)) ubv.push(c)
        }
        const v = [...lbv, ...ubv]
        if (v.length === 0) {
          for (const [c, j] of free.entries()) x[j] = z[c] ?? 0
          break
        }
        let alpha = Infinity
        let iBest = 0
        for (const [k, c] of v.entries()) {
          const j = free[c] as number
          const target = k < lbv.length ? (lb[j] ?? 0) : (ub[j] ?? 0)
          const al = (target - (xf[c] ?? 0)) / ((z[c] ?? 0) - (xf[c] ?? 0))
          if (al < alpha) {
            alpha = al
            iBest = k
          }
        }
        for (const [c, j] of free.entries()) x[j] = (xf[c] ?? 0) * (1 - alpha) + alpha * (z[c] ?? 0)
        const jb = free[v[iBest] as number] as number
        onBound[jb] = iBest < lbv.length ? -1 : 1
      }
      const r = residual(x)
      const cost = 0.5 * ddot(r, r)
      const g = jtv(A, m, n, r)
      const opt = kkt(g, onBound)
      const change = s.cost - cost
      return { x, onBound, cost, optimality: opt, done: opt < tol || change < tol * cost, moves: s.moves + 1 }
    },
    done: (s) => (s.done ? 'converged' : undefined),
    observe,
  }
}

/** Options of {@link lsqLinear}. */
export interface LsqLinearOptions extends LinearLeastSquaresOptions {
  /** Bounds on x (default none). */
  readonly bounds?: Bounds
}

/**
 * Solve a linear least-squares problem with bounds on the variables (scipy.optimize.lsq_linear) by BVLS; without bounds
 * the unbounded minimum-norm solution is returned directly.
 *
 * @param a - The matrix A (m x n).
 * @param b - The right-hand side.
 * @param options - Bounds, tolerance (default 1e-10), iteration cap (default n) and run options.
 * @returns The solution with its cost, optimality and active mask.
 * @concept least-squares
 * @example
 * ```ts
 * import { lsqLinear } from 'mdml-engine/optim'
 * import { toArray } from 'mdml-engine/tensor'
 * const r = lsqLinear([[1, 0], [0, 1], [1, 1]], [2, 2, 2], { bounds: { lower: 0, upper: 0.5 } })
 * [toArray(r.x), r.activeMask] // => [[0.5, 0.5], [1, 1]]
 * ```
 */
export function lsqLinear(a: TensorLike, b: TensorLike, options: LsqLinearOptions = {}): LinearLeastSquaresResult {
  const { A, bv, m, n } = matrix(a, b)
  const { bounds, ...rest } = options
  return finish(bvlsAlgorithm(a, b, bounds ?? {}, rest), rest, n + 1, A, bv, m, n)
}
