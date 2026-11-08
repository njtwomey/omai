/**
 * Convex quadratic programming: the `QuadraticProgram` type (quadratic objective, linear inequalities, equalities and
 * bounds), the active-set and interior-point solvers with KKT residuals to check a solution, the `quadprog` front end,
 * and Bertsekas' projected Newton method for box-constrained problems.
 */

import { OmaiError } from '../base/index.ts'
import { ddot } from '../numerics/linalg/index.ts'
import { type Tensor, type TensorLike, toVector } from '../tensor/index.ts'
import { type Algorithm, channels, type RunOptions, type RunStatus, runAlgorithm, runStatusOf } from '../trace/index.ts'
import { rethrowUnless } from './_catch.ts'
import {
  boundVectorFrom,
  type Dense,
  expectLength,
  finiteVectorFrom,
  matrixFrom,
  matVec,
  solveSpd,
  vectorFrom,
  vectorTensor,
} from './_programming.ts'
import { qpActiveSetAlgorithm } from './qp-active-set.ts'
import { qpInteriorPointAlgorithm } from './qp-interior-point.ts'

/**
 * A convex quadratic program: minimise 1/2 x^T P x + q^T x subject to G x <= h, A x = b and lb <= x <= ub
 * (P symmetric positive semidefinite).
 */
export interface QuadraticProgram {
  readonly P: TensorLike
  readonly q: TensorLike
  readonly G?: TensorLike
  readonly h?: TensorLike
  readonly A?: TensorLike
  readonly b?: TensorLike
  /** Lower bounds (-Infinity for none). */
  readonly lb?: TensorLike
  /** Upper bounds (Infinity for none). */
  readonly ub?: TensorLike
}

/**
 * A quadratic program as row-major number arrays, the form the solvers work on (finite bounds folded into G as rows
 * -x_j <= -lb_j and x_j <= ub_j after the general rows).
 */
export interface QpArrays {
  readonly n: number
  /** n x n, row-major. */
  readonly P: Float64Array
  readonly q: Float64Array
  readonly G: Dense
  readonly h: Float64Array
  readonly A: Dense
  readonly b: Float64Array
  /** For each row of G: -1 for a general row, otherwise the variable a bound row constrains. */
  readonly boundOf?: Int32Array
  /** For each row of G: 'general', 'lower' or 'upper'. */
  readonly rowKind?: readonly ('general' | 'lower' | 'upper')[]
}

/** A solution of a quadratic program in number arrays. */
export interface QpSolution {
  readonly x: Float64Array
  readonly fun: number
  /** Multipliers of A x = b. */
  readonly y: Float64Array
  /** Multipliers of G x <= h (non-negative). */
  readonly z: Float64Array
  readonly status: 'optimal' | 'infeasible' | 'unbounded' | 'iteration limit'
  readonly iterations: number
  /** Rows of G active at the solution (the final working set of the active-set method). */
  readonly active: readonly number[]
}

/** Options of the quadratic-programming solvers. */
export interface QpOptions extends Partial<Omit<RunOptions, 'maxSteps' | 'start'>> {
  /** The method, a function value: {@link qpActiveSet} (default) or {@link qpInteriorPoint}. */
  readonly method?: (qp: QpArrays, options?: QpOptions) => QpSolution
  readonly maxIterations?: number
  /** Optimality and feasibility tolerance (default 1e-10 active set, 1e-9 interior point). */
  readonly tol?: number
  /**
   * A feasible starting point for the active-set method (otherwise found by a phase-1 linear program): a vector
   * (migration, D-009: was `ArrayLike<number>`; a tensor, a typed array or a list).
   */
  readonly x0?: TensorLike
}

/**
 * Convert a quadratic program to number arrays, folding finite bounds into G.
 *
 * @param problem - The program.
 * @returns The arrays.
 * @throws OmaiError with code `shape` when the sizes disagree.
 * @throws OmaiError with code `value` when q, h, b or a matrix is NaN or infinite (bounds may be infinite, not NaN).
 * @example
 * ```ts
 * import { qpArrays } from 'omai-engine/optim'
 * const qp = qpArrays({ P: [[1, 0], [0, 1]], q: [0, 0], lb: [1, -Infinity] })
 * [qp.G.rows, Array.from(qp.h)] // => [1, [-1]]
 * ```
 */
export function qpArrays(problem: QuadraticProgram): QpArrays {
  const q = finiteVectorFrom(problem.q, 'q')
  const n = q.length
  const P = matrixFrom(problem.P, n, 'P')
  if (P.rows !== n) throw new OmaiError('shape', `P must be ${n} x ${n}`)
  const G = matrixFrom(problem.G, n, 'G')
  const h = finiteVectorFrom(problem.h, 'h')
  expectLength(h, G.rows, 'h')
  const A = matrixFrom(problem.A, n, 'A')
  const b = finiteVectorFrom(problem.b, 'b')
  expectLength(b, A.rows, 'b')
  const lb = problem.lb === undefined ? new Float64Array(n).fill(-Infinity) : boundVectorFrom(problem.lb, 'lb')
  const ub = problem.ub === undefined ? new Float64Array(n).fill(Infinity) : boundVectorFrom(problem.ub, 'ub')
  expectLength(lb, n, 'lb')
  expectLength(ub, n, 'ub')
  const rows: number[][] = []
  const rhs: number[] = []
  const boundOf: number[] = []
  const rowKind: ('general' | 'lower' | 'upper')[] = []
  for (let i = 0; i < G.rows; i++) {
    rows.push(Array.from(G.data.subarray(i * n, (i + 1) * n)))
    rhs.push(h[i] ?? 0)
    boundOf.push(-1)
    rowKind.push('general')
  }
  for (let j = 0; j < n; j++) {
    if (Number.isFinite(lb[j] ?? 0)) {
      const row = new Array<number>(n).fill(0)
      row[j] = -1
      rows.push(row)
      rhs.push(-(lb[j] ?? 0))
      boundOf.push(j)
      rowKind.push('lower')
    }
    if (Number.isFinite(ub[j] ?? 0)) {
      const row = new Array<number>(n).fill(0)
      row[j] = 1
      rows.push(row)
      rhs.push(ub[j] ?? 0)
      boundOf.push(j)
      rowKind.push('upper')
    }
  }
  return {
    n,
    P: Float64Array.from(P.data),
    q,
    G: { data: Float64Array.from(rows.flat()), rows: rows.length, cols: n },
    h: Float64Array.from(rhs),
    A,
    b,
    boundOf: Int32Array.from(boundOf),
    rowKind,
  }
}

/** Residuals of the Karush-Kuhn-Tucker conditions at a primal-dual point (infinity norms). */
export interface KktResiduals {
  /** |P x + q + A^T y + G^T z|. */
  readonly stationarity: number
  /** |A x - b|. */
  readonly primalEquality: number
  /** max(0, max(G x - h)). */
  readonly primalInequality: number
  /** max(0, -min z). */
  readonly dualFeasibility: number
  /** max |z_i (h - G x)_i|. */
  readonly complementarity: number
}

/**
 * The KKT residuals of a quadratic program at (x, y, z): all zero exactly at an optimal primal-dual pair.
 *
 * @param qp - The program (from {@link qpArrays}).
 * @param x - Primal point.
 * @param y - Equality multipliers.
 * @param z - Inequality multipliers.
 * @returns The residuals.
 * @example
 * ```ts
 * import { kktResiduals, qpArrays } from 'omai-engine/optim'
 * const qp = qpArrays({ P: [[2]], q: [-2], G: [[1]], h: [0.5] })
 * kktResiduals(qp, [0.5], [], [1]).stationarity // => 0
 * ```
 */
export function kktResiduals(
  qp: QpArrays,
  x: ArrayLike<number>,
  y: ArrayLike<number>,
  z: ArrayLike<number>,
): KktResiduals {
  const grad = matVec({ data: qp.P, rows: qp.n, cols: qp.n }, x)
  const aty = matVec(qp.A, y, true)
  const gtz = matVec(qp.G, z, true)
  let stationarity = 0
  for (let j = 0; j < qp.n; j++)
    stationarity = Math.max(stationarity, Math.abs((grad[j] ?? 0) + (qp.q[j] ?? 0) + (aty[j] ?? 0) + (gtz[j] ?? 0)))
  const ax = matVec(qp.A, x)
  let primalEquality = 0
  for (let i = 0; i < qp.A.rows; i++) primalEquality = Math.max(primalEquality, Math.abs((ax[i] ?? 0) - (qp.b[i] ?? 0)))
  const gx = matVec(qp.G, x)
  let primalInequality = 0
  let dualFeasibility = 0
  let complementarity = 0
  for (let i = 0; i < qp.G.rows; i++) {
    const slack = (qp.h[i] ?? 0) - (gx[i] ?? 0)
    primalInequality = Math.max(primalInequality, -slack)
    dualFeasibility = Math.max(dualFeasibility, -(z[i] ?? 0))
    complementarity = Math.max(complementarity, Math.abs((z[i] ?? 0) * slack))
  }
  return { stationarity, primalEquality, primalInequality, dualFeasibility, complementarity }
}

/** Run a QP algorithm whose state carries a `solution`. */
function solveWith<S extends { readonly solution: QpSolution }>(
  algorithm: Algorithm<S>,
  options: QpOptions,
  defaultIterations: number,
): QpSolution {
  const { method: _m, tol: _t, x0: _x, maxIterations, ...run } = options
  const r = runAlgorithm(algorithm, { ...run, maxSteps: maxIterations ?? defaultIterations })
  return r.state.solution
}

/**
 * Solve a convex quadratic program in number arrays by the primal active-set method (Nocedal and Wright, algorithm
 * 16.3): exact, finite, with the final working set as the active constraints.
 *
 * @param qp - The program (from {@link qpArrays}, or built directly).
 * @param options - Tolerance, feasible start, iteration cap and run options.
 * @returns The solution.
 * @concept quadratic-programming
 * @example
 * ```ts
 * import { qpActiveSet, qpArrays } from 'omai-engine/optim'
 * const qp = qpArrays({ P: [[2, 0], [0, 2]], q: [-2, -5], G: [[1, 2], [-1, 2], [-1, -2]], h: [6, 2, 2], lb: [0, 0] })
 * Array.from(qpActiveSet(qp).x) // => [1.4, 1.7]
 * ```
 */
export function qpActiveSet(qp: QpArrays, options: QpOptions = {}): QpSolution {
  return solveWith(qpActiveSetAlgorithm(qp, options), options, 50 * (qp.n + qp.G.rows + 1))
}

/**
 * Solve a convex quadratic program in number arrays by a primal-dual interior-point method (Mehrotra's
 * predictor-corrector on the KKT system reduced to x and y).
 *
 * @param qp - The program.
 * @param options - Tolerance, iteration cap and run options.
 * @returns The solution (active rows: multipliers above the tolerance).
 * @concept quadratic-programming
 * @example
 * ```ts
 * import { qpArrays, qpInteriorPoint } from 'omai-engine/optim'
 * const qp = qpArrays({ P: [[2, 0], [0, 2]], q: [-2, -5], G: [[1, 2], [-1, 2], [-1, -2]], h: [6, 2, 2], lb: [0, 0] })
 * Array.from(qpInteriorPoint(qp).x, (v) => Math.round(v * 1e6) / 1e6) // => [1.4, 1.7]
 * ```
 */
export function qpInteriorPoint(qp: QpArrays, options: QpOptions = {}): QpSolution {
  return solveWith(qpInteriorPointAlgorithm(qp, options), options, 100)
}

/** Result of {@link quadprog}. */
export interface QuadprogResult {
  readonly x: Tensor
  readonly fun: number
  /** Multipliers of A x = b. */
  readonly y: Tensor
  /** Multipliers of G x <= h. */
  readonly z: Tensor
  /** Multipliers of the lower and upper bounds (zero where a bound is infinite). */
  readonly zLower: Tensor
  readonly zUpper: Tensor
  readonly status: QpSolution['status']
  readonly success: boolean
  /** `success` as a synonym of `converged` in the one vocabulary shared with the vector methods. */
  readonly converged: boolean
  /** The run in the one vocabulary (`optimal` is `converged`, a limit is `completed`, the rest `failed`): `runStatusOf(status)`. */
  readonly runStatus: RunStatus
  readonly iterations: number
  /** Rows of G active at the solution. */
  readonly active: readonly number[]
  readonly kkt: KktResiduals
}

/**
 * Solve a convex quadratic program: minimise 1/2 x^T P x + q^T x subject to G x <= h, A x = b and bounds. The method
 * is a function value on number arrays: {@link qpActiveSet} (default) or {@link qpInteriorPoint}.
 *
 * @param problem - The program.
 * @param options - Method, tolerance, start, iteration cap and run options.
 * @returns The solution, its multipliers split by constraint type, the active rows and the KKT residuals.
 * @throws OmaiError with code `value` when q, h, b or a matrix is NaN or infinite (bounds may be infinite, not NaN).
 * @concept quadratic-programming
 * @index function Convex quadratic programming by active set or interior point.
 * @example
 * ```ts
 * import { quadprog } from 'omai-engine/optim'
 * import { toArray } from 'omai-engine/tensor'
 * const r = quadprog({ P: [[1, 0], [0, 1]], q: [-1, -1], A: [[1, 1]], b: [1] })
 * toArray(r.x) // => [0.5, 0.5]
 * ```
 */
export function quadprog(problem: QuadraticProgram, options: QpOptions = {}): QuadprogResult {
  const qp = qpArrays(problem)
  const sol = (options.method ?? qpActiveSet)(qp, options)
  const general = problem.h === undefined ? 0 : vectorFrom(problem.h).length
  const zLower = new Float64Array(qp.n)
  const zUpper = new Float64Array(qp.n)
  for (let i = general; i < qp.G.rows; i++) {
    const j = qp.boundOf?.[i] ?? -1
    if (j < 0) continue
    if (qp.rowKind?.[i] === 'lower') zLower[j] = sol.z[i] ?? 0
    else zUpper[j] = sol.z[i] ?? 0
  }
  return {
    x: vectorTensor(sol.x),
    fun: sol.fun,
    y: vectorTensor(sol.y),
    z: vectorTensor(sol.z.subarray(0, general)),
    zLower: vectorTensor(zLower),
    zUpper: vectorTensor(zUpper),
    status: sol.status,
    success: sol.status === 'optimal',
    converged: sol.status === 'optimal',
    runStatus: runStatusOf(sol.status),
    iterations: sol.iterations,
    active: sol.active,
    kkt: kktResiduals(qp, sol.x, sol.y, sol.z),
  }
}

/** The state of {@link boxQpAlgorithm}. */
export interface BoxQpState {
  readonly x: Float64Array
  readonly value: number
  readonly grad: Float64Array
  /** Infinity norm of x - proj(x - grad): zero exactly at a solution. */
  readonly projectedGradient: number
  /** Variables held at a bound in the last step. */
  readonly active: readonly number[]
  readonly step: number
  readonly evaluations: number
}

const project = (x: ArrayLike<number>, lb: Float64Array, ub: Float64Array): Float64Array =>
  Float64Array.from(x, (v, j) => Math.min(ub[j] ?? Infinity, Math.max(lb[j] ?? -Infinity, v)))

/**
 * Bertsekas' projected Newton method for box-constrained quadratic programs (min 1/2 x^T P x + q^T x subject to
 * lb <= x <= ub), as a traced algorithm: each step fixes the variables at a bound whose gradient pushes outward (the
 * epsilon-active set), takes a Newton step on the free variables (Cholesky of the free block of P, with a gradient
 * step if it is not positive definite), and backtracks along the projection arc with an Armijo test. Records the
 * objective, the projected-gradient norm, the step and the number of active bounds.
 *
 * @param P - Symmetric positive semidefinite n x n matrix.
 * @param q - Linear term.
 * @param lb - Lower bounds (-Infinity allowed).
 * @param ub - Upper bounds (Infinity allowed).
 * @param options - Tolerance (default 1e-10) and starting point `x0`, a vector (default the projection of 0; takes a
 *   tensor-like, migration, D-009: was `ArrayLike<number>`).
 * @returns The algorithm.
 * @throws OmaiError with code `value` when q or P is NaN or infinite (bounds may be infinite, not NaN); or when lb exceeds ub;
 *   `shape` unless `x0` has rank 1.
 * @concept quadratic-programming
 * @reference D. P. Bertsekas, Projected Newton methods for optimization problems with simple constraints (1982).
 * @example
 * ```ts
 * import { boxQpAlgorithm } from 'omai-engine/optim'
 * import { runAlgorithm } from 'omai-engine/trace'
 * const alg = boxQpAlgorithm([[2, 0], [0, 2]], [-4, 2], [0, 0], [1, 1])
 * Array.from(runAlgorithm(alg, { maxSteps: 50 }).state.x) // => [1, 0]
 * ```
 */
export function boxQpAlgorithm(
  P: TensorLike,
  q: TensorLike,
  lb: TensorLike,
  ub: TensorLike,
  options: { readonly tol?: number; readonly x0?: TensorLike } = {},
): Algorithm<BoxQpState> {
  const x0 = options.x0 === undefined ? undefined : toVector(options.x0, 'boxQp')
  const qv = finiteVectorFrom(q, 'q')
  const n = qv.length
  const pm = matrixFrom(P, n, 'P')
  const lo = boundVectorFrom(lb, 'lb')
  const hi = boundVectorFrom(ub, 'ub')
  expectLength(lo, n, 'lb')
  expectLength(hi, n, 'ub')
  const conflict = lo.findIndex((l, j) => l > (hi[j] ?? Infinity))
  if (conflict >= 0)
    throw new OmaiError('value', `boxQp: lb[${conflict}] = ${lo[conflict]} exceeds ub[${conflict}] = ${hi[conflict]}`)
  const tol = options.tol ?? 1e-10
  const f = (x: Float64Array): { value: number; grad: Float64Array } => {
    const px = matVec(pm, x)
    return { value: 0.5 * ddot(x, px) + ddot(qv, x), grad: Float64Array.from(px, (v, j) => v + (qv[j] ?? 0)) }
  }
  const measure = (x: Float64Array, g: Float64Array): number => {
    const p = project(
      Float64Array.from(x, (v, j) => v - (g[j] ?? 0)),
      lo,
      hi,
    )
    let m = 0
    for (let j = 0; j < n; j++) m = Math.max(m, Math.abs((x[j] ?? 0) - (p[j] ?? 0)))
    return m
  }
  const stateAt = (x: Float64Array, active: readonly number[], step: number, evaluations: number): BoxQpState => {
    const { value, grad } = f(x)
    return { x, value, grad, projectedGradient: measure(x, grad), active, step, evaluations }
  }
  return {
    name: 'boxQp',
    config: { tol },
    schema: channels({
      metrics: {
        loss: { description: 'objective value', direction: 'lower' },
        projectedGradient: { description: 'infinity norm of x - proj(x - grad)' },
        step: { description: 'accepted step along the projection arc' },
        active: { description: 'variables held at a bound' },
      },
      tensors: { x: { role: 'param', description: 'current iterate' } },
    }),
    init: () => stateAt(project(x0 ?? new Float64Array(n), lo, hi), [], 0, 1),
    step: (s) => {
      const eps = Math.min(1e-3, s.projectedGradient)
      const fixed = new Uint8Array(n)
      for (let j = 0; j < n; j++) {
        const xj = s.x[j] ?? 0
        const gj = s.grad[j] ?? 0
        if ((xj <= (lo[j] ?? -Infinity) + eps && gj > 0) || (xj >= (hi[j] ?? Infinity) - eps && gj < 0)) fixed[j] = 1
      }
      const free = Array.from({ length: n }, (_, j) => j).filter((j) => fixed[j] === 0)
      const d = Float64Array.from(s.grad, (g) => -g)
      if (free.length > 0) {
        const k = free.length
        const sub = new Float64Array(k * k)
        for (const [a, i] of free.entries())
          for (const [b, j] of free.entries()) sub[a * k + b] = pm.data[i * n + j] ?? 0
        try {
          const newton = solveSpd(
            sub,
            Float64Array.from(free, (j) => -(s.grad[j] ?? 0)),
            k,
          )
          for (const [a, j] of free.entries()) d[j] = newton[a] ?? 0
        } catch (error) {
          // Not positive definite on the free block: keep the gradient step.
          rethrowUnless(error, 'value', 'singular')
        }
      }
      let alpha = 1
      let evaluations = s.evaluations
      for (let trial = 0; trial < 60; trial++) {
        const xt = project(
          Float64Array.from(s.x, (v, j) => v + alpha * (d[j] ?? 0)),
          lo,
          hi,
        )
        const ft = f(xt)
        evaluations += 1
        let decrease = 0
        for (let j = 0; j < n; j++) decrease += (s.grad[j] ?? 0) * ((xt[j] ?? 0) - (s.x[j] ?? 0))
        if (ft.value <= s.value + 1e-4 * decrease || trial === 59) {
          const active = Array.from({ length: n }, (_, j) => j).filter(
            (j) => (xt[j] ?? 0) <= (lo[j] ?? -Infinity) || (xt[j] ?? 0) >= (hi[j] ?? Infinity),
          )
          return {
            x: xt,
            value: ft.value,
            grad: ft.grad,
            projectedGradient: measure(xt, ft.grad),
            active,
            step: alpha,
            evaluations,
          }
        }
        alpha /= 2
      }
      return s
    },
    done: (s) => (s.projectedGradient <= tol ? 'converged' : Number.isFinite(s.value) ? undefined : 'diverged'),
    observe: (s) => ({
      metrics: { loss: s.value, projectedGradient: s.projectedGradient, step: s.step, active: s.active.length },
      tensors: { x: { role: 'param', value: s.x } },
    }),
  }
}

/** Result of {@link boxQp}. */
export interface BoxQpResult {
  readonly x: Tensor
  readonly fun: number
  /** Multipliers of the lower and upper bounds (from the gradient at the solution). */
  readonly zLower: Tensor
  readonly zUpper: Tensor
  readonly iterations: number
  readonly converged: boolean
  /** How the run ended in the one vocabulary: `converged`, or `completed` when the iteration cap was reached. */
  readonly status: RunStatus
}

/**
 * Solve a box-constrained quadratic program by the projected Newton method.
 *
 * @param P - Symmetric positive semidefinite matrix.
 * @param q - Linear term.
 * @param lb - Lower bounds.
 * @param ub - Upper bounds.
 * @param options - Tolerance, start `x0` (a tensor-like vector; migration, D-009: was `ArrayLike<number>`), iteration cap
 *   (default 100 + 10 n) and run options.
 * @returns The solution and its bound multipliers.
 * @throws OmaiError with code `value` when q or P is NaN or infinite (bounds may be infinite, not NaN); or when lb exceeds ub.
 * @concept quadratic-programming
 * @example
 * ```ts
 * import { boxQp } from 'omai-engine/optim'
 * import { toArray } from 'omai-engine/tensor'
 * toArray(boxQp([[2, 1], [1, 2]], [-6, 0], [-Infinity, 0], [2, Infinity]).x) // => [2, 0]
 * ```
 */
export function boxQp(
  P: TensorLike,
  q: TensorLike,
  lb: TensorLike,
  ub: TensorLike,
  options: { readonly tol?: number; readonly x0?: TensorLike; readonly maxIterations?: number } & Partial<
    Omit<RunOptions, 'maxSteps' | 'start'>
  > = {},
): BoxQpResult {
  const { tol, x0, maxIterations, ...run } = options
  const n = vectorFrom(q).length
  const r = runAlgorithm(
    boxQpAlgorithm(P, q, lb, ub, { ...(tol === undefined ? {} : { tol }), ...(x0 === undefined ? {} : { x0 }) }),
    { ...run, maxSteps: maxIterations ?? 100 + 10 * n },
  )
  const s = r.state
  const lo = vectorFrom(lb)
  const hi = vectorFrom(ub)
  return {
    x: vectorTensor(s.x),
    fun: s.value,
    zLower: vectorTensor(
      Float64Array.from(s.x, (v, j) => (v <= (lo[j] ?? -Infinity) ? Math.max(0, s.grad[j] ?? 0) : 0)),
    ),
    zUpper: vectorTensor(
      Float64Array.from(s.x, (v, j) => (v >= (hi[j] ?? Infinity) ? Math.max(0, -(s.grad[j] ?? 0)) : 0)),
    ),
    iterations: r.steps,
    converged: r.status === 'converged',
    status: r.status,
  }
}
