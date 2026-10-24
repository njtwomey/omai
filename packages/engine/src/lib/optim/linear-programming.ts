import { ddot } from '../numerics/linalg/index.ts'
import { asTensor, type Tensor, toArray, toFlat } from '../tensor/index.ts'
import { type Algorithm, runAlgorithm } from '../trace/index.ts'
import { basisMatrix } from './_lp.ts'
import { matVec, solveDense, vectorTensor } from './_programming.ts'
import { lpInteriorPointAlgorithm } from './lp-interior-point.ts'
import { simplexAlgorithm } from './lp-simplex.ts'
import {
  fromStandard,
  type LinearProgram,
  type LinprogOptions,
  type LpPathPoint,
  type StandardForm,
  type StandardSolution,
  standardForm,
} from './lp-standard-form.ts'

/** Sensitivities of the optimal value (scipy's `marginals`): d fun / d of each right-hand side and bound. */
export interface Marginals {
  /** For each inequality row (non-positive). */
  readonly ineqlin: Tensor
  /** For each equality row. */
  readonly eqlin: Tensor
  /** For each lower bound (non-negative). */
  readonly lower: Tensor
  /** For each upper bound (non-positive). */
  readonly upper: Tensor
}

/** Allowable ranges (each `[lower, upper]` of the parameter's value) over which the optimal basis stays optimal. */
export interface Sensitivity {
  /** Range of each cost coefficient c_j keeping the basis optimal. */
  readonly cost: readonly (readonly [number, number])[]
  /** Range of each b_ub entry keeping the basis feasible. */
  readonly bUb: readonly (readonly [number, number])[]
  /** Range of each b_eq entry keeping the basis feasible. */
  readonly bEq: readonly (readonly [number, number])[]
}

/** Result of {@link linprog} (scipy's OptimizeResult for linprog). */
export interface LinprogResult {
  readonly x: Tensor
  readonly fun: number
  /** b_ub - A_ub x (non-negative when feasible). */
  readonly slack: Tensor
  /** b_eq - A_eq x (zero when feasible). */
  readonly con: Tensor
  readonly status: 'optimal' | 'infeasible' | 'unbounded' | 'iteration limit' | 'numerical difficulties'
  readonly success: boolean
  readonly message: string
  readonly iterations: number
  readonly marginals: Marginals
  /** Ranging at the optimal basis (simplex only). */
  readonly sensitivity?: Sensitivity
  /** The basis of the standard form at the end (simplex only). */
  readonly basis?: readonly number[]
  /** The vertices visited (simplex) or the iterates (interior point). */
  readonly path: readonly LpPathPoint[]
}

/** The marginals and the result in original terms from a standard-form solution. */
function assembleResult(form: StandardForm, sol: StandardSolution): LinprogResult {
  const { original: o } = form
  const x = fromStandard(form, sol.z)
  const fun = ddot(o.c, x)
  const ax = matVec(o.aUb, x)
  const slack = Float64Array.from(o.bUb, (v, i) => v - (ax[i] ?? 0))
  const ex = matVec(o.aEq, x)
  const con = Float64Array.from(o.bEq, (v, i) => v - (ex[i] ?? 0))
  const aty = matVec(form.a, sol.y, true)
  const d = sol.reducedCosts ?? Float64Array.from(form.c, (cj, k) => cj - (aty[k] ?? 0))
  const ineqlin = new Float64Array(o.aUb.rows)
  const eqlin = new Float64Array(o.aEq.rows)
  const lower = new Float64Array(form.n)
  const upper = new Float64Array(form.n)
  for (const [r, row] of form.rows.entries()) {
    const v = (sol.y[r] ?? 0) * row.sign
    if (row.kind === 'ub') ineqlin[row.index] = v
    else if (row.kind === 'eq') eqlin[row.index] = v
    else upper[row.index] = v
  }
  for (const [k, col] of form.columns.entries()) {
    if (col.kind === 'shift') lower[col.index] = d[k] ?? 0
    else if (col.kind === 'reflect') upper[col.index] = -(d[k] ?? 0)
  }
  const ok = sol.status === 'optimal'
  const messages: Record<LinprogResult['status'], string> = {
    optimal: 'optimization terminated successfully',
    infeasible: 'the problem is infeasible',
    unbounded: 'the problem is unbounded',
    'iteration limit': 'iteration limit reached',
    'numerical difficulties': 'numerical difficulties: the tableau would not settle after repeated re-inversion',
  }
  return {
    x: vectorTensor(x),
    fun: ok ? fun : sol.status === 'unbounded' ? -Infinity : Number.NaN,
    slack: vectorTensor(slack),
    con: vectorTensor(con),
    status: sol.status,
    success: ok,
    message: messages[sol.status],
    iterations: sol.iterations,
    marginals: {
      ineqlin: vectorTensor(ineqlin),
      eqlin: vectorTensor(eqlin),
      lower: vectorTensor(lower),
      upper: vectorTensor(upper),
    },
    ...(ok &&
    sol.tableau !== undefined &&
    sol.basis !== undefined &&
    sol.rowIds !== undefined &&
    sol.reducedCosts !== undefined
      ? { sensitivity: ranging(form, sol.basis, sol.rowIds, sol.tableau, sol.reducedCosts) }
      : {}),
    ...(sol.basis === undefined ? {} : { basis: Array.from(sol.basis) }),
    path: sol.path,
  }
}

function ranging(
  form: StandardForm,
  basis: Int32Array,
  rowIds: Int32Array,
  tableau: Float64Array,
  d: Float64Array,
): Sensitivity {
  const m = basis.length
  const N = form.a.cols
  const width = N + 1
  const isBasic = new Uint8Array(N)
  for (const k of basis) isBasic[k] = 1
  // Cost ranging: the cost of x_j moves every column that represents it; reduced costs must stay non-negative.
  const cost: [number, number][] = []
  for (let j = 0; j < form.n; j++) {
    const w = new Float64Array(N)
    for (const [k, col] of form.columns.entries()) {
      if (col.kind === 'slack' || col.index !== j) continue
      w[k] = col.kind === 'reflect' || col.kind === 'minus' ? -1 : 1
    }
    let lo = -Infinity
    let hi = Infinity
    for (let k = 0; k < N; k++) {
      if (isBasic[k] === 1) continue
      let g = w[k] ?? 0
      for (let i = 0; i < m; i++) g -= (w[basis[i] ?? 0] ?? 0) * (tableau[i * width + k] ?? 0)
      if (Math.abs(g) < 1e-12) continue
      const t = -(d[k] ?? 0) / g
      if (g > 0) lo = Math.max(lo, t)
      else hi = Math.min(hi, t)
    }
    const cj = form.original.c[j] ?? 0
    cost.push([cj + lo, cj + hi])
  }
  // Right-hand-side ranging: x_B(delta) = x_B + delta sign B^{-1} e_r must stay non-negative.
  const bmat = basisMatrix(form, basis, rowIds)
  const xb = Float64Array.from({ length: m }, (_, i) => tableau[i * width + N] ?? 0)
  const position = new Map<number, number>()
  for (const [i, r] of rowIds.entries()) position.set(r, i)
  const rhsRange = (r: number, value: number, sign: number): [number, number] => {
    const at = position.get(r)
    if (at === undefined) return [Number.NaN, Number.NaN]
    const e = new Float64Array(m)
    e[at] = sign
    const beta = solveDense(bmat, e, m)
    let lo = -Infinity
    let hi = Infinity
    for (let i = 0; i < m; i++) {
      const bi = beta[i] ?? 0
      if (Math.abs(bi) < 1e-12) continue
      const t = -(xb[i] ?? 0) / bi
      if (bi > 0) lo = Math.max(lo, t)
      else hi = Math.min(hi, t)
    }
    return [value + lo, value + hi]
  }
  const bUb: [number, number][] = []
  const bEq: [number, number][] = []
  for (const [r, row] of form.rows.entries()) {
    if (row.kind === 'ub') bUb[row.index] = rhsRange(r, form.original.bUb[row.index] ?? 0, row.sign)
    else if (row.kind === 'eq') bEq[row.index] = rhsRange(r, form.original.bEq[row.index] ?? 0, row.sign)
  }
  return { cost, bUb, bEq }
}

/** Rows of a constraint matrix (empty when absent). */
function rowsOf(a: LinearProgram['aUb']): number[][] {
  if (a === undefined) return []
  const v = toArray(asTensor(a)) as number[] | number[][]
  return v.length === 0 ? [] : Array.isArray(v[0]) ? (v as number[][]) : [v as number[]]
}

/**
 * Solve with every constraint row equilibrated to a largest coefficient of 1 (the solution is unchanged), then report
 * slacks and residuals against the original rows and rescale the row duals and ranges. Rows of 1e-6-sized coefficients
 * beside rows of order 1 (near-tied payoffs) otherwise fall under the solvers' tolerances: the simplex then reports
 * feasible programs infeasible and the interior point reports bounded ones unbounded.
 */
function equilibrated(problem: LinearProgram, solve: (p: LinearProgram) => LinprogResult): LinprogResult {
  const aUb = rowsOf(problem.aUb)
  const aEq = rowsOf(problem.aEq)
  const scaleOf = (row: readonly number[]): number => {
    let largest = 0
    for (const v of row) largest = Math.max(largest, Math.abs(v))
    return largest > 0 ? 1 / largest : 1
  }
  const sUb = aUb.map(scaleOf)
  const sEq = aEq.map(scaleOf)
  if (sUb.every((v) => v === 1) && sEq.every((v) => v === 1)) return solve(problem)
  const bUb = problem.bUb === undefined ? [] : toFlat(asTensor(problem.bUb))
  const bEq = problem.bEq === undefined ? [] : toFlat(asTensor(problem.bEq))
  const scaled: LinearProgram = {
    ...problem,
    ...(aUb.length === 0
      ? {}
      : { aUb: aUb.map((row, r) => row.map((v) => v * (sUb[r] ?? 1))), bUb: bUb.map((v, r) => v * (sUb[r] ?? 1)) }),
    ...(aEq.length === 0
      ? {}
      : { aEq: aEq.map((row, r) => row.map((v) => v * (sEq[r] ?? 1))), bEq: bEq.map((v, r) => v * (sEq[r] ?? 1)) }),
  }
  const r = solve(scaled)
  const x = toFlat(r.x)
  const residual = (rows: number[][], b: number[]): Tensor =>
    vectorTensor(Float64Array.from(rows, (row, i) => (b[i] ?? 0) - ddot(row, x)))
  // fun depends on b_r' = s_r b_r, so d fun / d b_r = s_r d fun / d b_r'; a range of b_r' is a range of b_r times 1 / s_r.
  const rescale = (t: Tensor, s: readonly number[]): Tensor =>
    vectorTensor(Float64Array.from(toFlat(t), (v, i) => v * (s[i] ?? 1)))
  const ranges = (rs: readonly (readonly [number, number])[], s: readonly number[]): [number, number][] =>
    rs.map(([lo, hi], i) => [lo / (s[i] ?? 1), hi / (s[i] ?? 1)])
  return {
    ...r,
    slack: aUb.length === 0 ? r.slack : residual(aUb, bUb),
    con: aEq.length === 0 ? r.con : residual(aEq, bEq),
    marginals: { ...r.marginals, ineqlin: rescale(r.marginals.ineqlin, sUb), eqlin: rescale(r.marginals.eqlin, sEq) },
    ...(r.sensitivity === undefined
      ? {}
      : {
          sensitivity: {
            ...r.sensitivity,
            bUb: ranges(r.sensitivity.bUb, sUb),
            bEq: ranges(r.sensitivity.bEq, sEq),
          },
        }),
  }
}

function run<S extends { readonly solution: StandardSolution; readonly finished: boolean }>(
  algorithm: Algorithm<S>,
  form: StandardForm,
  options: LinprogOptions,
  defaultIterations: number,
): LinprogResult {
  const { method: _m, maxIterations, tol: _t, pivotRule: _p, ...rest } = options
  const result = runAlgorithm(algorithm, { ...rest, maxSteps: maxIterations ?? defaultIterations })
  return assembleResult(form, result.state.solution)
}

/**
 * Solve a linear program by the two-phase simplex method on the standard form (Bland's rule by default, so it never
 * cycles), with duals, sensitivity ranges and the path of vertices.
 *
 * @param problem - The program.
 * @param options - Pivot rule, tolerance, iteration cap and run options.
 * @returns The result.
 * @concept linear-programming
 * @example
 * ```ts
 * import { linprogSimplex } from 'whitebox-engine/optim'
 * const r = linprogSimplex({ c: [-1, -2], aUb: [[1, 1], [1, 3]], bUb: [4, 6] })
 * [r.status, r.fun] // => ['optimal', -5]
 * ```
 */
export function linprogSimplex(problem: LinearProgram, options: LinprogOptions = {}): LinprogResult {
  return equilibrated(problem, (p) => {
    const form = standardForm(p)
    return run(simplexAlgorithm(form, options), form, options, 50 * (form.a.rows + form.a.cols + 1))
  })
}

/**
 * Solve a linear program by a primal-dual interior-point method: Mehrotra's predictor-corrector on the homogeneous
 * self-dual embedding (as scipy's legacy `interior-point`), which detects infeasible and unbounded programs.
 *
 * @param problem - The program.
 * @param options - Tolerance, iteration cap and run options.
 * @returns The result (no basis or ranging).
 * @concept linear-programming
 * @example
 * ```ts
 * import { linprogInteriorPoint } from 'whitebox-engine/optim'
 * const r = linprogInteriorPoint({ c: [-1, -2], aUb: [[1, 1], [1, 3]], bUb: [4, 6] })
 * Math.round(r.fun * 1e6) / 1e6 // => -5
 * ```
 */
export function linprogInteriorPoint(problem: LinearProgram, options: LinprogOptions = {}): LinprogResult {
  return equilibrated(problem, (p) => {
    const form = standardForm(p)
    return run(lpInteriorPointAlgorithm(form, options), form, options, 200)
  })
}

/**
 * Solve a linear program in scipy's form (scipy.optimize.linprog): minimise c^T x subject to A_ub x <= b_ub,
 * A_eq x = b_eq and bounds. The method is a function value; by default {@link linprogSimplex},
 * falling back to {@link linprogInteriorPoint} when the simplex reports numerical difficulties, and
 * returning an infeasible or unbounded verdict only when two methods (the simplex with Bland's and Dantzig's rules,
 * the interior point) agree; otherwise the status is `numerical difficulties`.
 *
 * @param problem - The program.
 * @param options - The method and its options.
 * @returns x, the optimal value, slacks, residuals, status, marginals (duals), and for the simplex the basis,
 * sensitivity ranges and visited vertices.
 * @concept linear-programming
 * @example
 * ```ts
 * import { linprog } from 'whitebox-engine/optim'
 * import { toArray } from 'whitebox-engine/tensor'
 * const r = linprog({ c: [-1, -2], aUb: [[1, 1], [1, 3]], bUb: [4, 6] })
 * toArray(r.x) // => [3, 1]
 * toArray(r.marginals.ineqlin) // => [-0.5, -0.5]
 * ```
 */
export function linprog(problem: LinearProgram, options: LinprogOptions = {}): LinprogResult {
  if (options.method !== undefined) return options.method(problem, options)
  const r = linprogSimplex(problem, options)
  if (r.status === 'optimal' || r.status === 'iteration limit') return r
  // Programs the simplex cannot settle numerically (near-tied coefficients) go to the interior-point method, and
  // infeasible or unbounded verdicts are cross-checked: one is returned only when two methods agree.
  const ip = linprogInteriorPoint(problem, options)
  if (ip.status === 'optimal' || (ip.status === r.status && r.status !== 'numerical difficulties')) return ip
  const dantzig = linprogSimplex(problem, { ...options, pivotRule: 'dantzig' })
  if (dantzig.status === 'optimal' || (dantzig.status === ip.status && ip.status !== 'iteration limit')) return dantzig
  return {
    ...ip,
    status: 'numerical difficulties',
    success: false,
    message: `numerical difficulties: the simplex (${r.status}, then ${dantzig.status}) and the interior point (${ip.status}) disagree`,
  }
}
