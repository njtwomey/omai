/**
 * Linear programs and their standard form: the `LinearProgram` type in scipy's form, conversion to standard form
 * (equalities with non-negative variables, by shifting, reflecting and splitting variables and adding slacks) and back,
 * and the shared option, path and solution types of the linear-programming methods.
 */

import { MdmlError } from '../base/index.ts'
import type { TensorLike } from '../tensor/index.ts'
import type { RunOptions } from '../trace/index.ts'
import { type Dense, expectLength, finiteVectorFrom, matrixFrom } from './_programming.ts'
import type { LinprogResult } from './linear-programming.ts'

/** Bounds of one variable: `[lower, upper]`, with `null` (or an infinity) for none. */
export type Bound = readonly [number | null, number | null]

/**
 * A linear program in scipy's form: minimise c^T x subject to A_ub x <= b_ub, A_eq x = b_eq and bounds on each
 * variable (default `[0, null]`, that is x >= 0).
 */
export interface LinearProgram {
  readonly c: TensorLike
  readonly aUb?: TensorLike
  readonly bUb?: TensorLike
  readonly aEq?: TensorLike
  readonly bEq?: TensorLike
  /** One bound for every variable, or one per variable. */
  readonly bounds?: Bound | readonly Bound[]
}

/** How an original variable appears in the standard form. */
type ColumnKind = 'shift' | 'reflect' | 'plus' | 'minus' | 'slack'

/**
 * A linear program in standard form: minimise c^T z + offset subject to A z = b, z >= 0, b >= 0, with the maps back to
 * the original variables and constraints.
 */
export interface StandardForm {
  /** Original variables. */
  readonly n: number
  /** Standard-form columns and rows. */
  readonly a: Dense
  readonly b: Float64Array
  readonly c: Float64Array
  readonly offset: number
  /** For each column: its kind, the original variable (or row for a slack) and its sign in x. */
  readonly columns: readonly { readonly kind: ColumnKind; readonly index: number }[]
  /** For each row: its kind, the original constraint index (or variable for a bound row) and the sign it was multiplied by. */
  readonly rows: readonly { readonly kind: 'ub' | 'eq' | 'bound'; readonly index: number; readonly sign: number }[]
  /** The column of each row's slack (-1 for equality rows). */
  readonly slackOf: Int32Array
  /** The parsed original problem. */
  readonly original: {
    readonly c: Float64Array
    readonly aUb: Dense
    readonly bUb: Float64Array
    readonly aEq: Dense
    readonly bEq: Float64Array
    readonly lower: Float64Array
    readonly upper: Float64Array
  }
}

function boundsOf(bounds: LinearProgram['bounds'], n: number): { lower: Float64Array; upper: Float64Array } {
  const lower = new Float64Array(n)
  const upper = new Float64Array(n).fill(Infinity)
  if (bounds === undefined) return { lower, upper }
  const single = bounds.length === 2 && !Array.isArray(bounds[0]) && !Array.isArray(bounds[1])
  for (let j = 0; j < n; j++) {
    const bj = (single ? bounds : (bounds as readonly Bound[])[j]) as Bound | undefined
    if (bj === undefined) throw new MdmlError('shape', `linprog: bounds must have one entry per variable (${n})`)
    lower[j] = bj[0] ?? -Infinity
    upper[j] = bj[1] ?? Infinity
    if (Number.isNaN(lower[j]) || Number.isNaN(upper[j])) throw new MdmlError('value', `linprog: bounds[${j}] is NaN`)
  }
  return { lower, upper }
}

/**
 * Convert a linear program to standard form: shift variables with a finite lower bound, reflect those with only an
 * upper bound, split free ones into a difference, add a row for every finite upper bound of a shifted variable, a slack
 * for every inequality, and flip rows to make the right-hand side non-negative.
 *
 * @param problem - The program.
 * @returns The standard form with its maps back.
 * @throws MdmlError with code `shape` when the sizes disagree.
 * @throws MdmlError with code `value` when c, bUb, bEq or a matrix is NaN or infinite (bounds may be infinite, not NaN).
 * @example
 * ```ts
 * import { standardForm } from 'mdml-engine/optim'
 * const s = standardForm({ c: [1, 1], aUb: [[1, 2]], bUb: [4], bounds: [[0, 3], [null, null]] })
 * [s.a.rows, s.a.cols] // => [2, 5]
 * ```
 */
export function standardForm(problem: LinearProgram): StandardForm {
  const c = finiteVectorFrom(problem.c, 'c')
  const n = c.length
  const aUb = matrixFrom(problem.aUb, n, 'aUb')
  const bUb = finiteVectorFrom(problem.bUb, 'bUb')
  const aEq = matrixFrom(problem.aEq, n, 'aEq')
  const bEq = finiteVectorFrom(problem.bEq, 'bEq')
  expectLength(bUb, aUb.rows, 'bUb')
  expectLength(bEq, aEq.rows, 'bEq')
  const { lower, upper } = boundsOf(problem.bounds, n)
  const columns: { kind: ColumnKind; index: number }[] = []
  const varCols: number[][] = []
  for (let j = 0; j < n; j++) {
    const l = lower[j] ?? 0
    const u = upper[j] ?? 0
    if (Number.isFinite(l)) {
      varCols.push([columns.length])
      columns.push({ kind: 'shift', index: j })
    } else if (Number.isFinite(u)) {
      varCols.push([columns.length])
      columns.push({ kind: 'reflect', index: j })
    } else {
      varCols.push([columns.length, columns.length + 1])
      columns.push({ kind: 'plus', index: j }, { kind: 'minus', index: j })
    }
  }
  const boundRows: number[] = []
  for (let j = 0; j < n; j++) if (Number.isFinite(lower[j] ?? 0) && Number.isFinite(upper[j] ?? 0)) boundRows.push(j)
  const m = aUb.rows + aEq.rows + boundRows.length
  const rowSpecs: { kind: 'ub' | 'eq' | 'bound'; index: number; sign: number }[] = []
  const slackOf = new Int32Array(m).fill(-1)
  for (let i = 0; i < aUb.rows; i++) {
    slackOf[rowSpecs.length] = columns.length
    columns.push({ kind: 'slack', index: rowSpecs.length })
    rowSpecs.push({ kind: 'ub', index: i, sign: 1 })
  }
  for (let i = 0; i < aEq.rows; i++) rowSpecs.push({ kind: 'eq', index: i, sign: 1 })
  for (const j of boundRows) {
    slackOf[rowSpecs.length] = columns.length
    columns.push({ kind: 'slack', index: rowSpecs.length })
    rowSpecs.push({ kind: 'bound', index: j, sign: 1 })
  }
  const N = columns.length
  const a = new Float64Array(m * N)
  const b = new Float64Array(m)
  const sc = new Float64Array(N)
  let offset = 0
  for (let j = 0; j < n; j++) {
    const cols = varCols[j] ?? []
    const kind = columns[cols[0] ?? 0]?.kind
    const cj = c[j] ?? 0
    if (kind === 'shift') {
      sc[cols[0] ?? 0] = cj
      offset += cj * (lower[j] ?? 0)
    } else if (kind === 'reflect') {
      sc[cols[0] ?? 0] = -cj
      offset += cj * (upper[j] ?? 0)
    } else {
      sc[cols[0] ?? 0] = cj
      sc[cols[1] ?? 0] = -cj
    }
  }
  // Each original row: sum_j A_ij x_j with x_j replaced by its columns.
  const fill = (r: number, coefficients: ArrayLike<number>, rhs: number): void => {
    let shifted = rhs
    for (let j = 0; j < n; j++) {
      const v = coefficients[j] ?? 0
      if (v === 0) continue
      const cols = varCols[j] ?? []
      const kind = columns[cols[0] ?? 0]?.kind
      if (kind === 'shift') {
        a[r * N + (cols[0] ?? 0)] = v
        shifted -= v * (lower[j] ?? 0)
      } else if (kind === 'reflect') {
        a[r * N + (cols[0] ?? 0)] = -v
        shifted -= v * (upper[j] ?? 0)
      } else {
        a[r * N + (cols[0] ?? 0)] = v
        a[r * N + (cols[1] ?? 0)] = -v
      }
    }
    b[r] = shifted
  }
  for (const [r, spec] of rowSpecs.entries()) {
    if (spec.kind === 'ub') fill(r, aUb.data.subarray(spec.index * n, (spec.index + 1) * n), bUb[spec.index] ?? 0)
    else if (spec.kind === 'eq') fill(r, aEq.data.subarray(spec.index * n, (spec.index + 1) * n), bEq[spec.index] ?? 0)
    else {
      a[r * N + (varCols[spec.index]?.[0] ?? 0)] = 1
      b[r] = (upper[spec.index] ?? 0) - (lower[spec.index] ?? 0)
    }
    const s = slackOf[r] ?? -1
    if (s >= 0) a[r * N + s] = 1
    if ((b[r] ?? 0) < 0) {
      spec.sign = -1
      b[r] = -(b[r] ?? 0)
      for (let k = 0; k < N; k++) a[r * N + k] = -(a[r * N + k] ?? 0)
    }
  }
  return {
    n,
    a: { data: a, rows: m, cols: N },
    b,
    c: sc,
    offset,
    columns,
    rows: rowSpecs,
    slackOf,
    original: { c, aUb, bUb, aEq, bEq, lower, upper },
  }
}

/**
 * The original variables from a standard-form point.
 *
 * @param form - The standard form.
 * @param z - A standard-form point.
 * @returns x.
 * @example
 * ```ts
 * import { fromStandard, standardForm } from 'mdml-engine/optim'
 * const s = standardForm({ c: [1], bounds: [[2, null]] })
 * Array.from(fromStandard(s, [3])) // => [5]
 * ```
 */
export function fromStandard(form: StandardForm, z: ArrayLike<number>): Float64Array {
  const x = new Float64Array(form.n)
  for (const [k, col] of form.columns.entries()) {
    const v = z[k] ?? 0
    const j = col.index
    if (col.kind === 'shift') x[j] = (form.original.lower[j] ?? 0) + v
    else if (col.kind === 'reflect') x[j] = (form.original.upper[j] ?? 0) - v
    else if (col.kind === 'plus') x[j] = (x[j] ?? 0) + v
    else if (col.kind === 'minus') x[j] = (x[j] ?? 0) - v
  }
  return x
}

/** One vertex (or iterate) of a linear-programming run, for figures. */
export interface LpPathPoint {
  /** 1 or 2 for the simplex phases, 0 for interior-point iterates. */
  readonly phase: number
  /** The point in the original variables. */
  readonly x: readonly number[]
  readonly objective: number
}

/** The final solution of a standard-form solve, from either method. */
export interface StandardSolution {
  readonly status: LinprogResult['status']
  readonly z: Float64Array
  /** Row duals (d objective / d b of the standard rows). */
  readonly y: Float64Array
  readonly iterations: number
  readonly path: readonly LpPathPoint[]
  readonly basis?: Int32Array
  /** The standard row of each basis row (redundant rows dropped). */
  readonly rowIds?: Int32Array
  /** B^{-1} [A b] at the optimal basis (rows x (N + 1)) and the reduced costs, for ranging. */
  readonly tableau?: Float64Array
  readonly reducedCosts?: Float64Array
}

/** Options of the linear-programming methods. */
export interface LinprogOptions extends Partial<Omit<RunOptions, 'maxSteps' | 'start'>> {
  /** The method, a function value: {@link linprogSimplex} (default) or {@link linprogInteriorPoint}. */
  readonly method?: (problem: LinearProgram, options?: LinprogOptions) => LinprogResult
  readonly maxIterations?: number
  /** Feasibility and optimality tolerance (default 1e-9 simplex, 1e-8 interior point). */
  readonly tol?: number
  /**
   * Simplex pivoting: `'dantzig'` (default), the most negative reduced cost, with Bland's rule taking over after a run
   * of degenerate pivots so it cannot cycle; or `'bland'`, Bland's rule throughout (never cycles, but takes many more
   * pivots).
   */
  readonly pivotRule?: 'bland' | 'dantzig'
  /**
   * The simplex's smallest acceptable pivot, relative to the largest entry of its column (default 1e-7; 0 keeps only
   * the absolute tolerance). `linprog` retries a program that fails with the floor without it.
   */
  readonly pivotFloor?: number
}
