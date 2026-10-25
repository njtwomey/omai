/**
 * Linear algebra the way it is taught, with every step recorded: Gaussian and Gauss-Jordan elimination to row echelon
 * and reduced row echelon form, solving linear systems, LU factorisation, determinants by cofactors and by elimination,
 * the four fundamental subspaces, projections and least squares, pictures of linear maps of the plane and space,
 * characteristic polynomials, small eigenproblems and diagonalisation, quadratic forms, and the geometry of the SVD.
 */

import { maxOf, WhiteboxError } from '../base/index.ts'
import { eigh, eigvals, eigvalsh, lstsq, nullSpace, pinv, svd } from '../numerics/linalg/index.ts'
import {
  arrow,
  axes3d,
  type Item3D,
  line,
  polygon,
  type Scene,
  type Scene3D,
  type SceneNode,
  scene,
  scene3d,
  segment3d,
  type Vec3,
} from '../scene/index.ts'
import { matmul, mul, promoteFloat, sum, type TensorLike, toArray, toFlat, type Value } from '../tensor/index.ts'

type Matrix = number[][]

function matrix(a: TensorLike, name: string): Matrix {
  const [v] = promoteFloat(a) as [Value]
  if (v.shape.length !== 2) throw new WhiteboxError('shape', `${name}: expected a matrix [m, n]`)
  return toArray(v) as Matrix
}

function square(a: TensorLike, name: string): Matrix {
  const m = matrix(a, name)
  if (m.some((row) => row.length !== m.length)) throw new WhiteboxError('shape', `${name}: expected a square matrix`)
  return m
}

const copy = (m: Matrix): Matrix => m.map((r) => r.map((x) => (Object.is(x, -0) ? 0 : x)))
const scaleOf = (m: Matrix): number => maxOf(m.flat().map(Math.abs))

/** An elementary row operation. */
export type RowOperation =
  | { readonly kind: 'swap'; readonly rows: readonly [number, number] }
  | { readonly kind: 'scale'; readonly row: number; readonly factor: number }
  /** `R_target <- R_target + factor R_source`. */
  | { readonly kind: 'add'; readonly target: number; readonly source: number; readonly factor: number }

/** One step of an elimination: the operation and the matrix after it. */
export interface EliminationStep {
  readonly operation: RowOperation
  readonly matrix: number[][]
  /** The operation in words, such as `R2 <- R2 - 3 R1` (rows numbered from 1). */
  readonly description: string
}

/** Options of the elimination procedures. */
export interface EliminationOptions {
  /** 'partial' (default) picks the largest pivot in the column, as LAPACK; 'first' the first nonzero, as by hand. */
  readonly pivoting?: 'partial' | 'first'
  /** Entries below `tol * max |a_ij|` count as zero (default `1e-12 * max(m, n)`). */
  readonly tol?: number
}

const fmt = (x: number): string => String(Math.round(x * 1e6) / 1e6)

function describe(op: RowOperation): string {
  if (op.kind === 'swap') return `R${op.rows[0] + 1} <-> R${op.rows[1] + 1}`
  if (op.kind === 'scale') return `R${op.row + 1} <- ${fmt(op.factor)} R${op.row + 1}`
  const sign = op.factor < 0 ? '-' : '+'
  return `R${op.target + 1} <- R${op.target + 1} ${sign} ${fmt(Math.abs(op.factor))} R${op.source + 1}`
}

function apply(m: Matrix, op: RowOperation): void {
  if (op.kind === 'swap') {
    const [i, j] = op.rows
    const t = m[i] as number[]
    m[i] = m[j] as number[]
    m[j] = t
  } else if (op.kind === 'scale') {
    m[op.row] = (m[op.row] ?? []).map((x) => x * op.factor)
  } else {
    const src = m[op.source] ?? []
    m[op.target] = (m[op.target] ?? []).map((x, j) => x + op.factor * (src[j] ?? 0))
  }
}

/** Run forward elimination (and, for the reduced form, back substitution) recording every step. */
function eliminate(
  a: Matrix,
  reduced: boolean,
  options: EliminationOptions,
  columns?: number,
): { m: Matrix; pivots: [number, number][]; steps: EliminationStep[]; swaps: number } {
  const m = copy(a)
  const rows = m.length
  const cols = columns ?? m[0]?.length ?? 0
  const tol = (options.tol ?? 1e-12 * Math.max(rows, cols, 1)) * Math.max(scaleOf(a), 1e-300)
  const steps: EliminationStep[] = []
  const pivots: [number, number][] = []
  let swaps = 0
  const record = (op: RowOperation): void => {
    apply(m, op)
    // Clean what elimination made (numerically) zero.
    for (const row of m) for (let j = 0; j < row.length; j++) if (Math.abs(row[j] ?? 0) <= tol) row[j] = 0
    steps.push({ operation: op, matrix: copy(m), description: describe(op) })
  }
  let r = 0
  for (let c = 0; c < cols && r < rows; c++) {
    let p = -1
    for (let i = r; i < rows; i++) {
      const v = Math.abs(m[i]?.[c] ?? 0)
      if (v <= tol) continue
      if (p < 0 || (options.pivoting !== 'first' && v > Math.abs(m[p]?.[c] ?? 0))) p = i
      if (options.pivoting === 'first') break
    }
    if (p < 0) continue
    if (p !== r) {
      record({ kind: 'swap', rows: [r, p] })
      swaps++
    }
    if (reduced) {
      const pivot = m[r]?.[c] ?? 1
      if (pivot !== 1) record({ kind: 'scale', row: r, factor: 1 / pivot })
    }
    for (let i = reduced ? 0 : r + 1; i < rows; i++) {
      if (i === r) continue
      const v = m[i]?.[c] ?? 0
      if (v === 0) continue
      record({ kind: 'add', target: i, source: r, factor: -v / (m[r]?.[c] ?? 1) })
    }
    pivots.push([r, c])
    r++
  }
  return { m, pivots, steps, swaps }
}

/** Result of {@link rowEchelon} and {@link rref}. */
export interface Elimination {
  readonly matrix: number[][]
  /** Pivot positions `[row, column]`. */
  readonly pivots: [number, number][]
  readonly pivotColumns: number[]
  readonly rank: number
  readonly steps: readonly EliminationStep[]
}

/**
 * Gaussian elimination to row echelon form, recording every row operation: below each pivot the column is cleared
 * by adding multiples of the pivot row (with row swaps to bring up a nonzero, by default the largest, pivot).
 *
 * @param a - A matrix `[m, n]` (augment it with a right-hand side to solve a system).
 * @param options - Pivoting rule and zero tolerance.
 * @returns The echelon form, pivots, rank and the steps.
 * @concept gaussian-elimination
 * @example
 * ```ts
 * import { rowEchelon } from 'whitebox-engine/foundations'
 * const e = rowEchelon([[1, 2], [3, 4]], { pivoting: 'first' })
 * e.matrix // => [[1, 2], [0, -2]]
 * e.steps[0].description // => 'R2 <- R2 - 3 R1'
 * ```
 */
export function rowEchelon(a: TensorLike, options: EliminationOptions = {}): Elimination {
  const { m, pivots, steps } = eliminate(matrix(a, 'rowEchelon'), false, options)
  return { matrix: m, pivots, pivotColumns: pivots.map((p) => p[1]), rank: pivots.length, steps }
}

/**
 * The reduced row echelon form (sympy's `rref`) by Gauss-Jordan elimination with every step: each pivot scaled to 1
 * and its column cleared above and below. The result is unique, whatever the pivoting.
 *
 * @param a - A matrix `[m, n]`.
 * @param options - Pivoting rule and zero tolerance.
 * @returns The reduced form, pivot columns, rank and steps.
 * @concept reduced-row-echelon-form
 * @example
 * ```ts
 * import { rref } from 'whitebox-engine/foundations'
 * rref([[1, 2, 3], [2, 4, 7]]).matrix // => [[1, 2, 0], [0, 0, 1]]
 * rref([[1, 2, 3], [2, 4, 7]]).pivotColumns // => [0, 2]
 * ```
 */
export function rref(a: TensorLike, options: EliminationOptions = {}): Elimination {
  const { m, pivots, steps } = eliminate(matrix(a, 'rref'), true, options)
  return { matrix: m, pivots, pivotColumns: pivots.map((p) => p[1]), rank: pivots.length, steps }
}

/** The general solution of a linear system from {@link solveByElimination}. */
export interface LinearSystemSolution {
  readonly consistent: boolean
  /** A particular solution (free variables set to zero), or null for an inconsistent system. */
  readonly particular: number[] | null
  /** A basis of the null space: every solution is `particular + sum t_i nullSpace[i]`. */
  readonly nullSpace: number[][]
  readonly pivotColumns: number[]
  readonly freeColumns: number[]
  /** The reduction of the augmented matrix `[A | b]`. */
  readonly steps: readonly EliminationStep[]
}

/**
 * Solve `A x = b` by Gauss-Jordan elimination of the augmented matrix: consistent exactly when no pivot falls in the
 * last column; then the pivot variables are read off with the free variables at zero, and each free variable gives a
 * null-space direction.
 *
 * @param a - The coefficients `[m, n]`.
 * @param b - The right-hand side `[m]`.
 * @param options - Pivoting rule and zero tolerance.
 * @returns Consistency, a particular solution, the null space and the steps.
 * @concept linear-systems
 * @example
 * ```ts
 * import { solveByElimination } from 'whitebox-engine/foundations'
 * const s = solveByElimination([[1, 1, 1], [0, 1, 2]], [6, 5])
 * s.particular // => [1, 5, 0]
 * s.nullSpace // => [[1, -2, 1]]
 * ```
 */
export function solveByElimination(
  a: TensorLike,
  b: TensorLike,
  options: EliminationOptions = {},
): LinearSystemSolution {
  const am = matrix(a, 'solveByElimination')
  const bv = toFlat(promoteFloat(b)[0] as Value)
  if (bv.length !== am.length) throw new WhiteboxError('shape', 'solveByElimination: b needs one entry per row of A')
  const n = am[0]?.length ?? 0
  const augmented = am.map((row, i) => [...row, bv[i] ?? 0])
  const { m, pivots, steps } = eliminate(augmented, true, options, n + 1)
  const pivotColumns = pivots.map((p) => p[1]).filter((c) => c < n)
  const consistent = !pivots.some((p) => p[1] === n)
  const freeColumns = Array.from({ length: n }, (_, j) => j).filter((j) => !pivotColumns.includes(j))
  const particular = consistent ? Array.from({ length: n }, () => 0) : null
  if (particular !== null) for (const [r, c] of pivots) if (c < n) particular[c] = m[r]?.[n] ?? 0
  const nullSpaceBasis = freeColumns.map((f) => {
    const v: number[] = Array.from({ length: n }, (_, j) => (j === f ? 1 : 0))
    for (const [r, c] of pivots) if (c < n) v[c] = -(m[r]?.[f] ?? 0)
    return v.map((x) => (Object.is(x, -0) ? 0 : x))
  })
  return { consistent, particular, nullSpace: nullSpaceBasis, pivotColumns, freeColumns, steps }
}

/** Result of {@link luSteps}. */
export interface LuSteps {
  /** The permutation matrix with `A = P L U` (scipy's convention). */
  readonly p: number[][]
  readonly l: number[][]
  readonly u: number[][]
  /** The elimination steps on U (swaps also swap the multipliers already stored in L). */
  readonly steps: readonly EliminationStep[]
}

/**
 * LU factorisation with partial pivoting, step by step: Gaussian elimination stores each multiplier in L as it clears a
 * column of U, giving `A = P L U` (as scipy.linalg.lu) with L unit lower-triangular.
 *
 * @param a - A square matrix `[n, n]`.
 * @returns P, L, U and the elimination steps.
 * @concept lu-decomposition
 * @example
 * ```ts
 * import { luSteps } from 'whitebox-engine/foundations'
 * const f = luSteps([[1, 2], [3, 4]])
 * f.u // => [[3, 4], [0, 0.6666666666666667]]
 * f.l // => [[1, 0], [0.3333333333333333, 1]]
 * ```
 */
export function luSteps(a: TensorLike): LuSteps {
  const am = square(a, 'luSteps')
  const n = am.length
  const u = copy(am)
  const l: Matrix = Array.from({ length: n }, () => Array.from({ length: n }, () => 0))
  const perm = Array.from({ length: n }, (_, i) => i)
  const steps: EliminationStep[] = []
  for (let k = 0; k < n; k++) {
    let p = k
    for (let i = k + 1; i < n; i++) if (Math.abs(u[i]?.[k] ?? 0) > Math.abs(u[p]?.[k] ?? 0)) p = i
    if (p !== k) {
      const op: RowOperation = { kind: 'swap', rows: [k, p] }
      apply(u, op)
      apply(l, op)
      const t = perm[k] as number
      perm[k] = perm[p] as number
      perm[p] = t
      steps.push({ operation: op, matrix: copy(u), description: describe(op) })
    }
    const pivot = u[k]?.[k] ?? 0
    if (pivot === 0) continue
    for (let i = k + 1; i < n; i++) {
      const factor = (u[i]?.[k] ?? 0) / pivot
      if (factor === 0) continue
      ;(l[i] as number[])[k] = factor
      const op: RowOperation = { kind: 'add', target: i, source: k, factor: -factor }
      apply(u, op)
      ;(u[i] as number[])[k] = 0
      steps.push({ operation: op, matrix: copy(u), description: describe(op) })
    }
  }
  for (let i = 0; i < n; i++) (l[i] as number[])[i] = 1
  // Row i of U came from row perm[i] of A: A = P L U with P[perm[i]][i] = 1.
  const pm = Array.from({ length: n }, (_, r) => Array.from({ length: n }, (_, c) => (perm[c] === r ? 1 : 0)))
  return { p: pm, l: copy(l), u: copy(u), steps }
}

/** One term of a cofactor expansion along the first row. */
export interface CofactorTerm {
  readonly column: number
  readonly entry: number
  /** `(-1)^(1 + j)`. */
  readonly sign: number
  readonly minor: number
}

/**
 * The determinant by cofactor (Laplace) expansion along the first row, recursively: `det A = sum_j (-1)^(1+j) a_1j
 * M_1j`. Exact for integer matrices of modest size but `n!` work, so limited to n <= 9.
 *
 * @param a - A square matrix `[n, n]`.
 * @returns The determinant and the first-row terms.
 * @throws WhiteboxError with code `value` for n > 9.
 * @concept determinant
 * @example
 * ```ts
 * import { detByCofactors } from 'whitebox-engine/foundations'
 * detByCofactors([[2, 0, 1], [1, 3, 2], [1, 1, 2]]).value // => 6
 * ```
 */
export function detByCofactors(a: TensorLike): { readonly value: number; readonly terms: CofactorTerm[] } {
  const am = square(a, 'detByCofactors')
  if (am.length > 9) throw new WhiteboxError('value', 'detByCofactors: n! work; use detByElimination beyond 9 x 9')
  const det = (m: Matrix): number => {
    const n = m.length
    if (n === 0) return 1
    if (n === 1) return m[0]?.[0] ?? 0
    if (n === 2) return (m[0]?.[0] ?? 0) * (m[1]?.[1] ?? 0) - (m[0]?.[1] ?? 0) * (m[1]?.[0] ?? 0)
    let s = 0
    for (let j = 0; j < n; j++) {
      const e = m[0]?.[j] ?? 0
      if (e !== 0) s += (j % 2 === 0 ? 1 : -1) * e * det(minorOf(m, 0, j))
    }
    return s
  }
  const terms = (am[0] ?? []).map((entry, j) => ({
    column: j,
    entry,
    sign: j % 2 === 0 ? 1 : -1,
    minor: am.length === 1 ? 1 : det(minorOf(am, 0, j)),
  }))
  return {
    value: am.length === 1 ? (am[0]?.[0] ?? 0) : terms.reduce((s, t) => s + t.sign * t.entry * t.minor, 0),
    terms,
  }
}

function minorOf(m: Matrix, row: number, col: number): Matrix {
  return m.filter((_, i) => i !== row).map((r) => r.filter((_, j) => j !== col))
}

/**
 * The determinant by elimination: reduce to upper-triangular form with partial pivoting; the determinant is the product
 * of the pivots, negated once per row swap (adding multiples of rows changes nothing).
 *
 * @param a - A square matrix `[n, n]`.
 * @returns The determinant, the pivots, the number of swaps and the steps.
 * @concept determinant
 * @example
 * ```ts
 * import { detByElimination } from 'whitebox-engine/foundations'
 * const d = detByElimination([[0, 1], [2, 3]])
 * d.value // => -2
 * d.swaps // => 1
 * ```
 */
export function detByElimination(a: TensorLike): {
  readonly value: number
  readonly pivots: number[]
  readonly swaps: number
  readonly steps: readonly EliminationStep[]
} {
  const am = square(a, 'detByElimination')
  const n = am.length
  const { m, pivots, steps, swaps } = eliminate(am, false, { pivoting: 'partial', tol: 0 })
  if (pivots.length < n) return { value: 0, pivots: pivots.map(([r, c]) => m[r]?.[c] ?? 0), swaps, steps }
  const values = pivots.map(([r, c]) => m[r]?.[c] ?? 0)
  const value = values.reduce((p, v) => p * v, swaps % 2 === 0 ? 1 : -1)
  return { value, pivots: values, swaps, steps }
}

/** The four fundamental subspaces of a matrix, as bases of vectors. */
export interface FundamentalSubspaces {
  readonly rank: number
  readonly nullity: number
  /** Solutions of `A x = 0` (one per free variable, from the reduced form). */
  readonly nullSpace: number[][]
  /** The pivot columns of A itself. */
  readonly columnSpace: number[][]
  /** The nonzero rows of the reduced form. */
  readonly rowSpace: number[][]
  /** The null space of `A^T`. */
  readonly leftNullSpace: number[][]
}

/**
 * The four fundamental subspaces of A (Strang): column space and left null space in `R^m`, row space and null space
 * in `R^n`, with `rank + nullity = n`. Bases come from the reduced row echelon form, as sympy's `columnspace`,
 * `rowspace` and `nullspace`.
 *
 * @param a - A matrix `[m, n]`.
 * @param options - Zero tolerance.
 * @returns Rank, nullity and the four bases.
 * @concept fundamental-subspaces
 * @example
 * ```ts
 * import { fundamentalSubspaces } from 'whitebox-engine/foundations'
 * const s = fundamentalSubspaces([[1, 2], [2, 4]])
 * s.rank // => 1
 * s.nullSpace // => [[-2, 1]]
 * s.leftNullSpace // => [[-2, 1]]
 * ```
 */
export function fundamentalSubspaces(a: TensorLike, options: EliminationOptions = {}): FundamentalSubspaces {
  const am = matrix(a, 'fundamentalSubspaces')
  const m = am.length
  const n = am[0]?.length ?? 0
  const zeros = Array.from({ length: m }, () => 0)
  const solved = solveByElimination(am, zeros, options)
  const reduced = rref(am, options)
  const transposed = Array.from({ length: n }, (_, j) => am.map((r) => r[j] ?? 0))
  const left = solveByElimination(
    transposed,
    Array.from({ length: n }, () => 0),
    options,
  )
  return {
    rank: reduced.rank,
    nullity: n - reduced.rank,
    nullSpace: solved.nullSpace,
    columnSpace: reduced.pivotColumns.map((c) => am.map((r) => r[c] ?? 0)),
    rowSpace: reduced.matrix.slice(0, reduced.rank),
    leftNullSpace: left.nullSpace,
  }
}

/**
 * A basis of the span of some vectors chosen from among them: the vectors in pivot columns when they are the columns
 * of a matrix (each one not a combination of the earlier ones).
 *
 * @param vectors - The vectors `[k][n]`.
 * @param options - Zero tolerance.
 * @returns The chosen vectors and their indices.
 * @concept basis
 * @example
 * ```ts
 * import { basisFromVectors } from 'whitebox-engine/foundations'
 * basisFromVectors([[1, 0, 1], [2, 0, 2], [0, 1, 0]]).indices // => [0, 2]
 * ```
 */
export function basisFromVectors(
  vectors: TensorLike,
  options: EliminationOptions = {},
): { readonly basis: number[][]; readonly indices: number[] } {
  const vs = matrix(vectors, 'basisFromVectors')
  const n = vs[0]?.length ?? 0
  const columns = Array.from({ length: n }, (_, i) => vs.map((v) => v[i] ?? 0))
  const indices = rref(columns, options).pivotColumns
  return { basis: indices.map((i) => [...(vs[i] ?? [])]), indices }
}

/**
 * The orthogonal projection matrix onto the column space of A, `P = A A^+` (`A (A^T A)^-1 A^T` for independent
 * columns): symmetric and idempotent. Differentiable.
 *
 * @param a - A matrix `[m, n]`.
 * @returns `P` `[m, m]`.
 * @concept orthogonal-projection
 * @example
 * ```ts
 * import { projectionMatrix } from 'whitebox-engine/foundations'
 * import { toArray } from 'whitebox-engine/tensor'
 * toArray(projectionMatrix([[1], [1]])).map((r) => r.map((v) => Math.round(v * 1e12) / 1e12)) // => [[0.5, 0.5], [0.5, 0.5]]
 * ```
 */
export function projectionMatrix(a: TensorLike): Value {
  const [av] = promoteFloat(a) as [Value]
  return matmul(av, pinv(av))
}

/** Least squares seen as a projection. */
export interface LeastSquaresProjection {
  /** The least-squares solution of `A x ~ b`. */
  readonly x: number[]
  /** `A x`: the projection of b onto the column space. */
  readonly projection: number[]
  /** `b - A x`, orthogonal to every column of A. */
  readonly residual: number[]
  readonly residualNorm: number
  /** The normal equations `A^T A x = A^T b`. */
  readonly normalMatrix: number[][]
  readonly normalRhs: number[]
  /** `A^T r`, zero up to rounding. */
  readonly orthogonality: number[]
}

/**
 * Least squares as projection: the best `x` makes `A x` the orthogonal projection of b onto the column space, so the
 * residual is perpendicular to every column (the normal equations `A^T A x = A^T b`). Solved by QR (numpy's `lstsq`
 * for full column rank).
 *
 * @param a - A matrix `[m, n]` with independent columns.
 * @param b - The target `[m]`.
 * @returns The solution, projection, residual and normal equations.
 * @concept least-squares
 * @example
 * ```ts
 * import { leastSquaresProjection } from 'whitebox-engine/foundations'
 * // The best line through (0, 0), (1, 1), (2, 1).
 * leastSquaresProjection([[1, 0], [1, 1], [1, 2]], [0, 1, 1]).x.map((v) => Math.round(v * 1e12) / 1e12) // => [0.166666666667, 0.5]
 * ```
 */
export function leastSquaresProjection(a: TensorLike, b: TensorLike): LeastSquaresProjection {
  const am = matrix(a, 'leastSquaresProjection')
  const bv = toFlat(promoteFloat(b)[0] as Value)
  const x = toFlat(lstsq(am, bv))
  const projection = am.map((row) => row.reduce((s, v, j) => s + v * (x[j] ?? 0), 0))
  const residual = bv.map((v, i) => v - (projection[i] ?? 0))
  const n = am[0]?.length ?? 0
  const col = (j: number): number[] => am.map((r) => r[j] ?? 0)
  const dot = (p: readonly number[], q: readonly number[]): number => p.reduce((s, v, i) => s + v * (q[i] ?? 0), 0)
  return {
    x,
    projection,
    residual,
    residualNorm: Math.sqrt(dot(residual, residual)),
    normalMatrix: Array.from({ length: n }, (_, i) => Array.from({ length: n }, (_, j) => dot(col(i), col(j)))),
    normalRhs: Array.from({ length: n }, (_, i) => dot(col(i), bv)),
    orthogonality: Array.from({ length: n }, (_, i) => dot(col(i), residual)),
  }
}

/** A grid and its image under a linear map, for drawing (2-D or 3-D). */
export interface GridTransformation {
  readonly matrix: number[][]
  readonly determinant: number
  /** Grid lines before the map, each a polyline of points. */
  readonly before: number[][][]
  /** The same lines after the map. */
  readonly after: number[][][]
  /** The images of the standard basis vectors (the columns of the matrix). */
  readonly basisImages: number[][]
  /** The unit square (or cube's vertices) and its image. */
  readonly unitCell: { readonly before: number[][]; readonly after: number[][] }
}

/**
 * The picture of a linear map of the plane or space: a grid of lines on `[-extent, extent]^d` and its image, the
 * images of the basis vectors, and the unit square or cube with its image (whose signed area or volume is the
 * determinant).
 *
 * @param m - A 2x2 or 3x3 matrix.
 * @param options - Half-width of the grid (default 2) and lines per direction (default 5).
 * @returns The grid before and after, basis images, unit cell and determinant.
 * @throws WhiteboxError with code `shape` for other sizes.
 * @concept linear-transformations
 * @example
 * ```ts
 * import { gridTransformation } from 'whitebox-engine/foundations'
 * const g = gridTransformation([[1, 1], [0, 1]])
 * g.basisImages // => [[1, 0], [1, 1]]
 * g.determinant // => 1
 * g.unitCell.after // => [[0, 0], [1, 0], [2, 1], [1, 1]]
 * ```
 */
export function gridTransformation(
  m: TensorLike,
  options: { readonly extent?: number; readonly lines?: number } = {},
): GridTransformation {
  const mat = square(m, 'gridTransformation')
  const d = mat.length
  if (d !== 2 && d !== 3) throw new WhiteboxError('shape', 'gridTransformation: a 2x2 or 3x3 matrix')
  const extent = options.extent ?? 2
  const k = options.lines ?? 5
  const ticks = Array.from({ length: k }, (_, i) => (k === 1 ? 0 : -extent + (2 * extent * i) / (k - 1)))
  const map = (p: readonly number[]): number[] => mat.map((row) => row.reduce((s, v, j) => s + v * (p[j] ?? 0), 0))
  const before: number[][][] = []
  for (let axis = 0; axis < d; axis++) {
    // Lines parallel to `axis`, through every grid point of the other coordinates.
    const others = d === 2 ? ticks.map((t) => [t]) : ticks.flatMap((s) => ticks.map((t) => [s, t]))
    for (const o of others) {
      const pts = [-extent, extent].map((t) => {
        const p: number[] = []
        let q = 0
        for (let j = 0; j < d; j++) p.push(j === axis ? t : (o[q++] ?? 0))
        return p
      })
      before.push(pts)
    }
  }
  const cell =
    d === 2
      ? [
          [0, 0],
          [1, 0],
          [1, 1],
          [0, 1],
        ]
      : [0, 1, 2, 3, 4, 5, 6, 7].map((b) => [b & 1, (b >> 1) & 1, (b >> 2) & 1])
  return {
    matrix: copy(mat),
    determinant: detByElimination(mat).value,
    before,
    after: before.map((pl) => pl.map(map)),
    basisImages: Array.from({ length: d }, (_, j) => mat.map((r) => r[j] ?? 0)),
    unitCell: { before: cell, after: cell.map(map) },
  }
}

/**
 * Draw a linear map of the plane as a scene: the original grid faintly, the transformed grid, the image of the unit
 * square (filled) and the images of the basis vectors as arrows (tab0 for e1, tab1 for e2).
 *
 * @param m - A 2x2 matrix.
 * @param options - Grid half-width (default 2), lines (default 5) and pixel size (default 320).
 * @returns A scene.
 * @concept linear-transformations
 * @example
 * ```ts
 * import { linearMapScene } from 'whitebox-engine/foundations'
 * linearMapScene([[0, -1], [1, 0]]).width // => 320
 * ```
 */
export function linearMapScene(
  m: TensorLike,
  options: { readonly extent?: number; readonly lines?: number; readonly size?: number } = {},
): Scene {
  const g = gridTransformation(m, options)
  if (g.matrix.length !== 2) throw new WhiteboxError('shape', 'linearMapScene: a 2x2 matrix (use linearMapScene3d)')
  const size = options.size ?? 320
  const extent = options.extent ?? 2
  const reach = Math.max(extent, maxOf(g.after.flat().map((p) => Math.max(Math.abs(p[0] ?? 0), Math.abs(p[1] ?? 0)))))
  const s = size / (2 * reach * 1.05)
  const px = (p: readonly number[]): [number, number] => [size / 2 + s * (p[0] ?? 0), size / 2 - s * (p[1] ?? 0)]
  const seg = (pl: readonly (readonly number[])[], style: Parameters<typeof line>[4]): SceneNode => {
    const [a, b] = [px(pl[0] ?? []), px(pl[1] ?? [])]
    return line(a[0], a[1], b[0], b[1], style)
  }
  const nodes: SceneNode[] = [
    ...g.before.map((pl) => seg(pl, { stroke: 'grid', strokeWidth: 1 })),
    polygon(
      g.unitCell.after.map((p) => px(p)),
      { fill: 'accent', opacity: 0.25, stroke: 'accent' },
    ),
    ...g.after.map((pl) => seg(pl, { stroke: 'muted', strokeWidth: 1.5 })),
    ...g.basisImages.map((v, i) => {
      const [x0, y0] = px([0, 0])
      const [x1, y1] = px(v)
      return arrow(x0, y0, x1, y1, { stroke: i === 0 ? 'tab0' : 'tab1', strokeWidth: 2.5 })
    }),
  ]
  return scene(size, size, nodes, { title: `linear map, det = ${fmt(g.determinant)}` })
}

/**
 * Draw a linear map of space as a 3-D scene: the transformed grid lines, the images of the basis vectors (tab0, tab1,
 * tab2) and the coordinate axes.
 *
 * @param m - A 3x3 matrix.
 * @param options - Grid half-width (default 1) and lines per direction (default 3).
 * @returns A 3-D scene (draw it with a camera from `scene`).
 * @concept linear-transformations
 * @example
 * ```ts
 * import { linearMapScene3d } from 'whitebox-engine/foundations'
 * linearMapScene3d([[1, 0, 0], [0, 2, 0], [0, 0, 1]]).items.length > 20 // => true
 * ```
 */
export function linearMapScene3d(
  m: TensorLike,
  options: { readonly extent?: number; readonly lines?: number } = {},
): Scene3D {
  const g = gridTransformation(m, { extent: options.extent ?? 1, lines: options.lines ?? 3 })
  if (g.matrix.length !== 3) throw new WhiteboxError('shape', 'linearMapScene3d: a 3x3 matrix')
  const v3 = (p: readonly number[]): Vec3 => [p[0] ?? 0, p[1] ?? 0, p[2] ?? 0]
  const items: Item3D[] = [
    ...axes3d(1),
    ...g.after.map((pl) => segment3d(v3(pl[0] ?? []), v3(pl[1] ?? []), { stroke: 'muted' })),
    ...g.basisImages.map((v, i) =>
      segment3d([0, 0, 0], v3(v), { stroke: (['tab0', 'tab1', 'tab2'] as const)[i] ?? 'tab0', strokeWidth: 2.5 }),
    ),
  ]
  return scene3d(items, { title: `linear map, det = ${fmt(g.determinant)}` })
}

/** The characteristic polynomial with the Faddeev-LeVerrier steps. */
export interface CharacteristicPolynomial {
  /** Coefficients of `det(t I - A)`, highest degree first (numpy's `poly`): `[1, c_1, ..., c_n]`. */
  readonly coefficients: number[]
  /** The matrices `M_k` of the recursion `M_k = A M_{k-1} + c_{k-1} I`, `c_k = -tr(A M_k) / k`. */
  readonly steps: readonly number[][][]
}

/**
 * The characteristic polynomial `det(t I - A)` by the Faddeev-LeVerrier algorithm (traces of a sequence of matrices, no
 * determinants), as numpy's `poly`. A by-product: `A^-1 = -M_n / c_n` when `c_n != 0`.
 *
 * @param a - A square matrix `[n, n]`.
 * @returns The coefficients (highest first) and the matrices of the recursion.
 * @concept characteristic-polynomial
 * @example
 * ```ts
 * import { characteristicPolynomial } from 'whitebox-engine/foundations'
 * characteristicPolynomial([[2, 1], [1, 2]]).coefficients // => [1, -4, 3]
 * ```
 */
export function characteristicPolynomial(a: TensorLike): CharacteristicPolynomial {
  const am = square(a, 'characteristicPolynomial')
  const n = am.length
  const mul2 = (p: Matrix, q: Matrix): Matrix =>
    p.map((row) => q[0]?.map((_, j) => row.reduce((s, v, k) => s + v * (q[k]?.[j] ?? 0), 0)) ?? [])
  const coefficients = [1]
  const steps: Matrix[] = []
  let mk: Matrix = am.map((r, i) => r.map((_, j) => (i === j ? 1 : 0)))
  for (let k = 1; k <= n; k++) {
    steps.push(copy(mk))
    const product = mul2(am, mk)
    let trace = 0
    for (let i = 0; i < n; i++) trace += product[i]?.[i] ?? 0
    const c = -trace / k
    coefficients.push(c)
    mk = product.map((r, i) => r.map((v, j) => v + (i === j ? c : 0)))
  }
  return { coefficients: coefficients.map((c) => (Object.is(c, -0) ? 0 : c)), steps }
}

/** An eigenvalue of a small matrix with its eigenvectors and multiplicities. */
export interface SmallEigenpair {
  readonly re: number
  readonly im: number
  readonly algebraicMultiplicity: number
  /** A basis of the eigenspace (real eigenvalues only; empty for complex ones). */
  readonly eigenvectors: number[][]
}

/**
 * Eigenvalues and eigenvectors of a small matrix the way they are taught: the eigenvalues (the roots of the
 * characteristic polynomial, see {@link characteristicPolynomial}; computed here by the QR algorithm, which stays
 * accurate where polynomial roots do not) grouped with their algebraic multiplicities, and for each real eigenvalue a
 * basis of the null space of `A - lambda I`, whose size is the geometric multiplicity. A defective matrix (such as a
 * shear) has fewer eigenvectors than its multiplicity.
 *
 * @param a - A square matrix.
 * @param tol - Relative distance at which computed eigenvalues are merged (default 1e-5: a repeated eigenvalue of a
 *   defective matrix is computed only to about the cube root of machine precision).
 * @returns The distinct eigenvalues (sorted by real then imaginary part) and their eigenspaces.
 * @concept eigenvalues
 * @example
 * ```ts
 * import { smallEigen } from 'whitebox-engine/foundations'
 * const e = smallEigen([[2, 1], [1, 2]])
 * e.map((p) => Math.round(p.re * 1e12) / 1e12) // => [1, 3]
 * e[1].eigenvectors[0].map((v) => Math.round(v * 1e12) / 1e12) // => [0.707106781187, 0.707106781187]
 * ```
 */
export function smallEigen(a: TensorLike, tol = 1e-5): SmallEigenpair[] {
  const am = square(a, 'smallEigen')
  const values = eigvals(am)
  const re = toFlat(values.re)
  const im = toFlat(values.im)
  const scale = Math.max(1, scaleOf(am))
  const groups: { re: number; im: number; count: number }[] = []
  for (const [i, r] of re.entries()) {
    const q = im[i] ?? 0
    const g = groups.find((x) => Math.hypot(x.re - r, x.im - q) <= tol * scale)
    if (g === undefined) groups.push({ re: r, im: q, count: 1 })
    else {
      // The mean of a cluster of perturbed multiple roots is far more accurate than any one of them.
      g.re = (g.re * g.count + r) / (g.count + 1)
      g.im = (g.im * g.count + q) / (g.count + 1)
      g.count++
    }
  }
  groups.sort((x, y) => x.re - y.re || x.im - y.im)
  return groups.map((g) => {
    const real = Math.abs(g.im) <= tol * scale
    const shifted = am.map((row, i) => row.map((v, j) => v - (i === j ? g.re : 0)))
    const ns = real ? nullSpace(shifted, 1e-8) : null
    const vectors =
      ns === null || (ns.shape[1] ?? 0) === 0
        ? []
        : (() => {
            const cols = toArray(ns) as Matrix
            const k = ns.shape[1] ?? 0
            return Array.from({ length: k }, (_, j) => {
              const v = cols.map((r) => r[j] ?? 0)
              // Sign convention: the largest-magnitude entry positive.
              let big = 0
              for (let i = 1; i < v.length; i++) if (Math.abs(v[i] ?? 0) > Math.abs(v[big] ?? 0)) big = i
              const s = (v[big] ?? 0) < 0 ? -1 : 1
              return v.map((x) => (x * s === 0 ? 0 : x * s))
            })
          })()
    return { re: g.re, im: real ? 0 : g.im, algebraicMultiplicity: g.count, eigenvectors: vectors }
  })
}

/** A diagonalisation `A = P D P^-1`. */
export interface Diagonalisation {
  /** Whether the eigenvectors span the space (every geometric multiplicity equals the algebraic one). */
  readonly diagonalisable: boolean
  /** Eigenvectors as columns (when diagonalisable). */
  readonly p: number[][] | null
  /** The eigenvalues on the diagonal, in the order of the columns of P. */
  readonly d: number[]
  readonly pInverse: number[][] | null
}

/**
 * Diagonalise a small real matrix with real eigenvalues: `A = P D P^-1` with the eigenvectors as the columns of P. Not
 * diagonalisable when an eigenvalue has fewer independent eigenvectors than its multiplicity (a defective matrix such
 * as a shear), or over the reals when an eigenvalue is complex.
 *
 * @param a - A square matrix.
 * @returns P, D, P^-1, or `diagonalisable: false`.
 * @concept diagonalisation
 * @example
 * ```ts
 * import { diagonalise } from 'whitebox-engine/foundations'
 * diagonalise([[1, 1], [0, 1]]).diagonalisable // => false
 * diagonalise([[2, 0], [0, 3]]).d // => [2, 3]
 * ```
 */
export function diagonalise(a: TensorLike): Diagonalisation {
  const am = square(a, 'diagonalise')
  const n = am.length
  const pairs = smallEigen(am)
  const ok = pairs.every((p) => p.im === 0 && p.eigenvectors.length === p.algebraicMultiplicity)
  if (!ok)
    return {
      diagonalisable: false,
      p: null,
      d: pairs.flatMap((p) => Array.from({ length: p.algebraicMultiplicity }, () => p.re)),
      pInverse: null,
    }
  const cols = pairs.flatMap((p) => p.eigenvectors)
  const d = pairs.flatMap((p) => p.eigenvectors.map(() => p.re))
  const pm = Array.from({ length: n }, (_, i) => cols.map((c) => c[i] ?? 0))
  const inv = toArray(pinv(pm)) as Matrix
  return { diagonalisable: true, p: pm, d, pInverse: inv }
}

/**
 * The quadratic form `x^T A x` at points (with A symmetrised). Differentiable.
 *
 * @param a - A square matrix `[n, n]`.
 * @param x - Points `[..., n]`.
 * @returns Values `[...]`.
 * @concept quadratic-forms
 * @example
 * ```ts
 * import { quadraticForm } from 'whitebox-engine/foundations'
 * import { toArray } from 'whitebox-engine/tensor'
 * toArray(quadraticForm([[1, 2], [0, 3]], [[1, 1]])) // => [6]
 * ```
 */
export function quadraticForm(a: TensorLike, x: TensorLike): Value {
  const [av, xv] = promoteFloat(a, x) as [Value, Value]
  return sum(mul(matmul(xv, av), xv), -1)
}

/** The definiteness of a quadratic form. */
export type Definiteness =
  | 'positive definite'
  | 'positive semidefinite'
  | 'negative definite'
  | 'negative semidefinite'
  | 'indefinite'

/**
 * Classify a quadratic form by the signs of the eigenvalues of its symmetric matrix (Sylvester's law of inertia: the
 * signs, not the values, are invariant under change of basis).
 *
 * @param a - A square matrix (symmetrised).
 * @param tol - Eigenvalues within `tol * max |lambda|` of zero count as zero (default 1e-12).
 * @returns The definiteness and the inertia (counts of positive, negative and zero eigenvalues).
 * @concept quadratic-forms
 * @example
 * ```ts
 * import { classifyQuadraticForm } from 'whitebox-engine/foundations'
 * classifyQuadraticForm([[1, 2], [2, 1]]).definiteness // => 'indefinite'
 * classifyQuadraticForm([[1, 1], [1, 1]]).inertia // => [1, 0, 1]
 * ```
 */
export function classifyQuadraticForm(
  a: TensorLike,
  tol = 1e-12,
): { readonly definiteness: Definiteness; readonly inertia: [number, number, number]; readonly eigenvalues: number[] } {
  const am = square(a, 'classifyQuadraticForm')
  const sym = am.map((r, i) => r.map((v, j) => (v + (am[j]?.[i] ?? 0)) / 2))
  const eigenvalues = toFlat(eigvalsh(sym))
  const cut = tol * Math.max(1e-300, maxOf(eigenvalues.map(Math.abs)))
  const pos = eigenvalues.filter((e) => e > cut).length
  const neg = eigenvalues.filter((e) => e < -cut).length
  const zero = eigenvalues.length - pos - neg
  const definiteness: Definiteness =
    pos > 0 && neg > 0
      ? 'indefinite'
      : pos === eigenvalues.length
        ? 'positive definite'
        : neg === eigenvalues.length
          ? 'negative definite'
          : pos > 0 || zero === eigenvalues.length
            ? 'positive semidefinite'
            : 'negative semidefinite'
  return { definiteness, inertia: [pos, neg, zero], eigenvalues }
}

/**
 * The principal axes of a quadratic form: the orthonormal eigenvectors of its symmetric matrix, in which the form
 * becomes `sum lambda_i y_i^2` (the axes of the ellipse or hyperbola `x^T A x = 1`).
 *
 * @param a - A square matrix (symmetrised).
 * @returns The eigenvalues (ascending), the axes as unit vectors, and the semi-axis lengths `1 / sqrt(lambda)` of the
 *   level set `x^T A x = 1` (NaN for non-positive eigenvalues).
 * @concept principal-axes
 * @example
 * ```ts
 * import { principalAxes } from 'whitebox-engine/foundations'
 * principalAxes([[4, 0], [0, 1]]).semiAxes // => [1, 0.5]
 * ```
 */
export function principalAxes(a: TensorLike): {
  readonly eigenvalues: number[]
  readonly axes: number[][]
  readonly semiAxes: number[]
} {
  const am = square(a, 'principalAxes')
  const sym = am.map((r, i) => r.map((v, j) => (v + (am[j]?.[i] ?? 0)) / 2))
  const { eigenvalues, eigenvectors } = eigh(sym)
  const values = toFlat(eigenvalues)
  const vecs = toArray(eigenvectors) as Matrix
  return {
    eigenvalues: values,
    axes: values.map((_, j) => vecs.map((r) => r[j] ?? 0)),
    semiAxes: values.map((l) => (l > 0 ? 1 / Math.sqrt(l) : Number.NaN)),
  }
}

/** The SVD as three simple maps: rotate (V^T), stretch (Sigma), rotate (U). */
export interface SvdGeometry {
  readonly u: number[][]
  readonly singularValues: number[]
  readonly vt: number[][]
  /** The unit circle (2-D) or a ring of the unit sphere's great circles (3-D) after each stage: start, V^T, Sigma V^T, A. */
  readonly stages: readonly number[][][]
  /** The semi-axes of the image ellipse(oid): `sigma_i u_i`. */
  readonly axes: number[][]
}

/**
 * The geometry of the singular value decomposition `A = U Sigma V^T` of a 2x2 or 3x3 matrix: the unit circle (or
 * sphere) is rotated by V^T, stretched along the axes by the singular values, and rotated by U into an ellipse (or
 * ellipsoid) with semi-axes `sigma_i u_i`.
 *
 * @param a - A 2x2 or 3x3 matrix.
 * @param n - Points on the circle (default 64).
 * @returns The factors, the points after each stage and the image's semi-axes.
 * @concept singular-value-decomposition
 * @example
 * ```ts
 * import { svdGeometry } from 'whitebox-engine/foundations'
 * svdGeometry([[3, 0], [0, 2]]).singularValues // => [3, 2]
 * ```
 */
export function svdGeometry(a: TensorLike, n = 64): SvdGeometry {
  const am = square(a, 'svdGeometry')
  const d = am.length
  if (d !== 2 && d !== 3) throw new WhiteboxError('shape', 'svdGeometry: a 2x2 or 3x3 matrix')
  const { u, s, vh } = svd(am)
  const um = toArray(u) as Matrix
  const sv = toFlat(s)
  const vt = toArray(vh) as Matrix
  const ts = Array.from({ length: n }, (_, i) => (2 * Math.PI * i) / n)
  const circle =
    d === 2
      ? ts.map((t) => [Math.cos(t), Math.sin(t)])
      : [
          ...ts.map((t) => [Math.cos(t), Math.sin(t), 0]),
          ...ts.map((t) => [Math.cos(t), 0, Math.sin(t)]),
          ...ts.map((t) => [0, Math.cos(t), Math.sin(t)]),
        ]
  const apply = (m: Matrix, pts: number[][]): number[][] =>
    pts.map((p) => m.map((r) => r.reduce((acc, v, j) => acc + v * (p[j] ?? 0), 0)))
  const rotated = apply(vt, circle)
  const stretched = rotated.map((p) => p.map((x, i) => x * (sv[i] ?? 0)))
  const image = apply(um, stretched)
  return {
    u: um,
    singularValues: sv,
    vt,
    stages: [circle, rotated, stretched, image],
    axes: sv.map((sigma, i) => um.map((r) => sigma * (r[i] ?? 0))),
  }
}
