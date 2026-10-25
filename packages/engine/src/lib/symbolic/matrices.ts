/**
 * Symbolic matrices: rows of expressions with transposes, products, determinants (Berkowitz's algorithm),
 * characteristic polynomials, inverses by the adjugate, null spaces, and exact eigenvalues in radicals with their
 * eigenvectors.
 */

import { MdmlError } from '../base/index.ts'
import { bounded } from './_budget.ts'
import { MINUS_ONE, makeAdd, makeMul, makePow, numFromRational, ONE, ZERO } from './_canon.ts'
import { expandExpr } from './_expand.ts'
import { evalComplexConstant, freeNames } from './_numeric.ts'
import { rat } from './_rational.ts'
import { cancel } from './_rational-functions.ts'
import { hasTrig, trigSimplifyExpr } from './_trigsimp.ts'
import { asExpr, type Expr, type ExprLike, sym } from './expr.ts'
import { exactRoots, type Root } from './polynomials.ts'

/** A symbolic matrix: rows of expressions (all rows the same length). */
export type Matrix = readonly (readonly Expr[])[]

/** An eigenvalue with its algebraic multiplicity and a basis of its eigenspace. */
export interface Eigenvector {
  readonly value: Expr
  readonly multiplicity: number
  readonly vectors: readonly (readonly Expr[])[]
}

function shape(a: Matrix): [number, number] {
  return [a.length, a[0]?.length ?? 0]
}

function square(a: Matrix, what: string): number {
  const [n, m] = shape(a)
  if (n !== m) throw new MdmlError('shape', `${what}: matrix must be square, got ${n}x${m}`)
  return n
}

/** Normal form of an entry: expanded, and cancelled when it has denominators. */
function normal(e: Expr): Expr {
  const x = expandExpr(e)
  return x.kind === 'mul' || x.kind === 'pow' || x.kind === 'add' ? cancel(x) : x
}

/** Whether an entry is zero (exactly after cancellation, or numerically for constants). */
function isZero(e: Expr): boolean {
  const c = normal(e)
  if (c.kind === 'num') return c.n === 0n
  if (freeNames(c).length === 0) {
    const v = evalComplexConstant(c)
    return v !== undefined && Math.hypot(v[0], v[1]) < 1e-12
  }
  return false
}

/**
 * A matrix from rows of expressions or numbers.
 *
 * @param rows - The rows.
 * @returns The matrix.
 * @throws MdmlError with code `shape` for ragged rows.
 * @example
 * ```ts
 * import { matrix } from 'mdml-engine/symbolic'
 * matrix([[1, 2], [3, 4]]).length // => 2
 * ```
 */
export function matrix(rows: readonly (readonly ExprLike[])[]): Matrix {
  const m = rows[0]?.length ?? 0
  if (rows.some((r) => r.length !== m)) throw new MdmlError('shape', 'matrix: rows must have equal length')
  return rows.map((r) => r.map(asExpr))
}

/**
 * The identity matrix.
 *
 * @param n - Size.
 * @returns The n x n identity.
 * @example
 * ```ts
 * import { identityMatrix, toText } from 'mdml-engine/symbolic'
 * toText(identityMatrix(2)[1]?.[1] ?? 0) // => '1'
 * ```
 */
export function identityMatrix(n: number): Matrix {
  return Array.from({ length: n }, (_, i) => Array.from({ length: n }, (_, j) => (i === j ? ONE : ZERO)))
}

/**
 * Transpose.
 *
 * @param a - The matrix.
 * @returns Its transpose.
 * @example
 * ```ts
 * import { matrix, toText, transpose } from 'mdml-engine/symbolic'
 * toText(transpose(matrix([[1, 2]]))[1]?.[0] ?? 0) // => '2'
 * ```
 */
export function transpose(a: Matrix): Matrix {
  const [n, m] = shape(a)
  return Array.from({ length: m }, (_, j) => Array.from({ length: n }, (_, i) => (a[i] as Expr[])[j] as Expr))
}

/**
 * Matrix product, entries expanded.
 *
 * @param a - Left factor (n x k).
 * @param b - Right factor (k x m).
 * @returns The n x m product.
 * @throws MdmlError with code `shape` on mismatched sizes.
 * @example
 * ```ts
 * import { matMul, matrix, parse, toText } from 'mdml-engine/symbolic'
 * toText(matMul(matrix([[parse('a'), 1]]), matrix([[2], [parse('b')]]))[0]?.[0] ?? 0) // => '2*a + b'
 * ```
 */
export function matMul(a: Matrix, b: Matrix): Matrix {
  return bounded('matMul', () => {
    const [n, k] = shape(a)
    const [k2, m] = shape(b)
    if (k !== k2) throw new MdmlError('shape', `matMul: ${n}x${k} times ${k2}x${m}`)
    return Array.from({ length: n }, (_, i) =>
      Array.from({ length: m }, (_, j) =>
        normal(
          makeAdd(
            Array.from({ length: k }, (_, t) => makeMul([(a[i] as Expr[])[t] as Expr, (b[t] as Expr[])[j] as Expr])),
          ),
        ),
      ),
    )
  })
}

/**
 * Coefficients of det(λI - A) by Berkowitz's division-free algorithm: [1, c1, ..., cn].
 */
function berkowitz(a: Matrix): Expr[] {
  const n = a.length
  let v: Expr[] = [ONE]
  for (let k = 0; k < n; k++) {
    const akk = (a[k] as Expr[])[k] as Expr
    const q: Expr[] = [ONE, makeMul([MINUS_ONE, akk])]
    let w: Expr[] = Array.from({ length: k }, (_, i) => (a[i] as Expr[])[k] as Expr)
    for (let j = 0; j < k; j++) {
      const rw = makeAdd(w.map((wi, i) => makeMul([(a[k] as Expr[])[i] as Expr, wi])))
      q.push(normal(makeMul([MINUS_ONE, rw])))
      w = Array.from({ length: k }, (_, i) =>
        normal(makeAdd(w.map((wt, t) => makeMul([(a[i] as Expr[])[t] as Expr, wt])))),
      )
    }
    const next: Expr[] = []
    for (let i = 0; i <= k + 1; i++) {
      const terms: Expr[] = []
      for (let j = 0; j <= i && j < v.length; j++) terms.push(makeMul([q[i - j] ?? ZERO, v[j] as Expr]))
      next.push(normal(makeAdd(terms)))
    }
    v = next
  }
  return v
}

/**
 * Determinant (Berkowitz's algorithm, entries kept expanded; rational entries are cancelled).
 *
 * @param a - A square matrix.
 * @returns The determinant.
 * @example
 * ```ts
 * import { det, matrix, parse, toText } from 'mdml-engine/symbolic'
 * toText(det(matrix([[parse('a'), parse('b')], [parse('c'), parse('d')]]))) // => 'a*d - b*c'
 * ```
 */
export function det(a: Matrix): Expr {
  return bounded('det', () => {
    const n = square(a, 'det')
    if (n === 0) return ONE
    const c = berkowitz(a)[n] as Expr
    return normal(n % 2 === 0 ? c : makeMul([MINUS_ONE, c]))
  })
}

/**
 * The characteristic polynomial det(λI - A), expanded in λ.
 *
 * @param a - A square matrix.
 * @param lambda - The variable.
 * @returns The polynomial.
 * @example
 * ```ts
 * import { charpoly, matrix, sym, toText } from 'mdml-engine/symbolic'
 * toText(charpoly(matrix([[1, 2], [2, 1]]), sym('lambda'))) // => 'lambda**2 - 2*lambda - 3'
 * ```
 */
export function charpoly(a: Matrix, lambda: Expr): Expr {
  return bounded('charpoly', () => {
    const n = square(a, 'charpoly')
    // As SymPy's Poly.as_expr: each coefficient (simplified) times its power of lambda, the products not expanded.
    const c = berkowitz(a).map((ci) => (hasTrig(ci) ? trigSimplifyExpr(ci) : ci))
    return makeAdd(c.map((ci, i) => makeMul([ci, makePow(lambda, numFromRational(rat(BigInt(n - i))))])))
  })
}

function minor(a: Matrix, r: number, c: number): Matrix {
  return a.filter((_, i) => i !== r).map((row) => row.filter((_, j) => j !== c))
}

/**
 * Inverse by the adjugate: entries cofactor / det, each cancelled.
 *
 * @param a - A square, non-singular matrix.
 * @returns The inverse.
 * @throws MdmlError with code `singular` when the determinant is zero.
 * @example
 * ```ts
 * import { inverse, matrix, toText } from 'mdml-engine/symbolic'
 * inverse(matrix([[2, 1], [1, 1]])).map((r) => r.map(toText)) // => [['1', '-1'], ['-1', '2']]
 * ```
 */
export function inverse(a: Matrix): Matrix {
  return bounded('inverse', () => {
    const n = square(a, 'inverse')
    const d = det(a)
    if (isZero(d)) throw new MdmlError('singular', 'inverse: the matrix is singular')
    const inv = makePow(d, MINUS_ONE)
    return Array.from({ length: n }, (_, i) =>
      Array.from({ length: n }, (_, j) => {
        const cof = n === 1 ? ONE : det(minor(a, j, i))
        return cancel(makeMul([(i + j) % 2 === 0 ? ONE : MINUS_ONE, cof, inv]))
      }),
    )
  })
}

/**
 * A basis of the null space {v : A v = 0}, from the reduced row echelon form (one vector per free column, with a 1
 * in that column).
 *
 * @param a - The matrix.
 * @returns The basis vectors.
 * @example
 * ```ts
 * import { matrix, nullspace, toText } from 'mdml-engine/symbolic'
 * nullspace(matrix([[1, 2], [2, 4]])).map((v) => v.map(toText)) // => [['-2', '1']]
 * ```
 */
export function nullspace(a: Matrix): Expr[][] {
  return bounded('nullspace', () => {
    const [n, m] = shape(a)
    const rows = a.map((r) => [...r])
    const pivots: number[] = []
    let r = 0
    for (let c = 0; c < m && r < n; c++) {
      let piv = -1
      for (let i = r; i < n; i++) {
        if (!isZero((rows[i] as Expr[])[c] as Expr)) {
          piv = i
          break
        }
      }
      if (piv < 0) continue
      ;[rows[r], rows[piv]] = [rows[piv] as Expr[], rows[r] as Expr[]]
      const row = rows[r] as Expr[]
      const inv = makePow(row[c] as Expr, MINUS_ONE)
      for (let j = 0; j < m; j++) row[j] = normal(makeMul([row[j] as Expr, inv]))
      for (let i = 0; i < n; i++) {
        if (i === r) continue
        const other = rows[i] as Expr[]
        const f = other[c] as Expr
        if (isZero(f)) {
          other[c] = ZERO
          continue
        }
        for (let j = 0; j < m; j++)
          other[j] = normal(makeAdd([other[j] as Expr, makeMul([MINUS_ONE, f, row[j] as Expr])]))
      }
      pivots.push(c)
      r++
    }
    const out: Expr[][] = []
    for (let free = 0; free < m; free++) {
      if (pivots.includes(free)) continue
      const v: Expr[] = new Array<Expr>(m).fill(ZERO)
      v[free] = ONE
      pivots.forEach((pc, i) => {
        v[pc] = normal(makeMul([MINUS_ONE, (rows[i] as Expr[])[free] as Expr]))
      })
      out.push(v)
    }
    return out
  })
}

/**
 * Eigenvalues: the roots of the characteristic polynomial in radicals, with algebraic multiplicities (complete when
 * its irreducible factors over the rationals have degree at most four).
 *
 * @param a - A square matrix.
 * @returns The eigenvalues.
 * @example
 * ```ts
 * import { eigenvals, matrix, toText } from 'mdml-engine/symbolic'
 * eigenvals(matrix([[2, 1], [1, 2]])).map((r) => toText(r.value)) // => ['1', '3']
 * ```
 */
export function eigenvals(a: Matrix): Root[] {
  return bounded('eigenvals', () => {
    square(a, 'eigenvals')
    const lam = sym('lambda_')
    return exactRoots(charpoly(a, lam), lam)
  })
}

/**
 * Eigenvalues with bases of their eigenspaces (null spaces of A - λI).
 *
 * @param a - A square matrix.
 * @returns One entry per distinct eigenvalue.
 * @example
 * ```ts
 * import { eigenvects, matrix, toText } from 'mdml-engine/symbolic'
 * eigenvects(matrix([[2, 0], [0, 3]])).map((e) => e.vectors[0]?.map(toText)) // => [['1', '0'], ['0', '1']]
 * ```
 */
export function eigenvects(a: Matrix): Eigenvector[] {
  return bounded('eigenvects', () => {
    square(a, 'eigenvects')
    return eigenvals(a).map(({ value, multiplicity }) => {
      const shifted = a.map((row, i) => row.map((x, j) => (i === j ? makeAdd([x, makeMul([MINUS_ONE, value])]) : x)))
      return { value, multiplicity, vectors: nullspace(shifted) }
    })
  })
}
