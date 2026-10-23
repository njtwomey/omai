import { WhiteboxError } from '../../base/index.ts'
import {
  add,
  concat,
  div,
  fromStorage,
  mul,
  primalOf,
  promoteFloat,
  reshape,
  sliceAxis,
  sqrt,
  sum,
  type Tensor,
  type TensorLike,
  toFlat,
  type Value,
  where,
} from '../../tensor/index.ts'
import { eigvals, lstsq } from '../linalg/index.ts'

// Coefficients are lowest degree first, as numpy.polynomial.polynomial: c[0] + c[1] x + c[2] x^2 + ...

/**
 * Evaluate a polynomial by Horner's rule (numpy.polynomial.polynomial.polyval), differentiable in the points and the
 * coefficients. Extra coefficient axes broadcast against x as numpy (with `tensor` true, as numpy's default).
 *
 * @param x - Points.
 * @param c - Coefficients `[k + 1]` (lowest degree first).
 * @returns The values, the shape of x.
 * @concept polynomials
 * @example
 * ```ts
 * import { polyval } from 'whitebox-engine/numerics/polynomial'
 * import { toArray } from 'whitebox-engine/tensor'
 * toArray(polyval([0, 1, 2], [1, 2, 3])) // => [1, 6, 17]
 * ```
 */
export function polyval(x: TensorLike, c: TensorLike): Value {
  const [xv, cv] = promoteFloat(x, c) as [Value, Value]
  const k = cv.shape[0] ?? 0
  if (k === 0) return mul(xv, 0)
  const coefficient = (i: number): Value => {
    const ci = reshape(sliceAxis(cv, 0, i, i + 1), cv.shape.slice(1))
    // Each further coefficient axis is placed before x's axes, as numpy.
    return cv.shape.length > 1 ? reshape(ci, [...ci.shape, ...new Array<number>(xv.shape.length).fill(1)]) : ci
  }
  let acc: Value = coefficient(k - 1)
  for (let i = k - 2; i >= 0; i--) acc = add(mul(acc, xv), coefficient(i))
  return cv.shape.length > 1 || acc.shape.length === xv.shape.length ? acc : add(acc, mul(xv, 0))
}

/**
 * The Vandermonde matrix V[i, j] = x_i^j for j = 0..deg (numpy.polynomial.polynomial.polyvander).
 *
 * @param x - Points `[m]`.
 * @param deg - Degree.
 * @returns `[m, deg + 1]`.
 * @example
 * ```ts
 * import { polyvander } from 'whitebox-engine/numerics/polynomial'
 * import { toArray } from 'whitebox-engine/tensor'
 * toArray(polyvander([2, 3], 2)) // => [[1, 2, 4], [1, 3, 9]]
 * ```
 */
export function polyvander(x: TensorLike, deg: number): Value {
  const [xv] = promoteFloat(x) as [Value]
  const col = reshape(xv, [-1, 1])
  const columns: Value[] = [add(mul(col, 0), 1)]
  for (let j = 1; j <= deg; j++) columns.push(mul(columns[j - 1] as Value, col))
  return concat(columns, 1)
}

/**
 * The least-squares polynomial fit of degree deg (numpy.polynomial.polynomial.polyfit): minimises
 * sum (w_i (y_i - p(x_i)))^2, with columns scaled for conditioning as numpy. Differentiable in x, y and w.
 *
 * @param x - Points `[m]`.
 * @param y - Values `[m]` or `[m, k]` (several fits).
 * @param deg - Degree.
 * @param w - Optional weights `[m]`.
 * @returns Coefficients `[deg + 1]` (or `[deg + 1, k]`), lowest first.
 * @example
 * ```ts
 * import { polyfit } from 'whitebox-engine/numerics/polynomial'
 * import { toArray } from 'whitebox-engine/tensor'
 * toArray(polyfit([0, 1, 2, 3], [1, 3, 5, 7], 1)).map((v) => Math.round(v * 1e12) / 1e12) // => [1, 2]
 * ```
 */
export function polyfit(x: TensorLike, y: TensorLike, deg: number, w?: TensorLike): Value {
  return weightedFit(polyvander(x, deg), y, w)
}

/** Weighted least squares with column scaling (numpy's *fit functions). */
export function weightedFit(vander: Value, y: TensorLike, w?: TensorLike): Value {
  const [yv] = promoteFloat(y) as [Value]
  let lhs = vander
  let rhs = yv
  if (w !== undefined) {
    const [wv] = promoteFloat(w) as [Value]
    lhs = mul(lhs, reshape(wv, [-1, 1]))
    rhs = yv.shape.length === 1 ? mul(rhs, wv) : mul(rhs, reshape(wv, [-1, 1]))
  }
  // Scale each column to unit norm, solve, and undo the scaling.
  const scale = sqrt(reshape(sumRows(mul(lhs, lhs)), [1, -1]))
  const solution = lstsq(safeDivide(lhs, scale), rhs)
  const s = reshape(scale, [-1])
  return yv.shape.length === 1 ? safeDivide(solution, s) : safeDivide(solution, reshape(s, [-1, 1]))
}

function sumRows(x: Value): Value {
  return sum(x, 0)
}

/** a / b, with zero divisors (all-zero columns) replaced by 1. */
function safeDivide(a: Value, b: Value): Value {
  const zero = fromStorage(
    Uint8Array.from(toFlat(primalOf(b)), (v) => (v === 0 ? 1 : 0)),
    b.shape,
    'bool',
  )
  return div(a, where(zero, 1, b))
}

/** Drop trailing (highest-degree) zero coefficients, keeping at least one. */
export function polytrim(c: readonly number[], tol = 0): number[] {
  let n = c.length
  while (n > 1 && Math.abs(c[n - 1] ?? 0) <= tol) n--
  return c.slice(0, n)
}

/**
 * The m-th derivative of a polynomial (numpy.polynomial.polynomial.polyder), scaled by scl^m.
 *
 * @param c - Coefficients, lowest first.
 * @param m - Order (default 1).
 * @param scl - Scale per differentiation (default 1).
 * @returns Coefficients of the derivative.
 * @example
 * ```ts
 * import { polyder } from 'whitebox-engine/numerics/polynomial'
 * polyder([1, 2, 3, 4]) // => [2, 6, 12]
 * ```
 */
export function polyder(c: readonly number[], m = 1, scl = 1): number[] {
  let out = [...c]
  for (let r = 0; r < m; r++) {
    if (out.length <= 1) return [0]
    out = out.slice(1).map((v, i) => v * (i + 1) * scl)
  }
  return out
}

/**
 * The m-th integral of a polynomial (numpy.polynomial.polynomial.polyint) with integration constants k (one per
 * integration, set so the integral is k at lbnd), each integration scaled by scl.
 *
 * @param c - Coefficients, lowest first.
 * @param m - Order (default 1).
 * @param k - Integration constants (default zeros).
 * @param lbnd - Lower bound (default 0).
 * @param scl - Scale (default 1).
 * @returns Coefficients of the integral.
 * @example
 * ```ts
 * import { polyint } from 'whitebox-engine/numerics/polynomial'
 * polyint([2, 6, 12]) // => [0, 2, 3, 4]
 * ```
 */
export function polyint(c: readonly number[], m = 1, k: readonly number[] = [], lbnd = 0, scl = 1): number[] {
  let out = [...c]
  for (let r = 0; r < m; r++) {
    const next = [0, ...out.map((v, i) => (v * scl) / (i + 1))]
    const at = horner(next, lbnd)
    next[0] = (next[0] ?? 0) + (k[r] ?? 0) - at
    out = next
  }
  return out
}

function horner(c: readonly number[], x: number): number {
  let acc = 0
  for (let i = c.length - 1; i >= 0; i--) acc = acc * x + (c[i] ?? 0)
  return acc
}

/**
 * The sum of two polynomials (numpy.polynomial.polynomial.polyadd).
 *
 * @param a - Coefficients, lowest first.
 * @param b - Coefficients, lowest first.
 * @returns The sum, trimmed of trailing zeros.
 * @example
 * ```ts
 * import { polyadd } from 'whitebox-engine/numerics/polynomial'
 * polyadd([1, 2], [3, 4, 5]) // => [4, 6, 5]
 * ```
 */
export function polyadd(a: readonly number[], b: readonly number[]): number[] {
  return polytrim(Array.from({ length: Math.max(a.length, b.length) }, (_, i) => (a[i] ?? 0) + (b[i] ?? 0)))
}

/**
 * The difference of two polynomials (numpy.polynomial.polynomial.polysub).
 *
 * @param a - Coefficients, lowest first.
 * @param b - Coefficients, lowest first.
 * @returns a - b.
 * @example
 * ```ts
 * import { polysub } from 'whitebox-engine/numerics/polynomial'
 * polysub([1, 2, 3], [1, 2]) // => [0, 0, 3]
 * ```
 */
export function polysub(a: readonly number[], b: readonly number[]): number[] {
  return polytrim(Array.from({ length: Math.max(a.length, b.length) }, (_, i) => (a[i] ?? 0) - (b[i] ?? 0)))
}

/**
 * The product of two polynomials (numpy.polynomial.polynomial.polymul): the convolution of the coefficients.
 *
 * @param a - Coefficients, lowest first.
 * @param b - Coefficients, lowest first.
 * @returns The product.
 * @example
 * ```ts
 * import { polymul } from 'whitebox-engine/numerics/polynomial'
 * polymul([1, 1], [-1, 1]) // => [-1, 0, 1]
 * ```
 */
export function polymul(a: readonly number[], b: readonly number[]): number[] {
  const out = new Array<number>(a.length + b.length - 1).fill(0)
  a.forEach((x, i) => {
    b.forEach((y, j) => {
      out[i + j] = (out[i + j] ?? 0) + x * y
    })
  })
  return out
}

/**
 * Polynomial division with remainder (numpy.polynomial.polynomial.polydiv): a = q b + r with deg r < deg b.
 *
 * @param a - Dividend, lowest first.
 * @param b - Divisor, lowest first (non-zero).
 * @returns Quotient and remainder.
 * @example
 * ```ts
 * import { polydiv } from 'whitebox-engine/numerics/polynomial'
 * polydiv([-1, 0, 1], [1, 1]) // => { quotient: [-1, 1], remainder: [0] }
 * ```
 */
export function polydiv(a: readonly number[], b: readonly number[]): { quotient: number[]; remainder: number[] } {
  const d = polytrim(b)
  const lead = d[d.length - 1] ?? 0
  if (lead === 0) throw new WhiteboxError('value', 'polydiv: division by the zero polynomial')
  const r = [...a]
  const nq = a.length - d.length + 1
  if (nq <= 0) return { quotient: [0], remainder: polytrim(r) }
  const q = new Array<number>(nq).fill(0)
  for (let i = nq - 1; i >= 0; i--) {
    const f = (r[i + d.length - 1] ?? 0) / lead
    q[i] = f
    d.forEach((v, j) => {
      r[i + j] = (r[i + j] ?? 0) - f * v
    })
  }
  return { quotient: q, remainder: polytrim(r.slice(0, Math.max(1, d.length - 1))) }
}

/**
 * The companion matrix of a polynomial (numpy.polynomial.polynomial.polycompanion's unscaled form): its eigenvalues
 * are the roots.
 *
 * @param c - Coefficients, lowest first (degree >= 1).
 * @returns `[n, n]` with n the degree.
 * @example
 * ```ts
 * import { companion } from 'whitebox-engine/numerics/polynomial'
 * import { toArray } from 'whitebox-engine/tensor'
 * toArray(companion([-2, 0, 1])) // => [[0, 2], [1, 0]]
 * ```
 */
export function companion(c: readonly number[]): Tensor {
  const p = polytrim(c)
  const n = p.length - 1
  if (n < 1) throw new WhiteboxError('value', 'companion: degree at least 1 is needed')
  const lead = p[n] ?? 1
  const m = new Float64Array(n * n)
  for (let i = 1; i < n; i++) m[i * n + i - 1] = 1
  for (let i = 0; i < n; i++) m[i * n + n - 1] = -(p[i] ?? 0) / lead
  return fromStorage(m, [n, n], 'float64')
}

/** Complex roots as parallel real and imaginary parts. */
export interface ComplexRoots {
  readonly re: number[]
  readonly im: number[]
}

/**
 * The roots of a polynomial as eigenvalues of its companion matrix (numpy.polynomial.polynomial.polyroots), sorted by
 * real then imaginary part as numpy.
 *
 * @param c - Coefficients, lowest first.
 * @returns Real and imaginary parts.
 * @concept polynomial-roots
 * @example
 * ```ts
 * import { polyroots } from 'whitebox-engine/numerics/polynomial'
 * polyroots([-2, 0, 1]).re.map((v) => Math.round(v * 1e12) / 1e12) // => [-1.414213562373, 1.414213562373]
 * ```
 */
export function polyroots(c: readonly number[]): ComplexRoots {
  const p = polytrim(c)
  const n = p.length - 1
  if (n < 1) return { re: [], im: [] }
  if (n === 1) return { re: [-(p[0] ?? 0) / (p[1] ?? 1)], im: [0] }
  const { re, im } = eigvals(companion(p))
  const pairs = toFlat(re).map((r, i) => [r, toFlat(im)[i] ?? 0] as const)
  pairs.sort((a, b) => a[0] - b[0] || a[1] - b[1])
  return { re: pairs.map((q) => q[0]), im: pairs.map((q) => q[1]) }
}

/**
 * The monic polynomial with the given real roots (numpy.polynomial.polynomial.polyfromroots), or with complex roots
 * in conjugate pairs (the result is then real).
 *
 * @param roots - Real roots, or real and imaginary parts.
 * @returns Coefficients, lowest first.
 * @example
 * ```ts
 * import { polyFromRoots } from 'whitebox-engine/numerics/polynomial'
 * polyFromRoots([1, -1]) // => [-1, 0, 1]
 * ```
 */
export function polyFromRoots(roots: readonly number[] | ComplexRoots): number[] {
  const re = Array.isArray(roots) ? (roots as readonly number[]) : (roots as ComplexRoots).re
  const im = Array.isArray(roots) ? re.map(() => 0) : (roots as ComplexRoots).im
  // Multiply (x - r) factors in complex arithmetic.
  let pr = [1]
  let pi = [0]
  re.forEach((r, k) => {
    const s = im[k] ?? 0
    const nr = new Array<number>(pr.length + 1).fill(0)
    const ni = new Array<number>(pr.length + 1).fill(0)
    pr.forEach((a, i) => {
      const b = pi[i] ?? 0
      nr[i + 1] = (nr[i + 1] ?? 0) + a
      ni[i + 1] = (ni[i + 1] ?? 0) + b
      nr[i] = (nr[i] ?? 0) - (a * r - b * s)
      ni[i] = (ni[i] ?? 0) - (a * s + b * r)
    })
    pr = nr
    pi = ni
  })
  return pr
}
