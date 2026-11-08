/**
 * Polynomials in the power basis, as `numpy.polynomial.polynomial`: Horner evaluation (differentiable in points and
 * coefficients), Vandermonde matrices and least-squares fits, derivatives and integrals, arithmetic and division,
 * companion matrices and roots, polynomials from roots, and the characteristic polynomial of a matrix with an exact
 * derivative rule.
 */

import { customJvp } from '../../autodiff/index.ts'
import { OmaiError } from '../../base/index.ts'
import {
  add,
  concat,
  div,
  eye,
  fromStorage,
  isValue,
  matmul,
  mul,
  neg,
  primalOf,
  promoteFloat,
  reshape,
  sliceAxis,
  sqrt,
  sum,
  type Tensor,
  type TensorLike,
  toFlat,
  toVector,
  transpose,
  type Value,
  where,
  zeros,
} from '../../tensor/index.ts'
import { matrixOf, vectorOf } from '../_vector.ts'
import { eigvals, lstsq } from '../linalg/index.ts'
import {
  addKernel,
  companionKernel,
  derKernel,
  divKernel,
  fromRootsKernel,
  intKernel,
  mulKernel,
  rootsKernel,
  subKernel,
  trimKernel,
} from './_power.ts'

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
 * import { polyval } from 'omai-engine/numerics/polynomial'
 * import { toArray } from 'omai-engine/tensor'
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
 * import { polyvander } from 'omai-engine/numerics/polynomial'
 * import { toArray } from 'omai-engine/tensor'
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
 * @index function Least-squares polynomial fit in the power basis.
 * @example
 * ```ts
 * import { polyfit } from 'omai-engine/numerics/polynomial'
 * import { toArray } from 'omai-engine/tensor'
 * toArray(polyfit([0, 1, 2, 3], [1, 3, 5, 7], 1)).map((v) => Math.round(v * 1e12) / 1e12) // => [1, 2]
 * ```
 */
export function polyfit(x: TensorLike, y: TensorLike, deg: number, w?: TensorLike): Value {
  return weightedFit(polyvander(x, deg), y, w)
}

/**
 * Weighted least squares with column scaling (numpy's *fit functions).
 *
 * @param vander - The design matrix `[n, m]`.
 * @param y - Targets `[n]` or `[n, k]`.
 * @param w - Optional weights `[n]`.
 * @returns The coefficients `[m]` or `[m, k]`.
 * @example
 * ```ts
 * import { polyvander, weightedFit } from 'omai-engine/numerics/polynomial'
 * import { toArray } from 'omai-engine/tensor'
 * (toArray(weightedFit(polyvander([0, 1, 2], 1), [1, 3, 5])) as number[]).map((v) => Math.round(v * 1e12) / 1e12) // => [1, 2]
 * ```
 */
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

/**
 * Drop trailing (highest-degree) zero coefficients, keeping at least one. Takes and returns tensors (migration, D-009:
 * `c` was `readonly number[]` and the result `number[]`); not differentiable (a traced `c` is rejected).
 *
 * @param c - Coefficients `[k]`, lowest first.
 * @param tol - Coefficients of absolute value at most `tol` count as zero (default 0).
 * @returns The trimmed coefficients `[k']`.
 * @throws OmaiError with code `shape` unless `c` is a vector, and `not-differentiable` for a traced value.
 * @example
 * ```ts
 * import { polytrim } from 'omai-engine/numerics/polynomial'
 * polytrim([1, 2, 0, 0]) // => [1, 2]
 * ```
 */
export function polytrim(c: TensorLike, tol = 0): Tensor<'float64', [number]> {
  return vectorOf(trimKernel(toVector(c, 'polytrim'), tol))
}

/**
 * The m-th derivative of a polynomial (numpy.polynomial.polynomial.polyder), scaled by scl^m. Takes and returns tensors
 * (migration, D-009: was `readonly number[]` to `number[]`); not differentiable (a traced `c` is rejected).
 *
 * @param c - Coefficients `[k]`, lowest first.
 * @param m - Order (default 1).
 * @param scl - Scale per differentiation (default 1).
 * @returns Coefficients of the derivative.
 * @throws OmaiError with code `shape` unless `c` is a vector, and `not-differentiable` for a traced value.
 * @example
 * ```ts
 * import { polyder } from 'omai-engine/numerics/polynomial'
 * polyder([1, 2, 3, 4]) // => [2, 6, 12]
 * ```
 */
export function polyder(c: TensorLike, m = 1, scl = 1): Tensor<'float64', [number]> {
  return vectorOf(derKernel(toVector(c, 'polyder'), m, scl))
}

/**
 * The m-th integral of a polynomial (numpy.polynomial.polynomial.polyint) with integration constants k (one per
 * integration, set so the integral is k at lbnd), each integration scaled by scl. Takes and returns tensors
 * (migration, D-009: `c` and `k` were `readonly number[]`, the result `number[]`); not differentiable.
 *
 * @param c - Coefficients `[k]`, lowest first.
 * @param m - Order (default 1).
 * @param k - Integration constants `[m]` (default zeros).
 * @param lbnd - Lower bound (default 0).
 * @param scl - Scale (default 1).
 * @returns Coefficients of the integral.
 * @throws OmaiError with code `shape` unless `c` and `k` are vectors, and `not-differentiable` for a traced value.
 * @example
 * ```ts
 * import { polyint } from 'omai-engine/numerics/polynomial'
 * polyint([2, 6, 12]) // => [0, 2, 3, 4]
 * ```
 */
export function polyint(c: TensorLike, m = 1, k: TensorLike = [], lbnd = 0, scl = 1): Tensor<'float64', [number]> {
  return vectorOf(intKernel(toVector(c, 'polyint'), m, toVector(k, 'polyint'), lbnd, scl))
}

/**
 * The sum of two polynomials (numpy.polynomial.polynomial.polyadd). Takes and returns tensors (migration, D-009: was
 * `readonly number[]` to `number[]`); not differentiable (a traced input is rejected).
 *
 * @param a - Coefficients `[p]`, lowest first.
 * @param b - Coefficients `[q]`, lowest first.
 * @returns The sum, trimmed of trailing zeros.
 * @throws OmaiError with code `shape` unless `a` and `b` are vectors, and `not-differentiable` for a traced value.
 * @example
 * ```ts
 * import { polyadd } from 'omai-engine/numerics/polynomial'
 * polyadd([1, 2], [3, 4, 5]) // => [4, 6, 5]
 * ```
 */
export function polyadd(a: TensorLike, b: TensorLike): Tensor<'float64', [number]> {
  return vectorOf(addKernel(toVector(a, 'polyadd'), toVector(b, 'polyadd')))
}

/**
 * The difference of two polynomials (numpy.polynomial.polynomial.polysub). Takes and returns tensors (migration, D-009:
 * was `readonly number[]` to `number[]`); not differentiable (a traced input is rejected).
 *
 * @param a - Coefficients `[p]`, lowest first.
 * @param b - Coefficients `[q]`, lowest first.
 * @returns a - b, trimmed of trailing zeros.
 * @throws OmaiError with code `shape` unless `a` and `b` are vectors, and `not-differentiable` for a traced value.
 * @example
 * ```ts
 * import { polysub } from 'omai-engine/numerics/polynomial'
 * polysub([1, 2, 3], [1, 2]) // => [0, 0, 3]
 * ```
 */
export function polysub(a: TensorLike, b: TensorLike): Tensor<'float64', [number]> {
  return vectorOf(subKernel(toVector(a, 'polysub'), toVector(b, 'polysub')))
}

/**
 * The product of two polynomials (numpy.polynomial.polynomial.polymul): the convolution of the coefficients. Takes and
 * returns tensors (migration, D-009: was `readonly number[]` to `number[]`); not differentiable (a traced input is
 * rejected).
 *
 * @param a - Coefficients `[p]`, lowest first.
 * @param b - Coefficients `[q]`, lowest first.
 * @returns The product `[p + q - 1]`.
 * @throws OmaiError with code `shape` unless `a` and `b` are vectors, and `not-differentiable` for a traced value.
 * @example
 * ```ts
 * import { polymul } from 'omai-engine/numerics/polynomial'
 * polymul([1, 1], [-1, 1]) // => [-1, 0, 1]
 * ```
 */
export function polymul(a: TensorLike, b: TensorLike): Tensor<'float64', [number]> {
  return vectorOf(mulKernel(toVector(a, 'polymul'), toVector(b, 'polymul')))
}

/** The quotient and remainder of {@link polydiv}. */
export interface PolynomialDivision {
  readonly quotient: Tensor<'float64', [number]>
  readonly remainder: Tensor<'float64', [number]>
}

/**
 * Polynomial division with remainder (numpy.polynomial.polynomial.polydiv): a = q b + r with deg r < deg b. Takes and
 * returns tensors (migration, D-009: was `readonly number[]` to `{ quotient: number[]; remainder: number[] }`); not
 * differentiable (a traced input is rejected).
 *
 * @param a - Dividend `[p]`, lowest first.
 * @param b - Divisor `[q]`, lowest first (non-zero).
 * @returns Quotient and remainder.
 * @throws OmaiError with code `value` for the zero divisor, `shape` unless `a` and `b` are vectors, and
 *   `not-differentiable` for a traced value.
 * @example
 * ```ts
 * import { polydiv } from 'omai-engine/numerics/polynomial'
 * polydiv([-1, 0, 1], [1, 1]) // => { quotient: [-1, 1], remainder: [0] }
 * ```
 */
export function polydiv(a: TensorLike, b: TensorLike): PolynomialDivision {
  const { quotient, remainder } = divKernel(toVector(a, 'polydiv'), toVector(b, 'polydiv'))
  return { quotient: vectorOf(quotient), remainder: vectorOf(remainder) }
}

/**
 * The companion matrix of a polynomial (numpy.polynomial.polynomial.polycompanion's unscaled form): its eigenvalues
 * are the roots. Takes a tensor (migration, D-009: `c` was `readonly number[]`); not differentiable.
 *
 * @param c - Coefficients `[k]`, lowest first (degree >= 1).
 * @returns `[n, n]` with n the degree.
 * @throws OmaiError with code `value` below degree 1, `shape` unless `c` is a vector, and `not-differentiable` for a
 *   traced value.
 * @example
 * ```ts
 * import { companion } from 'omai-engine/numerics/polynomial'
 * import { toArray } from 'omai-engine/tensor'
 * toArray(companion([-2, 0, 1])) // => [[0, 2], [1, 0]]
 * ```
 */
export function companion(c: TensorLike): Tensor<'float64', [number, number]> {
  const { data, n } = companionKernel(toVector(c, 'companion'))
  return matrixOf(data, n, n)
}

/** Complex roots as parallel real and imaginary parts, each a vector. */
export interface ComplexRoots {
  readonly re: Tensor<'float64', [number]>
  readonly im: Tensor<'float64', [number]>
}

/**
 * The roots of a polynomial as eigenvalues of its companion matrix (numpy.polynomial.polynomial.polyroots), sorted by
 * real then imaginary part as numpy. Takes a tensor and returns tensors (migration, D-009: `c` was `readonly number[]`,
 * the parts `number[]`); not differentiable.
 *
 * @param c - Coefficients `[k]`, lowest first.
 * @returns Real and imaginary parts, each `[n]` with n the degree.
 * @throws OmaiError with code `shape` unless `c` is a vector, and `not-differentiable` for a traced value.
 * @concept polynomial-roots
 * @example
 * ```ts
 * import { polyroots } from 'omai-engine/numerics/polynomial'
 * import { toFlat } from 'omai-engine/tensor'
 * toFlat(polyroots([-2, 0, 1]).re).map((v) => Math.round(v * 1e12) / 1e12) // => [-1.414213562373, 1.414213562373]
 * ```
 */
export function polyroots(c: TensorLike): ComplexRoots {
  const { re, im } = rootsKernel(toVector(c, 'polyroots'))
  return { re: vectorOf(re), im: vectorOf(im) }
}

/**
 * The monic polynomial with the given real roots (numpy.polynomial.polynomial.polyfromroots), or with complex roots
 * in conjugate pairs (the result is then real). Takes tensors and returns a tensor (migration, D-009: the roots were
 * `readonly number[]` or `{ re: number[]; im: number[] }`, the result `number[]`); not differentiable.
 *
 * @param roots - Real roots `[n]`, or real and imaginary parts (each `[n]`).
 * @returns Coefficients `[n + 1]`, lowest first.
 * @throws OmaiError with code `shape` unless the roots are vectors, and `not-differentiable` for a traced value.
 * @example
 * ```ts
 * import { polyFromRoots } from 'omai-engine/numerics/polynomial'
 * polyFromRoots([1, -1]) // => [-1, 0, 1]
 * ```
 */
export function polyFromRoots(
  roots: TensorLike | { readonly re: TensorLike; readonly im: TensorLike },
): Tensor<'float64', [number]> {
  if (isComplexRoots(roots)) {
    return vectorOf(fromRootsKernel(toVector(roots.re, 'polyFromRoots'), toVector(roots.im, 'polyFromRoots')))
  }
  const re = toVector(roots, 'polyFromRoots')
  return vectorOf(fromRootsKernel(re, new Array<number>(re.length).fill(0)))
}

function isComplexRoots(x: unknown): x is { readonly re: TensorLike; readonly im: TensorLike } {
  return typeof x === 'object' && x !== null && !Array.isArray(x) && !isValue(x) && 're' in x && 'im' in x
}

const charpolyPrimitive = customJvp({
  name: 'charpoly',
  forward: (a) => {
    const n = a.shape[0] ?? 0
    const { re, im } = eigvals(a)
    return fromStorage(Float64Array.from(fromRootsKernel(toFlat(re), toFlat(im))), [n + 1], 'float64')
  },
  // With adj(xI - A) = sum_k x^(n-1-k) B_k (B_0 = I, B_k = A B_(k-1) + c_k I, c highest degree first), the
  // coefficient of x^(n-1-k) moves by -tr(B_k dA).
  jvp: ([a], [t], out) => {
    if (t === null || t === undefined) return null
    const av = a as Value
    const n = av.shape[0] ?? 0
    const identity = eye(n)
    const tT = transpose(t)
    const parts: Value[] = [zeros([1])]
    let bk: Value = identity
    for (let k = 0; k < n; k++) {
      parts.unshift(reshape(neg(sum(mul(bk, tT))), [1]))
      bk = add(matmul(av, bk), mul(identity, sliceAxis(out, 0, n - 1 - k, n - k)))
    }
    return concat(parts, 0)
  },
})

/**
 * The characteristic polynomial det(x I - A) of a square matrix (numpy.poly of a matrix), from its eigenvalues, with
 * an exact derivative rule (the adjugate by the Faddeev-LeVerrier recurrence), so differentiable in A.
 *
 * @param a - Square matrix `[n, n]`.
 * @returns The n + 1 coefficients, lowest degree first (the last is 1).
 * @concept polynomials
 * @example
 * ```ts
 * import { charpoly } from 'omai-engine/numerics/polynomial'
 * import { toArray } from 'omai-engine/tensor'
 * toArray(charpoly([[2, 0], [0, 3]])) // => [6, -5, 1]
 * ```
 */
export function charpoly(a: TensorLike): Value {
  const [av] = promoteFloat(a) as [Value]
  if (av.shape.length !== 2 || av.shape[0] !== av.shape[1]) {
    throw new OmaiError('shape', 'charpoly: expected a square matrix')
  }
  if ((av.shape[0] ?? 0) === 0) return fromStorage(new Float64Array([1]), [1], 'float64')
  return charpolyPrimitive(av)
}
