/**
 * Functions of matrices, scipy.linalg style: the matrix exponential by Pade scaling and squaring with its Frechet
 * derivative, the principal square root (Denman-Beavers) and logarithm (inverse scaling and squaring), functions of
 * symmetric matrices through the eigendecomposition, and general `funm` by the Schur-Parlett method.
 */

import { MdmlError } from '../../base/index.ts'
import {
  abs,
  add,
  concat,
  concreteOrThrow,
  div,
  fromStorage,
  item,
  matmul,
  max,
  mul,
  primalOf,
  promoteFloat,
  sliceAxis,
  sub,
  sum,
  type Tensor,
  type TensorLike,
  toFlat,
  type Value,
  zeros,
} from '../../tensor/index.ts'
import { complexOps as C, type ComplexNumber } from '../differentiate/index.ts'
import { identity, mT, namingSingular, squareSize } from './_matrix.ts'
import { schurKernel } from './_schur.ts'
import { eigh } from './eigh.ts'
import { inv, solve } from './lu.ts'

const norm1 = (a: Value): number => item(primalOf(max(sum(abs(a), 0))))

// Pade coefficients and theta thresholds of Higham (2005), "The scaling and squaring method for the matrix exponential
// revisited".
const B: Readonly<Record<number, readonly number[]>> = {
  3: [120, 60, 12, 1],
  5: [30240, 15120, 3360, 420, 30, 1],
  7: [17297280, 8648640, 1995840, 277200, 25200, 1512, 56, 1],
  9: [17643225600, 8821612800, 2075673600, 302702400, 30270240, 2162160, 110880, 3960, 90, 1],
  13: [
    64764752532480000, 32382376266240000, 7771770303897600, 1187353796428800, 129060195264000, 10559470521600,
    670442572800, 33522128640, 1323241920, 40840800, 960960, 16380, 182, 1,
  ],
}
const THETA: readonly (readonly [number, number])[] = [
  [3, 1.495585217958292e-2],
  [5, 2.53939833006323e-1],
  [7, 9.504178996162932e-1],
  [9, 2.097847961257068],
]
const THETA13 = 5.371920351148152

/**
 * The matrix exponential by Pade approximation with scaling and squaring (Higham 2005, as scipy.linalg.expm), built from
 * matrix products and a linear solve, so it is differentiable.
 *
 * @param a - Square matrix `[n, n]`.
 * @returns exp(A).
 * @concept matrix-functions
 * @index function Matrix exponential by Pade scaling and squaring.
 * @example
 * ```ts
 * import { expm } from 'mdml-engine/numerics/linalg'
 * import { toArray } from 'mdml-engine/tensor'
 * toArray(expm([[0, 0], [0, 0]])) // => [[1, 0], [0, 1]]
 * ```
 */
export function expm(a: TensorLike): Value {
  const [m] = promoteFloat(a) as [Value]
  const n = squareSize(m.shape, 'expm')
  const eye = identity(n, m)
  const nrm = norm1(m)
  // An infinite entry has no exponential (the squaring count below would be infinite, a loop that never ends).
  if (nrm === Number.POSITIVE_INFINITY) return mul(m, Number.NaN)
  for (const [degree, theta] of THETA) {
    if (nrm <= theta) {
      const b = B[degree] as readonly number[]
      const a2 = matmul(m, m)
      let u = mul(eye, b[1] ?? 0)
      let v = mul(eye, b[0] ?? 0)
      let power = eye
      for (let k = 2; k <= degree; k += 2) {
        power = matmul(power, a2)
        u = add(u, mul(power, b[k + 1] ?? 0))
        v = add(v, mul(power, b[k] ?? 0))
      }
      u = matmul(m, u)
      return solve(sub(v, u), add(v, u))
    }
  }
  const s = Math.max(0, Math.ceil(Math.log2(nrm / THETA13)))
  const x = div(m, 2 ** s)
  const b = B[13] as readonly number[]
  const a2 = matmul(x, x)
  const a4 = matmul(a2, a2)
  const a6 = matmul(a2, a4)
  const c = (k: number): number => b[k] ?? 0
  const u = matmul(
    x,
    add(
      matmul(a6, add(add(mul(a6, c(13)), mul(a4, c(11))), mul(a2, c(9)))),
      add(add(add(mul(a6, c(7)), mul(a4, c(5))), mul(a2, c(3))), mul(eye, c(1))),
    ),
  )
  const v = add(
    matmul(a6, add(add(mul(a6, c(12)), mul(a4, c(10))), mul(a2, c(8)))),
    add(add(add(mul(a6, c(6)), mul(a4, c(4))), mul(a2, c(2))), mul(eye, c(0))),
  )
  let r = solve(sub(v, u), add(v, u))
  for (let i = 0; i < s; i++) r = matmul(r, r)
  return r
}

/**
 * exp(A) and its Frechet derivative in direction E, L(A, E) = d/dt exp(A + tE) at t = 0 (scipy.linalg.expm_frechet),
 * from the exponential of the block matrix [[A, E], [0, A]].
 *
 * @param a - Square matrix.
 * @param e - Direction (same shape).
 * @returns `{ expm, frechet }`.
 * @example
 * ```ts
 * import { expmFrechet } from 'mdml-engine/numerics/linalg'
 * import { toArray } from 'mdml-engine/tensor'
 * toArray(expmFrechet([[0]], [[1]]).frechet) // => [[1]]
 * ```
 */
export function expmFrechet(a: TensorLike, e: TensorLike): { readonly expm: Value; readonly frechet: Value } {
  const [m, d] = promoteFloat(a, e) as [Value, Value]
  const n = squareSize(m.shape, 'expmFrechet')
  const block = concat([concat([m, d], 1), concat([zeros([n, n]), m], 1)], 0)
  const big = expm(block)
  const top = sliceAxis(big, 0, 0, n)
  return { expm: sliceAxis(top, 1, 0, n), frechet: sliceAxis(top, 1, n, 2 * n) }
}

/** Options of {@link sqrtm}. */
export interface MatrixIterationOptions {
  readonly maxIterations?: number
  readonly tol?: number
}

/**
 * The principal matrix square root by the Denman-Beavers iteration (for matrices with no eigenvalues on the closed
 * negative real axis), built from inverses and so differentiable: Y <- (Y + Z^{-1}) / 2, Z <- (Z + Y^{-1}) / 2 from
 * Y = A, Z = I, with Y converging to sqrt(A).
 *
 * @param a - Square matrix.
 * @param options - Iteration cap and tolerance.
 * @returns sqrt(A).
 * @throws MdmlError with code `convergence` when the iteration does not converge.
 * @concept matrix-functions
 * @example
 * ```ts
 * import { sqrtm } from 'mdml-engine/numerics/linalg'
 * import { toArray } from 'mdml-engine/tensor'
 * toArray(sqrtm([[4, 0], [0, 9]])) // => [[2, 0], [0, 3]]
 * ```
 */
export function sqrtm(a: TensorLike, options: MatrixIterationOptions = {}): Value {
  const [m] = promoteFloat(a) as [Value]
  const n = squareSize(m.shape, 'sqrtm')
  if (toFlat(primalOf(m)).every((x) => x === 0)) return m
  let y = m
  let z = identity(n, m)
  const tol = options.tol ?? 1e-14
  return namingSingular(
    'sqrtm',
    'an iterate is singular (A has an eigenvalue on the closed negative real axis, or is singular)',
    () => {
      for (let i = 0; i < (options.maxIterations ?? 100); i++) {
        const yi = inv(y)
        const zi = inv(z)
        const next = mul(add(y, zi), 0.5)
        z = mul(add(z, yi), 0.5)
        const change = norm1(sub(next, y)) / Math.max(norm1(next), 1e-300)
        y = next
        if (change <= tol) return y
      }
      throw new MdmlError('convergence', 'sqrtm: the Denman-Beavers iteration did not converge')
    },
  )
}

// Gauss-Legendre nodes and weights on [0, 1] (8 points), for log(I + X) = sum_j w_j X (I + t_j X)^{-1}.
const GL8: readonly (readonly [number, number])[] = [
  [0.019855071751231856, 0.05061426814518813],
  [0.10166676129318664, 0.11119051722668724],
  [0.2372337950418355, 0.15685332293894363],
  [0.4082826787521751, 0.181341891689181],
  [0.5917173212478249, 0.181341891689181],
  [0.7627662049581645, 0.15685332293894363],
  [0.8983332387068134, 0.11119051722668724],
  [0.9801449282487681, 0.05061426814518813],
]

/**
 * The principal matrix logarithm by inverse scaling and squaring (Higham): take square roots until A is close to I,
 * apply a Gauss-Legendre (Pade-equivalent) formula for log(I + X), and scale back. For matrices with no eigenvalues
 * on the closed negative real axis; differentiable.
 *
 * @param a - Square matrix.
 * @returns log(A).
 * @concept matrix-functions
 * @example
 * ```ts
 * import { logm } from 'mdml-engine/numerics/linalg'
 * import { toArray } from 'mdml-engine/tensor'
 * toArray(logm([[1, 0], [0, 1]])) // => [[0, 0], [0, 0]]
 * ```
 */
export function logm(a: TensorLike): Value {
  const [m] = promoteFloat(a) as [Value]
  const n = squareSize(m.shape, 'logm')
  const eye = identity(n, m)
  return namingSingular('logm', 'A has an eigenvalue on the closed negative real axis, or is singular', () => {
    let y = m
    let k = 0
    while (norm1(sub(y, eye)) > 0.25 && k < 64) {
      y = sqrtm(y)
      k += 1
    }
    const x = sub(y, eye)
    let log = mul(eye, 0)
    for (const [t, w] of GL8) log = add(log, mul(matmul(x, inv(add(eye, mul(x, t)))), w))
    return mul(log, 2 ** k)
  })
}

/**
 * A function of a symmetric matrix through its eigendecomposition: V f(Lambda) V^T (the spectral calculus; scipy's
 * funm for the symmetric case). Differentiable where the eigenvalues are distinct.
 *
 * @param a - Symmetric matrix.
 * @param f - The scalar function, applied elementwise to the eigenvalues.
 * @returns f(A).
 * @example
 * ```ts
 * import { funmSymmetric } from 'mdml-engine/numerics/linalg'
 * import { toArray, square } from 'mdml-engine/tensor'
 * toArray(funmSymmetric([[2, 0], [0, 3]], (w) => square(w))) // => [[4, 0], [0, 9]]
 * ```
 */
export function funmSymmetric(a: TensorLike, f: (eigenvalues: Value) => Value): Value {
  const { eigenvalues, eigenvectors } = eigh(a)
  return matmul(mul(eigenvectors, f(eigenvalues)), mT(eigenvectors))
}

/** A complex matrix as parallel real and imaginary parts, row-major. */
interface CMat {
  re: Float64Array
  im: Float64Array
}

/** The real Schur form converted to complex upper-triangular form, T = Z^H A Z (scipy.linalg.rsf2csf). */
function complexSchur(a: Float64Array, n: number): { t: CMat; z: CMat } {
  const { t: tr, z: zr } = schurKernel(a, n)
  const t: CMat = { re: Float64Array.from(tr), im: new Float64Array(n * n) }
  const z: CMat = { re: Float64Array.from(zr), im: new Float64Array(n * n) }
  const eps = 2.220446049250313e-16
  for (let m = n - 1; m > 0; m--) {
    const sub = t.re[m * n + m - 1] ?? 0
    if (Math.abs(sub) > eps * (Math.abs(t.re[(m - 1) * n + m - 1] ?? 0) + Math.abs(t.re[m * n + m] ?? 0))) {
      // mu = (an eigenvalue of the 2 x 2 block) - T[m, m]. Earlier rotations have made T[m - 1, m] and T[m, m]
      // complex, so the block is complex in general.
      const at = (i: number, j: number): ComplexNumber => ({ re: t.re[i * n + j] ?? 0, im: t.im[i * n + j] ?? 0 })
      const p = at(m - 1, m - 1)
      const q = at(m - 1, m)
      const r0 = at(m, m - 1)
      const s0 = at(m, m)
      const half = C.mul(C.sub(p, s0), 0.5)
      const root = C.sqrt(C.add(C.mul(half, half), C.mul(q, r0)))
      const lambda = C.add(C.mul(C.add(p, s0), 0.5), root)
      const mu = C.sub(lambda, s0)
      const r = Math.hypot(mu.re, mu.im, sub)
      const c: ComplexNumber = { re: mu.re / r, im: mu.im / r }
      const s = sub / r
      // G = [[conj(c), s], [-s, c]]; T[m-1:m+1, m-1:] = G T; T[:m+1, m-1:m+1] = T G^H; Z[:, m-1:m+1] = Z G^H.
      for (let col = m - 1; col < n; col++) {
        const x = { re: t.re[(m - 1) * n + col] ?? 0, im: t.im[(m - 1) * n + col] ?? 0 }
        const y = { re: t.re[m * n + col] ?? 0, im: t.im[m * n + col] ?? 0 }
        const top = C.add(C.mul({ re: c.re, im: -c.im }, x), C.mul(y, s))
        const bottom = C.add(C.mul(x, -s), C.mul(c, y))
        t.re[(m - 1) * n + col] = top.re
        t.im[(m - 1) * n + col] = top.im
        t.re[m * n + col] = bottom.re
        t.im[m * n + col] = bottom.im
      }
      const rightMultiply = (mat: CMat, rows: number): void => {
        for (let row = 0; row < rows; row++) {
          const x = { re: mat.re[row * n + m - 1] ?? 0, im: mat.im[row * n + m - 1] ?? 0 }
          const y = { re: mat.re[row * n + m] ?? 0, im: mat.im[row * n + m] ?? 0 }
          // [x, y] G^H with G^H = [[c, -s], [s, conj(c)]].
          const left = C.add(C.mul(x, c), C.mul(y, s))
          const right = C.add(C.mul(x, -s), C.mul(y, { re: c.re, im: -c.im }))
          mat.re[row * n + m - 1] = left.re
          mat.im[row * n + m - 1] = left.im
          mat.re[row * n + m] = right.re
          mat.im[row * n + m] = right.im
        }
      }
      rightMultiply(t, m + 1)
      rightMultiply(z, n)
    }
    t.re[m * n + m - 1] = 0
    t.im[m * n + m - 1] = 0
  }
  return { t, z }
}

/** Result of {@link funm}. */
export interface FunmResult {
  /** f(A): real when A is real and the imaginary part is negligible, otherwise a complex pair. */
  readonly value: Tensor | { readonly re: Tensor; readonly im: Tensor }
  /** scipy's estimate of the relative error, large when eigenvalues nearly coincide. */
  readonly error: number
}

/**
 * A general matrix function f(A) by the Schur-Parlett method (scipy.linalg.funm): reduce A to complex triangular Schur
 * form T, compute F = f(T) by Parlett's recurrence F_ij = (T_ij (F_jj - F_ii) + sum_k (T_ik F_kj - F_ik T_kj)) /
 * (T_jj - T_ii) (from F T = T F), and transform back. Accurate when the eigenvalues are well separated (the error estimate says so).
 *
 * @param a - Square matrix.
 * @param f - The scalar function on complex numbers (built from complexOps).
 * @returns f(A) and the error estimate.
 * @concept matrix-functions
 * @example
 * ```ts
 * import { funm } from 'mdml-engine/numerics/linalg'
 * import { complexOps as C } from 'mdml-engine/numerics/differentiate'
 * import { toArray } from 'mdml-engine/tensor'
 * toArray(funm([[1, 0], [0, 2]], C.exp).value).map((r) => r.map((v) => Math.round(v * 1e12) / 1e12)) // => [[2.718281828459, 0], [0, 7.389056098931]]
 * ```
 */
export function funm(a: TensorLike, f: (z: ComplexNumber) => ComplexNumber): FunmResult {
  const t0 = concreteOrThrow(promoteFloat(a)[0] as Tensor, 'funm')
  const n = t0.shape[0] ?? 0
  if (t0.shape.length !== 2 || t0.shape[1] !== n) throw new MdmlError('shape', 'funm: the matrix must be square')
  const { t, z } = complexSchur(Float64Array.from(t0.data), n)
  const T = (i: number, j: number): ComplexNumber => ({ re: t.re[i * n + j] ?? 0, im: t.im[i * n + j] ?? 0 })
  const F: ComplexNumber[] = new Array(n * n).fill({ re: 0, im: 0 })
  for (let i = 0; i < n; i++) F[i * n + i] = f(T(i, i))
  let minDen = Number.POSITIVE_INFINITY
  for (let d = 1; d < n; d++) {
    for (let i = 0; i + d < n; i++) {
      const j = i + d
      let s = C.mul(T(i, j), C.sub(F[j * n + j] as ComplexNumber, F[i * n + i] as ComplexNumber))
      for (let k = i + 1; k < j; k++)
        s = C.add(
          s,
          C.sub(C.mul(T(i, k), F[k * n + j] as ComplexNumber), C.mul(F[i * n + k] as ComplexNumber, T(k, j))),
        )
      const den = C.sub(T(j, j), T(i, i))
      const absDen = Math.hypot(den.re, den.im)
      minDen = Math.min(minDen, absDen)
      F[i * n + j] = absDen === 0 ? { re: Number.NaN, im: Number.NaN } : C.div(s, den)
    }
  }
  // F <- Z F Z^H.
  const re = new Float64Array(n * n)
  const im = new Float64Array(n * n)
  for (let i = 0; i < n; i++) {
    for (let j = 0; j < n; j++) {
      let acc: ComplexNumber = { re: 0, im: 0 }
      for (let k = 0; k < n; k++) {
        const zik: ComplexNumber = { re: z.re[i * n + k] ?? 0, im: z.im[i * n + k] ?? 0 }
        let inner: ComplexNumber = { re: 0, im: 0 }
        for (let l = k; l < n; l++) {
          const zjl: ComplexNumber = { re: z.re[j * n + l] ?? 0, im: -(z.im[j * n + l] ?? 0) }
          inner = C.add(inner, C.mul(F[k * n + l] as ComplexNumber, zjl))
        }
        acc = C.add(acc, C.mul(zik, inner))
      }
      re[i * n + j] = acc.re
      im[i * n + j] = acc.im
    }
  }
  // scipy's error estimate.
  const tol = 2.220446049250313e-16
  let upper = 0
  for (let j = 0; j < n; j++) {
    let col = 0
    for (let i = 0; i < j; i++) col += Math.hypot(t.re[i * n + j] ?? 0, t.im[i * n + j] ?? 0)
    upper = Math.max(upper, col)
  }
  const error = n <= 1 ? tol : Math.min(1, Math.max(tol, (tol / minDen) * upper))
  let imagNorm = 0
  let realNorm = 0
  for (let k = 0; k < n * n; k++) {
    imagNorm = Math.max(imagNorm, Math.abs(im[k] ?? 0))
    realNorm = Math.max(realNorm, Math.abs(re[k] ?? 0))
  }
  const real = imagNorm <= 1e3 * tol * Math.max(1, realNorm)
  return {
    value: real
      ? fromStorage(re, [n, n], 'float64')
      : { re: fromStorage(re, [n, n], 'float64'), im: fromStorage(im, [n, n], 'float64') },
    error,
  }
}
