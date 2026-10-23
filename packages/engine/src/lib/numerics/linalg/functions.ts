import { WhiteboxError } from '../../base/index.ts'
import {
  abs,
  add,
  concat,
  div,
  item,
  matmul,
  max,
  mul,
  primalOf,
  promoteFloat,
  sliceAxis,
  sub,
  sum,
  type TensorLike,
  type Value,
  zeros,
} from '../../tensor/index.ts'
import { identity, mT, squareSize } from './_matrix.ts'
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
 * @example
 * ```ts
 * import { expm } from 'whitebox-engine/numerics/linalg'
 * import { toArray } from 'whitebox-engine/tensor'
 * toArray(expm([[0, 0], [0, 0]])) // => [[1, 0], [0, 1]]
 * ```
 */
export function expm(a: TensorLike): Value {
  const [m] = promoteFloat(a) as [Value]
  const n = squareSize(m.shape, 'expm')
  const eye = identity(n, m)
  const nrm = norm1(m)
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
 * import { expmFrechet } from 'whitebox-engine/numerics/linalg'
 * import { toArray } from 'whitebox-engine/tensor'
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
 * @throws WhiteboxError with code `convergence` when the iteration does not converge.
 * @concept matrix-functions
 * @example
 * ```ts
 * import { sqrtm } from 'whitebox-engine/numerics/linalg'
 * import { toArray } from 'whitebox-engine/tensor'
 * toArray(sqrtm([[4, 0], [0, 9]])) // => [[2, 0], [0, 3]]
 * ```
 */
export function sqrtm(a: TensorLike, options: MatrixIterationOptions = {}): Value {
  const [m] = promoteFloat(a) as [Value]
  const n = squareSize(m.shape, 'sqrtm')
  let y = m
  let z = identity(n, m)
  const tol = options.tol ?? 1e-14
  for (let i = 0; i < (options.maxIterations ?? 100); i++) {
    const yi = inv(y)
    const zi = inv(z)
    const next = mul(add(y, zi), 0.5)
    z = mul(add(z, yi), 0.5)
    const change = norm1(sub(next, y)) / Math.max(norm1(next), 1e-300)
    y = next
    if (change <= tol) return y
  }
  throw new WhiteboxError('convergence', 'sqrtm: the Denman-Beavers iteration did not converge')
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
 * import { logm } from 'whitebox-engine/numerics/linalg'
 * import { toArray } from 'whitebox-engine/tensor'
 * toArray(logm([[1, 0], [0, 1]])) // => [[0, 0], [0, 0]]
 * ```
 */
export function logm(a: TensorLike): Value {
  const [m] = promoteFloat(a) as [Value]
  const n = squareSize(m.shape, 'logm')
  const eye = identity(n, m)
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
 * import { funmSymmetric } from 'whitebox-engine/numerics/linalg'
 * import { toArray, square } from 'whitebox-engine/tensor'
 * toArray(funmSymmetric([[2, 0], [0, 3]], (w) => square(w))) // => [[4, 0], [0, 9]]
 * ```
 */
export function funmSymmetric(a: TensorLike, f: (eigenvalues: Value) => Value): Value {
  const { eigenvalues, eigenvectors } = eigh(a)
  return matmul(mul(eigenvectors, f(eigenvalues)), mT(eigenvectors))
}
