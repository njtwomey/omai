/**
 * Eigenvalues of general (non-symmetric) real matrices: Hessenberg reduction, the real Schur form by Francis
 * double-shift QR, `eig` and `eigvals` as in numpy.linalg (complex results as real and imaginary parts), and the
 * classic power, inverse and Rayleigh quotient iterations with their histories for convergence figures.
 */

import { WhiteboxError } from '../../base/index.ts'
import { type Key, key as makeKey, normal } from '../../random/index.ts'
import { concreteOrThrow, fromStorage, promoteFloat, type Tensor, type TensorLike, toFlat } from '../../tensor/index.ts'
import { solveKernel } from './_kernels.ts'
import { squareSize } from './_matrix.ts'
import { eigenvectorKernel, hessenbergKernel, schurEigenvalues, schurKernel } from './_schur.ts'
import { dnrm2, ddot as dot } from './blas.ts'

/** Complex values as their real and imaginary parts. */
export interface ComplexTensor {
  readonly re: Tensor
  readonly im: Tensor
}

function concreteSquare(a: TensorLike, name: string): { data: Float64Array; n: number } {
  const [v] = promoteFloat(a)
  const t = concreteOrThrow(v as Tensor, name)
  if (t.shape.length !== 2) throw new WhiteboxError('shape', `${name}: expected a single square matrix`)
  return { data: Float64Array.from(t.data), n: squareSize(t.shape, name) }
}

/**
 * The upper Hessenberg form A = Q H Q^T by Householder reflections (scipy.linalg.hessenberg with calc_q).
 *
 * @param a - Square matrix.
 * @returns H (zero below the first subdiagonal) and orthogonal Q.
 * @concept eigenvalues
 * @example
 * ```ts
 * import { hessenberg } from 'whitebox-engine/numerics/linalg'
 * hessenberg([[1, 2, 3], [4, 5, 6], [7, 8, 10]]).h.data[6] // => 0
 * ```
 */
export function hessenberg(a: TensorLike): { readonly h: Tensor; readonly q: Tensor } {
  const { data, n } = concreteSquare(a, 'hessenberg')
  const { h, q } = hessenbergKernel(data, n)
  return { h: fromStorage(h, [n, n], 'float64'), q: fromStorage(q, [n, n], 'float64') }
}

/**
 * The real Schur decomposition A = Z T Z^T by Francis double-shift QR (scipy.linalg.schur, output 'real'): T is
 * quasi-upper-triangular, with 2 x 2 diagonal blocks for complex-conjugate eigenvalue pairs.
 *
 * @param a - Square matrix.
 * @returns T and orthogonal Z.
 * @throws WhiteboxError with code `convergence` if the QR iteration stalls.
 * @concept eigenvalues
 * @example
 * ```ts
 * import { schur } from 'whitebox-engine/numerics/linalg'
 * schur([[2, 0], [0, 1]]).t.data[2] // => 0
 * ```
 */
export function schur(a: TensorLike): { readonly t: Tensor; readonly z: Tensor } {
  const { data, n } = concreteSquare(a, 'schur')
  const { t, z } = schurKernel(data, n)
  return { t: fromStorage(t, [n, n], 'float64'), z: fromStorage(z, [n, n], 'float64') }
}

/**
 * The eigenvalues of a general real matrix (numpy.linalg.eigvals), from its real Schur form, in Schur order.
 *
 * @param a - Square matrix.
 * @returns Real and imaginary parts.
 * @example
 * ```ts
 * import { eigvals } from 'whitebox-engine/numerics/linalg'
 * import { toFlat } from 'whitebox-engine/tensor'
 * toFlat(eigvals([[0, -1], [1, 0]]).im).sort((a, b) => a - b) // => [-1, 1]
 * ```
 */
export function eigvals(a: TensorLike): ComplexTensor {
  const { data, n } = concreteSquare(a, 'eigvals')
  const { re, im } = schurEigenvalues(schurKernel(data, n).t, n)
  return {
    re: fromStorage(Float64Array.from(re), [n], 'float64'),
    im: fromStorage(Float64Array.from(im), [n], 'float64'),
  }
}

/**
 * Eigenvalues and right eigenvectors of a general real matrix (numpy.linalg.eig): A v = lambda v. Eigenvectors (the
 * columns) have unit length, with their largest component real and positive. Not differentiable (use `eigh` for
 * symmetric matrices).
 *
 * @param a - Square matrix.
 * @returns The eigenvalues and eigenvectors, as real and imaginary parts.
 * @concept eigenvalues
 * @example
 * ```ts
 * import { eig } from 'whitebox-engine/numerics/linalg'
 * import { toFlat } from 'whitebox-engine/tensor'
 * toFlat(eig([[2, 1], [0, 3]]).values.re).sort((a, b) => a - b) // => [2, 3]
 * ```
 */
export function eig(a: TensorLike): { readonly values: ComplexTensor; readonly vectors: ComplexTensor } {
  const { data, n } = concreteSquare(a, 'eig')
  const { re, im } = schurEigenvalues(schurKernel(data, n).t, n)
  const vr = new Float64Array(n * n)
  const vi = new Float64Array(n * n)
  re.forEach((r, k) => {
    const v = eigenvectorKernel(data, n, r, im[k] ?? 0, (m, b, size) => solveKernel(m, b, size, 1))
    for (let i = 0; i < n; i++) {
      vr[i * n + k] = v.re[i] ?? 0
      vi[i * n + k] = v.im[i] ?? 0
    }
  })
  return {
    values: {
      re: fromStorage(Float64Array.from(re), [n], 'float64'),
      im: fromStorage(Float64Array.from(im), [n], 'float64'),
    },
    vectors: { re: fromStorage(vr, [n, n], 'float64'), im: fromStorage(vi, [n, n], 'float64') },
  }
}

/** The result of an eigenvalue iteration, with its history for convergence figures. */
export interface EigenIterationResult {
  readonly value: number
  readonly vector: Tensor
  readonly iterations: number
  readonly converged: boolean
  /** The eigenvalue estimate after each iteration. */
  readonly history: readonly number[]
}

function matvec(a: Float64Array, n: number, x: Float64Array): Float64Array {
  const y = new Float64Array(n)
  for (let i = 0; i < n; i++) {
    let s = 0
    for (let j = 0; j < n; j++) s += (a[i * n + j] ?? 0) * (x[j] ?? 0)
    y[i] = s
  }
  return y
}

const normalise = (x: Float64Array): Float64Array => {
  const length = dnrm2(x)
  return x.map((v) => v / length)
}

/** Options of the eigenvalue iterations. */
export interface EigenIterationOptions {
  readonly maxIterations?: number
  readonly tol?: number
  /** The starting vector (default: Gaussian from `key`). */
  readonly x0?: TensorLike
  readonly key?: Key
}

function start(n: number, options: EigenIterationOptions): Float64Array {
  if (options.x0 !== undefined) return normalise(Float64Array.from(toFlat(promoteFloat(options.x0)[0] as Tensor)))
  return normalise(Float64Array.from(normal(options.key ?? makeKey(0), [n]).data))
}

/**
 * Power iteration: x <- A x / |A x| converges to an eigenvector of the eigenvalue of largest magnitude, at the rate
 * |lambda_2 / lambda_1|; the estimate is the Rayleigh quotient.
 *
 * @param a - Square matrix.
 * @param options - Iterations, tolerance on the change of the estimate, start vector or key.
 * @returns The dominant eigenvalue estimate, its vector, and the history.
 * @concept eigenvalues
 * @example
 * ```ts
 * import { powerIteration } from 'whitebox-engine/numerics/linalg'
 * Math.round(powerIteration([[2, 0], [0, 1]]).value * 1e8) / 1e8 // => 2
 * ```
 */
export function powerIteration(a: TensorLike, options: EigenIterationOptions = {}): EigenIterationResult {
  const { data, n } = concreteSquare(a, 'powerIteration')
  const tol = options.tol ?? 1e-12
  let x = start(n, options)
  let value = Number.NaN
  const history: number[] = []
  for (let it = 1; it <= (options.maxIterations ?? 1000); it++) {
    const y = matvec(data, n, x)
    const next = dot(x, y)
    x = normalise(y)
    history.push(next)
    if (Math.abs(next - value) <= tol * Math.max(1, Math.abs(next))) {
      return { value: next, vector: fromStorage(x, [n], 'float64'), iterations: it, converged: true, history }
    }
    value = next
  }
  return { value, vector: fromStorage(x, [n], 'float64'), iterations: history.length, converged: false, history }
}

/**
 * Inverse iteration with a fixed shift: x <- (A - shift I)^{-1} x converges to the eigenvector of the eigenvalue closest
 * to the shift.
 *
 * @param a - Square matrix.
 * @param shift - The shift.
 * @param options - Iterations, tolerance, start vector or key.
 * @returns The eigenvalue nearest the shift, its vector and the history.
 * @example
 * ```ts
 * import { inverseIteration } from 'whitebox-engine/numerics/linalg'
 * Math.round(inverseIteration([[2, 0], [0, 1]], 0.9).value * 1e8) / 1e8 // => 1
 * ```
 */
export function inverseIteration(
  a: TensorLike,
  shift: number,
  options: EigenIterationOptions = {},
): EigenIterationResult {
  const { data, n } = concreteSquare(a, 'inverseIteration')
  const shifted = Float64Array.from(data)
  for (let i = 0; i < n; i++) shifted[i * n + i] = (shifted[i * n + i] ?? 0) - shift
  const tol = options.tol ?? 1e-12
  let x = start(n, options)
  let value = Number.NaN
  const history: number[] = []
  for (let it = 1; it <= (options.maxIterations ?? 200); it++) {
    x = normalise(solveKernel(shifted, x, n, 1))
    const next = dot(x, matvec(data, n, x))
    history.push(next)
    if (Math.abs(next - value) <= tol * Math.max(1, Math.abs(next))) {
      return { value: next, vector: fromStorage(x, [n], 'float64'), iterations: it, converged: true, history }
    }
    value = next
  }
  return { value, vector: fromStorage(x, [n], 'float64'), iterations: history.length, converged: false, history }
}

/**
 * Rayleigh quotient iteration: inverse iteration with the shift updated to the current Rayleigh quotient, converging
 * cubically for symmetric matrices.
 *
 * @param a - Square (ideally symmetric) matrix.
 * @param options - Iterations, tolerance, start vector or key.
 * @returns An eigenvalue, its vector and the history.
 * @example
 * ```ts
 * import { rayleighQuotientIteration } from 'whitebox-engine/numerics/linalg'
 * rayleighQuotientIteration([[2, 1], [1, 2]], { x0: [1, 0.9] }).converged // => true
 * ```
 */
export function rayleighQuotientIteration(a: TensorLike, options: EigenIterationOptions = {}): EigenIterationResult {
  const { data, n } = concreteSquare(a, 'rayleighQuotientIteration')
  const tol = options.tol ?? 1e-13
  let x = start(n, options)
  let value = dot(x, matvec(data, n, x))
  const history: number[] = [value]
  for (let it = 1; it <= (options.maxIterations ?? 50); it++) {
    const shifted = Float64Array.from(data)
    for (let i = 0; i < n; i++) shifted[i * n + i] = (shifted[i * n + i] ?? 0) - value
    let y: Float64Array
    try {
      y = solveKernel(shifted, x, n, 1)
    } catch {
      // The shift is an eigenvalue to working precision.
      return { value, vector: fromStorage(x, [n], 'float64'), iterations: it, converged: true, history }
    }
    x = normalise(y)
    const next = dot(x, matvec(data, n, x))
    history.push(next)
    if (Math.abs(next - value) <= tol * Math.max(1, Math.abs(next))) {
      return { value: next, vector: fromStorage(x, [n], 'float64'), iterations: it, converged: true, history }
    }
    value = next
  }
  return { value, vector: fromStorage(x, [n], 'float64'), iterations: history.length - 1, converged: false, history }
}
