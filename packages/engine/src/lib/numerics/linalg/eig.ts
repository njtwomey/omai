/**
 * Eigenvalues of general (non-symmetric) real matrices: Hessenberg reduction, the real Schur form by Francis
 * double-shift QR, `eig` and `eigvals` as in numpy.linalg (complex results as real and imaginary parts), and the
 * classic power, inverse and Rayleigh quotient iterations with their histories for convergence figures, each also a
 * steppable algorithm (`powerIterationAlgorithm`, `inverseIterationAlgorithm`, `rayleighQuotientIterationAlgorithm`).
 */

import { MdmlError } from '../../base/index.ts'
import { type Key, key as makeKey, normal } from '../../random/index.ts'
import { concreteOrThrow, fromStorage, promoteFloat, type Tensor, type TensorLike, toFlat } from '../../tensor/index.ts'
import {
  type Algorithm,
  type Contract,
  defineContract,
  type RunStatus,
  runAlgorithm,
  type StepContext,
  scalar,
  stateFromJson,
  stateToJson,
  tensor,
} from '../../trace/index.ts'
import { solveKernel } from './_kernels.ts'
import { squareSize } from './_matrix.ts'
import { hessenbergKernel, schurEigenvalues, schurEigenvectors, schurKernel } from './_schur.ts'
import { dgemv, dnrm2, ddot as dot } from './blas.ts'

/** Complex values as their real and imaginary parts. */
export interface ComplexTensor {
  readonly re: Tensor
  readonly im: Tensor
}

function concreteSquare(a: TensorLike, name: string): { data: Float64Array; n: number } {
  const [v] = promoteFloat(a)
  const t = concreteOrThrow(v as Tensor, name)
  if (t.shape.length !== 2) throw new MdmlError('shape', `${name}: expected a single square matrix`)
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
 * import { hessenberg } from 'mdml-engine/numerics/linalg'
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
 * @throws MdmlError with code `convergence` if the QR iteration stalls.
 * @concept eigenvalues
 * @example
 * ```ts
 * import { schur } from 'mdml-engine/numerics/linalg'
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
 * import { eigvals } from 'mdml-engine/numerics/linalg'
 * import { toFlat } from 'mdml-engine/tensor'
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
 * @index function Eigenvalues and eigenvectors of a general real matrix.
 * @example
 * ```ts
 * import { eig } from 'mdml-engine/numerics/linalg'
 * import { toFlat } from 'mdml-engine/tensor'
 * toFlat(eig([[2, 1], [0, 3]]).values.re).sort((a, b) => a - b) // => [2, 3]
 * ```
 */
export function eig(a: TensorLike): { readonly values: ComplexTensor; readonly vectors: ComplexTensor } {
  const { data, n } = concreteSquare(a, 'eig')
  const { t, z } = schurKernel(data, n)
  const { re, im } = schurEigenvalues(t, n)
  const { re: vr, im: vi } = schurEigenvectors(t, z, n, re, im)
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
  /** `converged`, or `completed` when the iteration cap was reached. */
  readonly status: RunStatus
  /** The eigenvalue estimate after each iteration. */
  readonly history: readonly number[]
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

/** The state of the eigenvalue iterations: the unit vector, the eigenvalue estimate after each step and the history. */
export interface EigenIterationState {
  /** The current unit vector. */
  readonly x: readonly number[]
  /** The eigenvalue estimate (the Rayleigh quotient); `null` before the first step of power and inverse iteration. */
  readonly value: number | null
  readonly iteration: number
  /** The estimate after each iteration (Rayleigh quotient iteration: starting with that of the start vector). */
  readonly history: readonly number[]
  readonly converged: boolean
}

/** The paths the eigenvalue iterations log, as a type. */
export type EigenIterationPath = 'eig/vector' | 'eig/value' | 'eig/residual' | 'eig/angle' | 'eig/shift'

/**
 * What the eigenvalue iterations log, by path: the unit vector, the eigenvalue estimate, the residual |A x - value x|, the
 * angle between this iterate and the last (in radians, ignoring the sign) and the shift in use.
 */
export const eigenIterationContract: Contract<EigenIterationState, EigenIterationPath> = defineContract({
  'eig/vector': tensor({ role: 'state', shape: ['n'], description: 'the current unit vector' }),
  'eig/value': scalar({ description: 'the eigenvalue estimate (the Rayleigh quotient of the vector)' }),
  'eig/residual': scalar({ description: 'the norm of A x - value x', direction: 'lower' }),
  'eig/angle': scalar({ description: 'the angle between this iterate and the last, in radians', direction: 'lower' }),
  'eig/shift': scalar({ description: 'the shift of the shifted solve (inverse and Rayleigh quotient iteration)' }),
})

function logIterate(
  ctx: StepContext<EigenIterationPath>,
  data: Float64Array,
  n: number,
  before: readonly number[],
  after: readonly number[],
  value: number,
  shift: number | undefined,
): void {
  if (ctx.wants('eig/vector')) ctx.log('eig/vector', Float64Array.from(after))
  if (ctx.wants('eig/value')) ctx.log('eig/value', value)
  if (ctx.wants('eig/residual')) {
    const ax = dgemv(data, n, n, Float64Array.from(after))
    ctx.log('eig/residual', dnrm2(ax.map((v, i) => v - value * (after[i] ?? 0))))
  }
  if (ctx.wants('eig/angle'))
    ctx.log('eig/angle', Math.acos(Math.min(1, Math.abs(dot(Float64Array.from(before), Float64Array.from(after))))))
  if (shift !== undefined && ctx.wants('eig/shift')) ctx.log('eig/shift', shift)
}

const statusOfIteration = (s: EigenIterationState, max: number): RunStatus | undefined =>
  s.converged ? 'converged' : s.iteration >= max ? 'completed' : undefined

function iterationResult(s: EigenIterationState, converged: boolean, iterations: number): EigenIterationResult {
  return {
    value: s.value ?? Number.NaN,
    vector: fromStorage(Float64Array.from(s.x), [s.x.length], 'float64'),
    iterations,
    converged,
    status: converged ? 'converged' : 'completed',
    history: s.history,
  }
}

/**
 * Power iteration as a steppable algorithm: each step applies A to the unit vector, takes the Rayleigh quotient of the
 * old vector against the product as the estimate and normalises the product. It converges to the eigenvector of the
 * eigenvalue of largest magnitude, at the rate |lambda_2 / lambda_1|, and stops when the estimate changes by less than
 * the tolerance.
 *
 * @param a - Square matrix.
 * @param options - Iterations, tolerance on the change of the estimate, start vector or key.
 * @returns The algorithm; its state is an {@link EigenIterationState}.
 * @throws MdmlError with code `shape` for a matrix that is not square.
 * @concept eigenvalues
 * @example
 * ```ts
 * import { powerIterationAlgorithm } from 'mdml-engine/numerics/linalg'
 * import { runAlgorithm } from 'mdml-engine/trace'
 * const s = runAlgorithm(powerIterationAlgorithm([[2, 0], [0, 1]], { x0: [1, 1] }), { maxSteps: 100 }).state
 * Math.round((s.value ?? 0) * 1e8) / 1e8 // => 2
 * ```
 */
export function powerIterationAlgorithm(
  a: TensorLike,
  options: EigenIterationOptions = {},
): Algorithm<EigenIterationState, EigenIterationPath> {
  const { data, n } = concreteSquare(a, 'powerIteration')
  const tol = options.tol ?? 1e-12
  const max = options.maxIterations ?? 1000
  return {
    name: 'power-iteration',
    config: { n, tol, maxIterations: max },
    init: () => ({ x: Array.from(start(n, options)), value: null, iteration: 0, history: [], converged: false }),
    step: (s, ctx) => {
      const y = dgemv(data, n, n, Float64Array.from(s.x))
      const next = dot(Float64Array.from(s.x), y)
      const x = Array.from(normalise(y))
      const converged = s.value !== null && Math.abs(next - s.value) <= tol * Math.max(1, Math.abs(next))
      logIterate(ctx, data, n, s.x, x, next, undefined)
      return { x, value: next, iteration: s.iteration + 1, history: [...s.history, next], converged }
    },
    done: (s) => statusOfIteration(s, max),
    serialise: (s) => stateToJson(s),
    deserialise: (j) => stateFromJson(j) as EigenIterationState,
  }
}

/**
 * Inverse iteration with a fixed shift as a steppable algorithm: each step solves (A - shift I) y = x and normalises y,
 * which converges to the eigenvector of the eigenvalue closest to the shift; the estimate is the Rayleigh quotient.
 *
 * @param a - Square matrix.
 * @param shift - The shift.
 * @param options - Iterations, tolerance, start vector or key.
 * @returns The algorithm; its state is an {@link EigenIterationState}.
 * @throws MdmlError with code `shape` for a matrix that is not square.
 * @concept eigenvalues
 * @example
 * ```ts
 * import { inverseIterationAlgorithm } from 'mdml-engine/numerics/linalg'
 * import { runAlgorithm } from 'mdml-engine/trace'
 * const s = runAlgorithm(inverseIterationAlgorithm([[2, 0], [0, 1]], 0.9, { x0: [1, 1] }), { maxSteps: 100 }).state
 * Math.round((s.value ?? 0) * 1e8) / 1e8 // => 1
 * ```
 */
export function inverseIterationAlgorithm(
  a: TensorLike,
  shift: number,
  options: EigenIterationOptions = {},
): Algorithm<EigenIterationState, EigenIterationPath> {
  const { data, n } = concreteSquare(a, 'inverseIteration')
  const shifted = Float64Array.from(data)
  for (let i = 0; i < n; i++) shifted[i * n + i] = (shifted[i * n + i] ?? 0) - shift
  const tol = options.tol ?? 1e-12
  const max = options.maxIterations ?? 200
  return {
    name: 'inverse-iteration',
    config: { n, shift, tol, maxIterations: max },
    init: () => ({ x: Array.from(start(n, options)), value: null, iteration: 0, history: [], converged: false }),
    step: (s, ctx) => {
      const x = Array.from(normalise(solveKernel(shifted, Float64Array.from(s.x), n, 1)))
      const next = dot(Float64Array.from(x), dgemv(data, n, n, Float64Array.from(x)))
      const converged = s.value !== null && Math.abs(next - s.value) <= tol * Math.max(1, Math.abs(next))
      logIterate(ctx, data, n, s.x, x, next, shift)
      return { x, value: next, iteration: s.iteration + 1, history: [...s.history, next], converged }
    },
    done: (s) => statusOfIteration(s, max),
    serialise: (s) => stateToJson(s),
    deserialise: (j) => stateFromJson(j) as EigenIterationState,
  }
}

/**
 * Rayleigh quotient iteration as a steppable algorithm: inverse iteration with the shift updated at every step to the
 * current Rayleigh quotient, converging cubically for symmetric matrices. A shift that is an eigenvalue to working
 * precision ends the run converged.
 *
 * @param a - Square (ideally symmetric) matrix.
 * @param options - Iterations, tolerance, start vector or key.
 * @returns The algorithm; its state is an {@link EigenIterationState}.
 * @throws MdmlError with code `shape` for a matrix that is not square.
 * @concept eigenvalues
 * @example
 * ```ts
 * import { rayleighQuotientIterationAlgorithm } from 'mdml-engine/numerics/linalg'
 * import { runAlgorithm } from 'mdml-engine/trace'
 * runAlgorithm(rayleighQuotientIterationAlgorithm([[2, 1], [1, 2]], { x0: [1, 0.9] }), { maxSteps: 50 }).state.converged // => true
 * ```
 */
export function rayleighQuotientIterationAlgorithm(
  a: TensorLike,
  options: EigenIterationOptions = {},
): Algorithm<EigenIterationState, EigenIterationPath> {
  const { data, n } = concreteSquare(a, 'rayleighQuotientIteration')
  const tol = options.tol ?? 1e-13
  const max = options.maxIterations ?? 50
  return {
    name: 'rayleigh-quotient-iteration',
    config: { n, tol, maxIterations: max },
    init: () => {
      const x = start(n, options)
      const value = dot(x, dgemv(data, n, n, x))
      return { x: Array.from(x), value, iteration: 0, history: [value], converged: false }
    },
    step: (s, ctx) => {
      const value = s.value ?? 0
      const shifted = Float64Array.from(data)
      for (let i = 0; i < n; i++) shifted[i * n + i] = (shifted[i * n + i] ?? 0) - value
      let y: Float64Array
      try {
        y = solveKernel(shifted, Float64Array.from(s.x), n, 1)
      } catch {
        // The shift is an eigenvalue to working precision.
        logIterate(ctx, data, n, s.x, s.x, value, value)
        return { ...s, iteration: s.iteration + 1, converged: true }
      }
      const x = Array.from(normalise(y))
      const next = dot(Float64Array.from(x), dgemv(data, n, n, Float64Array.from(x)))
      const converged = Math.abs(next - value) <= tol * Math.max(1, Math.abs(next))
      logIterate(ctx, data, n, s.x, x, next, value)
      return { x, value: next, iteration: s.iteration + 1, history: [...s.history, next], converged }
    },
    done: (s) => statusOfIteration(s, max),
    serialise: (s) => stateToJson(s),
    deserialise: (j) => stateFromJson(j) as EigenIterationState,
  }
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
 * import { powerIteration } from 'mdml-engine/numerics/linalg'
 * Math.round(powerIteration([[2, 0], [0, 1]]).value * 1e8) / 1e8 // => 2
 * ```
 */
export function powerIteration(a: TensorLike, options: EigenIterationOptions = {}): EigenIterationResult {
  const max = options.maxIterations ?? 1000
  const st = runAlgorithm(powerIterationAlgorithm(a, options), { maxSteps: max }).state
  return iterationResult(st, st.converged, st.history.length)
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
 * import { inverseIteration } from 'mdml-engine/numerics/linalg'
 * Math.round(inverseIteration([[2, 0], [0, 1]], 0.9).value * 1e8) / 1e8 // => 1
 * ```
 */
export function inverseIteration(
  a: TensorLike,
  shift: number,
  options: EigenIterationOptions = {},
): EigenIterationResult {
  const max = options.maxIterations ?? 200
  const st = runAlgorithm(inverseIterationAlgorithm(a, shift, options), { maxSteps: max }).state
  return iterationResult(st, st.converged, st.history.length)
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
 * import { rayleighQuotientIteration } from 'mdml-engine/numerics/linalg'
 * rayleighQuotientIteration([[2, 1], [1, 2]], { x0: [1, 0.9] }).converged // => true
 * ```
 */
export function rayleighQuotientIteration(a: TensorLike, options: EigenIterationOptions = {}): EigenIterationResult {
  const max = options.maxIterations ?? 50
  const st = runAlgorithm(rayleighQuotientIterationAlgorithm(a, options), { maxSteps: max }).state
  return iterationResult(st, st.converged, st.converged ? st.iteration : st.history.length - 1)
}
