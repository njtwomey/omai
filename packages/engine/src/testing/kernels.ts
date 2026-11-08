// Checks shared by the kernels tests: scikit-learn Gaussian-process cases (oracle/omai_oracle/kernels.py
// `gp_case`) and positive semi-definiteness of Gram matrices. Not published.
import type fc from 'fast-check'
import { expect } from 'vitest'
import { gramGradient, type Kernel, theta } from '../lib/kernels/index.ts'
import { eigvalsh } from '../lib/numerics/linalg/index.ts'
import { add, fromStorage, type Tensor, type TensorLike, take, toFlat, type Value } from '../lib/tensor/index.ts'
import { tensorOf } from './arbitrary.ts'
import { decode, expectMatches, type GoldenCase } from './golden.ts'

/**
 * Compare a kernel with a scikit-learn case: Gram matrix, cross-covariance, diagonal, theta and dK/dtheta. `order[i]`
 * is the index in omai's theta of scikit-learn's i-th hyperparameter (default the identity).
 */
export function expectGpCase(c: GoldenCase, kernel: Kernel, order?: readonly number[]): void {
  const [x, y] = c.inputs.map(decode) as [Value, Value]
  const out = c.output as Record<string, unknown>
  const match = (got: Value, key: string): void => expectMatches(got, out[key], c.rtol, c.atol, `${c.name} ${key}`)
  match(kernel.k(x), 'gram')
  match(kernel.k(x, y), 'cross')
  match(kernel.diag(x), 'diag')
  const ours = toFlat(theta(kernel))
  const want = c.params['theta'] as number[]
  const perm = order ?? want.map((_, i) => i)
  expect(perm.map((j) => ours[j] ?? Number.NaN)).toEqual(want.map((v) => expect.closeTo(v, 12)))
  const { gradient } = gramGradient(kernel, x)
  match(take(gradient, fromStorage(Int32Array.from(perm), [perm.length], 'int32'), 2), 'gradient')
}

/** Random point sets `[n, d]` with coordinates in `[lo, hi]`, for kernel property tests. */
export function points(n: number, d: number, lo = -2, hi = 2): fc.Arbitrary<Tensor<'float64'>> {
  return tensorOf([n, d], lo, hi)
}

/** The smallest eigenvalue of a symmetric matrix. */
export function minEigenvalue(k: TensorLike): number {
  // eigvalsh is ascending.
  return toFlat(eigvalsh(k))[0] ?? Number.NaN
}

/** Assert that a Gram matrix is symmetric and positive semi-definite up to a tolerance relative to its scale. */
export function expectPsd(k: Value, tol = 1e-9): void {
  const values = toFlat(k)
  const n = k.shape[0] ?? 0
  let scale = 1
  for (let i = 0; i < n; i++) {
    for (let j = 0; j < n; j++) {
      const a = values[i * n + j] ?? 0
      const b = values[j * n + i] ?? 0
      scale = Math.max(scale, Math.abs(a))
      expect(Math.abs(a - b)).toBeLessThanOrEqual(1e-12 * Math.max(1, Math.abs(a)))
    }
  }
  expect(minEigenvalue(k)).toBeGreaterThanOrEqual(-tol * scale * n)
}

/** Options of {@link expectKernelLaws}. */
export interface KernelLawOptions {
  /** Check positive semi-definiteness (default true). */
  readonly psd?: boolean
  /** Tolerance of the eigenvalue check, relative to the scale (default 1e-9). */
  readonly tol?: number
  /** Whether k(x, x) as a cross-kernel equals the Gram matrix (false for white noise; default true). */
  readonly crossIsGram?: boolean
  /** Tolerance of that comparison (default 1e-9; angular kernels lose ~sqrt(eps) on the diagonal of k(x, x)). */
  readonly crossTol?: number
}

function close(a: number, b: number, tol: number): void {
  expect(Math.abs(a - b)).toBeLessThanOrEqual(tol * Math.max(1, Math.abs(a)))
}

/**
 * The laws every kernel obeys on points x and y: a symmetric (PSD) Gram matrix whose diagonal is `diag`, a
 * cross-covariance that transposes when the arguments swap, and, for a stationary kernel, invariance to a common shift.
 */
export function expectKernelLaws(kernel: Kernel, x: Tensor, y: Tensor, options: KernelLawOptions = {}): void {
  const k = kernel.k(x)
  if (options.psd !== false) expectPsd(k, options.tol)
  const n = x.shape[0] ?? 0
  const m = y.shape[0] ?? 0
  const kv = toFlat(k)
  const d = toFlat(kernel.diag(x))
  for (let i = 0; i < n; i++) close(d[i] ?? Number.NaN, kv[i * n + i] ?? Number.NaN, 1e-10)
  const xy = toFlat(kernel.k(x, y))
  const yx = toFlat(kernel.k(y, x))
  for (let i = 0; i < n; i++) for (let j = 0; j < m; j++) close(xy[i * m + j] ?? 0, yx[j * n + i] ?? Number.NaN, 1e-12)
  if (options.crossIsGram !== false) {
    const xx = toFlat(kernel.k(x, x))
    for (let i = 0; i < kv.length; i++) close(kv[i] ?? 0, xx[i] ?? Number.NaN, options.crossTol ?? 1e-9)
  }
  if (kernel.stationary) {
    const shifted = toFlat(kernel.k(add(x, 0.37), add(y, 0.37)))
    for (let i = 0; i < xy.length; i++) close(xy[i] ?? 0, shifted[i] ?? Number.NaN, 1e-9)
  }
}
