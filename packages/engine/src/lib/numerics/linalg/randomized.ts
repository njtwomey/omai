/**
 * Randomised linear algebra: the range finder and truncated SVD of Halko, Martinsson and Tropp (as scikit-learn's
 * `randomized_svd`), random projection matrices with the Johnson-Lindenstrauss dimension that preserves pairwise
 * distances, and stochastic estimators for large implicit matrices, Hutchinson's trace estimator and log-determinants
 * by stochastic Lanczos quadrature.
 *
 * @index function Randomised SVD, range finder and random projections.
 */

import { MdmlError, minOf } from '../../base/index.ts'
import { foldIn, type Key, key as makeKey, normal, uniform } from '../../random/index.ts'
import {
  concreteOrThrow,
  fromStorage,
  matmul,
  promoteFloat,
  sliceAxis,
  type Tensor,
  type TensorLike,
  type Value,
} from '../../tensor/index.ts'
import { eighKernel } from './_kernels.ts'
import { mT } from './_matrix.ts'
import { dnrm2 } from './blas.ts'
import { qr } from './qr.ts'
import { svd } from './svd.ts'

/** Options of the randomised range finder and SVD. */
export interface RandomizedOptions {
  /** Extra sketch columns beyond the target rank (default 10). */
  readonly oversample?: number
  /** Power (subspace) iterations, sharpening the spectrum (default 4, as scikit-learn's 'auto' for small ranks). */
  readonly powerIterations?: number
  readonly key?: Key
}

/**
 * An orthonormal basis Q whose range approximates the range of A (Halko, Martinsson and Tropp 2011, algorithm 4.4):
 * QR of (A A^T)^q A Omega for a Gaussian sketch Omega, re-orthonormalised at each power iteration.
 *
 * @param a - Matrix `[m, n]`.
 * @param size - Number of columns of Q.
 * @param options - Power iterations and key.
 * @returns Q `[m, size]`, differentiable in A.
 * @concept randomised-linear-algebra
 * @example
 * ```ts
 * import { randomizedRangeFinder } from 'mdml-engine/numerics/linalg'
 * randomizedRangeFinder([[1, 0], [0, 1], [0, 0]], 2).shape // => [3, 2]
 * ```
 */
export function randomizedRangeFinder(a: TensorLike, size: number, options: RandomizedOptions = {}): Value {
  const [m] = promoteFloat(a) as [Value]
  const n = m.shape[1] ?? 0
  const k = options.key ?? makeKey(0)
  let q = qr(matmul(m, normal(k, [n, size]))).q
  for (let i = 0; i < (options.powerIterations ?? 4); i++) {
    q = qr(matmul(mT(m), q)).q
    q = qr(matmul(m, q)).q
  }
  return q
}

/**
 * A truncated SVD by randomised projection (Halko et al.; sklearn.utils.extmath.randomized_svd): project onto an
 * approximate range, take the exact SVD of the small projected matrix, and lift back.
 *
 * @param a - Matrix `[m, n]`.
 * @param rank - Number of singular triplets.
 * @param options - Oversampling, power iterations and key.
 * @returns `{ u, s, vh }` with `rank` components, differentiable.
 * @concept randomised-linear-algebra
 * @example
 * ```ts
 * import { randomizedSvd } from 'mdml-engine/numerics/linalg'
 * import { toArray } from 'mdml-engine/tensor'
 * toArray(randomizedSvd([[3, 0], [0, 1]], 1).s).map((v) => Math.round(v * 1e8) / 1e8) // => [3]
 * ```
 */
export function randomizedSvd(
  a: TensorLike,
  rank: number,
  options: RandomizedOptions = {},
): { readonly u: Value; readonly s: Value; readonly vh: Value } {
  const [m] = promoteFloat(a) as [Value]
  const size = Math.min(rank + (options.oversample ?? 10), minOf(m.shape))
  const q = randomizedRangeFinder(m, size, options)
  const { u, s, vh } = svd(matmul(mT(q), m))
  return { u: sliceAxis(matmul(q, u), 1, 0, rank), s: sliceAxis(s, 0, 0, rank), vh: sliceAxis(vh, 0, 0, rank) }
}

/**
 * A random projection matrix `[nComponents, nFeatures]` (scikit-learn's random projections): Gaussian with variance
 * 1 / nComponents, or the sparse Achlioptas/Li construction with entries +-sqrt(1 / (density nComponents)) with
 * probability density / 2 each (default density 1 / sqrt(nFeatures)).
 *
 * @param key - The random key.
 * @param nComponents - Target dimension.
 * @param nFeatures - Source dimension.
 * @param options - `kind` ('gaussian' or 'sparse') and `density`.
 * @returns The matrix.
 * @example
 * ```ts
 * import { randomProjectionMatrix } from 'mdml-engine/numerics/linalg'
 * import { key } from 'mdml-engine/random'
 * randomProjectionMatrix(key(0), 5, 100).shape // => [5, 100]
 * ```
 */
export function randomProjectionMatrix(
  key: Key,
  nComponents: number,
  nFeatures: number,
  options: { readonly kind?: 'gaussian' | 'sparse'; readonly density?: number } = {},
): Tensor {
  if ((options.kind ?? 'gaussian') === 'gaussian')
    return normal(key, [nComponents, nFeatures], { scale: 1 / Math.sqrt(nComponents) })
  const density = options.density ?? 1 / Math.sqrt(nFeatures)
  const value = Math.sqrt(1 / (density * nComponents))
  const u = uniform(key, [nComponents * nFeatures]).data
  return fromStorage(
    Float64Array.from(u, (v) => (v < density / 2 ? -value : v < density ? value : 0)),
    [nComponents, nFeatures],
    'float64',
  )
}

/**
 * The Johnson-Lindenstrauss dimension: a projection to this many dimensions preserves all pairwise distances of n
 * points within a factor 1 +- eps with high probability (sklearn.random_projection.johnson_lindenstrauss_min_dim).
 *
 * @param n - Number of points.
 * @param eps - Distortion (0 < eps < 1).
 * @returns The dimension.
 * @example
 * ```ts
 * import { johnsonLindenstraussDim } from 'mdml-engine/numerics/linalg'
 * johnsonLindenstraussDim(1e6, 0.5) // => 663
 * ```
 */
export function johnsonLindenstraussDim(n: number, eps: number): number {
  if (!(eps > 0 && eps < 1)) throw new MdmlError('value', 'johnsonLindenstraussDim: eps must be in (0, 1)')
  return Math.floor((4 * Math.log(n)) / (eps ** 2 / 2 - eps ** 3 / 3))
}

/** A matrix, or a function computing its product with a vector. */
export type Operator = TensorLike | ((x: Float64Array) => Float64Array)

function asMatvec(op: Operator): { matvec: (x: Float64Array) => Float64Array; n: number | undefined } {
  if (typeof op === 'function') return { matvec: op, n: undefined }
  const m = concreteOrThrow(promoteFloat(op)[0] as Tensor, 'operator')
  const n = m.shape[0] ?? 0
  const data = m.data
  return {
    n,
    matvec: (x) => {
      const y = new Float64Array(n)
      for (let i = 0; i < n; i++) {
        let s = 0
        for (let j = 0; j < n; j++) s += (data[i * n + j] ?? 0) * (x[j] ?? 0)
        y[i] = s
      }
      return y
    },
  }
}

function rademacher(k: Key, n: number): Float64Array {
  return Float64Array.from(uniform(k, [n]).data, (u) => (u < 0.5 ? -1 : 1))
}

/** Options of the stochastic estimators. */
export interface StochasticOptions {
  readonly samples?: number
  readonly key?: Key
  /** The dimension, needed when the operator is a function. */
  readonly n?: number
}

/**
 * Hutchinson's trace estimator: the mean of z^T A z over Rademacher probes z, with its standard error.
 *
 * @param operator - A square matrix or a matrix-vector product.
 * @param options - Number of probes, key and dimension.
 * @returns The estimate and its standard error.
 * @concept randomised-linear-algebra
 * @example
 * ```ts
 * import { hutchinsonTrace } from 'mdml-engine/numerics/linalg'
 * hutchinsonTrace([[2, 0], [0, 3]], { samples: 50 }).estimate // => 5
 * ```
 */
export function hutchinsonTrace(
  operator: Operator,
  options: StochasticOptions = {},
): { readonly estimate: number; readonly standardError: number } {
  const { matvec, n: dim } = asMatvec(operator)
  const n = options.n ?? dim
  if (n === undefined) throw new MdmlError('value', 'hutchinsonTrace: pass n for a function operator')
  const samples = options.samples ?? 100
  const values: number[] = []
  for (let s = 0; s < samples; s++) {
    const z = rademacher(foldIn(options.key ?? makeKey(0), s), n)
    const y = matvec(z)
    values.push(z.reduce((acc, v, i) => acc + v * (y[i] ?? 0), 0))
  }
  const mean = values.reduce((a, b) => a + b, 0) / samples
  const variance = values.reduce((a, b) => a + (b - mean) ** 2, 0) / Math.max(1, samples - 1)
  return { estimate: mean, standardError: Math.sqrt(variance / samples) }
}

/**
 * The log-determinant of a symmetric positive-definite operator by stochastic Lanczos quadrature (Ubaru, Chen and
 * Saad 2017): for each Rademacher probe, m Lanczos steps give a Gauss quadrature of z^T log(A) z.
 *
 * @param operator - A symmetric positive-definite matrix or its matrix-vector product.
 * @param options - Probes, key, dimension and Lanczos steps (default 30).
 * @returns The estimate and its standard error.
 * @example
 * ```ts
 * import { stochasticLogdet } from 'mdml-engine/numerics/linalg'
 * Math.abs(stochasticLogdet([[2, 0], [0, 3]], { samples: 20 }).estimate - Math.log(6)) < 1e-10 // => true
 * ```
 */
export function stochasticLogdet(
  operator: Operator,
  options: StochasticOptions & { readonly lanczosSteps?: number } = {},
): { readonly estimate: number; readonly standardError: number } {
  const { matvec, n: dim } = asMatvec(operator)
  const n = options.n ?? dim
  if (n === undefined) throw new MdmlError('value', 'stochasticLogdet: pass n for a function operator')
  const steps = Math.min(options.lanczosSteps ?? 30, n)
  const samples = options.samples ?? 30
  const values: number[] = []
  for (let s = 0; s < samples; s++) {
    const z = rademacher(foldIn(options.key ?? makeKey(0), s), n)
    const { alpha, beta } = lanczos(matvec, z, steps)
    const m = alpha.length
    const t = new Float64Array(m * m)
    for (let i = 0; i < m; i++) {
      t[i * m + i] = alpha[i] ?? 0
      if (i + 1 < m) {
        t[i * m + i + 1] = beta[i] ?? 0
        t[(i + 1) * m + i] = beta[i] ?? 0
      }
    }
    const { w, v } = eighKernel(t, m)
    let quad = 0
    for (let j = 0; j < m; j++) quad += (v[j] ?? 0) ** 2 * Math.log(w[j] ?? 1)
    values.push(n * quad)
  }
  const mean = values.reduce((a, b) => a + b, 0) / samples
  const variance = values.reduce((a, b) => a + (b - mean) ** 2, 0) / Math.max(1, samples - 1)
  return { estimate: mean, standardError: Math.sqrt(variance / samples) }
}

/** Lanczos tridiagonalisation from start vector z (normalised), with full reorthogonalisation. */
function lanczos(
  matvec: (x: Float64Array) => Float64Array,
  z: Float64Array,
  steps: number,
): { alpha: number[]; beta: number[] } {
  const length = dnrm2(z)
  const basis: Float64Array[] = [z.map((v) => v / length)]
  const alpha: number[] = []
  const beta: number[] = []
  for (let j = 0; j < steps; j++) {
    const q = basis[j] as Float64Array
    const w = matvec(q)
    const a = q.reduce((s, v, i) => s + v * (w[i] ?? 0), 0)
    alpha.push(a)
    for (const b of basis) {
      const c = b.reduce((s, v, i) => s + v * (w[i] ?? 0), 0)
      for (let i = 0; i < w.length; i++) w[i] = (w[i] ?? 0) - c * (b[i] ?? 0)
    }
    const nb = Math.sqrt(w.reduce((s, v) => s + v * v, 0))
    if (j + 1 === steps || nb < 1e-12) break
    beta.push(nb)
    basis.push(w.map((v) => v / nb))
  }
  return { alpha, beta }
}
