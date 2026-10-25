/**
 * Covariance matrix estimators, scikit-learn style: the maximum-likelihood (empirical) covariance and the shrinkage
 * estimators that pull it towards a scaled identity to stay well conditioned when variables are many and samples few,
 * with a fixed intensity, the Ledoit-Wolf intensity or the oracle approximating shrinkage (OAS).
 */

import { WhiteboxError } from '../base/index.ts'
import type { Tensor, TensorLike } from '../tensor/index.ts'
import { mat, rowsOf } from './_util.ts'

/** Options shared by the covariance estimators. */
export interface CovarianceOptions {
  /** The data are already centred: use zero as the location (default false). */
  readonly assumeCentered?: boolean
}

/** A shrunk covariance estimate with its shrinkage intensity. */
export interface ShrunkCovariance {
  readonly covariance: Tensor
  readonly shrinkage: number
  readonly location: Tensor
}

function centred(
  name: string,
  X: TensorLike,
  assume: boolean,
): { n: number; p: number; x: Float64Array; mean: Float64Array } {
  const m = rowsOf(name, X)
  const n = m.rows
  const p = m.cols
  if (n < 1) throw new WhiteboxError('value', `${name}: needs at least one observation`)
  const mean = new Float64Array(p)
  if (!assume)
    for (let i = 0; i < n; i++) for (let j = 0; j < p; j++) mean[j] = (mean[j] ?? 0) + (m.data[i * p + j] ?? 0) / n
  const x = Float64Array.from(m.data, (v, k) => v - (mean[k % p] ?? 0))
  return { n, p, x, mean }
}

function empirical(n: number, p: number, x: Float64Array): Float64Array {
  const c = new Float64Array(p * p)
  for (let i = 0; i < n; i++)
    for (let a = 0; a < p; a++) {
      const xa = x[i * p + a] ?? 0
      for (let b = a; b < p; b++) c[a * p + b] = (c[a * p + b] ?? 0) + xa * (x[i * p + b] ?? 0)
    }
  for (let a = 0; a < p; a++)
    for (let b = a; b < p; b++) {
      const v = (c[a * p + b] ?? 0) / n
      c[a * p + b] = v
      c[b * p + a] = v
    }
  return c
}

function shrink(c: Float64Array, p: number, shrinkage: number): Float64Array {
  let mu = 0
  for (let i = 0; i < p; i++) mu += (c[i * p + i] ?? 0) / p
  const out = Float64Array.from(c, (v) => (1 - shrinkage) * v)
  for (let i = 0; i < p; i++) out[i * p + i] = (out[i * p + i] ?? 0) + shrinkage * mu
  return out
}

/**
 * The maximum-likelihood covariance (sklearn.covariance.empirical_covariance): centred cross-products divided by n.
 *
 * @param X - Observations `[N, P]`.
 * @param options - assumeCentered.
 * @returns The P x P covariance.
 * @example
 * ```ts
 * import { empiricalCovariance } from 'whitebox-engine/stats'
 * empiricalCovariance([[0, 0], [2, 2]]) // => [[1, 1], [1, 1]]
 * ```
 */
export function empiricalCovariance(X: TensorLike, options: CovarianceOptions = {}): Tensor {
  const { n, p, x } = centred('empiricalCovariance', X, options.assumeCentered ?? false)
  return mat(empirical(n, p, x), p, p)
}

/**
 * A covariance shrunk towards a scaled identity (sklearn.covariance.ShrunkCovariance): (1 - a) S + a (tr S / p) I.
 *
 * @param X - Observations `[N, P]`.
 * @param options - shrinkage a in [0, 1] (default 0.1) and assumeCentered.
 * @returns The shrunk covariance.
 * @example
 * ```ts
 * import { shrunkCovariance } from 'whitebox-engine/stats'
 * shrunkCovariance([[0, 0], [2, 2]], { shrinkage: 0.5 }).covariance // => [[1, 0.5], [0.5, 1]]
 * ```
 */
export function shrunkCovariance(
  X: TensorLike,
  options: CovarianceOptions & { readonly shrinkage?: number } = {},
): ShrunkCovariance {
  const a = options.shrinkage ?? 0.1
  if (!(a >= 0 && a <= 1)) throw new WhiteboxError('value', 'shrunkCovariance: shrinkage must lie in [0, 1]')
  const { n, p, x, mean } = centred('shrunkCovariance', X, options.assumeCentered ?? false)
  return { covariance: mat(shrink(empirical(n, p, x), p, a), p, p), shrinkage: a, location: mat(mean, 1, p) }
}

/**
 * The Ledoit-Wolf covariance (sklearn.covariance.ledoit_wolf): the shrinkage towards (tr S / p) I that minimises the
 * asymptotic expected Frobenius loss, estimated from the data.
 *
 * @param X - Observations `[N, P]`.
 * @param options - assumeCentered.
 * @returns The covariance and the shrinkage.
 * @reference O. Ledoit and M. Wolf, A well-conditioned estimator for large-dimensional covariance matrices, Journal
 * of Multivariate Analysis 88 (2004).
 * @example
 * ```ts
 * import { ledoitWolf } from 'whitebox-engine/stats'
 * ledoitWolf([[1, 2], [2, 1], [3, 4], [4, 3]]).shrinkage // => 0.444444
 * ```
 */
export function ledoitWolf(X: TensorLike, options: CovarianceOptions = {}): ShrunkCovariance {
  const { n, p, x, mean } = centred('ledoitWolf', X, options.assumeCentered ?? false)
  const emp = empirical(n, p, x)
  if (p === 1) return { covariance: mat(emp, 1, 1), shrinkage: 0, location: mat(mean, 1, p) }
  let traceSum = 0
  for (let j = 0; j < p; j++) traceSum += emp[j * p + j] ?? 0
  const mu = traceSum / p
  let beta = 0
  let delta = 0
  for (let a = 0; a < p; a++)
    for (let b = 0; b < p; b++) {
      let s2 = 0
      let s = 0
      for (let i = 0; i < n; i++) {
        const xa = x[i * p + a] ?? 0
        const xb = x[i * p + b] ?? 0
        s += xa * xb
        s2 += xa * xa * xb * xb
      }
      delta += s * s
      beta += s2
    }
  delta /= n * n
  let b = (1 / (p * n)) * (beta / n - delta)
  let d = delta - 2 * mu * traceSum + p * mu * mu
  d /= p
  b = Math.min(b, d)
  const shrinkage = b === 0 ? 0 : b / d
  return { covariance: mat(shrink(emp, p, shrinkage), p, p), shrinkage, location: mat(mean, 1, p) }
}

/**
 * The oracle approximating shrinkage covariance (sklearn.covariance.oas, Chen et al. 2010), sklearn's corrected formula
 * for the shrinkage towards (tr S / p) I.
 *
 * @param X - Observations `[N, P]`.
 * @param options - assumeCentered.
 * @returns The covariance and the shrinkage.
 * @example
 * ```ts
 * import { oas } from 'whitebox-engine/stats'
 * oas([[1, 2], [2, 1], [3, 4], [4, 3]]).shrinkage // => 1
 * ```
 */
export function oas(X: TensorLike, options: CovarianceOptions = {}): ShrunkCovariance {
  const { n, p, x, mean } = centred('oas', X, options.assumeCentered ?? false)
  const emp = empirical(n, p, x)
  if (p === 1) return { covariance: mat(emp, 1, 1), shrinkage: 0, location: mat(mean, 1, p) }
  let alpha = 0
  for (const v of emp) alpha += v * v
  alpha /= p * p
  let mu = 0
  for (let j = 0; j < p; j++) mu += (emp[j * p + j] ?? 0) / p
  const num = alpha + mu * mu
  const den = (n + 1) * (alpha - (mu * mu) / p)
  const shrinkage = den === 0 ? 1 : Math.min(num / den, 1)
  return { covariance: mat(shrink(emp, p, shrinkage), p, p), shrinkage, location: mat(mean, 1, p) }
}
