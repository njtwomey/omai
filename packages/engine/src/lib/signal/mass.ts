/**
 * MASS (Mueen's algorithm for similarity search): the z-normalised Euclidean distance from a query to every window of a
 * time series, computed with FFT sliding dot products and rolling statistics, following stumpy.mass. Also the chunked
 * variant for very long series and the distance profile of one window of a series against all its others, a row of the
 * matrix profile.
 */

import { WhiteboxError } from '../base/index.ts'
import type { TensorLike } from '../tensor/index.ts'
import {
  applyExclusionZone,
  exclusionZone,
  nonNormalizedDistanceProfile,
  rollingMeanStd,
  seriesOf,
  slidingDotProduct,
  subsequenceStats,
  zNormalizedDistanceProfile,
} from './sliding.ts'

// Mueen's algorithm for similarity search (MASS): the distance profile of a query against every window of a series
// in O(n log n), by FFT sliding dot products (v2) or by chunks of the series (v3, memory proportional to the chunk),
// z-normalised as STUMPY's mass or non-normalised as its mass_absolute.

/** Options for {@link mass}. */
export interface MassOptions {
  /** z-normalised distances (default true) or raw Minkowski distances. */
  readonly normalize?: boolean
  /** The Minkowski norm for non-normalised distances (default 2). */
  readonly p?: number
}

function queryStats(q: Float64Array): { mu: number; sigma: number; constant: boolean } {
  const { mean, std } = rollingMeanStd(q, q.length)
  let lo = Number.POSITIVE_INFINITY
  let hi = Number.NEGATIVE_INFINITY
  for (const v of q) {
    lo = Math.min(lo, v)
    hi = Math.max(hi, v)
  }
  return { mu: mean[0] ?? 0, sigma: std[0] ?? 0, constant: lo === hi }
}

/**
 * The distance profile of a query against a series (MASS v2, stumpy.mass): z-normalised Euclidean distances from FFT
 * sliding dot products and rolling statistics, with STUMPY's conventions (0 between constant windows, sqrt(m)
 * between a constant and a non-constant one, infinity for non-finite windows or queries); or non-normalised
 * Minkowski distances.
 *
 * @param q - Query `[m]` (m >= 3).
 * @param t - Series `[n]`.
 * @param options - Normalisation and norm.
 * @returns The distances `[n - m + 1]`.
 * @concept mass
 * @example
 * ```ts
 * import { mass } from 'whitebox-engine/signal'
 * Array.from(mass([1, 2, 3], [0, 1, 2, 3, 9, 4, 5, 6])).map((v) => Math.round(v * 1e6) / 1e6).filter((v) => v === 0).length // => 3
 * ```
 */
export function mass(q: TensorLike, t: TensorLike, options: MassOptions = {}): Float64Array {
  const qa = q instanceof Float64Array ? q : seriesOf(q, 'mass')
  const ta = t instanceof Float64Array ? t : seriesOf(t, 'mass')
  const m = qa.length
  if (m < 3) throw new WhiteboxError('value', 'mass: the query must have at least 3 samples')
  if (m > ta.length) throw new WhiteboxError('value', 'mass: the query is longer than the series')
  if (!(options.normalize ?? true)) return nonNormalizedDistanceProfile(qa, ta, options.p ?? 2)
  if (qa.some((v) => !Number.isFinite(v))) return new Float64Array(ta.length - m + 1).fill(Number.POSITIVE_INFINITY)
  const stats = subsequenceStats(ta, m)
  const qs = queryStats(qa)
  return zNormalizedDistanceProfile(slidingDotProduct(qa, stats.t), qs.mu, qs.sigma, qs.constant, stats)
}

/**
 * MASS v3: the same distance profile with the sliding dot products computed chunk by chunk (each chunk of `chunk`
 * samples overlaps the next by m - 1), so FFT memory stays bounded for very long series.
 *
 * @param q - Query `[m]`.
 * @param t - Series `[n]`.
 * @param chunk - Chunk length (default 4096, at least m).
 * @returns The z-normalised distances `[n - m + 1]`.
 * @concept mass
 * @example
 * ```ts
 * import { mass, massChunked } from 'whitebox-engine/signal'
 * const t = Array.from({ length: 100 }, (_, i) => Math.sin(i / 3) + i / 50)
 * const a = mass([0, 1, 0.5, -1], t)
 * const b = massChunked([0, 1, 0.5, -1], t, 16)
 * Array.from(a).every((v, i) => Math.abs(v - (b[i] ?? 0)) < 1e-9) // => true
 * ```
 */
export function massChunked(q: TensorLike, t: TensorLike, chunk = 4096): Float64Array {
  const qa = q instanceof Float64Array ? q : seriesOf(q, 'massChunked')
  const ta = t instanceof Float64Array ? t : seriesOf(t, 'massChunked')
  const m = qa.length
  if (chunk < m) throw new WhiteboxError('value', 'massChunked: the chunk must be at least the query length')
  if (qa.some((v) => !Number.isFinite(v))) return new Float64Array(ta.length - m + 1).fill(Number.POSITIVE_INFINITY)
  const stats = subsequenceStats(ta, m)
  const l = ta.length - m + 1
  const qt = new Float64Array(l)
  const step = chunk - m + 1
  for (let start = 0; start < l; start += step) {
    const stop = Math.min(ta.length, start + chunk)
    const part = slidingDotProduct(qa, stats.t.subarray(start, stop))
    qt.set(part.subarray(0, Math.min(part.length, l - start)), start)
  }
  const qs = queryStats(qa)
  return zNormalizedDistanceProfile(qt, qs.mu, qs.sigma, qs.constant, stats)
}

/** Options for {@link distanceProfile}. */
export interface DistanceProfileOptions extends MassOptions {
  /** Exclude trivial matches around the query's own position (default true). */
  readonly excludeTrivial?: boolean
}

/**
 * The distance profile of the window at `idx` of a series against all its windows (a row of the matrix profile's
 * distance matrix), with the trivial-match exclusion zone ceil(m / 4) set to infinity.
 *
 * @param t - Series.
 * @param m - Window length.
 * @param idx - Start of the query window.
 * @param options - Normalisation, norm and exclusion.
 * @returns The distances `[n - m + 1]`.
 * @concept mass
 * @example
 * ```ts
 * import { distanceProfile } from 'whitebox-engine/signal'
 * distanceProfile([0, 1, 2, 3, 2, 1, 0, 1, 2, 3], 4, 0)[0] // => Infinity
 * ```
 */
export function distanceProfile(
  t: TensorLike,
  m: number,
  idx: number,
  options: DistanceProfileOptions = {},
): Float64Array {
  const ta = t instanceof Float64Array ? t : seriesOf(t, 'distanceProfile')
  if (!Number.isInteger(idx) || idx < 0 || idx > ta.length - m)
    throw new WhiteboxError('value', 'distanceProfile: idx out of range')
  const d = mass(ta.slice(idx, idx + m), ta, options)
  if (options.excludeTrivial ?? true) applyExclusionZone(d, idx, exclusionZone(m))
  return d
}
