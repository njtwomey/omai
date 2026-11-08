/**
 * The sliding-window machinery behind matrix profiles and similarity search, following STUMPY's conventions: rolling
 * means, standard deviations, constancy and finiteness of every length-m window, sliding dot products by FFT, the O(1)
 * diagonal update used by STOMP, z-normalised and non-normalised distance profiles, and the exclusion zone that ignores
 * trivial self-matches.
 */

import { OmaiError } from '../base/index.ts'
import { fftConvolve, oaConvolve } from '../numerics/convolution/index.ts'
import { primalOf, type TensorLike, toFlat } from '../tensor/index.ts'
import { float } from './_shared.ts'

// The shared machinery of subsequence (matrix-profile) analysis, as STUMPY computes it: statistics of every length-m
// window (Welford means and standard deviations with periodic exact recomputation, constancy by the range, finiteness),
// sliding dot products (by FFT or overlap-add, directly, and by the O(1) diagonal update), z-normalised and
// non-normalised distances from dot products, and exclusion zones. All work on typed arrays: these are the inner
// loops of O(n^2) algorithms.

/** Statistics of every length-m window of a series, computed with non-finite samples replaced by zero. */
export interface SubsequenceStats {
  /** Window length. */
  readonly m: number
  /** The series with non-finite values set to zero. */
  readonly t: Float64Array
  readonly mean: Float64Array
  readonly std: Float64Array
  /** Whether every value in the window is finite. */
  readonly isFinite: Uint8Array
  /** Whether the window is constant (range zero) and finite. */
  readonly isConstant: Uint8Array
}

/** The window length must be at least 3 (shorter z-normalised windows are meaningless, as STUMPY). */
export function checkWindow(m: number, n: number): void {
  if (!Number.isInteger(m) || m < 3) throw new OmaiError('value', 'the window length must be an integer of at least 3')
  if (m > n) throw new OmaiError('value', `the window length ${m} exceeds the series length ${n}`)
}

/** The series as a Float64Array of primal values. */
export function seriesOf(t: TensorLike, name: string): Float64Array {
  const v = float(t)
  if (v.shape.length !== 1) throw new OmaiError('shape', `${name}: expected a 1-D series`)
  return Float64Array.from(primalOf(v).data)
}

/**
 * Rolling means and standard deviations of every window of length m by Welford's updates, recomputed exactly every
 * `refresh` windows and whenever the running variance is small relative to the mean (where cancellation would lose
 * precision), so the results agree with direct computation to rounding.
 *
 * @param t - Series; windows holding a non-finite value get mean infinity and deviation NaN (as STUMPY).
 * @param m - Window length.
 * @param refresh - Exact recomputation interval (default 64).
 * @returns Means and (population) standard deviations `[n - m + 1]`.
 * @concept matrix-profile
 * @example
 * ```ts
 * import { rollingMeanStd } from 'omai-engine/signal'
 * const { mean, std } = rollingMeanStd([1, 2, 3, 4], 2)
 * [Array.from(mean), Array.from(std)] // => [[1.5, 2.5, 3.5], [0.5, 0.5, 0.5]]
 * ```
 */
export function rollingMeanStd(
  t: TensorLike,
  m: number,
  refresh = 64,
): { readonly mean: Float64Array; readonly std: Float64Array } {
  const raw = t instanceof Float64Array ? t : seriesOf(t, 'rollingMeanStd')
  const finite = raw.every(Number.isFinite)
  const x = finite ? raw : Float64Array.from(raw, (v) => (Number.isFinite(v) ? v : 0))
  const l = x.length - m + 1
  if (m < 1 || l < 1) throw new OmaiError('value', 'rollingMeanStd: the window must fit in the series')
  const mean = new Float64Array(l)
  const std = new Float64Array(l)
  let mu = 0
  let v = 0
  const exact = (s: number): void => {
    let a = 0
    for (let i = s; i < s + m; i++) a += x[i] ?? 0
    mu = a / m
    let b = 0
    for (let i = s; i < s + m; i++) b += ((x[i] ?? 0) - mu) ** 2
    v = b / m
  }
  for (let s = 0; s < l; s++) {
    if (s === 0 || s % refresh === 0) exact(s)
    else {
      const out = x[s - 1] ?? 0
      const inc = x[s + m - 1] ?? 0
      const nmu = mu + (inc - out) / m
      v += ((inc - out) * (inc - nmu + out - mu)) / m
      mu = nmu
      // Precision check: a variance far below the scale of the values has lost its digits; recompute it.
      if (v < 1e-7 * (mu * mu + 1)) exact(s)
    }
    mean[s] = mu
    std[s] = Math.sqrt(Math.max(v, 0))
  }
  if (!finite) {
    const ok = rollingIsFinite(raw, m)
    for (let s = 0; s < l; s++)
      if (ok[s] !== 1) {
        mean[s] = Number.POSITIVE_INFINITY
        std[s] = Number.NaN
      }
  }
  return { mean, std }
}

/**
 * Which windows of length m are constant (range zero), by monotone deques in O(n) (STUMPY's rolling_isconstant).
 *
 * @param t - Series.
 * @param m - Window length.
 * @returns 1 for a constant window, else 0.
 * @concept matrix-profile
 * @example
 * ```ts
 * import { rollingIsConstant } from 'omai-engine/signal'
 * Array.from(rollingIsConstant([1, 1, 1, 2], 3)) // => [1, 0]
 * ```
 */
export function rollingIsConstant(t: TensorLike, m: number): Uint8Array {
  const x = t instanceof Float64Array ? t : seriesOf(t, 'rollingIsConstant')
  const l = x.length - m + 1
  const out = new Uint8Array(Math.max(l, 0))
  const maxQ: number[] = []
  const minQ: number[] = []
  let maxHead = 0
  let minHead = 0
  for (let i = 0; i < x.length; i++) {
    const v = x[i] ?? 0
    while (maxQ.length > maxHead && (x[maxQ[maxQ.length - 1] ?? 0] ?? 0) <= v) maxQ.pop()
    maxQ.push(i)
    while (minQ.length > minHead && (x[minQ[minQ.length - 1] ?? 0] ?? 0) >= v) minQ.pop()
    minQ.push(i)
    const s = i - m + 1
    if (s < 0) continue
    while ((maxQ[maxHead] ?? 0) < s) maxHead++
    while ((minQ[minHead] ?? 0) < s) minHead++
    out[s] = (x[maxQ[maxHead] ?? 0] ?? 0) === (x[minQ[minHead] ?? 0] ?? 0) ? 1 : 0
  }
  return out
}

/**
 * Which windows of length m contain only finite values.
 *
 * @param t - Series (may contain NaN or infinities).
 * @param m - Window length.
 * @returns 1 for a finite window, else 0.
 * @concept matrix-profile
 * @example
 * ```ts
 * import { rollingIsFinite } from 'omai-engine/signal'
 * Array.from(rollingIsFinite([1, Number.NaN, 2, 3, 4], 2)) // => [0, 0, 1, 1]
 * ```
 */
export function rollingIsFinite(t: TensorLike, m: number): Uint8Array {
  const x = t instanceof Float64Array ? t : seriesOf(t, 'rollingIsFinite')
  const l = x.length - m + 1
  const out = new Uint8Array(Math.max(l, 0))
  let bad = 0
  for (let i = 0; i < x.length; i++) {
    if (!Number.isFinite(x[i] ?? Number.NaN)) bad++
    if (i >= m && !Number.isFinite(x[i - m] ?? Number.NaN)) bad--
    if (i >= m - 1) out[i - m + 1] = bad === 0 ? 1 : 0
  }
  return out
}

/**
 * The statistics of every window as STUMPY's preprocessing: non-finite samples set to zero, Welford means and
 * standard deviations of the zero-filled series, finiteness, and constancy (only for finite windows).
 *
 * @param t - Series.
 * @param m - Window length (at least 3).
 * @returns The statistics.
 * @concept matrix-profile
 * @example
 * ```ts
 * import { subsequenceStats } from 'omai-engine/signal'
 * Array.from(subsequenceStats([1, 1, 1, 2, 5], 3).isConstant) // => [1, 0, 0]
 * ```
 */
export function subsequenceStats(t: TensorLike, m: number): SubsequenceStats {
  const raw = t instanceof Float64Array ? t : seriesOf(t, 'subsequenceStats')
  checkWindow(m, raw.length)
  const finite = rollingIsFinite(raw, m)
  const filled = Float64Array.from(raw, (v) => (Number.isFinite(v) ? v : 0))
  const constant = rollingIsConstant(filled, m)
  const isConstant = Uint8Array.from(constant, (c, i) => (c === 1 && finite[i] === 1 ? 1 : 0))
  const { mean, std } = rollingMeanStd(filled, m)
  return { m, t: filled, mean, std, isFinite: finite, isConstant }
}

/**
 * The sliding dot products QT[i] = sum_j q[j] t[i + j] of a query against every window of a series, by FFT
 * convolution, or overlap-add for a long series and a short query.
 *
 * @param q - Query `[m]`.
 * @param t - Series `[n]` (n >= m).
 * @returns The dot products `[n - m + 1]`.
 * @concept matrix-profile
 * @example
 * ```ts
 * import { slidingDotProduct } from 'omai-engine/signal'
 * Array.from(slidingDotProduct([1, 1], [1, 2, 3, 4])).map((v) => Math.round(v * 1e12) / 1e12) // => [3, 5, 7]
 * ```
 */
export function slidingDotProduct(q: TensorLike, t: TensorLike): Float64Array {
  const qa = q instanceof Float64Array ? q : seriesOf(q, 'slidingDotProduct')
  const ta = t instanceof Float64Array ? t : seriesOf(t, 'slidingDotProduct')
  const m = qa.length
  const n = ta.length
  if (m > n) throw new OmaiError('value', 'slidingDotProduct: the query is longer than the series')
  if (m * n <= 4096) return slidingDotProductDirect(qa, ta)
  const rev = Float64Array.from(qa).reverse()
  const conv = n > 16 * m ? oaConvolve(ta, rev) : fftConvolve(ta, rev)
  return Float64Array.from(toFlat(conv).slice(m - 1, n))
}

/**
 * The sliding dot products by direct summation (O(n m), exact to rounding).
 *
 * @param q - Query `[m]`.
 * @param t - Series `[n]`.
 * @returns The dot products `[n - m + 1]`.
 * @concept matrix-profile
 * @example
 * ```ts
 * import { slidingDotProductDirect } from 'omai-engine/signal'
 * Array.from(slidingDotProductDirect([1, 1], [1, 2, 3, 4])) // => [3, 5, 7]
 * ```
 */
export function slidingDotProductDirect(q: TensorLike, t: TensorLike): Float64Array {
  const qa = q instanceof Float64Array ? q : seriesOf(q, 'slidingDotProductDirect')
  const ta = t instanceof Float64Array ? t : seriesOf(t, 'slidingDotProductDirect')
  const m = qa.length
  const l = ta.length - m + 1
  const out = new Float64Array(Math.max(l, 0))
  for (let i = 0; i < l; i++) {
    let s = 0
    for (let j = 0; j < m; j++) s += (qa[j] ?? 0) * (ta[i + j] ?? 0)
    out[i] = s
  }
  return out
}

/**
 * The O(1) diagonal update of a dot product between windows: QT(i + 1, j + 1) = QT(i, j) - a[i] b[j] + a[i + m]
 * b[j + m], the recurrence behind STOMP.
 *
 * @param qt - QT(i, j).
 * @param a - First series.
 * @param b - Second series.
 * @param i - Window start in a.
 * @param j - Window start in b.
 * @param m - Window length.
 * @returns QT(i + 1, j + 1).
 * @concept matrix-profile
 * @example
 * ```ts
 * import { nextDotProduct } from 'omai-engine/signal'
 * nextDotProduct(1 * 3 + 2 * 4, [1, 2, 3], [3, 4, 5], 0, 0, 2) // => 23
 * ```
 */
export function nextDotProduct(
  qt: number,
  a: ArrayLike<number>,
  b: ArrayLike<number>,
  i: number,
  j: number,
  m: number,
): number {
  return qt - (a[i] ?? 0) * (b[j] ?? 0) + (a[i + m] ?? 0) * (b[j + m] ?? 0)
}

/**
 * The squared z-normalised Euclidean distance between two windows from their dot product (STUMPY's
 * _calculate_squared_distance): 2 m (1 - rho) with the Pearson correlation rho = (QT - m muQ muT) / (m sigQ sigT)
 * capped at 1; 0 for two constant windows, m for one; infinite for a non-finite window (infinite mean).
 *
 * @param m - Window length.
 * @param qt - Dot product of the windows.
 * @param muQ - Mean of the query window.
 * @param sigmaQ - Standard deviation of the query window.
 * @param muT - Mean of the other window.
 * @param sigmaT - Standard deviation of the other window.
 * @param qConstant - Whether the query window is constant.
 * @param tConstant - Whether the other window is constant.
 * @returns The squared distance.
 * @concept matrix-profile
 * @example
 * ```ts
 * import { squaredZNormalizedDistance } from 'omai-engine/signal'
 * squaredZNormalizedDistance(3, 14, 2, Math.sqrt(2 / 3), 2, Math.sqrt(2 / 3), false, false) // => 0
 * ```
 */
export function squaredZNormalizedDistance(
  m: number,
  qt: number,
  muQ: number,
  sigmaQ: number,
  muT: number,
  sigmaT: number,
  qConstant: boolean,
  tConstant: boolean,
): number {
  if (!Number.isFinite(muT) || !Number.isFinite(muQ)) return Number.POSITIVE_INFINITY
  if (qConstant && tConstant) return 0
  if (qConstant || tConstant) return m
  const denom = Math.max(sigmaQ * sigmaT * m, 1e-14)
  const rho = Math.min((qt - muQ * muT * m) / denom, 1)
  return Math.abs(2 * m * (1 - rho))
}

/**
 * The z-normalised distance profile of a window against every window of a series from their dot products.
 *
 * @param qt - Dot products `[l]`.
 * @param muQ - Query mean.
 * @param sigmaQ - Query standard deviation.
 * @param qConstant - Whether the query is constant.
 * @param stats - The series' window statistics.
 * @returns Distances `[l]` (infinite for non-finite windows).
 * @concept matrix-profile
 * @example
 * ```ts
 * import { slidingDotProduct, subsequenceStats, zNormalizedDistanceProfile } from 'omai-engine/signal'
 * const s = subsequenceStats([1, 2, 3, 1, 2, 3], 3)
 * Array.from(zNormalizedDistanceProfile(slidingDotProduct([1, 2, 3], s.t), 2, Math.sqrt(2 / 3), false, s)).map((v) => Math.round(v * 1e6) / 1e6)[3] // => 0
 * ```
 */
export function zNormalizedDistanceProfile(
  qt: ArrayLike<number>,
  muQ: number,
  sigmaQ: number,
  qConstant: boolean,
  stats: SubsequenceStats,
): Float64Array {
  const l = stats.mean.length
  const out = new Float64Array(l)
  for (let i = 0; i < l; i++) {
    out[i] =
      stats.isFinite[i] === 1
        ? Math.sqrt(
            squaredZNormalizedDistance(
              stats.m,
              qt[i] ?? 0,
              muQ,
              sigmaQ,
              stats.mean[i] ?? 0,
              stats.std[i] ?? 0,
              qConstant,
              stats.isConstant[i] === 1,
            ),
          )
        : Number.POSITIVE_INFINITY
  }
  return out
}

/**
 * The non-normalised Minkowski distance profile (STUMPY's mass_absolute): ||q - t[i:i + m]||_p for every window,
 * infinite for windows with non-finite values; for p = 2 by sliding dot products and rolling sums of squares.
 *
 * @param q - Query `[m]`.
 * @param t - Series.
 * @param p - The norm (default 2).
 * @returns Distances `[n - m + 1]`.
 * @concept matrix-profile
 * @example
 * ```ts
 * import { nonNormalizedDistanceProfile } from 'omai-engine/signal'
 * Array.from(nonNormalizedDistanceProfile([0, 0, 0], [3, 4, 0, 0, 0])) // => [5, 4, 0]
 * ```
 */
export function nonNormalizedDistanceProfile(q: TensorLike, t: TensorLike, p = 2): Float64Array {
  const qa = q instanceof Float64Array ? q : seriesOf(q, 'nonNormalizedDistanceProfile')
  const raw = t instanceof Float64Array ? t : seriesOf(t, 'nonNormalizedDistanceProfile')
  const m = qa.length
  const l = raw.length - m + 1
  if (l < 1) throw new OmaiError('value', 'nonNormalizedDistanceProfile: the query is longer than the series')
  const finite = rollingIsFinite(raw, m)
  const out = new Float64Array(l)
  if (qa.some((v) => !Number.isFinite(v))) return out.fill(Number.POSITIVE_INFINITY)
  for (let i = 0; i < l; i++) {
    if (finite[i] !== 1) {
      out[i] = Number.POSITIVE_INFINITY
      continue
    }
    let s = 0
    for (let j = 0; j < m; j++) s += Math.abs((qa[j] ?? 0) - (raw[i + j] ?? 0)) ** p
    out[i] = s ** (1 / p)
  }
  return out
}

/**
 * The exclusion zone half-width of STUMPY: ceil(m / denom) with denom 4, within which matches are trivial.
 *
 * @param m - Window length.
 * @param denom - The denominator (default 4).
 * @returns The half-width.
 * @concept matrix-profile
 * @example
 * ```ts
 * import { exclusionZone } from 'omai-engine/signal'
 * exclusionZone(10) // => 3
 * ```
 */
export function exclusionZone(m: number, denom = 4): number {
  return Math.ceil(m / denom)
}

/**
 * Set a profile to a value over the exclusion zone [idx - zone, idx + zone] (clipped to the array), in place.
 *
 * @param a - The profile.
 * @param idx - The centre.
 * @param zone - Half-width.
 * @param value - The value (default infinity).
 * @returns The same array.
 * @concept matrix-profile
 * @example
 * ```ts
 * import { applyExclusionZone } from 'omai-engine/signal'
 * Array.from(applyExclusionZone(new Float64Array(5), 1, 1, 9)) // => [9, 9, 9, 0, 0]
 * ```
 */
export function applyExclusionZone(
  a: Float64Array,
  idx: number,
  zone: number,
  value: number = Number.POSITIVE_INFINITY,
): Float64Array {
  const lo = Math.max(0, idx - zone)
  const hi = Math.min(a.length - 1, idx + zone)
  for (let i = lo; i <= hi; i++) a[i] = value
  return a
}

/**
 * Z-normalise a sequence: (x - mean) / std, with the standard deviation replaced by 1 when below 1e-7 (STUMPY's
 * z_norm), so constant sequences map to zeros.
 *
 * @param x - Sequence.
 * @returns The z-normalised sequence.
 * @concept matrix-profile
 * @example
 * ```ts
 * import { zNormalize } from 'omai-engine/signal'
 * Array.from(zNormalize([1, 2, 3])).map((v) => Math.round(v * 1e6) / 1e6) // => [-1.224745, 0, 1.224745]
 * ```
 */
export function zNormalize(x: TensorLike): Float64Array {
  const s = x instanceof Float64Array ? x : seriesOf(x, 'zNormalize')
  let mu = 0
  for (const v of s) mu += v / s.length
  let sd = 0
  for (const v of s) sd += (v - mu) ** 2
  sd = Math.sqrt(sd / s.length)
  if (sd < 1e-7) sd = 1
  return Float64Array.from(s, (v) => (v - mu) / sd)
}
