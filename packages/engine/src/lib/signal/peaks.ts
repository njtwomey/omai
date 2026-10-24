import { WhiteboxError } from '../base/index.ts'
import type { TensorLike } from '../tensor/index.ts'
import { samples } from './_shared.ts'

// Peak finding as scipy.signal (find_peaks with its height, threshold, distance, prominence, width and plateau
// conditions; peak_prominences, peak_widths, argrelextrema) and zero crossings as librosa.

/** A condition on a peak property: a minimum, or a [min, max] interval (either end may be null). */
export type PeakCondition = number | readonly [number | null, number | null]

/** Options for {@link findPeaks}. */
export interface FindPeaksOptions {
  readonly height?: PeakCondition
  /** Vertical distance to the neighbouring samples. */
  readonly threshold?: PeakCondition
  /** Minimal horizontal distance between peaks (the higher wins). */
  readonly distance?: number
  readonly prominence?: PeakCondition
  readonly width?: PeakCondition
  /** Window length for prominences (default the whole signal). */
  readonly wlen?: number
  /** Relative height at which widths are measured (default 0.5). */
  readonly relHeight?: number
  readonly plateauSize?: PeakCondition
}

/** The peaks found and the properties computed for them (each array parallel to `peaks`). */
export interface PeakResult {
  readonly peaks: number[]
  readonly properties: Readonly<Record<string, number[]>>
}

/** Local maxima with plateaus: midpoints and edges (scipy's _local_maxima_1d). */
function localMaxima(x: Float64Array): { mid: number[]; left: number[]; right: number[] } {
  const mid: number[] = []
  const left: number[] = []
  const right: number[] = []
  let i = 1
  const iMax = x.length - 1
  while (i < iMax) {
    if ((x[i - 1] ?? 0) < (x[i] ?? 0)) {
      let ahead = i + 1
      while (ahead < iMax && x[ahead] === x[i]) ahead++
      if ((x[ahead] ?? 0) < (x[i] ?? 0)) {
        left.push(i)
        right.push(ahead - 1)
        mid.push(Math.floor((i + ahead - 1) / 2))
        i = ahead
      }
    }
    i++
  }
  return { mid, left, right }
}

function bounds(c: PeakCondition): [number | null, number | null] {
  return typeof c === 'number' ? [c, null] : [c[0], c[1]]
}

function within(values: readonly number[], c: PeakCondition): boolean[] {
  const [lo, hi] = bounds(c)
  return values.map((v) => (lo === null || lo <= v) && (hi === null || v <= hi))
}

/**
 * Prominences of peaks (scipy.signal.peak_prominences): the height of each peak above the higher of its two bases,
 * the lowest points on each side before the signal rises above the peak (within wlen samples when given).
 *
 * @param x - Signal.
 * @param peaks - Peak indices.
 * @param wlen - Window length (default the whole signal).
 * @returns Prominences and the left and right bases.
 * @concept peak-finding
 * @example
 * ```ts
 * import { peakProminences } from 'whitebox-engine/signal'
 * peakProminences([0, 2, 1, 3, 0], [1, 3]).prominences // => [1, 3]
 * ```
 */
export function peakProminences(
  x: TensorLike,
  peaks: readonly number[],
  wlen?: number,
): { readonly prominences: number[]; readonly leftBases: number[]; readonly rightBases: number[] } {
  const s = samples(x, 'peakProminences')
  const w = wlen === undefined ? -1 : Math.ceil(wlen)
  if (wlen !== undefined && !(wlen > 1)) throw new WhiteboxError('value', 'peakProminences: wlen must be larger than 1')
  const prominences: number[] = []
  const leftBases: number[] = []
  const rightBases: number[] = []
  for (const peak of peaks) {
    let iMin = 0
    let iMax = s.length - 1
    if (!(peak >= iMin && peak <= iMax))
      throw new WhiteboxError('value', `peakProminences: peak ${peak} is not a valid index`)
    if (w >= 2) {
      iMin = Math.max(peak - Math.floor(w / 2), iMin)
      iMax = Math.min(peak + Math.floor(w / 2), iMax)
    }
    const top = s[peak] ?? 0
    let i = peak
    let lb = peak
    let leftMin = top
    while (iMin <= i && (s[i] ?? 0) <= top) {
      if ((s[i] ?? 0) < leftMin) {
        leftMin = s[i] ?? 0
        lb = i
      }
      i--
    }
    i = peak
    let rb = peak
    let rightMin = top
    while (i <= iMax && (s[i] ?? 0) <= top) {
      if ((s[i] ?? 0) < rightMin) {
        rightMin = s[i] ?? 0
        rb = i
      }
      i++
    }
    prominences.push(top - Math.max(leftMin, rightMin))
    leftBases.push(lb)
    rightBases.push(rb)
  }
  return { prominences, leftBases, rightBases }
}

/**
 * Widths of peaks (scipy.signal.peak_widths): measured at the height prominence * relHeight below the peak, between
 * the linearly interpolated crossings on each side (bounded by the peak's bases).
 *
 * @param x - Signal.
 * @param peaks - Peak indices.
 * @param options - Relative height (default 0.5), prominence window, or precomputed prominence data.
 * @returns Widths, the heights where they were measured, and the interpolated left and right positions.
 * @concept peak-finding
 * @example
 * ```ts
 * import { peakWidths } from 'whitebox-engine/signal'
 * peakWidths([0, 1, 2, 1, 0], [2]).widths // => [2]
 * ```
 */
export function peakWidths(
  x: TensorLike,
  peaks: readonly number[],
  options: {
    readonly relHeight?: number
    readonly wlen?: number
    readonly prominenceData?: {
      readonly prominences: number[]
      readonly leftBases: number[]
      readonly rightBases: number[]
    }
  } = {},
): {
  readonly widths: number[]
  readonly widthHeights: number[]
  readonly leftIps: number[]
  readonly rightIps: number[]
} {
  const s = samples(x, 'peakWidths')
  const rel = options.relHeight ?? 0.5
  if (rel < 0) throw new WhiteboxError('value', 'peakWidths: relHeight must be non-negative')
  const pd = options.prominenceData ?? peakProminences(s, peaks, options.wlen)
  const widths: number[] = []
  const widthHeights: number[] = []
  const leftIps: number[] = []
  const rightIps: number[] = []
  for (const [p, peak] of peaks.entries()) {
    const iMin = pd.leftBases[p] ?? 0
    const iMax = pd.rightBases[p] ?? 0
    const height = (s[peak] ?? 0) - (pd.prominences[p] ?? 0) * rel
    let i = peak
    while (iMin < i && height < (s[i] ?? 0)) i--
    let left = i
    if ((s[i] ?? 0) < height) left += (height - (s[i] ?? 0)) / ((s[i + 1] ?? 0) - (s[i] ?? 0))
    i = peak
    while (i < iMax && height < (s[i] ?? 0)) i++
    let right = i
    if ((s[i] ?? 0) < height) right -= (height - (s[i] ?? 0)) / ((s[i - 1] ?? 0) - (s[i] ?? 0))
    widths.push(right - left)
    widthHeights.push(height)
    leftIps.push(left)
    rightIps.push(right)
  }
  return { widths, widthHeights, leftIps, rightIps }
}

/**
 * Find peaks (scipy.signal.find_peaks): local maxima (the middle of a plateau), filtered in scipy's order by plateau
 * size, height, threshold, distance (higher peaks win), prominence and width; the properties computed along the way
 * are returned.
 *
 * @param x - Signal.
 * @param options - Conditions (a minimum or a [min, max] interval) and the prominence and width parameters.
 * @returns Peak indices and their properties.
 * @concept peak-finding
 * @example
 * ```ts
 * import { findPeaks } from 'whitebox-engine/signal'
 * findPeaks([0, 3, 1, 5, 1, 2, 0], { height: 2.5 }).peaks // => [1, 3]
 * findPeaks([0, 3, 1, 5, 1, 2, 0], { distance: 3 }).peaks // => [3]
 * ```
 */
export function findPeaks(x: TensorLike, options: FindPeaksOptions = {}): PeakResult {
  const s = samples(x, 'findPeaks')
  if (options.distance !== undefined && options.distance < 1)
    throw new WhiteboxError('value', 'findPeaks: distance must be at least 1')
  const lm = localMaxima(s)
  let peaks = lm.mid
  let props: Record<string, number[]> = {}
  const filter = (keep: readonly boolean[]): void => {
    peaks = peaks.filter((_, i) => keep[i])
    props = Object.fromEntries(Object.entries(props).map(([k, v]) => [k, v.filter((_, i) => keep[i])]))
  }
  if (options.plateauSize !== undefined) {
    const sizes = lm.right.map((r, i) => r - (lm.left[i] ?? 0) + 1)
    props = { plateauSizes: sizes, leftEdges: lm.left, rightEdges: lm.right }
    filter(within(sizes, options.plateauSize))
  }
  if (options.height !== undefined) {
    props['peakHeights'] = peaks.map((p) => s[p] ?? 0)
    filter(within(props['peakHeights'], options.height))
  }
  if (options.threshold !== undefined) {
    const leftT = peaks.map((p) => (s[p] ?? 0) - (s[p - 1] ?? 0))
    const rightT = peaks.map((p) => (s[p] ?? 0) - (s[p + 1] ?? 0))
    props['leftThresholds'] = leftT
    props['rightThresholds'] = rightT
    const [lo, hi] = bounds(options.threshold)
    filter(
      leftT.map((l, i) => {
        const r = rightT[i] ?? 0
        return (lo === null || lo <= Math.min(l, r)) && (hi === null || Math.max(l, r) <= hi)
      }),
    )
  }
  if (options.distance !== undefined) {
    const d = Math.ceil(options.distance)
    const keep = new Array<boolean>(peaks.length).fill(true)
    const order = peaks.map((_, i) => i).sort((a, b) => (s[peaks[a] ?? 0] ?? 0) - (s[peaks[b] ?? 0] ?? 0) || a - b)
    for (let r = order.length - 1; r >= 0; r--) {
      const j = order[r] ?? 0
      if (!keep[j]) continue
      for (let k = j - 1; k >= 0 && (peaks[j] ?? 0) - (peaks[k] ?? 0) < d; k--) keep[k] = false
      for (let k = j + 1; k < peaks.length && (peaks[k] ?? 0) - (peaks[j] ?? 0) < d; k++) keep[k] = false
    }
    filter(keep)
  }
  if (options.prominence !== undefined || options.width !== undefined) {
    const pr = peakProminences(s, peaks, options.wlen)
    props['prominences'] = pr.prominences
    props['leftBases'] = pr.leftBases
    props['rightBases'] = pr.rightBases
  }
  if (options.prominence !== undefined) filter(within(props['prominences'] ?? [], options.prominence))
  if (options.width !== undefined) {
    const w = peakWidths(s, peaks, {
      relHeight: options.relHeight ?? 0.5,
      prominenceData: {
        prominences: props['prominences'] ?? [],
        leftBases: props['leftBases'] ?? [],
        rightBases: props['rightBases'] ?? [],
      },
    })
    props['widths'] = w.widths
    props['widthHeights'] = w.widthHeights
    props['leftIps'] = w.leftIps
    props['rightIps'] = w.rightIps
    filter(within(w.widths, options.width))
  }
  return { peaks, properties: props }
}

/**
 * Relative extrema (scipy.signal.argrelextrema, argrelmax, argrelmin): indices where the sample is strictly greater
 * (or less) than its `order` neighbours on each side, with 'clip' or 'wrap' handling of the ends.
 *
 * @param x - Signal.
 * @param kind - 'max' or 'min'.
 * @param options - Neighbourhood order (default 1) and end mode (default 'clip').
 * @returns The indices.
 * @concept peak-finding
 * @example
 * ```ts
 * import { argrelextrema } from 'whitebox-engine/signal'
 * argrelextrema([1, 3, 2, 5, 4, 4, 1], 'max') // => [1, 3]
 * ```
 */
export function argrelextrema(
  x: TensorLike,
  kind: 'max' | 'min',
  options: { readonly order?: number; readonly mode?: 'clip' | 'wrap' } = {},
): number[] {
  const s = samples(x, 'argrelextrema')
  const order = options.order ?? 1
  if (!Number.isInteger(order) || order < 1)
    throw new WhiteboxError('value', 'argrelextrema: order must be a positive integer')
  const n = s.length
  const at = (i: number): number => {
    if (options.mode === 'wrap') return s[((i % n) + n) % n] ?? 0
    return s[Math.min(Math.max(i, 0), n - 1)] ?? 0
  }
  const cmp = (a: number, b: number): boolean => (kind === 'max' ? a > b : a < b)
  const out: number[] = []
  for (let i = 0; i < n; i++) {
    let ok = true
    for (let k = 1; k <= order && ok; k++) ok = cmp(s[i] ?? 0, at(i + k)) && cmp(s[i] ?? 0, at(i - k))
    if (ok) out.push(i)
  }
  return out
}

/** Options for {@link zeroCrossings}. */
export interface ZeroCrossingOptions {
  /** Values with |y| <= threshold count as zero (default 1e-10). */
  readonly threshold?: number
  /** Scale the threshold by this reference (default none). */
  readonly refMagnitude?: number
  /** Whether the first sample counts as a crossing (default true). */
  readonly pad?: boolean
  /** Treat zero as positive (default true); otherwise zero is its own sign. */
  readonly zeroPos?: boolean
}

/**
 * Zero crossings (librosa.zero_crossings): positions where the sign differs from the previous sample's, after
 * values within the threshold are set to zero.
 *
 * @param y - Signal.
 * @param options - Threshold, reference, padding of the first position and the sign of zero.
 * @returns A boolean per sample.
 * @concept zero-crossings
 * @example
 * ```ts
 * import { zeroCrossings } from 'whitebox-engine/signal'
 * zeroCrossings([1, -1, -2, 3, 0, 1], { pad: false }) // => [false, true, false, true, false, false]
 * ```
 */
export function zeroCrossings(y: TensorLike, options: ZeroCrossingOptions = {}): boolean[] {
  const s = samples(y, 'zeroCrossings')
  const thr = (options.threshold ?? 1e-10) * (options.refMagnitude ?? 1)
  const zeroPos = options.zeroPos ?? true
  const sign = Array.from(s, (v) => {
    const c = thr > 0 && Math.abs(v) <= thr ? 0 : v
    if (zeroPos) return c < 0 || Object.is(c, -0) ? 1 : 0
    return Math.sign(c)
  })
  return sign.map((v, i) => (i === 0 ? (options.pad ?? true) : v !== sign[i - 1]))
}
