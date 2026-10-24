import { WhiteboxError } from '../base/index.ts'
import { correlate, correlationLags } from '../numerics/convolution/index.ts'
import { ndtri } from '../numerics/special/index.ts'
import { primalOf, type TensorLike, toFlat } from '../tensor/index.ts'
import { float, samples } from './_shared.ts'

// Similarity of time series: dynamic time warping with dtw-python's step patterns, windows and path, its lower
// bounds (LB_Kim, LB_Keogh), piecewise aggregate approximation and SAX symbols, and the lag of maximum
// cross-correlation.

/** A DTW step pattern (dtw-python's names). */
export type StepPattern = 'symmetric1' | 'symmetric2' | 'asymmetric'

/** Options for {@link dtw}. */
export interface DtwOptions {
  /** Step pattern (default 'symmetric2'). */
  readonly stepPattern?: StepPattern
  /** Sakoe-Chiba band: only cells with |i - j| <= window (default none). */
  readonly window?: number
}

/** A warping: the cumulative distance, its normalised value and the alignment path. */
export interface DtwResult {
  readonly distance: number
  /** distance / (n + m) for symmetric2, / n for asymmetric, NaN for symmetric1 (as dtw-python). */
  readonly normalizedDistance: number
  /** Matched indices into the first and second series, from (0, 0) to (n - 1, m - 1). */
  readonly index1: number[]
  readonly index2: number[]
}

function rows(x: TensorLike): { data: Float64Array; n: number; d: number } {
  const v = float(x)
  if (v.shape.length === 1) return { data: Float64Array.from(primalOf(v).data), n: v.shape[0] ?? 0, d: 1 }
  if (v.shape.length !== 2) throw new WhiteboxError('shape', 'dtw: series are [n] or [n, d]')
  return { data: Float64Array.from(primalOf(v).data), n: v.shape[0] ?? 0, d: v.shape[1] ?? 0 }
}

/**
 * Dynamic time warping (dtw-python's dtw): the alignment of two series minimising the cumulative Euclidean distance
 * under a step pattern (symmetric2 weights diagonal steps 2, symmetric1 counts every cell once, asymmetric steps
 * through the first series), optionally within a Sakoe-Chiba band, with the warping path. Ties go to the first step
 * of the pattern, as dtw-python.
 *
 * @param x - First series `[n]` or `[n, d]`.
 * @param y - Second series `[m]` or `[m, d]`.
 * @param options - Step pattern and window.
 * @returns The distance, normalised distance and path.
 * @throws WhiteboxError with code `value` when no alignment satisfies the window or pattern.
 * @concept dynamic-time-warping
 * @example
 * ```ts
 * import { dtw } from 'whitebox-engine/signal'
 * dtw([0, 1, 2], [0, 1, 1, 2]).distance // => 0
 * ```
 */
export function dtw(x: TensorLike, y: TensorLike, options: DtwOptions = {}): DtwResult {
  const a = rows(x)
  const b = rows(y)
  if (a.d !== b.d) throw new WhiteboxError('shape', 'dtw: series must have the same dimension')
  const n = a.n
  const m = b.n
  const pattern = options.stepPattern ?? 'symmetric2'
  const w = options.window ?? Number.POSITIVE_INFINITY
  const local = (i: number, j: number): number => {
    let s = 0
    for (let k = 0; k < a.d; k++) s += ((a.data[i * a.d + k] ?? 0) - (b.data[j * b.d + k] ?? 0)) ** 2
    return Math.sqrt(s)
  }
  const g = new Float64Array(n * m).fill(Number.POSITIVE_INFINITY)
  const dir = new Int8Array(n * m).fill(-1)
  const at = (i: number, j: number): number =>
    i < 0 || j < 0 ? Number.POSITIVE_INFINITY : (g[i * m + j] ?? Number.POSITIVE_INFINITY)
  for (let i = 0; i < n; i++) {
    for (let j = 0; j < m; j++) {
      if (Math.abs(i - j) > w) continue
      const d = local(i, j)
      if (i === 0 && j === 0) {
        g[0] = d
        continue
      }
      // Candidate steps in dtw-python's pattern order: [cost, direction].
      const steps: [number, number][] =
        pattern === 'symmetric2'
          ? [
              [at(i - 1, j - 1) + 2 * d, 0],
              [at(i, j - 1) + d, 1],
              [at(i - 1, j) + d, 2],
            ]
          : pattern === 'symmetric1'
            ? [
                [at(i - 1, j - 1) + d, 0],
                [at(i, j - 1) + d, 1],
                [at(i - 1, j) + d, 2],
              ]
            : [
                [at(i - 1, j) + d, 2],
                [at(i - 1, j - 1) + d, 0],
                [at(i - 1, j - 2) + d, 3],
              ]
      let best = Number.POSITIVE_INFINITY
      let choice = -1
      for (const [c, s] of steps) {
        if (c < best) {
          best = c
          choice = s
        }
      }
      g[i * m + j] = best
      dir[i * m + j] = choice
    }
  }
  const distance = at(n - 1, m - 1)
  if (!Number.isFinite(distance))
    throw new WhiteboxError('value', 'dtw: no alignment satisfies the window and step pattern')
  const index1: number[] = []
  const index2: number[] = []
  let i = n - 1
  let j = m - 1
  while (true) {
    index1.push(i)
    index2.push(j)
    if (i === 0 && j === 0) break
    const s = dir[i * m + j]
    if (s === 0) {
      i--
      j--
    } else if (s === 1) j--
    else if (s === 2) i--
    else if (s === 3) {
      i--
      j -= 2
      // The asymmetric (1, 2) step skips the intermediate cell of the second series.
    } else throw new WhiteboxError('value', 'dtw: broken path')
  }
  index1.reverse()
  index2.reverse()
  const normalizedDistance =
    pattern === 'symmetric2' ? distance / (n + m) : pattern === 'asymmetric' ? distance / n : Number.NaN
  return { distance, normalizedDistance, index1, index2 }
}

/**
 * The LB_Kim lower bound of the DTW distance (with symmetric1 steps and absolute differences, the simplified
 * four-feature version): the largest of the differences of first, last, maximum and minimum values.
 *
 * @param q - Query.
 * @param c - Candidate.
 * @returns The bound.
 * @concept dynamic-time-warping
 * @example
 * ```ts
 * import { lbKim } from 'whitebox-engine/signal'
 * lbKim([0, 2, 1], [1, 3, 1]) // => 1
 * ```
 */
export function lbKim(q: TensorLike, c: TensorLike): number {
  const a = samples(q, 'lbKim')
  const b = samples(c, 'lbKim')
  let amax = Number.NEGATIVE_INFINITY
  let amin = Number.POSITIVE_INFINITY
  let bmax = Number.NEGATIVE_INFINITY
  let bmin = Number.POSITIVE_INFINITY
  for (const v of a) {
    amax = Math.max(amax, v)
    amin = Math.min(amin, v)
  }
  for (const v of b) {
    bmax = Math.max(bmax, v)
    bmin = Math.min(bmin, v)
  }
  return Math.max(
    Math.abs((a[0] ?? 0) - (b[0] ?? 0)),
    Math.abs((a[a.length - 1] ?? 0) - (b[b.length - 1] ?? 0)),
    Math.abs(amax - bmax),
    Math.abs(amin - bmin),
  )
}

/**
 * The LB_Keogh lower bound of the windowed DTW distance (Keogh 2002): the Euclidean distance from the query to the
 * envelope [L, U] of the candidate over a band of radius r (series of equal length).
 *
 * @param q - Query.
 * @param c - Candidate (same length).
 * @param r - Band radius.
 * @returns sqrt(sum of squared excursions outside the envelope).
 * @concept dynamic-time-warping
 * @example
 * ```ts
 * import { lbKeogh } from 'whitebox-engine/signal'
 * lbKeogh([0, 5, 0], [0, 1, 0], 1) // => 4
 * ```
 */
export function lbKeogh(q: TensorLike, c: TensorLike, r: number): number {
  const a = samples(q, 'lbKeogh')
  const b = samples(c, 'lbKeogh')
  if (a.length !== b.length) throw new WhiteboxError('shape', 'lbKeogh: series must have the same length')
  let s = 0
  for (let i = 0; i < a.length; i++) {
    let lo = Number.POSITIVE_INFINITY
    let hi = Number.NEGATIVE_INFINITY
    for (let j = Math.max(0, i - r); j <= Math.min(b.length - 1, i + r); j++) {
      lo = Math.min(lo, b[j] ?? 0)
      hi = Math.max(hi, b[j] ?? 0)
    }
    const v = a[i] ?? 0
    if (v > hi) s += (v - hi) ** 2
    else if (v < lo) s += (v - lo) ** 2
  }
  return Math.sqrt(s)
}

/**
 * Piecewise aggregate approximation: the means of w equal-length segments (fractional samples split between
 * neighbouring segments when w does not divide n).
 *
 * @param x - Series.
 * @param w - Number of segments.
 * @returns The segment means `[w]`.
 * @concept symbolic-representation
 * @example
 * ```ts
 * import { paa } from 'whitebox-engine/signal'
 * paa([1, 2, 3, 4, 5, 6], 3) // => [1.5, 3.5, 5.5]
 * ```
 */
export function paa(x: TensorLike, w: number): number[] {
  const s = samples(x, 'paa')
  const n = s.length
  if (!Number.isInteger(w) || w < 1 || w > n) throw new WhiteboxError('value', 'paa: 1 <= w <= n')
  const out = new Array<number>(w).fill(0)
  // Sample i covers [i, i + 1) in units of 1 / n of the series; segment k covers [k n / w, (k + 1) n / w).
  for (let k = 0; k < w; k++) {
    const lo = (k * n) / w
    const hi = ((k + 1) * n) / w
    let acc = 0
    for (let i = Math.floor(lo); i < Math.ceil(hi); i++) {
      const overlap = Math.min(hi, i + 1) - Math.max(lo, i)
      if (overlap > 0) acc += overlap * (s[i] ?? 0)
    }
    out[k] = acc / (hi - lo)
  }
  return out
}

/** A symbolic aggregate approximation: the symbol indices and the word. */
export interface SaxResult {
  readonly symbols: number[]
  readonly word: string
  /** The breakpoints of the alphabet (standard normal quantiles). */
  readonly breakpoints: number[]
}

/**
 * Symbolic aggregate approximation (Lin et al. 2003): z-normalise, take the PAA, and map each mean to one of `a`
 * symbols with equiprobable standard-normal breakpoints.
 *
 * @param x - Series.
 * @param w - Word length.
 * @param a - Alphabet size (2 to 26).
 * @returns Symbols, the word (letters from 'a') and the breakpoints.
 * @concept symbolic-representation
 * @example
 * ```ts
 * import { sax } from 'whitebox-engine/signal'
 * sax([1, 2, 3, 4, 5, 6, 7, 8], 4, 3).word // => 'abbc'
 * ```
 */
export function sax(x: TensorLike, w: number, a: number): SaxResult {
  if (!Number.isInteger(a) || a < 2 || a > 26) throw new WhiteboxError('value', 'sax: the alphabet has 2 to 26 symbols')
  const s = samples(x, 'sax')
  let m = 0
  for (const v of s) m += v / s.length
  let sd = 0
  for (const v of s) sd += (v - m) ** 2
  sd = Math.sqrt(sd / s.length)
  const z = Float64Array.from(s, (v) => (sd > 0 ? (v - m) / sd : 0))
  const means = paa(z, w)
  const breakpoints = toFlat(ndtri(Array.from({ length: a - 1 }, (_, k) => (k + 1) / a)))
  const symbols = means.map((v) => breakpoints.filter((b) => v >= b).length)
  return { symbols, word: symbols.map((k) => String.fromCharCode(97 + k)).join(''), breakpoints }
}

/**
 * The MINDIST lower bound between two SAX words of series of length n (Lin et al.): sqrt(n / w) times the Euclidean
 * norm of the cell distances (zero for adjacent symbols, otherwise the gap between breakpoints).
 *
 * @param s1 - First word's symbols.
 * @param s2 - Second word's symbols.
 * @param breakpoints - The alphabet's breakpoints.
 * @param n - Length of the original series.
 * @returns The bound.
 * @concept symbolic-representation
 * @example
 * ```ts
 * import { sax, saxMindist } from 'whitebox-engine/signal'
 * const a = sax([1, 2, 3, 4, 5, 6, 7, 8], 4, 3)
 * saxMindist(a.symbols, a.symbols, a.breakpoints, 8) // => 0
 * ```
 */
export function saxMindist(
  s1: readonly number[],
  s2: readonly number[],
  breakpoints: readonly number[],
  n: number,
): number {
  if (s1.length !== s2.length) throw new WhiteboxError('shape', 'saxMindist: words must have the same length')
  let s = 0
  for (const [i, r] of s1.entries()) {
    const c = s2[i] ?? 0
    if (Math.abs(r - c) <= 1) continue
    const hi = Math.max(r, c)
    const lo = Math.min(r, c)
    s += ((breakpoints[hi - 1] ?? 0) - (breakpoints[lo] ?? 0)) ** 2
  }
  return Math.sqrt(n / s1.length) * Math.sqrt(s)
}

/**
 * The lag that maximises the cross-correlation sum_n x[n + k] y[n] (so a positive lag means x is delayed relative to
 * y), with scipy.signal.correlation_lags.
 *
 * @param x - First signal.
 * @param y - Second signal.
 * @returns The lag in samples.
 * @concept cross-correlation
 * @example
 * ```ts
 * import { crossCorrelationLag } from 'whitebox-engine/signal'
 * crossCorrelationLag([0, 0, 1, 2, 1, 0], [1, 2, 1, 0, 0, 0]) // => 2
 * ```
 */
export function crossCorrelationLag(x: TensorLike, y: TensorLike): number {
  const a = samples(x, 'crossCorrelationLag')
  const b = samples(y, 'crossCorrelationLag')
  const c = toFlat(correlate(a, b, 'full'))
  const lags = correlationLags(a.length, b.length, 'full')
  let best = 0
  for (let i = 1; i < c.length; i++) if ((c[i] ?? 0) > (c[best] ?? 0)) best = i
  return lags[best] ?? 0
}
