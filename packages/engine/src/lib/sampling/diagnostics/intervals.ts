/**
 * Credible intervals from posterior draws, as ArviZ: the highest density interval (the narrowest interval holding a
 * given mass), its multimodal form for discrete draws as a union of intervals, and the equal-tailed interval between
 * two quantiles.
 */

import { quantileOf, WhiteboxError } from '../../base/index.ts'
import { asTensor, type Tensor, type TensorLike, toFlat } from '../../tensor/index.ts'

function values(x: TensorLike, name: string): Float64Array {
  const v = Float64Array.from(toFlat(asTensor(x) as Tensor))
  if (v.length === 0) throw new WhiteboxError('value', `${name}: no draws`)
  return v
}

function checkProb(prob: number, name: string): void {
  if (!(prob > 0 && prob <= 1)) throw new WhiteboxError('value', `${name}: prob must lie in (0, 1]`)
}

/** numpy's mod for a positive divisor: always in [0, m). */
function mod(x: number, m: number): number {
  const r = x % m
  return r < 0 ? r + m : r
}

function standardise(x: number): number {
  return mod(x + Math.PI, 2 * Math.PI) - Math.PI
}

/**
 * The circular mean of angles in radians, in [-pi, pi) (scipy.stats.circmean with low -pi and high pi).
 *
 * @param x - Angles.
 * @returns The mean direction.
 */
function circularMean(x: Float64Array): number {
  let s = 0
  let c = 0
  for (const v of x) {
    s += Math.sin(v + Math.PI)
    c += Math.cos(v + Math.PI)
  }
  return mod(Math.atan2(s / x.length, c / x.length), 2 * Math.PI) - Math.PI
}

/** The narrowest window of sorted draws holding floor(prob n) + 1 of them (ArviZ's nearest HDI). */
function nearest(sorted: Float64Array, prob: number): [number, number] {
  const n = sorted.length
  const inc = Math.min(Math.floor(prob * n), n - 1)
  const count = n - inc
  let best = 0
  let width = Number.POSITIVE_INFINITY
  for (let i = 0; i < count; i++) {
    const w = (sorted[i + inc] ?? 0) - (sorted[i] ?? 0)
    if (w < width) {
      width = w
      best = i
    }
  }
  return [sorted[best] ?? Number.NaN, sorted[best + inc] ?? Number.NaN]
}

/** Options of {@link hdi}. */
export interface HdiOptions {
  /** Angles in radians on [-pi, pi): the interval is found around the circular mean and may wrap (default false). */
  readonly circular?: boolean
}

/**
 * The highest density interval of draws (ArviZ `hdi`, method `nearest`): the narrowest interval between two draws
 * that contains a fraction `prob` of them, the shortest credible interval for a unimodal posterior. All draws are
 * pooled (any shape). For circular draws the interval is found after centring on the circular mean, so it may wrap
 * around +-pi (then low > high).
 *
 * @param x - Draws (any shape, pooled).
 * @param prob - The probability mass in (0, 1] (ArviZ's default is 0.89; there is no default here).
 * @param options - `circular`.
 * @returns `[low, high]`.
 * @throws WhiteboxError with code `value` for an empty input or prob outside (0, 1].
 * @concept highest-density-interval
 * @reference R. J. Hyndman, Computing and graphing highest density regions (The American Statistician 1996).
 * @example
 * ```ts
 * import { hdi } from 'whitebox-engine/sampling'
 * hdi([0, 1, 2, 3, 4, 5, 6, 7, 8, 30], 0.8) // => [0, 8]
 * ```
 */
export function hdi(x: TensorLike, prob: number, options: HdiOptions = {}): [number, number] {
  checkProb(prob, 'hdi')
  let v = values(x, 'hdi')
  if (options.circular === true) {
    const m = circularMean(v)
    v = v.map((a) => standardise(a - m))
    const [lo, hi] = nearest(v.sort(), prob)
    return [standardise(lo + m), standardise(hi + m)]
  }
  return nearest(v.sort(), prob)
}

/** Options of {@link hdiModes}. */
export interface HdiModesOptions {
  /** The most intervals to return, by probability (default 10, as ArviZ). */
  readonly maxModes?: number
}

/**
 * The highest density region of discrete draws as a union of intervals (ArviZ `hdi` with method `multimodal` on
 * integer data): values are taken in order of decreasing frequency until their mass reaches `prob`, and runs of
 * consecutive values form the intervals, so a bimodal posterior gets two.
 *
 * @param x - Integer-valued draws (any shape, pooled).
 * @param prob - The probability mass in (0, 1].
 * @param options - The most intervals to return.
 * @returns The intervals `[low, high]` in increasing order, each with its probability.
 * @throws WhiteboxError with code `value` for an empty input, prob outside (0, 1] or non-integer draws.
 * @concept highest-density-interval
 * @example
 * ```ts
 * import { hdiModes } from 'whitebox-engine/sampling'
 * hdiModes([0, 0, 0, 1, 1, 5, 5, 5, 6, 6], 0.9).map((m) => m.interval) // => [[0, 1], [5, 6]]
 * ```
 */
export function hdiModes(
  x: TensorLike,
  prob: number,
  options: HdiModesOptions = {},
): { readonly interval: [number, number]; readonly prob: number }[] {
  checkProb(prob, 'hdiModes')
  const v = values(x, 'hdiModes')
  if (!v.every(Number.isInteger)) {
    throw new WhiteboxError('value', 'hdiModes: draws must be integers (continuous draws need a density estimate)')
  }
  const counts = new Map<number, number>()
  for (const a of v) counts.set(a, (counts.get(a) ?? 0) + 1)
  const bins = [...counts.keys()].sort((a, b) => a - b)
  const probs = bins.map((b) => (counts.get(b) ?? 0) / v.length)
  // Bins by decreasing probability (a stable sort reversed, as numpy's argsort(...)[::-1]).
  const order = bins
    .map((_, i) => i)
    .sort((a, b) => (probs[a] ?? 0) - (probs[b] ?? 0) || a - b)
    .reverse()
  // The first prefix whose mass reaches prob (numpy searchsorted on the cumulative sums, side left, plus one).
  let cum = 0
  let size = order.length
  for (let i = 0; i < order.length; i++) {
    cum += probs[order[i] ?? 0] ?? 0
    if (cum >= prob) {
      size = i + 1
      break
    }
  }
  const chosen = order.slice(0, size).sort((a, b) => a - b)
  const out: { interval: [number, number]; prob: number }[] = []
  for (const i of chosen) {
    const b = bins[i] ?? 0
    const last = out[out.length - 1]
    if (last !== undefined && b - last.interval[1] <= 1.01) {
      last.interval[1] = b
      last.prob += probs[i] ?? 0
    } else out.push({ interval: [b, b], prob: probs[i] ?? 0 })
  }
  const maxModes = options.maxModes ?? 10
  if (out.length <= maxModes) return out
  const top = new Set(
    out
      .map((m, i) => [m.prob, i] as const)
      .sort((a, b) => b[0] - a[0])
      .slice(0, maxModes)
      .map((p) => p[1]),
  )
  return out.filter((_, i) => top.has(i))
}

/**
 * The equal-tailed interval of draws (ArviZ `eti`): the (1 - prob) / 2 and (1 + prob) / 2 quantiles with numpy's
 * linear interpolation, so each tail holds (1 - prob) / 2 of the mass.
 *
 * @param x - Draws (any shape, pooled).
 * @param prob - The probability mass in (0, 1].
 * @returns `[low, high]`.
 * @throws WhiteboxError with code `value` for an empty input or prob outside (0, 1].
 * @concept credible-interval
 * @example
 * ```ts
 * import { eti } from 'whitebox-engine/sampling'
 * eti([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10], 0.8) // => [1, 9]
 * ```
 */
export function eti(x: TensorLike, prob: number): [number, number] {
  checkProb(prob, 'eti')
  const v = values(x, 'eti')
  const edge = (1 - prob) / 2
  return [quantileOf(v, edge), quantileOf(v, 1 - edge)]
}
