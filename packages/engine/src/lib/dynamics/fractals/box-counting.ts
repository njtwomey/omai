/**
 * Box-counting dimension: cover a set with boxes of side `eps`, count the boxes that touch it, and read the dimension
 * off how the count grows as the boxes shrink, `N(eps) ~ eps^-D`. For points (the chaos game, an attractor, the
 * vertices of a curve) and for a pixel mask (the deterministic iteration of a system of maps). Exact for the
 * self-similar sets whose dimension is known (Sierpinski's triangle, `log 3 / log 2`), which is how the estimate is
 * checked.
 */

import { OmaiError } from '../../base/index.ts'

/** The result of a box count. */
export interface BoxCount {
  /** The estimated dimension: the slope of `log N` against `log (1 / eps)`. */
  readonly dimension: number
  /** The intercept of that line. */
  readonly intercept: number
  /** The coefficient of determination of the fit. */
  readonly r2: number
  /** The box sides used, relative to the extent of the set (1 / 2^k). */
  readonly sizes: readonly number[]
  /** The boxes touching the set at each size. */
  readonly counts: readonly number[]
}

function fit(sizes: number[], counts: number[]): BoxCount {
  const x = sizes.map((s) => Math.log(1 / s))
  const y = counts.map((c) => Math.log(c))
  const n = x.length
  const mx = x.reduce((a, b) => a + b, 0) / n
  const my = y.reduce((a, b) => a + b, 0) / n
  let sxx = 0
  let sxy = 0
  let syy = 0
  for (let i = 0; i < n; i++) {
    sxx += ((x[i] ?? 0) - mx) ** 2
    sxy += ((x[i] ?? 0) - mx) * ((y[i] ?? 0) - my)
    syy += ((y[i] ?? 0) - my) ** 2
  }
  const dimension = sxy / sxx
  return { dimension, intercept: my - dimension * mx, r2: syy === 0 ? 1 : (sxy * sxy) / (sxx * syy), sizes, counts }
}

/** Options of {@link boxCountingDimension}. */
export interface BoxCountOptions {
  /** The coarsest level: boxes of side extent / 2^minLevel (default 2). */
  readonly minLevel?: number
  /** The finest level (default `floor(log2(n) / 2)`, so that the boxes are not mostly empty for want of points). */
  readonly maxLevel?: number
}

/**
 * The box-counting dimension of a planar point set: the bounding square is cut into `2^k` by `2^k` boxes for
 * `k = minLevel..maxLevel`, the touched boxes are counted, and the slope of `log N` against `k log 2` is the estimate.
 *
 * @param points - Flat `[x0, y0, x1, y1, ...]`.
 * @param options - The levels.
 * @returns The estimate with its fit and the counts.
 * @throws OmaiError with code `value` for fewer than 16 points, an odd length, or levels that leave fewer than two scales.
 * @concept box-counting-dimension
 * @example
 * ```ts
 * import { fractals } from 'omai-engine/dynamics'
 * // The points of a filled unit square have dimension 2.
 * const pts: number[] = []
 * for (let i = 0; i < 64; i++) for (let j = 0; j < 64; j++) pts.push((i + 0.5) / 64, (j + 0.5) / 64)
 * Math.round(fractals.boxCountingDimension(pts, { minLevel: 1, maxLevel: 5 }).dimension) // => 2
 * ```
 */
export function boxCountingDimension(points: ArrayLike<number>, options: BoxCountOptions = {}): BoxCount {
  const n = Math.floor(points.length / 2)
  if (points.length % 2 !== 0 || n < 16)
    throw new OmaiError('value', 'boxCountingDimension: need at least 16 points as x, y pairs')
  const lo = options.minLevel ?? 2
  const hi = options.maxLevel ?? Math.max(lo + 1, Math.floor(Math.log2(n) / 2))
  if (!(Number.isInteger(lo) && Number.isInteger(hi) && lo >= 0 && hi > lo && hi <= 26))
    throw new OmaiError('value', 'boxCountingDimension: need integer levels 0 <= minLevel < maxLevel <= 26')
  let x0 = Infinity
  let y0 = Infinity
  let x1 = -Infinity
  let y1 = -Infinity
  for (let i = 0; i < n; i++) {
    const x = points[2 * i] as number
    const y = points[2 * i + 1] as number
    if (x < x0) x0 = x
    if (x > x1) x1 = x
    if (y < y0) y0 = y
    if (y > y1) y1 = y
  }
  const side = Math.max(x1 - x0, y1 - y0) * (1 + 1e-12) || 1
  const sizes: number[] = []
  const counts: number[] = []
  for (let k = lo; k <= hi; k++) {
    const m = 2 ** k
    const seen = new Set<number>()
    for (let i = 0; i < n; i++) {
      const bx = Math.min(m - 1, Math.floor((((points[2 * i] as number) - x0) / side) * m))
      const by = Math.min(m - 1, Math.floor((((points[2 * i + 1] as number) - y0) / side) * m))
      seen.add(bx * m + by)
    }
    sizes.push(1 / m)
    counts.push(seen.size)
  }
  return fit(sizes, counts)
}

/**
 * The box-counting dimension of a set of lit pixels: boxes of `2^j` pixels a side for `j = 0..maxLevel`, counting those
 * with a lit pixel.
 *
 * @param mask - Row-major pixels, nonzero for lit; the image is `width` by `height`.
 * @param width - Columns.
 * @param height - Rows.
 * @param options - `minLevel` and `maxLevel` here are the box sides' powers of two in pixels, from `minLevel` (default 1)
 *   up to `maxLevel` (default `floor(log2(min(width, height))) - 2`).
 * @returns The estimate with its fit and the counts.
 * @throws OmaiError with code `value` for a mask of the wrong size or fewer than two usable scales.
 * @concept box-counting-dimension
 * @example
 * ```ts
 * import { fractals } from 'omai-engine/dynamics'
 * // A lit line across a 64 x 64 image has dimension 1.
 * const mask = new Uint8Array(64 * 64)
 * for (let c = 0; c < 64; c++) mask[32 * 64 + c] = 1
 * Math.round(fractals.boxCountingDimensionOfMask(mask, 64, 64, { minLevel: 1, maxLevel: 4 }).dimension) // => 1
 * ```
 */
export function boxCountingDimensionOfMask(
  mask: ArrayLike<number>,
  width: number,
  height: number,
  options: BoxCountOptions = {},
): BoxCount {
  if (mask.length !== width * height)
    throw new OmaiError('value', 'boxCountingDimensionOfMask: mask must be width x height')
  const lo = options.minLevel ?? 1
  const hi = options.maxLevel ?? Math.floor(Math.log2(Math.min(width, height))) - 2
  if (!(Number.isInteger(lo) && Number.isInteger(hi) && lo >= 0 && hi > lo))
    throw new OmaiError('value', 'boxCountingDimensionOfMask: need integer levels 0 <= minLevel < maxLevel')
  const sizes: number[] = []
  const counts: number[] = []
  const side = Math.max(width, height)
  for (let j = lo; j <= hi; j++) {
    const b = 2 ** j
    const cols = Math.ceil(width / b)
    const seen = new Set<number>()
    for (let r = 0; r < height; r++)
      for (let c = 0; c < width; c++)
        if ((mask[r * width + c] ?? 0) !== 0) seen.add(Math.floor(r / b) * cols + Math.floor(c / b))
    sizes.push(b / side)
    counts.push(seen.size)
  }
  return fit(sizes, counts)
}
