/**
 * Regions of images, OpenCV style: connected components with their statistics (`cv2.connectedComponentsWithStats`,
 * over the labelling of `signal/image`: bounding boxes, areas and centroids) and flood filling from a seed
 * (`cv2.floodFill`, OpenCV's scanline algorithm with floating or fixed tolerance ranges and 4- or 8-connectivity).
 */

import { MdmlError } from '../base/index.ts'
import { image } from '../signal/index.ts'
import { type Tensor, type TensorLike, toFlat } from '../tensor/index.ts'
import { f32, greyOf, imageTensor, rasterOf, rasterTensor } from './_image.ts'

/** The output of {@link connectedComponentsWithStats}. */
export interface ComponentStats {
  /** Labels and the background (0) count: `count` includes the background, as OpenCV's return value does. */
  readonly count: number
  readonly labels: Tensor
  /** Per label (background first): [left, top, width, height, area] (cv2.CC_STAT_*). */
  readonly stats: [number, number, number, number, number][]
  /** Per label: the centroid (x, y). */
  readonly centroids: [number, number][]
}

/**
 * Connected components of a binary image with their statistics (cv2.connectedComponentsWithStats): labels 1 ..
 * n - 1 in raster order of each component's first pixel (0 the background), and per label the bounding box, the
 * area and the centroid.
 *
 * @param x - Binary image `[H, W]` (nonzero is foreground).
 * @param options - connectivity 4 or 8 (default 8, OpenCV's).
 * @returns The count (with the background), labels, stats and centroids.
 * @concept connected-components
 * @example
 * ```ts
 * import { connectedComponentsWithStats } from 'mdml-engine/vision'
 * const r = connectedComponentsWithStats([[1, 1, 0], [0, 0, 0], [0, 1, 1]])
 * [r.count, r.stats[1], r.centroids[2]] // => [3, [0, 0, 2, 1, 2], [1.5, 2]]
 * ```
 */
export function connectedComponentsWithStats(
  x: TensorLike,
  options: { readonly connectivity?: 4 | 8 } = {},
): ComponentStats {
  const r = greyOf(x, 'connectedComponentsWithStats')
  const connectivity = options.connectivity ?? 8
  if (connectivity !== 4 && connectivity !== 8)
    throw new MdmlError('value', 'connectedComponentsWithStats: connectivity is 4 or 8')
  const { labels, count } = image.label(
    rasterTensor({ ...r, rank: 2 }),
    image.generateBinaryStructure(2, connectivity === 4 ? 1 : 2),
  )
  const lab = toFlat(labels)
  const n = count + 1
  const x0 = new Array<number>(n).fill(Number.POSITIVE_INFINITY)
  const y0 = new Array<number>(n).fill(Number.POSITIVE_INFINITY)
  const x1 = new Array<number>(n).fill(Number.NEGATIVE_INFINITY)
  const y1 = new Array<number>(n).fill(Number.NEGATIVE_INFINITY)
  const area = new Array<number>(n).fill(0)
  const sx = new Array<number>(n).fill(0)
  const sy = new Array<number>(n).fill(0)
  for (let y = 0; y < r.h; y++)
    for (let c = 0; c < r.w; c++) {
      const l = lab[y * r.w + c] ?? 0
      x0[l] = Math.min(x0[l] ?? 0, c)
      y0[l] = Math.min(y0[l] ?? 0, y)
      x1[l] = Math.max(x1[l] ?? 0, c)
      y1[l] = Math.max(y1[l] ?? 0, y)
      area[l] = (area[l] ?? 0) + 1
      sx[l] = (sx[l] ?? 0) + c
      sy[l] = (sy[l] ?? 0) + y
    }
  const stats: [number, number, number, number, number][] = []
  const centroids: [number, number][] = []
  for (let l = 0; l < n; l++) {
    const a = area[l] ?? 0
    if (a === 0) {
      stats.push([0, 0, 0, 0, 0])
      centroids.push([Number.NaN, Number.NaN])
      continue
    }
    stats.push([x0[l] ?? 0, y0[l] ?? 0, (x1[l] ?? 0) - (x0[l] ?? 0) + 1, (y1[l] ?? 0) - (y0[l] ?? 0) + 1, a])
    centroids.push([(sx[l] ?? 0) / a, (sy[l] ?? 0) / a])
  }
  return { count: n, labels, stats, centroids }
}

/** Options of {@link floodFill}. */
export interface FloodFillOptions {
  /** The largest step down (per channel) a neighbour may take and still be filled (default 0). */
  readonly loDiff?: number | readonly number[]
  /** The largest step up (default 0). */
  readonly upDiff?: number | readonly number[]
  /** 4- or 8-connected filling (default 4). */
  readonly connectivity?: 4 | 8
  /** Compare every pixel with the seed instead of its filled neighbour (cv2.FLOODFILL_FIXED_RANGE). */
  readonly fixedRange?: boolean
  /** Pixels already set here `[H, W]` are barriers (and stay unfilled). */
  readonly mask?: TensorLike
}

/** The output of {@link floodFill}. */
export interface FloodFillResult {
  /** The image with the filled region set to the new value. */
  readonly image: Tensor
  /** 1 where the region was filled, 0 elsewhere `[H, W]`. */
  readonly mask: Tensor
  readonly area: number
  /** The bounding box [x, y, width, height]. */
  readonly rect: [number, number, number, number]
}

/**
 * Flood fill from a seed (cv2.floodFill, Heckbert's scanline algorithm as OpenCV runs it): a pixel joins the region
 * when it differs from an already filled neighbour (or, with fixedRange, from the seed) by at most loDiff below and
 * upDiff above, in every channel; the region is painted with the new value. Differences are taken in single
 * precision, as OpenCV does for float images.
 *
 * @param x - Image `[H, W]` or `[H, W, C]`.
 * @param seed - The seed pixel (x, y).
 * @param newVal - The fill value (one per channel or shared).
 * @param options - Tolerances, connectivity, fixed range and a barrier mask.
 * @returns The filled image, the region mask, its area and bounding box.
 * @concept flood-fill
 * @example
 * ```ts
 * import { floodFill } from 'mdml-engine/vision'
 * import { toArray } from 'mdml-engine/tensor'
 * toArray(floodFill([[0, 0, 1], [0, 1, 0], [1, 0, 0]], [0, 0], 0.5).image) // => [[0.5, 0.5, 1], [0.5, 1, 0], [1, 0, 0]]
 * ```
 */
export function floodFill(
  x: TensorLike,
  seed: readonly [number, number],
  newVal: number | readonly number[],
  options: FloodFillOptions = {},
): FloodFillResult {
  const r = rasterOf(x, 'floodFill')
  const { h, w, c } = r
  const [sx, sy] = seed
  if (!(Number.isInteger(sx) && Number.isInteger(sy) && sx >= 0 && sy >= 0 && sx < w && sy < h))
    throw new MdmlError('value', 'floodFill: the seed lies outside the image')
  const perChannel = (v: number | readonly number[] | undefined): number[] =>
    Array.from({ length: c }, (_, k) => f32(typeof v === 'number' ? v : (v?.[k] ?? v?.[0] ?? 0)))
  const lo = perChannel(options.loDiff)
  const up = perChannel(options.upDiff)
  if (lo.some((v) => v < 0) || up.some((v) => v < 0)) throw new MdmlError('value', 'floodFill: negative tolerance')
  const fill = perChannel(newVal)
  const data = Float64Array.from(r.data)
  const img32 = Float64Array.from(data, f32)
  const mw = w + 2
  const mask = new Uint8Array(mw * (h + 2))
  for (let i = 0; i < mw; i++) {
    mask[i] = 1
    mask[(h + 1) * mw + i] = 1
  }
  for (let y = 0; y < h + 2; y++) {
    mask[y * mw] = 1
    mask[y * mw + w + 1] = 1
  }
  if (options.mask !== undefined) {
    const m = greyOf(options.mask, 'floodFill')
    if (m.h !== h || m.w !== w) throw new MdmlError('shape', 'floodFill: the mask is [H, W]')
    for (let y = 0; y < h; y++)
      for (let xx = 0; xx < w; xx++) if ((m.data[y * w + xx] ?? 0) !== 0) mask[(y + 1) * mw + xx + 1] = 1
  }
  const filled = new Uint8Array(h * w)
  const M = (y: number, xx: number): number => mask[(y + 1) * mw + xx + 1] ?? 1
  const setM = (y: number, xx: number): void => {
    mask[(y + 1) * mw + xx + 1] = 1
    filled[y * w + xx] = 1
  }
  const seedVal = Array.from({ length: c }, (_, k) => img32[(sy * w + sx) * c + k] ?? 0)
  // diff(a, b): a within [b - lo, b + up] per channel, the difference in single precision.
  const diffPx = (ya: number, xa: number, yb: number, xb: number): boolean => {
    for (let k = 0; k < c; k++) {
      const d = f32((img32[(ya * w + xa) * c + k] ?? 0) - (img32[(yb * w + xb) * c + k] ?? 0))
      if (!(-(lo[k] ?? 0) <= d && d <= (up[k] ?? 0))) return false
    }
    return true
  }
  const diffSeed = (ya: number, xa: number): boolean => {
    for (let k = 0; k < c; k++) {
      const d = f32((img32[(ya * w + xa) * c + k] ?? 0) - (seedVal[k] ?? 0))
      if (!(-(lo[k] ?? 0) <= d && d <= (up[k] ?? 0))) return false
    }
    return true
  }
  const eight = (options.connectivity ?? 4) === 8
  const fixed = options.fixedRange === true
  let area = 0
  let xMin = sx
  let xMax = sx
  let yMin = sy
  let yMax = sy
  if (M(sy, sx) === 0) {
    let L = sx
    let R = sx
    setM(sy, sx)
    while (M(sy, R + 1) === 0 && (fixed ? diffSeed(sy, R + 1) : diffPx(sy, R + 1, sy, R))) setM(sy, ++R)
    while (M(sy, L - 1) === 0 && (fixed ? diffSeed(sy, L - 1) : diffPx(sy, L - 1, sy, L))) setM(sy, --L)
    xMax = R
    xMin = L
    const stack: [number, number, number, number, number, number][] = [[sy, L, R, R + 1, R, 1]]
    while (stack.length > 0) {
      const [yc, l, rr, pl, pr, dir0] = stack.pop() as [number, number, number, number, number, number]
      const e = eight ? 1 : 0
      const spans: [number, number, number][] = [
        [-dir0, l - e, rr + e],
        [dir0, l - e, pl - 1],
        [dir0, pr + 1, rr + e],
      ]
      const length = rr - l
      area += length + 1
      xMax = Math.max(xMax, rr)
      xMin = Math.min(xMin, l)
      yMax = Math.max(yMax, yc)
      yMin = Math.min(yMin, yc)
      for (const [dir, left, right] of spans) {
        const yn = yc + dir
        for (let i = left; i <= right; i++) {
          if (M(yn, i) !== 0) continue
          // The pixel joins when it matches the seed (fixed range), the pixel above or below it (4-connected), or one
          // of the three filled pixels it touches on the current span (8-connected).
          let ok: boolean
          if (fixed) ok = diffSeed(yn, i)
          else if (!eight) ok = diffPx(yn, i, yc, i)
          else {
            const idx = i - l - 1
            ok =
              (idx >= 0 && idx <= length && diffPx(yn, i, yc, i - 1)) ||
              (idx + 1 >= 0 && idx + 1 <= length && diffPx(yn, i, yc, i)) ||
              (idx + 2 >= 0 && idx + 2 <= length && diffPx(yn, i, yc, i + 1))
          }
          if (!ok) continue
          let j = i
          setM(yn, i)
          while (M(yn, j - 1) === 0 && (fixed ? diffSeed(yn, j - 1) : diffPx(yn, j - 1, yn, j))) setM(yn, --j)
          for (;;) {
            const k = i + 1
            if (M(yn, k) !== 0) break
            let grow: boolean
            if (fixed) grow = diffSeed(yn, k)
            else if (!eight) grow = diffPx(yn, k, yn, k - 1) || (diffPx(yn, k, yc, k) && k <= rr)
            else {
              const idx = k - l - 1
              grow =
                diffPx(yn, k, yn, k - 1) ||
                (idx >= 0 && idx <= length && diffPx(yn, k, yc, k - 1)) ||
                (idx + 1 >= 0 && idx + 1 <= length && diffPx(yn, k, yc, k)) ||
                (idx + 2 >= 0 && idx + 2 <= length && diffPx(yn, k, yc, k + 1))
            }
            if (!grow) break
            i = k
            setM(yn, i)
          }
          stack.push([yn, j, i, l, rr, -dir])
          i++
        }
      }
    }
  }
  for (let i = 0; i < h * w; i++) if (filled[i] === 1) for (let k = 0; k < c; k++) data[i * c + k] = fill[k] ?? 0
  return {
    image: rasterTensor({ ...r, data }),
    mask: imageTensor(Float64Array.from(filled), h, w),
    area,
    rect: [xMin, yMin, xMax - xMin + 1, yMax - yMin + 1],
  }
}
