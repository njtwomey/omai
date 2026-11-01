/**
 * Regions of images, OpenCV style: connected components with their statistics (`cv2.connectedComponentsWithStats`,
 * over the labelling of `signal/image`: bounding boxes, areas and centroids) and flood filling from a seed
 * (`cv2.floodFill`, OpenCV's scanline algorithm with floating or fixed tolerance ranges and 4- or 8-connectivity).
 */

import { MdmlError } from '../base/index.ts'
import { image } from '../signal/index.ts'
import { type Tensor, type TensorLike, toFlat } from '../tensor/index.ts'
import { type Algorithm, runAlgorithm, stateFromJson, stateToJson } from '../trace/index.ts'
import { f32, greyOf, imageTensor, rasterOf, rasterTensor } from './_image.ts'
import type { GreedyPath } from './greedy-contract.ts'

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
 * The state of {@link floodFillAlgorithm}: the image as painted so far, the barrier mask (padded by one pixel all round),
 * which pixels were filled, the stack of spans still to scan and the region's size. Steps update them in place; keep a
 * copy of a state you want to look at later.
 */
export interface FloodFillState {
  readonly h: number
  readonly w: number
  readonly c: number
  readonly rank: 2 | 3
  /** The image with the filled pixels painted. */
  readonly data: Float64Array
  /** 1 for barriers and filled pixels, `(h + 2)` by `(w + 2)`. */
  readonly mask: Uint8Array
  /** 1 where the region was filled, `h` by `w`. */
  readonly filled: Uint8Array
  /** Spans waiting to be scanned: `[row, left, right, previous left, previous right, direction]`. */
  readonly stack: [number, number, number, number, number, number][]
  area: number
  xMin: number
  xMax: number
  yMin: number
  yMax: number
}

/**
 * Flood fill from a seed (cv2.floodFill, Heckbert's scanline algorithm as OpenCV runs it) as a stepped algorithm, one span
 * per step: the span popped from the stack is scanned in the rows above and below, each pixel that matches (a pixel joins
 * when it differs from an already filled neighbour, or with fixedRange from the seed, by at most loDiff below and upDiff
 * above in every channel) grows into a new span that is pushed, and the pixels are painted with the new value. Differences
 * are taken in single precision, as OpenCV does for float images. It logs the span's length and the area so far, and is
 * done (`converged`) when the stack is empty. The state serialises; read the result with {@link floodFillResult}.
 *
 * @param x - Image `[H, W]` or `[H, W, C]`.
 * @param seed - The seed pixel (x, y).
 * @param newVal - The fill value (one per channel or shared).
 * @param options - Tolerances, connectivity, fixed range and a barrier mask.
 * @returns The algorithm.
 * @throws MdmlError with code `value` for a seed outside the image or a negative tolerance, and `shape` for a mask of
 * the wrong shape.
 * @concept flood-fill
 * @example
 * ```ts
 * import { floodFillAlgorithm, floodFillResult } from 'mdml-engine/vision'
 * import { runAlgorithm } from 'mdml-engine/trace'
 * const run = runAlgorithm(floodFillAlgorithm([[0, 0, 1], [0, 1, 0], [1, 0, 0]], [0, 0], 0.5), { maxSteps: 20 })
 * floodFillResult(run.state).area // => 3
 * ```
 */
export function floodFillAlgorithm(
  x: TensorLike,
  seed: readonly [number, number],
  newVal: number | readonly number[],
  options: FloodFillOptions = {},
): Algorithm<FloodFillState, GreedyPath> {
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
  const img32 = Float64Array.from(r.data, f32)
  const mw = w + 2
  const barriers = new Uint8Array(mw * (h + 2))
  for (let i = 0; i < mw; i++) {
    barriers[i] = 1
    barriers[(h + 1) * mw + i] = 1
  }
  for (let y = 0; y < h + 2; y++) {
    barriers[y * mw] = 1
    barriers[y * mw + w + 1] = 1
  }
  if (options.mask !== undefined) {
    const m = greyOf(options.mask, 'floodFill')
    if (m.h !== h || m.w !== w) throw new MdmlError('shape', 'floodFill: the mask is [H, W]')
    for (let y = 0; y < h; y++)
      for (let xx = 0; xx < w; xx++) if ((m.data[y * w + xx] ?? 0) !== 0) barriers[(y + 1) * mw + xx + 1] = 1
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
  return {
    name: 'vision/flood-fill',
    config: { seed: [sx, sy], connectivity: eight ? 8 : 4, fixedRange: fixed, h, w },
    init: () => {
      const st: FloodFillState = {
        h,
        w,
        c,
        rank: r.rank,
        data: Float64Array.from(r.data),
        mask: Uint8Array.from(barriers),
        filled: new Uint8Array(h * w),
        stack: [],
        area: 0,
        xMin: sx,
        xMax: sx,
        yMin: sy,
        yMax: sy,
      }
      const M = (y: number, xx: number): number => st.mask[(y + 1) * mw + xx + 1] ?? 1
      const setM = (y: number, xx: number): void => {
        st.mask[(y + 1) * mw + xx + 1] = 1
        st.filled[y * w + xx] = 1
        for (let k = 0; k < c; k++) st.data[(y * w + xx) * c + k] = fill[k] ?? 0
      }
      if (M(sy, sx) === 0) {
        let L = sx
        let R = sx
        setM(sy, sx)
        while (M(sy, R + 1) === 0 && (fixed ? diffSeed(sy, R + 1) : diffPx(sy, R + 1, sy, R))) setM(sy, ++R)
        while (M(sy, L - 1) === 0 && (fixed ? diffSeed(sy, L - 1) : diffPx(sy, L - 1, sy, L))) setM(sy, --L)
        st.xMax = R
        st.xMin = L
        st.stack.push([sy, L, R, R + 1, R, 1])
      }
      return st
    },
    step: (st, ctx) => {
      const M = (y: number, xx: number): number => st.mask[(y + 1) * mw + xx + 1] ?? 1
      const setM = (y: number, xx: number): void => {
        st.mask[(y + 1) * mw + xx + 1] = 1
        st.filled[y * w + xx] = 1
        for (let k = 0; k < c; k++) st.data[(y * w + xx) * c + k] = fill[k] ?? 0
      }
      const [yc, l, rr, pl, pr, dir0] = st.stack.pop() as [number, number, number, number, number, number]
      const e = eight ? 1 : 0
      const spans: [number, number, number][] = [
        [-dir0, l - e, rr + e],
        [dir0, l - e, pl - 1],
        [dir0, pr + 1, rr + e],
      ]
      const length = rr - l
      st.area += length + 1
      st.xMax = Math.max(st.xMax, rr)
      st.xMin = Math.min(st.xMin, l)
      st.yMax = Math.max(st.yMax, yc)
      st.yMin = Math.min(st.yMin, yc)
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
          st.stack.push([yn, j, i, l, rr, -dir])
          i++
        }
      }
      ctx.log('fill/span', length + 1)
      ctx.log('fill/area', st.area)
      return st
    },
    done: (st) => (st.stack.length === 0 ? 'converged' : undefined),
    serialise: (st) => stateToJson(st),
    deserialise: (j) => stateFromJson(j) as FloodFillState,
  }
}

/**
 * The result of a flood-fill state: the painted image, the region mask, its area and bounding box.
 *
 * @param state - A (finished) state of {@link floodFillAlgorithm}.
 * @returns The filled image, the region mask, its area and bounding box.
 * @concept flood-fill
 * @example
 * ```ts
 * import { floodFillAlgorithm, floodFillResult } from 'mdml-engine/vision'
 * import { runAlgorithm } from 'mdml-engine/trace'
 * floodFillResult(runAlgorithm(floodFillAlgorithm([[0, 1]], [0, 0], 0.5), { maxSteps: 5 }).state).rect // => [0, 0, 1, 1]
 * ```
 */
export function floodFillResult(state: FloodFillState): FloodFillResult {
  return {
    image: rasterTensor({ h: state.h, w: state.w, c: state.c, rank: state.rank, data: state.data }),
    mask: imageTensor(Float64Array.from(state.filled), state.h, state.w),
    area: state.area,
    rect: [state.xMin, state.yMin, state.xMax - state.xMin + 1, state.yMax - state.yMin + 1],
  }
}

/**
 * Flood fill from a seed (cv2.floodFill, Heckbert's scanline algorithm as OpenCV runs it): a pixel joins the region
 * when it differs from an already filled neighbour (or, with fixedRange, from the seed) by at most loDiff below and
 * upDiff above, in every channel; the region is painted with the new value. Differences are taken in single
 * precision, as OpenCV does for float images. {@link floodFillAlgorithm} run to the end.
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
  return floodFillResult(
    runAlgorithm(floodFillAlgorithm(x, seed, newVal, options), { maxSteps: Number.MAX_SAFE_INTEGER }).state,
  )
}
