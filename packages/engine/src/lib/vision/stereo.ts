/**
 * Stereo correspondence on rectified pairs: block-matching disparity (SAD, SSD or normalised cross-correlation over a
 * window, winner-takes-all with parabolic sub-pixel refinement and a left-right consistency check), Hirschmüller's
 * semi-global matching (pixel costs aggregated along 4 or 8 paths with the P1 and P2 smoothness penalties), and depth
 * and 3-D points from disparity (cv2.reprojectImageTo3D).
 */

// TODO: OpenCV's StereoBM pre-filtering and texture thresholds, and SGBM's Birchfield-Tomasi sampling-insensitive cost.

import { MdmlError } from '../base/index.ts'
import type { Tensor, TensorLike } from '../tensor/index.ts'
import { greyOf, imageTensor } from './_image.ts'

/** The window costs of {@link blockMatching}. */
export type MatchCost = 'sad' | 'ssd' | 'ncc'

/** Options of the disparity estimators. */
export interface DisparityOptions {
  /** Disparities searched (default 16) from minDisparity (default 0). */
  readonly numDisparities?: number
  readonly minDisparity?: number
  /** The odd window side (default 5 for block matching, 3 for semi-global matching). */
  readonly blockSize?: number
  /** Check left-right consistency (default true), allowing this difference (default 1). */
  readonly lrCheck?: boolean
  readonly maxDiff?: number
}

/** The cost volume [h][w][d] of matching left (y, x) with right (y, x - d). */
function costVolume(
  L: Float64Array,
  R: Float64Array,
  h: number,
  w: number,
  dmin: number,
  nd: number,
  half: number,
  cost: MatchCost,
  rightReference: boolean,
): Float64Array {
  const vol = new Float64Array(h * w * nd).fill(Number.POSITIVE_INFINITY)
  const at = (img: Float64Array, y: number, x: number): number =>
    img[Math.min(h - 1, Math.max(0, y)) * w + Math.min(w - 1, Math.max(0, x))] ?? 0
  for (let y = 0; y < h; y++)
    for (let x = 0; x < w; x++)
      for (let k = 0; k < nd; k++) {
        const d = dmin + k
        // Reference left: compare left (x) with right (x - d); reference right: right (x) with left (x + d).
        const xo = rightReference ? x + d : x - d
        if (xo < 0 || xo >= w) continue
        const A = rightReference ? R : L
        const B = rightReference ? L : R
        let s = 0
        let sa = 0
        let sb = 0
        let saa = 0
        let sbb = 0
        let sab = 0
        let n = 0
        for (let dy = -half; dy <= half; dy++)
          for (let dx = -half; dx <= half; dx++) {
            const a = at(A, y + dy, x + dx)
            const b = at(B, y + dy, xo + dx)
            if (cost === 'sad') s += Math.abs(a - b)
            else if (cost === 'ssd') s += (a - b) ** 2
            else {
              sa += a
              sb += b
              saa += a * a
              sbb += b * b
              sab += a * b
              n++
            }
          }
        if (cost === 'ncc') {
          const va = saa - (sa * sa) / n
          const vb = sbb - (sb * sb) / n
          const cov = sab - (sa * sb) / n
          s = va > 1e-12 && vb > 1e-12 ? -cov / Math.sqrt(va * vb) : 0
        }
        vol[(y * w + x) * nd + k] = s
      }
  return vol
}

/** Winner-takes-all over a cost volume with parabolic sub-pixel refinement; -1 where nothing was comparable. */
function winners(vol: Float64Array, h: number, w: number, dmin: number, nd: number): Float64Array {
  const out = new Float64Array(h * w).fill(-1)
  for (let p = 0; p < h * w; p++) {
    let best = -1
    let bv = Number.POSITIVE_INFINITY
    for (let k = 0; k < nd; k++) {
      const v = vol[p * nd + k] ?? Number.POSITIVE_INFINITY
      if (v < bv) {
        bv = v
        best = k
      }
    }
    if (best < 0) continue
    let d = dmin + best
    if (best > 0 && best < nd - 1) {
      const a = vol[p * nd + best - 1] ?? 0
      const c = vol[p * nd + best + 1] ?? 0
      const denom = a - 2 * bv + c
      if (Number.isFinite(a) && Number.isFinite(c) && denom > 0) d += (0.5 * (a - c)) / denom
    }
    out[p] = d
  }
  return out
}

function consistency(dl: Float64Array, dr: Float64Array, h: number, w: number, maxDiff: number): Float64Array {
  return Float64Array.from(dl, (d, p) => {
    if (d < 0) return -1
    const x = p % w
    const xr = Math.round(x - d)
    if (xr < 0 || xr >= w) return -1
    const e = dr[p - x + xr] ?? -1
    return e >= 0 && Math.abs(d - e) <= maxDiff ? d : -1
  })
}

function pairOf(
  left: TensorLike,
  right: TensorLike,
  name: string,
): { L: Float64Array; R: Float64Array; h: number; w: number } {
  const l = greyOf(left, name)
  const r = greyOf(right, name)
  if (l.h !== r.h || l.w !== r.w) throw new MdmlError('shape', `${name}: the images differ in shape`)
  return { L: l.data, R: r.data, h: l.h, w: l.w }
}

/**
 * Disparity by block matching on a rectified pair: for each left pixel, the disparity d in the search range whose
 * window (borders replicated) best matches the right image at x - d by SAD, SSD or normalised cross-correlation, refined
 * to sub-pixel by a parabola through the neighbouring costs; with lrCheck, disparities that disagree with the
 * right-referenced match by more than maxDiff become -1 (invalid).
 *
 * @param left - Left image `[H, W]`.
 * @param right - Right image `[H, W]`.
 * @param options - Search range, window, cost and the consistency check.
 * @returns The disparity map `[H, W]` (-1 where invalid).
 * @concept stereo-matching
 * @example
 * ```ts
 * import { blockMatching } from 'mdml-engine/vision'
 * import { toArray } from 'mdml-engine/tensor'
 * const tex = (x: number, y: number) => Math.sin(x * 1.3) * Math.cos(y * 0.7) + Math.sin(x * 0.4 + y)
 * const L = Array.from({ length: 12 }, (_, y) => Array.from({ length: 24 }, (_, x) => tex(x, y)))
 * const R = Array.from({ length: 12 }, (_, y) => Array.from({ length: 24 }, (_, x) => tex(x + 3, y)))
 * Math.round((toArray(blockMatching(L, R, { numDisparities: 8 })) as number[][])[6][12]) // => 3
 * ```
 */
export function blockMatching(
  left: TensorLike,
  right: TensorLike,
  options: DisparityOptions & { readonly cost?: MatchCost } = {},
): Tensor {
  const { L, R, h, w } = pairOf(left, right, 'blockMatching')
  const nd = options.numDisparities ?? 16
  const dmin = options.minDisparity ?? 0
  const half = Math.floor((options.blockSize ?? 5) / 2)
  const cost = options.cost ?? 'sad'
  const dl = winners(costVolume(L, R, h, w, dmin, nd, half, cost, false), h, w, dmin, nd)
  if (options.lrCheck === false) return imageTensor(dl, h, w)
  const dr = winners(costVolume(L, R, h, w, dmin, nd, half, cost, true), h, w, dmin, nd)
  return imageTensor(consistency(dl, dr, h, w, options.maxDiff ?? 1), h, w)
}

/** Aggregate a cost volume along the given directions (Hirschmüller's recursion), summed over paths. */
function aggregate(
  C: Float64Array,
  h: number,
  w: number,
  nd: number,
  dirs: readonly (readonly [number, number])[],
  P1: number,
  P2: number,
): Float64Array {
  const S = new Float64Array(h * w * nd)
  for (const [dy, dx] of dirs) {
    const Lr = new Float64Array(h * w * nd)
    // Visit pixels so that p - r is always visited before p.
    const ys = dy > 0 ? [...Array(h).keys()] : dy < 0 ? [...Array(h).keys()].reverse() : [...Array(h).keys()]
    const xs = dx > 0 ? [...Array(w).keys()] : dx < 0 ? [...Array(w).keys()].reverse() : [...Array(w).keys()]
    for (const y of ys)
      for (const x of xs) {
        const p = y * w + x
        const py = y - dy
        const px = x - dx
        const inside = py >= 0 && py < h && px >= 0 && px < w
        let prevMin = 0
        if (inside) {
          prevMin = Number.POSITIVE_INFINITY
          for (let k = 0; k < nd; k++) prevMin = Math.min(prevMin, Lr[(py * w + px) * nd + k] ?? 0)
        }
        for (let k = 0; k < nd; k++) {
          const c = C[p * nd + k] ?? 0
          let v = c
          if (inside) {
            const q = (py * w + px) * nd
            const same = Lr[q + k] ?? 0
            const lo = k > 0 ? (Lr[q + k - 1] ?? 0) + P1 : Number.POSITIVE_INFINITY
            const hi = k < nd - 1 ? (Lr[q + k + 1] ?? 0) + P1 : Number.POSITIVE_INFINITY
            v = c + Math.min(same, lo, hi, prevMin + P2) - prevMin
          }
          Lr[p * nd + k] = v
          S[p * nd + k] = (S[p * nd + k] ?? 0) + v
        }
      }
  }
  return S
}

/**
 * Semi-global matching (Hirschmüller 2008): window SAD costs (out-of-range disparities cost the largest observed cost)
 * aggregated along 4 or 8 straight paths with penalty P1 for a disparity change of one and P2 for larger jumps, then
 * winner-takes-all with sub-pixel refinement and an optional left-right check.
 *
 * @param left - Left image `[H, W]`.
 * @param right - Right image `[H, W]`.
 * @param options - Search range, window (default 3), P1 and P2 (defaults 0.1 and 0.4 of a window's worth of unit
 *   cost), paths (4 or 8, default 8) and the consistency check.
 * @returns The disparity map (-1 where invalid).
 * @concept stereo-matching
 * @example
 * ```ts
 * import { semiGlobalMatching } from 'mdml-engine/vision'
 * import { toArray } from 'mdml-engine/tensor'
 * const tex = (x: number, y: number) => Math.sin(x * 1.3) * Math.cos(y * 0.7) + Math.sin(x * 0.4 + y)
 * const L = Array.from({ length: 10 }, (_, y) => Array.from({ length: 20 }, (_, x) => tex(x, y)))
 * const R = Array.from({ length: 10 }, (_, y) => Array.from({ length: 20 }, (_, x) => tex(x + 2, y)))
 * Math.round((toArray(semiGlobalMatching(L, R, { numDisparities: 6 })) as number[][])[5][10]) // => 2
 * ```
 */
export function semiGlobalMatching(
  left: TensorLike,
  right: TensorLike,
  options: DisparityOptions & { readonly P1?: number; readonly P2?: number; readonly paths?: 4 | 8 } = {},
): Tensor {
  const { L, R, h, w } = pairOf(left, right, 'semiGlobalMatching')
  const nd = options.numDisparities ?? 16
  const dmin = options.minDisparity ?? 0
  const bs = options.blockSize ?? 3
  const half = Math.floor(bs / 2)
  const P1 = options.P1 ?? 0.1 * bs * bs
  const P2 = options.P2 ?? 0.4 * bs * bs
  const dirs: [number, number][] =
    (options.paths ?? 8) === 4
      ? [
          [0, 1],
          [0, -1],
          [1, 0],
          [-1, 0],
        ]
      : [
          [0, 1],
          [0, -1],
          [1, 0],
          [-1, 0],
          [1, 1],
          [1, -1],
          [-1, 1],
          [-1, -1],
        ]
  const solve = (rightReference: boolean): Float64Array => {
    const C = costVolume(L, R, h, w, dmin, nd, half, 'sad', rightReference)
    let top = 0
    for (const v of C) if (Number.isFinite(v)) top = Math.max(top, v)
    for (let i = 0; i < C.length; i++) if (!Number.isFinite(C[i] ?? 0)) C[i] = top
    return winners(aggregate(C, h, w, nd, dirs, P1, P2), h, w, dmin, nd)
  }
  const dl = solve(false)
  if (options.lrCheck === false) return imageTensor(dl, h, w)
  return imageTensor(consistency(dl, solve(true), h, w, options.maxDiff ?? 1), h, w)
}

/**
 * Depth from disparity: Z = focal baseline / d for positive disparities, 0 elsewhere.
 *
 * @param disparity - Disparity `[H, W]`.
 * @param focal - The focal length in pixels.
 * @param baseline - The baseline (in the units of the depth).
 * @returns Depth `[H, W]`.
 * @concept stereo-matching
 * @example
 * ```ts
 * import { disparityToDepth } from 'mdml-engine/vision'
 * import { toArray } from 'mdml-engine/tensor'
 * toArray(disparityToDepth([[2, 4, -1]], 100, 0.1)) // => [[5, 2.5, 0]]
 * ```
 */
export function disparityToDepth(disparity: TensorLike, focal: number, baseline: number): Tensor {
  const d = greyOf(disparity, 'disparityToDepth')
  return imageTensor(
    Float64Array.from(d.data, (v) => (v > 0 ? (focal * baseline) / v : 0)),
    d.h,
    d.w,
  )
}

/**
 * Reproject a disparity map to 3-D (cv2.reprojectImageTo3D): [X, Y, Z, W] = Q [x, y, d, 1] for every pixel,
 * dehomogenised.
 *
 * @param disparity - Disparity `[H, W]`.
 * @param Q - The 4 x 4 disparity-to-depth matrix (from {@link stereoRectify}).
 * @returns Points `[H, W, 3]`.
 * @concept stereo-matching
 * @example
 * ```ts
 * import { reprojectImageTo3D } from 'mdml-engine/vision'
 * const Q = [[1, 0, 0, -1], [0, 1, 0, -1], [0, 0, 0, 100], [0, 0, 10, 0]]
 * reprojectImageTo3D([[2]], Q).shape // => [1, 1, 3]
 * ```
 */
export function reprojectImageTo3D(disparity: TensorLike, Q: readonly (readonly number[])[]): Tensor {
  const d = greyOf(disparity, 'reprojectImageTo3D')
  const q = (i: number, j: number): number => Q[i]?.[j] ?? 0
  const out = new Float64Array(d.h * d.w * 3)
  for (let y = 0; y < d.h; y++)
    for (let x = 0; x < d.w; x++) {
      const v = d.data[y * d.w + x] ?? 0
      const hom = [0, 1, 2, 3].map((i) => q(i, 0) * x + q(i, 1) * y + q(i, 2) * v + q(i, 3))
      const W = hom[3] ?? 1
      for (let k = 0; k < 3; k++) out[(y * d.w + x) * 3 + k] = (hom[k] ?? 0) / W
    }
  return imageTensor(out, d.h, d.w, 3)
}
