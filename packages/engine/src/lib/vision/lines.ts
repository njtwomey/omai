/**
 * Line segment detection after LSD (von Gioi, Jakubowicz, Morel and Randall 2012), following OpenCV's
 * LineSegmentDetector at 256 levels: level-line angles from a 2 x 2 gradient (OpenCV's fastAtan2), pixels pseudo-ordered
 * by gradient into 1024 bins, regions grown from the strongest over 8-neighbours of similar angle, and each region
 * approximated by the rectangle of its gradient-weighted inertia. Without validation this is OpenCV's LSD_REFINE_NONE;
 * the default validation keeps the rectangles dense enough whose number of false alarms (the a contrario binomial tail
 * of aligned pixels, times the number of tests) is below one.
 */

// TODO: LSD's Gaussian sub-sampling (scale 0.8), its region refinement and rectangle improvement steps.

import { OmaiError } from '../base/index.ts'
import { lgamma } from '../numerics/special/index.ts'
import { item, type TensorLike } from '../tensor/index.ts'
import { f32, fastAtan2, greyOf, level } from './_image.ts'

/** Options of {@link lineSegmentDetector}. */
export interface LsdOptions {
  /** The gradient quantisation error in grey levels (default 2). */
  readonly quant?: number
  /** The angle tolerance in degrees (default 22.5). */
  readonly angleTolerance?: number
  /** Validate rectangles by density and NFA (default true); false is OpenCV's LSD_REFINE_NONE at scale 1. */
  readonly validate?: boolean
  /** The least -log10 NFA of a kept segment (default 0, at most one false detection per image). */
  readonly logEps?: number
  /** The least fraction of the rectangle the region must fill (default 0.7). */
  readonly densityThreshold?: number
}

const DEG = Math.PI / 180

/** The angle between two orientations in [0, pi]. */
const angleDiff = (a: number, b: number): number => {
  let d = a - b
  while (d <= -Math.PI) d += 2 * Math.PI
  while (d > Math.PI) d -= 2 * Math.PI
  return Math.abs(d)
}

/** Whether a level-line angle is aligned with theta to precision prec (LSD's isAligned). */
const aligned = (angle: number, theta: number, prec: number): boolean => {
  if (Number.isNaN(angle)) return false
  let d = Math.abs(theta - angle)
  if (d > 1.5 * Math.PI) d = Math.abs(d - 2 * Math.PI)
  return d <= prec
}

/** -log10 of the number of false alarms of k aligned among n with probability p (LSD's nfa). */
function logNfa(n: number, k: number, p: number, logNT: number): number {
  if (n === 0 || k === 0) return -logNT
  if (n === k) return -logNT - n * Math.log10(p)
  const pTerm = p / (1 - p)
  const log1 =
    item(lgamma(n + 1)) - item(lgamma(k + 1)) - item(lgamma(n - k + 1)) + k * Math.log(p) + (n - k) * Math.log(1 - p)
  let term = Math.exp(log1)
  if (term === 0) return k > n * p ? -log1 / Math.LN10 - logNT : -logNT
  let tail = term
  for (let i = k + 1; i <= n; i++) {
    const bin = (n - i + 1) / i
    const mult = bin * pTerm
    term *= mult
    tail += term
    if (bin < 1) {
      const err = term * ((1 - mult ** (n - i + 1)) / (1 - mult) - 1)
      if (err < 0.1 * Math.abs(-Math.log10(tail) - logNT) * tail) break
    }
  }
  return -Math.log10(tail) - logNT
}

/**
 * Detect straight line segments (LSD at scale 1, without its refinement): pixels whose gradient exceeds
 * quant / sin(tolerance) are grown, strongest first, into regions of level-line angle within the tolerance of the
 * region's mean angle; each region of at least LSD's minimal size becomes the rectangle of its inertia, kept (when
 * validating) if it is dense enough and its -log10 NFA exceeds logEps. Coordinates are of pixel centres, as LSD's.
 *
 * @param x - Grey image `[H, W]` in [0, 1] (seen at 256 levels).
 * @param options - Quantisation, angle tolerance, validation and its NFA and density thresholds.
 * @returns Segments `[x1, y1, x2, y2]` (float32, as OpenCV's), their widths and -log10 NFA (NaN without validation).
 * @concept line-detection
 * @example
 * ```ts
 * import { lineSegmentDetector } from 'omai-engine/vision'
 * const img = Array.from({ length: 16 }, () => Array.from({ length: 16 }, (_, j) => (j < 8 ? 0 : 1)))
 * lineSegmentDetector(img).lines // => [[7.5, 0.5, 7.5, 14.5]]
 * ```
 */
export function lineSegmentDetector(
  x: TensorLike,
  options: LsdOptions = {},
): { readonly lines: number[][]; readonly widths: number[]; readonly logNfa: number[] } {
  const g = greyOf(x, 'lineSegmentDetector')
  const { h, w } = g
  if (h < 2 || w < 2) throw new OmaiError('shape', 'lineSegmentDetector: the image is at least 2 x 2')
  const tol = options.angleTolerance ?? 22.5
  if (!(tol > 0 && tol < 180)) throw new OmaiError('value', 'lineSegmentDetector: the tolerance is in (0, 180)')
  const prec = DEG * tol
  const p = tol / 180
  const rho = (options.quant ?? 2) / Math.sin(prec)
  const validate = options.validate ?? true
  const logEps = options.logEps ?? 0
  const densityTh = options.densityThreshold ?? 0.7
  const px = Uint8Array.from(g.data, level)
  const at = (yy: number, xx: number): number => px[yy * w + xx] ?? 0
  // Level-line angles (NaN where undefined: weak gradient, last row and column) and gradient norms.
  const angle = new Float64Array(h * w).fill(Number.NaN)
  const mag = new Float64Array(h * w)
  let maxGrad = -1
  for (let yy = 0; yy + 1 < h; yy++)
    for (let xx = 0; xx + 1 < w; xx++) {
      const DA = at(yy + 1, xx + 1) - at(yy, xx)
      const BC = at(yy, xx + 1) - at(yy + 1, xx)
      const gx = DA + BC
      const gy = DA - BC
      const norm = Math.sqrt((gx * gx + gy * gy) / 4)
      const i = yy * w + xx
      mag[i] = norm
      if (norm > rho) {
        angle[i] = fastAtan2(gx, -gy) * DEG
        maxGrad = Math.max(maxGrad, norm)
      }
    }
  // Pseudo-order: 1024 gradient bins, strongest first, raster order within a bin.
  const coef = maxGrad > 0 ? 1023 / maxGrad : 0
  const order: number[] = []
  const bin = new Int32Array(h * w)
  for (let yy = 0; yy + 1 < h; yy++)
    for (let xx = 0; xx + 1 < w; xx++) {
      const i = yy * w + xx
      bin[i] = Math.trunc((mag[i] ?? 0) * coef)
      order.push(i)
    }
  order.sort((a, b) => (bin[b] ?? 0) - (bin[a] ?? 0) || a - b)
  const logNT = (5 * (Math.log10(w) + Math.log10(h))) / 2 + Math.log10(11)
  const minRegion = Math.trunc(-logNT / Math.log10(p))
  const used = new Uint8Array(h * w)
  const lines: number[][] = []
  const widths: number[] = []
  const nfas: number[] = []
  for (const seed of order) {
    if (used[seed] || Number.isNaN(angle[seed] ?? Number.NaN)) continue
    // Grow the region (float32 sums, as OpenCV).
    const region = [seed]
    used[seed] = 1
    let regAngle = angle[seed] ?? 0
    let sx = f32(Math.cos(regAngle))
    let sy = f32(Math.sin(regAngle))
    for (let q = 0; q < region.length; q++) {
      const pix = region[q] as number
      const py = Math.floor(pix / w)
      const pxx = pix % w
      for (let yy = Math.max(0, py - 1); yy <= Math.min(h - 1, py + 1); yy++)
        for (let xx = Math.max(0, pxx - 1); xx <= Math.min(w - 1, pxx + 1); xx++) {
          const n = yy * w + xx
          if (used[n] || !aligned(angle[n] ?? Number.NaN, regAngle, prec)) continue
          used[n] = 1
          region.push(n)
          const a = f32(angle[n] ?? 0)
          sx = f32(sx + f32(Math.cos(a)))
          sy = f32(sy + f32(Math.sin(a)))
          regAngle = fastAtan2(sy, sx) * DEG
        }
    }
    if (region.length < minRegion) continue
    // The rectangle: gradient-weighted centre, inertia orientation and extents.
    let sw = 0
    let cx = 0
    let cy = 0
    for (const pix of region) {
      const m = mag[pix] ?? 0
      cx += (pix % w) * m
      cy += Math.floor(pix / w) * m
      sw += m
    }
    cx /= sw
    cy /= sw
    let ixx = 0
    let iyy = 0
    let ixy = 0
    for (const pix of region) {
      const m = mag[pix] ?? 0
      const dx = (pix % w) - cx
      const dy = Math.floor(pix / w) - cy
      ixx += dy * dy * m
      iyy += dx * dx * m
      ixy -= dx * dy * m
    }
    const lambda = 0.5 * (ixx + iyy - Math.sqrt((ixx - iyy) * (ixx - iyy) + 4 * ixy * ixy))
    let theta = (Math.abs(ixx) > Math.abs(iyy) ? fastAtan2(lambda - ixx, ixy) : fastAtan2(ixy, lambda - iyy)) * DEG
    if (angleDiff(theta, regAngle) > prec) theta += Math.PI
    const dx = Math.cos(theta)
    const dy = Math.sin(theta)
    let lMin = 0
    let lMax = 0
    let wMin = 0
    let wMax = 0
    for (const pix of region) {
      const rx = (pix % w) - cx
      const ry = Math.floor(pix / w) - cy
      const l = rx * dx + ry * dy
      const ww = -rx * dy + ry * dx
      if (l > lMax) lMax = l
      else if (l < lMin) lMin = l
      if (ww > wMax) wMax = ww
      else if (ww < wMin) wMin = ww
    }
    const width = Math.max(1, wMax - wMin)
    let score = Number.NaN
    if (validate) {
      const length = lMax - lMin
      if (region.length / (Math.max(length, 1) * width) < densityTh) continue
      // Count the pixels of the rectangle and those aligned with it.
      let n = 0
      let k = 0
      const ext = Math.ceil(length / 2 + width)
      for (let yy = Math.max(0, Math.floor(cy - ext)); yy <= Math.min(h - 1, Math.ceil(cy + ext)); yy++)
        for (let xx = Math.max(0, Math.floor(cx - ext)); xx <= Math.min(w - 1, Math.ceil(cx + ext)); xx++) {
          const l = (xx - cx) * dx + (yy - cy) * dy
          const ww = -(xx - cx) * dy + (yy - cy) * dx
          if (l < lMin - 1e-9 || l > lMax + 1e-9 || Math.abs(ww) > width / 2 + 1e-9) continue
          n++
          if (aligned(angle[yy * w + xx] ?? Number.NaN, theta, prec)) k++
        }
      score = logNfa(n, k, p, logNT)
      if (score <= logEps) continue
    }
    lines.push([
      f32(cx + lMin * dx + 0.5),
      f32(cy + lMin * dy + 0.5),
      f32(cx + lMax * dx + 0.5),
      f32(cy + lMax * dy + 0.5),
    ])
    widths.push(width)
    nfas.push(score)
  }
  return { lines, widths, logNfa: nfas }
}
