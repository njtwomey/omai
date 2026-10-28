/**
 * Keypoints and the FAST corner detector (Rosten and Drummond 2006), as OpenCV's FastFeatureDetector runs it on 256
 * levels: a pixel is a corner when nine contiguous pixels of the 16-pixel Bresenham circle around it are all brighter
 * or all darker than it by more than the threshold, its score is the largest threshold for which that still holds, and
 * non-maximum suppression keeps corners whose score beats their eight neighbours'.
 */

import { MdmlError } from '../base/index.ts'
import type { TensorLike } from '../tensor/index.ts'
import { greyOf, level } from './_image.ts'

/** A keypoint, OpenCV's cv::KeyPoint: position (x, y), diameter, orientation, strength and pyramid octave. */
export interface Keypoint {
  readonly x: number
  readonly y: number
  /** The diameter of the meaningful neighbourhood. */
  readonly size: number
  /** Orientation in degrees [0, 360), or -1 when not computed. */
  readonly angle: number
  /** The detector's response (larger is stronger). */
  readonly response: number
  /** The pyramid octave (or level) it was found in. */
  readonly octave: number
}

// The 16-pixel circle of radius 3 (dx, dy), clockwise from the top, as OpenCV's makeOffsets.
const CIRCLE: readonly (readonly [number, number])[] = [
  [0, 3],
  [1, 3],
  [2, 2],
  [3, 1],
  [3, 0],
  [3, -1],
  [2, -2],
  [1, -3],
  [0, -3],
  [-1, -3],
  [-2, -2],
  [-3, -1],
  [-3, 0],
  [-3, 1],
  [-2, 2],
  [-1, 3],
]

/** OpenCV's cornerScore<16>: the largest threshold at which the pixel is still a corner (d holds v - circle). */
function cornerScore(d: Int32Array, threshold: number): number {
  let a0 = threshold
  for (let k = 0; k < 16; k += 2) {
    let a = Math.min(d[k + 1] ?? 0, d[k + 2] ?? 0, d[k + 3] ?? 0)
    if (a <= a0) continue
    a = Math.min(a, d[k + 4] ?? 0, d[k + 5] ?? 0, d[k + 6] ?? 0, d[k + 7] ?? 0, d[k + 8] ?? 0)
    a0 = Math.max(a0, Math.min(a, d[k] ?? 0))
    a0 = Math.max(a0, Math.min(a, d[k + 9] ?? 0))
  }
  let b0 = -a0
  for (let k = 0; k < 16; k += 2) {
    let b = Math.max(d[k + 1] ?? 0, d[k + 2] ?? 0, d[k + 3] ?? 0, d[k + 4] ?? 0, d[k + 5] ?? 0)
    if (b >= b0) continue
    b = Math.max(b, d[k + 6] ?? 0, d[k + 7] ?? 0, d[k + 8] ?? 0)
    b0 = Math.min(b0, Math.max(b, d[k] ?? 0))
    b0 = Math.min(b0, Math.max(b, d[k + 9] ?? 0))
  }
  return -b0 - 1
}

/** Options of {@link fast}. */
export interface FastOptions {
  /** The intensity difference, in image units (default 10 / 255, OpenCV's 10). */
  readonly threshold?: number
  /** Keep only local maxima of the score (default true). */
  readonly nonmaxSuppression?: boolean
}

/**
 * FAST-9 corners (cv2.FastFeatureDetector with TYPE_9_16) at OpenCV's 256 levels: corners of the circle test, each
 * with OpenCV's corner score as its response, optionally thinned by 3 x 3 non-maximum suppression; keypoints in raster
 * order with size 7.
 *
 * @param x - Grey image `[H, W]` in [0, 1].
 * @param options - Threshold and non-maximum suppression.
 * @returns The keypoints.
 * @concept corner-detection
 * @example
 * ```ts
 * import { fast } from 'mdml-engine/vision'
 * const img = Array.from({ length: 12 }, (_, i) => Array.from({ length: 12 }, (_, j) => (i > 5 && j > 5 ? 1 : 0)))
 * fast(img, { nonmaxSuppression: false }).map((k) => [k.x, k.y]) // => [[6, 6], [7, 6], [8, 6], [6, 7], [7, 7], [6, 8]]
 * ```
 */
export function fast(x: TensorLike, options: FastOptions = {}): Keypoint[] {
  const r = greyOf(x, 'fast')
  const { h, w } = r
  const t = Math.min(255, Math.max(0, Math.round(255 * (options.threshold ?? 10 / 255) * 1e9) / 1e9))
  const threshold = Math.round(t)
  if (!Number.isFinite(threshold)) throw new MdmlError('value', 'fast: the threshold must be finite')
  const nms = options.nonmaxSuppression ?? true
  const img = Int32Array.from(r.data, level)
  const score = new Int32Array(h * w)
  const corners: number[] = []
  const d = new Int32Array(25)
  for (let y = 3; y < h - 3; y++)
    for (let xx = 3; xx < w - 3; xx++) {
      const v = img[y * w + xx] ?? 0
      for (let k = 0; k < 25; k++) {
        const [dx, dy] = CIRCLE[k % 16] as [number, number]
        d[k] = v - (img[(y + dy) * w + xx + dx] ?? 0)
      }
      // Nine contiguous pixels darker (d > t) or brighter (d < -t) than the centre, around the wrapped circle.
      let corner = false
      for (const sign of [1, -1]) {
        let count = 0
        for (let k = 0; k < 25 && !corner; k++) {
          if (sign * (d[k] ?? 0) > threshold) {
            if (++count > 8) corner = true
          } else count = 0
        }
      }
      if (!corner) continue
      corners.push(y * w + xx)
      score[y * w + xx] = nms ? cornerScore(d, threshold) : 0
    }
  const out: Keypoint[] = []
  for (const p of corners) {
    const s = score[p] ?? 0
    if (nms) {
      const y = Math.floor(p / w)
      const xx = p % w
      let keep = true
      for (let dy = -1; dy <= 1 && keep; dy++)
        for (let dx = -1; dx <= 1; dx++) {
          if (dx === 0 && dy === 0) continue
          if (s <= (score[(y + dy) * w + xx + dx] ?? 0)) {
            keep = false
            break
          }
        }
      if (!keep) continue
    }
    out.push({ x: p % w, y: Math.floor(p / w), size: 7, angle: -1, response: s, octave: 0 })
  }
  return out
}
