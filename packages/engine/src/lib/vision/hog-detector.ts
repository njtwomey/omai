/**
 * Sliding-window detection with HOG features (Dalal and Triggs 2005): the descriptor of each window (signal's hog,
 * scikit-image's) scored by a linear classifier over an image pyramid, and the windows above a threshold merged by
 * non-maximum suppression. The classifier is any linear one, such as a linear SVM or logistic regression trained on
 * {@link hogFeatures} of positive and negative windows.
 */

// TODO: OpenCV's HOGDescriptor layout (Gaussian block weighting, trilinear voting) and its default people detector.

import { MdmlError } from '../base/index.ts'
import { ddot } from '../numerics/linalg/index.ts'
import { image } from '../signal/index.ts'
import { type TensorLike, toFlat } from '../tensor/index.ts'
import { greyOf } from './_image.ts'
import { nms } from './boxes.ts'

/** A linear HOG detector for windows `[h, w]`: score = weights . hog(window) + bias. */
export interface HogDetector {
  readonly window: readonly [number, number]
  readonly weights: readonly number[]
  readonly bias: number
  /** The HOG options the weights were trained with (signal's hog defaults otherwise). */
  readonly hog?: image.HogOptions
}

const rowsOf = (data: Float64Array, h: number, w: number): number[][] =>
  Array.from({ length: h }, (_, i) => Array.from(data.subarray(i * w, (i + 1) * w)))

/**
 * HOG descriptors of windows (signal's hog), one row per window, for training a linear detector.
 *
 * @param windows - Grey windows `[h, w]` of one size.
 * @param options - HOG options.
 * @returns The descriptors.
 * @concept hog
 * @example
 * ```ts
 * import { hogFeatures } from 'mdml-engine/vision'
 * const win = Array.from({ length: 8 }, (_, i) => Array.from({ length: 8 }, (_, j) => (j < 4 ? 0 : 1)))
 * hogFeatures([win], { pixelsPerCell: [4, 4], cellsPerBlock: [1, 1] })[0].length // => 36
 * ```
 */
export function hogFeatures(windows: readonly TensorLike[], options: image.HogOptions = {}): number[][] {
  return windows.map((x) => toFlat(image.hog(x, options)))
}

/** Options of {@link hogDetectMultiScale}. */
export interface HogDetectOptions {
  /** The window step `[rows, columns]` at each level (default [8, 8]). */
  readonly stride?: readonly [number, number]
  /** The ratio between pyramid levels (default 1.2). */
  readonly scaleFactor?: number
  /** The most levels (default as many as fit the window). */
  readonly maxLevels?: number
  /** The least score of a detection (default 0). */
  readonly threshold?: number
  /** The overlap at which detections merge (default 0.3). */
  readonly iouThreshold?: number
}

/**
 * Detect objects with a linear HOG detector: the image is shrunk level by level (signal's anti-aliased resize), each
 * window at the stride is described and scored, and the boxes above the threshold (mapped back to the image) are merged
 * by non-maximum suppression.
 *
 * @param x - Grey image `[H, W]`.
 * @param detector - The window size, weights, bias and HOG options.
 * @param options - Stride, scale factor, levels, threshold and merge overlap.
 * @returns Boxes `[x1, y1, x2, y2]` and their scores, best first.
 * @concept hog
 * @example
 * ```ts
 * import { hogDetectMultiScale, hogFeatures } from 'mdml-engine/vision'
 * const edge = Array.from({ length: 8 }, () => Array.from({ length: 8 }, (_, j) => (j < 4 ? 0 : 1)))
 * const hog = { pixelsPerCell: [4, 4], cellsPerBlock: [1, 1] } as const
 * const weights = hogFeatures([edge], hog)[0]
 * const img = Array.from({ length: 8 }, () => Array.from({ length: 24 }, (_, j) => (j < 12 ? 0 : 1)))
 * hogDetectMultiScale(img, { window: [8, 8], weights, bias: -1, hog }, { stride: [4, 4], maxLevels: 1 }).boxes // => [[8, 0, 16, 8]]
 * ```
 */
export function hogDetectMultiScale(
  x: TensorLike,
  detector: HogDetector,
  options: HogDetectOptions = {},
): { readonly boxes: number[][]; readonly scores: number[] } {
  const g = greyOf(x, 'hogDetectMultiScale')
  const [wh, ww] = detector.window
  const [sy, sx] = options.stride ?? [8, 8]
  const factor = options.scaleFactor ?? 1.2
  if (!(factor > 1)) throw new MdmlError('value', 'hogDetectMultiScale: the scale factor must exceed 1')
  if (!(sy >= 1 && sx >= 1)) throw new MdmlError('value', 'hogDetectMultiScale: the stride is at least one pixel')
  const threshold = options.threshold ?? 0
  const boxes: number[][] = []
  const scores: number[] = []
  const full = rowsOf(g.data, g.h, g.w)
  for (let level = 0, scale = 1; level < (options.maxLevels ?? Number.POSITIVE_INFINITY); level++, scale *= factor) {
    const h = Math.round(g.h / scale)
    const w = Math.round(g.w / scale)
    if (h < wh || w < ww) break
    const plane = level === 0 ? g.data : Float64Array.from(toFlat(image.resize(full, [h, w])))
    for (let r = 0; r + wh <= h; r += sy)
      for (let c = 0; c + ww <= w; c += sx) {
        const win = Array.from({ length: wh }, (_, i) =>
          Array.from(plane.subarray((r + i) * w + c, (r + i) * w + c + ww)),
        )
        const f = toFlat(image.hog(win, detector.hog ?? {}))
        if (f.length !== detector.weights.length)
          throw new MdmlError('shape', 'hogDetectMultiScale: the weights must match the window descriptor')
        const score = ddot(f, detector.weights) + detector.bias
        if (score > threshold) {
          boxes.push([c * scale, r * scale, (c + ww) * scale, (r + wh) * scale])
          scores.push(score)
        }
      }
  }
  if (boxes.length === 0) return { boxes: [], scores: [] }
  const keep = nms(boxes, scores, options.iouThreshold ?? 0.3)
  return { boxes: keep.map((i) => boxes[i] as number[]), scores: keep.map((i) => scores[i] ?? 0) }
}
