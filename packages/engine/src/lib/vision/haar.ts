/**
 * Haar-like features and Viola-Jones cascades: the five rectangle feature types of scikit-image's haar_like_feature
 * (enumerated in its order and evaluated on an integral image), discrete AdaBoost over decision stumps as a traced
 * algorithm, attentional cascades trained stage by stage on the negatives that survive, and multi-scale detection that
 * scales the detector over the integral image and merges overlapping detections by non-maximum suppression.
 */

// TODO: OpenCV's cascade XML format, tilted (45 degree) features and Gentle or Real AdaBoost.

import { type Json, MdmlError } from '../base/index.ts'
import { bestStump, sortedRows } from '../ensembles/index.ts'
import { image } from '../signal/index.ts'
import { type Tensor, type TensorLike, tensor, toFlat } from '../tensor/index.ts'
import { type Algorithm, channels, type RunOptions, runAlgorithm } from '../trace/index.ts'
import { greyOf } from './_image.ts'
import { nms } from './boxes.ts'

/** The Haar-like feature types: two rectangles side by side or stacked, three in a row or column, and four. */
export type HaarType = 'type-2-x' | 'type-2-y' | 'type-3-x' | 'type-3-y' | 'type-4'

/** All Haar-like feature types, in scikit-image's order. */
export const HAAR_TYPES: readonly HaarType[] = ['type-2-x', 'type-2-y', 'type-3-x', 'type-3-y', 'type-4']

/** A Haar-like feature: rectangles `[row, col, height, width]` in its window, weighted -1, +1, -1, ... in order. */
export interface HaarFeature {
  readonly type: HaarType
  readonly rects: readonly (readonly [number, number, number, number])[]
}

/**
 * Every Haar-like feature of the given types that fits a window, in scikit-image's order (haar_like_feature_coord): by
 * type, then top row, left column, rectangle height and width.
 *
 * @param width - Window width.
 * @param height - Window height.
 * @param types - One type or several (default all, in {@link HAAR_TYPES} order).
 * @returns The features.
 * @concept haar-features
 * @example
 * ```ts
 * import { haarFeatures } from 'mdml-engine/vision'
 * haarFeatures(2, 2, 'type-2-x').map((f) => f.rects) // => [[[0, 0, 1, 1], [0, 1, 1, 1]], [[0, 0, 2, 1], [0, 1, 2, 1]], [[1, 0, 1, 1], [1, 1, 1, 1]]]
 * ```
 */
export function haarFeatures(
  width: number,
  height: number,
  types: HaarType | readonly HaarType[] = HAAR_TYPES,
): HaarFeature[] {
  if (!(Number.isInteger(width) && Number.isInteger(height) && width >= 1 && height >= 1))
    throw new MdmlError('value', 'haarFeatures: the window is positive integers')
  const out: HaarFeature[] = []
  for (const type of typeof types === 'string' ? [types] : types)
    for (let y = 0; y < height; y++)
      for (let x = 0; x < width; x++)
        for (let dy = 1; dy <= height; dy++)
          for (let dx = 1; dx <= width; dx++) {
            const r = (a: number, b: number): readonly [number, number, number, number] => [
              y + a * dy,
              x + b * dx,
              dy,
              dx,
            ]
            if (type === 'type-2-x' && x + 2 * dx <= width && y + dy <= height)
              out.push({ type, rects: [r(0, 0), r(0, 1)] })
            if (type === 'type-2-y' && y + 2 * dy <= height && x + dx <= width)
              out.push({ type, rects: [r(0, 0), r(1, 0)] })
            if (type === 'type-3-x' && x + 3 * dx <= width && y + dy <= height)
              out.push({ type, rects: [r(0, 0), r(0, 1), r(0, 2)] })
            if (type === 'type-3-y' && y + 3 * dy <= height && x + dx <= width)
              out.push({ type, rects: [r(0, 0), r(1, 0), r(2, 0)] })
            if (type === 'type-4' && x + 2 * dx <= width && y + 2 * dy <= height)
              out.push({ type, rects: [r(0, 0), r(0, 1), r(1, 1), r(1, 0)] })
          }
  return out
}

/** An integral image as a flat table. */
interface Table {
  readonly h: number
  readonly w: number
  readonly data: readonly number[]
}

/** The integral table of an image. */
const integralOf = (x: TensorLike): Table => {
  const v = image.integralImage(x)
  return { h: v.shape[0] ?? 0, w: v.shape[1] ?? 0, data: toFlat(v) }
}

/** An integral image given as such. */
const tableOf = (integral: TensorLike): Table => {
  const g = greyOf(integral, 'haarLikeFeature')
  return { h: g.h, w: g.w, data: Array.from(g.data) }
}

/** The sum of a box `[row, col, height, width]` from an inclusive integral table. */
function boxSum(t: Table, r: number, c: number, h: number, w: number): number {
  const at = (y: number, x: number): number => (y < 0 || x < 0 ? 0 : (t.data[y * t.w + x] ?? 0))
  const r1 = r + h - 1
  const c1 = c + w - 1
  return at(r1, c1) - at(r - 1, c1) - at(r1, c - 1) + at(r - 1, c - 1)
}

/** A feature's raw value at an offset and scale. */
function featureValue(t: Table, f: HaarFeature, r: number, c: number, scale = 1): number {
  let v = 0
  for (const [k, [y, x, h, w]] of f.rects.entries()) {
    const s = boxSum(
      t,
      r + Math.round(y * scale),
      c + Math.round(x * scale),
      Math.max(1, Math.round(h * scale)),
      Math.max(1, Math.round(w * scale)),
    )
    v += k % 2 === 0 ? -s : s
  }
  return v
}

/**
 * Haar-like feature values of a window of an integral image, as scikit-image's haar_like_feature: each feature's
 * rectangle sums weighted -1, +1, -1, ... in order.
 *
 * @param integral - The integral image (`image.integralImage` of signal) `[H, W]`.
 * @param r - Top row of the window.
 * @param c - Left column of the window.
 * @param width - Window width.
 * @param height - Window height.
 * @param features - Types (default all) or features from {@link haarFeatures}.
 * @returns The values, one per feature.
 * @concept haar-features
 * @example
 * ```ts
 * import { haarLikeFeature } from 'mdml-engine/vision'
 * import { image } from 'mdml-engine/signal'
 * import { toFlat } from 'mdml-engine/tensor'
 * toFlat(haarLikeFeature(image.integralImage([[1, 2], [8, 16]]), 0, 0, 2, 2, 'type-4')) // => [-7]
 * ```
 */
export function haarLikeFeature(
  integral: TensorLike,
  r: number,
  c: number,
  width: number,
  height: number,
  features: HaarType | readonly HaarType[] | readonly HaarFeature[] = HAAR_TYPES,
): Tensor {
  const t = tableOf(integral)
  if (r < 0 || c < 0 || r + height > t.h || c + width > t.w)
    throw new MdmlError('value', 'haarLikeFeature: the window must lie inside the image')
  const list =
    typeof features === 'string' || (features.length > 0 && typeof features[0] === 'string')
      ? haarFeatures(width, height, features as HaarType | readonly HaarType[])
      : (features as readonly HaarFeature[])
  return tensor(list.map((f) => featureValue(t, f, r, c)))
}

/** A window and its statistics for standardised features. */
interface Window {
  readonly sum: Table
  readonly sq: Table
}

const windowOf = (x: TensorLike): Window => {
  const g = greyOf(x, 'haar')
  const rows = Array.from({ length: g.h }, (_, i) => Array.from(g.data.subarray(i * g.w, (i + 1) * g.w)))
  return { sum: integralOf(rows), sq: integralOf(rows.map((row) => row.map((v) => v * v))) }
}

/**
 * The value of a feature on the window at (r, c) of the given base size and scale, standardised: computed on the window
 * scaled to zero mean and unit variance, per unit area of the base window (so it does not depend on the scale).
 */
function standardised(win: Window, f: HaarFeature, r: number, c: number, h: number, w: number, scale: number): number {
  const area = h * w
  const mean = boxSum(win.sum, r, c, h, w) / area
  const sd = Math.sqrt(Math.max(boxSum(win.sq, r, c, h, w) / area - mean * mean, 1e-12))
  let signedArea = 0
  for (const [k, [, , rh, rw]] of f.rects.entries()) {
    const a = Math.max(1, Math.round(rh * scale)) * Math.max(1, Math.round(rw * scale))
    signedArea += k % 2 === 0 ? -a : a
  }
  return (featureValue(win.sum, f, r, c, scale) - mean * signedArea) / (sd * scale * scale)
}

/** A decision stump on one feature: votes 1 when polarity * value < polarity * threshold, with weight alpha. */
export interface WeakClassifier {
  readonly feature: HaarFeature
  readonly threshold: number
  readonly polarity: 1 | -1
  readonly alpha: number
}

/** A cascade stage: a boosted vote of stumps, passing a window when the vote reaches the threshold. */
export interface CascadeStage {
  readonly weak: readonly WeakClassifier[]
  readonly threshold: number
}

/** A Viola-Jones cascade for windows of a base size. */
export interface Cascade {
  readonly width: number
  readonly height: number
  readonly stages: readonly CascadeStage[]
}

/** The state of {@link adaBoostAlgorithm}: sample weights, the stumps so far and their errors. */
export interface AdaBoostState {
  readonly weights: Float64Array
  readonly weak: readonly WeakClassifier[]
  /** The weighted error of the last stump. */
  readonly error: number
  /** The error of the strong classifier (threshold half the vote) on the training set. */
  readonly trainingError: number
}

/** Options of {@link adaBoostAlgorithm}. */
export interface AdaBoostOptions {
  /** The feature pool (default every feature of the window). */
  readonly features?: readonly HaarFeature[]
}

/** The vote of stumps on a window's standardised features. */
const voteOf = (weak: readonly WeakClassifier[], value: (f: HaarFeature) => number): number => {
  let s = 0
  for (const h of weak) if (h.polarity * value(h.feature) < h.polarity * h.threshold) s += h.alpha
  return s
}

/**
 * Discrete AdaBoost over Haar-feature decision stumps (Viola and Jones 2001) as an algorithm: positive and negative
 * weights start at 1 / 2m and 1 / 2l; each step normalises them, picks the stump of least weighted error over the pool
 * (thresholds between sorted values, both polarities), and down-weights the samples it gets right by beta = e / (1 - e),
 * voting with alpha = log(1 / beta).
 *
 * @param positives - Positive windows `[h, w]`, all of one size.
 * @param negatives - Negative windows of the same size.
 * @param options - The feature pool.
 * @returns The algorithm (converging when a stump separates the samples).
 * @concept boosting
 * @example
 * ```ts
 * import { adaBoostAlgorithm } from 'mdml-engine/vision'
 * import { runAlgorithm } from 'mdml-engine/trace'
 * const pos = [[[0, 1], [0, 1]], [[0.1, 0.9], [0, 1]]]
 * const neg = [[[1, 0], [1, 0]], [[0.5, 0.5], [0.4, 0.6]]]
 * runAlgorithm(adaBoostAlgorithm(pos, neg), { maxSteps: 3 }).state.trainingError // => 0
 * ```
 */
export function adaBoostAlgorithm(
  positives: readonly TensorLike[],
  negatives: readonly TensorLike[],
  options: AdaBoostOptions = {},
): Algorithm<AdaBoostState> {
  if (positives.length === 0 || negatives.length === 0)
    throw new MdmlError('value', 'adaBoostAlgorithm: positives and negatives are needed')
  const windows = [...positives, ...negatives].map(windowOf)
  const { h, w } = windows[0]?.sum ?? { h: 0, w: 0 }
  if (windows.some((v) => v.sum.h !== h || v.sum.w !== w))
    throw new MdmlError('shape', 'adaBoostAlgorithm: the windows must share one size')
  const pool = options.features ?? haarFeatures(w, h)
  if (pool.length === 0) throw new MdmlError('value', 'adaBoostAlgorithm: the feature pool is empty')
  const n = windows.length
  const m = positives.length
  const label = (i: number): number => (i < m ? 1 : 0)
  // Feature values and their sort order, computed once.
  const values = pool.map((f) => Float64Array.from(windows, (win) => standardised(win, f, 0, 0, h, w, 1)))
  const order = values.map(sortedRows)
  const featureIndex = new Map(pool.map((f, j) => [f, j]))
  const strongError = (weak: readonly WeakClassifier[]): number => {
    const half = weak.reduce((s, k) => s + k.alpha, 0) / 2
    let wrong = 0
    for (let i = 0; i < n; i++) {
      const vote = voteOf(weak, (f) => values[featureIndex.get(f) ?? 0]?.[i] ?? 0)
      if ((vote >= half ? 1 : 0) !== label(i)) wrong++
    }
    return wrong / n
  }
  return {
    name: 'adaboost',
    config: { positives: m, negatives: n - m, features: pool.length } as Json,
    schema: channels({
      metrics: {
        error: { description: 'weighted error of the chosen stump', direction: 'lower' },
        trainingError: { description: 'training error of the strong classifier', direction: 'lower' },
        alpha: { description: 'vote of the chosen stump' },
      },
    }),
    init: () => ({
      weights: Float64Array.from({ length: n }, (_, i) => (i < m ? 1 / (2 * m) : 1 / (2 * (n - m)))),
      weak: [],
      error: Number.POSITIVE_INFINITY,
      trainingError: 1,
    }),
    step: (s) => {
      const total = s.weights.reduce((a, b) => a + b, 0)
      const wt = Float64Array.from(s.weights, (v) => v / total)
      const found = bestStump(values, order, (i) => label(i) === 1, wt)
      const best = { error: found.error, j: found.feature, threshold: found.threshold, polarity: found.polarity }
      const e = Math.min(Math.max(best.error, 1e-10), 0.5 - 1e-10)
      const beta = e / (1 - e)
      const feature = pool[best.j] as HaarFeature
      const stump: WeakClassifier = {
        feature,
        threshold: best.threshold,
        polarity: best.polarity,
        alpha: Math.log(1 / beta),
      }
      const v = values[best.j] as Float64Array
      const weights = Float64Array.from(wt, (wi, i) => {
        const vote = stump.polarity * (v[i] ?? 0) < stump.polarity * stump.threshold ? 1 : 0
        return vote === label(i) ? wi * beta : wi
      })
      const weak = [...s.weak, stump]
      return { weights, weak, error: best.error, trainingError: strongError(weak) }
    },
    done: (s) => (s.weak.length > 0 && s.error <= 1e-12 ? 'converged' : undefined),
    observe: (s) => ({
      metrics: {
        error: Number.isFinite(s.error) ? s.error : -1,
        trainingError: s.trainingError,
        alpha: s.weak.at(-1)?.alpha ?? 0,
      },
    }),
  }
}

/** Options of {@link trainCascade}. */
export interface TrainCascadeOptions extends AdaBoostOptions {
  /** Stumps per stage (default [2, 5, 10]); training stops early once no negative survives. */
  readonly stages?: readonly number[]
  /** The fraction of positives each stage must pass (default 0.995), set by lowering the stage threshold. */
  readonly minDetection?: number
}

/**
 * Train an attentional cascade (Viola and Jones 2001): each stage is boosted ({@link adaBoostAlgorithm}, traced as a run)
 * on the positives and the negatives that passed every earlier stage, its threshold lowered until it passes
 * minDetection of the positives.
 *
 * @param positives - Positive windows `[h, w]`, all of one size.
 * @param negatives - Negative windows of the same size.
 * @param options - Stage sizes, detection rate, feature pool and tracing.
 * @returns The cascade.
 * @concept boosting
 * @example
 * ```ts
 * import { trainCascade } from 'mdml-engine/vision'
 * const pos = [[[0, 1], [0, 1]], [[0.1, 0.9], [0, 1]]]
 * const neg = [[[1, 0], [1, 0]], [[0.5, 0.5], [0.4, 0.6]]]
 * trainCascade(pos, neg, { stages: [1, 2] }).stages.length // => 1
 * ```
 */
export function trainCascade(
  positives: readonly TensorLike[],
  negatives: readonly TensorLike[],
  options: TrainCascadeOptions & Partial<Pick<RunOptions, 'tracer' | 'policy' | 'parent'>> = {},
): Cascade {
  const sizes = options.stages ?? [2, 5, 10]
  const minDetection = options.minDetection ?? 0.995
  const first = greyOf(positives[0] ?? [[0]], 'trainCascade')
  const pos = positives.map(windowOf)
  let neg = [...negatives]
  const stages: CascadeStage[] = []
  for (const size of sizes) {
    if (neg.length === 0) break
    const { weak } = runAlgorithm(adaBoostAlgorithm(positives, neg, options), {
      maxSteps: size,
      ...(options.tracer === undefined ? {} : { tracer: options.tracer }),
      ...(options.policy === undefined ? {} : { policy: options.policy }),
      ...(options.parent === undefined ? {} : { parent: options.parent }),
    }).state
    const score = (win: Window): number => voteOf(weak, (f) => standardised(win, f, 0, 0, first.h, first.w, 1))
    const scores = pos.map(score).sort((a, b) => a - b)
    const half = weak.reduce((s, k) => s + k.alpha, 0) / 2
    const keep = scores[Math.min(scores.length - 1, Math.floor((1 - minDetection) * scores.length))] ?? half
    const stage = { weak, threshold: Math.min(half, keep) - 1e-9 }
    stages.push(stage)
    neg = neg.filter((x) => score(windowOf(x)) >= stage.threshold)
  }
  return { width: first.w, height: first.h, stages }
}

/**
 * Whether a window passes a cascade, and its margin over the last stage's threshold (negative when rejected).
 *
 * @param cascade - The cascade.
 * @param x - A window `[h, w]` of the cascade's size.
 * @returns Pass or not, the stages passed and the final margin.
 * @concept boosting
 * @example
 * ```ts
 * import { cascadeClassify, trainCascade } from 'mdml-engine/vision'
 * const pos = [[[0, 1], [0, 1]], [[0.1, 0.9], [0, 1]]]
 * const neg = [[[1, 0], [1, 0]], [[0.5, 0.5], [0.4, 0.6]]]
 * cascadeClassify(trainCascade(pos, neg), [[0, 0.8], [0.1, 1]]).pass // => true
 * ```
 */
export function cascadeClassify(
  cascade: Cascade,
  x: TensorLike,
): { readonly pass: boolean; readonly stages: number; readonly margin: number } {
  const win = windowOf(x)
  if (win.sum.h !== cascade.height || win.sum.w !== cascade.width)
    throw new MdmlError('shape', 'cascadeClassify: the window must have the cascade size')
  return evaluate(cascade, win, 0, 0, 1)
}

function evaluate(
  cascade: Cascade,
  win: Window,
  r: number,
  c: number,
  scale: number,
): { readonly pass: boolean; readonly stages: number; readonly margin: number } {
  const h = Math.round(cascade.height * scale)
  const w = Math.round(cascade.width * scale)
  let margin = 0
  for (const [k, stage] of cascade.stages.entries()) {
    margin = voteOf(stage.weak, (f) => standardised(win, f, r, c, h, w, scale)) - stage.threshold
    if (margin < 0) return { pass: false, stages: k, margin }
  }
  return { pass: true, stages: cascade.stages.length, margin }
}

/** Options of {@link detectMultiScale}. */
export interface DetectMultiScaleOptions {
  /** The ratio between successive detector scales (default 1.25). */
  readonly scaleFactor?: number
  /** The step between windows as a fraction of the window width (default 0.1, at least 1 pixel). */
  readonly stepFraction?: number
  /** The smallest and largest scale (defaults 1 and as large as fits). */
  readonly minScale?: number
  readonly maxScale?: number
  /** The overlap at which detections merge (default 0.3). */
  readonly iouThreshold?: number
}

/**
 * Detect objects with a cascade at every position and scale, scaling the detector (its rectangles) over one integral
 * image as Viola and Jones do, then merging overlapping detections by non-maximum suppression on the final margin.
 *
 * @param x - Grey image `[H, W]`.
 * @param cascade - The cascade.
 * @param options - Scale factor, step, scale range and merge overlap.
 * @returns Boxes `[x1, y1, x2, y2]` and their margins, best first.
 * @concept object-detection
 * @example
 * ```ts
 * import { detectMultiScale, trainCascade } from 'mdml-engine/vision'
 * const pos = [[[0, 1], [0, 1]], [[0.1, 0.9], [0, 1]]]
 * const neg = [[[1, 0], [1, 0]], [[0.5, 0.5], [0.4, 0.6]]]
 * const img = [[0.5, 0.5, 0.5, 0.5], [0.5, 0, 1, 0.5], [0.5, 0, 1, 0.5], [0.5, 0.5, 0.5, 0.5]]
 * detectMultiScale(img, trainCascade(pos, neg), { maxScale: 1 }).boxes[0] // => [1, 1, 3, 3]
 * ```
 */
export function detectMultiScale(
  x: TensorLike,
  cascade: Cascade,
  options: DetectMultiScaleOptions = {},
): { readonly boxes: number[][]; readonly scores: number[] } {
  const win = windowOf(x)
  const { h: H, w: W } = win.sum
  const factor = options.scaleFactor ?? 1.25
  if (!(factor > 1)) throw new MdmlError('value', 'detectMultiScale: the scale factor must exceed 1')
  const maxScale = Math.min(options.maxScale ?? Number.POSITIVE_INFINITY, H / cascade.height, W / cascade.width)
  const boxes: number[][] = []
  const scores: number[] = []
  for (let scale = options.minScale ?? 1; scale <= maxScale + 1e-12; scale *= factor) {
    const h = Math.round(cascade.height * scale)
    const w = Math.round(cascade.width * scale)
    const step = Math.max(1, Math.round((options.stepFraction ?? 0.1) * w))
    for (let r = 0; r + h <= H; r += step)
      for (let c = 0; c + w <= W; c += step) {
        const res = evaluate(cascade, win, r, c, scale)
        if (res.pass) {
          boxes.push([c, r, c + w, r + h])
          scores.push(res.margin)
        }
      }
  }
  if (boxes.length === 0) return { boxes: [], scores: [] }
  const keep = nms(boxes, scores, options.iouThreshold ?? 0.3)
  return { boxes: keep.map((i) => boxes[i] as number[]), scores: keep.map((i) => scores[i] ?? 0) }
}
