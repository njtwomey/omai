/**
 * Thresholding, OpenCV and scikit-image style: the five global rules of `cv2.threshold` with a fixed value or one chosen
 * automatically (Otsu and Li from `signal/image`, the triangle method at OpenCV's 256 levels), OpenCV's adaptive mean
 * and Gaussian thresholds, multi-level Otsu, and the local Niblack and Sauvola thresholds for documents.
 */

import { MdmlError, roundHalfEven } from '../base/index.ts'
import { image } from '../signal/index.ts'
import { histogram } from '../stats/index.ts'
import { fromStorage, primalOf, type Tensor, type TensorLike, toFlat } from '../tensor/index.ts'
import { clamp, greyOf, imageTensor, level, rasterOf, rasterTensor, valuesOf } from './_image.ts'
import { gaussianBlur } from './blur.ts'

/** The rules of {@link threshold} (cv2.THRESH_*): pixels above the threshold t versus the rest. */
export type ThresholdType = 'binary' | 'binary-inv' | 'trunc' | 'tozero' | 'tozero-inv'

/** How {@link threshold} picks t: a value, or Otsu's, Li's or the triangle method. */
export type ThresholdChoice = number | 'otsu' | 'li' | 'triangle'

/**
 * Global thresholding (cv2.threshold): with t the threshold, 'binary' gives maxValue above t and 0 elsewhere,
 * 'binary-inv' the reverse, 'trunc' min(v, t), 'tozero' v above t and 0 elsewhere, 'tozero-inv' the reverse. The
 * threshold may be chosen automatically by Otsu's or Li's method (scikit-image's, from `signal/image`) or the
 * triangle method.
 *
 * @param x - Image (any shape; the automatic methods histogram all of it).
 * @param thresh - The threshold or the method that chooses it.
 * @param options - maxValue (default 1) and the rule (default 'binary').
 * @returns The threshold used and the thresholded image.
 * @concept thresholding
 * @index function Global thresholding: binary, truncate, to-zero and inverse variants.
 * @example
 * ```ts
 * import { threshold } from 'mdml-engine/vision'
 * import { toArray } from 'mdml-engine/tensor'
 * toArray(threshold([0.2, 0.5, 0.8], 0.5).image) // => [0, 0, 1]
 * toArray(threshold([0.2, 0.5, 0.8], 0.5, { type: 'trunc' }).image) // => [0.2, 0.5, 0.5]
 * ```
 */
export function threshold(
  x: TensorLike,
  thresh: ThresholdChoice,
  options: { readonly maxValue?: number; readonly type?: ThresholdType } = {},
): { readonly threshold: number; readonly image: Tensor } {
  const { data, shape } = valuesOf(x)
  const t =
    thresh === 'otsu'
      ? image.thresholdOtsu(x)
      : thresh === 'li'
        ? image.thresholdLi(x)
        : thresh === 'triangle'
          ? thresholdTriangle(x)
          : thresh
  const maxValue = options.maxValue ?? 1
  const type = options.type ?? 'binary'
  const rule: Readonly<Record<ThresholdType, (v: number) => number>> = {
    binary: (v) => (v > t ? maxValue : 0),
    'binary-inv': (v) => (v > t ? 0 : maxValue),
    trunc: (v) => Math.min(v, t),
    tozero: (v) => (v > t ? v : 0),
    'tozero-inv': (v) => (v > t ? 0 : v),
  }
  const f = rule[type]
  if (f === undefined) throw new MdmlError('value', `threshold: unknown type ${String(type)}`)
  return { threshold: t, image: fromStorage(Float64Array.from(data, f), [...shape], 'float64') }
}

/**
 * The triangle threshold (Zack et al. 1977; cv2.THRESH_TRIANGLE) at OpenCV's 256 levels: the line from the histogram
 * peak to the far end of the longer tail, and the level whose bar lies farthest below it (minus one level).
 *
 * @param x - Image in [0, 1] (any shape).
 * @returns The threshold, a level / 255.
 * @concept thresholding
 * @example
 * ```ts
 * import { thresholdTriangle } from 'mdml-engine/vision'
 * const x = [...new Array(40).fill(0.1), ...Array.from({ length: 20 }, (_, i) => 0.2 + i / 30)]
 * thresholdTriangle(x) * 255 // => 28
 * ```
 */
export function thresholdTriangle(x: TensorLike): number {
  const { data } = valuesOf(x)
  const n = 256
  let h = new Int32Array(n)
  for (const v of data) {
    const l = level(v)
    h[l] = (h[l] ?? 0) + 1
  }
  let left = 0
  let right = 0
  for (let i = 0; i < n; i++)
    if ((h[i] ?? 0) > 0) {
      left = i
      break
    }
  if (left > 0) left--
  for (let i = n - 1; i > 0; i--)
    if ((h[i] ?? 0) > 0) {
      right = i
      break
    }
  if (right < n - 1) right++
  let peak = 0
  let maxCount = 0
  for (let i = 0; i < n; i++)
    if ((h[i] ?? 0) > maxCount) {
      maxCount = h[i] ?? 0
      peak = i
    }
  let flipped = false
  if (peak - left < right - peak) {
    flipped = true
    h = h.reverse()
    left = n - 1 - right
    peak = n - 1 - peak
  }
  let t = left
  let dist = 0
  const a = maxCount
  const b = left - peak
  for (let i = left + 1; i <= peak; i++) {
    const d = a * i + b * (h[i] ?? 0)
    if (d > dist) {
      dist = d
      t = i
    }
  }
  t--
  if (flipped) t = n - 1 - t
  return t / 255
}

/** Options of {@link adaptiveThreshold}. */
export interface AdaptiveThresholdOptions {
  /** The value of pixels that pass (default 1). */
  readonly maxValue?: number
  /** The local level: the window mean or a Gaussian-weighted mean (default 'mean'). */
  readonly method?: 'mean' | 'gaussian'
  /** 'binary' (above the local level minus C) or 'binary-inv' (default 'binary'). */
  readonly type?: 'binary' | 'binary-inv'
  /** The odd window side (default 11). */
  readonly blockSize?: number
  /** The offset subtracted from the local level, in image units (default 0; OpenCV's C = 2 is 2 / 255). */
  readonly C?: number
}

/**
 * Adaptive thresholding (cv2.adaptiveThreshold) at OpenCV's 256 levels: each pixel is compared with the mean (or the
 * Gaussian-weighted mean, OpenCV's kernel for the block size) of its blockSize x blockSize neighbourhood, the image
 * extended by replication and the local level rounded to a level as OpenCV's 8-bit arithmetic does; 'binary' keeps
 * pixels with level - local > -C.
 *
 * @param x - Grey image `[H, W]` in [0, 1].
 * @param options - maxValue, method, type, block size and C.
 * @returns The binary image `[H, W]`.
 * @concept thresholding
 * @example
 * ```ts
 * import { adaptiveThreshold } from 'mdml-engine/vision'
 * import { toArray } from 'mdml-engine/tensor'
 * toArray(adaptiveThreshold([[0.2, 0.2, 0.9, 0.2, 0.2]], { blockSize: 3 })) // => [[0, 0, 1, 0, 0]]
 * ```
 */
export function adaptiveThreshold(x: TensorLike, options: AdaptiveThresholdOptions = {}): Tensor {
  const r = greyOf(x, 'adaptiveThreshold')
  const block = options.blockSize ?? 11
  if (!(Number.isInteger(block) && block % 2 === 1 && block > 1))
    throw new MdmlError('value', 'adaptiveThreshold: the block size must be odd and above 1')
  const levels = Float64Array.from(r.data, level)
  const lt = imageTensor(levels, r.h, r.w)
  const local =
    (options.method ?? 'mean') === 'mean'
      ? primalOf(image.uniformFilter(lt, block, { mode: 'nearest' }))
      : gaussianBlur(lt, [block, block], 0, { border: 'replicate' })
  const mean = Float64Array.from(local.data, (v) => clamp(roundHalfEven(Number(v)), 0, 255))
  // OpenCV rounds the offset towards keeping fewer pixels: up for 'binary', down for 'binary-inv'.
  const delta = Math.round(255 * (options.C ?? 0) * 1e9) / 1e9
  const inv = (options.type ?? 'binary') === 'binary-inv'
  const idelta = inv ? Math.floor(delta) : Math.ceil(delta)
  const maxValue = options.maxValue ?? 1
  const out = Float64Array.from(levels, (v, i) => {
    const pass = v - (mean[i] ?? 0) > -idelta
    return pass !== inv ? maxValue : 0
  })
  return rasterTensor({ ...r, data: out })
}

/**
 * Multi-level Otsu thresholds (Liao et al. 2001; skimage.filters.threshold_multiotsu): the classes - 1 thresholds
 * that maximise the between-class variance of the image histogram (nbins bins over the image's range), found by
 * dynamic programming over contiguous bin ranges; each threshold is the centre of the last bin of its class.
 *
 * @param x - Grey image (any shape).
 * @param options - classes (default 3) and bins (default 256).
 * @returns The increasing thresholds.
 * @concept thresholding
 * @example
 * ```ts
 * import { thresholdMultiOtsu } from 'mdml-engine/vision'
 * thresholdMultiOtsu([0, 0, 0.1, 0.5, 0.5, 0.6, 1, 1, 0.9], { classes: 3, nbins: 10 }) // => [0.15, 0.55]
 * ```
 */
export function thresholdMultiOtsu(
  x: TensorLike,
  options: { readonly classes?: number; readonly nbins?: number } = {},
): number[] {
  const classes = options.classes ?? 3
  const nbins = options.nbins ?? 256
  if (!(Number.isInteger(classes) && classes >= 2)) throw new MdmlError('value', 'thresholdMultiOtsu: classes >= 2')
  const h = histogram(x, { bins: nbins })
  const counts = toFlat(h.counts)
  const edges = toFlat(h.edges)
  const total = counts.reduce((a, b) => a + b, 0)
  const prob = counts.map((c) => c / total)
  const centres = counts.map((_, i) => ((edges[i] ?? 0) + (edges[i + 1] ?? 0)) / 2)
  const occupied = prob.flatMap((p, i) => (p > 0 ? [i] : []))
  if (occupied.length < classes)
    throw new MdmlError('value', 'thresholdMultiOtsu: fewer distinct histogram bins than classes')
  if (occupied.length === classes) return occupied.slice(0, -1).map((i) => centres[i] ?? 0)
  // Cumulative zeroth and first moments (bin index as the value), so a range's term is M^2 / P.
  const P = new Float64Array(nbins + 1)
  const S = new Float64Array(nbins + 1)
  for (let i = 0; i < nbins; i++) {
    P[i + 1] = (P[i] ?? 0) + (prob[i] ?? 0)
    S[i + 1] = (S[i] ?? 0) + i * (prob[i] ?? 0)
  }
  const term = (a: number, b: number): number => {
    const p = (P[b + 1] ?? 0) - (P[a] ?? 0)
    const s = (S[b + 1] ?? 0) - (S[a] ?? 0)
    return p > 0 ? (s * s) / p : 0
  }
  // best[c][j]: the largest sum over c + 1 classes covering bins 0..j; from[c][j] the end of the c-th class.
  const best = Array.from({ length: classes }, () => new Float64Array(nbins).fill(Number.NEGATIVE_INFINITY))
  const from = Array.from({ length: classes }, () => new Int32Array(nbins).fill(-1))
  for (let j = 0; j < nbins; j++) (best[0] as Float64Array)[j] = term(0, j)
  for (let c = 1; c < classes; c++)
    for (let j = c; j < nbins; j++)
      for (let i = c - 1; i < j; i++) {
        const v = (best[c - 1]?.[i] ?? 0) + term(i + 1, j)
        if (v > (best[c]?.[j] ?? 0)) {
          ;(best[c] as Float64Array)[j] = v
          ;(from[c] as Int32Array)[j] = i
        }
      }
  const ends: number[] = []
  let j = nbins - 1
  for (let c = classes - 1; c >= 1; c--) {
    j = from[c]?.[j] ?? 0
    ends.unshift(j)
  }
  return ends.map((i) => centres[i] ?? 0)
}

/** The local mean and standard deviation over a window (scikit-image's integral-image pass, by box filters). */
function localMeanStd(x: TensorLike, windowSize: number, name: string): { m: Float64Array; s: Float64Array } {
  if (!(Number.isInteger(windowSize) && windowSize % 2 === 1 && windowSize > 1))
    throw new MdmlError('value', `${name}: the window size must be odd and above 1`)
  const r = rasterOf(x, name)
  const t = rasterTensor(r)
  const sq = rasterTensor({ ...r, data: Float64Array.from(r.data, (v) => v * v) })
  const m = Float64Array.from(primalOf(image.uniformFilter(t, windowSize, { mode: 'mirror' })).data, Number)
  const g2 = primalOf(image.uniformFilter(sq, windowSize, { mode: 'mirror' })).data
  const s = Float64Array.from(m, (v, i) => Math.sqrt(Math.max(Number(g2[i] ?? 0) - v * v, 0)))
  return { m, s }
}

/**
 * Niblack's local threshold (skimage.filters.threshold_niblack): T = m - k s with m and s the mean and standard
 * deviation over a window (the image mirrored at the borders).
 *
 * @param x - Grey image.
 * @param options - The odd window size (default 15) and k (default 0.2).
 * @returns The threshold image (pixels above it are foreground).
 * @concept thresholding
 * @example
 * ```ts
 * import { thresholdNiblack } from 'mdml-engine/vision'
 * import { toArray } from 'mdml-engine/tensor'
 * toArray(thresholdNiblack([[0.5, 0.5, 0.5]], { windowSize: 3 })) // => [[0.5, 0.5, 0.5]]
 * ```
 */
export function thresholdNiblack(
  x: TensorLike,
  options: { readonly windowSize?: number; readonly k?: number } = {},
): Tensor {
  const { m, s } = localMeanStd(x, options.windowSize ?? 15, 'thresholdNiblack')
  const k = options.k ?? 0.2
  return fromStorage(
    Float64Array.from(m, (v, i) => v - k * (s[i] ?? 0)),
    [...valuesOf(x).shape],
    'float64',
  )
}

/**
 * Sauvola's local threshold (skimage.filters.threshold_sauvola): T = m (1 + k (s / r - 1)), the standard deviation
 * scaled by its dynamic range r (scikit-image's default for float images, 1).
 *
 * @param x - Grey image.
 * @param options - The odd window size (default 15), k (default 0.2) and r (default 1).
 * @returns The threshold image.
 * @concept thresholding
 * @example
 * ```ts
 * import { thresholdSauvola } from 'mdml-engine/vision'
 * import { toArray } from 'mdml-engine/tensor'
 * toArray(thresholdSauvola([[0.5, 0.5, 0.5]], { windowSize: 3 })) // => [[0.4, 0.4, 0.4]]
 * ```
 */
export function thresholdSauvola(
  x: TensorLike,
  options: { readonly windowSize?: number; readonly k?: number; readonly r?: number } = {},
): Tensor {
  const { m, s } = localMeanStd(x, options.windowSize ?? 15, 'thresholdSauvola')
  const k = options.k ?? 0.2
  const range = options.r ?? 1
  return fromStorage(
    Float64Array.from(m, (v, i) => v * (1 + k * ((s[i] ?? 0) / range - 1))),
    [...valuesOf(x).shape],
    'float64',
  )
}
