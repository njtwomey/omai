/**
 * Image histograms and the operations built on them, OpenCV and scikit-image style: N-dimensional histograms of
 * chosen channels, histogram equalisation and contrast-limited adaptive equalisation (CLAHE) at OpenCV's 256 levels,
 * histogram matching to a reference, intensity rescaling and percentile contrast stretching, normalisation, the
 * comparison measures of `cv2.compareHist`, and back-projection of a histogram onto an image.
 */

import { MdmlError, roundHalfEven } from '../base/index.ts'
import { extendIndex } from '../numerics/convolution/index.ts'
import { quantile } from '../stats/index.ts'
import { fromStorage, type Tensor, type TensorLike } from '../tensor/index.ts'
import { clamp, f32, greyOf, imageTensor, level, rasterOf, rasterTensor, valuesOf } from './_image.ts'

/** Options of {@link calcHist} and {@link calcBackProject}. */
export interface HistogramOptions {
  /** The channels to histogram (default [0] for a single-channel image, else all channels). */
  readonly channels?: readonly number[]
  /** Bins per channel, one value shared or one per channel (default 256). */
  readonly bins?: number | readonly number[]
  /** The value range [lo, hi] per channel, one shared or one per channel (default [0, 1]). */
  readonly ranges?: readonly [number, number] | readonly (readonly [number, number])[]
  /** A mask `[H, W]`: only pixels where it is nonzero count. */
  readonly mask?: TensorLike
}

interface Binning {
  readonly channels: number[]
  readonly bins: number[]
  readonly ranges: [number, number][]
}

function binningOf(c: number, options: HistogramOptions, name: string): Binning {
  const channels = [...(options.channels ?? (c === 1 ? [0] : Array.from({ length: c }, (_, k) => k)))]
  if (channels.some((k) => !Number.isInteger(k) || k < 0 || k >= c))
    throw new MdmlError('value', `${name}: channel out of range`)
  const d = channels.length
  const b = options.bins ?? 256
  const bins = typeof b === 'number' ? new Array<number>(d).fill(b) : [...b]
  const r = options.ranges ?? [0, 1]
  const ranges: [number, number][] =
    typeof r[0] === 'number'
      ? Array.from({ length: d }, () => [r[0] as number, r[1] as number])
      : (r as readonly (readonly [number, number])[]).map((p) => [p[0], p[1]])
  if (bins.length !== d || ranges.length !== d)
    throw new MdmlError('shape', `${name}: one bin count and range per channel`)
  if (bins.some((n) => !Number.isInteger(n) || n < 1)) throw new MdmlError('value', `${name}: bins must be positive`)
  if (ranges.some(([lo, hi]) => !(hi > lo))) throw new MdmlError('value', `${name}: each range needs lo < hi`)
  return { channels, bins, ranges }
}

/**
 * The flat histogram cell of a pixel, or -1 outside the ranges: bin floor((v - lo) bins / (hi - lo)) per channel
 * (cv2.calcHist with uniform ranges), the upper end hi falling in the last bin.
 */
function cellOf(data: Float64Array, offset: number, b: Binning): number {
  let cell = 0
  for (let d = 0; d < b.channels.length; d++) {
    const v = data[offset + (b.channels[d] ?? 0)] ?? 0
    const [lo, hi] = b.ranges[d] as [number, number]
    const n = b.bins[d] ?? 1
    if (!(v >= lo && v <= hi)) return -1
    const k = clamp(Math.floor(((v - lo) * n) / (hi - lo)), 0, n - 1)
    cell = cell * n + k
  }
  return cell
}

/**
 * The histogram of chosen channels of an image (cv2.calcHist with uniform bins): counts over the product of the
 * channels' bins, values outside a range ignored; the upper end of a range counts in the last bin (OpenCV's ranges
 * are half-open).
 *
 * @param x - Image `[H, W]` or `[H, W, C]`.
 * @param options - Channels, bins, ranges and mask.
 * @returns Counts `[bins_0, bins_1, ...]`.
 * @concept image-histogram
 * @example
 * ```ts
 * import { calcHist } from 'mdml-engine/vision'
 * import { toArray } from 'mdml-engine/tensor'
 * toArray(calcHist([[0, 0.3], [0.6, 1]], { bins: 2 })) // => [2, 2]
 * ```
 */
export function calcHist(x: TensorLike, options: HistogramOptions = {}): Tensor {
  const r = rasterOf(x, 'calcHist')
  const b = binningOf(r.c, options, 'calcHist')
  const mask = options.mask === undefined ? undefined : greyOf(options.mask, 'calcHist').data
  if (mask !== undefined && mask.length !== r.h * r.w) throw new MdmlError('shape', 'calcHist: the mask is [H, W]')
  const total = b.bins.reduce((p, n) => p * n, 1)
  const counts = new Float64Array(total)
  for (let i = 0; i < r.h * r.w; i++) {
    if (mask !== undefined && (mask[i] ?? 0) === 0) continue
    const cell = cellOf(r.data, i * r.c, b)
    if (cell >= 0) counts[cell] = (counts[cell] ?? 0) + 1
  }
  return fromStorage(counts, [...b.bins], 'float64')
}

/**
 * Back-projection of a histogram (cv2.calcBackProject): every pixel takes the histogram value of its cell (times a
 * scale), 0 outside the ranges; with a normalised hue histogram it is the likelihood map that mean shift and CamShift
 * track.
 *
 * @param x - Image `[H, W]` or `[H, W, C]`.
 * @param hist - Histogram `[bins_0, ...]` over the same channels.
 * @param options - Channels and ranges (the bins come from the histogram's shape) and a scale (default 1).
 * @returns The back-projection `[H, W]`.
 * @concept histogram-backprojection
 * @example
 * ```ts
 * import { calcBackProject } from 'mdml-engine/vision'
 * import { toArray } from 'mdml-engine/tensor'
 * toArray(calcBackProject([[0, 0.3], [0.6, 1]], [0.25, 0.75])) // => [[0.25, 0.25], [0.75, 0.75]]
 * ```
 */
export function calcBackProject(
  x: TensorLike,
  hist: TensorLike,
  options: Omit<HistogramOptions, 'bins' | 'mask'> & { readonly scale?: number } = {},
): Tensor {
  const r = rasterOf(x, 'calcBackProject')
  const h = valuesOf(hist)
  const b = binningOf(r.c, { ...options, bins: h.shape }, 'calcBackProject')
  const scale = options.scale ?? 1
  const out = new Float64Array(r.h * r.w)
  for (let i = 0; i < r.h * r.w; i++) {
    const cell = cellOf(r.data, i * r.c, b)
    out[i] = cell < 0 ? 0 : (h.data[cell] ?? 0) * scale
  }
  return imageTensor(out, r.h, r.w)
}

/**
 * Histogram equalisation (cv2.equalizeHist) at OpenCV's 256 levels: the image is read as levels round(255 v), the
 * cumulative histogram after the first occupied level is scaled to 0 .. 255 and rounded, and the result returned as
 * levels / 255.
 *
 * @param x - Grey image `[H, W]` in [0, 1].
 * @returns The equalised image in [0, 1].
 * @concept histogram-equalisation
 * @example
 * ```ts
 * import { equalizeHist } from 'mdml-engine/vision'
 * import { toArray } from 'mdml-engine/tensor'
 * toArray(equalizeHist([[0.2, 0.2, 0.4, 0.6]])) // => [[0, 0, 0.5019607843137255, 1]]
 * ```
 */
export function equalizeHist(x: TensorLike): Tensor {
  const r = greyOf(x, 'equalizeHist')
  const levels = Int32Array.from(r.data, level)
  const hist = new Int32Array(256)
  for (const v of levels) hist[v] = (hist[v] ?? 0) + 1
  let i = 0
  while (i < 256 && (hist[i] ?? 0) === 0) i++
  const total = levels.length
  const lut = new Float64Array(256)
  if ((hist[i] ?? 0) === total) lut.fill(i)
  else {
    const scale = f32(255 / (total - (hist[i] ?? 0)))
    let sum = 0
    lut[i] = 0
    for (let k = i + 1; k < 256; k++) {
      sum += hist[k] ?? 0
      lut[k] = clamp(roundHalfEven(f32(sum * scale)), 0, 255)
    }
  }
  return rasterTensor({ ...r, data: Float64Array.from(levels, (v) => (lut[v] ?? 0) / 255) })
}

/** Options of {@link clahe}. */
export interface ClaheOptions {
  /** The clip limit, as a multiple of the mean count per level (default 40, OpenCV's; 0 disables clipping). */
  readonly clipLimit?: number
  /** Tiles across and down (default [8, 8]). */
  readonly tileGridSize?: readonly [number, number]
}

/**
 * Contrast-limited adaptive histogram equalisation (Zuiderveld 1994; cv2.createCLAHE().apply) at OpenCV's 256
 * levels: the image (extended by reflection to a whole number of tiles) is cut into tiles, each tile's histogram is
 * clipped at the limit with the excess spread evenly, its cumulative sum gives the tile's mapping, and every pixel
 * blends the mappings of its four nearest tile centres bilinearly. The arithmetic follows OpenCV's single precision.
 *
 * @param x - Grey image `[H, W]` in [0, 1].
 * @param options - Clip limit and tile grid.
 * @returns The equalised image in [0, 1] (levels / 255).
 * @concept histogram-equalisation
 * @example
 * ```ts
 * import { clahe } from 'mdml-engine/vision'
 * const x = Array.from({ length: 16 }, (_, i) => Array.from({ length: 16 }, (_, j) => (i + j) / 60))
 * clahe(x, { tileGridSize: [2, 2] }).shape // => [16, 16]
 * ```
 */
export function clahe(x: TensorLike, options: ClaheOptions = {}): Tensor {
  const r = greyOf(x, 'clahe')
  const { h, w } = r
  const [tilesX, tilesY] = options.tileGridSize ?? [8, 8]
  if (!(Number.isInteger(tilesX) && Number.isInteger(tilesY) && tilesX >= 1 && tilesY >= 1))
    throw new MdmlError('value', 'clahe: the tile grid must be positive integers')
  const src = Int32Array.from(r.data, level)
  // OpenCV pads to whole tiles (by a full extra tile when a side already divides) unless both sides divide.
  const divides = w % tilesX === 0 && h % tilesY === 0
  const ew = divides ? w : w + tilesX - (w % tilesX)
  const eh = divides ? h : h + tilesY - (h % tilesY)
  const tw = ew / tilesX
  const th = eh / tilesY
  const area = tw * th
  const lutScale = f32(255 / area)
  const limit = (options.clipLimit ?? 40) > 0 ? Math.max(1, Math.trunc(((options.clipLimit ?? 40) * area) / 256)) : 0
  const luts = new Float64Array(tilesX * tilesY * 256)
  const hist = new Int32Array(256)
  for (let ty = 0; ty < tilesY; ty++)
    for (let tx = 0; tx < tilesX; tx++) {
      hist.fill(0)
      for (let y = ty * th; y < (ty + 1) * th; y++)
        for (let xx = tx * tw; xx < (tx + 1) * tw; xx++) {
          const v = src[extendIndex(y, h, 'reflect') * w + extendIndex(xx, w, 'reflect')] ?? 0
          hist[v] = (hist[v] ?? 0) + 1
        }
      if (limit > 0) {
        let clipped = 0
        for (let i = 0; i < 256; i++)
          if ((hist[i] ?? 0) > limit) {
            clipped += (hist[i] ?? 0) - limit
            hist[i] = limit
          }
        const batch = Math.trunc(clipped / 256)
        let residual = clipped - batch * 256
        for (let i = 0; i < 256; i++) hist[i] = (hist[i] ?? 0) + batch
        if (residual !== 0) {
          const step = Math.max(Math.trunc(256 / residual), 1)
          for (let i = 0; i < 256 && residual > 0; i += step, residual--) hist[i] = (hist[i] ?? 0) + 1
        }
      }
      let sum = 0
      const base = (ty * tilesX + tx) * 256
      for (let i = 0; i < 256; i++) {
        sum += hist[i] ?? 0
        luts[base + i] = clamp(roundHalfEven(f32(sum * lutScale)), 0, 255)
      }
    }
  const invW = f32(1 / tw)
  const invH = f32(1 / th)
  const out = new Float64Array(h * w)
  // Single precision as OpenCV's arm64 build evaluates it, with fused multiply-adds: the products of two singles are
  // exact in double, so f32(a b + c) is the fused result.
  for (let y = 0; y < h; y++) {
    const tyf = f32(y * invH - 0.5)
    const ty1 = Math.floor(tyf)
    const ya = f32(tyf - ty1)
    const ya1 = f32(1 - ya)
    const r1 = Math.max(ty1, 0) * tilesX
    const r2 = Math.min(ty1 + 1, tilesY - 1) * tilesX
    for (let xx = 0; xx < w; xx++) {
      const txf = f32(xx * invW - 0.5)
      const tx1 = Math.floor(txf)
      const xa = f32(txf - tx1)
      const xa1 = f32(1 - xa)
      const c1 = Math.max(tx1, 0)
      const c2 = Math.min(tx1 + 1, tilesX - 1)
      const v = src[y * w + xx] ?? 0
      const at = (row: number, col: number): number => luts[(row + col) * 256 + v] ?? 0
      const top = f32(at(r1, c1) * xa1 + f32(at(r1, c2) * xa))
      const bottom = f32(at(r2, c1) * xa1 + f32(at(r2, c2) * xa))
      const res = f32(top * ya1 + f32(bottom * ya))
      out[y * w + xx] = clamp(roundHalfEven(res), 0, 255) / 255
    }
  }
  return rasterTensor({ ...r, data: out })
}

/**
 * Histogram matching (skimage.exposure.match_histograms): each value of the image is replaced by the reference value
 * at the same quantile, interpolating linearly between the reference's distinct values; colour images are matched
 * channel by channel.
 *
 * @param x - Image `[H, W]` or `[H, W, C]`.
 * @param reference - Reference image with the same number of channels.
 * @returns The matched image.
 * @concept histogram-matching
 * @example
 * ```ts
 * import { matchHistograms } from 'mdml-engine/vision'
 * import { toArray } from 'mdml-engine/tensor'
 * toArray(matchHistograms([[0, 1], [2, 3]], [[10, 20], [30, 40]])) // => [[10, 20], [30, 40]]
 * ```
 */
export function matchHistograms(x: TensorLike, reference: TensorLike): Tensor {
  const src = rasterOf(x, 'matchHistograms')
  const ref = rasterOf(reference, 'matchHistograms')
  if (src.c !== ref.c) throw new MdmlError('shape', 'matchHistograms: the images differ in channels')
  const out = new Float64Array(src.data.length)
  const n = src.h * src.w
  const m = ref.h * ref.w
  for (let k = 0; k < src.c; k++) {
    const sv = Float64Array.from({ length: n }, (_, i) => src.data[i * src.c + k] ?? 0)
    const rv = Float64Array.from({ length: m }, (_, i) => ref.data[i * ref.c + k] ?? 0)
    const quantiles = (v: Float64Array): { values: number[]; q: number[] } => {
      const sorted = Float64Array.from(v).sort()
      const values: number[] = []
      const q: number[] = []
      for (let i = 0; i < sorted.length; i++) {
        if (i + 1 < sorted.length && sorted[i + 1] === sorted[i]) continue
        values.push(sorted[i] ?? 0)
        q.push((i + 1) / sorted.length)
      }
      return { values, q }
    }
    const s = quantiles(sv)
    const t = quantiles(rv)
    const mapped = new Map<number, number>()
    let j = 0
    for (let u = 0; u < s.values.length; u++) {
      const qu = s.q[u] ?? 0
      // numpy.interp: clamp to the ends, else linear between the bracketing reference quantiles.
      let value: number
      if (qu <= (t.q[0] ?? 0)) value = t.values[0] ?? 0
      else if (qu >= (t.q[t.q.length - 1] ?? 0)) value = t.values[t.values.length - 1] ?? 0
      else {
        while ((t.q[j + 1] ?? 0) < qu) j++
        const q0 = t.q[j] ?? 0
        const q1 = t.q[j + 1] ?? 0
        const v0 = t.values[j] ?? 0
        const v1 = t.values[j + 1] ?? 0
        value = v0 + ((qu - q0) * (v1 - v0)) / (q1 - q0)
      }
      mapped.set(s.values[u] ?? 0, value)
    }
    for (let i = 0; i < n; i++) out[i * src.c + k] = mapped.get(sv[i] ?? 0) ?? 0
  }
  return rasterTensor({ ...src, data: out })
}

/**
 * Rescale intensities (skimage.exposure.rescale_intensity): values are clipped to the input range and mapped
 * linearly onto the output range.
 *
 * @param x - Image (any shape).
 * @param options - inRange (default the image's minimum and maximum) and outRange (default [0, 1]).
 * @returns The rescaled image.
 * @concept contrast-stretching
 * @example
 * ```ts
 * import { rescaleIntensity } from 'mdml-engine/vision'
 * import { toArray } from 'mdml-engine/tensor'
 * toArray(rescaleIntensity([0.2, 0.4, 0.6])) // => [0, 0.5, 1]
 * ```
 */
export function rescaleIntensity(
  x: TensorLike,
  options: { readonly inRange?: readonly [number, number]; readonly outRange?: readonly [number, number] } = {},
): Tensor {
  const { data, shape } = valuesOf(x)
  let lo = Number.POSITIVE_INFINITY
  let hi = Number.NEGATIVE_INFINITY
  for (const v of data) {
    lo = Math.min(lo, v)
    hi = Math.max(hi, v)
  }
  const [a, b] = options.inRange ?? [lo, hi]
  const [c, d] = options.outRange ?? [0, 1]
  const out = Float64Array.from(data, (v) => {
    const t = clamp(v, a, b)
    return b === a ? c : ((t - a) / (b - a)) * (d - c) + c
  })
  return fromStorage(out, [...shape], 'float64')
}

/**
 * Percentile contrast stretching: the low and high percentiles of the image (numpy's linear interpolation) become 0
 * and 1, everything beyond them clipped.
 *
 * @param x - Image (any shape).
 * @param options - The percentiles (default 2 and 98).
 * @returns The stretched image in [0, 1].
 * @concept contrast-stretching
 * @example
 * ```ts
 * import { contrastStretch } from 'mdml-engine/vision'
 * import { toArray } from 'mdml-engine/tensor'
 * toArray(contrastStretch([0, 1, 2, 3, 4], { low: 25, high: 75 })) // => [0, 0, 0.5, 1, 1]
 * ```
 */
export function contrastStretch(
  x: TensorLike,
  options: { readonly low?: number; readonly high?: number } = {},
): Tensor {
  const lo = quantile(x, (options.low ?? 2) / 100)
  const hi = quantile(x, (options.high ?? 98) / 100)
  return rescaleIntensity(x, { inRange: [lo, hi] })
}

/** The norms of {@link normalize}. */
export type NormType = 'minmax' | 'l1' | 'l2' | 'inf'

/**
 * Normalise an array (cv2.normalize): 'minmax' maps the minimum and maximum onto alpha and beta; 'l1', 'l2' and
 * 'inf' scale so that norm equals alpha.
 *
 * @param x - Array (any shape).
 * @param options - alpha (default 1 for the norms, 0 for minmax), beta (default 1, minmax only), the norm (default
 *   'l2', OpenCV's).
 * @returns The normalised array.
 * @concept normalisation
 * @example
 * ```ts
 * import { normalize } from 'mdml-engine/vision'
 * import { toArray } from 'mdml-engine/tensor'
 * toArray(normalize([3, 4])) // => [0.6, 0.8]
 * toArray(normalize([1, 2, 3], { normType: 'minmax' })) // => [0, 0.5, 1]
 * ```
 */
export function normalize(
  x: TensorLike,
  options: { readonly alpha?: number; readonly beta?: number; readonly normType?: NormType } = {},
): Tensor {
  const { data, shape } = valuesOf(x)
  const type = options.normType ?? 'l2'
  let out: Float64Array
  if (type === 'minmax') {
    const alpha = options.alpha ?? 0
    const beta = options.beta ?? 1
    let lo = Number.POSITIVE_INFINITY
    let hi = Number.NEGATIVE_INFINITY
    for (const v of data) {
      lo = Math.min(lo, v)
      hi = Math.max(hi, v)
    }
    const lo2 = Math.min(alpha, beta)
    const hi2 = Math.max(alpha, beta)
    const scale = hi > lo ? (hi2 - lo2) / (hi - lo) : 0
    out = Float64Array.from(data, (v) => (v - lo) * scale + lo2)
  } else {
    let norm = 0
    for (const v of data) {
      if (type === 'l1') norm += Math.abs(v)
      else if (type === 'l2') norm += v * v
      else norm = Math.max(norm, Math.abs(v))
    }
    if (type === 'l2') norm = Math.sqrt(norm)
    const scale = norm > 0 ? (options.alpha ?? 1) / norm : 0
    out = Float64Array.from(data, (v) => v * scale)
  }
  return fromStorage(out, [...shape], 'float64')
}

/** The comparison measures of {@link compareHist} (cv2.HISTCMP_*). */
export type HistogramComparison =
  | 'correlation'
  | 'chi-square'
  | 'chi-square-alt'
  | 'intersection'
  | 'bhattacharyya'
  | 'kl-divergence'

/**
 * Compare two histograms (cv2.compareHist): Pearson correlation of the bin values, chi-square
 * sum (h1 - h2)^2 / h1 (and the symmetric 2 sum (h1 - h2)^2 / (h1 + h2)), intersection sum min(h1, h2), the
 * Bhattacharyya (Hellinger) distance sqrt(1 - sum sqrt(h1 h2) / sqrt(sum h1 sum h2)), or the Kullback-Leibler
 * divergence sum h1 log(h1 / h2).
 *
 * @param h1 - First histogram (any shape).
 * @param h2 - Second histogram, the same shape.
 * @param method - The measure.
 * @returns The comparison value.
 * @concept histogram-comparison
 * @example
 * ```ts
 * import { compareHist } from 'mdml-engine/vision'
 * compareHist([1, 2, 3], [1, 2, 3], 'correlation') // => 1
 * compareHist([1, 0], [0, 1], 'bhattacharyya') // => 1
 * ```
 */
export function compareHist(h1: TensorLike, h2: TensorLike, method: HistogramComparison): number {
  const a = valuesOf(h1).data
  const b = valuesOf(h2).data
  if (a.length !== b.length) throw new MdmlError('shape', 'compareHist: histograms differ in size')
  const n = a.length
  let result = 0
  let s1 = 0
  let s2 = 0
  let s11 = 0
  let s22 = 0
  let s12 = 0
  for (let i = 0; i < n; i++) {
    const p = a[i] ?? 0
    const q = b[i] ?? 0
    switch (method) {
      case 'chi-square':
      case 'chi-square-alt': {
        const den = method === 'chi-square' ? p : p + q
        if (Math.abs(den) > Number.EPSILON) result += ((p - q) * (p - q)) / den
        break
      }
      case 'correlation':
        s1 += p
        s2 += q
        s11 += p * p
        s22 += q * q
        s12 += p * q
        break
      case 'intersection':
        result += Math.min(p, q)
        break
      case 'bhattacharyya':
        result += Math.sqrt(p * q)
        s1 += p
        s2 += q
        break
      case 'kl-divergence':
        if (Math.abs(p) > Number.EPSILON) result += p * Math.log(p / (Math.abs(q) <= Number.EPSILON ? 1e-10 : q))
        break
      default:
        throw new MdmlError('value', `compareHist: unknown method ${String(method)}`)
    }
  }
  if (method === 'chi-square-alt') return 2 * result
  if (method === 'correlation') {
    const num = s12 - (s1 * s2) / n
    const den2 = (s11 - (s1 * s1) / n) * (s22 - (s2 * s2) / n)
    return Math.abs(den2) > Number.EPSILON ? num / Math.sqrt(den2) : 1
  }
  if (method === 'bhattacharyya') {
    const s = s1 * s2
    const inv = Math.abs(s) > 1.1920928955078125e-7 ? 1 / Math.sqrt(s) : 1
    return Math.sqrt(Math.max(1 - result * inv, 0))
  }
  return result
}
