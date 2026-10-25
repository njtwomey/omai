/**
 * Image feature detectors and descriptors, scikit-image style: the structure tensor and the Harris and Shi-Tomasi
 * corner responses, local peak finding, the Canny edge detector, local binary patterns (texture codes), histograms of
 * oriented gradients (HOG), and the straight-line and circular Hough transforms that vote for shapes.
 */

import { MdmlError } from '../../base/index.ts'
import { fromStorage, primalOf, type Tensor, type TensorLike } from '../../tensor/index.ts'
import { float } from '../_shared.ts'
import type { BorderOptions } from './correlate.ts'
import { gaussianFilter, maximumFilter } from './filters.ts'
import { label } from './morphology.ts'

// Image features as scikit-image computes them: structure tensors, Harris and Shi-Tomasi corners, local peaks,
// Canny edges, local binary patterns, histograms of oriented gradients and Hough transforms for lines and circles.
// TODO: Laplacian- and difference-of-Gaussian blob detection (blob_log, blob_dog with overlap pruning) and Hough peak
// extraction.

interface Plane {
  readonly h: number
  readonly w: number
  readonly data: Float64Array
}

function planeOf(x: TensorLike, name: string): Plane {
  const v = primalOf(float(x))
  if (v.shape.length !== 2) throw new MdmlError('shape', `${name}: images are [H, W]`)
  return { h: v.shape[0] ?? 0, w: v.shape[1] ?? 0, data: Float64Array.from(v.data, Number) }
}

const tensorOf = (p: Plane): Tensor => fromStorage(p.data, [p.h, p.w], 'float64')

/** scipy.ndimage.sobel (unnormalised [-1, 0, 1] along axis, [1, 2, 1] across). */
function ndSobel(p: Plane, axis: 0 | 1, mode: 'constant' | 'reflect' | 'nearest', cval: number): Plane {
  const at = (y: number, x: number): number => {
    if (y >= 0 && y < p.h && x >= 0 && x < p.w) return p.data[y * p.w + x] ?? 0
    if (mode === 'constant') return cval
    const clampY =
      mode === 'nearest' ? Math.min(p.h - 1, Math.max(0, y)) : y < 0 ? -y - 1 : y >= p.h ? 2 * p.h - y - 1 : y
    const clampX =
      mode === 'nearest' ? Math.min(p.w - 1, Math.max(0, x)) : x < 0 ? -x - 1 : x >= p.w ? 2 * p.w - x - 1 : x
    return p.data[clampY * p.w + clampX] ?? 0
  }
  const out = new Float64Array(p.h * p.w)
  for (let y = 0; y < p.h; y++)
    for (let x = 0; x < p.w; x++) {
      let s = 0
      for (let k = -1; k <= 1; k++) {
        const wgt = k === 0 ? 2 : 1
        s += axis === 0 ? wgt * (at(y + 1, x + k) - at(y - 1, x + k)) : wgt * (at(y + k, x + 1) - at(y + k, x - 1))
      }
      out[y * p.w + x] = s
    }
  return { h: p.h, w: p.w, data: out }
}

/**
 * The structure tensor of an image (skimage.feature.structure_tensor): the Gaussian-weighted products of its Sobel
 * derivatives, [A_rr, A_rc, A_cc].
 *
 * @param x - Image `[H, W]`.
 * @param sigma - Gaussian window standard deviation (default 1).
 * @param options - Border (default 'constant', 0).
 * @returns The three tensor fields.
 * @concept structure-tensor
 * @example
 * ```ts
 * import { image } from 'mdml-engine/signal'
 * image.structureTensor([[0, 0, 1, 1], [0, 0, 1, 1], [0, 0, 1, 1]]).length // => 3
 * ```
 */
export function structureTensor(x: TensorLike, sigma = 1, options: BorderOptions = {}): Tensor[] {
  const p = planeOf(x, 'structureTensor')
  const mode = options.mode ?? 'constant'
  if (mode !== 'constant' && mode !== 'reflect' && mode !== 'nearest')
    throw new MdmlError('value', 'structureTensor: mode must be constant, reflect or nearest')
  const cval = options.cval ?? 0
  const dr = ndSobel(p, 0, mode, cval)
  const dc = ndSobel(p, 1, mode, cval)
  const prod = (a: Plane, b: Plane): Tensor =>
    fromStorage(
      Float64Array.from(a.data, (v, i) => v * (b.data[i] ?? 0)),
      [p.h, p.w],
      'float64',
    )
  return [prod(dr, dr), prod(dr, dc), prod(dc, dc)].map(
    (t) => primalOf(gaussianFilter(t, sigma, { mode, cval })) as Tensor,
  )
}

/**
 * The Harris corner response (skimage.feature.corner_harris): det A - k tr(A)^2 of the structure tensor, or
 * 2 det A / (tr A + eps) (Noble's measure).
 *
 * @param x - Image `[H, W]`.
 * @param options - Method ('k' default or 'eps'), k (0.05), eps (1e-6), sigma (1).
 * @returns The response map.
 * @concept corner-detection
 * @example
 * ```ts
 * import { image } from 'mdml-engine/signal'
 * image.cornerHarris(Array.from({ length: 6 }, (_, i) => Array.from({ length: 6 }, (_, j) => (i < 3 && j < 3 ? 1 : 0)))).shape // => [6, 6]
 * ```
 */
export function cornerHarris(
  x: TensorLike,
  options: { readonly method?: 'k' | 'eps'; readonly k?: number; readonly eps?: number; readonly sigma?: number } = {},
): Tensor {
  const [arr, arc, acc] = structureTensor(x, options.sigma ?? 1).map((t) => t.data)
  const k = options.k ?? 0.05
  const eps = options.eps ?? 1e-6
  const p = planeOf(x, 'cornerHarris')
  const out = Float64Array.from({ length: p.h * p.w }, (_, i) => {
    const a = Number(arr?.[i] ?? 0)
    const b = Number(arc?.[i] ?? 0)
    const c = Number(acc?.[i] ?? 0)
    const det = a * c - b * b
    const tr = a + c
    return (options.method ?? 'k') === 'k' ? det - k * tr * tr : (2 * det) / (tr + eps)
  })
  return fromStorage(out, [p.h, p.w], 'float64')
}

/**
 * The Shi-Tomasi (minimum eigenvalue) corner response (skimage.feature.corner_shi_tomasi).
 *
 * @param x - Image `[H, W]`.
 * @param sigma - Window standard deviation (default 1).
 * @returns The smaller eigenvalue of the structure tensor at every pixel.
 * @concept corner-detection
 * @example
 * ```ts
 * import { image } from 'mdml-engine/signal'
 * image.cornerShiTomasi([[0, 0, 0], [0, 1, 0], [0, 0, 0]]).shape // => [3, 3]
 * ```
 */
export function cornerShiTomasi(x: TensorLike, sigma = 1): Tensor {
  const [arr, arc, acc] = structureTensor(x, sigma).map((t) => t.data)
  const p = planeOf(x, 'cornerShiTomasi')
  const out = Float64Array.from({ length: p.h * p.w }, (_, i) => {
    const a = Number(arr?.[i] ?? 0)
    const b = Number(arc?.[i] ?? 0)
    const c = Number(acc?.[i] ?? 0)
    return (a + c - Math.sqrt((a - c) ** 2 + 4 * b * b)) / 2
  })
  return fromStorage(out, [p.h, p.w], 'float64')
}

/** Options of {@link peakLocalMax}. */
export interface PeakOptions {
  /** Minimum separation of peaks (Chebyshev distance; default 1). */
  readonly minDistance?: number
  /** Lowest peak value (default the image minimum). */
  readonly thresholdAbs?: number
  /** Lowest peak value as a fraction of the maximum. */
  readonly thresholdRel?: number
  /** Ignore peaks within this many pixels of the border (default minDistance; false for none). */
  readonly excludeBorder?: number | boolean
  /** Most peaks (default all). */
  readonly numPeaks?: number
}

/**
 * Local maxima of an image (skimage.feature.peak_local_max): pixels equal to the maximum of their
 * (2 minDistance + 1)-box, above the threshold, away from the border, strongest first and at least minDistance apart.
 *
 * @param x - Image `[H, W]`.
 * @param options - Separation, thresholds, border exclusion and count.
 * @returns Peak coordinates [row, col], strongest first.
 * @concept peak-detection
 * @example
 * ```ts
 * import { image } from 'mdml-engine/signal'
 * image.peakLocalMax([[0, 0, 0, 0, 0], [0, 3, 0, 0, 0], [0, 0, 0, 0, 0], [0, 0, 0, 5, 0], [0, 0, 0, 0, 0]]) // => [[3, 3], [1, 1]]
 * ```
 */
export function peakLocalMax(x: TensorLike, options: PeakOptions = {}): [number, number][] {
  const p = planeOf(x, 'peakLocalMax')
  const minDistance = options.minDistance ?? 1
  if (minDistance < 1) throw new MdmlError('value', 'peakLocalMax: minDistance must be at least 1')
  let lo = Number.POSITIVE_INFINITY
  let hi = Number.NEGATIVE_INFINITY
  for (const v of p.data) {
    lo = Math.min(lo, v)
    hi = Math.max(hi, v)
  }
  let threshold = options.thresholdAbs ?? lo
  if (options.thresholdRel !== undefined) threshold = Math.max(threshold, options.thresholdRel * hi)
  const border =
    options.excludeBorder === undefined || options.excludeBorder === true
      ? minDistance
      : options.excludeBorder === false
        ? 0
        : options.excludeBorder
  const maxed = maximumFilter(tensorOf(p), 2 * minDistance + 1, { mode: 'nearest' }).data
  let trivial = true
  for (let i = 0; i < p.data.length; i++) if (Number(maxed[i]) !== p.data[i]) trivial = false
  const cands: { r: number; c: number; v: number }[] = []
  if (!trivial)
    for (let r = border; r < p.h - border; r++)
      for (let c = border; c < p.w - border; c++) {
        const v = p.data[r * p.w + c] ?? 0
        if (v === Number(maxed[r * p.w + c]) && v > threshold) cands.push({ r, c, v })
      }
  // Stable sort by decreasing value (raster order breaks ties, as numpy's stable argsort).
  const order = cands.map((_, i) => i).sort((a, b) => (cands[b]?.v ?? 0) - (cands[a]?.v ?? 0) || a - b)
  const sorted = order.map((i) => cands[i] as { r: number; c: number; v: number })
  const out: [number, number][] = []
  const limit = options.numPeaks ?? Number.POSITIVE_INFINITY
  if (minDistance > 1) {
    const rejected = new Uint8Array(sorted.length)
    for (let i = 0; i < sorted.length && out.length < limit; i++) {
      if (rejected[i] === 1) continue
      const a = sorted[i] as { r: number; c: number }
      out.push([a.r, a.c])
      for (let j = 0; j < sorted.length; j++) {
        const b = sorted[j] as { r: number; c: number }
        if (j !== i && Math.max(Math.abs(a.r - b.r), Math.abs(a.c - b.c)) < minDistance) rejected[j] = 1
      }
    }
  } else for (const s of sorted) if (out.length < limit) out.push([s.r, s.c])
  return out
}

/** Options of {@link canny}. */
export interface CannyOptions {
  /** Gaussian smoothing (default 1). */
  readonly sigma?: number
  /** Hysteresis thresholds on the gradient magnitude (defaults 0.1 and 0.2), or quantiles. */
  readonly lowThreshold?: number
  readonly highThreshold?: number
  readonly useQuantiles?: boolean
  /** Border of the smoothing (default 'constant', compensated as scikit-image does). */
  readonly mode?: 'constant' | 'reflect' | 'nearest'
  readonly cval?: number
}

function quantile(sorted: Float64Array, q: number): number {
  const pos = q * (sorted.length - 1)
  const lo = Math.floor(pos)
  const hi = Math.min(sorted.length - 1, lo + 1)
  return (sorted[lo] ?? 0) + (pos - lo) * ((sorted[hi] ?? 0) - (sorted[lo] ?? 0))
}

/**
 * The Canny edge detector (skimage.feature.canny): Gaussian smoothing (normalised against the zero border), Sobel
 * gradients, non-maximum suppression along the gradient direction with bilinear interpolation, and hysteresis
 * thresholding by connected components.
 *
 * @param x - Image `[H, W]`.
 * @param options - Smoothing, thresholds and border.
 * @returns A 0/1 edge map.
 * @concept edge-detection
 * @example
 * ```ts
 * import { image } from 'mdml-engine/signal'
 * import { toFlat } from 'mdml-engine/tensor'
 * const x = Array.from({ length: 12 }, () => Array.from({ length: 12 }, (_, j) => (j < 6 ? 0 : 1)))
 * toFlat(image.canny(x)).reduce((a, b) => a + b, 0) > 0 // => true
 * ```
 */
export function canny(x: TensorLike, options: CannyOptions = {}): Tensor {
  const p = planeOf(x, 'canny')
  const sigma = options.sigma ?? 1
  const mode = options.mode ?? 'constant'
  const cval = options.cval ?? 0
  let smoothed = Float64Array.from(primalOf(gaussianFilter(tensorOf(p), sigma, { mode, cval })).data, Number)
  if (mode === 'constant') {
    const bleed = primalOf(
      gaussianFilter(fromStorage(new Float64Array(p.h * p.w).fill(1), [p.h, p.w], 'float64'), sigma, { mode, cval }),
    ).data
    smoothed = Float64Array.from(smoothed, (v, i) => v / (Number(bleed[i] ?? 1) + Number.EPSILON))
  }
  const s: Plane = { h: p.h, w: p.w, data: smoothed }
  const jsob = ndSobel(s, 1, 'reflect', 0).data
  const isob = ndSobel(s, 0, 'reflect', 0).data
  const mag = Float64Array.from(isob, (v, i) => Math.sqrt(v * v + (jsob[i] ?? 0) ** 2))
  let low = options.lowThreshold ?? 0.1
  let high = options.highThreshold ?? 0.2
  if (high < low) throw new MdmlError('value', 'canny: the low threshold exceeds the high threshold')
  if (options.useQuantiles === true) {
    const sorted = Float64Array.from(mag).sort()
    low = quantile(sorted, low)
    high = quantile(sorted, high)
  }
  const { h, w } = p
  const m = (r: number, c: number): number => mag[r * w + c] ?? 0
  const kept = new Float64Array(h * w)
  for (let r = 1; r < h - 1; r++)
    for (let c = 1; c < w - 1; c++) {
      const v = m(r, c)
      if (!(v >= low) || v === 0) continue
      const is = isob[r * w + c] ?? 0
      const js = jsob[r * w + c] ?? 0
      const down = is <= 0
      const up = is >= 0
      const left = js <= 0
      const right = js >= 0
      const cond1 = (up && right) || (down && left)
      const cond2 = (down && right) || (up && left)
      if (!cond1 && !cond2) continue
      const ai = Math.abs(is)
      const aj = Math.abs(js)
      let wgt: number
      let n11: number
      let n12: number
      let n21: number
      let n22: number
      if (cond1) {
        if (ai > aj) {
          wgt = aj / ai
          n11 = m(r + 1, c)
          n12 = m(r + 1, c + 1)
          n21 = m(r - 1, c)
          n22 = m(r - 1, c - 1)
        } else {
          wgt = ai / aj
          n11 = m(r, c + 1)
          n12 = m(r + 1, c + 1)
          n21 = m(r, c - 1)
          n22 = m(r - 1, c - 1)
        }
      } else if (ai < aj) {
        wgt = ai / aj
        n11 = m(r, c + 1)
        n12 = m(r - 1, c + 1)
        n21 = m(r, c - 1)
        n22 = m(r + 1, c - 1)
      } else {
        wgt = aj / ai
        n11 = m(r - 1, c)
        n12 = m(r - 1, c + 1)
        n21 = m(r + 1, c)
        n22 = m(r + 1, c - 1)
      }
      if (n12 * wgt + n11 * (1 - wgt) <= v && n22 * wgt + n21 * (1 - wgt) <= v) kept[r * w + c] = v
    }
  const lowMask = fromStorage(
    Float64Array.from(kept, (v) => (v > 0 ? 1 : 0)),
    [h, w],
    'float64',
  )
  const { labels, count } = label(lowMask, fromStorage(new Float64Array(9).fill(1), [3, 3], 'float64'))
  const good = new Uint8Array(count + 1)
  const ld = labels.data
  for (let i = 0; i < h * w; i++) if ((kept[i] ?? 0) > 0 && (kept[i] ?? 0) >= high) good[Number(ld[i] ?? 0)] = 1
  good[0] = 0
  return fromStorage(
    Float64Array.from({ length: h * w }, (_, i) => good[Number(ld[i] ?? 0)] ?? 0),
    [h, w],
    'float64',
  )
}

/**
 * Local binary patterns (skimage.feature.local_binary_pattern): P neighbours on a circle of radius R (bilinearly
 * interpolated, zero outside) thresholded at the centre, as a P-bit code ('default'), its minimum over rotations
 * ('ror'), or the uniform code (the number of set bits for patterns with at most two transitions, else P + 1).
 *
 * @param x - Image `[H, W]`.
 * @param P - Neighbours.
 * @param R - Radius.
 * @param method - 'default', 'ror' or 'uniform'.
 * @returns The codes.
 * @concept local-binary-patterns
 * @example
 * ```ts
 * import { image } from 'mdml-engine/signal'
 * import { toArray } from 'mdml-engine/tensor'
 * (toArray(image.localBinaryPattern([[5, 5, 5], [5, 1, 5], [5, 5, 5]], 8, 1)) as number[][])[1]?.[1] // => 255
 * ```
 */
export function localBinaryPattern(
  x: TensorLike,
  P: number,
  R: number,
  method: 'default' | 'ror' | 'uniform' = 'default',
): Tensor {
  const p = planeOf(x, 'localBinaryPattern')
  const rp = Array.from({ length: P }, (_, i) => Math.round(-R * Math.sin((2 * Math.PI * i) / P) * 1e5) / 1e5)
  const cp = Array.from({ length: P }, (_, i) => Math.round(R * Math.cos((2 * Math.PI * i) / P) * 1e5) / 1e5)
  const px = (r: number, c: number): number => (r < 0 || r >= p.h || c < 0 || c >= p.w ? 0 : (p.data[r * p.w + c] ?? 0))
  const bilinear = (r: number, c: number): number => {
    const r0 = Math.floor(r)
    const c0 = Math.floor(c)
    const r1 = Math.ceil(r)
    const c1 = Math.ceil(c)
    const dr = r - r0
    const dc = c - c0
    const top = (1 - dc) * px(r0, c0) + dc * px(r0, c1)
    const bottom = (1 - dc) * px(r1, c0) + dc * px(r1, c1)
    return (1 - dr) * top + dr * bottom
  }
  const out = new Float64Array(p.h * p.w)
  const bits = new Uint8Array(P)
  for (let r = 0; r < p.h; r++)
    for (let c = 0; c < p.w; c++) {
      const centre = p.data[r * p.w + c] ?? 0
      for (let i = 0; i < P; i++) bits[i] = bilinear(r + (rp[i] ?? 0), c + (cp[i] ?? 0)) - centre >= 0 ? 1 : 0
      let code = 0
      if (method === 'uniform') {
        let changes = 0
        for (let i = 0; i < P - 1; i++) if (bits[i] !== bits[i + 1]) changes++
        let ones = 0
        for (const b of bits) ones += b
        code = changes <= 2 ? ones : P + 1
      } else {
        for (let i = 0; i < P; i++) code += (bits[i] ?? 0) * 2 ** i
        if (method === 'ror') {
          let best = code
          const full = 2 ** P
          for (let s = 1; s < P; s++) {
            const rot = (Math.floor(code / 2 ** s) + ((code * 2 ** (P - s)) % full)) % full
            best = Math.min(best, rot)
          }
          code = best
        }
      }
      out[r * p.w + c] = code
    }
  return fromStorage(out, [p.h, p.w], 'float64')
}

/** Options of {@link hog}. */
export interface HogOptions {
  /** Orientation bins over [0, 180) degrees (default 9). */
  readonly orientations?: number
  /** Cell size [rows, cols] (default [8, 8]). */
  readonly pixelsPerCell?: readonly [number, number]
  /** Block size in cells (default [3, 3]). */
  readonly cellsPerBlock?: readonly [number, number]
  /** Block normalisation (default 'L2-Hys'). */
  readonly blockNorm?: 'L1' | 'L1-sqrt' | 'L2' | 'L2-Hys'
  /** Square-root gamma compression first (default false). */
  readonly transformSqrt?: boolean
}

/**
 * Histograms of oriented gradients (Dalal and Triggs 2005, skimage.feature.hog): central-difference gradients,
 * per-cell histograms of unsigned orientation weighted by magnitude, and overlapping blocks of cells normalised.
 *
 * @param x - Image `[H, W]`.
 * @param options - Bins, cell and block sizes, normalisation and gamma.
 * @returns The feature vector (blocks row-major, then cells, then bins).
 * @concept histogram-of-oriented-gradients
 * @example
 * ```ts
 * import { image } from 'mdml-engine/signal'
 * const x = Array.from({ length: 16 }, (_, i) => Array.from({ length: 16 }, (_, j) => Math.sin(i / 2) + j / 8))
 * image.hog(x, { pixelsPerCell: [4, 4], cellsPerBlock: [2, 2] }).shape // => [324]
 * ```
 */
export function hog(x: TensorLike, options: HogOptions = {}): Tensor {
  const p0 = planeOf(x, 'hog')
  const p = options.transformSqrt === true ? { ...p0, data: Float64Array.from(p0.data, Math.sqrt) } : p0
  const { h, w } = p
  const gr = new Float64Array(h * w)
  const gc = new Float64Array(h * w)
  for (let r = 1; r < h - 1; r++)
    for (let c = 0; c < w; c++) gr[r * w + c] = (p.data[(r + 1) * w + c] ?? 0) - (p.data[(r - 1) * w + c] ?? 0)
  for (let r = 0; r < h; r++)
    for (let c = 1; c < w - 1; c++) gc[r * w + c] = (p.data[r * w + c + 1] ?? 0) - (p.data[r * w + c - 1] ?? 0)
  const nb = options.orientations ?? 9
  const [cr, cc] = options.pixelsPerCell ?? [8, 8]
  const [br, bc] = options.cellsPerBlock ?? [3, 3]
  const ncr = Math.floor(h / cr)
  const ncc = Math.floor(w / cc)
  const hist = new Float64Array(ncr * ncc * nb)
  const width = 180 / nb
  for (let r = 0; r < ncr * cr; r++)
    for (let c = 0; c < ncc * cc; c++) {
      const a = gr[r * w + c] ?? 0
      const b = gc[r * w + c] ?? 0
      const mag = Math.hypot(b, a)
      let o = ((Math.atan2(a, b) * 180) / Math.PI) % 180
      if (o < 0) o += 180
      const bin = Math.min(nb - 1, Math.floor(o / width))
      // scikit-image's half-open bins [i w, (i + 1) w): a value on a bin edge belongs to the upper bin.
      const k = (Math.floor(r / cr) * ncc + Math.floor(c / cc)) * nb + bin
      hist[k] = (hist[k] ?? 0) + mag / (cr * cc)
    }
  const nbr = ncr - br + 1
  const nbc = ncc - bc + 1
  if (nbr <= 0 || nbc <= 0) throw new MdmlError('value', 'hog: the image is smaller than a block')
  const method = options.blockNorm ?? 'L2-Hys'
  const eps = 1e-5
  const blockLen = br * bc * nb
  const out = new Float64Array(nbr * nbc * blockLen)
  for (let r = 0; r < nbr; r++)
    for (let c = 0; c < nbc; c++) {
      const block = new Float64Array(blockLen)
      let k = 0
      for (let i = 0; i < br; i++)
        for (let j = 0; j < bc; j++)
          for (let o = 0; o < nb; o++) block[k++] = hist[((r + i) * ncc + c + j) * nb + o] ?? 0
      let norm: Float64Array
      if (method === 'L1' || method === 'L1-sqrt') {
        let s = 0
        for (const v of block) s += Math.abs(v)
        norm = Float64Array.from(block, (v) => (method === 'L1' ? v / (s + eps) : Math.sqrt(v / (s + eps))))
      } else {
        let s = 0
        for (const v of block) s += v * v
        norm = Float64Array.from(block, (v) => v / Math.sqrt(s + eps * eps))
        if (method === 'L2-Hys') {
          norm = Float64Array.from(norm, (v) => Math.min(v, 0.2))
          let s2 = 0
          for (const v of norm) s2 += v * v
          norm = Float64Array.from(norm, (v) => v / Math.sqrt(s2 + eps * eps))
        }
      }
      out.set(norm, (r * nbc + c) * blockLen)
    }
  return fromStorage(out, [out.length], 'float64')
}

/**
 * The straight-line Hough transform (skimage.transform.hough_line): votes of every nonzero pixel for the lines
 * x cos(theta) + y sin(theta) = rho through it.
 *
 * @param x - Binary edge image `[H, W]`.
 * @param thetas - Angles (default 180 angles from -pi/2 inclusive to pi/2 exclusive).
 * @returns The accumulator `[distances, thetas]`, the angles and the distances.
 * @concept hough-transform
 * @example
 * ```ts
 * import { image } from 'mdml-engine/signal'
 * const x = Array.from({ length: 5 }, (_, i) => Array.from({ length: 5 }, (_, j) => (j === 2 ? 1 : 0)))
 * image.houghLine(x).accumulator.shape // => [17, 180]
 * ```
 */
export function houghLine(
  x: TensorLike,
  thetas?: readonly number[],
): { readonly accumulator: Tensor; readonly thetas: Float64Array; readonly distances: Float64Array } {
  const p = planeOf(x, 'houghLine')
  const th =
    thetas === undefined
      ? Float64Array.from({ length: 180 }, (_, i) => -Math.PI / 2 + (Math.PI * i) / 180)
      : Float64Array.from(thetas)
  const offset = Math.ceil(Math.sqrt(p.h * p.h + p.w * p.w))
  const nd = 2 * offset + 1
  const distances = Float64Array.from({ length: nd }, (_, i) => -offset + i)
  const acc = new Float64Array(nd * th.length)
  const cs = Float64Array.from(th, Math.cos)
  const sn = Float64Array.from(th, Math.sin)
  // C's round: half away from zero.
  const round = (v: number): number => (v < 0 ? -Math.round(-v) : Math.round(v))
  for (let y = 0; y < p.h; y++)
    for (let xx = 0; xx < p.w; xx++) {
      if ((p.data[y * p.w + xx] ?? 0) === 0) continue
      for (let j = 0; j < th.length; j++) {
        const idx = round((cs[j] ?? 0) * xx + (sn[j] ?? 0) * y) + offset
        acc[idx * th.length + j] = (acc[idx * th.length + j] ?? 0) + 1
      }
    }
  return { accumulator: fromStorage(acc, [nd, th.length], 'float64'), thetas: th, distances }
}

/** The points of a Bresenham circle of radius r about the origin (skimage.draw.circle_perimeter, with repeats). */
function circlePerimeter(radius: number): [number[], number[]] {
  const rr: number[] = []
  const cc: number[] = []
  let c = 0
  let r = radius
  let d = 3 - 2 * radius
  while (r >= c) {
    rr.push(r, -r, r, -r, c, -c, c, -c)
    cc.push(c, c, -c, -c, r, r, -r, -r)
    if (d < 0) d += 4 * c + 6
    else {
      d += 4 * (c - r) + 10
      r -= 1
    }
    c += 1
  }
  return [rr, cc]
}

/**
 * The circular Hough transform (skimage.transform.hough_circle): for each radius, every nonzero pixel votes for the
 * centres on a Bresenham circle around it; normalised by the circle's pixel count.
 *
 * @param x - Binary edge image `[H, W]`.
 * @param radii - Radii.
 * @param normalize - Divide by the number of perimeter pixels (default true).
 * @returns Accumulators `[radii, H, W]`.
 * @concept hough-transform
 * @example
 * ```ts
 * import { image } from 'mdml-engine/signal'
 * image.houghCircle([[0, 1, 0], [1, 0, 1], [0, 1, 0]], [1]).shape // => [1, 3, 3]
 * ```
 */
export function houghCircle(x: TensorLike, radii: readonly number[], normalize = true): Tensor {
  const p = planeOf(x, 'houghCircle')
  const out = new Float64Array(radii.length * p.h * p.w)
  const ys: number[] = []
  const xs: number[] = []
  for (let y = 0; y < p.h; y++)
    for (let xx = 0; xx < p.w; xx++)
      if ((p.data[y * p.w + xx] ?? 0) !== 0) {
        ys.push(y)
        xs.push(xx)
      }
  for (const [k, radius] of radii.entries()) {
    const [rr, cc] = circlePerimeter(radius)
    const inc = normalize ? 1 / rr.length : 1
    const base = k * p.h * p.w
    for (let i = 0; i < ys.length; i++)
      for (let q = 0; q < rr.length; q++) {
        const r = (ys[i] ?? 0) + (rr[q] ?? 0)
        const c = (xs[i] ?? 0) + (cc[q] ?? 0)
        if (r >= 0 && r < p.h && c >= 0 && c < p.w) out[base + r * p.w + c] = (out[base + r * p.w + c] ?? 0) + inc
      }
  }
  return fromStorage(out, [radii.length, p.h, p.w], 'float64')
}
