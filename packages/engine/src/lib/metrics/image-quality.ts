/**
 * Image quality metrics comparing a distorted image with a reference, skimage and sewar style: MSE and PSNR, structural
 * similarity (SSIM) and its multi-scale form, visual information fidelity, the universal quality index, the spectral
 * angle mapper, ERGAS and RASE for multispectral images, and LPIPS and DISTS from supplied network features.
 */

import { MdmlError, type MetricResult, type MetricSpec } from '../base/index.ts'
import { convNd } from '../numerics/convolution/index.ts'
import { image as filters } from '../signal/index.ts'
import { fromStorage, primalOf, reshape, type TensorLike, type Value } from '../tensor/index.ts'
import { defineMetric, type MetricFunction, metricResult, shapedArray } from './metric.ts'

// Images are handled as channel planes; the filtering itself is signal's (scipy.ndimage semantics).

/** A 2-D plane of values. */
interface Plane {
  readonly h: number
  readonly w: number
  readonly data: Float64Array
}

/** The channel planes of an image `[H, W]` or `[H, W, C]` (channels last). */
function planes(name: string, image: TensorLike): Plane[] {
  const m = shapedArray(image)
  const [h = 0, w = 0, c = 1] = m.shape
  if (m.shape.length < 2 || m.shape.length > 3) throw new MdmlError('shape', `${name}: images are [H, W] or [H, W, C]`)
  return Array.from({ length: c }, (_, k) => ({
    h,
    w,
    data: Float64Array.from({ length: h * w }, (_, i) => m.data[i * c + k] ?? 0),
  }))
}

/** Planes of two images, checked to have the same shape. */
function pairOfPlanes(name: string, a: TensorLike, b: TensorLike): [Plane[], Plane[]] {
  const pa = planes(name, a)
  const pb = planes(name, b)
  if (pa.length !== pb.length || pa[0]?.h !== pb[0]?.h || pa[0]?.w !== pb[0]?.w)
    throw new MdmlError('shape', `${name}: images differ in shape`)
  return [pa, pb]
}

/** Elementwise combination of planes. */
function zip(a: Plane, b: Plane, f: (x: number, y: number) => number): Plane {
  return { h: a.h, w: a.w, data: Float64Array.from(a.data, (x, i) => f(x, b.data[i] ?? 0)) }
}

/** Apply a tensor filter to a plane. */
function filterPlane(p: Plane, f: (t: Value) => Value): Plane {
  const out = primalOf(f(fromStorage(p.data, [p.h, p.w], 'float64')))
  return { h: out.shape[0] ?? 0, w: out.shape[1] ?? 0, data: Float64Array.from(out.data, Number) }
}

/** Valid 2-D correlation with a kernel `[k][k]` (scipy.signal.convolve2d with the kernel rotated, mode 'valid'). */
function correlateValid(p: Plane, kernel: readonly (readonly number[])[]): Plane {
  return filterPlane(p, (t) => {
    const kh = kernel.length
    const kw = kernel[0]?.length ?? 0
    const w = fromStorage(Float64Array.from(kernel.flat()), [1, 1, kh, kw], 'float64')
    const r = convNd(reshape(t, [1, 1, p.h, p.w]), w, { padding: 'valid' })
    return reshape(r, r.shape.slice(2))
  })
}

/** A normalised Gaussian window `[n][n]` as MATLAB's fspecial (sewar), tiny weights zeroed. */
function gaussianWindow(n: number, sigma: number): number[][] {
  // numpy.mgrid[-n//2 + 1 : n//2 + 1]: floor division, so even sizes run from -n/2 + 1 to n/2.
  const coords = Array.from({ length: n }, (_, i) => Math.floor(-n / 2) + 1 + i)
  const g = coords.map((y) => coords.map((x) => Math.exp(-(x * x + y * y) / (2 * sigma * sigma))))
  let max = 0
  for (const row of g) for (const v of row) max = Math.max(max, v)
  let total = 0
  for (const row of g)
    for (const [j, v] of row.entries()) {
      const kept = v < Number.EPSILON * max ? 0 : v
      row[j] = kept
      total += kept
    }
  return g.map((row) => row.map((v) => v / total))
}

/** The mean of a plane after cropping `s` cells from every border. */
function croppedMean(p: Plane, s: number): number {
  let sum = 0
  let n = 0
  for (let y = s; y < p.h - s; y++) {
    for (let x = s; x < p.w - s; x++) {
      sum += p.data[y * p.w + x] ?? 0
      n++
    }
  }
  return sum / n
}

/** The mean of a plane. */
function planeMean(p: Plane): number {
  return p.data.reduce((a, b) => a + b, 0) / p.data.length
}

/** Every other row and column (x[::2, ::2]). */
function decimate(p: Plane): Plane {
  const h = Math.ceil(p.h / 2)
  const w = Math.ceil(p.w / 2)
  return {
    h,
    w,
    data: Float64Array.from({ length: h * w }, (_, i) => p.data[2 * Math.floor(i / w) * p.w + 2 * (i % w)] ?? 0),
  }
}

/** scipy.ndimage.uniform_filter of a plane (reflect borders). */
const uniformFilter = (p: Plane, size: number): Plane => filterPlane(p, (t) => filters.uniformFilter(t, size))

const spec = (name: string, direction: 'higher' | 'lower', range: readonly [number, number]): MetricSpec => ({
  name,
  direction,
  range,
  inputs: 'values',
  category: 'image-quality',
  concept: 'image-quality',
})

const mean = (xs: readonly number[]): number => xs.reduce((a, b) => a + b, 0) / xs.length

/** The value range of an image (max - min), the default data range. */
function rangeOf(image: TensorLike): number {
  const d = shapedArray(image).data
  let lo = Number.POSITIVE_INFINITY
  let hi = Number.NEGATIVE_INFINITY
  for (const v of d) {
    lo = Math.min(lo, v)
    hi = Math.max(hi, v)
  }
  return hi - lo
}

const mseOf = (a: TensorLike, b: TensorLike): number => {
  const x = shapedArray(a).data
  const y = shapedArray(b).data
  if (x.length !== y.length) throw new MdmlError('shape', 'images differ in size')
  let s = 0
  for (const [i, v] of x.entries()) s += (v - (y[i] ?? 0)) ** 2
  return s / x.length
}

/**
 * Mean squared error between two images (skimage.metrics.mean_squared_error), over every pixel and channel.
 *
 * @param imageTrue - Reference image `[H, W]` or `[H, W, C]`.
 * @param imageTest - Test image, the same shape.
 * @returns The metric result.
 * @concept image-quality
 * @example
 * ```ts
 * import { imageMeanSquaredError } from 'mdml-engine/metrics'
 * imageMeanSquaredError([[0, 0], [0, 0]], [[1, 1], [1, 1]]).estimate.value // => 1
 * ```
 */
export const imageMeanSquaredError: MetricFunction<[TensorLike, TensorLike]> = defineMetric(
  spec('imageMeanSquaredError', 'lower', [0, Infinity]),
  (imageTrue: TensorLike, imageTest: TensorLike): MetricResult =>
    metricResult('imageMeanSquaredError', mseOf(imageTrue, imageTest), shapedArray(imageTrue).data.length),
)

/**
 * Peak signal-to-noise ratio 10 log10(R^2 / MSE) in decibels (skimage.metrics.peak_signal_noise_ratio).
 *
 * @param imageTrue - Reference image.
 * @param imageTest - Test image.
 * @param options - dataRange R (default the reference's max - min).
 * @returns The metric result (Infinity for identical images).
 * @example
 * ```ts
 * import { peakSignalNoiseRatio } from 'mdml-engine/metrics'
 * peakSignalNoiseRatio([[0, 1]], [[0.1, 0.9]], { dataRange: 1 }).estimate.value.toFixed(4) // => '20.0000'
 * ```
 */
export const peakSignalNoiseRatio: MetricFunction<[TensorLike, TensorLike, { readonly dataRange?: number }?]> =
  defineMetric(
    spec('peakSignalNoiseRatio', 'higher', [0, Infinity]),
    (imageTrue: TensorLike, imageTest: TensorLike, options: { readonly dataRange?: number } = {}): MetricResult => {
      const r = options.dataRange ?? rangeOf(imageTrue)
      const mse = mseOf(imageTrue, imageTest)
      return metricResult(
        'peakSignalNoiseRatio',
        mse === 0 ? Number.POSITIVE_INFINITY : 10 * Math.log10((r * r) / mse),
        1,
      )
    },
  )

/** Options of {@link structuralSimilarity}. */
export interface SsimOptions {
  /** The data range R (default the reference's max - min). */
  readonly dataRange?: number
  /** Gaussian weights (sigma, truncated at 3.5 sigma) instead of a uniform window (default false, as skimage). */
  readonly gaussianWeights?: boolean
  /** The Gaussian's standard deviation (default 1.5). */
  readonly sigma?: number
  /** The uniform window's side (default 7). */
  readonly winSize?: number
  readonly k1?: number
  readonly k2?: number
  /** Normalise covariances by N - 1 (default true, as skimage). */
  readonly useSampleCovariance?: boolean
}

/** skimage's SSIM map of one channel, and the crop. */
function ssimPlane(x: Plane, y: Plane, o: SsimOptions, r: number): number {
  const gaussian = o.gaussianWeights === true
  const sigma = o.sigma ?? 1.5
  const truncate = 3.5
  const win = gaussian ? 2 * Math.floor(truncate * sigma + 0.5) + 1 : (o.winSize ?? 7)
  const filter = (p: Plane): Plane =>
    gaussian ? filterPlane(p, (t) => filters.gaussianFilter(t, sigma, { truncate })) : uniformFilter(p, win)
  const np = win * win
  const covNorm = (o.useSampleCovariance ?? true) ? np / (np - 1) : 1
  const ux = filter(x)
  const uy = filter(y)
  const uxx = filter(zip(x, x, (a, b) => a * b))
  const uyy = filter(zip(y, y, (a, b) => a * b))
  const uxy = filter(zip(x, y, (a, b) => a * b))
  const c1 = ((o.k1 ?? 0.01) * r) ** 2
  const c2 = ((o.k2 ?? 0.03) * r) ** 2
  const s = Float64Array.from(ux.data, (mx, i) => {
    const my = uy.data[i] ?? 0
    const vx = covNorm * ((uxx.data[i] ?? 0) - mx * mx)
    const vy = covNorm * ((uyy.data[i] ?? 0) - my * my)
    const vxy = covNorm * ((uxy.data[i] ?? 0) - mx * my)
    return ((2 * mx * my + c1) * (2 * vxy + c2)) / ((mx * mx + my * my + c1) * (vx + vy + c2))
  })
  return croppedMean({ h: x.h, w: x.w, data: s }, (win - 1) >> 1)
}

/**
 * The structural similarity index (Wang et al., 2004; skimage.metrics.structural_similarity): the mean over windows
 * of luminance, contrast and structure agreement, with skimage's uniform 7 x 7 window (or Gaussian weights), sample
 * covariances and edge crop; channels (last axis) are averaged.
 *
 * @param imageTrue - Reference image `[H, W]` or `[H, W, C]`.
 * @param imageTest - Test image.
 * @param options - dataRange, gaussianWeights, sigma, winSize, k1, k2 and useSampleCovariance.
 * @returns The metric result, in [-1, 1].
 * @example
 * ```ts
 * import { structuralSimilarity } from 'mdml-engine/metrics'
 * const img = Array.from({ length: 8 }, (_, i) => Array.from({ length: 8 }, (_, j) => (i * j) % 5))
 * Number(structuralSimilarity(img, img).estimate.value.toFixed(12)) // => 1
 * ```
 */
export const structuralSimilarity: MetricFunction<[TensorLike, TensorLike, SsimOptions?]> = defineMetric(
  spec('structuralSimilarity', 'higher', [-1, 1]),
  (imageTrue: TensorLike, imageTest: TensorLike, options: SsimOptions = {}): MetricResult => {
    const [a, b] = pairOfPlanes('structuralSimilarity', imageTrue, imageTest)
    const r = options.dataRange ?? rangeOf(imageTrue)
    const values = a.map((p, c) => ssimPlane(p, b[c] as Plane, options, r))
    return metricResult('structuralSimilarity', mean(values), (a[0]?.h ?? 0) * (a[0]?.w ?? 0))
  },
)

/** The mean SSIM and contrast-structure of one plane with a Gaussian window and valid filtering (sewar's ssim). */
function ssimValid(
  x: Plane,
  y: Plane,
  window: readonly (readonly number[])[],
  c1: number,
  c2: number,
): { ssim: number; cs: number } {
  const ux = correlateValid(x, window)
  const uy = correlateValid(y, window)
  const sxx = correlateValid(
    zip(x, x, (a, b) => a * b),
    window,
  )
  const syy = correlateValid(
    zip(y, y, (a, b) => a * b),
    window,
  )
  const sxy = correlateValid(
    zip(x, y, (a, b) => a * b),
    window,
  )
  let ssim = 0
  let cs = 0
  for (const [i, mx] of ux.data.entries()) {
    const my = uy.data[i] ?? 0
    const vx = (sxx.data[i] ?? 0) - mx * mx
    const vy = (syy.data[i] ?? 0) - my * my
    const vxy = (sxy.data[i] ?? 0) - mx * my
    const contrast = (2 * vxy + c2) / (vx + vy + c2)
    ssim += ((2 * mx * my + c1) / (mx * mx + my * my + c1)) * contrast
    cs += contrast
  }
  return { ssim: ssim / ux.data.length, cs: cs / ux.data.length }
}

/** Real part of a product of real bases raised to real powers, negative bases taken through the complex logarithm. */
function complexPowerProduct(bases: readonly number[], powers: readonly number[]): number {
  let mag = 1
  let angle = 0
  for (const [i, b] of bases.entries()) {
    const p = powers[i] ?? 0
    mag *= Math.abs(b) ** p
    if (b < 0) angle += Math.PI * p
  }
  return mag * Math.cos(angle)
}

/**
 * Multi-scale SSIM (Wang, Simoncelli and Bovik, 2003; sewar's msssim): contrast-structure at each of up to five
 * dyadic scales and the full SSIM at the coarsest, combined with the standard weights
 * (0.0448, 0.2856, 0.3001, 0.2363, 0.1333); scales are capped so the image stays at least the window size.
 *
 * @param imageTrue - Reference image `[H, W]` (or `[H, W, C]`, channels averaged).
 * @param imageTest - Test image.
 * @param options - dataRange (default 1, sewar's for float images), winSize (default 11), weights, k1, k2.
 * @returns The metric result.
 * @example
 * ```ts
 * import { multiScaleSsim } from 'mdml-engine/metrics'
 * const img = Array.from({ length: 24 }, (_, i) => Array.from({ length: 24 }, (_, j) => Math.sin(i * j)))
 * Number(multiScaleSsim(img, img).estimate.value.toFixed(12)) // => 1
 * ```
 */
export const multiScaleSsim: MetricFunction<
  [
    TensorLike,
    TensorLike,
    {
      readonly dataRange?: number
      readonly winSize?: number
      readonly weights?: readonly number[]
      readonly k1?: number
      readonly k2?: number
    }?,
  ]
> = defineMetric(
  spec('multiScaleSsim', 'higher', [0, 1]),
  (
    imageTrue: TensorLike,
    imageTest: TensorLike,
    options: {
      readonly dataRange?: number
      readonly winSize?: number
      readonly weights?: readonly number[]
      readonly k1?: number
      readonly k2?: number
    } = {},
  ): MetricResult => {
    const [a, b] = pairOfPlanes('multiScaleSsim', imageTrue, imageTest)
    const ws = options.winSize ?? 11
    let weights = options.weights ?? [0.0448, 0.2856, 0.3001, 0.2363, 0.1333]
    const minDim = Math.min(a[0]?.h ?? 0, a[0]?.w ?? 0)
    if (minDim < ws) throw new MdmlError('shape', `multiScaleSsim: images must be at least ${ws} pixels on a side`)
    const maxScales = Math.floor(Math.log2(minDim / ws)) + 1
    if (weights.length > maxScales) weights = weights.slice(0, maxScales)
    const r = options.dataRange ?? 1
    const c1 = ((options.k1 ?? 0.01) * r) ** 2
    const c2 = ((options.k2 ?? 0.03) * r) ** 2
    const window = gaussianWindow(ws, 1.5)
    const values = a.map((p0, ch) => {
      let x = p0
      let y = b[ch] as Plane
      const ssims: number[] = []
      const css: number[] = []
      for (let s = 0; s < weights.length; s++) {
        const v = ssimValid(x, y, window, c1, c2)
        ssims.push(v.ssim)
        css.push(v.cs)
        x = decimate(uniformFilter(x, 2))
        y = decimate(uniformFilter(y, 2))
      }
      const last = weights.length - 1
      return complexPowerProduct(
        [...css.slice(0, last), ssims[last] ?? 0],
        [...weights.slice(0, last), weights[last] ?? 0],
      )
    })
    return metricResult('multiScaleSsim', mean(values), (a[0]?.h ?? 0) * (a[0]?.w ?? 0))
  },
)

/**
 * Pixel-domain visual information fidelity (Sheikh and Bovik, 2006; sewar's vifp): the information the distorted
 * image preserves about the reference over four scales of Gaussian-filtered local statistics, relative to the
 * information in the reference, with visual noise variance sigma_n^2.
 *
 * @param imageTrue - Reference image `[H, W]` (or `[H, W, C]`, channels averaged), large enough for four scales (about 48 x 48).
 * @param imageTest - Distorted image.
 * @param options - sigmaNsq (default 2).
 * @returns The metric result (1 for identical images, can exceed 1 for enhanced images).
 * @example
 * ```ts
 * import { visualInformationFidelity } from 'mdml-engine/metrics'
 * const img = Array.from({ length: 48 }, (_, i) => Array.from({ length: 48 }, (_, j) => 100 * Math.sin(i / 3) * Math.cos(j / 4)))
 * Number(visualInformationFidelity(img, img).estimate.value.toFixed(10)) // => 1
 * ```
 */
export const visualInformationFidelity: MetricFunction<[TensorLike, TensorLike, { readonly sigmaNsq?: number }?]> =
  defineMetric(
    spec('visualInformationFidelity', 'higher', [0, Infinity]),
    (imageTrue: TensorLike, imageTest: TensorLike, options: { readonly sigmaNsq?: number } = {}): MetricResult => {
      const [a, b] = pairOfPlanes('visualInformationFidelity', imageTrue, imageTest)
      const noise = options.sigmaNsq ?? 2
      const eps = 1e-10
      const values = a.map((p0, ch) => {
        let x = p0
        let y = b[ch] as Plane
        let num = 0
        let den = 0
        for (let scale = 1; scale <= 4; scale++) {
          const n = 2 ** (4 - scale + 1) + 1
          const win = gaussianWindow(n, n / 5)
          if (scale > 1) {
            x = decimate(correlateValid(x, win))
            y = decimate(correlateValid(y, win))
          }
          const mx = correlateValid(x, win)
          const my = correlateValid(y, win)
          const sxx = correlateValid(
            zip(x, x, (u, v) => u * v),
            win,
          )
          const syy = correlateValid(
            zip(y, y, (u, v) => u * v),
            win,
          )
          const sxy = correlateValid(
            zip(x, y, (u, v) => u * v),
            win,
          )
          for (const [i, ux] of mx.data.entries()) {
            const uy = my.data[i] ?? 0
            let sigX = Math.max((sxx.data[i] ?? 0) - ux * ux, 0)
            const sigY = Math.max((syy.data[i] ?? 0) - uy * uy, 0)
            const sigXY = (sxy.data[i] ?? 0) - ux * uy
            let g = sigXY / (sigX + eps)
            let sv = sigY - g * sigXY
            if (sigX < eps) {
              g = 0
              sv = sigY
              sigX = 0
            }
            if (sigY < eps) {
              g = 0
              sv = 0
            }
            if (g < 0) {
              sv = sigY
              g = 0
            }
            if (sv <= eps) sv = eps
            num += Math.log10(1 + (g * g * sigX) / (sv + noise))
            den += Math.log10(1 + sigX / noise)
          }
        }
        return num / den
      })
      return metricResult('visualInformationFidelity', mean(values), (a[0]?.h ?? 0) * (a[0]?.w ?? 0))
    },
  )

/**
 * The universal image quality index (Wang and Bovik, 2002; torchmetrics' universal_image_quality_index): SSIM without
 * stabilising constants, 4 sigma_xy mu_x mu_y / ((sigma_x^2 + sigma_y^2)(mu_x^2 + mu_y^2)) over an 11 x 11 Gaussian
 * window (sigma 1.5) with mirrored borders, edges cropped.
 *
 * @param imageTrue - Reference image `[H, W]` or `[H, W, C]`.
 * @param imageTest - Test image.
 * @param options - kernelSize (odd, default 11) and sigma (default 1.5).
 * @returns The metric result, in [-1, 1].
 * @example
 * ```ts
 * import { universalImageQualityIndex } from 'mdml-engine/metrics'
 * const img = Array.from({ length: 16 }, (_, i) => Array.from({ length: 16 }, (_, j) => 1 + ((i + 2 * j) % 7)))
 * Number(universalImageQualityIndex(img, img).estimate.value.toFixed(10)) // => 1
 * ```
 */
export const universalImageQualityIndex: MetricFunction<
  [TensorLike, TensorLike, { readonly kernelSize?: number; readonly sigma?: number }?]
> = defineMetric(
  spec('universalImageQualityIndex', 'higher', [-1, 1]),
  (
    imageTrue: TensorLike,
    imageTest: TensorLike,
    options: { readonly kernelSize?: number; readonly sigma?: number } = {},
  ): MetricResult => {
    const [a, b] = pairOfPlanes('universalImageQualityIndex', imageTest, imageTrue)
    const k = options.kernelSize ?? 11
    const sigma = options.sigma ?? 1.5
    const raw = Array.from({ length: k }, (_, i) => Math.exp(-0.5 * ((i - (k - 1) / 2) / sigma) ** 2))
    const total = raw.reduce((s, v) => s + v, 0)
    const kernel = raw.map((v) => v / total)
    const pad = (k - 1) >> 1
    // numpy 'reflect' borders (ndimage 'mirror'); the window starts pad samples before each pixel.
    const border = { mode: 'mirror', origin: pad - Math.floor(k / 2) } as const
    const filter = (p: Plane): Plane =>
      filterPlane(p, (t) => filters.correlate1d(filters.correlate1d(t, kernel, 1, border), kernel, 0, border))
    const values: number[] = []
    for (const [ch, p] of a.entries()) {
      const t = b[ch] as Plane
      const mp = filter(p)
      const mt = filter(t)
      const spp = filter(zip(p, p, (u, v) => u * v))
      const stt = filter(zip(t, t, (u, v) => u * v))
      const spt = filter(zip(p, t, (u, v) => u * v))
      const q = Float64Array.from(mp.data, (m1, i) => {
        const m2 = mt.data[i] ?? 0
        const vp = Math.max((spp.data[i] ?? 0) - m1 * m1, 0)
        const vt = Math.max((stt.data[i] ?? 0) - m2 * m2, 0)
        const cov = (spt.data[i] ?? 0) - m1 * m2
        return (2 * m1 * m2 * (2 * cov)) / ((m1 * m1 + m2 * m2) * (vp + vt) + Number.EPSILON)
      })
      values.push(croppedMean({ h: p.h, w: p.w, data: q }, pad))
    }
    return metricResult('universalImageQualityIndex', mean(values), (a[0]?.h ?? 0) * (a[0]?.w ?? 0))
  },
)

/** Spectral vectors `[pixels][channels]` of an `[H, W, C]` image. */
function spectra(image: TensorLike): { n: number; c: number; data: Float64Array } {
  const m = shapedArray(image)
  const c = m.shape.length === 3 ? (m.shape[2] ?? 1) : 1
  return { n: m.data.length / c, c, data: m.data }
}

/**
 * The spectral angle mapper (Kruse et al., 1993; sewar's sam): the mean angle in radians between the spectra of
 * corresponding pixels of two multispectral images; pixels with a zero spectrum are skipped.
 *
 * @param imageTrue - Reference image `[H, W, C]`.
 * @param imageTest - Test image `[H, W, C]`.
 * @returns The metric result.
 * @example
 * ```ts
 * import { spectralAngleMapper } from 'mdml-engine/metrics'
 * spectralAngleMapper([[[1, 0]]], [[[0, 1]]]).estimate.value === Math.PI / 2 // => true
 * ```
 */
export const spectralAngleMapper: MetricFunction<[TensorLike, TensorLike]> = defineMetric(
  spec('spectralAngleMapper', 'lower', [0, Math.PI]),
  (imageTrue: TensorLike, imageTest: TensorLike): MetricResult => {
    const a = spectra(imageTrue)
    const b = spectra(imageTest)
    const angles: number[] = []
    for (let i = 0; i < a.n; i++) {
      let dot = 0
      let na = 0
      let nb = 0
      for (let k = 0; k < a.c; k++) {
        const x = a.data[i * a.c + k] ?? 0
        const y = b.data[i * a.c + k] ?? 0
        dot += x * y
        na += x * x
        nb += y * y
      }
      const norm = Math.sqrt(na) * Math.sqrt(nb)
      if (norm > 0) angles.push(Math.acos(Math.min(1, Math.max(-1, dot / norm))))
    }
    return metricResult('spectralAngleMapper', angles.length === 0 ? Number.NaN : mean(angles), a.n)
  },
)

/**
 * ERGAS, the relative dimensionless global error in synthesis (Wald, 2000; torchmetrics and sewar):
 * (100 / ratio) sqrt(mean_c (RMSE_c / mu_c)^2) over the bands of a pansharpened image, with ratio the resolution
 * ratio of the high- and low-resolution images.
 *
 * @param imageTrue - Reference image `[H, W, C]`.
 * @param imageTest - Fused image `[H, W, C]`.
 * @param options - ratio (default 4).
 * @returns The metric result.
 * @example
 * ```ts
 * import { ergas } from 'mdml-engine/metrics'
 * ergas([[[2]]], [[[1]]], { ratio: 1 }).estimate.value // => 50
 * ```
 */
export const ergas: MetricFunction<[TensorLike, TensorLike, { readonly ratio?: number }?]> = defineMetric(
  spec('ergas', 'lower', [0, Infinity]),
  (imageTrue: TensorLike, imageTest: TensorLike, options: { readonly ratio?: number } = {}): MetricResult => {
    const [a, b] = pairOfPlanes('ergas', imageTrue, imageTest)
    const terms = a.map((p, c) => {
      const q = b[c] as Plane
      const rmse = Math.sqrt(planeMean(zip(p, q, (x, y) => (x - y) ** 2)))
      return (rmse / planeMean(p)) ** 2
    })
    return metricResult('ergas', (100 / (options.ratio ?? 4)) * Math.sqrt(mean(terms)), (a[0]?.h ?? 0) * (a[0]?.w ?? 0))
  },
)

/**
 * RASE, the relative average spectral error (Choi, 2006; sewar's rase): 100 / M sqrt(mean_c RMSE_c^2) from sliding
 * windows (uniform, reflect borders) of the squared errors and the band means, averaged over the image without its
 * borders.
 *
 * @param imageTrue - Reference image `[H, W, C]`.
 * @param imageTest - Test image `[H, W, C]`.
 * @param options - windowSize (default 8).
 * @returns The metric result.
 * @example
 * ```ts
 * import { relativeAverageSpectralError } from 'mdml-engine/metrics'
 * const img = Array.from({ length: 12 }, () => Array.from({ length: 12 }, () => [2]))
 * relativeAverageSpectralError(img, img).estimate.value // => 0
 * ```
 */
export const relativeAverageSpectralError: MetricFunction<[TensorLike, TensorLike, { readonly windowSize?: number }?]> =
  defineMetric(
    spec('relativeAverageSpectralError', 'lower', [0, Infinity]),
    (imageTrue: TensorLike, imageTest: TensorLike, options: { readonly windowSize?: number } = {}): MetricResult => {
      const [a, b] = pairOfPlanes('relativeAverageSpectralError', imageTrue, imageTest)
      const ws = options.windowSize ?? 8
      const n = a.length
      const h = a[0]?.h ?? 0
      const w = a[0]?.w ?? 0
      const sq = new Float64Array(h * w)
      const meanTrue = new Float64Array(h * w)
      for (const [c, p] of a.entries()) {
        const err = uniformFilter(
          zip(p, b[c] as Plane, (x, y) => (x - y) ** 2),
          ws,
        )
        const m = uniformFilter(p, ws)
        for (let i = 0; i < h * w; i++) {
          sq[i] = (sq[i] ?? 0) + (err.data[i] ?? 0)
          meanTrue[i] = (meanTrue[i] ?? 0) + (m.data[i] ?? 0) / n
        }
      }
      const map = Float64Array.from(sq, (s, i) => {
        const m = meanTrue[i] ?? 0
        return m === 0 ? 0 : (100 / m) * Math.sqrt(s / n)
      })
      // numpy's round half to even of ws / 2.
      const half = ws / 2
      const s = half % 1 === 0.5 ? 2 * Math.round(half / 2) : Math.round(half)
      return metricResult('relativeAverageSpectralError', croppedMean({ h, w, data: map }, s), h * w)
    },
  )

/** Feature maps of one image: one `[H, W, C]` map per layer (channels last). */
export type FeatureStack = readonly TensorLike[]

/**
 * LPIPS from supplied features (Zhang et al., 2018): sum over layers of the spatial mean of
 * sum_c w_lc (xhat - yhat)^2, with features unit-normalised over channels. mdml ships no network; pass the
 * activations of AlexNet, VGG or any extractor and its learned linear weights (or ones for the unweighted variant).
 *
 * @param featuresX - Feature maps of the first image, one `[H, W, C]` per layer.
 * @param featuresY - Feature maps of the second image.
 * @param weights - Non-negative channel weights per layer `[C_l]` (default all ones).
 * @returns The metric result (0 for identical features).
 * @example
 * ```ts
 * import { lpipsFromFeatures } from 'mdml-engine/metrics'
 * Number(lpipsFromFeatures([[[[1, 0]]]], [[[[0, 1]]]]).estimate.value.toFixed(6)) // => 2
 * ```
 */
export const lpipsFromFeatures: MetricFunction<[FeatureStack, FeatureStack, (readonly TensorLike[])?]> = defineMetric(
  spec('lpipsFromFeatures', 'lower', [0, Infinity]),
  (featuresX: FeatureStack, featuresY: FeatureStack, weights?: readonly TensorLike[]): MetricResult => {
    if (featuresX.length !== featuresY.length)
      throw new MdmlError('shape', 'lpips: one feature map per layer for each image')
    let total = 0
    for (const [l, fx] of featuresX.entries()) {
      const x = spectra(fx)
      const y = spectra(featuresY[l] as TensorLike)
      const wl = weights?.[l] === undefined ? undefined : shapedArray(weights[l] as TensorLike).data
      let layer = 0
      for (let i = 0; i < x.n; i++) {
        let nx = 0
        let ny = 0
        for (let k = 0; k < x.c; k++) {
          nx += (x.data[i * x.c + k] ?? 0) ** 2
          ny += (y.data[i * x.c + k] ?? 0) ** 2
        }
        nx = Math.sqrt(nx) + 1e-10
        ny = Math.sqrt(ny) + 1e-10
        for (let k = 0; k < x.c; k++) {
          const d = (x.data[i * x.c + k] ?? 0) / nx - (y.data[i * x.c + k] ?? 0) / ny
          layer += (wl?.[k] ?? 1) * d * d
        }
      }
      total += layer / x.n
    }
    return metricResult('lpipsFromFeatures', total, featuresX.length)
  },
)

/**
 * DISTS from supplied features (Ding et al., 2020): 1 - sum over layers and channels of alpha S_texture + beta
 * S_structure, with S_texture = (2 mu_x mu_y + c1) / (mu_x^2 + mu_y^2 + c1) and S_structure = (2 sigma_xy + c2) /
 * (sigma_x^2 + sigma_y^2 + c2) from the global statistics of each channel; alpha and beta are normalised to sum to 1.
 * Pass the VGG activations (the image itself first) and the learned weights; mdml ships no network.
 *
 * @param featuresX - Feature maps of the first image, one `[H, W, C]` per layer.
 * @param featuresY - Feature maps of the second image.
 * @param alpha - Texture weights per layer `[C_l]` (default all ones).
 * @param beta - Structure weights per layer `[C_l]` (default all ones).
 * @returns The metric result (0 for identical features).
 * @example
 * ```ts
 * import { distsFromFeatures } from 'mdml-engine/metrics'
 * distsFromFeatures([[[[1], [2]]]], [[[[1], [2]]]]).estimate.value // => 0
 * ```
 */
export const distsFromFeatures: MetricFunction<
  [FeatureStack, FeatureStack, (readonly TensorLike[])?, (readonly TensorLike[])?]
> = defineMetric(
  spec('distsFromFeatures', 'lower', [0, 1]),
  (
    featuresX: FeatureStack,
    featuresY: FeatureStack,
    alpha?: readonly TensorLike[],
    beta?: readonly TensorLike[],
  ): MetricResult => {
    const c1 = 1e-6
    const c2 = 1e-6
    let weightSum = 0
    let score = 0
    for (const [l, fx] of featuresX.entries()) {
      const x = spectra(fx)
      const y = spectra(featuresY[l] as TensorLike)
      const al = alpha?.[l] === undefined ? undefined : shapedArray(alpha[l] as TensorLike).data
      const bl = beta?.[l] === undefined ? undefined : shapedArray(beta[l] as TensorLike).data
      for (let k = 0; k < x.c; k++) {
        let mx = 0
        let my = 0
        for (let i = 0; i < x.n; i++) {
          mx += (x.data[i * x.c + k] ?? 0) / x.n
          my += (y.data[i * x.c + k] ?? 0) / x.n
        }
        let vx = 0
        let vy = 0
        let cxy = 0
        for (let i = 0; i < x.n; i++) {
          const dx = (x.data[i * x.c + k] ?? 0) - mx
          const dy = (y.data[i * x.c + k] ?? 0) - my
          vx += (dx * dx) / x.n
          vy += (dy * dy) / x.n
          cxy += (dx * dy) / x.n
        }
        const a = al?.[k] ?? 1
        const b = bl?.[k] ?? 1
        weightSum += a + b
        score += a * ((2 * mx * my + c1) / (mx * mx + my * my + c1)) + b * ((2 * cxy + c2) / (vx + vy + c2))
      }
    }
    return metricResult('distsFromFeatures', 1 - score / weightSum, featuresX.length)
  },
)
