/**
 * Image filters, scipy.ndimage and scikit-image style: Gaussian smoothing and derivatives, box (uniform) means, rank
 * filters (median, minimum, maximum, percentile), the Sobel, Scharr and Prewitt edge filters, the Laplacian and
 * Laplacian of Gaussian, the edge-preserving bilateral filter, and Gabor kernels and filter banks for texture.
 */

import { OmaiError } from '../../base/index.ts'
import { type BorderMode, extendIndex, padModeOf } from '../../numerics/convolution/index.ts'
import { add, fromStorage, mul, primalOf, sqrt, type Tensor, type TensorLike, type Value } from '../../tensor/index.ts'
import { axisOf, float } from '../_shared.ts'
import { type BorderOptions, convolveNd, correlate1d } from './correlate.ts'

// Image filters as scipy.ndimage and scikit-image define them: Gaussian (with derivatives), uniform, rank (median,
// minimum, maximum, percentile), edge (Sobel, Scharr, Prewitt), Laplacian and Laplacian of Gaussian, bilateral and
// Gabor filters. The linear ones are sums of shifted inputs (correlate1d), so they are differentiable in the image.

const perAxis = (v: number | readonly number[], rank: number, name: string): number[] => {
  if (typeof v === 'number') return new Array<number>(rank).fill(v)
  if (v.length !== rank) throw new OmaiError('shape', `${name}: expected one value per axis (${rank})`)
  return [...v]
}

/**
 * The sampled Gaussian kernel of scipy.ndimage (or its derivative of the given order) on -radius .. radius, the
 * weights normalised to sum to one before differentiation.
 *
 * @param sigma - Standard deviation.
 * @param order - Derivative order (default 0).
 * @param radius - Half-width (default round(4 sigma)).
 * @returns The kernel `[2 radius + 1]`.
 * @concept gaussian-filter
 * @example
 * ```ts
 * import { image } from 'omai-engine/signal'
 * Array.from(image.gaussianKernel1d(1, 0, 1)).map((v) => Math.round(v * 1e4) / 1e4) // => [0.2741, 0.4519, 0.2741]
 * ```
 */
export function gaussianKernel1d(sigma: number, order = 0, radius: number = Math.floor(4 * sigma + 0.5)): Float64Array {
  if (!(sigma > 0)) throw new OmaiError('value', 'gaussianKernel1d: sigma must be positive')
  if (!Number.isInteger(order) || order < 0) throw new OmaiError('value', 'gaussianKernel1d: order must be >= 0')
  const s2 = sigma * sigma
  const phi = Float64Array.from({ length: 2 * radius + 1 }, (_, i) => Math.exp((-0.5 / s2) * (i - radius) ** 2))
  let total = 0
  for (const v of phi) total += v
  for (let i = 0; i < phi.length; i++) phi[i] = (phi[i] ?? 0) / total
  if (order === 0) return phi
  // The derivative is a polynomial times the Gaussian: q' = D q + P q with D the power-lowering and P (-x / sigma^2).
  let q = new Float64Array(order + 1)
  q[0] = 1
  for (let r = 0; r < order; r++) {
    const next = new Float64Array(order + 1)
    for (let i = 0; i < order; i++) next[i] = (next[i] ?? 0) + (i + 1) * (q[i + 1] ?? 0)
    for (let i = 1; i <= order; i++) next[i] = (next[i] ?? 0) - (q[i - 1] ?? 0) / s2
    q = next
  }
  return Float64Array.from(phi, (p, i) => {
    const x = i - radius
    let poly = 0
    for (let e = order; e >= 0; e--) poly = poly * x + (q[e] ?? 0)
    return poly * p
  })
}

/** Options of the Gaussian filters. */
export interface GaussianOptions extends BorderOptions {
  /** Derivative order per axis or shared (default 0). */
  readonly order?: number | readonly number[]
  /** Kernel half-width in standard deviations (default 4). */
  readonly truncate?: number
  /** Explicit kernel half-width (overrides truncate). */
  readonly radius?: number
}

/**
 * One-dimensional Gaussian filter along an axis (scipy.ndimage.gaussian_filter1d), or a Gaussian derivative.
 *
 * @param x - Input.
 * @param sigma - Standard deviation.
 * @param axis - Axis (default -1).
 * @param options - Order, truncation, radius, border.
 * @returns The filtered input.
 * @concept gaussian-filter
 * @example
 * ```ts
 * import { image } from 'omai-engine/signal'
 * import { toArray } from 'omai-engine/tensor'
 * toArray(image.gaussianFilter1d([0, 0, 1, 0, 0], 1, -1, { radius: 1 })).map((v) => Math.round(v * 1e4) / 1e4) // => [0, 0.2741, 0.4519, 0.2741, 0]
 * ```
 */
export function gaussianFilter1d(x: TensorLike, sigma: number, axis = -1, options: GaussianOptions = {}): Value {
  const order = typeof options.order === 'number' ? options.order : 0
  const radius = options.radius ?? Math.floor((options.truncate ?? 4) * sigma + 0.5)
  const k = gaussianKernel1d(sigma, order, radius).reverse()
  return correlate1d(x, k, axis, { ...options, origin: 0 })
}

/**
 * Multidimensional Gaussian filter (scipy.ndimage.gaussian_filter): separable 1-D Gaussians (or derivatives) along
 * every axis whose sigma is positive.
 *
 * @param x - Input.
 * @param sigma - Standard deviation, per axis or shared.
 * @param options - Orders, truncation, radius, border.
 * @returns The filtered input.
 * @concept gaussian-filter
 * @example
 * ```ts
 * import { image } from 'omai-engine/signal'
 * import { toArray } from 'omai-engine/tensor'
 * const y = toArray(image.gaussianFilter([[0, 0, 0], [0, 1, 0], [0, 0, 0]], 1, { radius: 1 })) as number[][]
 * Math.round((y[1]?.[1] ?? 0) * 1e4) / 1e4 // => 0.2042
 * ```
 */
export function gaussianFilter(x: TensorLike, sigma: number | readonly number[], options: GaussianOptions = {}): Value {
  let v = float(x)
  const rank = v.shape.length
  const sigmas = perAxis(sigma, rank, 'gaussianFilter')
  const orders = perAxis(options.order ?? 0, rank, 'gaussianFilter')
  for (let d = 0; d < rank; d++) {
    const s = sigmas[d] ?? 0
    if (s <= 1e-15) continue
    v = gaussianFilter1d(v, s, d, { ...options, order: orders[d] ?? 0 })
  }
  return v
}

/**
 * One-dimensional uniform (box mean) filter along an axis (scipy.ndimage.uniform_filter1d).
 *
 * @param x - Input.
 * @param size - Box length.
 * @param axis - Axis (default -1).
 * @param options - Border and origin.
 * @returns The filtered input.
 * @concept uniform-filter
 * @example
 * ```ts
 * import { image } from 'omai-engine/signal'
 * import { toArray } from 'omai-engine/tensor'
 * toArray(image.uniformFilter1d([3, 0, 0, 3], 3)) // => [2, 1, 1, 2]
 * ```
 */
export function uniformFilter1d(
  x: TensorLike,
  size: number,
  axis = -1,
  options: BorderOptions & { readonly origin?: number } = {},
): Value {
  if (!Number.isInteger(size) || size < 1) throw new OmaiError('value', 'uniformFilter1d: size must be >= 1')
  return correlate1d(x, new Float64Array(size).fill(1 / size), axis, options)
}

/**
 * Multidimensional uniform filter (scipy.ndimage.uniform_filter): box means along every axis.
 *
 * @param x - Input.
 * @param size - Box length per axis or shared.
 * @param options - Border and origin (per axis or shared).
 * @returns The filtered input.
 * @concept uniform-filter
 * @example
 * ```ts
 * import { image } from 'omai-engine/signal'
 * import { toArray } from 'omai-engine/tensor'
 * toArray(image.uniformFilter([[4, 0], [0, 0]], 2)) // => [[4, 2], [2, 1]]
 * ```
 */
export function uniformFilter(
  x: TensorLike,
  size: number | readonly number[],
  options: BorderOptions & { readonly origin?: number | readonly number[] } = {},
): Value {
  let v = float(x)
  const rank = v.shape.length
  const sizes = perAxis(size, rank, 'uniformFilter')
  const origins = perAxis(options.origin ?? 0, rank, 'uniformFilter')
  for (let d = 0; d < rank; d++)
    if ((sizes[d] ?? 1) > 1) v = uniformFilter1d(v, sizes[d] ?? 1, d, { ...options, origin: origins[d] ?? 0 })
  return v
}

/** Options of the rank filters. */
export interface RankOptions extends BorderOptions {
  /** Window origin per axis or shared (default 0, centred). */
  readonly origin?: number | readonly number[]
}

function rankFilterOf(
  x: TensorLike,
  size: number | readonly number[],
  options: RankOptions,
  name: string,
  pick: (window: Float64Array) => number,
): Tensor {
  const v = primalOf(float(x))
  const shape = v.shape
  const rank = shape.length
  const sizes = perAxis(size, rank, name)
  const origins = perAxis(options.origin ?? 0, rank, name)
  const mode = padModeOf(options.mode ?? 'reflect')
  const cval = options.cval ?? 0
  const data = v.data
  const strides = new Array<number>(rank).fill(1)
  for (let d = rank - 2; d >= 0; d--) strides[d] = (strides[d + 1] ?? 1) * (shape[d + 1] ?? 1)
  const total = data.length
  const wsize = sizes.reduce((a, b) => a * b, 1)
  const window = new Float64Array(wsize)
  const out = new Float64Array(total)
  const pos = new Array<number>(rank).fill(0)
  const off = new Array<number>(rank).fill(0)
  for (let flat = 0; flat < total; flat++) {
    let rem = flat
    for (let d = rank - 1; d >= 0; d--) {
      pos[d] = rem % (shape[d] ?? 1)
      rem = Math.floor(rem / (shape[d] ?? 1))
    }
    for (let w = 0; w < wsize; w++) {
      let r = w
      for (let d = rank - 1; d >= 0; d--) {
        off[d] = r % (sizes[d] ?? 1)
        r = Math.floor(r / (sizes[d] ?? 1))
      }
      let idx = 0
      let outside = false
      for (let d = 0; d < rank; d++) {
        const n = shape[d] ?? 1
        const i = extendIndex(
          (pos[d] ?? 0) + (off[d] ?? 0) - Math.floor((sizes[d] ?? 1) / 2) - (origins[d] ?? 0),
          n,
          mode,
        )
        if (i < 0) {
          outside = true
          break
        }
        idx += i * (strides[d] ?? 1)
      }
      window[w] = outside ? cval : Number(data[idx] ?? 0)
    }
    out[flat] = pick(window)
  }
  return fromStorage(out, [...shape], 'float64')
}

/**
 * The rank-th smallest value in a box around every element (scipy.ndimage.rank_filter); negative ranks count from
 * the largest.
 *
 * @param x - Input.
 * @param rank - Rank in the window.
 * @param size - Box size per axis or shared.
 * @param options - Border and origin.
 * @returns The filtered input (not differentiable).
 * @concept rank-filter
 * @example
 * ```ts
 * import { image } from 'omai-engine/signal'
 * import { toArray } from 'omai-engine/tensor'
 * toArray(image.rankFilter([1, 5, 2, 8, 3], 1, 3)) // => [1, 2, 5, 3, 3]
 * ```
 */
export function rankFilter(
  x: TensorLike,
  rank: number,
  size: number | readonly number[],
  options: RankOptions = {},
): Tensor {
  return rankFilterOf(x, size, options, 'rankFilter', (w) => {
    const r = rank < 0 ? w.length + rank : rank
    if (r < 0 || r >= w.length) throw new OmaiError('value', 'rankFilter: rank out of range')
    return w.sort()[r] ?? 0
  })
}

/**
 * The median in a box around every element (scipy.ndimage.median_filter).
 *
 * @param x - Input.
 * @param size - Box size per axis or shared.
 * @param options - Border and origin.
 * @returns The filtered input.
 * @concept rank-filter
 * @example
 * ```ts
 * import { image } from 'omai-engine/signal'
 * import { toArray } from 'omai-engine/tensor'
 * toArray(image.medianFilter([1, 9, 2, 3, 4], 3)) // => [1, 2, 3, 3, 4]
 * ```
 */
export function medianFilter(x: TensorLike, size: number | readonly number[], options: RankOptions = {}): Tensor {
  return rankFilterOf(x, size, options, 'medianFilter', (w) => w.sort()[Math.floor(w.length / 2)] ?? 0)
}

/**
 * The minimum in a box (scipy.ndimage.minimum_filter, grey erosion with a flat box).
 *
 * @param x - Input.
 * @param size - Box size per axis or shared.
 * @param options - Border and origin.
 * @returns The filtered input.
 * @concept rank-filter
 * @example
 * ```ts
 * import { image } from 'omai-engine/signal'
 * import { toArray } from 'omai-engine/tensor'
 * toArray(image.minimumFilter([3, 1, 4, 1, 5], 3)) // => [1, 1, 1, 1, 1]
 * ```
 */
export function minimumFilter(x: TensorLike, size: number | readonly number[], options: RankOptions = {}): Tensor {
  return rankFilterOf(x, size, options, 'minimumFilter', (w) => {
    let m = Number.POSITIVE_INFINITY
    for (const v of w) m = Math.min(m, v)
    return m
  })
}

/**
 * The maximum in a box (scipy.ndimage.maximum_filter, grey dilation with a flat box).
 *
 * @param x - Input.
 * @param size - Box size per axis or shared.
 * @param options - Border and origin.
 * @returns The filtered input.
 * @concept rank-filter
 * @example
 * ```ts
 * import { image } from 'omai-engine/signal'
 * import { toArray } from 'omai-engine/tensor'
 * toArray(image.maximumFilter([3, 1, 4, 1, 5], 3)) // => [3, 4, 4, 5, 5]
 * ```
 */
export function maximumFilter(x: TensorLike, size: number | readonly number[], options: RankOptions = {}): Tensor {
  return rankFilterOf(x, size, options, 'maximumFilter', (w) => {
    let m = Number.NEGATIVE_INFINITY
    for (const v of w) m = Math.max(m, v)
    return m
  })
}

/**
 * A percentile in a box (scipy.ndimage.percentile_filter): the value of rank int(size p / 100) in the window (the
 * largest for p = 100).
 *
 * @param x - Input.
 * @param percentile - Percentile in [0, 100] (negative counts from 100).
 * @param size - Box size per axis or shared.
 * @param options - Border and origin.
 * @returns The filtered input.
 * @concept rank-filter
 * @example
 * ```ts
 * import { image } from 'omai-engine/signal'
 * import { toArray } from 'omai-engine/tensor'
 * toArray(image.percentileFilter([1, 5, 2, 8, 3], 50, 3)) // => [1, 2, 5, 3, 3]
 * ```
 */
export function percentileFilter(
  x: TensorLike,
  percentile: number,
  size: number | readonly number[],
  options: RankOptions = {},
): Tensor {
  const p = percentile < 0 ? 100 + percentile : percentile
  if (p < 0 || p > 100) throw new OmaiError('value', 'percentileFilter: percentile out of range')
  return rankFilterOf(x, size, options, 'percentileFilter', (w) => {
    let r = Math.trunc((w.length * p) / 100)
    if (r === w.length) r = w.length - 1
    return w.sort()[r] ?? 0
  })
}

/** Options of the edge filters. */
export interface EdgeOptions extends BorderOptions {
  /** The axis (or axes) to differentiate along; several axes give the gradient magnitude (default all). */
  readonly axis?: number | readonly number[]
}

function edgeFilter(x: TensorLike, smooth: readonly number[], options: EdgeOptions, name: string): Value {
  const v = float(x)
  const rank = v.shape.length
  const axes =
    options.axis === undefined
      ? Array.from({ length: rank }, (_, i) => i)
      : (typeof options.axis === 'number' ? [options.axis] : [...options.axis]).map((a) => axisOf(a, rank, name))
  const magnitude = axes.length > 1
  let total: Value | null = null
  for (const a of axes) {
    // skimage convolves with [1, 0, -1]: correlation with [-1, 0, 1].
    let y = correlate1d(v, [-1, 0, 1], a, options)
    for (let d = 0; d < rank; d++) if (d !== a) y = correlate1d(y, smooth, d, options)
    const term = magnitude ? mul(y, y) : y
    total = total === null ? term : add(total, term)
  }
  if (total === null) return v
  return magnitude ? mul(sqrt(total), 1 / Math.sqrt(rank)) : total
}

/**
 * The Sobel edge filter (skimage.filters.sobel): the derivative [1, 0, -1] along an axis smoothed by [1, 2, 1] / 4
 * along the others; with several axes the gradient magnitude divided by sqrt(ndim).
 *
 * @param x - Image.
 * @param options - Axis or axes, border.
 * @returns The response.
 * @concept edge-detection
 * @example
 * ```ts
 * import { image } from 'omai-engine/signal'
 * import { toArray } from 'omai-engine/tensor'
 * toArray(image.sobel([[0, 0, 1, 1], [0, 0, 1, 1]], { axis: 1 })) // => [[0, 1, 1, 0], [0, 1, 1, 0]]
 * ```
 */
export function sobel(x: TensorLike, options: EdgeOptions = {}): Value {
  return edgeFilter(x, [0.25, 0.5, 0.25], options, 'sobel')
}

/**
 * The Scharr edge filter (skimage.filters.scharr): smoothing [3, 10, 3] / 16, better rotational symmetry.
 *
 * @param x - Image.
 * @param options - Axis or axes, border.
 * @returns The response.
 * @concept edge-detection
 * @example
 * ```ts
 * import { image } from 'omai-engine/signal'
 * import { toArray } from 'omai-engine/tensor'
 * toArray(image.scharr([[0, 0, 1, 1], [0, 0, 1, 1]], { axis: 1 })) // => [[0, 1, 1, 0], [0, 1, 1, 0]]
 * ```
 */
export function scharr(x: TensorLike, options: EdgeOptions = {}): Value {
  return edgeFilter(x, [3 / 16, 10 / 16, 3 / 16], options, 'scharr')
}

/**
 * The Prewitt edge filter (skimage.filters.prewitt): smoothing [1, 1, 1] / 3.
 *
 * @param x - Image.
 * @param options - Axis or axes, border.
 * @returns The response.
 * @concept edge-detection
 * @example
 * ```ts
 * import { image } from 'omai-engine/signal'
 * import { toArray } from 'omai-engine/tensor'
 * toArray(image.prewitt([[0, 0, 1, 1], [0, 0, 1, 1]], { axis: 1 })) // => [[0, 1, 1, 0], [0, 1, 1, 0]]
 * ```
 */
export function prewitt(x: TensorLike, options: EdgeOptions = {}): Value {
  return edgeFilter(x, [1 / 3, 1 / 3, 1 / 3], options, 'prewitt')
}

/**
 * The discrete Laplacian (scipy.ndimage.laplace): the sum over axes of the second difference [1, -2, 1].
 *
 * @param x - Input.
 * @param options - Border.
 * @returns The Laplacian.
 * @concept laplacian
 * @example
 * ```ts
 * import { image } from 'omai-engine/signal'
 * import { toArray } from 'omai-engine/tensor'
 * toArray(image.laplace([[0, 0, 0], [0, 1, 0], [0, 0, 0]], { mode: 'constant' })) // => [[0, 1, 0], [1, -4, 1], [0, 1, 0]]
 * ```
 */
export function laplace(x: TensorLike, options: BorderOptions = {}): Value {
  const v = float(x)
  let total: Value | null = null
  for (let d = 0; d < v.shape.length; d++) {
    const t = correlate1d(v, [1, -2, 1], d, options)
    total = total === null ? t : add(total, t)
  }
  return total ?? v
}

/**
 * The Laplacian of Gaussian (scipy.ndimage.gaussian_laplace): the sum over axes of second Gaussian derivatives; the
 * blob detector's kernel.
 *
 * @param x - Input.
 * @param sigma - Standard deviation per axis or shared.
 * @param options - Truncation, radius, border.
 * @returns The response.
 * @concept laplacian
 * @example
 * ```ts
 * import { image } from 'omai-engine/signal'
 * import { toArray } from 'omai-engine/tensor'
 * const y = toArray(image.gaussianLaplace([[0, 0, 0], [0, 1, 0], [0, 0, 0]], 1, { mode: 'constant' })) as number[][]
 * (y[1]?.[1] ?? 0) < 0 // => true
 * ```
 */
export function gaussianLaplace(
  x: TensorLike,
  sigma: number | readonly number[],
  options: Omit<GaussianOptions, 'order'> = {},
): Value {
  const v = float(x)
  const rank = v.shape.length
  let total: Value | null = null
  for (let d = 0; d < rank; d++) {
    const order = Array.from({ length: rank }, (_, i) => (i === d ? 2 : 0))
    const t = gaussianFilter(v, sigma, { ...options, order })
    total = total === null ? t : add(total, t)
  }
  return total ?? v
}

/**
 * The gradient magnitude of the Gaussian-smoothed input (scipy.ndimage.gaussian_gradient_magnitude).
 *
 * @param x - Input.
 * @param sigma - Standard deviation per axis or shared.
 * @param options - Truncation, radius, border.
 * @returns The magnitude.
 * @concept edge-detection
 * @example
 * ```ts
 * import { image } from 'omai-engine/signal'
 * import { toArray } from 'omai-engine/tensor'
 * const y = toArray(image.gaussianGradientMagnitude([0, 0, 1, 1], 1)) as number[]
 * (y[1] ?? 0) > (y[0] ?? 0) // => true
 * ```
 */
export function gaussianGradientMagnitude(
  x: TensorLike,
  sigma: number | readonly number[],
  options: Omit<GaussianOptions, 'order'> = {},
): Value {
  const v = float(x)
  const rank = v.shape.length
  let total: Value | null = null
  for (let d = 0; d < rank; d++) {
    const order = Array.from({ length: rank }, (_, i) => (i === d ? 1 : 0))
    const g = gaussianFilter(v, sigma, { ...options, order })
    const t = mul(g, g)
    total = total === null ? t : add(total, t)
  }
  return total === null ? v : sqrt(total)
}

/** Options of {@link bilateralFilter}. */
export interface BilateralOptions {
  /** Window side (default max(5, 2 ceil(3 sigmaSpatial) + 1)). */
  readonly winSize?: number
  /** Range (intensity or colour distance) standard deviation (default the image's standard deviation). */
  readonly sigmaColor?: number
  /** Spatial standard deviation in pixels (default 1). */
  readonly sigmaSpatial?: number
  /** Border mode (default 'constant', as scikit-image). */
  readonly mode?: BorderMode
  /** Constant for mode 'constant' (default 0). */
  readonly cval?: number
}

/**
 * The bilateral filter (Tomasi and Manduchi): each pixel becomes the mean of its window weighted by a spatial
 * Gaussian and a Gaussian of the intensity (or colour, for `[H, W, C]`) difference, smoothing within regions while
 * keeping edges. Exact weights (scikit-image bins the range weights).
 *
 * @param x - Image `[H, W]` or `[H, W, C]`.
 * @param options - Window, standard deviations, border.
 * @returns The filtered image.
 * @concept bilateral-filter
 * @example
 * ```ts
 * import { image } from 'omai-engine/signal'
 * import { toArray } from 'omai-engine/tensor'
 * const y = toArray(image.bilateralFilter([[0, 0, 1, 1], [0, 0, 1, 1]], { sigmaColor: 0.1, mode: 'nearest' })) as number[][]
 * Math.round((y[0]?.[1] ?? 0) * 1e3) / 1e3 // => 0
 * ```
 */
export function bilateralFilter(x: TensorLike, options: BilateralOptions = {}): Tensor {
  const v = primalOf(float(x))
  const shape = v.shape
  if (shape.length !== 2 && shape.length !== 3)
    throw new OmaiError('shape', 'bilateralFilter: images are [H, W] or [H, W, C]')
  const h = shape[0] ?? 0
  const w = shape[1] ?? 0
  const c = shape[2] ?? 1
  const data = Float64Array.from(v.data, Number)
  const sigmaS = options.sigmaSpatial ?? 1
  const win = options.winSize ?? Math.max(5, 2 * Math.ceil(3 * sigmaS) + 1)
  let sigmaC = options.sigmaColor
  if (sigmaC === undefined) {
    let mean = 0
    for (const d of data) mean += d / data.length
    let s = 0
    for (const d of data) s += (d - mean) ** 2
    sigmaC = Math.sqrt(s / data.length)
  }
  const mode = padModeOf(options.mode ?? 'constant')
  const cval = options.cval ?? 0
  const half = Math.floor(win / 2)
  const out = new Float64Array(data.length)
  const value = (y: number, xx: number, k: number): number => {
    const yi = extendIndex(y, h, mode)
    const xi = extendIndex(xx, w, mode)
    return yi < 0 || xi < 0 ? cval : (data[(yi * w + xi) * c + k] ?? 0)
  }
  const acc = new Float64Array(c)
  for (let y = 0; y < h; y++)
    for (let xx = 0; xx < w; xx++) {
      acc.fill(0)
      let total = 0
      for (let dy = -half; dy <= half; dy++)
        for (let dx = -half; dx <= half; dx++) {
          let d2 = 0
          for (let k = 0; k < c; k++) d2 += (value(y + dy, xx + dx, k) - (data[(y * w + xx) * c + k] ?? 0)) ** 2
          const wt = Math.exp(-(dy * dy + dx * dx) / (2 * sigmaS * sigmaS) - d2 / (2 * sigmaC * sigmaC))
          total += wt
          for (let k = 0; k < c; k++) acc[k] = (acc[k] ?? 0) + wt * value(y + dy, xx + dx, k)
        }
      for (let k = 0; k < c; k++) out[(y * w + xx) * c + k] = (acc[k] ?? 0) / total
    }
  return fromStorage(out, [...shape], 'float64')
}

/** Options of the Gabor kernel and filter. */
export interface GaborOptions {
  /** Orientation in radians (default 0). */
  readonly theta?: number
  /** Bandwidth in octaves, setting sigma when it is not given (default 1). */
  readonly bandwidth?: number
  readonly sigmaX?: number
  readonly sigmaY?: number
  /** Kernel half-size in standard deviations (default 3). */
  readonly nStds?: number
  /** Phase offset (default 0). */
  readonly offset?: number
}

/**
 * The complex Gabor kernel (skimage.filters.gabor_kernel): a Gaussian envelope times a complex sinusoid of the given
 * spatial frequency along direction theta, normalised by 2 pi sigmaX sigmaY.
 *
 * @param frequency - Spatial frequency (cycles per pixel).
 * @param options - Orientation, bandwidth or sigmas, size and phase.
 * @returns The real and imaginary parts `[2 y0 + 1, 2 x0 + 1]`.
 * @concept gabor-filter
 * @example
 * ```ts
 * import { image } from 'omai-engine/signal'
 * image.gaborKernel(0.25).real.shape // => [15, 15]
 * ```
 */
export function gaborKernel(
  frequency: number,
  options: GaborOptions = {},
): { readonly real: Tensor; readonly imag: Tensor } {
  const b = options.bandwidth ?? 1
  const prefactor = (1 / Math.PI) * Math.sqrt(Math.log(2) / 2) * ((2 ** b + 1) / (2 ** b - 1))
  const sx = options.sigmaX ?? prefactor / frequency
  const sy = options.sigmaY ?? prefactor / frequency
  const theta = options.theta ?? 0
  const n = options.nStds ?? 3
  const ct = Math.cos(theta)
  const st = Math.sin(theta)
  const x0 = Math.ceil(Math.max(Math.abs(n * sx * ct), Math.abs(n * sy * st), 1))
  const y0 = Math.ceil(Math.max(Math.abs(n * sy * ct), Math.abs(n * sx * st), 1))
  const rows = 2 * y0 + 1
  const cols = 2 * x0 + 1
  const re = new Float64Array(rows * cols)
  const im = new Float64Array(rows * cols)
  const norm = 1 / (2 * Math.PI * sx * sy)
  for (let i = 0; i < rows; i++)
    for (let j = 0; j < cols; j++) {
      const y = i - y0
      const x = j - x0
      const rx = x * ct + y * st
      const ry = -x * st + y * ct
      const env = Math.exp(-0.5 * ((rx * rx) / (sx * sx) + (ry * ry) / (sy * sy))) * norm
      const phase = 2 * Math.PI * frequency * rx + (options.offset ?? 0)
      re[i * cols + j] = env * Math.cos(phase)
      im[i * cols + j] = env * Math.sin(phase)
    }
  return { real: fromStorage(re, [rows, cols], 'float64'), imag: fromStorage(im, [rows, cols], 'float64') }
}

/**
 * Gabor filtering of an image (skimage.filters.gabor): convolution with the real and imaginary parts of the kernel.
 *
 * @param x - Image `[H, W]`.
 * @param frequency - Spatial frequency.
 * @param options - Kernel options and border (default 'reflect').
 * @returns The real and imaginary responses.
 * @concept gabor-filter
 * @example
 * ```ts
 * import { image } from 'omai-engine/signal'
 * const x = Array.from({ length: 8 }, () => Array.from({ length: 8 }, (_, j) => Math.cos((Math.PI * j) / 2)))
 * image.gabor(x, 0.25).real.shape // => [8, 8]
 * ```
 */
export function gabor(
  x: TensorLike,
  frequency: number,
  options: GaborOptions & BorderOptions = {},
): { readonly real: Value; readonly imag: Value } {
  const v = float(x)
  if (v.shape.length !== 2) throw new OmaiError('shape', 'gabor: images are [H, W]')
  const k = gaborKernel(frequency, options)
  const border = { mode: options.mode ?? 'reflect', cval: options.cval ?? 0 }
  return { real: convolveNd(v, k.real, border), imag: convolveNd(v, k.imag, border) }
}

/**
 * A Gabor filter bank: the response magnitudes sqrt(real^2 + imag^2) for every frequency and orientation, the
 * texture features of Gabor-based classifiers.
 *
 * @param x - Image `[H, W]`.
 * @param frequencies - Spatial frequencies.
 * @param thetas - Orientations.
 * @param options - Kernel options (except theta) and border.
 * @returns Magnitudes `[frequencies, thetas, H, W]`.
 * @concept gabor-filter
 * @example
 * ```ts
 * import { image } from 'omai-engine/signal'
 * const x = Array.from({ length: 6 }, (_, i) => Array.from({ length: 6 }, (_, j) => (i + j) % 2))
 * image.gaborBank(x, [0.2, 0.4], [0, Math.PI / 2]).shape // => [2, 2, 6, 6]
 * ```
 */
export function gaborBank(
  x: TensorLike,
  frequencies: readonly number[],
  thetas: readonly number[],
  options: Omit<GaborOptions, 'theta'> & BorderOptions = {},
): Tensor {
  const v = float(x)
  const h = v.shape[0] ?? 0
  const w = v.shape[1] ?? 0
  const out = new Float64Array(frequencies.length * thetas.length * h * w)
  let o = 0
  for (const f of frequencies)
    for (const theta of thetas) {
      const r = gabor(v, f, { ...options, theta })
      const re = primalOf(r.real).data
      const im = primalOf(r.imag).data
      for (let i = 0; i < h * w; i++) out[o + i] = Math.hypot(Number(re[i] ?? 0), Number(im[i] ?? 0))
      o += h * w
    }
  return fromStorage(out, [frequencies.length, thetas.length, h, w], 'float64')
}
