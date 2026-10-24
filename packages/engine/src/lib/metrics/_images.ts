// 2-D image helpers of the image-quality metrics: planes, separable filtering with scipy.ndimage borders and valid
// correlation. TODO: move to signal/filters once the signal domain exists; the borders come from numerics/convolution.
import { WhiteboxError } from '../base/index.ts'
import { extendIndex, type PadMode } from '../numerics/convolution/index.ts'
import type { TensorLike } from '../tensor/index.ts'
import { matrix } from './metric.ts'

/** A 2-D plane of values. */
export interface Plane {
  readonly h: number
  readonly w: number
  readonly data: Float64Array
}

/** The channel planes of an image `[H, W]` or `[H, W, C]` (channels last). */
export function planes(name: string, image: TensorLike): Plane[] {
  const m = matrix(image)
  const [h = 0, w = 0, c = 1] = m.shape
  if (m.shape.length < 2 || m.shape.length > 3)
    throw new WhiteboxError('shape', `${name}: images are [H, W] or [H, W, C]`)
  return Array.from({ length: c }, (_, k) => ({
    h,
    w,
    data: Float64Array.from({ length: h * w }, (_, i) => m.data[i * c + k] ?? 0),
  }))
}

/** Planes of two images, checked to have the same shape. */
export function pairOfPlanes(name: string, a: TensorLike, b: TensorLike): [Plane[], Plane[]] {
  const pa = planes(name, a)
  const pb = planes(name, b)
  if (pa.length !== pb.length || pa[0]?.h !== pb[0]?.h || pa[0]?.w !== pb[0]?.w)
    throw new WhiteboxError('shape', `${name}: images differ in shape`)
  return [pa, pb]
}

/** Elementwise combination of planes. */
export function zip(a: Plane, b: Plane, f: (x: number, y: number) => number): Plane {
  return { h: a.h, w: a.w, data: Float64Array.from(a.data, (x, i) => f(x, b.data[i] ?? 0)) }
}

/** Correlate rows then columns with a 1-D kernel centred at index `origin` (borders: numpy pad modes; 'symmetric' is scipy.ndimage's 'reflect'). */
export function separable(p: Plane, kernel: readonly number[], origin: number, border: PadMode = 'symmetric'): Plane {
  const k = kernel.length
  const tmp = new Float64Array(p.h * p.w)
  for (let y = 0; y < p.h; y++) {
    for (let x = 0; x < p.w; x++) {
      let s = 0
      for (let j = 0; j < k; j++)
        s += (kernel[j] ?? 0) * (p.data[y * p.w + extendIndex(x + j - origin, p.w, border)] ?? 0)
      tmp[y * p.w + x] = s
    }
  }
  const out = new Float64Array(p.h * p.w)
  for (let y = 0; y < p.h; y++) {
    for (let x = 0; x < p.w; x++) {
      let s = 0
      for (let j = 0; j < k; j++) s += (kernel[j] ?? 0) * (tmp[extendIndex(y + j - origin, p.h, border) * p.w + x] ?? 0)
      out[y * p.w + x] = s
    }
  }
  return { h: p.h, w: p.w, data: out }
}

/** scipy.ndimage.uniform_filter (reflect borders; an even size covers [i - size/2, i + size/2 - 1]). */
export function uniformFilter(p: Plane, size: number): Plane {
  return separable(p, new Array<number>(size).fill(1 / size), Math.floor(size / 2))
}

/** scipy.ndimage.gaussian_filter (reflect borders, radius round(truncate sigma)). */
export function gaussianFilter(p: Plane, sigma: number, truncate = 4): Plane {
  const radius = Math.floor(truncate * sigma + 0.5)
  const raw = Array.from({ length: 2 * radius + 1 }, (_, i) => Math.exp(-0.5 * ((i - radius) / sigma) ** 2))
  const total = raw.reduce((a, b) => a + b, 0)
  return separable(
    p,
    raw.map((v) => v / total),
    radius,
  )
}

/** Valid 2-D correlation with a kernel `[k][k]` (scipy.signal.convolve2d with the kernel rotated, mode 'valid'). */
export function correlateValid(p: Plane, kernel: readonly (readonly number[])[]): Plane {
  const kh = kernel.length
  const kw = kernel[0]?.length ?? 0
  const h = p.h - kh + 1
  const w = p.w - kw + 1
  const out = new Float64Array(Math.max(h, 0) * Math.max(w, 0))
  for (let y = 0; y < h; y++) {
    for (let x = 0; x < w; x++) {
      let s = 0
      for (let i = 0; i < kh; i++)
        for (let j = 0; j < kw; j++) s += (kernel[i]?.[j] ?? 0) * (p.data[(y + i) * p.w + x + j] ?? 0)
      out[y * w + x] = s
    }
  }
  return { h, w, data: out }
}

/** A normalised Gaussian window `[n][n]` as MATLAB's fspecial (sewar), tiny weights zeroed. */
export function gaussianWindow(n: number, sigma: number): number[][] {
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
export function croppedMean(p: Plane, s: number): number {
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
export function planeMean(p: Plane): number {
  return p.data.reduce((a, b) => a + b, 0) / p.data.length
}

/** Every other row and column (x[::2, ::2]). */
export function decimate(p: Plane): Plane {
  const h = Math.ceil(p.h / 2)
  const w = Math.ceil(p.w / 2)
  return {
    h,
    w,
    data: Float64Array.from({ length: h * w }, (_, i) => p.data[2 * Math.floor(i / w) * p.w + 2 * (i % w)] ?? 0),
  }
}
