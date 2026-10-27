/**
 * Smoothing filters with OpenCV's parameterisation: `getGaussianKernel` (its fixed small kernels and the sigma it
 * derives from the size), `gaussianBlur`, the normalised box filter `blur`, `medianBlur` and `filter2D`, all with
 * OpenCV's default border (reflect-101, scipy's 'mirror') and run through the ndimage-style correlations of
 * `signal/image`.
 */

import { MdmlError } from '../base/index.ts'
import type { BorderMode } from '../numerics/convolution/index.ts'
import { image } from '../signal/index.ts'
import { primalOf, type Tensor, type TensorLike } from '../tensor/index.ts'
import { rasterOf, rasterTensor } from './_image.ts'

/** OpenCV's border types by their scipy.ndimage names: reflect-101 is 'mirror', replicate 'nearest', reflect 'reflect'. */
export type CvBorder = 'reflect101' | 'replicate' | 'reflect' | 'constant' | 'wrap'

const NDIMAGE: Readonly<Record<CvBorder, BorderMode>> = {
  reflect101: 'mirror',
  replicate: 'nearest',
  reflect: 'reflect',
  constant: 'constant',
  wrap: 'wrap',
}

/**
 * The scipy.ndimage border mode of an OpenCV border type (reflect-101 is 'mirror', replicate is 'nearest').
 *
 * @param border - The OpenCV border (default 'reflect101', OpenCV's default).
 * @returns The ndimage mode.
 * @example
 * ```ts
 * import { borderMode } from 'mdml-engine/vision'
 * borderMode('replicate') // => 'nearest'
 * ```
 */
export function borderMode(border: CvBorder = 'reflect101'): BorderMode {
  const m = NDIMAGE[border]
  if (m === undefined) throw new MdmlError('value', `unknown border ${String(border)}`)
  return m
}

const SMALL: readonly (readonly number[])[] = [
  [1],
  [0.25, 0.5, 0.25],
  [0.0625, 0.25, 0.375, 0.25, 0.0625],
  [0.03125, 0.109375, 0.21875, 0.28125, 0.21875, 0.109375, 0.03125],
  [4, 13, 30, 51, 60, 51, 30, 13, 4].map((v) => v / 256),
]

/**
 * The 1-D Gaussian kernel of OpenCV (cv2.getGaussianKernel): for odd sizes up to 9 with no sigma given, OpenCV's fixed
 * kernels; otherwise exp(-x^2 / (2 sigma^2)) on x = i - (n - 1) / 2, normalised, with sigma defaulting to
 * 0.3 ((n - 1) / 2 - 1) + 0.8. As OpenCV, an even-length kernel gives its two middle taps the weight of x = 0.
 *
 * @param ksize - The kernel length (positive).
 * @param sigma - Standard deviation (0 or negative: derived from ksize).
 * @returns The kernel.
 * @concept gaussian-filter
 * @example
 * ```ts
 * import { getGaussianKernel } from 'mdml-engine/vision'
 * Array.from(getGaussianKernel(3)) // => [0.25, 0.5, 0.25]
 * ```
 */
export function getGaussianKernel(ksize: number, sigma = 0): Float64Array {
  if (!(Number.isInteger(ksize) && ksize >= 1)) throw new MdmlError('value', 'getGaussianKernel: ksize must be >= 1')
  const fixed = ksize % 2 === 1 && ksize <= 9 && sigma <= 0 ? SMALL[ksize >> 1] : undefined
  if (fixed !== undefined) return Float64Array.from(fixed)
  const s = sigma > 0 ? sigma : ksize * 0.15 + 0.35
  const scale = -0.125 / (s * s)
  const half = (ksize - 1) >> 1
  // Taps at x = 2 i - (n - 1) on the doubled grid (so exp(-x^2 / (8 sigma^2))), the centre (or both centres) weight 1.
  const outer = Float64Array.from({ length: half }, (_, i) => Math.exp((1 - ksize + 2 * i) ** 2 * scale))
  const total = 2 * outer.reduce((a, b) => a + b, 0) + (ksize % 2 === 0 ? 2 : 1)
  const k = new Float64Array(ksize)
  for (let i = 0; i < half; i++) {
    k[i] = (outer[i] ?? 0) / total
    k[ksize - 1 - i] = k[i] ?? 0
  }
  k[half] = 1 / total
  if (ksize % 2 === 0) k[half + 1] = 1 / total
  return k
}

/** Separable correlation of every channel of an image with a row and a column kernel. */
function separable(x: TensorLike, kx: Float64Array, ky: Float64Array, border: CvBorder, name: string): Tensor {
  const r = rasterOf(x, name)
  const mode = borderMode(border)
  const t = rasterTensor(r)
  let v = image.correlate1d(t, kx, 1, { mode })
  v = image.correlate1d(v, ky, 0, { mode })
  return primalOf(v) as Tensor
}

/**
 * Gaussian smoothing (cv2.GaussianBlur): separable OpenCV Gaussian kernels; a zero size is derived from sigma as
 * round(8 sigma + 1) made odd, and sigmaY defaults to sigmaX.
 *
 * @param x - Image `[H, W]` or `[H, W, C]`.
 * @param ksize - Kernel size [width, height] (odd, or 0 to derive from sigma).
 * @param sigmaX - Horizontal standard deviation (0: from the size).
 * @param options - sigmaY and the border (default 'reflect101').
 * @returns The smoothed image.
 * @concept gaussian-filter
 * @example
 * ```ts
 * import { gaussianBlur } from 'mdml-engine/vision'
 * import { toArray } from 'mdml-engine/tensor'
 * toArray(gaussianBlur([[0, 0, 0], [0, 1, 0], [0, 0, 0]], [3, 3], 0, { border: 'constant' })) // => [[0.0625, 0.125, 0.0625], [0.125, 0.25, 0.125], [0.0625, 0.125, 0.0625]]
 * ```
 */
export function gaussianBlur(
  x: TensorLike,
  ksize: readonly [number, number],
  sigmaX: number,
  options: { readonly sigmaY?: number; readonly border?: CvBorder } = {},
): Tensor {
  const sx = sigmaX
  const sy = options.sigmaY !== undefined && options.sigmaY > 0 ? options.sigmaY : sigmaX
  const size = (k: number, s: number): number => (k > 0 ? k : Math.max(1, Math.round(s * 4 * 2 + 1) | 1))
  const kw = size(ksize[0], sx)
  const kh = size(ksize[1], sy)
  if (kw % 2 === 0 || kh % 2 === 0) throw new MdmlError('value', 'gaussianBlur: kernel sizes must be odd')
  return separable(
    x,
    getGaussianKernel(kw, sx),
    getGaussianKernel(kh, sy),
    options.border ?? 'reflect101',
    'gaussianBlur',
  )
}

/**
 * The normalised box filter (cv2.blur): the mean over a [width, height] window centred on each pixel.
 *
 * @param x - Image `[H, W]` or `[H, W, C]`.
 * @param ksize - Window size [width, height].
 * @param options - The border (default 'reflect101').
 * @returns The smoothed image.
 * @concept uniform-filter
 * @example
 * ```ts
 * import { blur } from 'mdml-engine/vision'
 * import { toArray } from 'mdml-engine/tensor'
 * toArray(blur([[0, 3, 0]], [3, 1])) // => [[2, 1, 2]]
 * ```
 */
export function blur(
  x: TensorLike,
  ksize: readonly [number, number],
  options: { readonly border?: CvBorder } = {},
): Tensor {
  const [kw, kh] = ksize
  if (!(kw >= 1 && kh >= 1)) throw new MdmlError('value', 'blur: the window must be positive')
  return separable(
    x,
    new Float64Array(kw).fill(1 / kw),
    new Float64Array(kh).fill(1 / kh),
    options.border ?? 'reflect101',
    'blur',
  )
}

/**
 * The median filter (cv2.medianBlur): the median over a ksize x ksize window, the image extended by replication.
 *
 * @param x - Image `[H, W]` or `[H, W, C]` (channels filtered separately).
 * @param ksize - Odd window size.
 * @returns The filtered image.
 * @concept rank-filter
 * @example
 * ```ts
 * import { medianBlur } from 'mdml-engine/vision'
 * import { toArray } from 'mdml-engine/tensor'
 * toArray(medianBlur([[0, 0, 0], [0, 9, 0], [0, 0, 0]], 3)) // => [[0, 0, 0], [0, 0, 0], [0, 0, 0]]
 * ```
 */
export function medianBlur(x: TensorLike, ksize: number): Tensor {
  if (!(Number.isInteger(ksize) && ksize % 2 === 1)) throw new MdmlError('value', 'medianBlur: ksize must be odd')
  const r = rasterOf(x, 'medianBlur')
  const size = r.rank === 3 ? [ksize, ksize, 1] : [ksize, ksize]
  return image.medianFilter(rasterTensor(r), size, { mode: 'nearest' })
}

/**
 * Correlate an image with a 2-D kernel (cv2.filter2D): out(x, y) = sum k(i, j) src(x + i - ax, y + j - ay) with the
 * anchor at the kernel centre, every channel alike.
 *
 * @param x - Image `[H, W]` or `[H, W, C]`.
 * @param kernel - Kernel `[kh, kw]`.
 * @param options - The border (default 'reflect101').
 * @returns The filtered image.
 * @concept correlation
 * @example
 * ```ts
 * import { filter2D } from 'mdml-engine/vision'
 * import { toArray } from 'mdml-engine/tensor'
 * toArray(filter2D([[1, 2, 3]], [[-1, 0, 1]], { border: 'replicate' })) // => [[1, 2, 1]]
 * ```
 */
export function filter2D(x: TensorLike, kernel: TensorLike, options: { readonly border?: CvBorder } = {}): Tensor {
  const r = rasterOf(x, 'filter2D')
  const k = rasterOf(kernel, 'filter2D')
  if (k.c !== 1 || k.rank !== 2) throw new MdmlError('shape', 'filter2D: the kernel is [kh, kw]')
  const kt = r.rank === 3 ? rasterTensor({ ...k, rank: 3 }) : rasterTensor(k)
  // ndimage centres a kernel of even length at k // 2 as OpenCV's default anchor does.
  return primalOf(
    image.correlateNd(rasterTensor(r), kt, { mode: borderMode(options.border ?? 'reflect101') }),
  ) as Tensor
}
