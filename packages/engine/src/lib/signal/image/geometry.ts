/**
 * Geometric transforms of images and arrays, scipy.ndimage and scikit-image style: spline interpolation at arbitrary
 * coordinates, affine and projective warps, shift, zoom, rotate and resize, Gaussian pyramids (successive smoothing and
 * downsampling), and the integral image (summed-area table) that gives any box sum in four lookups.
 */

import { OmaiError } from '../../base/index.ts'
import { extendIndex } from '../../numerics/convolution/index.ts'
import { fromStorage, primalOf, type Tensor, type TensorLike, toFlat } from '../../tensor/index.ts'
import { float } from '../_shared.ts'
import { gaussianFilter } from './filters.ts'

// Geometric transforms as scipy.ndimage and scikit-image define them: interpolation at arbitrary coordinates
// (nearest and linear, every ndimage boundary mode), affine transforms, shifts, zooms, rotations, scikit-image's
// resize and projective warp, Gaussian pyramids and integral images. TODO: spline orders 2 to 5 (ndimage's
// spline prefilter); they are rejected here.

/** ndimage boundary modes for interpolation. */
export type InterpolationMode = 'constant' | 'grid-constant' | 'nearest' | 'reflect' | 'mirror' | 'wrap' | 'grid-wrap'

/** Options of the interpolating transforms. */
export interface InterpolationOptions {
  /** Interpolation order: 0 (nearest) or 1 (linear; default). */
  readonly order?: 0 | 1
  /** Boundary mode (default 'constant'). */
  readonly mode?: InterpolationMode
  /** Value outside for the constant modes (default 0). */
  readonly cval?: number
}

interface Image {
  readonly data: Float64Array
  readonly shape: readonly number[]
  readonly strides: readonly number[]
}

function imageOf(x: TensorLike): Image {
  const v = primalOf(float(x))
  const shape = [...v.shape]
  const strides = new Array<number>(shape.length).fill(1)
  for (let d = shape.length - 2; d >= 0; d--) strides[d] = (strides[d + 1] ?? 1) * (shape[d + 1] ?? 1)
  return { data: Float64Array.from(v.data, Number), shape, strides }
}

/** The input index for an integer position under a mode, or -1 for the constant value. */
function indexOf(i: number, n: number, mode: InterpolationMode): number {
  if (i >= 0 && i < n) return i
  switch (mode) {
    case 'constant':
    case 'grid-constant':
      return -1
    case 'nearest':
      return extendIndex(i, n, 'edge')
    case 'reflect':
      return extendIndex(i, n, 'symmetric')
    case 'mirror':
      return n === 1 ? 0 : extendIndex(i, n, 'reflect')
    case 'grid-wrap':
      return extendIndex(i, n, 'wrap')
    case 'wrap': {
      // ndimage's wrap for interpolation: the first and last samples coincide (period n - 1).
      if (n === 1) return 0
      const p = n - 1
      return ((i % p) + p) % p
    }
  }
}

function sampler(img: Image, options: InterpolationOptions): (coord: readonly number[]) => number {
  const order = options.order ?? 1
  if (order !== 0 && order !== 1) throw new OmaiError('value', 'interpolation: only orders 0 and 1 are supported')
  const mode = options.mode ?? 'constant'
  const cval = options.cval ?? 0
  const rank = img.shape.length
  const corners = 2 ** rank
  return (coord) => {
    if (mode === 'constant')
      for (let d = 0; d < rank; d++) {
        const c = coord[d] ?? 0
        if (c < -1e-9 || c > (img.shape[d] ?? 1) - 1 + 1e-9) return cval
      }
    if (order === 0) {
      let idx = 0
      for (let d = 0; d < rank; d++) {
        const i = indexOf(Math.floor((coord[d] ?? 0) + 0.5), img.shape[d] ?? 1, mode)
        if (i < 0) return cval
        idx += i * (img.strides[d] ?? 1)
      }
      return img.data[idx] ?? 0
    }
    let total = 0
    for (let k = 0; k < corners; k++) {
      let w = 1
      let idx = 0
      let outside = false
      for (let d = 0; d < rank; d++) {
        const c = coord[d] ?? 0
        const lo = Math.floor(c)
        const t = c - lo
        const bit = (k >> d) & 1
        w *= bit === 1 ? t : 1 - t
        if (w === 0) break
        const i = indexOf(lo + bit, img.shape[d] ?? 1, mode)
        if (i < 0) outside = true
        else idx += i * (img.strides[d] ?? 1)
      }
      if (w === 0) continue
      total += w * (outside ? cval : (img.data[idx] ?? 0))
    }
    return total
  }
}

function transformInto(
  img: Image,
  outShape: readonly number[],
  map: (out: readonly number[], into: number[]) => void,
  options: InterpolationOptions,
): Tensor {
  const sample = sampler(img, options)
  const total = outShape.reduce((a, b) => a * b, 1)
  const out = new Float64Array(total)
  const o = new Array<number>(outShape.length).fill(0)
  const c = new Array<number>(img.shape.length).fill(0)
  for (let flat = 0; flat < total; flat++) {
    let rem = flat
    for (let d = outShape.length - 1; d >= 0; d--) {
      o[d] = rem % (outShape[d] ?? 1)
      rem = Math.floor(rem / (outShape[d] ?? 1))
    }
    map(o, c)
    out[flat] = sample(c)
  }
  return fromStorage(out, [...outShape], 'float64')
}

/**
 * Interpolate an array at coordinates (scipy.ndimage.map_coordinates): coordinates `[rank, ...points]`, one row per
 * input axis.
 *
 * @param x - Input array.
 * @param coordinates - Coordinates `[rank, ...]`.
 * @param options - Order, mode and constant.
 * @returns The values, shaped like one row of the coordinates.
 * @concept image-interpolation
 * @example
 * ```ts
 * import { image } from 'omai-engine/signal'
 * import { toArray } from 'omai-engine/tensor'
 * toArray(image.mapCoordinates([[0, 1], [2, 3]], [[0.5, 1], [0.5, 0]])) // => [1.5, 2]
 * ```
 */
export function mapCoordinates(x: TensorLike, coordinates: TensorLike, options: InterpolationOptions = {}): Tensor {
  const img = imageOf(x)
  const cv = primalOf(float(coordinates))
  const rank = img.shape.length
  if ((cv.shape[0] ?? 0) !== rank) throw new OmaiError('shape', 'mapCoordinates: one coordinate row per input axis')
  const points = cv.shape.slice(1)
  const n = points.reduce((a, b) => a * b, 1)
  const data = cv.data
  const flat = transformInto(
    img,
    [n],
    (o, into) => {
      for (let d = 0; d < rank; d++) into[d] = Number(data[d * n + (o[0] ?? 0)] ?? 0)
    },
    options,
  )
  return reshapeTensor(flat, points)
}

function reshapeTensor(t: Tensor, shape: readonly number[]): Tensor {
  return fromStorage(Float64Array.from(t.data, Number), [...shape], 'float64')
}

/**
 * An affine transform (scipy.ndimage.affine_transform): the output at o samples the input at matrix o + offset;
 * the matrix may be a vector (diagonal), square, or homogeneous (rank + 1 rows).
 *
 * @param x - Input array.
 * @param matrix - The output-to-input matrix.
 * @param options - Offset, output shape, order, mode and constant.
 * @returns The transformed array.
 * @concept geometric-transform
 * @example
 * ```ts
 * import { image } from 'omai-engine/signal'
 * import { toArray } from 'omai-engine/tensor'
 * toArray(image.affineTransform([[1, 2], [3, 4]], [[0, 1], [1, 0]])) // => [[1, 3], [2, 4]]
 * ```
 */
export function affineTransform(
  x: TensorLike,
  matrix: TensorLike,
  options: InterpolationOptions & {
    readonly offset?: number | readonly number[]
    readonly outputShape?: readonly number[]
  } = {},
): Tensor {
  const img = imageOf(x)
  const rank = img.shape.length
  const m = primalOf(float(matrix))
  const md = toFlat(m)
  let lin: number[][]
  let off =
    typeof options.offset === 'number'
      ? new Array<number>(rank).fill(options.offset)
      : [...(options.offset ?? new Array<number>(rank).fill(0))]
  if (m.shape.length === 1)
    lin = Array.from({ length: rank }, (_, i) => Array.from({ length: rank }, (_, j) => (i === j ? (md[i] ?? 0) : 0)))
  else {
    const cols = m.shape[1] ?? 0
    lin = Array.from({ length: rank }, (_, i) => Array.from({ length: rank }, (_, j) => md[i * cols + j] ?? 0))
    if (cols === rank + 1) off = Array.from({ length: rank }, (_, i) => md[i * cols + rank] ?? 0)
  }
  return transformInto(
    img,
    options.outputShape ?? img.shape,
    (o, into) => {
      for (let i = 0; i < rank; i++) {
        let s = off[i] ?? 0
        for (let j = 0; j < rank; j++) s += (lin[i]?.[j] ?? 0) * (o[j] ?? 0)
        into[i] = s
      }
    },
    options,
  )
}

/**
 * Shift an array (scipy.ndimage.shift): the output at o samples the input at o - shift.
 *
 * @param x - Input array.
 * @param shift - Shift per axis or shared.
 * @param options - Order, mode and constant.
 * @returns The shifted array.
 * @concept geometric-transform
 * @example
 * ```ts
 * import { image } from 'omai-engine/signal'
 * import { toArray } from 'omai-engine/tensor'
 * toArray(image.shift([1, 2, 3, 4], 1)) // => [0, 1, 2, 3]
 * ```
 */
export function shift(x: TensorLike, shift: number | readonly number[], options: InterpolationOptions = {}): Tensor {
  const img = imageOf(x)
  const s = typeof shift === 'number' ? new Array<number>(img.shape.length).fill(shift) : [...shift]
  return affineTransform(x, new Array<number>(img.shape.length).fill(1), { ...options, offset: s.map((v) => -v) })
}

/**
 * Zoom an array (scipy.ndimage.zoom): output shape round(n zoom); corner-aligned sampling (default) or, with
 * gridMode, pixel-centre-aligned sampling (o + 1/2) n / m - 1/2.
 *
 * @param x - Input array.
 * @param zoom - Factor per axis or shared.
 * @param options - Order, mode, constant and grid mode.
 * @returns The zoomed array.
 * @concept geometric-transform
 * @example
 * ```ts
 * import { image } from 'omai-engine/signal'
 * import { toArray } from 'omai-engine/tensor'
 * toArray(image.zoom([0, 2], 2)) // => [0, 0.6666666666666666, 1.3333333333333333, 2]
 * ```
 */
export function zoom(
  x: TensorLike,
  zoom: number | readonly number[],
  options: InterpolationOptions & { readonly gridMode?: boolean } = {},
): Tensor {
  const img = imageOf(x)
  const rank = img.shape.length
  const z = typeof zoom === 'number' ? new Array<number>(rank).fill(zoom) : [...zoom]
  const outShape = img.shape.map((n, d) => Math.round(n * (z[d] ?? 1)))
  return zoomTo(img, outShape, options)
}

function zoomTo(
  img: Image,
  outShape: readonly number[],
  options: InterpolationOptions & { readonly gridMode?: boolean },
): Tensor {
  const grid = options.gridMode ?? false
  const factor = img.shape.map((n, d) => {
    const m = outShape[d] ?? 1
    if (grid) return n / m
    return m > 1 ? (n - 1) / (m - 1) : 1
  })
  return transformInto(
    img,
    outShape,
    (o, into) => {
      for (let d = 0; d < img.shape.length; d++)
        into[d] = grid ? ((o[d] ?? 0) + 0.5) * (factor[d] ?? 1) - 0.5 : (o[d] ?? 0) * (factor[d] ?? 1)
    },
    options,
  )
}

/**
 * Rotate an image (scipy.ndimage.rotate) by an angle in degrees in the plane of the first two axes, enlarging the
 * output to hold the whole rotated input by default.
 *
 * @param x - Image `[H, W]`.
 * @param angle - Angle in degrees (counter-clockwise).
 * @param options - Reshape (default true), order, mode and constant.
 * @returns The rotated image.
 * @concept geometric-transform
 * @example
 * ```ts
 * import { image } from 'omai-engine/signal'
 * import { toArray } from 'omai-engine/tensor'
 * toArray(image.rotate([[1, 2], [3, 4]], 90)).map((r) => (r as number[]).map((v) => Math.round(v * 1e9) / 1e9)) // => [[2, 4], [1, 3]]
 * ```
 */
export function rotate(
  x: TensorLike,
  angle: number,
  options: InterpolationOptions & { readonly reshape?: boolean } = {},
): Tensor {
  const img = imageOf(x)
  if (img.shape.length !== 2) throw new OmaiError('shape', 'rotate: images are [H, W]')
  // scipy's cosdg and sindg: exact at multiples of 90 degrees.
  const r = ((angle % 360) + 360) % 360
  const quarter = r % 90 === 0 ? r / 90 : -1
  const c = quarter >= 0 ? ([1, 0, -1, 0][quarter] ?? 1) : Math.cos((angle * Math.PI) / 180)
  const s = quarter >= 0 ? ([0, 1, 0, -1][quarter] ?? 0) : Math.sin((angle * Math.PI) / 180)
  const [iy = 0, ix = 0] = img.shape
  let outShape: number[] = [iy, ix]
  if (options.reshape ?? true) {
    const xs = [0, 0, iy, iy].map((v, k) => c * v + s * ([0, ix, 0, ix][k] ?? 0))
    const ys = [0, 0, iy, iy].map((v, k) => -s * v + c * ([0, ix, 0, ix][k] ?? 0))
    const ptp = (a: number[]): number =>
      a.reduce((m, v) => Math.max(m, v), Number.NEGATIVE_INFINITY) -
      a.reduce((m, v) => Math.min(m, v), Number.POSITIVE_INFINITY)
    outShape = [Math.trunc(ptp(xs) + 0.5), Math.trunc(ptp(ys) + 0.5)]
  }
  const oc = [((outShape[0] ?? 1) - 1) / 2, ((outShape[1] ?? 1) - 1) / 2]
  const outCentre = [c * (oc[0] ?? 0) + s * (oc[1] ?? 0), -s * (oc[0] ?? 0) + c * (oc[1] ?? 0)]
  const offset = [(iy - 1) / 2 - (outCentre[0] ?? 0), (ix - 1) / 2 - (outCentre[1] ?? 0)]
  return affineTransform(
    x,
    [
      [c, s],
      [-s, c],
    ],
    { ...options, offset, outputShape: outShape },
  )
}

/** scikit-image's boundary modes (translated to ndimage's as skimage does). */
export type SkimageMode = 'constant' | 'edge' | 'symmetric' | 'reflect' | 'wrap'

const NDIMAGE_MODES: Readonly<Record<SkimageMode, InterpolationMode>> = {
  constant: 'grid-constant',
  edge: 'nearest',
  symmetric: 'reflect',
  reflect: 'mirror',
  wrap: 'grid-wrap',
}
const ndimageMode = (mode: SkimageMode): InterpolationMode => NDIMAGE_MODES[mode]

function clipToInput(out: Tensor, input: Image, mode: SkimageMode, cval: number): Tensor {
  let lo = Number.POSITIVE_INFINITY
  let hi = Number.NEGATIVE_INFINITY
  for (const v of input.data) {
    lo = Math.min(lo, v)
    hi = Math.max(hi, v)
  }
  const od = out.data
  let olo = Number.POSITIVE_INFINITY
  let ohi = Number.NEGATIVE_INFINITY
  for (const v of od) {
    olo = Math.min(olo, Number(v))
    ohi = Math.max(ohi, Number(v))
  }
  if (mode === 'constant' && !(lo <= cval && cval <= hi) && olo <= cval && cval <= ohi) {
    lo = Math.min(lo, cval)
    hi = Math.max(hi, cval)
  }
  return fromStorage(
    Float64Array.from(od, (v) => Math.min(hi, Math.max(lo, Number(v)))),
    [...out.shape],
    'float64',
  )
}

/** Options of {@link resize}. */
export interface ResizeOptions {
  /** Interpolation order (default 1). */
  readonly order?: 0 | 1
  /** Boundary mode (default 'reflect'). */
  readonly mode?: SkimageMode
  readonly cval?: number
  /** Clip to the input's range (default true). */
  readonly clip?: boolean
  /** Gaussian anti-aliasing before downsampling (default true when shrinking). */
  readonly antiAliasing?: boolean
  /** Its standard deviations (default max(0, (factor - 1) / 2)). */
  readonly antiAliasingSigma?: number | readonly number[]
}

/**
 * Resize an image (skimage.transform.resize): optional Gaussian anti-aliasing, then pixel-centre-aligned zooming to
 * the output shape, clipped to the input's range.
 *
 * @param x - Image.
 * @param outputShape - The output shape.
 * @param options - Order, mode, constant, clipping and anti-aliasing.
 * @returns The resized image.
 * @concept geometric-transform
 * @example
 * ```ts
 * import { image } from 'omai-engine/signal'
 * import { toArray } from 'omai-engine/tensor'
 * toArray(image.resize([[0, 1], [2, 3]], [4, 4])).length // => 4
 * ```
 */
export function resize(x: TensorLike, outputShape: readonly number[], options: ResizeOptions = {}): Tensor {
  const img = imageOf(x)
  if (outputShape.length !== img.shape.length)
    throw new OmaiError('shape', 'resize: the output shape must have the image rank')
  const factors = img.shape.map((n, d) => n / (outputShape[d] ?? 1))
  const mode = options.mode ?? 'reflect'
  const cval = options.cval ?? 0
  const nd = ndimageMode(mode)
  const anti = options.antiAliasing ?? factors.some((f) => f > 1)
  let filtered: Image = img
  if (anti) {
    const sig = options.antiAliasingSigma
    const sigma =
      sig === undefined
        ? factors.map((f) => Math.max(0, (f - 1) / 2))
        : typeof sig === 'number'
          ? factors.map(() => sig)
          : [...sig]
    const borderMode = nd === 'grid-constant' ? 'constant' : nd === 'grid-wrap' ? 'wrap' : nd
    filtered = imageOf(
      gaussianFilter(fromStorage(img.data, [...img.shape], 'float64'), sigma, { mode: borderMode, cval }),
    )
  }
  const out = zoomTo(filtered, outputShape, { order: options.order ?? 1, mode: nd, cval, gridMode: true })
  return (options.clip ?? true) ? clipToInput(out, img, mode, cval) : out
}

/** Options of the Gaussian pyramid. */
export interface PyramidOptions {
  /** Scale factor (default 2). */
  readonly factor?: number
  /** Smoothing standard deviation (default 2 factor / 6). */
  readonly sigma?: number
  /** Interpolation order (default 1). */
  readonly order?: 0 | 1
  /** Boundary mode (default 'reflect'). */
  readonly mode?: SkimageMode
  readonly cval?: number
}

/**
 * One level down a Gaussian pyramid (skimage.transform.pyramid_reduce): Gaussian smoothing, then resizing to
 * ceil(n / factor).
 *
 * @param x - Image.
 * @param options - Factor, smoothing, order, mode and constant.
 * @returns The reduced image.
 * @concept image-pyramid
 * @example
 * ```ts
 * import { image } from 'omai-engine/signal'
 * image.pyramidReduce(Array.from({ length: 8 }, () => new Array(8).fill(1))).shape // => [4, 4]
 * ```
 */
export function pyramidReduce(x: TensorLike, options: PyramidOptions = {}): Tensor {
  const img = imageOf(x)
  const f = options.factor ?? 2
  const sigma = options.sigma ?? (2 * f) / 6
  const mode = options.mode ?? 'reflect'
  const smoothed = gaussianFilter(fromStorage(img.data, [...img.shape], 'float64'), sigma, {
    mode: mode === 'edge' ? 'nearest' : mode === 'symmetric' ? 'reflect' : mode,
    cval: options.cval ?? 0,
  })
  return resize(
    smoothed,
    img.shape.map((n) => Math.ceil(n / f)),
    {
      order: options.order ?? 1,
      mode,
      cval: options.cval ?? 0,
      antiAliasing: false,
    },
  )
}

/**
 * One level up a Gaussian pyramid (skimage.transform.pyramid_expand): resizing to ceil(factor n), then smoothing.
 *
 * @param x - Image.
 * @param options - Factor, smoothing, order, mode and constant.
 * @returns The expanded image.
 * @concept image-pyramid
 * @example
 * ```ts
 * import { image } from 'omai-engine/signal'
 * image.pyramidExpand([[1, 2], [3, 4]]).shape // => [4, 4]
 * ```
 */
export function pyramidExpand(x: TensorLike, options: PyramidOptions = {}): Tensor {
  const img = imageOf(x)
  const f = options.factor ?? 2
  const sigma = options.sigma ?? (2 * f) / 6
  const mode = options.mode ?? 'reflect'
  const resized = resize(
    x,
    img.shape.map((n) => Math.ceil(f * n)),
    {
      order: options.order ?? 1,
      mode,
      cval: options.cval ?? 0,
      antiAliasing: false,
    },
  )
  return gaussianFilter(resized, sigma, {
    mode: mode === 'edge' ? 'nearest' : mode === 'symmetric' ? 'reflect' : mode,
    cval: options.cval ?? 0,
  }) as Tensor
}

/**
 * The Gaussian pyramid (skimage.transform.pyramid_gaussian): the image and its successive reductions, until a
 * level stops shrinking or maxLayer levels are made.
 *
 * @param x - Image.
 * @param maxLayer - Levels after the first (default until the shape stops changing).
 * @param options - Reduction options.
 * @returns The levels, finest first.
 * @concept image-pyramid
 * @example
 * ```ts
 * import { image } from 'omai-engine/signal'
 * image.pyramidGaussian(Array.from({ length: 8 }, () => new Array(8).fill(1))).map((l) => l.shape[0]) // => [8, 4, 2, 1]
 * ```
 */
export function pyramidGaussian(x: TensorLike, maxLayer = -1, options: PyramidOptions = {}): Tensor[] {
  const img = imageOf(x)
  const out: Tensor[] = [fromStorage(img.data, [...img.shape], 'float64')]
  let prev: Tensor = out[0] as Tensor
  for (let layer = 1; maxLayer < 0 || layer <= maxLayer; layer++) {
    const next = pyramidReduce(prev, options)
    if (next.shape.every((v, d) => v === prev.shape[d])) break
    out.push(next)
    prev = next
  }
  return out
}

/**
 * The integral image (summed-area table; skimage.transform.integral_image): cumulative sums along every axis, so
 * any box sum takes four lookups.
 *
 * @param x - Image `[H, W]`.
 * @returns The table, the shape of x.
 * @concept integral-image
 * @example
 * ```ts
 * import { image } from 'omai-engine/signal'
 * import { toArray } from 'omai-engine/tensor'
 * toArray(image.integralImage([[1, 2], [3, 4]])) // => [[1, 3], [4, 10]]
 * ```
 */
export function integralImage(x: TensorLike): Tensor {
  const img = imageOf(x)
  if (img.shape.length !== 2) throw new OmaiError('shape', 'integralImage: images are [H, W]')
  const [h = 0, w = 0] = img.shape
  const out = new Float64Array(h * w)
  for (let y = 0; y < h; y++) {
    let row = 0
    for (let xx = 0; xx < w; xx++) {
      row += img.data[y * w + xx] ?? 0
      out[y * w + xx] = row + (y > 0 ? (out[(y - 1) * w + xx] ?? 0) : 0)
    }
  }
  return fromStorage(out, [h, w], 'float64')
}

/**
 * The sum of an image over the box [r0, r1] x [c0, c1] (inclusive) from its integral image
 * (skimage.transform.integrate).
 *
 * @param table - Integral image.
 * @param r0 - First row.
 * @param c0 - First column.
 * @param r1 - Last row.
 * @param c1 - Last column.
 * @returns The box sum.
 * @concept integral-image
 * @example
 * ```ts
 * import { image } from 'omai-engine/signal'
 * image.integrate(image.integralImage([[1, 2], [3, 4]]), 0, 1, 1, 1) // => 6
 * ```
 */
export function integrate(table: TensorLike, r0: number, c0: number, r1: number, c1: number): number {
  const t = imageOf(table)
  const w = t.shape[1] ?? 0
  const at = (r: number, c: number): number => (r < 0 || c < 0 ? 0 : (t.data[r * w + c] ?? 0))
  return at(r1, c1) - at(r0 - 1, c1) - at(r1, c0 - 1) + at(r0 - 1, c0 - 1)
}

/**
 * A projective warp (skimage.transform.warp with a ProjectiveTransform as the inverse map): the output pixel at
 * (col, row) samples the input at H (col, row, 1) dehomogenised; clipped to the input range.
 *
 * @param x - Image `[H, W]`.
 * @param h - The 3 x 3 output-to-input homography in (x = col, y = row) coordinates.
 * @param options - Output shape, order, mode (default 'constant'), constant and clipping.
 * @returns The warped image.
 * @concept geometric-transform
 * @example
 * ```ts
 * import { image } from 'omai-engine/signal'
 * import { toArray } from 'omai-engine/tensor'
 * toArray(image.warpPerspective([[1, 2], [3, 4]], [[1, 0, 0], [0, 1, 0], [0, 0, 1]])) // => [[1, 2], [3, 4]]
 * ```
 */
export function warpPerspective(
  x: TensorLike,
  h: TensorLike,
  options: {
    readonly outputShape?: readonly [number, number]
    readonly order?: 0 | 1
    readonly mode?: SkimageMode
    readonly cval?: number
    readonly clip?: boolean
  } = {},
): Tensor {
  const img = imageOf(x)
  if (img.shape.length !== 2) throw new OmaiError('shape', 'warpPerspective: images are [H, W]')
  const m = toFlat(primalOf(float(h)))
  if (m.length !== 9) throw new OmaiError('shape', 'warpPerspective: the homography is 3 x 3')
  const mode = options.mode ?? 'constant'
  const cval = options.cval ?? 0
  const out = transformInto(
    img,
    options.outputShape ?? img.shape,
    (o, into) => {
      const col = o[1] ?? 0
      const row = o[0] ?? 0
      const zx = (m[0] ?? 0) * col + (m[1] ?? 0) * row + (m[2] ?? 0)
      const zy = (m[3] ?? 0) * col + (m[4] ?? 0) * row + (m[5] ?? 0)
      const zw = (m[6] ?? 0) * col + (m[7] ?? 0) * row + (m[8] ?? 0)
      into[0] = zy / zw
      into[1] = zx / zw
    },
    { order: options.order ?? 1, mode: ndimageMode(mode), cval },
  )
  return (options.clip ?? true) ? clipToInput(out, img, mode, cval) : out
}
