/**
 * Plumbing shared by the vision files: images as interleaved float64 rasters (`[H, W]` or `[H, W, C]`, channels
 * last), point lists in OpenCV's (x, y) order, single-precision rounding for parity with OpenCV's float code paths,
 * and the conversions between them and tensors. Private to vision.
 */

import { MdmlError, roundHalfEven } from '../base/index.ts'
import { type Point2, toPoints } from '../numerics/geometry/index.ts'
import { fromStorage, primalOf, promoteFloat, type Tensor, type TensorLike, type Value } from '../tensor/index.ts'

/** An image as interleaved samples: `h` rows of `w` pixels of `c` channels, row-major. */
export interface Raster {
  readonly h: number
  readonly w: number
  readonly c: number
  /** 2 for `[H, W]` inputs, 3 for `[H, W, C]`. */
  readonly rank: 2 | 3
  readonly data: Float64Array
}

/** Points given as a tensor `[n, 2]`, nested arrays or a list of pairs. */
export type PointsLike = TensorLike | readonly (readonly number[])[]

/** The primal values of an input as float64 (a fresh copy). */
export function valuesOf(x: TensorLike): { readonly data: Float64Array; readonly shape: readonly number[] } {
  const v = primalOf((promoteFloat(x) as [Value])[0])
  return { data: Float64Array.from(v.data, Number), shape: [...v.shape] }
}

/** An image input as a raster; `[H, W]` and `[H, W, C]` are accepted. */
export function rasterOf(x: TensorLike, name: string): Raster {
  const { data, shape } = valuesOf(x)
  if (shape.length === 2) return { h: shape[0] ?? 0, w: shape[1] ?? 0, c: 1, rank: 2, data }
  if (shape.length === 3) return { h: shape[0] ?? 0, w: shape[1] ?? 0, c: shape[2] ?? 1, rank: 3, data }
  throw new MdmlError('shape', `${name}: images are [H, W] or [H, W, C]`)
}

/** A single-channel image (`[H, W]` or `[H, W, 1]`). */
export function greyOf(x: TensorLike, name: string): Raster {
  const r = rasterOf(x, name)
  if (r.c !== 1) throw new MdmlError('shape', `${name}: expected a single-channel image`)
  return r
}

/** A three-channel image `[H, W, 3]`. */
export function rgbOf(x: TensorLike, name: string): Raster {
  const r = rasterOf(x, name)
  if (r.rank !== 3 || r.c !== 3) throw new MdmlError('shape', `${name}: expected an image [H, W, 3]`)
  return r
}

/** A float64 image tensor: `[h, w]` for one channel unless `rank` is 3, otherwise `[h, w, c]`. */
export function imageTensor(data: Float64Array, h: number, w: number, c = 1, rank: 2 | 3 = c === 1 ? 2 : 3): Tensor {
  return fromStorage(data, rank === 2 ? [h, w] : [h, w, c], 'float64')
}

/** A raster back as a tensor, keeping its rank. */
export function rasterTensor(r: Raster): Tensor {
  return imageTensor(r.data, r.h, r.w, r.c, r.rank)
}

/** Planar points as pairs. */
export function pointsOf(points: PointsLike, name: string): Point2[] {
  if (Array.isArray(points) && points.length === 0) return []
  try {
    return toPoints(points as TensorLike)
  } catch {
    throw new MdmlError('shape', `${name}: points are [n, 2]`)
  }
}

/** Single-precision rounding, for agreement with OpenCV's float32 arithmetic. */
export const f32: (x: number) => number = Math.fround
/** Clamp to [lo, hi]. */
export const clamp = (x: number, lo: number, hi: number): number => Math.min(hi, Math.max(lo, x))

/** An image's pixel at 256 levels (OpenCV's 8-bit view of a [0, 1] image): round(255 v) clamped to 0 .. 255. */
export const level = (v: number): number => clamp(roundHalfEven(255 * v), 0, 255)

const ATAN_P1 = f32(f32(0.9997878412794807) * f32(180 / Math.PI))
const ATAN_P3 = f32(f32(-0.3258083974640975) * f32(180 / Math.PI))
const ATAN_P5 = f32(f32(0.1555786518463281) * f32(180 / Math.PI))
const ATAN_P7 = f32(f32(-0.04432655554792128) * f32(180 / Math.PI))
const F32_DBL_EPSILON = f32(2.220446049250313e-16)

/**
 * OpenCV's fastAtan2 in degrees [0, 360): its float32 polynomial, with the multiply-adds fused as arm64 builds compile
 * them.
 */
export function fastAtan2(y: number, x: number): number {
  const fy = f32(y)
  const fx = f32(x)
  const ax = Math.abs(fx)
  const ay = Math.abs(fy)
  const poly = (c: number): number => {
    const c2 = f32(c * c)
    return f32(f32(f32(ATAN_P7 * c2 + ATAN_P5) * c2 + ATAN_P3) * c2 + ATAN_P1)
  }
  let a: number
  if (ax >= ay) {
    const c = f32(ay / f32(ax + F32_DBL_EPSILON))
    a = f32(poly(c) * c)
  } else {
    const c = f32(ax / f32(ay + F32_DBL_EPSILON))
    a = f32(90 - poly(c) * c)
  }
  if (fx < 0) a = f32(180 - a)
  if (fy < 0) a = f32(360 - a)
  return a
}

/** The sRGB decoding curve (gamma expansion) of one value in [0, 1]. */
export function srgbDecode(v: number): number {
  return v <= 0.04045 ? v / 12.92 : ((v + 0.055) / 1.055) ** 2.4
}

/** The sRGB encoding curve (gamma compression) of one value in [0, 1]. */
export function srgbEncode(v: number): number {
  return v <= 0.0031308 ? 12.92 * v : 1.055 * v ** (1 / 2.4) - 0.055
}
