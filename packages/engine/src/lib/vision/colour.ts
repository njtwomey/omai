/**
 * Colour spaces, as OpenCV's `cvtColor` defines them for floating-point images in [0, 1] (channels last, RGB order):
 * grey by the ITU-R BT.601 or BT.709 luma weights, HSV with the hue in degrees, CIE XYZ and L*a*b* under the D65
 * white (Lab through the sRGB transfer curve), and YCrCb; `cvtColor` dispatches on OpenCV's conversion names.
 */

import { OmaiError } from '../base/index.ts'
import type { Tensor, TensorLike } from '../tensor/index.ts'
import { greyOf, imageTensor, type Raster, rgbOf, srgbDecode, srgbEncode } from './_image.ts'

/** The luma weights of a grey conversion: ITU-R BT.601 (OpenCV's) or BT.709 (HDTV, sRGB). */
export type LumaStandard = '601' | '709'

const LUMA: Readonly<Record<LumaStandard, readonly [number, number, number]>> = {
  '601': [0.299, 0.587, 0.114],
  '709': [0.2126, 0.7152, 0.0722],
}

// OpenCV's linear sRGB to XYZ matrix and its inverse (colour_lab.cpp), and the D65 white point.
const RGB_TO_XYZ = [0.412453, 0.35758, 0.180423, 0.212671, 0.71516, 0.072169, 0.019334, 0.119193, 0.950227]
const XYZ_TO_RGB = [3.240479, -1.53715, -0.498535, -0.969256, 1.875991, 0.041556, 0.055648, -0.204043, 1.057311]
const D65: readonly [number, number, number] = [0.950456, 1, 1.088754]

/** Apply a per-pixel map from three channels to three channels. */
function map3(r: Raster, f: (a: number, b: number, c: number, out: Float64Array) => void): Tensor {
  const n = r.h * r.w
  const out = new Float64Array(3 * n)
  const px = new Float64Array(3)
  for (let i = 0; i < n; i++) {
    f(r.data[3 * i] ?? 0, r.data[3 * i + 1] ?? 0, r.data[3 * i + 2] ?? 0, px)
    out[3 * i] = px[0] ?? 0
    out[3 * i + 1] = px[1] ?? 0
    out[3 * i + 2] = px[2] ?? 0
  }
  return imageTensor(out, r.h, r.w, 3)
}

/**
 * Grey from RGB (cv2.COLOR_RGB2GRAY): the weighted sum of the channels by the BT.601 luma weights (0.299, 0.587,
 * 0.114), or the BT.709 ones (0.2126, 0.7152, 0.0722).
 *
 * @param x - Image `[H, W, 3]`.
 * @param options - The luma standard (default '601', OpenCV's).
 * @returns The grey image `[H, W]`.
 * @concept colour-space
 * @example
 * ```ts
 * import { rgbToGrey } from 'omai-engine/vision'
 * import { toArray } from 'omai-engine/tensor'
 * toArray(rgbToGrey([[[1, 0, 0], [1, 1, 1]]])) // => [[0.299, 1]]
 * ```
 */
export function rgbToGrey(x: TensorLike, options: { readonly standard?: LumaStandard } = {}): Tensor {
  const r = rgbOf(x, 'rgbToGrey')
  const [wr, wg, wb] = LUMA[options.standard ?? '601']
  const n = r.h * r.w
  const out = new Float64Array(n)
  for (let i = 0; i < n; i++)
    out[i] = wr * (r.data[3 * i] ?? 0) + wg * (r.data[3 * i + 1] ?? 0) + wb * (r.data[3 * i + 2] ?? 0)
  return imageTensor(out, r.h, r.w)
}

/**
 * RGB from grey (cv2.COLOR_GRAY2RGB): the grey value copied to the three channels.
 *
 * @param x - Image `[H, W]`.
 * @returns The image `[H, W, 3]`.
 * @concept colour-space
 * @example
 * ```ts
 * import { greyToRgb } from 'omai-engine/vision'
 * greyToRgb([[0.5, 1]]).shape // => [1, 2, 3]
 * ```
 */
export function greyToRgb(x: TensorLike): Tensor {
  const r = greyOf(x, 'greyToRgb')
  const out = new Float64Array(3 * r.h * r.w)
  for (let i = 0; i < r.h * r.w; i++) out.fill(r.data[i] ?? 0, 3 * i, 3 * i + 3)
  return imageTensor(out, r.h, r.w, 3)
}

/**
 * HSV from RGB (cv2.COLOR_RGB2HSV on float images): value V = max(R, G, B), saturation S = (V - min) / V and the hue
 * H in degrees [0, 360) by the hexagonal sector of the largest channel; grey pixels have H = S = 0.
 *
 * @param x - Image `[H, W, 3]` in [0, 1].
 * @returns The HSV image `[H, W, 3]` (H in degrees, S and V in [0, 1]).
 * @concept colour-space
 * @example
 * ```ts
 * import { rgbToHsv } from 'omai-engine/vision'
 * import { toArray } from 'omai-engine/tensor'
 * toArray(rgbToHsv([[[0, 0.5, 1]]])) // => [[[210, 1, 1]]]
 * ```
 */
export function rgbToHsv(x: TensorLike): Tensor {
  return map3(rgbOf(x, 'rgbToHsv'), (r, g, b, out) => {
    const v = Math.max(r, g, b)
    const diff = v - Math.min(r, g, b)
    const s = v > 0 ? diff / v : 0
    let h = 0
    if (diff > 0) {
      if (v === r) h = (60 * (g - b)) / diff
      else if (v === g) h = (60 * (b - r)) / diff + 120
      else h = (60 * (r - g)) / diff + 240
      if (h < 0) h += 360
    }
    out[0] = h
    out[1] = s
    out[2] = v
  })
}

/**
 * RGB from HSV (cv2.COLOR_HSV2RGB on float images): the hexagonal sector of the hue and linear ramps within it.
 *
 * @param x - HSV image `[H, W, 3]` (H in degrees, S and V in [0, 1]).
 * @returns The RGB image `[H, W, 3]`.
 * @concept colour-space
 * @example
 * ```ts
 * import { hsvToRgb } from 'omai-engine/vision'
 * import { toArray } from 'omai-engine/tensor'
 * toArray(hsvToRgb([[[210, 1, 1]]])) // => [[[0, 0.5, 1]]]
 * ```
 */
export function hsvToRgb(x: TensorLike): Tensor {
  return map3(rgbOf(x, 'hsvToRgb'), (h, s, v, out) => {
    if (s === 0) {
      out.fill(v)
      return
    }
    let hh = (h % 360) / 60
    if (hh < 0) hh += 6
    const sector = Math.min(5, Math.floor(hh))
    const f = hh - sector
    const p = v * (1 - s)
    const q = v * (1 - s * f)
    const t = v * (1 - s * (1 - f))
    const rgb = [
      [v, t, p],
      [q, v, p],
      [p, v, t],
      [p, q, v],
      [t, p, v],
      [v, p, q],
    ][sector] as number[]
    out[0] = rgb[0] ?? 0
    out[1] = rgb[1] ?? 0
    out[2] = rgb[2] ?? 0
  })
}

const linear3 = (m: readonly number[], a: number, b: number, c: number, out: Float64Array): void => {
  out[0] = (m[0] ?? 0) * a + (m[1] ?? 0) * b + (m[2] ?? 0) * c
  out[1] = (m[3] ?? 0) * a + (m[4] ?? 0) * b + (m[5] ?? 0) * c
  out[2] = (m[6] ?? 0) * a + (m[7] ?? 0) * b + (m[8] ?? 0) * c
}

/**
 * CIE XYZ from linear RGB (cv2.COLOR_RGB2XYZ): OpenCV's sRGB primaries matrix under the D65 white, applied without a
 * transfer curve.
 *
 * @param x - Linear RGB image `[H, W, 3]`.
 * @returns The XYZ image `[H, W, 3]`.
 * @concept colour-space
 * @example
 * ```ts
 * import { rgbToXyz } from 'omai-engine/vision'
 * import { toArray } from 'omai-engine/tensor'
 * toArray(rgbToXyz([[[1, 1, 1]]])) // => [[[0.950456, 1, 1.088754]]]
 * ```
 */
export function rgbToXyz(x: TensorLike): Tensor {
  return map3(rgbOf(x, 'rgbToXyz'), (r, g, b, out) => linear3(RGB_TO_XYZ, r, g, b, out))
}

/**
 * Linear RGB from CIE XYZ (cv2.COLOR_XYZ2RGB): OpenCV's inverse primaries matrix.
 *
 * @param x - XYZ image `[H, W, 3]`.
 * @returns The linear RGB image `[H, W, 3]`.
 * @concept colour-space
 * @example
 * ```ts
 * import { xyzToRgb } from 'omai-engine/vision'
 * import { toArray } from 'omai-engine/tensor'
 * toArray(xyzToRgb([[[0.950456, 1, 1.088754]]])).flat(2).map((v) => Math.round(v * 1e4) / 1e4) // => [1, 1, 1]
 * ```
 */
export function xyzToRgb(x: TensorLike): Tensor {
  return map3(rgbOf(x, 'xyzToRgb'), (a, b, c, out) => linear3(XYZ_TO_RGB, a, b, c, out))
}

// OpenCV's (and scikit-image's) rounded CIE constants: the threshold (6/29)^3, the toe slope (29/3)^3 / (29 4) and
// the dark-range lightness slope (29/3)^3.
const LAB_T = 0.008856
const LAB_K = 7.787
const LAB_DARK = 903.3
const labF = (t: number): number => (t > LAB_T ? Math.cbrt(t) : LAB_K * t + 16 / 116)

/** Options of the Lab conversions. */
export interface LabOptions {
  /** The reference white in XYZ (default OpenCV's D65, [0.950456, 1, 1.088754]). */
  readonly white?: readonly [number, number, number]
}

/**
 * CIE L*a*b* from sRGB (cv2.COLOR_RGB2Lab on float images): the sRGB curve decoded, XYZ relative to the white point,
 * then L = 116 f(Y) - 16 (903.3 Y in the dark), a = 500 (f(X) - f(Y)), b = 200 (f(Y) - f(Z)) with f the cube root
 * and its linear toe. Computed exactly, as scikit-image does; OpenCV interpolates a lookup table for float images and
 * differs by up to about 0.4.
 *
 * @param x - sRGB image `[H, W, 3]` in [0, 1] (values are clipped to it).
 * @param options - The reference white.
 * @returns The Lab image `[H, W, 3]` (L in [0, 100]).
 * @concept colour-space
 * @example
 * ```ts
 * import { rgbToLab } from 'omai-engine/vision'
 * import { toArray } from 'omai-engine/tensor'
 * toArray(rgbToLab([[[1, 1, 1]]])).flat(2).map((v) => Math.round(v * 1e4) / 1e4) // => [100, 0, 0]
 * ```
 */
export function rgbToLab(x: TensorLike, options: LabOptions = {}): Tensor {
  const [wx, wy, wz] = options.white ?? D65
  const xyz = new Float64Array(3)
  return map3(rgbOf(x, 'rgbToLab'), (r, g, b, out) => {
    const clip = (v: number): number => Math.min(1, Math.max(0, v))
    linear3(RGB_TO_XYZ, srgbDecode(clip(r)), srgbDecode(clip(g)), srgbDecode(clip(b)), xyz)
    const X = (xyz[0] ?? 0) / wx
    const Y = (xyz[1] ?? 0) / wy
    const Z = (xyz[2] ?? 0) / wz
    const fy = labF(Y)
    out[0] = Y > LAB_T ? 116 * fy - 16 : LAB_DARK * Y
    out[1] = 500 * (labF(X) - fy)
    out[2] = 200 * (fy - labF(Z))
  })
}

/**
 * sRGB from CIE L*a*b* (cv2.COLOR_Lab2RGB on float images): the inverse of {@link rgbToLab} by OpenCV's formulas,
 * clipped to [0, 1] before the sRGB curve is applied.
 *
 * @param x - Lab image `[H, W, 3]`.
 * @param options - The reference white.
 * @returns The sRGB image `[H, W, 3]`.
 * @concept colour-space
 * @example
 * ```ts
 * import { labToRgb } from 'omai-engine/vision'
 * import { toArray } from 'omai-engine/tensor'
 * toArray(labToRgb([[[100, 0, 0]]])).flat(2).map((v) => Math.round(v * 1e4) / 1e4) // => [1, 1, 1]
 * ```
 */
export function labToRgb(x: TensorLike, options: LabOptions = {}): Tensor {
  const [wx, wy, wz] = options.white ?? D65
  const rgb = new Float64Array(3)
  return map3(rgbOf(x, 'labToRgb'), (L, a, b, out) => {
    let y: number
    let fy: number
    if (L <= LAB_DARK * LAB_T) {
      y = L / LAB_DARK
      fy = LAB_K * y + 16 / 116
    } else {
      fy = (L + 16) / 116
      y = fy * fy * fy
    }
    const finv = (f: number): number => (f <= 6 / 29 ? (f - 16 / 116) / LAB_K : f * f * f)
    linear3(XYZ_TO_RGB, finv(a / 500 + fy) * wx, y * wy, finv(fy - b / 200) * wz, rgb)
    for (let k = 0; k < 3; k++) out[k] = srgbEncode(Math.min(1, Math.max(0, rgb[k] ?? 0)))
  })
}

/**
 * YCrCb from RGB (cv2.COLOR_RGB2YCrCb on float images): the BT.601 luma Y, then Cr = 0.713 (R - Y) + 0.5 and
 * Cb = 0.564 (B - Y) + 0.5, in that channel order.
 *
 * @param x - Image `[H, W, 3]`.
 * @returns The YCrCb image `[H, W, 3]`.
 * @concept colour-space
 * @example
 * ```ts
 * import { rgbToYcrcb } from 'omai-engine/vision'
 * import { toArray } from 'omai-engine/tensor'
 * toArray(rgbToYcrcb([[[1, 1, 1]]])) // => [[[1, 0.5, 0.5]]]
 * ```
 */
export function rgbToYcrcb(x: TensorLike): Tensor {
  return map3(rgbOf(x, 'rgbToYcrcb'), (r, g, b, out) => {
    const y = 0.299 * r + 0.587 * g + 0.114 * b
    out[0] = y
    out[1] = (r - y) * 0.713 + 0.5
    out[2] = (b - y) * 0.564 + 0.5
  })
}

/**
 * RGB from YCrCb (cv2.COLOR_YCrCb2RGB on float images): R = Y + 1.403 (Cr - 0.5), G = Y - 0.714 (Cr - 0.5) -
 * 0.344 (Cb - 0.5), B = Y + 1.773 (Cb - 0.5).
 *
 * @param x - YCrCb image `[H, W, 3]`.
 * @returns The RGB image `[H, W, 3]`.
 * @concept colour-space
 * @example
 * ```ts
 * import { ycrcbToRgb } from 'omai-engine/vision'
 * import { toArray } from 'omai-engine/tensor'
 * toArray(ycrcbToRgb([[[1, 0.5, 0.5]]])) // => [[[1, 1, 1]]]
 * ```
 */
export function ycrcbToRgb(x: TensorLike): Tensor {
  return map3(rgbOf(x, 'ycrcbToRgb'), (y, cr, cb, out) => {
    out[0] = y + 1.403 * (cr - 0.5)
    out[1] = y - 0.714 * (cr - 0.5) - 0.344 * (cb - 0.5)
    out[2] = y + 1.773 * (cb - 0.5)
  })
}

/** The conversions {@link cvtColor} knows, by OpenCV's names (RGB channel order). */
export type ColourConversion =
  | 'RGB2GRAY'
  | 'GRAY2RGB'
  | 'RGB2HSV'
  | 'HSV2RGB'
  | 'RGB2XYZ'
  | 'XYZ2RGB'
  | 'RGB2Lab'
  | 'Lab2RGB'
  | 'RGB2YCrCb'
  | 'YCrCb2RGB'

const CONVERSIONS: Readonly<Record<ColourConversion, (x: TensorLike) => Tensor>> = {
  RGB2GRAY: (x) => rgbToGrey(x),
  GRAY2RGB: greyToRgb,
  RGB2HSV: rgbToHsv,
  HSV2RGB: hsvToRgb,
  RGB2XYZ: rgbToXyz,
  XYZ2RGB: xyzToRgb,
  RGB2Lab: rgbToLab,
  Lab2RGB: labToRgb,
  RGB2YCrCb: rgbToYcrcb,
  YCrCb2RGB: ycrcbToRgb,
}

/**
 * Convert an image between colour spaces by OpenCV's conversion name (cv2.cvtColor with the COLOR_ prefix dropped).
 *
 * @param x - Image.
 * @param code - The conversion.
 * @returns The converted image.
 * @throws OmaiError with code `value` for an unknown conversion.
 * @concept colour-space
 * @index function Convert between colour spaces, like cv2.cvtColor.
 * @example
 * ```ts
 * import { cvtColor } from 'omai-engine/vision'
 * cvtColor([[[0.2, 0.4, 0.6]]], 'RGB2GRAY').shape // => [1, 1]
 * ```
 */
export function cvtColor(x: TensorLike, code: ColourConversion): Tensor {
  const f = CONVERSIONS[code]
  if (f === undefined) throw new OmaiError('value', `cvtColor: unknown conversion ${String(code)}`)
  return f(x)
}
